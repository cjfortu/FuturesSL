"""
NQ Futures PyTorch Dataset Module
=================================

This module implements the PyTorch Dataset and DataLoader utilities for the
NQ futures prediction model. It handles rolling window extraction, padding,
attention masking, and temporal feature generation for positional encoding.

Key Design Decisions (per claude-engineering.md Section 3.2):
- Rolling window sampling: ~1 trading week (6900 bars) context per sample
- Window positioning: LAST valid bar = prediction point
- Padding: Right-padding to T_MAX=7000 with attention mask
- Temporal features: 8-channel cyclical encodings for positional embedding
- Train/Val/Test splits: 2010-2019 / 2020-2022 / 2023-2025
- Training stride: 60 bars (non-overlapping hour samples)
- Validation/Test stride: 1 (full resolution for evaluation)

Per grok-scientific.md Section 3.1:
- T_max = 7000 (padded)
- V = 24 features
- Sequence must preserve temporal fidelity (no patching)

References:
    - grok-scientific.md Section 3.1: Input space specification
    - grok-scientific.md Section 3.2: Cyclical positional embeddings
    - claude-engineering.md Section 3.2: Dataset implementation
    
Authors: Claude (Engineering Lead), Gemini (Research), Grok (Scientific)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Configure module logger
logger = logging.getLogger(__name__)


# Constants per grok-scientific.md
T_MAX: int = 7000  # Maximum padded sequence length
BARS_PER_WEEK: int = 6900  # Approximate bars in one trading week
HORIZONS: List[int] = [5, 15, 30, 60, 120, 240]  # Target horizons in minutes
NUM_QUANTILES: int = 7  # [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

# Default data splits per claude-engineering.md
DEFAULT_TRAIN_END: str = "2020-01-01"  # Train: 2010-06-06 to 2019-12-31
DEFAULT_VAL_END: str = "2023-01-01"    # Val: 2020-01-01 to 2022-12-31
                                        # Test: 2023-01-01 to 2025-12-03


class NQFuturesDataset(Dataset):
    """
    PyTorch Dataset for NQ futures time series prediction.
    
    This dataset implements rolling window extraction for Transformer-based
    prediction models. Each sample consists of approximately one trading week
    of minute-bar data (context window), with the prediction point at the
    last valid bar of the window.
    
    Window Positioning:
        For a valid sample at index i:
        - Context window: [i - BARS_PER_WEEK + 1, i + 1) = ~6900 bars ending at i
        - Prediction targets: Forward returns from bar i to i+5, i+15, etc.
        - The model sees bars up to and including i, predicts from i forward
    
    Padding Strategy:
        - Sequences shorter than T_MAX are right-padded with zeros
        - Attention mask indicates valid (1.0) vs padded (0.0) positions
        - Padded positions are excluded from attention via mask
    
    Normalization:
        - Training: Computes and saves feature statistics (mean, std)
        - Val/Test: Loads pre-computed statistics for consistent normalization
        - Instance normalization is handled in the model, not here
    
    Attributes:
        mode: Dataset split ('train', 'val', or 'test').
        stride: Sampling stride (60 for training, 1 for val/test).
        feature_columns: List of feature column names (V=24).
        features: NumPy array of feature values (N, V).
        targets: NumPy array of target values (N, H=6).
        timestamps: NumPy array of timestamps.
        valid_indices: Array of valid sample center indices.
        feature_means: Feature means for normalization (V,).
        feature_stds: Feature stds for normalization (V,).
    
    Example:
        >>> dataset = NQFuturesDataset(
        ...     features_path='data/nq_features_v1.parquet',
        ...     targets_path='data/nq_targets_v1.parquet',
        ...     feature_columns=FEATURE_COLUMNS,
        ...     mode='train',
        ...     normalize_stats_path='data/feature_stats.json'
        ... )
        >>> sample = dataset[0]
        >>> print(sample['features'].shape)  # (7000, 24)
        >>> print(sample['mask'].shape)      # (7000,)
        >>> print(sample['targets'].shape)   # (6,)
    """
    
    def __init__(
        self,
        features_path: str,
        targets_path: str,
        feature_columns: List[str],
        mode: str = 'train',
        train_end_date: str = DEFAULT_TRAIN_END,
        val_end_date: str = DEFAULT_VAL_END,
        stride: Optional[int] = None,
        context_bars: int = BARS_PER_WEEK,
        normalize_stats_path: Optional[str] = None,
    ):
        """
        Initialize the NQ futures dataset.
        
        Args:
            features_path: Path to features parquet file.
                Type: str
                Expected columns: feature_columns + ['timestamp']
            targets_path: Path to targets parquet file.
                Type: str
                Expected columns: ['timestamp', 'target_5m', 'target_15m', ...]
            feature_columns: List of feature column names (V=24).
                Type: List[str]
                Must match FEATURE_COLUMNS from features.py
            mode: Dataset split identifier.
                Type: str
                Options: 'train', 'val', 'test'
            train_end_date: End date for training data (exclusive).
                Type: str
                Format: 'YYYY-MM-DD'
                Default: '2020-01-01' (train uses 2010-2019)
            val_end_date: End date for validation data (exclusive).
                Type: str
                Format: 'YYYY-MM-DD'
                Default: '2023-01-01' (val uses 2020-2022)
            stride: Sampling stride for window extraction.
                Type: Optional[int]
                Default: 60 for train (hourly non-overlapping), 1 for val/test
            context_bars: Number of context bars per window.
                Type: int
                Default: 6900 (~1 trading week)
            normalize_stats_path: Path to normalization statistics JSON.
                Type: Optional[str]
                For train: Statistics are computed and saved here
                For val/test: Statistics are loaded from here
        
        Raises:
            ValueError: If mode is not 'train', 'val', or 'test'.
            ValueError: If val/test mode but no normalize_stats_path provided.
            ValueError: If insufficient data for windowing.
        """
        # Validate mode
        if mode not in ('train', 'val', 'test'):
            raise ValueError(f"mode must be 'train', 'val', or 'test', got '{mode}'")
        
        self.mode = mode
        self.feature_columns = feature_columns
        self.context_bars = context_bars
        
        # Set stride based on mode
        # Per claude-engineering.md: 60 for training (non-overlapping hour)
        # Per claude-engineering.md: 1 for val/test (full resolution)
        if stride is not None:
            self.stride = stride
        else:
            self.stride = 60 if mode == 'train' else 1
        
        logger.info(f"Initializing NQFuturesDataset (mode={mode}, stride={self.stride})")
        
        # Load and merge data
        features_df, targets_df = self._load_data(features_path, targets_path)
        
        # Split by date
        df = self._apply_date_split(
            features_df, targets_df, train_end_date, val_end_date
        )
        
        # Extract arrays
        self.features = df[feature_columns].values.astype(np.float32)
        self.targets = df[[f'target_{h}m' for h in HORIZONS]].values.astype(np.float32)
        self.timestamps = df['timestamp'].values
        
        logger.info(f"Loaded {len(self.features):,} bars for {mode} set")
        
        # Handle normalization statistics
        self._setup_normalization(normalize_stats_path)
        
        # Compute valid sample indices
        self.valid_indices = self._compute_valid_indices()
        
        logger.info(f"Created {len(self.valid_indices):,} valid samples")
    
    def _load_data(
        self,
        features_path: str,
        targets_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and validate feature and target data.
        
        Args:
            features_path: Path to features parquet.
                Type: str
            targets_path: Path to targets parquet.
                Type: str
        
        Returns:
            Tuple of (features_df, targets_df).
            Type: Tuple[pd.DataFrame, pd.DataFrame]
        
        Raises:
            ValueError: If required columns are missing.
        """
        features_df = pd.read_parquet(features_path)
        targets_df = pd.read_parquet(targets_path)
        
        # Validate columns
        missing_features = set(self.feature_columns) - set(features_df.columns)
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        
        if 'timestamp' not in features_df.columns:
            raise ValueError("Features DataFrame must have 'timestamp' column")
        
        target_cols = [f'target_{h}m' for h in HORIZONS]
        missing_targets = set(target_cols) - set(targets_df.columns)
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")
        
        return features_df, targets_df
    
    def _apply_date_split(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        train_end: str,
        val_end: str
    ) -> pd.DataFrame:
        """
        Merge features and targets, then filter by date split.
        
        Data Splits (per claude-engineering.md):
            Train: 2010-06-06 to train_end (exclusive)
            Val: train_end to val_end (exclusive)
            Test: val_end to end of data
        
        CRITICAL TIMEZONE HANDLING:
            Databento returns timezone-aware UTC timestamps. We convert to
            timezone-naive for simpler comparison logic, as we don't need
            absolute UTC times - all temporal features are relative (time-of-day,
            day-of-week, etc.). This fixes the TypeError: "Invalid comparison
            between dtype=datetime64[ns, UTC] and Timestamp".
        
        Args:
            features_df: Features DataFrame.
                Type: pd.DataFrame
            targets_df: Targets DataFrame.
                Type: pd.DataFrame
            train_end: Training end date (YYYY-MM-DD).
                Type: str
            val_end: Validation end date (YYYY-MM-DD).
                Type: str
        
        Returns:
            Filtered and merged DataFrame.
            Type: pd.DataFrame
        """
        # Merge on timestamp
        df = features_df.merge(targets_df, on='timestamp', how='inner')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # CRITICAL: Convert timezone-aware (UTC) to timezone-naive
        # Databento returns UTC timestamps, but we don't need absolute UTC times.
        # All our temporal features (time-of-day, day-of-week, etc.) are relative
        # to local market time. Converting to naive allows simple date comparisons.
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Apply date filter with timezone-naive timestamps
        train_end_ts = pd.Timestamp(train_end)
        val_end_ts = pd.Timestamp(val_end)
        
        if self.mode == 'train':
            df = df[df['timestamp'] < train_end_ts]
        elif self.mode == 'val':
            df = df[(df['timestamp'] >= train_end_ts) & (df['timestamp'] < val_end_ts)]
        else:  # test
            df = df[df['timestamp'] >= val_end_ts]
        
        df = df.reset_index(drop=True)
        
        logger.info(
            f"Date range for {self.mode}: "
            f"{df['timestamp'].min()} to {df['timestamp'].max()}"
        )
        
        return df
    
    def _setup_normalization(self, stats_path: Optional[str]) -> None:
        """
        Set up feature normalization statistics.
        
        For training: Compute statistics from training data and optionally save.
        For val/test: Load pre-computed statistics.
        
        Note: These statistics are for reference/external use. Instance
        normalization in the model handles per-sample normalization.
        
        Args:
            stats_path: Path to statistics JSON file.
                Type: Optional[str]
        
        Raises:
            ValueError: If val/test mode but stats_path not provided or missing.
        """
        if self.mode == 'train':
            # Compute statistics from training data
            # Use nanmean/nanstd to handle NaN values in warmup period
            self.feature_means = np.nanmean(self.features, axis=0)
            self.feature_stds = np.nanstd(self.features, axis=0)
            
            # Prevent division by zero for constant features
            self.feature_stds[self.feature_stds < 1e-8] = 1.0
            
            # Save statistics if path provided
            if stats_path:
                self._save_stats(stats_path)
                
        else:
            # Load pre-computed statistics for val/test
            if stats_path is None:
                raise ValueError(
                    f"{self.mode} mode requires normalize_stats_path"
                )
            
            if not Path(stats_path).exists():
                raise ValueError(
                    f"Statistics file not found: {stats_path}. "
                    f"Run training first to generate statistics."
                )
            
            self._load_stats(stats_path)
    
    def _save_stats(self, path: str) -> None:
        """
        Save normalization statistics to JSON.
        
        Args:
            path: Output file path.
                Type: str
        """
        stats = {
            'means': self.feature_means.tolist(),
            'stds': self.feature_stds.tolist(),
            'columns': self.feature_columns,
            'context_bars': self.context_bars,
            't_max': T_MAX,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved normalization statistics to {path}")
    
    def _load_stats(self, path: str) -> None:
        """
        Load normalization statistics from JSON.
        
        Args:
            path: Input file path.
                Type: str
        """
        with open(path, 'r') as f:
            stats = json.load(f)
        
        self.feature_means = np.array(stats['means'], dtype=np.float32)
        self.feature_stds = np.array(stats['stds'], dtype=np.float32)
        
        # Validate column consistency
        if stats.get('columns') != self.feature_columns:
            logger.warning(
                "Feature columns in stats file don't match provided columns. "
                "Proceeding with provided columns."
            )
        
        logger.info(f"Loaded normalization statistics from {path}")
    
    def _compute_valid_indices(self) -> np.ndarray:
        """
        Compute indices that have sufficient history and valid targets.
        
        A valid sample at index i requires:
        1. At least context_bars of history available (i >= context_bars - 1)
        2. All target horizons available (i <= len - max_horizon - 1)
        3. No NaN in target values at index i
        
        Returns:
            Array of valid center indices.
            Type: np.ndarray
            Shape: (num_valid_samples,)
        """
        n = len(self.features)
        max_horizon = max(HORIZONS)
        
        # Valid range for center index
        # Need context_bars - 1 bars before (so context_bars total including center)
        # Need max_horizon bars after for targets
        valid_start = self.context_bars - 1
        valid_end = n - max_horizon
        
        if valid_end <= valid_start:
            raise ValueError(
                f"Insufficient data for windowing. "
                f"Have {n} bars, need at least {self.context_bars + max_horizon}"
            )
        
        # Check for NaN targets
        # Targets at index i are for returns starting from bar i
        target_valid = ~np.isnan(self.targets).any(axis=1)
        
        # Build valid indices with stride
        indices = []
        for i in range(valid_start, valid_end, self.stride):
            if target_valid[i]:
                indices.append(i)
        
        return np.array(indices, dtype=np.int64)
    
    def __len__(self) -> int:
        """
        Return number of valid samples.
        
        Returns:
            Number of samples.
            Type: int
        """
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with its context window.
        
        Window extraction:
            For center index i, extract [i - context_bars + 1, i + 1)
            This gives context_bars total, with prediction point at the end.
        
        Args:
            idx: Sample index (0 to len-1).
                Type: int
        
        Returns:
            Dictionary containing:
            - 'features': (T_MAX, V) padded feature tensor
            - 'mask': (T_MAX,) attention mask (1.0 = valid, 0.0 = pad)
            - 'targets': (H,) forward returns for each horizon
            - 'temporal_features': (T_MAX, 8) positional encoding inputs
            - 'seq_len': Scalar, actual sequence length before padding
            
            Type: Dict[str, torch.Tensor]
        """
        # Get center index (prediction point)
        center_idx = self.valid_indices[idx]
        
        # Extract window: [center_idx - context_bars + 1, center_idx + 1)
        start_idx = center_idx - self.context_bars + 1
        end_idx = center_idx + 1
        
        # Handle edge case where start_idx < 0 (shouldn't happen with valid_indices)
        if start_idx < 0:
            start_idx = 0
        
        # Extract window data
        window_features = self.features[start_idx:end_idx]  # (actual_len, V)
        window_timestamps = self.timestamps[start_idx:end_idx]
        actual_len = len(window_features)
        
        # Create padded output arrays
        padded_features = np.zeros((T_MAX, len(self.feature_columns)), dtype=np.float32)
        mask = np.zeros(T_MAX, dtype=np.float32)
        
        # Fill with actual data (left-aligned, padding on right)
        padded_features[:actual_len] = window_features
        mask[:actual_len] = 1.0
        
        # Handle NaN values in features (from warmup period)
        # Replace with 0 for padded positions - model uses instance norm anyway
        nan_mask = np.isnan(padded_features)
        padded_features[nan_mask] = 0.0
        
        # Extract temporal features for positional encoding
        temporal_features = self._extract_temporal_features(window_timestamps, actual_len)
        
        # Get targets at center_idx
        targets = self.targets[center_idx]
        
        return {
            'features': torch.from_numpy(padded_features),
            'mask': torch.from_numpy(mask),
            'targets': torch.from_numpy(targets),
            'temporal_features': torch.from_numpy(temporal_features),
            'seq_len': torch.tensor(actual_len, dtype=torch.long),
        }
    
    def _extract_temporal_features(
        self,
        timestamps: np.ndarray,
        actual_len: int
    ) -> np.ndarray:
        """
        Extract temporal features for cyclical positional encoding.
        
        Per grok-scientific.md Section 3.2:
        - Time-of-day: sin/cos with period 1440 minutes
        - Day-of-week: Index for learned embedding (7 classes)
        - Day-of-month: sin/cos with period 31
        - Day-of-year: sin/cos with period 365.25
        
        Output channels (8 total):
            [0]: sin(time_of_day)
            [1]: cos(time_of_day)
            [2]: day_of_week (0-6, integer for embedding)
            [3]: sin(day_of_month)
            [4]: cos(day_of_month)
            [5]: sin(day_of_year)
            [6]: cos(day_of_year)
            [7]: normalized minute (for reference, 0-1)
        
        Args:
            timestamps: Array of timestamps for the window.
                Type: np.ndarray
                Shape: (actual_len,)
            actual_len: Number of actual (non-padded) positions.
                Type: int
        
        Returns:
            Temporal features array.
            Type: np.ndarray
            Shape: (T_MAX, 8)
        """
        temporal = np.zeros((T_MAX, 8), dtype=np.float32)
        
        if actual_len == 0:
            return temporal
        
        # Convert to pandas datetime for feature extraction
        ts = pd.to_datetime(timestamps)
        
        # Time of day (minute of day: 0-1439)
        minutes = ts.hour * 60 + ts.minute
        temporal[:actual_len, 0] = np.sin(2 * np.pi * minutes / 1440)
        temporal[:actual_len, 1] = np.cos(2 * np.pi * minutes / 1440)
        
        # Day of week (0-6, Monday=0)
        temporal[:actual_len, 2] = ts.dayofweek
        
        # Day of month (1-31, period=31 per grok-scientific.md)
        dom = ts.day
        temporal[:actual_len, 3] = np.sin(2 * np.pi * dom / 31)
        temporal[:actual_len, 4] = np.cos(2 * np.pi * dom / 31)
        
        # Day of year (1-366, period=365.25 per grok-scientific.md)
        doy = ts.dayofyear
        temporal[:actual_len, 5] = np.sin(2 * np.pi * doy / 365.25)
        temporal[:actual_len, 6] = np.cos(2 * np.pi * doy / 365.25)
        
        # Normalized minute for reference
        temporal[:actual_len, 7] = minutes / 1440.0
        
        return temporal
    
    def get_sample_weight(self, idx: int) -> float:
        """
        Get sample weight for weighted sampling (optional).
        
        Can be used to implement importance sampling based on:
        - Volatility regime
        - Time of day
        - Market conditions
        
        Currently returns uniform weight.
        
        Args:
            idx: Sample index.
                Type: int
        
        Returns:
            Sample weight.
            Type: float
        """
        return 1.0


def create_dataloaders(
    features_path: str,
    targets_path: str,
    feature_columns: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    train_end_date: str = DEFAULT_TRAIN_END,
    val_end_date: str = DEFAULT_VAL_END,
    stats_path: str = 'feature_stats.json',
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    This convenience function creates all three DataLoaders with appropriate
    settings for each split:
    - Train: Shuffled, stride=60, batch_size as specified
    - Val: Not shuffled, stride=1, batch_size*2
    - Test: Not shuffled, stride=1, batch_size*2
    
    Args:
        features_path: Path to features parquet file.
            Type: str
        targets_path: Path to targets parquet file.
            Type: str
        feature_columns: List of feature column names.
            Type: List[str]
        batch_size: Training batch size.
            Type: int
            Default: 8 (per claude-engineering.md for A100)
        num_workers: Number of data loading workers.
            Type: int
            Default: 4 (adjust based on CPU cores)
        train_end_date: Training split end date.
            Type: str
            Default: '2020-01-01'
        val_end_date: Validation split end date.
            Type: str
            Default: '2023-01-01'
        stats_path: Path for normalization statistics.
            Type: str
            Default: 'feature_stats.json'
        pin_memory: Pin memory for faster GPU transfer.
            Type: bool
            Default: True
        prefetch_factor: Number of batches to prefetch.
            Type: int
            Default: 2
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
        Type: Tuple[DataLoader, DataLoader, DataLoader]
    
    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     'data/nq_features_v1.parquet',
        ...     'data/nq_targets_v1.parquet',
        ...     FEATURE_COLUMNS,
        ...     batch_size=8
        ... )
        >>> for batch in train_loader:
        ...     features = batch['features']  # (B, T_MAX, V)
        ...     break
    """
    logger.info("Creating DataLoaders...")
    
    # Create datasets
    train_ds = NQFuturesDataset(
        features_path=features_path,
        targets_path=targets_path,
        feature_columns=feature_columns,
        mode='train',
        train_end_date=train_end_date,
        val_end_date=val_end_date,
        normalize_stats_path=stats_path,
    )
    
    val_ds = NQFuturesDataset(
        features_path=features_path,
        targets_path=targets_path,
        feature_columns=feature_columns,
        mode='val',
        train_end_date=train_end_date,
        val_end_date=val_end_date,
        normalize_stats_path=stats_path,
    )
    
    test_ds = NQFuturesDataset(
        features_path=features_path,
        targets_path=targets_path,
        feature_columns=feature_columns,
        mode='test',
        train_end_date=train_end_date,
        val_end_date=val_end_date,
        normalize_stats_path=stats_path,
    )
    
    # Create DataLoaders
    # Train: shuffle for stochastic optimization
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,  # Drop incomplete batches for stable batch norm
    )
    
    # Val/Test: no shuffle, larger batch for faster evaluation
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
    )
    
    logger.info(
        f"DataLoaders created: "
        f"train={len(train_ds):,} samples, "
        f"val={len(val_ds):,} samples, "
        f"test={len(test_ds):,} samples"
    )
    
    return train_loader, val_loader, test_loader


def collate_fn_variable_length(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.
    
    This collate function stacks pre-padded sequences. Since NQFuturesDataset
    already pads to T_MAX, this is a simple stack operation.
    
    Provided for extensibility if dynamic padding is needed in the future.
    
    Args:
        batch: List of sample dictionaries from __getitem__.
            Type: List[Dict[str, torch.Tensor]]
    
    Returns:
        Batched dictionary with stacked tensors.
        Type: Dict[str, torch.Tensor]
        Keys: 'features', 'mask', 'targets', 'temporal_features', 'seq_len'
    """
    return {
        'features': torch.stack([s['features'] for s in batch]),
        'mask': torch.stack([s['mask'] for s in batch]),
        'targets': torch.stack([s['targets'] for s in batch]),
        'temporal_features': torch.stack([s['temporal_features'] for s in batch]),
        'seq_len': torch.stack([s['seq_len'] for s in batch]),
    }
