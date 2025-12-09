"""
PyTorch Dataset and DataModule for NQ futures model.

Implements efficient data loading with proper batching, normalization,
and temporal information extraction for the MIGT-TVDT architecture.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from .preprocessing import DataPreprocessor


class NQFuturesDataset(Dataset):
    """
    PyTorch Dataset for NQ futures prediction.
    
    Each sample consists of:
    - 24-hour lookback window of features (padded to 288)
    - Attention mask for valid data
    - Forward returns at 5 horizons (15m, 30m, 60m, 2h, 4h)
    - Temporal information for positional encodings
    - Normalization statistics for denormalization
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        preprocessor: DataPreprocessor,
        skip_first_n: int = 288,  # Skip first 24h (no full lookback)
        subsample_fraction: Optional[float] = None,  # NEW
        subsample_seed: int = 42                      # NEW
    ):
        """
        Initialize dataset.
        
        Args:
            df: Feature DataFrame with datetime index
            feature_cols: List of feature column names
            target_cols: List of target column names
            preprocessor: DataPreprocessor instance
            skip_first_n: Skip first N bars (insufficient history)
            subsample_fraction: Use only this fraction of samples (None = all)
            subsample_seed: Random seed for subsampling reproducibility
        """
        self.df = df  # Keep FULL DataFrame for lookback windows
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.preprocessor = preprocessor
        
        # Valid indices (skip first 24h for full lookback)
        # Also skip last samples with NaN targets
        valid_mask = ~df[target_cols].isna().any(axis=1)
        valid_indices = df.index[valid_mask]
        
        # Get integer positions
        all_positions = np.arange(len(df))
        valid_positions = all_positions[valid_mask.values]
        
        # Filter to skip first N
        self.valid_indices = valid_positions[valid_positions >= skip_first_n]
        
        # NEW: Subsample prediction points (not data!)
        if subsample_fraction is not None:
            if not (0.0 < subsample_fraction <= 1.0):
                raise ValueError(f"subsample_fraction must be in (0, 1], got {subsample_fraction}")
            
            n_original = len(self.valid_indices)
            np.random.seed(subsample_seed)
            n_subsample = int(n_original * subsample_fraction)
            
            # Randomly sample indices, then sort to preserve temporal order
            subsample_positions = np.sort(
                np.random.choice(n_original, n_subsample, replace=False)
            )
            self.valid_indices = self.valid_indices[subsample_positions]
        
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            dict with:
                features: (288, n_features) float32
                attention_mask: (288,) bool
                targets: (5,) float32
                bar_in_day: (actual_length,) int32
                day_of_week: int32 scalar
                day_of_month: int32 scalar
                day_of_year: int32 scalar
                norm_stats: dict (not tensor, for denormalization)
        """
        # Get DataFrame integer position and convert to timestamp
        end_idx = self.valid_indices[idx]
        end_timestamp = self.df.index[end_idx]
        
        # Create window (API requires timestamp)
        features, attention_mask, metadata = self.preprocessor.create_window(
            self.df, end_timestamp
        )
        
        # Normalize
        features_norm, norm_stats = self.preprocessor.normalize_window(
            features, attention_mask
        )
        
        # Get targets
        targets = self.df.iloc[end_idx][self.target_cols].values.astype(np.float32)
        
        # Extract temporal info (API requires timestamps array and end timestamp)
        temporal_info = self.preprocessor.extract_temporal_info(
            timestamps=metadata['timestamps'],
            end_timestamp=metadata['end_timestamp']
        )
        
        return {
            'features': torch.from_numpy(features_norm),
            'attention_mask': torch.from_numpy(attention_mask),
            'targets': torch.from_numpy(targets),
            'bar_in_day': torch.from_numpy(temporal_info['bar_in_day']),
            'day_of_week': torch.tensor(temporal_info['day_of_week'], dtype=torch.int32),
            'day_of_month': torch.tensor(temporal_info['day_of_month'], dtype=torch.int32),
            'day_of_year': torch.tensor(temporal_info['day_of_year'], dtype=torch.int32),
            'norm_stats': norm_stats  # Keep as dict for denormalization
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable-length temporal info.
    
    bar_in_day varies by sample (273-276 length). We pad to the maximum
    length in the batch for efficient tensor operations.
    
    Args:
        batch: List of sample dicts from __getitem__
        
    Returns:
        Batched dict with properly stacked tensors
    """
    # Stack regular tensors
    batch_dict = {
        'features': torch.stack([s['features'] for s in batch]),
        'attention_mask': torch.stack([s['attention_mask'] for s in batch]),
        'targets': torch.stack([s['targets'] for s in batch]),
        'day_of_week': torch.stack([s['day_of_week'] for s in batch]),
        'day_of_month': torch.stack([s['day_of_month'] for s in batch]),
        'day_of_year': torch.stack([s['day_of_year'] for s in batch]),
    }
    
    # Handle variable-length bar_in_day
    bar_in_day_list = [s['bar_in_day'] for s in batch]
    # CHANGE: Pad bar_in_day to max_seq_len=288, not just max_len in batch
    MAX_SEQ_LEN = 288  # Should match features padding

    padded_bar_in_day = []
    for b in bar_in_day_list:
        pad_len = MAX_SEQ_LEN - len(b)
        if pad_len > 0:
            padded = torch.cat([torch.zeros(pad_len, dtype=torch.int32), b])
        else:
            padded = b
        padded_bar_in_day.append(padded)
    
    batch_dict['bar_in_day'] = torch.stack(padded_bar_in_day)
    
    # Keep norm_stats as list (not batched - used for individual denormalization)
    batch_dict['norm_stats'] = [s['norm_stats'] for s in batch]
    
    return batch_dict


class NQDataModule:
    """
    Data module managing train/val/test splits and dataloaders.
    
    Handles:
    - Loading data from parquet
    - Creating time-based splits
    - Initializing datasets
    - Creating dataloaders with proper batching
    """
    
    def __init__(
        self,
        data_path: Path,
        feature_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        max_seq_len: int = 288,
        train_end: str = "2021-12-31",
        val_end: str = "2023-12-31",
        subsample_fraction: Optional[float] = None,
        apply_subsample_to_all_splits: bool = False,
        subsample_seed: int = 42
    ):
        """
        Initialize data module.
        
        Args:
            data_path: Path to feature parquet file
                Type: Path
            feature_cols: Feature column names (auto-detect if None)
                Type: Optional[List[str]]
            target_cols: Target column names (auto-detect if None)
                Type: Optional[List[str]]
            batch_size: Batch size for training
                Type: int
            num_workers: DataLoader workers for parallel data loading
                Type: int
            pin_memory: Pin memory for faster GPU transfer
                Type: bool
            prefetch_factor: Batches to prefetch per worker (overlaps I/O with compute)
                Type: int
            persistent_workers: Reuse workers across epochs (avoids reinit overhead)
                Type: bool
            max_seq_len: Maximum sequence length for padding
                Type: int
            train_end: Training split end date
                Type: str
            val_end: Validation split end date
                Type: str
            subsample_fraction: Fraction of samples to use (None = all)
                Type: Optional[float]
                Range: (0.0, 1.0]
            apply_subsample_to_all_splits: Apply subsampling to val/test (not just train)
                Type: bool
                Default: False (only training subsampled, val/test use full data)
            subsample_seed: Random seed for subsampling reproducibility
                Type: int
        """
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.max_seq_len = max_seq_len
        self.train_end = train_end
        self.val_end = val_end
        
        # Validate subsample_fraction
        if subsample_fraction is not None:
            if not (0.0 < subsample_fraction <= 1.0):
                raise ValueError(
                    f"subsample_fraction must be in (0.0, 1.0], got {subsample_fraction}"
                )
        
        self.subsample_fraction = subsample_fraction
        self.apply_subsample_to_all_splits = apply_subsample_to_all_splits
        self.subsample_seed = subsample_seed
        
        # Will be set in setup()
        self.df = None
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.preprocessor = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self):
        """
        Load data and create splits.
        
        Called once before training to prepare all datasets.
        """
        # Load data
        print(f"Loading data from {self.data_path}")
        self.df = pd.read_parquet(self.data_path)
        
        # Auto-detect columns if not provided
        if self.target_cols is None:
            self.target_cols = [col for col in self.df.columns if col.startswith('target_')]
        
        if self.feature_cols is None:
            self.feature_cols = [col for col in self.df.columns if col not in self.target_cols]
        
        print(f"Features: {len(self.feature_cols)}")
        print(f"Targets: {len(self.target_cols)}")
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(
            max_seq_len=self.max_seq_len,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols
        )
        
        # Create splits with purge gaps
        train_df, val_df, test_df = self.preprocessor.create_splits(
            self.df,
            train_end=self.train_end,
            val_end=self.val_end
            # Uses default purge_hours=24
        )
        
        # Validate no leakage
        self.preprocessor.validate_no_leakage(
            train_df, val_df, test_df, tolerance_hours=24
        )
        
        # Create datasets with conditional subsampling
        # If apply_subsample_to_all_splits=True, apply subsample_fraction to all splits
        # Otherwise, apply only to training (default behavior for backward compatibility)
        if self.apply_subsample_to_all_splits:
            # Apply same fraction to all splits
            self.train_dataset = NQFuturesDataset(
                train_df, 
                self.feature_cols, 
                self.target_cols, 
                self.preprocessor,
                subsample_fraction=self.subsample_fraction,
                subsample_seed=self.subsample_seed
            )
            self.val_dataset = NQFuturesDataset(
                val_df, 
                self.feature_cols, 
                self.target_cols, 
                self.preprocessor,
                subsample_fraction=self.subsample_fraction,
                subsample_seed=self.subsample_seed
            )
            self.test_dataset = NQFuturesDataset(
                test_df, 
                self.feature_cols, 
                self.target_cols, 
                self.preprocessor,
                subsample_fraction=self.subsample_fraction,
                subsample_seed=self.subsample_seed
            )
        else:
            # Default behavior: subsample only training
            self.train_dataset = NQFuturesDataset(
                train_df, 
                self.feature_cols, 
                self.target_cols, 
                self.preprocessor,
                subsample_fraction=self.subsample_fraction,
                subsample_seed=self.subsample_seed
            )
            self.val_dataset = NQFuturesDataset(
                val_df, self.feature_cols, self.target_cols, self.preprocessor
            )
            self.test_dataset = NQFuturesDataset(
                test_df, self.feature_cols, self.target_cols, self.preprocessor
            )
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(self.train_dataset):,}")
        print(f"  Val:   {len(self.val_dataset):,}")
        print(f"  Test:  {len(self.test_dataset):,}")
    
    def train_dataloader(self) -> DataLoader:
        """
        Training dataloader with shuffling.
        
        Returns:
            DataLoader: Training data loader with shuffle=True, drop_last=True
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate_fn,
            drop_last=True  # Drop incomplete batch for stable training
        )
    
    def val_dataloader(self) -> DataLoader:
        """
        Validation dataloader without shuffling.
        
        Returns:
            DataLoader: Validation data loader with shuffle=False
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """
        Test dataloader without shuffling.
        
        Returns:
            DataLoader: Test data loader with shuffle=False
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate_fn
        )
    
    def save_splits(self, output_dir: Path):
        """
        Save train/val/test splits to separate parquet files.
        
        Args:
            output_dir: Directory for split files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get DataFrames from datasets
        train_indices = self.train_dataset.valid_indices
        val_indices = self.val_dataset.valid_indices
        test_indices = self.test_dataset.valid_indices
        
        # Save splits
        self.train_dataset.df.iloc[train_indices].to_parquet(
            output_dir / "train_samples.parquet"
        )
        self.val_dataset.df.iloc[val_indices].to_parquet(
            output_dir / "val_samples.parquet"
        )
        self.test_dataset.df.iloc[test_indices].to_parquet(
            output_dir / "test_samples.parquet"
        )
        
        print(f"Splits saved to {output_dir}")