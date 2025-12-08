"""
Data preprocessing module for NQ futures model.

Handles window creation, padding, normalization, and train/val/test splits
with proper temporal separation to prevent data leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List


class DataPreprocessor:
    """
    Preprocessor for NQ futures time series data.
    
    Creates windowed samples with:
    - 24-hour lookback (273-276 bars depending on trading schedule)
    - Zero-padding to 288 max length
    - Instance normalization (RevIN)
    - Attention masks for valid data
    - Temporal info extraction for positional encodings
    """
    
    def __init__(
        self,
        max_seq_len: int = 288,
        feature_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
        eps: float = 1e-5
    ):
        """
        Initialize the preprocessor.
        
        Args:
            max_seq_len: Maximum sequence length (bars)
            feature_cols: List of feature column names
            target_cols: List of target column names
            eps: Epsilon for numerical stability in normalization
        """
        self.max_seq_len = max_seq_len
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.eps = eps
    
    def create_window(
        self,
        df: pd.DataFrame,
        end_timestamp: pd.Timestamp
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Create a 24-hour lookback window ending at end_timestamp.
        
        Looks back exactly 24 hours (288 bars at 5-min resolution) from the
        current bar. Actual number of valid bars varies (273-276) due to
        trading schedule variations. Zero-padding applied at the start.
        
        Args:
            df: Feature DataFrame with datetime index
            end_timestamp: Timestamp of current bar (must be in df.index)
            
        Returns:
            features: (max_seq_len, n_features) array, zero-padded
            attention_mask: (max_seq_len,) boolean array (True=valid)
            metadata: dict with timestamps, actual_length, etc.
        """
        # Calculate 24-hour lookback
        start_timestamp = end_timestamp - pd.Timedelta(hours=24)
        
        # Extract window (inclusive of end_timestamp)
        window_mask = (df.index > start_timestamp) & (df.index <= end_timestamp)
        window_df = df.loc[window_mask]
        
        # Get features
        if self.feature_cols is None:
            # Use all columns except targets
            features = window_df.drop(columns=self.target_cols).values
        else:
            features = window_df[self.feature_cols].values
        
        actual_length = len(features)
        
        # Create padded array (padding at start, data at end)
        padded_features = np.zeros((self.max_seq_len, features.shape[1]), dtype=np.float32)
        pad_length = self.max_seq_len - actual_length
        padded_features[pad_length:] = features
        
        # Create attention mask (False=padding, True=valid)
        attention_mask = np.zeros(self.max_seq_len, dtype=bool)
        attention_mask[pad_length:] = True
        
        # Metadata
        metadata = {
            'end_timestamp': end_timestamp,
            'start_timestamp': start_timestamp,
            'actual_length': actual_length,
            'pad_length': pad_length,
            'timestamps': window_df.index.to_numpy()
        }
        
        return padded_features, attention_mask, metadata
    
    def normalize_window(
        self,
        features: np.ndarray,
        attention_mask: np.ndarray,
        store_stats: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply Reversible Instance Normalization (RevIN) to the window.
        
        Normalizes only valid (non-padded) positions per feature independently:
        - Normal-variance features: normalized to zero mean, unit variance
        - Low-variance features (std <= eps): centered only, not scaled
          (avoids noise amplification in constant features)
        
        Args:
            features: (max_seq_len, n_features) array
            attention_mask: (max_seq_len,) boolean array
            store_stats: Whether to return stats (always True, for compatibility)
            
        Returns:
            normalized_features: (max_seq_len, n_features) array
            norm_stats: dict with mean, std per feature for denormalization
        """
        # Extract valid positions only
        valid_features = features[attention_mask]  # (actual_length, n_features)
        
        # Compute statistics over valid positions per feature
        mean = valid_features.mean(axis=0)  # (n_features,)
        std = valid_features.std(axis=0)    # (n_features,)
        
        # Handle low-variance features:
        # - If std > eps: normalize to unit variance
        # - If std <= eps: keep std=1.0 (centers without amplifying noise)
        std_safe = np.where(std > self.eps, std, 1.0)
        
        # Normalize only valid positions
        normalized_valid = (valid_features - mean) / std_safe
        
        # Create output array (padded positions remain zero)
        normalized_features = features.copy()
        normalized_features[attention_mask] = normalized_valid
        
        # Store statistics for denormalization
        norm_stats = {
            'mean': mean,
            'std': std,
            'std_safe': std_safe,
            'eps': self.eps
        }
        
        return normalized_features, norm_stats
    
    def denormalize_predictions(
        self,
        normalized_values: np.ndarray,
        norm_stats: Dict,
        feature_idx: int = 3  # Default: close price
    ) -> np.ndarray:
        """
        Reverse normalization for model outputs.
        
        Args:
            normalized_values: Normalized predictions
            norm_stats: Statistics dict from normalize_window()
            feature_idx: Index of feature to denormalize (default: close)
            
        Returns:
            Denormalized values in original scale
        """
        mean = norm_stats['mean'][feature_idx]
        std_safe = norm_stats['std_safe'][feature_idx]
        
        return normalized_values * std_safe + mean
    
    def extract_temporal_info(
        self,
        timestamps: np.ndarray,
        end_timestamp: pd.Timestamp
    ) -> Dict:
        """
        Extract temporal information for positional encodings.
        
        Computes:
        - bar_in_day: Position within 24h window (0-287)
        - day_of_week: 0=Monday, 4=Friday
        - day_of_month: 1-31
        - day_of_year: 1-366
        
        Args:
            timestamps: Array of timestamps for the window bars
            end_timestamp: Current bar timestamp (for day-level features)
            
        Returns:
            dict with temporal features
        """
        # Compute bar indices within the day (0-287)
        # For each timestamp, calculate minutes since midnight
        bar_in_day = []
        for ts in timestamps:
            minutes_since_midnight = ts.hour * 60 + ts.minute
            bar_idx = minutes_since_midnight // 5  # 5-minute bars
            bar_in_day.append(bar_idx)
        
        bar_in_day = np.array(bar_in_day, dtype=np.int32)
        
        # Day-level features (same for entire window)
        day_of_week = end_timestamp.dayofweek  # 0=Monday
        day_of_month = end_timestamp.day
        day_of_year = end_timestamp.dayofyear
        
        return {
            'bar_in_day': bar_in_day,
            'day_of_week': np.int32(day_of_week),
            'day_of_month': np.int32(day_of_month),
            'day_of_year': np.int32(day_of_year)
        }
    
    def create_splits(
        self,
        df: pd.DataFrame,
        train_end: str = "2021-12-31",
        val_end: str = "2023-12-31",
        purge_hours: int = 24
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-based train/val/test splits with purge gaps.
        
        Implements clean temporal separation by:
        1. Including full days in each split (end-of-day timestamps)
        2. Inserting purge periods between splits to prevent lookback overlap
        
        Split structure with purge_hours=24:
        - Train: start to 2021-12-31 23:59:59
        - Purge: 2022-01-01 00:00:00 to 2022-01-01 23:59:59 (dropped)
        - Val:   2022-01-02 00:00:00 to 2023-12-31 23:59:59
        - Purge: 2024-01-01 00:00:00 to 2024-01-01 23:59:59 (dropped)
        - Test:  2024-01-02 00:00:00 to end
        
        This ensures validation window at 2022-01-02 00:00 (looking back 24h
        to 2022-01-01 00:00) doesn't overlap with train data (ends 2021-12-31).
        
        Args:
            df: Feature DataFrame with datetime index (UTC)
            train_end: End date for training set (inclusive, end-of-day)
            val_end: End date for validation set (inclusive, end-of-day)
            purge_hours: Hours to drop between splits (default 24)
            
        Returns:
            train_df, val_df, test_df
        """
        # Convert to end-of-day UTC timestamps for full day inclusion
        train_end_ts = pd.Timestamp(f"{train_end} 23:59:59").tz_localize('UTC')
        val_end_ts = pd.Timestamp(f"{val_end} 23:59:59").tz_localize('UTC')
        
        # Calculate purge boundaries
        val_start_ts = train_end_ts + pd.Timedelta(hours=purge_hours)
        test_start_ts = val_end_ts + pd.Timedelta(hours=purge_hours)
        
        # Create splits with purge gaps
        train_df = df[df.index <= train_end_ts].copy()
        val_df = df[(df.index >= val_start_ts) & (df.index <= val_end_ts)].copy()
        test_df = df[df.index >= test_start_ts].copy()
        
        # Validate splits have data
        if len(train_df) == 0:
            raise ValueError(f"Training set is empty (end={train_end_ts})")
        if len(val_df) == 0:
            raise ValueError(f"Validation set is empty (start={val_start_ts}, end={val_end_ts})")
        if len(test_df) == 0:
            raise ValueError(f"Test set is empty (start={test_start_ts})")
        
        print(f"Split statistics:")
        print(f"  Train: {len(train_df):,} samples ({train_df.index.min().date()} to {train_df.index.max().date()})")
        print(f"  Val:   {len(val_df):,} samples ({val_df.index.min().date()} to {val_df.index.max().date()})")
        print(f"  Test:  {len(test_df):,} samples ({test_df.index.min().date()} to {test_df.index.max().date()})")
        
        # Verify gaps
        train_val_gap_hours = (val_df.index.min() - train_df.index.max()).total_seconds() / 3600
        val_test_gap_hours = (test_df.index.min() - val_df.index.max()).total_seconds() / 3600
        
        print(f"\nTemporal gaps:")
        print(f"  Train-Val gap: {train_val_gap_hours:.1f} hours")
        print(f"  Val-Test gap: {val_test_gap_hours:.1f} hours")
        print(f"  Purged samples: ~{int(purge_hours * 12 * 2)} total (~{purge_hours*12} per gap)")
        
        return train_df, val_df, test_df
    
    def validate_no_leakage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tolerance_hours: int = 24
    ) -> bool:
        """
        Validate temporal separation between splits.
        
        Ensures minimum gap between last bar of one split and first bar of
        next split. This prevents feature lookback overlap (e.g., if a val
        sample looks back 24 hours, it shouldn't overlap with train data).
        
        Args:
            train_df: Training split
            val_df: Validation split
            test_df: Test split
            tolerance_hours: Minimum required gap (hours)
            
        Returns:
            True if validation passes
            
        Raises:
            AssertionError if gaps are insufficient
        """
        train_end = train_df.index.max()
        val_start = val_df.index.min()
        val_end = val_df.index.max()
        test_start = test_df.index.min()
        
        # Calculate gaps in hours
        train_val_gap = (val_start - train_end).total_seconds() / 3600
        val_test_gap = (test_start - val_end).total_seconds() / 3600
        
        assert train_val_gap >= tolerance_hours, (
            f"Train-Val gap ({train_val_gap:.1f}h) < tolerance ({tolerance_hours}h)"
        )
        
        assert val_test_gap >= tolerance_hours, (
            f"Val-Test gap ({val_test_gap:.1f}h) < tolerance ({tolerance_hours}h)"
        )
        
        print(f"[PASS] No data leakage detected:")
        print(f"  Train-Val gap: {train_val_gap:.1f} hours")
        print(f"  Val-Test gap: {val_test_gap:.1f} hours")
        
        return True
