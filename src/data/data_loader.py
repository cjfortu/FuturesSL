"""
Data loading and aggregation module for NQ futures.

This module handles loading raw 1-minute OHLCV data and aggregating it to
5-minute bars following standard financial time series conventions.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


class DataLoader:
    """
    Load and aggregate NQ futures OHLCV data.
    
    Handles 1-minute to 5-minute bar aggregation with proper OHLCV rules:
    - Open: first value in period
    - High: maximum value in period
    - Low: minimum value in period
    - Close: last value in period
    - Volume: sum of volume in period
    """
    
    def __init__(self, raw_dir: Path, interim_dir: Path):
        """
        Initialize DataLoader with directory paths.
        
        Args:
            raw_dir: Path to raw data directory containing 1-min data
                Type: pathlib.Path
            interim_dir: Path to interim data directory for output
                Type: pathlib.Path
        """
        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)
        
        # Ensure directories exist
        self.interim_dir.mkdir(parents=True, exist_ok=True)
    
    def load_1min_data(self, filename: str = "nq_ohlcv_1m_raw.parquet") -> pd.DataFrame:
        """
        Load 1-minute OHLCV data from parquet file.
        
        Args:
            filename: Name of parquet file in raw_dir
                Type: str
                Default: "nq_ohlcv_1m_raw.parquet"
        
        Returns:
            DataFrame with 1-minute OHLCV data
                Type: pandas.DataFrame
                Columns: timestamp (index), open, high, low, close, volume
                Index: DatetimeIndex in UTC
                Shape: (n_bars, 5) where n_bars varies by date range
        """
        filepath = self.raw_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Load parquet file
        df = pd.read_parquet(filepath)
        
        # Ensure timestamp is datetime and set as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Ensure UTC timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        elif df.index.tz.zone != 'UTC':
            df.index = df.index.tz_convert('UTC')
        
        # Verify required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by timestamp to ensure chronological order
        df = df.sort_index()
        
        return df
    
    def aggregate_to_5min(
        self, 
        df_1min: pd.DataFrame,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate 1-minute bars to 5-minute bars.
        
        Uses standard OHLCV aggregation rules:
        - open: first value in 5-min period
        - high: maximum value in 5-min period
        - low: minimum value in 5-min period
        - close: last value in 5-min period
        - volume: sum of volume in 5-min period
        
        5-minute periods are aligned to clock times (00:00, 00:05, 00:10, etc.)
        to ensure consistency across trading sessions.
        
        Args:
            df_1min: DataFrame with 1-minute OHLCV data
                Type: pandas.DataFrame
                Columns: open, high, low, close, volume
                Index: DatetimeIndex
                Shape: (n_bars, 5)
            validate: Whether to run validation checks on output
                Type: bool
                Default: True
        
        Returns:
            DataFrame with 5-minute OHLCV data
                Type: pandas.DataFrame
                Columns: open, high, low, close, volume
                Index: DatetimeIndex in UTC
                Shape: (n_bars_5min, 5) where n_bars_5min ≈ n_bars / 5
        """
        # Define aggregation rules for each column
        # These follow standard financial time series conventions
        agg_rules = {
            'open': 'first',    # Opening price is first trade in period
            'high': 'max',      # High is maximum price in period
            'low': 'min',       # Low is minimum price in period
            'close': 'last',    # Close is last trade in period
            'volume': 'sum'     # Volume is total traded in period
        }
        
        # Resample to 5-minute frequency
        # '5min' aligns to clock times (00:00, 00:05, 00:10, etc.)
        # label='left' labels each bar with its start time
        # closed='left' includes the left boundary, excludes right
        df_5min = df_1min.resample('5min', label='left', closed='left').agg(agg_rules)
        
        # Remove bars with NaN values (can occur at data gaps)
        # These typically appear at trading halts or between sessions
        df_5min = df_5min.dropna()
        
        # Validation checks
        if validate:
            self._validate_aggregation(df_1min, df_5min)
        
        return df_5min
    
    def _validate_aggregation(
        self, 
        df_1min: pd.DataFrame, 
        df_5min: pd.DataFrame
    ) -> None:
        """
        Validate 5-minute aggregation results.
        
        Checks:
        1. High >= Low for all bars (OHLC constraint)
        2. High >= Open and High >= Close (OHLC constraint)
        3. Low <= Open and Low <= Close (OHLC constraint)
        4. Volume is non-negative
        5. No price values are zero (invalid ticks)
        6. Expected number of 5-min bars (approx 1/5 of 1-min bars)
        
        Args:
            df_1min: Original 1-minute data
                Type: pandas.DataFrame
                Shape: (n_bars_1min, 5)
            df_5min: Aggregated 5-minute data
                Type: pandas.DataFrame
                Shape: (n_bars_5min, 5)
        
        Raises:
            ValueError: If any validation check fails
        """
        # Check OHLC constraints
        if not (df_5min['high'] >= df_5min['low']).all():
            raise ValueError("Validation failed: High < Low in some bars")
        
        if not (df_5min['high'] >= df_5min['open']).all():
            raise ValueError("Validation failed: High < Open in some bars")
        
        if not (df_5min['high'] >= df_5min['close']).all():
            raise ValueError("Validation failed: High < Close in some bars")
        
        if not (df_5min['low'] <= df_5min['open']).all():
            raise ValueError("Validation failed: Low > Open in some bars")
        
        if not (df_5min['low'] <= df_5min['close']).all():
            raise ValueError("Validation failed: Low > Close in some bars")
        
        # Check for zero or negative prices (invalid ticks)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df_5min[col] <= 0).any():
                raise ValueError(f"Validation failed: Zero or negative {col} prices found")
        
        # Check volume is non-negative
        if (df_5min['volume'] < 0).any():
            raise ValueError("Validation failed: Negative volume found")
        
        # Check expected bar count ratio
        # Should be approximately 1:5 ratio (1-min to 5-min)
        # Allow some tolerance for gaps in trading
        expected_ratio = len(df_1min) / len(df_5min)
        if not (4.0 <= expected_ratio <= 6.0):
            print(
                f"Warning: Unexpected bar count ratio {expected_ratio:.2f} "
                f"(expected ~5.0). 1-min bars: {len(df_1min)}, 5-min bars: {len(df_5min)}. "
                f"This may indicate data gaps or incomplete trading sessions."
            )
    
    def filter_invalid_ticks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove invalid ticks from data.
        
        Invalid ticks are defined as:
        - Zero price in any OHLC field
        - Zero volume
        - Negative values in any field
        
        These typically occur due to data errors or trading halts
        and should be removed before analysis.
        
        Args:
            df: OHLCV DataFrame
                Type: pandas.DataFrame
                Columns: open, high, low, close, volume
                Shape: (n_bars, 5)
        
        Returns:
            Filtered DataFrame with invalid ticks removed
                Type: pandas.DataFrame
                Shape: (n_valid_bars, 5) where n_valid_bars <= n_bars
        """
        # Count initial rows for reporting
        initial_count = len(df)
        
        # Remove rows with zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df = df[df[col] > 0]
        
        # Remove rows with zero volume
        df = df[df['volume'] > 0]
        
        # Log filtering results
        removed_count = initial_count - len(df)
        if removed_count > 0:
            print(f"Filtered {removed_count} invalid ticks ({removed_count/initial_count*100:.2f}%)")
        
        return df
    
    def detect_trading_halts(
        self, 
        df: pd.DataFrame,
        max_gap_minutes: int = 15
    ) -> pd.DataFrame:
        """
        Detect and mark trading halts in the data.
        
        Trading halts are identified as gaps larger than max_gap_minutes
        between consecutive bars. These periods should be handled carefully
        in feature engineering to avoid lookahead bias.
        
        Args:
            df: OHLCV DataFrame with DatetimeIndex
                Type: pandas.DataFrame
                Shape: (n_bars, 5)
            max_gap_minutes: Maximum normal gap in minutes
                Type: int
                Default: 15 (for 5-min bars, 3 periods = 15 min is acceptable)
        
        Returns:
            DataFrame with additional 'is_post_halt' column
                Type: pandas.DataFrame
                Columns: open, high, low, close, volume, is_post_halt
                Shape: (n_bars, 6)
                is_post_halt: bool indicating if bar follows a trading halt
        """
        # Calculate time differences between consecutive bars
        time_diffs = df.index.to_series().diff()
        
        # Convert to minutes
        time_diffs_minutes = time_diffs.dt.total_seconds() / 60
        
        # Mark bars that follow gaps larger than threshold
        # These bars should be treated carefully in feature computation
        df = df.copy()
        df['is_post_halt'] = time_diffs_minutes > max_gap_minutes
        
        # Count and report halts
        halt_count = df['is_post_halt'].sum()
        if halt_count > 0:
            print(f"Detected {halt_count} trading halts (gaps > {max_gap_minutes} min)")
        
        return df
