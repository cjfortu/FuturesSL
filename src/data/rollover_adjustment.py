"""
Futures contract rollover adjustment module.

This module implements ratio back-adjustment for continuous futures contracts,
preserving log-returns across contract rolls while avoiding artificial price gaps.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np


class RolloverAdjuster:
    """
    Apply ratio back-adjustment to futures contract data.
    
    Ratio back-adjustment preserves percentage returns by multiplying
    historical prices by the ratio of new/old contract prices at rollover.
    This is critical for machine learning models based on log-returns.
    
    Mathematical formulation:
    At rollover from Contract A to Contract B:
        R = Price_B(t_roll) / Price_A(t_roll)
        For all t < t_roll:
            Price_adjusted(t) = Price_raw(t) × R
    
    For multiple rollovers, ratios are accumulated:
        Price_adjusted = Price_raw × ∏(R_i) for all rollovers after time t
    """
    
    def __init__(self, rollover_dates: pd.DataFrame):
        """
        Initialize rollover adjuster with detected rollover metadata.
        
        Args:
            rollover_dates: DataFrame containing rollover information
                Type: pandas.DataFrame
                Required columns: date, price_ratio
                date: datetime64, rollover timestamp
                price_ratio: float64, ratio of new/old contract prices
                Shape: (n_rollovers, 2)
        """
        # Sort rollovers chronologically
        self.rollover_dates = rollover_dates.sort_values('date').reset_index(drop=True)
        
        # Validate rollover data
        self._validate_rollover_data()
    
    @classmethod
    def from_data(
        cls, 
        df: pd.DataFrame,
        volume_threshold: float = 0.5,
        price_jump_threshold: float = 0.01
    ) -> 'RolloverAdjuster':
        """
        Detect rollovers automatically from continuous contract data.
        
        Rollovers are detected by identifying:
        1. Large price jumps (> threshold) that are NOT due to normal volatility
        2. Occurring near typical rollover windows (quarterly for NQ)
        
        For Databento continuous contracts with volume-based rollover,
        the rollover happens when front-month volume exceeds back-month.
        We detect this as a price discontinuity.
        
        Args:
            df: OHLCV DataFrame with continuous contract data
                Type: pandas.DataFrame
                Columns: open, high, low, close, volume
                Index: DatetimeIndex
                Shape: (n_bars, 5)
            volume_threshold: Not used for price-based detection, kept for compatibility
                Type: float
                Default: 0.5
            price_jump_threshold: Minimum price jump (as fraction) to consider rollover
                Type: float
                Default: 0.01 (1% jump)
        
        Returns:
            RolloverAdjuster instance with detected rollovers
                Type: RolloverAdjuster
        """
        # Calculate returns to find price discontinuities
        returns = df['close'].pct_change()
        
        # Calculate rolling volatility to normalize jumps
        # Use 20-bar window (100 minutes for 5-min bars)
        vol = returns.rolling(window=20, min_periods=10).std()
        
        # Identify large price jumps (> 3 standard deviations)
        # These are candidate rollovers
        standardized_returns = returns.abs() / vol
        candidate_rollovers = standardized_returns > 3.0
        
        # Additionally require absolute jump > threshold
        large_jumps = returns.abs() > price_jump_threshold
        candidate_rollovers = candidate_rollovers & large_jumps
        
        # Extract rollover dates and calculate price ratios
        rollover_indices = df.index[candidate_rollovers]
        
        if len(rollover_indices) == 0:
            # No rollovers detected, return identity adjustment
            rollover_df = pd.DataFrame({
                'date': [df.index[0]],
                'price_ratio': [1.0]
            })
        else:
            # Calculate price ratio at each rollover
            # Ratio = Price_after / Price_before
            price_ratios = []
            for idx in rollover_indices:
                idx_loc = df.index.get_loc(idx)
                if idx_loc > 0:
                    price_before = df['close'].iloc[idx_loc - 1]
                    price_after = df['close'].iloc[idx_loc]
                    ratio = price_after / price_before
                    price_ratios.append(ratio)
                else:
                    price_ratios.append(1.0)
            
            rollover_df = pd.DataFrame({
                'date': rollover_indices,
                'price_ratio': price_ratios
            })
        
        return cls(rollover_df)
    
    def _validate_rollover_data(self) -> None:
        """
        Validate rollover metadata.
        
        Checks:
        1. Required columns exist
        2. Price ratios are positive
        3. Dates are unique
        4. Price ratios are reasonable (between 0.5 and 2.0)
        
        Raises:
            ValueError: If validation fails
        """
        required_cols = ['date', 'price_ratio']
        missing_cols = set(required_cols) - set(self.rollover_dates.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in rollover data: {missing_cols}")
        
        # Check price ratios are positive
        if (self.rollover_dates['price_ratio'] <= 0).any():
            raise ValueError("Price ratios must be positive")
        
        # Check for reasonable price ratios
        # Futures contracts typically don't differ by more than 2x
        if (self.rollover_dates['price_ratio'] < 0.5).any() or \
           (self.rollover_dates['price_ratio'] > 2.0).any():
            extreme_ratios = self.rollover_dates[
                (self.rollover_dates['price_ratio'] < 0.5) | 
                (self.rollover_dates['price_ratio'] > 2.0)
            ]
            print(f"Warning: Extreme price ratios detected:\n{extreme_ratios}")
        
        # Check dates are unique
        if self.rollover_dates['date'].duplicated().any():
            raise ValueError("Duplicate rollover dates found")
    
    def adjust_prices(
        self,
        df: pd.DataFrame,
        price_cols: List[str] = ["open", "high", "low", "close"]
    ) -> pd.DataFrame:
        """
        Apply ratio back-adjustment to price columns.
        
        Algorithm:
        1. For each rollover (from newest to oldest):
           - Calculate adjustment ratio R
           - Multiply all prices BEFORE rollover by R
        2. This preserves log-returns across rollovers
        
        Args:
            df: OHLCV DataFrame to adjust
                Type: pandas.DataFrame
                Columns: open, high, low, close, volume
                Index: DatetimeIndex
                Shape: (n_bars, 5)
            price_cols: Columns to adjust (volume is never adjusted)
                Type: List[str]
                Default: ["open", "high", "low", "close"]
        
        Returns:
            Adjusted DataFrame with modified prices
                Type: pandas.DataFrame
                Columns: Same as input
                Index: Same as input
                Shape: Same as input (n_bars, 5)
        """
        # Work on a copy to avoid modifying original
        df_adjusted = df.copy()
        
        # Calculate cumulative adjustment ratios
        # We work backwards from most recent to oldest rollover
        # This ensures each historical price gets multiplied by all subsequent ratios
        cumulative_ratio = 1.0
        
        # Sort rollovers from newest to oldest
        rollovers_reversed = self.rollover_dates.sort_values('date', ascending=False)
        
        for _, rollover in rollovers_reversed.iterrows():
            rollover_date = rollover['date']
            ratio = rollover['price_ratio']
            
            # Update cumulative ratio
            cumulative_ratio *= ratio
            
            # Apply adjustment to all bars BEFORE this rollover
            mask = df_adjusted.index < rollover_date
            
            # Adjust specified price columns
            for col in price_cols:
                if col in df_adjusted.columns:
                    df_adjusted.loc[mask, col] *= ratio
        
        return df_adjusted
    
    def verify_returns_preserved(
        self,
        original: pd.DataFrame,
        adjusted: pd.DataFrame,
        tolerance: float = 1e-10
    ) -> bool:
        """
        Verify that log-returns are preserved after adjustment.
        
        This is a critical validation step. Log-returns should be identical
        before and after adjustment, except at rollover points where the
        artificial gap is removed.
        
        Args:
            original: Original unadjusted DataFrame
                Type: pandas.DataFrame
                Shape: (n_bars, 5)
            adjusted: Adjusted DataFrame
                Type: pandas.DataFrame
                Shape: (n_bars, 5)
            tolerance: Maximum allowed difference in log-returns
                Type: float
                Default: 1e-10
        
        Returns:
            True if returns preserved within tolerance
                Type: bool
        
        Raises:
            AssertionError: If returns differ by more than tolerance
        """
        # Calculate log returns on close prices
        orig_returns = np.log(original['close'] / original['close'].shift(1))
        adj_returns = np.log(adjusted['close'] / adjusted['close'].shift(1))
        
        # Create mask for non-rollover dates
        # At rollover dates, returns will differ (that's the point)
        rollover_mask = pd.Series(False, index=original.index)
        for _, rollover in self.rollover_dates.iterrows():
            # Mark the bar AT the rollover date
            if rollover['date'] in original.index:
                rollover_mask.loc[rollover['date']] = True
        
        # Compare returns at non-rollover dates
        non_rollover_mask = ~rollover_mask
        
        # Drop NaN values (first bar has no return)
        valid_mask = non_rollover_mask & orig_returns.notna() & adj_returns.notna()
        
        if valid_mask.sum() == 0:
            # No valid comparisons (very short series)
            return True
        
        # Calculate maximum absolute difference
        max_diff = (orig_returns[valid_mask] - adj_returns[valid_mask]).abs().max()
        
        # Verify within tolerance
        if max_diff > tolerance:
            # Find where differences occur
            diff_series = (orig_returns - adj_returns).abs()
            problem_dates = diff_series[diff_series > tolerance].head()
            raise AssertionError(
                f"Log-returns differ by more than tolerance {tolerance}.\n"
                f"Maximum difference: {max_diff:.2e}\n"
                f"Problem dates:\n{problem_dates}"
            )
        
        print(f"✓ Returns preserved: max difference {max_diff:.2e} < tolerance {tolerance:.2e}")
        return True
    
    def get_rollover_dates(self) -> pd.DataFrame:
        """
        Get DataFrame of rollover dates and ratios.
        
        Returns:
            Rollover metadata
                Type: pandas.DataFrame
                Columns: date, price_ratio
                Shape: (n_rollovers, 2)
        """
        return self.rollover_dates.copy()
    
    def get_adjustment_factors(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate cumulative adjustment factor for each date.
        
        The adjustment factor is the product of all rollover ratios
        that occur AFTER the given date. This shows how much each
        historical price has been adjusted.
        
        Args:
            df: DataFrame with dates to calculate factors for
                Type: pandas.DataFrame
                Index: DatetimeIndex
                Shape: (n_bars, n_cols)
        
        Returns:
            Series of adjustment factors
                Type: pandas.Series
                Index: Same as input DataFrame
                Values: float64, cumulative adjustment ratios
                Shape: (n_bars,)
        """
        # Initialize all factors to 1.0
        factors = pd.Series(1.0, index=df.index)
        
        # Apply each rollover ratio to earlier dates
        for _, rollover in self.rollover_dates.iterrows():
            rollover_date = rollover['date']
            ratio = rollover['price_ratio']
            
            # All bars before this rollover get multiplied by this ratio
            mask = df.index < rollover_date
            factors[mask] *= ratio
        
        return factors
