"""
NQ Futures Feature Engineering Module
=====================================

This module computes derived technical features from raw OHLCV data for the
NQ futures prediction model. It implements the 24-feature specification from
grok-scientific.md Section 3.1, organized into 6 semantic groups for
Two-Stage Attention (TSA) processing.

Feature Groups (per grok-scientific.md):
    F_P (Price Dynamics):    log_return, hl_range, close_location, open_return
    F_V (Volume):            log_volume, log_volume_delta, dollar_volume
    F_T (Trend):             vwap_deviation, macd_histogram, sma_20/50/200_dev
    F_M (Momentum):          rsi_norm, cci, plus_di, minus_di, adx, roc_norm
    F_σ (Volatility):        atr_norm, bb_pct_b, bb_bandwidth, realized_vol
    F_VW (Volume-Weighted):  mfi_norm, time_gap

Key Design Decisions:
- VWAP resets at RTH open (09:30 ET) per grok-scientific.md Section 3.1
- All features are normalized/bounded for numerical stability
- MFI uses typical price direction (not close price) per standard definition
- Gap feature uses log transform: ln(1 + max(Δt_minutes - 1, 0))

References:
    - grok-scientific.md Section 3.1: Feature specification (V=24)
    - gemini-research.md Section 4: Feature engineering insights
    - claude-engineering.md Section 3.1.3: Implementation details

Authors: Claude (Engineering Lead), Gemini (Research), Grok (Scientific)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


# Feature column names in canonical order (V=24 per grok-scientific.md)
FEATURE_COLUMNS: List[str] = [
    # Price dynamics (F_P) - 4 features
    'log_return', 'hl_range', 'close_location', 'open_return',
    # Volume (F_V) - 3 features
    'log_volume', 'log_volume_delta', 'dollar_volume',
    # Trend (F_T) - 5 features
    'vwap_deviation', 'macd_histogram', 'sma_20_dev', 'sma_50_dev', 'sma_200_dev',
    # Momentum (F_M) - 6 features
    'rsi_norm', 'cci', 'plus_di', 'minus_di', 'adx', 'roc_norm',
    # Volatility (F_σ) - 4 features
    'atr_norm', 'bb_pct_b', 'bb_bandwidth', 'realized_vol',
    # Volume-weighted (F_VW) - 1 feature + Temporal - 1 feature
    'mfi_norm', 'time_gap',
]

# Feature groups for TSA variable attention (per grok-scientific.md Section 3.4)
FEATURE_GROUPS: Dict[str, List[str]] = {
    'price': ['log_return', 'hl_range', 'close_location', 'open_return'],
    'volume': ['log_volume', 'log_volume_delta', 'dollar_volume'],
    'trend': ['vwap_deviation', 'macd_histogram', 'sma_20_dev', 'sma_50_dev', 'sma_200_dev'],
    'momentum': ['rsi_norm', 'cci', 'plus_di', 'minus_di', 'adx', 'roc_norm'],
    'volatility': ['atr_norm', 'bb_pct_b', 'bb_bandwidth', 'realized_vol'],
    'flow': ['mfi_norm', 'time_gap'],
}

# Target horizons in minutes (per problem statement)
TARGET_HORIZONS: List[int] = [5, 15, 30, 60, 120, 240]


class FeatureEngineer:
    """
    Computes derived technical features from OHLCV data.
    
    This class implements the complete feature engineering pipeline specified
    in grok-scientific.md Section 3.1. Features are computed using standard
    technical analysis formulas with modifications for ML consumption:
    - Normalization to bounded ranges where appropriate
    - Log transforms for scale-invariant measures
    - Session-aware VWAP with RTH reset
    
    Attributes:
        rsi_period: Lookback period for RSI calculation.
        atr_period: Lookback period for ATR calculation.
        bb_period: Lookback period for Bollinger Bands.
        bb_std: Standard deviation multiplier for Bollinger Bands.
        macd_fast: Fast EMA period for MACD.
        macd_slow: Slow EMA period for MACD.
        macd_signal: Signal line EMA period for MACD.
        sma_periods: List of SMA periods for trend features.
        cci_period: Lookback period for CCI.
        adx_period: Lookback period for DMI/ADX.
        mfi_period: Lookback period for Money Flow Index.
        roc_period: Lookback period for Rate of Change.
        realized_vol_period: Lookback period for realized volatility.
        rth_open_hour: RTH session start hour (Eastern Time).
        rth_open_minute: RTH session start minute.
    
    Example:
        >>> fe = FeatureEngineer()
        >>> features = fe.compute_all_features(ohlcv_df)
        >>> targets = fe.compute_targets(ohlcv_df)
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        atr_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        sma_periods: Optional[List[int]] = None,
        cci_period: int = 20,
        adx_period: int = 14,
        mfi_period: int = 14,
        roc_period: int = 10,
        realized_vol_period: int = 20,
        rth_open_hour: int = 9,
        rth_open_minute: int = 30
    ):
        """
        Initialize the feature engineer with indicator parameters.
        
        All parameters use standard technical analysis defaults unless
        specified otherwise in the scientific documentation.
        
        Args:
            rsi_period: RSI lookback period.
                Type: int, Default: 14
            atr_period: ATR lookback period.
                Type: int, Default: 14
            bb_period: Bollinger Bands lookback period.
                Type: int, Default: 20
            bb_std: Bollinger Bands standard deviation multiplier.
                Type: float, Default: 2.0
            macd_fast: MACD fast EMA period.
                Type: int, Default: 12
            macd_slow: MACD slow EMA period.
                Type: int, Default: 26
            macd_signal: MACD signal line EMA period.
                Type: int, Default: 9
            sma_periods: List of SMA periods for deviation features.
                Type: Optional[List[int]], Default: [20, 50, 200]
            cci_period: CCI lookback period.
                Type: int, Default: 20
            adx_period: ADX/DMI lookback period.
                Type: int, Default: 14
            mfi_period: Money Flow Index lookback period.
                Type: int, Default: 14
            roc_period: Rate of Change lookback period.
                Type: int, Default: 10
            realized_vol_period: Realized volatility lookback period.
                Type: int, Default: 20
            rth_open_hour: RTH session start hour (ET).
                Type: int, Default: 9
            rth_open_minute: RTH session start minute.
                Type: int, Default: 30
        """
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.sma_periods = sma_periods or [20, 50, 200]
        self.cci_period = cci_period
        self.adx_period = adx_period
        self.mfi_period = mfi_period
        self.roc_period = roc_period
        self.realized_vol_period = realized_vol_period
        self.rth_open_hour = rth_open_hour
        self.rth_open_minute = rth_open_minute
        
        # Minimum warmup period needed before features are valid
        self.warmup_period = max(
            self.macd_slow + self.macd_signal,
            max(self.sma_periods),
            self.adx_period * 2,
            self.realized_vol_period
        )
        
        logger.info(
            f"FeatureEngineer initialized. Warmup period: {self.warmup_period} bars"
        )
    
    def compute_all_features(
        self,
        df: pd.DataFrame,
        include_ohlcv: bool = True
    ) -> pd.DataFrame:
        """
        Compute all 24 derived features from OHLCV data.
        
        Processes the complete feature pipeline:
        1. Price dynamics (4 features)
        2. Volume features (3 features)
        3. Trend indicators (5 features)
        4. Momentum indicators (6 features)
        5. Volatility measures (4 features)
        6. Volume-weighted + temporal (2 features)
        
        Args:
            df: DataFrame with raw OHLCV data.
                Type: pd.DataFrame
                Required columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                Shape: (num_bars, 6)
            include_ohlcv: Whether to include raw OHLCV in output.
                Type: bool
                Default: True (needed for target computation)
        
        Returns:
            DataFrame with all computed features.
            Type: pd.DataFrame
            Columns: FEATURE_COLUMNS + optionally ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            Shape: (num_bars, 24 + optional 6)
        
        Raises:
            ValueError: If required columns are missing from input.
        """
        # Validate input
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Ensure proper types and sorting
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Computing features for {len(df):,} bars...")
        
        # Initialize output DataFrame
        features = pd.DataFrame(index=df.index)
        features['timestamp'] = df['timestamp']
        
        # Compute each feature group
        features = self._compute_price_dynamics(df, features)
        features = self._compute_volume_features(df, features)
        features = self._compute_trend_features(df, features)
        features = self._compute_momentum_features(df, features)
        features = self._compute_volatility_features(df, features)
        features = self._compute_volume_weighted_features(df, features)
        features = self._compute_gap_feature(df, features)
        
        # Optionally include raw OHLCV
        if include_ohlcv:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                features[col] = df[col]
        
        # Log feature statistics
        nan_counts = features[FEATURE_COLUMNS].isna().sum()
        nan_rate = nan_counts.sum() / (len(features) * len(FEATURE_COLUMNS))
        logger.info(
            f"Feature computation complete. NaN rate: {100*nan_rate:.2f}% "
            f"(expected high in warmup period)"
        )
        
        return features
    
    def _compute_price_dynamics(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute F_P: Price dynamics features (4 features).
        
        Features:
        - log_return: Log return from previous close
        - hl_range: High-low range normalized by close
        - close_location: Close position within bar (0=low, 1=high)
        - open_return: Intrabar return (open to close)
        
        Args:
            df: Raw OHLCV DataFrame.
                Type: pd.DataFrame
            features: Output DataFrame to populate.
                Type: pd.DataFrame
        
        Returns:
            Updated features DataFrame.
            Type: pd.DataFrame
        """
        # Log-returns: primary stationarity transform
        # Using log(P_t / P_{t-1}) per gemini-research.md Section 2.1
        features['log_return'] = np.log(
            df['close'] / df['close'].shift(1)
        )
        
        # High-low range normalized by close price
        # Measures intrabar volatility as fraction of price
        features['hl_range'] = (df['high'] - df['low']) / df['close']
        
        # Close location within bar: 0 = closed at low, 1 = closed at high
        # Indicates buying/selling pressure within bar
        hl_diff = df['high'] - df['low']
        features['close_location'] = np.where(
            hl_diff > 0,
            (df['close'] - df['low']) / hl_diff,
            0.5  # Doji bar (high == low): neutral
        )
        
        # Intrabar return: open to close
        # Captures momentum within single bar
        features['open_return'] = np.log(df['close'] / df['open'])
        
        return features
    
    def _compute_volume_features(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute F_V: Volume features (3 features).
        
        Features:
        - log_volume: Log-transformed volume (handles zero via log1p)
        - log_volume_delta: Change in log volume from previous bar
        - dollar_volume: Log of (close * volume) as liquidity proxy
        
        Args:
            df: Raw OHLCV DataFrame.
                Type: pd.DataFrame
            features: Output DataFrame to populate.
                Type: pd.DataFrame
        
        Returns:
            Updated features DataFrame.
            Type: pd.DataFrame
        """
        # Log volume: log1p handles zero volume bars
        features['log_volume'] = np.log1p(df['volume'])
        
        # Log volume delta: rate of change in volume
        features['log_volume_delta'] = (
            features['log_volume'] - features['log_volume'].shift(1)
        )
        
        # Dollar volume: proxy for liquidity/participation
        features['dollar_volume'] = np.log1p(df['close'] * df['volume'])
        
        return features
    
    def _compute_trend_features(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute F_T: Trend features (5 features).
        
        Features:
        - vwap_deviation: Price deviation from session VWAP (resets at RTH open)
        - macd_histogram: MACD histogram normalized by price
        - sma_20_dev: Log deviation from 20-bar SMA
        - sma_50_dev: Log deviation from 50-bar SMA
        - sma_200_dev: Log deviation from 200-bar SMA
        
        VWAP Implementation Note:
            Per grok-scientific.md Section 3.1, VWAP resets at RTH open (09:30 ET)
            to capture intraday session dynamics. This differs from a rolling VWAP.
        
        Args:
            df: Raw OHLCV DataFrame.
                Type: pd.DataFrame
            features: Output DataFrame to populate.
                Type: pd.DataFrame
        
        Returns:
            Updated features DataFrame.
            Type: pd.DataFrame
        """
        # VWAP deviation with session reset
        features['vwap_deviation'] = self._compute_vwap_deviation(df)
        
        # MACD histogram normalized by price
        # Per gemini-research.md: histogram provides stationary trend acceleration signal
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        features['macd_histogram'] = (macd_line - signal_line) / df['close']
        
        # SMA deviations: log(price / SMA) for stationarity
        # Per gemini-research.md Section 4.1: normalized relative to current close
        for period in self.sma_periods:
            sma = df['close'].rolling(window=period).mean()
            features[f'sma_{period}_dev'] = np.log(df['close'] / sma)
        
        return features
    
    def _compute_vwap_deviation(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute VWAP deviation with reset at RTH open (09:30 ET).
        
        Per grok-scientific.md Section 3.1:
        "VWAP deviation with reset at RTH open (09:30 ET) to capture session dynamics"
        
        This implementation:
        1. Identifies RTH session boundaries at 09:30 ET
        2. Computes cumulative typical price * volume within each session
        3. Divides by cumulative volume to get session VWAP
        4. Returns deviation as (close - VWAP) / close
        
        Args:
            df: Raw OHLCV DataFrame with timestamp column.
                Type: pd.DataFrame
        
        Returns:
            Series of VWAP deviations.
            Type: pd.Series
            Range: Typically (-0.05, 0.05) for NQ
        """
        df = df.copy()
        
        # Extract time components for session identification
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        # Identify RTH session starts (09:30 ET)
        # Session ID increments at each RTH open
        df['is_rth_start'] = (
            (df['hour'] == self.rth_open_hour) &
            (df['minute'] >= self.rth_open_minute) &
            (df['minute'] < self.rth_open_minute + 1)
        )
        df['session_id'] = df['is_rth_start'].cumsum()
        
        # Typical price for VWAP calculation
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_volume'] = df['typical_price'] * df['volume']
        
        # Cumulative sums within each session
        df['cum_tp_volume'] = df.groupby('session_id')['tp_volume'].cumsum()
        df['cum_volume'] = df.groupby('session_id')['volume'].cumsum()
        
        # VWAP: cumulative(TP * Volume) / cumulative(Volume)
        vwap = np.where(
            df['cum_volume'] > 0,
            df['cum_tp_volume'] / df['cum_volume'],
            df['close']  # Fallback for zero volume
        )
        
        # Deviation as percentage of price
        vwap_deviation = (df['close'].values - vwap) / df['close'].values
        
        return pd.Series(vwap_deviation, index=df.index)
    
    def _compute_momentum_features(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute F_M: Momentum features (6 features).
        
        Features:
        - rsi_norm: RSI normalized to [-1, 1] range
        - cci: Commodity Channel Index scaled by 100
        - plus_di: Positive Directional Indicator scaled to [0, 1]
        - minus_di: Negative Directional Indicator scaled to [0, 1]
        - adx: Average Directional Index scaled to [0, 1]
        - roc_norm: Rate of Change (10-bar)
        
        Args:
            df: Raw OHLCV DataFrame.
                Type: pd.DataFrame
            features: Output DataFrame to populate.
                Type: pd.DataFrame
        
        Returns:
            Updated features DataFrame.
            Type: pd.DataFrame
        """
        # RSI normalized to [-1, 1]
        # Per grok-scientific.md: (RSI - 50) / 50
        rsi = self._compute_rsi(df['close'], self.rsi_period)
        features['rsi_norm'] = (rsi - 50) / 50
        
        # CCI scaled down for numerical stability
        features['cci'] = self._compute_cci(df, self.cci_period) / 100
        
        # DMI/ADX components
        # Per gemini-research.md: distinguishes buying vs selling pressure
        plus_di, minus_di, adx = self._compute_dmi_adx(df, self.adx_period)
        features['plus_di'] = plus_di / 100
        features['minus_di'] = minus_di / 100
        features['adx'] = adx / 100
        
        # Rate of Change (simple momentum)
        features['roc_norm'] = df['close'] / df['close'].shift(self.roc_period) - 1
        
        return features
    
    def _compute_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Compute Relative Strength Index.
        
        Uses Wilder's smoothing method (exponential moving average with
        alpha = 1/period) for consistency with standard RSI implementations.
        
        Args:
            prices: Close price series.
                Type: pd.Series
            period: Lookback period.
                Type: int
        
        Returns:
            RSI values in range [0, 100].
            Type: pd.Series
        """
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        # Wilder's smoothing (EMA with alpha = 1/period)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        # RS and RSI
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _compute_cci(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Compute Commodity Channel Index.
        
        CCI = (Typical Price - SMA(TP)) / (0.015 * Mean Absolute Deviation)
        
        The 0.015 constant scales CCI so that ~70-80% of values fall
        within the [-100, +100] range.
        
        Args:
            df: OHLCV DataFrame.
                Type: pd.DataFrame
            period: Lookback period.
                Type: int
        
        Returns:
            CCI values (unbounded, typically in [-200, 200]).
            Type: pd.Series
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        
        # Mean absolute deviation
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(),
            raw=True
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mad.replace(0, np.finfo(float).eps))
        
        return cci
    
    def _compute_dmi_adx(
        self,
        df: pd.DataFrame,
        period: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute Directional Movement Index and Average Directional Index.
        
        Per gemini-research.md Section 4.1:
        "DMI deconstructs trend into Positive (+DI) and Negative (-DI)
        components, vital for the Variable Attention module to distinguish
        buying pressure from selling pressure."
        
        Args:
            df: OHLCV DataFrame.
                Type: pd.DataFrame
            period: Lookback period.
                Type: int
        
        Returns:
            Tuple of (+DI, -DI, ADX) as Series.
            Type: Tuple[pd.Series, pd.Series, pd.Series]
            Each in range [0, 100]
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed averages (Wilder's method)
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False).mean()
        
        # Directional Indicators
        eps = np.finfo(float).eps
        plus_di = 100 * plus_dm_smooth / atr.replace(0, eps)
        minus_di = 100 * minus_dm_smooth / atr.replace(0, eps)
        
        # ADX: smoothed absolute difference ratio
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        dx = 100 * di_diff / di_sum.replace(0, eps)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return plus_di, minus_di, adx
    
    def _compute_volatility_features(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute F_σ: Volatility features (4 features).
        
        Features:
        - atr_norm: ATR normalized by close price
        - bb_pct_b: Bollinger %B (price position within bands)
        - bb_bandwidth: Bollinger bandwidth (band width / middle band)
        - realized_vol: Annualized realized volatility
        
        Args:
            df: Raw OHLCV DataFrame.
                Type: pd.DataFrame
            features: Output DataFrame to populate.
                Type: pd.DataFrame
        
        Returns:
            Updated features DataFrame.
            Type: pd.DataFrame
        """
        # ATR normalized by price
        atr = self._compute_atr(df, self.atr_period)
        features['atr_norm'] = atr / df['close']
        
        # Bollinger Bands %B and Bandwidth
        sma = df['close'].rolling(window=self.bb_period).mean()
        std = df['close'].rolling(window=self.bb_period).std()
        
        upper_band = sma + self.bb_std * std
        lower_band = sma - self.bb_std * std
        band_diff = upper_band - lower_band
        
        # %B: position within bands (0 = lower, 1 = upper)
        # Per gemini-research.md: indicates volatility regime position
        features['bb_pct_b'] = np.where(
            band_diff > 0,
            (df['close'] - lower_band) / band_diff,
            0.5  # Undefined when bands collapse
        )
        
        # Bandwidth: volatility compression indicator
        # Per gemini-research.md: "squeeze" precursor to breakouts
        features['bb_bandwidth'] = band_diff / sma.replace(0, np.finfo(float).eps)
        
        # Realized volatility (annualized)
        # Using 252 trading days * 390 trading minutes per day
        log_returns = np.log(df['close'] / df['close'].shift(1))
        rolling_std = log_returns.rolling(window=self.realized_vol_period).std()
        features['realized_vol'] = rolling_std * np.sqrt(252 * 390)
        
        return features
    
    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Compute Average True Range.
        
        ATR measures volatility by capturing the full range of price
        movement including gaps from previous close.
        
        Args:
            df: OHLCV DataFrame.
                Type: pd.DataFrame
            period: Lookback period.
                Type: int
        
        Returns:
            ATR values.
            Type: pd.Series
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range: max of (H-L, |H-C_prev|, |L-C_prev|)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Average using Wilder's smoothing
        atr = tr.ewm(span=period, adjust=False).mean()
        
        return atr
    
    def _compute_volume_weighted_features(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute F_VW: Volume-weighted feature (1 feature).
        
        Features:
        - mfi_norm: Money Flow Index normalized to [-1, 1]
        
        MFI Implementation Note:
            MFI uses typical price direction (not close price direction)
            to determine positive vs negative money flow. This is the
            standard definition per gemini-research.md Section 4.2.
        
        Args:
            df: Raw OHLCV DataFrame.
                Type: pd.DataFrame
            features: Output DataFrame to populate.
                Type: pd.DataFrame
        
        Returns:
            Updated features DataFrame.
            Type: pd.DataFrame
        """
        # Money Flow Index
        mfi = self._compute_mfi(df, self.mfi_period)
        features['mfi_norm'] = (mfi - 50) / 50
        
        return features
    
    def _compute_mfi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Compute Money Flow Index.
        
        MFI is a volume-weighted RSI that uses typical price direction
        to classify money flow as positive or negative.
        
        Per gemini-research.md Section 4.2:
        "For NQ futures, volume is the fuel of price movement. Divergences
        between Price and MFI are high-probability reversal signals."
        
        Args:
            df: OHLCV DataFrame.
                Type: pd.DataFrame
            period: Lookback period.
                Type: int
        
        Returns:
            MFI values in range [0, 100].
            Type: pd.Series
        """
        # Typical price and raw money flow
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        
        # Direction based on typical price change (not close!)
        tp_diff = typical_price.diff()
        
        # Classify as positive or negative flow
        positive_flow = np.where(tp_diff > 0, raw_money_flow, 0)
        negative_flow = np.where(tp_diff < 0, raw_money_flow, 0)
        
        # Rolling sums
        positive_sum = pd.Series(positive_flow).rolling(window=period).sum()
        negative_sum = pd.Series(negative_flow).rolling(window=period).sum()
        
        # Money Flow Ratio and MFI
        money_ratio = positive_sum / negative_sum.replace(0, np.finfo(float).eps)
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    def _compute_gap_feature(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute temporal gap feature (1 feature).
        
        Features:
        - time_gap: Log-transformed time gap between bars
        
        Per grok-scientific.md Section 3.1:
        "Explicit gap feature Δt_i = ln(1 + ((timestamp_i - timestamp_{i-1}) / 60 - 1))"
        
        This captures:
        - Market closures (weekends, holidays): large positive values
        - Normal 1-minute intervals: 0
        - Helps model understand when time continuity breaks
        
        Args:
            df: Raw OHLCV DataFrame with timestamp column.
                Type: pd.DataFrame
            features: Output DataFrame to populate.
                Type: pd.DataFrame
        
        Returns:
            Updated features DataFrame.
            Type: pd.DataFrame
        """
        # Time difference in seconds, converted to minutes
        timestamps = pd.to_datetime(df['timestamp'])
        time_diff_seconds = timestamps.diff().dt.total_seconds()
        time_diff_minutes = time_diff_seconds / 60
        
        # Log-transformed gap: ln(1 + max(Δt - 1, 0))
        # This gives 0 for normal 1-minute intervals
        # Positive values for gaps > 1 minute
        features['time_gap'] = np.log1p(np.maximum(time_diff_minutes - 1, 0))
        
        return features
    
    def compute_targets(
        self,
        df: pd.DataFrame,
        horizons: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Compute forward log-return targets for multiple horizons.
        
        Per problem statement:
        "predict the NQ forward returns over multiple horizons (5m, 15m, 30m, 60m, 2h, 4h)"
        
        Args:
            df: DataFrame with close prices and timestamps.
                Type: pd.DataFrame
                Required columns: ['timestamp', 'close']
            horizons: List of forward horizons in minutes.
                Type: Optional[List[int]]
                Default: [5, 15, 30, 60, 120, 240]
        
        Returns:
            DataFrame with target columns.
            Type: pd.DataFrame
            Columns: ['timestamp', 'target_5m', 'target_15m', ...]
            Shape: (num_bars, 1 + len(horizons))
        """
        horizons = horizons or TARGET_HORIZONS
        
        targets = pd.DataFrame(index=df.index)
        targets['timestamp'] = df['timestamp']
        
        for h in horizons:
            # Forward log-return: ln(P_{t+h} / P_t)
            targets[f'target_{h}m'] = np.log(
                df['close'].shift(-h) / df['close']
            )
        
        logger.info(f"Computed targets for horizons: {horizons}")
        return targets
    
    @staticmethod
    def get_feature_columns() -> List[str]:
        """
        Return canonical list of feature column names.
        
        Returns:
            List of 24 feature column names in standard order.
            Type: List[str]
        """
        return FEATURE_COLUMNS.copy()
    
    @staticmethod
    def get_feature_groups() -> Dict[str, List[str]]:
        """
        Return feature groupings for TSA variable attention.
        
        Per grok-scientific.md Section 3.4:
        "Group projection: For each group, concatenate raw features,
        linearly project to dim d, yielding 6 group embeddings."
        
        Returns:
            Dictionary mapping group names to feature lists.
            Type: Dict[str, List[str]]
        """
        return {k: v.copy() for k, v in FEATURE_GROUPS.items()}
    
    @staticmethod
    def get_feature_indices() -> Dict[str, List[int]]:
        """
        Return feature group indices for model construction.
        
        Maps feature group names to indices in the FEATURE_COLUMNS array.
        Used by GroupProjection module in model architecture.
        
        Returns:
            Dictionary mapping group names to feature indices.
            Type: Dict[str, List[int]]
        """
        indices = {}
        for group_name, group_features in FEATURE_GROUPS.items():
            indices[group_name] = [
                FEATURE_COLUMNS.index(f) for f in group_features
            ]
        return indices


def compute_and_save_features(
    raw_data_path: str,
    output_dir: str,
    include_stats: bool = True
) -> Tuple[Path, Path, Optional[Path]]:
    """
    Convenience function to compute features from raw data and save to disk.
    
    This function orchestrates the complete feature engineering pipeline:
    1. Load raw OHLCV data from parquet
    2. Compute all 24 derived features
    3. Compute targets for all 6 horizons
    4. Save features, targets, and optionally normalization statistics
    
    Args:
        raw_data_path: Path to raw OHLCV parquet file.
            Type: str
        output_dir: Directory for output files.
            Type: str
        include_stats: Whether to compute and save feature statistics.
            Type: bool
            Default: True
    
    Returns:
        Tuple of (features_path, targets_path, stats_path or None).
        Type: Tuple[Path, Path, Optional[Path]]
    
    Example:
        >>> features_path, targets_path, stats_path = compute_and_save_features(
        ...     "data/raw/nq_ohlcv_1m_raw.parquet",
        ...     "data/processed"
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    logger.info(f"Loading raw data from {raw_data_path}")
    df = pd.read_parquet(raw_data_path)
    logger.info(f"Loaded {len(df):,} rows")
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Compute features and targets
    features = fe.compute_all_features(df, include_ohlcv=True)
    targets = fe.compute_targets(df)
    
    # Save features
    features_path = output_dir / 'nq_features_v1.parquet'
    features.to_parquet(features_path, engine='pyarrow', compression='snappy')
    logger.info(f"Saved features to {features_path}")
    
    # Save targets
    targets_path = output_dir / 'nq_targets_v1.parquet'
    targets.to_parquet(targets_path, engine='pyarrow', compression='snappy')
    logger.info(f"Saved targets to {targets_path}")
    
    # Compute and save statistics
    stats_path = None
    if include_stats:
        feature_cols = fe.get_feature_columns()
        stats = {
            'means': features[feature_cols].mean().tolist(),
            'stds': features[feature_cols].std().tolist(),
            'columns': feature_cols,
            'num_features': len(feature_cols),
            'feature_groups': fe.get_feature_groups(),
            'feature_indices': fe.get_feature_indices(),
            'warmup_period': fe.warmup_period,
        }
        
        stats_path = output_dir / 'feature_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved feature statistics to {stats_path}")
    
    return features_path, targets_path, stats_path
