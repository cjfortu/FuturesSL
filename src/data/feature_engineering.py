"""
Feature engineering module for NQ futures.

This module computes all derived features from OHLCV data following the
scientific document specifications. All features are computed causally
(using only past data) to prevent lookahead bias.

Features computed:
- Volatility: Garman-Klass, realized volatility at multiple horizons
- Liquidity: Amihud illiquidity measure
- Momentum: RSI, MACD, rate of change
- Trend: EMA slopes and deviations
- Range: Average True Range
- Volume: Log-volume and volume moving average ratio
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path


class FeatureEngineer:
    """
    Compute derived features from OHLCV data.
    
    All features are computed using only historical data to ensure causality.
    The module handles NaN values generated during warmup periods and provides
    validation checks for feature quality.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config: Optional configuration dictionary
                Type: Dict or None
                Default: None (uses default parameters)
        """
        self.config = config or self._default_config()
        
    @staticmethod
    def _default_config() -> Dict:
        """
        Get default configuration for feature computation.
        
        Returns:
            Default configuration dictionary
                Type: Dict
                Contains window sizes and parameters for all features
        """
        return {
            'rv_windows': [12, 36, 72],  # Realized volatility windows (bars)
            'rsi_period': 14,              # RSI period
            'macd_fast': 12,               # MACD fast EMA
            'macd_slow': 26,               # MACD slow EMA
            'macd_signal': 9,              # MACD signal line
            'roc_periods': [5, 10, 20],    # Rate of change periods
            'ema_periods': [9, 21, 50],    # EMA periods for trend
            'atr_period': 14,              # ATR period
            'volume_ma_period': 20,        # Volume moving average period
        }
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all derived features from OHLCV data.
        
        This is the main entry point that orchestrates feature computation.
        Features are computed in a specific order to handle dependencies.
        
        Args:
            df: OHLCV DataFrame with DatetimeIndex
                Type: pandas.DataFrame
                Required columns: open, high, low, close, volume
                Index: DatetimeIndex
                Shape: (n_bars, 5)
        
        Returns:
            DataFrame with all features
                Type: pandas.DataFrame
                Columns: Original OHLCV + 20 derived features
                Shape: (n_bars, 25)
                
        Feature computation order ensures dependencies are satisfied:
        1. Basic transforms (log returns, log volume)
        2. Volatility measures (GK, realized vol)
        3. Liquidity measures (Amihud)
        4. Momentum indicators (RSI, MACD, ROC)
        5. Trend indicators (EMA slopes and deviations)
        6. Range indicators (ATR)
        7. Volume indicators (MA ratio)
        """
        # Work on a copy to preserve original data
        df_features = df.copy()
        
        # 1. Basic transforms
        df_features['log_return'] = self._compute_log_returns(df_features['close'])
        df_features['log_volume'] = self._compute_log_volume(df_features['volume'])
        
        # 2. Volatility features
        df_features['gk_volatility'] = self._compute_garman_klass(
            df_features['open'],
            df_features['high'],
            df_features['low'],
            df_features['close']
        )
        
        # Realized volatilities at multiple horizons
        for window in self.config['rv_windows']:
            df_features[f'rv_{window}'] = self._compute_realized_volatility(
                df_features['log_return'],
                window=window
            )
        
        # 3. Liquidity features
        df_features['amihud_illiq'] = self._compute_amihud_illiquidity(
            df_features['log_return'],
            df_features['close'],
            df_features['volume']
        )
        
        # 4. Momentum features
        df_features['rsi_14'] = self._compute_rsi(
            df_features['close'],
            period=self.config['rsi_period']
        )
        
        macd, macd_signal = self._compute_macd(
            df_features['close'],
            fast=self.config['macd_fast'],
            slow=self.config['macd_slow'],
            signal=self.config['macd_signal']
        )
        df_features['macd'] = macd
        df_features['macd_signal'] = macd_signal
        
        for period in self.config['roc_periods']:
            df_features[f'roc_{period}'] = self._compute_roc(
                df_features['close'],
                period=period
            )
        
        # 5. Trend features
        for period in self.config['ema_periods']:
            slope, deviation = self._compute_ema_features(
                df_features['close'],
                period=period
            )
            df_features[f'ema_slope_{period}'] = slope
            df_features[f'ema_dev_{period}'] = deviation
        
        # 6. Range features
        df_features['atr_14'] = self._compute_atr(
            df_features['high'],
            df_features['low'],
            df_features['close'],
            period=self.config['atr_period']
        )
        
        # 7. Volume features
        df_features['volume_ma_ratio'] = self._compute_volume_ma_ratio(
            df_features['volume'],
            period=self.config['volume_ma_period']
        )
        
        return df_features
    
    def compute_targets(
        self,
        df: pd.DataFrame,
        horizons: List[int] = [3, 6, 12, 24, 48]
    ) -> pd.DataFrame:
        """
        Compute forward log-return targets for each prediction horizon.
        
        Targets are computed as log(P_{t+h} / P_t) where h is the horizon
        in number of 5-minute bars. This represents the future return from
        the current bar to h bars ahead.
        
        Args:
            df: DataFrame with close prices
                Type: pandas.DataFrame
                Required column: close
                Shape: (n_bars, ...)
            horizons: List of horizons in number of 5-min bars
                Type: List[int]
                Default: [3, 6, 12, 24, 48] for [15m, 30m, 1h, 2h, 4h]
                Each horizon represents number of bars to look ahead
        
        Returns:
            DataFrame with target columns
                Type: pandas.DataFrame
                Columns: target_15m, target_30m, target_60m, target_2h, target_4h
                Shape: Same as input
                Note: Last h bars will be NaN for horizon h
        
        Mathematical formulation:
            target_{h} = log(close_{t+h} / close_t)
        
        Causality note:
            During training, we only use targets where they are available
            (i.e., we drop the last h bars for each horizon during dataset
            construction). This ensures no lookahead bias.
        """
        df_targets = df.copy()
        
        # Horizon names for clarity
        horizon_names = {
            3: '15m',
            6: '30m', 
            12: '60m',
            24: '2h',
            48: '4h'
        }
        
        for h in horizons:
            # Compute forward return: log(P_{t+h} / P_t)
            # shift(-h) gets future price, then compute log return
            future_price = df_targets['close'].shift(-h)
            current_price = df_targets['close']
            
            # Log return from current bar to h bars ahead
            target = np.log(future_price / current_price)
            
            # Store with descriptive name
            horizon_name = horizon_names.get(h, f'{h}bars')
            df_targets[f'target_{horizon_name}'] = target
        
        return df_targets
    
    # ==================== Feature Computation Methods ====================
    
    @staticmethod
    def _compute_log_returns(close: pd.Series) -> pd.Series:
        """
        Compute log returns: ln(P_t / P_{t-1}).
        
        Args:
            close: Close price series
                Type: pandas.Series
                Shape: (n_bars,)
        
        Returns:
            Log returns series
                Type: pandas.Series
                Shape: (n_bars,)
                First value is NaN (no prior bar)
        
        Log returns are additive across time and approximately equal to
        percentage returns for small changes. They are the fundamental
        building block for many financial features.
        """
        return np.log(close / close.shift(1))
    
    @staticmethod
    def _compute_log_volume(volume: pd.Series) -> pd.Series:
        """
        Compute log-transformed volume: ln(V_t + 1).
        
        Args:
            volume: Volume series
                Type: pandas.Series
                Shape: (n_bars,)
        
        Returns:
            Log volume series
                Type: pandas.Series
                Shape: (n_bars,)
        
        The +1 handles zero volume bars (rare but possible during
        trading halts). Log transformation stabilizes variance and
        makes volume more normally distributed.
        """
        return np.log(volume + 1)
    
    @staticmethod
    def _compute_garman_klass(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """
        Compute Garman-Klass volatility estimator.
        
        This efficient estimator uses all OHLC information to estimate
        volatility within a single bar, providing better estimates than
        close-to-close volatility.
        
        Formula:
            GK = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
        
        Args:
            open_: Open price series
                Type: pandas.Series
                Shape: (n_bars,)
            high: High price series
                Type: pandas.Series
                Shape: (n_bars,)
            low: Low price series
                Type: pandas.Series
                Shape: (n_bars,)
            close: Close price series
                Type: pandas.Series
                Shape: (n_bars,)
        
        Returns:
            Garman-Klass volatility series
                Type: pandas.Series
                Shape: (n_bars,)
                Units: Variance (squared returns)
        
        Reference:
            Garman, M. B., & Klass, M. J. (1980). On the estimation of
            security price volatilities from historical data. Journal of
            Business, 53(1), 67-78.
        """
        # Log price ratios
        ln_hl = np.log(high / low)
        ln_co = np.log(close / open_)
        
        # Garman-Klass formula
        # First term: 0.5 * (ln(H/L))^2
        # Second term: -(2*ln(2) - 1) * (ln(C/O))^2
        gk_vol = 0.5 * ln_hl**2 - (2*np.log(2) - 1) * ln_co**2
        
        return gk_vol
    
    @staticmethod
    def _compute_realized_volatility(
        log_returns: pd.Series,
        window: int
    ) -> pd.Series:
        """
        Compute realized volatility over a rolling window.
        
        Realized volatility is the square root of the sum of squared
        returns, providing a backward-looking estimate of volatility.
        
        Formula:
            RV_t = sqrt(sum_{i=t-window+1}^{t} r_i^2)
        
        Args:
            log_returns: Log return series
                Type: pandas.Series
                Shape: (n_bars,)
            window: Number of bars to include in calculation
                Type: int
                Example: 12 bars = 1 hour, 72 bars = 6 hours (for 5-min bars)
        
        Returns:
            Realized volatility series
                Type: pandas.Series
                Shape: (n_bars,)
                First (window-1) values are NaN
        
        The rolling window only uses past data, ensuring causality.
        Larger windows provide smoother estimates but respond more
        slowly to regime changes.
        """
        # Square the returns
        squared_returns = log_returns ** 2
        
        # Sum over rolling window and take square root
        rv = np.sqrt(squared_returns.rolling(window=window, min_periods=window).sum())
        
        return rv
    
    @staticmethod
    def _compute_amihud_illiquidity(
        log_returns: pd.Series,
        price: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Compute Amihud illiquidity measure.
        
        This measure captures how much price moves per unit of volume,
        serving as a proxy for market impact and liquidity.
        
        Formula:
            ILLIQ_t = |r_t| / (P_t * V_t)
        
        Args:
            log_returns: Log return series
                Type: pandas.Series
                Shape: (n_bars,)
            price: Price series (typically close)
                Type: pandas.Series
                Shape: (n_bars,)
            volume: Volume series
                Type: pandas.Series
                Shape: (n_bars,)
        
        Returns:
            Amihud illiquidity series
                Type: pandas.Series
                Shape: (n_bars,)
                Higher values indicate lower liquidity
        
        The measure is normalized by price and volume, making it
        comparable across different market conditions. High values
        indicate that small volumes cause large price changes.
        
        Reference:
            Amihud, Y. (2002). Illiquidity and stock returns: Cross-section
            and time-series effects. Journal of Financial Markets, 5(1), 31-56.
        """
        # Absolute return divided by dollar volume
        # Add small epsilon to avoid division by zero
        dollar_volume = price * volume
        illiq = np.abs(log_returns) / (dollar_volume + 1e-10)
        
        return illiq
    
    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Compute Relative Strength Index (RSI).
        
        RSI is a momentum oscillator that measures the speed and magnitude
        of price changes, bounded between 0 and 100.
        
        Formula:
            RS = Average Gain / Average Loss over period
            RSI = 100 - 100 / (1 + RS)
        
        Args:
            close: Close price series
                Type: pandas.Series
                Shape: (n_bars,)
            period: Lookback period for averaging
                Type: int
                Default: 14 (Wilder's original specification)
        
        Returns:
            RSI series
                Type: pandas.Series
                Shape: (n_bars,)
                Range: [0, 100]
                First (period) values are NaN
        
        Implementation uses Wilder's smoothing (EWM with alpha=1/period)
        for the gain and loss averages, which gives more weight to recent
        observations while maintaining a long memory.
        
        Interpretation:
            RSI > 70: Overbought condition
            RSI < 30: Oversold condition
        """
        # Calculate price changes
        delta = close.diff()
        
        # Separate gains and losses
        # Gains are positive changes, losses are absolute value of negative changes
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Wilder's smoothing: EWM with alpha = 1/period
        # This is equivalent to a modified moving average
        avg_gains = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        # Relative Strength
        # Add small epsilon to avoid division by zero
        rs = avg_gains / (avg_losses + 1e-10)
        
        # RSI formula
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def _compute_macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute MACD (Moving Average Convergence Divergence) and signal line.
        
        MACD is a trend-following momentum indicator that shows the relationship
        between two moving averages of price.
        
        Formula:
            MACD = EMA_fast - EMA_slow
            Signal = EMA_signal(MACD)
        
        Args:
            close: Close price series
                Type: pandas.Series
                Shape: (n_bars,)
            fast: Fast EMA period
                Type: int
                Default: 12
            slow: Slow EMA period
                Type: int
                Default: 26
            signal: Signal line EMA period
                Type: int
                Default: 9
        
        Returns:
            Tuple of (MACD line, Signal line)
                Type: Tuple[pandas.Series, pandas.Series]
                Both series have shape (n_bars,)
                First (slow) values are NaN for MACD
                First (slow + signal - 1) values are NaN for signal
        
        The MACD histogram (MACD - Signal) indicates momentum strength.
        Crossovers of MACD and signal line generate trading signals.
        """
        # Compute exponential moving averages
        ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        
        # MACD line
        macd = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        macd_signal = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
        
        return macd, macd_signal
    
    @staticmethod
    def _compute_roc(close: pd.Series, period: int) -> pd.Series:
        """
        Compute Rate of Change (ROC) momentum indicator.
        
        ROC measures the percentage change in price over a specified period,
        providing a normalized momentum measure.
        
        Formula:
            ROC_t = (C_t - C_{t-period}) / C_{t-period}
        
        Args:
            close: Close price series
                Type: pandas.Series
                Shape: (n_bars,)
            period: Lookback period
                Type: int
                Typical values: 5 (short), 10 (medium), 20 (longer)
        
        Returns:
            ROC series
                Type: pandas.Series
                Shape: (n_bars,)
                First (period) values are NaN
        
        ROC is similar to log returns but uses percentage change.
        Positive values indicate upward momentum, negative values
        indicate downward momentum.
        """
        roc = close.pct_change(periods=period)
        return roc
    
    @staticmethod
    def _compute_ema_features(
        close: pd.Series,
        period: int
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute EMA-based trend features: slope and deviation.
        
        These features capture both the direction (slope) and magnitude
        (deviation) of trend relative to an exponential moving average.
        
        Formulas:
            EMA_slope = EMA_t - EMA_{t-1}
            EMA_deviation = (C_t - EMA_t) / EMA_t
        
        Args:
            close: Close price series
                Type: pandas.Series
                Shape: (n_bars,)
            period: EMA period
                Type: int
                Typical values: 9 (short), 21 (medium), 50 (long)
        
        Returns:
            Tuple of (slope, deviation)
                Type: Tuple[pandas.Series, pandas.Series]
                slope: First derivative of EMA (trend direction)
                    Shape: (n_bars,)
                    First (period + 1) values are NaN
                deviation: Normalized distance from EMA (trend strength)
                    Shape: (n_bars,)
                    First (period) values are NaN
        
        Slope indicates trend direction and acceleration.
        Deviation indicates how far price has moved from its moving average,
        useful for mean reversion signals.
        """
        # Compute EMA
        ema = close.ewm(span=period, adjust=False, min_periods=period).mean()
        
        # Slope: change in EMA (trend direction)
        ema_slope = ema.diff()
        
        # Deviation: normalized distance from EMA (trend strength)
        # Division by EMA makes it scale-invariant
        ema_dev = (close - ema) / (ema + 1e-10)
        
        return ema_slope, ema_dev
    
    @staticmethod
    def _compute_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Compute Average True Range (ATR).
        
        ATR measures market volatility by decomposing the range of an asset
        for a given period. It accounts for gaps between trading sessions.
        
        Formula:
            TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
            ATR_t = SMA_period(TR)
        
        Args:
            high: High price series
                Type: pandas.Series
                Shape: (n_bars,)
            low: Low price series
                Type: pandas.Series
                Shape: (n_bars,)
            close: Close price series
                Type: pandas.Series
                Shape: (n_bars,)
            period: Averaging period
                Type: int
                Default: 14 (Wilder's original specification)
        
        Returns:
            ATR series
                Type: pandas.Series
                Shape: (n_bars,)
                First (period) values are NaN
        
        ATR is widely used for position sizing and stop-loss placement.
        Higher ATR indicates higher volatility and wider price ranges.
        
        Reference:
            Wilder, J. W. (1978). New concepts in technical trading systems.
            Trend Research.
        """
        # Previous close for gap calculation
        prev_close = close.shift(1)
        
        # Three components of True Range
        # 1. High - Low (intraday range)
        tr1 = high - low
        # 2. |High - Previous Close| (gap up from yesterday)
        tr2 = (high - prev_close).abs()
        # 3. |Low - Previous Close| (gap down from yesterday)
        tr3 = (low - prev_close).abs()
        
        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is simple moving average of True Range
        atr = true_range.rolling(window=period, min_periods=period).mean()
        
        return atr
    
    @staticmethod
    def _compute_volume_ma_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Compute volume relative to its moving average.
        
        This normalized volume measure helps identify unusual trading activity
        independent of the absolute volume level.
        
        Formula:
            Volume_MA_Ratio = V_t / SMA_period(V)
        
        Args:
            volume: Volume series
                Type: pandas.Series
                Shape: (n_bars,)
            period: Moving average period
                Type: int
                Default: 20 bars (~100 minutes for 5-min bars)
        
        Returns:
            Volume MA ratio series
                Type: pandas.Series
                Shape: (n_bars,)
                First (period-1) values are NaN
        
        Values > 1 indicate above-average volume
        Values < 1 indicate below-average volume
        Extreme values (e.g., > 3) can signal important events
        """
        # Simple moving average of volume
        volume_ma = volume.rolling(window=period, min_periods=period).mean()
        
        # Ratio of current volume to average
        # Add small epsilon to avoid division by zero
        volume_ratio = volume / (volume_ma + 1e-10)
        
        return volume_ratio
    
    # ==================== Validation Methods ====================
    
    def validate_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Validate computed features for data quality.
        
        Checks performed:
        1. NaN count (expected during warmup periods)
        2. Infinite values (should be zero)
        3. Range checks (detect anomalies)
        4. Distribution statistics
        
        Args:
            df: DataFrame with features
                Type: pandas.DataFrame
                Shape: (n_bars, n_features)
            feature_cols: List of feature column names to validate
                Type: List[str]
        
        Returns:
            Validation summary DataFrame
                Type: pandas.DataFrame
                Rows: One per feature
                Columns: nan_count, inf_count, min, max, mean, std
                Shape: (n_features, 6)
        """
        results = {}
        
        for col in feature_cols:
            if col not in df.columns:
                continue
                
            series = df[col]
            results[col] = {
                'nan_count': series.isna().sum(),
                'inf_count': np.isinf(series).sum(),
                'min': series.min() if not series.isna().all() else np.nan,
                'max': series.max() if not series.isna().all() else np.nan,
                'mean': series.mean() if not series.isna().all() else np.nan,
                'std': series.std() if not series.isna().all() else np.nan
            }
        
        return pd.DataFrame(results).T
    
    def check_multicollinearity(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Check for highly correlated features.
        
        Identifies feature pairs with correlation above threshold.
        High correlation can cause multicollinearity issues in models.
        
        Args:
            df: DataFrame with features
                Type: pandas.DataFrame
                Shape: (n_bars, n_features)
            feature_cols: List of feature columns
                Type: List[str]
            threshold: Correlation threshold for flagging
                Type: float
                Default: 0.95
        
        Returns:
            DataFrame of highly correlated pairs
                Type: pandas.DataFrame
                Columns: feature_1, feature_2, correlation
                Sorted by absolute correlation (descending)
        """
        # Compute correlation matrix
        corr_matrix = df[feature_cols].corr()
        
        # Find high correlations (excluding diagonal)
        high_corr = []
        for i in range(len(feature_cols)):
            for j in range(i+1, len(feature_cols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > threshold:
                    high_corr.append({
                        'feature_1': feature_cols[i],
                        'feature_2': feature_cols[j],
                        'correlation': corr
                    })
        
        # Convert to DataFrame and sort
        if high_corr:
            corr_df = pd.DataFrame(high_corr)
            corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
            return corr_df
        else:
            return pd.DataFrame(columns=['feature_1', 'feature_2', 'correlation'])
