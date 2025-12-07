# claude-engineering.md

## NQ Futures Forward Return Prediction: Engineering Implementation Guide

**Document Version:** 1.0  
**Role:** Engineering & Planning Lead (Claude)  
**Based On:** grok-scientific.md, gemini-research.md  
**Target Environment:** Google Colab A100 (80GB VRAM, 167GB RAM, 12 CPU cores)

---

## 1. Executive Summary

This document translates the scientific specifications from grok-scientific.md into concrete, executable PyTorch implementations. It covers the complete system architecture, module specifications, and a phased development plan for building a Transformer-based distributional prediction model for NQ futures forward returns across 6 horizons (5m, 15m, 30m, 60m, 2h, 4h).

### 1.1 Key Engineering Decisions

| Decision | Specification | Rationale |
|----------|--------------|-----------|
| Framework | PyTorch 2.0+ | Native FlashAttention via `scaled_dot_product_attention` |
| Data Format | Parquet | Efficient columnar storage, pandas/PyArrow compatible |
| Precision | BF16 training, FP32 accumulation | A100-optimized, numerical stability |
| Sequence Length | T_max = 7000 (padded) | Accommodates ~6900 bars + buffer |
| Batch Size | 8 (training), 16 (inference) | VRAM-optimized for full sequence |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Databento API → Raw OHLCV → Feature Engineering → Parquet Storage          │
│                                     ↓                                        │
│                            Google Drive Sync                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Colab VM ← Copy from Drive                                                  │
│       ↓                                                                      │
│  NQDataset → DataLoader → Model → Loss → Optimizer → Checkpoints            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MODEL                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Input → InstanceNorm → GroupProjection → PositionalEmbed → TSA+LGU Blocks  │
│                                                    ↓                         │
│                                    Multi-Horizon Quantile Heads              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Directory Structure

```
/content/drive/MyDrive/Colab Notebooks/Transformers/FP/
├── data/
│   ├── raw/
│   │   └── nq_ohlcv_1m_raw.parquet      # Raw Databento data
│   ├── processed/
│   │   └── nq_features_v1.parquet       # Engineered features
│   └── metadata/
│       └── feature_stats.json           # Normalization statistics
├── checkpoints/
│   └── model_v{version}_epoch{n}.pt
├── logs/
│   └── training_{timestamp}.log
└── configs/
    └── model_config.yaml
```

---

## 3. Module Specifications

### 3.1 Data Acquisition Module

#### 3.1.1 Databento Configuration

```python
# config/databento_config.py
DATABENTO_CONFIG = {
    "api_key": "db-TwwdiEYy786bMK6Y4ahnuPPYkJVcg",
    "dataset": "GLBX.MDP3",
    # CRITICAL SYMBOLOGY NOTE:
    # - NQ.c.0 = Calendar-based rollover (by expiration date)
    # - NQ.v.0 = Volume-based rollover (per problem statement requirement)
    # - NQ.n.0 = Open interest-based rollover
    # We use NQ.v.0 to satisfy the "volume-based rollover" requirement
    "symbol": "NQ.v.0",
    "stype_in": "continuous",
    "schema": "ohlcv-1m",
    "start": "2010-06-06",
    "end": "2025-12-03",
}

# IMPORTANT: Databento's to_df() automatically converts fixed-point integers
# to floats. Do NOT apply additional scaling (the 1e-9 factor is already applied).
# If you need raw integers, pass price_type='fixed' to to_df().

# Train/Val/Test splits (per Grok recommendation)
DATA_SPLITS = {
    "train": ("2010-06-06", "2019-12-31"),  # ~9.5 years
    "val": ("2020-01-01", "2022-12-31"),    # 3 years
    "test": ("2023-01-01", "2025-12-03"),   # ~3 years
}

# Data cost estimate: ~$100-500 for 15 years of ohlcv-1m NQ data
# Use client.metadata.get_cost() to verify before download
```

#### 3.1.2 Data Acquisition Implementation

```python
# src/data/acquisition.py
"""
NQ Futures Data Acquisition from Databento
CRITICAL: Databento's to_df() automatically converts fixed-point prices to floats.
Do NOT apply additional 1e-9 scaling - this would corrupt your data!
"""
import databento as db
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NQDataAcquisition:
    """Handles data download from Databento with chunked requests."""
    
    CHUNK_DAYS = 90  # Request in 90-day chunks to avoid timeouts
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    
    def __init__(self, api_key: str, output_dir: str):
        self.client = db.Historical(api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def estimate_cost(
        self,
        start: str,
        end: str,
        symbol: str = "NQ.v.0",
        schema: str = "ohlcv-1m"
    ) -> float:
        """
        Estimate cost before downloading.
        IMPORTANT: Run this first to verify budget (~$100-500 for 15 years).
        """
        cost = self.client.metadata.get_cost(
            dataset="GLBX.MDP3",
            symbols=[symbol],
            stype_in="continuous",
            schema=schema,
            start=start,
            end=end,
        )
        logger.info(f"Estimated cost: ${cost:.2f}")
        return cost
        
    def download_range(
        self,
        start: str,
        end: str,
        symbol: str = "NQ.v.0",
        schema: str = "ohlcv-1m"
    ) -> pd.DataFrame:
        """
        Download OHLCV data in chunks to handle large date ranges.
        
        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            symbol: Databento symbol (NQ.v.0 for volume-based continuous)
            schema: Data schema (ohlcv-1m for 1-minute bars)
            
        Returns:
            DataFrame with OHLCV data (prices already as floats)
        """
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        
        all_data = []
        current_start = start_dt
        
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=self.CHUNK_DAYS), end_dt)
            
            logger.info(f"Downloading: {current_start.date()} to {current_end.date()}")
            
            # Retry logic for transient errors
            for attempt in range(self.MAX_RETRIES):
                try:
                    data = self.client.timeseries.get_range(
                        dataset="GLBX.MDP3",
                        symbols=symbol,
                        stype_in="continuous",
                        schema=schema,
                        start=current_start.strftime("%Y-%m-%dT00:00:00"),
                        end=current_end.strftime("%Y-%m-%dT23:59:59"),
                    )
                    
                    # CRITICAL: to_df() already converts fixed-point to floats
                    # Do NOT multiply by 1e-9 again!
                    df = data.to_df()
                    if len(df) > 0:
                        all_data.append(df)
                        logger.info(f"  Retrieved {len(df):,} bars")
                    break  # Success, exit retry loop
                        
                except Exception as e:
                    logger.warning(f"  Attempt {attempt+1}/{self.MAX_RETRIES} failed: {e}")
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(self.RETRY_DELAY * (attempt + 1))
                    else:
                        logger.error(f"  Failed after {self.MAX_RETRIES} attempts")
                        raise
                
            current_start = current_end
            
        if not all_data:
            raise ValueError("No data retrieved")
            
        df = pd.concat(all_data, ignore_index=False)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        
        return self._process_raw_data(df)
    
    def _process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Databento format to standard OHLCV.
        NOTE: Prices are already floats from to_df() - no scaling needed!
        """
        processed = pd.DataFrame(index=df.index)
        
        # Prices are already converted to floats by to_df()
        # DO NOT multiply by 1e-9 - that would corrupt the data!
        for col in ['open', 'high', 'low', 'close']:
            processed[col] = df[col]
            
        processed['volume'] = df['volume'].astype(np.int64)
        
        # Add timestamp column from index
        processed['timestamp'] = processed.index
        processed = processed.reset_index(drop=True)
        
        # Validate data integrity
        self._validate_ohlcv(processed)
        
        return processed
    
    def _validate_ohlcv(self, df: pd.DataFrame) -> None:
        """Validate OHLCV data integrity."""
        # Check OHLC relationships
        invalid_hl = df['high'] < df['low']
        invalid_oh = df['open'] > df['high']
        invalid_ol = df['open'] < df['low']
        invalid_ch = df['close'] > df['high']
        invalid_cl = df['close'] < df['low']
        
        total_invalid = (invalid_hl | invalid_oh | invalid_ol | invalid_ch | invalid_cl).sum()
        
        if total_invalid > 0:
            logger.warning(f"Found {total_invalid} bars with invalid OHLC relationships")
            
        # Check for zero/negative prices
        zero_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
        if zero_prices > 0:
            logger.warning(f"Found {zero_prices} bars with zero/negative prices")
            
        # Sanity check: NQ should be in thousands range (not microscopic)
        median_price = df['close'].median()
        if median_price < 100:
            logger.error(f"CRITICAL: Median price is {median_price:.6f} - check for scaling bug!")
            raise ValueError("Price data appears incorrectly scaled")
        else:
            logger.info(f"Price sanity check passed: median={median_price:.2f}")
            
    def save_parquet(self, df: pd.DataFrame, filename: str) -> Path:
        """Save DataFrame to parquet format."""
        filepath = self.output_dir / filename
        df.to_parquet(filepath, engine='pyarrow', compression='snappy')
        logger.info(f"Saved {len(df):,} rows to {filepath}")
        return filepath
```

#### 3.1.3 Feature Engineering Module

```python
# src/data/features.py
"""
Technical Indicator Feature Engineering for NQ Futures
Based on grok-scientific.md Section 3.1 specifications
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Computes derived features from OHLCV data.
    
    Feature Groups (per grok-scientific.md):
    - F_P: Price dynamics (log-returns, high-low range, close location)
    - F_V: Volume (log-volume, delta, dollar volume)
    - F_T: Trend (VWAP deviation, MACD histogram, normalized SMA)
    - F_M: Momentum (RSI, CCI, DMI/ADX)
    - F_sigma: Volatility (ATR, Bollinger %B, bandwidth)
    - F_VW: Volume-weighted (MFI)
    - Gap feature: temporal gap indicator
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
        sma_periods: List[int] = [20, 50, 200],
        cci_period: int = 20,
        adx_period: int = 14,
        mfi_period: int = 14,
        rth_open_hour: int = 9,
        rth_open_minute: int = 30,
    ):
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.sma_periods = sma_periods
        self.cci_period = cci_period
        self.adx_period = adx_period
        self.mfi_period = mfi_period
        self.rth_open_hour = rth_open_hour
        self.rth_open_minute = rth_open_minute
        
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all feature groups from OHLCV data.
        
        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
            
        Returns:
            DataFrame with all computed features
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        features = pd.DataFrame(index=df.index)
        features['timestamp'] = df['timestamp']
        
        # Compute feature groups
        features = self._compute_price_dynamics(df, features)
        features = self._compute_volume_features(df, features)
        features = self._compute_trend_features(df, features)
        features = self._compute_momentum_features(df, features)
        features = self._compute_volatility_features(df, features)
        features = self._compute_volume_weighted_features(df, features)
        features = self._compute_gap_feature(df, features)
        
        # Add raw OHLCV for reference
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['close'] = df['close']
        features['volume'] = df['volume']
        
        return features
    
    def _compute_price_dynamics(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """F_P: Price dynamics features (4 features)."""
        # Log-returns
        features['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # High-low range (normalized by close)
        features['hl_range'] = (df['high'] - df['low']) / df['close']
        
        # Close location within bar (0 = low, 1 = high)
        hl_diff = df['high'] - df['low']
        features['close_location'] = np.where(
            hl_diff > 0,
            (df['close'] - df['low']) / hl_diff,
            0.5  # Doji bar
        )
        
        # Open-to-close return (intrabar momentum)
        features['open_return'] = np.log(df['close'] / df['open'])
        
        return features
    
    def _compute_volume_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """F_V: Volume features."""
        # Log volume (add 1 to handle zero volume)
        features['log_volume'] = np.log1p(df['volume'])
        
        # Log volume delta
        features['log_volume_delta'] = features['log_volume'] - features['log_volume'].shift(1)
        
        # Dollar volume (proxy: close * volume, normalized)
        dollar_vol = df['close'] * df['volume']
        features['dollar_volume'] = np.log1p(dollar_vol)
        
        return features
    
    def _compute_trend_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """F_T: Trend features including VWAP with RTH reset."""
        
        # VWAP with RTH session reset
        features['vwap_deviation'] = self._compute_vwap_deviation(df)
        
        # MACD histogram
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        # Normalize by price
        features['macd_histogram'] = (macd_line - signal_line) / df['close']
        
        # Normalized SMA deviations
        for period in self.sma_periods:
            sma = df['close'].rolling(window=period).mean()
            features[f'sma_{period}_dev'] = np.log(df['close'] / sma)
            
        return features
    
    def _compute_vwap_deviation(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute VWAP deviation with reset at RTH open (09:30 ET).
        Per grok-scientific.md: VWAP resets at RTH open to capture session dynamics.
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Identify RTH session starts (09:30 ET)
        df['time'] = df['timestamp'].dt.time
        df['date'] = df['timestamp'].dt.date
        
        rth_start = pd.Timestamp('09:30:00').time()
        
        # Create session ID (changes at each RTH open)
        df['is_rth_start'] = (
            (df['time'] >= rth_start) & 
            (df['time'] < pd.Timestamp('09:31:00').time())
        )
        df['session_id'] = df['is_rth_start'].cumsum()
        
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_volume'] = df['typical_price'] * df['volume']
        
        # Cumulative sums within session
        df['cum_tp_volume'] = df.groupby('session_id')['tp_volume'].cumsum()
        df['cum_volume'] = df.groupby('session_id')['volume'].cumsum()
        
        # VWAP
        vwap = np.where(
            df['cum_volume'] > 0,
            df['cum_tp_volume'] / df['cum_volume'],
            df['close']
        )
        
        # Deviation as percentage of price
        vwap_deviation = (df['close'] - vwap) / df['close']
        
        return pd.Series(vwap_deviation, index=df.index)
    
    def _compute_momentum_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """F_M: Momentum features (6 features)."""
        
        # RSI (normalized to [-1, 1] range)
        features['rsi_norm'] = (self._compute_rsi(df['close'], self.rsi_period) - 50) / 50
        
        # CCI (already normalized by definition)
        features['cci'] = self._compute_cci(df, self.cci_period) / 100  # Scale down
        
        # DMI/ADX
        plus_di, minus_di, adx = self._compute_dmi_adx(df, self.adx_period)
        features['plus_di'] = plus_di / 100
        features['minus_di'] = minus_di / 100
        features['adx'] = adx / 100
        
        # Rate of Change (10-bar, normalized)
        features['roc_norm'] = (df['close'] / df['close'].shift(10) - 1)
        
        return features
    
    def _compute_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _compute_cci(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Commodity Channel Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        cci = (typical_price - sma_tp) / (0.015 * mad.replace(0, np.finfo(float).eps))
        return cci
    
    def _compute_dmi_adx(self, df: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute Directional Movement Index and ADX."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed averages
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False).mean()
        
        # DI values
        plus_di = 100 * plus_dm_smooth / atr.replace(0, np.finfo(float).eps)
        minus_di = 100 * minus_dm_smooth / atr.replace(0, np.finfo(float).eps)
        
        # ADX
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        dx = 100 * di_diff / di_sum.replace(0, np.finfo(float).eps)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return plus_di, minus_di, adx
    
    def _compute_volatility_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """F_sigma: Volatility features (4 features)."""
        
        # Normalized ATR
        atr = self._compute_atr(df, self.atr_period)
        features['atr_norm'] = atr / df['close']
        
        # Bollinger Bands %B and Bandwidth
        sma = df['close'].rolling(window=self.bb_period).mean()
        std = df['close'].rolling(window=self.bb_period).std()
        
        upper_band = sma + self.bb_std * std
        lower_band = sma - self.bb_std * std
        
        band_diff = upper_band - lower_band
        features['bb_pct_b'] = np.where(
            band_diff > 0,
            (df['close'] - lower_band) / band_diff,
            0.5
        )
        features['bb_bandwidth'] = band_diff / sma.replace(0, np.finfo(float).eps)
        
        # Realized volatility (20-bar rolling std of log returns, annualized)
        log_returns = np.log(df['close'] / df['close'].shift(1))
        features['realized_vol'] = log_returns.rolling(window=20).std() * np.sqrt(252 * 390)
        
        return features
    
    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(span=period, adjust=False).mean()
        return atr
    
    def _compute_volume_weighted_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """F_VW: Volume-weighted features (MFI)."""
        
        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        
        # Determine money flow direction
        tp_diff = typical_price.diff()
        positive_flow = np.where(tp_diff > 0, raw_money_flow, 0)
        negative_flow = np.where(tp_diff < 0, raw_money_flow, 0)
        
        positive_sum = pd.Series(positive_flow).rolling(window=self.mfi_period).sum()
        negative_sum = pd.Series(negative_flow).rolling(window=self.mfi_period).sum()
        
        money_ratio = positive_sum / negative_sum.replace(0, np.finfo(float).eps)
        mfi = 100 - (100 / (1 + money_ratio))
        
        # Normalize to [-1, 1]
        features['mfi_norm'] = (mfi - 50) / 50
        
        return features
    
    def _compute_gap_feature(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Compute temporal gap feature.
        Per grok-scientific.md: gap = ln(1 + ((t_i - t_{i-1}) / 60 - 1))
        """
        timestamps = pd.to_datetime(df['timestamp'])
        time_diff_seconds = timestamps.diff().dt.total_seconds()
        time_diff_minutes = time_diff_seconds / 60
        
        # Log-transformed gap (0 for 1-minute intervals, positive for longer gaps)
        features['time_gap'] = np.log1p(np.maximum(time_diff_minutes - 1, 0))
        
        return features
    
    def compute_targets(
        self, 
        df: pd.DataFrame, 
        horizons: List[int] = [5, 15, 30, 60, 120, 240]
    ) -> pd.DataFrame:
        """
        Compute forward log-return targets for multiple horizons.
        
        Args:
            df: DataFrame with 'close' and 'timestamp' columns
            horizons: List of forward horizons in minutes
            
        Returns:
            DataFrame with target columns
        """
        targets = pd.DataFrame(index=df.index)
        targets['timestamp'] = df['timestamp']
        
        for h in horizons:
            # Forward log-return
            targets[f'target_{h}m'] = np.log(df['close'].shift(-h) / df['close'])
            
        return targets
    
    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names (excluding timestamp and OHLCV).
        Per grok-scientific.md Section 3.1: V=24 features (23 derived + 1 gap)
        """
        return [
            # Price dynamics (F_P) - 4 features
            'log_return', 'hl_range', 'close_location', 'open_return',
            # Volume (F_V) - 3 features
            'log_volume', 'log_volume_delta', 'dollar_volume',
            # Trend (F_T) - 5 features
            'vwap_deviation', 'macd_histogram',
            'sma_20_dev', 'sma_50_dev', 'sma_200_dev',
            # Momentum (F_M) - 6 features
            'rsi_norm', 'cci', 'plus_di', 'minus_di', 'adx', 'roc_norm',
            # Volatility (F_σ) - 4 features
            'atr_norm', 'bb_pct_b', 'bb_bandwidth', 'realized_vol',
            # Volume-weighted (F_VW) - 1 feature
            'mfi_norm',
            # Temporal - 1 feature (gap)
            'time_gap',
        ]  # Total: 4+3+5+6+4+1+1 = 24 features
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return feature groupings for TSA variable attention (6 groups)."""
        return {
            'price': ['log_return', 'hl_range', 'close_location', 'open_return'],
            'volume': ['log_volume', 'log_volume_delta', 'dollar_volume'],
            'trend': ['vwap_deviation', 'macd_histogram', 'sma_20_dev', 'sma_50_dev', 'sma_200_dev'],
            'momentum': ['rsi_norm', 'cci', 'plus_di', 'minus_di', 'adx', 'roc_norm'],
            'volatility': ['atr_norm', 'bb_pct_b', 'bb_bandwidth', 'realized_vol'],
            'flow': ['mfi_norm', 'time_gap'],  # Volume-weighted + temporal
        }
```

### 3.2 Dataset and DataLoader Module

```python
# src/data/dataset.py
"""
PyTorch Dataset for NQ Futures Time Series
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class NQFuturesDataset(Dataset):
    """
    Dataset for NQ futures prediction with rolling windows.
    
    Per grok-scientific.md:
    - Sequence length: T_max = 7000 (padded)
    - Features: 24 (23 derived + 1 gap)
    - Targets: 6 horizons (5m, 15m, 30m, 60m, 2h, 4h)
    """
    
    HORIZONS = [5, 15, 30, 60, 120, 240]
    T_MAX = 7000
    BARS_PER_WEEK = 6900  # Approximate bars in trading week
    
    def __init__(
        self,
        features_path: str,
        targets_path: str,
        feature_columns: List[str],
        mode: str = 'train',
        train_end_date: str = '2020-01-01',  # Train: 2010-2019
        val_end_date: str = '2023-01-01',    # Val: 2020-2022, Test: 2023-2025
        stride: int = 60,  # Sample every 60 bars for training (non-overlapping hour)
        normalize_stats_path: Optional[str] = None,
    ):
        self.mode = mode
        self.stride = stride if mode == 'train' else 1
        self.feature_columns = feature_columns
        
        # Load data
        features_df = pd.read_parquet(features_path)
        targets_df = pd.read_parquet(targets_path)
        
        # Merge on timestamp
        df = features_df.merge(targets_df, on='timestamp', how='inner')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Split by date
        train_end = pd.Timestamp(train_end_date)
        val_end = pd.Timestamp(val_end_date)
        
        if mode == 'train':
            df = df[df['timestamp'] < train_end]
        elif mode == 'val':
            df = df[(df['timestamp'] >= train_end) & (df['timestamp'] < val_end)]
        else:  # test
            df = df[df['timestamp'] >= val_end]
            
        # Extract arrays
        self.features = df[feature_columns].values.astype(np.float32)
        self.targets = df[[f'target_{h}m' for h in self.HORIZONS]].values.astype(np.float32)
        self.timestamps = df['timestamp'].values
        
        # Handle normalization statistics
        if mode == 'train':
            self.feature_means = np.nanmean(self.features, axis=0)
            self.feature_stds = np.nanstd(self.features, axis=0)
            self.feature_stds[self.feature_stds < 1e-8] = 1.0  # Prevent division by zero
            
            if normalize_stats_path:
                self._save_stats(normalize_stats_path)
        else:
            if normalize_stats_path and Path(normalize_stats_path).exists():
                self._load_stats(normalize_stats_path)
            else:
                raise ValueError("Validation/test mode requires normalization stats")
                
        # Create valid sample indices (need BARS_PER_WEEK history + max horizon forward)
        self.valid_indices = self._compute_valid_indices()
        
    def _compute_valid_indices(self) -> np.ndarray:
        """Compute indices that have enough history and valid targets."""
        n = len(self.features)
        max_horizon = max(self.HORIZONS)
        
        # Need BARS_PER_WEEK bars before and max_horizon bars after
        valid_start = self.BARS_PER_WEEK
        valid_end = n - max_horizon
        
        if valid_end <= valid_start:
            raise ValueError("Insufficient data for windowing")
            
        # Check for NaN targets
        target_valid = ~np.isnan(self.targets).any(axis=1)
        
        indices = []
        for i in range(valid_start, valid_end, self.stride):
            if target_valid[i]:
                indices.append(i)
                
        return np.array(indices)
    
    def _save_stats(self, path: str):
        """Save normalization statistics."""
        stats = {
            'means': self.feature_means.tolist(),
            'stds': self.feature_stds.tolist(),
            'columns': self.feature_columns
        }
        with open(path, 'w') as f:
            json.dump(stats, f)
            
    def _load_stats(self, path: str):
        """Load normalization statistics."""
        with open(path, 'r') as f:
            stats = json.load(f)
        self.feature_means = np.array(stats['means'], dtype=np.float32)
        self.feature_stds = np.array(stats['stds'], dtype=np.float32)
        
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with its context window.
        
        Returns:
            Dict containing:
            - features: (T_MAX, V) padded feature tensor
            - mask: (T_MAX,) attention mask (1 = valid, 0 = pad)
            - targets: (6,) forward returns for each horizon
            - temporal_features: (T_MAX, 8) positional encoding inputs
        """
        center_idx = self.valid_indices[idx]
        
        # Extract window [center_idx - BARS_PER_WEEK + 1, center_idx + 1)
        start_idx = center_idx - self.BARS_PER_WEEK + 1
        end_idx = center_idx + 1
        
        # Get features for window
        window_features = self.features[start_idx:end_idx]  # (actual_len, V)
        window_timestamps = self.timestamps[start_idx:end_idx]
        actual_len = len(window_features)
        
        # Pad to T_MAX
        padded_features = np.zeros((self.T_MAX, len(self.feature_columns)), dtype=np.float32)
        padded_features[:actual_len] = window_features
        
        # Create attention mask
        mask = np.zeros(self.T_MAX, dtype=np.float32)
        mask[:actual_len] = 1.0
        
        # Extract temporal features for positional encoding
        temporal_features = self._extract_temporal_features(window_timestamps, actual_len)
        
        # Get targets (at center_idx)
        targets = self.targets[center_idx]
        
        return {
            'features': torch.from_numpy(padded_features),
            'mask': torch.from_numpy(mask),
            'targets': torch.from_numpy(targets),
            'temporal_features': torch.from_numpy(temporal_features),
        }
    
    def _extract_temporal_features(self, timestamps: np.ndarray, actual_len: int) -> np.ndarray:
        """
        Extract temporal features for positional encoding.
        
        Returns: (T_MAX, 8) array with:
        - sin/cos of time of day (2)
        - day of week index (1)
        - sin/cos of day of month (2)
        - sin/cos of day of year (2)
        - actual minute of day for reference (1)
        """
        temporal = np.zeros((self.T_MAX, 8), dtype=np.float32)
        
        if actual_len == 0:
            return temporal
            
        ts = pd.to_datetime(timestamps)
        
        # Time of day (minute: 0-1439)
        minutes = ts.hour * 60 + ts.minute
        temporal[:actual_len, 0] = np.sin(2 * np.pi * minutes / 1440)
        temporal[:actual_len, 1] = np.cos(2 * np.pi * minutes / 1440)
        
        # Day of week (0-6, will be embedded)
        temporal[:actual_len, 2] = ts.dayofweek
        
        # Day of month (1-31)
        dom = ts.day
        temporal[:actual_len, 3] = np.sin(2 * np.pi * dom / 31)
        temporal[:actual_len, 4] = np.cos(2 * np.pi * dom / 31)
        
        # Day of year (1-366)
        doy = ts.dayofyear
        temporal[:actual_len, 5] = np.sin(2 * np.pi * doy / 365.25)
        temporal[:actual_len, 6] = np.cos(2 * np.pi * doy / 365.25)
        
        # Raw minute of day
        temporal[:actual_len, 7] = minutes / 1440.0
        
        return temporal


def create_dataloaders(
    features_path: str,
    targets_path: str,
    feature_columns: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    train_end_date: str = '2023-01-01',
    val_end_date: str = '2024-01-01',
    stats_path: str = 'feature_stats.json',
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    train_ds = NQFuturesDataset(
        features_path, targets_path, feature_columns,
        mode='train', train_end_date=train_end_date, val_end_date=val_end_date,
        normalize_stats_path=stats_path
    )
    
    val_ds = NQFuturesDataset(
        features_path, targets_path, feature_columns,
        mode='val', train_end_date=train_end_date, val_end_date=val_end_date,
        normalize_stats_path=stats_path
    )
    
    test_ds = NQFuturesDataset(
        features_path, targets_path, feature_columns,
        mode='test', train_end_date=train_end_date, val_end_date=val_end_date,
        normalize_stats_path=stats_path
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
```

### 3.3 Model Architecture Module

```python
# src/model/architecture.py
"""
Grok-SL Transformer Architecture
Based on grok-scientific.md specifications
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class InstanceNorm1d(nn.Module):
    """
    Instance Normalization for time series.
    Per grok-scientific.md Section 3.3:
    Normalizes per sample, per feature across time dimension.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
            
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, V) input tensor
            mask: (B, T) attention mask (1 = valid, 0 = pad)
        Returns:
            Normalized tensor (B, T, V)
        """
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
            
        # Expand mask for broadcasting: (B, T, 1)
        mask_expanded = mask.unsqueeze(-1)
        
        # Compute mean and std only over valid positions
        valid_counts = mask.sum(dim=1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
        valid_counts = valid_counts.clamp(min=1)
        
        # Masked mean
        x_masked = x * mask_expanded
        mean = x_masked.sum(dim=1, keepdim=True) / valid_counts  # (B, 1, V)
        
        # Masked variance
        var = ((x_masked - mean * mask_expanded) ** 2).sum(dim=1, keepdim=True) / valid_counts
        std = torch.sqrt(var + self.eps)  # (B, 1, V)
        
        # Normalize
        x_norm = (x - mean) / std
        
        # Apply affine transformation
        if self.affine:
            x_norm = self.gamma * x_norm + self.beta
            
        # Zero out padded positions
        x_norm = x_norm * mask_expanded
        
        return x_norm


class CyclicalPositionalEncoding(nn.Module):
    """
    Cyclical positional encoding for temporal features.
    Per grok-scientific.md Section 3.2.
    """
    
    def __init__(self, d_model: int, max_len: int = 7000):
        super().__init__()
        self.d_model = d_model
        
        # Learnable embedding for day of week
        self.dow_embedding = nn.Embedding(7, d_model // 4)
        
        # Linear projection for cyclical features
        # Input: 6 cyclical features (sin/cos for time, dom, doy) + 1 dow
        self.proj = nn.Linear(6 + d_model // 4, d_model)
        
    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_features: (B, T, 8) from dataset
                [0-1]: sin/cos time of day
                [2]: day of week (int)
                [3-4]: sin/cos day of month
                [5-6]: sin/cos day of year
                [7]: raw minute (unused here)
        Returns:
            (B, T, d_model) positional encodings
        """
        B, T, _ = temporal_features.shape
        
        # Extract cyclical features
        cyclical = temporal_features[:, :, [0, 1, 3, 4, 5, 6]]  # (B, T, 6)
        
        # Embed day of week
        dow = temporal_features[:, :, 2].long().clamp(0, 6)  # (B, T)
        dow_emb = self.dow_embedding(dow)  # (B, T, d_model // 4)
        
        # Concatenate and project
        combined = torch.cat([cyclical, dow_emb], dim=-1)  # (B, T, 6 + d_model//4)
        pe = self.proj(combined)  # (B, T, d_model)
        
        return pe


class GroupProjection(nn.Module):
    """
    Projects raw features into grouped embeddings for TSA.
    Per grok-scientific.md: 6 feature groups projected to d_model.
    """
    
    def __init__(
        self,
        feature_groups: Dict[str, List[int]],  # group_name -> list of feature indices
        d_model: int
    ):
        super().__init__()
        self.feature_groups = feature_groups
        self.d_model = d_model
        self.num_groups = len(feature_groups)
        
        # Create projection for each group
        self.projections = nn.ModuleDict()
        for name, indices in feature_groups.items():
            self.projections[name] = nn.Linear(len(indices), d_model)
            
        self.group_names = list(feature_groups.keys())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, V) raw features
        Returns:
            (B, T, G, d_model) grouped embeddings where G = num_groups
        """
        B, T, V = x.shape
        
        group_embeddings = []
        for name in self.group_names:
            indices = self.feature_groups[name]
            group_features = x[:, :, indices]  # (B, T, group_size)
            group_emb = self.projections[name](group_features)  # (B, T, d_model)
            group_embeddings.append(group_emb)
            
        # Stack groups: (B, T, G, d_model)
        output = torch.stack(group_embeddings, dim=2)
        
        return output


class LiteGateUnit(nn.Module):
    """
    Lite Gate Unit for noise filtering.
    Per grok-scientific.md Section 3.5.
    """
    
    def __init__(self, d_model: int, init_bias: float = 0.0):
        super().__init__()
        
        # Gate computation
        self.W_z = nn.Linear(d_model, d_model)
        self.U_z = nn.Linear(d_model, d_model)
        
        # Candidate computation
        self.W_h = nn.Linear(d_model, d_model)
        self.U_h = nn.Linear(d_model, d_model)
        
        # Initialize biases to favor z ≈ 0.5 initially
        nn.init.constant_(self.W_z.bias, init_bias)
        nn.init.constant_(self.U_z.bias, init_bias)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (*, d_model) residual input
            y: (*, d_model) attention output
        Returns:
            (*, d_model) gated output
        """
        # Gate
        z = torch.sigmoid(self.W_z(x) + self.U_z(y))
        
        # Candidate
        h = torch.tanh(self.W_h(x) + self.U_h(z * y))
        
        # Gated combination
        output = (1 - z) * x + z * h
        
        return output


class TemporalAttention(nn.Module):
    """
    Temporal attention (Stage 1 of TSA).
    Attends across time for each variable independently.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, G, d_model) or (B*G, T, d_model)
            mask: (B, T) attention mask
        Returns:
            Same shape as input
        """
        # Handle 4D input by reshaping
        if x.dim() == 4:
            B, T, G, D = x.shape
            x = x.view(B * G, T, D)
            if mask is not None:
                mask = mask.unsqueeze(1).expand(-1, G, -1).reshape(B * G, T)
            reshape_back = True
        else:
            reshape_back = False
            B_G, T, D = x.shape
            
        # Self-attention with FlashAttention
        residual = x
        x = self.norm(x)
        
        qkv = self.qkv(x).reshape(-1, T, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.unbind(dim=2)  # Each: (B*G, T, heads, head_dim)
        q = q.transpose(1, 2)  # (B*G, heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Create attention mask if provided
        attn_mask = None
        if mask is not None:
            # Convert to attention bias (0 for valid, -inf for masked)
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (B*G, 1, 1, T)
            attn_mask = attn_mask.expand(-1, self.num_heads, T, -1)
            attn_mask = torch.where(attn_mask.bool(), 0.0, float('-inf'))
            
        # FlashAttention via PyTorch 2.0+
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        out = out.transpose(1, 2).reshape(-1, T, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        out = residual + out
        
        if reshape_back:
            out = out.view(B, G, T, D).permute(0, 2, 1, 3)  # (B, T, G, D)
            
        return out


class VariableAttention(nn.Module):
    """
    Variable attention (Stage 2 of TSA).
    Attends across variables at each time step.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, G, d_model)
        Returns:
            (B, T, G, d_model)
        """
        B, T, G, D = x.shape
        
        # Reshape to (B*T, G, D) for attention across variables
        x = x.view(B * T, G, D)
        
        residual = x
        x = self.norm(x)
        
        qkv = self.qkv(x).reshape(B * T, G, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B*T, heads, G, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # No masking needed for variable dimension
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        out = out.transpose(1, 2).reshape(B * T, G, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        out = residual + out
        
        # Reshape back
        out = out.view(B, T, G, D)
        
        return out


class TSABlock(nn.Module):
    """
    Two-Stage Attention Block.
    Combines Temporal Attention + Variable Attention + LGU + FFN.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Stage 1: Temporal attention
        self.temporal_attn = TemporalAttention(d_model, num_heads, dropout)
        
        # Stage 2: Variable attention
        self.variable_attn = VariableAttention(d_model, num_heads, dropout)
        
        # LGU after attention
        self.lgu = LiteGateUnit(d_model)
        
        # FFN
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, G, d_model)
            mask: (B, T) attention mask
        Returns:
            (B, T, G, d_model)
        """
        # Temporal attention
        attn_out = self.temporal_attn(x, mask)
        
        # Variable attention
        attn_out = self.variable_attn(attn_out)
        
        # LGU gating
        B, T, G, D = x.shape
        x_flat = x.reshape(B * T * G, D)
        attn_flat = attn_out.reshape(B * T * G, D)
        gated = self.lgu(x_flat, attn_flat)
        gated = gated.view(B, T, G, D)
        
        # FFN
        residual = gated
        gated = self.ffn_norm(gated)
        gated = residual + self.ffn(gated)
        
        return gated


class QuantileHead(nn.Module):
    """
    Quantile regression head for a single horizon.
    Outputs 7 quantiles: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    """
    
    QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.QUANTILES))
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, d_model) final hidden state
        Returns:
            (B, 7) quantile predictions
        """
        return self.mlp(x)


class MultiHorizonHead(nn.Module):
    """
    Multi-horizon autoregressive quantile heads.
    Per grok-scientific.md: shorter horizons condition longer ones.
    
    Teacher forcing improvement (per Gemini review):
    - Instead of naive scalar expansion (prev_value.expand(-1, 7)), we use
      a small MLP to embed the scalar target into a proper representation.
    - This preserves conditioning semantics and matches inference behavior.
    """
    
    HORIZONS = [5, 15, 30, 60, 120, 240]
    
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        
        self.d_model = d_model
        self.heads = nn.ModuleList()
        self.conditioning_projs = nn.ModuleList()
        
        # Embedding MLPs for teacher forcing (scalar -> 7-dim representation)
        # This properly embeds ground truth values for conditioning
        self.target_embeddings = nn.ModuleList()
        
        for i, h in enumerate(self.HORIZONS):
            self.heads.append(QuantileHead(d_model, hidden_dim))
            
            if i > 0:
                # Project previous horizon's quantiles + hidden to current hidden
                self.conditioning_projs.append(
                    nn.Linear(7 + d_model, d_model)
                )
                # MLP to embed scalar target into 7-dim (matching quantile output shape)
                # Architecture: scalar -> hidden -> 7, with tanh activation
                self.target_embeddings.append(
                    nn.Sequential(
                        nn.Linear(1, hidden_dim // 4),
                        nn.Tanh(),
                        nn.Linear(hidden_dim // 4, 7)
                    )
                )
            else:
                self.conditioning_projs.append(None)
                self.target_embeddings.append(None)
                
    def forward(
        self, 
        h: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True
    ) -> torch.Tensor:
        """
        Args:
            h: (B, d_model) final hidden state
            targets: (B, 6) ground truth returns (for teacher forcing)
            teacher_forcing: If True, use ground truth for conditioning
        Returns:
            (B, 6, 7) quantile predictions for each horizon
        """
        B = h.shape[0]
        outputs = []
        prev_output = None
        prev_hidden = h
        
        for i, head in enumerate(self.heads):
            if i == 0:
                # First horizon: direct prediction
                current_hidden = h
            else:
                # Autoregressive: condition on previous
                if teacher_forcing and targets is not None:
                    # Embed ground truth via learned MLP (not naive expansion)
                    prev_value = targets[:, i-1:i]  # (B, 1)
                    prev_cond = self.target_embeddings[i](prev_value)  # (B, 7)
                else:
                    # Use predicted quantile distribution
                    prev_cond = prev_output  # (B, 7)
                    
                # Concatenate quantile representation and hidden state
                combined = torch.cat([prev_cond, prev_hidden], dim=-1)  # (B, 7 + d_model)
                current_hidden = self.conditioning_projs[i](combined)  # (B, d_model)
                
            output = head(current_hidden)  # (B, 7)
            outputs.append(output)
            
            prev_output = output
            prev_hidden = current_hidden
            
        # Stack: (B, 6, 7)
        return torch.stack(outputs, dim=1)


class GrokSLTransformer(nn.Module):
    """
    Complete Grok-SL Transformer model.
    Per grok-scientific.md Section 6.
    """
    
    def __init__(
        self,
        num_features: int = 24,  # Per grok-scientific.md Section 3.1: V=24
        feature_groups: Dict[str, List[int]] = None,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 7000,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_features = num_features
        
        # Default feature groups if not provided
        # Per grok-scientific.md: 6 groups, V=24 total features
        if feature_groups is None:
            feature_groups = {
                'price': [0, 1, 2, 3],           # log_return, hl_range, close_location, open_return
                'volume': [4, 5, 6],             # log_volume, log_volume_delta, dollar_volume
                'trend': [7, 8, 9, 10, 11],      # vwap, macd, sma devs
                'momentum': [12, 13, 14, 15, 16, 17],  # rsi, cci, di, adx, roc
                'volatility': [18, 19, 20, 21],  # atr, bb_pct_b, bandwidth, realized_vol
                'flow': [22, 23],                # mfi, time_gap
            }
        self.feature_groups = feature_groups
        self.num_groups = len(feature_groups)
        
        # Instance normalization
        self.instance_norm = InstanceNorm1d(num_features)
        
        # Group projection
        self.group_proj = GroupProjection(feature_groups, d_model)
        
        # Positional encoding
        self.pos_encoding = CyclicalPositionalEncoding(d_model, max_len)
        
        # TSA blocks
        self.tsa_blocks = nn.ModuleList([
            TSABlock(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Pooling and output
        self.pool_norm = nn.LayerNorm(d_model)
        self.pool_proj = nn.Linear(d_model * self.num_groups, d_model)
        
        # Multi-horizon heads
        self.heads = MultiHorizonHead(d_model)
        
    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        temporal_features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True
    ) -> torch.Tensor:
        """
        Args:
            features: (B, T, V) raw features
            mask: (B, T) attention mask
            temporal_features: (B, T, 8) for positional encoding
            targets: (B, 6) ground truth (for teacher forcing)
            teacher_forcing: Use ground truth for autoregressive conditioning
        Returns:
            (B, 6, 7) quantile predictions
        """
        B, T, V = features.shape
        
        # Instance normalization
        x = self.instance_norm(features, mask)
        
        # Group projection: (B, T, G, d_model)
        x = self.group_proj(x)
        
        # Add positional encoding
        pe = self.pos_encoding(temporal_features)  # (B, T, d_model)
        x = x + pe.unsqueeze(2)  # Broadcast to all groups
        
        # TSA blocks
        for block in self.tsa_blocks:
            x = block(x, mask)
            
        # Extract last valid position for each sample
        # Find last valid index per sample
        last_valid_idx = mask.sum(dim=1).long() - 1  # (B,)
        batch_indices = torch.arange(B, device=x.device)
        
        # Extract: (B, G, d_model)
        h = x[batch_indices, last_valid_idx]
        
        # Pool across groups
        h = self.pool_norm(h)
        h = h.view(B, -1)  # (B, G * d_model)
        h = self.pool_proj(h)  # (B, d_model)
        
        # Multi-horizon prediction
        output = self.heads(h, targets, teacher_forcing)
        
        return output
```

### 3.4 Loss and Training Module

```python
# src/training/loss.py
"""
Loss functions for distributional prediction.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List


class PinballLoss(nn.Module):
    """
    Quantile (pinball) loss for distributional prediction.
    Per grok-scientific.md Section 3.6.
    """
    
    def __init__(self, quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, H, Q) predicted quantiles
            targets: (B, H) ground truth values
        Returns:
            Scalar loss
        """
        self.quantiles = self.quantiles.to(predictions.device)
        
        # Expand targets for broadcasting: (B, H, 1)
        targets = targets.unsqueeze(-1)
        
        # Compute errors: (B, H, Q)
        errors = targets - predictions
        
        # Pinball loss: (B, H, Q)
        loss = torch.where(
            errors >= 0,
            self.quantiles * errors,
            (self.quantiles - 1) * errors
        )
        
        return loss.mean()


class MonotonicityLoss(nn.Module):
    """
    Penalizes quantile crossing (non-monotonic predictions).
    """
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, H, Q) predicted quantiles (should be increasing in Q)
        Returns:
            Scalar penalty
        """
        # Check for quantile crossings
        diffs = predictions[:, :, 1:] - predictions[:, :, :-1]
        violations = torch.relu(-diffs)  # Positive where q_j > q_{j+1}
        
        return violations.mean()


class MultiHorizonQuantileLoss(nn.Module):
    """
    Combined loss for multi-horizon quantile prediction.
    Per grok-scientific.md: omega_h = 1 / log(h + 1)
    """
    
    HORIZONS = [5, 15, 30, 60, 120, 240]
    
    def __init__(
        self,
        quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
        monotonicity_weight: float = 0.1
    ):
        super().__init__()
        
        self.pinball = PinballLoss(quantiles)
        self.monotonicity = MonotonicityLoss()
        self.monotonicity_weight = monotonicity_weight
        
        # Compute horizon weights
        weights = [1.0 / np.log(h + 1) for h in self.HORIZONS]
        weights = np.array(weights) / np.sum(weights)  # Normalize
        self.horizon_weights = torch.tensor(weights, dtype=torch.float32)
        
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            predictions: (B, 6, 7) quantile predictions
            targets: (B, 6) ground truth returns
        Returns:
            total_loss: Scalar
            loss_dict: Dictionary of individual losses
        """
        self.horizon_weights = self.horizon_weights.to(predictions.device)
        
        # Mask out NaN targets
        valid_mask = ~torch.isnan(targets)
        
        # Compute per-horizon pinball loss
        horizon_losses = []
        for h in range(6):
            mask = valid_mask[:, h]
            if mask.sum() > 0:
                h_pred = predictions[mask, h, :]  # (valid_B, 7)
                h_target = targets[mask, h]  # (valid_B,)
                h_loss = self.pinball(h_pred.unsqueeze(1), h_target.unsqueeze(1))
                horizon_losses.append(h_loss * self.horizon_weights[h])
            else:
                horizon_losses.append(torch.tensor(0.0, device=predictions.device))
                
        pinball_loss = sum(horizon_losses)
        
        # Monotonicity penalty
        mono_loss = self.monotonicity(predictions)
        
        # Total loss
        total_loss = pinball_loss + self.monotonicity_weight * mono_loss
        
        loss_dict = {
            'total': total_loss,
            'pinball': pinball_loss,
            'monotonicity': mono_loss,
        }
        
        return total_loss, loss_dict


# src/training/trainer.py
"""
Training loop with mixed precision, gradient checkpointing, and WandB logging.
Incorporates Grok/Gemini suggestions for A100 optimization.
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.checkpoint import checkpoint_sequential
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, Optional
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training handler with A100 optimizations.
    - Mixed precision (BF16/TF32)
    - Gradient checkpointing for OOM fallback
    - WandB logging (optional)
    - torch.compile support (PyTorch 2.0+)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        max_epochs: int = 100,
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 100,
        use_amp: bool = True,
        use_wandb: bool = True,
        use_gradient_checkpointing: bool = False,
        use_compile: bool = False,
        project_name: str = 'nq-futures-transformer',
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.use_amp = use_amp
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Optional torch.compile (PyTorch 2.0+)
        if use_compile and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(model, mode='reduce-overhead')
        
        # WandB initialization
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project=project_name, config={
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'warmup_steps': warmup_steps,
                'max_epochs': max_epochs,
            })
            wandb.watch(model, log='gradients', log_freq=500)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Store initial LR for warmup
        for pg in self.optimizer.param_groups:
            pg['initial_lr'] = learning_rate
        
        # Scheduler with warmup
        self.warmup_steps = warmup_steps
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=len(train_loader) * 10,  # Restart every 10 epochs
            T_mult=2
        )
        
        # Loss function
        self.criterion = MultiHorizonQuantileLoss()
        
        # Mixed precision
        self.scaler = GradScaler(enabled=use_amp)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            # Move to GPU
            features = batch['features'].cuda()
            mask = batch['mask'].cuda()
            targets = batch['targets'].cuda()
            temporal = batch['temporal_features'].cuda()
            
            # Warmup learning rate
            if self.global_step < self.warmup_steps:
                lr_scale = self.global_step / self.warmup_steps
                for pg in self.optimizer.param_groups:
                    pg['lr'] = pg['initial_lr'] * lr_scale
                    
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                predictions = self.model(
                    features, mask, temporal, 
                    targets=targets, teacher_forcing=True
                )
                loss, loss_dict = self.criterion(predictions, targets)
                
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.global_step >= self.warmup_steps:
                self.scheduler.step()
                
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # WandB logging
            if self.use_wandb and self.global_step % self.log_interval == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/pinball': loss_dict['pinball'].item(),
                    'train/monotonicity': loss_dict['monotonicity'].item(),
                    'train/grad_norm': grad_norm.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step,
                })
            
        return {'train_loss': total_loss / num_batches}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            features = batch['features'].cuda()
            mask = batch['mask'].cuda()
            targets = batch['targets'].cuda()
            temporal = batch['temporal_features'].cuda()
            
            with autocast(enabled=self.use_amp):
                predictions = self.model(
                    features, mask, temporal,
                    targets=None, teacher_forcing=False
                )
                loss, _ = self.criterion(predictions, targets)
                
            total_loss += loss.item()
            num_batches += 1
            
            # Collect for IC computation
            all_predictions.append(predictions[:, :, 3].cpu())  # Median (tau=0.5)
            all_targets.append(targets.cpu())
            
        # Compute IC (Spearman correlation) for each horizon
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        ic_per_horizon = {}
        for h_idx, h in enumerate([5, 15, 30, 60, 120, 240]):
            valid_mask = ~torch.isnan(all_targets[:, h_idx])
            if valid_mask.sum() > 10:
                pred_h = all_predictions[valid_mask, h_idx].numpy()
                tgt_h = all_targets[valid_mask, h_idx].numpy()
                from scipy.stats import spearmanr
                ic, _ = spearmanr(pred_h, tgt_h)
                ic_per_horizon[f'ic_{h}m'] = ic
                
        val_loss = total_loss / num_batches
        
        # WandB logging
        if self.use_wandb:
            log_dict = {'val/loss': val_loss}
            log_dict.update({f'val/{k}': v for k, v in ic_per_horizon.items()})
            wandb.log(log_dict)
            
        return {'val_loss': val_loss, **ic_per_horizon}
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'global_step': self.global_step,
            'metrics': metrics,
        }
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        
        # Save best model
        if metrics.get('val_loss', float('inf')) < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f'New best model saved with val_loss: {self.best_val_loss:.4f}')
            
    def train(self):
        """Full training loop."""
        for epoch in range(self.max_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            metrics = {**train_metrics, **val_metrics}
            logger.info(f'Epoch {epoch}: {metrics}')
            
            self.save_checkpoint(epoch, metrics)
```

---

## 4. Phased Development Plan

### Phase Overview

| Phase | Focus | Duration | Key Deliverables |
|-------|-------|----------|------------------|
| 1 | Data Pipeline | 1 week | Raw data, features, storage |
| 2 | Dataset & Baseline | 2 weeks | DataLoader, vanilla Transformer |
| 3 | TSA + LGU | 3 weeks | Full architecture implementation |
| 4 | Training Infrastructure | 1 week | AMP, checkpointing, WandB logging |
| 5 | Multi-Horizon Heads | 1 week | Autoregressive prediction |
| 6 | Evaluation & Optimization | 2 weeks | Metrics, ablations, torch.compile |

**Total Duration:** ~10 weeks

---

### Phase 1: Data Acquisition and Feature Engineering

**Duration:** 1 week  
**Objective:** Download NQ data from Databento, compute all derived features, save to Google Drive

#### 1.1 Setup Colab Environment

```python
# phase1_setup.ipynb - Cell 1: Environment Setup
# ==================================================

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install core dependencies
!pip install -q databento pandas numpy pyarrow tqdm scipy

# REQUIRED: Install ta-lib for feature engineering (RSI, ATR, etc.)
# ta-lib requires system library installation first
!apt-get update -qq
!apt-get install -y -qq ta-lib libta-lib-dev
!pip install -q ta-lib

# Optional but recommended
!pip install -q wandb  # Experiment tracking

# Verify ta-lib installation
import talib
print(f"ta-lib version: {talib.__version__}")

# Create directory structure
import os
DATA_ROOT = '/content/drive/MyDrive/Colab Notebooks/Transformers/FP/data'
os.makedirs(f'{DATA_ROOT}/raw', exist_ok=True)
os.makedirs(f'{DATA_ROOT}/processed', exist_ok=True)
os.makedirs(f'{DATA_ROOT}/metadata', exist_ok=True)

print("Environment ready!")
```

#### 1.2 Data Acquisition Script

```python
# phase1_acquisition.ipynb - Cell 2: Data Download
# ==================================================
# CRITICAL: Databento's to_df() already converts prices to floats.
# Do NOT multiply by 1e-9 - this would corrupt your data!

import databento as db
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = "db-TwwdiEYy786bMK6Y4ahnuPPYkJVcg"
DATASET = "GLBX.MDP3"
SYMBOL = "NQ.v.0"  # Volume-based continuous (required by problem statement)
SCHEMA = "ohlcv-1m"
START_DATE = "2010-06-06"
END_DATE = "2025-12-03"
CHUNK_DAYS = 90
MAX_RETRIES = 3
OUTPUT_DIR = Path('/content/drive/MyDrive/Colab Notebooks/Transformers/FP/data/raw')

def estimate_and_confirm_cost():
    """Estimate cost before downloading. Budget: ~$100-500 for 15 years."""
    client = db.Historical(API_KEY)
    cost = client.metadata.get_cost(
        dataset=DATASET,
        symbols=[SYMBOL],
        stype_in="continuous",
        schema=SCHEMA,
        start=START_DATE,
        end=END_DATE,
    )
    print(f"Estimated cost: ${cost:.2f}")
    print("Proceed with download? (costs will be charged to your account)")
    return cost

def download_nq_data():
    """Download NQ futures data from Databento in chunks."""
    client = db.Historical(API_KEY)
    
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
    
    all_data = []
    current_start = start_dt
    
    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS), end_dt)
        
        logger.info(f"Downloading: {current_start.date()} to {current_end.date()}")
        
        # Retry logic for transient errors
        for attempt in range(MAX_RETRIES):
            try:
                data = client.timeseries.get_range(
                    dataset=DATASET,
                    symbols=SYMBOL,
                    stype_in="continuous",
                    schema=SCHEMA,
                    start=current_start.strftime("%Y-%m-%dT00:00:00"),
                    end=current_end.strftime("%Y-%m-%dT23:59:59"),
                )
                
                # CRITICAL: to_df() already converts fixed-point to floats
                # Prices will be in correct range (e.g., 20000.00 for NQ)
                df = data.to_df()
                if len(df) > 0:
                    all_data.append(df)
                    logger.info(f"  Retrieved {len(df):,} bars")
                break  # Success
                    
            except Exception as e:
                logger.warning(f"  Attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    raise
                
        current_start = current_end
        
    # Concatenate all data
    df = pd.concat(all_data, ignore_index=False)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    
    # Process raw data - NO PRICE SCALING NEEDED
    processed = pd.DataFrame()
    processed['timestamp'] = df.index
    processed['open'] = df['open'].values      # Already floats!
    processed['high'] = df['high'].values      # Already floats!
    processed['low'] = df['low'].values        # Already floats!
    processed['close'] = df['close'].values    # Already floats!
    processed['volume'] = df['volume'].values.astype(np.int64)
    
    # Validate - sanity check that prices are in correct range
    median_price = processed['close'].median()
    if median_price < 100:
        raise ValueError(f"CRITICAL: Median price {median_price:.6f} is too low - scaling bug!")
    logger.info(f"Price sanity check passed: median={median_price:.2f}")
    
    # Validate OHLC relationships
    invalid = (
        (processed['high'] < processed['low']) |
        (processed['open'] > processed['high']) |
        (processed['open'] < processed['low']) |
        (processed['close'] > processed['high']) |
        (processed['close'] < processed['low'])
    ).sum()
    
    if invalid > 0:
        logger.warning(f"Found {invalid} bars with invalid OHLC relationships")
        
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'nq_ohlcv_1m_raw.parquet'
    processed.to_parquet(output_path, engine='pyarrow', compression='snappy')
    logger.info(f"Saved {len(processed):,} rows to {output_path}")
    
    return processed

# Execute
if __name__ == "__main__":
    # Step 1: Check cost first
    cost = estimate_and_confirm_cost()
    
    # Step 2: Download (uncomment after confirming cost)
    # df = download_nq_data()
    # print(f"\nData shape: {df.shape}")
    # print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    # print(f"Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
    # print(f"\nSample:\n{df.head()}")
```

#### 1.3 Feature Engineering Script

```python
# phase1_features.ipynb - Cell 3: Feature Engineering
# ====================================================

import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

DATA_ROOT = Path('/content/drive/MyDrive/Colab Notebooks/Transformers/FP/data')
RAW_PATH = DATA_ROOT / 'raw' / 'nq_ohlcv_1m_raw.parquet'
FEATURES_PATH = DATA_ROOT / 'processed' / 'nq_features_v1.parquet'
TARGETS_PATH = DATA_ROOT / 'processed' / 'nq_targets_v1.parquet'
STATS_PATH = DATA_ROOT / 'metadata' / 'feature_stats.json'

# Load raw data
print("Loading raw data...")
df = pd.read_parquet(RAW_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"Loaded {len(df):,} rows")

# Feature computation functions
def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    return 100 - (100 / (1 + rs))

def compute_vwap_deviation(df: pd.DataFrame) -> pd.Series:
    """Compute VWAP deviation with RTH session reset."""
    df = df.copy()
    
    # Create session IDs based on RTH open (09:30 ET)
    df['time'] = df['timestamp'].dt.time
    df['is_rth_start'] = (
        (df['timestamp'].dt.hour == 9) & 
        (df['timestamp'].dt.minute >= 30) &
        (df['timestamp'].dt.minute < 31)
    )
    df['session_id'] = df['is_rth_start'].cumsum()
    
    # Typical price and cumulative sums
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_volume'] = df['typical_price'] * df['volume']
    df['cum_tp_volume'] = df.groupby('session_id')['tp_volume'].cumsum()
    df['cum_volume'] = df.groupby('session_id')['volume'].cumsum()
    
    # VWAP
    vwap = np.where(
        df['cum_volume'] > 0,
        df['cum_tp_volume'] / df['cum_volume'],
        df['close']
    )
    
    return pd.Series((df['close'].values - vwap) / df['close'].values)

def compute_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute CCI indicator."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.finfo(float).eps))

def compute_dmi_adx(df: pd.DataFrame, period: int = 14):
    """Compute DMI and ADX indicators."""
    high, low, close = df['high'], df['low'], df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False).mean()
    
    plus_di = 100 * plus_dm_smooth / atr.replace(0, np.finfo(float).eps)
    minus_di = 100 * minus_dm_smooth / atr.replace(0, np.finfo(float).eps)
    
    di_diff = abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    dx = 100 * di_diff / di_sum.replace(0, np.finfo(float).eps)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return plus_di, minus_di, adx

def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Money Flow Index."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    raw_mf = tp * df['volume']
    
    tp_diff = tp.diff()
    pos_flow = np.where(tp_diff > 0, raw_mf, 0)
    neg_flow = np.where(tp_diff < 0, raw_mf, 0)
    
    pos_sum = pd.Series(pos_flow).rolling(window=period).sum()
    neg_sum = pd.Series(neg_flow).rolling(window=period).sum()
    
    mf_ratio = pos_sum / neg_sum.replace(0, np.finfo(float).eps)
    return 100 - (100 / (1 + mf_ratio))

print("Computing features...")

features = pd.DataFrame()
features['timestamp'] = df['timestamp']

# Price dynamics
features['log_return'] = np.log(df['close'] / df['close'].shift(1))
features['hl_range'] = (df['high'] - df['low']) / df['close']
hl_diff = df['high'] - df['low']
features['close_location'] = np.where(hl_diff > 0, (df['close'] - df['low']) / hl_diff, 0.5)
features['open_return'] = np.log(df['close'] / df['open'])  # Intrabar return

# Volume
features['log_volume'] = np.log1p(df['volume'])
features['log_volume_delta'] = features['log_volume'] - features['log_volume'].shift(1)
features['dollar_volume'] = np.log1p(df['close'] * df['volume'])

# Trend
features['vwap_deviation'] = compute_vwap_deviation(df)
ema_fast = df['close'].ewm(span=12, adjust=False).mean()
ema_slow = df['close'].ewm(span=26, adjust=False).mean()
macd_line = ema_fast - ema_slow
signal_line = macd_line.ewm(span=9, adjust=False).mean()
features['macd_histogram'] = (macd_line - signal_line) / df['close']

for period in [20, 50, 200]:
    sma = df['close'].rolling(window=period).mean()
    features[f'sma_{period}_dev'] = np.log(df['close'] / sma)

# Momentum
features['rsi_norm'] = (compute_rsi(df['close'], 14) - 50) / 50
features['cci'] = compute_cci(df, 20) / 100
plus_di, minus_di, adx = compute_dmi_adx(df, 14)
features['plus_di'] = plus_di / 100
features['minus_di'] = minus_di / 100
features['adx'] = adx / 100
features['roc_norm'] = (df['close'] / df['close'].shift(10) - 1)  # 10-bar ROC, normalized

# Volatility
tr1 = df['high'] - df['low']
tr2 = abs(df['high'] - df['close'].shift(1))
tr3 = abs(df['low'] - df['close'].shift(1))
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
atr = tr.ewm(span=14, adjust=False).mean()
features['atr_norm'] = atr / df['close']

sma_bb = df['close'].rolling(window=20).mean()
std_bb = df['close'].rolling(window=20).std()
upper_band = sma_bb + 2 * std_bb
lower_band = sma_bb - 2 * std_bb
band_diff = upper_band - lower_band
features['bb_pct_b'] = np.where(band_diff > 0, (df['close'] - lower_band) / band_diff, 0.5)
features['bb_bandwidth'] = band_diff / sma_bb.replace(0, np.finfo(float).eps)
features['realized_vol'] = features['log_return'].rolling(window=20).std() * np.sqrt(252 * 390)  # Annualized

# Volume-weighted
features['mfi_norm'] = (compute_mfi(df, 14) - 50) / 50

# Temporal gap
time_diff = df['timestamp'].diff().dt.total_seconds() / 60
features['time_gap'] = np.log1p(np.maximum(time_diff - 1, 0))

# Add raw OHLCV
features['open'] = df['open']
features['high'] = df['high']
features['low'] = df['low']
features['close'] = df['close']
features['volume'] = df['volume']

print("Computing targets...")

# Targets
targets = pd.DataFrame()
targets['timestamp'] = df['timestamp']
for h in [5, 15, 30, 60, 120, 240]:
    targets[f'target_{h}m'] = np.log(df['close'].shift(-h) / df['close'])

# Save
print("Saving to parquet...")
features.to_parquet(FEATURES_PATH, engine='pyarrow', compression='snappy')
targets.to_parquet(TARGETS_PATH, engine='pyarrow', compression='snappy')

# Compute and save normalization statistics
# Per grok-scientific.md Section 3.1: V=24 features (23 derived + 1 gap)
feature_cols = [
    # Price dynamics (F_P) - 4
    'log_return', 'hl_range', 'close_location', 'open_return',
    # Volume (F_V) - 3
    'log_volume', 'log_volume_delta', 'dollar_volume',
    # Trend (F_T) - 5
    'vwap_deviation', 'macd_histogram', 'sma_20_dev', 'sma_50_dev', 'sma_200_dev',
    # Momentum (F_M) - 6
    'rsi_norm', 'cci', 'plus_di', 'minus_di', 'adx', 'roc_norm',
    # Volatility (F_σ) - 4
    'atr_norm', 'bb_pct_b', 'bb_bandwidth', 'realized_vol',
    # Volume-weighted (F_VW) - 1
    'mfi_norm',
    # Temporal - 1
    'time_gap'
]  # Total: 24 features

stats = {
    'means': features[feature_cols].mean().tolist(),
    'stds': features[feature_cols].std().tolist(),
    'columns': feature_cols
}
with open(STATS_PATH, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\nFeatures saved to: {FEATURES_PATH}")
print(f"Targets saved to: {TARGETS_PATH}")
print(f"Stats saved to: {STATS_PATH}")
print(f"\nFeature shape: {features.shape}")
print(f"Target shape: {targets.shape}")
print(f"\nFeature columns: {feature_cols}")
```

#### 1.4 Validation Criteria

| Criterion | Target | Validation Method |
|-----------|--------|-------------------|
| Data completeness | >5M rows | `len(df)` check |
| Date coverage | 2010-06-06 to 2025-12-03 | `df['timestamp'].min/max()` |
| OHLC validity | <0.1% invalid bars | Relationship checks |
| Feature NaN rate | <1% after warmup | `df.isna().mean()` |
| Parquet size | <500MB per file | File size check |

---

### Phase 2: Dataset and Baseline Model

**Duration:** 2 weeks  
**Objective:** Create PyTorch Dataset, DataLoader, train vanilla Transformer baseline

#### Key Deliverables
- `NQFuturesDataset` class with proper windowing
- Custom `WindowSampler` for non-overlapping 1-week windows
- Vanilla Transformer encoder with independent quantile heads (no TSA/LGU)
- MSE loss baseline before switching to Pinball

#### Validation Criteria
- Training completes without OOM on A100 (VRAM <40GB)
- Validation IC > 0.02 for 5m horizon
- Training loss decreases monotonically
- DataLoader throughput >500 samples/sec

---

### Phase 3: Two-Stage Attention and LGU

**Duration:** 3 weeks  
**Objective:** Implement full TSA + LGU architecture per grok-scientific.md

#### Key Deliverables
- `TemporalAttention` module with FlashAttention
- `VariableAttention` module for cross-variable interaction
- `LiteGateUnit` with proper bias initialization
- `TSABlock` combining all components

#### Validation Criteria
- VRAM usage <60GB with batch_size=8
- Validation CRPS drops >15% vs Phase 2 baseline
- Gradient norms stable (no explosions/vanishing)
- Ablation: TSA vs vanilla attention shows measurable IC improvement

---

### Phase 4: Training Infrastructure

**Duration:** 1 week  
**Objective:** Production-ready training with mixed precision, checkpointing, logging

#### Key Deliverables
- Mixed precision training (BF16/TF32 on A100)
- WandB integration for experiment tracking
- Checkpoint save/restore with optimizer state
- Gradient checkpointing as OOM fallback
- `torch.compile` integration (PyTorch 2.0+)

#### Validation Criteria
- Training throughput >100 samples/sec
- Checkpoint save/load verified (resume training works)
- WandB dashboard shows loss curves, IC metrics
- torch.compile provides >20% speedup

---

### Phase 5: Multi-Horizon Autoregressive Heads

**Duration:** 1 week  
**Objective:** Implement autoregressive quantile heads with teacher forcing

#### Key Deliverables
- `QuantileHead` for single-horizon prediction
- `MultiHorizonHead` with autoregressive conditioning
- Teacher forcing in training, predicted median in inference
- Monotonicity loss penalty

#### Validation Criteria
- All 6 horizons produce valid predictions
- Monotonicity loss <0.01 at convergence
- Long-horizon (2h, 4h) IC > 0.03
- No quantile crossing in >95% of predictions

---

### Phase 6: Evaluation and Optimization

**Duration:** 2 weeks  
**Objective:** Comprehensive evaluation, ablations, hyperparameter tuning

#### Key Deliverables
- Full evaluation suite (CRPS, IC, tail accuracy, calibration)
- Ablation studies for H1-H6 hypotheses from grok-scientific.md
- Optuna hyperparameter optimization
- Rolling backtest simulation
- Optional: ONNX export for deployment

#### Metrics to Compute
- CRPS per horizon (primary metric)
- IC (Spearman) per horizon
- Tail-weighted accuracy (top/bottom 10% of returns)
- Quantile calibration plots
- Sharpe ratio proxy from naive signal

#### Validation Criteria
- IC > 0.05 for 5m horizon
- CRPS < baseline - 25%
- Tail accuracy > 55%
- Calibration error < 5% across quantiles

---

## 5. Configuration Reference

### 5.1 Model Hyperparameters

```yaml
# config/model_config.yaml
model:
  d_model: 512
  num_heads: 8
  num_layers: 8
  ffn_dim: 2048
  dropout: 0.1
  max_len: 7000

training:
  batch_size: 8
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_steps: 2000
  max_epochs: 100
  gradient_clip: 1.0
  use_amp: true
  use_wandb: true
  use_compile: true  # PyTorch 2.0+

data:
  train_end_date: "2020-01-01"  # Train: 2010-2019
  val_end_date: "2023-01-01"    # Val: 2020-2022, Test: 2023-2025
  stride: 60
  num_workers: 12  # Match A100 CPU cores
  prefetch_factor: 2

optimization:
  use_optuna: false  # Enable for Phase 6
  n_trials: 50
  
quantiles: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
horizons: [5, 15, 30, 60, 120, 240]
```

### 5.2 Feature Groups (for TSA)

```python
# Per grok-scientific.md Section 3.1: V=24 features, 6 groups
FEATURE_GROUPS = {
    'price': [0, 1, 2, 3],           # log_return, hl_range, close_location, open_return (4)
    'volume': [4, 5, 6],             # log_volume, log_volume_delta, dollar_volume (3)
    'trend': [7, 8, 9, 10, 11],      # vwap, macd, sma_20/50/200 devs (5)
    'momentum': [12, 13, 14, 15, 16, 17],  # rsi, cci, +di, -di, adx, roc (6)
    'volatility': [18, 19, 20, 21],  # atr, bb_pct_b, bandwidth, realized_vol (4)
    'flow': [22, 23],                # mfi, time_gap (2)
}
# Total: 4+3+5+6+4+2 = 24 features
```

---

## 6. References

[1] F. Gu et al., "MIGT: Memory Instance Gated Transformer Framework for Financial Portfolio Management," arXiv:2502.07280, 2025.

[2] Y. Li et al., "Reinforcement Learning with Temporal and Variable Dependency-aware Transformer for Stock Trading Optimization," Neural Networks, vol. 192, 2025.

[3] Databento Python Client Documentation. [Online]. Available: https://databento.com/docs

[4] PyTorch Documentation: scaled_dot_product_attention. [Online]. Available: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

[5] W. Dabney et al., "Distributional Reinforcement Learning with Quantile Regression," AAAI, 2018.

---

## Appendix A: Data Copy Script for Colab VM

```python
# Copy data from Drive to VM for fast access
import shutil
from pathlib import Path

DRIVE_DATA = Path('/content/drive/MyDrive/Colab Notebooks/Transformers/FP/data')
LOCAL_DATA = Path('/content/data')

def copy_to_vm():
    """Copy data from Drive to VM for faster training."""
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)
    
    # Copy processed data
    for src in DRIVE_DATA.glob('processed/*.parquet'):
        dst = LOCAL_DATA / src.name
        if not dst.exists():
            print(f"Copying {src.name}...")
            shutil.copy(src, dst)
            
    # Copy metadata
    for src in DRIVE_DATA.glob('metadata/*.json'):
        dst = LOCAL_DATA / src.name
        if not dst.exists():
            print(f"Copying {src.name}...")
            shutil.copy(src, dst)
            
    print("Data copied to VM!")
    return LOCAL_DATA

# Usage: local_path = copy_to_vm()
```

---

## Appendix B: VRAM Management and OOM Handling

Per Gemini's recommendation, gradient checkpointing provides OOM fallback for large sequences.

```python
# src/model/checkpointing.py
"""
Gradient checkpointing utilities for memory-efficient training.
Use when batch_size=8 causes OOM on A100.
"""
import torch
from torch.utils.checkpoint import checkpoint

def enable_gradient_checkpointing(model):
    """
    Enable gradient checkpointing on TSA blocks.
    Trades ~30% compute for ~40% memory reduction.
    """
    for block in model.tsa_blocks:
        block.gradient_checkpointing = True
        
    # Override forward to use checkpointing
    original_forward = model.forward
    
    def checkpointed_forward(features, mask, temporal_features, **kwargs):
        # Checkpoint the heavy TSA blocks
        x = model.instance_norm(features, mask)
        x = model.group_proj(x)
        pe = model.pos_encoding(temporal_features)
        x = x + pe.unsqueeze(2)
        
        # Checkpoint each TSA block
        for block in model.tsa_blocks:
            x = checkpoint(block, x, mask, use_reentrant=False)
            
        # Continue with rest of forward pass (not checkpointed)
        # ... (pooling and heads)
        return x
    
    return model


# Usage in training:
# if OOM:
#     model = enable_gradient_checkpointing(model)
#     batch_size = 4  # Can also reduce batch size
```

### VRAM Budget Estimates (A100 80GB)

| Configuration | Estimated VRAM | Notes |
|--------------|----------------|-------|
| batch_size=8, FP32, no checkpoint | ~45-55GB | FP32 baseline (Phase 2 initial) |
| batch_size=8, BF16, no checkpoint | ~25-35GB | **With AMP (recommended)** |
| batch_size=8, BF16, with checkpoint | ~20-25GB | Maximum efficiency |
| batch_size=16, BF16, no checkpoint | ~45-60GB | Larger batch with AMP |
| batch_size=4, BF16, no checkpoint | ~15-20GB | Conservative config |

**Phase 2 actual measurements (BF16):**
- Inference: ~25-30 GB
- Training: ~30-40 GB

---

## Appendix C: Optuna Hyperparameter Search (Phase 6)

```python
# src/optimization/hyperparameter_search.py
"""
Optuna hyperparameter optimization for model tuning.
Per Grok's recommendation.
"""
import optuna
from optuna.pruners import MedianPruner

def objective(trial):
    # Sample hyperparameters
    config = {
        'd_model': trial.suggest_categorical('d_model', [256, 512, 768]),
        'num_layers': trial.suggest_int('num_layers', 6, 12),
        'num_heads': trial.suggest_categorical('num_heads', [4, 8, 16]),
        'dropout': trial.suggest_float('dropout', 0.05, 0.2),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-4, 1e-1),
    }
    
    # Create model and train for N epochs
    model = GrokSLTransformer(**config)
    trainer = Trainer(model, train_loader, val_loader, **config)
    
    for epoch in range(10):  # Quick validation
        trainer.train_epoch(epoch)
        metrics = trainer.validate()
        
        # Report for pruning
        trial.report(metrics['val_loss'], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return metrics['val_loss']

# Run optimization
study = optuna.create_study(
    direction='minimize',
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
)
study.optimize(objective, n_trials=50)

print(f"Best params: {study.best_params}")
```

---

*Document prepared by Claude (Engineering Lead)*  
*Version 1.1 - Incorporated Grok/Gemini peer review improvements*

**Key Improvements from Peer Review:**
- Grok: Better train/val/test splits (2010-2019/2020-2022/2023-2025), WandB logging, Optuna hyperparameter search, realistic phase timelines (~10 weeks)
- Gemini: Symbology clarification (NQ.v.0 vs NQ.c.0), gradient checkpointing for OOM, torch.compile optimization, Polars suggestion