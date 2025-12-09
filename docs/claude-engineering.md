# claude-engineering.md: Engineering Implementation Guide for Distributional Supervised Learning on NASDAQ Futures

## 1. Executive Summary

This document provides the engineering implementation guide for the NQ futures distributional forecasting model. It translates the scientific concepts from `grok-scientific.md` into actionable architecture designs, specifications, and a phased development plan suitable for execution on Google Colab with an A100 GPU (80GB VRAM, 167.1GB system RAM, 12 CPU cores).

The implementation follows a modular design pattern enabling independent development, testing, and iteration of each component. All code will be developed from this specification document.

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE / PREDICTION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────────┐    │
│  │ Raw 5-min   │───▶│  Preprocessing  │───▶│  Feature Engineering     │    │
│  │ OHLCV Data  │    │  (RevIN + Pad)  │    │  (Derived Features)      │    │
│  └─────────────┘    └─────────────────┘    └──────────────────────────┘    │
│                                                      │                      │
│                                                      ▼                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     HYBRID MIGT-TVDT MODEL                           │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │  │
│  │  │ Input Embedding│─▶│ Temporal       │─▶│ Variable Attention     │ │  │
│  │  │ + Positional   │  │ Attention      │  │ + Gated InstanceNorm   │ │  │
│  │  │ Encodings      │  │ (per variable) │  │                        │ │  │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘ │  │
│  │                                                      │               │  │
│  │                                                      ▼               │  │
│  │  ┌──────────────────────────────────────────────────────────────┐   │  │
│  │  │              Multi-Horizon Quantile Heads                    │   │  │
│  │  │  [15m] [30m] [60m] [2h] [4h] × [τ=0.05,0.1,0.25,0.5,0.75,0.9,0.95] │  │
│  │  └──────────────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                      │                      │
│                                                      ▼                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     OUTPUT: Quantile Distributions                   │  │
│  │       Per horizon: 7 quantiles → calibrated prediction intervals     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Dependency Graph

```
data_loader.py
        │
        ▼
data_preprocessing.py ──────────────────┐
        │                               │
        ▼                               ▼
feature_engineering.py          rollover_adjustment.py
        │
        ▼
dataset.py ◀──────────────────── positional_encodings.py
        │
        ▼
model/
├── embeddings.py
├── temporal_attention.py
├── variable_attention.py
├── gated_instance_norm.py
├── quantile_heads.py
└── migt_tvdt.py (main model)
        │
        ▼
training/
├── loss_functions.py
├── trainer.py
└── scheduler.py
        │
        ▼
evaluation/
├── metrics.py
├── calibration.py
└── backtest.py
```

---

## 3. Data Pipeline Specifications

### 3.1 Directory Structure

```
/content/drive/MyDrive/Colab Notebooks/Transformers/FP/
├── data/
│   ├── raw/
│   │   ├── nq_ohlcv_1m_raw.parquet            # Raw 1-min bars (USER PROVIDED)
│   │   └── rollover_dates.csv                  # Contract rollover metadata (generated)
│   ├── interim/
│   │   ├── nq_ohlcv_5min_aggregated.parquet   # Aggregated to 5-min bars
│   │   └── nq_ohlcv_5min_adjusted.parquet     # Ratio back-adjusted
│   └── processed/
│       ├── nq_features_full.parquet           # All derived features
│       ├── train_samples.parquet              # 2010-2021 (or .h5)
│       ├── val_samples.parquet                # 2022-2023
│       └── test_samples.parquet               # 2024-Dec 2025
├── credentials/
│   └── databento.txt                          # API credentials (GITIGNORE - not committed)
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── rollover_adjustment.py
│   │   ├── feature_engineering.py
│   │   ├── preprocessing.py
│   │   └── dataset.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── positional_encodings.py
│   │   ├── temporal_attention.py
│   │   ├── variable_attention.py
│   │   ├── gated_instance_norm.py
│   │   ├── quantile_heads.py
│   │   └── migt_tvdt.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── loss_functions.py
│   │   ├── trainer.py
│   │   ├── scheduler.py
│   │   └── hyperopt.py
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       ├── calibration.py
│       ├── backtest.py
│       └── inference.py
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_hyperopt.ipynb
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
└── tests/
    ├── test_data_pipeline.py
    ├── test_model_components.py
    └── test_training.py
```

### 3.2 Databento Data Acquisition

#### 3.2.1 Module: `data_loader.py`

**Purpose:** Download NQ front-month OHLCV bars from Databento or load existing data, then aggregate to 5-minute bars.

**API Configuration:**
- Credentials file: `/content/drive/MyDrive/Colab Notebooks/Transformers/FP/credentials/databento.txt`
  - Line 1: Username
  - Line 2: API key
  - **Note:** This file is in `.gitignore` and never committed to repository
- Dataset: `GLBX.MDP3` (CME Globex)
- Schema: `ohlcv-1m` (1-minute bars, aggregated to 5-min locally)
- Symbol: `NQ.v.0` (front-month with **volume-based** rollover)
- Symbol type: `stype_in="continuous"`
- Date range: 2010-06-06 to 2025-12-03

**Important:** Databento does not provide native 5-minute OHLCV bars. Per their documentation, download `ohlcv-1m` and aggregate to 5-minute bars locally.

**If you already have 1-minute data:** Skip the download step and proceed directly to aggregation. Place existing data in `/content/drive/MyDrive/Colab Notebooks/Transformers/FP/data/raw/nq_ohlcv_1m_raw.parquet`.

**Interface:**

```python
class DatabentoLoader:
    """
    Download and manage NQ futures data from Databento.
    
    Handles API authentication, data retrieval, and local caching.
    """
    
    def __init__(
        self,
        credentials_path: Path,
        output_dir: Path,
        cache_enabled: bool = True
    ):
        """
        Initialize the Databento loader.
        
        Args:
            credentials_path: Path to databento.txt (username line 1, API key line 2)
            output_dir: Directory for raw data storage
            cache_enabled: Whether to use local cache
        """
        self.credentials_path = credentials_path
        self.output_dir = output_dir
        self._load_credentials()
    
    def _load_credentials(self):
        """Load API credentials from file (never hardcode)."""
        with open(self.credentials_path) as f:
            lines = f.read().strip().split('\n')
            self.username = lines[0]
            self.api_key = lines[1]
    
    def download_ohlcv(
        self,
        symbol: str = "NQ.v.0",           # Volume-based rollover
        start_date: str = "2010-06-06",
        end_date: str = "2025-12-03",
        schema: str = "ohlcv-1m",          # 1-minute bars
        stype_in: str = "continuous"       # Continuous contract
    ) -> pd.DataFrame:
        """
        Download OHLCV data for the specified parameters.
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            Index: DatetimeIndex in UTC
        """
        import databento as db
        
        client = db.Historical(self.api_key)
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[symbol],
            schema=schema,
            stype_in=stype_in,
            start=start_date,
            end=end_date
        )
        return data.to_df()
    
    def aggregate_to_5min(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 1-minute data to 5-minute bars.
        
        Uses standard OHLCV aggregation:
        - open: first
        - high: max
        - low: min
        - close: last
        - volume: sum
        """
        df_5min = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return df_5min
    
    def get_rollover_metadata(self) -> pd.DataFrame:
        """
        Extract rollover dates and price ratios from continuous contract data.
        
        Detects rollovers by identifying price discontinuities that exceed
        normal volatility thresholds coinciding with contract transitions.
        
        Returns:
            DataFrame with columns: date, from_contract, to_contract, price_ratio
        """
        ...
```

**Output Schema (raw):**

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime64[ns, UTC] | Bar timestamp (start of 5-min period) |
| `open` | float64 | Opening price |
| `high` | float64 | High price |
| `low` | float64 | Low price |
| `close` | float64 | Closing price |
| `volume` | int64 | Volume traded |

### 3.3 Rollover Adjustment

#### 3.3.1 Module: `rollover_adjustment.py`

**Purpose:** Apply ratio back-adjustment to preserve log-return integrity across contract rolls.

**Algorithm:**

```
For each rollover event at time t_roll from Contract A to Contract B:
    1. Calculate ratio: R = Price_B(t_roll) / Price_A(t_roll)
    2. Multiply all historical prices (O, H, L, C) prior to t_roll by R
    3. Accumulate ratios for multiple rolls: P_adj = P_raw × Π(R_i)
```

**Interface:**

```python
class RolloverAdjuster:
    """
    Apply ratio back-adjustment for futures contract rolls.
    """
    
    def __init__(self, rollover_metadata: pd.DataFrame):
        """
        Args:
            rollover_metadata: DataFrame with rollover dates and ratios
        """
        ...
    
    def adjust_prices(
        self,
        df: pd.DataFrame,
        price_cols: List[str] = ["open", "high", "low", "close"]
    ) -> pd.DataFrame:
        """
        Apply cumulative ratio adjustment to price columns.
        
        Returns:
            DataFrame with adjusted prices, original volume preserved
        """
        ...
    
    def verify_returns_preserved(
        self,
        original: pd.DataFrame,
        adjusted: pd.DataFrame,
        tolerance: float = 1e-10
    ) -> bool:
        """
        Verify that log-returns are preserved after adjustment.
        """
        ...
```

### 3.4 Feature Engineering

#### 3.4.1 Module: `feature_engineering.py`

**Purpose:** Compute all derived features specified in the scientific document. All computations are strictly causal (use only past data).

**Feature Specifications:**

| Feature | Formula | Rolling Window | Notes |
|---------|---------|----------------|-------|
| `log_return` | ln(C_t / C_{t-1}) | 1 | Target computation base |
| `gk_volatility` | 0.5(ln H - ln L)² - (2ln2-1)(ln C - ln O)² | 1 | Garman-Klass estimator |
| `rv_12` | √(Σ r²) over 12 bars | 12 | ~1 hour realized vol |
| `rv_36` | √(Σ r²) over 36 bars | 36 | ~3 hour realized vol |
| `rv_72` | √(Σ r²) over 72 bars | 72 | ~6 hour realized vol |
| `amihud_illiq` | \|r_t\| / (P_t × V_t) | 1 | Illiquidity proxy |
| `rsi_14` | 100 - 100/(1 + RS) | 14 | Wilder's RSI |
| `macd` | EMA_12 - EMA_26 | 26 | Momentum |
| `macd_signal` | EMA_9(MACD) | 9 | Signal line |
| `roc_5` | (C_t - C_{t-5}) / C_{t-5} | 5 | Short-term momentum |
| `roc_10` | (C_t - C_{t-10}) / C_{t-10} | 10 | Rate of change |
| `roc_20` | (C_t - C_{t-20}) / C_{t-20} | 20 | Medium-term momentum |
| `ema_slope_9` | EMA_9(t) - EMA_9(t-1) | 9 | Trend slope |
| `ema_slope_21` | EMA_21(t) - EMA_21(t-1) | 21 | Trend slope |
| `ema_slope_50` | EMA_50(t) - EMA_50(t-1) | 50 | Trend slope |
| `ema_dev_9` | (C_t - EMA_9) / EMA_9 | 9 | Deviation from EMA |
| `ema_dev_21` | (C_t - EMA_21) / EMA_21 | 21 | Deviation from EMA |
| `ema_dev_50` | (C_t - EMA_50) / EMA_50 | 50 | Deviation from EMA |
| `atr_14` | SMA_14(TR) | 14 | Average true range |
| `log_volume` | ln(V_t + 1) | 1 | Log-transformed volume |
| `volume_ma_ratio` | V_t / SMA_20(V) | 20 | Volume relative to average |

**Data Cleaning Requirements:**
- Filter invalid ticks: Remove bars with zero price or zero volume
- Handle trading halts: Forward-fill for gaps ≤15 minutes, mask for longer gaps
- Detect anomalies: Flag bars with returns > 5 standard deviations for review

**Note on Feature Selection:**
The feature list above extends the scientific document's explicit list with:
- `macd_signal` (EMA_9 of MACD) - standard MACD interpretation requires signal line
- `volume_ma_ratio` - normalized volume for cross-regime comparability  
- `log_return` - explicit inclusion for target computation and standardization

These additions align with the scientific document's intent (Section 3.2) while maintaining low multicollinearity. If ablation shows no value, they can be removed.

**Interface:**

```python
class FeatureEngineer:
    """
    Compute derived features from OHLCV data.
    
    All features are computed causally (no lookahead bias).
    Supports both Pandas and Polars backends for efficiency.
    """
    
    def __init__(self, config: Dict[str, Any], backend: str = "polars"):
        """
        Args:
            config: Feature computation parameters (windows, etc.)
            backend: "pandas" or "polars" (polars recommended for large datasets)
        """
        ...
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all derived features.
        
        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
        
        Returns:
            DataFrame with original columns + all derived features
        """
        ...
    
    def compute_targets(
        self,
        df: pd.DataFrame,
        horizons: List[int] = [3, 6, 12, 24, 48]  # in 5-min bars
    ) -> pd.DataFrame:
        """
        Compute forward log-returns for each horizon.
        
        Horizons: 15m=3, 30m=6, 60m=12, 2h=24, 4h=48 bars
        
        Returns:
            DataFrame with target columns: target_15m, target_30m, etc.
        """
        ...
    
    @staticmethod
    def garman_klass_volatility(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """Single-bar Garman-Klass volatility estimator."""
        ...
    
    @staticmethod
    def amihud_illiquidity(
        returns: pd.Series,
        price: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Amihud illiquidity measure."""
        ...
```

### 3.5 Preprocessing & Dataset

#### 3.5.1 Module: `preprocessing.py`

**Purpose:** Prepare data windows for model input, including padding and attention mask generation.

**Specifications:**
- Maximum sequence length: T_max = 288 (theoretical max 5-min bars in 24h)
- Valid sequence lengths: 273-276 bars (actual trading day lengths)
- Padding strategy: Zero-padding at sequence end
- Attention mask: Boolean tensor where True = valid, False = padding

**Interface:**

```python
class DataPreprocessor:
    """
    Prepare input windows for the model.
    """
    
    def __init__(
        self,
        max_seq_len: int = 288,
        feature_cols: List[str] = None,
        target_cols: List[str] = None
    ):
        ...
    
    def create_window(
        self,
        df: pd.DataFrame,
        end_timestamp: pd.Timestamp
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Create a single 24-hour lookback window.
        
        Args:
            df: Full feature DataFrame
            end_timestamp: Current bar timestamp
        
        Returns:
            features: (T_max, V) array, zero-padded
            attention_mask: (T_max,) boolean array
            metadata: dict with actual_length, timestamps, etc.
        """
        ...
    
    def normalize_window(
        self,
        features: np.ndarray,
        attention_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply instance normalization (RevIN) to valid positions only.
        
        Returns:
            normalized_features: (T_max, V) array
            norm_stats: dict with mean, std per variable for denormalization
        """
        ...
```

#### 3.5.2 Module: `dataset.py`

**Purpose:** PyTorch Dataset implementation for efficient data loading.

**Interface:**

```python
class NQFuturesDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for NQ futures prediction.
    """
    
    def __init__(
        self,
        data_path: Path,
        feature_cols: List[str],
        target_cols: List[str],
        max_seq_len: int = 288,
        transform: Optional[Callable] = None
    ):
        ...
    
    def __len__(self) -> int:
        ...
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                - features: (T_max, V) float tensor
                - attention_mask: (T_max,) bool tensor
                - targets: (H,) float tensor for H horizons
                - timestamps: temporal info for positional encoding
                - norm_stats: for output denormalization
        """
        ...


class NQDataModule:
    """
    Data module handling train/val/test splits and dataloaders.
    """
    
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        ...
    
    def setup(self):
        """Load and prepare all datasets."""
        ...
    
    def train_dataloader(self) -> DataLoader:
        ...
    
    def val_dataloader(self) -> DataLoader:
        ...
    
    def test_dataloader(self) -> DataLoader:
        ...
```

---

## 4. Model Architecture Specifications

### 4.1 Configuration Schema

```yaml
# model_config.yaml
model:
  d_model: 256              # Model dimension
  n_heads: 8                # Attention heads
  n_temporal_layers: 4      # Temporal attention depth
  n_variable_layers: 2      # Variable attention depth
  d_ff: 1024               # Feed-forward dimension
  dropout: 0.1              # Dropout rate
  max_seq_len: 288          # Maximum sequence length
  n_variables: 25           # Number of input variables (see Appendix A)
  n_horizons: 5             # Prediction horizons
  n_quantiles: 7            # Quantiles per horizon

positional_encoding:
  time_of_day:
    type: "sinusoidal"
    dim: 32
  day_of_week:
    type: "learnable"
    dim: 16
  day_of_month:
    type: "time2vec"
    dim: 16
  day_of_year:
    type: "time2vec"
    dim: 32

quantile_regression:
  quantiles: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
  crossing_penalty: 0.1     # Penalty for quantile crossing
```

### 4.2 Positional Encoding Module

#### 4.2.1 Module: `positional_encodings.py`

**CRITICAL: Tensor Broadcasting (The "4D Trap")**

Variable embedding produces 4D tensors `(B, T, V, D)`. Positional encodings produce lower-dimensional tensors that must be broadcast correctly:

```python
# Time-of-day encoding: (B, T, D) → (B, T, 1, D) to broadcast across Variables
time_enc = time_enc.unsqueeze(2)

# Day-of-week encoding: (B, D) → (B, 1, 1, D) to broadcast across Time and Variables  
dow_enc = dow_enc.unsqueeze(1).unsqueeze(1)

# Combined: embedded_input + time_enc + dow_enc  (all broadcast to B, T, V, D)
```

**Interface:**

```python
class TimeOfDayEncoding(nn.Module):
    """
    Sinusoidal encoding for intraday position.
    
    PE(t) = [sin(2πt/288), cos(2πt/288), ...]
    Ensures 23:55 is close to 00:00 in embedding space.
    """
    
    def __init__(self, d_model: int):
        ...
    
    def forward(self, bar_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bar_indices: (B, T) integer tensor, 0-287 for bar position in day
        
        Returns:
            (B, T, d_model) encoding tensor
            NOTE: Caller must unsqueeze(2) before adding to 4D variable embeddings
        """
        ...


class DayOfWeekEncoding(nn.Module):
    """
    Learnable embedding for day of week (0=Monday, 4=Friday).
    
    Markets behave differently on different days (e.g., OpEx Fridays).
    """
    
    def __init__(self, d_model: int):
        ...
    
    def forward(self, day_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            day_indices: (B,) integer tensor, 0-4
        
        Returns:
            (B, d_model) encoding, broadcast across time dimension
        """
        ...


class Time2VecEncoding(nn.Module):
    """
    Learnable periodic encoding (Kazemi et al., 2019).
    
    T2V(τ)[0] = ω₀τ + φ₀  (linear term)
    T2V(τ)[i] = sin(ωᵢτ + φᵢ) for i ≥ 1  (periodic terms)
    
    Learns non-obvious cycles (e.g., quarterly rebalancing).
    """
    
    def __init__(self, d_model: int, n_frequencies: int = None):
        ...
    
    def forward(self, time_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_values: (B, T) or (B,) normalized time values
        
        Returns:
            (B, T, d_model) or (B, d_model) encoding tensor
        """
        ...


class CompositePositionalEncoding(nn.Module):
    """
    Combines all positional encodings into unified representation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        ...
    
    def forward(
        self,
        bar_in_day: torch.Tensor,      # (B, T) 0-287
        day_of_week: torch.Tensor,     # (B,) 0-4
        day_of_month: torch.Tensor,    # (B,) 1-31
        day_of_year: torch.Tensor      # (B,) 1-366
    ) -> torch.Tensor:
        """
        Returns:
            (B, T, d_pos) combined positional encoding
        """
        ...
```

### 4.3 Input Embedding

#### 4.3.1 Module: `embeddings.py`

**Interface:**

```python
class VariableEmbedding(nn.Module):
    """
    Project each variable's time series to model dimension.
    
    Per scientific document: Each variable gets its own linear projection,
    preserving the time dimension for temporal attention.
    
    Input: (B, T, V) raw features
    Output: (B, T, V, D) embedded features
    """
    
    def __init__(
        self,
        n_variables: int,
        d_model: int,
        dropout: float = 0.1
    ):
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, V) input features
        
        Returns:
            (B, T, V, D) variable embeddings
        """
        ...


class InputEmbedding(nn.Module):
    """
    Complete input embedding combining variable projection and positional encoding.
    
    CRITICAL BROADCASTING NOTE:
    Variable embedding produces (B, T, V, D) 4D tensor. Positional encodings must
    be unsqueezed for proper broadcasting:
    - Time-of-day: (B, T, D) → unsqueeze(2) → (B, T, 1, D) broadcasts across V
    - Day-of-week: (B, D) → unsqueeze(1).unsqueeze(2) → (B, 1, 1, D) broadcasts across T and V
    - Time2Vec: (B, T, D) → unsqueeze(2) → (B, T, 1, D) broadcasts across V
    """
    
    def __init__(
        self,
        n_variables: int,
        d_model: int,
        positional_config: Dict[str, Any],
        dropout: float = 0.1
    ):
        ...
    
    def forward(
        self,
        features: torch.Tensor,        # (B, T, V)
        temporal_info: Dict[str, torch.Tensor]  # positional encoding inputs
    ) -> torch.Tensor:
        """
        Returns:
            (B, T, V, D) embedded input with positional information
        
        Implementation notes:
            # Variable embedding: (B, T, V) → (B, T, V, D)
            x = self.variable_embed(features)
            
            # Positional encodings with proper broadcasting
            tod_enc = self.time_of_day(temporal_info['bar_in_day'])  # (B, T, D)
            dow_enc = self.day_of_week(temporal_info['day_of_week'])  # (B, D)
            t2v_enc = self.time2vec(temporal_info['day_of_year'])     # (B, T, D)
            
            # Unsqueeze for broadcasting to (B, T, V, D)
            x = x + tod_enc.unsqueeze(2)               # (B, T, 1, D)
            x = x + dow_enc.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, D)
            x = x + t2v_enc.unsqueeze(2)               # (B, T, 1, D)
            
            return self.dropout(x)
        """
        ...
```

### 4.4 Two-Stage Attention

#### 4.4.1 Module: `temporal_attention.py`

**Interface:**

```python
class TemporalAttentionBlock(nn.Module):
    """
    Self-attention over time dimension, applied independently per variable.
    
    Per scientific document (RL-TVDT): Learns temporal dynamics (trends,
    seasonality) of each variable separately before cross-variable interaction.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        ...
    
    def forward(
        self,
        x: torch.Tensor,               # (B, T, V, D)
        attention_mask: torch.Tensor   # (B, T) boolean mask
    ) -> torch.Tensor:
        """
        Apply temporal attention independently to each variable.
        
        Returns:
            (B, T, V, D) temporally-attended features
        """
        ...


class TemporalAggregation(nn.Module):
    """
    Aggregate temporal sequence into fixed-size representation per variable.
    
    Uses attention-weighted pooling with learnable query (bidirectional
    over historical lookback, per scientific document justification).
    """
    
    def __init__(self, d_model: int, n_heads: int = 1):
        ...
    
    def forward(
        self,
        x: torch.Tensor,               # (B, T, V, D)
        attention_mask: torch.Tensor   # (B, T)
    ) -> torch.Tensor:
        """
        Returns:
            (B, V, D) aggregated variable representations
        """
        ...
```

#### 4.4.2 Module: `variable_attention.py`

**Interface:**

```python
class VariableAttentionBlock(nn.Module):
    """
    Self-attention over variable dimension.
    
    Per scientific document: Learns inter-variable correlations (e.g., how
    volume shocks impact price movements, RSI-Volume relationships).
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, V, D) aggregated variable representations
        
        Returns:
            (B, V, D) cross-variable attended features
        """
        ...
```

### 4.5 Gated Instance Normalization

#### 4.5.1 Module: `gated_instance_norm.py`

**Interface:**

```python
class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    
    Applied to raw input before embedding; reversed after decoder
    to restore original scale.
    """
    
    def __init__(self, n_variables: int, eps: float = 1e-5, affine: bool = True):
        ...
    
    def forward(
        self,
        x: torch.Tensor,
        mode: str = "normalize"  # "normalize" or "denormalize"
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, V) input tensor
            mode: "normalize" stores stats and normalizes,
                  "denormalize" restores original scale
        
        Returns:
            Normalized or denormalized tensor
        """
        ...


class LiteGateUnit(nn.Module):
    """
    Lite Gate Unit from MIGT paper.
    
    Filters attention output to suppress noisy signals and pass
    through only high-confidence feature updates.
    
    G = σ(Wx + b)
    output = x + G ⊙ attention(x)
    """
    
    def __init__(self, d_model: int):
        ...
    
    def forward(
        self,
        x: torch.Tensor,               # Pre-attention input
        attention_output: torch.Tensor  # Post-attention output
    ) -> torch.Tensor:
        """
        Returns:
            Gated residual output
        """
        ...


class GatedInstanceNorm(nn.Module):
    """
    Combined instance normalization with gating (MIGT-style).
    
    Applied post-attention as specified in scientific document.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        ...
    
    def forward(
        self,
        x: torch.Tensor,
        attention_output: torch.Tensor
    ) -> torch.Tensor:
        ...
```

### 4.6 Output Heads

#### 4.6.1 Module: `quantile_heads.py`

**Interface:**

```python
class QuantileHead(nn.Module):
    """
    Quantile regression head for a single horizon.
    
    Outputs non-crossing quantiles using cumulative softplus parameterization:
    q_τ = q_min + Σ softplus(δ_i) for i up to τ
    
    Implementation: Use torch.cumsum(F.softplus(deltas), dim=-1) to enforce
    strict monotonicity without quantile crossing.
    """
    
    def __init__(
        self,
        d_model: int,
        n_quantiles: int = 7,
        hidden_dim: int = 128
    ):
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) pooled representation
        
        Returns:
            (B, n_quantiles) non-crossing quantile predictions
        
        Implementation notes:
            # Project to get base value and deltas
            hidden = self.mlp(x)                      # (B, hidden_dim)
            base = self.base_proj(hidden)             # (B, 1)
            deltas = self.delta_proj(hidden)          # (B, n_quantiles)
            
            # Cumulative softplus ensures monotonicity
            cumulative = torch.cumsum(F.softplus(deltas), dim=-1)  # (B, n_quantiles)
            quantiles = base + cumulative             # (B, n_quantiles)
            return quantiles
        """
        ...


class MultiHorizonQuantileHead(nn.Module):
    """
    Multi-horizon quantile output with horizon embeddings.
    
    Shared representation with horizon-specific MLPs.
    """
    
    def __init__(
        self,
        d_model: int,
        n_horizons: int = 5,
        n_quantiles: int = 7,
        hidden_dim: int = 128
    ):
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) pooled representation from encoder
        
        Returns:
            (B, n_horizons, n_quantiles) quantile predictions
        """
        ...
```

### 4.7 Complete Model

#### 4.7.1 Module: `migt_tvdt.py`

**Interface:**

```python
class MIGT_TVDT(nn.Module):
    """
    Hybrid MIGT-TVDT model for distributional NQ futures forecasting.
    
    Architecture flow:
    1. RevIN normalization on raw input
    2. Variable embedding + positional encoding
    3. Temporal attention (per variable) + aggregation
    4. Variable attention (cross-variable) + gating
    5. Multi-horizon quantile output
    6. RevIN denormalization
    
    Implements specifications from scientific document Sections 4.1-4.2.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        
        # RevIN for input/output normalization
        self.revin = RevIN(config['n_variables'])
        
        # Input embedding
        self.input_embedding = InputEmbedding(
            n_variables=config['n_variables'],
            d_model=config['d_model'],
            positional_config=config['positional_encoding'],
            dropout=config['dropout']
        )
        
        # Temporal attention stack
        self.temporal_layers = nn.ModuleList([
            TemporalAttentionBlock(
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                d_ff=config['d_ff'],
                dropout=config['dropout']
            )
            for _ in range(config['n_temporal_layers'])
        ])
        
        # Temporal aggregation
        self.temporal_aggregation = TemporalAggregation(
            d_model=config['d_model']
        )
        
        # Variable attention stack with gating
        self.variable_layers = nn.ModuleList([
            VariableAttentionBlock(
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                d_ff=config['d_ff'],
                dropout=config['dropout']
            )
            for _ in range(config['n_variable_layers'])
        ])
        
        self.gated_norms = nn.ModuleList([
            GatedInstanceNorm(config['d_model'])
            for _ in range(config['n_variable_layers'])
        ])
        
        # Output pooling and quantile heads
        self.output_pool = nn.Linear(
            config['n_variables'] * config['d_model'],
            config['d_model']
        )
        
        self.quantile_head = MultiHorizonQuantileHead(
            d_model=config['d_model'],
            n_horizons=config['n_horizons'],
            n_quantiles=config['n_quantiles']
        )
    
    def forward(
        self,
        features: torch.Tensor,              # (B, T, V)
        attention_mask: torch.Tensor,        # (B, T)
        temporal_info: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Returns:
            dict with:
                - quantiles: (B, H, Q) quantile predictions
                - norm_stats: RevIN statistics for potential denormalization
        """
        # 1. RevIN normalize
        x = self.revin(features, mode="normalize")
        
        # 2. Embed + positional encoding
        x = self.input_embedding(x, temporal_info)  # (B, T, V, D)
        
        # 3. Temporal attention per variable
        for layer in self.temporal_layers:
            x = layer(x, attention_mask)
        
        # 4. Aggregate time dimension
        x = self.temporal_aggregation(x, attention_mask)  # (B, V, D)
        
        # 5. Variable attention with gating
        for layer, gated_norm in zip(self.variable_layers, self.gated_norms):
            attn_out = layer(x)
            x = gated_norm(x, attn_out)
        
        # 6. Pool and predict
        x = x.flatten(start_dim=1)  # (B, V*D)
        x = self.output_pool(x)     # (B, D)
        
        quantiles = self.quantile_head(x)  # (B, H, Q)
        
        return {
            'quantiles': quantiles,
            'norm_stats': self.revin.get_stats()
        }
    
    @classmethod
    def from_config(cls, config_path: Path) -> 'MIGT_TVDT':
        """Load model from YAML configuration file."""
        ...
    
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

**Estimated Model Size:**

| Component | Parameters (d_model=256) |
|-----------|-------------------------|
| Variable Embedding (23 vars) | ~150K |
| Positional Encodings | ~50K |
| Temporal Attention (4 layers) | ~2.1M |
| Temporal Aggregation | ~65K |
| Variable Attention (2 layers) | ~1.0M |
| Gated Norms | ~130K |
| Output Pool + Heads | ~500K |
| **Total** | **~4M** |

Memory footprint estimate: ~16-20GB peak VRAM with batch size 128 (well within A100 80GB capacity).

---

## 5. Training Pipeline Specifications

### 5.1 Loss Functions

#### 5.1.1 Module: `loss_functions.py`

**Interface:**

```python
class PinballLoss(nn.Module):
    """
    Quantile regression loss (pinball loss).
    
    L_τ(y, ŷ_τ) = τ(y - ŷ_τ)₊ + (1-τ)(ŷ_τ - y)₊
    
    Optimizes model to find quantiles; asymmetric penalty ensures
    τ% of observations fall below the predicted quantile.
    """
    
    def __init__(self, quantiles: List[float]):
        ...
    
    def forward(
        self,
        predictions: torch.Tensor,  # (B, H, Q)
        targets: torch.Tensor       # (B, H)
    ) -> torch.Tensor:
        """
        Returns:
            Scalar loss averaged over batch, horizons, and quantiles
        """
        ...


class QuantileCrossingPenalty(nn.Module):
    """
    Penalty for quantile crossing violations.
    
    Adds soft penalty when predicted quantiles are not monotonically increasing.
    """
    
    def __init__(self, weight: float = 0.1):
        ...
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, H, Q) quantile predictions
        
        Returns:
            Scalar crossing penalty
        """
        ...


class CombinedQuantileLoss(nn.Module):
    """
    Combined loss: Pinball + crossing penalty.
    """
    
    def __init__(
        self,
        quantiles: List[float],
        crossing_weight: float = 0.1
    ):
        ...
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with 'total', 'pinball', 'crossing' loss components
        """
        ...
```

### 5.2 Training Configuration

```yaml
# training_config.yaml
training:
  batch_size: 128
  gradient_accumulation_steps: 2  # Effective batch = 256
  max_epochs: 100
  early_stopping_patience: 10
  early_stopping_metric: "val_crps"
  mixed_precision: true            # Enable torch.amp for A100 optimization
  
optimizer:
  name: "AdamW"
  lr: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]
  
scheduler:
  name: "CosineAnnealingWarmRestarts"
  warmup_steps: 1000
  warmup_type: "linear"           # Linear warmup from 0 to lr
  T_0: 10  # Restart period in epochs
  T_mult: 2
  eta_min: 1e-6
  
regularization:
  dropout: 0.1
  gradient_clip_norm: 1.0        # Clip gradients by global norm (torch.nn.utils.clip_grad_norm_)
  gradient_clip_value: null      # Alternative: clip by value (not used)
  label_smoothing: 0.0

checkpointing:
  save_top_k: 3
  metric: "val_crps"
  mode: "min"
  
logging:
  backend: "wandb"                # "wandb" or "tensorboard"
  project_name: "nq-futures-dist"
  log_every_n_steps: 50
  val_check_interval: 1.0  # Validate every epoch
```

### 5.3 Trainer Module

#### 5.3.1 Module: `trainer.py`

**Interface:**

```python
class Trainer:
    """
    Training orchestrator for MIGT-TVDT model.
    
    Handles training loop, validation, checkpointing, logging,
    early stopping, mixed precision, and gradient accumulation.
    """
    
    def __init__(
        self,
        model: MIGT_TVDT,
        config: Dict[str, Any],
        data_module: NQDataModule,
        output_dir: Path,
        use_wandb: bool = True
    ):
        ...
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['training']['mixed_precision'])
        self.grad_accum_steps = config['training']['gradient_accumulation_steps']
    
    def train(self) -> Dict[str, Any]:
        """
        Execute full training procedure.
        
        Returns:
            Training history dict with losses, metrics per epoch
        """
        ...
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Single training step with mixed precision support.
        
        Uses torch.cuda.amp.autocast for forward pass,
        scaler for backward pass.
        """
        with torch.cuda.amp.autocast(enabled=self.config['training']['mixed_precision']):
            outputs = self.model(**batch)
            loss = self.loss_fn(outputs['quantiles'], batch['targets'])
        
        # Scale loss for gradient accumulation
        loss = loss / self.grad_accum_steps
        self.scaler.scale(loss).backward()
        
        return loss
    
    def _optimizer_step(self, step: int):
        """
        Execute optimizer step with gradient accumulation and clipping.
        """
        if (step + 1) % self.grad_accum_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['regularization']['gradient_clip_norm']
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
    
    def validate(self) -> Dict[str, float]:
        """Run validation epoch. Returns validation metrics dict."""
        ...
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint with training state."""
        ...
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Restore training from checkpoint."""
        ...
```

---

## 6. Evaluation Framework Specifications

### 6.1 Metrics

#### 6.1.1 Module: `metrics.py`

**Interface:**

```python
class DistributionalMetrics:
    """
    Metrics for evaluating distributional predictions.
    """
    
    @staticmethod
    def crps(
        predictions: torch.Tensor,  # (N, Q) quantiles
        targets: torch.Tensor,      # (N,) actual values
        quantiles: List[float]
    ) -> float:
        """
        Continuous Ranked Probability Score.
        
        Approximated via pinball loss at predicted quantiles.
        Lower is better.
        """
        ...
    
    @staticmethod
    def calibration_error(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        quantiles: List[float]
    ) -> Dict[str, float]:
        """
        Per-quantile calibration error.
        
        |actual_coverage - expected_coverage| for each quantile.
        """
        ...
    
    @staticmethod
    def prediction_interval_coverage_probability(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        lower_q: int = 1,  # Index for 0.1 quantile
        upper_q: int = 5   # Index for 0.9 quantile
    ) -> float:
        """
        PICP: Proportion of targets falling within [q_lower, q_upper].
        
        For [0.1, 0.9] interval, target coverage is 80%.
        """
        ...
    
    @staticmethod
    def mean_prediction_interval_width(
        predictions: torch.Tensor,
        lower_q: int = 1,
        upper_q: int = 5
    ) -> float:
        """
        MPIW: Average width of prediction intervals.
        
        Measures sharpness (narrower is better, given calibration).
        """
        ...


class PointMetrics:
    """
    Point forecast metrics using median (0.5 quantile).
    """
    
    @staticmethod
    def information_coefficient(
        predictions: torch.Tensor,  # Median predictions
        targets: torch.Tensor
    ) -> float:
        """
        Spearman rank correlation between predictions and actuals.
        
        Measures ranking quality of predictions.
        """
        ...
    
    @staticmethod
    def directional_accuracy(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        Proportion of correct sign predictions.
        """
        ...
    
    @staticmethod
    def rmse(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Root Mean Squared Error on median predictions."""
        ...


class FinancialMetrics:
    """
    Trading-oriented performance metrics.
    """
    
    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        periods_per_year: int = 252 * 78  # 5-min bars
    ) -> float:
        """
        Annualized Sharpe ratio.
        
        sqrt(periods_per_year) * mean(returns) / std(returns)
        """
        ...
    
    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        """Maximum peak-to-trough drawdown."""
        ...
    
    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """Gross profits / gross losses."""
        ...
```

### 6.2 Backtesting

#### 6.2.1 Module: `backtest.py`

**Interface:**

```python
class SimpleBacktester:
    """
    Simple backtesting framework for model evaluation.
    
    Strategy: Long/short based on median prediction sign,
    position sized inversely to prediction interval width.
    """
    
    def __init__(
        self,
        predictions: pd.DataFrame,  # With quantile columns
        actuals: pd.DataFrame,
        config: Dict[str, Any]
    ):
        ...
    
    def run(self) -> Dict[str, Any]:
        """
        Execute backtest.
        
        Returns:
            dict with:
                - equity_curve: pd.Series
                - returns: pd.Series
                - trades: pd.DataFrame
                - metrics: dict of FinancialMetrics results
        """
        ...
    
    def generate_signals(self) -> pd.Series:
        """
        Generate trading signals from predictions.
        
        Signal = sign(median) * (1 / interval_width)
        """
        ...
```

---

## 7. Phased Development Plan

### Development Timeline Summary

| Phase | Duration | Objective |
|-------|----------|-----------|
| 1 | Week 1-2 | Data Acquisition & Preprocessing |
| 2 | Week 3 | Feature Engineering |
| 3 | Week 4 | Dataset & DataLoader |
| 4 | Week 5-6 | Model Implementation |
| 5 | Week 7 | Training Pipeline |
| 6 | Week 8 | Evaluation & Analysis |
| 7 | Week 9-10 | Hyperparameter Optimization & Deployment |

**Total Duration:** ~10 weeks

---

### Phase 1: Data Acquisition & Preprocessing

**Objective:** Download raw data from Databento, apply ratio back-adjustment, and validate data integrity.

**Duration:** Week 1-2

**Deliverables:**
1. `data_loader.py` - Data loading and aggregation module
2. `rollover_adjustment.py` - Price adjustment module
3. `01_data_acquisition.ipynb` - Notebook for execution
4. Aggregated data: `nq_ohlcv_5min_aggregated.parquet` (in `interim/`)
5. Adjusted data: `nq_ohlcv_5min_adjusted.parquet` (in `interim/`)
6. Validation report: Return continuity verification

**Data Source:** User-provided `nq_ohlcv_1m_raw.parquet` (1-minute bars) → skip Databento download

**Acceptance Criteria:**
- [ ] Data spans 2010-06-06 to 2025-12-03
- [ ] Volume-based rollover correctly identifies ~60 contract rolls
- [ ] Log-returns preserved within 1e-10 tolerance after adjustment
- [ ] No gaps > 15 minutes during trading hours (excluding scheduled halts)
- [ ] Invalid ticks filtered (zero price/volume)
- [ ] Trading halts handled (forward-fill ≤15min, mask longer)
- [ ] Data stored in specified Google Drive paths

**Unit Tests:**
```python
# tests/test_data_pipeline.py

def test_rollover_preserves_returns():
    """Verify log-returns unchanged after ratio back-adjustment."""
    ...

def test_no_lookahead_in_rollover():
    """Ensure rollover detection uses only past data."""
    ...

def test_split_no_leakage():
    """Verify no temporal overlap between train/val/test."""
    ...

def test_invalid_tick_removal():
    """Confirm zero price/volume bars are removed."""
    ...
```

**Implementation Notes:**

```python
# Notebook: 01_data_acquisition.ipynb

# Cell 1: Setup
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')

# Set paths
BASE_DIR = Path("/content/drive/MyDrive/Colab Notebooks/Transformers/FP")
RAW_DIR = BASE_DIR / "data/raw"
INTERIM_DIR = BASE_DIR / "data/interim"
CREDENTIALS_PATH = BASE_DIR / "credentials/databento.txt"

for d in [RAW_DIR, INTERIM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Cell 2: Check for existing data OR download
EXISTING_1MIN = RAW_DIR / "nq_ohlcv_1m_raw.parquet"

if EXISTING_1MIN.exists():
    print(f"Using existing 1-min data: {EXISTING_1MIN}")
    import pandas as pd
    df_1min = pd.read_parquet(EXISTING_1MIN)
else:
    # Download from Databento
    from src.data.data_loader import DataLoader
    loader = DataLoader(raw_dir=RAW_DIR, interim_dir=INTERIM_DIR)
    
    # Check for existing 1-min data (USER PROVIDED)
    existing_file = RAW_DIR / "nq_ohlcv_1m_raw.parquet"
    if existing_file.exists():
        print(f"Using existing 1-min data: {existing_file}")
        df_1min = loader.load_1min_data()
    else:
        # Download from Databento if needed (requires credits)
        print("No existing data found. Downloading from Databento...")
        from src.data.databento_download import download_from_databento
        df_1min = download_from_databento(
            credentials_path=CREDENTIALS_PATH,
            symbol="NQ.v.0",           # Volume-based front month
            stype_in="continuous",
            schema="ohlcv-1m",
            start_date="2010-06-06",
            end_date="2025-12-03"
        )
        df_1min.to_parquet(existing_file)

# Cell 3: Aggregate to 5-minute bars
from src.data.data_loader import DataLoader
loader = DataLoader(raw_dir=RAW_DIR, interim_dir=INTERIM_DIR)
df_5min = loader.aggregate_to_5min(df_1min)
df_5min.to_parquet(INTERIM_DIR / "nq_ohlcv_5min_aggregated.parquet")

# Cell 4: Detect rollover dates from instrument_id changes (if available)
# OR from volume threshold crossing
from src.data.rollover_adjustment import RolloverAdjuster
adjuster = RolloverAdjuster.from_data(df_5min)  # Auto-detect rollovers
rollover_df = adjuster.get_rollover_dates()
rollover_df.to_csv(RAW_DIR / "rollover_dates.csv", index=False)

# Cell 5: Apply ratio back-adjustment
df_adjusted = adjuster.adjust_prices(df_5min)
assert adjuster.verify_returns_preserved(df_5min, df_adjusted)
df_adjusted.to_parquet(INTERIM_DIR / "nq_ohlcv_5min_adjusted.parquet")

print(f"Adjusted data saved: {len(df_adjusted)} rows, {len(rollover_df)} rollovers detected")
```

---

### Phase 2: Feature Engineering

**Objective:** Compute all derived features and prepare target variables.

**Duration:** Week 3

**Deliverables:**
1. `feature_engineering.py` - Feature computation module
2. `02_preprocessing.ipynb` - Feature engineering notebook
3. `nq_features_full.parquet` - Complete feature dataset

**Acceptance Criteria:**
- [ ] All 24 features computed without lookahead bias (see Appendix A)
- [ ] Targets computed for 5 horizons (15m, 30m, 60m, 2h, 4h)
- [ ] NaN handling: drop warmup period rows
- [ ] Feature distributions validated (no infinities, reasonable ranges)
- [ ] Correlation analysis completed (remove >0.95 correlated features if any)

**Feature Validation Checks:**

```python
# In notebook
def validate_features(df: pd.DataFrame, feature_cols: List[str]):
    """Run validation checks on computed features."""
    results = {}
    
    for col in feature_cols:
        results[col] = {
            'nan_count': df[col].isna().sum(),
            'inf_count': np.isinf(df[col]).sum(),
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std()
        }
    
    return pd.DataFrame(results).T
```

---

### Phase 3: Dataset & DataLoader Implementation

**Objective:** Create efficient PyTorch data pipeline with proper windowing, padding, and train/val/test splits.

**Duration:** Week 4

**Deliverables:**
1. `preprocessing.py` - Window creation and normalization
2. `dataset.py` - PyTorch Dataset and DataModule
3. `03_dataset_preparation.ipynb` - Dataset preparation notebook
4. Split files: `train_samples.parquet`, `val_samples.parquet`, `test_samples.parquet`

**Acceptance Criteria:**
- [ ] Windows correctly padded to 288 length
- [ ] Attention masks correctly mark valid/invalid positions
- [ ] Train: 2010-2021, Val: 2022-2023, Test: 2024-Dec 2025
- [ ] No data leakage across splits
- [ ] DataLoader yields expected tensor shapes
- [ ] Temporal info correctly extracted for positional encodings

**Data Split Statistics (Approximate):**

| Split | Date Range | Samples | % |
|-------|-----------|---------|---|
| Train | 2010-06-06 to 2021-12-31 | ~580K | 70% |
| Val | 2022-01-01 to 2023-12-31 | ~165K | 20% |
| Test | 2024-01-01 to 2025-12-03 | ~85K | 10% |

---

### Phase 4: Model Implementation

**Objective:** Implement complete MIGT-TVDT architecture per specifications.

**Duration:** Week 5-6

**Deliverables:**
1. `embeddings.py` - Input embedding module
2. `positional_encodings.py` - All positional encoding variants
3. `temporal_attention.py` - Temporal attention and aggregation
4. `variable_attention.py` - Variable attention
5. `gated_instance_norm.py` - RevIN and LGU
6. `quantile_heads.py` - Output heads
7. `migt_tvdt.py` - Complete model
8. Unit tests for each component

**Acceptance Criteria:**
- [ ] Forward pass executes without error
- [ ] Output shapes: (B, 5, 7) for 5 horizons × 7 quantiles
- [ ] Parameter count ~4M (within expected range)
- [ ] Gradient flow verified (no vanishing/exploding)
- [ ] VRAM usage <25GB with batch_size=128
- [ ] Model can be saved/loaded correctly

**Component Testing Framework:**

```python
# tests/test_model_components.py

import torch
import pytest
from src.model import *

@pytest.fixture
def batch():
    B, T, V, D = 4, 288, 24, 256
    return {
        'features': torch.randn(B, T, V),
        'attention_mask': torch.ones(B, T, dtype=torch.bool),
        'temporal_info': {
            'bar_in_day': torch.randint(0, 288, (B, T)),
            'day_of_week': torch.randint(0, 5, (B,)),
            'day_of_month': torch.randint(1, 32, (B,)),
            'day_of_year': torch.randint(1, 366, (B,))
        }
    }

def test_temporal_attention_shape(batch):
    layer = TemporalAttentionBlock(d_model=256, n_heads=8, d_ff=1024)
    x = torch.randn(4, 288, 23, 256)
    out = layer(x, batch['attention_mask'])
    assert out.shape == (4, 288, 23, 256)

def test_quantile_head_non_crossing():
    head = QuantileHead(d_model=256, n_quantiles=7)
    x = torch.randn(4, 256)
    out = head(x)
    # Verify quantiles are monotonically increasing
    assert torch.all(out[:, 1:] >= out[:, :-1])
```

---

### Phase 5: Training Pipeline

**Objective:** Implement training loop, loss functions, optimization, and checkpointing.

**Duration:** Week 7

**Deliverables:**
1. `loss_functions.py` - Pinball loss and crossing penalty
2. `trainer.py` - Training orchestrator
3. `scheduler.py` - Learning rate scheduling
4. Training configuration files
5. `04_model_training.ipynb` - Training notebook

**Acceptance Criteria:**
- [ ] Training loop executes on A100 without OOM
- [ ] Loss decreases over epochs
- [ ] Checkpointing saves/restores correctly
- [ ] Early stopping triggers on val_crps plateau
- [ ] Gradient clipping prevents instability
- [ ] TensorBoard/WandB logging functional

**Training Configuration:**

Default production settings:
- Batch size: 128 (optimal for time series)
- Gradient accumulation: 2 (effective batch = 256)
- Max epochs: 100
- Early stopping: 10 epochs patience

Development/tuning settings for faster iteration:
- Increase `training.batch_size: 192` (~30% faster per epoch)
- Set `data.subsample_fraction: 0.25` (4× fewer training samples)
- For testing: Set `data.apply_subsample_to_all_splits: true` (subsamples val/test too)
- Combined: ~5× faster for hyperparameter sweeps

**Training Notebook Structure:**
```python
# Notebook: 05_training.ipynb (Dev Phase 5 test notebook)
# Notebook: 06_full_training.ipynb (Production training)

# Cell 1: Copy data to VM for faster I/O
!cp -r /content/drive/MyDrive/Colab\ Notebooks/Transformers/FP/data/processed/*.parquet /content/data/

# Cell 2: Setup
import yaml
from src.model.migt_tvdt import MIGT_TVDT
from src.data.dataset import NQDataModule
from src.training.trainer import create_trainer

# Load configs
with open('configs/model_config.yaml') as f:
    model_config = yaml.safe_load(f)
with open('configs/training_config.yaml') as f:
    train_config = yaml.safe_load(f)

# Cell 3: Initialize
model = MIGT_TVDT(model_config['model'])
data_module = NQDataModule(
    data_path=Path('/content/data/nq_features_full.parquet'),
    batch_size=train_config['training']['batch_size'],
    num_workers=train_config['data']['num_workers'],
    pin_memory=train_config['data']['pin_memory'],
    train_end="2021-12-31",
    val_end="2023-12-31",
    subsample_fraction=train_config['data'].get('subsample_fraction'),
    apply_subsample_to_all_splits=train_config['data'].get('apply_subsample_to_all_splits', False),
    subsample_seed=train_config['data'].get('subsample_seed', 42)
)
data_module.setup()

# Cell 4: Train
trainer = create_trainer(
    model=model,
    config={**model_config, **train_config},
    data_module=data_module,
    output_dir=Path('/content/outputs'),
    use_wandb=False
)

history = trainer.train()

# Cell 5: Save to Drive
!cp -r /content/outputs/* /content/drive/MyDrive/Colab\ Notebooks/Transformers/FP/outputs/
```

---

### Phase 6: Evaluation & Analysis

**Objective:** Implement evaluation metrics, calibration analysis, and backtesting.

**Duration:** Week 8

**Deliverables:**
1. `metrics.py` - All evaluation metrics
2. `calibration.py` - Calibration analysis tools
3. `backtest.py` - Simple backtesting framework
4. `05_evaluation.ipynb` - Evaluation notebook
5. Final evaluation report

**Acceptance Criteria:**
- [ ] CRPS computed correctly
- [ ] Calibration plots show actual vs. expected coverage
- [ ] IC and DA metrics reported per horizon
- [ ] Backtest produces Sharpe ratio >1.0 (pre-costs target)
- [ ] Ablation study results documented

---

### Phase 7: Hyperparameter Optimization & Deployment

**Objective:** Systematic hyperparameter tuning and inference preparation.

**Duration:** Week 9-10

**Deliverables:**
1. `hyperopt.py` - Optuna optimization module
2. `06_hyperopt.ipynb` - Tuning notebook
3. `inference.py` - Production inference module
4. Optimized model weights
5. Hyperparameter search report

**Optuna Search Space:**

```python
# src/training/hyperopt.py

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    """
    config = {
        'd_model': trial.suggest_categorical('d_model', [128, 256, 384, 512]),
        'n_heads': trial.suggest_categorical('n_heads', [4, 8]),
        'n_temporal_layers': trial.suggest_int('n_temporal_layers', 2, 6),
        'n_variable_layers': trial.suggest_int('n_variable_layers', 1, 3),
        'd_ff_mult': trial.suggest_categorical('d_ff_mult', [2, 4]),
        'dropout': trial.suggest_float('dropout', 0.05, 0.2),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
    }
    
    # Train with config, return val_crps
    val_crps = train_with_config(config, max_epochs=30)
    return val_crps


def run_optimization(n_trials: int = 20) -> optuna.Study:
    """
    Run Optuna hyperparameter search.
    
    Args:
        n_trials: Number of trials (10-20 recommended for A100 budget)
    
    Returns:
        Completed study with best parameters
    """
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=n_trials)
    return study
```

**Acceptance Criteria:**
- [ ] Optuna search completes 15-20 trials
- [ ] Best model achieves val_crps < baseline
- [ ] Inference module processes single sample in <100ms
- [ ] Model exports correctly (TorchScript or ONNX optional)

**Evaluation Report Template:**

```markdown
# Model Evaluation Report

## Distributional Metrics (Test Set)

| Horizon | CRPS | PICP (80%) | MPIW |
|---------|------|------------|------|
| 15m     |      |            |      |
| 30m     |      |            |      |
| 60m     |      |            |      |
| 2h      |      |            |      |
| 4h      |      |            |      |

## Point Metrics (Median)

| Horizon | IC | DA | RMSE |
|---------|----|----|------|
| ...     |    |    |      |

## Financial Metrics

| Metric | Value |
|--------|-------|
| Sharpe Ratio (gross) | |
| Max Drawdown | |
| Profit Factor | |

## Calibration Analysis

[Include calibration plots]

## Ablation Results

| Configuration | Val CRPS | Test CRPS | Notes |
|--------------|----------|-----------|-------|
| Full model | | | Baseline |
| No Variable Embedding | | | H8 test |
| No TSA | | | H8 test |
| No Instance Norm | | | H9 test |
| No LGU | | | H9 test |
| Point prediction (MSE) | | | H10 test |
```

---

## 8. Dependencies

### 8.1 Python Environment

```
# requirements.txt

# Core
python>=3.10
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
polars>=0.19.0              # Efficient alternative for feature engineering

# Data
databento>=0.25.0
pyarrow>=12.0.0
fastparquet>=2023.4.0
h5py>=3.8.0                 # HDF5 support for large datasets

# Model
einops>=0.6.0
flash-attn>=2.0.0           # Optional, for A100 optimization

# Training
wandb>=0.15.0
tensorboard>=2.13.0
optuna>=3.2.0               # Hyperparameter optimization
pyyaml>=6.0

# Evaluation
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Testing
pytest>=7.3.0
pytest-cov>=4.1.0

# Utilities
tqdm>=4.65.0
python-dotenv>=1.0.0
```

### 8.2 Colab Setup Script

```python
# setup_colab.py

def setup_environment():
    """Setup Colab environment for training."""
    import subprocess
    import sys
    
    # Install dependencies
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-q',
        'databento', 'einops', 'wandb', 'flash-attn'
    ])
    
    # Verify GPU
    import torch
    assert torch.cuda.is_available(), "GPU not available"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Mount Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Create local directories
    import os
    os.makedirs('/content/data', exist_ok=True)
    os.makedirs('/content/outputs', exist_ok=True)
    
    return True
```

---

## 9. Risk Mitigation

### 9.1 Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Databento API changes | Data acquisition blocked | Cache raw data locally; document API version |
| VRAM overflow | Training failure | Profile memory; reduce batch size; use gradient checkpointing; mixed precision |
| Quantile crossing | Invalid predictions | Softplus parameterization enforces monotonicity |
| Overfitting | Poor generalization | Early stopping; dropout; validation monitoring |
| Non-convergence | Wasted compute | LR warmup; gradient clipping; monitor loss curves |
| Hyperopt budget exceeded | Incomplete search | Use pruning; limit epochs per trial; TPE sampler |

### 9.2 Data Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Missing data gaps | Invalid windows | Filter windows with >5% missing; interpolate small gaps |
| Rollover miscalculation | Corrupted returns | Verify log-returns before/after adjustment |
| Lookahead bias | Inflated performance | Strict causal feature computation; code review |

---

## 10. Appendices

### Appendix A: Variable List

| Index | Variable | Source | Category |
|-------|----------|--------|----------|
| 0 | open | Raw | Price |
| 1 | high | Raw | Price |
| 2 | low | Raw | Price |
| 3 | close | Raw | Price |
| 4 | log_volume | Derived | Volume |
| 5 | log_return | Derived | Returns |
| 6 | gk_volatility | Derived | Volatility |
| 7 | rv_12 | Derived | Volatility |
| 8 | rv_36 | Derived | Volatility |
| 9 | rv_72 | Derived | Volatility |
| 10 | amihud_illiq | Derived | Liquidity |
| 11 | rsi_14 | Derived | Momentum |
| 12 | macd | Derived | Momentum |
| 13 | roc_5 | Derived | Momentum |
| 14 | roc_10 | Derived | Momentum |
| 15 | roc_20 | Derived | Momentum |
| 16 | ema_slope_9 | Derived | Trend |
| 17 | ema_slope_21 | Derived | Trend |
| 18 | ema_slope_50 | Derived | Trend |
| 19 | ema_dev_9 | Derived | Trend |
| 20 | ema_dev_21 | Derived | Trend |
| 21 | ema_dev_50 | Derived | Trend |
| 22 | atr_14 | Derived | Range |
| 23 | volume_ma_ratio | Derived | Volume |

**Note:** macd_signal was removed during Phase 2 due to high multicollinearity (0.955 correlation with macd). 
Final feature count: 24 (down from originally planned 25).

### Appendix B: Horizon Mapping

| Horizon Name | Minutes | 5-min Bars |
|--------------|---------|------------|
| 15m | 15 | 3 |
| 30m | 30 | 6 |
| 60m | 60 | 12 |
| 2h | 120 | 24 |
| 4h | 240 | 48 |

### Appendix C: Quantile Mapping

| Index | Quantile τ | Interpretation |
|-------|-----------|----------------|
| 0 | 0.05 | 5th percentile (extreme low) |
| 1 | 0.10 | 10th percentile (low bound for 80% CI) |
| 2 | 0.25 | 25th percentile (Q1) |
| 3 | 0.50 | 50th percentile (median, point forecast) |
| 4 | 0.75 | 75th percentile (Q3) |
| 5 | 0.90 | 90th percentile (high bound for 80% CI) |
| 6 | 0.95 | 95th percentile (extreme high) |

---

*Document Version: 1.2*  
*Last Updated: December 2025*  
*Engineering Lead: Claude*  
*Peer Review: Grok, Gemini (Feedback Round 1 incorporated)*

**Key changes in v1.2:**
- Fixed Databento API parameters: `NQ.v.0` with `stype_in="continuous"` for volume rollover
- Confirmed `ohlcv-1m` schema required (no native 5-min); aggregate locally
- Added tensor broadcasting notes for 4D positional encoding (Gemini)
- Added `ema_dev_50` to feature list (Grok)
- Added credentials location: `/credentials/databento.txt`
- Confirmed existing 1-min data can skip download step