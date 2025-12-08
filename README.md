# FuturesSL: Distributional Forecasting for NASDAQ Futures

**Author:** Clemente Fortuna  
**Project Status:** Phase 2 Complete (Data Acquisition & Feature Engineering)

## Overview

FuturesSL is a transformer-based distributional forecasting system for NASDAQ 100 (NQ) futures trading. The model predicts log-return distributions across multiple time horizons (15m, 30m, 60m, 2h, 4h) using quantile regression, providing both point forecasts and calibrated uncertainty estimates for enhanced risk management.

This project implements a novel hybrid architecture combining:
- **Variable Embeddings** from RL-TVDT [1] for improved multivariate dependency modeling
- **Two-Stage Attention** decoupling temporal and cross-variable dynamics
- **Gated Instance Normalization** from MIGT [2] for non-stationary financial data
- **Quantile Regression** for distributional outputs with calibrated prediction intervals

## Project Motivation

Traditional point-forecast models fail to capture the uncertainty inherent in financial markets. FuturesSL addresses this through distributional forecasting, outputting a full predicted distribution (via 7 quantiles: 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95) rather than single point estimates. This enables:

1. **Risk-Aware Predictions:** Quantile spread indicates forecast uncertainty
2. **Tail Risk Assessment:** Explicit modeling of extreme outcomes
3. **Trading Strategy Integration:** Position sizing based on confidence intervals
4. **Regime Adaptation:** Instance normalization handles non-stationary market regimes

## Technical Architecture

### Data Pipeline

**Source:** Databento NQ front-month continuous contract (volume-based rollover)  
**Period:** June 2010 - December 2025 (15.5 years)  
**Frequency:** 5-minute OHLCV bars (~1.1M samples)

Key preprocessing steps:
- Ratio back-adjustment for futures rollover (preserves log-returns)
- Invalid tick filtering (outliers, zero prices/volume)
- Trading halt detection and flagging
- Comprehensive feature engineering (24 features across 6 categories)

### Feature Engineering

**Raw Features (5):**
- Price: open, high, low, close
- Volume: volume

**Derived Features (19):**
- **Volatility (4):** Garman-Klass estimator, Realized Volatility (1h, 3h, 6h windows)
- **Liquidity (1):** Amihud illiquidity proxy
- **Momentum (5):** RSI-14, MACD, Rate of Change (5, 10, 20 periods)
- **Trend (6):** EMA slopes (9, 21, 50), EMA deviations (9, 21, 50)
- **Range (1):** Average True Range (14-period)
- **Volume (2):** Log volume, volume MA ratio

All features computed causally to prevent lookahead bias.

### Model Architecture (Planned - Phase 4)

```
Input: (Batch, Time=273-276, Variables=24)
  â†" Padding + Masking → (Batch, 288, 24)
  â†" Variable Embedding → (Batch, 24, D_model)
  â†" Positional Encoding (Time of Day, Day of Week, Time2Vec)
  
[Temporal Attention Stage]
  For each variable independently:
    â†" Self-Attention over time dimension
    â†" Attention-weighted pooling → (Batch, 24, D_model)
  
[Variable Attention Stage]
  â†" Cross-attention between variables
  â†" Gated Instance Normalization (MIGT)
  â†" Residual connections
  
[Multi-Horizon Quantile Heads]
  For each horizon h ∈ {15m, 30m, 60m, 2h, 4h}:
    â†" Horizon embedding
    â†" MLP decoder
    â†" Output: 7 quantiles (τ = 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
```

**Key Design Choices:**

1. **Variable-Centric Embeddings:** Unlike standard transformers that embed timesteps, we embed the entire time series of each variable. This allows the model to learn correlations between indicators (e.g., RSI-Volume relationships) rather than just temporal patterns.

2. **Two-Stage Attention:** Decouples temporal dynamics (intraday patterns within each feature) from cross-variable dependencies (how features interact). Reduces computational complexity and aligns with financial market structure.

3. **Instance Normalization with Gating:** Normalizes each 24-hour window independently, allowing the model to learn shape patterns invariant to price level. The Lite Gate Unit (LGU) filters noisy signals common in high-frequency data.

4. **Distributional Outputs:** Quantile regression with pinball loss directly optimizes prediction intervals. Softplus parameterization ensures monotonic quantile ordering (no crossing).

### Positional Encodings

Multi-scale temporal encoding capturing financial market cycles:

- **Time of Day:** Cyclical sine/cosine (288 5-min periods)
- **Day of Week:** Learnable embeddings (Mon-Fri market effects)
- **Seasonal Patterns:** Time2Vec [3] with learned frequencies for quarterly rebalancing, option expiry effects

## Implementation Status

### ✅ Phase 1: Data Acquisition & Preprocessing (Complete)

**Deliverables:**
- `data_loader.py` - 1-minute to 5-minute aggregation
- `rollover_adjustment.py` - Ratio back-adjustment with validation
- `01_data_acquisition.ipynb` - Testing notebook

**Outputs:**
- `nq_ohlcv_5min_aggregated.parquet` (~1.6M bars)
- `nq_ohlcv_5min_adjusted.parquet` (ratio-adjusted)
- `rollover_dates.csv` (~60 detected rollovers)

**Key Validations:**
- ✓ OHLC relationship constraints maintained
- ✓ Returns preservation (max diff < 1e-10)
- ✓ Chronological ordering verified
- ✓ Price positivity enforced

### ✅ Phase 2: Feature Engineering (Complete)

**Deliverables:**
- `feature_engineering.py` - 24 causal features + 5 targets
- `02_preprocessing.ipynb` - Comprehensive testing

**Outputs:**
- `nq_features_full.parquet` (24 features + 5 target horizons, ~234 MB)

**Quality Metrics:**
- Zero NaN values (after 72-bar warmup)
- Zero infinite values
- No high multicollinearity (max correlation: 0.94)
- All features within expected statistical ranges

**Target Statistics:**

| Horizon | Mean Return | Std Dev | Ann. Sharpe |
|---------|-------------|---------|-------------|
| 15m     | 0.000008    | 0.00129 | 0.96        |
| 30m     | 0.000016    | 0.00182 | 0.96        |
| 60m     | 0.000032    | 0.00258 | 0.96        |
| 2h      | 0.000064    | 0.00365 | 0.96        |
| 4h      | 0.000128    | 0.00515 | 0.97        |

### 🔄 Phase 3-6: In Development

- **Phase 3:** PyTorch Dataset & DataLoader (24-hour sliding windows, RevIN normalization)
- **Phase 4:** Transformer model implementation (MIGT-TVDT hybrid)
- **Phase 5:** Training pipeline (quantile loss, validation monitoring)
- **Phase 6:** Evaluation & backtesting (IC, CRPS, Sharpe ratio, calibration analysis)

## Compute Environment

**Primary:** Google Colab with A100 GPU (80GB VRAM)  
**Secondary Resources:** 167.1 GB RAM, 12 CPU cores  
**Storage:** Google Drive integration for data persistence

Memory profile estimates:
- Model parameters: ~10-40M (20-40 GB peak VRAM with batch size 64-256)
- Training data: In-memory loading after initial GDrive copy
- Checkpointing: Periodic saves to GDrive

## Installation & Setup

### Prerequisites

```bash
python >= 3.10
torch >= 2.0.0
numpy >= 1.24.0
pandas >= 2.0.0
```

### Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- **Data:** `databento`, `pyarrow`, `fastparquet`
- **Model:** `torch`, `einops`, `flash-attn` (optional)
- **Training:** `wandb`, `tensorboard`, `optuna`
- **Evaluation:** `scipy`, `scikit-learn`, `matplotlib`, `seaborn`

### Google Colab Setup

1. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Install dependencies:
```python
!pip install -q databento einops wandb flash-attn
```

3. Verify GPU:
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Data Directory Structure

```
/content/drive/MyDrive/Colab Notebooks/Transformers/FP/
├── data/
│   ├── raw/
│   │   ├── nq_ohlcv_1m_raw.parquet          # User-provided 1-min data
│   │   └── rollover_dates.csv               # Generated metadata
│   ├── interim/
│   │   ├── nq_ohlcv_5min_aggregated.parquet
│   │   └── nq_ohlcv_5min_adjusted.parquet
│   └── processed/
│       ├── nq_features_full.parquet         # 24 features + 5 targets
│       └── column_info.csv
├── src/
│   └── data/
│       ├── data_loader.py
│       ├── rollover_adjustment.py
│       └── feature_engineering.py
└── notebooks/
    ├── 01_data_acquisition.ipynb
    └── 02_preprocessing.ipynb
```

## Usage

### Phase 1-2: Data Preparation (Current Status)

```python
from src.data import DataLoader, RolloverAdjuster, FeatureEngineer

# Load and aggregate 1-min to 5-min bars
loader = DataLoader(
    raw_path='/path/to/nq_ohlcv_1m_raw.parquet',
    output_dir='/path/to/interim/'
)
df_5min = loader.aggregate_to_5min()

# Apply rollover adjustment
adjuster = RolloverAdjuster.from_data(df_5min, threshold_pct=0.01, threshold_std=3.0)
df_adjusted = adjuster.adjust_prices(df_5min)
adjuster.verify_returns_preserved(df_5min, df_adjusted)

# Compute features
engineer = FeatureEngineer()
df_features = engineer.compute_all_features(df_adjusted)

# Add target variables
horizons = {'15m': 3, '30m': 6, '60m': 12, '2h': 24, '4h': 48}
df_final = engineer.add_targets(df_features, horizons)
```

### Phase 3+: Model Training (Planned)

```python
from src.model import MIGTTVDTModel
from src.training import QuantileLoss, Trainer

# Initialize model
model = MIGTTVDTModel(
    input_vars=24,
    d_model=256,
    n_heads=8,
    n_layers=6,
    horizons=5,
    quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
)

# Setup training
criterion = QuantileLoss(quantiles=model.quantiles)
trainer = Trainer(model, criterion, optimizer='AdamW', lr=1e-4)

# Train with validation monitoring
trainer.fit(train_loader, val_loader, epochs=100, early_stopping_patience=10)
```

## Performance Metrics

### ML Metrics
- **Information Coefficient (IC):** Spearman correlation (predicted vs. actual)
- **Directional Accuracy (DA):** Sign prediction accuracy
- **CRPS:** Continuous Ranked Probability Score (distributional quality)
- **Calibration Error:** Deviation from nominal coverage

### Financial Metrics
- **Sharpe Ratio:** Annualized risk-adjusted returns (target > 1.5 gross)
- **Maximum Drawdown:** Peak-to-trough decline
- **Profit Factor:** Gross wins / gross losses
- **Tail Metrics:** 5th/95th percentile coverage accuracy

## Scientific Foundations

### Key Hypotheses

**H1 (Non-Stationarity):** Instance normalization enables learning across market regimes with different volatility/price levels.

**H2 (Multivariate Dependencies):** Variable-centric embeddings capture feature interactions (e.g., volume-RSI) more effectively than timestep embeddings.

**H3 (Distributional Superiority):** Quantile regression achieves lower CRPS and better tail calibration than MSE point predictions.

**H4 (Temporal Cycles):** Composite positional encodings capture intraday, weekly, and seasonal patterns missed by standard sinusoidal encodings.

### Ablation Studies (Planned)

| Configuration              | Modification                | Expected Impact          |
|----------------------------|----------------------------|--------------------------|
| No Variable Embedding      | Standard timestep tokens   | ↓ IC (entangled)         |
| No Two-Stage Attention     | Single attention stage     | ↑ compute, ↓ performance |
| No Instance Normalization  | LayerNorm only             | ↓ regime adaptation      |
| No Gating (LGU)            | Remove gate mechanism      | ↑ noise, wider intervals |
| Point Prediction (MSE)     | MSE loss instead of QR     | No uncertainty, ↑ CRPS   |

## Data Considerations

### Rollover Adjustment

Futures contracts expire quarterly. Volume-based rollover switches to the next contract when its volume exceeds the current. This creates price discontinuities that must be adjusted:

```
Ratio Back-Adjustment:
For rollover from Contract A to B at time t:
  R = Price_B(t) / Price_A(t)
  Price_historical_adjusted = Price_historical_raw × R
```

**Why Ratio vs. Difference:** Ratio adjustment preserves percentage returns, critical for log-return-based ML. Difference adjustment distorts returns over long periods.

**Validation:** Log-returns are identical before/after adjustment (tolerance < 1e-10) at non-rollover timestamps.

### Causality Enforcement

All derived features use strictly historical data:
- Rolling windows: `rolling(window).mean()` with no forward data
- Exponential moving averages: `ewm(span=n, adjust=False)`
- Target variables: Computed with `.shift(-h)` but dropped during training

**Critical:** The 24-hour lookback window contains only historical bars. Target computation uses future data for labeling but these columns are excluded from model inputs.

## References

[1] Y. Li et al., "Reinforcement learning with temporal and variable dependency-aware transformer for stock trading optimization," *Neural Networks*, vol. 192, p. 107905, 2025. https://arxiv.org/abs/2408.12446

[2] F. Gu et al., "MIGT: Memory Instance Gated Transformer Framework for Financial Portfolio Management," *arXiv preprint arXiv:2502.07280*, 2025. https://arxiv.org/abs/2502.07280

[3] S. M. Kazemi et al., "Time2Vec: Learning a Vector Representation of Time," *arXiv preprint arXiv:1907.05321*, 2019. https://arxiv.org/abs/1907.05321

[4] Y. Liu et al., "iTransformer: Inverted Transformers are Effective for Time Series Forecasting," *ICLR*, 2024. https://arxiv.org/abs/2310.06625

[5] A. Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017. https://arxiv.org/abs/1706.03762

[6] Y. Zhang and J. Yan, "Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting," *ICLR*, 2023. https://openreview.net/forum?id=vSVLM2j9eie

## License

This project is provided as-is for research and educational purposes.

## Contact

**Author:** Clemente Fortuna  
**Project:** FuturesSL - Distributional Forecasting for NASDAQ Futures

---

*Last Updated: December 2025*  
*Project Status: Phase 2 Complete - Feature Engineering Delivered*
