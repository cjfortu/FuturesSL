# FuturesSL: Distributional Forecasting for NASDAQ Futures

**Author:** Clemente Fortuna  
**Project Status:** Complete (Dev Phases 1-6) - Model Trained and Evaluated

## Overview

FuturesSL is a transformer-based distributional forecasting system for NASDAQ 100 (NQ) futures trading. The model predicts log-return distributions across multiple time horizons (15m, 30m, 60m, 2h, 4h) using quantile regression, providing both point forecasts and calibrated uncertainty estimates for enhanced risk management.

This project implements a novel hybrid architecture combining:
- **Variable Embeddings** from RL-TVDT for improved multivariate dependency modeling
- **Two-Stage Attention** decoupling temporal and cross-variable dynamics
- **Gated Instance Normalization** from MIGT for non-stationary financial data
- **Quantile Regression** for distributional outputs with calibrated prediction intervals
- **Flash Attention** for memory-efficient training on A100 GPUs

## Project Motivation

Traditional point-forecast models fail to capture the uncertainty inherent in financial markets. FuturesSL addresses this through distributional forecasting, outputting a full predicted distribution (via 7 quantiles: 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95) rather than single point estimates. This enables:

1. **Risk-Aware Predictions:** Quantile spread indicates forecast uncertainty
2. **Tail Risk Assessment:** Explicit modeling of extreme outcomes
3. **Trading Strategy Integration:** Position sizing based on confidence intervals
4. **Regime Adaptation:** Instance normalization handles non-stationary market regimes

## Technical Architecture

### Model Architecture Diagram

![MIGT-TVDT Architecture](https://github.com/cjfortu/FuturesSL/blob/master/evaluation_results/migt_tvdt_architecture.png)

The architecture implements a hybrid MIGT-TVDT design with the following key stages:

1. **Preprocessing Layer**: Reversible Instance Normalization (RevIN) with padding to T_max=288
2. **Variable Embedding Layer**: Per-variable linear projection preserving temporal dimension
3. **Positional Encodings**: Composite multi-scale temporal encoding (time-of-day, day-of-week, day-of-month, day-of-year)
4. **Stage 1 - Temporal Attention**: Self-attention over time dimension for each variable independently (4 layers)
5. **Stage 2 - Variable Attention**: Cross-variable attention with Gated Instance Normalization (2 layers)
6. **Multi-Horizon Quantile Heads**: Horizon-specific decoders producing 7 quantiles per horizon
7. **Training Loss**: Pinball loss with monotonicity constraint

### Data Pipeline

**Source:** Databento NQ front-month continuous contract (volume-based rollover)  
**Period:** June 2010 - December 2025 (15.5 years)  
**Frequency:** 5-minute OHLCV bars (~1.1M samples)

Key preprocessing steps:
- Ratio back-adjustment for futures rollover (preserves log-returns)
- Invalid tick filtering (outliers, zero prices/volume)
- Trading halt detection and flagging
- Comprehensive feature engineering (24 features across 6 categories)
- Time-based splits with purge gaps (prevents data leakage)

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

### Architecture Specifications

**Model Parameters:**
- d_model: 256
- n_heads: 8 (32D per head)
- d_ff: 1024
- dropout: 0.1
- Temporal layers: 4
- Variable layers: 2
- Total parameters: 6.87M

**Input Dimensions:**
- Batch size: 128 (effective 256 with gradient accumulation)
- Sequence length: 288 timesteps (padded from 273-276)
- Variables: 24 features
- Horizons: 5 (15m, 30m, 60m, 2h, 4h)
- Quantiles: 7 per horizon

**Key Design Choices:**

1. **Variable-Centric Embeddings:** Unlike standard transformers that embed timesteps, we embed the entire time series of each variable. This allows the model to learn correlations between indicators (e.g., RSI-Volume relationships) rather than just temporal patterns.

2. **Two-Stage Attention:** Decouples temporal dynamics (intraday patterns within each feature) from cross-variable dependencies (how features interact). Reduces computational complexity and aligns with financial market structure.

3. **Instance Normalization with Gating:** Normalizes each 24-hour window independently, allowing the model to learn shape patterns invariant to price level. The Lite Gate Unit (LGU) filters noisy signals common in high-frequency data.

4. **Distributional Outputs:** Quantile regression with pinball loss directly optimizes prediction intervals. Cumulative softplus parameterization ensures monotonic quantile ordering (no crossing).

5. **Flash Attention:** PyTorch 2.0+ SDPA backend reduces memory complexity from O(T²) to O(T) per layer, enabling batch sizes up to 128 on A100 (vs. 64 with manual attention).

### Positional Encodings

Multi-scale temporal encoding capturing financial market cycles:

- **Time of Day (32D):** Sinusoidal encoding with integer frequency multipliers (k=1,2,...,16) for exact periodicity over 288 5-minute bars. Captures market open, lunch, close patterns.
- **Day of Week (16D):** Learnable embeddings for Mon-Fri market effects (e.g., Monday reversals, Friday position squaring).
- **Day of Month (16D):** Time2Vec with learned frequencies for monthly patterns (option expiry, rebalancing).
- **Day of Year (32D):** Time2Vec for seasonal patterns (quarterly earnings, tax deadlines, holiday effects).

Total: 96D projected to d_model=256 via learned linear layer.

## Implementation Status

All six development phases have been completed:

### ✅ Phase 1: Data Acquisition & Preprocessing

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

### ✅ Phase 2: Feature Engineering

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

### ✅ Phase 3: Dataset & DataLoader Implementation

**Deliverables:**
- `preprocessing.py` - Window creation, normalization, temporal splits with purge gaps
- `dataset.py` - PyTorch Dataset and DataModule
- `03_dataset_preparation.ipynb` - Testing notebook

**Key Features:**
- 24-hour sliding windows with padding to 288 bars
- RevIN (Reversible Instance Normalization) per sample
- Time-based splits with 24-hour purge gaps (prevents data leakage)
- Attention masks for variable-length sequences
- Temporal information extraction for positional encodings

**Split Statistics:**

| Split | Date Range | Samples | % |
|-------|-----------|---------|----|
| Train | 2010-06-07 to 2021-12-31 | ~580K | 70% |
| Val | 2022-01-02 to 2023-12-31 | ~164K | 20% |
| Test | 2024-01-02 to 2025-12-03 | ~85K | 10% |

**Purge Gap Implementation:**
- Train-Val gap: 24 hours (2022-01-01 excluded)
- Val-Test gap: 24 hours (2024-01-01 excluded)
- Total purged: ~576 bars (~0.05% of data)
- Rationale: Prevents validation/test lookback windows from overlapping with train data

### ✅ Phase 4: Model Architecture Implementation

**Deliverables:**
- `positional_encodings.py` - Composite positional encoding with Time2Vec
- `embeddings.py` - Variable embedding with 3D to 4D projection
- `temporal_attention.py` - Temporal attention with Flash Attention
- `variable_attention.py` - Cross-variable attention with Flash Attention
- `gated_instance_norm.py` - RevIN with LGU gating
- `quantile_heads.py` - Non-crossing quantile regression heads
- `migt_tvdt.py` - Complete MIGT-TVDT hybrid architecture
- `model_config.yaml` - Architecture configuration
- `04_model_testing.ipynb` - Comprehensive testing notebook

**Key Achievements:**
- Flash Attention implementation reduces memory from 77GB (manual) to 35GB at batch_size=64
- Batch size 128 achieves ~70GB VRAM usage (88% A100 utilization, safe margin for training)
- All 10 test suites passing (architecture, positional encoding, attention mechanisms, quantile heads, save/load, integration)
- Phase 3 integration verified (dataset outputs compatible with model inputs)
- Model parameters: 6.87M

**Memory Profile (Flash Attention, A100 80GB):**
- Batch size 64: 35.44 GB (44% utilization)
- Batch size 128: ~70 GB (88% utilization)
- Batch size 180: OOM (exceeds capacity)

**Technical Highlights:**
- Composite positional encoding with integer frequency multipliers for exact periodicity
- Two-stage attention architecture (temporal then variable)
- Gated Instance Normalization (RevIN + LGU) for non-stationary data
- Non-crossing quantile outputs via cumulative softplus parameterization
- Full PyTorch 2.0+ SDPA compatibility

### ✅ Phase 5: Training Pipeline

**Deliverables:**
- `loss_functions.py` - Quantile regression loss with per-quantile/horizon breakdowns
- `scheduler.py` - Learning rate scheduling (warmup + cosine annealing)
- `trainer.py` - Complete training orchestration with checkpointing and early stopping
- `training_config.yaml` - Training hyperparameters
- `05_training.ipynb` - Unit and integration testing
- `05_full_training.ipynb` - Production training execution

**Key Features:**
- **Mixed Precision Training (AMP):** Automatic on A100 for faster training
- **Gradient Accumulation:** Effective batch size 256 (128 × 2)
- **Learning Rate Schedule:** Linear warmup (1000 steps) + cosine annealing with restarts
- **Gradient Clipping:** Global norm clipping at 1.0
- **Early Stopping:** Patience 10 epochs, min delta 1e-5
- **Checkpointing:** Latest, best, and top-3 epoch checkpoints
- **WandB Integration:** For experiment tracking

**Training Configuration:**
- Data: 10% subsample for prototyping validation
- Epochs: 4 (best at epoch 3)
- Batch size: 128 (effective 256 with gradient accumulation)
- Hardware: A100 GPU (80GB VRAM)
- Training time: ~15 minutes

**Training Results:**
- Best validation loss: 0.014659 (epoch 3)
- Model size: 82.7 MB
- Status: Pipeline validated successfully

**Metrics Computed:**
- Per-quantile coverage (calibration check)
- PICP-80 (80% prediction interval coverage)
- Mean interval widths (50% and 80%)
- Per-quantile and per-horizon loss breakdowns

### ✅ Phase 6: Evaluation & Analysis

**Deliverables:**
- `metrics.py` - Distributional (CRPS, calibration), point (IC, DA), and financial metrics (Sharpe, MDD)
- `calibration.py` - Calibration analysis with reliability diagrams and PIT histograms
- `backtest.py` - Signal generation and multi-horizon backtesting
- `inference.py` - Model prediction and evaluation pipeline
- `06_evaluation.ipynb` - Comprehensive testing notebook
- `06_production_evaluation.ipynb` - Production evaluation execution

**Distributional Metrics:**
- **CRPS:** Continuous Ranked Probability Score (approximated via quantile loss)
- **Calibration Error:** Per-quantile coverage deviation
- **PICP:** Prediction Interval Coverage Probability (80% intervals)
- **MPIW:** Mean Prediction Interval Width (sharpness metric)

**Point Metrics:**
- **IC:** Spearman Information Coefficient (rank correlation)
- **DA:** Directional Accuracy (sign prediction)
- **RMSE/MAE:** Root Mean Squared Error and Mean Absolute Error

**Financial Metrics:**
- **Sharpe Ratio:** Risk-adjusted returns (annualized)
- **Sortino Ratio:** Downside deviation only
- **Calmar Ratio:** Return / Max Drawdown
- **Maximum Drawdown:** Largest peak-to-trough decline
- **Profit Factor:** Gross wins / gross losses

**Backtesting Strategy:**
- **Signal Generation:** Direction from median prediction, size inversely proportional to interval width
- **Multi-Horizon Comparison:** Comparative backtest across all 5 horizons
- **Risk Filters:** Max interval width threshold, minimum signal strength
- **Returns:** Gross (pre-transaction costs)

**Calibration Analysis:**
- **Reliability Diagrams:** Expected vs. observed quantile coverage
- **PIT Histograms:** Probability Integral Transform uniformity check
- **Coverage by Horizon:** Per-horizon calibration breakdown

## Performance Results

### Test Set Evaluation Metrics

**Evaluated on 13,599 test samples (10% subsample)**

#### Distributional Metrics

| Horizon | CRPS | PICP-80 | PICP-50 | MPIW-80 | MPIW-50 |
|---------|---------|---------|---------|---------|---------|
| 15m | 0.03331 | 0.000 | 0.000 | 0.00689 | 0.00314 |
| 30m | 0.00144 | 0.169 | 0.026 | 0.00772 | 0.00315 |
| 60m | 0.00869 | 0.001 | 0.000 | 0.00755 | 0.00346 |
| 2h | 0.01238 | 0.003 | 0.001 | 0.00652 | 0.00294 |
| 4h | 0.00431 | 0.030 | 0.017 | 0.00487 | 0.00254 |

**Note:** Low PICP values indicate miscalibration, suggesting the model produces overly narrow prediction intervals in this prototyping run. Full-scale training expected to improve calibration.

#### Point Metrics (Median Predictions)

| Horizon | IC | DA | RMSE | MAE |
|---------|---------|-------|----------|----------|
| 15m | 0.0011 | 0.517 | 0.07021 | 0.07016 |
| 30m | -0.0060 | 0.518 | 0.00488 | 0.00447 |
| 60m | -0.0014 | 0.529 | 0.02135 | 0.02116 |
| 2h | -0.0226 | 0.536 | 0.02702 | 0.02668 |
| 4h | -0.0260 | 0.545 | 0.01087 | 0.00989 |

**Directional Accuracy** ranges from 51.7% to 54.5%, showing slight edge over random (50%).

#### Backtest Results (Gross Returns)

| Horizon | Sharpe | Sortino | Max DD | Profit Factor | Hit Rate | Calmar | Total Return | Trades |
|---------|--------|---------|--------|---------------|----------|--------|--------------|--------|
| 15m | 2.09 | 2.97 | 6.95% | 1.052 | 51.7% | 5.80 | 26.4% | 13,599 |
| 30m | 2.42 | 3.37 | 15.6% | 1.059 | 51.8% | 4.58 | 45.1% | 13,599 |
| 60m | 1.44 | 1.95 | 29.4% | 1.034 | 52.9% | 1.81 | 34.3% | 13,599 |
| 2h | 1.94 | 2.64 | 50.4% | 1.046 | 53.6% | 2.46 | 74.6% | 13,599 |
| 4h | 3.31 | 4.52 | 70.7% | 1.076 | 54.5% | 9.11 | 300.1% | 13,599 |

**Key Observations:**
- **Best Sharpe:** 4h horizon (3.31) demonstrates strongest risk-adjusted performance
- **Trade-off:** Longer horizons show higher returns but larger drawdowns
- **Profit Factors:** All > 1.0 indicating profitable backtests (gross, pre-costs)
- **Hit Rates:** Modest edge (51.7-54.5%) consistent with IC results

**Calibration Summary:**
- Mean calibration error: 0.4721
- Max calibration error: 0.8539
- **Status:** Requires recalibration via full-scale training or post-hoc temperature scaling

### Interpretation

These results are from a **prototyping run** (10% data subsample, 4 epochs) designed to validate the complete pipeline. The model demonstrates:

1. **Functional architecture** - All components integrate correctly
2. **Positive signal** - Directional accuracy > 50%, positive Sharpe ratios
3. **Calibration issues** - Narrow prediction intervals (low PICP) require addressing
4. **Horizon patterns** - Longer horizons show stronger performance metrics

**Recommended Next Steps:**
1. Full-scale training on 100% data
2. Extended epochs (50-100 with early stopping)
3. Calibration tuning (temperature scaling or recalibration layer)
4. Transaction cost modeling for realistic backtest returns
5. Hyperparameter optimization (learning rate, architecture depth)

## Compute Environment

**Primary:** Google Colab with A100 GPU (80GB VRAM)  
**Secondary Resources:** 167.1 GB RAM, 12 CPU cores  
**Storage:** Google Drive integration for data persistence

**Memory Profile:**
- Model parameters: 6.87M
- Forward pass (B=128): ~70 GB
- Training overhead (Adam + gradients): +3 GB
- Total training memory: ~73 GB (91% utilization, safe margin)
- Gradient checkpointing: Not required (sufficient headroom)

**Optimal Batch Sizes:**
- Primary: 128 (optimal for financial time series, fits comfortably)
- Alternative: 64 (more gradient noise, useful for exploration)
- Avoid: >180 (exceeds memory, no performance gain)

## Installation & Setup

### Prerequisites

```bash
python >= 3.10
torch >= 2.0.0
numpy >= 1.24.0
pandas >= 2.0.0
```

### Dependencies

Key packages:
- **Data:** `databento`, `pyarrow`, `fastparquet`
- **Model:** `torch`, `einops`
- **Training:** `wandb`, `tensorboard`
- **Evaluation:** `scipy`, `scikit-learn`, `matplotlib`, `seaborn`

### Google Colab Setup

1. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Install dependencies:
```python
!pip install -q databento einops wandb
```

3. Verify GPU:
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
```

### Data Directory Structure

```
/content/drive/MyDrive/Colab Notebooks/Transformers/FP/
├── data/
│   ├── raw/
│   │   ├── nq_ohlcv_1m_raw.parquet          # 1-min data from Databento
│   │   └── rollover_dates.csv               # Generated metadata
│   ├── interim/
│   │   ├── nq_ohlcv_5min_aggregated.parquet
│   │   └── nq_ohlcv_5min_adjusted.parquet
│   └── processed/
│       ├── nq_features_full.parquet         # 24 features + 5 targets
│       ├── train_samples.parquet
│       ├── val_samples.parquet
│       ├── test_samples.parquet
│       └── column_info.csv
├── outputs/
│   ├── checkpoint_best.pt                   # Best model weights
│   ├── checkpoint_latest.pt                 # For resuming training
│   ├── training_history.json                # Loss curves
│   └── training_curves.png                  # Visualization
├── src/
│   ├── data/
│   ├── model/
│   ├── training/
│   └── evaluation/
├── configs/
│   ├── model_config.yaml
│   └── training_config.yaml
└── notebooks/
    ├── 01_data_acquisition.ipynb
    ├── 02_preprocessing.ipynb
    ├── 03_dataset_preparation.ipynb
    ├── 04_model_testing.ipynb
    ├── 05_training.ipynb
    ├── 05_full_training.ipynb
    ├── 06_evaluation.ipynb
    └── 06_production_evaluation.ipynb
```

## Usage

### Data Preparation (Phases 1-3)

Run notebooks `01_data_acquisition.ipynb`, `02_preprocessing.ipynb`, `03_dataset_preparation.ipynb` to run testing plus execution, or simply:  

```python
from src.data import DataLoader, RolloverAdjuster, FeatureEngineer
from src.data import DataPreprocessor, NQDataModule

# Load and aggregate 1-min to 5-min bars
loader = DataLoader(
    raw_path='/path/to/nq_ohlcv_1m_raw.parquet',
    output_dir='/path/to/interim/'
)
df_5min = loader.aggregate_to_5min()

# Apply rollover adjustment
adjuster = RolloverAdjuster.from_data(df_5min, threshold_pct=0.01, threshold_std=3.0)
df_adjusted = adjuster.adjust_prices(df_5min)

# Compute features and targets
engineer = FeatureEngineer()
df_features = engineer.compute_all_features(df_adjusted)
horizons = {'15m': 3, '30m': 6, '60m': 12, '2h': 24, '4h': 48}
df_final = engineer.add_targets(df_features, horizons)

# Create dataset and dataloaders
data_module = NQDataModule(
    data_path='/path/to/nq_features_full.parquet',
    batch_size=128,
    num_workers=4
)
data_module.setup()
```

### Model Instantiation (Phase 4)

Run notebook `04_model_testing.ipynb` to run testing.  
To execute, simply run:  

```python
from src.model import MIGT_TVDT
import yaml

# Load configuration
with open('configs/model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize model
model = MIGT_TVDT(config['model'])

# Model summary
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# Forward pass
batch = next(iter(data_module.train_dataloader()))
predictions = model(batch['features'], batch['mask'], batch['temporal_info'])
# predictions: (B, n_horizons=5, n_quantiles=7)
```

### Model Training (Phase 5)

For testing run `05_training.ipynb`.  
For execution run `05_full_training.ipynb`.  
The core training code is:  

```python
from src.training import QuantileLoss, Trainer
import torch

# Setup training
criterion = QuantileLoss(quantiles=config['quantile_regression']['quantiles'])
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device='cuda'
)

# Train with validation monitoring
trainer.fit(
    train_loader=data_module.train_dataloader(),
    val_loader=data_module.val_dataloader(),
    epochs=100,
    early_stopping_patience=10,
    checkpoint_dir='/path/to/checkpoints'
)
```

### Model Evaluation (Phase 6)

For testing run `06_evaluation.ipynb`.  
For execution run `06_production_evaluation.ipynb`.  
The core eval/analysis code is:  

```python
from src.evaluation import run_evaluation, format_evaluation_report
from src.evaluation import CalibrationByHorizon, MultiHorizonBacktester

# Load best checkpoint
checkpoint = torch.load('/path/to/checkpoint_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run comprehensive evaluation
results = run_evaluation(
    model=model,
    test_loader=data_module.test_dataloader(),
    device='cuda',
    horizons=['15m', '30m', '60m', '2h', '4h']
)

# Generate evaluation report
report = format_evaluation_report(results)
print(report)

# Calibration analysis
calibration = CalibrationByHorizon(n_quantiles=7)
calibration_results = calibration.analyze(
    predictions=results['predictions'],
    targets=results['targets'],
    horizons=results['horizons']
)

# Multi-horizon backtesting
backtester = MultiHorizonBacktester()
backtest_results = backtester.backtest_all_horizons(
    predictions=results['predictions'],
    targets=results['targets'],
    horizons=results['horizons']
)
```

## Scientific Foundations

### Key Hypotheses

**H1 (Non-Stationarity):** Instance normalization enables learning across market regimes with different volatility/price levels.

**H2 (Multivariate Dependencies):** Variable-centric embeddings capture feature interactions (e.g., volume-RSI) more effectively than timestep embeddings.

**H3 (Distributional Superiority):** Quantile regression achieves lower CRPS and better tail calibration than MSE point predictions.

**H4 (Temporal Cycles):** Composite positional encodings capture intraday, weekly, and seasonal patterns missed by standard sinusoidal encodings.

### Future Ablation Studies

| Configuration | Modification | Expected Impact |
|---------------|-------------|-----------------|
| No Variable Embedding | Standard timestep tokens | ↓ IC (entangled) |
| No Two-Stage Attention | Single attention stage | ↑ compute, ↓ performance |
| No Instance Normalization | LayerNorm only | ↓ regime adaptation |
| No Gating (LGU) | Remove gate mechanism | ↑ noise, wider intervals |
| Point Prediction (MSE) | MSE loss instead of QR | No uncertainty, ↑ CRPS |
| Manual Attention | Disable Flash Attention | ↑ memory (77GB), ↓ batch |

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

The 24-hour lookback window contains only historical bars. Target computation uses future data for labeling but these columns are excluded from model inputs.

### Data Leakage Prevention

**Purge Gaps:** 24-hour gaps between train/val and val/test splits ensure validation/test lookback windows don't overlap with training data.

**Example:**
- Train ends: 2021-12-31 23:59:59
- Purge: 2022-01-01 (288 bars)
- Val starts: 2022-01-02 00:00:00
- Val window at 2022-01-02 00:00 looks back 24h to 2022-01-01 00:00
- No overlap with train data (ends Dec 31)

This is standard practice for time series ML and prevents subtle leakage through temporal context mixing.

## Development Timeline

| Phase | Status | Duration | Objective |
|-------|--------|----------|-----------|
| 1 | ✅ Complete | Week 1 | Data Acquisition & Preprocessing |
| 2 | ✅ Complete | Week 2 | Feature Engineering |
| 3 | ✅ Complete | Week 3 | Dataset & DataLoader |
| 4 | ✅ Complete | Week 4 | Model Implementation |
| 5 | ✅ Complete | Week 5 | Training Pipeline |
| 6 | ✅ Complete | Week 6 | Evaluation & Analysis |

**Total Duration:** ~6 weeks (All phases complete)

## Project Outputs

### Trained Model
- **Checkpoint:** `checkpoint_best.pt` (82.7 MB)
- **Parameters:** 6.87M
- **Training:** 4 epochs on 10% data subsample
- **Validation Loss:** 0.014659

### Evaluation Results
- **Metrics:** CRPS, IC, DA, Sharpe, Sortino, Calmar ratios
- **Calibration:** Reliability diagrams, PIT histograms
- **Backtest:** Multi-horizon strategy evaluation
- **Reports:** JSON metrics, CSV summaries, PNG visualizations

### Documentation
- **Evaluation Report:** `evaluation_report.md` - Comprehensive test results

## References

[1] Y. Li et al., "Reinforcement learning with temporal and variable dependency-aware transformer for stock trading optimization," *Neural Networks*, vol. 192, p. 107905, 2025. https://arxiv.org/abs/2408.12446

[2] F. Gu et al., "MIGT: Memory Instance Gated Transformer Framework for Financial Portfolio Management," *arXiv preprint arXiv:2502.07280*, 2025. https://arxiv.org/abs/2502.07280

[3] S. M. Kazemi et al., "Time2Vec: Learning a Vector Representation of Time," *arXiv preprint arXiv:1907.05321*, 2019. https://arxiv.org/abs/1907.05321

[4] Y. Liu et al., "iTransformer: Inverted Transformers are Effective for Time Series Forecasting," *ICLR*, 2024. https://arxiv.org/abs/2310.06625

[5] A. Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017. https://arxiv.org/abs/1706.03762

[6] Y. Zhang and J. Yan, "Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting," *ICLR*, 2023. https://openreview.net/forum?id=vSVLM2j9eie

[7] T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," *NeurIPS*, 2022. https://arxiv.org/abs/2205.14135

[8] J. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," *arXiv preprint arXiv:2104.09864*, 2021. https://arxiv.org/abs/2104.09864

[9] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, 1997.

[10] B. Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting," *International Journal of Forecasting*, 2021.

## Repository Structure

```
FuturesSL/
├── README.md                          # This file
├── .gitignore                         # Git ignore patterns
├── configs/
│   ├── model_config.yaml              # Model architecture configuration
│   └── training_config.yaml           # Training hyperparameters
├── src/
│   ├── data/
│   │   ├── data_loader.py             # Data loading and aggregation
│   │   ├── rollover_adjustment.py     # Futures contract rollover
│   │   ├── feature_engineering.py     # Feature computation
│   │   ├── preprocessing.py           # Window creation and normalization
│   │   └── dataset.py                 # PyTorch Dataset and DataModule
│   ├── model/
│   │   ├── positional_encodings.py    # Multi-scale temporal encodings
│   │   ├── embeddings.py              # Variable embeddings
│   │   ├── temporal_attention.py      # Per-variable temporal attention
│   │   ├── variable_attention.py      # Cross-variable attention
│   │   ├── gated_instance_norm.py     # RevIN + LGU normalization
│   │   ├── quantile_heads.py          # Multi-horizon quantile decoders
│   │   └── migt_tvdt.py               # Complete model architecture
│   ├── training/
│   │   ├── loss_functions.py          # Quantile regression loss
│   │   ├── scheduler.py               # Learning rate scheduling
│   │   └── trainer.py                 # Training orchestration
│   └── evaluation/
│       ├── metrics.py                 # Distributional and financial metrics
│       ├── calibration.py             # Calibration analysis
│       ├── backtest.py                # Multi-horizon backtesting
│       └── inference.py               # Prediction pipeline
├── notebooks/
│   ├── 01_data_acquisition.ipynb      # Phase 1 testing
│   ├── 02_preprocessing.ipynb         # Phase 2 testing
│   ├── 03_dataset_preparation.ipynb   # Phase 3 testing
│   ├── 04_model_testing.ipynb         # Phase 4 testing
│   ├── 05_training.ipynb              # Phase 5 testing
│   ├── 05_full_training.ipynb         # Production training
│   ├── 06_evaluation.ipynb            # Phase 6 testing
│   └── 06_production_evaluation.ipynb # Production evaluation
└── evaluation_results/
    ├── metrics.json                   # Evaluation metrics
    ├── calibration.json               # Calibration results
    ├── backtest.json                  # Backtest results
    ├── predictions_targets.npz        # Model predictions and targets
    ├── evaluation_report.md           # Formatted report
    ├── migt_tvdt_architecture.png     # Architecture diagram
    └── *.png, *.csv                   # Visualizations and summaries
```

## License

This project is provided as-is for research and educational purposes.

## Contact

**Author:** Clemente Fortuna  
**Project:** FuturesSL - Distributional Forecasting for NASDAQ Futures  
**Repository:** https://github.com/cjfortu/FuturesSL

---

*Project Completed: December 2025*  
*Status: All 6 Development Phases Complete - Model Trained and Evaluated*