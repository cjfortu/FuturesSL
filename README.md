# FuturesSL: Transformer-Based Distributional Forecasting for NQ Futures

A sophisticated machine learning system for predicting NASDAQ 100 E-mini futures (NQ) forward returns across multiple time horizons using transformer architectures and distributional learning methods.

## Overview

FuturesSL is a supervised learning model designed to predict NQ forward log-returns over six horizons (5m, 15m, 30m, 60m, 2h, 4h) using approximately one trading week of minute-level market data (~6,825-6,900 bars). The project prioritizes prediction accuracy through a novel architecture that combines:

- **Two-Stage Attention (TSA)**: Separates temporal and variable-level attention mechanisms to capture both time-series patterns and cross-feature dependencies
- **Instance Normalization**: Enables cross-regime generalization by learning price trajectory shapes rather than absolute levels
- **Distributional Outputs**: Produces quantile predictions rather than point estimates to capture market uncertainty and heteroskedasticity
- **Lite Gate Units (LGU)**: Filters noise in low signal-to-noise ratio environments while maintaining gradient stability

### Key Features

- Full-sequence processing without patching to preserve temporal fidelity at 1-minute resolution
- Cyclical positional embeddings encoding time-of-day, day-of-week, day-of-month, and day-of-year patterns
- 24 engineered technical features grouped into 6 semantic categories (price, volume, trend, momentum, volatility, flow)
- Autoregressive multi-horizon prediction heads with teacher forcing during training
- Optimized for Google Colab A100 GPU (80GB VRAM) environment

## Project Architecture

### Collaborative Development Approach

FuturesSL is developed through collaboration between three AI systems, each with specialized expertise:

- **Claude (Opus 4.1)**: Engineering & Planning Lead - Architecture design, implementation specifications, phased development
- **Gemini (2.5 Pro)**: Research Lead - Literature review, mathematical foundations, methodology selection
- **Grok (Super 4/4.1)**: Chief Quant/ML Scientist - Scientific ideation, hypothesis generation, model concepts

This multi-AI approach ensures rigorous peer review at each stage, from research through scientific formulation to engineering implementation.

### Documentation Structure

The project maintains three authoritative documents:

1. **[gemini-research.md](docs/gemini-research.md)**: Research guide covering transformer architectures, positional encoding for financial time series, distributional forecasting methods, and references to source literature
2. **[grok-scientific.md](docs/grok-scientific.md)**: Scientific blueprint translating research into mathematical formulations, algorithmic strategies, and testable hypotheses
3. **[claude-engineering.md](docs/claude-engineering.md)**: Engineering implementation guide with PyTorch specifications, module designs, and phased development plan

### System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│  Databento API → Raw OHLCV → Feature Engineering → Parquet Storage │
│                                ↓                                    │
│                       Google Drive Sync                             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────┤
│  Colab VM ← Copy from Drive                                        │
│       ↓                                                             │
│  NQDataset → DataLoader → Model → Loss → Optimizer → Checkpoints   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         MODEL                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Input → InstanceNorm → Linear → PositionalEmbed                   │
│                              ↓                                      │
│                    Transformer Encoder Blocks                       │
│                              ↓                                      │
│              Multi-Horizon Quantile Prediction Heads                │
└─────────────────────────────────────────────────────────────────────┘
```

### Model Architecture: Grok-SL Transformer

The core architecture processes input through the following stages:

1. **Instance Normalization**: Per-sample, per-feature normalization across time to enforce stationarity
2. **Feature Embedding**: Linear projection of 24 raw features to d_model=512 dimensions
3. **Positional Encoding**: Cyclical temporal embeddings + learnable day-of-week embeddings
4. **Transformer Encoder**: 6-layer encoder with multi-head attention
   - *Phase 3 will replace with TSA (Temporal + Variable attention) + LGU blocks*
5. **Quantile Heads**: Multi-horizon quantile regression heads
   - *Phase 3 will add autoregressive conditioning between horizons*

**Key Specifications:**
- Sequence Length: T_max = 7,000 (padded)
- Model Dimension: d = 512
- Attention Heads: 8
- Encoder Layers: 6 (Phase 2 baseline)
- Quantiles: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
- Horizons: [5, 15, 30, 60, 120, 240] minutes

## Current Status: Dev Phase 2 Complete ✓

### Phase 1: Data Acquisition and Feature Engineering ✓

**Data Acquisition Module** (`src/data/acquisition.py`):
- Databento Historical API integration with NQ.v.0 (volume-based rollover)
- Chunked downloading (90-day segments) with retry logic
- Cost estimation before download
- Data validation (OHLC relationships, price sanity checks)
- Parquet storage with snappy compression

**Feature Engineering Module** (`src/data/features.py`):
- 24 technical indicators across 6 semantic groups:
  - **Price Dynamics** (4): Log-returns, high-low range, close location, open returns
  - **Volume** (3): Log-volume, volume delta, dollar volume
  - **Trend** (5): VWAP deviation (with RTH reset), MACD histogram, SMA deviations
  - **Momentum** (6): RSI, CCI, +DI/-DI, ADX, Rate of Change
  - **Volatility** (4): ATR, Bollinger %B, bandwidth, realized volatility
  - **Flow** (2): Money Flow Index, time gap feature
- Instance normalization support (per-sample statistics)
- Vectorized computation for efficiency

**Data Coverage:**
- **Period**: June 6, 2010 - December 3, 2025 (15.5 years)
- **Resolution**: 1-minute OHLCV bars
- **Contract**: NQ.v.0 (volume-based continuous)
- **Data Splits**:
  - Training: 2010-06-06 to 2019-12-31 (~9.5 years)
  - Validation: 2020-01-01 to 2022-12-31 (3 years)
  - Test: 2023-01-01 to 2025-12-03 (~3 years)

### Phase 2: Dataset and Baseline Model ✓

**Dataset Module** (`src/data/dataset.py`):
- `NQFuturesDataset`: Rolling window dataset with proper padding/masking
- Window strategy: ~6900 bars context, right-padded to T_max=7000
- Train stride: 60 bars (hourly non-overlapping samples)
- Val/Test stride: 1 bar (full sequential coverage)
- Temporal features: 8-channel encoding for cyclical positional embeddings
- Automatic normalization statistics handling
- Efficient batch generation with DataLoader utilities

**Baseline Model** (`src/model/baseline.py`):
- `InstanceNorm1d`: Per-sample, per-feature normalization with attention masking
- `CyclicalPositionalEncoding`: Time-of-day, day-of-week, day-of-month, day-of-year
- `BaselineTransformer`: Vanilla encoder with independent quantile heads (Phase 2)
- Forward pass validated: (B, T=7000, V=24) → (B, H=6, Q=7)
- Gradient flow tested: All parameters receive gradients
- Memory efficient: <40GB VRAM on A100 with batch_size=8

**Testing Infrastructure**:
- `tests/phase1_tests.ipynb`: Data acquisition and feature engineering validation
- `tests/phase2_tests.ipynb`: Dataset, model, and integration tests
- Comprehensive unit tests for all components
- Performance benchmarking and shape validation

**Critical Issues Resolved:**
- Timezone handling: Converted timestamps to naive (market-local time) for consistency
- Feature normalization: Proper handling of NaN values and padding
- Sequence boundary handling: Eliminated future data leakage

## Installation

### Prerequisites

- Python 3.8+
- Google Colab account (for A100 GPU access)
- Databento API key (available at https://databento.com)

### Dependencies

```bash
pip install databento pandas numpy pyarrow torch scipy tqdm
```

### Repository Structure

```
FuturesSL/
├── docs/
│   ├── gemini-research.md           # Research guide
│   ├── grok-scientific.md           # Scientific blueprint
│   ├── claude-engineering.md        # Engineering specifications
│   ├── phase1_documentation.md      # Phase 1 implementation notes
│   └── phase2_documentation.md      # Phase 2 implementation notes
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── acquisition.py           # Databento download module
│   │   ├── features.py              # Technical indicator computation
│   │   └── dataset.py               # PyTorch Dataset/DataLoader (Phase 2)
│   └── model/
│       ├── __init__.py
│       └── baseline.py              # Baseline Transformer (Phase 2)
├── tests/
│   ├── phase1_tests.ipynb           # Phase 1 testing notebook
│   └── phase2_tests.ipynb           # Phase 2 testing notebook (Colab)
├── futuresSL_prob-statement.txt     # Original problem statement
└── README.md
```

## Usage

### Data Acquisition (Phase 1)

```python
from src.data.acquisition import NQDataAcquisition

# Initialize acquisition module
acquisition = NQDataAcquisition(
    api_key="your_databento_key",
    output_dir="./data/raw"
)

# Estimate cost before downloading
cost = acquisition.estimate_cost("2020-01-01", "2020-12-31")
print(f"Estimated cost: ${cost:.2f}")

# Download data
df = acquisition.download_range("2020-01-01", "2020-12-31")

# Save to parquet
acquisition.save_parquet(df, "nq_2020.parquet")
```

### Feature Engineering (Phase 1)

```python
from src.data.features import NQFeatureEngineer, FEATURE_COLUMNS
import pandas as pd

# Load raw data
df = pd.read_parquet("./data/raw/nq_2020.parquet")

# Initialize feature engineer
engineer = NQFeatureEngineer()

# Compute all 24 features
features_df = engineer.compute_all_features(df)

# Compute forward return targets
targets_df = engineer.compute_targets(df)

# Save processed data
features_df.to_parquet("./data/processed/nq_features_2020.parquet")
targets_df.to_parquet("./data/processed/nq_targets_2020.parquet")

print(f"Feature columns: {FEATURE_COLUMNS}")
print(f"Feature shape: {features_df.shape}")
```

### Dataset and Model (Phase 2)

```python
from src.data import create_dataloaders, FEATURE_COLUMNS
from src.model import create_model
import torch

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    features_path='data/nq_features_v1.parquet',
    targets_path='data/nq_targets_v1.parquet',
    feature_columns=FEATURE_COLUMNS,
    batch_size=8,
)

# Create baseline model
model = create_model(num_features=24, d_model=512).cuda()

# Forward pass
for batch in train_loader:
    features = batch['features'].cuda()
    mask = batch['mask'].cuda()
    temporal = batch['temporal_features'].cuda()
    targets = batch['targets'].cuda()
    
    # Model prediction: (B, 6, 7) = (batch, horizons, quantiles)
    predictions = model(features, mask, temporal)
    
    print(f"Input shape: {features.shape}")       # (8, 7000, 24)
    print(f"Output shape: {predictions.shape}")   # (8, 6, 7)
    break
```

### Running Tests (Colab)

**Phase 1 Tests:**
1. Upload `tests/phase1_tests.ipynb` to Google Colab
2. Connect to A100 GPU runtime
3. Mount Google Drive
4. Run all cells sequentially

**Phase 2 Tests:**
1. Upload `tests/phase2_tests.ipynb` to Google Colab
2. Connect to A100 GPU runtime  
3. Ensure Phase 1 data is available in Drive
4. Run all cells sequentially

Expected output: All tests passing with validation metrics.

## Development Roadmap

### Phase 1: Data Acquisition and Feature Engineering ✓ COMPLETE
- ✓ Databento API integration
- ✓ Volume-based rollover handling
- ✓ 24 technical indicators with proper session handling
- ✓ Comprehensive unit and integration tests

### Phase 2: Dataset and Baseline Model ✓ COMPLETE
- ✓ Rolling window sampler with padding/masking
- ✓ Cyclical positional encoding computation
- ✓ Train/val/test splits with proper striding
- ✓ Instance normalization layer
- ✓ Baseline Transformer encoder
- ✓ Independent multi-horizon quantile heads
- ✓ Forward pass and gradient flow validation

### Phase 3: TSA + LGU Architecture (In Progress)
- [ ] Two-Stage Attention: Temporal + Variable attention blocks
- [ ] Lite Gate Units for noise filtering
- [ ] Group projection (6 semantic feature groups)
- [ ] Autoregressive multi-horizon heads with teacher forcing
- [ ] Full Grok-SL Transformer implementation

### Phase 4: Training Infrastructure (Weeks 5-6)
- [ ] Pinball loss with monotonicity constraints
- [ ] AdamW optimizer with cosine scheduling + warmup
- [ ] Mixed precision training (BF16/TF32)
- [ ] Gradient checkpointing for memory efficiency
- [ ] WandB logging integration
- [ ] Checkpoint save/restore

### Phase 5: Evaluation Framework (Weeks 7-8)
- [ ] CRPS (Continuous Ranked Probability Score)
- [ ] Information Coefficient (IC)
- [ ] Tail-weighted accuracy
- [ ] Sharpe/Sortino ratios (backtest proxy)
- [ ] Quantile calibration analysis
- [ ] Ablation studies for hypothesis validation

### Phase 6: Optimization and Deployment (Weeks 9-10)
- [ ] Hyperparameter tuning (Optuna)
- [ ] torch.compile optimization
- [ ] Ensemble methods
- [ ] Real-time inference pipeline
- [ ] Model export (ONNX if needed)

## Performance Targets

Based on hypotheses in grok-scientific.md:

| Metric | Target | Baseline Improvement |
|--------|--------|---------------------|
| Information Coefficient (IC) | > 0.05 | +0.03 from baseline |
| CRPS Reduction | -25% | vs. point estimate baseline |
| Cross-Regime Accuracy | +15-20% | via Instance Normalization |
| Tail Accuracy (top/bottom 10%) | > 55% | vs. ~50% random |
| Validation IC (5m horizon) | > 0.02 | Phase 2 validation criterion |

## Technical Stack

- **Framework**: PyTorch 2.0+ (native FlashAttention-2 planned for Phase 3)
- **Data Format**: Parquet with Snappy compression
- **Precision**: FP32 (Phase 2), BF16 training planned (Phase 4)
- **Hardware**: Google Colab A100 (80GB VRAM, 167GB RAM, 12 CPU cores)
- **Data Provider**: Databento Historical API (GLBX.MDP3 dataset)
- **Version Control**: Git + GitHub
- **Experiment Tracking**: WandB (Phase 4)

## Scientific Foundation

FuturesSL adapts insights from recent literature on transformer-based financial forecasting:

1. **Memory Instance Gated Transformer (MIGT)** - Instance normalization for cross-regime generalization, Lite Gate Units for noise filtering
2. **RL-TVDT** - Two-Stage Attention (Temporal + Variable), variable dependency modeling
3. **PatchTST** - Evaluated but not adopted (granularity mismatch for short horizons)
4. **Temporal Fusion Transformer (TFT)** - Multi-horizon forecasting architecture inspiration
5. **Quantile Regression Networks** - Distributional learning for heteroskedastic returns

See [gemini-research.md](docs/gemini-research.md) for complete references and theoretical justification.

## Evaluation Philosophy

FuturesSL bridges AI/ML and quantitative finance evaluation methodologies:

**AI/ML Metrics:**
- Continuous Ranked Probability Score (CRPS)
- Negative Log-Likelihood (NLL)
- Quantile calibration

**Finance Metrics:**
- Information Coefficient (Pearson/Spearman correlation)
- Sharpe/Sortino/Omega ratios
- Tail-weighted accuracy
- Risk-adjusted returns

This dual evaluation ensures the model is both technically sound and economically viable.

## Contributing

This project follows a structured peer review process:

1. Research proposals reviewed by Gemini (Research Lead)
2. Scientific formulations reviewed by Grok (Quant/ML Scientist)
3. Engineering implementations reviewed by Claude (Engineering Lead)

All changes must pass peer review from at least two AI systems before merging.

## License

[Add your license here]

## Citation

If you use FuturesSL in your research, please cite:

```
@software{futuressl2025,
  title={FuturesSL: Transformer-Based Distributional Forecasting for NQ Futures},
  author={Multi-AI Collaborative Team: Claude (Anthropic), Gemini (Google), Grok (xAI)},
  year={2025},
  url={https://github.com/cjfortu/FuturesSL}
}
```

## Acknowledgments

- **Databento** for high-quality futures market data
- **Google Colab** for A100 GPU infrastructure
- Research foundations from MIGT, RL-TVDT, and TFT papers

---

**Project Status**: Dev Phase 2 Complete | Active Development  
**Last Updated**: December 2025  
**Next Milestone**: Phase 3 - TSA + LGU Architecture