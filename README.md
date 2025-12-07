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

1. **[gemini-research.md](gemini-research.md)**: Research guide covering transformer architectures, positional encoding for financial time series, distributional forecasting methods, and references to source literature
2. **[grok-scientific.md](grok-scientific.md)**: Scientific blueprint translating research into mathematical formulations, algorithmic strategies, and testable hypotheses
3. **[claude-engineering.md](claude-engineering.md)**: Engineering implementation guide with PyTorch specifications, module designs, and phased development plan

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
│  Input → InstanceNorm → GroupProjection → PositionalEmbed          │
│                              ↓                                      │
│                    TSA+LGU Encoder Blocks                          │
│                              ↓                                      │
│              Multi-Horizon Quantile Prediction Heads                │
└─────────────────────────────────────────────────────────────────────┘
```

### Model Architecture: Grok-SL Transformer

The core architecture processes input through the following stages:

1. **Instance Normalization**: Per-sample, per-feature normalization across time to enforce stationarity
2. **Group Projection**: 24 raw features projected into 6 semantic groups (price, volume, trend, momentum, volatility, flow)
3. **Positional Encoding**: Cyclical temporal embeddings + learnable day-of-week embeddings
4. **TSA Encoder**: 8-12 stacked blocks, each containing:
   - Temporal attention (per variable, across time)
   - Variable attention (per timestep, across features)
   - Lite Gate Units for noise filtering
   - Feed-forward networks
5. **Autoregressive Heads**: Multi-horizon quantile regression heads where shorter horizons condition longer ones

**Key Specifications:**
- Sequence Length: T_max = 7,000 (padded)
- Model Dimension: d = 512
- Attention Heads: 8 per stage
- Quantiles: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
- Horizons: [5, 15, 30, 60, 120, 240] minutes

## Current Status: Dev Phase 1 Complete ✓

### Implemented Components

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

**Testing Infrastructure** (`phase1_tests.ipynb`):
- Comprehensive unit tests for all 24 features
- Integration tests for end-to-end pipeline
- Validation of computational correctness
- Performance benchmarking

### Critical Bug Fixes

Through multi-AI peer review, the following issues were identified and resolved:

1. **VWAP Calculation Error**: Fixed numerator aggregation to prevent look-ahead bias
2. **Money Flow Index Computation**: Corrected typical price calculation and positive/negative flow logic
3. **Dataset Boundary Handling**: Eliminated potential future data leakage in rolling window construction

### Data Coverage

- **Period**: June 6, 2010 - December 3, 2025 (15.5 years)
- **Resolution**: 1-minute OHLCV bars
- **Contract**: NQ.v.0 (volume-based continuous)
- **Data Splits**:
  - Training: 2010-06-06 to 2019-12-31 (~9.5 years)
  - Validation: 2020-01-01 to 2022-12-31 (3 years)
  - Test: 2023-01-01 to 2025-12-03 (~3 years)

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
├── data/
│   ├── __init__.py
│   ├── acquisition.py          # Databento download module
│   └── features.py              # Technical indicator computation
├── src/
│   └── __init__.py
├── gemini-research.md           # Research guide
├── grok-scientific.md           # Scientific blueprint
├── claude-engineering.md        # Engineering specifications
├── phase1_tests.ipynb           # Dev Phase 1 testing notebook
├── futuresSL_prob-statement.txt # Original problem statement
└── README.md
```

## Usage

### Data Acquisition

```python
from data.acquisition import NQDataAcquisition, DatabentoConfig

# Initialize acquisition module
acquisition = NQDataAcquisition(
    api_key=DatabentoConfig.API_KEY,
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

### Feature Engineering

```python
from data.features import NQFeatureEngineer
import pandas as pd

# Load raw data
df = pd.read_parquet("./data/raw/nq_2020.parquet")

# Initialize feature engineer
engineer = NQFeatureEngineer()

# Compute all 24 features
features_df = engineer.compute_all_features(df)

# Save processed features
features_df.to_parquet("./data/processed/nq_features_2020.parquet")
```

### Running Tests (Colab)

1. Upload `phase1_tests.ipynb` to Google Colab
2. Connect to A100 GPU runtime
3. Mount Google Drive
4. Run all cells sequentially

Expected output: All tests passing with performance metrics for each feature computation.

## Development Roadmap

### Phase 1: Data Acquisition and Feature Engineering ✓ COMPLETE
- ✓ Databento API integration
- ✓ Volume-based rollover handling
- ✓ 24 technical indicators
- ✓ Comprehensive testing

### Phase 2: Dataset and DataLoader (Weeks 1-2)
- Rolling window sampler with padding/masking
- Positional encoding computation
- Train/val/test splits
- Efficient batching for A100

### Phase 3: Model Architecture (Weeks 3-4)
- Instance normalization layer
- Group projection module
- Two-stage attention blocks
- Lite gate units
- Multi-horizon quantile heads

### Phase 4: Training Infrastructure (Weeks 5-6)
- Pinball loss with monotonicity constraints
- AdamW optimizer with cosine scheduling
- Mixed precision training (BF16)
- Gradient checkpointing for memory efficiency
- WandB logging integration

### Phase 5: Evaluation Framework (Weeks 7-8)
- CRPS (Continuous Ranked Probability Score)
- Information Coefficient (IC)
- Tail-weighted accuracy
- Sharpe/Sortino ratios (backtest proxy)
- Quantile calibration analysis

### Phase 6: Optimization and Deployment (Weeks 9-10)
- Hyperparameter tuning (Optuna)
- Ensemble methods
- Real-time inference pipeline
- Model compression (if needed)

## Performance Targets

Based on hypotheses in grok-scientific.md:

| Metric | Target | Baseline Improvement |
|--------|--------|---------------------|
| Information Coefficient (IC) | > 0.05 | +0.03 from baseline |
| CRPS Reduction | -25% | vs. point estimate baseline |
| Cross-Regime Accuracy | +15-20% | via Instance Normalization |
| Tail Accuracy (top/bottom 10%) | > 55% | vs. ~50% random |

## Technical Stack

- **Framework**: PyTorch 2.0+ (native FlashAttention-2)
- **Data Format**: Parquet with Snappy compression
- **Precision**: BF16 training, FP32 accumulation
- **Hardware**: Google Colab A100 (80GB VRAM, 167GB RAM, 12 CPU cores)
- **Data Provider**: Databento Historical API
- **Version Control**: Git + GitHub

## Scientific Foundation

FuturesSL adapts insights from recent literature on transformer-based financial forecasting:

1. **Memory Instance Gated Transformer (MIGT)** - Instance normalization for cross-regime generalization
2. **RL-TVDT** - Variable dependency attention for multi-variate time series
3. **PatchTST** - Evaluated but not adopted (granularity mismatch for short horizons)
4. **Temporal Fusion Transformer (TFT)** - Multi-horizon forecasting architecture inspiration
5. **Quantile Regression Networks** - Distributional learning for heteroskedastic returns

See [gemini-research.md](gemini-research.md) for complete references and theoretical justification.

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

**Project Status**: Dev Phase 1 Complete | Active Development  
**Last Updated**: December 2025  
**Contact**: [Your contact information]
