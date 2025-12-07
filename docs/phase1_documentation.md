# Dev Phase 1: Data Acquisition and Feature Engineering

**Version:** 1.0  
**Status:** Complete  
**Date:** 2025-12-06

## Overview

This document captures the Dev Phase 1 implementation of the NQ Futures Forward Return Prediction system, covering data acquisition from Databento and feature engineering per the specifications in grok-scientific.md and claude-engineering.md.

## Modules Delivered

### 1. `src/data/acquisition.py`

**Purpose:** Download and validate NQ futures OHLCV data from Databento.

**Key Classes:**
- `DatabentoConfig`: Configuration constants for API access
- `NQDataAcquisition`: Main data acquisition handler

**Features:**
- Volume-based rollover continuous contract (NQ.v.0) per problem statement
- Chunked downloading (90-day segments) to prevent API timeouts
- Exponential backoff retry logic for transient failures
- OHLCV validation with price sanity checks
- Parquet storage with snappy compression

**Critical Implementation Note:**
Databento's `to_df()` method automatically converts fixed-point integer prices to floats. The module does NOT apply additional scaling (e.g., 1e-9 multiplication) as this would corrupt the data.

### 2. `src/data/features.py`

**Purpose:** Compute 24 derived technical features from OHLCV data.

**Key Classes:**
- `FeatureEngineer`: Main feature computation class

**Feature Specification (V=24 per grok-scientific.md Section 3.1):**

| Group | Features | Count |
|-------|----------|-------|
| F_P (Price) | log_return, hl_range, close_location, open_return | 4 |
| F_V (Volume) | log_volume, log_volume_delta, dollar_volume | 3 |
| F_T (Trend) | vwap_deviation, macd_histogram, sma_20/50/200_dev | 5 |
| F_M (Momentum) | rsi_norm, cci, plus_di, minus_di, adx, roc_norm | 6 |
| F_σ (Volatility) | atr_norm, bb_pct_b, bb_bandwidth, realized_vol | 4 |
| F_VW (Flow) | mfi_norm, time_gap | 2 |

**Key Implementation Details:**

1. **VWAP Session Reset:** Per grok-scientific.md, VWAP resets at RTH open (09:30 ET) to capture intraday session dynamics, not a rolling calculation.

2. **MFI Direction:** Uses typical price direction `(H+L+C)/3` for positive/negative flow classification, not close price direction.

3. **Time Gap Formula:** `ln(1 + max(Δt_minutes - 1, 0))` where normal 1-minute intervals produce 0.

4. **Normalization:** Features bounded to reasonable ranges:
   - RSI, MFI: normalized to [-1, 1]
   - DI, ADX: scaled to [0, 1]
   - CCI: scaled by 100

## Test Coverage

### Unit Tests
- Feature engineer initialization
- Feature column specification (24 features, 6 groups)
- Price dynamics computation
- Volume feature computation
- Momentum indicator computation
- Volatility feature computation
- VWAP session reset behavior
- MFI typical price direction
- Time gap formula verification
- Target computation

### Integration Tests
- Full pipeline execution
- NaN rate after warmup period
- Parquet roundtrip integrity
- Feature-target timestamp alignment

## Validation Criteria Status

| Criterion | Target | Status |
|-----------|--------|--------|
| Feature count | 24 | ✓ |
| Feature groups | 6 | ✓ |
| OHLC validity | <0.1% invalid | ✓ |
| NaN rate after warmup | <1% | ✓ |
| Parquet size estimate | <500MB | ✓ |

## API Compatibility

### Internal API

```python
# Feature computation
from src.data.features import FeatureEngineer, FEATURE_COLUMNS, FEATURE_GROUPS

fe = FeatureEngineer()
features = fe.compute_all_features(ohlcv_df)
targets = fe.compute_targets(ohlcv_df)

# Get indices for model's GroupProjection
indices = fe.get_feature_indices()
```

### Expected by Phase 2

The following are consumed by Dev Phase 2 (Dataset/DataLoader):
- `FEATURE_COLUMNS`: List of 24 feature names in canonical order
- `FEATURE_GROUPS`: Dict mapping group names to feature lists
- `TARGET_HORIZONS`: List [5, 15, 30, 60, 120, 240]
- Parquet files: `nq_features_v1.parquet`, `nq_targets_v1.parquet`
- Stats file: `feature_stats.json` with means, stds, indices

## Directory Structure

```
/content/drive/MyDrive/Colab Notebooks/Transformers/FP/
├── src/
│   ├── init.py
│   └── data/
│       ├── init.py
│       ├── acquisition.py
│       └── features.py
├── tests/
│   └── phase1_tests.ipynb
├── docs/
│   └── phase1_documentation.md
└── data/
├── raw/
│   └── nq_ohlcv_1m_raw.parquet
└── processed/
├── nq_features_v1.parquet
├── nq_targets_v1.parquet
└── feature_stats.json
```

## Usage Example

```python
# Data acquisition (requires Databento API key)
from src.data.acquisition import NQDataAcquisition, download_full_dataset

# Check cost first
acquisition = NQDataAcquisition(api_key="db-xxx", output_dir="./data/raw")
cost = acquisition.estimate_cost("2010-06-06", "2025-12-03")

# Download (set dry_run=False after confirming cost)
df = download_full_dataset("./data/raw", dry_run=False)

# Feature engineering
from src.data.features import compute_and_save_features

features_path, targets_path, stats_path = compute_and_save_features(
    raw_data_path="./data/raw/nq_ohlcv_1m_raw.parquet",
    output_dir="./data/processed"
)
```

## Update Log

| Date | Change | Reason |
|------|--------|--------|
| 2025-12-06 | Initial release | Dev Phase 1 complete |

| Date | Change | Reason |
|------|--------|--------|
| 2025-12-06 | Added degraded days inspection cell | Validate Databento quality warnings for scientific rigor |

## Data Quality Notes

### Degraded Dates Handling

Databento flagged 15 dates (2017-2025) as "degraded" during download. Inspection revealed:

- **Cause**: Exchange outages, packet loss (<1%), or symbology gaps
- **Impact**: <0.1% of dataset (~15k of 5.2M bars)
- **Handling**: 
  - `time_gap` feature captures temporal discontinuities
  - Rolling indicators use `min_periods=1` for robustness
  - Instance normalization makes model regime-invariant
  - Phase 2 Dataset will mask via `stats.json['degraded_days']`

**Validation Criteria**:
- Completeness >90%: Usable
- Zeros in OHLC: Corrupted (filter)
- Max 1-min return >10%: Corrupted (filter)

All 15 degraded dates passed validation (no corruption detected).

## References

- grok-scientific.md Section 3.1: Feature specification
- gemini-research.md Section 4: Feature engineering insights
- claude-engineering.md Section 3.1: Implementation details
- Databento API: https://databento.com/docs
