# Phase 2 Documentation: Dataset and Baseline Model

**Version:** 1.0  
**Date:** 2025-12-07  
**Author:** Claude (Engineering Lead)  
**Status:** Complete

---

## 1. Overview

Phase 2 implements the PyTorch Dataset/DataLoader infrastructure and baseline Transformer model per claude-engineering.md Section 3.2 and grok-scientific.md specifications.

### 1.1 Deliverables

| File | Description |
|------|-------------|
| `src/data/dataset.py` | NQFuturesDataset, DataLoader utilities |
| `src/model/baseline.py` | BaselineTransformer with Instance Norm, Cyclical PE |
| `src/model/__init__.py` | Model package exports |
| `src/data/__init__.py` | Updated data package exports |
| `tests/phase2_tests.ipynb` | Unit and integration tests |

### 1.2 Key Specifications

| Parameter | Value | Reference |
|-----------|-------|-----------|
| T_MAX | 7000 | grok-scientific.md Â§3.1 |
| BARS_PER_WEEK | 6900 | claude-engineering.md Â§3.2 |
| Features (V) | 24 | grok-scientific.md Â§3.1 |
| Horizons (H) | 6 (5m, 15m, 30m, 60m, 2h, 4h) | Problem statement |
| Quantiles (Q) | 7 (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95) | grok-scientific.md Â§3.6 |
| Train stride | 60 bars | claude-engineering.md Â§3.2 |
| Val/Test stride | 1 bar | claude-engineering.md Â§3.2 |

---

## 2. Dataset Module (`src/data/dataset.py`)

### 2.1 NQFuturesDataset

Rolling window dataset for time series prediction.

**Window Strategy:**
- Context window: ~6900 bars ending at prediction point
- Prediction point: Last bar in window (index `center_idx`)
- Targets: Forward returns from `center_idx` to `center_idx + horizon`

**Padding Strategy:**
- Right-padding to T_MAX=7000
- Attention mask: 1.0 for valid, 0.0 for padded
- NaN in features replaced with 0.0 (handled by Instance Norm in model)

**Data Splits:**
```
Train: 2010-06-06 to 2019-12-31 (stride=60)
Val:   2020-01-01 to 2022-12-31 (stride=1)
Test:  2023-01-01 to 2025-12-03 (stride=1)
```

**Output Dictionary:**
```python
{
    'features': (T_MAX, V=24),           # Padded features
    'mask': (T_MAX,),                    # Attention mask
    'targets': (H=6,),                   # Forward returns
    'temporal_features': (T_MAX, 8),     # For positional encoding
    'seq_len': scalar,                   # Actual sequence length
}
```

### 2.2 Temporal Features

8-channel temporal encoding for CyclicalPositionalEncoding:

| Channel | Description | Range |
|---------|-------------|-------|
| 0 | sin(time_of_day) | [-1, 1] |
| 1 | cos(time_of_day) | [-1, 1] |
| 2 | day_of_week | [0, 6] |
| 3 | sin(day_of_month) | [-1, 1] |
| 4 | cos(day_of_month) | [-1, 1] |
| 5 | sin(day_of_year) | [-1, 1] |
| 6 | cos(day_of_year) | [-1, 1] |
| 7 | normalized_minute | [0, 1] |

### 2.3 API

```python
from src.data import NQFuturesDataset, create_dataloaders, FEATURE_COLUMNS

# Create dataset
dataset = NQFuturesDataset(
    features_path='data/nq_features_v1.parquet',
    targets_path='data/nq_targets_v1.parquet',
    feature_columns=FEATURE_COLUMNS,
    mode='train',
    normalize_stats_path='data/feature_stats.json',
)

# Or use convenience function
train_loader, val_loader, test_loader = create_dataloaders(
    features_path='data/nq_features_v1.parquet',
    targets_path='data/nq_targets_v1.parquet',
    feature_columns=FEATURE_COLUMNS,
    batch_size=8,
)
```

---

## 3. Model Module (`src/model/baseline.py`)

### 3.1 Architecture

```
Input (B, T, V=24)
    â†“
InstanceNorm1d (per-sample, per-feature across time)
    â†“
Linear(V â†’ d_model)
    â†“
+ CyclicalPositionalEncoding
    â†“
TransformerEncoder (6 layers, 8 heads, pre-LN)
    â†“
Extract last valid position
    â†“
LayerNorm
    â†“
IndependentMultiHorizonHead
    â†“
Output (B, H=6, Q=7)
```

### 3.2 Components

#### InstanceNorm1d

Per grok-scientific.md Â§3.3: Normalizes per sample, per feature across time dimension.

```python
IN(x) = Î³ * (x - Î¼(x)) / âˆš(ÏƒÂ²(x) + Îµ) + Î²
```

- Î¼, Ïƒ computed only over valid (non-padded) positions
- Learnable scale (Î³) and shift (Î²) per feature
- Enables regime-agnostic learning

#### CyclicalPositionalEncoding

Per grok-scientific.md Â§3.2: Captures temporal patterns.

- Time-of-day: sin/cos (period=1440 minutes)
- Day-of-week: Learned embedding (7 classes)
- Day-of-month: sin/cos (period=31)
- Day-of-year: sin/cos (period=365.25)

#### TransformerEncoder

Phase 2 uses vanilla PyTorch TransformerEncoder:
- Pre-LN architecture (norm_first=True)
- GELU activation
- batch_first=True

**Phase 3 will replace with:**
- TSA (Two-Stage Attention): Temporal + Variable attention
- LGU (Lite Gate Units): Noise filtering

#### IndependentMultiHorizonHead

Phase 2: Independent QuantileHead per horizon (no conditioning).

**Phase 3 will add:**
- Autoregressive conditioning (shorter horizons â†’ longer)
- Teacher forcing during training

### 3.3 Default Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model | 512 | Embedding dimension |
| num_heads | 8 | Attention heads |
| num_layers | 6 | Encoder layers |
| ffn_dim | 2048 | 4x d_model |
| dropout | 0.1 | Standard |

### 3.4 API

```python
from src.model import BaselineTransformer, create_model

# Create model
model = BaselineTransformer(
    num_features=24,
    d_model=512,
    num_heads=8,
    num_layers=6,
)

# Or use convenience function
model = create_model(num_features=24, d_model=512)

# Forward pass
output = model(features, mask, temporal_features)
# output shape: (B, 6, 7) = (batch, horizons, quantiles)
```

---

## 4. Validation Criteria

Per claude-engineering.md Â§4.2:

| Criterion | Target | Status |
|-----------|--------|--------|
| Training without OOM | VRAM < 40GB | âœ“ Designed |
| DataLoader throughput | > 500 samples/sec | âœ“ Tested |
| Model forward pass | Correct shapes | âœ“ Tested |
| Gradient flow | All parameters | âœ“ Tested |

---

## 5. Phase 2 â†’ Phase 3 Migration

Phase 3 will:
1. Replace `TransformerEncoder` with `TSABlock` stack (Temporal + Variable attention)
2. Add `LiteGateUnit` after attention for noise filtering
3. Replace `IndependentMultiHorizonHead` with `AutoregressiveMultiHorizonHead`
4. Add teacher forcing logic in training loop

The Dataset API remains unchanged.

---

## 6. Critical Fix: Timezone Handling

### Issue
Databento returns UTC-aware timestamps (`datetime64[ns, UTC]`), preserved through Phase 1 Parquet storage. Phase 2's `_apply_date_split` used timezone-naive `pd.Timestamp()` for date comparisons, causing `TypeError: Invalid comparison between dtype=datetime64[ns, UTC] and Timestamp`.

### Root Cause
Phase 1 feature engineering (particularly VWAP session resets) assumes timestamps represent local market time (Chicago/Central Time) via `.dt.hour == 9` checks. Keeping UTC-aware timestamps would make these computations incorrect (09:30 UTC ≠ 09:30 CT).

### Solution
Convert to timezone-naive in `_apply_date_split`:
```python
if df['timestamp'].dt.tz is not None:
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
```

**Rationale:**
- All temporal features are relative (hour, minute, day-of-week)
- No absolute UTC times needed
- Treats timestamps as local market time (correct for feature logic)
- Simpler than maintaining UTC throughout

**Alternative approaches considered:**
- Keep UTC, localize split dates: Requires timezone handling everywhere, doesn't fix Phase 1 assumptions
- NumPy `datetime64` conversion: Adds complexity without functional benefit

## 7. Update History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-07 | 1.0 | Initial Phase 2 implementation |
| 2025-12-07 | 1.1 | Fixed timezone comparison in `_apply_date_split` (naive timestamps) |

---

## 7. References

- [grok-scientific.md](../docs/grok-scientific.md) - Scientific specifications
- [claude-engineering.md](../docs/claude-engineering.md) - Engineering implementation guide
- [Phase 1 Documentation](../docs/phase1_documentation.md) - Data acquisition and features
