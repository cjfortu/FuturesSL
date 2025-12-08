# Dev Phase 4: Model Architecture - Implementation Documentation

**Status:** Complete  
**Last Updated:** December 2025

## Overview

Phase 4 implements the complete MIGT-TVDT hybrid model architecture for distributional NQ futures forecasting per the engineering specification (claude-engineering.md) and scientific document (grok-scientific.md).

## Deliverables

### Source Modules (`src/model/`)

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `__init__.py` | Package exports | All public classes |
| `positional_encodings.py` | Temporal encodings | TimeOfDayEncoding, DayOfWeekEncoding, Time2VecEncoding, CompositePositionalEncoding |
| `embeddings.py` | Variable projection | VariableEmbedding, InputEmbedding |
| `temporal_attention.py` | Per-variable attention | TemporalAttentionBlock, TemporalAggregation |
| `variable_attention.py` | Cross-variable attention | VariableAttentionBlock |
| `gated_instance_norm.py` | Normalization + gating | RevIN, LiteGateUnit, GatedInstanceNorm |
| `quantile_heads.py` | Quantile output | QuantileHead, MultiHorizonQuantileHead |
| `migt_tvdt.py` | Complete model | MIGT_TVDT |

### Configuration (`configs/`)

- `model_config.yaml` - Model hyperparameters

### Testing (`notebooks/`)

- `04_model_testing.ipynb` - Comprehensive test suite

## Architecture Summary

```
Input: (B, 288, 24) features + mask + temporal_info

1. RevIN normalize         (B, 288, 24) -> (B, 288, 24)
2. Variable embed + pos    (B, 288, 24) -> (B, 288, 24, 256)
3. Temporal attention x4   (B, 288, 24, 256) -> (B, 288, 24, 256)
4. Temporal aggregation    (B, 288, 24, 256) -> (B, 24, 256)
5. Variable attention x2   (B, 24, 256) -> (B, 24, 256) with gating
6. Pool variables          (B, 24, 256) -> (B, 256)
7. Quantile heads          (B, 256) -> (B, 5, 7)

Output: {quantiles: (B, 5, 7), norm_stats: dict}
```

## Key Implementation Details

### n_variables Correction

**Engineering doc specified:** 25  
**Actual from Phase 2:** 24 (4 OHLC + 20 derived)  
**Implementation uses:** 24

### Non-Crossing Quantiles

Guaranteed by cumulative softplus parameterization:
```python
base = self.base_proj(hidden)           # (B, 1)
deltas = self.delta_proj(hidden)        # (B, Q)
positive_deltas = F.softplus(deltas)    # Always > 0
cumulative = torch.cumsum(positive_deltas, dim=-1)
quantiles = base + cumulative           # Strictly increasing
```

### 4D Broadcasting

Positional encodings broadcast correctly to 4D variable embeddings:
```python
# Variable embedding: (B, T, V, D)
# Positional encoding: (B, T, D) -> unsqueeze(2) -> (B, T, 1, D)
x = x + pos_enc.unsqueeze(2)  # Broadcasts across V
```

## Parameter Count

| Component | Parameters | Percentage |
|-----------|------------|------------|
| RevIN | ~100 | <0.1% |
| Input embedding | ~160K | 4% |
| Temporal layers (4) | ~2.1M | 52% |
| Temporal aggregation | ~200K | 5% |
| Variable layers (2) | ~1.0M | 26% |
| Gated norms | ~130K | 3% |
| Output pool + heads | ~400K | 10% |
| **Total** | **~4M** | 100% |

## Memory Usage (A100)

| Batch Size | Peak Memory |
|------------|-------------|
| 64 | ~12 GB |
| 128 | ~20 GB |
| 256 | ~35 GB |

Well within 80GB A100 limit.

## Scientific Alignment

| Hypothesis | Implementation |
|------------|----------------|
| H8: TSA + variable embeddings | Two-stage attention (temporal → variable) |
| H9: Gated normalization | RevIN + LiteGateUnit per variable layer |
| H10: Quantile outputs | 7 quantiles with non-crossing guarantee |
| H11: Composite embeddings | 4-component positional encoding |

## Integration Points

### From Phase 3 (Dataset)

Expected batch dict:
- `features`: (B, 288, 24) float32
- `attention_mask`: (B, 288) bool
- `bar_in_day`: (B, 288) int32 ← **Fixed to 288**
- `day_of_week`: (B,) int32
- `day_of_month`: (B,) int32
- `day_of_year`: (B,) int32
- `targets`: (B, 5) float32

### For Phase 5 (Training)

Model returns:
- `quantiles`: (B, 5, 7) for pinball loss
- `norm_stats`: {mean, std} for optional denormalization

## Acceptance Criteria

- [x] Forward pass executes without error
- [x] Output shape: (B, 5, 7) for 5 horizons × 7 quantiles
- [x] Parameter count ~4M (within expected range)
- [x] Gradient flow verified (no vanishing/exploding)
- [x] VRAM <25GB with batch_size=128
- [x] Model save/load works correctly
- [x] Non-crossing quantiles guaranteed
- [x] Phase 3 dataloader compatible

## Updates/Changes Record

### Initial Implementation

**Date:** December 2025

- Implemented complete MIGT-TVDT architecture
- Used n_variables=24 (corrected from engineering doc's 25)
- All tests pass

### Phase 3 Compatibility Fix

**Date:** December 2025  
**Issue:** Phase 3 `collate_fn` padded `bar_in_day` to max_len (~275), not 288

**Resolution:** Phase 3 `collate_fn` updated to pad `bar_in_day` to MAX_SEQ_LEN=288, matching features shape. See Phase 3 documentation for details.

## Next Phase

**Phase 5: Training Pipeline**
- Loss functions (pinball, CRPS)
- Trainer class
- Learning rate scheduler
- Checkpointing
- Early stopping
