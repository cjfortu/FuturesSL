# Dev Phase 3 Documentation: Dataset & DataLoader Implementation

**Status:** Complete (Updated)  
**Duration:** As specified in engineering plan  
**Last Updated:** December 2025

---

## Updates & Changes

### Update 4: DataLoader Optimization (prefetch_factor, persistent_workers)

**Date:** December 2025  
**Issue:** Training time discrepancy between estimate and actual runtime

**Observed Behavior:**
- Estimated training: 0.4 min/epoch, Actual: ~2.5 min (6x slower)
- Estimated validation: 0.6 min/epoch, Actual: ~9 min (15x slower)

**Root Cause Analysis (team consensus):**

1. **Validation bottleneck (primary):** GPU-to-CPU synchronization from `.cpu()` calls in `trainer.py` for VRAM safety. Each of 736 validation batches incurs ~700ms sync overhead. This is architectural and not addressable via DataLoader tuning.

2. **Training bottleneck (secondary):** Per-batch `collate_fn` operations (padding, tensor construction) and worker reinit overhead per epoch.

**Changes Applied to `NQDataModule`:**

```python
# New parameters with defaults
prefetch_factor: int = 2,      # Overlap I/O with compute
persistent_workers: bool = True # Reuse workers across epochs

# Updated DataLoader creation (all three loaders)
return DataLoader(
    ...,
    prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
    persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
    ...
)
```

**Rationale:**
- `prefetch_factor=2`: Each worker prefetches 2 batches ahead, overlapping I/O with GPU compute
- `persistent_workers=True`: Avoids 5-10s worker process reinit per epoch
- `num_workers=4`: Kept at 4 (stable baseline; increasing to 8 requires profiling)
- Conditional application: `prefetch_factor` and `persistent_workers` only valid when `num_workers > 0`

**Expected Impact:**
- Marginal improvement (~5-15%) from worker persistence and prefetching
- Major validation bottleneck unaffected (requires architectural change to trainer)

**Note:** The timing estimate in `06_full_training.ipynb` should be updated with realistic benchmarks (~715ms/train batch, ~735ms/val batch) based on observed performance.

---

### Update 3: collate_fn bar_in_day Padding Fix

**Date:** December 2025  
**Issue:** Shape mismatch between `bar_in_day` and `features` tensors

**Problem:** Original `collate_fn` padded `bar_in_day` to `max_len` within each batch (273-276), but `features` is always padded to 288. This causes broadcasting errors in Phase 4 model when positional encodings are added to variable embeddings.

**Root Cause:** 
```python
# OLD - variable padding per batch
max_len = max(len(b) for b in bar_in_day_list)
```

**Fix Applied:**
```python
# NEW - fixed padding to match features
MAX_SEQ_LEN = 288  # Must match features padding
```

**Impact:** All batched outputs now have consistent shapes:
- `features`: (B, 288, 24)
- `attention_mask`: (B, 288)
- `bar_in_day`: (B, 288)

**Verification:** Phase 4 model forward pass now executes without shape errors.

---

### Temporal Split Purge Gap Fix

**Date:** December 2025  
**Issue:** AssertionError: Train-Val gap (0.1h) < tolerance (24h)

**Root Cause - Combined Problems:**
1. Date strings defaulted to midnight (00:00:00), excluding most bars from boundary days
2. Contiguous splits caused validation lookback windows to overlap with train data

**Solution - Purge Gaps with End-of-Day Timestamps:**

```python
def create_splits(
    df: pd.DataFrame,
    train_end: str = "2021-12-31",
    val_end: str = "2023-12-31",
    purge_hours: int = 24
):
    # End-of-day timestamps ensure full days included
    train_end_ts = pd.Timestamp(f"{train_end} 23:59:59").tz_localize('UTC')
    val_end_ts = pd.Timestamp(f"{val_end} 23:59:59").tz_localize('UTC')
    
    # Purge gaps prevent lookback overlap
    val_start_ts = train_end_ts + pd.Timedelta(hours=purge_hours)
    test_start_ts = val_end_ts + pd.Timedelta(hours=purge_hours)
    
    # Clean separation
    train_df = df[df.index <= train_end_ts]
    val_df = df[(df.index >= val_start_ts) & (df.index <= val_end_ts)]
    test_df = df[df.index >= test_start_ts]
```

**Split Structure:**
- Train: 2010-06-07 to 2021-12-31 23:59:59
- Purge: 2022-01-01 (288 bars dropped)
- Val: 2022-01-02 to 2023-12-31 23:59:59
- Purge: 2024-01-01 (288 bars dropped)
- Test: 2024-01-02 to 2025-12-03

---

### Normalization Logic Correction (RevIN)

**Date:** December 2025  
**Change:** Line 233 in normalize_window():
```python
std_safe = np.where(std > eps, std, 1.0)
```

**Rationale:** Low-variance features (std <= 1e-5) centered without noise amplification.

---

## Deliverables

### 1. Package Structure

```
src/data/
  __init__.py (updated)
  preprocessing.py (with purge gaps)
  dataset.py (with DataLoader optimizations)

notebooks/
  03_dataset_preparation.ipynb
```

### 2. Module: `src/data/preprocessing.py`

**Key Components:**
- `create_window()`: 24-hour lookback with padding to 288
- `normalize_window()`: RevIN with low-variance handling
- `extract_temporal_info()`: Positional encoding features
- `create_splits()`: Time-based splits with purge gaps
- `validate_no_leakage()`: Validates 24h separation

**Technical Specifications:**
- Max sequence: 288 bars
- Actual length: 273-276 bars
- Padding: Zero-padding at start
- Normalization eps: 1e-5
- Purge gap: 24 hours (configurable)

### 3. Module: `src/data/dataset.py`

**Key Components:**

**`NQFuturesDataset`:**
- Variable-length lookback (273-276 bars)
- Instance normalization per sample
- Skips first 288 bars (insufficient history)
- Optional subsampling for training

**Sample Structure:**
```python
{
    'features': (288, 24) float32
    'attention_mask': (288,) bool
    'targets': (5,) float32
    'bar_in_day': (288,) int32  # Padded to fixed length
    'day_of_week': int32
    'day_of_month': int32
    'day_of_year': int32
    'norm_stats': dict
}
```

**`collate_fn`:**
- Pads bar_in_day to 288 (fixed, matches features)
- Stacks other tensors

**`NQDataModule`:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| batch_size | 128 | Samples per batch |
| num_workers | 4 | Parallel data loading processes |
| pin_memory | True | Pin tensors for faster GPU transfer |
| prefetch_factor | 2 | Batches prefetched per worker |
| persistent_workers | True | Reuse workers across epochs |
| subsample_fraction | None | Training subsample ratio |

---

## Testing Results

### All Tests Passed

**Split Statistics (with purge gaps):**

| Split | Date Range | Samples | % |
|-------|-----------|---------|---|
| Train | 2010-06-07 to 2021-12-31 | ~580K | 70% |
| Val | 2022-01-02 to 2023-12-31 | ~164K | 20% |
| Test | 2024-01-02 to 2025-12-03 | ~85K | 10% |

**Temporal Gaps:**
- Train-Val: 24.0h
- Val-Test: 24.0h
- Purged: ~576 bars total

---

## Known Limitations

1. **Purge Period Data Loss:** ~576 bars dropped (~0.05% of data)
   - Standard practice, theoretically necessary
   - Minimal impact on model training

2. **Variable Window Length:** 273-276 bars
   - Handled via padding + masks
   - No performance impact

3. **Validation Speed:** ~735ms/batch due to VRAM safety mechanism
   - CPU offload in trainer.py causes GPU-CPU sync per batch
   - Not addressable via DataLoader tuning

---

## Integration with Future Phases

**Phase 4 (Model Architecture):**
- Input: (B, 288, 24) features, (B, 288) masks, temporal info
- Uses: Variable embedding, positional encodings

**Phase 5 (Training):**
- Uses: train/val dataloaders with prefetching
- Monitors: Quantile loss, CRPS
- Note: Update timing estimates with realistic benchmarks

**Phase 6 (Evaluation):**
- Uses: test dataloader
- Same DataLoader configuration as validation

---

## Acceptance Criteria

- [x] Purge gaps implemented correctly
- [x] End-of-day timestamps for intuitive behavior
- [x] No data leakage (24h gaps validated)
- [x] Normalization handles low-variance features
- [x] DataLoader optimization (prefetch, persistent workers)
- [x] All tests pass
- [x] Documentation complete

**Phase 3 Status: COMPLETE (Updated)**

---

## Next Steps

**Phase 4: Model Architecture**
- Variable embedding (24 features)
- Positional encodings (temporal info from preprocessor)
- Temporal & variable attention
- Gated instance normalization
- Multi-horizon quantile heads (5 x 7)

---

**Last Updated:** December 2025  
**Engineering Lead:** Claude  
**Contributors:** Grok (DataLoader tuning), Gemini (bottleneck diagnosis)