# Dev Phase 3 Documentation: Dataset & DataLoader Implementation

**Status:** Complete ✓  
**Duration:** As specified in engineering plan  
**Last Updated:** December 2025

---

## Updates & Changes

### Temporal Split Purge Gap Fix

**Date:** December 2025  
**Issue:** AssertionError: Train-Val gap (0.1h) < tolerance (24h)

**Root Cause - Combined Problems:**
1. Date strings defaulted to midnight (00:00:00), excluding most bars from boundary days
2. Contiguous splits caused validation lookback windows to overlap with train data

**Problem Example:**
- train_end="2021-12-31" → included only up to 2021-12-31 00:00:00
- Val started at 2021-12-31 00:00:05
- Val window at 2022-01-01 00:00 looks back 24h to 2021-12-31 00:00
- Lookback overlaps with train data from Dec 31 (subtle leakage via context mixing)

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
- **Purge: 2022-01-01 (288 bars dropped)**
- Val: 2022-01-02 to 2023-12-31 23:59:59
- **Purge: 2024-01-01 (288 bars dropped)**
- Test: 2024-01-02 to 2025-12-03

**Rationale:**
- End-of-day timestamps: Intuitive behavior (train_end includes full day)
- Purge gaps: Standard ML practice, prevents context mixing
- Clean separation: Val lookbacks don't touch train data
- Minimal loss: ~576 bars (~0.05% of 1.1M)

**Impact:**
- Gaps: 0.1h → 24.0h ✓
- Purged: ~576 bars total
- Ratios: Preserved (~70%/20%/10%)

---

### Normalization Logic Correction (RevIN)

**Date:** December 2025  
**Change:** Line 233 in normalize_window():
```python
std_safe = np.where(std > eps, std, 1.0)
```

**Rationale:** Low-variance features (std ≤ 1e-5) centered without noise amplification.

---

## Deliverables

### 1. Package Structure

```
src/data/
  __init__.py (updated)
  preprocessing.py (with purge gaps)
  dataset.py

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

**Sample Structure:**
```python
{
    'features': (288, 24) float32
    'attention_mask': (288,) bool
    'targets': (5,) float32
    'bar_in_day': (actual_length,) int32
    'day_of_week': int32
    'day_of_month': int32
    'day_of_year': int32
    'norm_stats': dict
}
```

**`collate_fn`:**
- Pads bar_in_day to batch max length
- Stacks other tensors

**`NQDataModule`:**
- `setup()`: Loads data, creates splits, initializes datasets
- DataLoaders with proper batching

---

## Testing Results

### All Tests Passed ✓

**Split Statistics (with purge gaps):**

| Split | Date Range | Samples | % |
|-------|-----------|---------|---|
| Train | 2010-06-07 to 2021-12-31 | ~580K | 70% |
| Val | 2022-01-02 to 2023-12-31 | ~164K | 20% |
| Test | 2024-01-02 to 2025-12-03 | ~85K | 10% |

**Temporal Gaps:**
- Train-Val: 24.0h ✓
- Val-Test: 24.0h ✓
- Purged: ~576 bars total

### Feature Variance

Typical distribution:
- High variance (std > 0.01): 85-90% of features in 95%+ windows
- Low variance (std ≤ 1e-5): 0-5% of features (ema_slope_*, roc_* in flat markets)

---

## Known Limitations

1. **Purge Period Data Loss:** ~576 bars dropped (~0.05% of data)
   - Standard practice, theoretically necessary
   - Minimal impact on model training

2. **Variable Window Length:** 273-276 bars
   - Handled via padding + masks
   - No performance impact

3. **Low-Variance Windows:** Some features near-constant
   - Correctly normalized (centered, not amplified)
   - Model learns low predictability signal

---

## Integration with Future Phases

**Phase 4 (Model Architecture):**
- Input: (B, 288, 24) features, (B, 288) masks, temporal info
- Uses: Variable embedding, positional encodings

**Phase 5 (Training):**
- Uses: train/val dataloaders
- Monitors: Quantile loss, CRPS
- Expects: Proper temporal separation (no leakage)

---

## Acceptance Criteria

- [x] Purge gaps implemented correctly
- [x] End-of-day timestamps for intuitive behavior
- [x] No data leakage (24h gaps validated)
- [x] Normalization handles low-variance features
- [x] All tests pass
- [x] Documentation complete

**Phase 3 Status: ✓ COMPLETE**

---

## Next Steps

**Phase 4: Model Architecture**
- Variable embedding (24 features)
- Positional encodings (temporal info from preprocessor)
- Temporal & variable attention
- Gated instance normalization
- Multi-horizon quantile heads (5 × 7)

---

**Last Updated:** December 2025  
**Engineering Lead:** Claude  
**Contributors:** Grok (end-of-day insight), Gemini (purge gap approach)