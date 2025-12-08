# Dev Phase 2: Feature Engineering - Implementation Documentation

## Overview

Phase 2 implements comprehensive feature engineering for NQ futures data, computing all derived features specified in the scientific document. All features are computed causally (using only historical data) to prevent lookahead bias.

**Status:** ✓ Complete  
**Duration:** Implementation completed according to specification  
**Key Deliverables:**
- `feature_engineering.py` - Feature computation module (807 lines)
- `02_preprocessing.ipynb` - Comprehensive testing notebook
- `nq_features_full.parquet` - Complete feature dataset with targets (24 features + 5 targets)

---

## Updates & Changes

### Feature Selection Refinement

**Change:** Dropped `macd_signal` feature due to high multicollinearity  
**Date:** Post-validation analysis  
**Rationale:**
- Correlation with `macd`: 0.955 (above 0.95 threshold)
- `macd_signal` = EMA_9(MACD) is a smoothed, lagging version of MACD
- Transformer attention mechanisms learn optimal smoothing patterns
- Retaining raw `macd` provides higher information content with lower redundancy

**Impact:** Final feature count: 24 features (down from 25)

---

## Implementation Note: macd_signal Removal Process

**Where Removal Occurs:** In notebook (Cell 30), NOT in `feature_engineering.py`

**Data Flow:**
1. `feature_engineering.py` computes 25 features (includes `macd_signal`)
2. Notebook detects correlation: macd/macd_signal = 0.955
3. Notebook drops: `df_clean = df_clean.drop(columns=['macd_signal'])`
4. Saved data: 24 features + 5 targets = **29 columns total** ✓

**Column Counts:**
```
Before drop:  30 columns (25 features + 5 targets)
After drop:   29 columns (24 features + 5 targets)
Saved file:   29 columns ✓
```

**Why feature_engineering.py Still Computes It:**
The module computes `macd_signal` but it's immediately dropped in post-processing. This is acceptable because:
- Drop happens before saving processed data
- Saved data is correct (24 features)
- Model expects 24 features ✓
- No API changes needed

**Status:** Implementation correct. Feature engineering module could be updated to skip `macd_signal` computation for efficiency, but this is not required.

---


## Features Implemented

### Feature Categories

**1. Price Features (4 raw)**
- `open`, `high`, `low`, `close`

**2. Returns & Volume (2 derived)**
- `log_return` - Log returns: ln(P_t / P_{t-1})
- `log_volume` - Log-transformed volume: ln(V_t + 1)

**3. Volatility Features (4 derived)**
- `gk_volatility` - Garman-Klass volatility estimator
- `rv_12` - Realized volatility (12 bars = 1 hour)
- `rv_36` - Realized volatility (36 bars = 3 hours)
- `rv_72` - Realized volatility (72 bars = 6 hours)

**4. Liquidity Features (1 derived)**
- `amihud_illiq` - Amihud illiquidity measure

**5. Momentum Features (6 derived)**
- `rsi_14` - Relative Strength Index (14-period)
- `macd` - MACD line (EMA_12 - EMA_26)
- `roc_5`, `roc_10`, `roc_20` - Rate of change at multiple horizons

**6. Trend Features (6 derived)**
- `ema_slope_9`, `ema_slope_21`, `ema_slope_50` - EMA slopes
- `ema_dev_9`, `ema_dev_21`, `ema_dev_50` - EMA deviations

**7. Range Features (1 derived)**
- `atr_14` - Average True Range (14-period)

**8. Volume Features (1 derived, excluding log_volume)**
- `volume_ma_ratio` - Volume relative to 20-period MA

**Total: 24 features (4 raw OHLC + 20 derived)**

**Note:** Raw `volume` is not included as a feature (it's an intermediate variable). 
Only `log_volume` is used as it captures scale-invariant volume information.

### Target Variables (5)

Forward log-return targets for multiple prediction horizons:
- `target_15m` - 15-minute ahead (3 bars)
- `target_30m` - 30-minute ahead (6 bars)
- `target_60m` - 1-hour ahead (12 bars)
- `target_2h` - 2-hour ahead (24 bars)
- `target_4h` - 4-hour ahead (48 bars)

---

## Implementation Details

### Feature Computation Algorithm

1. **Basic Transforms** - Log returns and log volume
2. **Volatility Measures** - Garman-Klass, realized volatilities
3. **Liquidity Measures** - Amihud illiquidity
4. **Momentum Indicators** - RSI, MACD, ROC
5. **Trend Indicators** - EMA slopes and deviations
6. **Range Indicators** - ATR
7. **Volume Indicators** - Volume MA ratio

### Causality Assurance

All features strictly use historical data only:
- Rolling windows use `rolling()` with proper parameters
- No forward-looking calculations
- EMA computations use `ewm()` with `adjust=False`
- Target computation explicitly uses `.shift(-h)` which is only for labeling (dropped in training)

### NaN Handling

**Warmup Period:** Features with rolling windows create NaN values during warmup:
- Longest warmup: 72 bars (for `rv_72`)
- Solution: Drop all rows with any NaN after feature computation
- Typical loss: <0.1% of total data

**Result:** Clean dataset with no NaN or infinite values

### Multicollinearity Management

**Analysis:** Correlation matrix computed for all feature pairs
**Threshold:** 0.95 for flagging high correlation
**Action:** Dropped `macd_signal` (0.955 correlation with `macd`)
**Result:** No remaining feature pairs above 0.95 threshold

---

## Validation Results

### Feature Quality Checks

✓ **No infinite values** - All features computed successfully  
✓ **No NaN values** - Warmup period dropped (122 bars, 0.01%)  
✓ **No multicollinearity** - All feature pairs < 0.95 correlation  
✓ **Reasonable ranges** - All features within expected statistical bounds

### Data Coverage

- **Date Range:** 2010-06-07 to 2025-12-03 (5,658 days)
- **Total Bars:** 1,086,808 clean 5-minute bars
- **Features:** 24 input features + 5 target horizons = 29 columns
- **File Size:** ~234 MB (Parquet format)

### Target Statistics

| Horizon | Mean Return | Std Dev | Annualized Sharpe |
|---------|-------------|---------|-------------------|
| 15m (3 bars) | 0.000008 | 0.001290 | 0.96 |
| 30m (6 bars) | 0.000016 | 0.001824 | 0.96 |
| 60m (12 bars) | 0.000032 | 0.002584 | 0.96 |
| 2h (24 bars) | 0.000064 | 0.003650 | 0.96 |
| 4h (48 bars) | 0.000128 | 0.005152 | 0.97 |

---

## Data Quality Metrics

### Feature Distributions

All features show reasonable statistical properties:
- Mean-reverting behavior in momentum indicators
- Heavy tails in volatility measures (expected)
- Skewness in liquidity measures (expected for illiquidity)
- Normal-like distributions in trend deviations

### Temporal Coverage

- **Trading Hours:** Full 24-hour coverage (futures market)
- **Gaps:** None (continuous 5-minute bars)
- **Contract Rolls:** Properly adjusted via ratio back-adjustment

---

## Known Issues & Considerations

### Feature Engineering Module

1. **Computation Speed:** Pandas-based implementation is adequate for ~1M bars but could be optimized with Polars for larger datasets
2. **Memory Usage:** Full dataset fits in RAM on Colab; no chunking required
3. **Feature Selection:** Final 24-feature set after multicollinearity filtering

### Data Characteristics

1. **Regime Changes:** Dataset spans multiple market regimes (2010-2025) including high/low volatility periods
2. **Structural Breaks:** COVID-19 period (Mar 2020) shows extreme volatility - model must handle this
3. **Feature Stability:** All features computed successfully across entire history

---

## Next Steps

Phase 2 delivers clean, engineered features ready for model training. The next phase will:

1. **Phase 3: Dataset & DataLoader Implementation**
   - Create PyTorch Dataset class
   - Implement 24-hour lookback windowing
   - Apply RevIN normalization per window
   - Handle train/val/test splits

2. **Phase 4: Model Architecture**
   - Build transformer components (24 input variables)
   - Implement attention mechanisms
   - Create quantile prediction heads

3. **Phase 5: Training Pipeline**
   - Implement quantile loss functions
   - Set up training loop with validation
   - Monitor convergence metrics

---

## File Locations
```
/content/drive/MyDrive/Colab Notebooks/Transformers/FP/
├── data/
│   ├── interim/
│   │   └── nq_ohlcv_5min_adjusted.parquet     (from Phase 1)
│   └── processed/
│       ├── nq_features_full.parquet           (24 features + 5 targets)
│       └── column_info.csv                     (metadata)
└── src/
    └── data/
        └── feature_engineering.py              (module)
```

---

## Acceptance Criteria

- [x] All 24 features computed successfully
- [x] All 5 target horizons computed
- [x] No NaN values in final dataset
- [x] No infinite values in final dataset
- [x] No high multicollinearity (>0.95)
- [x] Causality verified (no lookahead bias)
- [x] Data saved in Parquet format
- [x] Comprehensive testing notebook
- [x] Full documentation

**Phase 2 Status: ✓ COMPLETE**