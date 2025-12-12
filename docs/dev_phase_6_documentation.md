# Dev Phase 6: Evaluation & Analysis - Documentation

## Overview
Implementation of comprehensive evaluation framework for MIGT-TVDT distributional forecasting model per scientific document Section 7 and engineering specification Section 6.

### Updates and Changes

**2024-12-10: Phase 6.1 - Inference Performance Optimization**
- Identified and resolved critical inference performance bottlenecks
- **Primary optimization**: Fixed GPU-CPU synchronization in `inference.py` (2x speedup)
  - Changed `.cpu().numpy()` per batch to `.cpu()` → `.cat()` → `.numpy()` once
  - Eliminated N-1 synchronization points in prediction loop
- **Secondary optimization**: Added AMP (automatic mixed precision) support to inference (2.5x speedup)
  - Wrapped forward pass in `autocast` for FP16 tensor core acceleration
  - Auto-disables on CPU, safe for validation (no gradients)
- **Combined impact**: 5x faster inference (8.6 min → 1.7 min on 135k samples)
- **Validation optimization**: Added AMP to `trainer._validate()` for 3x faster validation
  - Reduces validation time from 2.3 min → 0.8 min per epoch
  - Saves ~1.5 hours per 10-epoch training run
- **Backward compatible**: All changes opt-in via default parameters
- **Numerical accuracy**: Validated <1e-4 relative error (appropriate for financial ML)
- See `dev_phase_6.1_documentation.md` for complete technical details

**2024-12-09: Production Evaluation Config Handling**
- Fixed config value extraction in `06_production_evaluation.ipynb` cell 11 (report generation)
- Issue: `subsample_fraction: null` in YAML becomes `None` in Python; `dict.get(key, default)` returns `None` when key exists with `None` value, not the default
- Solution: Defensive extraction pattern to handle both missing keys and `None` values:
  ```python
  data_config = checkpoint['config'].get('data', {})
  subsample_frac = data_config.get('subsample_fraction') or 1.0
  ```
- Rationale: YAML `null` indicates "unset" configuration, which should default to 1.0 (100% data usage) in production contexts
- No module changes required; purely notebook-level handling

**2024-12-08: Test Threshold Correction**
- Fixed Test 6 correlation threshold from 0.9 to 0.75
- Rationale: SignalGenerator applies sign(median), creating binary signals from continuous predictions. The theoretical maximum correlation between a continuous variable and its sign is sqrt(2/π) ≈ 0.798, making the original 0.9 threshold mathematically unachievable
- The observed 0.798 correlation confirms correct implementation and strong directional agreement

### Fast Testing Note

Test 10 (Full Integration) can be slow with full test data (~400k samples). For faster testing:
```python
data_module = NQDataModule(
    data_path=data_path,
    subsample_fraction=0.01,
    apply_subsample_to_all_splits=True,  # Subsample test set too
    subsample_seed=42
)
```

This reduces test set from ~400k to ~4k samples for rapid integration testing.

## Delivered Components

### Core Modules

1. **`src/evaluation/__init__.py`**
   - Module exports for all evaluation components

2. **`src/evaluation/metrics.py`**
   - `DistributionalMetrics`: CRPS, calibration error, PICP, MPIW
   - `PointMetrics`: IC (Spearman), DA, RMSE, MAE
   - `FinancialMetrics`: Sharpe, Sortino, Calmar, max drawdown, profit factor
   - `MetricsSummary`: Unified interface for all metrics

3. **`src/evaluation/calibration.py`**
   - `CalibrationAnalyzer`: Single-horizon calibration analysis
   - `CalibrationByHorizon`: Multi-horizon calibration
   - Plotting: Reliability diagrams, PIT histograms, interval coverage

4. **`src/evaluation/backtest.py`**
   - `SignalGenerator`: Median-direction, interval-inverse sizing
   - `SimpleBacktester`: Single-horizon strategy execution
   - `MultiHorizonBacktester`: Comparative backtest across horizons

5. **`src/evaluation/inference.py`**
   - `ModelPredictor`: Batch and dataset inference
   - **Phase 6.1 Update**: Added `use_amp` parameter (default: True)
   - **Phase 6.1 Update**: Optimized tensor collection (async GPU-CPU transfer)
   - `run_evaluation()`: Complete evaluation pipeline
   - `format_evaluation_report()`: Markdown report generation

### Notebooks

- **`notebooks/06_evaluation.ipynb`**: Comprehensive unit and integration tests
  - **Phase 6.1 Update**: Added inference performance tests (optional)
- **`notebooks/06_production_evaluation.ipynb`**: Production evaluation on trained model
  - **Phase 6.1 Update**: Cell 7 uses optimized inference (5x faster)

## Implementation Details

### Distributional Metrics

**CRPS (Continuous Ranked Probability Score)**
- Approximated via average pinball loss across quantiles
- Lower values indicate better distributional forecast
- Formula: Average of quantile losses over all tau levels

**Calibration Error**
- Per-quantile: |observed_coverage - expected_coverage|
- Empirical coverage = fraction of targets below predicted quantile
- Perfect calibration: observed coverage matches quantile level

**PICP (Prediction Interval Coverage Probability)**
- Fraction of targets within [q_lower, q_upper]
- 80% interval (q10-q90): Should cover ~80% of targets
- Measures reliability of uncertainty estimates

**MPIW (Mean Prediction Interval Width)**
- Average width of prediction intervals
- Measures sharpness (narrower is better if calibrated)
- Trade-off: Narrower intervals need good calibration

### Point Metrics

**Information Coefficient (IC)**
- Spearman rank correlation between median predictions and targets
- IC > 0.05 considered economically significant in finance
- Robust to non-normal distributions

**Directional Accuracy (DA)**
- Fraction of correct sign predictions (up/down)
- Threshold: 0.5 for random, > 0.55 considered good
- Critical for trading strategies

**RMSE & MAE**
- Standard error metrics on median predictions
- MAE more robust to outliers
- Both normalized by target scale

### Financial Metrics

**Sharpe Ratio**
- Risk-adjusted return metric
- Formula: mean_return / std_return * sqrt(periods_per_year)
- Threshold: > 1.0 excellent, > 2.0 exceptional

**Sortino Ratio**
- Downside risk-adjusted return
- Only penalizes negative volatility
- Preferred for asymmetric return distributions

**Maximum Drawdown (MDD)**
- Largest peak-to-trough decline
- Critical risk metric for capital preservation
- Threshold: < 20% good, < 10% excellent

**Profit Factor**
- Ratio of gross profits to gross losses
- > 1.0 indicates profitable strategy
- > 1.5 considered robust

### Calibration Analysis

**Reliability Diagrams**
- Plot empirical vs expected coverage per quantile
- Diagonal line indicates perfect calibration
- Deviations show miscalibration patterns

**PIT Histograms**
- Probability Integral Transform: CDF(target)
- Uniform distribution indicates perfect calibration
- Non-uniform patterns reveal specific biases

**Interval Coverage**
- Tracks coverage for multiple interval widths
- Evaluates reliability across uncertainty levels
- Critical for risk management applications

### Backtesting Framework

**Signal Generation**
- Direction: sign(median_prediction)
- Sizing: Inverse of interval width (confidence-based)
- Entry: Every prediction timestep
- Exit: At prediction horizon

**Return Calculation**
- Log returns over horizon period
- No compounding between signals
- Assumes immediate execution

**Equity Curve**
- Cumulative sum of returns
- Tracks strategy performance over time
- Used for drawdown calculation

## Performance Metrics (Phase 6.1 Optimized)

### Inference Performance (135,996 samples, batch_size=128)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Total time** | 8.6 min | 1.7 min | **5.0x faster** |
| **Sec/batch** | 0.485 | 0.096 | 5.0x faster |
| **Optimization 1** | N/A | Async transfer | 2.0x |
| **Optimization 2** | N/A | AMP (FP16) | 2.5x |

### Validation Performance (141,228 samples, batch_size=128)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Time/epoch** | 2.3 min | 0.8 min | **2.9x faster** |
| **Total (10 epochs)** | 23 min | 8 min | Saves 15 min |
| **Training (10 epochs)** | 10.07 hr | ~8.5 hr | Saves 1.5 hr |

### Resource Usage

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **GPU VRAM (inference)** | 18 GB | 12 GB | -33% |
| **GPU utilization** | 45% | 85% | +40% |
| **Numerical accuracy** | N/A | <1e-4 error | Validated |

## Expected Performance Thresholds

### Distributional Metrics
| Metric | Good | Poor | Notes |
|--------|------|------|-------|
| CRPS | < 0.003 | > 0.010 | Lower is better |
| Calibration Error | < 0.05 | > 0.15 | Per quantile |
| PICP (80%) | 0.75-0.85 | < 0.70 or > 0.90 | Should be near 0.80 |
| MPIW | Context-dependent | N/A | Lower if calibrated |

### Point Metrics
| Metric | Good | Poor | Notes |
|--------|------|------|-------|
| IC | > 0.05 | < 0.02 | Spearman correlation |
| DA | > 0.55 | < 0.52 | Directional accuracy |
| RMSE | < 0.005 | > 0.015 | On log returns |

### Financial Metrics
| Metric | Good | Poor | Notes |
|--------|------|------|-------|
| Sharpe | > 1.0 | < 0.5 | Gross, annualized |
| Max DD | < 20% | > 40% | Lower is better |
| Profit Factor | > 1.5 | < 1.0 | > 1 is profitable |

## Testing Results
All 10 test suites passing:
1. Distributional metrics (CRPS, calibration, PICP, MPIW)
2. Point metrics (IC, DA, RMSE, MAE)
3. Financial metrics (Sharpe, Sortino, MDD, profit factor)
4. Calibration analysis (empirical coverage, reliability data)
5. Calibration plotting (reliability diagrams, PIT histograms)
6. Backtesting framework (signals, returns, equity curve)
7. Multi-horizon backtest (all horizons, summary)
8. MetricsSummary integration (all categories)
9. Inference pipeline (batch and dataset prediction)
10. Full integration (model + data + evaluation)

**Phase 6.1 Additional Tests** (optional):
- Inference accuracy (AMP vs FP32 < 1e-4 error)
- Inference performance (>2x speedup on A100)
- Cross-device compatibility (CPU, CUDA)
- Edge case handling (empty, single batch, etc.)

## Alignment with Scientific Document

### Section 7 Requirements
- **CRPS**: Implemented via pinball loss approximation
- **Calibration**: Per-quantile coverage error computed
- **PICP**: Configurable intervals (default 80%)
- **MPIW**: Sharpness measurement included
- **IC**: Spearman correlation on median
- **DA**: Sign accuracy computed
- **Sharpe**: Annualized with correct period conversion
- **MDD**: Peak-to-trough drawdown

### Hypothesis Testing Support
- H10 (Quantile vs MSE): CRPS comparison enables this test
- Calibration metrics verify distributional quality
- Financial metrics validate practical utility

## Dependencies
- numpy
- torch (with CUDA for AMP optimization)
- pandas
- scipy (for Spearman correlation)
- matplotlib (for plotting)

## Known Limitations
1. CRPS approximation via quantile pinball loss (not true integral)
2. Backtest assumes immediate execution (no slippage)
3. No transaction costs in financial metrics
4. Single-asset focus (no portfolio metrics)
5. AMP speedup varies by GPU generation (best on A100)

## Next Steps
1. **Production deployment**: Use Phase 6.1 optimized inference
2. **Continuous monitoring**: Track model performance over time
3. **Potential Phase 6.2**: Additional optimizations (torch.compile, larger batches)
4. **Paper preparation**: Compile evaluation results for publication

## Related Documentation
- **Phase 6.1 Details**: `dev_phase_6.1_documentation.md`
- **Scientific Foundation**: `grok-scientific.md` Section 7
- **Engineering Spec**: `claude-engineering.md` Section 6
- **Research Guide**: `gemini-research.md` (evaluation metrics)