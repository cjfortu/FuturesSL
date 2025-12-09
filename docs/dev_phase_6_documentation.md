# Dev Phase 6: Evaluation & Analysis - Documentation

## Overview
Implementation of comprehensive evaluation framework for MIGT-TVDT distributional forecasting model per scientific document Section 7 and engineering specification Section 6.

### Updates and Changes

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
   - `run_evaluation()`: Complete evaluation pipeline
   - `format_evaluation_report()`: Markdown report generation

### Notebooks

- **`notebooks/06_evaluation.ipynb`**: Comprehensive unit and integration tests
- **`notebooks/06_production_evaluation.ipynb`**: Production evaluation on trained model

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
- Fraction of correct sign predictions
- Random baseline: 50%
- Critical for trading strategy profitability

### Financial Metrics

**Sharpe Ratio**
- `sqrt(periods_per_year) * mean(returns) / std(returns)`
- Annualization: 252 days * 78 bars/day for 5-min data
- Target: > 1.0 (gross, pre-costs)

**Sortino Ratio**
- Like Sharpe but uses downside deviation only
- Better reflects investor preferences

**Maximum Drawdown**
- Largest peak-to-trough decline in equity
- Risk measure for worst-case scenario

### Calibration Analysis

**Reliability Diagram**
- Plot expected vs observed coverage for each quantile
- Perfect calibration: points on diagonal
- Above diagonal: underconfident (intervals too wide)
- Below diagonal: overconfident (intervals too narrow)

**PIT Histogram**
- Probability Integral Transform values should be uniform
- Flat histogram = well-calibrated
- U-shaped: underconfident
- Inverted-U: overconfident

### Backtesting Strategy

**Signal Generation**
- Direction: sign(median prediction)
- Size: 1 / interval_width (inverse of uncertainty)
- Filters: Max interval width, minimum signal threshold

**Strategy Returns**
- Return = signal * realized_return
- No transaction costs (gross returns)
- No market impact assumed

## API Examples

### Basic Metrics
```python
from evaluation.metrics import DistributionalMetrics, PointMetrics

# CRPS
crps = DistributionalMetrics.crps_quantile(predictions, targets, quantiles)

# IC
median_preds = predictions[:, 3]  # q50
ic = PointMetrics.information_coefficient(median_preds, targets)
```

### Calibration Analysis
```python
from evaluation.calibration import CalibrationAnalyzer

analyzer = CalibrationAnalyzer(quantiles)
fig, summary = analyzer.create_calibration_report(predictions, targets)
```

### Backtesting
```python
from evaluation.backtest import MultiHorizonBacktester

bt = MultiHorizonBacktester(predictions, targets, horizon_names)
results = bt.run()
summary_df = bt.get_metrics_summary()
```

### Full Evaluation Pipeline
```python
from evaluation.inference import run_evaluation, format_evaluation_report

results = run_evaluation(model, dataloader, quantiles, horizon_names)
report = format_evaluation_report(results)
```

## Config Handling Best Practices

When accessing checkpoint configurations in notebooks:

```python
# GOOD: Defensive extraction handles None and missing keys
data_config = checkpoint['config'].get('data', {})
subsample_frac = data_config.get('subsample_fraction') or 1.0

# BAD: Fails when key exists with None value
subsample_frac = checkpoint['config']['data'].get('subsample_fraction', 1.0) * 100  # TypeError if None
```

**Rationale**: YAML `null` values become Python `None`. The `dict.get(key, default)` method only returns the default when the key is **missing**, not when it exists with value `None`. Using `or 1.0` provides a fallback for both cases.

## Metric Interpretation Guide

### Distributional Quality
| Metric | Good | Poor | Notes |
|--------|------|------|-------|
| CRPS | < 0.001 | > 0.005 | Scale-dependent |
| Calibration Error | < 0.02 | > 0.05 | Absolute coverage deviation |
| PICP-80 | 0.78-0.82 | < 0.70 or > 0.90 | Target: 80% |

### Point Prediction Quality
| Metric | Good | Poor | Notes |
|--------|------|------|-------|
| IC | > 0.05 | < 0.02 | Rank correlation |
| DA | > 0.52 | < 0.50 | Direction accuracy |

### Trading Performance
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
- torch
- pandas
- scipy (for Spearman correlation)
- matplotlib (for plotting)

## Known Limitations
1. CRPS approximation via quantile pinball loss (not true integral)
2. Backtest assumes immediate execution (no slippage)
3. No transaction costs in financial metrics
4. Single-asset focus (no portfolio metrics)

## Next Steps
Proceed to production deployment and continuous monitoring of model performance.