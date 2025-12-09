# MIGT-TVDT Evaluation Report

**Date:** 2025-12-09 04:38:40

## Training Info
- Epochs: 2
- Val loss: 0.014659
- Subsample: 10%

## Architecture
- Variables: 24
- Max seq len: 288
- Horizons: ['15m', '30m', '60m', '2h', '4h']
- Quantiles: 7
- d_model: 256
- Parameters: 6,866,984

## Test Set
- Samples: 13,599

---

# Model Evaluation Report

Samples evaluated: 13,599


## Distributional Metrics

| Horizon | CRPS | PICP-80 | PICP-50 | MPIW-80 | MPIW-50 |
|---------|------|---------|---------|---------|---------|
| 15m | 0.03331 | 0.000 | 0.000 | 0.00689 | 0.00314 |
| 30m | 0.00144 | 0.169 | 0.026 | 0.00772 | 0.00315 |
| 60m | 0.00869 | 0.001 | 0.000 | 0.00755 | 0.00346 |
| 2h | 0.01238 | 0.003 | 0.001 | 0.00652 | 0.00294 |
| 4h | 0.00431 | 0.030 | 0.017 | 0.00487 | 0.00254 |

## Point Metrics (Median)

| Horizon | IC | DA | RMSE | MAE |
|---------|----|----|------|-----|
| 15m | 0.0011 | 0.517 | 0.07021 | 0.07016 |
| 30m | -0.0060 | 0.518 | 0.00488 | 0.00447 |
| 60m | -0.0014 | 0.529 | 0.02135 | 0.02116 |
| 2h | -0.0226 | 0.536 | 0.02702 | 0.02668 |
| 4h | -0.0260 | 0.545 | 0.01087 | 0.00989 |

## Calibration Summary

- Mean calibration error: 0.4721
- Max calibration error: 0.8539
## Backtest Results

           sharpe   sortino  max_drawdown  profit_factor  hit_rate    calmar  total_return  n_trades  mean_return  std_return
horizon                                                                                                                      
15m      2.088242  2.966822      0.069485       1.051661  0.517023  5.801443      0.263772     13599     0.000018    0.001204
30m      2.419255  3.374956      0.155986       1.058892  0.518494  4.578941      0.451304     13599     0.000029    0.001668
60m      1.438116  1.950728      0.293658       1.034214  0.528862  1.812895      0.343300     13599     0.000025    0.002396
2h       1.942117  2.639945      0.504079       1.045922  0.536363  2.459614      0.745920     13599     0.000047    0.003368
4h       3.311340  4.523842      0.706536       1.076169  0.544746  9.106573      3.000511     13599     0.000114    0.004806
