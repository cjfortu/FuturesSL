"""
Evaluation module for distributional forecasting.

Provides metrics, calibration analysis, and backtesting for
assessing quantile prediction models per scientific document Section 7.

Modules:
    metrics: Distributional, point, and financial metrics
    calibration: Calibration analysis and visualization
    backtest: Simple trading strategy backtest
    inference: Model prediction utilities
"""

from .metrics import (
    DistributionalMetrics,
    PointMetrics,
    FinancialMetrics,
    MetricsSummary
)

from .calibration import (
    CalibrationAnalyzer,
    CalibrationByHorizon
)

from .backtest import (
    SignalGenerator,
    SimpleBacktester,
    MultiHorizonBacktester
)

from .inference import (
    ModelPredictor,
    run_evaluation,
    format_evaluation_report
)


__all__ = [
    # Metrics
    'DistributionalMetrics',
    'PointMetrics',
    'FinancialMetrics',
    'MetricsSummary',
    # Calibration
    'CalibrationAnalyzer',
    'CalibrationByHorizon',
    # Backtest
    'SignalGenerator',
    'SimpleBacktester',
    'MultiHorizonBacktester',
    # Inference
    'ModelPredictor',
    'run_evaluation',
    'format_evaluation_report'
]
