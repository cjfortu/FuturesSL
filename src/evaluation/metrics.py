"""
Evaluation metrics for distributional forecasting.

Implements metrics spanning AI/ML (CRPS, calibration) and quantitative finance
(IC, Sharpe ratio) perspectives per scientific document Section 7.

Metrics categories:
1. Distributional: CRPS, calibration error, PICP, MPIW
2. Point: IC (Information Coefficient), DA (Directional Accuracy), RMSE
3. Financial: Sharpe ratio, maximum drawdown, profit factor
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats


class DistributionalMetrics:
    """
    Metrics for evaluating distributional (quantile) predictions.
    
    Quantile predictions enable uncertainty quantification. These metrics
    assess both accuracy (CRPS) and calibration (whether predicted quantiles
    match empirical frequencies).
    """
    
    @staticmethod
    def crps_quantile(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        quantiles: List[float]
    ) -> float:
        """
        Continuous Ranked Probability Score approximated via quantile predictions.
        
        CRPS measures the integrated squared difference between the predicted
        cumulative distribution and the empirical CDF (step function at observation).
        With quantile predictions, CRPS is approximated as the average pinball loss
        across all quantiles.
        
        Lower CRPS indicates better distributional forecast.
        
        Args:
            predictions: Quantile predictions
                Type: ndarray or Tensor (float)
                Shape: (N, Q) where N=samples, Q=quantiles
            targets: Observed values
                Type: ndarray or Tensor (float)
                Shape: (N,)
            quantiles: Quantile levels
                Type: List[float]
                Example: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
                
        Returns:
            CRPS value (lower is better)
                Type: float
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Ensure proper shapes
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        else:
            targets = targets.reshape(-1, 1)
        
        quantiles = np.array(quantiles).reshape(1, -1)
        
        # Pinball loss: tau * max(y - q, 0) + (1 - tau) * max(q - y, 0)
        errors = targets - predictions  # (N, Q)
        
        pinball = np.where(
            errors >= 0,
            quantiles * errors,
            (quantiles - 1) * errors
        )
        
        # Average across quantiles and samples
        return float(pinball.mean())
    
    @staticmethod
    def crps_per_horizon(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        quantiles: List[float],
        horizon_names: List[str] = ['15m', '30m', '60m', '2h', '4h']
    ) -> Dict[str, float]:
        """
        CRPS computed separately for each prediction horizon.
        
        Args:
            predictions: Quantile predictions
                Type: ndarray or Tensor (float)
                Shape: (N, H, Q) where H=horizons, Q=quantiles
            targets: Observed values
                Type: ndarray or Tensor (float)
                Shape: (N, H)
            quantiles: Quantile levels
                Type: List[float]
            horizon_names: Names for each horizon
                Type: List[str]
                
        Returns:
            Dictionary mapping horizon names to CRPS values
                Type: Dict[str, float]
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        results = {}
        n_horizons = predictions.shape[1]
        
        for h in range(n_horizons):
            name = horizon_names[h] if h < len(horizon_names) else f'h{h}'
            results[name] = DistributionalMetrics.crps_quantile(
                predictions[:, h, :],
                targets[:, h],
                quantiles
            )
        
        return results
    
    @staticmethod
    def calibration_error(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        quantiles: List[float]
    ) -> Dict[str, float]:
        """
        Per-quantile calibration error.
        
        For a well-calibrated model, tau fraction of observations should
        fall below the tau-th quantile. Calibration error is the absolute
        difference between actual and expected coverage.
        
        Args:
            predictions: Quantile predictions
                Type: ndarray or Tensor (float)
                Shape: (N, Q)
            targets: Observed values
                Type: ndarray or Tensor (float)
                Shape: (N,)
            quantiles: Quantile levels
                Type: List[float]
                
        Returns:
            Dictionary with 'errors' (per-quantile), 'mean', 'max' calibration errors
                Type: Dict[str, float]
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        targets = targets.reshape(-1, 1)
        
        # Empirical coverage: fraction of targets <= predicted quantile
        below = (targets <= predictions).astype(float)
        empirical_coverage = below.mean(axis=0)  # (Q,)
        
        # Expected coverage is the quantile level itself
        expected_coverage = np.array(quantiles)
        
        # Calibration error per quantile
        errors = np.abs(empirical_coverage - expected_coverage)
        
        result = {
            f'q{int(q*100):02d}': float(errors[i])
            for i, q in enumerate(quantiles)
        }
        result['mean'] = float(errors.mean())
        result['max'] = float(errors.max())
        result['empirical_coverage'] = {
            f'q{int(q*100):02d}': float(empirical_coverage[i])
            for i, q in enumerate(quantiles)
        }
        
        return result
    
    @staticmethod
    def prediction_interval_coverage_probability(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        lower_idx: int = 1,
        upper_idx: int = 5
    ) -> float:
        """
        Prediction Interval Coverage Probability (PICP).
        
        Measures what fraction of observations fall within the prediction
        interval [q_lower, q_upper]. For indices 1 and 5 with standard
        quantiles [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], this gives
        the 80% interval (q10 to q90).
        
        Args:
            predictions: Quantile predictions
                Type: ndarray or Tensor (float)
                Shape: (N, Q)
            targets: Observed values
                Type: ndarray or Tensor (float)
                Shape: (N,)
            lower_idx: Index of lower quantile in predictions
                Type: int
                Default: 1 (q10 with standard quantiles)
            upper_idx: Index of upper quantile in predictions
                Type: int
                Default: 5 (q90 with standard quantiles)
                
        Returns:
            Coverage probability (ideally should match expected interval width)
                Type: float
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        lower = predictions[:, lower_idx]
        upper = predictions[:, upper_idx]
        
        in_interval = (targets >= lower) & (targets <= upper)
        
        return float(in_interval.mean())
    
    @staticmethod
    def mean_prediction_interval_width(
        predictions: Union[np.ndarray, torch.Tensor],
        lower_idx: int = 1,
        upper_idx: int = 5
    ) -> float:
        """
        Mean Prediction Interval Width (MPIW).
        
        Measures average width of prediction intervals. Narrower intervals
        are better (sharper forecasts), but only if calibration is maintained.
        MPIW should be evaluated alongside PICP.
        
        Args:
            predictions: Quantile predictions
                Type: ndarray or Tensor (float)
                Shape: (N, Q)
            lower_idx: Index of lower quantile
                Type: int
            upper_idx: Index of upper quantile
                Type: int
                
        Returns:
            Mean interval width
                Type: float
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        
        widths = predictions[:, upper_idx] - predictions[:, lower_idx]
        
        return float(widths.mean())


class PointMetrics:
    """
    Point forecast metrics using median (0.5 quantile) as point prediction.
    
    While distributional forecasting provides uncertainty, many applications
    require a single point estimate. The median (q50) is the natural choice
    as it minimizes expected absolute error.
    """
    
    @staticmethod
    def information_coefficient(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Information Coefficient (IC): Spearman rank correlation.
        
        Measures how well predictions rank outcomes. IC captures monotonic
        relationships without assuming linearity, making it robust to
        non-normal distributions common in financial returns.
        
        IC > 0.05 is considered economically significant in finance.
        
        Args:
            predictions: Point predictions (typically median)
                Type: ndarray or Tensor (float)
                Shape: (N,)
            targets: Observed values
                Type: ndarray or Tensor (float)
                Shape: (N,)
                
        Returns:
            Spearman correlation coefficient
                Type: float
                Range: [-1, 1]
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Handle NaN values
        mask = ~(np.isnan(predictions) | np.isnan(targets))
        if mask.sum() < 2:
            return np.nan
        
        corr, _ = stats.spearmanr(predictions[mask], targets[mask])
        
        return float(corr)
    
    @staticmethod
    def directional_accuracy(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Directional Accuracy (DA): Fraction of correct sign predictions.
        
        In trading, predicting direction correctly is often more important
        than magnitude. DA measures how often the model correctly predicts
        whether returns will be positive or negative.
        
        Random baseline: 0.5. Values > 0.52 are typically significant.
        
        Args:
            predictions: Point predictions
                Type: ndarray or Tensor (float)
                Shape: (N,)
            targets: Observed values
                Type: ndarray or Tensor (float)
                Shape: (N,)
                
        Returns:
            Fraction of correct sign predictions
                Type: float
                Range: [0, 1]
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        pred_sign = np.sign(predictions)
        target_sign = np.sign(targets)
        
        # Handle zeros (treat as correct if both are zero or same sign)
        correct = pred_sign == target_sign
        
        return float(correct.mean())
    
    @staticmethod
    def rmse(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Root Mean Squared Error.
        
        Standard regression metric. While not ideal for heavy-tailed
        financial returns, RMSE provides comparability with other studies.
        
        Args:
            predictions: Point predictions
                Type: ndarray or Tensor (float)
                Shape: (N,)
            targets: Observed values
                Type: ndarray or Tensor (float)
                Shape: (N,)
                
        Returns:
            RMSE value (same units as targets)
                Type: float
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        return float(np.sqrt(np.mean((predictions - targets) ** 2)))
    
    @staticmethod
    def mae(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Mean Absolute Error.
        
        More robust to outliers than RMSE. The median minimizes MAE,
        so this is the natural loss for median predictions.
        
        Args:
            predictions: Point predictions
                Type: ndarray or Tensor (float)
                Shape: (N,)
            targets: Observed values
                Type: ndarray or Tensor (float)
                Shape: (N,)
                
        Returns:
            MAE value
                Type: float
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        return float(np.mean(np.abs(predictions - targets)))
    
    @staticmethod
    def metrics_per_horizon(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        horizon_names: List[str] = ['15m', '30m', '60m', '2h', '4h']
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute all point metrics for each horizon.
        
        Args:
            predictions: Median predictions
                Type: ndarray or Tensor (float)
                Shape: (N, H)
            targets: Observed values
                Type: ndarray or Tensor (float)
                Shape: (N, H)
            horizon_names: Names for each horizon
                Type: List[str]
                
        Returns:
            Nested dictionary: {horizon: {metric: value}}
                Type: Dict[str, Dict[str, float]]
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        results = {}
        n_horizons = predictions.shape[1]
        
        for h in range(n_horizons):
            name = horizon_names[h] if h < len(horizon_names) else f'h{h}'
            pred_h = predictions[:, h]
            targ_h = targets[:, h]
            
            results[name] = {
                'ic': PointMetrics.information_coefficient(pred_h, targ_h),
                'da': PointMetrics.directional_accuracy(pred_h, targ_h),
                'rmse': PointMetrics.rmse(pred_h, targ_h),
                'mae': PointMetrics.mae(pred_h, targ_h)
            }
        
        return results


class FinancialMetrics:
    """
    Trading-oriented performance metrics.
    
    These metrics evaluate model performance in the context of actual
    trading strategies, bridging the gap between statistical and economic
    significance.
    """
    
    @staticmethod
    def sharpe_ratio(
        returns: Union[np.ndarray, torch.Tensor],
        periods_per_year: int = 252 * 78,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Annualized Sharpe Ratio.
        
        Sharpe = sqrt(periods_per_year) * (mean(returns) - rf) / std(returns)
        
        For 5-min bars: ~78 bars per trading day, 252 trading days/year.
        
        Args:
            returns: Strategy returns (not prices)
                Type: ndarray or Tensor (float)
                Shape: (N,)
            periods_per_year: Number of return periods in one year
                Type: int
                Default: 252 * 78 = 19,656 (5-min bars)
            risk_free_rate: Annualized risk-free rate
                Type: float
                Default: 0.0
                
        Returns:
            Annualized Sharpe ratio
                Type: float
        """
        if isinstance(returns, torch.Tensor):
            returns = returns.detach().cpu().numpy()
        
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 2:
            return np.nan
        
        mean_return = returns.mean()
        std_return = returns.std(ddof=1)
        
        if std_return < 1e-10:
            return np.nan
        
        # Convert risk-free rate to per-period
        rf_per_period = risk_free_rate / periods_per_year
        
        sharpe = np.sqrt(periods_per_year) * (mean_return - rf_per_period) / std_return
        
        return float(sharpe)
    
    @staticmethod
    def max_drawdown(equity_curve: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Maximum Drawdown (MDD).
        
        Largest peak-to-trough decline in equity. Measures the worst
        historical loss experienced by the strategy.
        
        Args:
            equity_curve: Cumulative equity values (not returns)
                Type: ndarray or Tensor (float)
                Shape: (N,)
                Note: Should start at initial capital (e.g., 1.0)
                
        Returns:
            Maximum drawdown as a positive fraction
                Type: float
                Range: [0, 1] where 1 = 100% loss
        """
        if isinstance(equity_curve, torch.Tensor):
            equity_curve = equity_curve.detach().cpu().numpy()
        
        # Running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Drawdown at each point
        drawdowns = (running_max - equity_curve) / running_max
        
        return float(np.max(drawdowns))
    
    @staticmethod
    def profit_factor(returns: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Profit Factor: Gross profits / Gross losses.
        
        Ratio of total winning returns to total losing returns.
        Values > 1 indicate profitable strategy, > 1.5 is strong.
        
        Args:
            returns: Strategy returns
                Type: ndarray or Tensor (float)
                Shape: (N,)
                
        Returns:
            Profit factor
                Type: float
                Range: [0, inf]
        """
        if isinstance(returns, torch.Tensor):
            returns = returns.detach().cpu().numpy()
        
        profits = returns[returns > 0].sum()
        losses = np.abs(returns[returns < 0].sum())
        
        if losses < 1e-10:
            return float('inf') if profits > 0 else 1.0
        
        return float(profits / losses)
    
    @staticmethod
    def sortino_ratio(
        returns: Union[np.ndarray, torch.Tensor],
        periods_per_year: int = 252 * 78,
        target_return: float = 0.0
    ) -> float:
        """
        Sortino Ratio: Risk-adjusted return using downside deviation.
        
        Like Sharpe but only penalizes downside volatility, which better
        reflects investor preferences (upside volatility is desirable).
        
        Args:
            returns: Strategy returns
                Type: ndarray or Tensor (float)
                Shape: (N,)
            periods_per_year: Number of return periods in one year
                Type: int
            target_return: Minimum acceptable return (annualized)
                Type: float
                Default: 0.0
                
        Returns:
            Annualized Sortino ratio
                Type: float
        """
        if isinstance(returns, torch.Tensor):
            returns = returns.detach().cpu().numpy()
        
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 2:
            return np.nan
        
        # Downside returns relative to target
        target_per_period = target_return / periods_per_year
        downside_returns = returns - target_per_period
        downside_returns = np.minimum(downside_returns, 0)
        
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_std < 1e-10:
            return np.nan
        
        mean_return = returns.mean()
        sortino = np.sqrt(periods_per_year) * (mean_return - target_per_period) / downside_std
        
        return float(sortino)
    
    @staticmethod
    def calmar_ratio(
        returns: Union[np.ndarray, torch.Tensor],
        periods_per_year: int = 252 * 78
    ) -> float:
        """
        Calmar Ratio: Annualized return / Maximum drawdown.
        
        Measures return relative to worst-case risk.
        
        Args:
            returns: Strategy returns
                Type: ndarray or Tensor (float)
                Shape: (N,)
            periods_per_year: Number of return periods in one year
                Type: int
                
        Returns:
            Calmar ratio
                Type: float
        """
        if isinstance(returns, torch.Tensor):
            returns = returns.detach().cpu().numpy()
        
        # Compute equity curve
        equity = np.cumprod(1 + returns)
        
        # Annualized return
        total_return = equity[-1] / equity[0] - 1
        n_periods = len(returns)
        ann_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        
        # Max drawdown
        mdd = FinancialMetrics.max_drawdown(equity)
        
        if mdd < 1e-10:
            return np.nan
        
        return float(ann_return / mdd)
    
    @staticmethod
    def hit_rate(returns: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Hit Rate: Fraction of profitable trades/periods.
        
        Args:
            returns: Strategy returns
                Type: ndarray or Tensor (float)
                Shape: (N,)
                
        Returns:
            Fraction of positive returns
                Type: float
                Range: [0, 1]
        """
        if isinstance(returns, torch.Tensor):
            returns = returns.detach().cpu().numpy()
        
        return float((returns > 0).mean())


class MetricsSummary:
    """
    Comprehensive metrics summary combining all metric categories.
    
    Provides a unified interface for computing all metrics at once,
    suitable for generating evaluation reports.
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
        horizon_names: List[str] = ['15m', '30m', '60m', '2h', '4h'],
        periods_per_year: int = 252 * 78
    ):
        """
        Initialize metrics summary calculator.
        
        Args:
            quantiles: Quantile levels for distributional metrics
                Type: List[float]
            horizon_names: Names for prediction horizons
                Type: List[str]
            periods_per_year: Trading periods per year for financial metrics
                Type: int
        """
        self.quantiles = quantiles
        self.horizon_names = horizon_names
        self.periods_per_year = periods_per_year
        self.median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    
    def compute_all(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        strategy_returns: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, any]:
        """
        Compute comprehensive metrics summary.
        
        Args:
            predictions: Quantile predictions
                Type: ndarray or Tensor (float)
                Shape: (N, H, Q) or (N, Q) for single horizon
            targets: Observed values
                Type: ndarray or Tensor (float)
                Shape: (N, H) or (N,) for single horizon
            strategy_returns: Optional strategy returns for financial metrics
                Type: ndarray or Tensor (float) or None
                Shape: (N,) or (N, H)
                
        Returns:
            Comprehensive metrics dictionary
                Type: Dict[str, any]
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        results = {}
        
        # Handle single vs multi-horizon
        if predictions.ndim == 2:
            predictions = predictions[:, np.newaxis, :]
            targets = targets[:, np.newaxis]
        
        n_samples, n_horizons, n_quantiles = predictions.shape
        
        # Distributional metrics per horizon
        results['distributional'] = {}
        for h in range(n_horizons):
            name = self.horizon_names[h] if h < len(self.horizon_names) else f'h{h}'
            pred_h = predictions[:, h, :]
            targ_h = targets[:, h]
            
            results['distributional'][name] = {
                'crps': DistributionalMetrics.crps_quantile(pred_h, targ_h, self.quantiles),
                'picp_80': DistributionalMetrics.prediction_interval_coverage_probability(
                    pred_h, targ_h, lower_idx=1, upper_idx=5
                ),
                'picp_50': DistributionalMetrics.prediction_interval_coverage_probability(
                    pred_h, targ_h, lower_idx=2, upper_idx=4
                ),
                'mpiw_80': DistributionalMetrics.mean_prediction_interval_width(
                    pred_h, lower_idx=1, upper_idx=5
                ),
                'mpiw_50': DistributionalMetrics.mean_prediction_interval_width(
                    pred_h, lower_idx=2, upper_idx=4
                )
            }
        
        # Overall calibration
        all_pred = predictions.reshape(-1, n_quantiles)
        all_targ = targets.reshape(-1)
        results['calibration'] = DistributionalMetrics.calibration_error(
            all_pred, all_targ, self.quantiles
        )
        
        # Point metrics per horizon
        median_preds = predictions[:, :, self.median_idx]
        results['point'] = PointMetrics.metrics_per_horizon(
            median_preds, targets, self.horizon_names[:n_horizons]
        )
        
        # Financial metrics (if returns provided)
        if strategy_returns is not None:
            if isinstance(strategy_returns, torch.Tensor):
                strategy_returns = strategy_returns.detach().cpu().numpy()
            
            if strategy_returns.ndim == 1:
                strategy_returns = strategy_returns[:, np.newaxis]
            
            results['financial'] = {}
            for h in range(min(strategy_returns.shape[1], n_horizons)):
                name = self.horizon_names[h] if h < len(self.horizon_names) else f'h{h}'
                ret_h = strategy_returns[:, h]
                
                results['financial'][name] = {
                    'sharpe': FinancialMetrics.sharpe_ratio(ret_h, self.periods_per_year),
                    'sortino': FinancialMetrics.sortino_ratio(ret_h, self.periods_per_year),
                    'max_drawdown': FinancialMetrics.max_drawdown(np.cumprod(1 + ret_h)),
                    'profit_factor': FinancialMetrics.profit_factor(ret_h),
                    'hit_rate': FinancialMetrics.hit_rate(ret_h)
                }
        
        return results
