"""
Simple backtesting framework for quantile forecast evaluation.

Implements a basic trading strategy that uses median predictions for direction
and interval width for position sizing, per engineering specification Section 6.2.

This is intended for model evaluation, not production trading.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .metrics import FinancialMetrics


class SignalGenerator:
    """
    Generate trading signals from quantile predictions.
    
    Strategy logic:
    - Direction: Long if median > 0, short if median < 0
    - Size: Inversely proportional to prediction interval width
    - Confidence filter: Skip trades when interval is too wide
    """
    
    def __init__(
        self,
        median_idx: int = 3,
        lower_idx: int = 1,
        upper_idx: int = 5,
        max_interval_width: Optional[float] = None,
        min_abs_signal: float = 0.0
    ):
        """
        Initialize signal generator.
        
        Args:
            median_idx: Index of median quantile (q50)
                Type: int
                Default: 3 (for standard 7 quantiles)
            lower_idx: Index of lower bound for interval
                Type: int
                Default: 1 (q10)
            upper_idx: Index of upper bound for interval
                Type: int
                Default: 5 (q90)
            max_interval_width: Maximum interval width for trading
                Type: float or None
                If exceeded, signal = 0 (skip trade)
            min_abs_signal: Minimum absolute median value to trade
                Type: float
                Default: 0.0
        """
        self.median_idx = median_idx
        self.lower_idx = lower_idx
        self.upper_idx = upper_idx
        self.max_interval_width = max_interval_width
        self.min_abs_signal = min_abs_signal
    
    def generate_signals(
        self,
        predictions: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Generate trading signals from predictions.
        
        Signal = sign(median) * (1 / interval_width)
        
        Args:
            predictions: Quantile predictions
                Type: ndarray or Tensor (float)
                Shape: (N, Q)
                
        Returns:
            Trading signals
                Type: ndarray (float)
                Shape: (N,)
                Positive = long, negative = short, 0 = no trade
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        
        median = predictions[:, self.median_idx]
        lower = predictions[:, self.lower_idx]
        upper = predictions[:, self.upper_idx]
        
        # Interval width (80% CI by default)
        width = upper - lower
        
        # Avoid division by zero
        width = np.maximum(width, 1e-8)
        
        # Base signal: direction from median, size from inverse width
        direction = np.sign(median)
        confidence = 1.0 / width
        
        signals = direction * confidence
        
        # Apply filters
        if self.max_interval_width is not None:
            signals = np.where(width > self.max_interval_width, 0.0, signals)
        
        signals = np.where(np.abs(median) < self.min_abs_signal, 0.0, signals)
        
        # Normalize to max absolute value of 1
        max_signal = np.abs(signals).max()
        if max_signal > 0:
            signals = signals / max_signal
        
        return signals


class SimpleBacktester:
    """
    Simple backtesting framework for model evaluation.
    
    Executes a signal-based strategy and computes financial metrics.
    Assumes:
    - Immediate execution at predicted time
    - No transaction costs (gross returns)
    - No market impact
    """
    
    def __init__(
        self,
        predictions: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        targets: Union[np.ndarray, torch.Tensor, pd.Series],
        timestamps: Optional[pd.DatetimeIndex] = None,
        signal_generator: Optional[SignalGenerator] = None
    ):
        """
        Initialize backtester.
        
        Args:
            predictions: Quantile predictions
                Type: ndarray, Tensor, or DataFrame
                Shape: (N, Q)
            targets: Realized returns
                Type: ndarray, Tensor, or Series
                Shape: (N,)
            timestamps: Optional timestamps for each observation
                Type: pd.DatetimeIndex or None
            signal_generator: Signal generator instance
                Type: SignalGenerator or None (uses default)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.values
        
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(targets, pd.Series):
            targets = targets.values
        
        self.predictions = predictions
        self.targets = targets
        self.timestamps = timestamps
        self.signal_generator = signal_generator or SignalGenerator()
        
        # Results (populated by run())
        self.signals = None
        self.returns = None
        self.equity_curve = None
        self.metrics = None
    
    def run(self) -> Dict[str, any]:
        """
        Execute backtest.
        
        Returns:
            Backtest results
                Type: Dict[str, any]
                Keys:
                    signals: Trading signals
                    returns: Strategy returns
                    equity_curve: Cumulative equity
                    metrics: Financial performance metrics
        """
        # Generate signals
        self.signals = self.signal_generator.generate_signals(self.predictions)
        
        # Compute strategy returns
        # Return = signal * realized_return
        self.returns = self.signals * self.targets
        
        # Equity curve (starting at 1.0)
        self.equity_curve = np.cumprod(1 + self.returns)
        
        # Compute metrics
        self.metrics = self._compute_metrics()
        
        return {
            'signals': self.signals,
            'returns': self.returns,
            'equity_curve': self.equity_curve,
            'metrics': self.metrics
        }
    
    def _compute_metrics(self) -> Dict[str, float]:
        """
        Compute financial performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        metrics = {
            'sharpe': FinancialMetrics.sharpe_ratio(self.returns),
            'sortino': FinancialMetrics.sortino_ratio(self.returns),
            'max_drawdown': FinancialMetrics.max_drawdown(self.equity_curve),
            'profit_factor': FinancialMetrics.profit_factor(self.returns),
            'hit_rate': FinancialMetrics.hit_rate(self.returns),
            'calmar': FinancialMetrics.calmar_ratio(self.returns)
        }
        
        # Additional statistics
        metrics['total_return'] = float(self.equity_curve[-1] - 1)
        metrics['n_trades'] = int((self.signals != 0).sum())
        metrics['mean_return'] = float(self.returns.mean())
        metrics['std_return'] = float(self.returns.std())
        
        return metrics
    
    def plot_results(self, figsize: Tuple[int, int] = (14, 10)) -> Figure:
        """
        Plot backtest results.
        
        Args:
            figsize: Figure size
                Type: Tuple[int, int]
                
        Returns:
            Matplotlib figure
        """
        if self.returns is None:
            raise ValueError("Run backtest first with .run()")
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # Equity curve
        ax = axes[0, 0]
        if self.timestamps is not None:
            ax.plot(self.timestamps, self.equity_curve)
        else:
            ax.plot(self.equity_curve)
        ax.set_title('Equity Curve')
        ax.set_ylabel('Equity')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Drawdown
        ax = axes[0, 1]
        running_max = np.maximum.accumulate(self.equity_curve)
        drawdown = (running_max - self.equity_curve) / running_max
        if self.timestamps is not None:
            ax.fill_between(self.timestamps, drawdown, alpha=0.7, color='red')
        else:
            ax.fill_between(range(len(drawdown)), drawdown, alpha=0.7, color='red')
        ax.set_title('Drawdown')
        ax.set_ylabel('Drawdown')
        ax.grid(True, alpha=0.3)
        
        # Return distribution
        ax = axes[1, 0]
        ax.hist(self.returns, bins=50, density=True, alpha=0.7, color='steelblue')
        ax.axvline(x=0, color='red', linestyle='--')
        ax.set_title('Return Distribution')
        ax.set_xlabel('Return')
        ax.set_ylabel('Density')
        
        # Signal distribution
        ax = axes[1, 1]
        ax.hist(self.signals, bins=50, density=True, alpha=0.7, color='green')
        ax.axvline(x=0, color='red', linestyle='--')
        ax.set_title('Signal Distribution')
        ax.set_xlabel('Signal')
        ax.set_ylabel('Density')
        
        # Rolling Sharpe
        ax = axes[2, 0]
        window = min(252 * 78, len(self.returns) // 4)  # ~1 year or 25% of data
        if window > 20:
            rolling_mean = pd.Series(self.returns).rolling(window).mean()
            rolling_std = pd.Series(self.returns).rolling(window).std()
            rolling_sharpe = np.sqrt(252 * 78) * rolling_mean / rolling_std
            ax.plot(rolling_sharpe.values)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'Rolling Sharpe ({window} periods)')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        
        # Metrics summary
        ax = axes[2, 1]
        ax.axis('off')
        metrics_text = '\n'.join([
            f"Sharpe Ratio: {self.metrics['sharpe']:.3f}",
            f"Sortino Ratio: {self.metrics['sortino']:.3f}",
            f"Max Drawdown: {self.metrics['max_drawdown']:.2%}",
            f"Profit Factor: {self.metrics['profit_factor']:.2f}",
            f"Hit Rate: {self.metrics['hit_rate']:.2%}",
            f"Total Return: {self.metrics['total_return']:.2%}",
            f"N Trades: {self.metrics['n_trades']:,}"
        ])
        ax.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                verticalalignment='center', transform=ax.transAxes)
        ax.set_title('Performance Metrics')
        
        fig.tight_layout()
        return fig
    
    def get_trade_statistics(self) -> Dict[str, any]:
        """
        Compute detailed trade statistics.
        
        Returns:
            Trade statistics dictionary
        """
        if self.returns is None:
            raise ValueError("Run backtest first with .run()")
        
        # Active trades only
        active_mask = self.signals != 0
        active_returns = self.returns[active_mask]
        active_signals = self.signals[active_mask]
        
        wins = active_returns > 0
        losses = active_returns < 0
        
        stats = {
            'total_trades': int(active_mask.sum()),
            'winning_trades': int(wins.sum()),
            'losing_trades': int(losses.sum()),
            'win_rate': float(wins.mean()) if len(wins) > 0 else 0.0,
            'avg_win': float(active_returns[wins].mean()) if wins.sum() > 0 else 0.0,
            'avg_loss': float(active_returns[losses].mean()) if losses.sum() > 0 else 0.0,
            'largest_win': float(active_returns.max()) if len(active_returns) > 0 else 0.0,
            'largest_loss': float(active_returns.min()) if len(active_returns) > 0 else 0.0,
            'avg_trade': float(active_returns.mean()) if len(active_returns) > 0 else 0.0,
            'std_trade': float(active_returns.std()) if len(active_returns) > 0 else 0.0,
            'long_trades': int((active_signals > 0).sum()),
            'short_trades': int((active_signals < 0).sum())
        }
        
        return stats


class MultiHorizonBacktester:
    """
    Backtest across multiple prediction horizons.
    """
    
    def __init__(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        horizon_names: List[str] = ['15m', '30m', '60m', '2h', '4h'],
        signal_generator: Optional[SignalGenerator] = None
    ):
        """
        Initialize multi-horizon backtester.
        
        Args:
            predictions: Quantile predictions
                Shape: (N, H, Q)
            targets: Realized returns
                Shape: (N, H)
            horizon_names: Names for each horizon
                Type: List[str]
            signal_generator: Signal generator instance
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        self.predictions = predictions
        self.targets = targets
        self.horizon_names = horizon_names
        self.signal_generator = signal_generator or SignalGenerator()
        
        self.results = {}
    
    def run(self) -> Dict[str, Dict[str, any]]:
        """
        Run backtest for all horizons.
        
        Returns:
            Results per horizon
                Type: Dict[str, Dict[str, any]]
        """
        n_horizons = self.predictions.shape[1]
        
        for h in range(n_horizons):
            name = self.horizon_names[h] if h < len(self.horizon_names) else f'h{h}'
            
            bt = SimpleBacktester(
                predictions=self.predictions[:, h, :],
                targets=self.targets[:, h],
                signal_generator=self.signal_generator
            )
            
            self.results[name] = bt.run()
            self.results[name]['trade_stats'] = bt.get_trade_statistics()
        
        return self.results
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """
        Get summary of metrics across all horizons.
        
        Returns:
            DataFrame with metrics per horizon
        """
        if not self.results:
            raise ValueError("Run backtest first with .run()")
        
        rows = []
        for horizon, result in self.results.items():
            row = {'horizon': horizon}
            row.update(result['metrics'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.set_index('horizon')
        
        return df
    
    def plot_equity_curves(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Axes:
        """
        Plot equity curves for all horizons.
        
        Args:
            figsize: Figure size
                
        Returns:
            Matplotlib axes
        """
        if not self.results:
            raise ValueError("Run backtest first with .run()")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for horizon, result in self.results.items():
            ax.plot(result['equity_curve'], label=horizon)
        
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Equity Curves by Horizon')
        ax.set_xlabel('Period')
        ax.set_ylabel('Equity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_metrics_comparison(self, figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Plot comparison of metrics across horizons.
        
        Args:
            figsize: Figure size
                
        Returns:
            Matplotlib figure
        """
        if not self.results:
            raise ValueError("Run backtest first with .run()")
        
        metrics_df = self.get_metrics_summary()
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        metrics_to_plot = ['sharpe', 'sortino', 'max_drawdown', 'profit_factor', 'hit_rate', 'total_return']
        titles = ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Profit Factor', 'Hit Rate', 'Total Return']
        
        for ax, metric, title in zip(axes.flat, metrics_to_plot, titles):
            values = metrics_df[metric].values
            horizons = metrics_df.index.tolist()
            
            colors = ['green' if v > 0 else 'red' for v in values]
            if metric in ['max_drawdown']:
                colors = ['red' for _ in values]
            
            ax.bar(horizons, values, color=colors, alpha=0.7)
            ax.set_title(title)
            ax.set_ylabel(metric)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        return fig
