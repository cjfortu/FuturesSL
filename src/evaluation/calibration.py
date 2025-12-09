"""
Calibration analysis tools for distributional forecasts.

Provides visualization and diagnostic functions to assess whether
predicted quantiles match empirical frequencies per scientific
document Section 7 requirements.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class CalibrationAnalyzer:
    """
    Analyzer for quantile forecast calibration.
    
    A well-calibrated distributional forecast has empirical coverage
    matching predicted quantile levels. This class provides tools to
    assess and visualize calibration.
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    ):
        """
        Initialize calibration analyzer.
        
        Args:
            quantiles: Quantile levels being predicted
                Type: List[float]
        """
        self.quantiles = np.array(quantiles)
        self.n_quantiles = len(quantiles)
    
    def compute_empirical_coverage(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Compute empirical coverage for each quantile.
        
        Empirical coverage = fraction of targets <= predicted quantile.
        For perfect calibration, this equals the quantile level.
        
        Args:
            predictions: Quantile predictions
                Type: ndarray or Tensor (float)
                Shape: (N, Q)
            targets: Observed values
                Type: ndarray or Tensor (float)
                Shape: (N,)
                
        Returns:
            Empirical coverage for each quantile
                Type: ndarray (float)
                Shape: (Q,)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        targets = targets.reshape(-1, 1)
        
        below = (targets <= predictions).astype(float)
        return below.mean(axis=0)
    
    def compute_pit_histogram(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Probability Integral Transform (PIT) histogram.
        
        For a calibrated forecast, the PIT values (CDF evaluated at observations)
        should be uniformly distributed on [0, 1]. The histogram should be flat.
        
        Uses linear interpolation between predicted quantiles to estimate CDF.
        
        Args:
            predictions: Quantile predictions
                Type: ndarray or Tensor (float)
                Shape: (N, Q)
            targets: Observed values
                Type: ndarray or Tensor (float)
                Shape: (N,)
            n_bins: Number of histogram bins
                Type: int
                
        Returns:
            Tuple of (bin_counts, bin_edges)
                bin_counts: Normalized histogram counts
                    Type: ndarray (float)
                    Shape: (n_bins,)
                bin_edges: Bin boundaries
                    Type: ndarray (float)
                    Shape: (n_bins + 1,)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        n_samples = len(targets)
        pit_values = np.zeros(n_samples)
        
        # For each sample, find where target falls in quantile distribution
        for i in range(n_samples):
            pred_i = predictions[i]
            targ_i = targets[i]
            
            if targ_i <= pred_i[0]:
                # Below lowest quantile
                pit_values[i] = self.quantiles[0] * (targ_i / pred_i[0]) if pred_i[0] != 0 else 0
            elif targ_i >= pred_i[-1]:
                # Above highest quantile
                pit_values[i] = self.quantiles[-1] + (1 - self.quantiles[-1]) * 0.5
            else:
                # Interpolate between quantiles
                idx = np.searchsorted(pred_i, targ_i) - 1
                idx = max(0, min(idx, len(pred_i) - 2))
                
                # Linear interpolation
                frac = (targ_i - pred_i[idx]) / (pred_i[idx + 1] - pred_i[idx] + 1e-10)
                pit_values[i] = self.quantiles[idx] + frac * (self.quantiles[idx + 1] - self.quantiles[idx])
        
        # Compute histogram
        counts, edges = np.histogram(pit_values, bins=n_bins, range=(0, 1), density=True)
        
        return counts, edges
    
    def compute_reliability_data(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        Compute data for reliability diagram.
        
        Returns expected coverage (quantile levels) and observed coverage
        (empirical frequencies) for plotting.
        
        Args:
            predictions: Quantile predictions
                Type: ndarray or Tensor (float)
                Shape: (N, Q)
            targets: Observed values
                Type: ndarray or Tensor (float)
                Shape: (N,)
                
        Returns:
            Dictionary with reliability data
                Type: Dict[str, ndarray]
                Keys: 'expected', 'observed', 'errors'
        """
        observed = self.compute_empirical_coverage(predictions, targets)
        
        return {
            'expected': self.quantiles.copy(),
            'observed': observed,
            'errors': np.abs(observed - self.quantiles)
        }
    
    def calibration_error_summary(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute summary calibration error statistics.
        
        Args:
            predictions: Quantile predictions
                Shape: (N, Q)
            targets: Observed values
                Shape: (N,)
                
        Returns:
            Summary statistics
                Type: Dict[str, float]
                Keys: 'mean_error', 'max_error', 'rmse'
        """
        rel_data = self.compute_reliability_data(predictions, targets)
        errors = rel_data['errors']
        
        return {
            'mean_error': float(errors.mean()),
            'max_error': float(errors.max()),
            'rmse': float(np.sqrt((errors ** 2).mean()))
        }
    
    def plot_reliability_diagram(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        title: str = 'Calibration (Reliability) Diagram',
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot reliability diagram showing expected vs observed coverage.
        
        Perfect calibration appears as points on the diagonal.
        Points above the line indicate underconfident predictions.
        Points below indicate overconfident predictions.
        
        Args:
            predictions: Quantile predictions
                Shape: (N, Q)
            targets: Observed values
                Shape: (N,)
            title: Plot title
                Type: str
            ax: Existing axes to plot on (creates new if None)
                Type: plt.Axes or None
                
        Returns:
            Matplotlib axes object
                Type: plt.Axes
        """
        rel_data = self.compute_reliability_data(predictions, targets)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.7)
        
        # Plot actual calibration
        ax.scatter(
            rel_data['expected'],
            rel_data['observed'],
            s=60,
            c='steelblue',
            edgecolors='white',
            linewidths=1.5,
            label='Model'
        )
        ax.plot(rel_data['expected'], rel_data['observed'], 'b-', alpha=0.5)
        
        # Annotate quantiles
        for i, q in enumerate(self.quantiles):
            ax.annotate(
                f'{int(q*100)}%',
                (rel_data['expected'][i], rel_data['observed'][i]),
                textcoords='offset points',
                xytext=(5, 5),
                fontsize=8
            )
        
        ax.set_xlabel('Expected Coverage (Quantile Level)')
        ax.set_ylabel('Observed Coverage')
        ax.set_title(title)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc='upper left')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_pit_histogram(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        n_bins: int = 10,
        title: str = 'PIT Histogram',
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot Probability Integral Transform histogram.
        
        A flat histogram indicates good calibration.
        U-shaped: underconfident (intervals too wide).
        Inverted-U: overconfident (intervals too narrow).
        
        Args:
            predictions: Quantile predictions
                Shape: (N, Q)
            targets: Observed values
                Shape: (N,)
            n_bins: Number of histogram bins
                Type: int
            title: Plot title
                Type: str
            ax: Existing axes
                Type: plt.Axes or None
                
        Returns:
            Matplotlib axes object
        """
        counts, edges = self.compute_pit_histogram(predictions, targets, n_bins)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot histogram bars
        width = edges[1] - edges[0]
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, counts, width=width * 0.9, color='steelblue', edgecolor='white')
        
        # Plot uniform reference line
        ax.axhline(y=1.0, color='r', linestyle='--', label='Uniform (ideal)')
        
        ax.set_xlabel('PIT Value')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        return ax
    
    def plot_interval_coverage(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        intervals: List[Tuple[int, int]] = [(1, 5), (2, 4)],
        interval_names: List[str] = ['80%', '50%'],
        title: str = 'Prediction Interval Coverage',
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot prediction interval coverage comparison.
        
        Shows expected vs actual coverage for multiple prediction intervals.
        
        Args:
            predictions: Quantile predictions
                Shape: (N, Q)
            targets: Observed values
                Shape: (N,)
            intervals: List of (lower_idx, upper_idx) tuples
                Type: List[Tuple[int, int]]
            interval_names: Names for each interval
                Type: List[str]
            title: Plot title
                Type: str
            ax: Existing axes
                Type: plt.Axes or None
                
        Returns:
            Matplotlib axes object
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        expected = []
        observed = []
        
        for (lower_idx, upper_idx), name in zip(intervals, interval_names):
            lower = predictions[:, lower_idx]
            upper = predictions[:, upper_idx]
            
            in_interval = (targets >= lower) & (targets <= upper)
            obs_coverage = in_interval.mean()
            
            exp_coverage = self.quantiles[upper_idx] - self.quantiles[lower_idx]
            
            expected.append(exp_coverage)
            observed.append(obs_coverage)
        
        x = np.arange(len(interval_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, expected, width, label='Expected', color='lightgray', edgecolor='black')
        bars2 = ax.bar(x + width/2, observed, width, label='Observed', color='steelblue', edgecolor='black')
        
        ax.set_ylabel('Coverage')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(interval_names)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars1, expected):
            ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, observed):
            ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)
        
        return ax
    
    def create_calibration_report(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        figsize: Tuple[int, int] = (15, 5)
    ) -> Tuple[Figure, Dict[str, float]]:
        """
        Create comprehensive calibration report with multiple plots.
        
        Args:
            predictions: Quantile predictions
                Shape: (N, Q)
            targets: Observed values
                Shape: (N,)
            figsize: Figure size
                Type: Tuple[int, int]
                
        Returns:
            Tuple of (figure, summary_stats)
                figure: Matplotlib figure with subplots
                summary_stats: Calibration error summary
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Reliability diagram
        self.plot_reliability_diagram(predictions, targets, ax=axes[0])
        
        # PIT histogram
        self.plot_pit_histogram(predictions, targets, ax=axes[1])
        
        # Interval coverage
        self.plot_interval_coverage(predictions, targets, ax=axes[2])
        
        fig.tight_layout()
        
        summary = self.calibration_error_summary(predictions, targets)
        
        return fig, summary


class CalibrationByHorizon:
    """
    Calibration analysis across multiple prediction horizons.
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
        horizon_names: List[str] = ['15m', '30m', '60m', '2h', '4h']
    ):
        """
        Initialize multi-horizon calibration analyzer.
        
        Args:
            quantiles: Quantile levels
                Type: List[float]
            horizon_names: Names for prediction horizons
                Type: List[str]
        """
        self.quantiles = quantiles
        self.horizon_names = horizon_names
        self.analyzer = CalibrationAnalyzer(quantiles)
    
    def compute_per_horizon(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute calibration metrics for each horizon.
        
        Args:
            predictions: Quantile predictions
                Shape: (N, H, Q)
            targets: Observed values
                Shape: (N, H)
                
        Returns:
            Nested dictionary: {horizon: {metric: value}}
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        results = {}
        n_horizons = predictions.shape[1]
        
        for h in range(n_horizons):
            name = self.horizon_names[h] if h < len(self.horizon_names) else f'h{h}'
            pred_h = predictions[:, h, :]
            targ_h = targets[:, h]
            
            results[name] = self.analyzer.calibration_error_summary(pred_h, targ_h)
            
            # Add empirical coverage
            coverage = self.analyzer.compute_empirical_coverage(pred_h, targ_h)
            results[name]['empirical_coverage'] = {
                f'q{int(q*100):02d}': float(coverage[i])
                for i, q in enumerate(self.quantiles)
            }
        
        return results
    
    def plot_reliability_by_horizon(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        figsize: Tuple[int, int] = None
    ) -> Figure:
        """
        Create reliability diagrams for each horizon.
        
        Args:
            predictions: Quantile predictions
                Shape: (N, H, Q)
            targets: Observed values
                Shape: (N, H)
            figsize: Figure size (auto if None)
                Type: Tuple[int, int] or None
                
        Returns:
            Matplotlib figure
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        n_horizons = predictions.shape[1]
        
        if figsize is None:
            figsize = (5 * n_horizons, 4)
        
        fig, axes = plt.subplots(1, n_horizons, figsize=figsize)
        if n_horizons == 1:
            axes = [axes]
        
        for h in range(n_horizons):
            name = self.horizon_names[h] if h < len(self.horizon_names) else f'h{h}'
            self.analyzer.plot_reliability_diagram(
                predictions[:, h, :],
                targets[:, h],
                title=f'Calibration - {name}',
                ax=axes[h]
            )
        
        fig.tight_layout()
        return fig
    
    def plot_calibration_heatmap(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Axes:
        """
        Plot heatmap of calibration errors across horizons and quantiles.
        
        Args:
            predictions: Quantile predictions
                Shape: (N, H, Q)
            targets: Observed values
                Shape: (N, H)
            figsize: Figure size
                Type: Tuple[int, int]
                
        Returns:
            Matplotlib axes
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        n_horizons = predictions.shape[1]
        n_quantiles = len(self.quantiles)
        
        # Compute error matrix
        error_matrix = np.zeros((n_horizons, n_quantiles))
        
        for h in range(n_horizons):
            rel_data = self.analyzer.compute_reliability_data(
                predictions[:, h, :],
                targets[:, h]
            )
            error_matrix[h, :] = rel_data['observed'] - rel_data['expected']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(error_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
        
        # Labels
        horizon_labels = self.horizon_names[:n_horizons]
        quantile_labels = [f'{int(q*100)}%' for q in self.quantiles]
        
        ax.set_xticks(range(n_quantiles))
        ax.set_xticklabels(quantile_labels)
        ax.set_yticks(range(n_horizons))
        ax.set_yticklabels(horizon_labels)
        
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Horizon')
        ax.set_title('Calibration Error (Observed - Expected)')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coverage Error')
        
        # Annotate cells
        for h in range(n_horizons):
            for q in range(n_quantiles):
                val = error_matrix[h, q]
                color = 'white' if abs(val) > 0.05 else 'black'
                ax.text(q, h, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)
        
        fig.tight_layout()
        return ax
