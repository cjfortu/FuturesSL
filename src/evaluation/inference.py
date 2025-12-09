"""
Inference utilities for running model predictions.

Provides functions for batch inference on datasets with proper
device handling and output formatting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm


class ModelPredictor:
    """
    Utility class for running model inference on datasets.
    
    Handles batch processing, device management, and output formatting
    for evaluation and analysis.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained MIGT-TVDT model
                Type: nn.Module
            device: Device for inference (auto-detect if None)
                Type: torch.device or None
        """
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu')
        )
        self.model = model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def predict_batch(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference on a single batch.
        
        Args:
            batch: Batch from DataLoader
                Type: Dict[str, torch.Tensor]
                Required keys: features, attention_mask, bar_in_day,
                               day_of_week, day_of_month, day_of_year
                
        Returns:
            Model outputs
                Type: Dict[str, torch.Tensor]
                Keys: quantiles, norm_stats
        """
        # Move inputs to device
        features = batch['features'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        temporal_info = {
            'bar_in_day': batch['bar_in_day'].to(self.device),
            'day_of_week': batch['day_of_week'].to(self.device),
            'day_of_month': batch['day_of_month'].to(self.device),
            'day_of_year': batch['day_of_year'].to(self.device)
        }
        
        # Forward pass
        outputs = self.model(
            features=features,
            attention_mask=attention_mask,
            temporal_info=temporal_info
        )
        
        return outputs
    
    @torch.no_grad()
    def predict_dataset(
        self,
        dataloader: DataLoader,
        return_targets: bool = True,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on entire dataset.
        
        Args:
            dataloader: DataLoader for dataset
                Type: DataLoader
            return_targets: Whether to also return targets
                Type: bool
            show_progress: Show progress bar
                Type: bool
                
        Returns:
            Dictionary with predictions and optionally targets
                Type: Dict[str, ndarray]
                Keys: 'predictions' (N, H, Q), optionally 'targets' (N, H)
        """
        all_predictions = []
        all_targets = []
        
        iterator = tqdm(dataloader, desc='Predicting') if show_progress else dataloader
        
        for batch in iterator:
            outputs = self.predict_batch(batch)
            all_predictions.append(outputs['quantiles'].cpu().numpy())
            
            if return_targets and 'targets' in batch:
                all_targets.append(batch['targets'].numpy())
        
        result = {
            'predictions': np.concatenate(all_predictions, axis=0)
        }
        
        if return_targets and all_targets:
            result['targets'] = np.concatenate(all_targets, axis=0)
        
        return result
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        model_class: type,
        model_config: Dict,
        device: Optional[torch.device] = None
    ) -> 'ModelPredictor':
        """
        Create predictor from saved checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
                Type: Path
            model_class: Model class (MIGT_TVDT)
                Type: type
            model_config: Model configuration dict
                Type: Dict
            device: Device for inference
                Type: torch.device or None
                
        Returns:
            Initialized predictor with loaded weights
        """
        # Determine device
        device = device or (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu')
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Initialize model
        model = model_class(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, device)


def run_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
    horizon_names: List[str] = ['15m', '30m', '60m', '2h', '4h'],
    device: Optional[torch.device] = None
) -> Dict[str, any]:
    """
    Run full evaluation pipeline.
    
    Convenience function that combines prediction, metric computation,
    and result formatting.
    
    Args:
        model: Trained model
            Type: nn.Module
        dataloader: Test/validation dataloader
            Type: DataLoader
        quantiles: Quantile levels
            Type: List[float]
        horizon_names: Horizon names
            Type: List[str]
        device: Inference device
            Type: torch.device or None
            
    Returns:
        Comprehensive evaluation results
            Type: Dict[str, any]
    """
    from .metrics import MetricsSummary
    from .calibration import CalibrationByHorizon
    from .backtest import MultiHorizonBacktester
    
    # Run predictions
    predictor = ModelPredictor(model, device)
    pred_result = predictor.predict_dataset(dataloader)
    
    predictions = pred_result['predictions']
    targets = pred_result['targets']
    
    results = {
        'predictions': predictions,
        'targets': targets,
        'n_samples': len(targets)
    }
    
    # Compute metrics
    metrics_calc = MetricsSummary(quantiles, horizon_names)
    results['metrics'] = metrics_calc.compute_all(predictions, targets)
    
    # Calibration analysis
    cal_analyzer = CalibrationByHorizon(quantiles, horizon_names)
    results['calibration'] = cal_analyzer.compute_per_horizon(predictions, targets)
    
    # Backtest
    backtester = MultiHorizonBacktester(predictions, targets, horizon_names)
    results['backtest'] = backtester.run()
    results['backtest_summary'] = backtester.get_metrics_summary()
    
    return results


def format_evaluation_report(
    results: Dict[str, any],
    horizon_names: List[str] = ['15m', '30m', '60m', '2h', '4h']
) -> str:
    """
    Format evaluation results as markdown report.
    
    Args:
        results: Results from run_evaluation()
            Type: Dict[str, any]
        horizon_names: Horizon names
            Type: List[str]
            
    Returns:
        Formatted markdown report
            Type: str
    """
    lines = []
    lines.append("# Model Evaluation Report\n")
    lines.append(f"Samples evaluated: {results['n_samples']:,}\n")
    
    # Distributional metrics
    lines.append("\n## Distributional Metrics\n")
    lines.append("| Horizon | CRPS | PICP-80 | PICP-50 | MPIW-80 | MPIW-50 |")
    lines.append("|---------|------|---------|---------|---------|---------|")
    
    dist_metrics = results['metrics']['distributional']
    for h in horizon_names:
        if h in dist_metrics:
            m = dist_metrics[h]
            lines.append(
                f"| {h} | {m['crps']:.5f} | {m['picp_80']:.3f} | "
                f"{m['picp_50']:.3f} | {m['mpiw_80']:.5f} | {m['mpiw_50']:.5f} |"
            )
    
    # Point metrics
    lines.append("\n## Point Metrics (Median)\n")
    lines.append("| Horizon | IC | DA | RMSE | MAE |")
    lines.append("|---------|----|----|------|-----|")
    
    point_metrics = results['metrics']['point']
    for h in horizon_names:
        if h in point_metrics:
            m = point_metrics[h]
            lines.append(
                f"| {h} | {m['ic']:.4f} | {m['da']:.3f} | "
                f"{m['rmse']:.5f} | {m['mae']:.5f} |"
            )
    
    # Calibration
    lines.append("\n## Calibration Summary\n")
    cal = results['metrics']['calibration']
    lines.append(f"- Mean calibration error: {cal['mean']:.4f}")
    lines.append(f"- Max calibration error: {cal['max']:.4f}")
    
    # Backtest
    if 'backtest_summary' in results:
        lines.append("\n## Financial Metrics (Backtest)\n")
        lines.append("| Horizon | Sharpe | Sortino | Max DD | Profit Factor | Hit Rate |")
        lines.append("|---------|--------|---------|--------|---------------|----------|")
        
        bt_df = results['backtest_summary']
        for h in bt_df.index:
            row = bt_df.loc[h]
            lines.append(
                f"| {h} | {row['sharpe']:.3f} | {row['sortino']:.3f} | "
                f"{row['max_drawdown']:.2%} | {row['profit_factor']:.2f} | {row['hit_rate']:.2%} |"
            )
    
    return "\n".join(lines)
