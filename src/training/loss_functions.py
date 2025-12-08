"""
Loss functions for quantile regression in the MIGT-TVDT model.

Implements pinball (quantile) loss for distributional forecasting.
Per scientific document Section 4.2, quantile regression captures
uncertainty and tail behavior better than point predictions.

The quantile heads already guarantee non-crossing via cumulative
softplus parameterization, so no crossing penalty is needed.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional


class PinballLoss(nn.Module):
    """
    Pinball (quantile) loss for quantile regression.
    
    For each quantile tau, the loss is:
        L_tau(y, q_tau) = tau * max(y - q_tau, 0) + (1 - tau) * max(q_tau - y, 0)
    
    This asymmetric loss encourages the model to predict the tau-th quantile:
    - When y > q_tau (underprediction): penalized by factor tau
    - When y < q_tau (overprediction): penalized by factor (1 - tau)
    
    For tau=0.5, this reduces to MAE and predicts the median.
    For tau=0.9, the model learns to predict the 90th percentile.
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    ):
        """
        Initialize PinballLoss.
        
        Args:
            quantiles: List of quantile levels to predict
                Type: List[float]
                Default: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
                Values should be in (0, 1)
        """
        super().__init__()
        
        # Validate quantiles
        for q in quantiles:
            if not 0 < q < 1:
                raise ValueError(f"Quantile {q} must be in (0, 1)")
        
        # Store as buffer (non-trainable tensor that moves with model)
        self.register_buffer(
            'quantiles',
            torch.tensor(quantiles, dtype=torch.float32)
        )
        self.n_quantiles = len(quantiles)
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute pinball loss across all quantiles and horizons.
        
        The loss averages over batch, horizons, and quantiles. If a mask is
        provided (for valid samples only), masked positions are excluded.
        
        Args:
            predictions: Predicted quantiles
                Type: torch.Tensor (float32)
                Shape: (B, H, Q) where H=horizons, Q=n_quantiles
            targets: Actual values
                Type: torch.Tensor (float32)
                Shape: (B, H) - one target per horizon
            mask: Optional mask for valid samples
                Type: torch.Tensor (bool) or None
                Shape: (B, H) where True=valid
                
        Returns:
            Scalar loss value
                Type: torch.Tensor (float32)
                Shape: ()
        """
        # Expand targets to match predictions: (B, H) -> (B, H, Q)
        targets_expanded = targets.unsqueeze(-1)  # (B, H, 1)
        
        # Compute errors for all quantiles: predictions - targets
        errors = predictions - targets_expanded  # (B, H, Q)
        
        # Get quantile weights: (Q,) -> (1, 1, Q) for broadcasting
        quantiles = self.quantiles.view(1, 1, -1)
        
        # Pinball loss formula:
        # When error > 0 (overprediction): weight = (1 - tau)
        # When error < 0 (underprediction): weight = tau
        # Loss = tau * max(-error, 0) + (1-tau) * max(error, 0)
        #      = max(tau * (-error), (tau-1) * (-error))
        #      = max((tau-1) * error, tau * error)  [when error is either sign]
        loss = torch.max(
            (quantiles - 1) * errors,  # Contribution when error > 0
            quantiles * errors          # Contribution when error < 0
        )  # (B, H, Q)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask: (B, H) -> (B, H, Q)
            mask_expanded = mask.unsqueeze(-1).expand_as(loss)
            # Zero out invalid positions
            loss = loss * mask_expanded.float()
            # Mean over valid positions only
            n_valid = mask_expanded.sum()
            if n_valid > 0:
                return loss.sum() / n_valid
            else:
                return loss.sum() * 0.0  # Return zero with gradients
        
        # Average over all dimensions
        return loss.mean()
    
    def per_quantile_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss broken down by quantile (for logging/analysis).
        
        Args:
            predictions: (B, H, Q)
            targets: (B, H)
            
        Returns:
            Dictionary mapping quantile strings to loss values
                Type: Dict[str, torch.Tensor]
        """
        targets_expanded = targets.unsqueeze(-1)
        errors = predictions - targets_expanded
        quantiles = self.quantiles.view(1, 1, -1)
        
        loss = torch.max(
            (quantiles - 1) * errors,
            quantiles * errors
        )
        
        # Mean over batch and horizons, keep quantiles separate
        per_q_loss = loss.mean(dim=(0, 1))  # (Q,)
        
        return {
            f"q{int(q*100):02d}": per_q_loss[i].item()
            for i, q in enumerate(self.quantiles.tolist())
        }
    
    def per_horizon_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        horizon_names: List[str] = ['15m', '30m', '60m', '2h', '4h']
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss broken down by horizon (for logging/analysis).
        
        Args:
            predictions: (B, H, Q)
            targets: (B, H)
            horizon_names: Names for each horizon
            
        Returns:
            Dictionary mapping horizon names to loss values
        """
        targets_expanded = targets.unsqueeze(-1)
        errors = predictions - targets_expanded
        quantiles = self.quantiles.view(1, 1, -1)
        
        loss = torch.max(
            (quantiles - 1) * errors,
            quantiles * errors
        )
        
        # Mean over batch and quantiles, keep horizons separate
        per_h_loss = loss.mean(dim=(0, 2))  # (H,)
        
        return {
            name: per_h_loss[i].item()
            for i, name in enumerate(horizon_names)
        }


class QuantileCrossingPenalty(nn.Module):
    """
    Penalty for quantile crossing violations.
    
    Adds soft penalty when predicted quantiles are not monotonically increasing.
    
    Note: This penalty is typically NOT needed when using cumulative softplus
    parameterization in quantile heads, which mathematically guarantees
    non-crossing. Included for API completeness per engineering specification.
    """
    
    def __init__(self, weight: float = 0.1):
        """
        Initialize crossing penalty.
        
        Args:
            weight: Penalty weight multiplier
                Type: float
                Default: 0.1
        """
        super().__init__()
        self.weight = weight
        
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute crossing penalty.
        
        Penalizes cases where q[i+1] < q[i] (crossing).
        
        Args:
            predictions: (B, H, Q) quantile predictions
            
        Returns:
            Scalar crossing penalty
        """
        # Compute differences between adjacent quantiles
        diffs = predictions[:, :, 1:] - predictions[:, :, :-1]  # (B, H, Q-1)
        
        # Penalize negative differences (crossings)
        # Using ReLU(-diff) to only penalize when diff < 0
        crossings = torch.relu(-diffs)
        
        return self.weight * crossings.mean()


class CombinedQuantileLoss(nn.Module):
    """
    Combined loss: Pinball + crossing penalty.
    
    Primary component is PinballLoss. Crossing penalty is included for
    API completeness but typically not needed when quantile heads use
    cumulative softplus parameterization (which guarantees non-crossing).
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
        crossing_weight: float = 0.0
    ):
        """
        Initialize CombinedQuantileLoss.
        
        Args:
            quantiles: Quantile levels
                Type: List[float]
            crossing_weight: Weight for crossing penalty
                Type: float
                Default: 0.0 (disabled, as cumulative softplus guarantees non-crossing)
        """
        super().__init__()
        self.pinball = PinballLoss(quantiles)
        self.crossing_penalty = QuantileCrossingPenalty(weight=1.0)
        self.crossing_weight = crossing_weight
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: (B, H, Q) predicted quantiles
            targets: (B, H) actual values
            mask: Optional (B, H) validity mask
            
        Returns:
            Dictionary with loss components
                Type: Dict[str, torch.Tensor]
                Keys: 'total', 'pinball', 'crossing'
        """
        # Primary pinball loss
        pinball_loss = self.pinball(predictions, targets, mask)
        
        # Crossing penalty (typically 0 with cumulative softplus heads)
        if self.crossing_weight > 0:
            crossing_loss = self.crossing_penalty(predictions)
        else:
            crossing_loss = torch.zeros_like(pinball_loss)
        
        total_loss = pinball_loss + self.crossing_weight * crossing_loss
        
        return {
            'total': total_loss,
            'pinball': pinball_loss,
            'crossing': crossing_loss
        }
    
    def get_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute detailed metrics for logging.
        
        Args:
            predictions: (B, H, Q)
            targets: (B, H)
            
        Returns:
            Dictionary with per-quantile and per-horizon losses
        """
        metrics = {}
        
        # Per-quantile losses
        q_losses = self.pinball.per_quantile_loss(predictions, targets)
        metrics.update({f"loss_{k}": v for k, v in q_losses.items()})
        
        # Per-horizon losses
        h_losses = self.pinball.per_horizon_loss(predictions, targets)
        metrics.update({f"loss_{k}": v for k, v in h_losses.items()})
        
        # Coverage statistics (what fraction of targets fall below each quantile)
        targets_expanded = targets.unsqueeze(-1)  # (B, H, 1)
        below = (targets_expanded <= predictions).float()  # (B, H, Q)
        coverage = below.mean(dim=(0, 1))  # (Q,)
        
        for i, q in enumerate(self.pinball.quantiles.tolist()):
            metrics[f"coverage_q{int(q*100):02d}"] = coverage[i].item()
        
        # Interval statistics
        interval_80 = predictions[:, :, 5] - predictions[:, :, 1]  # q90 - q10
        interval_50 = predictions[:, :, 4] - predictions[:, :, 2]  # q75 - q25
        
        metrics['interval_80_mean'] = interval_80.mean().item()
        metrics['interval_50_mean'] = interval_50.mean().item()
        
        # 80% interval coverage (should be ~0.8 if well-calibrated)
        in_interval = (
            (targets >= predictions[:, :, 1]) &
            (targets <= predictions[:, :, 5])
        ).float()
        metrics['picp_80'] = in_interval.mean().item()
        
        return metrics
