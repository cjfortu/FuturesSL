"""
Quantile regression output heads for the MIGT-TVDT architecture.

Implements non-crossing quantile prediction using cumulative softplus:
- Base value + cumulative positive deltas guarantees monotonicity
- No post-hoc sorting or penalty terms needed
- Each horizon has its own head with shared representation

Per scientific document Section 4.2, quantile regression captures uncertainty
and tail behavior better than point predictions (H10).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class QuantileHead(nn.Module):
    """
    Quantile regression head for a single horizon.
    
    Outputs non-crossing quantiles using cumulative softplus parameterization:
        q_τ = base + cumsum(softplus(deltas))
    
    This guarantees strict monotonicity without any crossing penalties or
    post-processing. The softplus ensures all deltas are positive, and
    cumsum ensures each quantile is strictly greater than the previous.
    """
    
    def __init__(
        self,
        d_model: int,
        n_quantiles: int = 7,
        hidden_dim: int = 128
    ):
        """
        Initialize quantile head.
        
        Args:
            d_model: Input dimension from encoder
                Type: int
                Typical value: 256
            n_quantiles: Number of quantiles to predict
                Type: int
                Default: 7 (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
            hidden_dim: Hidden layer dimension
                Type: int
                Default: 128
        """
        super().__init__()
        self.n_quantiles = n_quantiles
        
        # MLP to process encoder output
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        # Base value projection (lowest quantile baseline)
        self.base_proj = nn.Linear(hidden_dim, 1)
        
        # Delta projections (increments between quantiles)
        # We predict n_quantiles deltas, cumsum gives quantile values
        self.delta_proj = nn.Linear(hidden_dim, n_quantiles)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict non-crossing quantiles.
        
        Uses cumulative softplus to guarantee monotonicity:
        1. Compute base value (roughly the minimum quantile)
        2. Compute positive deltas via softplus
        3. Cumsum deltas and add to base
        
        Args:
            x: Pooled representation from encoder
                Type: torch.Tensor (float32)
                Shape: (B, D) where B=batch, D=d_model
                
        Returns:
            Non-crossing quantile predictions
                Type: torch.Tensor (float32)
                Shape: (B, n_quantiles)
                Guaranteed: output[:, i] < output[:, i+1] for all i
        """
        # Process through MLP
        hidden = self.mlp(x)  # (B, hidden_dim)
        
        # Predict base (scalar per sample)
        base = self.base_proj(hidden)  # (B, 1)
        
        # Predict deltas and ensure positivity via softplus
        deltas = self.delta_proj(hidden)  # (B, n_quantiles)
        positive_deltas = F.softplus(deltas)  # Always > 0
        
        # Cumulative sum gives increasing quantiles
        cumulative = torch.cumsum(positive_deltas, dim=-1)  # (B, n_quantiles)
        
        # Add base to get final quantiles
        quantiles = base + cumulative  # (B, n_quantiles)
        
        return quantiles


class MultiHorizonQuantileHead(nn.Module):
    """
    Multi-horizon quantile output with shared representation.
    
    Each horizon (15m, 30m, 60m, 2h, 4h) has its own quantile head,
    but they share the initial representation from the encoder.
    Horizon embeddings provide horizon-specific context.
    """
    
    def __init__(
        self,
        d_model: int,
        n_horizons: int = 5,
        n_quantiles: int = 7,
        hidden_dim: int = 128
    ):
        """
        Initialize multi-horizon quantile head.
        
        Args:
            d_model: Input dimension from encoder
                Type: int
                Typical value: 256
            n_horizons: Number of prediction horizons
                Type: int
                Default: 5 (15m, 30m, 60m, 2h, 4h)
            n_quantiles: Number of quantiles per horizon
                Type: int
                Default: 7
            hidden_dim: Hidden dimension for each head
                Type: int
                Default: 128
        """
        super().__init__()
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
        
        # Learnable horizon embeddings
        self.horizon_embedding = nn.Embedding(n_horizons, d_model)
        
        # Separate quantile head per horizon
        self.heads = nn.ModuleList([
            QuantileHead(d_model, n_quantiles, hidden_dim)
            for _ in range(n_horizons)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict quantiles for all horizons.
        
        Each horizon head receives the encoder output plus its horizon
        embedding, then predicts non-crossing quantiles.
        
        Args:
            x: Pooled representation from encoder
                Type: torch.Tensor (float32)
                Shape: (B, D)
                
        Returns:
            Quantile predictions for all horizons
                Type: torch.Tensor (float32)
                Shape: (B, n_horizons, n_quantiles) = (B, 5, 7)
        """
        B = x.shape[0]
        device = x.device
        
        outputs = []
        for h in range(self.n_horizons):
            # Get horizon embedding
            h_idx = torch.tensor([h], device=device)
            h_embed = self.horizon_embedding(h_idx)  # (1, D)
            
            # Add horizon context to encoder output
            x_h = x + h_embed  # (B, D) + (1, D) broadcasts to (B, D)
            
            # Predict quantiles for this horizon
            q_h = self.heads[h](x_h)  # (B, n_quantiles)
            outputs.append(q_h)
            
        # Stack: list of (B, Q) -> (B, H, Q)
        output = torch.stack(outputs, dim=1)
        
        return output
