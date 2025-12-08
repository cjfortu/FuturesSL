"""
Gated instance normalization modules for the MIGT-TVDT architecture.

Implements:
- RevIN: Reversible Instance Normalization for regime adaptation
- LiteGateUnit: MIGT-style gating for filtering noisy updates
- GatedInstanceNorm: Combined normalization with gating

Per scientific document Section 4.1, these components handle non-stationarity
(H1) and noise dominance (H5) by normalizing per instance and gating
attention outputs to suppress noisy updates.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    
    Normalizes each instance (sample) to zero mean and unit variance before
    processing, then reverses the normalization after the decoder. This helps
    the model learn shape-invariant patterns that generalize across different
    volatility regimes.
    
    Per scientific document: Applied to raw input before embedding;
    statistics stored for optional denormalization after output.
    """
    
    def __init__(
        self,
        n_variables: int,
        eps: float = 1e-5,
        affine: bool = True
    ):
        """
        Initialize RevIN.
        
        Args:
            n_variables: Number of input variables
                Type: int
                Value: 24
            eps: Small constant for numerical stability
                Type: float
                Default: 1e-5
            affine: Whether to learn scale/bias parameters
                Type: bool
                Default: True
        """
        super().__init__()
        self.n_variables = n_variables
        self.eps = eps
        self.affine = affine
        
        if affine:
            # Learnable affine parameters per variable
            self.gamma = nn.Parameter(torch.ones(n_variables))
            self.beta = nn.Parameter(torch.zeros(n_variables))
            
        # Storage for normalization statistics (set during forward)
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        
    def forward(
        self,
        x: torch.Tensor,
        mode: str = "normalize"
    ) -> torch.Tensor:
        """
        Normalize or denormalize input.
        
        In 'normalize' mode, computes and stores statistics, then normalizes.
        In 'denormalize' mode, uses stored statistics to restore original scale.
        
        Args:
            x: Input tensor
                Type: torch.Tensor (float32)
                Shape: (B, T, V) for normalize
                       (B, H, Q) or similar for denormalize (uses stored stats)
            mode: Operation mode
                Type: str
                Values: "normalize" or "denormalize"
                
        Returns:
            Normalized or denormalized tensor
                Type: torch.Tensor (float32)
                Shape: Same as input
        """
        if mode == "normalize":
            # Compute statistics over time dimension (dim=1)
            # Shape: (B, V)
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
            
            # Normalize: (B, T, V)
            x_norm = (x - self.mean) / self.std
            
            # Apply affine transformation if enabled
            if self.affine:
                x_norm = x_norm * self.gamma + self.beta
                
            return x_norm
            
        elif mode == "denormalize":
            if self.mean is None or self.std is None:
                raise ValueError("Must normalize before denormalize")
                
            # Remove affine transformation if applied
            if self.affine:
                x = (x - self.beta) / self.gamma
                
            # Denormalize
            # Note: For output predictions, we may need to handle different shapes
            # This assumes the statistics broadcast appropriately
            x_denorm = x * self.std + self.mean
            
            return x_denorm
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
    def get_stats(self) -> dict:
        """
        Get stored normalization statistics.
        
        Returns:
            Dictionary with mean and std tensors
                Type: Dict[str, torch.Tensor]
                Keys: 'mean' (B, 1, V), 'std' (B, 1, V)
        """
        return {
            'mean': self.mean,
            'std': self.std
        }


class LiteGateUnit(nn.Module):
    """
    Lite Gate Unit from MIGT paper.
    
    Filters attention output to suppress noisy signals and pass through
    only high-confidence feature updates. The gate learns which updates
    are informative vs. noise.
    
    Formula:
        G = sigmoid(W @ x + b)
        output = x + G * attention_output
        
    When G -> 0, the attention update is suppressed (noisy).
    When G -> 1, the attention update is fully applied (confident).
    """
    
    def __init__(self, d_model: int):
        """
        Initialize LiteGateUnit.
        
        Args:
            d_model: Model dimension
                Type: int
                Typical value: 256
        """
        super().__init__()
        
        # Gate projection: learns which dimensions to update
        self.gate_proj = nn.Linear(d_model, d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply gated residual update.
        
        The gate is computed from the pre-attention input to decide
        how much of the attention output to incorporate.
        
        Args:
            x: Pre-attention input
                Type: torch.Tensor (float32)
                Shape: (B, V, D)
            attention_output: Post-attention output (residual already removed)
                Type: torch.Tensor (float32)
                Shape: (B, V, D)
                
        Returns:
            Gated output
                Type: torch.Tensor (float32)
                Shape: (B, V, D)
        """
        # Compute gate from pre-attention state
        gate = torch.sigmoid(self.gate_proj(x))
        
        # Gated residual: x + gate * attention_output
        output = x + gate * attention_output
        
        return output


class GatedInstanceNorm(nn.Module):
    """
    Combined instance normalization with gating.
    
    Applies instance normalization followed by gating, as specified in
    MIGT architecture. Used post-variable-attention to stabilize training
    and filter noisy updates.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Initialize GatedInstanceNorm.
        
        Args:
            d_model: Model dimension
                Type: int
            eps: Numerical stability constant
                Type: float
        """
        super().__init__()
        
        # Instance normalization (per sample, per variable)
        self.norm = nn.LayerNorm(d_model, eps=eps)
        
        # Gating mechanism
        self.gate = LiteGateUnit(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply normalized gated update.
        
        Steps:
        1. Normalize the attention output
        2. Compute gated residual update
        
        Args:
            x: Pre-attention input
                Type: torch.Tensor (float32)
                Shape: (B, V, D)
            attention_output: Post-attention output (before residual)
                Type: torch.Tensor (float32)
                Shape: (B, V, D)
                
        Returns:
            Gated normalized output
                Type: torch.Tensor (float32)
                Shape: (B, V, D)
        """
        # Normalize attention output
        attn_normed = self.norm(attention_output)
        
        # Apply gated update
        output = self.gate(x, attn_normed - x)  # Pass the delta, not full output
        
        return output
