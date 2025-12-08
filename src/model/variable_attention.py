"""
Variable attention module for the MIGT-TVDT architecture.

Implements cross-variable self-attention to learn inter-variable correlations:
- Price-volume relationships (volume confirming price moves)
- RSI-momentum interactions
- Volatility-liquidity dependencies

Per scientific document Section 4.1, this is the second stage of the two-stage
attention mechanism, applied after temporal aggregation has collapsed the time
dimension to (B, V, D).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VariableAttentionBlock(nn.Module):
    """
    Self-attention over variable dimension.
    
    After temporal aggregation produces (B, V, D), this layer learns
    relationships between variables. For example:
    - How volume changes relate to price volatility
    - RSI-MACD confirmation patterns
    - Liquidity impact on momentum
    
    Standard transformer block with multi-head attention and feed-forward.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Initialize variable attention block.
        
        Args:
            d_model: Model dimension
                Type: int
                Typical value: 256
            n_heads: Number of attention heads
                Type: int
                Typical value: 8
            d_ff: Feed-forward hidden dimension
                Type: int
                Typical value: 1024
            dropout: Dropout probability
                Type: float
                Typical value: 0.1
        """
        super().__init__()
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Multi-head self-attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention over variable dimension.
        
        Each variable attends to all other variables, learning which
        cross-variable relationships are most predictive.
        
        Args:
            x: Aggregated variable representations
                Type: torch.Tensor (float32)
                Shape: (B, V, D) where B=batch, V=24 variables, D=256
                
        Returns:
            Cross-variable attended features
                Type: torch.Tensor (float32)
                Shape: (B, V, D) - same as input
        """
        B, V, D = x.shape
        
        # Pre-norm + self-attention
        residual = x
        x_norm = self.norm1(x)
        
        # Q, K, V projections: (B, V, D)
        Q = self.q_proj(x_norm)
        K = self.k_proj(x_norm)
        V_attn = self.v_proj(x_norm)
        
        # Reshape for multi-head: (B, V, D) -> (B, V, n_heads, d_head) -> (B, n_heads, V, d_head)
        Q = Q.view(B, V, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, V, self.n_heads, self.d_head).transpose(1, 2)
        V_attn = V_attn.view(B, V, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention scores: (B, n_heads, V, V)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention: (B, n_heads, V, d_head)
        attn_output = torch.matmul(attn_probs, V_attn)
        
        # Reshape: (B, n_heads, V, d_head) -> (B, V, n_heads, d_head) -> (B, V, D)
        attn_output = attn_output.transpose(1, 2).reshape(B, V, D)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection
        x = residual + self.dropout(attn_output)
        
        # Pre-norm + feed-forward
        residual = x
        x = residual + self.ff(self.norm2(x))
        
        return x
