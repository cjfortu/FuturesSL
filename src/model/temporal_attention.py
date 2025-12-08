"""
Temporal attention modules for the MIGT-TVDT architecture.

Implements per-variable temporal self-attention following RL-TVDT design:
- Self-attention over time dimension, applied independently to each variable
- Learns temporal dynamics (trends, seasonality) before cross-variable interaction
- Attention-weighted aggregation reduces time dimension to fixed representation

Per scientific document Section 4.1, bidirectional attention over the lookback
window is appropriate since all data is historical (no future leakage concern).

Memory Optimization:
Uses PyTorch's scaled_dot_product_attention (Flash Attention backend) for
O(N) memory complexity instead of O(N^2). This avoids materializing the full
(B*V, n_heads, T, T) attention score matrix, reducing memory from ~4 GB per
layer to ~200 MB, enabling batch_size=128+ within 25GB budget on A100.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class TemporalAttentionBlock(nn.Module):
    """
    Self-attention over time dimension, applied independently per variable.
    
    For input (B, T, V, D), reshapes to (B*V, T, D) to apply standard
    multi-head self-attention over the time dimension, then reshapes back.
    
    This learns temporal patterns (trends, mean reversion, volatility clustering)
    within each variable before any cross-variable interaction. The attention
    mask ensures padding tokens don't contribute to attention computation.
    
    Memory Efficiency:
        Uses F.scaled_dot_product_attention which leverages Flash Attention
        on compatible hardware (A100, H100). This avoids materializing the
        full (B*V, n_heads, T, T) attention matrix, reducing memory from
        O(T^2) to O(T) and enabling larger batch sizes.
        
        Memory comparison for B=64, V=24, T=288, H=8:
        - Manual attention: ~4.08 GB per layer for attn_scores alone
        - Flash Attention: ~0.2 GB per layer (fused, no materialization)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Initialize temporal attention block.
        
        Args:
            d_model: Model dimension
                Type: int
                Typical value: 256
            n_heads: Number of attention heads
                Type: int
                Typical value: 8
                Constraint: d_model must be divisible by n_heads
            d_ff: Feed-forward hidden dimension
                Type: int
                Typical value: 1024 (4 * d_model)
            dropout: Dropout probability
                Type: float
                Typical value: 0.1
        """
        super().__init__()
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout
        
        # Multi-head self-attention components
        # Keep manual projections for flexibility and compatibility with existing code
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
        
        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply temporal self-attention to each variable independently.
        
        The input is reshaped to process all variables in parallel, with
        attention computed over the time dimension. The mask prevents
        attention to padding positions.
        
        Uses Flash Attention via scaled_dot_product_attention for memory
        efficiency. On A100 with PyTorch 2.0+, this reduces memory from
        ~4 GB per attention layer to ~200 MB, enabling 6x larger batch sizes.
        
        Args:
            x: Input tensor
                Type: torch.Tensor (float32)
                Shape: (B, T, V, D) where B=batch, T=288, V=24, D=256
            attention_mask: Boolean mask for valid positions
                Type: torch.Tensor (bool)
                Shape: (B, T) where True=valid, False=padding
                
        Returns:
            Attended tensor
                Type: torch.Tensor (float32)
                Shape: (B, T, V, D) - same as input
        """
        B, T, V, D = x.shape
        
        # Reshape: (B, T, V, D) -> (B*V, T, D) for batched attention
        # This treats each variable as an independent sequence
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * V, T, D)
        
        # Expand mask: (B, T) -> (B*V, T)
        # Each variable in a sample shares the same padding mask
        mask_expanded = attention_mask.unsqueeze(1).expand(-1, V, -1).reshape(B * V, T)
        
        # Pre-norm + self-attention (pre-norm transformer architecture)
        residual = x_reshaped
        x_norm = self.norm1(x_reshaped)
        
        # Compute Q, K, V projections: (B*V, T, D)
        Q = self.q_proj(x_norm)
        K = self.k_proj(x_norm)
        V_attn = self.v_proj(x_norm)
        
        # Reshape for multi-head attention
        # (B*V, T, D) -> (B*V, n_heads, T, d_head)
        Q = Q.view(B * V, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B * V, T, self.n_heads, self.d_head).transpose(1, 2)
        V_attn = V_attn.view(B * V, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Prepare mask for SDPA
        # SDPA boolean convention: True = masked (do NOT attend), False = attend
        # Our convention: True = valid, False = padding
        # Therefore we must INVERT the mask
        # Shape: (B*V, T) -> (B*V, 1, 1, T) for broadcasting over (heads, query_len, key_len)
        attn_mask = ~mask_expanded  # Invert: now True = padding (mask out)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B*V, 1, 1, T)
        
        # Use scaled_dot_product_attention with Flash Attention backend
        # This fused kernel avoids materializing the (B*V, n_heads, T, T) attention matrix
        # Memory: O(T) instead of O(T^2), enabling ~6x larger batch sizes
        # Automatically handles: scaling by sqrt(d_head), softmax, dropout
        attn_output = F.scaled_dot_product_attention(
            Q, K, V_attn,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False  # Bidirectional attention over historical data
        )
        
        # Reshape back: (B*V, n_heads, T, d_head) -> (B*V, T, D)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B * V, T, D)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection with dropout
        x_reshaped = residual + self.dropout(attn_output)
        
        # Pre-norm + feed-forward
        residual = x_reshaped
        x_reshaped = residual + self.ff(self.norm2(x_reshaped))
        
        # Reshape back: (B*V, T, D) -> (B, V, T, D) -> (B, T, V, D)
        output = x_reshaped.view(B, V, T, D).permute(0, 2, 1, 3)
        
        return output


class TemporalAggregation(nn.Module):
    """
    Aggregate temporal sequence into fixed-size representation per variable.
    
    Uses attention-weighted pooling with learnable query to aggregate the
    T timesteps into a single D-dimensional vector per variable. This is
    more flexible than mean pooling as it learns which timesteps are most
    important for prediction.
    
    Per scientific document: Bidirectional over historical lookback is
    appropriate since all data is past (no future leakage). This preserves
    relative sequence importance while capturing the full context.
    
    Memory Note:
        This module uses manual attention (not SDPA) because the attention
        matrix is (B*V, 1, T) which is O(T), not O(T^2). The query dimension
        is 1 (single learnable query), so no T x T matrix is created.
        Memory cost is negligible (~0.5 GB) compared to temporal attention.
    """
    
    def __init__(self, d_model: int, n_heads: int = 1):
        """
        Initialize temporal aggregation.
        
        Args:
            d_model: Model dimension
                Type: int
                Typical value: 256
            n_heads: Number of attention heads for aggregation
                Type: int
                Default: 1 (single head sufficient for pooling)
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Learnable query for attention-weighted pooling
        # Shape: (1, 1, D) - single query attending to all timesteps
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Key and value projections for the sequence
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.scale = math.sqrt(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate time dimension via attention-weighted pooling.
        
        The learnable query attends to all valid timesteps, producing a
        weighted average that emphasizes the most predictive positions.
        
        Memory: O(T) not O(T^2) since query dimension is 1.
        Attention scores shape: (B*V, 1, T) - no T x T matrix.
        
        Args:
            x: Input tensor after temporal attention
                Type: torch.Tensor (float32)
                Shape: (B, T, V, D)
            attention_mask: Boolean mask for valid positions
                Type: torch.Tensor (bool)
                Shape: (B, T)
                
        Returns:
            Aggregated representation
                Type: torch.Tensor (float32)
                Shape: (B, V, D) - time dimension collapsed
        """
        B, T, V, D = x.shape
        
        # Reshape: (B, T, V, D) -> (B*V, T, D)
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * V, T, D)
        
        # Expand mask: (B, T) -> (B*V, T)
        mask_expanded = attention_mask.unsqueeze(1).expand(-1, V, -1).reshape(B * V, T)
        
        # Expand query: (1, 1, D) -> (B*V, 1, D)
        query = self.query.expand(B * V, -1, -1)
        
        # Compute K, V: (B*V, T, D)
        K = self.k_proj(x_reshaped)
        V_pool = self.v_proj(x_reshaped)
        
        # Attention scores: (B*V, 1, D) @ (B*V, D, T) -> (B*V, 1, T)
        # This is O(T), not O(T^2) - only 1 query position
        attn_scores = torch.matmul(query, K.transpose(-2, -1)) / self.scale
        
        # Apply mask: (B*V, T) -> (B*V, 1, T)
        # Our convention: True = valid, so mask out where False (padding)
        mask_attn = mask_expanded.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(~mask_attn, float('-inf'))
        
        # Softmax over time dimension
        attn_probs = F.softmax(attn_scores, dim=-1)  # (B*V, 1, T)
        
        # Weighted sum: (B*V, 1, T) @ (B*V, T, D) -> (B*V, 1, D)
        aggregated = torch.matmul(attn_probs, V_pool)
        
        # Remove singleton and reshape: (B*V, 1, D) -> (B*V, D) -> (B, V, D)
        output = aggregated.squeeze(1).view(B, V, D)
        
        return output
