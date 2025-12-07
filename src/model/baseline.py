"""
Phase 2 Baseline Transformer Model
===================================

This module implements the Phase 2 baseline Transformer for NQ futures prediction.
It provides a vanilla Transformer encoder with Instance Normalization, Cyclical
Positional Encoding, and independent multi-horizon quantile heads.

Phase 2 vs Phase 3 Differences:
    - Phase 2: Vanilla TransformerEncoder (standard multi-head attention)
    - Phase 3: Will replace with TSA (Two-Stage Attention) + LGU (Lite Gate Units)
    - Phase 2: Independent quantile heads per horizon (no autoregressive conditioning)
    - Phase 3: Will add autoregressive heads with teacher forcing

Key Components (per grok-scientific.md):
    - InstanceNorm1d: Per-sample, per-feature normalization across time
    - CyclicalPositionalEncoding: Time-of-day, day-of-week, etc.
    - Standard TransformerEncoder: Multi-head self-attention (8 heads, 6 layers)
    - QuantileHead: 7 quantiles [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    - IndependentMultiHorizonHead: 6 horizons, independent predictions

References:
    - grok-scientific.md Section 3.3: Instance Normalization
    - grok-scientific.md Section 3.2: Cyclical Positional Embeddings
    - grok-scientific.md Section 3.6: Distributional Output (quantiles)
    - claude-engineering.md Phase 2: Baseline model specification

Authors: Claude (Engineering Lead), Gemini (Research), Grok (Scientific)
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure module logger
logger = logging.getLogger(__name__)


# Model constants per grok-scientific.md
QUANTILES: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
NUM_QUANTILES: int = len(QUANTILES)
HORIZONS: List[int] = [5, 15, 30, 60, 120, 240]
NUM_HORIZONS: int = len(HORIZONS)


class InstanceNorm1d(nn.Module):
    """
    Instance Normalization for time series.
    
    Per grok-scientific.md Section 3.3:
    "Normalizes per sample, per feature across time dimension. Statistics μ, σ
    computed strictly on input window X_{1:T}, excluding targets to prevent
    look-ahead bias. Hypothesis: Enforces stationarity, enabling regime-agnostic
    learning."
    
    Unlike BatchNorm, Instance Normalization:
    - Computes statistics per sample (not across batch)
    - Computes statistics per feature (not across features)
    - Handles variable-length sequences via attention mask
    
    Mathematical formulation:
        IN(x) = γ * (x - μ(x)) / √(σ²(x) + ε) + β
    
    Where μ and σ are computed only over valid (non-padded) positions.
    
    Attributes:
        num_features: Number of input features (V=24).
        eps: Small constant for numerical stability.
        affine: Whether to learn scale (γ) and shift (β) parameters.
        gamma: Learnable scale parameter (V,).
        beta: Learnable shift parameter (V,).
    
    Example:
        >>> instance_norm = InstanceNorm1d(num_features=24)
        >>> x = torch.randn(8, 7000, 24)  # (B, T, V)
        >>> mask = torch.ones(8, 7000)    # (B, T)
        >>> x_norm = instance_norm(x, mask)
        >>> print(x_norm.shape)  # (8, 7000, 24)
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True
    ):
        """
        Initialize Instance Normalization layer.
        
        Args:
            num_features: Number of features to normalize (V).
                Type: int
            eps: Small constant added to variance for stability.
                Type: float
                Default: 1e-5
            affine: If True, learn scale (γ) and shift (β) parameters.
                Type: bool
                Default: True
        """
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            # Learnable affine parameters per feature
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply instance normalization with mask support.
        
        Computation:
        1. Compute mean and variance only over valid (masked) positions
        2. Normalize each sample, each feature independently
        3. Apply learnable scale and shift if affine=True
        4. Zero out padded positions
        
        Args:
            x: Input tensor of shape (B, T, V).
                Type: torch.Tensor
                B = batch size, T = sequence length, V = num_features
            mask: Attention mask of shape (B, T).
                Type: Optional[torch.Tensor]
                1.0 = valid position, 0.0 = padded position
                If None, all positions are treated as valid.
        
        Returns:
            Normalized tensor of shape (B, T, V).
            Type: torch.Tensor
        """
        # Default mask: all valid
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
        
        # Expand mask for broadcasting: (B, T) -> (B, T, 1)
        mask_expanded = mask.unsqueeze(-1)
        
        # Count valid positions per sample: (B, 1, 1)
        valid_counts = mask.sum(dim=1, keepdim=True).unsqueeze(-1)
        valid_counts = valid_counts.clamp(min=1)  # Prevent division by zero
        
        # Masked input: zero out padded positions for statistics computation
        x_masked = x * mask_expanded
        
        # Compute mean: sum over time / count, per sample, per feature
        # Shape: (B, 1, V)
        mean = x_masked.sum(dim=1, keepdim=True) / valid_counts
        
        # Compute variance: E[(x - μ)²] over valid positions
        # Shape: (B, 1, V)
        diff = (x_masked - mean * mask_expanded)
        var = (diff ** 2).sum(dim=1, keepdim=True) / valid_counts
        
        # Normalize
        std = torch.sqrt(var + self.eps)  # (B, 1, V)
        x_norm = (x - mean) / std
        
        # Apply learnable affine transformation
        if self.affine:
            x_norm = self.gamma * x_norm + self.beta
        
        # Zero out padded positions
        x_norm = x_norm * mask_expanded
        
        return x_norm


class CyclicalPositionalEncoding(nn.Module):
    """
    Cyclical positional encoding for temporal patterns.
    
    Per grok-scientific.md Section 3.2:
    "For cyclical c with period P: φ(c) = [sin(2πc/P), cos(2πc/P)]
    Specifications: Time-of-day (P=1440), day-of-month (P=31, fixed),
    day-of-year (P=365.25); day-of-week learnable E ∈ R^{7×d}"
    
    This module takes pre-computed temporal features from the dataset and
    projects them into model dimension for addition to input embeddings.
    
    Input temporal features (8 channels from dataset):
        [0]: sin(time_of_day)
        [1]: cos(time_of_day)
        [2]: day_of_week (0-6)
        [3]: sin(day_of_month)
        [4]: cos(day_of_month)
        [5]: sin(day_of_year)
        [6]: cos(day_of_year)
        [7]: normalized minute (unused in encoding)
    
    Attributes:
        d_model: Model embedding dimension.
        dow_embedding: Learnable embedding for day-of-week (7 classes).
        proj: Linear projection from cyclical features to d_model.
    
    Example:
        >>> pe = CyclicalPositionalEncoding(d_model=512)
        >>> temporal = torch.randn(8, 7000, 8)  # (B, T, 8)
        >>> pos_enc = pe(temporal)
        >>> print(pos_enc.shape)  # (8, 7000, 512)
    """
    
    def __init__(self, d_model: int, max_len: int = 7000):
        """
        Initialize cyclical positional encoding.
        
        Args:
            d_model: Model embedding dimension.
                Type: int
            max_len: Maximum sequence length (for reference).
                Type: int
                Default: 7000
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Learnable embedding for day-of-week (7 days)
        # Per grok-scientific.md: "day-of-week learnable E ∈ R^{7×d}"
        dow_dim = d_model // 4  # Use 1/4 of d_model for day-of-week
        self.dow_embedding = nn.Embedding(7, dow_dim)
        
        # Linear projection: 6 cyclical features + dow_embedding -> d_model
        # 6 = sin/cos for: time_of_day (2), day_of_month (2), day_of_year (2)
        input_dim = 6 + dow_dim
        self.proj = nn.Linear(input_dim, d_model)
    
    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Compute positional encodings from temporal features.
        
        Args:
            temporal_features: Pre-computed temporal features from dataset.
                Type: torch.Tensor
                Shape: (B, T, 8)
                Channels: [sin_tod, cos_tod, dow, sin_dom, cos_dom, sin_doy, cos_doy, norm_min]
        
        Returns:
            Positional encodings of shape (B, T, d_model).
            Type: torch.Tensor
        """
        B, T, _ = temporal_features.shape
        
        # Extract cyclical features (indices 0,1,3,4,5,6)
        # Exclude index 2 (dow - will embed) and index 7 (normalized minute - unused)
        cyclical = temporal_features[:, :, [0, 1, 3, 4, 5, 6]]  # (B, T, 6)
        
        # Embed day of week
        dow = temporal_features[:, :, 2].long().clamp(0, 6)  # (B, T)
        dow_emb = self.dow_embedding(dow)  # (B, T, dow_dim)
        
        # Concatenate cyclical features and day-of-week embedding
        combined = torch.cat([cyclical, dow_emb], dim=-1)  # (B, T, 6 + dow_dim)
        
        # Project to model dimension
        pe = self.proj(combined)  # (B, T, d_model)
        
        return pe


class QuantileHead(nn.Module):
    """
    Quantile regression head for a single horizon.
    
    Outputs 7 quantiles: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    
    Per grok-scientific.md Section 3.6:
    "For each horizon h, output quantiles {ŷ_τk}_{k=1}^7"
    
    Architecture:
        Linear(d_model -> hidden) -> GELU -> Linear(hidden -> hidden) -> GELU -> Linear(hidden -> 7)
    
    Attributes:
        mlp: Multi-layer perceptron for quantile prediction.
        quantiles: List of quantile values for reference.
    
    Example:
        >>> head = QuantileHead(d_model=512)
        >>> h = torch.randn(8, 512)  # (B, d_model)
        >>> q = head(h)
        >>> print(q.shape)  # (8, 7)
    """
    
    QUANTILES: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    
    def __init__(self, d_model: int, hidden_dim: int = 256):
        """
        Initialize quantile prediction head.
        
        Args:
            d_model: Input embedding dimension.
                Type: int
            hidden_dim: Hidden layer dimension.
                Type: int
                Default: 256
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.QUANTILES)),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict quantiles from hidden state.
        
        Args:
            x: Hidden state of shape (B, d_model).
                Type: torch.Tensor
        
        Returns:
            Quantile predictions of shape (B, 7).
            Type: torch.Tensor
        """
        return self.mlp(x)


class IndependentMultiHorizonHead(nn.Module):
    """
    Independent multi-horizon quantile heads (Phase 2).
    
    This is the Phase 2 implementation with independent predictions per horizon.
    Phase 3 will add autoregressive conditioning where shorter horizons
    condition longer horizons per grok-scientific.md Section 3.6.
    
    Phase 2 simplification:
        - Each horizon has independent QuantileHead
        - No autoregressive conditioning
        - No teacher forcing (not needed without conditioning)
    
    Phase 3 enhancement (deferred):
        - Shorter horizon outputs condition longer horizon inputs
        - Teacher forcing during training (use ground truth for conditioning)
        - Predicted median for inference
    
    Attributes:
        heads: ModuleList of QuantileHead, one per horizon.
    
    Example:
        >>> mh_head = IndependentMultiHorizonHead(d_model=512)
        >>> h = torch.randn(8, 512)  # (B, d_model)
        >>> preds = mh_head(h)
        >>> print(preds.shape)  # (8, 6, 7) = (B, H, Q)
    """
    
    HORIZONS: List[int] = [5, 15, 30, 60, 120, 240]
    
    def __init__(self, d_model: int, hidden_dim: int = 256):
        """
        Initialize independent multi-horizon heads.
        
        Args:
            d_model: Input embedding dimension.
                Type: int
            hidden_dim: Hidden layer dimension for each head.
                Type: int
                Default: 256
        """
        super().__init__()
        
        # One independent head per horizon
        self.heads = nn.ModuleList([
            QuantileHead(d_model, hidden_dim)
            for _ in self.HORIZONS
        ])
    
    def forward(
        self,
        h: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing: bool = False
    ) -> torch.Tensor:
        """
        Predict quantiles for all horizons independently.
        
        Note: targets and teacher_forcing parameters are included for API
        compatibility with Phase 3 autoregressive heads but are unused in
        Phase 2 independent heads.
        
        Args:
            h: Hidden state of shape (B, d_model).
                Type: torch.Tensor
            targets: Ground truth (unused in Phase 2).
                Type: Optional[torch.Tensor]
                Shape: (B, 6) if provided
            teacher_forcing: Whether to use teacher forcing (unused in Phase 2).
                Type: bool
        
        Returns:
            Quantile predictions of shape (B, H, Q).
            Type: torch.Tensor
            H = num_horizons (6), Q = num_quantiles (7)
        """
        # Predict each horizon independently
        outputs = [head(h) for head in self.heads]
        
        # Stack: (B, 6, 7)
        return torch.stack(outputs, dim=1)


class BaselineTransformer(nn.Module):
    """
    Phase 2 Baseline Transformer for NQ futures prediction.
    
    This is a vanilla Transformer encoder with:
    - Instance Normalization (per grok-scientific.md Section 3.3)
    - Cyclical Positional Encoding (per grok-scientific.md Section 3.2)
    - Standard multi-head self-attention (PyTorch TransformerEncoder)
    - Independent multi-horizon quantile heads
    
    Phase 2 architecture (vanilla):
        Input (B, T, V=24)
        → InstanceNorm (per-sample, per-feature)
        → Linear projection (V → d_model)
        → + CyclicalPositionalEncoding
        → TransformerEncoder (standard MHA, 6 layers)
        → Extract last valid position (prediction point)
        → LayerNorm
        → IndependentMultiHorizonHead
        → Output (B, H=6, Q=7)
    
    Phase 3 will replace TransformerEncoder with TSA + LGU blocks.
    
    Attributes:
        num_features: Number of input features (V=24).
        d_model: Model embedding dimension.
        instance_norm: Instance normalization layer.
        input_proj: Linear projection from features to d_model.
        pos_encoding: Cyclical positional encoding.
        transformer_encoder: PyTorch TransformerEncoder.
        output_norm: LayerNorm before prediction heads.
        heads: Independent multi-horizon quantile heads.
    
    Example:
        >>> model = BaselineTransformer(num_features=24, d_model=512)
        >>> features = torch.randn(8, 7000, 24)
        >>> mask = torch.ones(8, 7000)
        >>> temporal = torch.randn(8, 7000, 8)
        >>> preds = model(features, mask, temporal)
        >>> print(preds.shape)  # (8, 6, 7)
    """
    
    def __init__(
        self,
        num_features: int = 24,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 7000,
    ):
        """
        Initialize baseline Transformer model.
        
        Args:
            num_features: Number of input features (V).
                Type: int
                Default: 24 (per grok-scientific.md)
            d_model: Embedding and Transformer dimension.
                Type: int
                Default: 512
            num_heads: Number of attention heads.
                Type: int
                Default: 8
            num_layers: Number of Transformer encoder layers.
                Type: int
                Default: 6 (baseline; Phase 3 may use 8-12)
            ffn_dim: Feedforward network dimension.
                Type: int
                Default: 2048 (4x d_model is common)
            dropout: Dropout probability.
                Type: float
                Default: 0.1
            max_len: Maximum sequence length.
                Type: int
                Default: 7000 (T_MAX)
        """
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Instance normalization (per grok-scientific.md Section 3.3)
        self.instance_norm = InstanceNorm1d(num_features)
        
        # Input projection: V -> d_model
        self.input_proj = nn.Linear(num_features, d_model)
        
        # Cyclical positional encoding (per grok-scientific.md Section 3.2)
        self.pos_encoding = CyclicalPositionalEncoding(d_model, max_len)
        
        # Standard Transformer encoder
        # Phase 2: Vanilla attention; Phase 3 will use TSA + LGU
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # Input shape: (B, T, d_model)
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)
        
        # Multi-horizon quantile heads
        # Phase 2: Independent; Phase 3 will add autoregressive
        self.heads = IndependentMultiHorizonHead(d_model)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"BaselineTransformer initialized: "
            f"d_model={d_model}, layers={num_layers}, heads={num_heads}"
        )
    
    def _init_weights(self):
        """
        Initialize model weights.
        
        Uses Xavier/Glorot uniform for linear layers and normal for embeddings,
        following standard Transformer initialization practices.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        temporal_features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through baseline Transformer.
        
        Processing pipeline:
        1. Instance normalization (per-sample, per-feature)
        2. Linear projection to d_model
        3. Add positional encoding
        4. Transformer encoder with attention mask
        5. Extract last valid position
        6. Output normalization
        7. Multi-horizon prediction heads
        
        Args:
            features: Input features of shape (B, T, V).
                Type: torch.Tensor
                V = num_features (24)
            mask: Attention mask of shape (B, T).
                Type: torch.Tensor
                1.0 = valid, 0.0 = padded
            temporal_features: Temporal features of shape (B, T, 8).
                Type: torch.Tensor
                From NQFuturesDataset._extract_temporal_features
            targets: Ground truth targets (unused in Phase 2).
                Type: Optional[torch.Tensor]
                Shape: (B, 6) if provided
            teacher_forcing: Whether to use teacher forcing (unused in Phase 2).
                Type: bool
        
        Returns:
            Quantile predictions of shape (B, H, Q).
            Type: torch.Tensor
            H = num_horizons (6), Q = num_quantiles (7)
        """
        B, T, V = features.shape
        
        # Step 1: Instance normalization
        # Per grok-scientific.md: "per sample, per feature across time"
        x = self.instance_norm(features, mask)
        
        # Step 2: Project to model dimension
        x = self.input_proj(x)  # (B, T, d_model)
        
        # Step 3: Add positional encoding
        pe = self.pos_encoding(temporal_features)  # (B, T, d_model)
        x = x + pe
        
        # Step 4: Transformer encoder
        # Convert padding mask: PyTorch expects True for positions to IGNORE
        # Our mask has 1.0 for valid, 0.0 for padded
        src_key_padding_mask = (mask == 0)  # True where padded
        
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # (B, T, d_model)
        
        # Step 5: Extract last valid position for each sample
        # This is the prediction point (end of context window)
        h = self._extract_last_valid(x, mask)  # (B, d_model)
        
        # Step 6: Output normalization
        h = self.output_norm(h)
        
        # Step 7: Multi-horizon prediction
        output = self.heads(h, targets, teacher_forcing)  # (B, 6, 7)
        
        return output
    
    def _extract_last_valid(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract the hidden state at the last valid position for each sample.
        
        The last valid position is the prediction point - the end of the
        context window where we make forward predictions.
        
        Args:
            x: Transformer output of shape (B, T, d_model).
                Type: torch.Tensor
            mask: Attention mask of shape (B, T).
                Type: torch.Tensor
                1.0 = valid, 0.0 = padded
        
        Returns:
            Hidden states at last valid positions, shape (B, d_model).
            Type: torch.Tensor
        """
        B = x.shape[0]
        
        # Find last valid index for each sample
        # mask.sum(dim=1) gives count of valid positions
        # Subtract 1 for 0-indexed
        last_valid_idx = mask.sum(dim=1).long() - 1  # (B,)
        
        # Clamp to valid range (safety for edge cases)
        last_valid_idx = last_valid_idx.clamp(min=0, max=x.shape[1] - 1)
        
        # Extract: use advanced indexing
        batch_indices = torch.arange(B, device=x.device)
        h = x[batch_indices, last_valid_idx]  # (B, d_model)
        
        return h
    
    def get_num_parameters(self) -> int:
        """
        Count total number of trainable parameters.
        
        Returns:
            Total parameter count.
            Type: int
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_vram_mb(self, batch_size: int = 8, seq_len: int = 7000) -> float:
        """
        Estimate VRAM usage for training.
        
        This is a rough estimate based on:
        - Model parameters
        - Activations for forward pass
        - Gradients for backward pass
        - Optimizer states (Adam: 2x parameters)
        
        Args:
            batch_size: Training batch size.
                Type: int
            seq_len: Sequence length.
                Type: int
        
        Returns:
            Estimated VRAM in MB.
            Type: float
        """
        # Parameter memory
        param_bytes = sum(p.numel() * 4 for p in self.parameters())  # FP32
        
        # Activation memory (rough estimate)
        # Main activations: (B, T, d_model) * num_layers * 2 (forward + attention)
        activation_bytes = (
            batch_size * seq_len * self.d_model * 4 *  # FP32
            self.num_layers * 2
        )
        
        # Gradient memory (same as parameters)
        gradient_bytes = param_bytes
        
        # Optimizer states (Adam: m and v for each parameter)
        optimizer_bytes = param_bytes * 2
        
        # Total with some overhead
        total_bytes = (param_bytes + activation_bytes + gradient_bytes + optimizer_bytes) * 1.2
        
        return total_bytes / (1024 * 1024)


def create_model(
    num_features: int = 24,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    ffn_dim: int = 2048,
    dropout: float = 0.1,
) -> BaselineTransformer:
    """
    Create a Phase 2 baseline Transformer model.
    
    Convenience function with default hyperparameters per claude-engineering.md.
    
    Args:
        num_features: Number of input features.
            Type: int
            Default: 24
        d_model: Model dimension.
            Type: int
            Default: 512
        num_heads: Number of attention heads.
            Type: int
            Default: 8
        num_layers: Number of encoder layers.
            Type: int
            Default: 6
        ffn_dim: Feedforward dimension.
            Type: int
            Default: 2048
        dropout: Dropout probability.
            Type: float
            Default: 0.1
    
    Returns:
        Initialized BaselineTransformer model.
        Type: BaselineTransformer
    """
    model = BaselineTransformer(
        num_features=num_features,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
    )
    
    num_params = model.get_num_parameters()
    vram_est = model.estimate_vram_mb()
    
    logger.info(f"Created model with {num_params:,} parameters")
    logger.info(f"Estimated VRAM usage: {vram_est:.0f} MB")
    
    return model
