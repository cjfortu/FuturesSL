"""
Input embedding modules for the MIGT-TVDT architecture.

Implements variable embedding following the iTransformer paradigm:
- Each variable's time series is projected independently to model dimension
- Preserves time dimension for subsequent temporal attention
- Output is 4D: (B, T, V, D) enabling per-variable temporal processing

Per scientific document Section 4.1, variable embedding treats each feature
as a separate "token" with its own projection, allowing the model to learn
variable-specific representations before cross-variable interaction.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from .positional_encodings import CompositePositionalEncoding


class VariableEmbedding(nn.Module):
    """
    Project each variable's time series to model dimension.
    
    Unlike standard transformers that embed timesteps, this embeds variables:
    - Input (B, T, V): V variables, each with T timesteps
    - Output (B, T, V, D): Each variable has its own D-dimensional representation
    
    This preserves temporal structure within each variable while enabling
    the model to learn variable-specific patterns. Each variable gets its
    own linear projection (not shared), allowing different features to be
    processed differently.
    """
    
    def __init__(
        self,
        n_variables: int,
        d_model: int,
        dropout: float = 0.1
    ):
        """
        Initialize variable embedding.
        
        Args:
            n_variables: Number of input variables (features)
                Type: int
                Value: 24 for our feature set (4 OHLC + 20 derived)
            d_model: Model/embedding dimension
                Type: int
                Typical value: 256
            dropout: Dropout probability
                Type: float
                Typical value: 0.1
        """
        super().__init__()
        self.n_variables = n_variables
        self.d_model = d_model
        
        # Separate linear projection for each variable
        # This allows different features to be embedded differently
        # Shape transformation: (B, T, 1) -> (B, T, D) per variable
        self.projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_variables)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed each variable independently.
        
        Each variable's time series is projected through its own linear layer,
        resulting in variable-specific embeddings that preserve temporal order.
        
        Args:
            x: Input features
                Type: torch.Tensor (float32)
                Shape: (B, T, V) where B=batch, T=288, V=24
                
        Returns:
            Variable embeddings
                Type: torch.Tensor (float32)
                Shape: (B, T, V, D) where D=d_model
        """
        B, T, V = x.shape
        
        # Project each variable separately
        # x[:, :, v] has shape (B, T), needs (B, T, 1) for linear
        embedded = []
        for v in range(V):
            var_input = x[:, :, v:v+1]  # (B, T, 1)
            var_embedded = self.projections[v](var_input)  # (B, T, D)
            embedded.append(var_embedded)
            
        # Stack along variable dimension: list of (B, T, D) -> (B, T, V, D)
        output = torch.stack(embedded, dim=2)
        
        return self.dropout(output)


class InputEmbedding(nn.Module):
    """
    Complete input embedding combining variable projection and positional encoding.
    
    CRITICAL BROADCASTING NOTE (from engineering spec):
    Variable embedding produces 4D tensor (B, T, V, D). Positional encodings
    produce (B, T, D) and must be unsqueezed for proper broadcasting:
    - Positional: (B, T, D) -> unsqueeze(2) -> (B, T, 1, D)
    - Addition: (B, T, V, D) + (B, T, 1, D) broadcasts across V dimension
    
    This ensures all variables receive the same positional information while
    maintaining their distinct learned representations.
    """
    
    def __init__(
        self,
        n_variables: int,
        d_model: int,
        positional_config: Dict[str, Any],
        dropout: float = 0.1
    ):
        """
        Initialize input embedding.
        
        Args:
            n_variables: Number of input variables
                Type: int
                Value: 24
            d_model: Model dimension
                Type: int
                Typical value: 256
            positional_config: Configuration for positional encodings
                Type: Dict[str, Any]
                Required keys: time_of_day, day_of_week, day_of_month, day_of_year
                Each with 'dim' key specifying encoding dimension
            dropout: Dropout probability
                Type: float
                Typical value: 0.1
        """
        super().__init__()
        
        # Variable embedding
        self.variable_embed = VariableEmbedding(
            n_variables=n_variables,
            d_model=d_model,
            dropout=0.0  # Dropout applied after positional encoding
        )
        
        # Add d_model to positional config for projection
        pos_config = positional_config.copy()
        pos_config['d_model'] = d_model
        
        # Composite positional encoding
        self.positional_encoding = CompositePositionalEncoding(pos_config)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        features: torch.Tensor,
        temporal_info: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Embed features with positional information.
        
        Steps:
        1. Project each variable to d_model: (B, T, V) -> (B, T, V, D)
        2. Compute positional encoding: (B, T, D)
        3. Unsqueeze positional for broadcasting: (B, T, D) -> (B, T, 1, D)
        4. Add: (B, T, V, D) + (B, T, 1, D) -> (B, T, V, D)
        
        Args:
            features: Input feature tensor
                Type: torch.Tensor (float32)
                Shape: (B, T, V) where T=288, V=24
            temporal_info: Dictionary with temporal indices
                Type: Dict[str, torch.Tensor]
                Required keys:
                    bar_in_day: (B, T) int32, values 0-287
                    day_of_week: (B,) int32, values 0-4
                    day_of_month: (B,) int32, values 1-31
                    day_of_year: (B,) int32, values 1-366
                    
        Returns:
            Embedded input with positional information
                Type: torch.Tensor (float32)
                Shape: (B, T, V, D)
        """
        # Variable embedding: (B, T, V) -> (B, T, V, D)
        x = self.variable_embed(features)
        
        # Positional encoding: (B, T, D)
        pos_enc = self.positional_encoding(
            bar_in_day=temporal_info['bar_in_day'],
            day_of_week=temporal_info['day_of_week'],
            day_of_month=temporal_info['day_of_month'],
            day_of_year=temporal_info['day_of_year']
        )
        
        # CRITICAL: Unsqueeze for broadcasting to 4D
        # (B, T, D) -> (B, T, 1, D) broadcasts across V dimension
        pos_enc = pos_enc.unsqueeze(2)
        
        # Add positional encoding (broadcasts across variables)
        x = x + pos_enc
        
        return self.dropout(x)
