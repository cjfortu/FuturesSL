"""
Complete MIGT-TVDT hybrid model for distributional NQ futures forecasting.

Orchestrates all components into the full architecture:
1. RevIN normalization (regime adaptation)
2. Variable embedding + positional encoding
3. Temporal attention (per-variable) + aggregation
4. Variable attention (cross-variable) + gating
5. Multi-horizon quantile output

Per scientific document Section 4.1-4.2, this implements:
- H8: TSA + variable embeddings for feature interactions
- H9: Gated normalization for cross-regime generalization
- H10: Quantile outputs for uncertainty quantification
- H11: Composite embeddings for cyclical patterns
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from pathlib import Path
import yaml

from .embeddings import InputEmbedding
from .temporal_attention import TemporalAttentionBlock, TemporalAggregation
from .variable_attention import VariableAttentionBlock
from .gated_instance_norm import RevIN, GatedInstanceNorm
from .quantile_heads import MultiHorizonQuantileHead


class MIGT_TVDT(nn.Module):
    """
    Hybrid MIGT-TVDT model for distributional NQ futures forecasting.
    
    Architecture flow:
        Input: (B, T, V) raw features, T=288 padded, V=24 variables
        
        1. RevIN normalize: (B, T, V) -> (B, T, V) normalized
        2. Variable embed + positional: (B, T, V) -> (B, T, V, D)
        3. Temporal attention stack: (B, T, V, D) -> (B, T, V, D)
        4. Temporal aggregation: (B, T, V, D) -> (B, V, D)
        5. Variable attention + gating: (B, V, D) -> (B, V, D)
        6. Pool variables: (B, V, D) -> (B, V*D) -> (B, D)
        7. Quantile heads: (B, D) -> (B, H, Q) for H=5 horizons, Q=7 quantiles
        
    Output: Dictionary with 'quantiles' (B, 5, 7) and 'norm_stats' for
    optional denormalization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MIGT-TVDT model.
        
        Args:
            config: Model configuration dictionary
                Type: Dict[str, Any]
                Required keys:
                    d_model: int (256)
                    n_heads: int (8)
                    n_temporal_layers: int (4)
                    n_variable_layers: int (2)
                    d_ff: int (1024)
                    dropout: float (0.1)
                    max_seq_len: int (288)
                    n_variables: int (24)
                    n_horizons: int (5)
                    n_quantiles: int (7)
                    positional_encoding: dict with time_of_day, day_of_week, etc.
        """
        super().__init__()
        self.config = config
        
        # Extract config values
        d_model = config['d_model']
        n_heads = config['n_heads']
        n_temporal_layers = config['n_temporal_layers']
        n_variable_layers = config['n_variable_layers']
        d_ff = config['d_ff']
        dropout = config['dropout']
        n_variables = config['n_variables']
        n_horizons = config['n_horizons']
        n_quantiles = config['n_quantiles']
        
        # 1. RevIN for input normalization
        self.revin = RevIN(n_variables)
        
        # 2. Input embedding (variable projection + positional)
        self.input_embedding = InputEmbedding(
            n_variables=n_variables,
            d_model=d_model,
            positional_config=config['positional_encoding'],
            dropout=dropout
        )
        
        # 3. Temporal attention stack
        self.temporal_layers = nn.ModuleList([
            TemporalAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_temporal_layers)
        ])
        
        # 4. Temporal aggregation
        self.temporal_aggregation = TemporalAggregation(
            d_model=d_model,
            n_heads=1
        )
        
        # 5. Variable attention stack with gating
        self.variable_layers = nn.ModuleList([
            VariableAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_variable_layers)
        ])
        
        self.gated_norms = nn.ModuleList([
            GatedInstanceNorm(d_model)
            for _ in range(n_variable_layers)
        ])
        
        # 6. Output pooling: flatten variables and project
        self.output_pool = nn.Sequential(
            nn.Linear(n_variables * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 7. Multi-horizon quantile heads
        self.quantile_head = MultiHorizonQuantileHead(
            d_model=d_model,
            n_horizons=n_horizons,
            n_quantiles=n_quantiles,
            hidden_dim=128
        )
        
    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        temporal_info: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            features: Input feature tensor
                Type: torch.Tensor (float32)
                Shape: (B, T, V) where B=batch, T=288, V=24
            attention_mask: Boolean mask for valid positions
                Type: torch.Tensor (bool)
                Shape: (B, T) where True=valid, False=padding
            temporal_info: Dictionary with temporal indices for positional encoding
                Type: Dict[str, torch.Tensor]
                Required keys:
                    bar_in_day: (B, T) int32, values 0-287
                    day_of_week: (B,) int32, values 0-4
                    day_of_month: (B,) int32, values 1-31
                    day_of_year: (B,) int32, values 1-366
                    
        Returns:
            Output dictionary
                Type: Dict[str, torch.Tensor]
                Keys:
                    quantiles: (B, H, Q) = (B, 5, 7) quantile predictions
                    norm_stats: dict with 'mean' and 'std' from RevIN
        """
        # 1. RevIN normalize
        x = self.revin(features, mode="normalize")
        
        # 2. Variable embedding + positional encoding
        # (B, T, V) -> (B, T, V, D)
        x = self.input_embedding(x, temporal_info)
        
        # 3. Temporal attention per variable
        for layer in self.temporal_layers:
            x = layer(x, attention_mask)
            
        # 4. Aggregate time dimension
        # (B, T, V, D) -> (B, V, D)
        x = self.temporal_aggregation(x, attention_mask)
        
        # 5. Variable attention with gating
        for layer, gated_norm in zip(self.variable_layers, self.gated_norms):
            attn_out = layer(x)
            x = gated_norm(x, attn_out)
            
        # 6. Pool variables
        # (B, V, D) -> (B, V*D) -> (B, D)
        x = x.flatten(start_dim=1)
        x = self.output_pool(x)
        
        # 7. Predict quantiles
        # (B, D) -> (B, H, Q)
        quantiles = self.quantile_head(x)
        
        return {
            'quantiles': quantiles,
            'norm_stats': self.revin.get_stats()
        }
    
    @classmethod
    def from_config(cls, config_path: Path) -> 'MIGT_TVDT':
        """
        Load model from YAML configuration file.
        
        Args:
            config_path: Path to model_config.yaml
                Type: Path or str
                
        Returns:
            Initialized model
                Type: MIGT_TVDT
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        return cls(config['model'])
    
    def count_parameters(self) -> int:
        """
        Return total number of trainable parameters.
        
        Returns:
            Parameter count
                Type: int
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_parameters_by_component(self) -> Dict[str, int]:
        """
        Return parameter count broken down by component.
        
        Returns:
            Dictionary mapping component names to parameter counts
                Type: Dict[str, int]
        """
        counts = {}
        
        counts['revin'] = sum(p.numel() for p in self.revin.parameters())
        counts['input_embedding'] = sum(p.numel() for p in self.input_embedding.parameters())
        counts['temporal_layers'] = sum(p.numel() for p in self.temporal_layers.parameters())
        counts['temporal_aggregation'] = sum(p.numel() for p in self.temporal_aggregation.parameters())
        counts['variable_layers'] = sum(p.numel() for p in self.variable_layers.parameters())
        counts['gated_norms'] = sum(p.numel() for p in self.gated_norms.parameters())
        counts['output_pool'] = sum(p.numel() for p in self.output_pool.parameters())
        counts['quantile_head'] = sum(p.numel() for p in self.quantile_head.parameters())
        counts['total'] = self.count_parameters()
        
        return counts
