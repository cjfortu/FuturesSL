"""
Model package for MIGT-TVDT hybrid architecture.

This package implements the distributional forecasting model for NQ futures
as specified in the scientific document (grok-scientific.md). The architecture
combines ideas from RL-TVDT (two-stage attention) and MIGT (gated instance
normalization) for robust multi-horizon quantile prediction.

Architecture Overview:
    1. RevIN normalization for regime adaptation
    2. Variable embedding with composite positional encodings
    3. Temporal attention (per-variable) + aggregation
    4. Variable attention (cross-variable) + gating
    5. Multi-horizon quantile output heads

Modules:
    positional_encodings: Time-of-day, day-of-week, Time2Vec encodings
    embeddings: Variable projection and input embedding
    temporal_attention: Per-variable temporal self-attention
    variable_attention: Cross-variable attention
    gated_instance_norm: RevIN and LiteGateUnit
    quantile_heads: Non-crossing quantile regression heads
    migt_tvdt: Complete model orchestration
"""

from .positional_encodings import (
    TimeOfDayEncoding,
    DayOfWeekEncoding,
    Time2VecEncoding,
    CompositePositionalEncoding
)
from .embeddings import VariableEmbedding, InputEmbedding
from .temporal_attention import TemporalAttentionBlock, TemporalAggregation
from .variable_attention import VariableAttentionBlock
from .gated_instance_norm import RevIN, LiteGateUnit, GatedInstanceNorm
from .quantile_heads import QuantileHead, MultiHorizonQuantileHead
from .migt_tvdt import MIGT_TVDT

__all__ = [
    # Positional encodings
    'TimeOfDayEncoding',
    'DayOfWeekEncoding',
    'Time2VecEncoding',
    'CompositePositionalEncoding',
    # Embeddings
    'VariableEmbedding',
    'InputEmbedding',
    # Attention
    'TemporalAttentionBlock',
    'TemporalAggregation',
    'VariableAttentionBlock',
    # Normalization and gating
    'RevIN',
    'LiteGateUnit',
    'GatedInstanceNorm',
    # Output heads
    'QuantileHead',
    'MultiHorizonQuantileHead',
    # Complete model
    'MIGT_TVDT'
]
