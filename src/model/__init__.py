"""
NQ Futures Model Package
========================

This package contains the Transformer-based model components for NQ futures
prediction across multiple time horizons.

Phase 2 Components (baseline):
    BaselineTransformer: Vanilla Transformer encoder with standard attention
    InstanceNorm1d: Per-sample, per-feature normalization
    CyclicalPositionalEncoding: Time-aware positional encoding
    QuantileHead: Single-horizon quantile prediction
    IndependentMultiHorizonHead: Independent predictions per horizon

Phase 3 Components (planned):
    TSABlock: Two-Stage Attention (Temporal + Variable)
    LiteGateUnit: Noise filtering gate
    AutoregressiveMultiHorizonHead: Conditioned predictions

Usage:
    >>> from src.model import BaselineTransformer, create_model
    >>> model = create_model(num_features=24, d_model=512)
"""

from .baseline import (
    BaselineTransformer,
    InstanceNorm1d,
    CyclicalPositionalEncoding,
    QuantileHead,
    IndependentMultiHorizonHead,
    create_model,
    QUANTILES,
    NUM_QUANTILES,
    HORIZONS,
    NUM_HORIZONS,
)

__all__ = [
    # Model classes
    'BaselineTransformer',
    'InstanceNorm1d',
    'CyclicalPositionalEncoding',
    'QuantileHead',
    'IndependentMultiHorizonHead',
    # Factory functions
    'create_model',
    # Constants
    'QUANTILES',
    'NUM_QUANTILES',
    'HORIZONS',
    'NUM_HORIZONS',
]
