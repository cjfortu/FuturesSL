"""
Training pipeline modules for the MIGT-TVDT architecture.

This package provides:
- loss_functions: Pinball loss for quantile regression
- scheduler: Learning rate scheduling with warmup
- trainer: Training orchestrator with mixed precision support
"""

from .loss_functions import PinballLoss, QuantileCrossingPenalty, CombinedQuantileLoss
from .scheduler import WarmupCosineScheduler
from .trainer import Trainer

__all__ = [
    'PinballLoss',
    'QuantileCrossingPenalty',
    'CombinedQuantileLoss',
    'WarmupCosineScheduler',
    'Trainer'
]
