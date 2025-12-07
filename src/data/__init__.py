"""
NQ Futures Data Module
======================

This module handles data acquisition and feature engineering for the
NQ futures prediction model.

Components:
    acquisition: Databento API client for downloading historical data
    features: Technical indicator computation and feature engineering

Key Classes:
    NQDataAcquisition: Downloads and validates OHLCV data from Databento
    FeatureEngineer: Computes 24 derived features per grok-scientific.md

Usage:
    >>> from src.data.acquisition import NQDataAcquisition
    >>> from src.data.features import FeatureEngineer, compute_and_save_features
"""

from .acquisition import (
    NQDataAcquisition,
    DatabentoConfig,
    download_full_dataset,
)
from .features import (
    FeatureEngineer,
    compute_and_save_features,
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    TARGET_HORIZONS,
)

__all__ = [
    'NQDataAcquisition',
    'DatabentoConfig',
    'download_full_dataset',
    'FeatureEngineer',
    'compute_and_save_features',
    'FEATURE_COLUMNS',
    'FEATURE_GROUPS',
    'TARGET_HORIZONS',
]
