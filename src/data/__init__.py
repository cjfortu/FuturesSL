"""
NQ Futures Data Package
=======================

This package handles data acquisition, feature engineering, and PyTorch
Dataset/DataLoader creation for the NQ futures prediction model.

Phase 1 Components:
    NQDataAcquisition: Databento API client for OHLCV data
    FeatureEngineer: Technical indicator computation (V=24 features)
    FEATURE_COLUMNS, FEATURE_GROUPS, TARGET_HORIZONS: Constants

Phase 2 Components:
    NQFuturesDataset: PyTorch Dataset with rolling window extraction
    create_dataloaders: Convenience function for train/val/test loaders
    T_MAX, BARS_PER_WEEK: Sequence length constants

Usage:
    >>> from src.data import NQFuturesDataset, create_dataloaders, FEATURE_COLUMNS
    >>> train_loader, val_loader, test_loader = create_dataloaders(
    ...     'data/features.parquet',
    ...     'data/targets.parquet',
    ...     FEATURE_COLUMNS
    ... )
"""

# Phase 1 exports
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

# Phase 2 exports
from .dataset import (
    NQFuturesDataset,
    create_dataloaders,
    collate_fn_variable_length,
    T_MAX,
    BARS_PER_WEEK,
    HORIZONS,
    NUM_QUANTILES,
    DEFAULT_TRAIN_END,
    DEFAULT_VAL_END,
)

__all__ = [
    # Phase 1: Acquisition
    'NQDataAcquisition',
    'DatabentoConfig',
    'download_full_dataset',
    # Phase 1: Features
    'FeatureEngineer',
    'compute_and_save_features',
    'FEATURE_COLUMNS',
    'FEATURE_GROUPS',
    'TARGET_HORIZONS',
    # Phase 2: Dataset
    'NQFuturesDataset',
    'create_dataloaders',
    'collate_fn_variable_length',
    'T_MAX',
    'BARS_PER_WEEK',
    'HORIZONS',
    'NUM_QUANTILES',
    'DEFAULT_TRAIN_END',
    'DEFAULT_VAL_END',
]
