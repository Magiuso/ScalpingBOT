#!/usr/bin/env python3
"""
Shared Components - CONSOLIDATED MODULE
REGOLE CLAUDE_RESTAURO.md APPLICATE:
- ✅ Zero duplications
- ✅ Centralized shared resources
- ✅ Clean module structure

Modulo per componenti condivisi tra tutti i moduli del sistema.
"""

from .enums import ModelType, OptimizationProfile
from .exceptions import (
    InsufficientDataError,
    ModelNotInitializedError,
    InvalidInputError,
    PredictionError,
    ConfigurationError,
    DataValidationError,
    AlgorithmExecutionError,
    CompetitionSystemError,
    MLTrainingError,
    TensorShapeError,
    AlgorithmErrors,
    SystemErrors,
    MLErrors,
    create_algorithm_error
)

__all__ = [
    'ModelType',
    'OptimizationProfile',
    # Exception classes
    'InsufficientDataError',
    'ModelNotInitializedError',
    'InvalidInputError',
    'PredictionError',
    'ConfigurationError',
    'DataValidationError',
    'AlgorithmExecutionError',
    'CompetitionSystemError',
    'MLTrainingError',
    'TensorShapeError',
    # Exception collections
    'AlgorithmErrors',
    'SystemErrors',
    'MLErrors',
    # Utilities
    'create_algorithm_error'
]