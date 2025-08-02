"""
Preprocessing Module - Data Preprocessing and Normalization
==========================================================

Advanced data preprocessing with intelligent normalization and outlier detection.

Classes:
- AdvancedDataPreprocessor: Smart normalization and outlier handling
- PreprocessingConfig: Configuration for preprocessing operations
- DataStabilityTracker: Data drift and stability monitoring

Author: ScalpingBOT Team
Version: 1.0.0
"""

from .data_preprocessing import (
    AdvancedDataPreprocessor,
    PreprocessingConfig,
    DataStabilityTracker,
    create_optimized_preprocessor
)

__all__ = [
    'AdvancedDataPreprocessor',
    'PreprocessingConfig',
    'DataStabilityTracker',
    'create_optimized_preprocessor'
]