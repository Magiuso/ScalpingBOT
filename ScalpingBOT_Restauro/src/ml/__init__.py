"""
ML Module - Machine Learning Utilities
======================================

Advanced machine learning utilities for ScalpingBOT system.
Includes training optimization, data preprocessing, monitoring, and integration.

Modules:
- training: Adaptive training algorithms and optimization
- preprocessing: Data preprocessing and normalization
- monitoring: Training monitoring and observability
- integration: ML integration with main analyzer system

Author: ScalpingBOT Team
Version: 1.0.0
"""

# Import key classes for convenience
from .training.adaptive_trainer import AdaptiveTrainer, TrainingConfig
from .preprocessing.data_preprocessing import AdvancedDataPreprocessor, PreprocessingConfig
from .monitoring.training_monitor import TrainingMonitor, MonitorConfig
from .integration.analyzer_ml_integration import EnhancedLSTMTrainer

__all__ = [
    'AdaptiveTrainer',
    'TrainingConfig', 
    'AdvancedDataPreprocessor',
    'PreprocessingConfig',
    'TrainingMonitor',
    'MonitorConfig',
    'EnhancedLSTMTrainer'
]