"""
Training Module - Adaptive Training Algorithms
==============================================

Adaptive training algorithms and optimization for ML models.

Classes:
- AdaptiveTrainer: Intelligent training management with automatic optimization
- TrainingConfig: Configuration for adaptive training
- LossTracker: Training loss monitoring and stability tracking
- AdaptiveLRScheduler: Learning rate scheduling with multiple strategies

Author: ScalpingBOT Team
Version: 1.0.0
"""

from .adaptive_trainer import (
    AdaptiveTrainer,
    TrainingConfig,
    LossTracker,
    AdaptiveLRScheduler,
    create_adaptive_trainer_config
)

__all__ = [
    'AdaptiveTrainer',
    'TrainingConfig',
    'LossTracker', 
    'AdaptiveLRScheduler',
    'create_adaptive_trainer_config'
]