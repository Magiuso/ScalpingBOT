"""
Integration Module - ML Integration with Analyzer System
========================================================

Integration of ML optimizations with the main Analyzer system.

Classes:
- EnhancedLSTMTrainer: Enhanced LSTM trainer with full optimization
- OptimizedTrainingPipeline: Training pipeline with optimization profiles
- OptimizedTrainingManager: Training management system

Factory Functions:
- create_enhanced_sr_trainer: Support/Resistance trainer
- create_enhanced_pattern_trainer: Pattern recognition trainer  
- create_enhanced_bias_trainer: Bias detection trainer

Author: ScalpingBOT Team
Version: 1.0.0
"""

from .analyzer_ml_integration import (
    EnhancedLSTMTrainer,
    OptimizedTrainingPipeline,
    OptimizedTrainingManager,
    ModelType,
    OptimizationProfile,
    create_enhanced_sr_trainer,
    create_enhanced_pattern_trainer,
    create_enhanced_bias_trainer,
    create_stable_training_pipeline
)

from .algorithm_bridge import (
    AlgorithmBridge,
    create_algorithm_bridge
)

__all__ = [
    'EnhancedLSTMTrainer',
    'OptimizedTrainingPipeline',
    'OptimizedTrainingManager',
    'ModelType',
    'OptimizationProfile',
    'create_enhanced_sr_trainer',
    'create_enhanced_pattern_trainer', 
    'create_enhanced_bias_trainer',
    'create_stable_training_pipeline',
    
    # Algorithm Bridge
    'AlgorithmBridge',
    'create_algorithm_bridge'
]