"""
ML Models - Machine Learning Model Definitions
==============================================

Moduli per i modelli ML del sistema ScalpingBOT.
Estratti dal monolite Analyzer.py per organizzazione modulare.

Available modules:
- base_models: ModelType, Prediction, AlgorithmPerformance
- advanced_lstm: AdvancedLSTM neural network model
- transformer_models: TransformerPredictor for advanced pattern recognition
- cnn_models: CNNPatternRecognizer for convolutional pattern detection
- competition: AlgorithmCompetition system with all supporting classes
"""

from .base_models import (
    ModelType,
    OptimizationProfile,
    Prediction,
    AlgorithmPerformance,
    create_prediction,
    create_algorithm_performance
)

from .advanced_lstm import (
    AdvancedLSTM
)

from .transformer_models import (
    TransformerPredictor
)

from .cnn_models import (
    CNNPatternRecognizer
)

from .competition import (
    ChampionPreserver,
    RealityChecker,
    EmergencyStopSystem,
    PostErrorReanalyzer,
    AlgorithmCompetition
)

__all__ = [
    # Base model types
    'ModelType',
    'OptimizationProfile',
    'Prediction',
    'AlgorithmPerformance',
    
    # Neural networks
    'AdvancedLSTM',
    'TransformerPredictor',
    'CNNPatternRecognizer',
    
    # Competition system
    'ChampionPreserver',
    'RealityChecker',
    'EmergencyStopSystem',
    'PostErrorReanalyzer',
    'AlgorithmCompetition',
    
    # Factory functions
    'create_prediction',
    'create_algorithm_performance'
]