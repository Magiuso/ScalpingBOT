#!/usr/bin/env python3
"""
ML Algorithms Module - Core Trading Algorithms
==============================================

Modulo contenente gli algoritmi concreti di trading estratti dal monolite.
Ogni algoritmo mantiene la logica identica all'originale.

Moduli disponibili:
- support_resistance_algorithms: 5 algoritmi S/R (PivotPoints, VolumeProfile, LSTM, Statistical, Transformer)
- pattern_recognition_algorithms: 5 algoritmi Pattern Recognition (CNN, Classical, LSTM_Sequences, Transformer, Ensemble)
- bias_detection_algorithms: 5 algoritmi Bias Detection (Sentiment_LSTM, VolumePrice, Momentum_ML, Transformer, MultiModal)
- trend_analysis_algorithms: 5 algoritmi Trend Analysis
- volatility_prediction_algorithms: 3 algoritmi Volatility
- momentum_analysis_algorithms: 3 algoritmi Momentum
"""

from .support_resistance_algorithms import (
    SupportResistanceAlgorithms,
    create_support_resistance_algorithms,
    InsufficientDataError,
    ModelNotInitializedError,
    InvalidInputError,
    PredictionError
)

from .pattern_recognition_algorithms import (
    PatternRecognitionAlgorithms,
    create_pattern_recognition_algorithms
)

from .bias_detection_algorithms import (
    BiasDetectionAlgorithms,
    create_bias_detection_algorithms
)

from .trend_analysis_algorithms import (
    TrendAnalysisAlgorithms,
    create_trend_analysis_algorithms
)

__all__ = [
    # Support/Resistance
    'SupportResistanceAlgorithms',
    'create_support_resistance_algorithms',
    
    # Pattern Recognition
    'PatternRecognitionAlgorithms',
    'create_pattern_recognition_algorithms',
    
    # Bias Detection
    'BiasDetectionAlgorithms',
    'create_bias_detection_algorithms',
    
    # Trend Analysis
    'TrendAnalysisAlgorithms',
    'create_trend_analysis_algorithms',
    
    # Exceptions
    'InsufficientDataError',
    'ModelNotInitializedError', 
    'InvalidInputError',
    'PredictionError'
]