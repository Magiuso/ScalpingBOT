#!/usr/bin/env python3
"""
Shared Enums - CONSOLIDATED AND CENTRALIZED
REGOLE CLAUDE_RESTAURO.md APPLICATE:
- ✅ Zero duplications  
- ✅ Single source of truth
- ✅ No redundant definitions
- ✅ Centralized enum management

Enum condivisi per evitare duplicazioni tra moduli.
Consolidato da base_models.py e analyzer_ml_integration.py.
"""

from enum import Enum


class ModelType(Enum):
    """Model types - CONSOLIDATED from base_models.py and analyzer_ml_integration.py"""
    SUPPORT_RESISTANCE = "support_resistance"
    PATTERN_RECOGNITION = "pattern_recognition"
    BIAS_DETECTION = "bias_detection"
    TREND_ANALYSIS = "trend_analysis"
    VOLATILITY_PREDICTION = "volatility_prediction"


class OptimizationProfile(Enum):
    """Optimization profiles - CONSOLIDATED from base_models.py and analyzer_ml_integration.py"""
    HIGH_PERFORMANCE = "high_performance"
    STABLE_TRAINING = "stable_training"
    RESEARCH_MODE = "research_mode"
    PRODUCTION_READY = "production_ready"


# Export all enums
__all__ = [
    'ModelType',
    'OptimizationProfile'
]