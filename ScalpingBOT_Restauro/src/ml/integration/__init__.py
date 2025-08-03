"""
Integration Module - ML Integration with Analyzer System
========================================================

Integration of ML optimizations with the main Analyzer system.

BIBBIA COMPLIANCE: Only AdaptiveTrainer is used for ALL ML training.
All EnhancedLSTMTrainer and create_enhanced_* functions have been removed.

Classes:
- AlgorithmBridge: Bridge between algorithms and ML system

Author: ScalpingBOT Team
Version: 2.0.0 - BIBBIA Compliant
"""

from .algorithm_bridge import (
    AlgorithmBridge,
    create_algorithm_bridge
)

__all__ = [
    # Algorithm Bridge
    'AlgorithmBridge',
    'create_algorithm_bridge'
]