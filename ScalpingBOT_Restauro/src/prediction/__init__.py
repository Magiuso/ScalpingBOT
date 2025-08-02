#!/usr/bin/env python3
"""
Prediction Systems - FASE 6 PREDICTION & ORCHESTRATION
REGOLE CLAUDE_RESTAURO.md APPLICATE:
- ✅ Zero fallback/defaults
- ✅ Fail fast error handling
- ✅ No debug prints/spam
- ✅ No test code embedded
- ✅ Modular architecture

Sistema di predizione centralizzato che orchestra tutti i moduli migrati.
"""

from .core import (
    AssetAnalyzer,
    create_asset_analyzer,
    AdvancedMarketAnalyzer,
    create_advanced_market_analyzer
)
from .unified_system import (
    UnifiedAnalyzerSystem,
    SystemMode,
    create_unified_system,
    create_production_system,
    create_testing_system
)

__all__ = [
    'AssetAnalyzer',
    'create_asset_analyzer', 
    'AdvancedMarketAnalyzer',
    'create_advanced_market_analyzer',
    'UnifiedAnalyzerSystem',
    'SystemMode',
    'create_unified_system',
    'create_production_system',
    'create_testing_system'
]