"""
Data Module - Consolidated Data Processing Infrastructure
========================================================

Modulo consolidato per gestione e elaborazione dati.
ESTRATTO e RIORGANIZZATO dal monolite Analyzer.py.
CONSOLIDATO per eliminare duplicazioni con moduli esistenti.

Components:
- collectors: Raccolta dati tick e aggregazioni real-time
- processors: Elaborazione dati e feature engineering consolidata

Note: storage e validators rimossi per eliminare duplicazioni con
StorageManager e AdvancedDataPreprocessor esistenti.

Author: ScalpingBOT Team
Version: 1.0.0
"""

from .collectors.tick_collector import TickCollector, create_tick_collector
from .processors.market_data_processor import MarketDataProcessor, create_market_data_processor

__all__ = [
    # Collectors
    'TickCollector',
    'create_tick_collector',
    
    # Processors
    'MarketDataProcessor', 
    'create_market_data_processor'
]