"""
Processors Module - Data Processing and Feature Engineering
==========================================================

Sistema di elaborazione dati con feature engineering avanzato.

Classes:
- MarketDataProcessor: Elaborazione dati di mercato e calcolo indicatori

Author: ScalpingBOT Team
Version: 1.0.0
"""

from .market_data_processor import MarketDataProcessor, create_market_data_processor

__all__ = [
    'MarketDataProcessor',
    'create_market_data_processor'
]