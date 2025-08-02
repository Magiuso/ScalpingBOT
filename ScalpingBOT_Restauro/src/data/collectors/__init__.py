"""
Collectors Module - Data Collection and Aggregation
===================================================

Sistema di raccolta dati con aggregazioni real-time e thread-safety.

Classes:
- TickCollector: Raccolta tick con aggregazioni temporali multiple

Author: ScalpingBOT Team
Version: 1.0.0
"""

from .tick_collector import TickCollector, create_tick_collector

__all__ = [
    'TickCollector',
    'create_tick_collector'
]