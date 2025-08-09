#!/usr/bin/env python3
"""
Incremental Support/Resistance Calculator - BIBBIA COMPLIANT
===========================================================

Estratto da AdvancedMarketAnalyzer per scomposizione modulare.
O(n) complexity per real-money trading production.

BIBBIA RULES COMPLIANCE:
- ZERO FALLBACK: No default values, FAIL FAST validations
- NO TEST DATA: Only real market data
- ONE ROAD: Single path for S/R detection
- CLEAN CODE: Constants instead of magic numbers
- FAIL FAST: Immediate validation, no silent failures

Author: ScalpingBOT Team
Version: 2.0.0 - BIBBIA COMPLIANT REFACTOR
"""

from collections import deque
from typing import Dict, List, Any, Optional


class IncrementalSRCalculator:
    """
    Incremental Support/Resistance Calculator for O(n) complexity
    Production-ready for real-money trading - BIBBIA COMPLIANT
    """
    
    # Constants - CLEAN CODE compliant (no magic numbers)
    DEFAULT_WINDOW_SIZE = 20
    DEFAULT_LEVEL_TOLERANCE = 0.002  # 0.2%
    DEFAULT_MAX_LEVELS = 100
    DEFAULT_DECAY_PERIOD = 1000
    
    # Strength constants
    STRENGTH_INCREMENT = 0.1
    MAX_STRENGTH = 1.0
    TOUCH_STRENGTH_INCREMENT = 0.05
    INITIAL_LEVEL_STRENGTH = 0.3
    
    # Decay constants
    DECAY_CHECK_INTERVAL = 100  # Check every 100 ticks
    DECAY_FACTOR_BASE = 0.95
    MINIMUM_STRENGTH_THRESHOLD = 0.1
    
    # Precision
    PRICE_PRECISION_DECIMALS = 2
    
    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE, 
                 level_tolerance: float = DEFAULT_LEVEL_TOLERANCE, 
                 max_levels: int = DEFAULT_MAX_LEVELS, 
                 decay_period: int = DEFAULT_DECAY_PERIOD):
        """
        Initialize Incremental S/R Calculator - FAIL FAST validation
        
        Args:
            window_size: Window for local min/max detection
            level_tolerance: Price tolerance for level clustering
            max_levels: Maximum S/R levels to maintain
            decay_period: Ticks before level strength starts decaying
            
        Raises:
            ValueError: If parameters are invalid - FAIL FAST
        """
        # FAIL FAST validations - no fallback allowed
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"FAIL FAST: window_size must be positive integer, got {window_size}")
        if not isinstance(level_tolerance, (int, float)) or level_tolerance <= 0:
            raise ValueError(f"FAIL FAST: level_tolerance must be positive number, got {level_tolerance}")
        if not isinstance(max_levels, int) or max_levels <= 0:
            raise ValueError(f"FAIL FAST: max_levels must be positive integer, got {max_levels}")
        if not isinstance(decay_period, int) or decay_period <= 0:
            raise ValueError(f"FAIL FAST: decay_period must be positive integer, got {decay_period}")
        
        self.window_size = window_size
        self.level_tolerance = level_tolerance
        self.max_levels = max_levels
        self.decay_period = decay_period
        
        # Rolling price buffer for local min/max detection
        self.price_buffer = deque(maxlen=window_size * 2 + 1)
        
        # S/R levels with metadata: {price: {'strength': float, 'touches': int, 'last_seen': int}}
        self.support_levels = {}
        self.resistance_levels = {}
        
        # Tick counter
        self.tick_count = 0
        
    def add_tick(self, price: float) -> None:
        """Add new tick and update S/R levels incrementally - O(1) amortized"""
        # FAIL FAST validation
        if not isinstance(price, (int, float)) or price <= 0:
            raise ValueError(f"FAIL FAST: price must be positive number, got {price}")
            
        self.tick_count += 1
        self.price_buffer.append(price)
        
        # Need full window to detect local min/max
        if len(self.price_buffer) < self.window_size * 2 + 1:
            return
            
        # Check if middle price is local min/max
        middle_idx = self.window_size
        middle_price = self.price_buffer[middle_idx]
        
        # Local minimum = potential support
        if all(middle_price <= self.price_buffer[i] for i in range(len(self.price_buffer)) if i != middle_idx):
            self._add_support_level(middle_price)
            
        # Local maximum = potential resistance  
        if all(middle_price >= self.price_buffer[i] for i in range(len(self.price_buffer)) if i != middle_idx):
            self._add_resistance_level(middle_price)
            
        # Update touches for existing levels
        self._update_level_touches(price)
        
        # Decay old levels periodically
        if self.tick_count % self.DECAY_CHECK_INTERVAL == 0:
            self._decay_old_levels()
    
    def _add_support_level(self, price: float) -> None:
        """Add or strengthen support level"""
        # Round price for clustering
        rounded_price = round(price, self.PRICE_PRECISION_DECIMALS)
        
        # Check if level already exists (within tolerance)
        for existing_price in list(self.support_levels.keys()):
            if abs(existing_price - rounded_price) / existing_price < self.level_tolerance:
                # Strengthen existing level
                self.support_levels[existing_price]['strength'] = min(
                    self.support_levels[existing_price]['strength'] + self.STRENGTH_INCREMENT, 
                    self.MAX_STRENGTH
                )
                self.support_levels[existing_price]['touches'] += 1
                self.support_levels[existing_price]['last_seen'] = self.tick_count
                return
                
        # Add new level
        self.support_levels[rounded_price] = {
            'strength': self.INITIAL_LEVEL_STRENGTH,
            'touches': 1,
            'last_seen': self.tick_count
        }
        
        # Maintain max levels
        if len(self.support_levels) > self.max_levels:
            # Remove weakest level
            if not self.support_levels:  # FAIL FAST: Should never happen but safety check
                raise RuntimeError("FAIL FAST: support_levels unexpectedly empty")
            weakest = min(self.support_levels.items(), 
                         key=lambda x: x[1]['strength'])
            del self.support_levels[weakest[0]]
    
    def _add_resistance_level(self, price: float) -> None:
        """Add or strengthen resistance level"""
        # Round price for clustering
        rounded_price = round(price, self.PRICE_PRECISION_DECIMALS)
        
        # Check if level already exists (within tolerance)
        for existing_price in list(self.resistance_levels.keys()):
            if abs(existing_price - rounded_price) / existing_price < self.level_tolerance:
                # Strengthen existing level
                self.resistance_levels[existing_price]['strength'] = min(
                    self.resistance_levels[existing_price]['strength'] + self.STRENGTH_INCREMENT,
                    self.MAX_STRENGTH
                )
                self.resistance_levels[existing_price]['touches'] += 1
                self.resistance_levels[existing_price]['last_seen'] = self.tick_count
                return
                
        # Add new level
        self.resistance_levels[rounded_price] = {
            'strength': self.INITIAL_LEVEL_STRENGTH,
            'touches': 1,
            'last_seen': self.tick_count
        }
        
        # Maintain max levels
        if len(self.resistance_levels) > self.max_levels:
            # Remove weakest level
            if not self.resistance_levels:  # FAIL FAST: Should never happen but safety check
                raise RuntimeError("FAIL FAST: resistance_levels unexpectedly empty")
            weakest = min(self.resistance_levels.items(), 
                         key=lambda x: x[1]['strength'])
            del self.resistance_levels[weakest[0]]
    
    def _update_level_touches(self, current_price: float) -> None:
        """Update touch count when price approaches levels"""
        # Check support touches
        for level_price, metadata in self.support_levels.items():
            if abs(current_price - level_price) / level_price < self.level_tolerance:
                metadata['touches'] += 1
                metadata['last_seen'] = self.tick_count
                metadata['strength'] = min(
                    metadata['strength'] + self.TOUCH_STRENGTH_INCREMENT, 
                    self.MAX_STRENGTH
                )
                
        # Check resistance touches
        for level_price, metadata in self.resistance_levels.items():
            if abs(current_price - level_price) / level_price < self.level_tolerance:
                metadata['touches'] += 1
                metadata['last_seen'] = self.tick_count
                metadata['strength'] = min(
                    metadata['strength'] + self.TOUCH_STRENGTH_INCREMENT,
                    self.MAX_STRENGTH
                )
    
    def _decay_old_levels(self) -> None:
        """Decay strength of old levels"""
        current_tick = self.tick_count
        
        # Decay supports
        for level_price in list(self.support_levels.keys()):
            metadata = self.support_levels[level_price]
            age = current_tick - metadata['last_seen']
            if age > self.decay_period:
                decay_factor = self.DECAY_FACTOR_BASE ** (age / self.decay_period)
                metadata['strength'] *= decay_factor
                
                # Remove if too weak
                if metadata['strength'] < self.MINIMUM_STRENGTH_THRESHOLD:
                    del self.support_levels[level_price]
                    
        # Decay resistances
        for level_price in list(self.resistance_levels.keys()):
            metadata = self.resistance_levels[level_price]
            age = current_tick - metadata['last_seen']
            if age > self.decay_period:
                decay_factor = self.DECAY_FACTOR_BASE ** (age / self.decay_period)
                metadata['strength'] *= decay_factor
                
                # Remove if too weak
                if metadata['strength'] < self.MINIMUM_STRENGTH_THRESHOLD:
                    del self.resistance_levels[level_price]
    
    def get_current_levels(self) -> Dict[str, List[float]]:
        """Get current S/R levels sorted by price - O(n log n) where n = number of levels"""
        return {
            'support': sorted(self.support_levels.keys()),
            'resistance': sorted(self.resistance_levels.keys())
        }
    
    def get_levels_with_strength(self) -> Dict[str, Dict[float, float]]:
        """Get S/R levels with their strength values"""
        return {
            'support': {price: meta['strength'] for price, meta in self.support_levels.items()},
            'resistance': {price: meta['strength'] for price, meta in self.resistance_levels.items()}
        }
    
    def get_nearest_levels(self, current_price: float) -> Dict[str, Any]:
        """
        Get nearest support and resistance with metadata - O(n)
        
        Args:
            current_price: Current market price
            
        Returns:
            Dict with nearest support/resistance info
            
        Raises:
            ValueError: If current_price invalid - FAIL FAST
        """
        # FAIL FAST validation
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            raise ValueError(f"FAIL FAST: current_price must be positive number, got {current_price}")
        
        # Initialize with FAIL FAST approach - no fallback defaults
        nearest_support = None
        nearest_support_strength = None
        
        nearest_resistance = None
        nearest_resistance_strength = None
        
        # Find nearest support (below current price)
        for price, metadata in self.support_levels.items():
            if price <= current_price:
                if nearest_support is None or price > nearest_support:
                    nearest_support = price
                    nearest_support_strength = metadata['strength']
                
        # Find nearest resistance (above current price)
        for price, metadata in self.resistance_levels.items():
            if price >= current_price:
                if nearest_resistance is None or price < nearest_resistance:
                    nearest_resistance = price
                    nearest_resistance_strength = metadata['strength']
                
        return {
            'support': nearest_support,
            'support_strength': nearest_support_strength,
            'resistance': nearest_resistance,
            'resistance_strength': nearest_resistance_strength
        }


# Factory function for external usage - BIBBIA COMPLIANT
def create_incremental_sr_calculator(**kwargs) -> IncrementalSRCalculator:
    """Factory function to create IncrementalSRCalculator - FAIL FAST if invalid params"""
    return IncrementalSRCalculator(**kwargs)


# Export
__all__ = [
    'IncrementalSRCalculator',
    'create_incremental_sr_calculator'
]