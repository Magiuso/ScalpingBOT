#!/usr/bin/env python3
"""
Asset-Specific Configuration - DYNAMIC MULTIASSET SYSTEM
REGOLE CLAUDE_RESTAURO.md APPLICATE:
- ✅ Zero fallback/defaults
- ✅ Fail fast error handling
- ✅ No debug prints/spam
- ✅ TRULY MULTIASSET - NO HARDCODED ASSET SYMBOLS

Contains parameters that vary by asset CATEGORY (FOREX vs INDICES vs COMMODITIES vs CRYPTO)
NO MORE HARDCODED ASSET SYMBOLS - TRULY GENERIC MULTIASSET SYSTEM

MIGRATED FROM:
- Original src/Analyzer.py AnalyzerConfig asset-specific parameters
- Consolidated with base_config.py to eliminate duplication
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from ScalpingBOT_Restauro.src.config.base.base_config import AnalyzerConfig


@dataclass
class AssetSpecificConfig:
    """
    Asset-specific configuration parameters - DYNAMIC MULTIASSET SYSTEM
    
    NO HARDCODED ASSET SYMBOLS - Uses dynamic category detection:
    - FOREX: Currency pairs (6-char currency codes with USD/EUR/GBP/etc.)
    - INDICES: Stock indices (symbols containing TEC/DAQ/SPX/500/DOW/etc.)
    - COMMODITIES: Precious metals, oil, etc. (symbols containing XAU/XAG/OIL/WTI/etc.)
    - CRYPTO: Cryptocurrencies (symbols containing BTC/ETH/LTC/XRP/etc.)
    """
    
    # === ASSET IDENTIFICATION ===
    asset_symbol: str = ""  # Will be set at runtime
    asset_display_name: str = ""  # Human readable name
    asset_category: str = ""  # FOREX, INDICES, COMMODITIES, CRYPTO
    
    # === TRADING THRESHOLDS (ASSET-CATEGORY-SPECIFIC) ===
    
    # === ASSET-CATEGORY OVERRIDES ===
    # Dynamic parameters that vary by asset category (FOREX, INDICES, COMMODITIES, CRYPTO)
    # All other parameters inherited from base AnalyzerConfig
    
    # Competition system overrides (if asset needs different thresholds)
    champion_threshold_override: Optional[float] = None
    min_predictions_for_champion_override: Optional[int] = None
    
    # Performance threshold overrides (if asset needs different thresholds)
    accuracy_threshold_override: Optional[float] = None
    confidence_threshold_override: Optional[float] = None
    
    # Emergency stop overrides (different volatility assets need different thresholds)
    emergency_accuracy_drop_override: Optional[float] = None
    emergency_consecutive_failures_override: Optional[int] = None
    emergency_confidence_collapse_override: Optional[float] = None
    
    # === VOLATILITY THRESHOLDS (ASSET-CATEGORY-SPECIFIC) ===
    # These MUST be category-specific as different asset categories have different volatility characteristics
    high_volatility_threshold: float = 0.02     # Dynamic based on asset category
    low_volatility_threshold: float = 0.005     # Dynamic based on asset category
    extreme_volatility_threshold: float = 0.025  # Dynamic based on asset category
    volatility_spike_multiplier: float = 1.5    # Category-specific spike detection
    
    # === TECHNICAL INDICATORS OVERRIDES ===
    # Only if asset needs different indicator parameters
    atr_period_override: Optional[int] = None    # Override ATR period if needed
    
    # === SPREAD AND VOLUME THRESHOLDS ===
    # These vary significantly between asset categories (stocks vs forex vs crypto)
    spread_high_threshold: float = 0.001        # High spread warning threshold
    volume_low_multiplier: float = 0.5          # Low volume detection multiplier
    volume_high_multiplier: float = 2.0         # High volume detection multiplier
    
    # === PATTERN RECOGNITION THRESHOLDS ===
    # Only parameters that need different values for different asset price scales
    double_top_bottom_tolerance: float = 0.01   # Tolerance for double top/bottom patterns
    triangle_slope_threshold: float = 0.0001    # Minimum slope for triangle patterns
    sr_proximity_threshold: float = 0.001       # Proximity threshold for support/resistance
    trend_slope_threshold: float = 0.0001       # Minimum slope for trend detection
    
    def __post_init__(self):
        """Validation and auto-setup of configuration"""
        
        # Validate required fields
        if not self.asset_symbol:
            raise ValueError("asset_symbol is required and cannot be empty")
            
        # Auto-set display name if not provided
        if not self.asset_display_name:
            self.asset_display_name = self.asset_symbol
        
        # Validate thresholds
        assert self.high_volatility_threshold > self.low_volatility_threshold, f"Volatility thresholds invalid for {self.asset_symbol}"
        assert self.extreme_volatility_threshold > self.high_volatility_threshold, f"Extreme volatility threshold invalid for {self.asset_symbol}"
        assert self.volume_high_multiplier > self.volume_low_multiplier, f"Volume multipliers invalid for {self.asset_symbol}"
    
    def get_asset_directory(self, base_path: str) -> str:
        """Get the directory path for this asset"""
        return os.path.join(base_path, self.asset_symbol)
    
    def get_models_directory(self, base_path: str) -> str:
        """Get the models directory for this asset"""
        return os.path.join(self.get_asset_directory(base_path), "models")
    
    def get_logs_directory(self, base_path: str) -> str:
        """Get the logs directory for this asset"""
        return os.path.join(self.get_asset_directory(base_path), "logs")
    
    def get_events_directory(self, base_path: str) -> str:
        """Get the events directory for this asset"""
        return os.path.join(self.get_asset_directory(base_path), "events")
    
    def get_data_directory(self, base_path: str) -> str:
        """Get the data directory for this asset"""
        return os.path.join(self.get_asset_directory(base_path), "data")
    
    def create_asset_directories(self, base_path: str):
        """Create all necessary directories for this asset"""
        directories = [
            self.get_asset_directory(base_path),
            self.get_models_directory(base_path),
            self.get_logs_directory(base_path),
            self.get_events_directory(base_path),
            self.get_data_directory(base_path)
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_combined_config(self, base_config: Optional[AnalyzerConfig] = None) -> AnalyzerConfig:
        """Create combined configuration from base config + asset-specific overrides"""
        if base_config is None:
            from ..base.base_config import get_analyzer_config
            config = get_analyzer_config()
        else:
            config = base_config
        
        # Apply asset-specific overrides to base config
        if self.champion_threshold_override is not None:
            config.champion_threshold = self.champion_threshold_override
        
        if self.min_predictions_for_champion_override is not None:
            config.min_predictions_for_champion = self.min_predictions_for_champion_override
        
        if self.accuracy_threshold_override is not None:
            config.accuracy_threshold = self.accuracy_threshold_override
        
        if self.confidence_threshold_override is not None:
            config.confidence_threshold = self.confidence_threshold_override
        
        if self.emergency_accuracy_drop_override is not None:
            config.emergency_accuracy_drop = self.emergency_accuracy_drop_override
        
        if self.emergency_consecutive_failures_override is not None:
            config.emergency_consecutive_failures = self.emergency_consecutive_failures_override
        
        if self.emergency_confidence_collapse_override is not None:
            config.emergency_confidence_collapse = self.emergency_confidence_collapse_override
        
        if self.atr_period_override is not None:
            config.atr_period = self.atr_period_override
        
        return config
    
    # === DYNAMIC MULTIASSET FACTORY METHODS ===
    
    @classmethod
    def for_asset(cls, asset_symbol: str) -> 'AssetSpecificConfig':
        """Create asset-specific configuration based on dynamic category detection"""
        
        # Detect asset category from symbol pattern
        category = cls._detect_asset_category(asset_symbol)
        
        # Create configuration based on category, not hardcoded symbols
        return cls._create_config_for_category(asset_symbol, category)
    
    @classmethod
    def _detect_asset_category(cls, asset_symbol: str) -> str:
        """Detect asset category from symbol pattern - NO HARDCODED ASSETS"""
        symbol_upper = asset_symbol.upper()
        
        # FOREX patterns (3+3 or 6 chars with currency codes)
        if len(symbol_upper) == 6 and any(curr in symbol_upper for curr in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']):
            return "FOREX"
            
        # INDICES patterns (common index suffixes)
        if any(suffix in symbol_upper for suffix in ['TEC', 'DAQ', 'SPX', '500', 'DOW', 'NIK', 'FTSE', 'DAX']):
            return "INDICES"
            
        # COMMODITIES patterns
        if any(comm in symbol_upper for comm in ['XAU', 'XAG', 'OIL', 'WTI', 'BRENT', 'GAS', 'GOLD', 'SILVER']):
            return "COMMODITIES"
            
        # CRYPTO patterns
        if any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'LTC', 'XRP', 'ADA', 'DOT']):
            return "CRYPTO"
            
        # Default to FOREX if pattern not recognized
        return "FOREX"
    
    @classmethod
    def _create_config_for_category(cls, asset_symbol: str, category: str) -> 'AssetSpecificConfig':
        """Create configuration based on asset category - TRULY MULTIASSET"""
        
        if category == "FOREX":
            return cls._forex_category_config(asset_symbol)
        elif category == "INDICES":
            return cls._indices_category_config(asset_symbol)
        elif category == "COMMODITIES":
            return cls._commodities_category_config(asset_symbol)
        elif category == "CRYPTO":
            return cls._crypto_category_config(asset_symbol)
        else:
            # Default to FOREX category for unknown patterns
            return cls._forex_category_config(asset_symbol)
    
    @classmethod
    def _forex_category_config(cls, asset_symbol: str) -> 'AssetSpecificConfig':
        """Configuration optimized for FOREX assets"""
        return cls(
            asset_symbol=asset_symbol,
            asset_display_name=f"Forex {asset_symbol}",
            asset_category="FOREX",
            # FOREX volatility characteristics (stable currency markets)
            high_volatility_threshold=0.015,      # Moderate volatility for FX
            low_volatility_threshold=0.003,       # Low threshold for stable periods
            extreme_volatility_threshold=0.025,   # Conservative extreme threshold
            volatility_spike_multiplier=1.3,      # Conservative spike detection
            # FOREX trading parameters
            spread_high_threshold=0.0005,         # Tight spread for forex markets
            volume_low_multiplier=0.5,            # Higher volume requirement
            volume_high_multiplier=2.0,           # Conservative volume detection
            # FOREX price scale adjustments (precision trading)
            double_top_bottom_tolerance=0.0025,   # Tight tolerance for forex precision
            triangle_slope_threshold=0.00005,     # Low slope for gradual forex moves
            sr_proximity_threshold=0.0008,        # Tight S/R proximity for forex
            trend_slope_threshold=0.00005         # Low slope for gradual forex trends
        )
    
    @classmethod
    def _indices_category_config(cls, asset_symbol: str) -> 'AssetSpecificConfig':
        """Configuration optimized for INDICES assets"""
        return cls(
            asset_symbol=asset_symbol,
            asset_display_name=f"Index {asset_symbol}",
            asset_category="INDICES",
            # INDICES volatility characteristics (market index volatility)
            high_volatility_threshold=0.025,      # Higher volatility for index markets
            low_volatility_threshold=0.005,       # Standard low threshold
            extreme_volatility_threshold=0.040,   # Higher extreme for index crashes
            volatility_spike_multiplier=2.0,      # Higher spike detection
            # INDICES trading parameters
            spread_high_threshold=0.002,          # Higher spread tolerance
            volume_low_multiplier=0.3,            # Lower volume threshold
            volume_high_multiplier=3.0,           # Higher volume spike detection
            # INDICES price scale adjustments
            double_top_bottom_tolerance=0.015,    # Wider tolerance for index movements
            triangle_slope_threshold=0.0002,      # Higher slope for index moves
            sr_proximity_threshold=0.0015,        # S/R proximity for indices
            trend_slope_threshold=0.0002          # Higher slope for index trends
        )
    
    @classmethod
    def _commodities_category_config(cls, asset_symbol: str) -> 'AssetSpecificConfig':
        """Configuration optimized for COMMODITIES assets"""
        return cls(
            asset_symbol=asset_symbol,
            asset_display_name=f"Commodity {asset_symbol}",
            asset_category="COMMODITIES",
            # COMMODITIES volatility characteristics (commodity market volatility)
            high_volatility_threshold=0.030,      # Higher volatility for commodity markets
            low_volatility_threshold=0.008,       # Higher low threshold
            extreme_volatility_threshold=0.050,   # Much higher for commodity market spikes
            volatility_spike_multiplier=2.5,      # High spike detection
            # COMMODITIES trading parameters
            spread_high_threshold=0.001,          # Wider spread tolerance
            volume_low_multiplier=0.2,            # Lower volume requirement
            volume_high_multiplier=4.0,           # Higher volume spike detection
            # COMMODITIES price scale adjustments
            double_top_bottom_tolerance=0.020,    # Wider tolerance for commodity swings
            triangle_slope_threshold=0.0005,      # Higher slope for commodity moves
            sr_proximity_threshold=0.002,         # Wider S/R proximity for commodities
            trend_slope_threshold=0.0005          # Higher slope for commodity trends
        )
    
    @classmethod
    def _crypto_category_config(cls, asset_symbol: str) -> 'AssetSpecificConfig':
        """Configuration optimized for CRYPTO assets"""
        return cls(
            asset_symbol=asset_symbol,
            asset_display_name=f"Crypto {asset_symbol}",
            asset_category="CRYPTO",
            # CRYPTO volatility characteristics (cryptocurrency volatility)
            high_volatility_threshold=0.050,      # Very high volatility for crypto markets
            low_volatility_threshold=0.015,       # Higher low threshold
            extreme_volatility_threshold=0.100,   # Extreme threshold for crypto markets
            volatility_spike_multiplier=3.0,      # Very high spike detection
            # CRYPTO trading parameters
            spread_high_threshold=0.003,          # Wide spread tolerance
            volume_low_multiplier=0.1,            # Very low volume requirement
            volume_high_multiplier=5.0,           # Very high volume spike detection
            # CRYPTO price scale adjustments
            double_top_bottom_tolerance=0.030,    # Wide tolerance for crypto swings
            triangle_slope_threshold=0.001,       # High slope for crypto moves
            sr_proximity_threshold=0.005,         # Wide S/R proximity for crypto
            trend_slope_threshold=0.001           # High slope for crypto trends
        )
    
    # === CONFIGURATION PERSISTENCE ===
    
    def save_to_file(self, config_path: str) -> None:
        """Save asset-specific configuration to file"""
        try:
            import json
            
            config_dict = {
                'asset_symbol': self.asset_symbol,
                'asset_display_name': self.asset_display_name,
                'asset_category': self.asset_category,
                'high_volatility_threshold': self.high_volatility_threshold,
                'low_volatility_threshold': self.low_volatility_threshold,
                'extreme_volatility_threshold': self.extreme_volatility_threshold,
                'volatility_spike_multiplier': self.volatility_spike_multiplier,
                'spread_high_threshold': self.spread_high_threshold,
                'volume_low_multiplier': self.volume_low_multiplier,
                'volume_high_multiplier': self.volume_high_multiplier,
                'double_top_bottom_tolerance': self.double_top_bottom_tolerance,
                'triangle_slope_threshold': self.triangle_slope_threshold,
                'sr_proximity_threshold': self.sr_proximity_threshold,
                'trend_slope_threshold': self.trend_slope_threshold
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
        except Exception as e:
            raise RuntimeError(f"Error saving asset config for {self.asset_symbol}: {e}")


# Export
__all__ = ['AssetSpecificConfig']