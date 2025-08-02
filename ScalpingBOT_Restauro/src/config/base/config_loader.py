"""
Unified configuration loader - NEW FILE
Loads and manages all system configurations
"""

import os
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Import all configuration classes
from ScalpingBOT_Restauro.src.config.base.base_config import AnalyzerConfig, get_analyzer_config, set_analyzer_config
from ScalpingBOT_Restauro.src.config.domain.system_config import UnifiedConfig, SystemMode, PerformanceProfile
from ScalpingBOT_Restauro.src.config.domain.monitoring_config import MLTrainingLoggerConfig, VerbosityLevel
from ScalpingBOT_Restauro.src.config.domain.asset_config import AssetSpecificConfig

# Import universal encoding fix for system-wide Unicode support
from ScalpingBOT_Restauro.src.monitoring.utils.universal_encoding_fix import init_universal_encoding


# ============================================================================
# CONFIGURATION ENUMS
# ============================================================================

# ENUM RIMOSSA - Usare VerbosityLevel da monitoring_config.py per evitare duplicazioni


# ============================================================================
# CONFIGURATION CONTAINERS
# ============================================================================

@dataclass
class SystemConfiguration:
    """Complete system configuration container"""
    analyzer: AnalyzerConfig          # General analyzer parameters
    unified: UnifiedConfig            # System-wide settings
    monitoring: MLTrainingLoggerConfig # Monitoring settings
    asset: AssetSpecificConfig        # Asset-specific parameters


class ConfigurationManager:
    """Manages all system configurations"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = config_dir or os.path.join(os.path.dirname(__file__), "..", "..", "..", "config")
        self._current_config: Optional[SystemConfiguration] = None
        
        # Initialize universal encoding for system-wide Unicode support
        # This fixes emoji/Unicode issues across the entire application
        init_universal_encoding(silent=True)
    
    def load_default_configuration(self, asset_symbol: str) -> SystemConfiguration:
        """
        Load default configuration for all components
        
        Args:
            asset_symbol: Asset symbol (REQUIRED - from startup selection)
        """
        analyzer_config = AnalyzerConfig()
        unified_config = UnifiedConfig()
        monitoring_config = MLTrainingLoggerConfig()
        asset_config = AssetSpecificConfig.for_asset(asset_symbol)
        
        # Create asset directories
        asset_config.create_asset_directories(unified_config.base_directory)
        
        # Update paths to use asset-specific directories
        monitoring_config.storage.output_directory = asset_config.get_events_directory(unified_config.base_directory)
        
        self._current_config = SystemConfiguration(
            analyzer=analyzer_config,
            unified=unified_config,
            monitoring=monitoring_config,
            asset=asset_config
        )
        
        return self._current_config
    
    def load_configuration_for_mode(self, mode: str, asset_symbol: str) -> SystemConfiguration:
        """
        Load configuration optimized for specific mode and asset
        
        Args:
            mode: One of 'production', 'development', 'testing', 'demo'
            asset_symbol: Asset symbol (REQUIRED - from startup selection)
        """
        # Load base analyzer config
        analyzer_config = AnalyzerConfig()
        
        # Create unified config based on mode (no asset parameter needed)
        if mode == "production":
            unified_config = UnifiedConfig.for_production()
            verbosity = VerbosityLevel.MINIMAL
        elif mode == "development":
            unified_config = UnifiedConfig.for_development()
            verbosity = VerbosityLevel.VERBOSE
        elif mode == "demo":
            unified_config = UnifiedConfig.for_demo()
            verbosity = VerbosityLevel.DEBUG
        else:  # testing
            unified_config = UnifiedConfig()
            unified_config.system_mode = SystemMode.TESTING
            verbosity = VerbosityLevel.VERBOSE
        
        # Create asset-specific configuration
        asset_config = AssetSpecificConfig.for_asset(asset_symbol)
        
        # Create asset directories
        asset_config.create_asset_directories(unified_config.base_directory)
        
        # Create monitoring config with appropriate verbosity
        monitoring_config = MLTrainingLoggerConfig(verbosity=verbosity)
        
        # Sync settings using asset-specific paths
        monitoring_config.storage.output_directory = asset_config.get_events_directory(unified_config.base_directory)
        monitoring_config.storage.flush_interval_seconds = analyzer_config.ml_logger_flush_interval
        
        self._current_config = SystemConfiguration(
            analyzer=analyzer_config,
            unified=unified_config,
            monitoring=monitoring_config,
            asset=asset_config
        )
        
        return self._current_config
    
    def load_from_files(self, 
                       asset_symbol: str,
                       analyzer_file: Optional[str] = None,
                       unified_file: Optional[str] = None,
                       monitoring_file: Optional[str] = None) -> SystemConfiguration:
        """
        Load configuration from specific files
        
        Args:
            asset_symbol: Asset symbol for asset-specific configuration
            analyzer_file: Path to analyzer config JSON
            unified_file: Path to unified config JSON
            monitoring_file: Path to monitoring config JSON
        """
        # Load analyzer config
        if analyzer_file and os.path.exists(analyzer_file):
            analyzer_config = AnalyzerConfig.load_from_file(analyzer_file)
        else:
            analyzer_config = AnalyzerConfig()
        
        # Load unified config
        if unified_file and os.path.exists(unified_file):
            with open(unified_file, 'r') as f:
                unified_data = json.load(f)
            unified_config = UnifiedConfig(**unified_data)
        else:
            unified_config = UnifiedConfig()
        
        # Load monitoring config
        if monitoring_file and os.path.exists(monitoring_file):
            monitoring_config = MLTrainingLoggerConfig()
            monitoring_config.load_from_file(monitoring_file)
        else:
            monitoring_config = MLTrainingLoggerConfig()
        
        # Create asset-specific configuration
        asset_config = AssetSpecificConfig.for_asset(asset_symbol)
        
        # Create asset directories
        asset_config.create_asset_directories(unified_config.base_directory)
        
        # Update monitoring paths to use asset-specific directories
        monitoring_config.storage.output_directory = asset_config.get_events_directory(unified_config.base_directory)
        
        self._current_config = SystemConfiguration(
            analyzer=analyzer_config,
            unified=unified_config,
            monitoring=monitoring_config,
            asset=asset_config
        )
        
        return self._current_config
    
    def save_current_configuration(self, 
                                 analyzer_file: Optional[str] = None,
                                 unified_file: Optional[str] = None,
                                 monitoring_file: Optional[str] = None):
        """Save current configuration to files"""
        if not self._current_config:
            raise RuntimeError("No configuration loaded")
        
        # Save analyzer config
        if analyzer_file:
            self._current_config.analyzer.save_to_file(analyzer_file)
        
        # Save unified config
        if unified_file:
            # Convert to dict manually for unified config
            unified_dict = {
                "system_mode": self._current_config.unified.system_mode.value,
                "performance_profile": self._current_config.unified.performance_profile.value,
                "log_level": self._current_config.unified.log_level,
                "enable_console_output": self._current_config.unified.enable_console_output,
                "enable_file_output": self._current_config.unified.enable_file_output,
                "base_directory": self._current_config.unified.base_directory,
                "rate_limits": self._current_config.unified.rate_limits,
                # Add other fields as needed
            }
            with open(unified_file, 'w') as f:
                json.dump(unified_dict, f, indent=2)
        
        # Save monitoring config
        if monitoring_file:
            self._current_config.monitoring.save_to_file(monitoring_file)
    
    def get_current_configuration(self) -> SystemConfiguration:
        """Get current active configuration"""
        if not self._current_config:
            raise RuntimeError("No configuration loaded - must call load_configuration_for_mode(mode, asset_symbol) first")
        return self._current_config
    
    def update_analyzer_config(self, updates: Dict[str, Any]):
        """Update analyzer configuration values"""
        if not self._current_config:
            raise RuntimeError("No configuration loaded - cannot update analyzer config")
        
        for key, value in updates.items():
            if hasattr(self._current_config.analyzer, key):
                setattr(self._current_config.analyzer, key, value)
    
    def get_model_architecture(self, model_name: str) -> Dict[str, Any]:
        """Get model architecture from analyzer config"""
        if not self._current_config:
            raise RuntimeError("No configuration loaded - cannot get model architecture")
        return self._current_config.analyzer.get_model_architecture(model_name)
    
    def get_rate_limits(self) -> Dict[str, int]:
        """Get rate limits from unified config"""
        if not self._current_config:
            raise RuntimeError("No configuration loaded - cannot get rate limits")
        return self._current_config.unified.rate_limits
    
    # ============================================================================
    # FACTORY METHODS eliminati - Usare load_configuration_for_mode() direttamente
    # UnifiedConfig.for_production/development/demo() già forniscono factory specifici
    # ============================================================================
    
    def get_verbosity_from_config(self) -> VerbosityLevel:
        """Ritorna VerbosityLevel direttamente"""
        if not self._current_config:
            raise RuntimeError("No configuration loaded")
        return self._current_config.monitoring.verbosity


# Global configuration manager instance
_config_manager = ConfigurationManager()


def get_configuration_manager() -> ConfigurationManager:
    """Get global configuration manager instance"""
    return _config_manager


def load_configuration_for_mode(mode: str, asset_symbol: str) -> SystemConfiguration:
    """Convenience function to load configuration for specific mode"""
    return _config_manager.load_configuration_for_mode(mode, asset_symbol)


def get_current_configuration() -> SystemConfiguration:
    """Convenience function to get current configuration"""
    return _config_manager.get_current_configuration()


