"""
Configuration module - Unified configuration management
"""

# Import main configuration classes
from src.config.base.base_config import AnalyzerConfig
from src.config.domain.system_config import UnifiedConfig, SystemMode, PerformanceProfile
from src.config.domain.monitoring_config import MLTrainingLoggerConfig, VerbosityLevel, EventSeverity

# Import configuration manager
from src.config.base.config_loader import (
    ConfigurationManager,
    SystemConfiguration,
    get_configuration_manager,
    load_configuration_for_mode,
    get_current_configuration
)

__all__ = [
    # Base configs
    'AnalyzerConfig',
    'UnifiedConfig',
    'MLTrainingLoggerConfig',
    
    # Enums
    'SystemMode',
    'PerformanceProfile',
    'VerbosityLevel',
    'EventSeverity',
    
    # Configuration management
    'ConfigurationManager',
    'SystemConfiguration',
    'get_configuration_manager',
    'load_configuration_for_mode',
    'get_current_configuration'
]