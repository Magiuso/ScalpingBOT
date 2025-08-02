"""
System configuration - MIGRATED from src/Unified_Analyzer_System.py (lines 58-236)
NO LOGIC CHANGES - Only reorganized
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime
from enum import Enum
from collections import deque


class SystemMode(Enum):
    """Modalità operative del sistema"""
    PRODUCTION = "production"      # Minimal logging, max performance
    DEVELOPMENT = "development"    # Normal logging, debugging enabled
    TESTING = "testing"           # Verbose logging, full diagnostics
    DEMO = "demo"                 # Rich console output, showcasing
    BACKTESTING = "backtesting"   # Ultra-high performance backtesting mode


class PerformanceProfile(Enum):
    """Profili di performance predefiniti"""
    HIGH_FREQUENCY = "high_frequency"    # Trading ad alta frequenza
    NORMAL = "normal"                    # Trading normale
    RESEARCH = "research"                # Ricerca e backtesting


@dataclass
class UnifiedConfig:
    """Configurazione unificata per tutto il sistema (OTTIMIZZATA)"""
    
    # === SYSTEM SETTINGS ===
    system_mode: SystemMode = SystemMode.PRODUCTION
    performance_profile: PerformanceProfile = PerformanceProfile.NORMAL
    
    # === ANALYZER SETTINGS ===
    # NOTE: asset_symbol removed - will be set at startup via asset selection
    max_tick_buffer_size: int = 1000000  # Increased to 1M to support 100K chunks
    learning_phase_enabled: bool = True
    min_learning_days: int = 7
    
    # === LOGGING SETTINGS (OTTIMIZZATE) ===
    log_level: str = "NORMAL"              # MINIMAL, NORMAL, VERBOSE, DEBUG, SILENT
    enable_console_output: bool = True
    enable_file_output: bool = True
    enable_csv_export: bool = True
    enable_json_export: bool = False
    
    # === RATE LIMITING (UNIFIED) ===
    # MIGRATED TO: src/config/shared/rate_limiting_config.py
    # Use get_legacy_system_rate_limits() for backward compatibility
    @property
    def rate_limits(self) -> Dict[str, int]:
        """Rate limits - now unified in shared configuration"""
        from ..shared.rate_limiting_config import get_legacy_system_rate_limits
        return get_legacy_system_rate_limits()
    
    # === PERFORMANCE SETTINGS (OTTIMIZZATE) ===
    event_processing_interval: float = 5.0    # Secondi tra processing eventi
    batch_size: int = 100                     # AUMENTATO per performance
    max_queue_size: int = 20000               # AUMENTATO per buffer maggiori
    async_processing: bool = True
    max_workers: int = 2
    
    # === PREDICTION SETTINGS (NUOVE) ===
    # Note: Demo predictor removed - violates CLAUDE_RESTAURO.md "one implementation" rule
    prediction_confidence_threshold: float = 0.8  # Solo confidence > 80%
    prediction_duplicate_window: int = 30    # Finestra anti-duplicati (secondi)
    
    # === STORAGE SETTINGS ===
    base_directory: str = "./ScalpingBOT_Data"  # Root directory for all assets
    log_rotation_hours: int = 24
    max_log_files: int = 30
    compress_old_logs: bool = True
    
    # === MONITORING SETTINGS (SELETTIVE) ===
    enable_performance_monitoring: bool = True
    performance_report_interval: float = 60.0  # Secondi
    memory_threshold_mb: int = 1000
    cpu_threshold_percent: float = 80.0
    enable_memory_cleanup: bool = True        # NUOVO
    memory_cleanup_interval: int = 1000       # Cleanup ogni N ticks
    
    @classmethod
    def for_production(cls) -> 'UnifiedConfig':
        """Configurazione ottimizzata per produzione"""
        return cls(
            system_mode=SystemMode.PRODUCTION,
            performance_profile=PerformanceProfile.HIGH_FREQUENCY,
            log_level="MINIMAL",
            enable_console_output=False,
            enable_json_export=False,
            event_processing_interval=10.0,
            batch_size=200,
            performance_report_interval=300.0
        )
    
    
    @classmethod
    def for_development(cls) -> 'UnifiedConfig':
        """Configurazione ottimizzata per sviluppo"""
        return cls(
            system_mode=SystemMode.DEVELOPMENT,
            performance_profile=PerformanceProfile.NORMAL,
            log_level="VERBOSE",
            enable_console_output=True,
            enable_json_export=True,
            event_processing_interval=2.0,
            performance_report_interval=30.0
        )
    
    @classmethod
    def for_demo(cls) -> 'UnifiedConfig':
        """Configurazione ottimizzata per demo"""
        return cls(
            system_mode=SystemMode.DEMO,
            performance_profile=PerformanceProfile.NORMAL,
            log_level="DEBUG",
            enable_console_output=True,
            enable_json_export=True,
            event_processing_interval=1.0,
            performance_report_interval=10.0
        )


# ================== STANDALONE TEST ==================
