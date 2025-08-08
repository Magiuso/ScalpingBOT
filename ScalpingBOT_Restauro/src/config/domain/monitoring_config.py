"""
Monitoring configuration - MIGRATED from ML_Training_Logger/Config_Manager.py (lines 1-400+)
NO LOGIC CHANGES - Only reorganized
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum


class VerbosityLevel(Enum):
    """Livelli di verbosità del logging"""
    MINIMAL = "minimal"
    STANDARD = "standard" 
    VERBOSE = "verbose"
    DEBUG = "debug"


class OutputFormat(Enum):
    """Formati di output supportati"""
    JSON = "json"
    CSV = "csv"
    BOTH = "both"


class TerminalMode(Enum):
    """Modalità di visualizzazione terminale"""
    SCROLL = "scroll"          # Output tradizionale scorrevole
    SILENT = "silent"          # Solo file output
    AUTO = "auto"             # Detect capabilities


class EventSeverity(Enum):
    """Livelli di severità eventi"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DisplaySettings:
    """Configurazione display terminale"""
    terminal_mode: TerminalMode = TerminalMode.AUTO
    refresh_rate_seconds: float = 2.0
    max_recent_events: int = 10
    show_timestamps: bool = True
    show_progress_bar: bool = True
    show_champions_status: bool = True
    show_performance_metrics: bool = True
    color_enabled: bool = True
    
    # Removed dashboard layout settings since DASHBOARD mode is eliminated


@dataclass
class StorageSettings:
    """Configurazione storage e persistenza"""
    enable_file_output: bool = True
    output_directory: str = "./analyzer_data"
    
    # File naming
    session_prefix: str = "ml_training"
    timestamp_format: str = "%Y%m%d_%H%M%S"
    
    # Formats
    output_formats: List[OutputFormat] = field(default_factory=lambda: [OutputFormat.JSON, OutputFormat.CSV])
    
    # JSON settings
    json_indent: int = 2
    json_ensure_ascii: bool = False
    
    # CSV settings
    csv_delimiter: str = ","
    csv_include_headers: bool = True
    
    # File rotation
    enable_rotation: bool = True
    max_file_size_mb: int = 100
    max_files_per_session: int = 10
    compress_old_logs: bool = True 
    
    # Flush settings
    flush_interval_seconds: float = 5.0
    flush_on_critical: bool = True
    buffer_size: int = 1000
    
    # Batch processing
    batch_processing_size: int = 100


@dataclass
class EventFilterSettings:
    """Configurazione filtri eventi"""
    verbosity_level: VerbosityLevel = VerbosityLevel.STANDARD
    
    # Severity filtering
    min_severity: EventSeverity = EventSeverity.INFO
    excluded_severities: List[EventSeverity] = field(default_factory=list)
    
    # Event type filtering
    enabled_event_types: List[str] = field(default_factory=lambda: [
        "learning_progress",
        "champion_change", 
        "model_training",
        "performance_metrics",
        "emergency_stop",
        "validation_complete"
    ])
    disabled_event_types: List[str] = field(default_factory=list)
    
    # Source filtering
    enabled_sources: List[str] = field(default_factory=lambda: [
        "AdvancedMarketAnalyzer",
        "UnifiedAnalyzerSystem",
        "Manual"  # 🔧 FIX: Allow manual events from _emit_ml_event
    ])
    disabled_sources: List[str] = field(default_factory=list)
    
    # Rate limiting per event type - MIGRATED TO UNIFIED CONFIG
    # MIGRATED TO: src/config/shared/rate_limiting_config.py
    @property
    def event_rate_limits(self) -> Dict[str, float]:
        """Event rate limits - now unified in shared configuration"""
        from ..shared.rate_limiting_config import get_legacy_monitoring_rate_limits
        return get_legacy_monitoring_rate_limits()


@dataclass
class PerformanceSettings:
    """Configurazione performance e ottimizzazioni"""
    # Threading
    enable_async_processing: bool = True
    max_worker_threads: int = 2
    event_queue_size: int = 10000
    
    # Memory management
    max_memory_mb: int = 200
    cleanup_interval_seconds: float = 30.0
    max_buffer_size: int = 10000
    
    # Processing optimization
    batch_processing_size: int = 100
    event_compression: bool = False
    
    # Rate limiting per event type - MIGRATED TO UNIFIED CONFIG
    # MIGRATED TO: src/config/shared/rate_limiting_config.py
    @property  
    def rate_limits(self) -> Dict[str, float]:
        """Rate limits - now unified in shared configuration"""
        from ..shared.rate_limiting_config import get_legacy_monitoring_rate_limits
        return get_legacy_monitoring_rate_limits()
    
    # Emergency handling
    emergency_fallback_enabled: bool = True
    emergency_queue_size: int = 1000
    emergency_timeout_seconds: float = 5.0


@dataclass
class IntegrationSettings:
    """Configurazione integrazione con sistemi esistenti"""
    # AdvancedMarketAnalyzer integration
    analyzer_hook_enabled: bool = True
    analyzer_hook_methods: List[str] = field(default_factory=lambda: [
        "on_learning_progress",
        "on_champion_change",
        "on_model_training",
        "on_emergency_stop"
    ])
    
    # UnifiedAnalyzerSystem integration  
    unified_system_hook_enabled: bool = True
    unified_system_events: List[str] = field(default_factory=lambda: [
        "performance_metrics",
        "system_status",
        "error_events"
    ])
    
    # Callback configuration
    callback_timeout_seconds: float = 1.0
    callback_retry_attempts: int = 3
    callback_error_handling: str = "ignore"  # ignore, log, raise


class MLTrainingLoggerConfig:
    """
    Configurazione completa per MLTrainingLogger
    """
    
    def __init__(self, 
                 verbosity: VerbosityLevel = VerbosityLevel.STANDARD,
                 config_file: Optional[str] = None):
        """
        Inizializza configurazione
        
        Args:
            verbosity: Livello di verbosità predefinito
            config_file: File di configurazione opzionale
        """
        self.verbosity = verbosity
        self.config_file = config_file
        
        # Initialize settings with defaults
        self.display = DisplaySettings()
        self.storage = StorageSettings()
        self.event_filter = EventFilterSettings(verbosity_level=verbosity)
        self.performance = PerformanceSettings()
        self.integration = IntegrationSettings()
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # Apply verbosity-specific defaults
        self._apply_verbosity_defaults()
        
        # Validate configuration
        self._validate_config()
    
    def _apply_verbosity_defaults(self):
        """Applica impostazioni predefinite basate sul livello di verbosità"""
        
        if self.verbosity == VerbosityLevel.MINIMAL:
            # Minimal: solo eventi critici e progress essenziale
            self.event_filter.min_severity = EventSeverity.WARNING
            self.event_filter.enabled_event_types = [
                "learning_progress", 
                "champion_change",
                "emergency_stop"
            ]
            self.display.max_recent_events = 5
            self.display.show_performance_metrics = False
            self.storage.flush_interval_seconds = 10.0
            
        elif self.verbosity == VerbosityLevel.STANDARD:
            # Standard: eventi importanti + metriche base
            self.event_filter.min_severity = EventSeverity.INFO
            self.event_filter.enabled_event_types = [
                "learning_progress",
                "champion_change",
                "model_training", 
                "performance_metrics",
                "emergency_stop",
                "validation_complete"
            ]
            self.display.max_recent_events = 10
            self.display.show_performance_metrics = True
            self.storage.flush_interval_seconds = 5.0
            
        elif self.verbosity == VerbosityLevel.VERBOSE:
            # Verbose: tutti gli eventi + dettagli estesi
            self.event_filter.min_severity = EventSeverity.DEBUG
            self.event_filter.enabled_event_types.extend([
                "prediction_generated",
                "algorithm_update",
                "diagnostics_event"
            ])
            self.display.max_recent_events = 20
            self.display.show_timestamps = True
            self.storage.flush_interval_seconds = 2.0
            self.storage.json_indent = 4
            
        elif self.verbosity == VerbosityLevel.DEBUG:
            # Debug: tutto + diagnostica interna
            self.event_filter.min_severity = EventSeverity.DEBUG
            self.event_filter.enabled_event_types.extend([
                "prediction_generated",
                "algorithm_update", 
                "diagnostics_event",
                "internal_state",
                "performance_debug",
                "memory_usage"
            ])
            self.display.max_recent_events = 50
            self.display.refresh_rate_seconds = 1.0
            self.storage.flush_interval_seconds = 1.0
            self.storage.json_indent = 4
            self.performance.enable_async_processing = False  # Sync per debug
    
    def _validate_config(self):
        """Valida la configurazione e corregge valori non validi"""
        
        # Validate display settings
        if self.display.refresh_rate_seconds <= 0:
            self.display.refresh_rate_seconds = 1.0
            
        if self.display.max_recent_events < 1:
            self.display.max_recent_events = 5
            
        # Validate storage settings
        if self.storage.flush_interval_seconds <= 0:
            self.storage.flush_interval_seconds = 1.0
            
        if self.storage.max_file_size_mb < 1:
            self.storage.max_file_size_mb = 10
            
        # Validate performance settings
        if self.performance.max_worker_threads < 1:
            self.performance.max_worker_threads = 1
            
        if self.performance.event_queue_size < 100:
            self.performance.event_queue_size = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte configurazione in dizionario"""
        return {
            "verbosity": self.verbosity.value,
            "display": asdict(self.display),
            "storage": asdict(self.storage),
            "event_filter": asdict(self.event_filter),
            "performance": asdict(self.performance),
            "integration": asdict(self.integration)
        }
    
    def save_to_file(self, filepath: str):
        """Salva configurazione su file"""
        try:
            config_dict = self.to_dict()
            # Convert enums to strings
            config_dict["display"]["terminal_mode"] = self.display.terminal_mode.value
            config_dict["event_filter"]["verbosity_level"] = self.event_filter.verbosity_level.value
            config_dict["event_filter"]["min_severity"] = self.event_filter.min_severity.value
            config_dict["event_filter"]["excluded_severities"] = [
                s.value for s in self.event_filter.excluded_severities
            ]
            config_dict["storage"]["output_formats"] = [
                f.value for f in self.storage.output_formats
            ]
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
        except Exception as e:
            raise RuntimeError(f"Error saving config to {filepath}: {e}")
    
    def load_from_file(self, filepath: str):
        """Carica configurazione da file"""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # Update verbosity
            if "verbosity" in config_dict:
                self.verbosity = VerbosityLevel(config_dict["verbosity"])
            
            # Update settings
            if "display" in config_dict:
                for key, value in config_dict["display"].items():
                    if key == "terminal_mode":
                        value = TerminalMode(value)
                    setattr(self.display, key, value)
                    
            if "storage" in config_dict:
                for key, value in config_dict["storage"].items():
                    if key == "output_formats":
                        value = [OutputFormat(f) for f in value]
                    setattr(self.storage, key, value)
                    
            if "event_filter" in config_dict:
                for key, value in config_dict["event_filter"].items():
                    if key == "verbosity_level":
                        value = VerbosityLevel(value)
                    elif key == "min_severity":
                        value = EventSeverity(value)
                    elif key == "excluded_severities":
                        value = [EventSeverity(s) for s in value]
                    setattr(self.event_filter, key, value)
                    
            if "performance" in config_dict:
                for key, value in config_dict["performance"].items():
                    setattr(self.performance, key, value)
                    
            if "integration" in config_dict:
                for key, value in config_dict["integration"].items():
                    setattr(self.integration, key, value)
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading config from {filepath}: {e}")


# ================== STANDALONE TEST ==================
