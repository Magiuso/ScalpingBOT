#!/usr/bin/env python3
"""
MLTrainingLogger - Configuration Manager
========================================

Gestisce la configurazione del sistema di logging ML training.
Supporta diversi livelli di verbosità e modalità operative.

Author: ScalpingBOT Team
Version: 1.0.0
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
    output_directory: str = "./test_analyzer_data"
    
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
        "UnifiedAnalyzerSystem"
    ])
    disabled_sources: List[str] = field(default_factory=list)
    
    # Rate limiting per event type - 🔧 SPAM FIX: Rate limits molto più aggressivi
    event_rate_limits: Dict[str, float] = field(default_factory=lambda: {
        "learning_progress": 0.1,    # 🔧 SPAM FIX: Era 1.0 -> 10x meno frequente (1 ogni 10 sec)
        "champion_change": 0.0,      # No limit (eventi rari)
        "model_training": 0.05,      # 🔧 SPAM FIX: Era 0.5 -> 10x meno frequente (1 ogni 20 sec)
        "performance_metrics": 0.01, # 🔧 SPAM FIX: Era 0.1 -> 10x meno frequente (1 ogni 100 sec)
        "prediction_generated": 1.0, # 🔧 SPAM FIX: Era 10.0 -> 10x meno frequente (1 al secondo)
        "overfitting_debug": 0.05,   # 🔧 NUOVO: Rate limit per debug overfitting (1 ogni 20 sec)
        "tensor_validation": 0.01,   # 🔧 NUOVO: Rate limit per tensor validation (1 ogni 100 sec)
        "gradient_debug": 0.02       # 🔧 NUOVO: Rate limit per gradient debug (1 ogni 50 sec)
    })


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
    
    # Processing optimization
    batch_processing_size: int = 100
    event_compression: bool = False
    
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
            self.performance.event_queue_size = 1000
            
        # Create output directory if it doesn't exist
        if self.storage.enable_file_output:
            os.makedirs(self.storage.output_directory, exist_ok=True)
    
    def save_to_file(self, filename: Optional[str] = None) -> str:
        """
        Salva configurazione su file JSON
        
        Args:
            filename: Nome file (opzionale, genera automaticamente se None)
            
        Returns:
            str: Path del file salvato
        """
        if filename is None:
            timestamp = datetime.now().strftime(self.storage.timestamp_format)
            filename = f"ml_logger_config_{timestamp}.json"
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = os.path.join(self.storage.output_directory, filename)
        
        config_dict = {
            'verbosity': self.verbosity.value,
            'display': asdict(self.display),
            'storage': asdict(self.storage),
            'event_filter': asdict(self.event_filter),
            'performance': asdict(self.performance),
            'integration': asdict(self.integration),
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0.0'
            }
        }
        
        # Convert enums to strings for JSON serialization
        config_dict = self._serialize_enums(config_dict)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=self.storage.json_indent, 
                     ensure_ascii=self.storage.json_ensure_ascii)
        
        return filepath
    
    def load_from_file(self, filename: str):
        """
        Carica configurazione da file JSON
        
        Args:
            filename: Path del file di configurazione
        """
        with open(filename, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Load verbosity
        if 'verbosity' in config_dict:
            self.verbosity = VerbosityLevel(config_dict['verbosity'])
        
        # Load sections
        if 'display' in config_dict:
            self.display = self._deserialize_dataclass(DisplaySettings, config_dict['display'])
        
        if 'storage' in config_dict:
            self.storage = self._deserialize_dataclass(StorageSettings, config_dict['storage'])
        
        if 'event_filter' in config_dict:
            self.event_filter = self._deserialize_dataclass(EventFilterSettings, config_dict['event_filter'])
        
        if 'performance' in config_dict:
            self.performance = self._deserialize_dataclass(PerformanceSettings, config_dict['performance'])
        
        if 'integration' in config_dict:
            self.integration = self._deserialize_dataclass(IntegrationSettings, config_dict['integration'])
    
    def _serialize_enums(self, obj):
        """Converte enum in stringhe per serializzazione JSON"""
        if isinstance(obj, dict):
            return {k: self._serialize_enums(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_enums(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return obj
    
    def _deserialize_dataclass(self, cls, data):
        """Deserializza dict in dataclass gestendo enum"""
        # Convert enum strings back to enums
        if cls == DisplaySettings and 'terminal_mode' in data:
            data['terminal_mode'] = TerminalMode(data['terminal_mode'])
        
        if cls == StorageSettings and 'output_formats' in data:
            data['output_formats'] = [OutputFormat(fmt) for fmt in data['output_formats']]
        
        if cls == EventFilterSettings:
            if 'verbosity_level' in data:
                data['verbosity_level'] = VerbosityLevel(data['verbosity_level'])
            if 'min_severity' in data:
                data['min_severity'] = EventSeverity(data['min_severity'])
            if 'excluded_severities' in data:
                data['excluded_severities'] = [EventSeverity(sev) for sev in data['excluded_severities']]
        
        return cls(**data)
    
    @classmethod
    def create_preset(cls, preset_name: str) -> 'MLTrainingLoggerConfig':
        """
        Crea configurazione da preset predefinito
        
        Args:
            preset_name: Nome preset (minimal, standard, verbose, debug, production)
            
        Returns:
            MLTrainingLoggerConfig: Configurazione inizializzata
        """
        preset_map = {
            'minimal': VerbosityLevel.MINIMAL,
            'standard': VerbosityLevel.STANDARD,
            'verbose': VerbosityLevel.VERBOSE,
            'debug': VerbosityLevel.DEBUG,
            'production': VerbosityLevel.STANDARD
        }
        
        if preset_name not in preset_map:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(preset_map.keys())}")
        
        config = cls(verbosity=preset_map[preset_name])
        
        # Production-specific adjustments
        if preset_name == 'production':
            config.display.terminal_mode = TerminalMode.SILENT
            config.performance.enable_async_processing = True
            config.performance.max_worker_threads = 4
            config.storage.flush_interval_seconds = 10.0
            config.storage.enable_rotation = True
        
        return config
    
    def get_display_config(self) -> DisplaySettings:
        """Ottieni configurazione display"""
        return self.display
    
    def get_storage_config(self) -> StorageSettings:
        """Ottieni configurazione storage"""
        return self.storage
    
    def get_filter_config(self) -> EventFilterSettings:
        """Ottieni configurazione filtri eventi"""
        return self.event_filter
    
    def get_performance_config(self) -> PerformanceSettings:
        """Ottieni configurazione performance"""
        return self.performance
    
    def get_integration_config(self) -> IntegrationSettings:
        """Ottieni configurazione integrazione"""
        return self.integration
    
    def is_event_enabled(self, event_type: str, severity: EventSeverity = EventSeverity.INFO) -> bool:
        """
        Verifica se un evento è abilitato dalla configurazione
        
        Args:
            event_type: Tipo di evento
            severity: Severità evento
            
        Returns:
            bool: True se evento è abilitato
        """
        # Check severity filter
        severity_values = {
            EventSeverity.DEBUG: 0,
            EventSeverity.INFO: 1, 
            EventSeverity.WARNING: 2,
            EventSeverity.ERROR: 3,
            EventSeverity.CRITICAL: 4
        }
        
        if severity_values[severity] < severity_values[self.event_filter.min_severity]:
            return False
        
        if severity in self.event_filter.excluded_severities:
            return False
        
        # Check event type filter
        if event_type in self.event_filter.disabled_event_types:
            return False
        
        if (self.event_filter.enabled_event_types and 
            event_type not in self.event_filter.enabled_event_types):
            return False
        
        return True
    
    def get_rate_limit(self, event_type: str) -> float:
        """
        Ottieni rate limit per tipo di evento
        
        Args:
            event_type: Tipo di evento
            
        Returns:
            float: Limite eventi/secondo (0 = no limit)
        """
        return self.event_filter.event_rate_limits.get(event_type, 0.0)
    
    def __str__(self) -> str:
        """Rappresentazione stringa della configurazione"""
        return f"MLTrainingLoggerConfig(verbosity={self.verbosity.value}, " \
               f"terminal_mode={self.display.terminal_mode.value}, " \
               f"output_formats={[fmt.value for fmt in self.storage.output_formats]})"
    
    def summary(self) -> Dict[str, Any]:
        """Ottieni riassunto configurazione"""
        return {
            'verbosity': self.verbosity.value,
            'terminal_mode': self.display.terminal_mode.value,
            'output_formats': [fmt.value for fmt in self.storage.output_formats],
            'refresh_rate': self.display.refresh_rate_seconds,
            'enabled_event_types': len(self.event_filter.enabled_event_types),
            'file_output_enabled': self.storage.enable_file_output,
            'async_processing': self.performance.enable_async_processing,
            'max_memory_mb': self.performance.max_memory_mb
        }


# Factory functions per configurazioni comuni
def create_minimal_config() -> MLTrainingLoggerConfig:
    """Crea configurazione minimale per test veloci"""
    return MLTrainingLoggerConfig.create_preset('minimal')


def create_standard_config() -> MLTrainingLoggerConfig:
    """Crea configurazione standard per uso generale"""
    return MLTrainingLoggerConfig.create_preset('standard')


def create_verbose_config() -> MLTrainingLoggerConfig:
    """Crea configurazione verbosa per analisi dettagliate"""
    return MLTrainingLoggerConfig.create_preset('verbose')


def create_debug_config() -> MLTrainingLoggerConfig:
    """Crea configurazione debug per sviluppo"""
    return MLTrainingLoggerConfig.create_preset('debug')


def create_production_config() -> MLTrainingLoggerConfig:
    """Crea configurazione ottimizzata per produzione"""
    return MLTrainingLoggerConfig.create_preset('production')


# Example usage e testing
if __name__ == "__main__":
    # Test delle configurazioni
    print("Testing MLTrainingLogger Configuration...")
    
    # Test preset configurations
    configs = {
        'minimal': create_minimal_config(),
        'standard': create_standard_config(), 
        'verbose': create_verbose_config(),
        'debug': create_debug_config(),
        'production': create_production_config()
    }
    
    for name, config in configs.items():
        print(f"\n{name.upper()} CONFIG:")
        print(f"  {config}")
        print(f"  Summary: {config.summary()}")
        
        # Test event filtering
        test_events = [
            ('learning_progress', EventSeverity.INFO),
            ('prediction_generated', EventSeverity.DEBUG),
            ('emergency_stop', EventSeverity.CRITICAL)
        ]
        
        for event_type, severity in test_events:
            enabled = config.is_event_enabled(event_type, severity)
            rate_limit = config.get_rate_limit(event_type)
            print(f"    {event_type} ({severity.value}): {'✓' if enabled else '✗'} (rate: {rate_limit}/s)")
    
    # Test save/load
    print("\nTesting save/load functionality...")
    test_config = create_standard_config()
    saved_file = test_config.save_to_file("test_config.json")
    print(f"Saved to: {saved_file}")
    
    loaded_config = MLTrainingLoggerConfig(config_file=saved_file)
    print(f"Loaded: {loaded_config}")
    print("✓ Save/load test completed")


# Advanced Configuration Methods
class AdvancedConfigManager:
    """
    Gestione avanzata della configurazione con tuning dinamico
    """
    
    def __init__(self, base_config: MLTrainingLoggerConfig):
        self.base_config = base_config
        self.runtime_adjustments = {}
        self.performance_history = []
        self.auto_tuning_enabled = False
        
    def enable_auto_tuning(self, performance_target: Optional[Dict[str, float]] = None):
        """
        Abilita auto-tuning basato su performance
        
        Args:
            performance_target: Target performance metrics
                - 'memory_usage_percent': 70.0
                - 'cpu_usage_percent': 50.0
                - 'event_processing_delay_ms': 100.0
                - 'storage_write_delay_ms': 50.0
        """
        self.auto_tuning_enabled = True
        self.performance_target = performance_target or {
            'memory_usage_percent': 70.0,
            'cpu_usage_percent': 50.0,
            'event_processing_delay_ms': 100.0,
            'storage_write_delay_ms': 50.0
        }
        
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """
        Aggiorna metriche performance e applica auto-tuning se abilitato
        
        Args:
            metrics: Current performance metrics
        """
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics.copy()
        })
        
        # Keep only last 100 measurements
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        if self.auto_tuning_enabled:
            self._apply_auto_tuning(metrics)
    
    def _apply_auto_tuning(self, current_metrics: Dict[str, float]):
        """Applica aggiustamenti automatici basati su performance"""
        
        adjustments_made = []
        
        # Memory usage tuning
        memory_usage = current_metrics.get('memory_usage_percent', 0)
        if memory_usage > self.performance_target['memory_usage_percent']:
            if self.base_config.performance.batch_processing_size > 50:
                self.base_config.performance.batch_processing_size -= 10
                adjustments_made.append(f"Reduced batch size to {self.base_config.performance.batch_processing_size}")
            
            if self.base_config.storage.buffer_size > 500:
                self.base_config.storage.buffer_size -= 100
                adjustments_made.append(f"Reduced buffer size to {self.base_config.storage.buffer_size}")
        
        # CPU usage tuning
        cpu_usage = current_metrics.get('cpu_usage_percent', 0)
        if cpu_usage > self.performance_target['cpu_usage_percent']:
            if self.base_config.display.refresh_rate_seconds < 5.0:
                self.base_config.display.refresh_rate_seconds += 0.5
                adjustments_made.append(f"Increased refresh rate to {self.base_config.display.refresh_rate_seconds}s")
            
            if self.base_config.performance.max_worker_threads > 1:
                self.base_config.performance.max_worker_threads -= 1
                adjustments_made.append(f"Reduced workers to {self.base_config.performance.max_worker_threads}")
        
        # Event processing delay tuning
        processing_delay = current_metrics.get('event_processing_delay_ms', 0)
        if processing_delay > self.performance_target['event_processing_delay_ms']:
            if self.base_config.storage.flush_interval_seconds < 10.0:
                self.base_config.storage.flush_interval_seconds += 1.0
                adjustments_made.append(f"Increased flush interval to {self.base_config.storage.flush_interval_seconds}s")
        
        # Storage write delay tuning
        storage_delay = current_metrics.get('storage_write_delay_ms', 0)
        if storage_delay > self.performance_target['storage_write_delay_ms']:
            if not self.base_config.performance.enable_async_processing:
                self.base_config.performance.enable_async_processing = True
                adjustments_made.append("Enabled async processing")
        
        if adjustments_made:
            self.runtime_adjustments[datetime.now().isoformat()] = adjustments_made
    
    def get_optimized_config_for_system(self, system_info: Dict[str, Any]) -> MLTrainingLoggerConfig:
        """
        Ottimizza configurazione basata su info sistema
        
        Args:
            system_info: System information
                - 'cpu_cores': int
                - 'memory_gb': float  
                - 'disk_type': 'ssd'|'hdd'
                - 'terminal_capabilities': Dict
                
        Returns:
            MLTrainingLoggerConfig: Configurazione ottimizzata
        """
        optimized_config = MLTrainingLoggerConfig(verbosity=self.base_config.verbosity)
        
        # CPU-based optimizations
        cpu_cores = system_info.get('cpu_cores', 2)
        optimized_config.performance.max_worker_threads = min(cpu_cores - 1, 4)
        
        if cpu_cores >= 8:
            optimized_config.performance.enable_async_processing = True
            optimized_config.display.refresh_rate_seconds = 1.0
        elif cpu_cores <= 2:
            optimized_config.performance.enable_async_processing = False
            optimized_config.display.refresh_rate_seconds = 3.0
        
        # Memory-based optimizations  
        memory_gb = system_info.get('memory_gb', 4.0)
        if memory_gb >= 16:
            optimized_config.performance.max_memory_mb = 500
            optimized_config.performance.event_queue_size = 20000
            optimized_config.storage.buffer_size = 2000
        elif memory_gb <= 4:
            optimized_config.performance.max_memory_mb = 100
            optimized_config.performance.event_queue_size = 5000
            optimized_config.storage.buffer_size = 500
        
        # Storage-based optimizations
        disk_type = system_info.get('disk_type', 'hdd')
        if disk_type == 'ssd':
            optimized_config.storage.flush_interval_seconds = 2.0
            optimized_config.storage.enable_rotation = True
        else:
            optimized_config.storage.flush_interval_seconds = 10.0
            optimized_config.storage.enable_rotation = False
        
        # Terminal capabilities optimization
        terminal_caps = system_info.get('terminal_capabilities', {})
        if terminal_caps.get('supports_ansi', False) and terminal_caps.get('supports_colors', False):
            optimized_config.display.terminal_mode = TerminalMode.SCROLL
            optimized_config.display.color_enabled = True
        else:
            optimized_config.display.terminal_mode = TerminalMode.SCROLL
            optimized_config.display.color_enabled = False
        
        return optimized_config
    
    def create_session_config(self, session_info: Dict[str, Any]) -> MLTrainingLoggerConfig:
        """
        Crea configurazione specifica per una sessione di training
        
        Args:
            session_info: Informazioni sessione
                - 'asset': str
                - 'expected_duration_hours': float
                - 'expected_tick_volume': int
                - 'learning_phase': bool
                - 'production_mode': bool
                
        Returns:
            MLTrainingLoggerConfig: Configurazione per la sessione
        """
        session_config = MLTrainingLoggerConfig(verbosity=self.base_config.verbosity)
        
        # Asset-specific naming
        asset = session_info.get('asset', 'UNKNOWN')
        session_config.storage.session_prefix = f"ml_training_{asset.lower()}"
        
        # Duration-based adjustments
        duration_hours = session_info.get('expected_duration_hours', 1.0)
        if duration_hours > 24:  # Long sessions
            session_config.storage.enable_rotation = True
            session_config.storage.max_file_size_mb = 50
            session_config.display.refresh_rate_seconds = 5.0
        elif duration_hours < 1:  # Short sessions
            session_config.storage.enable_rotation = False
            session_config.display.refresh_rate_seconds = 1.0
            session_config.storage.flush_interval_seconds = 2.0
        
        # Volume-based adjustments
        tick_volume = session_info.get('expected_tick_volume', 100000)
        if tick_volume > 1000000:  # High volume
            session_config.performance.batch_processing_size = 200
            session_config.event_filter.event_rate_limits['prediction_generated'] = 1.0
        elif tick_volume < 10000:  # Low volume
            session_config.performance.batch_processing_size = 50
            session_config.event_filter.event_rate_limits['prediction_generated'] = 0.0
        
        # Phase-specific adjustments
        if session_info.get('learning_phase', False):
            session_config.event_filter.enabled_event_types.extend([
                'learning_progress',
                'model_training',
                'validation_complete'
            ])
            session_config.display.show_progress_bar = True
        
        if session_info.get('production_mode', False):
            session_config.display.terminal_mode = TerminalMode.SILENT
            session_config.event_filter.min_severity = EventSeverity.WARNING
            session_config.storage.flush_interval_seconds = 30.0
        
        return session_config
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Genera report delle performance e aggiustamenti
        
        Returns:
            Dict: Report completo
        """
        if not self.performance_history:
            return {'status': 'no_data', 'message': 'No performance data available'}
        
        # Calculate averages
        recent_metrics = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        
        avg_metrics = {}
        for metric_name in ['memory_usage_percent', 'cpu_usage_percent', 'event_processing_delay_ms', 'storage_write_delay_ms']:
            values = [m['metrics'].get(metric_name, 0) for m in recent_metrics if metric_name in m['metrics']]
            avg_metrics[metric_name] = sum(values) / len(values) if values else 0
        
        # Performance status
        status = 'optimal'
        issues = []
        
        if self.auto_tuning_enabled:
            for metric_name, target_value in self.performance_target.items():
                current_value = avg_metrics.get(metric_name, 0)
                if current_value > target_value * 1.2:  # 20% tolerance
                    status = 'degraded'
                    issues.append(f"{metric_name}: {current_value:.1f} > {target_value:.1f}")
        
        return {
            'status': status,
            'average_metrics': avg_metrics,
            'performance_targets': self.performance_target if self.auto_tuning_enabled else None,
            'issues': issues,
            'total_adjustments': len(self.runtime_adjustments),
            'recent_adjustments': list(self.runtime_adjustments.values())[-5:] if self.runtime_adjustments else [],
            'auto_tuning_enabled': self.auto_tuning_enabled,
            'measurement_count': len(self.performance_history)
        }


class ConfigValidationError(Exception):
    """Eccezione per errori di validazione configurazione"""
    pass


class ConfigProfiler:
    """
    Profiling e analisi della configurazione
    """
    
    @staticmethod
    def analyze_config_impact(config: MLTrainingLoggerConfig) -> Dict[str, Any]:
        """
        Analizza l'impatto della configurazione su performance
        
        Args:
            config: Configurazione da analizzare
            
        Returns:
            Dict: Analisi impatto
        """
        analysis = {
            'estimated_memory_usage_mb': 0,
            'estimated_cpu_overhead_percent': 0,
            'estimated_disk_usage_mb_per_hour': 0,
            'performance_level': 'unknown',
            'recommendations': []
        }
        
        # Memory usage estimation
        base_memory = 50  # Base overhead
        base_memory += config.performance.event_queue_size * 0.001  # ~1KB per event
        base_memory += config.storage.buffer_size * 0.002  # ~2KB per buffered event
        if config.performance.enable_async_processing:
            base_memory += config.performance.max_worker_threads * 10  # Thread overhead
        
        analysis['estimated_memory_usage_mb'] = base_memory
        
        # CPU overhead estimation
        cpu_overhead = 5  # Base overhead
        if config.display.terminal_mode == TerminalMode.SCROLL:
            cpu_overhead += 1 / config.display.refresh_rate_seconds  # More frequent updates = more CPU
        if config.performance.enable_async_processing:
            cpu_overhead -= 1  # Async reduces blocking
        cpu_overhead += len(config.event_filter.enabled_event_types) * 0.1  # Event processing overhead
        
        analysis['estimated_cpu_overhead_percent'] = min(cpu_overhead, 20)  # Cap at 20%
        
        # Disk usage estimation
        events_per_hour = 3600 / max(config.storage.flush_interval_seconds, 1)  # Rough estimate
        avg_event_size_kb = 1 if OutputFormat.JSON in config.storage.output_formats else 0.5
        disk_usage_mb = (events_per_hour * avg_event_size_kb) / 1024
        
        analysis['estimated_disk_usage_mb_per_hour'] = disk_usage_mb
        
        # Performance level assessment
        if analysis['estimated_memory_usage_mb'] < 100 and analysis['estimated_cpu_overhead_percent'] < 5:
            analysis['performance_level'] = 'high'
        elif analysis['estimated_memory_usage_mb'] < 200 and analysis['estimated_cpu_overhead_percent'] < 10:
            analysis['performance_level'] = 'medium'
        else:
            analysis['performance_level'] = 'low'
        
        # Recommendations
        if analysis['estimated_memory_usage_mb'] > 200:
            analysis['recommendations'].append("Consider reducing event_queue_size or buffer_size to lower memory usage")
        
        if analysis['estimated_cpu_overhead_percent'] > 10:
            analysis['recommendations'].append("Consider increasing refresh_rate_seconds or reducing enabled_event_types")
        
        if config.display.terminal_mode == TerminalMode.SCROLL and analysis['estimated_cpu_overhead_percent'] > 15:
            analysis['recommendations'].append("Consider switching to SCROLL mode for better performance")
        
        if not config.performance.enable_async_processing and len(config.event_filter.enabled_event_types) > 8:
            analysis['recommendations'].append("Consider enabling async_processing for better throughput")
        
        return analysis
    
    @staticmethod
    def validate_config_compatibility(config: MLTrainingLoggerConfig, 
                                    system_constraints: Dict[str, Any]) -> List[str]:
        """
        Valida compatibilità configurazione con vincoli sistema
        
        Args:
            config: Configurazione da validare
            system_constraints: Vincoli sistema (memory_mb, cpu_cores, etc.)
            
        Returns:
            List[str]: Lista di errori/warning di compatibilità
        """
        issues = []
        
        # Memory constraints
        max_memory = system_constraints.get('memory_mb', float('inf'))
        estimated_memory = ConfigProfiler.analyze_config_impact(config)['estimated_memory_usage_mb']
        
        if estimated_memory > max_memory:
            issues.append(f"Estimated memory usage ({estimated_memory:.1f}MB) exceeds constraint ({max_memory}MB)")
        
        # CPU constraints
        if config.performance.max_worker_threads > system_constraints.get('cpu_cores', 1):
            issues.append(f"Worker threads ({config.performance.max_worker_threads}) exceeds available CPU cores")
        
        # Disk constraints
        max_disk_mb = system_constraints.get('max_disk_usage_mb_per_hour', float('inf'))
        estimated_disk = ConfigProfiler.analyze_config_impact(config)['estimated_disk_usage_mb_per_hour']
        
        if estimated_disk > max_disk_mb:
            issues.append(f"Estimated disk usage ({estimated_disk:.1f}MB/h) exceeds constraint ({max_disk_mb}MB/h)")
        
        # Terminal capabilities
        if (config.display.terminal_mode == TerminalMode.SCROLL and 
            not system_constraints.get('supports_ansi', True)):
            issues.append("SCROLL mode requires ANSI terminal support")
        
        if (config.display.color_enabled and 
            not system_constraints.get('supports_colors', True)):
            issues.append("Color output requires color terminal support")
        
        return issues


# Configuration Templates for specific use cases
class ConfigTemplates:
    """Templates di configurazione per casi d'uso specifici"""
    
    @staticmethod
    def for_quick_testing() -> MLTrainingLoggerConfig:
        """Configurazione per test rapidi"""
        config = create_minimal_config()
        config.display.refresh_rate_seconds = 5.0
        config.storage.flush_interval_seconds = 10.0
        config.performance.enable_async_processing = False
        return config
    
    @staticmethod
    def for_development() -> MLTrainingLoggerConfig:
        """Configurazione per sviluppo"""
        config = create_debug_config()
        config.display.terminal_mode = TerminalMode.SCROLL
        config.storage.json_indent = 4
        config.event_filter.event_rate_limits = {}  # No rate limiting in dev
        return config
    
    @staticmethod
    def for_research_analysis() -> MLTrainingLoggerConfig:
        """Configurazione per analisi di ricerca"""
        config = create_verbose_config()
        config.storage.output_formats = [OutputFormat.JSON, OutputFormat.CSV]
        config.storage.enable_rotation = True
        config.display.max_recent_events = 50
        return config
    
    @staticmethod
    def for_production_monitoring() -> MLTrainingLoggerConfig:
        """Configurazione per monitoraggio produzione"""
        config = create_production_config()
        config.event_filter.min_severity = EventSeverity.WARNING
        config.storage.flush_interval_seconds = 30.0
        config.performance.emergency_fallback_enabled = True
        return config
    
    @staticmethod
    def for_high_frequency_trading() -> MLTrainingLoggerConfig:
        """Configurazione per trading ad alta frequenza"""
        config = create_minimal_config()
        config.display.terminal_mode = TerminalMode.SILENT
        config.event_filter.enabled_event_types = ['emergency_stop', 'critical_error']
        config.performance.enable_async_processing = True
        config.performance.max_worker_threads = 1  # Minimize resource usage
        config.storage.flush_interval_seconds = 60.0
        return config


# Export main classes and functions
__all__ = [
    'MLTrainingLoggerConfig',
    'VerbosityLevel', 
    'OutputFormat',
    'TerminalMode',
    'EventSeverity',
    'DisplaySettings',
    'StorageSettings', 
    'EventFilterSettings',
    'PerformanceSettings',
    'IntegrationSettings',
    'AdvancedConfigManager',
    'ConfigProfiler',
    'ConfigTemplates',
    'ConfigValidationError',
    'create_minimal_config',
    'create_standard_config',
    'create_verbose_config', 
    'create_debug_config',
    'create_production_config'
]