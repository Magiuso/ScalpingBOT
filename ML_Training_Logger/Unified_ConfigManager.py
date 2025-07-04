"""
Unified ConfigManager - ML Training Logger
==========================================

Sistema di configurazione centralizzato e intelligente per il ML_Training_Logger.
Unifica tutte le configurazioni sparse del sistema fornendo:

- 4 Livelli di verbosità (MINIMAL, STANDARD, VERBOSE, DEBUG)
- Timing settings ottimizzati (refresh_rate, flush_interval, buffer_size)
- Output settings flessibili (terminal_mode, file_output, formats)
- Profili predefiniti per ogni scenario d'uso
- Integrazione perfetta con UnifiedConfig, LoggingConfig, AnalyzerConfig

Usage:
    # Quick setup con profilo predefinito
    config = UnifiedConfigManager.create_production_config("EURUSD")
    
    # Setup avanzato con customizzazione
    config = UnifiedConfigManager.create_custom_config(
        verbosity=ConfigVerbosity.VERBOSE,
        timing_preset=ConfigTimingPreset.HIGH_FREQUENCY,
        output_preset=ConfigOutputPreset.FULL_LOGGING
    )
    
    # Applicazione al sistema
    unified_system = UnifiedAnalyzerSystem(config.get_unified_config())
    logging_slave = create_logging_slave(config.get_logging_config())
    analyzer = Analyzer(config.get_analyzer_config())
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging


# ================================
# CORE ENUMS AND TYPES
# ================================

class ConfigVerbosity(Enum):
    """4 livelli di verbosità principali del sistema ConfigManager"""
    MINIMAL = "minimal"      # Solo errori critici e champion changes
    STANDARD = "standard"    # Eventi importanti + summary periodici
    VERBOSE = "verbose"      # Tutti gli eventi + diagnostics
    DEBUG = "debug"          # Debug completo + timing dettagliato


class ConfigTimingPreset(Enum):
    """Preset di timing ottimizzati per diversi scenari"""
    HIGH_FREQUENCY = "high_frequency"    # Trading ultra-rapido
    NORMAL_TRADING = "normal_trading"    # Trading standard
    RESEARCH_MODE = "research_mode"      # Backtesting e ricerca
    DEMO_MODE = "demo_mode"              # Presentazioni e demo


class ConfigOutputPreset(Enum):
    """Preset di output per diversi use cases"""
    PRODUCTION = "production"      # Minimal console, file ottimizzati
    DEVELOPMENT = "development"    # Rich console, tutti i formati
    MONITORING = "monitoring"      # Focus su metriche e performance
    FULL_LOGGING = "full_logging"  # Tutto attivato per debug


class ConfigSystemProfile(Enum):
    """Profili di sistema completi predefiniti"""
    PRODUCTION_TRADING = "production_trading"
    DEVELOPMENT_TESTING = "development_testing"
    RESEARCH_ANALYSIS = "research_analysis"
    DEMO_SHOWCASE = "demo_showcase"
    MONITORING_ONLY = "monitoring_only"


# ================================
# TIMING CONFIGURATIONS
# ================================

@dataclass
class ConfigTimingSettings:
    """Configurazioni di timing ottimizzate"""
    
    # Core timing intervals
    refresh_rate: float = 1.0           # Secondi tra refresh display
    flush_interval: float = 5.0         # Secondi tra flush su disco
    event_processing_interval: float = 2.0  # Processing eventi accumulati
    
    # Buffer configurations
    buffer_size: int = 1000            # Dimensione buffer principale
    max_queue_size: int = 10000        # Coda massima eventi
    batch_size: int = 50               # Eventi per batch di processing
    
    # Performance thresholds
    max_processing_time: float = 10.0  # Tempo massimo processing (sec)
    performance_report_interval: float = 60.0  # Report performance (sec)
    
    # Rate limiting intervals
    tick_log_interval: int = 100       # Log ogni N ticks
    prediction_log_interval: int = 50  # Log ogni N predizioni
    training_log_interval: int = 1     # Log ogni N training events
    
    @classmethod
    def for_high_frequency(cls) -> 'ConfigTimingSettings':
        """Configurazione ottimizzata per high frequency trading"""
        return cls(
            refresh_rate=0.1,              # Display ultra-rapido
            flush_interval=10.0,           # Flush meno frequente
            event_processing_interval=5.0, # Processing batch più raro
            buffer_size=5000,              # Buffer più grandi
            max_queue_size=50000,          # Code enormi
            batch_size=100,                # Batch più grandi
            tick_log_interval=1000,        # Log meno frequente
            prediction_log_interval=500,
            performance_report_interval=300.0
        )
    
    @classmethod
    def for_normal_trading(cls) -> 'ConfigTimingSettings':
        """Configurazione bilanciata per trading normale"""
        return cls(
            refresh_rate=1.0,
            flush_interval=5.0,
            event_processing_interval=2.0,
            buffer_size=1000,
            max_queue_size=10000,
            batch_size=50,
            tick_log_interval=100,
            prediction_log_interval=50,
            performance_report_interval=60.0
        )
    
    @classmethod
    def for_research_mode(cls) -> 'ConfigTimingSettings':
        """Configurazione per ricerca e backtesting"""
        return cls(
            refresh_rate=2.0,              # Display meno frequente
            flush_interval=1.0,            # Flush frequente per sicurezza
            event_processing_interval=1.0, # Processing frequente per debug
            buffer_size=500,               # Buffer più piccoli
            max_queue_size=5000,
            batch_size=25,                 # Batch piccoli per feedback
            tick_log_interval=10,          # Log molto frequente
            prediction_log_interval=5,
            performance_report_interval=30.0
        )
    
    @classmethod
    def for_demo_mode(cls) -> 'ConfigTimingSettings':
        """Configurazione per demo e presentazioni"""
        return cls(
            refresh_rate=0.5,              # Display molto frequente
            flush_interval=2.0,
            event_processing_interval=1.0,
            buffer_size=200,               # Buffer piccoli per reattività
            max_queue_size=2000,
            batch_size=10,                 # Batch piccolissimi
            tick_log_interval=1,           # Log quasi tutto
            prediction_log_interval=1,
            performance_report_interval=15.0
        )


# ================================
# OUTPUT CONFIGURATIONS
# ================================

@dataclass
class ConfigOutputSettings:
    """Configurazioni di output complete"""
    
    # Terminal/Console settings
    terminal_mode: str = "rich"         # "simple", "rich", "minimal", "disabled"
    console_colors: bool = True         # Colori nel terminale
    console_timestamps: bool = True     # Timestamp nei messaggi
    console_progress_bars: bool = True  # Progress bar animate
    
    # File output settings
    file_output: bool = True           # Abilitazione file logging
    log_rotation: str = "monthly"      # "daily", "weekly", "monthly", "size"
    max_log_files: int = 30           # File di log da mantenere
    compress_old_logs: bool = True     # Compressione log vecchi
    
    # Export formats
    formats: Dict[str, bool] = field(default_factory=lambda: {
        'csv': True,           # Export CSV per analisi
        'json': False,         # Export JSON per API
        'parquet': False,      # Export Parquet per big data
        'sqlite': False,       # Database SQLite locale
        'real_time_feed': False  # Feed real-time per monitoring
    })
    
    # Advanced output options
    enable_metrics_dashboard: bool = False    # Dashboard web real-time
    enable_telegram_alerts: bool = False      # Alert via Telegram
    enable_email_reports: bool = False        # Report periodici via email
    
    # File paths
    base_directory: str = "./ml_logger_output"
    log_file_prefix: str = "ml_training_logger"
    export_directory: str = "./exports"
    
    @classmethod
    def for_production(cls) -> 'ConfigOutputSettings':
        """Output ottimizzato per produzione"""
        return cls(
            terminal_mode="minimal",
            console_colors=False,
            console_progress_bars=False,
            file_output=True,
            log_rotation="daily",
            compress_old_logs=True,
            formats={
                'csv': True,
                'json': False,
                'parquet': False,
                'sqlite': False,
                'real_time_feed': False
            },
            enable_metrics_dashboard=False,
            enable_telegram_alerts=True,  # Solo alert critici
            base_directory="./production_logs"
        )
    
    @classmethod
    def for_development(cls) -> 'ConfigOutputSettings':
        """Output completo per sviluppo"""
        return cls(
            terminal_mode="rich",
            console_colors=True,
            console_progress_bars=True,
            file_output=True,
            log_rotation="weekly",
            compress_old_logs=False,
            formats={
                'csv': True,
                'json': True,
                'parquet': False,
                'sqlite': True,
                'real_time_feed': False
            },
            enable_metrics_dashboard=True,
            base_directory="./development_logs"
        )
    
    @classmethod
    def for_monitoring(cls) -> 'ConfigOutputSettings':
        """Output focalizzato su monitoring"""
        return cls(
            terminal_mode="simple",
            console_colors=True,
            console_timestamps=True,
            console_progress_bars=False,
            file_output=True,
            log_rotation="size",  # Rotazione per dimensione
            formats={
                'csv': True,
                'json': True,
                'parquet': True,  # Per analisi big data
                'sqlite': True,
                'real_time_feed': True
            },
            enable_metrics_dashboard=True,
            enable_telegram_alerts=True,
            enable_email_reports=True,
            base_directory="./monitoring_logs"
        )
    
    @classmethod
    def for_full_logging(cls) -> 'ConfigOutputSettings':
        """Output completo - tutti i formati attivati"""
        return cls(
            terminal_mode="rich",
            console_colors=True,
            console_timestamps=True,
            console_progress_bars=True,
            file_output=True,
            log_rotation="daily",
            compress_old_logs=True,
            formats={
                'csv': True,
                'json': True,
                'parquet': True,
                'sqlite': True,
                'real_time_feed': True
            },
            enable_metrics_dashboard=True,
            enable_telegram_alerts=True,
            enable_email_reports=True,
            base_directory="./full_logging"
        )


# ================================
# RATE LIMITING CONFIGURATIONS
# ================================

@dataclass
class ConfigRateLimitSettings:
    """Configurazioni di rate limiting intelligenti"""
    
    # Core event limits (log ogni N eventi)
    tick_processing: int = 100
    predictions: int = 50
    validations: int = 25
    training_events: int = 1
    champion_changes: int = 1
    emergency_events: int = 1
    
    # Advanced event limits
    diagnostics: int = 1000
    performance_metrics: int = 500
    memory_checks: int = 100
    model_updates: int = 10
    
    # Dynamic rate limiting
    adaptive_limiting: bool = True      # Adatta automaticamente i limiti
    burst_allowance: int = 5           # Eventi burst consentiti
    cooldown_period: float = 60.0      # Periodo di cooldown dopo burst
    
    @classmethod
    def for_verbosity_level(cls, level: ConfigVerbosity) -> 'ConfigRateLimitSettings':
        """Crea rate limits ottimizzati per livello di verbosità"""
        
        if level == ConfigVerbosity.MINIMAL:
            return cls(
                tick_processing=10000,     # Log molto raro
                predictions=1000,
                validations=500,
                diagnostics=50000,
                performance_metrics=5000,
                adaptive_limiting=True
            )
        
        elif level == ConfigVerbosity.STANDARD:
            return cls(
                tick_processing=1000,
                predictions=100,
                validations=50,
                diagnostics=5000,
                performance_metrics=1000,
                adaptive_limiting=True
            )
        
        elif level == ConfigVerbosity.VERBOSE:
            return cls(
                tick_processing=100,
                predictions=25,
                validations=10,
                diagnostics=1000,
                performance_metrics=200,
                adaptive_limiting=True
            )
        
        else:  # DEBUG
            return cls(
                tick_processing=10,        # Log quasi tutto
                predictions=5,
                validations=1,
                diagnostics=100,
                performance_metrics=50,
                adaptive_limiting=False    # No limiting in debug
            )


# ================================
# MAIN UNIFIED CONFIG MANAGER
# ================================

@dataclass
class UnifiedConfigManager:
    """
    Manager centralizzato per tutte le configurazioni del sistema ML Training Logger.
    
    Fornisce un'interfaccia unificata per gestire:
    - Livelli di verbosità
    - Timing ottimizzato 
    - Output configurabile
    - Integrazione con tutti i moduli esistenti
    """
    
    # Core settings
    verbosity_level: ConfigVerbosity = ConfigVerbosity.STANDARD
    timing_settings: ConfigTimingSettings = field(default_factory=ConfigTimingSettings.for_normal_trading)
    output_settings: ConfigOutputSettings = field(default_factory=ConfigOutputSettings.for_development)
    rate_limit_settings: ConfigRateLimitSettings = field(default_factory=lambda: ConfigRateLimitSettings.for_verbosity_level(ConfigVerbosity.STANDARD))
    
    # System identification
    asset_symbol: str = "EURUSD"
    system_mode: str = "development"  # "production", "development", "testing", "demo"
    config_version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    
    # Integration settings
    enable_unified_system: bool = True     # Usa UnifiedAnalyzerSystem
    enable_logging_slave: bool = True      # Usa logging slave separato
    enable_analyzer_integration: bool = True  # Integrazione con Analyzer esistente
    
    def __post_init__(self):
        """Validazione e setup automatico post-inizializzazione"""
        self._validate_configuration()
        self._auto_adjust_settings()
        self._setup_directories()
    
    def _validate_configuration(self) -> None:
        """Valida che la configurazione sia consistente"""
        
        # Valida timing settings
        assert self.timing_settings.refresh_rate > 0, "Refresh rate deve essere positivo"
        assert self.timing_settings.flush_interval > 0, "Flush interval deve essere positivo"
        assert self.timing_settings.buffer_size > 0, "Buffer size deve essere positivo"
        
        # Valida rate limits
        assert self.rate_limit_settings.tick_processing > 0, "Tick processing rate deve essere positivo"
        assert self.rate_limit_settings.predictions > 0, "Predictions rate deve essere positivo"
        
        # Valida output settings
        assert len(self.output_settings.base_directory) > 0, "Base directory deve essere specificata"
        
    def _auto_adjust_settings(self) -> None:
        """Aggiustamenti automatici intelligenti tra i settings"""
        
        # Auto-adjust rate limits based on verbosity
        if self.verbosity_level != ConfigVerbosity.DEBUG:
            self.rate_limit_settings = ConfigRateLimitSettings.for_verbosity_level(self.verbosity_level)
        
        # Auto-adjust timing based on verbosity for better performance
        if self.verbosity_level == ConfigVerbosity.MINIMAL:
            # Minimize overhead for minimal logging
            self.timing_settings.flush_interval = max(self.timing_settings.flush_interval, 10.0)
            self.timing_settings.event_processing_interval = max(self.timing_settings.event_processing_interval, 5.0)
        
        elif self.verbosity_level == ConfigVerbosity.DEBUG:
            # Maximize responsiveness for debug
            self.timing_settings.flush_interval = min(self.timing_settings.flush_interval, 2.0)
            self.timing_settings.event_processing_interval = min(self.timing_settings.event_processing_interval, 1.0)
    
    def _setup_directories(self) -> None:
        """Crea le directory necessarie se non esistono"""
        try:
            base_dir = Path(self.output_settings.base_directory)
            base_dir.mkdir(parents=True, exist_ok=True)
            
            export_dir = Path(self.output_settings.export_directory) 
            export_dir.mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not create directories: {e}")

    # ================================
    # INTEGRATION METHODS
    # ================================

    def get_unified_config(self) -> Dict[str, Any]:
        """Genera UnifiedConfig compatibile con UnifiedAnalyzerSystem"""
        
        # Map verbosity to system modes
        system_mode_map = {
            ConfigVerbosity.MINIMAL: "production",
            ConfigVerbosity.STANDARD: "development", 
            ConfigVerbosity.VERBOSE: "testing",
            ConfigVerbosity.DEBUG: "demo"
        }
        
        # Map timing to performance profiles
        performance_profile = "normal"
        if hasattr(self.timing_settings, '__class__'):
            if self.timing_settings.refresh_rate <= 0.5:
                performance_profile = "high_frequency"
            elif self.timing_settings.refresh_rate >= 2.0:
                performance_profile = "research"
        
        # Build complete unified config
        unified_config = {
            # System settings
            'system_mode': system_mode_map.get(self.verbosity_level, "development"),
            'performance_profile': performance_profile,
            'asset_symbol': self.asset_symbol,
            
            # Logging settings mapped from verbosity
            'log_level': self.verbosity_level.value.upper(),
            'enable_console_output': self.output_settings.terminal_mode != "disabled",
            'enable_file_output': self.output_settings.file_output,
            'enable_csv_export': self.output_settings.formats.get('csv', True),
            'enable_json_export': self.output_settings.formats.get('json', False),
            
            # Rate limiting from our intelligent system
            'rate_limits': {
                'tick_processing': self.rate_limit_settings.tick_processing,
                'predictions': self.rate_limit_settings.predictions,
                'validations': self.rate_limit_settings.validations,
                'training_events': self.rate_limit_settings.training_events,
                'champion_changes': self.rate_limit_settings.champion_changes,
                'emergency_events': self.rate_limit_settings.emergency_events,
                'diagnostics': self.rate_limit_settings.diagnostics
            },
            
            # Performance settings from timing
            'event_processing_interval': self.timing_settings.event_processing_interval,
            'batch_size': self.timing_settings.batch_size,
            'max_queue_size': self.timing_settings.max_queue_size,
            'async_processing': True,
            'max_workers': 2,
            
            # Storage settings
            'base_directory': self.output_settings.base_directory,
            'log_rotation_hours': 24 if self.output_settings.log_rotation == "daily" else 168,
            'max_log_files': self.output_settings.max_log_files,
            'compress_old_logs': self.output_settings.compress_old_logs,
            
            # Monitoring settings
            'enable_performance_monitoring': True,
            'performance_report_interval': self.timing_settings.performance_report_interval,
            'memory_threshold_mb': 1000,
            'cpu_threshold_percent': 80.0
        }
        
        return unified_config
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Genera LoggingConfig compatibile con Analyzer_Logging_SlaveModule"""
        
        # Map verbosity to LogLevel enum values  
        log_level_map = {
            ConfigVerbosity.MINIMAL: "minimal",
            ConfigVerbosity.STANDARD: "normal",
            ConfigVerbosity.VERBOSE: "verbose", 
            ConfigVerbosity.DEBUG: "debug"
        }
        
        logging_config = {
            # Core logging level
            'log_level': log_level_map[self.verbosity_level],
            
            # Rate limiting perfettamente mappato
            'rate_limits': {
                'process_tick': self.rate_limit_settings.tick_processing,
                'predictions': self.rate_limit_settings.predictions,
                'validations': self.rate_limit_settings.validations,
                'training_progress': self.rate_limit_settings.training_events,
                'diagnostics': self.rate_limit_settings.diagnostics,
                'champion_changes': self.rate_limit_settings.champion_changes,
                'emergency_events': self.rate_limit_settings.emergency_events
            },
            
            # Output configuration
            'enable_console_output': self.output_settings.terminal_mode != "disabled",
            'enable_file_output': self.output_settings.file_output,
            'enable_csv_export': self.output_settings.formats.get('csv', True),
            'enable_json_export': self.output_settings.formats.get('json', False),
            
            # Aggregation settings from timing
            'batch_size': self.timing_settings.batch_size,
            'batch_interval': self.timing_settings.event_processing_interval,
            'max_queue_size': self.timing_settings.max_queue_size,
            
            # Advanced settings
            'enable_adaptive_rate_limiting': self.rate_limit_settings.adaptive_limiting,
            'burst_allowance': self.rate_limit_settings.burst_allowance,
            'cooldown_period': self.rate_limit_settings.cooldown_period
        }
        
        return logging_config
    
    def get_analyzer_config(self) -> Dict[str, Any]:
        """Genera AnalyzerConfig compatibile con Analyzer principale"""
        
        analyzer_config = {
            # Asset settings
            'asset': self.asset_symbol,
            'max_tick_buffer_size': self.timing_settings.buffer_size * 100,  # Scala per analyzer
            
            # Learning settings ottimizzate per verbosity
            'min_learning_days': 7 if self.verbosity_level in [ConfigVerbosity.MINIMAL, ConfigVerbosity.STANDARD] else 3,
            'learning_phase_enabled': True,
            
            # Performance settings
            'max_processing_time': self.timing_settings.max_processing_time,
            'performance_window_size': 100,
            'feature_vector_size': 10,
            
            # Logging integration
            'log_milestone_interval': self.rate_limit_settings.tick_processing,
            'log_level_system': "INFO" if self.verbosity_level != ConfigVerbosity.DEBUG else "DEBUG",
            'log_level_errors': "ERROR",
            'log_level_predictions': "DEBUG" if self.verbosity_level == ConfigVerbosity.DEBUG else "INFO",
            
            # Rate limiting per analyzer
            'diagnostics_monitor_interval': max(30, self.timing_settings.performance_report_interval // 2),
            
            # Model settings ottimizzate per performance
            'lstm_hidden_size': 128 if self.verbosity_level == ConfigVerbosity.MINIMAL else 256,
            'lstm_num_layers': 2 if self.verbosity_level == ConfigVerbosity.MINIMAL else 3,
            'lstm_dropout': 0.1,
            
            # Training settings adaptive
            'training_test_split': 0.8,
            'champion_threshold': 0.65,
            'accuracy_threshold': 0.6,
            
            # Validation criteria adaptive
            'validation_default_minutes': 60 if self.verbosity_level == ConfigVerbosity.MINIMAL else 30,
            'validation_sr_ticks': 1000 if self.verbosity_level == ConfigVerbosity.MINIMAL else 500,
            'validation_pattern_ticks': 500 if self.verbosity_level == ConfigVerbosity.MINIMAL else 250
        }
        
        return analyzer_config

# ================================
    # FACTORY METHODS - SYSTEM PROFILES
    # ================================

    @classmethod
    def create_production_config(cls, asset_symbol: str = "EURUSD") -> 'UnifiedConfigManager':
        """Configurazione ottimizzata per produzione trading reale"""
        return cls(
            verbosity_level=ConfigVerbosity.MINIMAL,
            timing_settings=ConfigTimingSettings.for_high_frequency(),
            output_settings=ConfigOutputSettings.for_production(),
            rate_limit_settings=ConfigRateLimitSettings.for_verbosity_level(ConfigVerbosity.MINIMAL),
            asset_symbol=asset_symbol,
            system_mode="production",
            enable_unified_system=True,
            enable_logging_slave=True,
            enable_analyzer_integration=True
        )
    
    @classmethod 
    def create_development_config(cls, asset_symbol: str = "EURUSD") -> 'UnifiedConfigManager':
        """Configurazione completa per sviluppo e testing"""
        return cls(
            verbosity_level=ConfigVerbosity.VERBOSE,
            timing_settings=ConfigTimingSettings.for_research_mode(),
            output_settings=ConfigOutputSettings.for_development(),
            rate_limit_settings=ConfigRateLimitSettings.for_verbosity_level(ConfigVerbosity.VERBOSE),
            asset_symbol=asset_symbol,
            system_mode="development",
            enable_unified_system=True,
            enable_logging_slave=True,
            enable_analyzer_integration=True
        )
    
    @classmethod
    def create_research_config(cls, asset_symbol: str = "EURUSD") -> 'UnifiedConfigManager':
        """Configurazione per ricerca e backtesting"""
        return cls(
            verbosity_level=ConfigVerbosity.VERBOSE,
            timing_settings=ConfigTimingSettings.for_research_mode(),
            output_settings=ConfigOutputSettings.for_full_logging(),
            rate_limit_settings=ConfigRateLimitSettings.for_verbosity_level(ConfigVerbosity.VERBOSE),
            asset_symbol=asset_symbol,
            system_mode="testing",
            enable_unified_system=True,
            enable_logging_slave=True,
            enable_analyzer_integration=True
        )
    
    @classmethod
    def create_demo_config(cls, asset_symbol: str = "EURUSD") -> 'UnifiedConfigManager':
        """Configurazione per demo e presentazioni"""
        return cls(
            verbosity_level=ConfigVerbosity.DEBUG,
            timing_settings=ConfigTimingSettings.for_demo_mode(),
            output_settings=ConfigOutputSettings.for_full_logging(),
            rate_limit_settings=ConfigRateLimitSettings.for_verbosity_level(ConfigVerbosity.DEBUG),
            asset_symbol=asset_symbol,
            system_mode="demo",
            enable_unified_system=True,
            enable_logging_slave=True,
            enable_analyzer_integration=True
        )
    
    @classmethod
    def create_monitoring_config(cls, asset_symbol: str = "EURUSD") -> 'UnifiedConfigManager':
        """Configurazione focalizzata su monitoring e metriche"""
        return cls(
            verbosity_level=ConfigVerbosity.STANDARD,
            timing_settings=ConfigTimingSettings.for_normal_trading(),
            output_settings=ConfigOutputSettings.for_monitoring(),
            rate_limit_settings=ConfigRateLimitSettings.for_verbosity_level(ConfigVerbosity.STANDARD),
            asset_symbol=asset_symbol,
            system_mode="production",
            enable_unified_system=True,
            enable_logging_slave=True,
            enable_analyzer_integration=True
        )
    
    @classmethod
    def create_custom_config(
        cls,
        asset_symbol: str = "EURUSD",
        verbosity: ConfigVerbosity = ConfigVerbosity.STANDARD,
        timing_preset: ConfigTimingPreset = ConfigTimingPreset.NORMAL_TRADING,
        output_preset: ConfigOutputPreset = ConfigOutputPreset.DEVELOPMENT,
        **kwargs
    ) -> 'UnifiedConfigManager':
        """Factory method per configurazioni completamente personalizzate"""
        
        # Map timing preset to settings
        timing_map = {
            ConfigTimingPreset.HIGH_FREQUENCY: ConfigTimingSettings.for_high_frequency(),
            ConfigTimingPreset.NORMAL_TRADING: ConfigTimingSettings.for_normal_trading(),
            ConfigTimingPreset.RESEARCH_MODE: ConfigTimingSettings.for_research_mode(),
            ConfigTimingPreset.DEMO_MODE: ConfigTimingSettings.for_demo_mode()
        }
        
        # Map output preset to settings
        output_map = {
            ConfigOutputPreset.PRODUCTION: ConfigOutputSettings.for_production(),
            ConfigOutputPreset.DEVELOPMENT: ConfigOutputSettings.for_development(),
            ConfigOutputPreset.MONITORING: ConfigOutputSettings.for_monitoring(),
            ConfigOutputPreset.FULL_LOGGING: ConfigOutputSettings.for_full_logging()
        }
        
        return cls(
            verbosity_level=verbosity,
            timing_settings=timing_map[timing_preset],
            output_settings=output_map[output_preset],
            rate_limit_settings=ConfigRateLimitSettings.for_verbosity_level(verbosity),
            asset_symbol=asset_symbol,
            **kwargs
        )

    # ================================
    # RUNTIME MANAGEMENT
    # ================================

    def update_verbosity(self, new_level: ConfigVerbosity) -> None:
        """Aggiorna il livello di verbosità runtime con auto-adjustment"""
        old_level = self.verbosity_level
        self.verbosity_level = new_level
        
        # Auto-adjust rate limits
        self.rate_limit_settings = ConfigRateLimitSettings.for_verbosity_level(new_level)
        
        # Auto-adjust timing for performance
        self._auto_adjust_settings()
        
        print(f"🔄 Verbosity updated: {old_level.value} → {new_level.value}")
        print(f"📊 Rate limits auto-adjusted for {new_level.value} mode")
    
    def update_timing_preset(self, preset: ConfigTimingPreset) -> None:
        """Aggiorna il preset di timing runtime"""
        timing_map = {
            ConfigTimingPreset.HIGH_FREQUENCY: ConfigTimingSettings.for_high_frequency(),
            ConfigTimingPreset.NORMAL_TRADING: ConfigTimingSettings.for_normal_trading(),
            ConfigTimingPreset.RESEARCH_MODE: ConfigTimingSettings.for_research_mode(),
            ConfigTimingPreset.DEMO_MODE: ConfigTimingSettings.for_demo_mode()
        }
        
        old_refresh = self.timing_settings.refresh_rate
        self.timing_settings = timing_map[preset]
        new_refresh = self.timing_settings.refresh_rate
        
        print(f"⏱️ Timing preset updated: {preset.value}")
        print(f"🔄 Refresh rate: {old_refresh}s → {new_refresh}s")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Ottieni summary delle performance della configurazione attuale"""
        return {
            'config_profile': {
                'verbosity': self.verbosity_level.value,
                'system_mode': self.system_mode,
                'asset': self.asset_symbol
            },
            'timing_performance': {
                'refresh_rate': self.timing_settings.refresh_rate,
                'event_processing_interval': self.timing_settings.event_processing_interval,
                'buffer_size': self.timing_settings.buffer_size,
                'expected_throughput_ticks_per_sec': int(1.0 / self.timing_settings.refresh_rate * self.timing_settings.buffer_size)
            },
            'rate_limiting': {
                'tick_log_frequency': f"1 ogni {self.rate_limit_settings.tick_processing} ticks",
                'prediction_log_frequency': f"1 ogni {self.rate_limit_settings.predictions} predizioni",
                'adaptive_enabled': self.rate_limit_settings.adaptive_limiting
            },
            'output_capabilities': {
                'terminal_mode': self.output_settings.terminal_mode,
                'file_output': self.output_settings.file_output,
                'active_formats': [fmt for fmt, enabled in self.output_settings.formats.items() if enabled],
                'advanced_features': {
                    'metrics_dashboard': self.output_settings.enable_metrics_dashboard,
                    'telegram_alerts': self.output_settings.enable_telegram_alerts,
                    'email_reports': self.output_settings.enable_email_reports
                }
            },
            'estimated_overhead': self._calculate_overhead_estimate()
        }
    
    def _calculate_overhead_estimate(self) -> Dict[str, str]:
        """Calcola stima dell'overhead di logging"""
        
        # Base overhead per verbosity level
        base_overhead = {
            ConfigVerbosity.MINIMAL: 0.1,    # 0.1% overhead
            ConfigVerbosity.STANDARD: 0.5,   # 0.5% overhead  
            ConfigVerbosity.VERBOSE: 2.0,    # 2% overhead
            ConfigVerbosity.DEBUG: 5.0       # 5% overhead
        }[self.verbosity_level]
        
        overhead_pct = base_overhead
        
        # Adjust per output formats
        format_count = sum(1 for enabled in self.output_settings.formats.values() if enabled)
        if format_count > 2:
            overhead_pct *= 1.2
        
        # Adjust per timing
        if self.timing_settings.refresh_rate < 1.0:
            overhead_pct *= 1.3
            
        return {
            'estimated_cpu_overhead': f"{overhead_pct:.1f}%",
            'estimated_memory_overhead': f"{overhead_pct * 2:.1f}MB per 1000 events",
            'performance_impact': "minimal" if overhead_pct < 1.0 else "low" if overhead_pct < 3.0 else "moderate"
        }

    # ================================
    # PERSISTENCE AND SERIALIZATION
    # ================================

    def save_to_file(self, filepath: str) -> bool:
        """Salva la configurazione su file JSON"""
        try:
            config_dict = {
                'metadata': {
                    'config_version': self.config_version,
                    'created_at': self.created_at.isoformat(),
                    'saved_at': datetime.now().isoformat(),
                    'asset_symbol': self.asset_symbol,
                    'system_mode': self.system_mode
                },
                'verbosity_level': self.verbosity_level.value,
                'timing_settings': asdict(self.timing_settings),
                'output_settings': asdict(self.output_settings),
                'rate_limit_settings': asdict(self.rate_limit_settings),
                'integration_settings': {
                    'enable_unified_system': self.enable_unified_system,
                    'enable_logging_slave': self.enable_logging_slave,
                    'enable_analyzer_integration': self.enable_analyzer_integration
                }
            }
            
            filepath_obj = Path(filepath)
            filepath_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Configuration saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'UnifiedConfigManager':
        """Carica configurazione da file JSON"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Extract core settings
            verbosity = ConfigVerbosity(config_data['verbosity_level'])
            timing_settings = ConfigTimingSettings(**config_data['timing_settings'])
            output_settings = ConfigOutputSettings(**config_data['output_settings'])
            rate_limit_settings = ConfigRateLimitSettings(**config_data['rate_limit_settings'])
            
            # Extract metadata
            metadata = config_data.get('metadata', {})
            integration = config_data.get('integration_settings', {})
            
            instance = cls(
                verbosity_level=verbosity,
                timing_settings=timing_settings,
                output_settings=output_settings,
                rate_limit_settings=rate_limit_settings,
                asset_symbol=metadata.get('asset_symbol', 'EURUSD'),
                system_mode=metadata.get('system_mode', 'development'),
                config_version=metadata.get('config_version', '1.0.0'),
                enable_unified_system=integration.get('enable_unified_system', True),
                enable_logging_slave=integration.get('enable_logging_slave', True),
                enable_analyzer_integration=integration.get('enable_analyzer_integration', True)
            )
            
            print(f"✅ Configuration loaded from {filepath}")
            return instance
            
        except FileNotFoundError:
            print(f"⚠️ Config file {filepath} not found, creating default configuration")
            return cls()
        except Exception as e:
            print(f"❌ Error loading configuration: {e}, using defaults")
            return cls()
    
    def export_to_multiple_formats(self, base_path: str) -> Dict[str, bool]:
        """Esporta la configurazione in formati multipli per compatibilità"""
        results = {}
        base_path_obj = Path(base_path)
        
        # JSON format (primary)
        json_path = base_path_obj.with_suffix('.json')
        results['json'] = self.save_to_file(str(json_path))
        
        # YAML format per human readability
        try:
            import yaml
            yaml_path = base_path_obj.with_suffix('.yaml')
            config_dict = self._to_export_dict()
            
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            results['yaml'] = True
            
        except ImportError:
            results['yaml'] = False
            print("⚠️ YAML export requires 'pyyaml' package")
        except Exception as e:
            results['yaml'] = False
            print(f"❌ YAML export failed: {e}")
        
        # INI format per legacy compatibility
        try:
            import configparser
            ini_path = base_path_obj.with_suffix('.ini')
            config = configparser.ConfigParser()
            
            config['METADATA'] = {
                'config_version': self.config_version,
                'asset_symbol': self.asset_symbol,
                'system_mode': self.system_mode,
                'verbosity_level': self.verbosity_level.value
            }
            
            config['TIMING'] = {
                'refresh_rate': str(self.timing_settings.refresh_rate),
                'flush_interval': str(self.timing_settings.flush_interval),
                'buffer_size': str(self.timing_settings.buffer_size)
            }
            
            config['OUTPUT'] = {
                'terminal_mode': self.output_settings.terminal_mode,
                'file_output': str(self.output_settings.file_output),
                'base_directory': self.output_settings.base_directory
            }
            
            with open(ini_path, 'w', encoding='utf-8') as f:
                config.write(f)
            results['ini'] = True
            
        except Exception as e:
            results['ini'] = False
            print(f"❌ INI export failed: {e}")
        
        return results
    
    def _to_export_dict(self) -> Dict[str, Any]:
        """Converte la configurazione in dizionario per export"""
        return {
            'metadata': {
                'config_version': self.config_version,
                'created_at': self.created_at.isoformat(),
                'exported_at': datetime.now().isoformat(),
                'asset_symbol': self.asset_symbol,
                'system_mode': self.system_mode
            },
            'verbosity': {
                'level': self.verbosity_level.value,
                'description': self._get_verbosity_description()
            },
            'timing': asdict(self.timing_settings),
            'output': asdict(self.output_settings),
            'rate_limits': asdict(self.rate_limit_settings),
            'integration': {
                'unified_system': self.enable_unified_system,
                'logging_slave': self.enable_logging_slave,
                'analyzer_integration': self.enable_analyzer_integration
            }
        }
    
    def _get_verbosity_description(self) -> str:
        """Ottieni descrizione del livello di verbosità"""
        descriptions = {
            ConfigVerbosity.MINIMAL: "Solo errori critici e champion changes - massima performance",
            ConfigVerbosity.STANDARD: "Eventi importanti + summary periodici - bilanciato",
            ConfigVerbosity.VERBOSE: "Tutti gli eventi + diagnostics - completo",
            ConfigVerbosity.DEBUG: "Debug completo + timing dettagliato - massima informazione"
        }
        return descriptions.get(self.verbosity_level, "Unknown verbosity level")

    # ================================
    # VALIDATION AND DIAGNOSTICS
    # ================================

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validazione completa della configurazione con dettagli errori"""
        errors = []
        
        # Validate timing settings
        if self.timing_settings.refresh_rate <= 0:
            errors.append("Refresh rate deve essere maggiore di 0")
        if self.timing_settings.refresh_rate > 60:
            errors.append("Refresh rate troppo alto (>60s), potrebbe causare ritardi")
        
        if self.timing_settings.flush_interval <= 0:
            errors.append("Flush interval deve essere maggiore di 0")
        if self.timing_settings.flush_interval < self.timing_settings.refresh_rate:
            errors.append("Flush interval dovrebbe essere >= refresh_rate per efficienza")
        
        if self.timing_settings.buffer_size < 100:
            errors.append("Buffer size troppo piccolo (<100), può causare perdita dati")
        if self.timing_settings.buffer_size > 1000000:
            errors.append("Buffer size troppo grande (>1M), può causare problemi di memoria")
        
        # Validate rate limits
        if self.rate_limit_settings.tick_processing <= 0:
            errors.append("Tick processing rate deve essere positivo")
        if self.rate_limit_settings.predictions <= 0:
            errors.append("Predictions rate deve essere positivo")
        
        # Cross-validation timing vs rate limits
        if self.verbosity_level == ConfigVerbosity.DEBUG and self.rate_limit_settings.tick_processing > 100:
            errors.append("Rate limiting troppo restrittivo per DEBUG mode")
        
        if self.verbosity_level == ConfigVerbosity.MINIMAL and self.rate_limit_settings.tick_processing < 1000:
            errors.append("Rate limiting troppo permissivo per MINIMAL mode")
        
        # Validate output settings
        if not self.output_settings.base_directory:
            errors.append("Base directory deve essere specificata")
        
        if not any(self.output_settings.formats.values()) and self.output_settings.file_output:
            errors.append("Almeno un formato di export deve essere abilitato se file_output=True")
        
        # Validate paths
        try:
            base_path = Path(self.output_settings.base_directory)
            if not base_path.parent.exists():
                errors.append(f"Directory parent {base_path.parent} non esiste")
        except Exception as e:
            errors.append(f"Errore validazione path: {e}")
        
        # Performance warnings
        warnings = self._get_performance_warnings()
        if warnings:
            errors.extend([f"WARNING: {w}" for w in warnings])
        
        return len(errors) == 0, errors
    
    def _get_performance_warnings(self) -> List[str]:
        """Ottieni warning sulle performance della configurazione"""
        warnings = []
        
        # High frequency + verbose logging warning
        if (self.timing_settings.refresh_rate < 1.0 and 
            self.verbosity_level in [ConfigVerbosity.VERBOSE, ConfigVerbosity.DEBUG]):
            warnings.append("Combinazione high-frequency + verbosity alta può impattare performance")
        
        # Multiple formats warning
        active_formats = sum(1 for enabled in self.output_settings.formats.values() if enabled)
        if active_formats > 3:
            warnings.append(f"Troppi formati attivi ({active_formats}) possono rallentare il sistema")
        
        # Large buffer + frequent flush warning
        if (self.timing_settings.buffer_size > 10000 and 
            self.timing_settings.flush_interval < 5.0):
            warnings.append("Buffer grandi + flush frequenti possono causare I/O intensivo")
        
        # Debug mode in production warning
        if self.verbosity_level == ConfigVerbosity.DEBUG and self.system_mode == "production":
            warnings.append("DEBUG mode non raccomandato in produzione")
        
        return warnings
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Esegue diagnostics completa della configurazione"""
        start_time = time.time()
        
        # Configuration validation
        is_valid, validation_errors = self.validate_configuration()
        
        # Performance analysis
        performance_summary = self.get_performance_summary()
        
        # Compatibility check
        compatibility = self._check_system_compatibility()
        
        # Resource estimation
        resource_estimate = self._estimate_resource_usage()
        
        # Integration test
        integration_status = self._test_integration_configs()
        
        diagnostics_time = time.time() - start_time
        
        return {
            'timestamp': datetime.now().isoformat(),
            'diagnostics_duration_ms': round(diagnostics_time * 1000, 2),
            'configuration_validation': {
                'is_valid': is_valid,
                'errors': validation_errors,
                'error_count': len(validation_errors)
            },
            'performance_analysis': performance_summary,
            'system_compatibility': compatibility,
            'resource_estimation': resource_estimate,
            'integration_status': integration_status,
            'recommendations': self._generate_recommendations(validation_errors, compatibility)
        }
    
    def _check_system_compatibility(self) -> Dict[str, Any]:
        """Verifica compatibilità con i sistemi esistenti"""
        compatibility = {
            'unified_analyzer_system': True,
            'logging_slave_module': True,
            'analyzer_main': True,
            'missing_dependencies': [],
            'version_conflicts': []
        }
        
        # Check UnifiedConfig compatibility
        try:
            unified_config = self.get_unified_config()
            required_fields = ['system_mode', 'asset_symbol', 'log_level', 'rate_limits']
            missing = [field for field in required_fields if field not in unified_config]
            if missing:
                compatibility['unified_analyzer_system'] = False
                compatibility['missing_dependencies'].extend(missing)
        except Exception as e:
            compatibility['unified_analyzer_system'] = False
            compatibility['version_conflicts'].append(f"UnifiedConfig generation failed: {e}")
        
        # Check LoggingConfig compatibility
        try:
            logging_config = self.get_logging_config()
            required_fields = ['log_level', 'rate_limits', 'enable_console_output']
            missing = [field for field in required_fields if field not in logging_config]
            if missing:
                compatibility['logging_slave_module'] = False
                compatibility['missing_dependencies'].extend(missing)
        except Exception as e:
            compatibility['logging_slave_module'] = False
            compatibility['version_conflicts'].append(f"LoggingConfig generation failed: {e}")
        
        return compatibility
    
    def _estimate_resource_usage(self) -> Dict[str, Any]:
        """Stima l'utilizzo delle risorse"""
        
        # Base resource usage per verbosity level
        base_cpu = {
            ConfigVerbosity.MINIMAL: 0.1,
            ConfigVerbosity.STANDARD: 0.5,
            ConfigVerbosity.VERBOSE: 2.0,
            ConfigVerbosity.DEBUG: 5.0
        }[self.verbosity_level]
        
        base_memory_mb = {
            ConfigVerbosity.MINIMAL: 50,
            ConfigVerbosity.STANDARD: 100,
            ConfigVerbosity.VERBOSE: 250,
            ConfigVerbosity.DEBUG: 500
        }[self.verbosity_level]
        
        # Adjust for buffer size
        memory_buffer_mb = (self.timing_settings.buffer_size * 0.001)  # ~1KB per event
        
        # Adjust for output formats
        format_multiplier = 1 + (sum(1 for enabled in self.output_settings.formats.values() if enabled) * 0.2)
        
        # Adjust for timing
        timing_multiplier = max(0.5, 2.0 / self.timing_settings.refresh_rate)
        
        final_cpu = base_cpu * format_multiplier * timing_multiplier
        final_memory = (base_memory_mb + memory_buffer_mb) * format_multiplier
        
        return {
            'estimated_cpu_usage_percent': round(final_cpu, 2),
            'estimated_memory_usage_mb': round(final_memory, 1),
            'estimated_disk_usage_mb_per_hour': self._estimate_disk_usage(),
            'network_usage_assessment': self._assess_network_usage(),
            'scalability_assessment': self._assess_scalability()
        }
    
    def _estimate_disk_usage(self) -> float:
        """Stima utilizzo disco per ora"""
        
        # Base events per hour estimate
        events_per_hour = {
            ConfigVerbosity.MINIMAL: 100,
            ConfigVerbosity.STANDARD: 1000,
            ConfigVerbosity.VERBOSE: 5000,
            ConfigVerbosity.DEBUG: 20000
        }[self.verbosity_level]
        
        # Average event size in bytes
        avg_event_size = 200  # JSON event ~200 bytes
        
        # Format multipliers
        format_sizes = {
            'csv': 0.5,     # CSV più compatto
            'json': 1.0,    # Base size
            'parquet': 0.3, # Molto compatto
            'sqlite': 0.7   # Compressa
        }
        
        total_multiplier = sum(
            format_sizes.get(fmt, 1.0) 
            for fmt, enabled in self.output_settings.formats.items() 
            if enabled
        )
        
        total_bytes_per_hour = events_per_hour * avg_event_size * total_multiplier
        return round(total_bytes_per_hour / (1024 * 1024), 2)  # Convert to MB
    
    def _assess_network_usage(self) -> str:
        """Valuta l'utilizzo di rete"""
        network_features = [
            self.output_settings.enable_metrics_dashboard,
            self.output_settings.enable_telegram_alerts,
            self.output_settings.enable_email_reports,
            self.output_settings.formats.get('real_time_feed', False)
        ]
        
        active_network = sum(1 for feature in network_features if feature)
        
        if active_network == 0:
            return "minimal"
        elif active_network <= 2:
            return "low"
        else:
            return "moderate"
    
    def _assess_scalability(self) -> str:
        """Valuta la scalabilità della configurazione"""
        
        # Factors che impattano scalabilità
        scalability_score = 100
        
        # Verbosity impact
        verbosity_penalty = {
            ConfigVerbosity.MINIMAL: 0,
            ConfigVerbosity.STANDARD: 10,
            ConfigVerbosity.VERBOSE: 30,
            ConfigVerbosity.DEBUG: 50
        }[self.verbosity_level]
        
        scalability_score -= verbosity_penalty
        
        # Buffer size impact
        if self.timing_settings.buffer_size < 1000:
            scalability_score -= 20  # Small buffers limit scalability
        elif self.timing_settings.buffer_size > 50000:
            scalability_score -= 15  # Too large buffers can cause memory issues
        
        # Format count impact
        format_count = sum(1 for enabled in self.output_settings.formats.values() if enabled)
        if format_count > 3:
            scalability_score -= (format_count - 3) * 10
        
        # Timing impact
        if self.timing_settings.refresh_rate < 0.5:
            scalability_score -= 20  # Very high frequency impacts scalability
        
        if scalability_score >= 80:
            return "excellent"
        elif scalability_score >= 60:
            return "good"
        elif scalability_score >= 40:
            return "fair"
        else:
            return "poor"
    
    def _test_integration_configs(self) -> Dict[str, bool]:
        """Testa la generazione delle configurazioni di integrazione"""
        integration_tests = {}
        
        try:
            unified_config = self.get_unified_config()
            integration_tests['unified_config'] = isinstance(unified_config, dict) and len(unified_config) > 0
        except Exception:
            integration_tests['unified_config'] = False
        
        try:
            logging_config = self.get_logging_config()
            integration_tests['logging_config'] = isinstance(logging_config, dict) and len(logging_config) > 0
        except Exception:
            integration_tests['logging_config'] = False
        
        try:
            analyzer_config = self.get_analyzer_config()
            integration_tests['analyzer_config'] = isinstance(analyzer_config, dict) and len(analyzer_config) > 0
        except Exception:
            integration_tests['analyzer_config'] = False
        
        return integration_tests
    
    def _generate_recommendations(self, validation_errors: List[str], compatibility: Dict[str, Any]) -> List[str]:
        """Genera raccomandazioni per ottimizzare la configurazione"""
        recommendations = []
        
        # Performance recommendations
        if self.verbosity_level == ConfigVerbosity.DEBUG and self.system_mode == "production":
            recommendations.append("Considera di usare STANDARD o MINIMAL verbosity in produzione")
        
        if self.timing_settings.refresh_rate < 1.0 and self.verbosity_level in [ConfigVerbosity.VERBOSE, ConfigVerbosity.DEBUG]:
            recommendations.append("Riduci verbosity o aumenta refresh_rate per migliori performance")
        
        # Resource recommendations
        if self._estimate_disk_usage() > 1000:  # >1GB per hour
            recommendations.append("Utilizzo disco elevato: considera log rotation più aggressiva")
        
        # Compatibility recommendations
        if not compatibility['unified_analyzer_system']:
            recommendations.append("Problemi compatibilità UnifiedAnalyzerSystem: verifica versioni")
        
        # Configuration recommendations
        active_formats = sum(1 for enabled in self.output_settings.formats.values() if enabled)
        if active_formats > 3:
            recommendations.append("Troppi formati attivi: disabilita quelli non necessari")
        
        if len(validation_errors) > 5:
            recommendations.append("Molti errori di validazione: considera di usare un profilo predefinito")
        
        return recommendations

    # ================================
    # UTILITIES AND HELPERS
    # ================================

    def compare_with(self, other: 'UnifiedConfigManager') -> Dict[str, Any]:
        """Confronta questa configurazione con un'altra"""
        differences = {
            'verbosity_level': {
                'this': self.verbosity_level.value,
                'other': other.verbosity_level.value,
                'different': self.verbosity_level != other.verbosity_level
            },
            'timing_differences': self._compare_timing(other.timing_settings),
            'output_differences': self._compare_output(other.output_settings),
            'rate_limit_differences': self._compare_rate_limits(other.rate_limit_settings),
            'performance_impact': self._compare_performance_impact(other)
        }
        
        return differences
    
    def _compare_timing(self, other_timing: ConfigTimingSettings) -> Dict[str, Any]:
        """Confronta timing settings"""
        return {
            'refresh_rate': {
                'this': self.timing_settings.refresh_rate,
                'other': other_timing.refresh_rate,
                'difference_percent': round(((self.timing_settings.refresh_rate - other_timing.refresh_rate) / other_timing.refresh_rate) * 100, 1) if other_timing.refresh_rate > 0 else 0
            },
            'buffer_size': {
                'this': self.timing_settings.buffer_size,
                'other': other_timing.buffer_size,
                'difference_percent': round(((self.timing_settings.buffer_size - other_timing.buffer_size) / other_timing.buffer_size) * 100, 1) if other_timing.buffer_size > 0 else 0
            }
        }
    
    def _compare_output(self, other_output: ConfigOutputSettings) -> Dict[str, Any]:
        """Confronta output settings"""
        this_formats = set(fmt for fmt, enabled in self.output_settings.formats.items() if enabled)
        other_formats = set(fmt for fmt, enabled in other_output.formats.items() if enabled)
        
        return {
            'terminal_mode': {
                'this': self.output_settings.terminal_mode,
                'other': other_output.terminal_mode,
                'different': self.output_settings.terminal_mode != other_output.terminal_mode
            },
            'formats': {
                'this_only': list(this_formats - other_formats),
                'other_only': list(other_formats - this_formats),
                'common': list(this_formats & other_formats)
            }
        }
    
    def _compare_rate_limits(self, other_rates: ConfigRateLimitSettings) -> Dict[str, Any]:
        """Confronta rate limit settings"""
        return {
            'tick_processing': {
                'this': self.rate_limit_settings.tick_processing,
                'other': other_rates.tick_processing,
                'ratio': round(self.rate_limit_settings.tick_processing / other_rates.tick_processing, 2) if other_rates.tick_processing > 0 else 0
            },
            'predictions': {
                'this': self.rate_limit_settings.predictions,
                'other': other_rates.predictions,
                'ratio': round(self.rate_limit_settings.predictions / other_rates.predictions, 2) if other_rates.predictions > 0 else 0
            }
        }
    
    def _compare_performance_impact(self, other: 'UnifiedConfigManager') -> Dict[str, str]:
        """Confronta l'impatto sulle performance"""
        this_overhead = self._calculate_overhead_estimate()
        other_overhead = other._calculate_overhead_estimate()
        
        this_cpu = float(this_overhead['estimated_cpu_overhead'].rstrip('%'))
        other_cpu = float(other_overhead['estimated_cpu_overhead'].rstrip('%'))
        
        if this_cpu < other_cpu:
            performance_comparison = "better"
        elif this_cpu > other_cpu:
            performance_comparison = "worse"
        else:
            performance_comparison = "similar"
        
        return {
            'performance_comparison': performance_comparison,
            'cpu_overhead_difference': f"{this_cpu - other_cpu:+.1f}%",
            'recommendation': "Use this config" if performance_comparison == "better" else "Use other config" if performance_comparison == "worse" else "Both configs have similar performance"
        }
    
    def clone(self) -> 'UnifiedConfigManager':
        """Crea una copia identica della configurazione"""
        return UnifiedConfigManager(
            verbosity_level=self.verbosity_level,
            timing_settings=ConfigTimingSettings(**asdict(self.timing_settings)),
            output_settings=ConfigOutputSettings(**asdict(self.output_settings)),
            rate_limit_settings=ConfigRateLimitSettings(**asdict(self.rate_limit_settings)),
            asset_symbol=self.asset_symbol,
            system_mode=self.system_mode,
            config_version=self.config_version,
            enable_unified_system=self.enable_unified_system,
            enable_logging_slave=self.enable_logging_slave,
            enable_analyzer_integration=self.enable_analyzer_integration
        )
    
    def __str__(self) -> str:
        """String representation leggibile"""
        return f"""UnifiedConfigManager(
    Asset: {self.asset_symbol}
    Verbosity: {self.verbosity_level.value}
    System Mode: {self.system_mode}
    Timing: {self.timing_settings.refresh_rate}s refresh, {self.timing_settings.buffer_size} buffer
    Output: {self.output_settings.terminal_mode} terminal, {sum(1 for x in self.output_settings.formats.values() if x)} formats
    Performance: {self._calculate_overhead_estimate()['performance_impact']} impact
)"""
    
    def __repr__(self) -> str:
        """Representation for debugging"""
        return f"UnifiedConfigManager(verbosity={self.verbosity_level.value}, asset={self.asset_symbol}, mode={self.system_mode})"


# ================================
# GLOBAL INSTANCE AND UTILITIES
# ================================

# Default global instance
_global_config_manager: Optional[UnifiedConfigManager] = None

def get_global_config_manager() -> UnifiedConfigManager:
    """Ottieni l'istanza globale del ConfigManager"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = UnifiedConfigManager.create_development_config()
    return _global_config_manager

def set_global_config_manager(config_manager: UnifiedConfigManager) -> None:
    """Imposta l'istanza globale del ConfigManager"""
    global _global_config_manager
    _global_config_manager = config_manager
    print(f"🔧 Global ConfigManager updated: {config_manager.verbosity_level.value} mode for {config_manager.asset_symbol}")

def create_quick_config(profile: str, asset: str = "EURUSD") -> UnifiedConfigManager:
    """Factory method rapido per profili comuni"""
    profile_map = {
        'prod': UnifiedConfigManager.create_production_config,
        'production': UnifiedConfigManager.create_production_config,
        'dev': UnifiedConfigManager.create_development_config,
        'development': UnifiedConfigManager.create_development_config,
        'research': UnifiedConfigManager.create_research_config,
        'demo': UnifiedConfigManager.create_demo_config,
        'monitor': UnifiedConfigManager.create_monitoring_config,
        'monitoring': UnifiedConfigManager.create_monitoring_config
    }
    
    factory_method = profile_map.get(profile.lower())
    if factory_method is None:
        raise ValueError(f"Unknown profile '{profile}'. Available: {list(profile_map.keys())}")
    
    return factory_method(asset)


# ================================
# MAIN EXECUTION AND TESTING
# ================================

def main():
    """Test e demo del ConfigManager"""
    print("🔧 Unified ConfigManager - ML Training Logger")
    print("=" * 60)
    
    # Test creazione profili
    print("\n📋 Testing predefined profiles...")
    
    profiles = {
        'Production': UnifiedConfigManager.create_production_config("EURUSD"),
        'Development': UnifiedConfigManager.create_development_config("EURUSD"),
        'Research': UnifiedConfigManager.create_research_config("EURUSD"),
        'Demo': UnifiedConfigManager.create_demo_config("EURUSD")
    }
    
    for name, config in profiles.items():
        print(f"\n{name} Profile:")
        print(f"  Verbosity: {config.verbosity_level.value}")
        print(f"  Refresh Rate: {config.timing_settings.refresh_rate}s")
        print(f"  Terminal Mode: {config.output_settings.terminal_mode}")
        
        # Test performance summary
        perf = config.get_performance_summary()
        print(f"  Est. CPU Overhead: {perf['estimated_overhead']['estimated_cpu_overhead']}")
    
    # Test validazione
    print("\n🔍 Testing validation...")
    test_config = UnifiedConfigManager.create_development_config("BTCUSD")
    is_valid, errors = test_config.validate_configuration()
    print(f"Configuration valid: {is_valid}")
    if errors:
        print(f"Errors found: {len(errors)}")
        for error in errors[:3]:  # Show first 3 errors
            print(f"  - {error}")
    
    # Test diagnostics
    print("\n🏥 Running diagnostics...")
    diagnostics = test_config.run_diagnostics()
    print(f"Diagnostics completed in {diagnostics['diagnostics_duration_ms']}ms")
    print(f"Validation: {diagnostics['configuration_validation']['is_valid']}")
    print(f"System compatibility: {all(diagnostics['system_compatibility'].values())}")
    print(f"Scalability: {diagnostics['resource_estimation']['scalability_assessment']}")
    
    # Test integration configs
    print("\n🔗 Testing integration configs...")
    unified_config = test_config.get_unified_config()
    logging_config = test_config.get_logging_config()
    analyzer_config = test_config.get_analyzer_config()
    
    print(f"UnifiedConfig generated: {len(unified_config)} fields")
    print(f"LoggingConfig generated: {len(logging_config)} fields")
    print(f"AnalyzerConfig generated: {len(analyzer_config)} fields")
    
    # Test serialization
    print("\n💾 Testing serialization...")
    temp_file = "test_config.json"
    save_success = test_config.save_to_file(temp_file)
    if save_success:
        loaded_config = UnifiedConfigManager.load_from_file(temp_file)
        print(f"Save/Load successful: {loaded_config.verbosity_level == test_config.verbosity_level}")
        
        # Cleanup
        try:
            import os
            os.remove(temp_file)
        except:
            pass
    
    # Test comparison
    print("\n⚖️ Testing configuration comparison...")
    prod_config = UnifiedConfigManager.create_production_config("EURUSD")
    dev_config = UnifiedConfigManager.create_development_config("EURUSD")
    
    comparison = prod_config.compare_with(dev_config)
    print(f"Verbosity different: {comparison['verbosity_level']['different']}")
    print(f"Performance comparison: {comparison['performance_impact']['performance_comparison']}")
    
    # Test runtime updates
    print("\n🔄 Testing runtime updates...")
    print(f"Initial verbosity: {test_config.verbosity_level.value}")
    test_config.update_verbosity(ConfigVerbosity.MINIMAL)
    print(f"Updated verbosity: {test_config.verbosity_level.value}")
    
    # Test quick config factory
    print("\n⚡ Testing quick config factory...")
    quick_configs = ['prod', 'dev', 'research', 'demo']
    for profile in quick_configs:
        try:
            config = create_quick_config(profile, "GBPUSD")
            print(f"  {profile}: {config.verbosity_level.value} - ✅")
        except Exception as e:
            print(f"  {profile}: Failed - {e}")
    
    print("\n✅ All tests completed!")
    print("\n📖 Usage examples:")
    print("""
# Quick setup for production
config = UnifiedConfigManager.create_production_config("EURUSD")

# Custom configuration
config = UnifiedConfigManager.create_custom_config(
    verbosity=ConfigVerbosity.VERBOSE,
    timing_preset=ConfigTimingPreset.HIGH_FREQUENCY,
    output_preset=ConfigOutputPreset.MONITORING
)

# Apply to systems
unified_system = UnifiedAnalyzerSystem(config.get_unified_config())
logging_slave = create_logging_slave(config.get_logging_config())
analyzer = Analyzer(config.get_analyzer_config())

# Runtime management
config.update_verbosity(ConfigVerbosity.DEBUG)
summary = config.get_performance_summary()
diagnostics = config.run_diagnostics()

# Persistence
config.save_to_file("my_config.json")
loaded = UnifiedConfigManager.load_from_file("my_config.json")
""")


if __name__ == "__main__":
    main()