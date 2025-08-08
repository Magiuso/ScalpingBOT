"""
Base configuration class - MIGRATED from src/Analyzer.py (lines 198-537)
NO LOGIC CHANGES - Only reorganized
"""

from dataclasses import dataclass, field, fields
from typing import Dict, List, Any, Optional
import json


@dataclass
class AnalyzerConfig:
    """Configurazione centralizzata per tutti i magic numbers dell'Analyzer"""
    
    # ========== LEARNING PHASE CONFIGURATION ==========
    min_learning_days: int = 30  # Giorni minimi di learning
    learning_ticks_threshold: int = 100000  # Tick minimi per completare learning - Increased to 100K
    learning_mini_training_interval: int = 500   # Era 1000 -> 2x più frequente per GPU
    
    # ========== DATA MANAGEMENT ==========
    max_tick_buffer_size: int = 1000000  # Era 500K -> 2x per batch grandi GPU
    data_cleanup_days: int = 180  # Giorni di dati da mantenere
    aggregation_windows: List[int] = field(default_factory=lambda: [5, 15, 30, 60])  # Minuti
    
    # ========== COMPETITION SYSTEM ==========
    champion_threshold: float = 0.10  # FIXED: Was 0.20, now 10% improvement to dethrone
    min_predictions_for_champion: int = 50  # FIXED: Was 100, now 50 for faster promotion
    reality_check_interval_hours: int = 6  # Ore tra reality check
    
    # ========== PERFORMANCE THRESHOLDS ==========
    accuracy_threshold: float = 0.6  # Threshold per predizione corretta
    confidence_threshold: float = 0.5  # Threshold minimo confidence
    emergency_accuracy_drop: float = 0.5  # 🔧 RILASSATO: Drop 50% per emergency stop (era 30%)
    emergency_consecutive_failures: int = 20  # 🔧 RILASSATO: 20 fallimenti consecutivi (era 10)
    emergency_confidence_collapse: float = 0.15  # 🔧 RILASSATO: Confidence sotto 15% (era 40%)
    emergency_rejection_rate: float = 0.9  # 🔧 RILASSATO: 90% feedback negativi (era 80%)
    emergency_score_decline: float = 0.4  # 🔧 RILASSATO: Declino 40% in 24h (era 25%)
    
    # ========== MODEL TRAINING ==========
    training_batch_size: int = 32  # BIBBIA COMPLIANT: Smaller batch for better generalization
    training_epochs: int = 100  # Epoch per training
    training_patience: int = 15  # Early stopping patience
    training_test_split: float = 0.8  # Train/test split ratio
    max_grad_norm: float = 1.0  # 🔧 FIXED: Gradient clipping meno aggressivo per permettere apprendimento
    learning_rate: float = 5e-4  # 🔧 BIBBIA COMPLIANT: Reduced for stability as per AdaptiveTrainer config
    
    # ========== LSTM CONFIGURATION ==========
    lstm_sequence_length: int = 30  # Lunghezza sequenza LSTM
    lstm_hidden_size: int = 256  # ANTI-OVERFITTING: Reduced from 512 to 256
    lstm_num_layers: int = 2     # ANTI-OVERFITTING: Reduced from 4 to 2 layers
    lstm_dropout: float = 0.6    # ANTI-OVERFITTING: Increased from 0.5 to 0.6 for stronger regularization
    
    # ========== TECHNICAL INDICATORS ==========
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14
    
    # ========== VOLATILITY THRESHOLDS ==========
    high_volatility_threshold: float = 0.02  # 2%
    low_volatility_threshold: float = 0.005  # 0.5%
    extreme_volatility_threshold: float = 0.025  # 2.5%
    volatility_spike_multiplier: float = 1.5  # 1.5x average per spike
    
    # ========== CACHE SYSTEM ==========
    indicators_cache_size: int = 1000  # Max indicatori in cache
    cache_cleanup_threshold: float = 0.8  # Cleanup quando cache > 80%
    cache_hit_rate_threshold: float = 50.0  # Hit rate minimo desiderato
    
    # ========== ASYNC I/O ==========
    async_queue_size: int = 2000  # Dimensione coda async
    async_workers: int = 3  # Worker threads per I/O
    async_batch_timeout: float = 1.0  # Timeout batch (secondi)
    
    # ========== RISK MANAGEMENT ==========
    risk_high_threshold: float = 0.6  # Score rischio alto
    risk_moderate_threshold: float = 0.3  # Score rischio moderato
    spread_high_threshold: float = 0.001  # 0.1% spread alto
    volume_low_multiplier: float = 0.5  # Volume basso < 50% media
    volume_high_multiplier: float = 2.0  # Volume alto > 200% media
    
    # ========== PATTERN RECOGNITION ==========
    pattern_probability_threshold: float = 0.7  # Threshold pattern detection
    pattern_confidence_threshold: float = 0.65  # Confidence minima pattern
    double_top_bottom_tolerance: float = 0.02  # 2% tolleranza per double patterns
    triangle_slope_threshold: float = 0.0001  # Threshold pendenza triangoli
    
    # ========== TREND ANALYSIS ==========
    trend_strength_threshold: float = 0.7  # Forza trend significativa
    trend_r_squared_threshold: float = 0.7  # R² minimo per trend forte
    trend_slope_threshold: float = 0.0001  # Slope minimo per trend
    trend_age_threshold: int = 30  # Età massima trend (periodi)
    
    # ========== BIAS DETECTION ==========
    bias_confidence_threshold: float = 0.7  # Confidence minima bias
    order_flow_threshold: float = 0.1  # Soglia order flow imbalance
    sentiment_threshold: float = 0.65  # Soglia sentiment analysis
    smart_money_threshold: int = 3  # Segnali minimi smart money
    
    # ========== SUPPORT/RESISTANCE ==========
    sr_test_count_threshold: int = 3  # Test minimi per livello S/R
    sr_proximity_threshold: float = 0.002  # 0.2% vicinanza a S/R
    sr_confidence_threshold: float = 0.75  # Confidence minima S/R
    volume_profile_percentile: float = 70.0  # Percentile volume profile
    
    # ========== VALIDATION TIMING ==========
    validation_default_minutes: int = 5  # Minuti default validazione
    validation_sr_ticks: int = 100  # Tick per validazione S/R
    validation_pattern_ticks: int = 200  # Tick per validazione pattern
    validation_bias_ticks: int = 50  # Tick per validazione bias
    validation_trend_ticks: int = 300  # Tick per validazione trend
    
    # ========== DIAGNOSTICS ==========
    diagnostics_memory_threshold: float = 80.0  # % memoria per allarme
    diagnostics_processing_time_threshold: float = 5.0  # Secondi max processing
    diagnostics_tick_rate_min: float = 0.1  # Tick/secondo minimo
    diagnostics_monitor_interval: int = 30  # Secondi tra monitor checks
    
    # ========== LOGGING ==========
    log_rotation_months: int = 1  # Rotazione log mensile
    log_level_system: str = "INFO"
    log_level_errors: str = "ERROR" 
    log_level_predictions: str = "DEBUG"
    log_milestone_interval: int = 1000  # Log ogni N tick
    
    # ========== PERFORMANCE OPTIMIZATION ==========
    performance_window_size: int = 100  # Finestra rolling performance
    latency_history_size: int = 100  # Storia latency
    max_processing_time: float = 10.0  # Tempo max processing (sec)
    feature_vector_size: int = 50  # Dimensione ridotta per evitare compressione eccessiva (200→50 invece di 200→10)
    
    # ========== PRESERVATION SYSTEM ==========
    max_preserved_champions: int = 5  # Champion preservati per tipo
    preservation_score_threshold: float = 70.0  # Score minimo preservazione
    preservation_improvement_threshold: float = 0.2  # 20% miglioramento
    
    # ========== ML TRAINING LOGGER CONFIGURATION ==========
    ml_logger_enabled: bool = True  # Abilita ML Training Logger
    ml_logger_verbosity: str = "verbose"   # Mostra log dettagliati del training ML
    ml_logger_terminal_mode: str = "scroll"  # dashboard, scroll, minimal - scroll mostra i log in console
    ml_logger_file_output: bool = True  # Abilita output su file
    ml_logger_formats: List[str] = field(default_factory=lambda: ["json", "csv"])  # Formati output
    ml_logger_base_directory: str = "./analyzer_data"  # Directory base log
    ml_logger_rate_limit_ticks: int = 100  # Rate limit per tick processing
    ml_logger_flush_interval: float = 5.0  # Intervallo flush su disco (secondi)
    
    def __post_init__(self):
        """Validazione configurazione"""
        self._validate_config()
    
    def _validate_config(self):
        """Valida che la configurazione sia sensata"""
        assert 0 < self.champion_threshold < 1, "Champion threshold deve essere tra 0 e 1"
        assert 0 < self.training_test_split < 1, "Test split deve essere tra 0 e 1"
        assert self.min_learning_days > 0, "Giorni learning devono essere positivi"
        assert self.max_tick_buffer_size > 1000, "Buffer size troppo piccolo"
        assert 0 < self.accuracy_threshold < 1, "Accuracy threshold deve essere tra 0 e 1"
        assert len(self.sma_periods) > 0, "Serve almeno un periodo SMA"
        assert self.high_volatility_threshold > self.low_volatility_threshold, "Soglie volatilità inconsistenti"
    
    def get_validation_criteria(self, model_type: str) -> Dict[str, int]:
        """Ottieni criteri di validazione per tipo di modello"""
        # NOTE: Original uses ModelType enum - we use string for now to avoid import
        validation_map = {
            'support_resistance': {'ticks': self.validation_sr_ticks, 'minutes': self.validation_default_minutes},
            'pattern_recognition': {'ticks': self.validation_pattern_ticks, 'minutes': self.validation_default_minutes * 2},
            'bias_detection': {'ticks': self.validation_bias_ticks, 'minutes': self.validation_default_minutes // 2},
            'trend_analysis': {'ticks': self.validation_trend_ticks, 'minutes': self.validation_default_minutes * 3},
            'volatility_prediction': {'ticks': self.validation_sr_ticks + 50, 'minutes': self.validation_default_minutes + 2}
        }
        
        # FAIL FAST - No default validation map
        if model_type not in validation_map:
            raise KeyError(f"No validation configuration found for model_type: {model_type}")
        return validation_map[model_type]
    
    def get_emergency_stop_triggers(self) -> Dict[str, float]:
        """Ottieni triggers per emergency stop"""
        return {
            'accuracy_drop': self.emergency_accuracy_drop,
            'consecutive_failures': self.emergency_consecutive_failures,
            'confidence_collapse': self.emergency_confidence_collapse,
            'observer_rejection_rate': self.emergency_rejection_rate,
            'rapid_score_decline': self.emergency_score_decline
        }
    
    def get_risk_thresholds(self) -> Dict[str, float]:
        """Ottieni soglie per risk assessment"""
        return {
            'high_risk': self.risk_high_threshold,
            'moderate_risk': self.risk_moderate_threshold,
            'high_volatility': self.high_volatility_threshold,
            'low_volatility': self.low_volatility_threshold,
            'wide_spread': self.spread_high_threshold,
            'low_volume_ratio': self.volume_low_multiplier,
            'high_volume_ratio': self.volume_high_multiplier
        }
    
    def get_model_architecture(self, model_name: str) -> Dict[str, Any]:
        """Ottieni architettura per modelli ML"""
        architectures = {
            'LSTM_SupportResistance': {
                'input_size': self.feature_vector_size * 8,  # 50 * 8 = 400 features (prices, volumes, sma, rsi, vwap, order_flow, buying_pressure, hvn)
                'hidden_size': self.lstm_hidden_size,
                'num_layers': self.lstm_num_layers,
                'output_size': 2,  # S/R = 2 livelli
                'dropout': self.lstm_dropout
            },
            'LSTM_Sequences': {
                'input_size': 200,  # ✅ FIXED: window_size * 4 = 50 * 4 = 200 features
                'hidden_size': self.lstm_hidden_size,  # 🚀 OPTIMIZED: 256 (not 512) for better generalization
                'num_layers': self.lstm_num_layers,    # 🚀 OPTIMIZED: 2 layers (not 3) for faster training
                'output_size': 5,  # ✅ FIXED: 5 pattern types (classical, cnn, lstm, transformer, ensemble)
                'dropout': self.lstm_dropout
            },
            'Sentiment_LSTM': {
                'input_size': 200,  # ✅ FIXED: 200 features per compatibilità con _prepare_bias_dataset
                'hidden_size': self.lstm_hidden_size // 2,
                'num_layers': self.lstm_num_layers - 1,
                'output_size': 6,  # 🚀 FIXED: 6 output dal bias dataset (non 3)
                'dropout': self.lstm_dropout
            },
            'LSTM_TrendPrediction': {
                'input_size': 23,               # 🚀 FIXED: 23 features da _prepare_trend_dataset (verified from debug)
                'hidden_size': self.lstm_hidden_size,
                'num_layers': self.lstm_num_layers,
                'output_size': 1,               # 🚀 FIXED: Regressione (1 valore continuo) non classificazione
                'dropout': self.lstm_dropout
            },
            'LSTM_Volatility': {
                'input_size': 6,  # Fixed: volatility dataset has exactly 6 features
                'hidden_size': self.lstm_hidden_size // 2,
                'num_layers': self.lstm_num_layers - 1,
                'output_size': 1,  # Volatility = 1 valore
                'dropout': self.lstm_dropout
            },
            'Neural_Momentum': {
                'input_size': 250,  # Fixed: momentum dataset has 250 features (5 x window_size)
                'hidden_size': self.lstm_hidden_size // 2,
                'num_layers': 2,
                'output_size': 4,  # 4 momentum indicators
                'dropout': self.lstm_dropout
            }
        }
        
        # FAIL FAST - No default architecture allowed
        if model_name not in architectures:
            raise KeyError(f"No architecture configuration found for model_name: {model_name}")
        return architectures[model_name]
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'AnalyzerConfig':
        """Carica configurazione da file JSON"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return cls(**config_data)
        except FileNotFoundError:
            # REMOVED safe_print - will raise proper error
            raise FileNotFoundError(f"Config file {config_path} not found")
        except Exception as e:
            # REMOVED safe_print - will raise proper error
            raise RuntimeError(f"Error loading config: {e}")
    
    def create_ml_logger_config(self, asset_name: str):
        """Crea configurazione ML Training Logger basata su AnalyzerConfig"""
        try:
            # NOTE: Import da sistema UNIFICATO - FASE 1 - CONFIG completata  
            from src.config.domain.monitoring_config import (
                MLTrainingLoggerConfig, VerbosityLevel, 
                TerminalMode, OutputFormat
            )
            
            # Mappa la verbosity string direttamente a VerbosityLevel (eliminata duplicazione)
            verbosity_map = {
                "minimal": VerbosityLevel.MINIMAL,
                "standard": VerbosityLevel.STANDARD,
                "verbose": VerbosityLevel.VERBOSE,
                "debug": VerbosityLevel.DEBUG
            }
            
            if self.ml_logger_verbosity not in verbosity_map:
                raise ValueError(f"Invalid ml_logger_verbosity: '{self.ml_logger_verbosity}'. Must be one of: {list(verbosity_map.keys())}")
            ml_verbosity = verbosity_map[self.ml_logger_verbosity]
            
            # Convert string types to enums
            terminal_mode_map = {
                "scroll": TerminalMode.SCROLL,
                "silent": TerminalMode.SILENT,
                "auto": TerminalMode.AUTO
            }
            
            output_format_map = {
                "json": OutputFormat.JSON,
                "csv": OutputFormat.CSV,
                "both": OutputFormat.BOTH
            }
            
            # Crea configurazione diretta
            ml_config = MLTrainingLoggerConfig(
                verbosity=ml_verbosity
            )
            
            # Personalizza con i parametri dall'AnalyzerConfig (FAIL FAST - NO FALLBACK)
            if self.ml_logger_terminal_mode not in terminal_mode_map:
                raise ValueError(f"Invalid terminal_mode: '{self.ml_logger_terminal_mode}'. Must be one of: {list(terminal_mode_map.keys())}")
            ml_config.display.terminal_mode = terminal_mode_map[self.ml_logger_terminal_mode]
            
            ml_config.storage.enable_file_output = self.ml_logger_file_output
            
            # Convert formats list from strings to OutputFormat enums (FAIL FAST)
            converted_formats = []
            for fmt in self.ml_logger_formats:
                if fmt not in output_format_map:
                    raise ValueError(f"Invalid output format: '{fmt}'. Must be one of: {list(output_format_map.keys())}")
                converted_formats.append(output_format_map[fmt])
            
            if not converted_formats:
                raise ValueError("ml_logger_formats cannot be empty")
            ml_config.storage.output_formats = converted_formats
            
            ml_config.storage.output_directory = self.ml_logger_base_directory
            ml_config.storage.flush_interval_seconds = self.ml_logger_flush_interval
            
            return ml_config
            
        except ImportError as e:
            # REMOVED safe_print - will raise proper error
            raise ImportError(f"ML_Training_Logger not available: {e}")
        except Exception as e:
            # REMOVED safe_print - will raise proper error
            raise RuntimeError(f"Error creating ML logger config: {e}")
    
    def save_to_file(self, config_path: str) -> None:
        """Salva configurazione su file JSON"""
        try:
            # Convert dataclass to dict
            config_dict = {}
            for field in fields(self):
                value = getattr(self, field.name)
                if isinstance(value, (list, dict, str, int, float, bool)):
                    config_dict[field.name] = value
                else:
                    config_dict[field.name] = str(value)
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            # REMOVED safe_print - operation succeeds silently
            
        except Exception as e:
            # REMOVED safe_print - will raise proper error
            raise RuntimeError(f"Error saving config: {e}")


# ================== GLOBAL CONFIG INSTANCE ==================

# Istanza globale della configurazione
DEFAULT_CONFIG = AnalyzerConfig()

def get_analyzer_config() -> AnalyzerConfig:
    """Ottieni l'istanza della configurazione corrente"""
    return DEFAULT_CONFIG

def set_analyzer_config(config: AnalyzerConfig) -> None:
    """Imposta una nuova configurazione globale"""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config
    # REMOVED safe_print

def load_config_from_file(config_path: str) -> None:
    """Carica e imposta configurazione da file"""
    config = AnalyzerConfig.load_from_file(config_path)
    set_analyzer_config(config)


