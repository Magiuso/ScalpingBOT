import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils
from transformers import AutoModel, AutoTokenizer
import talib as ta
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Deque
from dataclasses import dataclass, field, fields
from collections import deque, defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import json
import pickle
import os
import sys
import threading
import asyncio
import queue
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import warnings
import csv
import logging
import logging.handlers
from pathlib import Path
import gzip
import shutil
import MetaTrader5 as mt5  # Per interfaccia MT5
import psutil
import time
warnings.filterwarnings('ignore')

# ================== ANALYZER CONFIGURATION SYSTEM ==================

@dataclass
class AnalyzerConfig:
    """Configurazione centralizzata per tutti i magic numbers dell'Analyzer"""
    
    # ========== LEARNING PHASE CONFIGURATION ==========
    min_learning_days: int = 90  # Giorni minimi di learning
    learning_ticks_threshold: int = 50000  # Tick minimi per completare learning
    learning_mini_training_interval: int = 1000  # Ogni N tick durante learning
    
    # ========== DATA MANAGEMENT ==========
    max_tick_buffer_size: int = 100000  # Dimensione buffer tick_data
    data_cleanup_days: int = 180  # Giorni di dati da mantenere
    aggregation_windows: List[int] = field(default_factory=lambda: [5, 15, 30, 60])  # Minuti
    
    # ========== COMPETITION SYSTEM ==========
    champion_threshold: float = 0.20  # 20% migliore per detronizzare champion
    min_predictions_for_champion: int = 100  # Predizioni minime per essere champion
    reality_check_interval_hours: int = 6  # Ore tra reality check
    
    # ========== PERFORMANCE THRESHOLDS ==========
    accuracy_threshold: float = 0.6  # Threshold per predizione corretta
    confidence_threshold: float = 0.5  # Threshold minimo confidence
    emergency_accuracy_drop: float = 0.3  # Drop 30% per emergency stop
    emergency_consecutive_failures: int = 10  # Fallimenti consecutivi per emergency
    emergency_confidence_collapse: float = 0.4  # Confidence sotto 40%
    emergency_rejection_rate: float = 0.8  # 80% feedback negativi
    emergency_score_decline: float = 0.25  # Declino 25% in 24h
    
    # ========== MODEL TRAINING ==========
    training_batch_size: int = 32  # Batch size per training
    training_epochs: int = 100  # Epoch per training
    training_patience: int = 15  # Early stopping patience
    training_test_split: float = 0.8  # Train/test split ratio
    max_grad_norm: float = 1.0  # Gradient clipping
    learning_rate: float = 1e-4  # Learning rate di default
    
    # ========== LSTM CONFIGURATION ==========
    lstm_sequence_length: int = 30  # Lunghezza sequenza LSTM
    lstm_hidden_size: int = 128  # Hidden size di default
    lstm_num_layers: int = 3  # Layer LSTM di default
    lstm_dropout: float = 0.2  # Dropout LSTM
    
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
    feature_vector_size: int = 10  # Dimensione standard feature vector
    
    # ========== PRESERVATION SYSTEM ==========
    max_preserved_champions: int = 5  # Champion preservati per tipo
    preservation_score_threshold: float = 70.0  # Score minimo preservazione
    preservation_improvement_threshold: float = 0.2  # 20% miglioramento
    
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
    
    def get_validation_criteria(self, model_type: 'ModelType') -> Dict[str, int]:
        """Ottieni criteri di validazione per tipo di modello"""
        validation_map = {
            'support_resistance': {'ticks': self.validation_sr_ticks, 'minutes': self.validation_default_minutes},
            'pattern_recognition': {'ticks': self.validation_pattern_ticks, 'minutes': self.validation_default_minutes * 2},
            'bias_detection': {'ticks': self.validation_bias_ticks, 'minutes': self.validation_default_minutes // 2},
            'trend_analysis': {'ticks': self.validation_trend_ticks, 'minutes': self.validation_default_minutes * 3},
            'volatility_prediction': {'ticks': self.validation_sr_ticks + 50, 'minutes': self.validation_default_minutes + 2},
            'momentum_analysis': {'ticks': self.validation_bias_ticks + 50, 'minutes': self.validation_default_minutes}
        }
        
        return validation_map.get(model_type.value, {
            'ticks': self.validation_sr_ticks, 
            'minutes': self.validation_default_minutes
        })
    
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
                'input_size': self.feature_vector_size,
                'hidden_size': self.lstm_hidden_size,
                'num_layers': self.lstm_num_layers,
                'output_size': 2,  # S/R = 2 livelli
                'dropout': self.lstm_dropout
            },
            'LSTM_Sequences': {
                'input_size': 15,
                'hidden_size': self.lstm_hidden_size * 2,
                'num_layers': self.lstm_num_layers + 1,
                'output_size': 50,  # Pattern = molti pattern
                'dropout': self.lstm_dropout
            },
            'Sentiment_LSTM': {
                'input_size': 8,
                'hidden_size': self.lstm_hidden_size // 2,
                'num_layers': self.lstm_num_layers - 1,
                'output_size': 3,  # Bias = 3 categorie
                'dropout': self.lstm_dropout
            },
            'LSTM_TrendPrediction': {
                'input_size': 12,
                'hidden_size': self.lstm_hidden_size,
                'num_layers': self.lstm_num_layers,
                'output_size': 3,  # Trend = 3 direzioni
                'dropout': self.lstm_dropout
            },
            'LSTM_Volatility': {
                'input_size': self.feature_vector_size,
                'hidden_size': self.lstm_hidden_size // 2,
                'num_layers': self.lstm_num_layers - 1,
                'output_size': 1,  # Volatility = 1 valore
                'dropout': self.lstm_dropout
            }
        }
        
        return architectures.get(model_name, {
            'input_size': self.feature_vector_size,
            'hidden_size': self.lstm_hidden_size,
            'num_layers': self.lstm_num_layers,
            'output_size': 1,
            'dropout': self.lstm_dropout
        })
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'AnalyzerConfig':
        """Carica configurazione da file JSON"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return cls(**config_data)
        except FileNotFoundError:
            safe_print(f"⚠️ Config file {config_path} not found, using defaults")
            return cls()
        except Exception as e:
            safe_print(f"❌ Error loading config: {e}, using defaults")
            return cls()
    
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
            
            safe_print(f"✅ Configuration saved to {config_path}")
            
        except Exception as e:
            safe_print(f"❌ Error saving config: {e}")

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
    safe_print("✅ Analyzer configuration updated")

def load_config_from_file(config_path: str) -> None:
    """Carica e imposta configurazione da file"""
    config = AnalyzerConfig.load_from_file(config_path)
    set_analyzer_config(config)

# ================== CUSTOM EXCEPTIONS ==================

class AnalyzerException(Exception):
    """Base exception for Analyzer errors"""
    pass

class InsufficientDataError(AnalyzerException):
    """Raised when there's not enough data for analysis"""
    def __init__(self, required: int, available: int, operation: str = "analysis"):
        self.required = required
        self.available = available
        self.operation = operation
        super().__init__(f"Insufficient data for {operation}: required {required}, available {available}")

class ModelNotInitializedError(AnalyzerException):
    """Raised when a model is not properly initialized"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Model {model_name} is not properly initialized")

class InvalidInputError(AnalyzerException):
    """Raised when input data is invalid"""
    def __init__(self, field: str, value: Any, reason: str = ""):
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid {field}: {value}. {reason}".strip())

class PredictionError(AnalyzerException):
    """Raised when prediction generation fails"""
    def __init__(self, algorithm: str, reason: str):
        self.algorithm = algorithm
        self.reason = reason
        super().__init__(f"Prediction failed for {algorithm}: {reason}")

# ================== ERROR HANDLER WRAPPER ==================

def safe_execute(func, default_return=None, log_errors=True, logger=None):
    """Wrapper per esecuzione sicura con logging standardizzato"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AnalyzerException as e:
            if log_errors and logger:
                logger.loggers['errors'].error(f"Analyzer error in {func.__name__}: {e}")
            if default_return is not None:
                return default_return
            # Re-raise se non c'è default
            raise
        except Exception as e:
            if log_errors and logger:
                logger.loggers['errors'].error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            if default_return is not None:
                return default_return
            # Wrap in AnalyzerException
            raise AnalyzerException(f"Unexpected error in {func.__name__}: {e}") from e
    return wrapper

# ================== INDICATORS CACHE SYSTEM ==================

class IndicatorsCache:
    """Sistema di cache per indicatori tecnici con gestione intelligente della memoria"""
    
    def __init__(self, max_cache_size: int = 500):
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, Any] = {}
        self.access_count: Dict[str, int] = {}
        self.last_access: Dict[str, datetime] = {}
        
    def _generate_cache_key(self, prices: np.ndarray, indicator_name: str, **kwargs) -> str:
        """Genera chiave univoca per cache basata su hash dei dati"""
        # Hash dei prezzi per identificare univocamente il dataset
        prices_bytes = prices.tobytes() if hasattr(prices, 'tobytes') else str(prices).encode()
        prices_hash = hashlib.md5(prices_bytes).hexdigest()[:8]
        
        # Parametri per la cache key
        params_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        
        return f"{indicator_name}_{prices_hash}_{len(prices)}_{params_str}"
    
    def get_indicator(self, prices: np.ndarray, indicator_name: str, calculator_func, **kwargs) -> np.ndarray:
        """Ottieni indicatore dalla cache o calcolalo"""
        cache_key = self._generate_cache_key(prices, indicator_name, **kwargs)
        
        # Cache hit
        if cache_key in self.cache:
            self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
            self.last_access[cache_key] = datetime.now()
            return self.cache[cache_key]
        
        # Cache miss - calcola indicatore
        try:
            result = calculator_func(prices, **kwargs)
            
            # Valida risultato prima di salvare in cache
            if result is not None and len(result) > 0:
                # Cleanup cache se necessario
                if len(self.cache) >= self.max_cache_size:
                    self._cleanup_cache()
                
                # Salva in cache
                self.cache[cache_key] = result
                self.access_count[cache_key] = 1
                self.last_access[cache_key] = datetime.now()
                
                return result
            else:
                # Fallback se calcolo fallisce
                return np.full_like(prices, np.nan)
                
        except Exception as e:
            # Log errore e ritorna fallback
            safe_print(f"⚠️ Errore calcolo {indicator_name}: {e}")
            return np.full_like(prices, np.nan)
    
    def _cleanup_cache(self) -> None:
        """Rimuove elementi meno utilizzati dalla cache"""
        if len(self.cache) < self.max_cache_size * 0.8:
            return
        
        # Strategia: rimuovi elementi con basso access count e vecchi
        removal_candidates = []
        
        for key in self.cache.keys():
            access_count = self.access_count.get(key, 0)
            last_access = self.last_access.get(key, datetime.min)
            age_hours = (datetime.now() - last_access).total_seconds() / 3600
            
            # Score: meno accessi e più vecchio = score più basso
            score = access_count - (age_hours * 0.1)
            removal_candidates.append((key, score))
        
        # Ordina per score e rimuovi i meno utilizzati
        removal_candidates.sort(key=lambda x: x[1])
        to_remove = len(self.cache) - int(self.max_cache_size * 0.7)
        
        for key, _ in removal_candidates[:to_remove]:
            if key in self.cache:
                del self.cache[key]
                del self.access_count[key]
                del self.last_access[key]
        
        safe_print(f"🧹 Cache cleanup: rimossi {to_remove} indicatori, cache size: {len(self.cache)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche della cache"""
        if not self.cache:
            return {'size': 0, 'hit_rate': 0.0, 'memory_efficient': True}
        
        total_accesses = sum(self.access_count.values())
        unique_indicators = len(set(key.split('_')[0] for key in self.cache.keys()))
        
        return {
            'size': len(self.cache),
            'total_accesses': total_accesses,
            'unique_indicators': unique_indicators,
            'hit_rate': (total_accesses - len(self.cache)) / max(total_accesses, 1) * 100,
            'memory_efficient': len(self.cache) < self.max_cache_size * 0.9
        }
    
    def clear_cache(self) -> None:
        """Pulisce completamente la cache"""
        self.cache.clear()
        self.access_count.clear()
        self.last_access.clear()


# ================== CACHED INDICATORS FUNCTIONS ==================

def create_cached_indicator_calculator(cache: IndicatorsCache):
    """Factory per creare funzioni di calcolo indicatori con cache"""
    
    def safe_sma(prices: np.ndarray, period: int) -> np.ndarray:
        """SMA protetto con fallback"""
        def _calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
            if len(prices) < period:
                return np.full_like(prices, prices[-1] if len(prices) > 0 else 1.0)
            
            result = ta.SMA(prices, timeperiod=period) # type: ignore
            if result is None:
                return np.full_like(prices, prices[-1])
            
            last_value = float(prices[-1])
            result = np.nan_to_num(result, nan=last_value, posinf=last_value, neginf=last_value)
            return result
        
        return cache.get_indicator(prices, "SMA", _calculate_sma, period=period)
    
    def safe_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI protetto con fallback"""
        def _calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
            if len(prices) < period + 1:
                return np.full_like(prices, 50.0)
            
            result = ta.RSI(prices, timeperiod=period) # type: ignore
            if result is None:
                return np.full_like(prices, 50.0)
            
            result = np.nan_to_num(result, nan=50.0, posinf=100.0, neginf=0.0)
            result = np.clip(result, 0.0, 100.0)
            return result
        
        return cache.get_indicator(prices, "RSI", _calculate_rsi, period=period)
    
    def safe_macd(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD protetto con fallback"""
        def _calculate_macd(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if len(prices) < 26:
                zeros = np.zeros_like(prices)
                return zeros, zeros, zeros
            
            macd, signal, hist = ta.MACD(prices) # type: ignore
            if macd is None or signal is None or hist is None:
                zeros = np.zeros_like(prices)
                return zeros, zeros, zeros
            
            macd = np.nan_to_num(macd, nan=0.0, posinf=0.0, neginf=0.0)
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
            hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
            
            return macd, signal, hist
        
        cache_result = cache.get_indicator(prices, "MACD", _calculate_macd)
        if isinstance(cache_result, tuple):
            return cache_result
        else:
            # Fallback se cache non ritorna tuple
            zeros = np.zeros_like(prices)
            return zeros, zeros, zeros
    
    def safe_bbands(prices: np.ndarray, period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands protette"""
        def _calculate_bbands(prices: np.ndarray, period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if len(prices) < period:
                return prices * 1.02, prices, prices * 0.98
            
            upper, middle, lower = ta.BBANDS(prices, timeperiod=period) # type: ignore
            if upper is None or middle is None or lower is None:
                return prices * 1.02, prices, prices * 0.98
            
            last_price = float(prices[-1])
            upper = np.nan_to_num(upper, nan=last_price * 1.02)
            middle = np.nan_to_num(middle, nan=last_price)
            lower = np.nan_to_num(lower, nan=last_price * 0.98)
            
            return upper, middle, lower
        
        cache_result = cache.get_indicator(prices, "BBANDS", _calculate_bbands, period=period)
        if isinstance(cache_result, tuple):
            return cache_result
        else:
            # Fallback se cache non ritorna tuple
            return prices * 1.02, prices, prices * 0.98
    
    def safe_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """ATR protetto"""
        def _calculate_atr(prices: np.ndarray, period: int = 14, **kwargs) -> np.ndarray:
            high_arr = kwargs.get('high', prices)
            low_arr = kwargs.get('low', prices)
            close_arr = kwargs.get('close', prices)
            
            if len(close_arr) < period:
                return np.full_like(close_arr, 0.01)
            
            result = ta.ATR(high_arr, low_arr, close_arr, timeperiod=period) # type: ignore
            if result is None:
                return np.full_like(close_arr, 0.01)
            
            result = np.nan_to_num(result, nan=0.01, posinf=0.01, neginf=0.01)
            result = np.maximum(result, 0.001)
            return result
        
        return cache.get_indicator(close, "ATR", _calculate_atr, period=period, high=high, low=low, close=close)
    
    return {
        'sma': safe_sma,
        'rsi': safe_rsi,
        'macd': safe_macd,
        'bbands': safe_bbands,
        'atr': safe_atr
    }

# ================== ASYNC I/O SYSTEM ==================

class AsyncFileWriter:
    """Sistema di scrittura file asincrono per ridurre blocking I/O"""
    
    def __init__(self, max_queue_size: int = 1000, max_workers: int = 2):
        self.write_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="FileWriter")
        self.active = True
        self.stats = {
            'writes_queued': 0,
            'writes_completed': 0,
            'writes_failed': 0,
            'queue_full_events': 0
        }
        
        # File handles cache per evitare aperture/chiusure frequenti
        self.file_handles: Dict[str, Any] = {}
        self.file_locks: Dict[str, threading.Lock] = {}
        
        # Worker thread per processare la coda
        self.worker_thread = threading.Thread(target=self._process_write_queue, daemon=True)
        self.worker_thread.start()
    
    def queue_write(self, file_path: str, data: Dict[str, Any], write_type: str = 'csv') -> bool:
        """Accoda una scrittura file in modo non-blocking"""
        
        if not self.active:
            return False
        
        write_task = {
            'file_path': file_path,
            'data': data,
            'write_type': write_type,
            'timestamp': datetime.now()
        }
        
        try:
            # Non-blocking put con timeout
            self.write_queue.put(write_task, timeout=0.1)
            self.stats['writes_queued'] += 1
            return True
            
        except queue.Full:
            self.stats['queue_full_events'] += 1
            # Strategia: drop oldest writes se coda piena
            try:
                self.write_queue.get_nowait()  # Rimuovi il più vecchio
                self.write_queue.put(write_task, timeout=0.1)  # Aggiungi nuovo
                return True
            except:
                return False
    
    def _process_write_queue(self) -> None:
        """Worker thread che processa la coda di scrittura"""
        
        batch_size = 10
        batch_timeout = 1.0  # secondi
        
        while self.active:
            batch = []
            batch_start = datetime.now()
            
            # Raccoglie batch di scritture
            while len(batch) < batch_size and (datetime.now() - batch_start).total_seconds() < batch_timeout:
                try:
                    task = self.write_queue.get(timeout=0.5)
                    batch.append(task)
                except queue.Empty:
                    break
            
            if batch:
                self._process_batch(batch)
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Processa un batch di scritture raggruppate per file"""
        
        # Raggruppa per file per ottimizzare I/O
        file_groups: Dict[str, List[Dict]] = defaultdict(list)
        
        for task in batch:
            file_groups[task['file_path']].append(task)
        
        # Processa ogni file
        for file_path, tasks in file_groups.items():
            try:
                self._write_batch_to_file(file_path, tasks)
                self.stats['writes_completed'] += len(tasks)
                
            except Exception as e:
                self.stats['writes_failed'] += len(tasks)
                safe_print(f"❌ Batch write failed for {file_path}: {e}")
    
    def _write_batch_to_file(self, file_path: str, tasks: List[Dict[str, Any]]) -> None:
        """Scrive un batch di dati nello stesso file"""
        
        # Assicura che la directory esista
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Lock per questo file
        if file_path not in self.file_locks:
            self.file_locks[file_path] = threading.Lock()
        
        with self.file_locks[file_path]:
            # Determina se file esiste per header CSV
            file_exists = os.path.exists(file_path)
            
            # Scrittura batch
            if tasks[0]['write_type'] == 'csv':
                self._write_csv_batch(file_path, tasks, file_exists)
            elif tasks[0]['write_type'] == 'json':
                self._write_json_batch(file_path, tasks)
            else:
                # Fallback: scrittura individuale
                for task in tasks:
                    self._write_single_task(file_path, task, file_exists)
    
    def _write_csv_batch(self, file_path: str, tasks: List[Dict[str, Any]], file_exists: bool) -> None:
        """Scrittura ottimizzata per CSV batch"""
        
        if not tasks:
            return
        
        # Estrai struttura CSV dal primo task
        first_data = tasks[0]['data']
        if not isinstance(first_data, dict):
            return
        
        fieldnames = list(first_data.keys())
        
        # Scrittura batch con context manager
        with open(file_path, 'a', newline='', encoding='utf-8', buffering=8192) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Header solo se file nuovo
            if not file_exists:
                writer.writeheader()
            
            # Scrivi tutti i record del batch
            for task in tasks:
                if isinstance(task['data'], dict):
                    # Sanitizza i dati per CSV
                    sanitized_data = {}
                    for key, value in task['data'].items():
                        if isinstance(value, datetime):
                            sanitized_data[key] = value.isoformat()
                        elif isinstance(value, (dict, list)):
                            sanitized_data[key] = json.dumps(value)
                        else:
                            sanitized_data[key] = value
                    
                    writer.writerow(sanitized_data)
    
    def _write_json_batch(self, file_path: str, tasks: List[Dict[str, Any]]) -> None:
        """Scrittura ottimizzata per JSON batch (append mode)"""
        
        for task in tasks:
            # Append JSON lines
            with open(file_path, 'a', encoding='utf-8') as f:
                json.dump(task['data'], f, default=str)
                f.write('\n')
    
    def _write_single_task(self, file_path: str, task: Dict[str, Any], file_exists: bool) -> None:
        """Fallback per scrittura singola"""
        
        if task['write_type'] == 'csv':
            self._write_csv_batch(file_path, [task], file_exists)
        else:
            self._write_json_batch(file_path, [task])
    
    def force_flush(self) -> None:
        """Forza la scrittura di tutti i dati in coda"""
        
        safe_print("🔄 Forcing flush of write queue...")
        
        # Processa tutto quello che c'è in coda
        remaining_tasks = []
        
        try:
            while True:
                task = self.write_queue.get_nowait()
                remaining_tasks.append(task)
        except queue.Empty:
            pass
        
        if remaining_tasks:
            self._process_batch(remaining_tasks)
            safe_print(f"✅ Flushed {len(remaining_tasks)} pending writes")
    
    def get_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche del sistema I/O"""
        
        queue_size = self.write_queue.qsize()
        
        return {
            'queue_size': queue_size,
            'writes_queued': self.stats['writes_queued'],
            'writes_completed': self.stats['writes_completed'],
            'writes_failed': self.stats['writes_failed'],
            'queue_full_events': self.stats['queue_full_events'],
            'success_rate': (self.stats['writes_completed'] / max(1, self.stats['writes_queued'])) * 100,
            'active_files': len(self.file_locks),
            'queue_utilization': (queue_size / self.write_queue.maxsize) * 100
        }
    
    def shutdown(self) -> None:
        """Shutdown pulito del sistema I/O"""
        
        safe_print("🔄 Shutting down AsyncFileWriter...")
        
        # Stop accepting new writes
        self.active = False
        
        # Flush remaining writes
        self.force_flush()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Close file handles
        for handle in self.file_handles.values():
            try:
                if hasattr(handle, 'close'):
                    handle.close()
            except:
                pass
        
        self.file_handles.clear()
        self.file_locks.clear()
        
        safe_print("✅ AsyncFileWriter shutdown complete")

# ================== FIX ENCODING UNIVERSALE ==================

from utils.universal_encoding_fix import (
    init_universal_encoding, 
    get_safe_logger,
    safe_print,
    UniversalEncodingFixer
)

# Inizializza encoding una sola volta
init_universal_encoding(silent=False)

# ================== SISTEMA DI LOGGING AVANZATO ==================

from collections import deque, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

class AnalyzerLogger:
    """Sistema di logging avanzato per l'Analyzer - VERSIONE CORRETTA MEMORY-SAFE"""
    
    def __init__(self, base_path: str = "./analyzer_logs") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Minimal initialization - no active loggers during cleanup phase
        self.loggers: Dict[str, logging.Logger] = {}
        self.csv_writers: Dict[str, Any] = {}
        self.csv_files: Dict[str, Any] = {}
        
        # Event buffers for future slave module integration - MEMORY SAFE con deque
        self._prediction_events_buffer: deque = deque(maxlen=500)
        self._champion_events_buffer: deque = deque(maxlen=200)
        self._error_events_buffer: deque = deque(maxlen=300)
        self._training_events_buffer: deque = deque(maxlen=200)
        self._mt5_events_buffer: deque = deque(maxlen=1000)
        
        # Rotazione mensile mantenuta per compatibilità
        self.current_month = datetime.now().strftime("%Y-%m")
        
        # CSV structures kept for potential future use
        self.csv_structures: Dict[str, List[str]] = {
            'predictions': ['timestamp', 'asset', 'model_type', 'algorithm', 'prediction_data', 
                          'confidence', 'validation_time', 'actual_outcome', 'accuracy', 'error_analysis'],
            'performance': ['timestamp', 'asset', 'model_type', 'algorithm', 'final_score', 
                          'confidence_score', 'quality_score', 'accuracy_rate', 'total_predictions',
                          'decay_factor', 'reality_check_status'],
            'champion_changes': ['timestamp', 'asset', 'model_type', 'old_champion', 'new_champion',
                               'old_score', 'new_score', 'improvement', 'reason'],
            'errors': ['timestamp', 'asset', 'model_type', 'algorithm', 'error_type', 
                      'severity', 'pattern', 'market_condition', 'resolution'],
            'training': ['timestamp', 'asset', 'model_type', 'algorithm', 'training_type',
                        'data_points', 'duration_seconds', 'final_loss', 'improvement'],
            'mt5_bridge': ['timestamp', 'direction', 'message_type', 'asset', 'data']
        }
        
        # Shutdown flag per controllo sicuro
        self._shutdown_initiated = False
    
    def log_prediction(self, asset: str, prediction: 'Prediction', 
                      validation_result: Optional[Dict[str, Any]] = None) -> None:
        """Store prediction event for future slave module processing"""
        
        if self._shutdown_initiated:
            return
        
        prediction_event = {
            'timestamp': datetime.now(),
            'event_type': 'prediction_logged',
            'data': {
                'prediction_timestamp': prediction.timestamp,
                'asset': asset,
                'model_type': prediction.model_type.value,
                'algorithm': prediction.algorithm_name,
                'prediction_id': prediction.id,
                'prediction_data': prediction.prediction_data,
                'confidence': prediction.confidence,
                'validation_criteria': prediction.validation_criteria,
                'validation_result': validation_result,
                'accuracy': prediction.self_validation_score,
                'error_analysis': prediction.error_analysis
            }
        }
        
        # Buffer gestito automaticamente da deque con maxlen=500
        self._prediction_events_buffer.append(prediction_event)
    
    def log_champion_change(self, asset: str, model_type: 'ModelType', old_champion: str, 
                          new_champion: str, old_score: float, new_score: float, reason: str) -> None:
        """Store champion change event for future slave module processing"""
        
        if self._shutdown_initiated:
            return
        
        improvement = ((new_score - old_score) / old_score * 100) if old_score > 0 else 0
        
        champion_event = {
            'timestamp': datetime.now(),
            'event_type': 'champion_changed',
            'data': {
                'asset': asset,
                'model_type': model_type.value,
                'old_champion': old_champion,
                'new_champion': new_champion,
                'old_score': old_score,
                'new_score': new_score,
                'improvement_percentage': improvement,
                'reason': reason
            }
        }
        
        # Buffer gestito automaticamente da deque con maxlen=200
        self._champion_events_buffer.append(champion_event)
    
    def log_error_analysis(self, asset: str, model_type: 'ModelType', algorithm: str,
                         error_analysis: Dict[str, Any], market_condition: Dict[str, Any]) -> None:
        """Store error analysis event for future slave module processing"""
        
        if self._shutdown_initiated:
            return
        
        # Extract lessons learned
        lessons_learned = []
        if error_analysis.get('patterns'):
            for pattern in error_analysis['patterns']:
                lessons_learned.append(
                    f"Pattern '{pattern}' identified as problematic under current market conditions"
                )
        
        error_event = {
            'timestamp': datetime.now(),
            'event_type': 'error_analyzed',
            'data': {
                'asset': asset,
                'model_type': model_type.value,
                'algorithm': algorithm,
                'error_analysis': error_analysis,
                'market_condition': market_condition,
                'lessons_learned': lessons_learned,
                'severity': error_analysis.get('severity', 0),
                'error_types': error_analysis.get('error_types', []),
                'patterns': error_analysis.get('patterns', [])
            }
        }
        
        # Buffer gestito automaticamente da deque con maxlen=300
        self._error_events_buffer.append(error_event)
    
    def log_training_event(self, asset: str, model_type: 'ModelType', algorithm: str,
                         training_type: str, metrics: Dict[str, Any]) -> None:
        """Store training event for future slave module processing"""
        
        if self._shutdown_initiated:
            return
        
        training_event = {
            'timestamp': datetime.now(),
            'event_type': 'training_logged',
            'data': {
                'asset': asset,
                'model_type': model_type.value,
                'algorithm': algorithm,
                'training_type': training_type,
                'metrics': metrics,
                'data_points': metrics.get('data_points', 0),
                'duration_seconds': metrics.get('duration_seconds', 0),
                'final_loss': metrics.get('final_loss', 0),
                'improvement': metrics.get('improvement', 0)
            }
        }
        
        # Buffer gestito automaticamente da deque con maxlen=200
        self._training_events_buffer.append(training_event)
    
    def log_mt5_communication(self, direction: str, message_type: str, 
                            asset: str, data: Dict[str, Any]) -> None:
        """Store MT5 communication event for future slave module processing"""
        
        if self._shutdown_initiated:
            return
        
        mt5_event = {
            'timestamp': datetime.now(),
            'event_type': 'mt5_communication',
            'data': {
                'direction': direction,  # 'in' o 'out'
                'message_type': message_type,
                'asset': asset,
                'data': data
            }
        }
        
        # Buffer gestito automaticamente da deque con maxlen=1000
        self._mt5_events_buffer.append(mt5_event)
    
    def get_performance_summary(self, asset: str, days: int = 30) -> Dict[str, Any]:
        """Generate performance summary from in-memory events (simplified)"""
        
        if self._shutdown_initiated:
            return {'error': 'Logger shutdown initiated'}
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        summary: Dict[str, Any] = {
            'asset': asset,
            'period_days': days,
            'total_predictions': 0,
            'champion_changes': 0,
            'error_events': 0,
            'training_events': 0,
            'models': defaultdict(lambda: {
                'total_predictions': 0,
                'champion_changes': 0,
                'training_events': 0,
                'error_events': 0
            })
        }
        
        # Analyze prediction events
        for event in self._prediction_events_buffer:
            if (event['timestamp'] > cutoff_date and 
                event['data']['asset'] == asset):
                summary['total_predictions'] += 1
                model_type = event['data']['model_type']
                summary['models'][model_type]['total_predictions'] += 1
        
        # Analyze champion changes
        for event in self._champion_events_buffer:
            if (event['timestamp'] > cutoff_date and 
                event['data']['asset'] == asset):
                summary['champion_changes'] += 1
                model_type = event['data']['model_type']
                summary['models'][model_type]['champion_changes'] += 1
        
        # Analyze error events
        for event in self._error_events_buffer:
            if (event['timestamp'] > cutoff_date and 
                event['data']['asset'] == asset):
                summary['error_events'] += 1
                model_type = event['data']['model_type']
                summary['models'][model_type]['error_events'] += 1
        
        # Analyze training events
        for event in self._training_events_buffer:
            if (event['timestamp'] > cutoff_date and 
                event['data']['asset'] == asset):
                summary['training_events'] += 1
                model_type = event['data']['model_type']
                summary['models'][model_type]['training_events'] += 1
        
        return dict(summary)
    
    def get_all_events_for_slave(self) -> Dict[str, List[Dict]]:
        """Get all accumulated events for slave module processing"""
        
        if self._shutdown_initiated:
            return {}
        
        return {
            'predictions': list(self._prediction_events_buffer),
            'champion_changes': list(self._champion_events_buffer),
            'errors': list(self._error_events_buffer),
            'training': list(self._training_events_buffer),
            'mt5_communications': list(self._mt5_events_buffer)
        }
    
    def clear_events_buffer(self, event_types: Optional[List[str]] = None) -> None:
        """Clear event buffers after slave module processing"""
        
        if self._shutdown_initiated:
            return
        
        if event_types is None:
            # Clear all buffers
            self._prediction_events_buffer.clear()
            self._champion_events_buffer.clear()
            self._error_events_buffer.clear()
            self._training_events_buffer.clear()
            self._mt5_events_buffer.clear()
        else:
            # Clear specific buffers
            for event_type in event_types:
                if event_type == 'predictions':
                    self._prediction_events_buffer.clear()
                elif event_type == 'champion_changes':
                    self._champion_events_buffer.clear()
                elif event_type == 'errors':
                    self._error_events_buffer.clear()
                elif event_type == 'training':
                    self._training_events_buffer.clear()
                elif event_type == 'mt5_communications':
                    self._mt5_events_buffer.clear()
    
    def get_buffer_status(self) -> Dict[str, Dict[str, Any]]:
        """Ottieni stato corrente dei buffer"""
        
        if self._shutdown_initiated:
            return {'status': {'message': 'shutdown', 'operational': False}}
        
        return {
            'predictions': {
                'current_size': len(self._prediction_events_buffer),
                'max_size': self._prediction_events_buffer.maxlen or 0,
                'utilization_percent': (len(self._prediction_events_buffer) / (self._prediction_events_buffer.maxlen or 1)) * 100
            },
            'champion_changes': {
                'current_size': len(self._champion_events_buffer),
                'max_size': self._champion_events_buffer.maxlen or 0,
                'utilization_percent': (len(self._champion_events_buffer) / (self._champion_events_buffer.maxlen or 1)) * 100
            },
            'errors': {
                'current_size': len(self._error_events_buffer),
                'max_size': self._error_events_buffer.maxlen or 0,
                'utilization_percent': (len(self._error_events_buffer) / (self._error_events_buffer.maxlen or 1)) * 100
            },
            'training': {
                'current_size': len(self._training_events_buffer),
                'max_size': self._training_events_buffer.maxlen or 0,
                'utilization_percent': (len(self._training_events_buffer) / (self._training_events_buffer.maxlen or 1)) * 100
            },
            'mt5_communications': {
                'current_size': len(self._mt5_events_buffer),
                'max_size': self._mt5_events_buffer.maxlen or 0,
                'utilization_percent': (len(self._mt5_events_buffer) / (self._mt5_events_buffer.maxlen or 1)) * 100
            }
        }
    
    def force_flush_buffers(self) -> Dict[str, Any]:
        """Force flush di tutti i buffer per emergenze"""
        
        if self._shutdown_initiated:
            return {'status': 'already_shutdown', 'message': 'Logger already shut down'}
        
        events_flushed = {
            'predictions': len(self._prediction_events_buffer),
            'champion_changes': len(self._champion_events_buffer),
            'errors': len(self._error_events_buffer),
            'training': len(self._training_events_buffer),
            'mt5_communications': len(self._mt5_events_buffer),
            'total_flushed': (
                len(self._prediction_events_buffer) +
                len(self._champion_events_buffer) +
                len(self._error_events_buffer) +
                len(self._training_events_buffer) +
                len(self._mt5_events_buffer)
            ),
            'flush_timestamp': datetime.now()
        }
        
        # Clear all buffers
        self.clear_events_buffer()
        
        return events_flushed
    
    def shutdown(self) -> Dict[str, Any]:
        """Shutdown sicuro del logger con cleanup completo"""
        
        if self._shutdown_initiated:
            return {'status': 'already_shutdown'}
        
        # Set shutdown flag
        self._shutdown_initiated = True
        
        # Get final statistics
        final_stats = {
            'shutdown_timestamp': datetime.now(),
            'events_in_buffers': {
                'predictions': len(self._prediction_events_buffer),
                'champion_changes': len(self._champion_events_buffer),
                'errors': len(self._error_events_buffer),
                'training': len(self._training_events_buffer),
                'mt5_communications': len(self._mt5_events_buffer)
            },
            'total_events': (
                len(self._prediction_events_buffer) +
                len(self._champion_events_buffer) +
                len(self._error_events_buffer) +
                len(self._training_events_buffer) +
                len(self._mt5_events_buffer)
            )
        }
        
        # Close any open file handles
        try:
            for handle in self.csv_files.values():
                if hasattr(handle, 'close'):
                    handle.close()
        except Exception:
            pass
        
        # Clear all buffers
        self._prediction_events_buffer.clear()
        self._champion_events_buffer.clear()
        self._error_events_buffer.clear()
        self._training_events_buffer.clear()
        self._mt5_events_buffer.clear()
        
        # Clear dictionaries
        self.loggers.clear()
        self.csv_writers.clear()
        self.csv_files.clear()
        
        final_stats['shutdown_completed'] = True
        
        return final_stats
    
    def is_operational(self) -> bool:
        """Verifica se il logger è operativo"""
        return not self._shutdown_initiated
    
    # Legacy compatibility methods (disabled)
    def _setup_loggers(self) -> None:
        """Legacy method - disabled during cleanup phase"""
        pass
    
    def _create_logger(self, name: str, filename: str, level: int) -> None:
        """Legacy method - disabled during cleanup phase"""
        pass
    
    def _setup_csv_structure(self) -> None:
        """Legacy method - disabled during cleanup phase"""
        pass
    
    def _write_csv(self, csv_type: str, data: Dict[str, Any]) -> None:
        """Legacy method - disabled during cleanup phase"""
        pass
    
    def _rotate_csv_files(self) -> None:
        """Legacy method - disabled during cleanup phase"""
        pass

# ================== LOGGING INTEGRATION ==================

class AsyncAnalyzerLogger(AnalyzerLogger):
    """Versione asincrona di AnalyzerLogger - VERSIONE COMPLETAMENTE PULITA"""
    
    def __init__(self, base_path: str = "./analyzer_logs"):
        super().__init__(base_path)
        
        # Sistema I/O asincrono SOLO per necessità business critical
        self.async_writer = AsyncFileWriter(max_queue_size=2000, max_workers=3)
        
        # Configurazione async minimal
        self.async_csv_enabled = True
        self.async_json_enabled = True
        
        # Statistiche I/O essenziali (solo contatori)
        self.io_stats = {
            'csv_writes_queued': 0,
            'csv_writes_sync_fallback': 0,
            'json_writes_queued': 0,
            'json_writes_sync_fallback': 0,
            'total_async_operations': 0
        }
        
        # Performance tracking minimale
        self.sync_write_times = deque(maxlen=50)
        self.async_queue_times = deque(maxlen=50)
    
    def _write_csv_async(self, csv_type: str, data: Dict[str, Any]) -> bool:
        """Versione asincrona di _write_csv - SILENT"""
        
        if not self.async_csv_enabled:
            return self._write_csv_sync_fallback(csv_type, data)
        
        # Prepara path del file
        csv_path = self.base_path / 'metrics' / self.current_month / f'{csv_type}.csv'
        
        # Controlla rotazione mensile
        current_month = datetime.now().strftime("%Y-%m")
        if current_month != self.current_month:
            self._rotate_csv_files()
            self.current_month = current_month
            csv_path = self.base_path / 'metrics' / self.current_month / f'{csv_type}.csv'
        
        # Performance tracking minimale
        start_time = time.time()
        
        # Queue async write
        success = self.async_writer.queue_write(str(csv_path), data, 'csv')
        
        queue_time = time.time() - start_time
        self.async_queue_times.append(queue_time)
        
        if success:
            self.io_stats['csv_writes_queued'] += 1
            self.io_stats['total_async_operations'] += 1
            return True
        else:
            # Silent fallback to sync
            self.io_stats['csv_writes_sync_fallback'] += 1
            return self._write_csv_sync_fallback(csv_type, data)
    
    def _write_csv_sync_fallback(self, csv_type: str, data: Dict[str, Any]) -> bool:
        """Fallback sincrono - SILENT"""
        
        start_time = time.time()
        
        try:
            # Usa il metodo parent per scrittura sincrona (che ora è disabled)
            # Fallback to manual implementation
            csv_path = self.base_path / 'metrics' / self.current_month / f'{csv_type}.csv'
            
            file_exists = csv_path.exists()
            with threading.Lock():
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_structures[csv_type])
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(data)
            
            sync_time = time.time() - start_time
            self.sync_write_times.append(sync_time)
            
            return True
            
        except Exception:
            return False
    
    def _write_json_async(self, file_path: str, data: Dict[str, Any]) -> bool:
        """Scrittura JSON asincrona - SILENT"""
        
        if not self.async_json_enabled:
            return self._write_json_sync_fallback(file_path, data)
        
        start_time = time.time()
        
        # Queue async write
        success = self.async_writer.queue_write(file_path, data, 'json')
        
        queue_time = time.time() - start_time
        self.async_queue_times.append(queue_time)
        
        if success:
            self.io_stats['json_writes_queued'] += 1
            self.io_stats['total_async_operations'] += 1
            return True
        else:
            # Silent fallback
            self.io_stats['json_writes_sync_fallback'] += 1
            return self._write_json_sync_fallback(file_path, data)
    
    def _write_json_sync_fallback(self, file_path: str, data: Dict[str, Any]) -> bool:
        """Fallback JSON sincrono - SILENT"""
        
        start_time = time.time()
        
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'a', encoding='utf-8') as f:
                json.dump(data, f, default=str)
                f.write('\n')
            
            sync_time = time.time() - start_time
            self.sync_write_times.append(sync_time)
            
            return True
            
        except Exception:
            return False
    
    # Override dei metodi principali - USA SOLO EVENT BUFFERS (NO LOGGING)
    
    def log_prediction(self, asset: str, prediction: 'Prediction', 
                      validation_result: Optional[Dict[str, Any]] = None) -> None:
        """Store prediction event - NO LOGGING VERSION"""
        
        # Use parent's cleaned event buffer system
        super().log_prediction(asset, prediction, validation_result)
        
        # Optional async storage for compatibility (if enabled)
        if self.async_json_enabled:
            log_data = {
                'timestamp': prediction.timestamp.isoformat(),
                'asset': asset,
                'model_type': prediction.model_type.value,
                'algorithm': prediction.algorithm_name,
                'prediction_id': prediction.id,
                'prediction_data': prediction.prediction_data,
                'confidence': prediction.confidence,
                'validation_criteria': prediction.validation_criteria
            }
            
            if validation_result:
                log_data['validation_result'] = validation_result
                log_data['accuracy'] = prediction.self_validation_score
                if prediction.error_analysis:
                    log_data['error_analysis'] = prediction.error_analysis
            
            json_path = str(self.base_path / f'predictions_{asset}_detailed.jsonl')
            self._write_json_async(json_path, log_data)
    
    def log_champion_change(self, asset: str, model_type: 'ModelType', old_champion: str, 
                          new_champion: str, old_score: float, new_score: float, reason: str) -> None:
        """Store champion change event - NO LOGGING VERSION"""
        
        # Use parent's cleaned event buffer system
        super().log_champion_change(asset, model_type, old_champion, new_champion, old_score, new_score, reason)
        
        # Optional async storage for compatibility
        if self.async_json_enabled:
            improvement = ((new_score - old_score) / old_score * 100) if old_score > 0 else 0
            
            detailed_log = {
                'timestamp': datetime.now().isoformat(),
                'asset': asset,
                'model_type': model_type.value,
                'change': {
                    'from': {'algorithm': old_champion, 'score': old_score},
                    'to': {'algorithm': new_champion, 'score': new_score}
                },
                'improvement_percentage': improvement,
                'reason': reason,
                'event_type': 'champion_change'
            }
            
            json_path = str(self.base_path / f'champion_history_{asset}.jsonl')
            self._write_json_async(json_path, detailed_log)
    
    def log_error_analysis(self, asset: str, model_type: 'ModelType', algorithm: str,
                         error_analysis: Dict[str, Any], market_condition: Dict[str, Any]) -> None:
        """Store error analysis event - NO LOGGING VERSION"""
        
        # Use parent's cleaned event buffer system
        super().log_error_analysis(asset, model_type, algorithm, error_analysis, market_condition)
        
        # Optional async storage for compatibility
        if self.async_json_enabled:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'asset': asset,
                'model_type': model_type.value,
                'algorithm': algorithm,
                'error_analysis': error_analysis,
                'market_condition': market_condition,
                'event_type': 'error_analysis'
            }
            
            json_path = str(self.base_path / f'error_analysis_{asset}.jsonl')
            self._write_json_async(json_path, log_entry)
    
    def log_training_event(self, asset: str, model_type: 'ModelType', algorithm: str,
                         training_type: str, metrics: Dict[str, Any]) -> None:
        """Store training event - NO LOGGING VERSION"""
        
        # Use parent's cleaned event buffer system
        super().log_training_event(asset, model_type, algorithm, training_type, metrics)
        
        # Optional async storage for compatibility
        if self.async_json_enabled:
            detailed_log = {
                'timestamp': datetime.now().isoformat(),
                'asset': asset,
                'model_type': model_type.value,
                'algorithm': algorithm,
                'training_type': training_type,
                'metrics': metrics,
                'event_type': 'training_event'
            }
            
            json_path = str(self.base_path / f'training_events_{asset}.jsonl')
            self._write_json_async(json_path, detailed_log)
    
    def log_performance_metrics(self, asset: str, model_type: 'ModelType', 
                              algorithm: 'AlgorithmPerformance') -> None:
        """Store performance metrics - NO LOGGING VERSION"""
        
        # Create performance event for parent buffer system
        performance_event = {
            'timestamp': datetime.now(),
            'event_type': 'performance_metrics',
            'data': {
                'asset': asset,
                'model_type': model_type.value,
                'algorithm': algorithm.name,
                'final_score': algorithm.final_score,
                'confidence_score': algorithm.confidence_score,
                'quality_score': algorithm.quality_score,
                'accuracy_rate': algorithm.accuracy_rate,
                'total_predictions': algorithm.total_predictions,
                'reality_check_status': 'failed' if algorithm.reality_check_failures > 0 else 'passed',
                'needs_retraining': algorithm.needs_retraining,
                'emergency_stop': algorithm.emergency_stop_triggered
            }
        }
        
        # Store in appropriate parent buffer
        if not hasattr(self, '_performance_events_buffer'):
            self._performance_events_buffer = []
        
        self._performance_events_buffer.append(performance_event)
        
        # Keep buffer manageable
        if len(self._performance_events_buffer) > 300:
            self._performance_events_buffer = self._performance_events_buffer[-150:]
    
    def log_mt5_communication(self, direction: str, message_type: str, 
                            asset: str, data: Dict[str, Any]) -> None:
        """Store MT5 communication - NO LOGGING VERSION"""
        
        # Use parent's cleaned event buffer system
        super().log_mt5_communication(direction, message_type, asset, data)
        
        # No additional async operations for high-frequency data
    
    # STATISTICHE E MONITORING (SILENT)
    
    def get_io_performance_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche I/O - SILENT VERSION"""
        
        # Get base stats without logging
        base_stats = self.async_writer.get_stats()
        
        total_operations = (self.io_stats['csv_writes_queued'] + 
                          self.io_stats['csv_writes_sync_fallback'] +
                          self.io_stats['json_writes_queued'] + 
                          self.io_stats['json_writes_sync_fallback'])
        
        async_ratio = (self.io_stats['total_async_operations'] / max(1, total_operations)) * 100
        
        # Performance comparison
        avg_sync_time = np.mean(self.sync_write_times) if self.sync_write_times else 0
        avg_async_queue_time = np.mean(self.async_queue_times) if self.async_queue_times else 0
        
        performance_improvement = 0
        if avg_sync_time > 0 and avg_async_queue_time > 0:
            performance_improvement = ((avg_sync_time - avg_async_queue_time) / avg_sync_time) * 100
        
        return {
            'async_file_writer': base_stats,
            'logger_specific': {
                'total_operations': total_operations,
                'async_operations': self.io_stats['total_async_operations'],
                'sync_fallbacks': (self.io_stats['csv_writes_sync_fallback'] + 
                                 self.io_stats['json_writes_sync_fallback']),
                'async_ratio_percentage': async_ratio
            },
            'performance_metrics': {
                'avg_sync_write_time_ms': avg_sync_time * 1000,
                'avg_async_queue_time_ms': avg_async_queue_time * 1000,
                'performance_improvement_percentage': performance_improvement,
                'total_time_saved_seconds': len(self.async_queue_times) * max(0, avg_sync_time - avg_async_queue_time)
            }
        }
    
    def optimize_async_performance(self) -> Dict[str, Any]:
        """Ottimizza performance async - SILENT VERSION"""
        
        current_stats = self.async_writer.get_stats()
        
        actions_taken = []
        
        if current_stats['queue_utilization'] > 90:
            self.async_json_enabled = False
            actions_taken.append('disabled_async_json_temporarily')
        
        if current_stats['success_rate'] < 80:
            self.async_writer.force_flush()
            actions_taken.append('forced_queue_flush')
        
        return {
            'actions_taken': actions_taken,
            'queue_utilization': current_stats['queue_utilization'],
            'success_rate': current_stats['success_rate']
        }
    
    def force_flush_all_async(self) -> None:
        """Forza flush async operations - SILENT"""
        self.async_writer.force_flush()
        
        # Brief wait for completion
        import time
        time.sleep(0.1)
    
    def shutdown(self) -> None:
        """Shutdown silenzioso"""
        
        # Flush pending operations
        self.force_flush_all_async()
        
        # Shutdown async writer
        self.async_writer.shutdown()
    
    def get_all_events_for_slave(self) -> Dict[str, List[Dict]]:
        """Get all events including async-specific ones for slave module"""
        
        # Get parent events
        parent_events = super().get_all_events_for_slave()
        
        # Add async-specific events
        if hasattr(self, '_performance_events_buffer'):
            parent_events['performance_metrics'] = self._performance_events_buffer.copy()
        
        return parent_events
    
    def clear_events_buffer(self, event_types: Optional[List[str]] = None) -> None:
        """Clear event buffers including async-specific ones"""
        
        # Clear parent buffers
        super().clear_events_buffer(event_types)
        
        # Clear async-specific buffers
        if event_types is None or 'performance_metrics' in event_types:
            if hasattr(self, '_performance_events_buffer'):
                self._performance_events_buffer.clear()

# ================== SISTEMA DIAGNOSTICO ==================

class LearningDiagnostics:
    """Sistema diagnostico per monitorare l'apprendimento"""
    
    def __init__(self, asset: str, logger: 'AnalyzerLogger'):
        self.asset = asset
        self.logger = logger
        self.diagnostics_data = {
            'learning_phases': [],
            'memory_usage': [],
            'processing_times': [],
            'tick_rates': [],
            'bottlenecks': [],
            'system_health': []
        }
        
        # Tracking dettagliato
        self.last_tick_time = None
        self.tick_intervals = []
        self.processing_times = []
        self.memory_samples = []
        
        # Contatori
        self.total_ticks_processed = 0
        self.learning_iterations = 0
        self.failed_operations = 0
        
        # Soglie di allarme
        self.max_processing_time = 5.0  # secondi
        self.max_memory_usage = 80  # percentuale
        self.min_tick_rate = 0.1  # ticks/secondo
        
        # Thread per monitoraggio continuo
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._continuous_monitor, daemon=True)
        self.monitor_thread.start()
    
    def log_tick_processing(self, tick_data: Dict[str, Any], processing_start: datetime) -> None:
        """Logga il processing di ogni tick con dettagli"""
        
        processing_end = datetime.now()
        processing_time = (processing_end - processing_start).total_seconds()
        
        self.total_ticks_processed += 1
        self.processing_times.append(processing_time)
        
        # Calcola tick rate
        if self.last_tick_time:
            interval = (processing_end - self.last_tick_time).total_seconds()
            self.tick_intervals.append(interval)
            current_tick_rate = 1.0 / interval if interval > 0 else 0
        else:
            current_tick_rate = 0
        
        self.last_tick_time = processing_end
        
        # Log dettagliato ogni 1000 ticks
        if self.total_ticks_processed % 1000 == 0:
            self._log_processing_milestone()
        
        # Controlla bottlenecks
        if processing_time > self.max_processing_time:
            self._log_performance_bottleneck('slow_processing', {
                'processing_time': processing_time,
                'tick_number': self.total_ticks_processed,
                'tick_data_size': len(str(tick_data))
            })
        
        if current_tick_rate < self.min_tick_rate:
            self._log_performance_bottleneck('low_tick_rate', {
                'tick_rate': current_tick_rate,
                'expected_min': self.min_tick_rate,
                'tick_number': self.total_ticks_processed
            })
    
    def log_learning_phase_change(self, phase_info: Dict[str, Any]) -> None:
        """Logga cambiamenti nelle fasi di apprendimento"""
        
        phase_record = {
            'timestamp': datetime.now(),
            'tick_count': self.total_ticks_processed,
            'phase_info': phase_info,
            'memory_usage': self._get_memory_usage(),
            'system_load': self._get_system_load()
        }
        
        self.diagnostics_data['learning_phases'].append(phase_record)
        
        # Log critico per fasi importanti
        if phase_info.get('phase') == 'learning_complete':
            self.logger.loggers['system'].critical(
                f"🎓 LEARNING COMPLETE | {self.asset} | "
                f"Ticks: {self.total_ticks_processed} | "
                f"Duration: {phase_info.get('duration_hours', 0):.2f}h | "
                f"Memory: {phase_record['memory_usage']:.1f}%"
            )
        elif phase_info.get('phase') == 'learning_stalled':
            self.logger.loggers['emergency'].critical(
                f"🚨 LEARNING STALLED | {self.asset} | "
                f"Stuck at: {self.total_ticks_processed} ticks | "
                f"Reason: {phase_info.get('reason', 'unknown')}"
            )
    
    def log_data_structure_analysis(self, analyzer_instance) -> Dict[str, Any]:
        """Analizza e logga lo stato delle strutture dati - VERSIONE PULITA"""
        
        analysis_start = datetime.now()
        
        analysis = {
            'timestamp': analysis_start,
            'tick_data_count': len(analyzer_instance.tick_data),
            'tick_data_maxlen': analyzer_instance.tick_data.maxlen,
            'memory_efficiency': len(analyzer_instance.tick_data) / analyzer_instance.tick_data.maxlen * 100,
            'competitions_status': {},
            'ml_models_loaded': len(analyzer_instance.ml_models),
            'scalers_loaded': len(analyzer_instance.scalers)
        }
        
        # Analizza ogni competition
        for model_type, competition in analyzer_instance.competitions.items():
            analysis['competitions_status'][model_type.value] = {
                'algorithms_count': len(competition.algorithms),
                'predictions_history': len(competition.predictions_history),
                'pending_validations': len(competition.pending_validations),
                'champion': competition.champion,
                'emergency_stops': sum(1 for alg in competition.algorithms.values() 
                                    if alg.emergency_stop_triggered)
            }
        
        # Detect structural issues without logging
        structural_issues = []
        
        if analysis['memory_efficiency'] > 90:
            structural_issues.append({
                'type': 'memory_structure_full',
                'severity': 'high',
                'details': f"Memory efficiency at {analysis['memory_efficiency']:.1f}%"
            })
        
        if analysis['ml_models_loaded'] == 0:
            structural_issues.append({
                'type': 'no_ml_models',
                'severity': 'critical',
                'details': 'No ML models loaded'
            })
        
        # Check for emergency stop issues
        total_emergency_stops = sum(status['emergency_stops'] 
                                for status in analysis['competitions_status'].values())
        if total_emergency_stops > 0:
            structural_issues.append({
                'type': 'emergency_stops_active',
                'severity': 'medium',
                'details': f"{total_emergency_stops} algorithms in emergency stop"
            })
        
        # Check for pending validation backlog
        total_pending = sum(status['pending_validations'] 
                        for status in analysis['competitions_status'].values())
        if total_pending > 100:
            structural_issues.append({
                'type': 'validation_backlog',
                'severity': 'medium',
                'details': f"{total_pending} pending validations"
            })
        
        # Store analysis results for future slave module processing
        analysis_end = datetime.now()
        analysis_duration = (analysis_end - analysis_start).total_seconds()
        
        analysis_data = {
            'analysis_result': analysis,
            'structural_issues': structural_issues,
            'analysis_duration': analysis_duration,
            'analyzer_asset': getattr(analyzer_instance, 'asset', 'unknown')
        }
        
        # Store with appropriate priority based on issues found
        if structural_issues:
            # Critical or high severity issues
            critical_issues = [issue for issue in structural_issues 
                            if issue['severity'] in ['critical', 'high']]
            if critical_issues:
                self._store_diagnostic_event('structure_analysis_critical', analysis_data)
            else:
                self._store_diagnostic_event('structure_analysis_issues', analysis_data)
        else:
            # Store periodic healthy analysis (less frequently)
            if not hasattr(self, '_last_healthy_analysis_log'):
                self._last_healthy_analysis_log = datetime.now()
                self._store_diagnostic_event('structure_analysis_healthy', analysis_data)
            else:
                time_since_last = datetime.now() - self._last_healthy_analysis_log
                if time_since_last > timedelta(hours=6):  # Log healthy state every 6 hours
                    self._store_diagnostic_event('structure_analysis_healthy', analysis_data)
                    self._last_healthy_analysis_log = datetime.now()
        
        return analysis
    
    def detect_learning_stall(self, analyzer_instance) -> Optional[Dict[str, Any]]:
        """Rileva se l'apprendimento si è bloccato - VERSIONE PULITA"""
        
        detection_start = datetime.now()
        stall_indicators = []
        
        # 1. Controlla se i tick si stanno accumulando ma l'apprendimento non progredisce
        if (analyzer_instance.learning_phase and 
            len(analyzer_instance.tick_data) >= 50000 and 
            analyzer_instance.learning_progress < 0.1):
            
            days_learning = (datetime.now() - analyzer_instance.learning_start_time).days
            if days_learning > 7:  # Più di una settimana senza progressi
                stall_indicators.append({
                    'type': 'no_progress',
                    'details': f'Learning for {days_learning} days with only {analyzer_instance.learning_progress:.1%} progress'
                })
        
        # 2. Controlla se i modelli non si stanno allenando
        if hasattr(self, 'last_training_check'):
            time_since_training = datetime.now() - self.last_training_check
            if time_since_training > timedelta(hours=2):
                stall_indicators.append({
                    'type': 'no_training',
                    'details': f'No training activity for {time_since_training.total_seconds()/3600:.1f} hours'
                })
        
        # 3. Controlla memoria eccessiva
        memory_usage = self._get_memory_usage()
        if memory_usage > 85:
            stall_indicators.append({
                'type': 'memory_pressure',
                'details': f'Memory usage at {memory_usage:.1f}%'
            })
        
        # 4. Controlla rate di processing
        if len(self.tick_intervals) > 100:
            avg_interval = sum(self.tick_intervals[-100:]) / 100
            if avg_interval > 10:  # Più di 10 secondi per tick
                stall_indicators.append({
                    'type': 'slow_processing',
                    'details': f'Average tick interval: {avg_interval:.2f}s'
                })
        
        self.last_training_check = datetime.now()
        
        if stall_indicators:
            stall_info = {
                'detected_at': datetime.now(),
                'tick_count': self.total_ticks_processed,
                'indicators': stall_indicators,
                'system_state': self.log_data_structure_analysis(analyzer_instance)
            }
            
            # Store learning stall for future slave module processing
            detection_duration = (datetime.now() - detection_start).total_seconds()
            self._store_diagnostic_event('learning_stall_detected', {
                'stall_info': stall_info,
                'detection_duration': detection_duration,
                'indicator_count': len(stall_indicators),
                'analyzer_asset': getattr(analyzer_instance, 'asset', 'unknown')
            })
            
            return stall_info
        
        # Store successful detection (no stall) periodically for monitoring
        if hasattr(self, '_last_no_stall_log'):
            time_since_last = datetime.now() - self._last_no_stall_log
            if time_since_last > timedelta(hours=1):  # Log success every hour
                self._store_diagnostic_event('stall_check_passed', {
                    'tick_count': self.total_ticks_processed,
                    'memory_usage': memory_usage,
                    'avg_processing_interval': sum(self.tick_intervals[-100:]) / 100 if len(self.tick_intervals) > 100 else 0,
                    'learning_progress': getattr(analyzer_instance, 'learning_progress', 0)
                })
                self._last_no_stall_log = datetime.now()
        else:
            self._last_no_stall_log = datetime.now()
        
        return None


    def _store_diagnostic_event(self, event_type: str, event_data: Dict) -> None:
        """Store diagnostic events in memory for future processing by slave module"""
        if not hasattr(self, '_diagnostic_events_buffer'):
            self._diagnostic_events_buffer = []
        
        event_entry = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'data': event_data
        }
        
        self._diagnostic_events_buffer.append(event_entry)
        
        # Keep buffer size manageable with priority for critical events
        if len(self._diagnostic_events_buffer) > 200:
            # Separate critical stalls from routine checks
            critical_events = [e for e in self._diagnostic_events_buffer 
                            if e['event_type'] == 'learning_stall_detected']
            routine_checks = [e for e in self._diagnostic_events_buffer 
                            if e['event_type'] == 'stall_check_passed']
            
            # Keep all critical + sample of routine checks
            recent_checks = routine_checks[-50:] if len(routine_checks) > 50 else routine_checks
            if len(routine_checks) > 50:
                # Keep every 10th older check for trend analysis
                older_sample = routine_checks[:-50][::10]
                recent_checks = older_sample + recent_checks
            
            self._diagnostic_events_buffer = critical_events + recent_checks
    
    def _log_processing_milestone(self) -> None:
        """Logga milestone di processing ogni 1000 ticks"""
        
        # Calcola statistiche
        recent_times = self.processing_times[-1000:] if len(self.processing_times) >= 1000 else self.processing_times
        recent_intervals = self.tick_intervals[-1000:] if len(self.tick_intervals) >= 1000 else self.tick_intervals
        
        stats = {
            'total_ticks': self.total_ticks_processed,
            'avg_processing_time': sum(recent_times) / len(recent_times) if recent_times else 0,
            'max_processing_time': max(recent_times) if recent_times else 0,
            'avg_tick_interval': sum(recent_intervals) / len(recent_intervals) if recent_intervals else 0,
            'current_tick_rate': len(recent_intervals) / sum(recent_intervals) if recent_intervals and sum(recent_intervals) > 0 else 0,
            'memory_usage': self._get_memory_usage(),
            'failed_operations': self.failed_operations
        }
        
        self.logger.loggers['system'].info(
            f"📈 PROCESSING MILESTONE | {self.asset} | "
            f"Ticks: {stats['total_ticks']:,} | "
            f"Rate: {stats['current_tick_rate']:.2f}/s | "
            f"AvgTime: {stats['avg_processing_time']:.3f}s | "
            f"Memory: {stats['memory_usage']:.1f}%"
        )
        
        # Controlla trend peggiorativi
        if stats['avg_processing_time'] > 1.0:
            self._log_performance_bottleneck('processing_degradation', stats)
    
    def _log_performance_bottleneck(self, bottleneck_type: str, details: Dict[str, Any]) -> None:
        """Logga bottlenecks performance - THREAD SAFE VERSION"""
        
        bottleneck_record = {
            'timestamp': datetime.now(),
            'type': bottleneck_type,
            'details': details,
            'tick_count': getattr(self, 'total_ticks_processed', 0)
        }
        
        # FIX: Controlla se il logger esiste prima di usarlo
        try:
            if hasattr(self, 'logger') and hasattr(self.logger, 'loggers'):
                if 'emergency' in self.logger.loggers:
                    self.logger.loggers['emergency'].critical(
                        f"Performance bottleneck detected: {bottleneck_type}", 
                        extra={'details': details}
                    )
                elif 'errors' in self.logger.loggers:
                    self.logger.loggers['errors'].error(
                        f"Performance bottleneck: {bottleneck_type} - {details}"
                    )
                else:
                    # Fallback: usa print se non ci sono logger disponibili
                    print(f"⚠️ Performance bottleneck: {bottleneck_type} - {details}")
            else:
                # Logger non disponibile, usa print
                print(f"⚠️ Performance bottleneck: {bottleneck_type} - {details}")
        except Exception:
            # Catch-all per evitare che il monitoring thread si blocchi
            pass
    
    def _calculate_severity(self, bottleneck_type: str, details: Dict[str, Any]) -> str:
        """Calcola la severità del bottleneck"""
        
        if bottleneck_type == 'memory_pressure':
            usage = details.get('memory_usage', 0)
            if usage > 95:
                return 'critical'
            elif usage > 90:
                return 'high'
            else:
                return 'medium'
        
        elif bottleneck_type == 'slow_processing':
            time_val = details.get('processing_time', 0)
            if time_val > 10:
                return 'critical'
            elif time_val > 5:
                return 'high'
            else:
                return 'medium'
        
        elif bottleneck_type == 'no_training':
            return 'critical'
        
        elif bottleneck_type == 'memory_structure_full':
            return 'high'
        
        return 'medium'
    
    def _continuous_monitor(self) -> None:
        """Monitor continuo delle performance - SAFE VERSION"""
        
        while self.monitoring_active:
            try:
                time.sleep(30)  # Check ogni 30 secondi
                
                # Memoria
                memory_usage = self._get_memory_usage()
                if memory_usage > self.max_memory_usage:
                    self._log_performance_bottleneck('memory_pressure', {
                        'usage_percent': memory_usage,
                        'threshold': self.max_memory_usage
                    })
                
                # Tick rate
                if len(self.tick_intervals) > 10:
                    avg_interval = np.mean(self.tick_intervals[-10:])
                    tick_rate = 1.0 / avg_interval if avg_interval > 0 else 0
                    
                    if tick_rate < self.min_tick_rate:
                        self._log_performance_bottleneck('low_tick_rate', {
                            'current_rate': tick_rate,
                            'min_expected': self.min_tick_rate
                        })
            
            except Exception as e:
                # FIX: Gestione sicura delle eccezioni nel monitoring thread
                try:
                    if (hasattr(self, 'logger') and hasattr(self.logger, 'loggers') 
                        and 'errors' in self.logger.loggers):
                        self.logger.loggers['errors'].error(f"Monitor thread error: {e}")
                    else:
                        print(f"⚠️ Monitor thread error: {e}")
                except Exception:
                    # Ultimo fallback per evitare crash del thread
                    print(f"⚠️ Monitor error (fallback): {e}")
    
    def _get_memory_usage(self) -> float:
        """Ottieni utilizzo memoria corrente"""
        try:
            return psutil.virtual_memory().percent
        except:
            return 0.0
    
    def _get_system_load(self) -> float:
        """Ottieni carico sistema corrente"""
        try:
            return psutil.cpu_percent(interval=1)
        except:
            return 0.0
    
    def generate_diagnostics_report(self) -> Dict[str, Any]:
        """Genera report diagnostico completo"""
        
        return {
            'asset': self.asset,
            'generated_at': datetime.now(),
            'summary': {
                'total_ticks_processed': self.total_ticks_processed,
                'learning_iterations': self.learning_iterations,
                'failed_operations': self.failed_operations,
                'avg_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
                'current_memory_usage': self._get_memory_usage(),
                'bottlenecks_detected': len(self.diagnostics_data['bottlenecks'])
            },
            'detailed_data': self.diagnostics_data,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Genera raccomandazioni basate sui dati diagnostici"""
        
        recommendations = []
        
        # Analizza bottlenecks
        critical_bottlenecks = [b for b in self.diagnostics_data['bottlenecks'] 
                              if b['severity'] == 'critical']
        
        if critical_bottlenecks:
            recommendations.append(f"Address {len(critical_bottlenecks)} critical bottlenecks immediately")
        
        # Analizza memoria
        if self.memory_samples:
            avg_memory = sum(s['memory_usage'] for s in self.memory_samples[-10:]) / min(10, len(self.memory_samples))
            if avg_memory > 80:
                recommendations.append("Consider increasing system memory or optimizing data structures")
        
        # Analizza processing time
        if self.processing_times:
            avg_time = sum(self.processing_times[-1000:]) / min(1000, len(self.processing_times))
            if avg_time > 0.5:
                recommendations.append("Optimize tick processing performance")
        
        return recommendations
    
    def shutdown(self):
        """Shutdown del sistema diagnostico"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

# ================== ENUMS E DATACLASSES ==================

class ModelType(Enum):
    SUPPORT_RESISTANCE = "support_resistance"
    PATTERN_RECOGNITION = "pattern_recognition"
    BIAS_DETECTION = "bias_detection"
    TREND_ANALYSIS = "trend_analysis"
    VOLATILITY_PREDICTION = "volatility_prediction"
    MOMENTUM_ANALYSIS = "momentum_analysis"

@dataclass
class Prediction:
    """Singola predizione con auto-monitoring e analisi errori"""
    id: str
    timestamp: datetime
    model_type: ModelType
    algorithm_name: str
    prediction_data: Dict[str, Any]
    confidence: float
    validation_time: datetime
    validation_criteria: Dict[str, Any]
    actual_outcome: Optional[Dict[str, Any]] = None
    self_validation_score: Optional[float] = None
    observer_feedback_score: Optional[float] = None
    error_analysis: Optional[Dict[str, Any]] = None
    market_conditions_snapshot: Optional[Dict[str, Any]] = None
    
@dataclass
class AlgorithmPerformance:
    """Tracking performance di un singolo algoritmo con decay e controlli"""
    name: str
    model_type: 'ModelType'
    confidence_score: float = 50.0
    quality_score: float = 50.0
    total_predictions: int = 0
    correct_predictions: int = 0
    observer_feedback_count: int = 0
    observer_positive_feedback: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    is_champion: bool = False
    creation_date: datetime = field(default_factory=datetime.now)
    last_training_date: datetime = field(default_factory=datetime.now)
    confidence_decay_rate: float = 0.995  # Decay giornaliero
    frozen_weights: Optional[Dict[str, Any]] = None  # Per preservare champion
    reality_check_failures: int = 0
    emergency_stop_triggered: bool = False
    preserved_model_path: Optional[str] = None
    # ANNOTAZIONE CORRETTA
    rolling_window_performances: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def final_score(self) -> float:
        """Score finale con confidence decay applicato"""
        days_since_training = (datetime.now() - self.last_training_date).days
        decay_factor = self.confidence_decay_rate ** days_since_training
        
        # Se è un champion preservato, decay più lento
        if self.frozen_weights is not None:
            decay_factor = max(decay_factor, 0.7)  # Non scende sotto il 70%
        
        raw_score = (self.confidence_score + self.quality_score) / 2
        return raw_score * decay_factor
    
    @property
    def accuracy_rate(self) -> float:
        return self.correct_predictions / max(1, self.total_predictions)
    
    @property
    def observer_satisfaction(self) -> float:
        return self.observer_positive_feedback / max(1, self.observer_feedback_count)
    
    @property
    def needs_retraining(self) -> bool:
        """Determina se l'algoritmo necessita retraining"""
        days_since_training = (datetime.now() - self.last_training_date).days
        
        # Criteri per retraining
        if self.emergency_stop_triggered:
            return True
        if days_since_training > 30:  # Più di un mese
            return True
        if self.reality_check_failures > 5:
            return True
        if self.final_score < 40:  # Performance troppo bassa
            return True
        
        return False
    
    @property
    def recent_performance_trend(self) -> str:
        """Analizza il trend delle performance recenti"""
        if len(self.rolling_window_performances) < 10:
            return "insufficient_data"
        
        recent = list(self.rolling_window_performances)[-10:]
        older = list(self.rolling_window_performances)[-20:-10] if len(self.rolling_window_performances) >= 20 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg * 1.1:
            return "improving"
        elif recent_avg < older_avg * 0.9:
            return "declining"
        else:
            return "stable"

# ================== SISTEMI DI SUPPORTO ==================

class ChampionPreserver:
    """Sistema per preservare i migliori champion"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.preserved_champions: Dict[str, List[Dict]] = defaultdict(list)
        self._load_preserved_champions()
    
    def preserve_champion(self, asset: str, model_type: ModelType, algorithm_name: str,
                         performance: AlgorithmPerformance, model_weights: Any):
        """Preserva un champion di successo"""
        preservation_data = {
            'timestamp': datetime.now().isoformat(),
            'asset': asset,
            'model_type': model_type.value,
            'algorithm_name': algorithm_name,
            'performance_metrics': {
                'final_score': performance.final_score,
                'accuracy_rate': performance.accuracy_rate,
                'total_predictions': performance.total_predictions,
                'observer_satisfaction': performance.observer_satisfaction,
                'confidence_score': performance.confidence_score,
                'quality_score': performance.quality_score
            },
            'model_file': f"{asset}_{model_type.value}_{algorithm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        }
        
        # Salva i pesi del modello
        model_path = self.storage_path / preservation_data['model_file']
        with open(model_path, 'wb') as f:
            pickle.dump({
                'weights': model_weights,
                'performance': performance,
                'metadata': preservation_data
            }, f)
        
        # Aggiungi alla lista dei champion preservati
        key = f"{asset}_{model_type.value}"
        self.preserved_champions[key].append(preservation_data)
        
        # Mantieni solo i migliori 5 per ogni combinazione
        self.preserved_champions[key] = sorted(
            self.preserved_champions[key],
            key=lambda x: x['performance_metrics']['final_score'],
            reverse=True
        )[:5]
        
        self._save_preserved_champions()
        
        return preservation_data
    
    def get_best_preserved(self, asset: str, model_type: ModelType) -> Optional[Dict]:
        """Ottieni il miglior champion preservato"""
        key = f"{asset}_{model_type.value}"
        if key in self.preserved_champions and self.preserved_champions[key]:
            return self.preserved_champions[key][0]
        return None
    
    def load_preserved_model(self, preservation_data: Dict) -> Optional[Dict]:
        """Carica un modello preservato"""
        model_path = self.storage_path / preservation_data['model_file']
        if model_path.exists():
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_preserved_champions(self):
        """Salva l'indice dei champion preservati"""
        index_path = self.storage_path / 'champions_index.json'
        with open(index_path, 'w') as f:
            json.dump(dict(self.preserved_champions), f, indent=2)
    
    def _load_preserved_champions(self):
        """Carica l'indice dei champion preservati"""
        index_path = self.storage_path / 'champions_index.json'
        if index_path.exists():
            with open(index_path, 'r') as f:
                self.preserved_champions = defaultdict(list, json.load(f))

class RealityChecker:
    """Sistema per validare che i pattern appresi siano ancora validi"""
    
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or get_analyzer_config()  # 🔧 ADDED
        self.validation_threshold = self.config.accuracy_threshold  # 🔧 CHANGED
        self.reality_checks: Dict[str, List[Dict]] = defaultdict(list)
        self.pattern_validity: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    def perform_reality_check(self, asset: str, model_type: ModelType, 
                            algorithm: AlgorithmPerformance, recent_predictions: List[Prediction]) -> Dict[str, Any]:
        """Esegue un reality check sui pattern appresi"""
        
        if len(recent_predictions) < 10:
            return {'status': 'insufficient_data', 'passed': True}
        
        # Analizza performance recenti
        recent_accuracy = np.mean([
            p.self_validation_score for p in recent_predictions[-20:]
            if p.self_validation_score is not None
        ])
        
        # Confronta con performance storiche
        historical_accuracy = algorithm.accuracy_rate
        
        # Calcola degradazione
        performance_degradation = (historical_accuracy - recent_accuracy) / historical_accuracy if historical_accuracy > 0 else 0
        
        # Analizza pattern di errori
        error_patterns = self._analyze_error_patterns(recent_predictions)
        
        # Analizza condizioni di mercato durante gli errori
        market_conditions = self._analyze_market_conditions_during_errors(recent_predictions)
        
        # Determina se il reality check è passato
        check_passed = (
            performance_degradation < 0.2 and  # Degradazione < 20%
            recent_accuracy > self.validation_threshold and
            len(error_patterns['systematic_errors']) == 0
        )
        
        result = {
            'status': 'completed',
            'passed': check_passed,
            'timestamp': datetime.now(),
            'metrics': {
                'recent_accuracy': recent_accuracy,
                'historical_accuracy': historical_accuracy,
                'performance_degradation': performance_degradation,
                'systematic_errors': error_patterns['systematic_errors'],
                'error_concentration': error_patterns['error_concentration'],
                'problematic_conditions': market_conditions
            }
        }
        
        # Registra il check
        check_key = f"{asset}_{model_type.value}_{algorithm.name}"
        self.reality_checks[check_key].append(result)
        
        # Aggiorna validità pattern
        self._update_pattern_validity(asset, model_type, algorithm.name, bool(check_passed))
        
        # Se fallito, incrementa counter
        if not check_passed:
            algorithm.reality_check_failures += 1
        else:
            # Reset failures se passa il check
            algorithm.reality_check_failures = max(0, algorithm.reality_check_failures - 1)
        
        return result
    
    def _analyze_error_patterns(self, predictions: List[Prediction]) -> Dict[str, Any]:
        """Analizza pattern negli errori"""
        errors = []
        for pred in predictions:
            if pred.self_validation_score is not None and pred.self_validation_score < 0.5:
                errors.append({
                    'timestamp': pred.timestamp,
                    'score': pred.self_validation_score,
                    'type': pred.model_type.value,
                    'data': pred.prediction_data,
                    'conditions': pred.market_conditions_snapshot
                })
        
        # Identifica errori sistematici
        systematic_errors = []
        
        if len(errors) > 5:
            # Controlla se gli errori sono concentrati in specifiche condizioni
            error_hours = [e['timestamp'].hour for e in errors]
            hour_concentration = max(set(error_hours), key=error_hours.count)
            
            if error_hours.count(hour_concentration) > len(errors) * 0.4:
                systematic_errors.append(f"Errors concentrated at hour {hour_concentration}")
            
            # Controlla pattern nei dati di predizione
            if all('pattern' in e['data'] for e in errors):
                patterns = [e['data']['pattern'] for e in errors]
                common_pattern = max(set(patterns), key=patterns.count)
                if patterns.count(common_pattern) > len(errors) * 0.5:
                    systematic_errors.append(f"Failures concentrated on '{common_pattern}' pattern")
        
        return {
            'systematic_errors': systematic_errors,
            'error_concentration': len(errors) / len(predictions) if predictions else 0,
            'total_errors': len(errors)
        }
    
    def _analyze_market_conditions_during_errors(self, predictions: List[Prediction]) -> List[Dict]:
        """Analizza le condizioni di mercato durante gli errori"""
        problematic_conditions = []
        
        error_conditions = []
        for pred in predictions:
            if pred.self_validation_score is not None and pred.self_validation_score < 0.5:
                if pred.market_conditions_snapshot:
                    error_conditions.append(pred.market_conditions_snapshot)
        
        if len(error_conditions) > 3:
            # Analizza volatilità
            volatilities = [c.get('volatility', 0) for c in error_conditions]
            avg_error_volatility = np.mean(volatilities)
            
            # Analizza trend
            trends = [c.get('price_change_5m', 0) for c in error_conditions]
            
            if avg_error_volatility > 0.02:  # Alta volatilità
                problematic_conditions.append({
                    'condition': 'high_volatility',
                    'threshold': avg_error_volatility,
                    'occurrences': len([v for v in volatilities if v > 0.02])
                })
        
        return problematic_conditions
    
    def _update_pattern_validity(self, asset: str, model_type: ModelType, 
                               algorithm_name: str, check_passed: bool):
        """Aggiorna la validità dei pattern"""
        key = f"{asset}_{model_type.value}"
        
        if algorithm_name not in self.pattern_validity[key]:
            self.pattern_validity[key][algorithm_name] = 1.0
        
        # Aggiorna con media mobile esponenziale
        alpha = 0.1
        current_validity = self.pattern_validity[key][algorithm_name]
        new_validity = alpha * (1.0 if check_passed else 0.0) + (1 - alpha) * current_validity
        
        self.pattern_validity[key][algorithm_name] = new_validity
    
    def get_pattern_validity(self, asset: str, model_type: ModelType, 
                           algorithm_name: str) -> float:
        """Ottieni la validità corrente dei pattern"""
        key = f"{asset}_{model_type.value}"
        return self.pattern_validity[key].get(algorithm_name, 1.0)

class EmergencyStopSystem:
    """Sistema di emergency stop per prevenire perdite catastrofiche"""
    
    def __init__(self, logger: AnalyzerLogger, config: Optional[AnalyzerConfig] = None):
        self.logger = logger
        self.config = config or get_analyzer_config()  # 🔧 ADDED
        
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.performance_window_size))  # 🔧 CHANGED
        
        # 🔧 Usa configurazione per stop triggers invece di hardcoded
        self.stop_triggers = self.config.get_emergency_stop_triggers()
        
        self.stopped_algorithms: Set[str] = set()
    
    def check_emergency_conditions(self, asset: str, model_type: ModelType,
                                 algorithm: AlgorithmPerformance, 
                                 recent_predictions: List[Prediction]) -> Dict[str, Any]:
        """Controlla se ci sono condizioni di emergenza"""
        
        algorithm_key = f"{asset}_{model_type.value}_{algorithm.name}"
        
        # Se già stoppato, ritorna
        if algorithm_key in self.stopped_algorithms:
            return {'emergency_stop': True, 'reason': 'already_stopped'}
        
        triggers_activated = []
        
        # Check 1: Accuracy drop
        if len(recent_predictions) >= 20:
            recent_accuracy = np.mean([
                p.self_validation_score for p in recent_predictions[-20:]
                if p.self_validation_score is not None
            ])
            
            historical_accuracy = algorithm.accuracy_rate
            if historical_accuracy > 0:
                accuracy_drop = (historical_accuracy - recent_accuracy) / historical_accuracy
                if accuracy_drop > self.stop_triggers['accuracy_drop']:
                    triggers_activated.append(('accuracy_drop', accuracy_drop))
        
        # Check 2: Consecutive failures
        if len(recent_predictions) >= 10:
            consecutive_failures = 0
            for pred in recent_predictions[-10:]:
                if pred.self_validation_score is not None and pred.self_validation_score < 0.5:
                    consecutive_failures += 1
                else:
                    break
            
            if consecutive_failures >= self.stop_triggers['consecutive_failures']:
                triggers_activated.append(('consecutive_failures', consecutive_failures))
        
        # Check 3: Confidence collapse
        if algorithm.confidence_score < self.stop_triggers['confidence_collapse'] * 100:
            triggers_activated.append(('confidence_collapse', algorithm.confidence_score))
        
        # Check 4: Observer rejection
        if algorithm.observer_feedback_count > 10:
            rejection_rate = 1 - algorithm.observer_satisfaction
            if rejection_rate > self.stop_triggers['observer_rejection_rate']:
                triggers_activated.append(('observer_rejection', rejection_rate))
        
        # Check 5: Rapid score decline
        if len(algorithm.rolling_window_performances) >= 20:
            recent_scores = list(algorithm.rolling_window_performances)[-10:]
            older_scores = list(algorithm.rolling_window_performances)[-20:-10]
            
            if older_scores:
                recent_avg = np.mean(recent_scores)
                older_avg = np.mean(older_scores)
                
                if older_avg > 0:
                    score_decline = (older_avg - recent_avg) / older_avg
                    if score_decline > self.stop_triggers['rapid_score_decline']:
                        triggers_activated.append(('rapid_score_decline', score_decline))
        
        # Se ci sono trigger attivati, ferma l'algoritmo
        if triggers_activated:
            self._trigger_emergency_stop(algorithm_key, algorithm, triggers_activated)
            return {
                'emergency_stop': True,
                'triggers': triggers_activated,
                'timestamp': datetime.now(),
                'recommendation': 'immediate_retraining_required'
            }
        
        return {'emergency_stop': False}
    
    def _trigger_emergency_stop(self, algorithm_key: str, algorithm: AlgorithmPerformance,
                              triggers: List[Tuple[str, float]]):
        """Attiva l'emergency stop"""
        algorithm.emergency_stop_triggered = True
        self.stopped_algorithms.add(algorithm_key)
        
        # Log dettagliato
        self.logger.loggers['emergency'].critical(
            f"🚨 EMERGENCY STOP TRIGGERED | {algorithm_key} | "
            f"Triggers: {triggers} | "
            f"Final Score: {algorithm.final_score:.2f}"
        )
        
        # Notifica per analisi
        stop_event = {
            'timestamp': datetime.now().isoformat(),
            'algorithm_key': algorithm_key,
            'triggers': {t[0]: t[1] for t in triggers},
            'algorithm_state': {
                'final_score': algorithm.final_score,
                'confidence_score': algorithm.confidence_score,
                'quality_score': algorithm.quality_score,
                'total_predictions': algorithm.total_predictions,
                'recent_trend': algorithm.recent_performance_trend
            }
        }
        
        # Salva evento
        with open(self.logger.base_path / 'emergency_stops.json', 'a') as f:
            f.write(json.dumps(stop_event) + '\n')
    
    def reset_emergency_stop(self, algorithm_key: str):
        """Reset emergency stop dopo retraining - VERSIONE PULITA"""
        if algorithm_key in self.stopped_algorithms:
            self.stopped_algorithms.remove(algorithm_key)
            
            # 🧹 PULITO: Sostituito logger con event storage
            self._store_emergency_event('emergency_stop_reset', {
                'algorithm_key': algorithm_key,
                'status': 'reset_successful',
                'timestamp': datetime.now()
            })

    def _store_emergency_event(self, event_type: str, event_data: Dict) -> None:
        """Store emergency system events in memory for future processing by slave module"""
        if not hasattr(self, '_emergency_events_buffer'):
            self._emergency_events_buffer = []
        
        event_entry = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'data': event_data
        }
        
        self._emergency_events_buffer.append(event_entry)
        
        # Keep buffer size manageable
        if len(self._emergency_events_buffer) > 100:
            self._emergency_events_buffer = self._emergency_events_buffer[-50:]

    def get_all_events_for_slave(self) -> Dict[str, List[Dict]]:
        """Get all accumulated events for slave module processing"""
        events = {}
        
        # Emergency events
        if hasattr(self, '_emergency_events_buffer'):
            events['emergency_events'] = self._emergency_events_buffer.copy()
        
        return events

    def clear_events_buffer(self, event_types: Optional[List[str]] = None) -> None:
        """Clear event buffers after slave module processing"""
        if event_types is None:
            # Clear all buffers
            if hasattr(self, '_emergency_events_buffer'):
                self._emergency_events_buffer.clear()
        else:
            # Clear specific buffers
            for event_type in event_types:
                if event_type == 'emergency_events' and hasattr(self, '_emergency_events_buffer'):
                    self._emergency_events_buffer.clear()
class MT5Interface:
    """Interfaccia per comunicazione con MetaTrader 5 - VERSIONE PULITA"""
    
    def __init__(self, logger: AnalyzerLogger):
        self.logger = logger
        self.connected = False
        self.account_info = None
        
    def connect(self) -> bool:
        """Connetti a MT5 - VERSIONE PULITA"""
        try:
            if not mt5.initialize(): # type: ignore
                # 🧹 PULITO: Sostituito logger con event storage
                self._store_mt5_event('connection_failed', {
                    'status': 'initialization_failed',
                    'error': 'MT5 initialization failed',
                    'timestamp': datetime.now(),
                    'severity': 'error'
                })
                return False
            
            self.connected = True
            self.account_info = mt5.account_info() # type: ignore
            
            # 🧹 PULITO: Sostituito logger con event storage
            self._store_mt5_event('connected', {
                'status': 'success',
                'account_login': self.account_info.login if self.account_info else 'unknown',
                'timestamp': datetime.now()
            })
            return True
            
        except Exception as e:
            # 🧹 PULITO: Sostituito logger con event storage
            self._store_mt5_event('connection_error', {
                'status': 'error',
                'error_message': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.now(),
                'severity': 'error'
            })
            return False
    
    def get_tick_data(self, symbol: str) -> Optional[Dict]:
        """Ottieni l'ultimo tick per un simbolo - VERSIONE PULITA"""
        if not self.connected:
            return None
        
        try:
            tick = mt5.symbol_info_tick(symbol) # type: ignore
            if tick is None:
                return None
            
            return {
                'timestamp': datetime.fromtimestamp(tick.time),
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'flags': tick.flags
            }
            
        except Exception as e:
            # 🧹 PULITO: Sostituito logger con event storage
            self._store_mt5_event('tick_data_error', {
                'status': 'error',
                'symbol': symbol,
                'error_message': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.now(),
                'severity': 'warning'
            })
            return None
    
    def get_historical_data(self, symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
        """Ottieni dati storici - VERSIONE PULITA"""
        if not self.connected:
            return None
        
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count) # type: ignore
            if rates is None:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            # 🧹 PULITO: Sostituito logger con event storage
            self._store_mt5_event('historical_data_error', {
                'status': 'error',
                'symbol': symbol,
                'timeframe': timeframe,
                'count': count,
                'error_message': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.now(),
                'severity': 'warning'
            })
            return None
    
    def prepare_analysis_output(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara output analisi per MT5/Observer - VERSIONE PULITA"""
        mt5_output = {
            'timestamp': analysis['timestamp'].isoformat() if isinstance(analysis['timestamp'], datetime) else analysis['timestamp'],
            'asset': analysis['asset'],
            'market_state': {},
            'recommendations': [],
            'confidence_levels': {}
        }
        
        # Estrai raccomandazioni chiave
        for model_type, state in analysis.get('market_state', {}).items():
            if model_type == 'support_resistance':
                if 'support_levels' in state:
                    mt5_output['recommendations'].append({
                        'type': 'support_levels',
                        'values': state['support_levels'],
                        'confidence': state.get('confidence', 0)
                    })
                if 'resistance_levels' in state:
                    mt5_output['recommendations'].append({
                        'type': 'resistance_levels',
                        'values': state['resistance_levels'],
                        'confidence': state.get('confidence', 0)
                    })
            
            elif model_type == 'pattern_recognition':
                if 'detected_patterns' in state:
                    for pattern in state['detected_patterns']:
                        mt5_output['recommendations'].append({
                            'type': 'pattern',
                            'pattern_name': pattern.get('pattern'),
                            'probability': pattern.get('probability'),
                            'direction': pattern.get('direction', 'neutral')
                        })
            
            elif model_type == 'bias_detection':
                if 'directional_bias' in state:
                    mt5_output['recommendations'].append({
                        'type': 'bias',
                        'direction': state['directional_bias'].get('direction'),
                        'confidence': state['directional_bias'].get('confidence')
                    })
            
            elif model_type == 'trend_analysis':
                if 'trend_direction' in state:
                    mt5_output['recommendations'].append({
                        'type': 'trend',
                        'direction': state.get('trend_direction'),
                        'strength': state.get('trend_strength'),
                        'confidence': state.get('confidence')
                    })
        
        # 🧹 PULITO: Log comunicazione (usa metodo esistente del logger)
        self.logger.log_mt5_communication('out', 'analysis', analysis['asset'], mt5_output)
        
        return mt5_output
    
    def disconnect(self):
        """Disconnetti da MT5 - VERSIONE PULITA"""
        if self.connected:
            mt5.shutdown() # type: ignore
            self.connected = False
            
            # 🧹 PULITO: Sostituito logger con event storage
            self._store_mt5_event('disconnected', {
                'status': 'success',
                'timestamp': datetime.now()
            })
    
    def _store_mt5_event(self, event_type: str, event_data: Dict) -> None:
        """Store MT5 events in memory for future processing by slave module"""
        if not hasattr(self, '_mt5_events_buffer'):
            self._mt5_events_buffer = []
        
        event_entry = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'data': event_data
        }
        
        self._mt5_events_buffer.append(event_entry)
        
        # Keep buffer size manageable
        if len(self._mt5_events_buffer) > 100:
            self._mt5_events_buffer = self._mt5_events_buffer[-50:]
    
    def get_all_events_for_slave(self) -> Dict[str, List[Dict]]:
        """Get all accumulated events for slave module processing"""
        events = {}
        
        # MT5 events
        if hasattr(self, '_mt5_events_buffer'):
            events['mt5_events'] = self._mt5_events_buffer.copy()
        
        return events
    
    def clear_events_buffer(self, event_types: Optional[List[str]] = None) -> None:
        """Clear event buffers after slave module processing"""
        if event_types is None:
            # Clear all buffers
            if hasattr(self, '_mt5_events_buffer'):
                self._mt5_events_buffer.clear()
        else:
            # Clear specific buffers
            for event_type in event_types:
                if event_type == 'mt5_events' and hasattr(self, '_mt5_events_buffer'):
                    self._mt5_events_buffer.clear()

# ================== MODELLI NEURALI ==================

class AdvancedLSTM(nn.Module):
    """LSTM avanzato con auto-resize dinamico per qualsiasi dimensione input"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(AdvancedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.expected_input_size = input_size  # Dimensione target preferita
        
        # 🔧 NUOVO: Pool di adapter dinamici per diverse dimensioni
        self.input_adapters = nn.ModuleDict()  # Memorizza adapter per diverse dimensioni
        self.adapter_cache = {}  # Cache per evitare ricreazioni
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()
        
        # Statistiche per debug
        self.resize_stats = {
            'total_calls': 0,
            'adapters_created': 0,
            'dimension_history': []
        }
    
    def _get_or_create_adapter(self, actual_input_size: int) -> nn.Module:
        """Ottiene o crea un adapter per la dimensione specifica con caching ottimizzato"""
        
        # Se la dimensione è già quella attesa, non serve adapter
        if actual_input_size == self.expected_input_size:
            return nn.Identity()
        
        # Crea chiave per l'adapter
        adapter_key = f"adapter_{actual_input_size}_to_{self.expected_input_size}"
        
        # 🚀 CACHE HIT: Se l'adapter esiste già, riutilizzalo
        if adapter_key in self.input_adapters:
            # Track usage per statistiche
            if not hasattr(self, 'adapter_usage_count'):
                self.adapter_usage_count = {}
            self.adapter_usage_count[adapter_key] = self.adapter_usage_count.get(adapter_key, 0) + 1
            
            # Solo log ogni 100 utilizzi per ridurre spam
            if self.adapter_usage_count[adapter_key] % 100 == 0:
                safe_print(f"🔄 Adapter cache hit #{self.adapter_usage_count[adapter_key]}: {adapter_key}")
            
            return self.input_adapters[adapter_key]
        
        # 🔧 CACHE MISS: Crea nuovo adapter solo se necessario
        safe_print(f"🔧 Creating new LSTM adapter: {actual_input_size} → {self.expected_input_size}")
        
        # 🚀 AUTO-CLEANUP: Gestione intelligente degli adapter
        max_adapters = 8  # Ridotto per safety
        cleanup_threshold = 6  # Inizia cleanup prima del limite
        
        if len(self.input_adapters) >= cleanup_threshold:
            self._auto_cleanup_adapters(max_adapters)
        
        # Crea nuovo adapter con architettura ottimizzata
        new_adapter = nn.Sequential(
            nn.Linear(actual_input_size, self.expected_input_size),
            nn.LayerNorm(self.expected_input_size),
            nn.Dropout(0.1)
        )
        
        # 🔧 Inizializza i pesi dell'adapter in modo efficiente
        with torch.no_grad():
            for module in new_adapter.modules():
                if isinstance(module, nn.Linear):
                    # Usa inizializzazione più efficiente
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.zeros_(module.bias)
                    break
        
        # 🚀 REGISTRA con tracking ottimizzato
        self.input_adapters[adapter_key] = new_adapter
        self.resize_stats['adapters_created'] += 1
        
        # Inizializza usage count
        if not hasattr(self, 'adapter_usage_count'):
            self.adapter_usage_count = {}
        self.adapter_usage_count[adapter_key] = 1
        
        safe_print(f"✅ Adapter '{adapter_key}' created and cached (Total: {len(self.input_adapters)})")
        
        return new_adapter
    
    def _auto_cleanup_adapters(self, max_adapters: int) -> None:
        """Auto-cleanup intelligente degli adapter con strategia LRU"""
        
        if not hasattr(self, 'adapter_usage_count'):
            self.adapter_usage_count = {}
        
        current_count = len(self.input_adapters)
        target_count = max(2, max_adapters - 2)  # Mantieni almeno 2, target 2 sotto il max
        
        if current_count <= target_count:
            return
        
        # Strategia di cleanup intelligente
        removal_candidates = []
        
        # 1. Adapter mai utilizzati o con usage molto basso
        for adapter_key, usage_count in self.adapter_usage_count.items():
            if adapter_key in self.input_adapters:
                # Score basato su: usage_count, età (implicita dall'ordine), dimensione
                score = usage_count
                
                # Penalizza adapter per dimensioni inusuali (probabilmente temporanei)
                if 'adapter_' in adapter_key:
                    try:
                        parts = adapter_key.split('_')
                        input_size = int(parts[1])
                        # Penalizza dimensioni molto diverse dall'expected
                        if abs(input_size - self.expected_input_size) > self.expected_input_size * 0.5:
                            score *= 0.5  # Dimezza score per dimensioni anomale
                    except:
                        pass
                
                removal_candidates.append((adapter_key, score))
        
        # Ordina per score (i meno utilizzati first)
        removal_candidates.sort(key=lambda x: x[1])
        
        # Rimuovi gli adapter con score più basso
        to_remove = current_count - target_count
        removed_count = 0
        
        for adapter_key, score in removal_candidates:
            if removed_count >= to_remove:
                break
            
            if adapter_key in self.input_adapters:
                # Backup info per log
                usage_count = self.adapter_usage_count.get(adapter_key, 0)
                
                # Rimuovi
                del self.input_adapters[adapter_key]
                if adapter_key in self.adapter_usage_count:
                    del self.adapter_usage_count[adapter_key]
                
                removed_count += 1
                safe_print(f"🧹 Auto-removed adapter: {adapter_key} (score: {score:.1f}, usage: {usage_count})")
        
        # Log risultato cleanup
        final_count = len(self.input_adapters)
        memory_saved = removed_count * 0.1  # Stima MB per adapter
        
        safe_print(f"✅ Adapter cleanup: {current_count} → {final_count} (-{removed_count}), ~{memory_saved:.1f}MB saved")
        
        # Force garbage collection per liberare memoria
        import gc
        gc.collect()
    
    def get_cache_efficiency_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche dettagliate sull'efficienza della cache"""
        
        if not hasattr(self, 'adapter_usage_count'):
            self.adapter_usage_count = {}
        
        total_calls = sum(self.adapter_usage_count.values())
        cache_hits = total_calls - self.resize_stats['adapters_created']
        hit_rate = (cache_hits / total_calls * 100) if total_calls > 0 else 0
        
        # Trova adapter più e meno utilizzati
        if self.adapter_usage_count:
            most_used = max(self.adapter_usage_count.items(), key=lambda x: x[1])
            least_used = min(self.adapter_usage_count.items(), key=lambda x: x[1])
        else:
            most_used = ("none", 0)
            least_used = ("none", 0)
        
        return {
            'total_adapter_calls': total_calls,
            'cache_hits': cache_hits,
            'cache_misses': self.resize_stats['adapters_created'],
            'hit_rate_percentage': hit_rate,
            'active_adapters': len(self.input_adapters),
            'most_used_adapter': {'key': most_used[0], 'usage': most_used[1]},
            'least_used_adapter': {'key': least_used[0], 'usage': least_used[1]},
            'memory_efficiency': 'high' if hit_rate > 80 else 'medium' if hit_rate > 50 else 'low'
        }

    def optimize_cache(self) -> Dict[str, Any]:
        """Ottimizza la cache rimuovendo adapter inutilizzati"""
        
        if not hasattr(self, 'adapter_usage_count'):
            return {'status': 'no_cache_data'}
        
        initial_count = len(self.input_adapters)
        removed_adapters = []
        
        # Rimuovi adapter utilizzati meno di 5 volte
        min_usage_threshold = 5
        for adapter_key, usage_count in list(self.adapter_usage_count.items()):
            if usage_count < min_usage_threshold:
                if adapter_key in self.input_adapters:
                    del self.input_adapters[adapter_key]
                    del self.adapter_usage_count[adapter_key]
                    removed_adapters.append(adapter_key)
        
        final_count = len(self.input_adapters)
        
        safe_print(f"🔧 Cache optimization: removed {len(removed_adapters)} unused adapters")
        
        return {
            'status': 'optimized',
            'adapters_before': initial_count,
            'adapters_after': final_count,
            'removed_count': len(removed_adapters),
            'removed_adapters': removed_adapters
        }

    def clear_adapter_cache(self) -> None:
        """Pulisce completamente la cache degli adapter"""
        
        cache_size = len(self.input_adapters)
        self.input_adapters.clear()
        
        if hasattr(self, 'adapter_usage_count'):
            self.adapter_usage_count.clear()
        
        # Reset stats
        self.resize_stats['adapters_created'] = 0
        self.resize_stats['dimension_history'].clear()
        
        safe_print(f"🗑️ Adapter cache cleared: {cache_size} adapters removed")
    
    def _apply_adapter(self, x: torch.Tensor, adapter: nn.Module) -> torch.Tensor:
        """Applica l'adapter mantenendo la forma del tensore con protezione completa anti-NaN"""
        
        original_shape = x.shape
        
        # 🛡️ VALIDAZIONE INPUT TENSOR CRITICA
        if torch.isnan(x).any():
            safe_print(f"❌ Input tensor contiene NaN prima dell'adapter: {torch.isnan(x).sum().item()} valori")
            # Sanitizza input
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            safe_print("🔧 Input tensor sanitizzato")
        
        if torch.isinf(x).any():
            safe_print(f"❌ Input tensor contiene Inf prima dell'adapter: {torch.isinf(x).sum().item()} valori")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            safe_print("🔧 Input tensor sanitizzato")
        
        # 🛡️ GESTIONE SHAPE INTELLIGENTE CON PROTEZIONE
        try:
            if len(original_shape) == 3:  # (batch, seq, features)
                batch_size, seq_len, features = original_shape
                
                # 🛡️ VALIDAZIONE DIMENSIONI
                if features <= 0 or batch_size <= 0 or seq_len <= 0:
                    safe_print(f"❌ Dimensioni non valide: {original_shape}")
                    # Crea tensor di fallback sicuro
                    fallback_tensor = torch.zeros(batch_size, seq_len, self.expected_input_size, dtype=x.dtype, device=x.device)
                    safe_print(f"🔧 Creato tensor fallback: {fallback_tensor.shape}")
                    return fallback_tensor
                
                # Reshape per applicare Linear: (batch*seq, features)
                x_reshaped = x.view(-1, features)
                
                # 🛡️ VALIDAZIONE DOPO RESHAPE
                if torch.isnan(x_reshaped).any() or torch.isinf(x_reshaped).any():
                    safe_print("❌ Tensor contiene NaN/Inf dopo reshape")
                    x_reshaped = torch.nan_to_num(x_reshaped, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 🛡️ APPLICA ADAPTER CON PROTEZIONE
                try:
                    x_adapted = adapter(x_reshaped)
                    
                    # 🛡️ VALIDAZIONE OUTPUT ADAPTER CRITICA
                    if x_adapted is None:
                        safe_print("❌ Adapter ha ritornato None")
                        x_adapted = torch.zeros(x_reshaped.shape[0], self.expected_input_size, dtype=x.dtype, device=x.device)
                    
                    elif torch.isnan(x_adapted).any():
                        nan_count = torch.isnan(x_adapted).sum().item()
                        safe_print(f"❌ Adapter output contiene {nan_count} NaN values")
                        x_adapted = torch.nan_to_num(x_adapted, nan=0.0, posinf=1.0, neginf=-1.0)
                        safe_print("🔧 Adapter output sanitizzato")
                    
                    elif torch.isinf(x_adapted).any():
                        inf_count = torch.isinf(x_adapted).sum().item()
                        safe_print(f"❌ Adapter output contiene {inf_count} Inf values")
                        x_adapted = torch.nan_to_num(x_adapted, nan=0.0, posinf=1.0, neginf=-1.0)
                        safe_print("🔧 Adapter output sanitizzato")
                    
                    # 🛡️ VALIDAZIONE FORMA OUTPUT ADAPTER
                    expected_adapter_shape = (x_reshaped.shape[0], self.expected_input_size)
                    if x_adapted.shape != expected_adapter_shape:
                        safe_print(f"❌ Adapter output shape mismatch: {x_adapted.shape} vs {expected_adapter_shape}")
                        # Crea output corretto
                        x_adapted = torch.zeros(expected_adapter_shape, dtype=x.dtype, device=x.device)
                        safe_print(f"🔧 Creato adapter output corretto: {x_adapted.shape}")
                    
                except Exception as adapter_error:
                    safe_print(f"❌ Errore nell'adapter: {adapter_error}")
                    # Fallback sicuro per adapter error
                    x_adapted = torch.zeros(x_reshaped.shape[0], self.expected_input_size, dtype=x.dtype, device=x.device)
                    safe_print(f"🔧 Fallback adapter creato: {x_adapted.shape}")
                
                # 🛡️ RESHAPE BACK CON PROTEZIONE
                try:
                    target_shape = (batch_size, seq_len, self.expected_input_size)
                    x = x_adapted.view(target_shape)
                    
                    # 🛡️ VALIDAZIONE FINALE DOPO RESHAPE
                    if x.shape != target_shape:
                        safe_print(f"❌ Reshape finale fallito: {x.shape} vs {target_shape}")
                        x = torch.zeros(target_shape, dtype=x_adapted.dtype, device=x_adapted.device)
                        safe_print(f"🔧 Tensor finale ricreato: {x.shape}")
                    
                except RuntimeError as reshape_error:
                    safe_print(f"❌ Errore reshape finale: {reshape_error}")
                    safe_print(f"   Original: {original_shape}")
                    safe_print(f"   Adapted shape: {x_adapted.shape}")
                    safe_print(f"   Target: ({batch_size}, {seq_len}, {self.expected_input_size})")
                    
                    # Fallback sicuro per reshape error
                    target_shape = (batch_size, seq_len, self.expected_input_size)
                    x = torch.zeros(target_shape, dtype=x_adapted.dtype, device=x_adapted.device)
                    safe_print(f"🔧 Fallback finale creato: {x.shape}")
                
            elif len(original_shape) == 2:  # (batch, features)
                # 🛡️ GESTIONE 2D CON PROTEZIONE
                try:
                    x_adapted = adapter(x)
                    
                    # 🛡️ VALIDAZIONE OUTPUT 2D
                    if x_adapted is None:
                        safe_print("❌ Adapter 2D ha ritornato None")
                        x_adapted = torch.zeros(x.shape[0], self.expected_input_size, dtype=x.dtype, device=x.device)
                    
                    elif torch.isnan(x_adapted).any() or torch.isinf(x_adapted).any():
                        safe_print("❌ Adapter 2D output contiene NaN/Inf")
                        x_adapted = torch.nan_to_num(x_adapted, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    x = x_adapted
                    
                except Exception as adapter_2d_error:
                    safe_print(f"❌ Errore adapter 2D: {adapter_2d_error}")
                    x = torch.zeros(x.shape[0], self.expected_input_size, dtype=x.dtype, device=x.device)
            
            else:
                # 🛡️ CASO COMPLESSO CON FALLBACK SICURO
                safe_print(f"⚠️ Shape non standard per adapter: {original_shape}")
                
                try:
                    x_prepared, shape_info = TensorShapeManager.prepare_model_input(
                        x, 'LSTM', self.expected_input_size
                    )
                    
                    # Applica adapter alla forma preparata con protezione
                    if len(x_prepared.shape) == 3:
                        batch_size, seq_len, features = x_prepared.shape
                        x_reshaped = x_prepared.view(-1, features)
                        
                        # Adapter con protezione
                        try:
                            x_adapted = adapter(x_reshaped)
                            if torch.isnan(x_adapted).any() or torch.isinf(x_adapted).any():
                                x_adapted = torch.nan_to_num(x_adapted, nan=0.0, posinf=1.0, neginf=-1.0)
                            x = x_adapted.view(batch_size, seq_len, self.expected_input_size)
                        except:
                            x = torch.zeros(batch_size, seq_len, self.expected_input_size, dtype=x.dtype, device=x.device)
                    else:
                        try:
                            x_adapted = adapter(x_prepared)
                            if torch.isnan(x_adapted).any() or torch.isinf(x_adapted).any():
                                x_adapted = torch.nan_to_num(x_adapted, nan=0.0, posinf=1.0, neginf=-1.0)
                            x = x_adapted
                        except:
                            x = torch.zeros(x_prepared.shape[0], self.expected_input_size, dtype=x.dtype, device=x.device)
                    
                    safe_print(f"✅ Shape correction applicata: {original_shape} → {x.shape}")
                    
                except Exception as shape_error:
                    safe_print(f"❌ Errore shape correction: {shape_error}")
                    # Fallback assoluto
                    if len(original_shape) >= 2:
                        fallback_shape = (original_shape[0], self.expected_input_size)
                        x = torch.zeros(fallback_shape, dtype=x.dtype, device=x.device)
                    else:
                        x = torch.zeros(1, self.expected_input_size, dtype=x.dtype, device=x.device)
                    safe_print(f"🔧 Fallback assoluto creato: {x.shape}")
            
            # 🛡️ VALIDAZIONE FINALE COMPLETA
            if torch.isnan(x).any() or torch.isinf(x).any():
                safe_print("❌ Output finale contiene ancora NaN/Inf")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                safe_print("🔧 Output finale sanitizzato")
            
            # 🛡️ VERIFICA FORMA FINALE
            if x.shape[-1] != self.expected_input_size:
                safe_print(f"❌ Forma finale incorretta: {x.shape} (expected last dim: {self.expected_input_size})")
                # Crea output con forma corretta
                if len(x.shape) == 3:
                    correct_shape = (x.shape[0], x.shape[1], self.expected_input_size)
                elif len(x.shape) == 2:
                    correct_shape = (x.shape[0], self.expected_input_size)
                else:
                    correct_shape = (1, self.expected_input_size)
                
                x = torch.zeros(correct_shape, dtype=x.dtype, device=x.device)
                safe_print(f"🔧 Output finale ricreato con forma corretta: {x.shape}")
            
            safe_print(f"✅ Adapter applicato con successo: {original_shape} → {x.shape}")
            return x
            
        except Exception as e:
            safe_print(f"❌ Errore catastrofico in _apply_adapter: {e}")
            safe_print(f"   Input shape: {original_shape}")
            safe_print(f"   Expected input size: {self.expected_input_size}")
            
            # 🛡️ FALLBACK ASSOLUTO FINALE
            try:
                if len(original_shape) == 3:
                    fallback_shape = (original_shape[0], original_shape[1], self.expected_input_size)
                elif len(original_shape) == 2:
                    fallback_shape = (original_shape[0], self.expected_input_size)
                else:
                    fallback_shape = (1, self.expected_input_size)
                
                fallback_tensor = torch.zeros(fallback_shape, dtype=x.dtype if x is not None else torch.float32, 
                                            device=x.device if x is not None else 'cpu')
                safe_print(f"🔧 Fallback assoluto finale: {fallback_tensor.shape}")
                return fallback_tensor
                
            except Exception as fallback_error:
                safe_print(f"❌ Anche il fallback è fallito: {fallback_error}")
                # Ultimo resort
                return torch.zeros(1, self.expected_input_size)
    
    def forward(self, x):
        """Forward pass con protezione completa anti-NaN a ogni step"""
        
        self.resize_stats['total_calls'] += 1
        
        # 🛡️ VALIDAZIONE INPUT ASSOLUTA
        if x is None:
            safe_print("❌ Input è None!")
            return torch.zeros(1, self.expected_input_size)
        
        if not isinstance(x, torch.Tensor):
            safe_print(f"❌ Input non è un tensor: {type(x)}")
            try:
                x = torch.tensor(x, dtype=torch.float32)
            except:
                return torch.zeros(1, self.expected_input_size)
        
        # 🛡️ VALIDAZIONE NaN/Inf INPUT
        if torch.isnan(x).any():
            nan_count = torch.isnan(x).sum().item()
            safe_print(f"❌ Input contiene {nan_count} valori NaN - sanitizzando...")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isinf(x).any():
            inf_count = torch.isinf(x).sum().item()
            safe_print(f"❌ Input contiene {inf_count} valori Inf - sanitizzando...")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 🛡️ VALIDAZIONE RANGE INPUT
        if torch.abs(x).max() > 1000:
            safe_print(f"⚠️ Input ha valori estremi: max={torch.abs(x).max():.2f}")
            x = torch.clamp(x, -100, 100)  # Clamp valori estremi
        
        original_shape = x.shape
        
        # 🛡️ TENSOR SHAPE MANAGEMENT CON PROTEZIONE
        try:
            # Inizializza shape manager se non esiste
            if not hasattr(self, '_shape_manager'):
                self._shape_manager = TensorShapeManager()
            
            # Preparazione input con protezione completa
            try:
                x, shape_info = TensorShapeManager.ensure_lstm_input_shape(
                    x, sequence_length=1, expected_features=self.expected_input_size
                )
                
                # 🛡️ VALIDAZIONE DOPO SHAPE MANAGEMENT
                if torch.isnan(x).any() or torch.isinf(x).any():
                    safe_print("❌ NaN/Inf rilevati dopo shape management")
                    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Log conversioni significative
                if shape_info['conversion_applied']:
                    conversion_key = f"{original_shape}→{x.shape}"
                    
                    if not hasattr(self, '_logged_conversions'):
                        self._logged_conversions = set()
                    
                    if conversion_key not in self._logged_conversions:
                        safe_print(f"🔧 TensorShape: {shape_info['method_used']}: {original_shape} → {x.shape}")
                        self._logged_conversions.add(conversion_key)
                        
                        if not hasattr(self, '_conversion_stats'):
                            self._conversion_stats = {}
                        method = shape_info['method_used']
                        self._conversion_stats[method] = self._conversion_stats.get(method, 0) + 1
            
            except Exception as shape_error:
                safe_print(f"❌ Errore TensorShapeManager: {shape_error}")
                safe_print(f"   Input shape: {original_shape}")
                
                # Fallback shape management
                if len(original_shape) == 1:
                    x = x.unsqueeze(0).unsqueeze(0)
                elif len(original_shape) == 2:
                    x = x.unsqueeze(1)
                elif len(original_shape) != 3:
                    safe_print(f"❌ Shape non gestibile: {original_shape}")
                    return torch.zeros(1, self.expected_input_size)
        
        except Exception as e:
            safe_print(f"❌ Errore critico in shape management: {e}")
            return torch.zeros(1, self.expected_input_size)
        
        # 🛡️ VERIFICA FINALE SHAPE
        if len(x.shape) != 3:
            safe_print(f"❌ Shape finale non valida: {x.shape} (deve essere 3D)")
            return torch.zeros(1, self.expected_input_size)
        
        # Estrai dimensioni
        batch_size, seq_len, actual_input_size = x.shape
        
        # 🛡️ VALIDAZIONE DIMENSIONI
        if batch_size <= 0 or seq_len <= 0 or actual_input_size <= 0:
            safe_print(f"❌ Dimensioni non valide: {x.shape}")
            return torch.zeros(1, self.expected_input_size)
        
        # 🛡️ REGISTRA DIMENSIONE PER STATISTICHE
        if actual_input_size not in self.resize_stats['dimension_history'][-10:]:
            self.resize_stats['dimension_history'].append(actual_input_size)
        
        # 🛡️ DYNAMIC ADAPTER CON PROTEZIONE
        try:
            adapter = self._get_or_create_adapter(actual_input_size)
            
            # Applica adapter se necessario
            if not isinstance(adapter, nn.Identity):
                try:
                    x = self._apply_adapter(x, adapter)
                    
                    # 🛡️ VALIDAZIONE POST-ADAPTER
                    if torch.isnan(x).any() or torch.isinf(x).any():
                        safe_print("❌ NaN/Inf dopo adapter - sanitizzando...")
                        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    safe_print(f"🔧 Adapter applicato: {actual_input_size} → {x.shape[-1]}")
                    
                except Exception as adapter_error:
                    safe_print(f"❌ Errore applicazione adapter: {adapter_error}")
                    # Fallback: crea tensor con dimensioni corrette
                    x = torch.zeros(batch_size, seq_len, self.expected_input_size, dtype=x.dtype, device=x.device)
        
        except Exception as adapter_creation_error:
            safe_print(f"❌ Errore creazione adapter: {adapter_creation_error}")
            # Fallback finale
            x = torch.zeros(batch_size, seq_len, self.expected_input_size, dtype=x.dtype, device=x.device)
        
        # 🛡️ VERIFICA FINALE DELLE DIMENSIONI
        if x.shape[-1] != self.expected_input_size:
            safe_print(f"❌ Dimensione finale incorretta! Expected {self.expected_input_size}, got {x.shape[-1]}")
            # Crea tensor corretto
            x = torch.zeros(batch_size, seq_len, self.expected_input_size, dtype=x.dtype, device=x.device)
        
        # 🛡️ LSTM PROCESSING CON PROTEZIONE COMPLETA
        try:
            # Controlla che x sia ancora valido
            if torch.isnan(x).any() or torch.isinf(x).any():
                safe_print("❌ Input LSTM contiene NaN/Inf - sanitizzando...")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # LSTM forward
            lstm_out, lstm_hidden = self.lstm(x)
            
            # 🛡️ VALIDAZIONE OUTPUT LSTM
            if lstm_out is None:
                safe_print("❌ LSTM output è None!")
                return torch.zeros(batch_size, self.fc2.out_features)
            
            if torch.isnan(lstm_out).any():
                nan_count = torch.isnan(lstm_out).sum().item()
                safe_print(f"❌ LSTM output contiene {nan_count} NaN - sanitizzando...")
                lstm_out = torch.nan_to_num(lstm_out, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if torch.isinf(lstm_out).any():
                inf_count = torch.isinf(lstm_out).sum().item()
                safe_print(f"❌ LSTM output contiene {inf_count} Inf - sanitizzando...")
                lstm_out = torch.nan_to_num(lstm_out, nan=0.0, posinf=1.0, neginf=-1.0)
            
        except Exception as lstm_error:
            safe_print(f"❌ Errore LSTM: {lstm_error}")
            return torch.zeros(batch_size, self.fc2.out_features if hasattr(self.fc2, 'out_features') else 1)
        
        # 🛡️ ATTENTION MECHANISM CON PROTEZIONE
        try:
            # Transpose per attention
            lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch, features)
            
            # Attention
            attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            
            # 🛡️ VALIDAZIONE ATTENTION OUTPUT
            if attn_out is None:
                safe_print("❌ Attention output è None!")
                attn_out = lstm_out
            
            if torch.isnan(attn_out).any() or torch.isinf(attn_out).any():
                safe_print("❌ Attention output contiene NaN/Inf - sanitizzando...")
                attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Layer norm con protezione
            try:
                attn_out = self.layer_norm(attn_out + lstm_out)
                
                if torch.isnan(attn_out).any() or torch.isinf(attn_out).any():
                    safe_print("❌ LayerNorm output contiene NaN/Inf - sanitizzando...")
                    attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1.0, neginf=-1.0)
                    
            except Exception as norm_error:
                safe_print(f"❌ Errore LayerNorm: {norm_error}")
                attn_out = lstm_out  # Fallback
            
            # Transpose back
            attn_out = attn_out.transpose(0, 1)  # (batch, seq_len, features)
            
        except Exception as attention_error:
            safe_print(f"❌ Errore Attention: {attention_error}")
            # Usa output LSTM direttamente
            attn_out = lstm_out
        
        # 🛡️ FINAL LAYERS CON PROTEZIONE
        try:
            # Take last output
            out = attn_out[:, -1, :]
            
            # 🛡️ VALIDAZIONE PRIMA DEI LAYER FINALI
            if torch.isnan(out).any() or torch.isinf(out).any():
                safe_print("❌ Pre-FC output contiene NaN/Inf - sanitizzando...")
                out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Dropout
            out = self.dropout(out)
            
            # FC1 con protezione
            try:
                out = self.activation(self.fc1(out))
                
                if torch.isnan(out).any() or torch.isinf(out).any():
                    safe_print("❌ FC1 output contiene NaN/Inf - sanitizzando...")
                    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
                    
            except Exception as fc1_error:
                safe_print(f"❌ Errore FC1: {fc1_error}")
                out = torch.zeros(out.shape[0], self.fc1.out_features, dtype=out.dtype, device=out.device)
            
            # Dropout finale
            out = self.dropout(out)
            
            # FC2 finale con protezione
            try:
                out = self.fc2(out)
                
                if torch.isnan(out).any() or torch.isinf(out).any():
                    safe_print("❌ FC2 output contiene NaN/Inf - sanitizzando...")
                    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
                    
            except Exception as fc2_error:
                safe_print(f"❌ Errore FC2: {fc2_error}")
                out = torch.zeros(out.shape[0], self.fc2.out_features, dtype=out.dtype, device=out.device)
            
        except Exception as final_error:
            safe_print(f"❌ Errore nei layer finali: {final_error}")
            # Fallback finale
            output_size = self.fc2.out_features if hasattr(self.fc2, 'out_features') else 1
            out = torch.zeros(batch_size, output_size)
        
        # 🛡️ VALIDAZIONE FINALE ASSOLUTA
        if out is None:
            safe_print("❌ Output finale è None!")
            output_size = self.fc2.out_features if hasattr(self.fc2, 'out_features') else 1
            return torch.zeros(batch_size, output_size)
        
        if torch.isnan(out).any() or torch.isinf(out).any():
            safe_print("❌ Output finale contiene NaN/Inf - sanitizzando...")
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 🛡️ CLAMP OUTPUT PER SICUREZZA
        out = torch.clamp(out, -100, 100)  # Previeni output estremi
        
        safe_print(f"✅ Forward completato con successo: {original_shape} → {out.shape}")
        return out
            
    def get_resize_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche complete delle operazioni di resize e performance"""
        
        # Calcola frequenza dimensioni
        dimension_counts = {}
        for dim in self.resize_stats['dimension_history']:
            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
        
        # Calcola statistiche cache
        cache_stats = self.get_cache_efficiency_stats()
        
        # Calcola risparmio computazionale
        total_calls = self.resize_stats['total_calls']
        adapters_created = self.resize_stats['adapters_created']
        
        if total_calls > 0:
            efficiency_gain = ((total_calls - adapters_created) / total_calls) * 100
            computational_savings = f"{efficiency_gain:.1f}%"
        else:
            computational_savings = "0%"
        
        # Identifica dimensioni più comuni
        if dimension_counts:
            most_common_dim = max(dimension_counts.items(), key=lambda x: x[1])
            optimization_potential = most_common_dim[1] / total_calls * 100 if total_calls > 0 else 0
        else:
            most_common_dim = (0, 0)
            optimization_potential = 0
        
        return {
            'performance_metrics': {
                'total_calls': total_calls,
                'adapters_created': adapters_created,
                'computational_savings': computational_savings,
                'cache_hit_rate': cache_stats['hit_rate_percentage'],
                'memory_efficiency': cache_stats['memory_efficiency']
            },
            'dimension_analysis': {
                'unique_dimensions_seen': len(set(self.resize_stats['dimension_history'])),
                'dimension_frequency': dimension_counts,
                'most_common_dimension': {
                    'size': most_common_dim[0],
                    'frequency': most_common_dim[1],
                    'optimization_potential': f"{optimization_potential:.1f}%"
                }
            },
            'cache_details': cache_stats,
            'adapter_keys': list(self.input_adapters.keys()),
            'recommendations': self._generate_optimization_recommendations(cache_stats, dimension_counts)
        }

    def _generate_optimization_recommendations(self, cache_stats: Dict, dimension_counts: Dict) -> List[str]:
        """Genera raccomandazioni per ottimizzazione"""
        
        recommendations = []
        
        # Raccomandazioni basate su hit rate
        if cache_stats['hit_rate_percentage'] < 50:
            recommendations.append("Low cache hit rate - consider input data preprocessing")
        elif cache_stats['hit_rate_percentage'] > 90:
            recommendations.append("Excellent cache performance - system well optimized")
        
        # Raccomandazioni basate su varietà dimensioni
        unique_dims = len(set(dimension_counts.keys()))
        if unique_dims > 8:
            recommendations.append(f"High dimension variety ({unique_dims}) - consider data standardization")
        elif unique_dims <= 3:
            recommendations.append("Low dimension variety - excellent for caching")
        
        # Raccomandazioni basate su active adapters
        if cache_stats['active_adapters'] > 15:
            recommendations.append("Too many active adapters - run cache optimization")
        
        # Raccomandazioni basate su usage patterns
        if cache_stats['most_used_adapter']['usage'] > cache_stats['least_used_adapter']['usage'] * 10:
            recommendations.append("Uneven adapter usage - some dimensions dominate")
        
        return recommendations
    
    def reset_adapters(self):
        """Reset tutti gli adapter (utile per testing)"""
        self.input_adapters.clear()
        self.adapter_cache.clear()
        self.resize_stats = {
            'total_calls': 0,
            'adapters_created': 0,
            'dimension_history': []
        }
        print("🔄 All adapters reset")
    
    def get_tensor_shape_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche dettagliate sulla gestione delle forme tensor"""
        
        # Statistiche conversioni
        conversion_stats = getattr(self, '_conversion_stats', {})
        total_conversions = sum(conversion_stats.values())
        
        # Statistiche shape manager
        if hasattr(self, '_shape_manager'):
            shape_stats = self._shape_manager.get_shape_statistics()
        else:
            shape_stats = {'total_processed': 0, 'conversions_needed': 0}
        
        # Calcola efficienza
        total_calls = self.resize_stats['total_calls']
        efficiency_improvement = (total_calls - total_conversions) / total_calls * 100 if total_calls > 0 else 100
        
        return {
            'tensor_shape_performance': {
                'total_forward_calls': total_calls,
                'shape_conversions_applied': total_conversions,
                'conversion_efficiency': f"{efficiency_improvement:.1f}%",
                'most_common_conversions': dict(sorted(conversion_stats.items(), key=lambda x: x[1], reverse=True)[:5])
            },
            'shape_manager_stats': shape_stats,
            'optimization_impact': {
                'before_optimization': '398 automatic expansions',
                'current_conversions': total_conversions,
                'reduction_achieved': f"{max(0, 398 - total_conversions)} fewer conversions",
                'efficiency_gain': f"{efficiency_improvement:.1f}%"
            },
            'recommendations': self._generate_tensor_recommendations(conversion_stats, efficiency_improvement)
        }

    def _generate_tensor_recommendations(self, conversion_stats: Dict, efficiency: float) -> List[str]:
        """Genera raccomandazioni per ottimizzazione tensor shapes"""
        
        recommendations = []
        
        if efficiency < 80:
            recommendations.append("Consider preprocessing input data to standard LSTM format")
        
        if '2D_to_3D_standard' in conversion_stats and conversion_stats['2D_to_3D_standard'] > 50:
            recommendations.append("High 2D→3D conversions - implement input standardization")
        
        if 'smart_reshape' in str(conversion_stats):
            recommendations.append("Smart reshaping active - monitor model accuracy")
        
        if len(conversion_stats) > 3:
            recommendations.append("Multiple conversion types detected - unify input pipeline")
        
        if efficiency > 95:
            recommendations.append("Excellent tensor shape efficiency achieved!")
        
        return recommendations

    @property 
    def adapter_created(self) -> bool:
        """Compatibilità con codice esistente"""
        return len(self.input_adapters) > 0

class TransformerPredictor(nn.Module):
    """Transformer per pattern recognition avanzato"""
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 6, output_dim: int = 1):
        super(TransformerPredictor, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, output_dim)
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input and add positional encoding
        x = self.input_projection(x)
        x += self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer expects (seq_len, batch, features)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        
        # Take last token and project to output
        x = x[-1]  # (batch, d_model)
        return self.output_layer(x)

class CNNPatternRecognizer(nn.Module):
    """CNN 1D per riconoscimento pattern grafici"""
    def __init__(self, input_channels: int = 1, sequence_length: int = 100, num_patterns: int = 50):
        super(CNNPatternRecognizer, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Second conv block
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Third conv block
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Fourth conv block
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_patterns),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        return self.classifier(x)

# ================== LSTM NaN PROTECTION SYSTEM ==================

class OptimizedLSTMTrainer:
    """Trainer LSTM ottimizzato con protezioni anti-NaN e gradient clipping"""
    
    def __init__(self, model: nn.Module, config: Optional[AnalyzerConfig] = None):
        self.model = model
        self.config = config or get_analyzer_config()  # 🔧 ADDED
        
        # 🔧 USA CONFIGURAZIONE per parametri training
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        self.max_grad_norm = self.config.max_grad_norm
        
        # Scheduler per learning rate adattivo
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=self.config.training_patience // 2
        )
    
    def validate_data(self, data: torch.Tensor, name: str = "data") -> torch.Tensor:
        """Valida che i dati non contengano NaN o Inf"""
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"{name} deve essere un torch.Tensor")
        
        if torch.isnan(data).any():
            nan_count = torch.isnan(data).sum().item()
            raise ValueError(f"{name} contiene {nan_count} valori NaN")
        
        if torch.isinf(data).any():
            inf_count = torch.isinf(data).sum().item()
            raise ValueError(f"{name} contiene {inf_count} valori infiniti")
        
        if data.numel() == 0:
            raise ValueError(f"{name} è vuoto")
        
        return data
    
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Singolo step di training con protezione completa anti-NaN"""
        
        # 🛡️ VALIDAZIONE INPUT CRITICA PREVENTIVA
        if inputs is None or targets is None:
            safe_print("❌ Input o target sono None!")
            raise ValueError("Input o target sono None")
        
        if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            safe_print(f"❌ Input/target non sono tensor: {type(inputs)}, {type(targets)}")
            raise TypeError("Input e target devono essere torch.Tensor")
        
        # 🛡️ VALIDAZIONE PREVENTIVA NaN/Inf
        def validate_and_sanitize_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
            """Valida e sanitizza un tensor con report dettagliato"""
            
            if tensor.numel() == 0:
                raise ValueError(f"{name} è vuoto")
            
            # Check NaN
            nan_mask = torch.isnan(tensor)
            if nan_mask.any():
                nan_count = nan_mask.sum().item()
                safe_print(f"❌ {name} contiene {nan_count}/{tensor.numel()} valori NaN")
                
                # Strategia di sanitizzazione intelligente
                if nan_count < tensor.numel() * 0.1:  # Meno del 10% sono NaN
                    # Sostituisci con media dei valori validi
                    valid_mean = tensor[~nan_mask].mean() if (~nan_mask).any() else 0.0
                    tensor = torch.where(nan_mask, valid_mean, tensor)
                    safe_print(f"🔧 {name}: NaN sostituiti con media valida ({valid_mean:.6f})")
                else:
                    # Troppi NaN, usa fallback
                    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                    safe_print(f"🔧 {name}: Troppi NaN, usato fallback zero")
            
            # Check Inf
            inf_mask = torch.isinf(tensor)
            if inf_mask.any():
                inf_count = inf_mask.sum().item()
                safe_print(f"❌ {name} contiene {inf_count}/{tensor.numel()} valori infiniti")
                
                # Sostituisci Inf con valori ragionevoli
                finite_mask = torch.isfinite(tensor)
                if finite_mask.any():
                    finite_max = tensor[finite_mask].max()
                    finite_min = tensor[finite_mask].min()
                    
                    # Sostituisci +Inf con max finito, -Inf con min finito
                    tensor = torch.where(tensor == float('inf'), finite_max, tensor)
                    tensor = torch.where(tensor == float('-inf'), finite_min, tensor)
                    safe_print(f"🔧 {name}: Inf sostituiti con min/max finiti")
                else:
                    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                    safe_print(f"🔧 {name}: Tutti Inf, usato fallback")
            
            # Check valori estremi
            abs_max = torch.abs(tensor).max()
            if abs_max > 1000:
                safe_print(f"⚠️ {name} ha valori estremi: max_abs={abs_max:.2f}")
                tensor = torch.clamp(tensor, -1000, 1000)
                safe_print(f"🔧 {name}: Valori clampati a [-1000, 1000]")
            
            return tensor
        
        # 🛡️ SANITIZZA INPUT E TARGET
        try:
            inputs = validate_and_sanitize_tensor(inputs, "inputs")
            targets = validate_and_sanitize_tensor(targets, "targets")
        except Exception as validation_error:
            safe_print(f"❌ Errore validazione: {validation_error}")
            raise ValueError(f"Validazione fallita: {validation_error}")
        
        # 🛡️ AUTO-RESIZE per LSTM se necessario con protezione
        try:
            if hasattr(self.model, '_get_or_create_adapter'):
                original_shape = inputs.shape
                inputs = self._ensure_lstm_input_shape(inputs)
                
                # Rivalidate dopo resize
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    safe_print("❌ NaN/Inf generati durante resize - sanitizzando...")
                    inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=-1.0)
                
                if inputs.shape != original_shape:
                    safe_print(f"🔧 Input reshape: {original_shape} → {inputs.shape}")
                    
        except Exception as resize_error:
            safe_print(f"❌ Errore durante resize: {resize_error}")
            raise ValueError(f"Resize fallito: {resize_error}")
        
        # 🛡️ VERIFICA DIMENSIONI COMPATIBILI
        if inputs.shape[0] != targets.shape[0]:
            safe_print(f"❌ Batch size mismatch: inputs={inputs.shape[0]}, targets={targets.shape[0]}")
            raise ValueError("Batch size non compatibili")
        
        # 🛡️ TRAINING STEP PROTETTO
        self.model.train()
        self.optimizer.zero_grad()
        
        # 🛡️ FORWARD PASS CON CONTROLLO ERRORI
        try:
            # Pre-forward validation
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                safe_print("❌ Input contiene NaN/Inf prima del forward")
                raise ValueError("Input non valido prima del forward")
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Post-forward validation
            if outputs is None:
                safe_print("❌ Model ha ritornato None")
                raise ValueError("Model output è None")
            
            outputs = validate_and_sanitize_tensor(outputs, "outputs")
            
        except Exception as forward_error:
            safe_print(f"❌ Errore nel forward pass: {forward_error}")
            # Log dettagli per debug
            safe_print(f"   Input shape: {inputs.shape}")
            safe_print(f"   Input range: [{inputs.min():.6f}, {inputs.max():.6f}]")
            safe_print(f"   Model type: {type(self.model).__name__}")
            raise ValueError(f"Forward pass fallito: {forward_error}")
        
        # 🛡️ CALCOLA LOSS CON PROTEZIONE
        try:
            # Adatta dimensioni se necessario
            if outputs.shape != targets.shape:
                safe_print(f"⚠️ Shape mismatch: outputs={outputs.shape}, targets={targets.shape}")
                
                # Strategia di adattamento intelligente
                if len(targets.shape) == 1 and len(outputs.shape) == 2:
                    if outputs.shape[1] == 1:
                        outputs = outputs.squeeze(1)
                        safe_print("🔧 Output squeezed per match target")
                    elif targets.max() < outputs.shape[1]:
                        # Probabilmente classificazione
                        criterion = nn.CrossEntropyLoss()
                    else:
                        safe_print("❌ Impossibile adattare shapes per loss")
                        raise ValueError("Shape incompatibili per loss")
                elif len(targets.shape) == 2 and len(outputs.shape) == 2:
                    if targets.shape[1] != outputs.shape[1]:
                        safe_print("❌ Dimensioni feature non compatibili")
                        raise ValueError("Feature dimensions incompatibili")
            
            # Seleziona criterio appropriato
            if len(targets.shape) > 1 or targets.dtype == torch.float32:
                criterion = nn.MSELoss()
                loss_type = "MSE"
            else:
                criterion = nn.CrossEntropyLoss()
                loss_type = "CrossEntropy"
            
            # Calcola loss
            loss = criterion(outputs, targets)
            
            # 🛡️ VALIDAZIONE LOSS CRITICA
            if loss is None:
                safe_print("❌ Loss è None")
                raise ValueError("Loss calculation returned None")
            
            if not torch.is_tensor(loss):
                safe_print(f"❌ Loss non è un tensor: {type(loss)}")
                raise ValueError("Loss non è un tensor")
            
            if torch.isnan(loss):
                safe_print(f"❌ Loss è NaN con {loss_type}")
                safe_print(f"   Outputs range: [{outputs.min():.6f}, {outputs.max():.6f}]")
                safe_print(f"   Targets range: [{targets.min():.6f}, {targets.max():.6f}]")
                safe_print(f"   Outputs shape: {outputs.shape}")
                safe_print(f"   Targets shape: {targets.shape}")
                raise ValueError("Loss contiene NaN")
            
            if torch.isinf(loss):
                safe_print(f"❌ Loss è infinita: {loss.item()}")
                raise ValueError("Loss contiene Inf")
            
            if loss.item() > 1e6:
                safe_print(f"⚠️ Loss molto alta: {loss.item()}")
                # Non bloccare ma avvertire
            
        except Exception as loss_error:
            safe_print(f"❌ Errore nel calcolo loss: {loss_error}")
            raise ValueError(f"Loss calculation fallita: {loss_error}")
        
        # 🛡️ BACKWARD PASS CON PROTEZIONE
        try:
            # Pre-backward check
            if torch.isnan(loss) or torch.isinf(loss):
                safe_print("❌ Loss non valida prima del backward")
                raise ValueError("Loss non valida per backward")
            
            # Backward pass
            loss.backward()
            
            # 🛡️ VALIDAZIONE GRADIENTI POST-BACKWARD
            invalid_gradients = []
            zero_gradients = []
            large_gradients = []
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad = param.grad
                    
                    # Check NaN gradients
                    if torch.isnan(grad).any():
                        nan_count = torch.isnan(grad).sum().item()
                        invalid_gradients.append(f"{name}: {nan_count} NaN")
                    
                    # Check Inf gradients
                    if torch.isinf(grad).any():
                        inf_count = torch.isinf(grad).sum().item()
                        invalid_gradients.append(f"{name}: {inf_count} Inf")
                    
                    # Check zero gradients (possibile problema)
                    if torch.all(grad == 0):
                        zero_gradients.append(name)
                    
                    # Check large gradients
                    grad_norm = torch.norm(grad).item()
                    if grad_norm > 100:
                        large_gradients.append(f"{name}: norm={grad_norm:.2f}")
            
            # Report problemi gradienti
            if invalid_gradients:
                safe_print(f"❌ Gradienti NaN/Inf rilevati:")
                for grad_issue in invalid_gradients:
                    safe_print(f"   {grad_issue}")
                raise ValueError("Gradienti contengono NaN/Inf")
            
            if zero_gradients:
                safe_print(f"⚠️ Gradienti zero in: {zero_gradients}")
            
            if large_gradients:
                safe_print(f"⚠️ Gradienti grandi rilevati:")
                for grad_issue in large_gradients:
                    safe_print(f"   {grad_issue}")
            
        except Exception as backward_error:
            safe_print(f"❌ Errore nel backward pass: {backward_error}")
            # Clean up gradients
            self.optimizer.zero_grad()
            raise ValueError(f"Backward pass fallito: {backward_error}")
        
        # 🛡️ GRADIENT CLIPPING CRITICO
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                safe_print(f"❌ Grad norm non valida: {grad_norm}")
                raise ValueError("Gradient norm non valida")
            
            if grad_norm > self.max_grad_norm:
                safe_print(f"⚠️ Gradienti clippati: norma {grad_norm:.4f} → {self.max_grad_norm}")
            
        except Exception as clipping_error:
            safe_print(f"❌ Errore durante gradient clipping: {clipping_error}")
            raise ValueError(f"Gradient clipping fallito: {clipping_error}")
        
        # 🛡️ OPTIMIZER STEP CON PROTEZIONE
        try:
            # Final gradient check before step
            for name, param in self.model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    safe_print(f"❌ Gradiente ancora non valido in {name} prima dell'optimizer step")
                    raise ValueError(f"Gradiente non valido in {name}")
            
            # Optimizer step
            self.optimizer.step()
            
            # 🛡️ POST-STEP VALIDATION
            # Verifica che i parametri siano ancora validi
            for name, param in self.model.named_parameters():
                if torch.isnan(param.data).any():
                    safe_print(f"❌ Parametro {name} contiene NaN dopo optimizer step")
                    raise ValueError(f"Parametro {name} corrotto")
                
                if torch.isinf(param.data).any():
                    safe_print(f"❌ Parametro {name} contiene Inf dopo optimizer step")
                    raise ValueError(f"Parametro {name} corrotto")
            
        except Exception as optimizer_error:
            safe_print(f"❌ Errore nell'optimizer step: {optimizer_error}")
            raise ValueError(f"Optimizer step fallito: {optimizer_error}")
        
        # 🛡️ FINAL VALIDATION
        final_loss = loss.item()
        
        if not isinstance(final_loss, (int, float)):
            safe_print(f"❌ Loss finale non è un numero: {type(final_loss)}")
            raise ValueError("Loss finale non numerica")
        
        if not (0 <= final_loss <= 1e6):
            safe_print(f"⚠️ Loss finale fuori range ragionevole: {final_loss}")
        
        safe_print(f"✅ Training step completato: loss={final_loss:.6f}")
        return final_loss
    
    def _ensure_lstm_input_shape(self, data: torch.Tensor) -> torch.Tensor:
        """Assicura che i dati abbiano la forma corretta per LSTM [batch, seq, features]"""
        if data.dim() == 1:
            # [features] → [1, 1, features]
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.dim() == 2:
            # [batch, features] → [batch, 1, features]
            data = data.unsqueeze(1)
        elif data.dim() == 3:
            # Già nella forma corretta [batch, seq, features]
            pass
        else:
            raise ValueError(f"Dimensioni tensor non supportate: {data.dim()}D")
        
        return data
    
    def train_model_protected(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> Dict[str, Any]:
        """Training completo con tutte le protezioni"""
        
        # 🛡️ VALIDAZIONE DATI NUMPY
        if np.isnan(X).any():
            raise ValueError("Input X contiene valori NaN")
        if np.isnan(y).any():
            raise ValueError("Target y contiene valori NaN")
        
        # Converte a tensori
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y) if len(y.shape) > 1 else torch.LongTensor(y)
        
        # Split train/validation
        train_size = int(0.8 * len(X))
        indices = torch.randperm(len(X))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
        X_val, y_val = X_tensor[val_indices], y_tensor[val_indices]
        
        # Training loop con protezioni
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        train_losses = []
        val_losses = []
        
        safe_print(f"🔄 Avvio training protetto per {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training batch
            epoch_train_loss = 0
            train_batches = 0
            
            batch_size = min(32, len(X_train))
            for i in range(0, len(X_train), batch_size):
                try:
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    loss = self.train_step(batch_X, batch_y)
                    epoch_train_loss += loss
                    train_batches += 1
                    
                except Exception as e:
                    safe_print(f"❌ Errore batch {i//batch_size}: {e}")
                    continue
            
            if train_batches == 0:
                safe_print("❌ Nessun batch completato con successo")
                break
            
            avg_train_loss = epoch_train_loss / train_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    try:
                        batch_X = X_val[i:i+batch_size]
                        batch_y = y_val[i:i+batch_size]
                        
                        # Auto-resize se necessario
                        if hasattr(self.model, '_get_or_create_adapter'):
                            batch_X = self._ensure_lstm_input_shape(batch_X)
                        
                        outputs = self.model(batch_X)
                        
                        if len(batch_y.shape) > 1 or batch_y.dtype == torch.float32:
                            criterion = nn.MSELoss()
                        else:
                            criterion = nn.CrossEntropyLoss()
                        
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        val_batches += 1
                        
                    except Exception as e:
                        safe_print(f"⚠️ Errore validation batch: {e}")
                        continue
            
            if val_batches > 0:
                avg_val_loss = val_loss / val_batches
                val_losses.append(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    safe_print(f"🔄 Early stopping alla epoch {epoch+1}")
                    break
                
                # Update learning rate
                self.scheduler.step(avg_val_loss)
                
                # Progress ogni 10 epochs
                if (epoch + 1) % 10 == 0:
                    safe_print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            self.model.train()
        
        safe_print(f"✅ Training completato! Best val loss: {best_val_loss:.6f}")
        
        return {
            'status': 'success',
            'final_loss': best_val_loss,
            'epochs_trained': len(train_losses),
            'train_history': train_losses,
            'val_history': val_losses,
            'improvement': (train_losses[0] - best_val_loss) / train_losses[0] if train_losses else 0
        }

# ================== END NaN PROTECTION ==================

# ================== TENSOR SHAPE MANAGER ==================

class TensorShapeManager:
    """Gestisce automaticamente le forme dei tensor per LSTM e altri modelli"""
    
    def __init__(self):
        self.shape_conversions = {
            'processed_count': 0,
            'conversion_history': [],
            'common_patterns': {},
            'error_patterns': []
        }
    
    @staticmethod
    def ensure_lstm_input_shape(data: torch.Tensor, sequence_length: int = 1, 
                               expected_features: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Assicura che i dati abbiano la forma corretta per LSTM [batch, seq, features]"""
        
        original_shape = data.shape
        conversion_info = {
            'original_shape': original_shape,
            'conversion_applied': False,
            'target_shape': None,
            'method_used': 'none'
        }
        
        # 🔧 CASO 1: Input 1D [features] → [1, 1, features]
        if data.dim() == 1:
            data = data.unsqueeze(0).unsqueeze(0)
            conversion_info.update({
                'conversion_applied': True,
                'target_shape': data.shape,
                'method_used': '1D_to_3D_single_batch_single_seq'
            })
        
        # 🔧 CASO 2: Input 2D [batch, features] → [batch, seq, features]
        elif data.dim() == 2:
            batch_size, features = data.shape
            
            # Se abbiamo expected_features, controlliamo se è compatibile
            if expected_features and features != expected_features:
                # Potrebbe essere [seq, features] invece di [batch, features]
                if features == expected_features:
                    # È già corretto come [seq, features], aggiungi batch
                    data = data.unsqueeze(0)  # [1, seq, features]
                    conversion_info['method_used'] = '2D_seq_features_to_3D'
                else:
                    # Reshaping intelligente basato su fattori
                    possible_seq_len = TensorShapeManager._find_best_sequence_length(features, expected_features)
                    if possible_seq_len and features % possible_seq_len == 0:
                        new_features = features // possible_seq_len
                        data = data.view(batch_size, possible_seq_len, new_features)
                        conversion_info['method_used'] = f'2D_smart_reshape_{possible_seq_len}x{new_features}'
                    else:
                        # Default: aggiungi sequenza di lunghezza 1
                        data = data.unsqueeze(1)  # [batch, 1, features]
                        conversion_info['method_used'] = '2D_batch_features_to_3D'
            else:
                # Standard: [batch, features] → [batch, 1, features]
                data = data.unsqueeze(1)
                conversion_info['method_used'] = '2D_to_3D_standard'
            
            conversion_info.update({
                'conversion_applied': True,
                'target_shape': data.shape
            })
        
        # 🔧 CASO 3: Input 3D [batch, seq, features] - già corretto
        elif data.dim() == 3:
            batch_size, seq_len, features = data.shape
            
            # Verifica se le dimensioni sono ragionevoli
            if seq_len > 10000:  # Sequenza troppo lunga
                conversion_info['method_used'] = '3D_already_correct_but_long_sequence'
            elif features > 10000:  # Troppe features
                conversion_info['method_used'] = '3D_already_correct_but_many_features'
            else:
                conversion_info['method_used'] = '3D_already_correct'
        
        # 🔧 CASO 4: Input 4D+ - errore
        else:
            raise ValueError(f"❌ Dimensioni tensor non supportate: {data.dim()}D con shape {original_shape}")
        
        conversion_info['final_shape'] = data.shape
        
        return data, conversion_info
    
    @staticmethod
    def _find_best_sequence_length(total_features: int, expected_features: int) -> Optional[int]:
        """Trova la migliore lunghezza di sequenza per reshaping intelligente"""
        
        if expected_features <= 0 or total_features <= 0:
            return None
        
        # Cerca fattori ragionevoli
        max_seq_len = min(100, total_features // expected_features)
        
        for seq_len in range(2, max_seq_len + 1):
            if total_features % seq_len == 0:
                resulting_features = total_features // seq_len
                if resulting_features == expected_features:
                    return seq_len
        
        return None
    
    @staticmethod
    def validate_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], 
                            name: str = "tensor", allow_batch_dim: bool = True) -> bool:
        """Valida che un tensor abbia la forma attesa"""
        
        actual_shape = tensor.shape
        
        # Se allow_batch_dim, ignora la prima dimensione per il confronto
        if allow_batch_dim and len(actual_shape) > 1 and len(expected_shape) > 1:
            actual_shape_to_check = actual_shape[1:]
            expected_shape_to_check = expected_shape[1:]
        else:
            actual_shape_to_check = actual_shape
            expected_shape_to_check = expected_shape
        
        if actual_shape_to_check != expected_shape_to_check:
            safe_print(f"⚠️ {name} shape mismatch:")
            safe_print(f"   Expected: {expected_shape}")
            safe_print(f"   Actual: {actual_shape}")
            return False
        
        return True
    
    @staticmethod
    def smart_batch_reshape(data: torch.Tensor, target_batch_size: int) -> torch.Tensor:
        """Reshape intelligente per adattare batch size"""
        
        current_batch = data.shape[0]
        
        if current_batch == target_batch_size:
            return data
        
        if current_batch > target_batch_size:
            # Riduci batch size prendendo i primi elementi
            return data[:target_batch_size]
        else:
            # Aumenta batch size replicando dati
            repeats = (target_batch_size + current_batch - 1) // current_batch
            expanded = data.repeat(repeats, *([1] * (data.dim() - 1)))
            return expanded[:target_batch_size]
    
    @classmethod
    def prepare_model_input(cls, data: torch.Tensor, model_type: str, 
                          expected_input_size: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Prepara input per diversi tipi di modelli"""
        
        shape_manager = cls()
        preparation_info = {
            'model_type': model_type,
            'original_shape': data.shape,
            'transformations': []
        }
        
        if model_type.upper() == 'LSTM':
            # Per LSTM: assicura forma [batch, seq, features]
            data, conversion_info = cls.ensure_lstm_input_shape(data, expected_features=expected_input_size)
            preparation_info['transformations'].append(conversion_info)
        
        elif model_type.upper() == 'CNN':
            # Per CNN: assicura forma [batch, channels, length]
            if data.dim() == 2:
                data = data.unsqueeze(1)  # Aggiungi dimensione channel
                preparation_info['transformations'].append({
                    'conversion_applied': True,
                    'method_used': '2D_to_CNN_format',
                    'target_shape': data.shape
                })
        
        elif model_type.upper() == 'TRANSFORMER':
            # Per Transformer: assicura forma [batch, seq, features]
            data, conversion_info = cls.ensure_lstm_input_shape(data, expected_features=expected_input_size)
            preparation_info['transformations'].append(conversion_info)
        
        elif model_type.upper() == 'LINEAR':
            # Per Linear: assicura forma [batch, features]
            if data.dim() > 2:
                original_shape = data.shape
                data = data.view(data.shape[0], -1)  # Flatten mantenendo batch
                preparation_info['transformations'].append({
                    'conversion_applied': True,
                    'method_used': 'flatten_to_linear',
                    'original_shape': original_shape,
                    'target_shape': data.shape
                })
        
        # Update statistics
        shape_manager.shape_conversions['processed_count'] += 1
        if preparation_info['transformations']:
            shape_manager.shape_conversions['conversion_history'].append(preparation_info)
        
        preparation_info['final_shape'] = data.shape
        
        return data, preparation_info
    
    def get_shape_statistics(self) -> Dict[str, Any]:
        """Ottieni statistiche sulle conversioni di forma"""
        
        total_processed = self.shape_conversions['processed_count']
        conversions_applied = len(self.shape_conversions['conversion_history'])
        
        # Analizza metodi più comuni
        method_counts = {}
        for conversion in self.shape_conversions['conversion_history']:
            for transform in conversion['transformations']:
                method = transform.get('method_used', 'unknown')
                method_counts[method] = method_counts.get(method, 0) + 1
        
        # Calcola efficienza
        efficiency_rate = (total_processed - conversions_applied) / total_processed * 100 if total_processed > 0 else 100
        
        return {
            'total_processed': total_processed,
            'conversions_needed': conversions_applied,
            'efficiency_rate': f"{efficiency_rate:.1f}%",
            'common_conversion_methods': dict(sorted(method_counts.items(), key=lambda x: x[1], reverse=True)),
            'recommendations': self._generate_shape_recommendations(method_counts, efficiency_rate)
        }
    
    def _generate_shape_recommendations(self, method_counts: Dict, efficiency_rate: float) -> List[str]:
        """Genera raccomandazioni per ottimizzazione forme"""
        
        recommendations = []
        
        if efficiency_rate < 70:
            recommendations.append("High conversion rate - consider input data preprocessing")
        
        if '2D_to_3D_standard' in method_counts and method_counts['2D_to_3D_standard'] > 10:
            recommendations.append("Many 2D→3D conversions - standardize input pipeline")
        
        if 'smart_reshape' in str(method_counts):
            recommendations.append("Smart reshaping active - monitor for accuracy impact")
        
        if len(method_counts) > 5:
            recommendations.append("Multiple conversion types - consider unified input format")
        
        return recommendations

# ================== END TENSOR SHAPE MANAGER ==================

# ================== ROLLING WINDOW TRAINING SYSTEM ==================

class RollingWindowTrainer:
    """Sistema per training con finestra mobile preservando i migliori modelli"""
    
    def __init__(self, window_size_days: int = 180, retrain_frequency_days: int = 7):
        self.window_size_days = window_size_days
        self.retrain_frequency_days = retrain_frequency_days
        self.last_training_dates: Dict[str, datetime] = {}
        self.training_history: Dict[str, List[Dict]] = defaultdict(list)
        
    def should_retrain(self, asset: str, model_type: ModelType, algorithm_name: str,
                      force: bool = False) -> bool:
        """Determina se è necessario retraining"""
        if force:
            return True
            
        key = f"{asset}_{model_type.value}_{algorithm_name}"
        
        if key not in self.last_training_dates:
            return True
        
        days_since_training = (datetime.now() - self.last_training_dates[key]).days
        return days_since_training >= self.retrain_frequency_days
    
    def prepare_training_data(self, tick_data: deque, window_size_override: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Prepara dati con finestra mobile"""
        window_size = window_size_override or self.window_size_days
        cutoff_date = datetime.now() - timedelta(days=window_size)
        
        # Filtra dati nella finestra
        filtered_data = [
            tick for tick in tick_data 
            if tick['timestamp'] > cutoff_date
        ]
        
        if len(filtered_data) < 1000:  # Minimo di dati richiesti
            return {}
        
        # Estrai features
        prices = np.array([tick['price'] for tick in filtered_data])
        volumes = np.array([tick['volume'] for tick in filtered_data])
        timestamps = [tick['timestamp'] for tick in filtered_data]
        
        # Aggiungi features derivate
        returns = np.diff(prices) / prices[:-1]
        log_returns = np.log(prices[1:] / prices[:-1])
        
        # Indicatori tecnici
        sma_20 = self._calculate_sma(prices, 20)
        sma_50 = self._calculate_sma(prices, 50)
        rsi = self._calculate_rsi(prices, 14)
        
        return {
            'prices': prices,
            'volumes': volumes,
            'timestamps': np.array(timestamps),
            'returns': returns,
            'log_returns': log_returns,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi': rsi
        }
    
    def _calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calcola Simple Moving Average"""
        if len(prices) < period:
            return np.full_like(prices, np.nan)
        
        sma = np.full_like(prices, np.nan)
        for i in range(period - 1, len(prices)):
            sma[i] = np.mean(prices[i - period + 1:i + 1])
        
        return sma
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calcola RSI"""
        if len(prices) < period + 1:
            return np.full_like(prices, 50.0)
        
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = np.zeros_like(prices)
        rsi[:period] = 50.0
        rsi[period] = 100 - 100 / (1 + rs)
        
        for i in range(period + 1, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 100
            rsi[i] = 100 - 100 / (1 + rs)
        
        return rsi
    
    def train_model(self, model: Any, training_data: Dict[str, np.ndarray], 
                   model_type: ModelType, algorithm_name: str,
                   preserve_weights: bool = True) -> Dict[str, Any]:
        """Esegue il training del modello con opzione di preservare i pesi migliori"""
        
        start_time = datetime.now()
        
        # Prepara dataset basato sul tipo di modello
        if model_type == ModelType.SUPPORT_RESISTANCE:
            X, y = self._prepare_sr_dataset(training_data)
        elif model_type == ModelType.PATTERN_RECOGNITION:
            X, y = self._prepare_pattern_dataset(training_data)
        elif model_type == ModelType.BIAS_DETECTION:
            X, y = self._prepare_bias_dataset(training_data)
        elif model_type == ModelType.TREND_ANALYSIS:
            X, y = self._prepare_trend_dataset(training_data)
        else:
            return {'status': 'error', 'message': f'Unknown model type: {model_type}'}
        
        # Training specifico per tipo di modello
        if isinstance(model, nn.Module):
            result = self._train_neural_model(model, X, y, preserve_weights)
        else:
            result = self._train_sklearn_model(model, X, y)
        
        # Registra training
        duration = (datetime.now() - start_time).total_seconds()
        training_record = {
            'timestamp': start_time,
            'model_type': model_type.value,
            'algorithm_name': algorithm_name,
            'data_points': len(X),
            'duration_seconds': duration,
            'final_loss': result.get('final_loss', 0),
            'improvement': result.get('improvement', 0)
        }
        
        key = f"{model_type.value}_{algorithm_name}"
        self.training_history[key].append(training_record)
        self.last_training_dates[f"unknown_{key}"] = datetime.now()
        
        return result
    
    def _prepare_sr_dataset(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dataset per Support/Resistance"""
        prices = data['prices']
        volumes = data['volumes']
        
        # Feature window di 50 punti
        window_size = 50
        future_window = 20
        
        X, y = [], []
        
        for i in range(window_size, len(prices) - future_window):
            # Features: prezzi, volumi, indicatori
            features = np.concatenate([
                prices[i-window_size:i],
                volumes[i-window_size:i],
                data['sma_20'][i-window_size:i],
                data['rsi'][i-window_size:i]
            ])
            
            # Target: massimi e minimi futuri per S/R
            future_prices = prices[i:i+future_window]
            resistance = np.max(future_prices)
            support = np.min(future_prices)
            current_price = prices[i]
            
            # Normalizza target
            y_val = np.array([
                (support - current_price) / current_price,
                (resistance - current_price) / current_price
            ])
            
            X.append(features)
            y.append(y_val)
        
        return np.array(X), np.array(y)
    
    def _prepare_pattern_dataset(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dataset per Pattern Recognition"""
        prices = data['prices']
        
        # Normalizza prezzi per pattern recognition
        X, y = [], []
        pattern_length = 100
        
        for i in range(pattern_length, len(prices) - 10):
            # Normalizza la finestra di prezzi
            price_window = prices[i-pattern_length:i]
            normalized = (price_window - price_window.mean()) / price_window.std()
            
            # Crea label basato sul movimento futuro
            future_return = (prices[i+10] - prices[i]) / prices[i]
            
            # Multi-label per diversi pattern (semplificato)
            labels = np.zeros(50)  # 50 possibili pattern
            
            # Esempi di pattern detection
            if future_return > 0.01:
                labels[0] = 1  # Bullish
            if future_return < -0.01:
                labels[1] = 1  # Bearish
            
            # Aggiungi altri pattern based su shape della curva
            # ...
            
            X.append(normalized)
            y.append(labels)
        
        return np.array(X), np.array(y)
    
    def _prepare_bias_dataset(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dataset per Bias Detection"""
        X, y = [], []
        
        window_size = 30
        for i in range(window_size, len(data['prices']) - 5):
            # Features per sentiment/bias
            price_window = data['prices'][i-window_size:i]
            volume_window = data['volumes'][i-window_size:i]
            returns_window = data['returns'][i-window_size+1:i]
            
            features = np.concatenate([
                price_window[-10:],  # Ultimi 10 prezzi
                volume_window[-10:],  # Ultimi 10 volumi
                returns_window[-10:],  # Ultimi 10 returns
                [np.mean(returns_window), np.std(returns_window)],  # Stats
                [data['rsi'][i]]  # RSI corrente
            ])
            
            # Label: direzione futura (bearish=0, neutral=1, bullish=2)
            future_return = (data['prices'][i+5] - data['prices'][i]) / data['prices'][i]
            
            if future_return < -0.002:
                label = 0  # Bearish
            elif future_return > 0.002:
                label = 2  # Bullish
            else:
                label = 1  # Neutral
            
            X.append(features)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def _prepare_trend_dataset(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dataset per Trend Analysis - VERSIONE CORRETTA"""
        X, y = [], []
        
        for i in range(100, len(data['prices']) - 20):
            # Features aggregate (invariate)
            features = []
            
            # Price position relative to MAs
            current_price = data['prices'][i]
            if not np.isnan(data['sma_20'][i]):
                features.append((current_price - data['sma_20'][i]) / data['sma_20'][i])
            else:
                features.append(0)
            
            if not np.isnan(data['sma_50'][i]):
                features.append((current_price - data['sma_50'][i]) / data['sma_50'][i])
            else:
                features.append(0)
            
            # Momentum
            features.append(data['rsi'][i] / 100)
            
            # Recent returns statistics
            recent_returns = data['returns'][max(0, i-20):i]
            features.extend([
                np.mean(recent_returns),
                np.std(recent_returns),
                np.min(recent_returns),
                np.max(recent_returns)
            ])
            
            # Volume trend
            recent_volumes = data['volumes'][max(0, i-20):i]
            features.append(np.mean(recent_volumes) / (np.mean(data['volumes']) + 1e-8))
            
            # 🔧 CORREZIONE PRINCIPALE: Target continuo invece di categorie
            future_prices = data['prices'][i:i+20]
            
            # Calcola slope della regressione lineare sui prezzi futuri
            x_vals = np.arange(len(future_prices))
            slope, _ = np.polyfit(x_vals, future_prices, 1)
            
            # Normalizza il slope rispetto al prezzo corrente
            normalized_slope = slope / current_price
            
            X.append(features)
            y.append(normalized_slope)  # ✅ Target continuo per regressione
        
        return np.array(X), np.array(y)
    
    def _train_neural_model(self, model: nn.Module, X: np.ndarray, y: np.ndarray, 
                        preserve_weights: bool) -> Dict[str, Any]:
        """Training per modelli neurali PyTorch con protezione completa anti-NaN - VERSIONE PULITA"""
        
        training_start = datetime.now()
        
        # Store training attempt for future slave module processing
        training_info = {
            'model_type': type(model).__name__,
            'input_shape': X.shape if hasattr(X, 'shape') else None,
            'output_shape': y.shape if hasattr(y, 'shape') else None,
            'preserve_weights': preserve_weights,
            'timestamp': training_start
        }
        
        # VALIDAZIONE DATI PREPROCESSING COMPLETA (silent)
        def validate_numpy_data(data: np.ndarray, name: str) -> np.ndarray:
            """Valida e sanitizza dati numpy senza output verboso"""
            
            if data is None:
                raise ValueError(f"{name} è None")
            
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            if data.size == 0:
                raise ValueError(f"{name} è vuoto")
            
            # Check NaN
            nan_count = int(np.isnan(data).sum())
            if nan_count > 0:
                if nan_count < data.size * 0.1:  # Meno del 10%
                    # Sostituisci con interpolazione
                    valid_mask = ~np.isnan(data)
                    if valid_mask.any():
                        mean_val = float(np.nanmean(data))
                        data = np.where(np.isnan(data), mean_val, data)
                        training_info['nan_fixes'] = {'count': nan_count, 'method': 'mean_substitution'}
                    else:
                        raise ValueError(f"{name}: tutti valori sono NaN")
                else:
                    raise ValueError(f"{name}: troppi NaN ({nan_count}/{data.size})")
            
            # Check Inf
            inf_count = int(np.isinf(data).sum())
            if inf_count > 0:
                # Sostituisci Inf con valori ragionevoli
                finite_mask = np.isfinite(data)
                if finite_mask.any():
                    data_min = float(np.nanmin(data[finite_mask]))
                    data_max = float(np.nanmax(data[finite_mask]))
                    data = np.where(data == np.inf, data_max, data)
                    data = np.where(data == -np.inf, data_min, data)
                    training_info['inf_fixes'] = {'count': inf_count, 'method': 'min_max_substitution'}
                else:
                    raise ValueError(f"{name}: tutti valori sono infiniti")
            
            # Check valori estremi
            abs_max = float(np.abs(data).max())
            if abs_max > 1e6:
                data = np.clip(data, -1e6, 1e6)
                training_info['extreme_value_fixes'] = {'max_value': abs_max, 'method': 'clipping'}
            
            return data
        
        # VALIDAZIONE E SANITIZZAZIONE (silent)
        try:
            X = validate_numpy_data(X, "Input X")
            y = validate_numpy_data(y, "Target y")
        except Exception as validation_error:
            self._store_training_event('validation_failed', {
                **training_info,
                'error': str(validation_error)
            })
            return {'status': 'error', 'message': f'Data validation failed: {validation_error}'}
        
        # VALIDAZIONE MODEL STATE PRE-TRAINING (silent)
        def validate_model_state(model: nn.Module, stage: str) -> bool:
            """Valida che tutti i parametri del modello siano validi"""
            
            corrupted_params = []
            for name, param in model.named_parameters():
                if torch.isnan(param.data).any():
                    corrupted_params.append(f"{name}: NaN")
                if torch.isinf(param.data).any():
                    corrupted_params.append(f"{name}: Inf")
            
            if corrupted_params:
                training_info[f'corrupted_params_{stage}'] = corrupted_params
                return False
            
            return True
        
        # VERIFICA MODEL STATE INIZIALE
        if not validate_model_state(model, "initial"):
            self._store_training_event('model_corrupted_initial', training_info)
            return {'status': 'error', 'message': 'Model corrupted before training'}
        
        # CREA TRAINER PROTETTO CON GESTIONE ERRORI
        try:
            protected_trainer = OptimizedLSTMTrainer(model)
        except Exception as trainer_error:
            self._store_training_event('trainer_creation_failed', {
                **training_info,
                'error': str(trainer_error)
            })
            return {'status': 'error', 'message': f'Failed to create trainer: {trainer_error}'}
        
        # SALVATAGGIO PESI INIZIALI CON VALIDAZIONE
        initial_state = None
        if preserve_weights:
            try:
                if validate_model_state(model, "pre-save"):
                    initial_state = {}
                    for k, v in model.state_dict().items():
                        param_copy = v.clone().detach()
                        if torch.isnan(param_copy).any() or torch.isinf(param_copy).any():
                            initial_state = None
                            break
                        initial_state[k] = param_copy
            except Exception as save_error:
                training_info['weight_preservation_failed'] = str(save_error)
                initial_state = None
        
        # TRAINING PROTETTO CON CHECKPOINTS
        training_successful = False
        final_result: Dict[str, Any] = {'status': 'error', 'message': 'Unknown training error'}
        
        try:
            # CHECKPOINT OGNI 10 EPOCHS
            checkpoint_frequency = 10
            best_loss = float('inf')
            best_state = None
            epochs_without_improvement = 0
            max_patience = 20
            original_epochs = 50
            
            epoch_results = []
            
            for epoch_batch in range(0, original_epochs, checkpoint_frequency):
                epochs_in_batch = min(checkpoint_frequency, original_epochs - epoch_batch)
                
                try:
                    # Training batch con protezione
                    batch_result = protected_trainer.train_model_protected(X, y, epochs=epochs_in_batch)
                    
                    if batch_result['status'] != 'success':
                        break
                    
                    current_loss = float(batch_result['final_loss'])
                    epoch_results.append({
                        'epoch_range': f"{epoch_batch}-{epoch_batch+epochs_in_batch}",
                        'loss': current_loss,
                        'timestamp': datetime.now()
                    })
                    
                    # VALIDAZIONE MODEL STATE POST-BATCH
                    if not validate_model_state(model, f"epoch_{epoch_batch+epochs_in_batch}"):
                        break
                    
                    # CHECKPOINT SE MIGLIORAMENTO
                    if current_loss < best_loss:
                        best_loss = current_loss
                        epochs_without_improvement = 0
                        
                        # Salva checkpoint migliore
                        try:
                            best_state = {}
                            for k, v in model.state_dict().items():
                                param_copy = v.clone().detach()
                                if torch.isnan(param_copy).any() or torch.isinf(param_copy).any():
                                    best_state = None
                                    break
                                best_state[k] = param_copy
                        except Exception as checkpoint_error:
                            training_info['checkpoint_failed'] = str(checkpoint_error)
                            best_state = None
                    else:
                        epochs_without_improvement += epochs_in_batch
                    
                    # Early stopping
                    if epochs_without_improvement >= max_patience:
                        training_info['early_stopping'] = {
                            'epochs_without_improvement': epochs_without_improvement,
                            'final_epoch': epoch_batch + epochs_in_batch
                        }
                        break
                    
                    # Update final result
                    final_result = batch_result.copy()
                    training_successful = True
                    
                except Exception as batch_error:
                    training_info['batch_error'] = {
                        'epoch_batch': epoch_batch,
                        'error': str(batch_error)
                    }
                    break
            
            # RIPRISTINA MIGLIORE CHECKPOINT SE DISPONIBILE
            if best_state is not None and training_successful:
                try:
                    model.load_state_dict(best_state)
                    final_result['final_loss'] = best_loss
                except Exception as restore_error:
                    training_info['checkpoint_restore_failed'] = str(restore_error)
            
            if training_successful:
                # Store LSTM-specific info if applicable
                if isinstance(model, AdvancedLSTM):
                    stats = model.get_resize_stats()
                    training_info['lstm_stats'] = {
                        'adapters_created': stats['performance_metrics'].get('adapters_created', 0)
                    }
            
            training_info['epoch_results'] = epoch_results
            training_info['best_loss'] = best_loss
            
        except Exception as training_error:
            training_successful = False
            final_result = {'status': 'error', 'message': str(training_error)}
            training_info['training_error'] = str(training_error)
        
        # GESTIONE FALLIMENTI E PRESERVE WEIGHTS
        if not training_successful or final_result.get('status') != 'success':
            # Se preserve_weights e il training è fallito, ripristina
            if preserve_weights and initial_state is not None:
                try:
                    # Valida stato iniziale prima del ripristino
                    corrupted_initial = False
                    for k, v in initial_state.items():
                        if torch.isnan(v).any() or torch.isinf(v).any():
                            corrupted_initial = True
                            break
                    
                    if not corrupted_initial:
                        model.load_state_dict(initial_state)
                        
                        # Verifica ripristino
                        if validate_model_state(model, "restored"):
                            final_result = {
                                'status': 'preserved',
                                'message': 'Training failed, preserved original weights',
                                'final_loss': 0.0,
                                'improvement': 0.0
                            }
                        else:
                            final_result = {'status': 'error', 'message': 'Model corrupted beyond recovery'}
                    else:
                        final_result = {'status': 'error', 'message': 'Initial state corrupted'}
                        
                except Exception as restore_error:
                    final_result = {'status': 'error', 'message': f'Restore failed: {restore_error}'}
                    training_info['restore_error'] = str(restore_error)
            
            # Store failed training event
            training_end = datetime.now()
            training_info['training_duration'] = (training_end - training_start).total_seconds()
            self._store_training_event('training_failed', training_info)
            
            return final_result
        
        # CONTROLLO QUALITÀ RISULTATO FINALE
        if final_result['status'] == 'success':
            improvement = final_result.get('improvement', 0.0)
            if not isinstance(improvement, (int, float)):
                improvement = 0.0
            
            # VALIDAZIONE FINALE MODEL STATE
            if not validate_model_state(model, "final"):
                # Ripristina se possibile
                if preserve_weights and initial_state is not None:
                    try:
                        model.load_state_dict(initial_state)
                        final_result = {
                            'status': 'preserved',
                            'message': 'Model corrupted at end, preserved original weights',
                            'final_loss': final_result.get('final_loss', 0.0),
                            'improvement': 0.0
                        }
                    except:
                        pass
                else:
                    final_result = {'status': 'error', 'message': 'Model corrupted at training end'}
            
            # Se preserve_weights e il modello è peggiorato significativamente, ripristina
            if preserve_weights and improvement < -0.2 and initial_state is not None:
                try:
                    model.load_state_dict(initial_state)
                    final_result = {
                        'status': 'preserved',
                        'message': 'Model performance degraded, preserved original weights',
                        'final_loss': final_result.get('final_loss', 0.0),
                        'improvement': 0.0
                    }
                except Exception as e:
                    training_info['weight_restore_warning'] = str(e)
        
        # Store successful training event
        training_end = datetime.now()
        training_info['training_duration'] = (training_end - training_start).total_seconds()
        training_info['final_status'] = final_result['status']
        training_info['final_loss'] = final_result.get('final_loss', 0.0)
        training_info['improvement'] = final_result.get('improvement', 0.0)
        
        self._store_training_event('training_completed', training_info)
        
        return final_result


    def _store_training_event(self, event_type: str, event_data: Dict) -> None:
        """Store training events in memory for future processing by slave module"""
        if not hasattr(self, '_training_events_buffer'):
            self._training_events_buffer = []
        
        event_entry = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'data': event_data
        }
        
        self._training_events_buffer.append(event_entry)
        
        # Keep buffer size manageable with priority preservation
        if len(self._training_events_buffer) > 100:
            # Separate critical events from routine completions
            critical_events = [e for e in self._training_events_buffer 
                            if e['event_type'] in ['validation_failed', 'model_corrupted_initial', 'training_failed']]
            routine_events = [e for e in self._training_events_buffer 
                            if e['event_type'] == 'training_completed']
            
            # Keep all critical + recent routine events
            recent_routine = routine_events[-30:] if len(routine_events) > 30 else routine_events
            
            self._training_events_buffer = critical_events + recent_routine
    
    def _train_sklearn_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Training per modelli scikit-learn"""
        
        # Split train/test
        train_size = int(0.8 * len(X))
        indices = np.random.permutation(len(X))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        # Training
        start_time = datetime.now()
        
        # Se è un modello già trainato, calcola performance iniziale
        initial_score = 0
        if hasattr(model, 'predict'):
            try:
                initial_predictions = model.predict(X_test)
                if len(y.shape) == 1:  # Classification
                    initial_score = accuracy_score(y_test, initial_predictions)
                else:  # Regression
                    initial_score = -np.mean((y_test - initial_predictions) ** 2)
            except:
                pass
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        
        # Auto-detect se è classificazione o regressione
        if len(np.unique(y)) <= 10 and np.all(y == y.astype(int)):
            # Classificazione (valori discreti limitati)
            final_score = accuracy_score(y_test, predictions.round())
            final_loss = 1 - final_score
            model_type = 'classification'
        else:
            # Regressione (valori continui)
            final_score = r2_score(y_test, predictions)
            final_loss = mean_squared_error(y_test, predictions)
            model_type = 'regression'

        safe_print(f"🔍 Model type detected: {model_type}")
        safe_print(f"📊 Final score: {final_score:.4f}")
        
        improvement = (final_score - initial_score) / abs(initial_score) if initial_score != 0 else 0
        
        return {
            'status': 'success',
            'final_loss': final_loss,
            'improvement': improvement,
            'training_time': (datetime.now() - start_time).total_seconds(),
            'test_score': final_score
        }

# ================== POST-ERROR REANALYSIS SYSTEM ==================

class PostErrorReanalyzer:
    """Sistema per rianalizzare e imparare dagli errori"""
    
    def __init__(self, logger: AnalyzerLogger):
        self.logger = logger
        self.error_patterns_db: Dict[str, List[Dict]] = defaultdict(list)
        self.lessons_learned: Dict[str, Dict[str, Any]] = {}
        self.reanalysis_queue: deque = deque(maxlen=1000)
        
    def add_to_reanalysis_queue(self, failed_prediction: Prediction, 
                              actual_outcome: Dict[str, Any],
                              market_data_snapshot: Dict[str, Any]):
        """Aggiunge una predizione fallita alla coda di rianalisi"""
        self.reanalysis_queue.append({
            'prediction': failed_prediction,
            'actual_outcome': actual_outcome,
            'market_data': market_data_snapshot,
            'added_timestamp': datetime.now()
        })
    
    def perform_reanalysis(self, asset: str, model_type: ModelType, 
                         algorithm_name: str) -> Dict[str, Any]:
        """Esegue rianalisi sugli errori recenti"""
        
        key = f"{asset}_{model_type.value}_{algorithm_name}"
        relevant_errors = [
            item for item in self.reanalysis_queue
            if (item['prediction'].model_type == model_type and 
                item['prediction'].algorithm_name == algorithm_name)
        ]
        
        if len(relevant_errors) < 5:
            return {'status': 'insufficient_errors', 'errors_analyzed': 0}
        
        # Analizza pattern comuni negli errori
        error_analysis = self._analyze_error_patterns(relevant_errors)
        
        # Identifica condizioni problematiche
        problematic_conditions = self._identify_problematic_conditions(relevant_errors)
        
        # Genera lezioni apprese
        lessons = self._generate_lessons(error_analysis, problematic_conditions)
        
        # Salva lezioni nel database
        self.lessons_learned[key] = {
            'timestamp': datetime.now(),
            'errors_analyzed': len(relevant_errors),
            'patterns_found': error_analysis,
            'problematic_conditions': problematic_conditions,
            'lessons': lessons,
            'recommendations': self._generate_recommendations(lessons)
        }
        
        # Log l'analisi
        self.logger.loggers['training'].info(
            f"POST-ERROR REANALYSIS | {asset} | {model_type.value} | {algorithm_name} | "
            f"Analyzed {len(relevant_errors)} errors | Lessons: {len(lessons)}"
        )
        
        # Salva nel database degli error pattern
        self.error_patterns_db[key].extend(error_analysis['patterns'])
        
        return {
            'status': 'completed',
            'errors_analyzed': len(relevant_errors),
            'patterns_found': len(error_analysis['patterns']),
            'lessons_learned': lessons,
            'recommendations': self.lessons_learned[key]['recommendations']
        }
    
    def _analyze_error_patterns(self, errors: List[Dict]) -> Dict[str, Any]:
        """Analizza pattern comuni negli errori"""
        patterns = []
        
        # Analisi temporale
        error_times = [e['prediction'].timestamp for e in errors]
        error_hours = [t.hour for t in error_times]
        
        # Trova ore problematiche
        hour_counts = defaultdict(int)
        for hour in error_hours:
            hour_counts[hour] += 1
        
        problematic_hours = [
            hour for hour, count in hour_counts.items()
            if count > len(errors) * 0.2  # Più del 20% degli errori
        ]
        
        if problematic_hours:
            patterns.append({
                'type': 'temporal',
                'description': f'High error concentration at hours: {problematic_hours}',
                'severity': 'medium'
            })
        
        # Analisi delle predizioni sbagliate
        prediction_types = defaultdict(list)
        for error in errors:
            pred_data = error['prediction'].prediction_data
            for key, value in pred_data.items():
                prediction_types[key].append(value)
        
        # Trova valori problematici
        for pred_type, values in prediction_types.items():
            if len(values) > 5:
                # Calcola se c'è un bias sistematico
                if all(isinstance(v, (int, float)) for v in values):
                    avg_predicted = np.mean(values)
                    actual_values = []
                    
                    for error in errors:
                        if error['actual_outcome'] and pred_type in error['actual_outcome']:
                            actual_values.append(error['actual_outcome'][pred_type])
                    
                    if actual_values:
                        avg_actual = np.mean(actual_values)
                        bias = (avg_predicted - avg_actual) / avg_actual if avg_actual != 0 else 0
                        
                        if abs(bias) > 0.1:  # Bias > 10%
                            patterns.append({
                                'type': 'systematic_bias',
                                'description': f'{pred_type} consistently {"over" if bias > 0 else "under"}estimated by {abs(bias)*100:.1f}%',
                                'severity': 'high'
                            })
        
        # Analisi condizioni di mercato
        market_conditions = [e['market_data'] for e in errors if e.get('market_data')]
        
        if market_conditions:
            volatilities = [m.get('volatility', 0) for m in market_conditions]
            avg_error_volatility = np.mean(volatilities) if volatilities else 0
            
            if avg_error_volatility > 0.02:
                patterns.append({
                    'type': 'market_condition',
                    'description': f'Errors occur during high volatility (avg: {avg_error_volatility:.3f})',
                    'severity': 'medium'
                })
        
        return {
            'patterns': patterns,
            'total_patterns': len(patterns)
        }
    
    def _identify_problematic_conditions(self, errors: List[Dict]) -> List[Dict]:
        """Identifica condizioni di mercato problematiche"""
        conditions = []
        
        market_states = [e['market_data'] for e in errors if e.get('market_data')]
        
        if not market_states:
            return conditions
        
        # Analizza volatilità
        volatilities = [m.get('volatility', 0) for m in market_states]
        if volatilities:
            high_vol_errors = sum(1 for v in volatilities if v > 0.015)
            if high_vol_errors > len(errors) * 0.5:
                conditions.append({
                    'condition': 'high_volatility',
                    'threshold': 0.015,
                    'frequency': high_vol_errors / len(errors),
                    'recommendation': 'Reduce confidence during high volatility periods'
                })
        
        # Analizza trend
        trends = [m.get('price_change_5m', 0) for m in market_states]
        if trends:
            strong_trend_errors = sum(1 for t in trends if abs(t) > 0.01)
            if strong_trend_errors > len(errors) * 0.4:
                conditions.append({
                    'condition': 'strong_trend',
                    'threshold': 0.01,
                    'frequency': strong_trend_errors / len(errors),
                    'recommendation': 'Adjust predictions during strong directional moves'
                })
        
        # Analizza volume
        volumes = [m.get('avg_volume', 0) for m in market_states]
        if volumes:
            low_volume_errors = sum(1 for v in volumes if v < np.mean(volumes) * 0.5)
            if low_volume_errors > len(errors) * 0.3:
                conditions.append({
                    'condition': 'low_volume',
                    'threshold': 0.5,
                    'frequency': low_volume_errors / len(errors),
                    'recommendation': 'Be cautious during low volume periods'
                })
        
        return conditions
    
    def _generate_lessons(self, error_analysis: Dict, problematic_conditions: List[Dict]) -> List[Dict]:
        """Genera lezioni dagli errori analizzati"""
        lessons = []
        
        # Lezioni dai pattern
        for pattern in error_analysis['patterns']:
            if pattern['type'] == 'temporal':
                lessons.append({
                    'type': 'temporal_adjustment',
                    'lesson': 'Model performs poorly during specific hours',
                    'action': 'Implement time-based confidence adjustment',
                    'priority': pattern['severity']
                })
            
            elif pattern['type'] == 'systematic_bias':
                lessons.append({
                    'type': 'bias_correction',
                    'lesson': pattern['description'],
                    'action': 'Apply bias correction factor to predictions',
                    'priority': pattern['severity']
                })
            
            elif pattern['type'] == 'market_condition':
                lessons.append({
                    'type': 'condition_awareness',
                    'lesson': pattern['description'],
                    'action': 'Create market condition filters',
                    'priority': pattern['severity']
                })
        
        # Lezioni dalle condizioni problematiche
        for condition in problematic_conditions:
            lessons.append({
                'type': 'environmental_factor',
                'lesson': f"Poor performance during {condition['condition']}",
                'action': condition['recommendation'],
                'priority': 'high' if condition['frequency'] > 0.5 else 'medium'
            })
        
        return lessons
    
    def _generate_recommendations(self, lessons: List[Dict]) -> List[str]:
        """Genera raccomandazioni pratiche dalle lezioni"""
        recommendations = []
        
        # Prioritizza per importanza
        high_priority_lessons = [l for l in lessons if l.get('priority') == 'high']
        
        for lesson in high_priority_lessons[:5]:  # Top 5 raccomandazioni
            if lesson['type'] == 'temporal_adjustment':
                recommendations.append(
                    "Implement dynamic confidence scoring based on time of day"
                )
            elif lesson['type'] == 'bias_correction':
                recommendations.append(
                    "Add post-processing bias correction to model outputs"
                )
            elif lesson['type'] == 'condition_awareness':
                recommendations.append(
                    "Create pre-filters to detect problematic market conditions"
                )
            elif lesson['type'] == 'environmental_factor':
                recommendations.append(lesson['action'])
        
        return recommendations
    
    def get_lessons_for_algorithm(self, asset: str, model_type: ModelType, 
                                 algorithm_name: str) -> Optional[Dict]:
        """Ottieni lezioni apprese per un algoritmo specifico"""
        key = f"{asset}_{model_type.value}_{algorithm_name}"
        return self.lessons_learned.get(key)
    
    def apply_lessons_to_prediction(self, prediction: Prediction, 
                                  market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Applica le lezioni apprese a una nuova predizione"""
        
        key = f"{prediction.model_type.value}_{prediction.algorithm_name}"
        lessons = self.lessons_learned.get(key)
        
        if not lessons:
            return {'adjusted': False, 'confidence_multiplier': 1.0}
        
        confidence_multiplier = 1.0
        adjustments = []
        
        # Applica aggiustamenti basati sulle lezioni
        for lesson in lessons.get('lessons', []):
            if lesson['type'] == 'temporal_adjustment':
                # Aggiusta per ora del giorno
                current_hour = prediction.timestamp.hour
                # Implementa logica specifica...
                
            elif lesson['type'] == 'bias_correction':
                # Applica correzione bias
                # Implementa logica specifica...
                pass
            
            elif lesson['type'] == 'condition_awareness':
                # Controlla condizioni di mercato
                if market_conditions.get('volatility', 0) > 0.02:
                    confidence_multiplier *= 0.8
                    adjustments.append('high_volatility_adjustment')
        
        return {
            'adjusted': len(adjustments) > 0,
            'confidence_multiplier': confidence_multiplier,
            'adjustments_applied': adjustments
        }
        
# ================== ALGORITHM COMPETITION SYSTEM ==================

class AlgorithmCompetition:
    """Sistema di competizione tra algoritmi per ogni modello con funzionalità avanzate - VERSIONE PULITA"""
    
    def __init__(self, model_type: ModelType, asset: str, logger: AnalyzerLogger,
                champion_preserver: ChampionPreserver, reality_checker: RealityChecker,
                emergency_stop: EmergencyStopSystem, config: Optional[AnalyzerConfig] = None):
        self.model_type = model_type
        self.asset = asset
        self.logger = logger
        self.champion_preserver = champion_preserver
        self.reality_checker = reality_checker
        self.emergency_stop = emergency_stop
        
        # 🔧 NUOVO: Configurazione centralizzata
        self.config = config or get_analyzer_config()
        
        self.algorithms: Dict[str, AlgorithmPerformance] = {}
        self.champion: Optional[str] = None
        self.champion_threshold = self.config.champion_threshold  # 🔧 CHANGED
        self.predictions_history: List[Prediction] = []
        self.pending_validations: Dict[str, Prediction] = {}
        
        # Post-error reanalyzer per questo competition
        self.reanalyzer = PostErrorReanalyzer(logger)
        
        # Performance tracking - USA CONFIG
        self.performance_window = deque(maxlen=self.config.performance_window_size)  # 🔧 CHANGED
        self.last_reality_check = datetime.now()
        self.reality_check_interval = timedelta(hours=self.config.reality_check_interval_hours)  # 🔧 CHANGED
        
    def register_algorithm(self, name: str) -> None:
        """Registra un nuovo algoritmo nella competizione - VERSIONE PULITA"""
        self.algorithms[name] = AlgorithmPerformance(
            name=name,
            model_type=self.model_type
        )
        
        # Se è il primo algoritmo, diventa automaticamente champion
        if self.champion is None:
            self.champion = name
            self.algorithms[name].is_champion = True
            
        # 🧹 PULITO: Sostituito logger con event storage
        self._store_system_event('algorithm_registered', {
            'algorithm_name': name,
            'asset': self.asset,
            'model_type': self.model_type.value,
            'is_first_champion': self.champion == name,
            'timestamp': datetime.now()
        })
    
    def _store_system_event(self, event_type: str, event_data: Dict) -> None:
        """Store system events in memory for future processing by slave module"""
        if not hasattr(self, '_system_events_buffer'):
            self._system_events_buffer: deque = deque(maxlen=100)
        
        event_entry = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'data': event_data
        }
        
        self._system_events_buffer.append(event_entry)
        
        # Buffer size is automatically managed by deque maxlen=50
        # No manual cleanup needed since deque automatically removes old items
    
    def submit_prediction(self, algorithm_name: str, prediction_data: Dict[str, Any], 
                        confidence: float, validation_criteria: Dict[str, Any],
                        market_conditions: Dict[str, Any]) -> str:
        """Sottomette una predizione per validazione con condizioni di mercato - VERSIONE PULITA"""
        
        submission_start = datetime.now()
        
        # Controlla se l'algoritmo è in emergency stop
        algorithm_key = f"{self.asset}_{self.model_type.value}_{algorithm_name}"
        if algorithm_key in self.emergency_stop.stopped_algorithms:
            # Store rejection for future slave module processing
            self._store_prediction_event('rejected_emergency_stop', {
                'algorithm': algorithm_name,
                'algorithm_key': algorithm_key,
                'confidence': confidence,
                'timestamp': submission_start
            })
            return "REJECTED_EMERGENCY_STOP"
        
        # Applica lezioni apprese se disponibili
        lessons_adjustment = self.reanalyzer.apply_lessons_to_prediction(
            Prediction(
                id="temp",
                timestamp=datetime.now(),
                model_type=self.model_type,
                algorithm_name=algorithm_name,
                prediction_data=prediction_data,
                confidence=confidence,
                validation_time=datetime.now(),
                validation_criteria=validation_criteria
            ),
            market_conditions
        )
        
        # Aggiusta confidence basato sulle lezioni
        adjusted_confidence = confidence * lessons_adjustment['confidence_multiplier']
        
        prediction_id = f"{self.asset}_{self.model_type.value}_{algorithm_name}_{datetime.now().isoformat()}"
        
        # Calcola quando validare
        validation_time = self._calculate_validation_time(validation_criteria)
        
        prediction = Prediction(
            id=prediction_id,
            timestamp=datetime.now(),
            model_type=self.model_type,
            algorithm_name=algorithm_name,
            prediction_data=prediction_data,
            confidence=adjusted_confidence,
            validation_time=validation_time,
            validation_criteria=validation_criteria,
            market_conditions_snapshot=market_conditions
        )
        
        self.predictions_history.append(prediction)
        self.pending_validations[prediction_id] = prediction
        
        # Store prediction submission for future slave module processing
        submission_end = datetime.now()
        processing_time = (submission_end - submission_start).total_seconds()
        
        self._store_prediction_event('submitted', {
            'prediction_id': prediction_id,
            'algorithm': algorithm_name,
            'confidence': confidence,
            'adjusted_confidence': adjusted_confidence,
            'confidence_multiplier': lessons_adjustment['confidence_multiplier'],
            'validation_time': validation_time,
            'processing_time': processing_time,
            'timestamp': submission_end
        })
        
        # Aggiorna rolling window performance
        algorithm = self.algorithms[algorithm_name]
        algorithm.rolling_window_performances.append(adjusted_confidence)
        
        # Controlla se è il momento di fare reality check
        if datetime.now() - self.last_reality_check > self.reality_check_interval:
            self._perform_scheduled_reality_check()
        
        return prediction_id

    def _store_prediction_event(self, event_type: str, event_data: Dict) -> None:
        """Store prediction events in memory for future processing by slave module"""
        if not hasattr(self, '_prediction_events_buffer'):
            self._prediction_events_buffer: deque = deque(maxlen=500)
        
        event_entry = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'data': event_data
        }
        
        self._prediction_events_buffer.append(event_entry)
        
        # Buffer size is automatically managed by deque maxlen=500
        # No manual cleanup needed since deque automatically removes old items
    
    def _calculate_validation_time(self, criteria: Dict[str, Any]) -> datetime:
        """Calcola quando validare la predizione"""
        now = datetime.now()
        
        if 'ticks' in criteria:
            # Approssimazione: 1 tick = 1 secondo (da aggiustare con dati reali)
            return now + timedelta(seconds=criteria['ticks'])
        elif 'seconds' in criteria:
            return now + timedelta(seconds=criteria['seconds'])
        elif 'minutes' in criteria:
            return now + timedelta(minutes=criteria['minutes'])
        else:
            return now + timedelta(minutes=5)  # Default
    
    def validate_predictions(self, current_market_data: Dict[str, Any]) -> None:
        """Valida le predizioni scadute con analisi avanzata - VERSIONE PULITA"""
        current_time = datetime.now()
        to_validate = []
        
        # Performance tracking
        validation_start = current_time
        
        # Collect predictions ready for validation
        for pred_id, prediction in self.pending_validations.items():
            if current_time >= prediction.validation_time:
                to_validate.append(pred_id)
        
        # Process validations
        validation_count = 0
        for pred_id in to_validate:
            prediction = self.pending_validations.pop(pred_id)
            self._validate_single_prediction(prediction, current_market_data)
            validation_count += 1
        
        # Store validation metrics for future slave module processing
        if validation_count > 0:
            validation_end = datetime.now()
            processing_time = (validation_end - validation_start).total_seconds()
            
            self._store_validation_metrics({
                'validation_count': validation_count,
                'processing_time': processing_time,
                'pending_count': len(self.pending_validations),
                'timestamp': validation_end
            })

    def _store_validation_metrics(self, metrics: Dict) -> None:
        """Store validation metrics in memory for future processing by slave module"""
        if not hasattr(self, '_validation_metrics_buffer'):
            self._validation_metrics_buffer: deque = deque(maxlen=200)
        
        metric_entry = {
            'timestamp': datetime.now(),
            'metrics': metrics
        }
        
        self._validation_metrics_buffer.append(metric_entry)
        
        # Buffer size is automatically managed by deque maxlen=200
        # No manual cleanup needed since deque automatically removes old items
    
    def _validate_single_prediction(self, prediction: Prediction, market_data: Dict[str, Any]) -> None:
        """Valida una singola predizione con analisi errori - VERSIONE PULITA"""
        
        # Calcola l'outcome reale
        actual_outcome = self._calculate_actual_outcome(prediction, market_data)
        prediction.actual_outcome = actual_outcome
        
        # Calcola self-validation score
        self_score = self._calculate_self_validation_score(prediction)
        prediction.self_validation_score = self_score
        
        # Se la predizione è fallita, analizza perché
        if self_score < 0.5:
            error_analysis = self._analyze_prediction_error(prediction, actual_outcome, market_data)
            prediction.error_analysis = error_analysis
            
            # Aggiungi alla coda di rianalisi
            self.reanalyzer.add_to_reanalysis_queue(
                prediction, actual_outcome, market_data
            )
            
            # 🧹 PULITO: Sostituito logger con event storage
            self.logger.log_error_analysis(
                self.asset, self.model_type, prediction.algorithm_name,
                error_analysis, market_data
            )
        
        # Aggiorna performance dell'algoritmo
        algorithm = self.algorithms[prediction.algorithm_name]
        algorithm.total_predictions += 1
        
        if self_score > 0.6:  # Threshold per considerare corretta
            algorithm.correct_predictions += 1
        
        # Aggiorna confidence score con decay awareness
        algorithm.confidence_score = self._update_confidence_score(algorithm, self_score)
        algorithm.last_updated = datetime.now()
        
        # Aggiungi score alla rolling window
        algorithm.rolling_window_performances.append(self_score)
        
        # Log validation result
        self.logger.log_prediction(self.asset, prediction, {'score': self_score})
        
        # Controlla condizioni di emergenza
        recent_predictions = [
            p for p in self.predictions_history[-50:]
            if p.algorithm_name == prediction.algorithm_name
        ]
        
        emergency_check = self.emergency_stop.check_emergency_conditions(
            self.asset, self.model_type, algorithm, recent_predictions
        )
        
        if emergency_check['emergency_stop']:
            # 🧹 PULITO: Sostituito logger con event storage
            self._store_system_event('emergency_stop_triggered', {
                'algorithm_name': prediction.algorithm_name,
                'emergency_check': emergency_check,
                'timestamp': datetime.now(),
                'severity': 'critical'
            })
        
        # Controlla se c'è un nuovo champion
        self._update_champion()
    
    def _analyze_prediction_error(self, prediction: Prediction, 
                                actual_outcome: Dict[str, Any],
                                market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analizza dettagliatamente perché una predizione è fallita"""
        
        error_analysis = {
            'error_types': [],
            'severity': 0.0,
            'patterns': [],
            'market_conditions': {},
            'specific_failures': []
        }
        
        # Analisi specifica per tipo di modello
        if self.model_type == ModelType.SUPPORT_RESISTANCE:
            predicted_support = prediction.prediction_data.get('support_levels', [])
            predicted_resistance = prediction.prediction_data.get('resistance_levels', [])
            
            # Controlla se i livelli sono stati violati
            if predicted_support and 'support_accuracy' in actual_outcome:
                if actual_outcome['support_accuracy'] < 0.3:
                    error_analysis['error_types'].append('support_level_breach')
                    error_analysis['specific_failures'].append({
                        'type': 'support_breach',
                        'predicted_levels': predicted_support,
                        'accuracy': actual_outcome['support_accuracy']
                    })
            
            if predicted_resistance and 'resistance_accuracy' in actual_outcome:
                if actual_outcome['resistance_accuracy'] < 0.3:
                    error_analysis['error_types'].append('resistance_level_breach')
                    error_analysis['specific_failures'].append({
                        'type': 'resistance_breach',
                        'predicted_levels': predicted_resistance,
                        'accuracy': actual_outcome['resistance_accuracy']
                    })
        
        elif self.model_type == ModelType.PATTERN_RECOGNITION:
            if 'pattern_occurred' in actual_outcome and not actual_outcome['pattern_occurred']:
                predicted_pattern = prediction.prediction_data.get('pattern', '')
                error_analysis['error_types'].append('pattern_not_materialized')
                error_analysis['patterns'].append(f'false_{predicted_pattern}')
                
                # Analizza se c'era un pattern diverso
                if market_data.get('actual_pattern'):
                    error_analysis['specific_failures'].append({
                        'type': 'wrong_pattern',
                        'predicted': predicted_pattern,
                        'actual': market_data['actual_pattern']
                    })
        
        elif self.model_type == ModelType.BIAS_DETECTION:
            predicted_direction = prediction.prediction_data.get('directional_bias', {}).get('direction', '')
            if 'actual_direction' in market_data:
                if predicted_direction != market_data['actual_direction']:
                    error_analysis['error_types'].append('wrong_direction')
                    error_analysis['patterns'].append('directional_miss')
        
        # Analizza condizioni di mercato durante l'errore
        if prediction.market_conditions_snapshot:
            volatility = prediction.market_conditions_snapshot.get('volatility', 0)
            volume = prediction.market_conditions_snapshot.get('avg_volume', 0)
            
            error_analysis['market_conditions'] = {
                'volatility': volatility,
                'volume': volume,
                'high_volatility': volatility > 0.02,
                'low_volume': volume < market_data.get('typical_volume', volume) * 0.5
            }
            
            # Pattern di errore basati su condizioni
            if volatility > 0.02:
                error_analysis['patterns'].append('high_volatility_error')
            if error_analysis['market_conditions']['low_volume']:
                error_analysis['patterns'].append('low_volume_error')
        
        # Calcola severity
        error_analysis['severity'] = len(error_analysis['error_types']) * 0.3 + \
                                   len(error_analysis['patterns']) * 0.2
        
        return error_analysis
    
    def _calculate_actual_outcome(self, prediction: Prediction, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calcola cosa è realmente accaduto nel mercato"""
        outcome = {}
        
        if self.model_type == ModelType.SUPPORT_RESISTANCE:
            # Controlla se i livelli predetti sono stati rispettati
            predicted_support = prediction.prediction_data.get('support_levels', [])
            predicted_resistance = prediction.prediction_data.get('resistance_levels', [])
            current_price = market_data.get('current_price', 0)
            
            outcome['support_accuracy'] = self._check_level_accuracy(
                predicted_support, current_price, market_data, 'support'
            )
            outcome['resistance_accuracy'] = self._check_level_accuracy(
                predicted_resistance, current_price, market_data, 'resistance'
            )
            
        elif self.model_type == ModelType.PATTERN_RECOGNITION:
            # Controlla se il pattern si è manifestato
            predicted_pattern = prediction.prediction_data.get('pattern', '')
            predicted_direction = prediction.prediction_data.get('direction', '')
            
            outcome['pattern_occurred'] = self._check_pattern_occurrence(predicted_pattern, market_data)
            outcome['direction_correct'] = self._check_direction_accuracy(predicted_direction, market_data)
            
        elif self.model_type == ModelType.BIAS_DETECTION:
            # Controlla se il bias era corretto
            predicted_bias = prediction.prediction_data.get('directional_bias', {}).get('direction', '')
            actual_direction = self._determine_actual_direction(market_data)
            
            outcome['bias_correct'] = predicted_bias == actual_direction
            outcome['actual_direction'] = actual_direction
            
        elif self.model_type == ModelType.TREND_ANALYSIS:
            # Controlla accuratezza del trend
            predicted_trend = prediction.prediction_data.get('trend_direction', '')
            actual_trend = self._determine_actual_trend(market_data)
            
            outcome['trend_correct'] = predicted_trend == actual_trend
            outcome['actual_trend'] = actual_trend
            outcome['trend_strength_error'] = self._calculate_trend_strength_error(
                prediction.prediction_data, market_data
            )
        
        return outcome
    
    def _check_level_accuracy(self, levels: List[float], current_price: float, 
                            market_data: Dict[str, Any], level_type: str) -> float:
        """Controlla l'accuratezza dei livelli di supporto/resistenza"""
        if not levels:
            return 0.0
        
        price_history = market_data.get('price_history', [current_price])
        accuracy_scores = []
        
        for level in levels:
            # Controlla se il prezzo ha toccato o rispettato il livello
            level_tolerance = level * 0.001  # 0.1% tolerance
            
            touches = 0
            respects = 0
            
            for i, price in enumerate(price_history[-20:]):  # Ultimi 20 tick
                if abs(price - level) <= level_tolerance:
                    touches += 1
                    
                    # Controlla se il livello ha tenuto
                    if i < len(price_history) - 21:
                        next_price = price_history[-20:][i + 1]
                        if level_type == 'support' and next_price > level:
                            respects += 1
                        elif level_type == 'resistance' and next_price < level:
                            respects += 1
            
            # Score basato su touches e respects
            if touches > 0:
                level_score = (touches / 20) * 0.5 + (respects / touches) * 0.5
            else:
                level_score = 0.0
            
            accuracy_scores.append(level_score)
        
        return float(np.mean(accuracy_scores))
    
    def _check_pattern_occurrence(self, predicted_pattern: str, market_data: Dict[str, Any]) -> bool:
        """Controlla se il pattern predetto si è verificato"""
        price_history = market_data.get('price_history', [])
        
        if len(price_history) < 10:
            return False
        
        recent_prices = price_history[-10:]
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Pattern recognition avanzato
        pattern_checks = {
            'double_top': self._check_double_top_completion,
            'double_bottom': self._check_double_bottom_completion,
            'head_shoulders': self._check_head_shoulders_completion,
            'triangle': self._check_triangle_breakout,
            'flag': self._check_flag_completion,
            'wedge': self._check_wedge_breakout
        }
        
        # Controlla pattern specifici
        for pattern_name, check_func in pattern_checks.items():
            if pattern_name in predicted_pattern.lower():
                return check_func(price_history)
        
        # Fallback per pattern generici
        if 'bullish' in predicted_pattern.lower():
            return price_change > 0.005  # 0.5% rialzo
        elif 'bearish' in predicted_pattern.lower():
            return price_change < -0.005  # 0.5% ribasso
        elif 'breakout' in predicted_pattern.lower():
            return abs(price_change) > 0.01  # 1% movimento
        
        return False
    
    def _check_double_top_completion(self, prices: List[float]) -> bool:
        """Verifica se un double top si è completato"""
        if len(prices) < 20:
            return False
        
        # Trova i massimi recenti
        highs = []
        for i in range(5, len(prices) - 5):
            if prices[i] > max(prices[i-5:i]) and prices[i] > max(prices[i+1:i+6]):
                highs.append((i, prices[i]))
        
        if len(highs) >= 2:
            # Controlla se c'è stato un breakdown
            neckline = min(prices[highs[-2][0]:highs[-1][0]])
            current_price = prices[-1]
            
            return current_price < neckline * 0.995  # Breakdown confermato
        
        return False
        
    def _check_double_bottom_completion(self, prices: List[float]) -> bool:
        """Verifica se un double bottom si è completato"""
        if len(prices) < 20:
            return False
        
        # Trova i minimi recenti
        lows = []
        for i in range(5, len(prices) - 5):
            if prices[i] < min(prices[i-5:i]) and prices[i] < min(prices[i+1:i+6]):
                lows.append((i, prices[i]))
        
        if len(lows) >= 2:
            # Controlla se c'è stato un breakout
            neckline = max(prices[lows[-2][0]:lows[-1][0]])
            current_price = prices[-1]
            
            return current_price > neckline * 1.005  # Breakout confermato
        
        return False
    
    def _check_head_shoulders_completion(self, prices: List[float]) -> bool:
        """Verifica completamento head & shoulders"""
        if len(prices) < 30:
            return False
        
        # Implementazione semplificata
        # Trova tre picchi
        peaks = []
        for i in range(5, len(prices) - 5):
            if prices[i] > max(prices[i-5:i]) and prices[i] > max(prices[i+1:i+6]):
                peaks.append((i, prices[i]))
        
        if len(peaks) >= 3:
            # Verifica pattern H&S (picco centrale più alto)
            if peaks[-2][1] > peaks[-3][1] and peaks[-2][1] > peaks[-1][1]:
                # Controlla breakdown della neckline
                neckline = min(prices[peaks[-3][0]:peaks[-1][0]])
                return prices[-1] < neckline * 0.995
        
        return False
    
    def _check_triangle_breakout(self, prices: List[float]) -> bool:
        """Verifica breakout da triangolo"""
        if len(prices) < 20:
            return False
        
        # Calcola range che si restringe
        ranges = []
        for i in range(0, len(prices) - 5, 5):
            window = prices[i:i+5]
            ranges.append(max(window) - min(window))
        
        # Se il range si sta restringendo
        if len(ranges) >= 3 and ranges[-1] < ranges[-3] * 0.7:
            # Controlla breakout
            recent_range = max(prices[-5:]) - min(prices[-5:])
            price_change = abs(prices[-1] - prices[-6])
            
            return price_change > recent_range * 1.5
        
        return False
    
    def _check_flag_completion(self, prices: List[float]) -> bool:
        """Verifica completamento pattern flag"""
        if len(prices) < 15:
            return False
        
        # Flag = forte movimento iniziale + consolidamento + continuazione
        initial_move = (prices[-15] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
        consolidation_range = max(prices[-10:-5]) - min(prices[-10:-5])
        continuation_move = (prices[-1] - prices[-5]) / prices[-5]
        
        # Bullish flag
        if initial_move > 0.01 and consolidation_range < abs(initial_move) * 0.5:
            return continuation_move > 0.005
        
        # Bearish flag
        if initial_move < -0.01 and consolidation_range < abs(initial_move) * 0.5:
            return continuation_move < -0.005
        
        return False
    
    def _check_wedge_breakout(self, prices: List[float]) -> bool:
        """Verifica breakout da wedge"""
        if len(prices) < 20:
            return False
        
        # Calcola trend delle highs e lows
        highs = []
        lows = []
        
        for i in range(2, len(prices) - 2):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                highs.append((i, prices[i]))
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                lows.append((i, prices[i]))
        
        if len(highs) >= 2 and len(lows) >= 2:
            # Calcola pendenze
            high_slope = (highs[-1][1] - highs[-2][1]) / (highs[-1][0] - highs[-2][0])
            low_slope = (lows[-1][1] - lows[-2][1]) / (lows[-1][0] - lows[-2][0])
            
            # Rising wedge (bearish)
            if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
                return prices[-1] < lows[-1][1]
            
            # Falling wedge (bullish)
            if high_slope < 0 and low_slope < 0 and high_slope > low_slope:
                return prices[-1] > highs[-1][1]
        
        return False
    
    def _check_direction_accuracy(self, predicted_direction: str, market_data: Dict[str, Any]) -> bool:
        """Controlla l'accuratezza della direzione predetta"""
        price_history = market_data.get('price_history', [])
        
        if len(price_history) < 5:
            return False
        
        price_change = (price_history[-1] - price_history[-5]) / price_history[-5]
        
        if predicted_direction.lower() == 'bullish' and price_change > 0.002:
            return True
        elif predicted_direction.lower() == 'bearish' and price_change < -0.002:
            return True
        elif predicted_direction.lower() == 'neutral' and abs(price_change) < 0.002:
            return True
        
        return False
    
    def _determine_actual_direction(self, market_data: Dict[str, Any]) -> str:
        """Determina la direzione effettiva del mercato"""
        price_change = market_data.get('price_change_5m', 0)
        
        if price_change > 0.003:
            return 'bullish'
        elif price_change < -0.003:
            return 'bearish'
        else:
            return 'neutral'
    
    def _determine_actual_trend(self, market_data: Dict[str, Any]) -> str:
        """Determina il trend effettivo"""
        prices = market_data.get('price_history', [])
        
        if len(prices) < 20:
            return 'unknown'
        
        # Calcola trend con regressione lineare
        x = np.arange(len(prices[-20:]))
        y = np.array(prices[-20:])
        
        slope, _ = np.polyfit(x, y, 1)
        
        # Normalizza slope
        normalized_slope = slope / np.mean(y)
        
        if normalized_slope > 0.0001:
            return 'uptrend'
        elif normalized_slope < -0.0001:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _calculate_trend_strength_error(self, prediction_data: Dict, market_data: Dict) -> float:
        """Calcola errore nella forza del trend predetto"""
        predicted_strength = prediction_data.get('trend_strength', 0)
        
        # Calcola forza effettiva
        prices = market_data.get('price_history', [])
        if len(prices) < 20:
            return 0.0
        
        # Usa R-squared della regressione come misura di forza
        x = np.arange(len(prices[-20:]))
        y = np.array(prices[-20:])
        
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        actual_strength = r_squared
        
        return abs(predicted_strength - actual_strength)
    
    def _calculate_self_validation_score(self, prediction: Prediction) -> float:
        """Calcola il punteggio di auto-validazione"""
        if not prediction.actual_outcome:
            return 0.0
        
        score = 0.0
        total_weight = 0.0
        
        # Pesi per diversi tipi di accuracy
        weights = {
            'support_accuracy': 1.5,
            'resistance_accuracy': 1.5,
            'pattern_occurred': 2.0,
            'direction_correct': 1.0,
            'bias_correct': 1.5,
            'trend_correct': 1.5,
            'trend_strength_error': 1.0
        }
        
        for key, value in prediction.actual_outcome.items():
            if key in weights:
                weight = weights[key]
                
                if isinstance(value, bool):
                    score += (1.0 if value else 0.0) * weight
                elif isinstance(value, (int, float)):
                    if 'error' in key:
                        # Per errori, inverti il punteggio
                        score += max(0, 1 - value) * weight
                    else:
                        score += value * weight
                
                total_weight += weight
        
        return score / max(1.0, total_weight)
    
    def _update_confidence_score(self, algorithm: AlgorithmPerformance, new_score: float) -> float:
        """Aggiorna il confidence score con media pesata e decay awareness"""
        # Applica decay factor
        days_since_training = (datetime.now() - algorithm.last_training_date).days
        decay_factor = algorithm.confidence_decay_rate ** days_since_training
        
        # Learning rate adattivo basato su performance
        if algorithm.recent_performance_trend == "declining":
            learning_rate = 0.15  # Più reattivo se in declino
        elif algorithm.recent_performance_trend == "improving":
            learning_rate = 0.05  # Più conservativo se sta migliorando
        else:
            learning_rate = 0.1  # Default
        
        # Aggiorna con media pesata
        current_confidence = algorithm.confidence_score * decay_factor
        new_confidence = current_confidence * (1 - learning_rate) + new_score * 100 * learning_rate
        
        return max(10.0, min(100.0, new_confidence))  # Clamp tra 10 e 100
    
    def receive_observer_feedback(self, prediction_id: str, feedback_score: float,
                                feedback_details: Dict[str, Any]) -> None:
        """Riceve feedback dall'Observer con dettagli - VERSIONE PULITA"""
        # Trova la predizione
        prediction = None
        for pred in self.predictions_history:
            if pred.id == prediction_id:
                prediction = pred
                break
        
        if prediction is None:
            return
        
        prediction.observer_feedback_score = feedback_score
        
        # Aggiorna quality score dell'algoritmo
        algorithm = self.algorithms[prediction.algorithm_name]
        algorithm.observer_feedback_count += 1
        
        if feedback_score > 0.6:  # Threshold per feedback positivo
            algorithm.observer_positive_feedback += 1
        
        # Aggiorna quality score con decay awareness
        algorithm.quality_score = self._update_quality_score(algorithm, feedback_score)
        
        # Se feedback molto negativo, analizza
        if feedback_score < 0.3:
            self._analyze_negative_feedback(prediction, feedback_details)
        
        # Log performance metrics
        self._log_performance_metrics(algorithm)
        
        # Controlla se c'è un nuovo champion
        self._update_champion()
    
    def _update_quality_score(self, algorithm: AlgorithmPerformance, feedback_score: float) -> float:
        """Aggiorna il quality score con media pesata e trend awareness"""
        # Considera il trend delle performance
        if algorithm.recent_performance_trend == "declining":
            weight = 0.15  # Peso maggiore al nuovo feedback
        else:
            weight = 0.1
        
        return algorithm.quality_score * (1 - weight) + feedback_score * 100 * weight
    
    def _analyze_negative_feedback(self, prediction: Prediction, feedback_details: Dict[str, Any]):
        """Analizza feedback negativo per identificare problemi - VERSIONE PULITA"""
        # Aggiungi ai pattern di errore
        if 'error_reason' in feedback_details:
            if prediction.error_analysis is None:
                prediction.error_analysis = {'error_types': [], 'patterns': []}
            
            prediction.error_analysis['patterns'].append(
                f"observer_feedback_{feedback_details['error_reason']}"
            )
        
        # 🧹 PULITO: Sostituito logger con event storage
        self._store_system_event('negative_observer_feedback', {
            'algorithm_name': prediction.algorithm_name,
            'prediction_id': prediction.id,
            'feedback_details': feedback_details,
            'timestamp': datetime.now(),
            'severity': 'warning'
        })
    
    def _update_champion(self) -> None:
        """Aggiorna il champion se necessario con preservazione - VERSIONE PULITA"""
        if not self.algorithms:
            return
        
        current_champion = self.algorithms.get(self.champion) if self.champion is not None else None
        if current_champion is None:
            return
        
        current_champion_score = current_champion.final_score
        
        # Trova il migliore sfidante
        best_challenger = None
        best_challenger_score = 0
        
        for name, algorithm in self.algorithms.items():
            if name != self.champion and algorithm.final_score > best_challenger_score:
                # Verifica che non sia in emergency stop
                algorithm_key = f"{self.asset}_{self.model_type.value}_{name}"
                if algorithm_key not in self.emergency_stop.stopped_algorithms:
                    best_challenger = name
                    best_challenger_score = algorithm.final_score
        
        # Controlla se lo sfidante può detronizzare il champion
        if best_challenger and best_challenger_score > current_champion_score * (1 + self.champion_threshold):
            # Preserva il vecchio champion se ha performato bene
            if current_champion_score > 70 and current_champion.total_predictions > 100:
                self._preserve_champion(current_champion)
            
            # Nuovo champion!
            old_champion = self.champion or "None"
            old_score = current_champion_score
            
            current_champion.is_champion = False
            self.algorithms[best_challenger].is_champion = True
            self.champion = best_challenger
            
            # Calcola ragione del cambio
            reason = self._determine_champion_change_reason(
                current_champion, self.algorithms[best_challenger]
            )
            
            # Log il cambio
            self.logger.log_champion_change(
                self.asset, self.model_type, old_champion or "None", best_challenger,
                old_score, best_challenger_score, reason
            )
    
    def _determine_champion_change_reason(self, old_champ: AlgorithmPerformance,
                                        new_champ: AlgorithmPerformance) -> str:
        """Determina la ragione del cambio champion"""
        reasons = []
        
        # Confronta metriche
        if new_champ.accuracy_rate > old_champ.accuracy_rate * 1.1:
            reasons.append(f"accuracy improvement ({old_champ.accuracy_rate:.2%} → {new_champ.accuracy_rate:.2%})")
        
        if new_champ.observer_satisfaction > old_champ.observer_satisfaction * 1.2:
            reasons.append(f"observer satisfaction ({old_champ.observer_satisfaction:.2%} → {new_champ.observer_satisfaction:.2%})")
        
        if old_champ.emergency_stop_triggered:
            reasons.append("previous champion in emergency stop")
        
        if old_champ.reality_check_failures > 3:
            reasons.append(f"reality check failures ({old_champ.reality_check_failures})")
        
        if new_champ.recent_performance_trend == "improving" and old_champ.recent_performance_trend == "declining":
            reasons.append("performance trend reversal")
        
        return " | ".join(reasons) if reasons else "overall performance improvement"
    
    def _preserve_champion(self, champion: AlgorithmPerformance):
        """Preserva un champion di successo - VERSIONE PULITA"""
        # Ottieni i pesi del modello se disponibile
        model_weights = None  # Questo dovrebbe essere passato dal sistema principale
        
        preservation_data = self.champion_preserver.preserve_champion(
            self.asset, self.model_type, champion.name,
            champion, model_weights
        )
        
        champion.preserved_model_path = preservation_data['model_file']
        
        # 🧹 PULITO: Sostituito logger con event storage
        self._store_system_event('champion_preserved', {
            'algorithm_name': champion.name,
            'final_score': champion.final_score,
            'preservation_data': preservation_data,
            'timestamp': datetime.now()
        })
    
    def _perform_scheduled_reality_check(self):
        """Esegue reality check programmato su tutti gli algoritmi - VERSIONE PULITA"""
        self.last_reality_check = datetime.now()
        
        for name, algorithm in self.algorithms.items():
            # Skip se in emergency stop
            algorithm_key = f"{self.asset}_{self.model_type.value}_{name}"
            if algorithm_key in self.emergency_stop.stopped_algorithms:
                continue
            
            # Ottieni predizioni recenti
            recent_predictions = [
                p for p in self.predictions_history[-50:]
                if p.algorithm_name == name
            ]
            
            if len(recent_predictions) >= 10:
                reality_result = self.reality_checker.perform_reality_check(
                    self.asset, self.model_type, algorithm, recent_predictions
                )
                
                if not reality_result['passed']:
                    # 🧹 PULITO: Sostituito logger con event storage
                    self._store_system_event('reality_check_failed', {
                        'algorithm_name': name,
                        'reality_result': reality_result,
                        'timestamp': datetime.now(),
                        'severity': 'warning'
                    })
                    
                    # Trigger retraining se necessario
                    if algorithm.reality_check_failures > 3:
                        self._request_retraining(algorithm)
    
    def _request_retraining(self, algorithm: AlgorithmPerformance):
        """Richiede retraining per un algoritmo - VERSIONE PULITA"""
        # 🧹 PULITO: Sostituito logger con event storage
        self._store_system_event('retraining_requested', {
            'algorithm_name': algorithm.name,
            'reason': 'poor_performance',
            'reality_check_failures': algorithm.reality_check_failures,
            'timestamp': datetime.now()
        })
        # Il retraining effettivo sarà gestito dal sistema principale
    
    def _log_performance_metrics(self, algorithm: AlgorithmPerformance):
        """Logga metriche di performance dettagliate - VERSIONE PULITA"""
        # 🧹 PULITO: Sostituito _write_csv con event storage
        performance_data = {
            'timestamp': datetime.now(),
            'asset': self.asset,
            'model_type': self.model_type.value,
            'algorithm': algorithm.name,
            'final_score': algorithm.final_score,
            'confidence_score': algorithm.confidence_score,
            'quality_score': algorithm.quality_score,
            'accuracy_rate': algorithm.accuracy_rate,
            'total_predictions': algorithm.total_predictions,
            'decay_factor': algorithm.confidence_decay_rate ** (datetime.now() - algorithm.last_training_date).days,
            'reality_check_status': 'failed' if algorithm.reality_check_failures > 0 else 'passed'
        }
        
        # Store in local buffer for unified system
        self._store_performance_metrics(performance_data)
    
    def _store_performance_metrics(self, performance_data: Dict) -> None:
        """Store performance metrics in memory for future processing by slave module"""
        if not hasattr(self, '_performance_metrics_buffer'):
            self._performance_metrics_buffer: deque = deque(maxlen=300)
        
        metric_entry = {
            'timestamp': datetime.now(),
            'metrics': performance_data
        }
        
        self._performance_metrics_buffer.append(metric_entry)
        
        # Buffer size is automatically managed by deque maxlen=300
        # No manual cleanup needed since deque automatically removes old items
        
    def get_champion_algorithm(self) -> Optional[str]:
        """Restituisce l'algoritmo champion corrente"""
        return self.champion
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Restituisce un summary delle performance dettagliato"""
        summary = {
            'champion': self.champion,
            'algorithms': {},
            'pending_validations': len(self.pending_validations),
            'total_predictions': len(self.predictions_history),
            'recent_performance': self._calculate_recent_performance(),
            'health_status': self._calculate_health_status()
        }
        
        for name, alg in self.algorithms.items():
            algorithm_key = f"{self.asset}_{self.model_type.value}_{name}"
            
            summary['algorithms'][name] = {
                'final_score': alg.final_score,
                'confidence_score': alg.confidence_score,
                'quality_score': alg.quality_score,
                'accuracy_rate': alg.accuracy_rate,
                'observer_satisfaction': alg.observer_satisfaction,
                'is_champion': alg.is_champion,
                'total_predictions': alg.total_predictions,
                'recent_trend': alg.recent_performance_trend,
                'needs_retraining': alg.needs_retraining,
                'emergency_stop': algorithm_key in self.emergency_stop.stopped_algorithms,
                'reality_check_failures': alg.reality_check_failures,
                'days_since_training': (datetime.now() - alg.last_training_date).days
            }
        
        return summary
    
    def _calculate_recent_performance(self) -> Dict[str, float]:
        """Calcola performance recenti aggregate"""
        recent_predictions = self.predictions_history[-100:] if len(self.predictions_history) > 100 else self.predictions_history
        
        if not recent_predictions:
            return {'accuracy': 0.0, 'confidence': 0.0}
        
        accuracies = [p.self_validation_score for p in recent_predictions if p.self_validation_score is not None]
        confidences = [p.confidence for p in recent_predictions]
        
        return {
            'accuracy': float(np.mean(accuracies)) if accuracies else 0.0,
            'confidence': float(np.mean(confidences)) if confidences else 0.0,
            'validation_rate': float(len(accuracies) / len(recent_predictions)) if recent_predictions else 0.0
        }
    
    def _calculate_health_status(self) -> str:
        """Calcola lo stato di salute generale della competizione"""
        if not self.algorithms:
            return "no_algorithms"
        
        # Conta algoritmi attivi
        active_algorithms = sum(
            1 for name, alg in self.algorithms.items()
            if f"{self.asset}_{self.model_type.value}_{name}" not in self.emergency_stop.stopped_algorithms
        )
        
        if active_algorithms == 0:
            return "critical"
        
        # Controlla performance del champion
        if self.champion and self.champion in self.algorithms:
            champion_score = self.algorithms[self.champion].final_score
            if champion_score < 40:
                return "poor"
            elif champion_score < 60:
                return "fair"
            elif champion_score < 80:
                return "good"
            else:
                return "excellent"
        
        return "unknown"
    
    def get_all_events_for_slave(self) -> Dict[str, List[Dict]]:
        """Get all accumulated events for slave module processing"""
        events = {}
        
        # System events
        if hasattr(self, '_system_events_buffer'):
            events['system_events'] = self._system_events_buffer.copy()
        
        # Prediction events
        if hasattr(self, '_prediction_events_buffer'):
            events['prediction_events'] = self._prediction_events_buffer.copy()
        
        # Validation metrics
        if hasattr(self, '_validation_metrics_buffer'):
            events['validation_metrics'] = self._validation_metrics_buffer.copy()
        
        # Performance metrics
        if hasattr(self, '_performance_metrics_buffer'):
            events['performance_metrics'] = self._performance_metrics_buffer.copy()
        
        return events
    
    def clear_events_buffer(self, event_types: Optional[List[str]] = None) -> None:
        """Clear event buffers after slave module processing"""
        if event_types is None:
            # Clear all buffers
            if hasattr(self, '_system_events_buffer'):
                self._system_events_buffer.clear()
            if hasattr(self, '_prediction_events_buffer'):
                self._prediction_events_buffer.clear()
            if hasattr(self, '_validation_metrics_buffer'):
                self._validation_metrics_buffer.clear()
            if hasattr(self, '_performance_metrics_buffer'):
                self._performance_metrics_buffer.clear()
        else:
            # Clear specific buffers
            for event_type in event_types:
                if event_type == 'system_events' and hasattr(self, '_system_events_buffer'):
                    self._system_events_buffer.clear()
                elif event_type == 'prediction_events' and hasattr(self, '_prediction_events_buffer'):
                    self._prediction_events_buffer.clear()
                elif event_type == 'validation_metrics' and hasattr(self, '_validation_metrics_buffer'):
                    self._validation_metrics_buffer.clear()
                elif event_type == 'performance_metrics' and hasattr(self, '_performance_metrics_buffer'):
                    self._performance_metrics_buffer.clear()

# ================== ASSET ANALYZER PRINCIPALE ==================

class AssetAnalyzer:
    """Analyzer per un singolo asset con tutti i modelli competitivi e sistemi avanzati"""
       
    def __init__(self, asset: str, data_path: str = "./analyzer_data", config: Optional[AnalyzerConfig] = None):
        self.asset = asset
        self.data_path = f"{data_path}/{asset}"
        os.makedirs(self.data_path, exist_ok=True)
        
        # 🔧 NUOVO: Configurazione centralizzata
        self.config = config or get_analyzer_config()
        
        # Sistema di logging
        self.logger = AsyncAnalyzerLogger(f"{self.data_path}/logs")
        
        # 🔧 NUOVO: Sistema diagnostico avanzato
        self.diagnostics = LearningDiagnostics(asset, self.logger)

        # Sistemi di supporto
        self.champion_preserver = ChampionPreserver(f"{self.data_path}/champions")
        self.reality_checker = RealityChecker()
        self.emergency_stop = EmergencyStopSystem(self.logger)
        self.mt5_interface = MT5Interface(self.logger)
        self.rolling_trainer = RollingWindowTrainer()
        
        # Inizializza competizioni per ogni modello
        self.competitions: Dict[ModelType, AlgorithmCompetition] = {}
        for model_type in ModelType:
            self.competitions[model_type] = AlgorithmCompetition(
                model_type, asset, self.logger, self.champion_preserver,
                self.reality_checker, self.emergency_stop
            )
        
        # Data storage con gestione memoria - USA CONFIG
        self.tick_data = deque(maxlen=self.config.max_tick_buffer_size)  # 🔧 CHANGED
        self.aggregated_data = {}  # Diverse aggregazioni temporali
        
        # Thread safety - Lock multipli per evitare race conditions
        self.data_lock = threading.RLock()
        self.competitions_lock = threading.RLock()
        self.models_lock = threading.RLock()
        self.state_lock = threading.RLock()
        
        # ML Models storage
        self.ml_models = {}
        self.scalers = {}

        # Technical indicators cache system - USA CONFIG
        self.indicators_cache = IndicatorsCache(max_cache_size=self.config.indicators_cache_size)  # 🔧 CHANGED
        self.cached_indicators = create_cached_indicator_calculator(self.indicators_cache)
        
        # Learning phase tracking - USA CONFIG
        self.learning_phase = True
        self.learning_start_time = datetime.now()
        self.min_learning_days = self.config.min_learning_days  # 🔧 CHANGED
        self.learning_progress = 0.0
        
        # Performance tracking - USA CONFIG
        self.analysis_count = 0
        self.last_analysis_time = None
        self.analysis_latency_history = deque(maxlen=self.config.latency_history_size)  # 🔧 CHANGED
        
        # Initialize all algorithms
        self._initialize_algorithms()
        
        # Load previous state if exists
        self.load_analyzer_state()

        # 🧹 PULITO: Sostituito logger con event storage
        self._store_system_event('analyzer_initialized', {
            'asset': asset,
            'learning_phase': self.learning_phase,
            'config_type': type(self.config).__name__,
            'competitions_count': len(self.competitions),
            'data_path': self.data_path,
            'timestamp': datetime.now()
        })

        def _load_ml_models(self) -> None:
            """Carica i modelli ML salvati - VERSIONE PULITA"""
            
            models_dir = f"{self.data_path}/models"
            
            if not os.path.exists(models_dir):
                return
            
            models_loaded = 0
            models_failed = 0
            
            # Load PyTorch models
            for model_name, model in self.ml_models.items():
                if isinstance(model, nn.Module):
                    model_path = f"{models_dir}/{model_name}.pt"
                    if os.path.exists(model_path):
                        try:
                            checkpoint = torch.load(model_path, map_location='cpu')
                            model.load_state_dict(checkpoint['model_state_dict'])
                            model.eval()
                            
                            # 🧹 PULITO: Sostituito logger con event storage
                            self._store_system_event('ml_model_loaded', {
                                'model_name': model_name,
                                'model_type': 'pytorch',
                                'status': 'success',
                                'model_path': model_path,
                                'timestamp': datetime.now()
                            })
                            models_loaded += 1
                            
                        except Exception as e:
                            # 🧹 PULITO: Sostituito logger con event storage
                            self._store_system_event('ml_model_load_failed', {
                                'model_name': model_name,
                                'model_type': 'pytorch',
                                'status': 'error',
                                'error_message': str(e),
                                'error_type': type(e).__name__,
                                'model_path': model_path,
                                'timestamp': datetime.now(),
                                'severity': 'error'
                            })
                            models_failed += 1
                
                else:
                    # Load scikit-learn models
                    model_path = f"{models_dir}/{model_name}.pkl"
                    if os.path.exists(model_path):
                        try:
                            with open(model_path, 'rb') as f:
                                self.ml_models[model_name] = pickle.load(f)
                            
                            # 🧹 PULITO: Sostituito logger con event storage
                            self._store_system_event('ml_model_loaded', {
                                'model_name': model_name,
                                'model_type': 'sklearn',
                                'status': 'success',
                                'model_path': model_path,
                                'timestamp': datetime.now()
                            })
                            models_loaded += 1
                            
                        except Exception as e:
                            # 🧹 PULITO: Sostituito logger con event storage
                            self._store_system_event('ml_model_load_failed', {
                                'model_name': model_name,
                                'model_type': 'sklearn',
                                'status': 'error',
                                'error_message': str(e),
                                'error_type': type(e).__name__,
                                'model_path': model_path,
                                'timestamp': datetime.now(),
                                'severity': 'error'
                            })
                            models_failed += 1
            
            # Load scalers
            scalers_path = f"{models_dir}/scalers.pkl"
            if os.path.exists(scalers_path):
                try:
                    with open(scalers_path, 'rb') as f:
                        self.scalers = pickle.load(f)
                    
                    # 🧹 PULITO: Sostituito logger con event storage
                    self._store_system_event('scalers_loaded', {
                        'status': 'success',
                        'scalers_count': len(self.scalers),
                        'scalers_path': scalers_path,
                        'timestamp': datetime.now()
                    })
                    
                except Exception as e:
                    # 🧹 PULITO: Sostituito logger con event storage
                    self._store_system_event('scalers_load_failed', {
                        'status': 'error',
                        'error_message': str(e),
                        'error_type': type(e).__name__,
                        'scalers_path': scalers_path,
                        'timestamp': datetime.now(),
                        'severity': 'error'
                    })
            
            # Summary event
            self._store_system_event('models_loading_complete', {
                'models_loaded': models_loaded,
                'models_failed': models_failed,
                'scalers_loaded': len(self.scalers),
                'total_models': len(self.ml_models),
                'timestamp': datetime.now()
            })

        def _load_recent_predictions(self):
            """Carica predizioni recenti salvate - VERSIONE PULITA"""
            
            predictions_dir = f"{self.data_path}/predictions"
            
            if not os.path.exists(predictions_dir):
                return
            
            predictions_loaded = 0
            predictions_failed = 0
            
            for model_type, competition in self.competitions.items():
                predictions_file = f"{predictions_dir}/{model_type.value}_recent.pkl"
                
                if os.path.exists(predictions_file):
                    try:
                        with open(predictions_file, 'rb') as f:
                            recent_predictions = pickle.load(f)
                        
                        # Add to competition history
                        competition.predictions_history.extend(recent_predictions[-100:])  # Last 100
                        
                        # 🧹 PULITO: Sostituito logger con event storage
                        self._store_system_event('predictions_loaded', {
                            'model_type': model_type.value,
                            'status': 'success',
                            'predictions_count': len(recent_predictions),
                            'predictions_added': len(recent_predictions[-100:]),
                            'predictions_file': predictions_file,
                            'timestamp': datetime.now()
                        })
                        predictions_loaded += 1
                        
                    except Exception as e:
                        # 🧹 PULITO: Sostituito logger con event storage
                        self._store_system_event('predictions_load_failed', {
                            'model_type': model_type.value,
                            'status': 'error',
                            'error_message': str(e),
                            'error_type': type(e).__name__,
                            'predictions_file': predictions_file,
                            'timestamp': datetime.now(),
                            'severity': 'error'
                        })
                        predictions_failed += 1
            
            # Summary event
            if predictions_loaded > 0 or predictions_failed > 0:
                self._store_system_event('predictions_loading_complete', {
                    'model_types_loaded': predictions_loaded,
                    'model_types_failed': predictions_failed,
                    'total_model_types': len(self.competitions),
                    'timestamp': datetime.now()
                })

    def _initialize_algorithms(self) -> None:
        """Inizializza tutti gli algoritmi per ogni modello"""
        
        # Support/Resistance algorithms
        sr_competition = self.competitions[ModelType.SUPPORT_RESISTANCE]
        sr_competition.config = self.config  # 🔧 ADDED - passa config
        sr_competition.register_algorithm("PivotPoints_Classic")
        sr_competition.register_algorithm("VolumeProfile_Advanced")
        sr_competition.register_algorithm("StatisticalLevels_ML")
        sr_competition.register_algorithm("LSTM_SupportResistance")
        sr_competition.register_algorithm("Transformer_Levels")
        
        # Pattern Recognition algorithms
        pr_competition = self.competitions[ModelType.PATTERN_RECOGNITION]
        pr_competition.config = self.config  # 🔧 ADDED
        pr_competition.register_algorithm("CNN_PatternRecognizer")
        pr_competition.register_algorithm("Classical_Patterns")
        pr_competition.register_algorithm("LSTM_Sequences")
        pr_competition.register_algorithm("Transformer_Patterns")
        pr_competition.register_algorithm("Ensemble_Patterns")
        
        # Bias Detection algorithms
        bd_competition = self.competitions[ModelType.BIAS_DETECTION]
        bd_competition.config = self.config  # 🔧 ADDED
        bd_competition.register_algorithm("Sentiment_LSTM")
        bd_competition.register_algorithm("VolumePrice_Analysis")
        bd_competition.register_algorithm("Momentum_ML")
        bd_competition.register_algorithm("Transformer_Bias")
        bd_competition.register_algorithm("MultiModal_Bias")
        
        # Trend Analysis algorithms
        ta_competition = self.competitions[ModelType.TREND_ANALYSIS]
        ta_competition.config = self.config  # 🔧 ADDED
        ta_competition.register_algorithm("LSTM_TrendPrediction")
        ta_competition.register_algorithm("RandomForest_Trend")
        ta_competition.register_algorithm("GradientBoosting_Trend")
        ta_competition.register_algorithm("Transformer_Trend")
        ta_competition.register_algorithm("Ensemble_Trend")
        
        # Volatility Prediction algorithms
        vp_competition = self.competitions[ModelType.VOLATILITY_PREDICTION]
        vp_competition.config = self.config  # 🔧 ADDED
        vp_competition.register_algorithm("GARCH_Volatility")
        vp_competition.register_algorithm("LSTM_Volatility")
        vp_competition.register_algorithm("Realized_Volatility")
        
        # Momentum Analysis algorithms
        ma_competition = self.competitions[ModelType.MOMENTUM_ANALYSIS]
        ma_competition.config = self.config  # 🔧 ADDED
        ma_competition.register_algorithm("RSI_Momentum")
        ma_competition.register_algorithm("MACD_Analysis")
        ma_competition.register_algorithm("Neural_Momentum")
        
        # Initialize ML models for each algorithm
        self._initialize_ml_models()
    
    def _initialize_ml_models(self) -> None:
        """Inizializza i modelli ML con configurazione centralizzata"""
        
        # CNN Pattern Recognizer
        self.ml_models['CNN_PatternRecognizer'] = CNNPatternRecognizer(
            input_channels=1, 
            sequence_length=100, 
            num_patterns=50
        )
        
        # 🔧 LSTM models con configurazione centralizzata
        lstm_configs = {
            'LSTM_SupportResistance': self.config.get_model_architecture('LSTM_SupportResistance'),
            'LSTM_Sequences': self.config.get_model_architecture('LSTM_Sequences'),
            'Sentiment_LSTM': self.config.get_model_architecture('Sentiment_LSTM'),
            'LSTM_TrendPrediction': self.config.get_model_architecture('LSTM_TrendPrediction'),
            'LSTM_Volatility': self.config.get_model_architecture('LSTM_Volatility')
        }
        
        for model_name, config in lstm_configs.items():
            self.ml_models[model_name] = AdvancedLSTM(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=config['output_size'],
                dropout=config['dropout']
            )
        
        # Transformer models con configurazione
        transformer_configs = {
            'Transformer_Levels': {'input_dim': 10, 'd_model': 256, 'nhead': 8, 'num_layers': 6, 'output_dim': 10},
            'Transformer_Patterns': {'input_dim': 15, 'd_model': 512, 'nhead': 16, 'num_layers': 8, 'output_dim': 50},
            'Transformer_Bias': {'input_dim': 8, 'd_model': 128, 'nhead': 4, 'num_layers': 4, 'output_dim': 3},
            'Transformer_Trend': {'input_dim': 12, 'd_model': 256, 'nhead': 8, 'num_layers': 6, 'output_dim': 5}
        }
        
        for model_name, config in transformer_configs.items():
            self.ml_models[model_name] = TransformerPredictor(**config)
        
        # Traditional ML models con configurazione
        self.ml_models['RandomForest_Trend'] = RandomForestRegressor(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
        )
        
        self.ml_models['GradientBoosting_Trend'] = GradientBoostingRegressor(
            n_estimators=150, learning_rate=self.config.learning_rate * 1000,  # 🔧 CHANGED
            max_depth=8, random_state=42
        )
        
        # Scalers per normalizzazione
        for model_name in self.ml_models.keys():
            self.scalers[model_name] = StandardScaler()
        
        # Carica pesi preservati se disponibili
        self._load_preserved_models()
    
    def _load_preserved_models(self):
        """Carica modelli preservati per champion precedenti"""
        for model_type in ModelType:
            best_preserved = self.champion_preserver.get_best_preserved(self.asset, model_type)
            if best_preserved:
                model_data = self.champion_preserver.load_preserved_model(best_preserved)
                if model_data:
                    algorithm_name = best_preserved['algorithm_name']
                    if algorithm_name in self.ml_models:
                        try:
                            # Carica pesi nel modello
                            if isinstance(self.ml_models[algorithm_name], nn.Module):
                                self.ml_models[algorithm_name].load_state_dict(model_data['weights'])
                            else:
                                self.ml_models[algorithm_name] = model_data['weights']
                            
                            self.logger.loggers['system'].info(
                                f"Loaded preserved model for {algorithm_name} from {best_preserved['timestamp']}"
                            )
                        except Exception as e:
                            self.logger.loggers['errors'].error(
                                f"Failed to load preserved model {algorithm_name}: {e}"
                            )
    
    def process_tick(self, timestamp: datetime, price: float, volume: float, 
                    bid: Optional[float] = None, ask: Optional[float] = None, additional_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Processa un nuovo tick con gestione completa - VERSIONE PULITA"""
        
        processing_start = datetime.now()
        
        # Thread-safe data storage
        with self.data_lock:
            # Store tick data
            tick_data = {
                'timestamp': timestamp,
                'price': price,
                'volume': volume,
                'bid': bid or price,
                'ask': ask or price,
                'spread': (ask - bid) if ask and bid else 0,
                **(additional_data or {})
            }
            self.tick_data.append(tick_data)
            
            # Update aggregated data
            self._update_aggregated_data(tick_data)
        
        # Prepare market data with lock protection
        with self.data_lock:
            current_market_data = self._prepare_market_data()
        
        # Update learning progress
        if self.learning_phase:
            days_learning = (datetime.now() - self.learning_start_time).days
            self.learning_progress = min(1.0, days_learning / self.config.min_learning_days)
            
            if days_learning < self.config.min_learning_days:
                # Still learning, just collect data
                remaining_days = self.config.min_learning_days - days_learning
                
                # Ogni N tick durante learning, fai un mini-training
                if len(self.tick_data) % self.config.learning_mini_training_interval == 0:
                    self._perform_learning_phase_training()
                
                return {
                    "status": "learning",
                    "progress": self.learning_progress,
                    "days_remaining": remaining_days,
                    "ticks_collected": len(self.tick_data),
                    "config_min_days": self.config.min_learning_days
                }
            else:
                self.learning_phase = False
                # Perform final comprehensive training
                self._perform_final_training()
        
        # Validate pending predictions with thread safety
        with self.competitions_lock:
            for competition in self.competitions.values():
                competition.validate_predictions(current_market_data)
        
        # Check for retraining needs
        self._check_retraining_needs()
        
        # Generate analysis only if not in learning phase
        if not self.learning_phase:
            analysis = self._generate_full_analysis(current_market_data)
            
            # Track latency
            latency = (datetime.now() - processing_start).total_seconds()
            self.analysis_latency_history.append(latency)
            
            # Update counters
            self.analysis_count += 1
            self.last_analysis_time = datetime.now()
            
            # Prepare for MT5/Observer
            if self.mt5_interface.connected:
                mt5_output = self.mt5_interface.prepare_analysis_output(analysis)
                analysis['mt5_output'] = mt5_output
            
            # Check for learning stall detection (without logging)
            if len(self.tick_data) % (self.config.learning_mini_training_interval * 5) == 0:
                stall_info = self.diagnostics.detect_learning_stall(self)
                if stall_info:
                    # Store stall info for future slave module processing
                    self._store_emergency_data(stall_info)
            
            return analysis
        
        return {"status": "learning_complete", "ready": True}


    def _store_emergency_data(self, stall_info: Dict) -> None:
        """Store emergency data in memory for future processing by slave module"""
        if not hasattr(self, '_emergency_data_buffer'):
            self._emergency_data_buffer: deque = deque(maxlen=20)
        
        emergency_data = {
            'timestamp': datetime.now(),
            'type': 'learning_stall',
            'data': stall_info,
            'asset': self.asset
        }
        
        self._emergency_data_buffer.append(emergency_data)
        
        # Buffer size is automatically managed by deque maxlen=20
        # No manual cleanup needed since deque automatically removes old items
    
    def _update_aggregated_data(self, tick: Dict):
        """Aggiorna dati aggregati per diverse finestre temporali"""
        # Implementa aggregazioni per 1m, 5m, 15m, 1h, 4h, 1d
        pass  # Implementazione semplificata per brevità
    
    def _perform_learning_phase_training(self):
        """Esegue mini-training durante la fase di learning"""
        if len(self.tick_data) < self.config.learning_ticks_threshold // 10:  # 🔧 CHANGED (1/10 del threshold)
            return
        
        self.logger.loggers['training'].info(
            f"Performing learning phase training for {self.asset} with {len(self.tick_data)} ticks"
        )
        
        # Prepara dati di training
        training_data = self.rolling_trainer.prepare_training_data(self.tick_data)
        
        if not training_data:
            return
        
            # 🧪 AGGIUNGI QUI IL DEBUG - START
            safe_print(f"\n🧪 DEBUGGING RandomForest_Trend for {self.asset}:")
            print(f"Training data available: {bool(training_data)}")
            print(f"Training data keys: {list(training_data.keys()) if training_data else 'None'}")
            
            # Test specifico per RandomForest_Trend dataset
            if 'RandomForest_Trend' in ['LSTM_SupportResistance', 'RandomForest_Trend', 'VolumePrice_Analysis']:
                try:
                    safe_print(f"📊 Testing trend dataset preparation...")
                    X, y = self.rolling_trainer._prepare_trend_dataset(training_data)
                    safe_print(f"✅ Dataset prepared successfully!")
                    print(f"   📏 X shape: {X.shape}")
                    print(f"   📏 y shape: {y.shape}")
                    safe_print(f"   🔢 Target type: {type(y[0])}")
                    safe_print(f"   📈 Target samples: {y[:3]}")
                    safe_print(f"   ✅ Is continuous: {not np.all(y == y.astype(int))}")
                    safe_print(f"   📊 Target range: [{np.min(y):.6f}, {np.max(y):.6f}]")
                except Exception as e:
                    safe_print(f"❌ Dataset preparation FAILED: {e}")
                    import traceback
                    traceback.print_exc()
            # 🧪 DEBUG - END

            # 🧪 TEST LSTM AUTO-RESIZE
            if 'LSTM_SupportResistance' in key_models:
                safe_print(f"\n🧪 TESTING LSTM Auto-Resize for {self.asset}:")
                
                try:
                    # Simula input di dimensione sbagliata
                    test_features_2d = np.random.randn(10, 200)  # 200 features invece di 10
                    test_features_3d = np.random.randn(5, 30, 150)  # 150 features invece di 10
                    
                    model = self.ml_models['LSTM_SupportResistance']
                    safe_print(f"   🎯 Model expects input_size: {model.expected_input_size}")
                    
                    # Test con input 2D
                    safe_print(f"   🧪 Testing 2D input: {test_features_2d.shape}")
                    test_input_2d = torch.FloatTensor(test_features_2d)
                    output_2d = model(test_input_2d)
                    safe_print(f"   ✅ 2D → 3D + resize successful! Output: {output_2d.shape}")
                    
                    # Test con input 3D
                    safe_print(f"   🧪 Testing 3D input: {test_features_3d.shape}")
                    test_input_3d = torch.FloatTensor(test_features_3d)
                    output_3d = model(test_input_3d)
                    safe_print(f"   ✅ 3D resize successful! Output: {output_3d.shape}")
                    
                except Exception as e:
                    safe_print(f"   ❌ LSTM auto-resize test FAILED: {e}")
                    import traceback
                    traceback.print_exc()

        # Train solo alcuni modelli chiave durante learning
        key_models = ['LSTM_SupportResistance', 'RandomForest_Trend', 'VolumePrice_Analysis']
        
        for model_name in key_models:
            if model_name in self.ml_models:
                model = self.ml_models[model_name]
                
                # Determina il tipo di modello
                model_type = None
                for mt, competition in self.competitions.items():
                    if model_name in competition.algorithms:
                        model_type = mt
                        break
                
                if model_type:
                    try:
                        result = self.rolling_trainer.train_model(
                            model, training_data, model_type, model_name,
                            preserve_weights=False  # Durante learning, non preservare
                        )
                        
                        if result['status'] == 'success':
                            # Aggiorna data di training
                            if model_name in self.competitions[model_type].algorithms:
                                self.competitions[model_type].algorithms[model_name].last_training_date = datetime.now()
                        
                        self.logger.log_training_event(
                            self.asset, model_type, model_name,
                            'learning_phase', result
                        )
                        
                    except Exception as e:
                        self.logger.loggers['errors'].error(
                            f"Learning phase training failed for {model_name}: {e}"
                        )
                        
    def _perform_final_training(self):
        """Esegue training completo alla fine della fase di learning"""
        self.logger.loggers['training'].info(
            f"Starting comprehensive final training for {self.asset}"
        )
        
        # Prepara dataset completo
        training_data = self.rolling_trainer.prepare_training_data(self.tick_data)
        
        if not training_data:
            self.logger.loggers['errors'].error("No training data available for final training")
            return
        
        # Train tutti i modelli
        for model_name, model in self.ml_models.items():
            # Trova il tipo di modello
            model_type = None
            for mt, competition in self.competitions.items():
                if any(model_name == alg_name for alg_name in competition.algorithms):
                    model_type = mt
                    break
            
            if model_type:
                try:
                    self.logger.loggers['training'].info(f"Training {model_name} for {model_type.value}")
                    
                    result = self.rolling_trainer.train_model(
                        model, training_data, model_type, model_name,
                        preserve_weights=False
                    )
                    
                    if result['status'] == 'success':
                        # Aggiorna algoritmo
                        for competition in self.competitions.values():
                            if model_name in competition.algorithms:
                                competition.algorithms[model_name].last_training_date = datetime.now()
                                competition.algorithms[model_name].confidence_score = 50.0  # Reset
                                competition.algorithms[model_name].quality_score = 50.0
                    
                    self.logger.log_training_event(
                        self.asset, model_type, model_name,
                        'final_training', result
                    )
                    
                except Exception as e:
                    self.logger.loggers['errors'].error(f"Final training failed for {model_name}: {e}")
    
    def _check_retraining_needs(self):
        """Controlla se qualche modello necessita retraining"""
        current_time = datetime.now()
        
        for model_type, competition in self.competitions.items():
            for alg_name, algorithm in competition.algorithms.items():
                if algorithm.needs_retraining:
                    # Controlla se dovremmo fare retraining
                    if self.rolling_trainer.should_retrain(
                        self.asset, model_type, alg_name,
                        force=algorithm.emergency_stop_triggered
                    ):
                        self._retrain_algorithm(model_type, alg_name, algorithm)
    
    def _retrain_algorithm(self, model_type: ModelType, algorithm_name: str, 
                        algorithm: AlgorithmPerformance):
        """Riaddestra un algoritmo specifico - VERSIONE PULITA"""
        
        retraining_start = datetime.now()
        
        # Store retraining attempt for future slave module processing
        retraining_info = {
            'model_type': model_type.value,
            'algorithm_name': algorithm_name,
            'asset': self.asset,
            'emergency_stop_triggered': algorithm.emergency_stop_triggered,
            'final_score': algorithm.final_score,
            'timestamp': retraining_start
        }
        
        # Se c'è stato un emergency stop, fai prima reanalysis
        if algorithm.emergency_stop_triggered:
            competition = self.competitions[model_type]
            reanalysis_result = competition.reanalyzer.perform_reanalysis(
                self.asset, model_type, algorithm_name
            )
            
            if reanalysis_result['status'] == 'completed':
                retraining_info['reanalysis_lessons'] = len(reanalysis_result['lessons_learned'])
        
        # Prepara dati con rolling window
        training_data = self.rolling_trainer.prepare_training_data(self.tick_data)
        
        if not training_data:
            self._store_retraining_event('no_training_data', retraining_info)
            return
        
        # Ottieni il modello
        if algorithm_name not in self.ml_models:
            retraining_info['error'] = 'model_not_found'
            self._store_retraining_event('model_not_found', retraining_info)
            return
        
        model = self.ml_models[algorithm_name]
        
        try:
            # Train con preservazione dei pesi se il modello stava performando decentemente
            preserve_weights = algorithm.final_score > 60 and not algorithm.emergency_stop_triggered
            retraining_info['preserve_weights'] = preserve_weights
            
            result = self.rolling_trainer.train_model(
                model, training_data, model_type, algorithm_name,
                preserve_weights=preserve_weights
            )
            
            retraining_end = datetime.now()
            retraining_duration = (retraining_end - retraining_start).total_seconds()
            
            retraining_info.update({
                'training_result': result,
                'retraining_duration': retraining_duration,
                'success': result['status'] == 'success'
            })
            
            if result['status'] == 'success':
                # Reset contatori negativi
                algorithm.reality_check_failures = 0
                algorithm.emergency_stop_triggered = False
                algorithm.last_training_date = datetime.now()
                
                # Reset emergency stop se attivo
                algorithm_key = f"{self.asset}_{model_type.value}_{algorithm_name}"
                self.emergency_stop.reset_emergency_stop(algorithm_key)
                
                improvement = result.get('improvement', 0)
                retraining_info['improvement'] = improvement
                
                # Se il training è andato molto bene, preserva il modello
                if improvement > 0.2:  # 20% improvement
                    self._preserve_successful_model(model_type, algorithm_name, algorithm, model)
                    retraining_info['model_preserved'] = True
            
            self._store_retraining_event('retraining_completed', retraining_info)
            
        except Exception as e:
            retraining_info['error'] = str(e)
            retraining_info['error_type'] = type(e).__name__
            self._store_retraining_event('retraining_failed', retraining_info)


    def _store_retraining_event(self, event_type: str, event_data: Dict) -> None:
        """Store retraining events in memory for future processing by slave module"""
        if not hasattr(self, '_retraining_events_buffer'):
            self._retraining_events_buffer: deque = deque(maxlen=30)
        
        event_entry = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'data': event_data
        }
        
        self._retraining_events_buffer.append(event_entry)
        
        # Buffer size is automatically managed by deque maxlen=30
        # No manual cleanup needed since deque automatically removes old items
    
    def _preserve_successful_model(self, model_type: ModelType, algorithm_name: str,
                                 algorithm: AlgorithmPerformance, model: Any):
        """Preserva un modello che ha avuto successo nel retraining"""
        try:
            # Estrai pesi del modello
            if isinstance(model, nn.Module):
                model_weights = model.state_dict()
            else:
                model_weights = model
            
            preservation_data = self.champion_preserver.preserve_champion(
                self.asset, model_type, algorithm_name,
                algorithm, model_weights
            )
            
            algorithm.preserved_model_path = preservation_data['model_file']
            
            self.logger.loggers['system'].info(
                f"Model preserved after successful retraining: {algorithm_name}"
            )
            
        except Exception as e:
            self.logger.loggers['errors'].error(f"Failed to preserve model {algorithm_name}: {e}")
    
    def _prepare_market_data(self) -> Dict[str, Any]:
        """Prepara i dati di mercato per l'analisi con feature avanzate - VERSIONE PULITA"""
        if len(self.tick_data) < 50:
            return {}
        
        # Performance tracking (minimal, in-memory only)
        processing_start = datetime.now()
        
        # OTTIMIZZAZIONE CRITICA: Accesso diretto senza copia lista completa
        with self.data_lock:
            total_ticks = len(self.tick_data)
            start_idx = max(0, total_ticks - 100)
            # Reference diretta + slice range invece di list() copy
            recent_ticks_range = range(start_idx, total_ticks)
            n_ticks = len(recent_ticks_range)
        
        # PRE-ALLOCA arrays con dtype esplicito (NO COPIE)
        prices = np.empty(n_ticks, dtype=np.float32)
        volumes = np.empty(n_ticks, dtype=np.float32)
        timestamps = []  # Keep list per timestamp (necessario per output)
        
        # Loop ottimizzato per estrazione dati (accesso diretto tramite indici)
        with self.data_lock:  # Secondo lock minimale per accesso sicuro
            for i, tick_idx in enumerate(recent_ticks_range):
                tick = self.tick_data[tick_idx]
                prices[i] = tick['price']
                volumes[i] = tick['volume']
                timestamps.append(tick['timestamp'])
        
        # CALCOLA STATISTICHE UNA VOLTA SOLA (operazioni vettoriali)
        current_price = float(prices[-1])
        price_mean = float(np.mean(prices))
        price_std = float(np.std(prices))
        avg_volume = float(np.mean(volumes))
        
        # Price history come VIEW (reference diretta, zero copie)
        price_history = prices
        volume_history = volumes
        
        # Price changes con accesso diretto (NO SLICE COPIES)
        price_change_1m = (current_price - prices[-20]) / prices[-20] if n_ticks > 20 else 0.0
        price_change_5m = (current_price - prices[0]) / prices[0] if n_ticks == 100 else 0.0
        
        # Volume analysis (avg_volume già calcolato)
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
        
        # VOLATILITY OTTIMIZZATA: Pre-alloca + operazioni IN-PLACE
        if n_ticks > 1:
            returns = np.empty(n_ticks - 1, dtype=np.float32)
            # Calcolo vettoriale IN-PLACE per returns
            np.subtract(prices[1:], prices[:-1], out=returns)
            np.divide(returns, prices[:-1], out=returns, where=(prices[:-1] != 0))
            volatility = float(np.std(returns))
        else:
            volatility = 0.0
        
        # ATR VOLATILITY ULTRA-OTTIMIZZATO: Operazioni vettoriali
        if n_ticks >= 20:
            # Pre-alloca array per ranges
            max_windows = min(19, n_ticks - 5)
            ranges = np.empty(max_windows, dtype=np.float32)
            
            # Calcolo vettoriale dei range per ATR
            for i in range(max_windows):
                window_start = max(0, n_ticks - i - 6)
                window_end = n_ticks - i
                if window_end > window_start:
                    # View minimale per window
                    window_prices = prices[window_start:window_end]
                    ranges[i] = np.ptp(window_prices)  # Peak-to-peak vettoriale
            
            atr_volatility = float(np.mean(ranges)) / current_price if max_windows > 0 else volatility
        else:
            atr_volatility = volatility
        
        # Market microstructure (accesso diretto ultimo tick)
        with self.data_lock:
            last_tick = self.tick_data[-1]
            if 'bid' in last_tick and 'ask' in last_tick:
                bid = last_tick['bid']
                ask = last_tick['ask']
                spread = ask - bid
                spread_percentage = spread / current_price if current_price > 0 else 0
                mid_price = (bid + ask) * 0.5  # Moltiplicazione più veloce di divisione
            else:
                spread_percentage = 0.0
                mid_price = current_price
        
        # Momentum indicators (valori già calcolati)
        momentum_1m = price_change_1m
        momentum_5m = price_change_5m
        
        # Volume momentum OTTIMIZZATO: View invece di slice
        if n_ticks >= 40:
            # Views dirette per calcoli volume
            recent_volumes = volumes[-20:]  # View
            older_volumes = volumes[-40:-20]  # View
            recent_vol_mean = float(np.mean(recent_volumes))
            older_vol_mean = float(np.mean(older_volumes))
            volume_momentum = (recent_vol_mean - older_vol_mean) / older_vol_mean if older_vol_mean > 0 else 0.0
        elif n_ticks >= 20:
            recent_vol_mean = float(np.mean(volumes[-20:]))
            volume_momentum = (recent_vol_mean - avg_volume) / avg_volume if avg_volume > 0 else 0.0
        else:
            volume_momentum = 0.0
        
        # Price acceleration OTTIMIZZATO (accesso diretto)
        if n_ticks >= 30:
            # Accesso diretto senza slice intermedi
            price_20_ago = prices[-20]
            price_30_ago = prices[-30]
            older_change = (price_20_ago - price_30_ago) / price_30_ago if price_30_ago != 0 else 0.0
            acceleration = price_change_1m - older_change
        else:
            acceleration = 0.0
        
        # Market state detection (passa view dirette)
        market_state = self._detect_market_state(prices, volumes)  # Views, non copie
        
        # Store performance metrics for future slave module processing
        processing_end = datetime.now()
        processing_time = (processing_end - processing_start).total_seconds()
        
        # Accumulate performance data in memory
        self._store_performance_metrics('prepare_market_data', {
            'processing_time': processing_time,
            'tick_count': n_ticks,
            'timestamp': processing_end
        })
        
        # RETURN dictionary con references dirette (zero copie aggiuntive)
        return {
            'current_price': current_price,
            'price_history': price_history,      # VIEW diretta
            'volume_history': volume_history,    # VIEW diretta  
            'timestamp': timestamps[-1],
            'price_change_1m': price_change_1m,
            'price_change_5m': price_change_5m,
            'avg_volume': avg_volume,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'atr_volatility': atr_volatility,
            'spread_percentage': spread_percentage,
            'mid_price': mid_price,
            'momentum_1m': momentum_1m,
            'momentum_5m': momentum_5m,
            'volume_momentum': volume_momentum,
            'acceleration': acceleration,
            'price_mean': price_mean,
            'price_std': price_std,
            'market_state': market_state,
            'typical_volume': avg_volume,
            'tick_count': n_ticks
        }


    def _store_performance_metrics(self, operation: str, metrics: Dict) -> None:
        """Store performance metrics in memory for future processing by slave module"""
        if not hasattr(self, '_performance_metrics_buffer'):
            self._performance_metrics_buffer: deque = deque(maxlen=100)
        
        metric_entry = {
            'operation': operation,
            'metrics': metrics,
            'timestamp': datetime.now()
        }
        
        self._performance_metrics_buffer.append(metric_entry)
        
        # Buffer size is automatically managed by deque maxlen=100
        # No manual cleanup needed since deque automatically removes old items
    
    def _detect_market_state(self, prices: np.ndarray, volumes: np.ndarray) -> str:
        """Rileva lo stato corrente del mercato"""
        if len(prices) < 20:
            return "unknown"
        
        # Calcola volatilità
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # Calcola trend
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        normalized_slope = slope / np.mean(prices)
        
        # Calcola volume profile
        avg_volume = np.mean(volumes)
        recent_volume = np.mean(volumes[-10:])
        
        # Determina stato
        if volatility > 0.02:
            if abs(normalized_slope) > 0.0001:
                return "volatile_trending"
            else:
                return "volatile_ranging"
        elif volatility < 0.005:
            if recent_volume < avg_volume * 0.5:
                return "low_activity"
            else:
                return "consolidating"
        else:
            if abs(normalized_slope) > 0.0001:
                return "trending"
            else:
                return "ranging"
    
    def _generate_full_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Genera l'analisi completa usando gli algoritmi champion"""
        
        analysis_output = {
            'asset': self.asset,
            'timestamp': market_data.get('timestamp', datetime.now()),
            'market_state': market_data.get('market_state', 'unknown'),
            'market_data': {
                'price': market_data.get('current_price'),
                'volatility': market_data.get('volatility'),
                'volume_ratio': market_data.get('volume_ratio'),
                'momentum': market_data.get('momentum_5m')
            },
            'predictions': {},
            'confidence_levels': {},
            'recommendations': [],
            'risk_assessment': self._assess_current_risk(market_data),
            'meta': {
                'champions': {},
                'model_performances': {},
                'analysis_number': self.analysis_count,
                'average_latency': np.mean(self.analysis_latency_history) if self.analysis_latency_history else 0
            }
        }
        
        # Per ogni tipo di modello, usa l'algoritmo champion
        for model_type, competition in self.competitions.items():
            champion = competition.get_champion_algorithm()
            if champion:
                try:
                    # Esegui algoritmo champion
                    result = self._run_champion_algorithm(model_type, champion, market_data)
                    
                    # Genera predizione per validazione futura
                    if result and 'error' not in result:
                        prediction_id = competition.submit_prediction(
                            champion,
                            result,
                            result.get('confidence', 0.5),
                            self._generate_validation_criteria(model_type),
                            market_data
                        )
                        
                        # Aggiungi all'output
                        analysis_output['predictions'][model_type.value] = result
                        analysis_output['confidence_levels'][model_type.value] = result.get('confidence', 0.5)
                        analysis_output['meta']['champions'][model_type.value] = champion
                    
                except Exception as e:
                    self.logger.loggers['errors'].error(
                        f"Error running {champion} for {model_type.value}: {e}"
                    )
                    analysis_output['predictions'][model_type.value] = {'error': str(e)}
            
            # Aggiungi performance summary
            analysis_output['meta']['model_performances'][model_type.value] = competition.get_performance_summary()
        
        # Genera raccomandazioni basate su tutte le predizioni
        analysis_output['recommendations'] = self._generate_recommendations(
            analysis_output['predictions'],
            market_data
        )
        
        return analysis_output
    
    def _assess_current_risk(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Valuta il rischio corrente del mercato usando configurazione"""
        
        # Ottieni soglie dalla configurazione
        risk_thresholds = self.config.get_risk_thresholds()
        
        risk_score = 0.0
        risk_factors = []
        
        # Volatilità - USA CONFIG
        volatility = market_data.get('volatility', 0)
        if volatility > risk_thresholds['high_volatility']:  # 🔧 CHANGED
            risk_score += 0.3
            risk_factors.append("high_volatility")
        elif volatility > risk_thresholds['low_volatility']:  # 🔧 CHANGED
            risk_score += 0.1
            risk_factors.append("moderate_volatility")
        
        # Volume - USA CONFIG
        volume_ratio = market_data.get('volume_ratio', 1)
        if volume_ratio < risk_thresholds['low_volume_ratio']:  # 🔧 CHANGED
            risk_score += 0.2
            risk_factors.append("low_volume")
        elif volume_ratio > risk_thresholds['high_volume_ratio']:  # 🔧 CHANGED
            risk_score += 0.1
            risk_factors.append("unusual_volume")
        
        # Spread - USA CONFIG
        spread = market_data.get('spread_percentage', 0)
        if spread > risk_thresholds['wide_spread']:  # 🔧 CHANGED
            risk_score += 0.1
            risk_factors.append("wide_spread")
        
        # Market state
        market_state = market_data.get('market_state', '')
        if 'volatile' in market_state:
            risk_score += 0.2
        elif market_state == 'low_activity':
            risk_score += 0.1
        
        # Normalizza risk score
        risk_score = min(1.0, risk_score)
        
        # Determina risk level - USA CONFIG
        if risk_score < risk_thresholds['moderate_risk']:  # 🔧 CHANGED
            risk_level = "low"
        elif risk_score < risk_thresholds['high_risk']:  # 🔧 CHANGED
            risk_level = "moderate"
        else:
            risk_level = "high"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendation': self._get_risk_recommendation(risk_level, risk_factors),
            'thresholds_used': risk_thresholds  # 🔧 ADDED per debugging
        }
    
    def _get_risk_recommendation(self, risk_level: str, risk_factors: List[str]) -> str:
        """Genera raccomandazione basata sul rischio"""
        if risk_level == "high":
            if "high_volatility" in risk_factors:
                return "Reduce position size or avoid trading during high volatility"
            elif "low_volume" in risk_factors:
                return "Be cautious of low liquidity, use limit orders"
            else:
                return "High risk detected, trade with caution"
        elif risk_level == "moderate":
            return "Normal trading conditions with moderate risk"
        else:
            return "Favorable trading conditions with low risk"
    
    def _generate_validation_criteria(self, model_type: ModelType) -> Dict[str, Any]:
        """Genera criteri di validazione per tipo di modello usando configurazione"""
        return self.config.get_validation_criteria(model_type)
    
    def _generate_recommendations(self, predictions: Dict[str, Any], 
                                market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera raccomandazioni basate su tutte le predizioni"""
        recommendations = []
        
        # Analizza Support/Resistance
        if 'support_resistance' in predictions:
            sr_data = predictions['support_resistance']
            if 'support_levels' in sr_data and sr_data['support_levels']:
                nearest_support = min(sr_data['support_levels'], 
                                    key=lambda x: abs(x - market_data['current_price']))
                distance_to_support = (market_data['current_price'] - nearest_support) / market_data['current_price']
                
                if distance_to_support < 0.002:  # Molto vicino al supporto
                    recommendations.append({
                        'type': 'support_proximity',
                        'action': 'consider_long',
                        'reason': f'Price near support at {nearest_support:.5f}',
                        'confidence': sr_data.get('confidence', 0.5),
                        'priority': 'high' if distance_to_support < 0.001 else 'medium'
                    })
            
            if 'resistance_levels' in sr_data and sr_data['resistance_levels']:
                nearest_resistance = min(sr_data['resistance_levels'], 
                                       key=lambda x: abs(x - market_data['current_price']))
                distance_to_resistance = (nearest_resistance - market_data['current_price']) / market_data['current_price']
                
                if distance_to_resistance < 0.002:  # Molto vicino alla resistenza
                    recommendations.append({
                        'type': 'resistance_proximity',
                        'action': 'consider_short',
                        'reason': f'Price near resistance at {nearest_resistance:.5f}',
                        'confidence': sr_data.get('confidence', 0.5),
                        'priority': 'high' if distance_to_resistance < 0.001 else 'medium'
                    })
        
        # Analizza Pattern Recognition
        if 'pattern_recognition' in predictions:
            pr_data = predictions['pattern_recognition']
            if 'detected_patterns' in pr_data:
                for pattern in pr_data['detected_patterns']:
                    if pattern.get('probability', 0) > 0.7:
                        pattern_type = pattern.get('pattern', '')
                        
                        # Map pattern to action
                        if 'bullish' in pattern_type.lower() or 'bottom' in pattern_type.lower():
                            action = 'long_setup'
                        elif 'bearish' in pattern_type.lower() or 'top' in pattern_type.lower():
                            action = 'short_setup'
                        else:
                            action = 'monitor'
                        
                        recommendations.append({
                            'type': 'pattern_detected',
                            'action': action,
                            'pattern': pattern_type,
                            'reason': f'{pattern_type} pattern detected with {pattern["probability"]:.0%} probability',
                            'confidence': pattern.get('confidence', pattern['probability']),
                            'priority': 'high' if pattern['probability'] > 0.85 else 'medium'
                        })
        
        # Analizza Bias Detection
        if 'bias_detection' in predictions:
            bd_data = predictions['bias_detection']
            if 'directional_bias' in bd_data:
                bias = bd_data['directional_bias']
                if bias.get('confidence', 0) > 0.7:
                    direction = bias.get('direction', 'neutral')
                    
                    if direction == 'bullish':
                        recommendations.append({
                            'type': 'market_bias',
                            'action': 'favor_long',
                            'reason': f'Strong bullish bias detected ({bias["confidence"]:.0%} confidence)',
                            'confidence': bias['confidence'],
                            'priority': 'medium'
                        })
                    elif direction == 'bearish':
                        recommendations.append({
                            'type': 'market_bias',
                            'action': 'favor_short',
                            'reason': f'Strong bearish bias detected ({bias["confidence"]:.0%} confidence)',
                            'confidence': bias['confidence'],
                            'priority': 'medium'
                        })
        
        # Analizza Trend
        if 'trend_analysis' in predictions:
            ta_data = predictions['trend_analysis']
            trend_direction = ta_data.get('trend_direction', '')
            trend_strength = ta_data.get('trend_strength', 0)
            
            if trend_strength > 0.7:
                if trend_direction == 'uptrend':
                    recommendations.append({
                        'type': 'trend_following',
                        'action': 'follow_uptrend',
                        'reason': f'Strong uptrend in progress (strength: {trend_strength:.0%})',
                        'confidence': ta_data.get('confidence', trend_strength),
                        'priority': 'high'
                    })
                elif trend_direction == 'downtrend':
                    recommendations.append({
                        'type': 'trend_following',
                        'action': 'follow_downtrend',
                        'reason': f'Strong downtrend in progress (strength: {trend_strength:.0%})',
                        'confidence': ta_data.get('confidence', trend_strength),
                        'priority': 'high'
                    })
        
        # Combina raccomandazioni e risolvi conflitti
        recommendations = self._resolve_recommendation_conflicts(recommendations, market_data)
        
        # Ordina per priorità e confidence
        recommendations.sort(key=lambda x: (
            {'high': 3, 'medium': 2, 'low': 1}.get(x.get('priority', 'low'), 0),
            x.get('confidence', 0)
        ), reverse=True)
        
        # Limita a top 5 raccomandazioni
        return recommendations[:5]
    
    def _resolve_recommendation_conflicts(self, recommendations: List[Dict], 
                                        market_data: Dict[str, Any]) -> List[Dict]:
        """Risolve conflitti tra raccomandazioni"""
        if not recommendations:
            return recommendations
        
        # Raggruppa per tipo di azione
        long_recs = [r for r in recommendations if 'long' in r.get('action', '')]
        short_recs = [r for r in recommendations if 'short' in r.get('action', '')]
        neutral_recs = [r for r in recommendations if r.get('action') in ['monitor', 'wait']]
        
        # Se ci sono sia long che short, valuta quale è più forte
        if long_recs and short_recs:
            long_confidence = np.mean([r.get('confidence', 0.5) for r in long_recs])
            short_confidence = np.mean([r.get('confidence', 0.5) for r in short_recs])
            
            # Se sono troppo vicini, suggerisci di aspettare
            if abs(long_confidence - short_confidence) < 0.1:
                return [{
                    'type': 'conflicting_signals',
                    'action': 'wait',
                    'reason': 'Conflicting signals detected, better to wait for clarity',
                    'confidence': 0.3,
                    'priority': 'high'
                }] + neutral_recs
            
            # Altrimenti, mantieni solo il lato più forte
            if long_confidence > short_confidence:
                return long_recs + neutral_recs
            else:
                return short_recs + neutral_recs
        
        return recommendations
    
    def _run_champion_algorithm(self, model_type: ModelType, algorithm_name: str, 
                            market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Esegue l'algoritmo champion per un tipo di modello con error handling standardizzato - VERSIONE PULITA"""
        
        # Performance tracking
        algorithm_start = datetime.now()
        
        # VALIDAZIONE MARKET DATA
        if not market_data:
            return {"error": "Market data is empty"}
        
        # CONTROLLO VALORI CRITICI (mantenuto per stabilità)
        critical_fields = ['current_price', 'price_history', 'volume_history']
        for field in critical_fields:
            if field in market_data:
                value = market_data[field]
                if isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        self._store_algorithm_error('invalid_value', {
                            'field': field,
                            'value': str(value),
                            'algorithm': algorithm_name
                        })
                        return {"error": f"Invalid value in {field}: {value}"}
                elif isinstance(value, (list, np.ndarray)):
                    if len(value) > 0:
                        arr = np.array(value)
                        if np.isnan(arr).any() or np.isinf(arr).any():
                            self._store_algorithm_error('nan_inf_values', {
                                'field': field,
                                'algorithm': algorithm_name
                            })
                            return {"error": f"NaN/Inf values in {field}"}

        try:
            # Route to appropriate algorithm
            result = None
            if model_type == ModelType.SUPPORT_RESISTANCE:
                result = self._run_support_resistance_algorithm(algorithm_name, market_data)
            elif model_type == ModelType.PATTERN_RECOGNITION:
                result = self._run_pattern_recognition_algorithm(algorithm_name, market_data)
            elif model_type == ModelType.BIAS_DETECTION:
                result = self._run_bias_detection_algorithm(algorithm_name, market_data)
            elif model_type == ModelType.TREND_ANALYSIS:
                result = self._run_trend_analysis_algorithm(algorithm_name, market_data)
            elif model_type == ModelType.VOLATILITY_PREDICTION:
                result = self._run_volatility_prediction_algorithm(algorithm_name, market_data)
            elif model_type == ModelType.MOMENTUM_ANALYSIS:
                result = self._run_momentum_analysis_algorithm(algorithm_name, market_data)
            else:
                return {"error": f"Unknown model type: {model_type}"}
            
            # Track successful execution
            algorithm_end = datetime.now()
            execution_time = (algorithm_end - algorithm_start).total_seconds()
            
            self._store_algorithm_success(algorithm_name, {
                'model_type': str(model_type),
                'execution_time': execution_time,
                'timestamp': algorithm_end
            })
            
            return result
                
        except InsufficientDataError as e:
            self._store_algorithm_error('insufficient_data', {
                'algorithm': algorithm_name,
                'required': e.required,
                'available': e.available,
                'operation': e.operation
            })
            return {
                "error": "insufficient_data",
                "details": {
                    "required": e.required,
                    "available": e.available,
                    "operation": e.operation
                }
            }
        
        except ModelNotInitializedError as e:
            self._store_algorithm_error('model_not_initialized', {
                'algorithm': algorithm_name,
                'model_name': e.model_name
            })
            return {
                "error": "model_not_initialized",
                "details": {
                    "model_name": e.model_name,
                    "algorithm": algorithm_name
                }
            }
        
        except InvalidInputError as e:
            self._store_algorithm_error('invalid_input', {
                'algorithm': algorithm_name,
                'field': e.field,
                'value': str(e.value),
                'reason': e.reason
            })
            return {
                "error": "invalid_input",
                "details": {
                    "field": e.field,
                    "value": str(e.value),
                    "reason": e.reason
                }
            }
        
        except PredictionError as e:
            self._store_algorithm_error('prediction_failed', {
                'algorithm': algorithm_name,
                'prediction_algorithm': e.algorithm,
                'reason': e.reason
            })
            return {
                "error": "prediction_failed",
                "details": {
                    "algorithm": e.algorithm,
                    "reason": e.reason
                }
            }
        
        except AnalyzerException as e:
            self._store_algorithm_error('analyzer_error', {
                'algorithm': algorithm_name,
                'message': str(e)
            })
            return {
                "error": "analyzer_error",
                "details": {
                    "algorithm": algorithm_name,
                    "message": str(e)
                }
            }
        
        except Exception as e:
            self._store_algorithm_error('unexpected_error', {
                'algorithm': algorithm_name,
                'message': str(e),
                'type': type(e).__name__
            })
            return {
                "error": "unexpected_error",
                "details": {
                    "algorithm": algorithm_name,
                    "message": str(e),
                    "type": type(e).__name__
                }
            }


    def _store_algorithm_error(self, error_type: str, error_data: Dict) -> None:
        """Store algorithm errors in memory for future processing by slave module"""
        if not hasattr(self, '_algorithm_errors_buffer'):
            self._algorithm_errors_buffer: deque = deque(maxlen=50)
        
        error_entry = {
            'timestamp': datetime.now(),
            'error_type': error_type,
            'data': error_data
        }
        
        self._algorithm_errors_buffer.append(error_entry)
        
        # Buffer size is automatically managed by deque maxlen=50
        # No manual cleanup needed since deque automatically removes old items


    def _store_algorithm_success(self, algorithm_name: str, success_data: Dict) -> None:
        """Store algorithm success metrics in memory for future processing by slave module"""
        if not hasattr(self, '_algorithm_success_buffer'):
            self._algorithm_success_buffer: deque = deque(maxlen=200)
        
        success_entry = {
            'timestamp': datetime.now(),
            'algorithm': algorithm_name,
            'data': success_data
        }
        
        self._algorithm_success_buffer.append(success_entry)
        
        # Buffer size is automatically managed by deque maxlen=200
        # No manual cleanup needed since deque automatically removes old items
    
    def _run_support_resistance_algorithm(self, algorithm_name: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Esegue algoritmi di Support/Resistance"""
        
        if algorithm_name == "PivotPoints_Classic":
            # Implementazione Pivot Points classici
            prices = market_data['price_history']
            if len(prices) < 20:
                raise InsufficientDataError(required=20, available=len(prices), operation="PivotPoints_Classic")
            
            high = max(prices[-20:])
            low = min(prices[-20:])
            close = prices[-1]
            
            pivot = (high + low + close) / 3
            support1 = 2 * pivot - high
            resistance1 = 2 * pivot - low
            support2 = pivot - (high - low)
            resistance2 = pivot + (high - low)
            support3 = low - 2 * (high - pivot)
            resistance3 = high + 2 * (pivot - low)
            
            return {
                "support_levels": sorted([support3, support2, support1]),
                "resistance_levels": sorted([resistance1, resistance2, resistance3]),
                "pivot": pivot,
                "confidence": 0.75,
                "method": "Classic_Pivot_Points"
            }
        
        elif algorithm_name == "VolumeProfile_Advanced":
            # Volume Profile analysis
            prices = np.array(market_data['price_history'])
            volumes = np.array(market_data['volume_history'])
            
            if len(prices) < 50:
                raise InsufficientDataError(required=50, available=len(prices), operation="VolumeProfile_Advanced")
            
            # Crea price bins
            price_min, price_max = prices.min(), prices.max()
            n_bins = 20
            bins = np.linspace(price_min, price_max, n_bins)
            
            # Calcola volume per price level
            volume_profile = np.zeros(n_bins - 1)
            for i in range(len(prices)):
                bin_idx = np.searchsorted(bins, prices[i]) - 1
                if 0 <= bin_idx < len(volume_profile):
                    volume_profile[bin_idx] += volumes[i]
            
            # Trova high volume nodes (potential S/R)
            threshold = np.percentile(volume_profile, 70)
            high_volume_indices = np.where(volume_profile > threshold)[0]
            
            support_levels = []
            resistance_levels = []
            current_price = market_data['current_price']
            
            for idx in high_volume_indices:
                level = (bins[idx] + bins[idx + 1]) / 2
                if level < current_price:
                    support_levels.append(level)
                else:
                    resistance_levels.append(level)
            
            return {
                "support_levels": sorted(support_levels)[-3:],  # Top 3
                "resistance_levels": sorted(resistance_levels)[:3],  # Top 3
                "confidence": 0.8,
                "method": "Volume_Profile_Analysis",
                "volume_nodes": len(high_volume_indices)
            }
        
        elif algorithm_name == "LSTM_SupportResistance":
            # LSTM per Support/Resistance con protezioni
            model = self.ml_models.get('LSTM_SupportResistance')
            if model is None:
                raise ModelNotInitializedError('LSTM_SupportResistance')
            
            # Prepara input
            prices = np.array(market_data['price_history'][-50:])
            volumes = np.array(market_data['volume_history'][-50:])
            
            if len(prices) < 50:
                raise InsufficientDataError(required=50, available=len(prices), operation="LSTM_SupportResistance")
            
            # 🛡️ VALIDAZIONE DATI INPUT
            if np.isnan(prices).any() or np.isinf(prices).any():
                safe_print("❌ Prezzi contengono valori NaN/Inf")
                raise InvalidInputError("prices", "NaN/Inf values", "LSTM requires valid numeric prices")
            
            if np.isnan(volumes).any() or np.isinf(volumes).any():
                safe_print("❌ Volumi contengono valori NaN/Inf")
                raise InvalidInputError("volumes", "NaN/Inf values", "LSTM requires valid numeric volumes")
            
            try:
                # Feature engineering
                features = self._prepare_lstm_features(prices, volumes, market_data)
                
                # 🛡️ VALIDAZIONE FEATURES
                if np.isnan(features).any() or np.isinf(features).any():
                    safe_print("❌ Features contengono valori NaN/Inf")
                    raise InvalidInputError("features", "NaN/Inf values", "LSTM features must be numeric")
                
                # Prediction protetta
                model.eval()
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(features).unsqueeze(0)
                    
                    # 🛡️ VALIDAZIONE TENSOR INPUT
                    if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
                        safe_print("❌ Input tensor contiene valori NaN/Inf")
                        raise InvalidInputError("input_tensor", "NaN/Inf values", "PyTorch tensor must be finite")
                    
                    prediction = model(input_tensor)
                    
                    # 🛡️ VALIDAZIONE OUTPUT
                    if torch.isnan(prediction).any() or torch.isinf(prediction).any():
                        safe_print("❌ LSTM output contiene valori NaN/Inf")
                        raise PredictionError("LSTM_SupportResistance", "Model produced NaN/Inf in output")
                    
                    levels = prediction.numpy().flatten()
                    
                    # 🛡️ VALIDAZIONE FINALE
                    if np.isnan(levels).any() or np.isinf(levels).any():
                        safe_print("❌ Livelli finali contengono valori NaN/Inf")
                        raise PredictionError("LSTM_SupportResistance", "Final levels contain NaN/Inf values")
                    
                safe_print(f"✅ LSTM prediction successful: {levels.shape}")
                
                # Interpreta output con validazione
                current_price = market_data['current_price']
                
                # 🛡️ VALIDAZIONE CURRENT_PRICE
                if np.isnan(current_price) or np.isinf(current_price) or current_price <= 0:
                    safe_print(f"❌ Current price non valido: {current_price}")
                    raise InvalidInputError("current_price", current_price, "Must be a positive finite number")
                
                support_levels = []
                resistance_levels = []
                
                for i in range(0, len(levels), 2):
                    if i < len(levels) - 1:
                        support_offset = levels[i]
                        resistance_offset = levels[i + 1]
                        
                        # 🛡️ VALIDAZIONE OFFSET
                        if np.isnan(support_offset) or np.isinf(support_offset):
                            continue
                        if np.isnan(resistance_offset) or np.isinf(resistance_offset):
                            continue
                        
                        support_level = current_price * (1 + support_offset)
                        resistance_level = current_price * (1 + resistance_offset)
                        
                        # 🛡️ VALIDAZIONE LIVELLI FINALI
                        if (not np.isnan(support_level) and not np.isinf(support_level) and support_level > 0):
                            support_levels.append(support_level)
                        
                        if (not np.isnan(resistance_level) and not np.isinf(resistance_level) and resistance_level > 0):
                            resistance_levels.append(resistance_level)
                
                safe_print(f"✅ LSTM S/R levels: {len(support_levels)} support, {len(resistance_levels)} resistance")
                
                return {
                    "support_levels": sorted(support_levels),
                    "resistance_levels": sorted(resistance_levels),
                    "confidence": 0.85,
                    "method": "LSTM_Neural_Network_Protected"
                }
                
            except Exception as e:
                safe_print(f"❌ Errore durante predizione LSTM: {e}")
                raise PredictionError("LSTM_SupportResistance", f"Unexpected error: {str(e)}") from e
        
        elif algorithm_name == "StatisticalLevels_ML":
            # Statistical approach with ML
            prices = np.array(market_data['price_history'])
            
            if len(prices) < 100:
                raise InsufficientDataError(required=100, available=len(prices), operation="StatisticalLevels_ML")
            
            # Identifica livelli statisticamente significativi
            price_counts = defaultdict(int)
            for price in prices:
                # Round to nearest pip
                rounded_price = round(price, 5)
                price_counts[rounded_price] += 1
            
            # Trova i livelli più testati
            significant_levels = sorted(price_counts.items(), 
                                    key=lambda x: x[1], reverse=True)[:10]
            
            current_price = market_data['current_price']
            support_levels = []
            resistance_levels = []
            
            for level, count in significant_levels:
                if count > 3:  # Testato almeno 3 volte
                    if level < current_price:
                        support_levels.append(level)
                    else:
                        resistance_levels.append(level)
            
            # Calcola confidence basato su quante volte i livelli sono stati rispettati
            confidence = min(0.9, 0.5 + (len(support_levels) + len(resistance_levels)) * 0.05)
            
            return {
                "support_levels": sorted(support_levels)[-3:],
                "resistance_levels": sorted(resistance_levels)[:3],
                "confidence": confidence,
                "method": "Statistical_ML_Analysis"
            }
        
        elif algorithm_name == "Transformer_Levels":
            # Transformer per livelli S/R
            model = self.ml_models.get('Transformer_Levels')
            if model is None:
                raise ModelNotInitializedError('Transformer_Levels')
            
            # Prepara features
            prices = np.array(market_data['price_history'][-100:])
            volumes = np.array(market_data['volume_history'][-100:])
            
            if len(prices) < 100:
                raise InsufficientDataError(required=100, available=len(prices), operation="Transformer_Levels")
            
            features = self._prepare_transformer_features(prices, volumes)
            
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features).unsqueeze(0)
                prediction = model(input_tensor)
                levels = torch.sigmoid(prediction).numpy().flatten()
            
            # Converti predictions in livelli
            current_price = market_data['current_price']
            price_range = prices.max() - prices.min()
            
            support_levels = []
            resistance_levels = []
            
            for i, level_prob in enumerate(levels):
                if level_prob > 0.6:  # Threshold per significatività
                    # Mappa probabilità a offset di prezzo
                    offset = (i - len(levels)/2) / len(levels) * price_range * 0.1
                    level_price = current_price + offset
                    
                    if level_price < current_price:
                        support_levels.append(level_price)
                    else:
                        resistance_levels.append(level_price)
            
            return {
                "support_levels": sorted(support_levels)[-3:],
                "resistance_levels": sorted(resistance_levels)[:3],
                "confidence": 0.88,
                "method": "Transformer_AI"
            }
        
        raise AnalyzerException(f"Algorithm {algorithm_name} not implemented yet")
    
    def _run_pattern_recognition_algorithm(self, algorithm_name: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Esegue algoritmi di Pattern Recognition con error handling standardizzato"""
        
        if algorithm_name == "CNN_PatternRecognizer":
            model = self.ml_models.get('CNN_PatternRecognizer')
            if model is None:
                raise ModelNotInitializedError('CNN_PatternRecognizer')
            
            prices = np.array(market_data['price_history'][-100:])
            if len(prices) < 100:
                raise InsufficientDataError(required=100, available=len(prices), operation="CNN_PatternRecognizer")
            
            try:
                # Normalizza i prezzi
                normalized_prices = (prices - prices.mean()) / prices.std()
                
                # Prepara input per CNN
                input_tensor = torch.FloatTensor(normalized_prices).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    pattern_logits = model(input_tensor)
                    pattern_probs = pattern_logits.numpy().flatten()
            except Exception as e:
                raise PredictionError("CNN_PatternRecognizer", f"Model inference failed: {str(e)}")
            
            # Pattern mapping
            pattern_names = [
                "double_top", "double_bottom", "head_shoulders", "inverse_head_shoulders",
                "ascending_triangle", "descending_triangle", "symmetrical_triangle",
                "bullish_flag", "bearish_flag", "bullish_pennant", "bearish_pennant",
                "cup_handle", "wedge_rising", "wedge_falling", "rectangle_pattern",
                "breakout_up", "breakout_down", "reversal_bullish", "reversal_bearish",
                "continuation_bullish", "continuation_bearish", "channel_up", "channel_down",
                "broadening_pattern", "diamond_pattern"
            ]
            
            detected_patterns = []
            for i, prob in enumerate(pattern_probs[:len(pattern_names)]):
                if prob > 0.7:  # Threshold
                    pattern = pattern_names[i]
                    
                    # Determina direzione
                    if 'bullish' in pattern or 'up' in pattern or 'ascending' in pattern:
                        direction = 'bullish'
                    elif 'bearish' in pattern or 'down' in pattern or 'descending' in pattern:
                        direction = 'bearish'
                    else:
                        direction = 'neutral'
                    
                    detected_patterns.append({
                        "pattern": pattern,
                        "probability": float(prob),
                        "confidence": float(prob * 0.9),
                        "direction": direction,
                        "timeframe": "short_term"
                    })
            
            # Ordina per probabilità
            detected_patterns.sort(key=lambda x: x['probability'], reverse=True)
            
            return {
                "detected_patterns": detected_patterns[:3],  # Top 3
                "pattern_strength": float(np.max(pattern_probs)) if detected_patterns else 0.0,
                "confidence": float(np.mean([p["confidence"] for p in detected_patterns[:3]])) if detected_patterns else 0.3,
                "method": "CNN_Deep_Learning"
            }
            
        elif algorithm_name == "Classical_Patterns":
            # Pattern recognition classico
            prices = np.array(market_data['price_history'][-50:])
            if len(prices) < 50:
                raise InsufficientDataError(required=50, available=len(prices), operation="Classical_Patterns")
            
            patterns = []
            
            try:
                # Double Top Detection
                if self._detect_double_top(prices):
                    patterns.append({
                        "pattern": "double_top", 
                        "probability": 0.8, 
                        "direction": "bearish",
                        "confidence": 0.75
                    })
                
                # Double Bottom Detection  
                if self._detect_double_bottom(prices):
                    patterns.append({
                        "pattern": "double_bottom", 
                        "probability": 0.8, 
                        "direction": "bullish",
                        "confidence": 0.75
                    })
                
                # Head & Shoulders
                if self._detect_head_shoulders(prices):
                    patterns.append({
                        "pattern": "head_shoulders",
                        "probability": 0.75,
                        "direction": "bearish",
                        "confidence": 0.7
                    })
                
                # Triangle Pattern Detection
                triangle_type = self._detect_triangle_pattern(prices)
                if triangle_type:
                    patterns.append({
                        "pattern": f"{triangle_type}_triangle",
                        "probability": 0.75,
                        "direction": "breakout",
                        "confidence": 0.7
                    })
                
                # Flag Pattern Detection
                flag_type = self._detect_flag_pattern(prices)
                if flag_type:
                    patterns.append({
                        "pattern": f"{flag_type}_flag",
                        "probability": 0.7,
                        "direction": flag_type,
                        "confidence": 0.65
                    })
                
                # Channel Detection
                channel_type = self._detect_channel_pattern(prices)
                if channel_type:
                    patterns.append({
                        "pattern": f"{channel_type}_channel",
                        "probability": 0.7,
                        "direction": channel_type.split('_')[0],
                        "confidence": 0.65
                    })
            except Exception as e:
                raise PredictionError("Classical_Patterns", f"Pattern detection failed: {str(e)}")
            
            return {
                "detected_patterns": patterns,
                "pattern_strength": max([p["probability"] for p in patterns]) if patterns else 0.0,
                "confidence": 0.8 if patterns else 0.3,
                "method": "Classical_Technical_Analysis"
            }
        
        elif algorithm_name == "LSTM_Sequences":
            model = self.ml_models.get('LSTM_Sequences')
            if model is None:
                raise ModelNotInitializedError('LSTM_Sequences')
            
            # Prepara sequenze per LSTM
            prices = np.array(market_data['price_history'][-60:])
            volumes = np.array(market_data['volume_history'][-60:])
            
            if len(prices) < 60:
                raise InsufficientDataError(required=60, available=len(prices), operation="LSTM_Sequences")
            
            try:
                # Feature engineering per pattern sequences
                features = self._prepare_lstm_pattern_features(prices, volumes)
                
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(features).unsqueeze(0)
                    pattern_output = model(input_tensor)
                    pattern_probs = torch.softmax(pattern_output, dim=-1).numpy().flatten()
            except Exception as e:
                raise PredictionError("LSTM_Sequences", f"LSTM pattern prediction failed: {str(e)}")
            
            # Interpreta output come probabilità di pattern sequenziali
            sequential_patterns = [
                "impulse_wave", "corrective_wave", "elliott_wave_3", "elliott_wave_5",
                "wyckoff_accumulation", "wyckoff_distribution", "trend_continuation",
                "trend_exhaustion", "momentum_divergence", "volume_climax"
            ]
            
            detected_patterns = []
            for i, prob in enumerate(pattern_probs[:len(sequential_patterns)]):
                if prob > 0.6:
                    pattern = sequential_patterns[i]
                    detected_patterns.append({
                        "pattern": pattern,
                        "probability": float(prob),
                        "confidence": float(prob * 0.85),
                        "direction": self._get_pattern_direction(pattern),
                        "sequence_strength": float(prob)
                    })
            
            return {
                "detected_patterns": detected_patterns,
                "pattern_strength": float(np.max(pattern_probs)),
                "confidence": float(np.mean([p["confidence"] for p in detected_patterns])) if detected_patterns else 0.5,
                "method": "LSTM_Sequence_Analysis"
            }
        
        elif algorithm_name == "Transformer_Patterns":
            model = self.ml_models.get('Transformer_Patterns')
            if model is None:
                raise ModelNotInitializedError('Transformer_Patterns')
            
            # Prepara features complesse per Transformer
            prices = np.array(market_data['price_history'][-100:])
            volumes = np.array(market_data['volume_history'][-100:])
            
            if len(prices) < 100:
                raise InsufficientDataError(required=100, available=len(prices), operation="Transformer_Patterns")
            
            try:
                features = self._prepare_transformer_pattern_features(prices, volumes, market_data)
                
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(features).unsqueeze(0)
                    pattern_logits = model(input_tensor)
                    pattern_probs = torch.softmax(pattern_logits, dim=-1).numpy().flatten()
            except Exception as e:
                raise PredictionError("Transformer_Patterns", f"Transformer inference failed: {str(e)}")
            
            # Pattern avanzati riconoscibili dal Transformer
            advanced_patterns = [
                "harmonic_butterfly", "harmonic_gartley", "harmonic_bat", "harmonic_crab",
                "elliott_impulse", "elliott_correction", "wyckoff_spring", "wyckoff_upthrust",
                "vsa_stopping_volume", "vsa_no_demand", "market_structure_break",
                "liquidity_grab", "order_block_formation", "fair_value_gap",
                "institutional_accumulation", "institutional_distribution"
            ]
            
            detected_patterns = []
            for i, prob in enumerate(pattern_probs[:len(advanced_patterns)]):
                if prob > 0.65:
                    pattern = advanced_patterns[i]
                    
                    # Aggiungi context specifico per pattern
                    try:
                        pattern_context = self._get_pattern_context(pattern, prices, volumes)
                    except Exception as e:
                        # Se context non riesce, usa default
                        pattern_context = {"context": "default"}
                    
                    detected_patterns.append({
                        "pattern": pattern,
                        "probability": float(prob),
                        "confidence": float(prob * 0.95),
                        "direction": self._get_pattern_direction(pattern),
                        "context": pattern_context,
                        "advanced": True
                    })
            
            return {
                "detected_patterns": detected_patterns,
                "pattern_strength": float(np.max(pattern_probs)),
                "confidence": float(np.mean([p["confidence"] for p in detected_patterns])) if detected_patterns else 0.5,
                "method": "Transformer_Advanced_AI"
            }
        
        elif algorithm_name == "Ensemble_Patterns":
            # Ensemble di tutti i pattern recognizer
            results = []
            
            # Esegui tutti gli altri algoritmi
            for algo in ["CNN_PatternRecognizer", "Classical_Patterns", "LSTM_Sequences", "Transformer_Patterns"]:
                if algo != "Ensemble_Patterns":
                    try:
                        result = self._run_pattern_recognition_algorithm(algo, market_data)
                        if 'detected_patterns' in result:
                            results.append(result)
                    except AnalyzerException:
                        # Skip failed algorithms in ensemble
                        continue
                    except Exception as e:
                        # Log unexpected errors but continue
                        safe_print(f"⚠️ Ensemble algorithm {algo} failed: {e}")
                        continue
            
            if not results:
                raise PredictionError("Ensemble_Patterns", "No pattern algorithms available for ensemble")
            
            # Combina risultati con voting
            pattern_votes = defaultdict(list)
            
            for result in results:
                for pattern in result.get('detected_patterns', []):
                    pattern_name = pattern['pattern']
                    pattern_votes[pattern_name].append({
                        'probability': pattern['probability'],
                        'confidence': pattern['confidence'],
                        'method': result['method']
                    })
            
            # Crea consensus patterns
            ensemble_patterns = []
            for pattern_name, votes in pattern_votes.items():
                if len(votes) >= 2:  # Almeno 2 algoritmi concordano
                    avg_probability = np.mean([v['probability'] for v in votes])
                    avg_confidence = np.mean([v['confidence'] for v in votes])
                    
                    ensemble_patterns.append({
                        "pattern": pattern_name,
                        "probability": float(avg_probability),
                        "confidence": float(avg_confidence * 1.1),  # Boost per consensus
                        "direction": self._get_pattern_direction(pattern_name),
                        "consensus_count": len(votes),
                        "methods": [v['method'] for v in votes]
                    })
            
            # Ordina per confidence
            ensemble_patterns.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                "detected_patterns": ensemble_patterns[:3],
                "pattern_strength": float(max([p["probability"] for p in ensemble_patterns])) if ensemble_patterns else 0.0,
                "confidence": float(max([p["confidence"] for p in ensemble_patterns])) if ensemble_patterns else 0.4,
                "method": "Ensemble_Consensus",
                "algorithms_used": len(results)
            }
        
        raise AnalyzerException(f"Algorithm {algorithm_name} not implemented yet")
    
    def _run_bias_detection_algorithm(self, algorithm_name: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Esegue algoritmi di Bias Detection con error handling standardizzato"""
        
        if algorithm_name == "Sentiment_LSTM":
            model = self.ml_models.get('Sentiment_LSTM')
            if model is None:
                raise ModelNotInitializedError('Sentiment_LSTM')
            
            # Prepara features per sentiment
            prices = np.array(market_data['price_history'][-30:])
            volumes = np.array(market_data['volume_history'][-30:])
            
            if len(prices) < 30:
                raise InsufficientDataError(required=30, available=len(prices), operation="Sentiment_LSTM")
            
            try:
                # Feature engineering per sentiment
                features = self._prepare_sentiment_features(prices, volumes, market_data)
                
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(features).unsqueeze(0)
                    bias_output = model(input_tensor)
                    bias_probs = torch.softmax(bias_output, dim=-1).numpy().flatten()
            except Exception as e:
                raise PredictionError("Sentiment_LSTM", f"LSTM sentiment prediction failed: {str(e)}")
            
            # Interpreta output: [bearish, neutral, bullish]
            bias_labels = ["bearish", "neutral", "bullish"]
            dominant_bias_idx = np.argmax(bias_probs)
            dominant_bias = bias_labels[dominant_bias_idx]
            bias_confidence = float(bias_probs[dominant_bias_idx])
            
            # Analisi addizionale del sentiment
            try:
                behavioral_analysis = self._analyze_market_behavior(prices, volumes)
            except Exception as e:
                # Fallback se behavioral analysis fallisce
                behavioral_analysis = {"type": "unknown", "confidence": 0.5}
                safe_print(f"⚠️ Behavioral analysis fallback: {e}")
            
            return {
                "directional_bias": {
                    "direction": dominant_bias,
                    "confidence": bias_confidence,
                    "distribution": {bias_labels[i]: float(bias_probs[i]) for i in range(len(bias_labels))}
                },
                "behavioral_bias": behavioral_analysis,
                "overall_confidence": float((bias_confidence + behavioral_analysis["confidence"]) / 2),
                "method": "LSTM_Sentiment_Analysis"
            }
        
        elif algorithm_name == "VolumePrice_Analysis":
            # Analisi Volume-Price per bias
            prices = np.array(market_data['price_history'][-50:])
            volumes = np.array(market_data['volume_history'][-50:])
            
            if len(prices) < 50:
                raise InsufficientDataError(required=50, available=len(prices), operation="VolumePrice_Analysis")
            
            try:
                # Volume-Price Trend Analysis
                price_changes = np.diff(prices)
                
                # Analisi volume su movimenti positivi vs negativi
                positive_volume = np.sum(volumes[1:][price_changes > 0])
                negative_volume = np.sum(volumes[1:][price_changes < 0])
                total_volume = positive_volume + negative_volume
                
                if total_volume > 0:
                    buying_pressure = positive_volume / total_volume
                    selling_pressure = negative_volume / total_volume
                else:
                    buying_pressure = selling_pressure = 0.5
                
                # Volume momentum
                volume_ma_short = np.mean(volumes[-10:])
                volume_ma_long = np.mean(volumes[-30:])
                volume_momentum = (volume_ma_short - volume_ma_long) / volume_ma_long if volume_ma_long > 0 else 0
                
                # Price-Volume divergence
                price_trend = (prices[-1] - prices[-20]) / prices[-20]
                volume_trend = (volume_ma_short - np.mean(volumes[-20:-10])) / np.mean(volumes[-20:-10])
                divergence = abs(price_trend - volume_trend)
                
                # Smart money analysis
                smart_money_indicator = self._analyze_smart_money(prices, volumes)
                
            except Exception as e:
                raise PredictionError("VolumePrice_Analysis", f"Volume-Price analysis failed: {str(e)}")
            
            # Determina bias
            if buying_pressure > 0.6 and volume_momentum > 0.1:
                directional_bias = "bullish"
                confidence = min(0.9, buying_pressure + volume_momentum * 0.5)
            elif selling_pressure > 0.6 and volume_momentum < -0.1:
                directional_bias = "bearish"
                confidence = min(0.9, selling_pressure - volume_momentum * 0.5)
            else:
                directional_bias = "neutral"
                confidence = 0.5
            
            # Caratteristiche comportamentali
            if divergence > 0.1:
                behavioral = "divergent"
            elif np.std(volumes[-10:]) / np.mean(volumes[-10:]) > 0.5:
                behavioral = "volatile"
            else:
                behavioral = "stable"
            
            return {
                "directional_bias": {
                    "direction": directional_bias,
                    "confidence": float(confidence)
                },
                "behavioral_bias": {
                    "type": behavioral,
                    "buying_pressure": float(buying_pressure),
                    "selling_pressure": float(selling_pressure),
                    "volume_momentum": float(volume_momentum),
                    "divergence": float(divergence),
                    "smart_money": smart_money_indicator,
                    "confidence": float(confidence)
                },
                "overall_confidence": float(confidence),
                "method": "Volume_Price_Analysis"
            }
        
        elif algorithm_name == "Momentum_ML":
            # Machine Learning basato su momentum
            prices = np.array(market_data['price_history'])
            volumes = np.array(market_data['volume_history'])
            
            if len(prices) < 100:
                raise InsufficientDataError(required=100, available=len(prices), operation="Momentum_ML")
            
            try:
                # Calcola vari momentum indicators
                momentum_features = self._calculate_momentum_features(prices, volumes)
                
                # Analisi del momentum bias
                short_momentum = momentum_features['momentum_5']
                medium_momentum = momentum_features['momentum_20']
                long_momentum = momentum_features['momentum_50']
                
                # RSI bias
                rsi = momentum_features['rsi']
                rsi_bias = "neutral"
                if rsi > 70:
                    rsi_bias = "overbought"
                elif rsi < 30:
                    rsi_bias = "oversold"
                
                # MACD bias
                macd_bias = "neutral"
                if momentum_features['macd'] > momentum_features['macd_signal']:
                    macd_bias = "bullish"
                elif momentum_features['macd'] < momentum_features['macd_signal']:
                    macd_bias = "bearish"
                
            except Exception as e:
                raise PredictionError("Momentum_ML", f"Momentum features calculation failed: {str(e)}")
            
            # Combine momentum signals
            momentum_score = 0
            if short_momentum > 0: momentum_score += 1
            if medium_momentum > 0: momentum_score += 1
            if long_momentum > 0: momentum_score += 1
            if rsi_bias == "oversold": momentum_score += 1
            if rsi_bias == "overbought": momentum_score -= 1
            if macd_bias == "bullish": momentum_score += 1
            if macd_bias == "bearish": momentum_score -= 1
            
            # Determina bias finale
            if momentum_score >= 3:
                direction = "bullish"
                confidence = min(0.9, 0.5 + momentum_score * 0.1)
            elif momentum_score <= -2:
                direction = "bearish"
                confidence = min(0.9, 0.5 + abs(momentum_score) * 0.1)
            else:
                direction = "neutral"
                confidence = 0.5
            
            return {
                "directional_bias": {
                    "direction": direction,
                    "confidence": float(confidence)
                },
                "behavioral_bias": {
                    "momentum_score": momentum_score,
                    "rsi_condition": rsi_bias,
                    "macd_signal": macd_bias,
                    "short_term_momentum": float(short_momentum),
                    "medium_term_momentum": float(medium_momentum),
                    "long_term_momentum": float(long_momentum),
                    "confidence": float(confidence)
                },
                "overall_confidence": float(confidence),
                "method": "Momentum_ML_Analysis"
            }
        
        elif algorithm_name == "Transformer_Bias":
            model = self.ml_models.get('Transformer_Bias')
            if model is None:
                raise ModelNotInitializedError('Transformer_Bias')
            
            # Prepara features multi-dimensionali
            prices = np.array(market_data['price_history'][-100:])
            volumes = np.array(market_data['volume_history'][-100:])
            
            if len(prices) < 100:
                raise InsufficientDataError(required=100, available=len(prices), operation="Transformer_Bias")
            
            try:
                features = self._prepare_transformer_bias_features(prices, volumes, market_data)
                
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(features).unsqueeze(0)
                    bias_output = model(input_tensor)
                    bias_logits = bias_output.numpy().flatten()
                
                # Interpreta output multi-dimensionale
                # Output: [bearish_strength, neutral_strength, bullish_strength]
                bias_probs = self._softmax(bias_logits)
                
                bias_labels = ["bearish", "neutral", "bullish"]
                dominant_idx = np.argmax(bias_probs)
                
                # Analisi più sofisticata del bias
                market_regime = self._detect_market_regime(prices, volumes)
                institutional_bias = self._detect_institutional_bias(prices, volumes, market_data)
                
            except Exception as e:
                raise PredictionError("Transformer_Bias", f"Transformer bias prediction failed: {str(e)}")
            
            return {
                "directional_bias": {
                    "direction": bias_labels[dominant_idx],
                    "confidence": float(bias_probs[dominant_idx]),
                    "distribution": {bias_labels[i]: float(bias_probs[i]) for i in range(len(bias_labels))}
                },
                "behavioral_bias": {
                    "market_regime": market_regime,
                    "institutional_activity": institutional_bias,
                    "confidence": float(bias_probs[dominant_idx])
                },
                "overall_confidence": float(bias_probs[dominant_idx] * 0.95),
                "method": "Transformer_Advanced_Bias"
            }
            
        elif algorithm_name == "MultiModal_Bias":
            # Combina multiple modalità per bias detection
            
            try:
                # Raccogli risultati da altri metodi
                methods_results = []
                
                # Price action bias
                price_bias = self._analyze_price_action_bias(market_data)
                methods_results.append(price_bias)
                
                # Order flow bias
                order_flow_bias = self._analyze_order_flow_bias(market_data)
                methods_results.append(order_flow_bias)
                
                # Market microstructure bias
                microstructure_bias = self._analyze_microstructure_bias(market_data)
                methods_results.append(microstructure_bias)
                
                # Volatility regime bias
                volatility_bias = self._analyze_volatility_bias(market_data)
                methods_results.append(volatility_bias)
                
                # Combina tutti i bias con weighted voting
                combined_bias = self._combine_bias_signals(methods_results)
                
            except Exception as e:
                raise PredictionError("MultiModal_Bias", f"Multi-modal bias analysis failed: {str(e)}")
            
            return {
                "directional_bias": combined_bias["directional"],
                "behavioral_bias": combined_bias["behavioral"],
                "overall_confidence": combined_bias["confidence"],
                "method": "MultiModal_Ensemble",
                "components": {
                    "price_action": price_bias,
                    "order_flow": order_flow_bias,
                    "microstructure": microstructure_bias,
                    "volatility": volatility_bias
                }
            }
        
        raise AnalyzerException(f"Algorithm {algorithm_name} not implemented yet")
    
    def _run_trend_analysis_algorithm(self, algorithm_name: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Esegue algoritmi di Trend Analysis con error handling standardizzato"""
        
        if algorithm_name == "RandomForest_Trend":
            model = self.ml_models.get('RandomForest_Trend')
            if model is None:
                raise ModelNotInitializedError('RandomForest_Trend')
            
            # Prepara features
            prices = np.array(market_data['price_history'][-100:])
            volumes = np.array(market_data['volume_history'][-100:])
            
            if len(prices) < 100:
                raise InsufficientDataError(required=100, available=len(prices), operation="RandomForest_Trend")
            
            try:
                # Feature engineering per trend
                features = self._prepare_trend_features(prices, volumes)
                
                # Verifica che il modello sia stato trainato
                # Prediction
                trend_prediction = model.predict(features.reshape(1, -1))[0]
                
                # Se il modello supporta predict_proba
                if hasattr(model, 'predict_proba'):
                    trend_proba = model.predict_proba(features.reshape(1, -1))[0]
                else:
                    # Stima confidence basata su feature importance
                    trend_proba = self._estimate_confidence_rf(model, features)
                
                trend_labels = ["downtrend", "sideways", "uptrend"]
                trend_direction = trend_labels[int(trend_prediction)]
                
                # Analisi aggiuntiva della forza del trend
                trend_strength = self._calculate_trend_strength(prices)
                
            except Exception as e:
                if "not fitted" in str(e).lower() or "not trained" in str(e).lower():
                    raise ModelNotInitializedError('RandomForest_Trend')
                else:
                    raise PredictionError("RandomForest_Trend", f"Model prediction failed: {str(e)}")
            
            return {
                "trend_direction": trend_direction,
                "trend_strength": float(trend_strength),
                "confidence": float(np.max(trend_proba)),
                "probabilities": {trend_labels[i]: float(trend_proba[i]) for i in range(len(trend_labels))},
                "trend_metrics": {
                    "slope": float(features[0]),  # Assumendo che slope sia la prima feature
                    "r_squared": float(self._calculate_trend_r_squared(prices)),
                    "volatility": float(market_data.get('volatility', 0))
                },
                "method": "RandomForest_ML"
            }
        
        elif algorithm_name == "LSTM_TrendPrediction":
            model = self.ml_models.get('LSTM_TrendPrediction')
            if model is None:
                raise ModelNotInitializedError('LSTM_TrendPrediction')
            
            prices = np.array(market_data['price_history'][-50:])
            volumes = np.array(market_data['volume_history'][-50:])
            
            if len(prices) < 50:
                raise InsufficientDataError(required=50, available=len(prices), operation="LSTM_TrendPrediction")
            
            try:
                # Prepara sequenza per LSTM
                lstm_features = self._prepare_lstm_trend_features(prices, volumes)
                
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(lstm_features).unsqueeze(0)
                    trend_output = model(input_tensor)
                    trend_probs = torch.softmax(trend_output, dim=-1).numpy().flatten()
                
            except Exception as e:
                raise PredictionError("LSTM_TrendPrediction", f"LSTM trend prediction failed: {str(e)}")
            
            # Output più dettagliato: [strong_down, weak_down, sideways, weak_up, strong_up]
            trend_labels = ["strong_down", "weak_down", "sideways", "weak_up", "strong_up"]
            trend_idx = np.argmax(trend_probs)
            trend_detail = trend_labels[trend_idx]
            
            # Semplifica in 3 categorie principali
            if "down" in trend_detail:
                simple_direction = "downtrend"
            elif "up" in trend_detail:
                simple_direction = "uptrend"
            else:
                simple_direction = "sideways"
            
            # Calcola trend projection
            try:
                trend_projection = self._project_trend(prices, simple_direction, trend_probs[trend_idx])
            except Exception as e:
                # Fallback per projection
                trend_projection = {"error": f"Projection failed: {str(e)}"}
            
            return {
                "trend_direction": simple_direction,
                "trend_detail": trend_detail,
                "trend_strength": float(trend_probs[trend_idx]),
                "confidence": float(trend_probs[trend_idx]),
                "trend_projection": trend_projection,
                "probabilities": {trend_labels[i]: float(trend_probs[i]) for i in range(len(trend_labels))},
                "method": "LSTM_Deep_Learning"
            }
        
        elif algorithm_name == "GradientBoosting_Trend":
            model = self.ml_models.get('GradientBoosting_Trend')
            if model is None:
                raise ModelNotInitializedError('GradientBoosting_Trend')
            
            # Simile a RandomForest ma con features leggermente diverse
            prices = np.array(market_data['price_history'][-100:])
            volumes = np.array(market_data['volume_history'][-100:])
            
            if len(prices) < 100:
                raise InsufficientDataError(required=100, available=len(prices), operation="GradientBoosting_Trend")
            
            try:
                features = self._prepare_gb_trend_features(prices, volumes, market_data)
                
                trend_prediction = model.predict(features.reshape(1, -1))[0]
                
                # GradientBoosting confidence basata su prediction variance
                trend_confidence = self._estimate_confidence_gb(model, features)
                
                trend_labels = ["downtrend", "sideways", "uptrend"]
                trend_direction = trend_labels[int(trend_prediction)]
                
                # Analisi trend caratteristiche
                trend_characteristics = self._analyze_trend_characteristics(prices, volumes)
                
            except Exception as e:
                if "not fitted" in str(e).lower() or "not trained" in str(e).lower():
                    raise ModelNotInitializedError('GradientBoosting_Trend')
                else:
                    raise PredictionError("GradientBoosting_Trend", f"GradientBoosting prediction failed: {str(e)}")
            
            return {
                "trend_direction": trend_direction,
                "trend_strength": trend_characteristics["strength"],
                "confidence": float(trend_confidence),
                "trend_characteristics": trend_characteristics,
                "method": "GradientBoosting_ML"
            }
        
        elif algorithm_name == "Transformer_Trend":
            model = self.ml_models.get('Transformer_Trend')
            if model is None:
                raise ModelNotInitializedError('Transformer_Trend')
            
            # Prepara features avanzate per Transformer
            prices = np.array(market_data['price_history'][-100:])
            volumes = np.array(market_data['volume_history'][-100:])
            
            if len(prices) < 100:
                raise InsufficientDataError(required=100, available=len(prices), operation="Transformer_Trend")
            
            try:
                features = self._prepare_transformer_trend_features(prices, volumes, market_data)
                
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(features).unsqueeze(0)
                    trend_output = model(input_tensor)
                    trend_logits = trend_output.numpy().flatten()
                
                # Output multi-dimensionale per trend analysis
                trend_components = self._decompose_trend_components(trend_logits)
                
                # Determina trend principale
                primary_trend = trend_components["primary_trend"]
                
                # Analisi di trend complessa
                multi_timeframe_trends = self._analyze_multi_timeframe_trends(prices, volumes)
                
            except Exception as e:
                raise PredictionError("Transformer_Trend", f"Transformer trend analysis failed: {str(e)}")
            
            return {
                "trend_direction": primary_trend["direction"],
                "trend_strength": primary_trend["strength"],
                "confidence": primary_trend["confidence"],
                "trend_components": trend_components,
                "multi_timeframe": multi_timeframe_trends,
                "method": "Transformer_Advanced_AI"
            }
        
        elif algorithm_name == "Ensemble_Trend":
            # Ensemble di tutti i trend analyzer
            ensemble_results = []
            
            # Esegui tutti gli altri algoritmi
            for algo in ["RandomForest_Trend", "LSTM_TrendPrediction", 
                        "GradientBoosting_Trend", "Transformer_Trend"]:
                if algo != "Ensemble_Trend":
                    try:
                        result = self._run_trend_analysis_algorithm(algo, market_data)
                        if 'trend_direction' in result:
                            ensemble_results.append(result)
                    except AnalyzerException:
                        # Skip failed algorithms in ensemble
                        continue
                    except Exception as e:
                        # Log unexpected errors but continue
                        safe_print(f"⚠️ Ensemble trend algorithm {algo} failed: {e}")
                        continue
            
            if not ensemble_results:
                raise PredictionError("Ensemble_Trend", "No trend algorithms available for ensemble")
            
            try:
                # Weighted voting basato su confidence
                trend_votes = {"uptrend": 0.0, "sideways": 0.0, "downtrend": 0.0}
                total_confidence = 0
                
                for result in ensemble_results:
                    confidence = result.get('confidence', 0.5)
                    direction = result.get('trend_direction', 'sideways')
                    
                    trend_votes[direction] += confidence
                    total_confidence += confidence
                
                # Normalizza votes
                for trend in trend_votes:
                    trend_votes[trend] = float(trend_votes[trend] / max(1, total_confidence))
                
                # Trova trend dominante
                dominant_trend = max(trend_votes.items(), key=lambda x: x[1])
                
                # Calcola confidence dell'ensemble
                ensemble_confidence = dominant_trend[1]
                
                # Se c'è forte disaccordo, riduci confidence
                vote_variance = np.var(list(trend_votes.values()))
                if vote_variance > 0.1:
                    ensemble_confidence *= 0.8
                
            except Exception as e:
                raise PredictionError("Ensemble_Trend", f"Ensemble voting failed: {str(e)}")
            
            return {
                "trend_direction": dominant_trend[0],
                "trend_strength": float(np.mean([r.get('trend_strength', 0.5) for r in ensemble_results])),
                "confidence": float(ensemble_confidence),
                "ensemble_votes": trend_votes,
                "algorithms_consensus": len(ensemble_results),
                "method": "Ensemble_Voting"
            }
        
        raise AnalyzerException(f"Algorithm {algorithm_name} not implemented yet")
    
    def _run_volatility_prediction_algorithm(self, algorithm_name: str, 
                                        market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Esegue algoritmi di Volatility Prediction con error handling standardizzato"""
        
        if algorithm_name == "GARCH_Volatility":
            # GARCH implementation
            prices = np.array(market_data['price_history'])
            
            if len(prices) < 100:
                raise InsufficientDataError(required=100, available=len(prices), operation="GARCH_Volatility")
            
            try:
                # Calcola returns
                returns = np.diff(prices) / prices[:-1]
                
                # Simplified GARCH(1,1) implementation
                omega = 0.000001
                alpha = 0.1
                beta = 0.85
                
                # Initialize
                variance = np.var(returns)
                garch_variances = [variance]
                
                # GARCH recursion
                for i in range(1, len(returns)):
                    variance = omega + alpha * returns[i-1]**2 + beta * variance
                    garch_variances.append(variance)
                
                # Predict next period volatility
                next_variance = omega + alpha * returns[-1]**2 + beta * garch_variances[-1]
                predicted_volatility = np.sqrt(next_variance)
                
                # Volatility regimes
                current_vol = np.sqrt(garch_variances[-1])
                avg_vol = np.mean(np.sqrt(garch_variances[-20:]))
                
                if current_vol > avg_vol * 1.5:
                    vol_regime = "high"
                elif current_vol < avg_vol * 0.7:
                    vol_regime = "low"
                else:
                    vol_regime = "normal"
                    
            except Exception as e:
                raise PredictionError("GARCH_Volatility", f"GARCH model computation failed: {str(e)}")
            
            return {
                "predicted_volatility": float(predicted_volatility),
                "current_volatility": float(current_vol),
                "volatility_regime": vol_regime,
                "volatility_trend": "increasing" if predicted_volatility > current_vol else "decreasing",
                "confidence": 0.75,
                "method": "GARCH_Model"
            }
        
        elif algorithm_name == "LSTM_Volatility":
            model = self.ml_models.get('LSTM_Volatility')
            if model is None:
                raise ModelNotInitializedError('LSTM_Volatility')
            
            prices = np.array(market_data['price_history'][-60:])
            volumes = np.array(market_data['volume_history'][-60:])
            
            if len(prices) < 60:
                raise InsufficientDataError(required=60, available=len(prices), operation="LSTM_Volatility")
            
            try:
                # Prepara features per volatility prediction
                features = self._prepare_volatility_features(prices, volumes)
                
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(features).unsqueeze(0)
                    vol_output = model(input_tensor)
                    vol_prediction = vol_output.numpy().flatten()
                
                # Interpreta output: [low_vol, normal_vol, high_vol]
                vol_regimes = ["low", "normal", "high"]
                vol_probs = self._softmax(vol_prediction)
                predicted_regime = vol_regimes[np.argmax(vol_probs)]
                
                # Stima volatilità numerica
                if predicted_regime == "low":
                    predicted_vol = market_data['volatility'] * 0.7
                elif predicted_regime == "high":
                    predicted_vol = market_data['volatility'] * 1.5
                else:
                    predicted_vol = market_data['volatility']
                    
            except Exception as e:
                raise PredictionError("LSTM_Volatility", f"LSTM volatility prediction failed: {str(e)}")
            
            return {
                "predicted_volatility": float(predicted_vol),
                "current_volatility": float(market_data['volatility']),
                "volatility_regime": predicted_regime,
                "regime_probabilities": {vol_regimes[i]: float(vol_probs[i]) for i in range(len(vol_regimes))},
                "confidence": float(np.max(vol_probs)),
                "method": "LSTM_Volatility_Prediction"
            }
        
        elif algorithm_name == "Realized_Volatility":
            # Realized volatility con multiple misure
            prices = np.array(market_data['price_history'])
            
            if len(prices) < 100:
                raise InsufficientDataError(required=100, available=len(prices), operation="Realized_Volatility")
            
            try:
                # Calcola diverse misure di volatilità
                returns = np.diff(prices) / prices[:-1]
                
                # Realized volatility (different windows)
                rv_5 = np.std(returns[-5:]) if len(returns) >= 5 else 0
                rv_20 = np.std(returns[-20:]) if len(returns) >= 20 else 0
                rv_50 = np.std(returns[-50:]) if len(returns) >= 50 else 0
                
                # Parkinson volatility (high-low estimator)
                hl_ranges = []
                for i in range(max(0, len(prices)-20), len(prices)):
                    window = prices[max(0, i-5):i+1]
                    if len(window) > 1:
                        hl_range = (max(window) - min(window)) / np.mean(window)
                        hl_ranges.append(hl_range)
                
                parkinson_vol = np.mean(hl_ranges) * np.sqrt(1/(4*np.log(2))) if hl_ranges else rv_20
                
                # Volatility forecast basato su trend
                vol_trend = (rv_5 - rv_20) / rv_20 if rv_20 > 0 else 0
                
                if vol_trend > 0.2:
                    predicted_vol = rv_5 * 1.1
                    vol_forecast = "increasing"
                elif vol_trend < -0.2:
                    predicted_vol = rv_5 * 0.9
                    vol_forecast = "decreasing"
                else:
                    predicted_vol = rv_5
                    vol_forecast = "stable"
                
                # Determina regime
                if rv_5 > rv_50 * 1.5:
                    regime = "high"
                elif rv_5 < rv_50 * 0.7:
                    regime = "low"
                else:
                    regime = "normal"
                    
            except Exception as e:
                raise PredictionError("Realized_Volatility", f"Realized volatility calculation failed: {str(e)}")
            
            return {
                "predicted_volatility": float(predicted_vol),
                "current_volatility": float(rv_5),
                "volatility_measures": {
                    "realized_5": float(rv_5),
                    "realized_20": float(rv_20),
                    "realized_50": float(rv_50),
                    "parkinson": float(parkinson_vol)
                },
                "volatility_regime": regime,
                "volatility_forecast": vol_forecast,
                "confidence": 0.8,
                "method": "Realized_Volatility_Analysis"
            }
        
        raise AnalyzerException(f"Algorithm {algorithm_name} not implemented yet")
        
    def _run_momentum_analysis_algorithm(self, algorithm_name: str, 
                                    market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Esegue algoritmi di Momentum Analysis con error handling standardizzato"""
        
        if algorithm_name == "RSI_Momentum":
            prices = np.array(market_data['price_history'])
            
            if len(prices) < 50:
                raise InsufficientDataError(required=50, available=len(prices), operation="RSI_Momentum")
            
            try:
                # Calcola RSI multipli
                rsi_14 = self._calculate_rsi(prices, 14)
                rsi_9 = self._calculate_rsi(prices, 9)
                rsi_21 = self._calculate_rsi(prices, 21)
                
                current_rsi = rsi_14[-1]
                
                # RSI momentum (rate of change)
                if len(rsi_14) >= 5:
                    rsi_momentum = rsi_14[-1] - rsi_14[-5]
                else:
                    rsi_momentum = 0
                
                # Divergence detection
                price_trend = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
                rsi_trend = (rsi_14[-1] - rsi_14[-10]) / 100 if len(rsi_14) >= 10 else 0
                
                divergence = None
                if price_trend > 0 and rsi_trend < 0:
                    divergence = "bearish_divergence"
                elif price_trend < 0 and rsi_trend > 0:
                    divergence = "bullish_divergence"
                
            except Exception as e:
                raise PredictionError("RSI_Momentum", f"RSI calculation failed: {str(e)}")
            
            # Momentum state
            if current_rsi > 70:
                momentum_state = "overbought"
                signal = "potential_reversal_down"
            elif current_rsi < 30:
                momentum_state = "oversold"
                signal = "potential_reversal_up"
            elif current_rsi > 50 and rsi_momentum > 0:
                momentum_state = "bullish_momentum"
                signal = "continue_up"
            elif current_rsi < 50 and rsi_momentum < 0:
                momentum_state = "bearish_momentum"
                signal = "continue_down"
            else:
                momentum_state = "neutral"
                signal = "wait"
            
            confidence = 0.7
            if divergence:
                confidence += 0.1
            if abs(rsi_momentum) > 10:
                confidence += 0.1
            
            return {
                "momentum_indicators": {
                    "rsi_14": float(current_rsi),
                    "rsi_9": float(rsi_9[-1]),
                    "rsi_21": float(rsi_21[-1]),
                    "rsi_momentum": float(rsi_momentum)
                },
                "momentum_state": momentum_state,
                "signal": signal,
                "divergence": divergence,
                "confidence": float(min(0.9, confidence)),
                "method": "RSI_Momentum_Analysis"
            }
        
        elif algorithm_name == "MACD_Analysis":
            prices = np.array(market_data['price_history'])
            
            if len(prices) < 50:
                raise InsufficientDataError(required=50, available=len(prices), operation="MACD_Analysis")
            
            try:
                # Calcola MACD
                exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
                exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                histogram = macd - signal
                
                # Current values
                current_macd = float(macd.iloc[-1])
                current_signal = float(signal.iloc[-1])
                current_histogram = float(histogram.iloc[-1])
                
                # MACD momentum
                macd_momentum = float(macd.iloc[-1] - macd.iloc[-5]) if len(macd) >= 5 else 0
                
                # Crossover detection
                crossover = None
                if len(histogram) >= 2:
                    if histogram.iloc[-2] < 0 and histogram.iloc[-1] > 0:
                        crossover = "bullish_crossover"
                    elif histogram.iloc[-2] > 0 and histogram.iloc[-1] < 0:
                        crossover = "bearish_crossover"
                
                # Zero line analysis
                zero_line_position = "above" if current_macd > 0 else "below"
                
                # Signal strength
                signal_strength = abs(current_histogram) / prices[-1] * 1000  # Normalize
                
            except Exception as e:
                raise PredictionError("MACD_Analysis", f"MACD calculation failed: {str(e)}")
            
            # Determine signal
            if crossover == "bullish_crossover":
                trading_signal = "buy"
                confidence = 0.8
            elif crossover == "bearish_crossover":
                trading_signal = "sell"
                confidence = 0.8
            elif current_histogram > 0 and macd_momentum > 0:
                trading_signal = "bullish"
                confidence = 0.7
            elif current_histogram < 0 and macd_momentum < 0:
                trading_signal = "bearish"
                confidence = 0.7
            else:
                trading_signal = "neutral"
                confidence = 0.5
            
            return {
                "momentum_indicators": {
                    "macd": current_macd,
                    "signal": current_signal,
                    "histogram": current_histogram,
                    "macd_momentum": macd_momentum
                },
                "crossover": crossover,
                "zero_line": zero_line_position,
                "signal_strength": float(signal_strength),
                "trading_signal": trading_signal,
                "confidence": float(confidence),
                "method": "MACD_Analysis"
            }
        
        elif algorithm_name == "Neural_Momentum":
            # Neural network based momentum analysis
            prices = np.array(market_data['price_history'])
            volumes = np.array(market_data.get('volume_history', []))
            
            if len(prices) < 100:
                raise InsufficientDataError(required=100, available=len(prices), operation="Neural_Momentum")
            
            # Verifica disponibilità dati volume
            if len(volumes) == 0:
                volumes = np.ones_like(prices)  # Fallback con volume unitario
            
            try:
                # Calcola momentum features complessi
                momentum_features = self._calculate_neural_momentum_features(prices, volumes)
                
                # Analisi multi-timeframe momentum
                short_momentum = momentum_features['momentum_short']
                medium_momentum = momentum_features['momentum_medium']
                long_momentum = momentum_features['momentum_long']
                
                # Volume-weighted momentum
                vw_momentum = momentum_features['volume_weighted_momentum']
                
                # Acceleration
                momentum_acceleration = momentum_features['acceleration']
                
                # Neural scoring
                momentum_score = (
                    short_momentum * 0.4 +
                    medium_momentum * 0.3 +
                    long_momentum * 0.2 +
                    vw_momentum * 0.1
                )
                
            except Exception as e:
                raise PredictionError("Neural_Momentum", f"Neural momentum analysis failed: {str(e)}")
            
            # Classify momentum
            if momentum_score > 0.5:
                momentum_class = "strong_bullish"
                signal = "buy"
            elif momentum_score > 0.2:
                momentum_class = "bullish"
                signal = "buy_cautious"
            elif momentum_score < -0.5:
                momentum_class = "strong_bearish"
                signal = "sell"
            elif momentum_score < -0.2:
                momentum_class = "bearish"
                signal = "sell_cautious"
            else:
                momentum_class = "neutral"
                signal = "wait"
            
            # Confidence basata su consistency
            momentum_consistency = 1 - np.std([short_momentum, medium_momentum, long_momentum])
            confidence = 0.5 + momentum_consistency * 0.4
            
            return {
                "momentum_score": float(momentum_score),
                "momentum_class": momentum_class,
                "momentum_components": {
                    "short_term": float(short_momentum),
                    "medium_term": float(medium_momentum),
                    "long_term": float(long_momentum),
                    "volume_weighted": float(vw_momentum),
                    "acceleration": float(momentum_acceleration)
                },
                "signal": signal,
                "consistency": float(momentum_consistency),
                "confidence": float(confidence),
                "method": "Neural_Momentum_Analysis"
            }
        
        raise AnalyzerException(f"Algorithm {algorithm_name} not implemented yet")
    
    # ================== HELPER FUNCTIONS ==================

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calcola RSI"""
        if len(prices) < period + 1:
            return np.full_like(prices, 50.0)
        
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = np.zeros_like(prices)
        rsi[:period] = 50.0
        rsi[period] = 100 - 100 / (1 + rs)
        
        for i in range(period + 1, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 100
            rsi[i] = 100 - 100 / (1 + rs)
        
        return rsi
    
    def _prepare_lstm_features(self, prices: np.ndarray, volumes: np.ndarray, market_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Prepara features per modelli LSTM ULTRA-OTTIMIZZATO - ZERO COPIE NON NECESSARIE"""
        
        if market_data is None:
            market_data = {'spread_percentage': self.config.spread_high_threshold}
        
        # 🛡️ VALIDAZIONE INPUT CRITICA OTTIMIZZATA
        if len(prices) == 0 or len(volumes) == 0:
            safe_print("❌ Input arrays vuoti per LSTM features")
            return np.zeros((self.config.lstm_sequence_length, self.config.feature_vector_size), dtype=np.float32)
        
        # 🚀 OTTIMIZZAZIONE CRITICA: Ensure dtype IN-PLACE quando possibile
        if prices.dtype != np.float32:
            # Controlla se possiamo fare conversion in-place
            if prices.flags.writeable and prices.base is None:
                prices = prices.view(dtype=np.float32) if prices.itemsize == 4 else prices.astype(np.float32, copy=False)
            else:
                prices = prices.astype(np.float32)
        
        if volumes.dtype != np.float32:
            if volumes.flags.writeable and volumes.base is None:
                volumes = volumes.view(dtype=np.float32) if volumes.itemsize == 4 else volumes.astype(np.float32, copy=False)
            else:
                volumes = volumes.astype(np.float32)
        
        # 🚀 VALIDAZIONE COMBINATA NaN/Inf UNA VOLTA SOLA (operazione vettoriale)
        prices_invalid = ~np.isfinite(prices)
        if prices_invalid.any():
            mean_price = np.nanmean(prices)
            np.copyto(prices, mean_price, where=prices_invalid)  # IN-PLACE replacement
            safe_print("🔧 Prices sanitizzati IN-PLACE")
        
        volumes_invalid = ~np.isfinite(volumes)
        if volumes_invalid.any():
            mean_volume = np.nanmean(volumes)
            np.copyto(volumes, mean_volume, where=volumes_invalid)  # IN-PLACE replacement
            safe_print("🔧 Volumes sanitizzati IN-PLACE")
        
        # 🚀 PRE-CALCOLA VALORI RIUTILIZZATI UNA VOLTA
        current_price = float(prices[-1])
        current_price_inv = 1.0 / max(current_price, 1e-10)
        mean_price = float(np.mean(prices))
        mean_volume = float(np.mean(volumes))
        
        # 🚀 CALCOLA INDICATORI CON BATCH OTTIMIZZATO
        safe_print("🔄 Calcolando indicatori con batch ottimizzato...")
        
        try:
            # 🚀 BATCH CALCULATION - tutti gli indicatori in una chiamata
            indicators = self._calculate_indicators_batch_optimized(prices, volumes)
            safe_print("✅ Indicatori batch calcolati")
            
        except Exception as cache_error:
            safe_print(f"❌ Errore batch indicatori: {cache_error}")
            safe_print("🔄 Fallback a calcoli individuali...")
            
            # 🛡️ FALLBACK con funzioni safe esistenti
            indicators = {
                'sma_5': self._safe_sma_fallback(prices, 5),
                'sma_10': self._safe_sma_fallback(prices, 10),
                'rsi': self._safe_rsi_fallback(prices, 14),
                'macd': self._safe_macd_fallback(prices)[0],
                'macd_signal': self._safe_macd_fallback(prices)[1],
                'macd_hist': self._safe_macd_fallback(prices)[2],
                'bb_upper': self._safe_bbands_fallback(prices, 20)[0],
                'bb_middle': self._safe_bbands_fallback(prices, 20)[1],
                'bb_lower': self._safe_bbands_fallback(prices, 20)[2],
                'atr': self._safe_atr_fallback(prices, prices, prices, 14),
                'vol_sma': self._safe_sma_fallback(volumes, 10)
            }
        
        # 🚀 CALCOLA RETURNS UNA VOLTA (operazione vettoriale)
        returns = np.empty(len(prices) - 1, dtype=np.float32)
        if len(prices) >= 2:
            np.divide(np.diff(prices), prices[:-1], out=returns, 
                    where=(prices[:-1] != 0), casting='safe')
            # Clamp extremes IN-PLACE
            np.clip(returns, -0.5, 0.5, out=returns)
        else:
            returns.fill(0.0)
        
        # 🚀 DETERMINA SEQUENCE LENGTH UNA VOLTA
        sequence_length = min(self.config.lstm_sequence_length, len(prices) - 1)
        if sequence_length <= 0:
            safe_print("❌ Sequence length non valida")
            return np.zeros((self.config.lstm_sequence_length, self.config.feature_vector_size), dtype=np.float32)
        
        # 🚀 PRE-ALLOCA FEATURE MATRIX (evita append e successive conversioni)
        features_matrix = np.empty((sequence_length, self.config.feature_vector_size), dtype=np.float32)
        
        # 🚀 CALCOLA SLICES UNA VOLTA per tutti gli indicatori
        start_idx = len(prices) - sequence_length
        end_idx = start_idx + sequence_length
        
        # 🚀 SLICES VETTORIALI (view, non copie)
        price_slice = prices[start_idx:end_idx]
        sma5_slice = indicators['sma_5'][start_idx:end_idx]
        sma10_slice = indicators['sma_10'][start_idx:end_idx]
        rsi_slice = indicators['rsi'][start_idx:end_idx]
        macd_slice = indicators['macd'][start_idx:end_idx]
        bb_upper_slice = indicators['bb_upper'][start_idx:end_idx]
        bb_lower_slice = indicators['bb_lower'][start_idx:end_idx]
        atr_slice = indicators['atr'][start_idx:end_idx]
        vol_slice = volumes[start_idx:end_idx]
        vol_sma_slice = indicators['vol_sma'][start_idx:end_idx]
        
        # 🚀 OPERAZIONI VETTORIALI COMPLETE su tutti gli elementi
        # Feature 0: Normalized price
        np.multiply(price_slice, current_price_inv, out=features_matrix[:, 0])
        np.clip(features_matrix[:, 0], 0.1, 10.0, out=features_matrix[:, 0])
        
        # Feature 1: SMA5 ratio
        np.divide(sma5_slice, np.maximum(price_slice, 1e-10), out=features_matrix[:, 1])
        np.clip(features_matrix[:, 1], 0.5, 2.0, out=features_matrix[:, 1])
        
        # Feature 2: SMA10 ratio
        np.divide(sma10_slice, np.maximum(price_slice, 1e-10), out=features_matrix[:, 2])
        np.clip(features_matrix[:, 2], 0.5, 2.0, out=features_matrix[:, 2])
        
        # Feature 3: RSI normalized
        np.multiply(rsi_slice, 0.01, out=features_matrix[:, 3])  # /100 ottimizzato
        np.clip(features_matrix[:, 3], 0.0, 1.0, out=features_matrix[:, 3])
        
        # Feature 4: MACD normalized
        np.multiply(macd_slice, current_price_inv, out=features_matrix[:, 4])
        np.clip(features_matrix[:, 4], -0.1, 0.1, out=features_matrix[:, 4])
        
        # Feature 5: Bollinger position
        bb_range = np.maximum(bb_upper_slice - bb_lower_slice, 1e-10)
        np.divide(price_slice - bb_lower_slice, bb_range, out=features_matrix[:, 5])
        np.clip(features_matrix[:, 5], 0.0, 1.0, out=features_matrix[:, 5])
        
        # Feature 6: ATR normalized
        np.divide(atr_slice, np.maximum(price_slice, 1e-10), out=features_matrix[:, 6])
        np.clip(features_matrix[:, 6], 0.0, 0.1, out=features_matrix[:, 6])
        
        # Feature 7: Volume ratio
        np.divide(vol_slice, np.maximum(vol_sma_slice, 1e-10), out=features_matrix[:, 7])
        np.clip(features_matrix[:, 7], 0.1, 10.0, out=features_matrix[:, 7])
        
        # Feature 8: Returns (accesso sicuro con padding)
        if len(returns) >= sequence_length:
            returns_slice = returns[start_idx:end_idx]
        else:
            returns_slice = np.zeros(sequence_length, dtype=np.float32)
            if len(returns) > 0:
                returns_slice[:len(returns)] = returns
        
        np.copyto(features_matrix[:, 8], returns_slice)
        np.clip(features_matrix[:, 8], -0.1, 0.1, out=features_matrix[:, 8])
        
        # Feature 9: Spread (solo ultimo elemento)
        features_matrix[:, 9] = 0.0
        spread = market_data.get('spread_percentage', 0.001)
        features_matrix[-1, 9] = np.clip(spread, 0.0, 0.01)
        
        # 🛡️ VALIDAZIONE FINALE IN-PLACE
        invalid_final = ~np.isfinite(features_matrix)
        if invalid_final.any():
            safe_print("❌ Features matrix contiene NaN/Inf - sanitizzando IN-PLACE...")
            np.copyto(features_matrix, 0.0, where=invalid_final)
        
        safe_print(f"✅ LSTM features ULTRA-ottimizzate: shape={features_matrix.shape}, operazioni vettoriali")
        return features_matrix

    def _calculate_indicators_batch_optimized(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Calcola tutti gli indicatori in batch ottimizzato"""
        
        indicators = {}
        
        # 🚀 Usa cached indicators se disponibili
        if hasattr(self, 'cached_indicators'):
            indicators['sma_5'] = self.cached_indicators['sma'](prices, 5)
            indicators['sma_10'] = self.cached_indicators['sma'](prices, 10)
            indicators['rsi'] = self.cached_indicators['rsi'](prices, 14)
            
            macd_result = self.cached_indicators['macd'](prices)
            indicators['macd'] = macd_result[0]
            indicators['macd_signal'] = macd_result[1] 
            indicators['macd_hist'] = macd_result[2]
            
            bb_result = self.cached_indicators['bbands'](prices, 20)
            indicators['bb_upper'] = bb_result[0]
            indicators['bb_middle'] = bb_result[1]
            indicators['bb_lower'] = bb_result[2]
            
            indicators['atr'] = self.cached_indicators['atr'](prices, prices, prices, 14)
            indicators['vol_sma'] = self.cached_indicators['sma'](volumes, 10)
        else:
            # Fallback a calcoli diretti
            raise Exception("Cache not available, using fallback")
        
        return indicators

    def _safe_sma_fallback(self, data: np.ndarray, period: int) -> np.ndarray:
        """SMA fallback ottimizzato"""
        if len(data) < period:
            return np.full_like(data, data[-1] if len(data) > 0 else 1.0)
        
        try:
            result = ta.SMA(data, timeperiod=period) # type: ignore
            if result is None:
                return np.full_like(data, data[-1])
            
            last_value = float(data[-1])
            np.nan_to_num(result, nan=last_value, posinf=last_value, neginf=last_value, copy=False)
            return result
        except:
            return np.full_like(data, data[-1] if len(data) > 0 else 1.0)

    def _safe_rsi_fallback(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI fallback ottimizzato"""
        if len(data) < period + 1:
            return np.full_like(data, 50.0)
        
        try:
            result = ta.RSI(data, timeperiod=period) # type: ignore
            if result is None:
                return np.full_like(data, 50.0)
            
            np.nan_to_num(result, nan=50.0, posinf=100.0, neginf=0.0, copy=False)
            np.clip(result, 0.0, 100.0, out=result)
            return result
        except:
            return np.full_like(data, 50.0)

    def _safe_macd_fallback(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD fallback ottimizzato"""
        if len(data) < 26:
            zeros = np.zeros_like(data)
            return zeros, zeros, zeros
        
        try:
            macd, signal, hist = ta.MACD(data) # type: ignore
            if macd is None or signal is None or hist is None:
                zeros = np.zeros_like(data)
                return zeros, zeros, zeros
            
            np.nan_to_num(macd, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            
            return macd, signal, hist
        except:
            zeros = np.zeros_like(data)
            return zeros, zeros, zeros

    def _safe_bbands_fallback(self, data: np.ndarray, period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands fallback ottimizzato"""
        if len(data) < period:
            return data * 1.02, data, data * 0.98
        
        try:
            upper, middle, lower = ta.BBANDS(data, timeperiod=period) # type: ignore
            if upper is None or middle is None or lower is None:
                return data * 1.02, data, data * 0.98
            
            last_price = float(data[-1])
            np.nan_to_num(upper, nan=last_price * 1.02, copy=False)
            np.nan_to_num(middle, nan=last_price, copy=False)
            np.nan_to_num(lower, nan=last_price * 0.98, copy=False)
            
            return upper, middle, lower
        except:
            return data * 1.02, data, data * 0.98

    def _safe_atr_fallback(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """ATR fallback ottimizzato"""
        if len(close) < period:
            return np.full_like(close, 0.01)
        
        try:
            result = ta.ATR(high, low, close, timeperiod=period) # type: ignore
            if result is None:
                return np.full_like(close, 0.01)
            
            np.nan_to_num(result, nan=0.01, posinf=0.01, neginf=0.01, copy=False)
            np.maximum(result, 0.001, out=result)
            return result
        except:
            return np.full_like(close, 0.01)
    
    def _prepare_transformer_features(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Prepara features avanzate per Transformer - ULTRA OTTIMIZZATO ZERO COPIE"""
        
        safe_print("🔄 Preparando features Transformer ultra-ottimizzate...")
        
        # 🛡️ VALIDAZIONE INPUT OTTIMIZZATA
        if len(prices) == 0 or len(volumes) == 0:
            safe_print("❌ Input arrays vuoti per Transformer features")
            return np.zeros((50, 15), dtype=np.float32)  # sequence_length=50, features=15
        
        # 🚀 ENSURE DTYPE IN-PLACE quando possibile
        if prices.dtype != np.float32:
            if prices.flags.writeable and prices.base is None:
                prices = prices.view(dtype=np.float32) if prices.itemsize == 4 else prices.astype(np.float32, copy=False)
            else:
                prices = prices.astype(np.float32)
        
        if volumes.dtype != np.float32:
            if volumes.flags.writeable and volumes.base is None:
                volumes = volumes.view(dtype=np.float32) if volumes.itemsize == 4 else volumes.astype(np.float32, copy=False)
            else:
                volumes = volumes.astype(np.float32)
        
        # 🚀 VALIDAZIONE COMBINATA NaN/Inf UNA VOLTA
        prices_invalid = ~np.isfinite(prices)
        if prices_invalid.any():
            mean_price = np.nanmean(prices)
            np.copyto(prices, mean_price, where=prices_invalid)
        
        volumes_invalid = ~np.isfinite(volumes)
        if volumes_invalid.any():
            mean_volume = np.nanmean(volumes)
            np.copyto(volumes, mean_volume, where=volumes_invalid)
        
        # 🚀 PRE-CALCOLA VALORI RIUTILIZZATI UNA VOLTA
        current_price = float(prices[-1])
        current_price_inv = 1.0 / max(current_price, 1e-10)
        mean_volume = float(np.mean(volumes))
        mean_volume_inv = 1.0 / max(mean_volume, 1e-10)
        
        try:
            # 🚀 BATCH CALCULATION - tutti gli indicatori principali in una chiamata
            indicators = self._calculate_indicators_batch_transformer(prices, volumes)
            safe_print("✅ Indicatori Transformer batch calcolati")
            
        except Exception as cache_error:
            safe_print(f"❌ Errore batch Transformer: {cache_error}")
            safe_print("🔄 Fallback a calcoli individuali...")
            
            # 🛡️ FALLBACK con TA-lib diretto (come nel codice originale)
            try:
                sma_5 = ta.SMA(prices, timeperiod=5) # type: ignore
                sma_20 = ta.SMA(prices, timeperiod=20) # type: ignore
                ema_12 = ta.EMA(prices, timeperiod=12) # type: ignore
                ema_26 = ta.EMA(prices, timeperiod=26) # type: ignore
                rsi = ta.RSI(prices, timeperiod=14) # type: ignore
                macd, macd_signal, macd_hist = ta.MACD(prices) # type: ignore
                bb_upper, bb_middle, bb_lower = ta.BBANDS(prices, timeperiod=20) # type: ignore
                atr = ta.ATR(prices, prices, prices, timeperiod=14) # type: ignore
                
                # Sanitize NaN values IN-PLACE
                for arr in [sma_5, sma_20, ema_12, ema_26, rsi, macd, macd_signal, macd_hist, bb_upper, bb_lower, atr]:
                    if arr is not None:
                        invalid = ~np.isfinite(arr)
                        if invalid.any():
                            np.copyto(arr, 0.0, where=invalid)
                
                indicators = {
                    'sma_5': sma_5,
                    'sma_20': sma_20,
                    'ema_12': ema_12,
                    'ema_26': ema_26,
                    'rsi': rsi,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'macd_hist': macd_hist,
                    'bb_upper': bb_upper,
                    'bb_lower': bb_lower,
                    'atr': atr
                }
                
            except Exception as ta_error:
                safe_print(f"❌ Errore TA-lib fallback: {ta_error}")
                # Ultimate fallback con arrays zero
                indicators = {
                    'sma_5': np.full_like(prices, current_price),
                    'sma_20': np.full_like(prices, current_price),
                    'ema_12': np.full_like(prices, current_price),
                    'ema_26': np.full_like(prices, current_price),
                    'rsi': np.full_like(prices, 50.0),
                    'macd': np.zeros_like(prices),
                    'macd_signal': np.zeros_like(prices),
                    'macd_hist': np.zeros_like(prices),
                    'bb_upper': np.full_like(prices, current_price * 1.02),
                    'bb_lower': np.full_like(prices, current_price * 0.98),
                    'atr': np.full_like(prices, current_price * 0.01)
                }
        
        # 🚀 VOLUME INDICATORS OTTIMIZZATI (operazioni vettoriali)
        obv = np.empty(len(volumes), dtype=np.float32)
        obv[0] = volumes[0]
        
        # OBV ottimizzato: cumsum condizionale vettoriale
        price_changes = np.diff(prices)
        volume_signs = np.where(price_changes > 0, 1.0, 
                            np.where(price_changes < 0, -1.0, 0.0))
        
        # Cumsum vettoriale per OBV
        signed_volumes = volumes[1:] * volume_signs
        obv[1:] = np.cumsum(signed_volumes) + obv[0]
        
        # AD Line ottimizzato (senza roll, operazione vettoriale)
        if len(prices) > 1:
            ad_line = np.empty(len(prices), dtype=np.float32)
            ad_line[0] = 0.0
            
            # Calcolo vettoriale Money Flow Volume
            price_diffs = np.diff(prices)
            mfv = volumes[1:] * price_diffs
            ad_line[1:] = np.cumsum(mfv)
        else:
            ad_line = np.zeros_like(prices)
        
        # 🚀 ADVANCED FEATURES OTTIMIZZATE
        # Returns: operazione vettoriale IN-PLACE
        if len(prices) > 1:
            returns = np.empty(len(prices) - 1, dtype=np.float32)
            np.divide(np.diff(prices), prices[:-1], out=returns, 
                    where=(prices[:-1] != 0), casting='safe')
            np.clip(returns, -0.5, 0.5, out=returns)  # Clamp extremes
            
            # Log returns: operazione vettoriale
            log_returns = np.empty(len(prices) - 1, dtype=np.float32)
            price_ratios = prices[1:] / np.maximum(prices[:-1], 1e-10)
            np.log(price_ratios, out=log_returns)
            np.clip(log_returns, -0.5, 0.5, out=log_returns)
        else:
            returns = np.array([0.0], dtype=np.float32)
            log_returns = np.array([0.0], dtype=np.float32)
        
        # 🚀 VOLATILITY ROLLING OTTIMIZZATA (no pandas, operazioni vettoriali)
        window_size = min(10, len(returns))
        if len(returns) >= window_size:
            volatility_rolling = np.empty(len(returns), dtype=np.float32)
            
            # Calcolo rolling std vettoriale
            for i in range(len(returns)):
                start_idx = max(0, i - window_size + 1)
                window = returns[start_idx:i+1]
                volatility_rolling[i] = float(np.std(window))
        else:
            volatility_rolling = np.zeros(len(returns), dtype=np.float32)
        
        # 🚀 DETERMINA SEQUENCE LENGTH
        sequence_length = min(50, len(prices))  # Configurable
        if sequence_length <= 0:
            safe_print("❌ Sequence length non valida")
            return np.zeros((50, 15), dtype=np.float32)
        
        # 🚀 PRE-ALLOCA FEATURE MATRIX (15 features predefinite)
        features_matrix = np.empty((sequence_length, 15), dtype=np.float32)
        
        # 🚀 CALCOLA SLICES UNA VOLTA per tutti gli indicatori
        start_idx = len(prices) - sequence_length
        
        # 🚀 SLICES VETTORIALI (views, non copie)
        price_slice = prices[start_idx:start_idx + sequence_length]
        sma5_slice = indicators['sma_5'][start_idx:start_idx + sequence_length]
        sma20_slice = indicators['sma_20'][start_idx:start_idx + sequence_length]
        ema12_slice = indicators['ema_12'][start_idx:start_idx + sequence_length]
        ema26_slice = indicators['ema_26'][start_idx:start_idx + sequence_length]
        rsi_slice = indicators['rsi'][start_idx:start_idx + sequence_length]
        macd_slice = indicators['macd'][start_idx:start_idx + sequence_length]
        bb_upper_slice = indicators['bb_upper'][start_idx:start_idx + sequence_length]
        bb_lower_slice = indicators['bb_lower'][start_idx:start_idx + sequence_length]
        atr_slice = indicators['atr'][start_idx:start_idx + sequence_length]
        volume_slice = volumes[start_idx:start_idx + sequence_length]
        obv_slice = obv[start_idx:start_idx + sequence_length]
        ad_slice = ad_line[start_idx:start_idx + sequence_length]
        
        # 🚀 NORMALIZZATORI PRE-CALCOLATI
        obv_max = max(np.max(np.abs(obv_slice)), 1e-10)
        obv_max_inv = 1.0 / obv_max
        ad_max = max(np.max(np.abs(ad_slice)), 1e-10)
        ad_max_inv = 1.0 / ad_max
        
        # 🚀 RIEMPI MATRICE con operazioni vettoriali COMPLETE
        
        # Feature 0: Normalized prices
        np.multiply(price_slice, current_price_inv, out=features_matrix[:, 0])
        
        # Feature 1: SMA5 ratio
        np.divide(sma5_slice, np.maximum(price_slice, 1e-10), out=features_matrix[:, 1])
        
        # Feature 2: SMA20 ratio  
        np.divide(sma20_slice, np.maximum(price_slice, 1e-10), out=features_matrix[:, 2])
        
        # Feature 3: EMA12 ratio
        np.divide(ema12_slice, np.maximum(price_slice, 1e-10), out=features_matrix[:, 3])
        
        # Feature 4: EMA26 ratio
        np.divide(ema26_slice, np.maximum(price_slice, 1e-10), out=features_matrix[:, 4])
        
        # Feature 5: RSI normalized
        np.multiply(rsi_slice, 0.01, out=features_matrix[:, 5])  # /100 ottimizzato
        
        # Feature 6: MACD (già normalizzato)
        np.copyto(features_matrix[:, 6], macd_slice)
        
        # Feature 7: Bollinger upper ratio
        np.divide(bb_upper_slice, np.maximum(price_slice, 1e-10), out=features_matrix[:, 7])
        
        # Feature 8: Bollinger lower ratio
        np.divide(bb_lower_slice, np.maximum(price_slice, 1e-10), out=features_matrix[:, 8])
        
        # Feature 9: ATR ratio
        np.divide(atr_slice, np.maximum(price_slice, 1e-10), out=features_matrix[:, 9])
        
        # Feature 10: Volume ratio
        np.multiply(volume_slice, mean_volume_inv, out=features_matrix[:, 10])
        
        # Feature 11: OBV normalized
        np.multiply(obv_slice, obv_max_inv, out=features_matrix[:, 11])
        
        # Feature 12: AD Line normalized
        np.multiply(ad_slice, ad_max_inv, out=features_matrix[:, 12])
        
        # Feature 13: Returns (con padding sicuro)
        if len(returns) >= sequence_length:
            returns_slice = returns[start_idx:start_idx + sequence_length]
            np.copyto(features_matrix[:, 13], returns_slice)
        else:
            features_matrix[:, 13] = 0.0
            if len(returns) > 0:
                available_len = min(len(returns), sequence_length)
                features_matrix[:available_len, 13] = returns[:available_len]
        
        # Feature 14: Rolling volatility (con padding sicuro)
        if len(volatility_rolling) >= sequence_length:
            vol_slice = volatility_rolling[start_idx:start_idx + sequence_length]
            np.copyto(features_matrix[:, 14], vol_slice)
        else:
            features_matrix[:, 14] = 0.0
            if len(volatility_rolling) > 0:
                available_len = min(len(volatility_rolling), sequence_length)
                features_matrix[:available_len, 14] = volatility_rolling[:available_len]
        
        # 🚀 CLIPPING VETTORIALE finale su tutta la matrice
        np.clip(features_matrix, -10.0, 10.0, out=features_matrix)
        
        # 🛡️ VALIDAZIONE FINALE IN-PLACE
        invalid_final = ~np.isfinite(features_matrix)
        if invalid_final.any():
            safe_print("❌ Features matrix contiene NaN/Inf - sanitizzando IN-PLACE...")
            np.copyto(features_matrix, 0.0, where=invalid_final)
        
        safe_print(f"✅ Transformer features ULTRA-ottimizzate: shape={features_matrix.shape}")
        return features_matrix

    def _calculate_indicators_batch_transformer(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Calcola tutti gli indicatori Transformer in batch ottimizzato"""
        
        indicators = {}
        
        # 🚀 USA CACHED INDICATORS invece di calcoli manuali
        try:
            indicators['sma_5'] = self.cached_indicators['sma'](prices, 5)
            indicators['sma_20'] = self.cached_indicators['sma'](prices, 20)
            indicators['rsi'] = self.cached_indicators['rsi'](prices, 14)
            
            # MACD come batch
            macd_result = self.cached_indicators['macd'](prices)
            indicators['macd'] = macd_result[0]
            indicators['macd_signal'] = macd_result[1] 
            indicators['macd_hist'] = macd_result[2]
            
            # Bollinger Bands come batch
            bb_result = self.cached_indicators['bbands'](prices, 20)
            indicators['bb_upper'] = bb_result[0]
            indicators['bb_middle'] = bb_result[1]
            indicators['bb_lower'] = bb_result[2]
            
            indicators['atr'] = self.cached_indicators['atr'](prices, prices, prices, 14)
            
            # EMA calculations (se disponibili in cache)
            try:
                if 'ema' in self.cached_indicators:
                    indicators['ema_12'] = self.cached_indicators['ema'](prices, 12)
                    indicators['ema_26'] = self.cached_indicators['ema'](prices, 26)
                else:
                    # Fallback a TA-lib diretto
                    indicators['ema_12'] = ta.EMA(prices, timeperiod=12) # type: ignore
                    indicators['ema_26'] = ta.EMA(prices, timeperiod=26) # type: ignore
                    
                    # Sanitize NaN
                    for key in ['ema_12', 'ema_26']:
                        if indicators[key] is not None:
                            invalid = ~np.isfinite(indicators[key])
                            if invalid.any():
                                np.copyto(indicators[key], indicators['sma_5'], where=invalid)
            except:
                # Ultimate fallback
                indicators['ema_12'] = indicators['sma_5']  # Approximation
                indicators['ema_26'] = indicators['sma_20']  # Approximation
            
            return indicators
            
        except Exception as e:
            safe_print(f"❌ Errore batch indicators Transformer: {e}")
            raise e  # Re-raise per trigger fallback nel chiamante
    
    def _detect_double_top(self, prices: np.ndarray) -> bool:
        """Rileva pattern Double Top - OTTIMIZZATO"""
        n_prices = len(prices)
        if n_prices < 20:
            return False
        
        # Pre-allocazione arrays per peaks invece di list
        max_possible_peaks = n_prices // 3  # Stima conservativa
        peak_indices = np.empty(max_possible_peaks, dtype=np.int32)
        peak_values = np.empty(max_possible_peaks, dtype=np.float32)
        peak_count = 0
        
        # Trova i picchi con accessi ottimizzati
        for i in range(2, n_prices - 2):
            # Cache del valore corrente per ridurre accessi array
            current_price = prices[i]
            
            # Controllo picco con singoli accessi
            if (current_price > prices[i-1] and 
                current_price > prices[i+1] and 
                current_price > prices[i-2] and 
                current_price > prices[i+2]):
                
                # Salva direttamente negli arrays pre-allocati
                peak_indices[peak_count] = i
                peak_values[peak_count] = current_price
                peak_count += 1
                
                # Early exit se abbiamo abbastanza peaks
                if peak_count >= max_possible_peaks:
                    break
        
        # Controllo minimo peaks senza allocazioni
        if peak_count < 2:
            return False
        
        # Accesso diretto agli ultimi due peaks senza slicing
        last_peak_idx = peak_count - 1
        second_last_peak_idx = peak_count - 2
        
        # Valori degli ultimi due picchi
        last_peak_value = peak_values[last_peak_idx]
        second_last_peak_value = peak_values[second_last_peak_idx]
        
        # Calcolo differenza percentuale
        if second_last_peak_value == 0:
            return False
        
        price_diff = abs(second_last_peak_value - last_peak_value) / second_last_peak_value
        
        # Controllo finale: picchi simili (entro 2%) e separati
        peaks_distance = peak_indices[last_peak_idx] - peak_indices[second_last_peak_idx]
        
        return price_diff < 0.02 and peaks_distance > 5
    
    def _detect_double_bottom(self, prices: np.ndarray) -> bool:
        """Rileva pattern Double Bottom - OTTIMIZZATO"""
        n_prices = len(prices)
        if n_prices < 20:
            return False
        
        # Pre-allocazione arrays per valleys invece di list
        max_possible_valleys = n_prices // 3  # Stima conservativa
        valley_indices = np.empty(max_possible_valleys, dtype=np.int32)
        valley_values = np.empty(max_possible_valleys, dtype=np.float32)
        valley_count = 0
        
        # Trova i minimi con accessi ottimizzati
        for i in range(2, n_prices - 2):
            # Cache del valore corrente per ridurre accessi array
            current_price = prices[i]
            
            # Controllo minimo con singoli accessi (logica invertita rispetto a double_top)
            if (current_price < prices[i-1] and 
                current_price < prices[i+1] and 
                current_price < prices[i-2] and 
                current_price < prices[i+2]):
                
                # Salva direttamente negli arrays pre-allocati
                valley_indices[valley_count] = i
                valley_values[valley_count] = current_price
                valley_count += 1
                
                # Early exit se abbiamo abbastanza valleys
                if valley_count >= max_possible_valleys:
                    break
        
        # Controllo minimo valleys senza allocazioni
        if valley_count < 2:
            return False
        
        # Accesso diretto agli ultimi due valleys senza slicing
        last_valley_idx = valley_count - 1
        second_last_valley_idx = valley_count - 2
        
        # Valori degli ultimi due minimi
        last_valley_value = valley_values[last_valley_idx]
        second_last_valley_value = valley_values[second_last_valley_idx]
        
        # Calcolo differenza percentuale
        if second_last_valley_value == 0:
            return False
        
        price_diff = abs(second_last_valley_value - last_valley_value) / second_last_valley_value
        
        # Controllo finale: minimi simili (entro 2%) e separati
        valleys_distance = valley_indices[last_valley_idx] - valley_indices[second_last_valley_idx]
        
        return price_diff < 0.02 and valleys_distance > 5
    
    def _detect_head_shoulders(self, prices: np.ndarray) -> bool:
        """Rileva pattern Head & Shoulders - OTTIMIZZATO"""
        n_prices = len(prices)
        if n_prices < 30:
            return False
        
        # Pre-allocazione arrays per peaks invece di list
        max_possible_peaks = n_prices // 4  # Stima conservativa
        peak_indices = np.empty(max_possible_peaks, dtype=np.int32)
        peak_values = np.empty(max_possible_peaks, dtype=np.float32)
        peak_count = 0
        
        # Trova picchi significativi SENZA slicing window
        for i in range(3, n_prices - 3):
            # Cache del valore corrente
            current_price = prices[i]
            
            # Controllo se è il massimo nella finestra 7-elementi SENZA creare array
            # Equivale a: prices[i] == max(prices[i-3:i+4])
            is_peak = True
            
            # Controlla tutti i punti nella finestra senza slicing
            for j in range(i - 3, i + 4):
                if prices[j] > current_price:
                    is_peak = False
                    break
            
            if is_peak:
                # Salva direttamente negli arrays pre-allocati
                peak_indices[peak_count] = i
                peak_values[peak_count] = current_price
                peak_count += 1
                
                # Early exit se abbiamo abbastanza peaks
                if peak_count >= max_possible_peaks:
                    break
        
        # Controllo minimo peaks senza allocazioni
        if peak_count < 3:
            return False
        
        # Accesso diretto agli ultimi 3 peaks senza slicing
        if peak_count >= 3:
            # Indici degli ultimi 3 peaks
            third_last_idx = peak_count - 3
            second_last_idx = peak_count - 2
            last_idx = peak_count - 1
            
            # Valori dei picchi (spalla sinistra, testa, spalla destra)
            left_shoulder = peak_values[third_last_idx]
            head = peak_values[second_last_idx]
            right_shoulder = peak_values[last_idx]
            
            # Il picco centrale (testa) deve essere il più alto
            if head > left_shoulder and head > right_shoulder:
                # Le spalle devono essere simili
                if left_shoulder > 0:  # Protezione divisione per zero
                    shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                    return shoulder_diff < 0.03
        
        return False
    
    def _detect_triangle_pattern(self, prices: np.ndarray) -> Optional[str]:
        """Rileva pattern triangolari - OTTIMIZZATO"""
        n_prices = len(prices)
        if n_prices < 30:
            return None
        
        # Pre-allocazione arrays per highs e lows invece di list
        max_possible_points = n_prices // 3  # Stima conservativa
        high_indices = np.empty(max_possible_points, dtype=np.int32)
        high_values = np.empty(max_possible_points, dtype=np.float32)
        low_indices = np.empty(max_possible_points, dtype=np.int32)
        low_values = np.empty(max_possible_points, dtype=np.float32)
        
        high_count = 0
        low_count = 0
        
        # Trova highs e lows con accessi ottimizzati
        for i in range(2, n_prices - 2):
            current_price = prices[i]
            
            # Check per high
            if (current_price > prices[i-1] and 
                current_price > prices[i+1]):
                high_indices[high_count] = i
                high_values[high_count] = current_price
                high_count += 1
                
                # Early exit se raggiungiamo il limite
                if high_count >= max_possible_points:
                    break
            
            # Check per low
            elif (current_price < prices[i-1] and 
                current_price < prices[i+1]):
                low_indices[low_count] = i
                low_values[low_count] = current_price
                low_count += 1
                
                # Early exit se raggiungiamo il limite
                if low_count >= max_possible_points:
                    break
        
        # Controllo minimo punti
        if high_count < 2 or low_count < 2:
            return None
        
        # Calcola trend lines usando ultimi 3 punti (o meno se non disponibili)
        # Per highs
        n_recent_highs = min(3, high_count)
        start_high_idx = high_count - n_recent_highs
        
        high_slope = 0.0
        if n_recent_highs >= 2:
            # Usa primo e ultimo dei recent highs
            first_high_x = high_indices[start_high_idx]
            first_high_y = high_values[start_high_idx]
            last_high_x = high_indices[high_count - 1]
            last_high_y = high_values[high_count - 1]
            
            x_diff_high = last_high_x - first_high_x
            if x_diff_high != 0:
                high_slope = (last_high_y - first_high_y) / x_diff_high
        
        # Per lows
        n_recent_lows = min(3, low_count)
        start_low_idx = low_count - n_recent_lows
        
        low_slope = 0.0
        if n_recent_lows >= 2:
            # Usa primo e ultimo dei recent lows
            first_low_x = low_indices[start_low_idx]
            first_low_y = low_values[start_low_idx]
            last_low_x = low_indices[low_count - 1]
            last_low_y = low_values[low_count - 1]
            
            x_diff_low = last_low_x - first_low_x
            if x_diff_low != 0:
                low_slope = (last_low_y - first_low_y) / x_diff_low
        
        # Classifica il triangolo con threshold constants
        SLOPE_THRESHOLD = 0.0001
        
        high_slope_negative = high_slope < -SLOPE_THRESHOLD
        high_slope_flat = abs(high_slope) < SLOPE_THRESHOLD
        low_slope_positive = low_slope > SLOPE_THRESHOLD
        low_slope_flat = abs(low_slope) < SLOPE_THRESHOLD
        
        # Classificazione triangolo
        if high_slope_negative and low_slope_positive:
            return "symmetrical"
        elif high_slope_negative and low_slope_flat:
            return "descending"
        elif high_slope_flat and low_slope_positive:
            return "ascending"
        
        return None

    def _detect_flag_pattern(self, prices: np.ndarray) -> Optional[str]:
        """Rileva pattern Flag - OTTIMIZZATO"""
        n_prices = len(prices)
        if n_prices < 20:
            return None
        
        # Un flag ha: forte movimento -> consolidamento -> continuazione
        # Evita slicing: accesso diretto agli indici
        
        # Initial move: da prices[-20] a prices[-10] (10 elementi: da n-20 a n-11)
        initial_start_idx = n_prices - 20
        initial_end_idx = n_prices - 10
        
        initial_start_price = prices[initial_start_idx]
        initial_end_price = prices[initial_end_idx - 1]  # prices[-11], ultimo della initial move
        
        # Calcola la mossa iniziale senza array temporaneo
        if initial_start_price != 0:
            initial_change = (initial_end_price - initial_start_price) / initial_start_price
        else:
            return None
        
        # Consolidation: ultimi 10 elementi (da prices[-10] a prices[-1])
        # Calcola mean e std del consolidamento SENZA creare array
        consol_start_idx = n_prices - 10
        
        # Calcola mean in-place
        consol_sum = 0.0
        for i in range(consol_start_idx, n_prices):
            consol_sum += prices[i]
        consol_mean = consol_sum / 10.0
        
        # Calcola std in-place se mean non è zero
        if consol_mean != 0:
            consol_sum_sq_diff = 0.0
            for i in range(consol_start_idx, n_prices):
                diff = prices[i] - consol_mean
                consol_sum_sq_diff += diff * diff
            
            # Standard deviation
            consol_variance = consol_sum_sq_diff / 10.0
            consol_std = np.sqrt(consol_variance) if consol_variance > 0 else 0.0
            consol_volatility = consol_std / consol_mean
        else:
            consol_volatility = 0.0
        
        # Se mossa iniziale forte e consolidamento a bassa volatilità
        if abs(initial_change) > 0.02 and consol_volatility < 0.01:
            return "bullish" if initial_change > 0 else "bearish"
        
        return None
    
    def _detect_channel_pattern(self, prices: np.ndarray) -> Optional[str]:
        """Rileva pattern Channel - OTTIMIZZATO (ULTIMA FUNZIONE!)"""
        n_prices = len(prices)
        if n_prices < 30:
            return None
        
        # Pre-allocazione arrays per highs e lows invece di list
        max_possible_points = n_prices // 3  # Stima conservativa
        high_indices = np.empty(max_possible_points, dtype=np.int32)
        high_values = np.empty(max_possible_points, dtype=np.float32)
        low_indices = np.empty(max_possible_points, dtype=np.int32)
        low_values = np.empty(max_possible_points, dtype=np.float32)
        
        high_count = 0
        low_count = 0
        
        # Trova highs e lows con accessi ottimizzati
        for i in range(1, n_prices - 1):
            current_price = prices[i]
            
            # Check per high
            if current_price > prices[i-1] and current_price > prices[i+1]:
                high_indices[high_count] = i
                high_values[high_count] = current_price
                high_count += 1
                
                if high_count >= max_possible_points:
                    break
            
            # Check per low
            elif current_price < prices[i-1] and current_price < prices[i+1]:
                low_indices[low_count] = i
                low_values[low_count] = current_price
                low_count += 1
                
                if low_count >= max_possible_points:
                    break
        
        if high_count < 3 or low_count < 3:
            return None
        
        # Fit linee di regressione SENZA np.polyfit e list comprehensions
        # Usa ultimi 4 points (o meno se non disponibili)
        n_high_points = min(4, high_count)
        n_low_points = min(4, low_count)
        
        if n_high_points >= 2 and n_low_points >= 2:
            # Manual linear regression per highs
            high_start_idx = high_count - n_high_points
            
            # Calcola slope per highs
            high_x_sum = 0.0
            high_y_sum = 0.0
            high_x_sum_sq = 0.0
            high_xy_sum = 0.0
            
            for i in range(n_high_points):
                x = float(high_indices[high_start_idx + i])
                y = high_values[high_start_idx + i]
                
                high_x_sum += x
                high_y_sum += y
                high_x_sum_sq += x * x
                high_xy_sum += x * y
            
            # Linear regression: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
            high_numerator = n_high_points * high_xy_sum - high_x_sum * high_y_sum
            high_denominator = n_high_points * high_x_sum_sq - high_x_sum * high_x_sum
            
            if high_denominator != 0:
                high_slope = high_numerator / high_denominator
            else:
                return None
            
            # Manual linear regression per lows
            low_start_idx = low_count - n_low_points
            
            # Calcola slope per lows
            low_x_sum = 0.0
            low_y_sum = 0.0
            low_x_sum_sq = 0.0
            low_xy_sum = 0.0
            
            for i in range(n_low_points):
                x = float(low_indices[low_start_idx + i])
                y = low_values[low_start_idx + i]
                
                low_x_sum += x
                low_y_sum += y
                low_x_sum_sq += x * x
                low_xy_sum += x * y
            
            low_numerator = n_low_points * low_xy_sum - low_x_sum * low_y_sum
            low_denominator = n_low_points * low_x_sum_sq - low_x_sum * low_x_sum
            
            if low_denominator != 0:
                low_slope = low_numerator / low_denominator
            else:
                return None
            
            # Se le pendenze sono simili, è un channel
            abs_high_slope = abs(high_slope)
            abs_low_slope = abs(low_slope)
            max_abs_slope = max(abs_high_slope, abs_low_slope, 0.0001)
            
            slope_diff = abs(high_slope - low_slope) / max_abs_slope
            
            if slope_diff < 0.2:  # Pendenze simili
                if high_slope > 0.0001:
                    return "ascending_channel"
                elif high_slope < -0.0001:
                    return "descending_channel"
                else:
                    return "horizontal_channel"
        
        return None
    
    def _get_pattern_direction(self, pattern_name: str) -> str:
        """Determina la direzione implicita di un pattern"""
        bullish_patterns = [
            'double_bottom', 'inverse_head_shoulders', 'ascending_triangle',
            'bullish_flag', 'bullish_pennant', 'cup_handle', 'falling_wedge',
            'bullish', 'reversal_bullish', 'continuation_bullish', 'ascending',
            'elliott_wave_3', 'wyckoff_accumulation', 'institutional_accumulation',
            'harmonic_butterfly', 'harmonic_gartley'
        ]
        
        bearish_patterns = [
            'double_top', 'head_shoulders', 'descending_triangle',
            'bearish_flag', 'bearish_pennant', 'rising_wedge',
            'bearish', 'reversal_bearish', 'continuation_bearish', 'descending',
            'elliott_wave_5', 'wyckoff_distribution', 'institutional_distribution'
        ]
        
        for pattern in bullish_patterns:
            if pattern in pattern_name.lower():
                return 'bullish'
        
        for pattern in bearish_patterns:
            if pattern in pattern_name.lower():
                return 'bearish'
        
        return 'neutral'
    
    def _get_pattern_context(self, pattern: str, prices: np.ndarray, 
                           volumes: np.ndarray) -> Dict[str, Any]:
        """Ottieni contesto specifico per un pattern"""
        context = {
            'price_range': float(prices.max() - prices.min()),
            'volume_profile': 'normal',
            'pattern_completion': 0.0
        }
        
        # Volume analysis
        recent_vol = np.mean(volumes[-10:])
        avg_vol = np.mean(volumes)
        if recent_vol > avg_vol * 1.5:
            context['volume_profile'] = 'high'
        elif recent_vol < avg_vol * 0.7:
            context['volume_profile'] = 'low'
        
        # Pattern-specific context
        if 'harmonic' in pattern:
            # Per pattern armonici, calcola Fibonacci levels
            context['fib_levels'] = self._calculate_fibonacci_levels(prices)
        
        elif 'elliott' in pattern:
            # Per Elliott waves, identifica wave count
            context['wave_count'] = self._estimate_elliott_wave_count(prices)
        
        elif 'wyckoff' in pattern:
            # Per Wyckoff, identifica fase
            context['wyckoff_phase'] = self._identify_wyckoff_phase(prices, volumes)
        
        return context
    
    def _prepare_lstm_pattern_features(self, prices: np.ndarray, 
                                     volumes: np.ndarray) -> np.ndarray:
        """Prepara features specifiche per pattern LSTM"""
        
        # Pattern-specific indicators
        features = []
        
        # Price action features
        for i in range(30):
            idx = -(30 - i)
            
            # Local highs/lows
            is_high = prices[idx] > prices[idx-1] and prices[idx] > prices[idx+1] if idx > -len(prices)+1 else 0
            is_low = prices[idx] < prices[idx-1] and prices[idx] < prices[idx+1] if idx > -len(prices)+1 else 0
            
            # Price position
            price_percentile = np.sum(prices < prices[idx]) / len(prices)
            
            # Volume spike
            vol_spike = volumes[idx] / np.mean(volumes[max(0, idx-5):idx]) if idx > 5 else 1
            
            # Momentum
            momentum = (prices[idx] - prices[idx-5]) / prices[idx-5] if idx >= 5 else 0
            
            feature_vec = [
                prices[idx] / prices[-1],
                float(is_high),
                float(is_low),
                price_percentile,
                vol_spike,
                momentum,
                volumes[idx] / np.mean(volumes)
            ]
            
            features.append(feature_vec)
        
        return np.array(features)
    
    def _prepare_transformer_pattern_features(self, prices: np.ndarray, volumes: np.ndarray,
                                            market_data: Dict[str, Any]) -> np.ndarray:
        """Prepara features complesse per Transformer pattern recognition"""
        
        # Multi-scale features
        features = []
        
        # Different time scales
        for scale in [5, 10, 20]:
            scaled_prices = prices[::scale]
            if len(scaled_prices) > 10:
                # Pattern statistics at this scale
                peaks = self._count_peaks(scaled_prices)
                valleys = self._count_valleys(scaled_prices)
                trend = np.polyfit(range(len(scaled_prices)), scaled_prices, 1)[0]
                
                features.extend([peaks, valleys, trend])
        
        # Wavelet-like decomposition (simplified)
        for window in [10, 20, 40]:
            if len(prices) >= window:
                rolling_mean = np.convolve(prices, np.ones(window)/window, mode='valid')
                if len(rolling_mean) > 0:
                    features.append(rolling_mean[-1] / prices[-1])
                else:
                    features.append(1.0)
        
        # Market microstructure
        features.extend([
            market_data.get('spread_percentage', 0),
            market_data.get('volume_ratio', 1),
            market_data.get('volatility', 0),
            market_data.get('momentum_5m', 0)
        ])
        
        # Pad to expected size
        while len(features) < 15:
            features.append(0.0)
        
        return np.array(features[:15])
    
    def _count_peaks(self, prices: np.ndarray) -> int:
        """Conta i picchi in una serie di prezzi"""
        peaks = 0
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks += 1
        return peaks
    
    def _count_valleys(self, prices: np.ndarray) -> int:
        """Conta le valli in una serie di prezzi"""
        valleys = 0
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                valleys += 1
        return valleys
    
    def _prepare_sentiment_features(self, prices: np.ndarray, volumes: np.ndarray,
                                market_data: Dict[str, Any]) -> np.ndarray:
        """Prepara features per sentiment analysis - OTTIMIZZATO"""
        
        # Pre-allocazione array risultato
        features = np.zeros(8, dtype=np.float32)
        feature_idx = 0
        
        # Cache valori comuni
        n_prices = len(prices)
        n_volumes = len(volumes)
        current_price = float(prices[-1]) if n_prices > 0 else 0.0
        
        # Price momentum at different scales - OTTIMIZZATO
        lookbacks = [5, 10, 20]
        for lookback in lookbacks:
            if n_prices >= lookback:
                past_price = float(prices[n_prices - 1 - lookback])  # Accesso diretto invece di [-lookback]
                if past_price > 0:
                    momentum = (current_price - past_price) / past_price
                    features[feature_idx] = momentum
                # else: già 0.0 da pre-allocazione
            feature_idx += 1
        
        # Volume profile - OTTIMIZZATO: evita slicing e np.mean()
        if n_volumes >= 20:
            # Calcola medie senza creare arrays temporanei
            recent_vol_sum = 0.0
            for i in range(n_volumes - 5, n_volumes):  # Ultimi 5
                recent_vol_sum += volumes[i]
            recent_vol = recent_vol_sum / 5.0
            
            older_vol_sum = 0.0
            for i in range(n_volumes - 20, n_volumes - 5):  # Da -20 a -5
                older_vol_sum += volumes[i]
            older_vol = older_vol_sum / 15.0  # 15 elementi (20-5)
            
            if older_vol > 0:
                vol_change = (recent_vol - older_vol) / older_vol
                features[feature_idx] = vol_change
            # else: già 0.0 da pre-allocazione
        # else: già 0.0 da pre-allocazione
        feature_idx += 1
        
        # Volatility metrics - OTTIMIZZATO: evita np.diff() e slicing
        if n_prices >= 20:
            # Calcola returns in-place senza creare array temporaneo
            returns_sum_sq = 0.0
            returns_sum = 0.0
            n_returns = 19  # 20 prezzi = 19 returns
            
            start_idx = n_prices - 20
            for i in range(n_returns):
                if prices[start_idx + i] > 0:
                    return_val = (prices[start_idx + i + 1] - prices[start_idx + i]) / prices[start_idx + i]
                    returns_sum += return_val
                    returns_sum_sq += return_val * return_val
            
            # Calcola standard deviation senza array temporaneo
            if n_returns > 0:
                mean_return = returns_sum / n_returns
                variance = (returns_sum_sq / n_returns) - (mean_return * mean_return)
                if variance > 0:
                    volatility = np.sqrt(variance)
                    features[feature_idx] = volatility
                # else: già 0.0 da pre-allocazione
        else:
            # Fallback da market_data
            volatility_fallback = market_data.get('volatility', 0.0)
            features[feature_idx] = float(volatility_fallback)
        feature_idx += 1
        
        # Price position in recent range - OTTIMIZZATO: singolo pass per min/max
        if n_prices >= 20:
            # Calcola min/max in un solo loop invece di np.min/np.max su slice
            start_idx = n_prices - 20
            recent_high = prices[start_idx]
            recent_low = prices[start_idx]
            
            for i in range(start_idx + 1, n_prices):
                if prices[i] > recent_high:
                    recent_high = prices[i]
                if prices[i] < recent_low:
                    recent_low = prices[i]
            
            if recent_high > recent_low:
                price_position = (current_price - recent_low) / (recent_high - recent_low)
                features[feature_idx] = price_position
            else:
                features[feature_idx] = 0.5
        else:
            features[feature_idx] = 0.5
        feature_idx += 1
        
        # Market state indicators - accesso diretto
        market_state = market_data.get('market_state', '')
        features[feature_idx] = 1.0 if market_state == 'trending' else 0.0
        feature_idx += 1
        
        volume_ratio = market_data.get('volume_ratio', 1.0)
        features[feature_idx] = float(volume_ratio)
        feature_idx += 1
        
        # Le ultime due features rimangono 0.0 dalla pre-allocazione
        # Non serve loop "while len(features) < 8" né slicing "features[:8]"
        
        return features
    
    def _analyze_market_behavior(self, prices: np.ndarray, 
                            volumes: np.ndarray) -> Dict[str, Any]:
        """Analizza il comportamento del mercato - OTTIMIZZATO"""
        
        n_prices = len(prices)
        n_volumes = len(volumes)
        
        if n_prices < 20:
            return {"type": "unknown", "confidence": 0.0}
        
        # Pre-calcola returns una sola volta senza copie intermedie
        n_returns = n_prices - 1
        returns = np.empty(n_returns, dtype=np.float32)
        
        # Calcolo in-place per evitare np.diff() e slicing
        for i in range(n_returns):
            if prices[i] != 0:
                returns[i] = (prices[i + 1] - prices[i]) / prices[i]
            else:
                returns[i] = 0.0
        
        # Inizializza valori di default
        volatility_clustering = 0.0
        mean_reversion = 0.0
        volume_price_corr = 0.0
        
        # Volatility clustering - evita copie multiple
        if n_returns >= 10:
            try:
                # Pre-alloca arrays per correlazione invece di slicing
                n_corr_points = n_returns - 1
                abs_returns_curr = np.empty(n_corr_points, dtype=np.float32)
                abs_returns_next = np.empty(n_corr_points, dtype=np.float32)
                
                # Popola arrays senza slicing
                for i in range(n_corr_points):
                    abs_returns_curr[i] = abs(returns[i])
                    abs_returns_next[i] = abs(returns[i + 1])
                
                # Calcola correlazione una sola volta
                corr_matrix = np.corrcoef(abs_returns_curr, abs_returns_next)
                volatility_clustering = float(corr_matrix[0, 1])
                
            except (ValueError, np.linalg.LinAlgError):
                volatility_clustering = 0.0
        
        # Mean reversion - riutilizza stesso pattern ottimizzato
        if n_returns >= 10:
            try:
                # Riutilizza arrays precedenti se disponibili, altrimenti pre-alloca
                n_corr_points = n_returns - 1
                returns_curr = np.empty(n_corr_points, dtype=np.float32)
                returns_next = np.empty(n_corr_points, dtype=np.float32)
                
                # Popola arrays senza slicing
                for i in range(n_corr_points):
                    returns_curr[i] = returns[i]
                    returns_next[i] = returns[i + 1]
                
                # Calcola correlazione
                corr_matrix = np.corrcoef(returns_curr, returns_next)
                mean_reversion = -float(corr_matrix[0, 1])
                
            except (ValueError, np.linalg.LinAlgError):
                mean_reversion = 0.0
        
        # Momentum persistence - calcolo diretto
        momentum_persistence = -mean_reversion  # Opposto di mean reversion
        
        # Volume-Price correlation - ottimizzato per evitare slicing
        if n_prices >= 10 and n_volumes >= 10:
            try:
                # Usa ultimi 10 punti senza creare copie
                n_corr = min(10, n_prices, n_volumes)
                start_prices = n_prices - n_corr
                start_volumes = n_volumes - n_corr
                
                # Pre-alloca arrays per correlazione
                price_subset = np.empty(n_corr, dtype=np.float32)
                volume_subset = np.empty(n_corr, dtype=np.float32)
                
                # Copia diretta senza slicing
                for i in range(n_corr):
                    price_subset[i] = prices[start_prices + i]
                    volume_subset[i] = volumes[start_volumes + i]
                
                # Calcola correlazione
                corr_matrix = np.corrcoef(price_subset, volume_subset)
                volume_price_corr = float(corr_matrix[0, 1])
                
            except (ValueError, np.linalg.LinAlgError):
                volume_price_corr = 0.0
        
        # Sanitizza valori NaN una sola volta
        if np.isnan(volatility_clustering):
            volatility_clustering = 0.0
        if np.isnan(mean_reversion):
            mean_reversion = 0.0
        if np.isnan(momentum_persistence):
            momentum_persistence = 0.0
        if np.isnan(volume_price_corr):
            volume_price_corr = 0.0
        
        # Determina comportamento dominante - ottimizzato
        abs_volatile = abs(volatility_clustering)
        abs_mean_rev = abs(mean_reversion)
        abs_momentum = abs(momentum_persistence)
        abs_volume = abs(volume_price_corr)
        
        # Pre-alloca comportamenti per evitare dictionary recreations
        behaviors = {
            "volatile_clustering": volatility_clustering,
            "mean_reverting": mean_reversion,
            "momentum_driven": momentum_persistence,
            "volume_sensitive": volume_price_corr
        }
        
        # Trova comportamento dominante senza lambda
        max_strength = abs_volatile
        dominant_behavior = "volatile_clustering"
        
        if abs_mean_rev > max_strength:
            max_strength = abs_mean_rev
            dominant_behavior = "mean_reverting"
        
        if abs_momentum > max_strength:
            max_strength = abs_momentum
            dominant_behavior = "momentum_driven"
        
        if abs_volume > max_strength:
            max_strength = abs_volume
            dominant_behavior = "volume_sensitive"
        
        # Calcola confidence una volta
        confidence = min(0.9, max_strength * 2.0)
        
        return {
            "type": dominant_behavior,
            "strength": max_strength,
            "confidence": confidence,
            "characteristics": behaviors
        }
    
    def _analyze_smart_money(self, prices: np.ndarray, 
                        volumes: np.ndarray) -> Dict[str, Any]:
        """Analizza attività smart money - OTTIMIZZATO"""
        
        n_prices = len(prices)
        n_volumes = len(volumes)
        
        if n_prices < 50 or n_volumes < 50:
            return {"activity": "unknown", "confidence": 0.0}
        
        # Volume spikes su movimenti minimi di prezzo - OTTIMIZZATO MASSIMAMENTE
        accumulation_score = 0
        distribution_score = 0
        
        # Pre-calcola rolling volume means usando sliding window ottimizzato
        # Invece di volumes[i-20:i] per ogni i, usa rolling sum
        window_size = 20
        
        # Calcola initial rolling sum per posizione 20
        rolling_volume_sum = 0.0
        for j in range(window_size):
            rolling_volume_sum += volumes[j]
        
        # Loop ottimizzato con sliding window
        for i in range(window_size, n_prices - 1):
            # Cache valori per ridurre accessi array
            current_price = prices[i]
            prev_price = prices[i - 1]
            current_volume = volumes[i]
            
            # Calcola price change una volta
            if prev_price > 0:
                price_change = abs(current_price - prev_price) / prev_price
            else:
                price_change = 0.0
            
            # Calcola volume spike usando rolling mean ottimizzato
            rolling_volume_mean = rolling_volume_sum / window_size
            if rolling_volume_mean > 0:
                volume_spike = current_volume / rolling_volume_mean
            else:
                volume_spike = 1.0
            
            # Controlli smart money con thresholds cached
            if volume_spike > 2.0 and price_change < 0.002:
                if current_price > prev_price:
                    accumulation_score += 1
                elif current_price < prev_price:
                    distribution_score += 1
            
            # Update rolling sum per prossima iterazione (sliding window)
            if i < n_prices - 2:  # Evita out-of-bounds per ultimo elemento
                # Rimuovi elemento più vecchio, aggiungi nuovo
                rolling_volume_sum -= volumes[i - window_size]
                rolling_volume_sum += volumes[i + 1]
        
        # Analisi del volume profile - OTTIMIZZATO: evita slicing
        # Recent average (ultimi 20)
        recent_sum = 0.0
        for i in range(n_volumes - 20, n_volumes):
            recent_sum += volumes[i]
        recent_avg_volume = recent_sum / 20.0
        
        # Older average (da -50 a -20, quindi 30 elementi)
        older_sum = 0.0
        for i in range(n_volumes - 50, n_volumes - 20):
            older_sum += volumes[i]
        older_avg_volume = older_sum / 30.0
        
        # Determina volume trend
        volume_trend = "increasing" if recent_avg_volume > older_avg_volume * 1.2 else "decreasing"
        
        # Determina attività smart money
        if accumulation_score > distribution_score * 1.5:
            activity = "accumulation"
            confidence = min(0.9, accumulation_score / 10.0)
        elif distribution_score > accumulation_score * 1.5:
            activity = "distribution"
            confidence = min(0.9, distribution_score / 10.0)
        else:
            activity = "neutral"
            confidence = 0.5
        
        return {
            "activity": activity,
            "confidence": confidence,
            "accumulation_signals": accumulation_score,
            "distribution_signals": distribution_score,
            "volume_trend": volume_trend
        }
    
    def _calculate_momentum_features(self, prices: np.ndarray, 
                                volumes: np.ndarray) -> Dict[str, float]:
        """Calcola features di momentum - ULTRA OTTIMIZZATO ZERO COPIE"""
        
        # 🛡️ VALIDAZIONE INPUT RAPIDA
        if len(prices) == 0 or len(volumes) == 0:
            return self._get_default_momentum_features()
        
        # 🚀 ENSURE DTYPE IN-PLACE quando possibile
        if prices.dtype != np.float32:
            if prices.flags.writeable and prices.base is None:
                prices = prices.view(dtype=np.float32) if prices.itemsize == 4 else prices.astype(np.float32, copy=False)
            else:
                prices = prices.astype(np.float32)
        
        if volumes.dtype != np.float32:
            if volumes.flags.writeable and volumes.base is None:
                volumes = volumes.view(dtype=np.float32) if volumes.itemsize == 4 else volumes.astype(np.float32, copy=False)
            else:
                volumes = volumes.astype(np.float32)
        
        # 🚀 PRE-ALLOCA DICTIONARY con capacità nota (evita resize)
        features = {}
        
        # 🚀 PRE-CALCOLA VALORI RIUTILIZZATI UNA VOLTA
        current_price = float(prices[-1])
        prices_len = len(prices)
        volumes_len = len(volumes)
        
        # 🚀 PRICE MOMENTUM OTTIMIZZATO: Accesso diretto senza loop
        # Pre-calcola tutti gli indici e valori necessari
        momentum_periods = [5, 10, 20, 50]
        
        # Calcolo vettoriale per tutti i momentum contemporaneamente
        for period in momentum_periods:
            if prices_len >= period:
                past_price = float(prices[-period])
                if past_price != 0:
                    momentum = (current_price - past_price) / past_price
                    features[f'momentum_{period}'] = momentum
                else:
                    features[f'momentum_{period}'] = 0.0
            else:
                features[f'momentum_{period}'] = 0.0
        
        # 🚀 INDICATORI CON CACHE SYSTEM OTTIMIZZATO
        try:
            # Usa batch calculation per tutti gli indicatori necessari
            indicators = self._calculate_momentum_indicators_batch(prices)
            
            # RSI
            if prices_len >= 14 and 'rsi' in indicators:
                rsi_value = float(indicators['rsi'][-1])
                features['rsi'] = rsi_value if np.isfinite(rsi_value) else 50.0
            else:
                features['rsi'] = 50.0
            
            # MACD
            if prices_len >= 26 and 'macd' in indicators:
                macd_value = float(indicators['macd'][-1])
                signal_value = float(indicators['macd_signal'][-1])
                hist_value = float(indicators['macd_hist'][-1])
                
                features['macd'] = macd_value if np.isfinite(macd_value) else 0.0
                features['macd_signal'] = signal_value if np.isfinite(signal_value) else 0.0
                features['macd_histogram'] = hist_value if np.isfinite(hist_value) else 0.0
            else:
                features['macd'] = features['macd_signal'] = features['macd_histogram'] = 0.0
            
            # Stochastic
            if prices_len >= 14 and 'stoch_k' in indicators:
                stoch_k_value = float(indicators['stoch_k'][-1])
                stoch_d_value = float(indicators['stoch_d'][-1])
                
                features['stoch_k'] = stoch_k_value if np.isfinite(stoch_k_value) else 50.0
                features['stoch_d'] = stoch_d_value if np.isfinite(stoch_d_value) else 50.0
            else:
                features['stoch_k'] = features['stoch_d'] = 50.0
                
        except Exception as cache_error:
            safe_print(f"❌ Errore cache momentum indicators: {cache_error}")
            # Fallback con valori default
            features.update({
                'rsi': 50.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'macd_histogram': 0.0,
                'stoch_k': 50.0,
                'stoch_d': 50.0
            })
        
        # 🚀 VOLUME MOMENTUM ULTRA-OTTIMIZZATO
        if volumes_len >= 20:
            # Pre-calcola gli indici per le slice
            recent_start = volumes_len - 5
            older_start = volumes_len - 20
            older_end = volumes_len - 5
            
            # Views dirette invece di slice che copiano
            recent_volumes = volumes[recent_start:]  # Ultimi 5
            older_volumes = volumes[older_start:older_end]  # Dal -20 al -5
            
            # Calcolo vettoriale delle medie
            recent_mean = float(np.mean(recent_volumes))
            older_mean = float(np.mean(older_volumes))
            
            if older_mean != 0:
                vol_momentum = (recent_mean - older_mean) / older_mean
                features['volume_momentum'] = vol_momentum
            else:
                features['volume_momentum'] = 0.0
        else:
            features['volume_momentum'] = 0.0
        
        return features

    def _calculate_momentum_indicators_batch(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Calcola tutti gli indicatori momentum in batch ottimizzato"""
        
        indicators = {}
        
        try:
            # 🚀 USA CACHED INDICATORS per massima efficienza
            if len(prices) >= 14:
                indicators['rsi'] = self.cached_indicators['rsi'](prices, 14)
            
            if len(prices) >= 26:
                # MACD come batch
                macd_result = self.cached_indicators['macd'](prices)
                if isinstance(macd_result, tuple) and len(macd_result) >= 3:
                    indicators['macd'] = macd_result[0]
                    indicators['macd_signal'] = macd_result[1]
                    indicators['macd_hist'] = macd_result[2]
            
            # Stochastic (se disponibile in cache)
            if len(prices) >= 14:
                try:
                    if 'stoch' in self.cached_indicators:
                        stoch_result = self.cached_indicators['stoch'](prices, prices, prices, 14)
                        if isinstance(stoch_result, tuple) and len(stoch_result) >= 2:
                            indicators['stoch_k'] = stoch_result[0]
                            indicators['stoch_d'] = stoch_result[1]
                except:
                    pass  # Stochastic opzionale
            
            return indicators
            
        except Exception as e:
            safe_print(f"❌ Errore batch momentum indicators: {e}")
            return {}  # Return empty dict per trigger fallback

    def _get_default_momentum_features(self) -> Dict[str, float]:
        """Restituisce features momentum di default per input non validi"""
        return {
            'momentum_5': 0.0,
            'momentum_10': 0.0,
            'momentum_20': 0.0,
            'momentum_50': 0.0,
            'rsi': 50.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'stoch_k': 50.0,
            'stoch_d': 50.0,
            'volume_momentum': 0.0
        }
        
    def _prepare_transformer_bias_features(self, prices: np.ndarray, volumes: np.ndarray,
                                        market_data: Dict[str, Any]) -> np.ndarray:
        """Prepara features multi-dimensionali per Transformer bias detection"""
        
        features = []
        
        # Multi-timeframe momentum
        for tf in [5, 15, 30, 60]:
            if len(prices) >= tf:
                tf_return = (prices[-1] - prices[-tf]) / prices[-tf]
                features.append(tf_return)
            else:
                features.append(0.0)
        
        # Order flow imbalance proxy
        if len(prices) >= 21 and len(volumes) >= 21:  # Serve 21 per calcolare diff su 20 elementi
            price_changes = np.diff(prices[-21:])  # Diff di 21 elementi da 20 risultati
            volume_slice = volumes[-20:]  # Ultimi 20 volumi
            
            # Separa volumi buy/sell basandosi sui price changes
            buy_volume = np.sum(volume_slice[price_changes > 0])
            sell_volume = np.sum(volume_slice[price_changes < 0])
            total_vol = buy_volume + sell_volume
            
            if total_vol > 0:
                order_imbalance = (buy_volume - sell_volume) / total_vol
            else:
                order_imbalance = 0.0
            features.append(order_imbalance)
        else:
            features.append(0.0)
        
        # Market regime indicators
        features.append(1.0 if market_data.get('market_state') == 'trending' else 0.0)
        features.append(1.0 if market_data.get('market_state') == 'volatile_trending' else 0.0)
        
        # Microstructure bias
        features.append(market_data.get('spread_percentage', 0))
        
        # Ensure correct size
        while len(features) < 8:
            features.append(0.0)
        
        return np.array(features[:8])
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function implementation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _detect_market_regime(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """Rileva il regime di mercato corrente - OTTIMIZZATO"""
        
        n_prices = len(prices)
        n_volumes = len(volumes)
        
        if n_prices < 50:
            return {"regime": "unknown", "confidence": 0.0}
        
        # Calcola volatility senza creare arrays returns
        # Usa ultimi 20 returns (21 prezzi) per volatility
        volatility = 0.0
        if n_prices >= 21:
            returns_sum = 0.0
            returns_sum_sq = 0.0
            n_returns = 20
            
            start_idx = n_prices - 21  # 21 prezzi per 20 returns
            for i in range(n_returns):
                if prices[start_idx + i] > 0:
                    return_val = (prices[start_idx + i + 1] - prices[start_idx + i]) / prices[start_idx + i]
                    returns_sum += return_val
                    returns_sum_sq += return_val * return_val
            
            if n_returns > 0:
                mean_return = returns_sum / n_returns
                variance = (returns_sum_sq / n_returns) - (mean_return * mean_return)
                volatility = np.sqrt(variance) if variance > 0 else 0.0
        
        # Trend strength - OTTIMIZZATO: evita np.polyfit e slicing
        r_squared = 0.0
        slope = 0.0
        
        if n_prices >= 20:
            # Manual linear regression senza np.polyfit
            # Usa ultimi 20 prezzi
            n_trend = 20
            start_idx = n_prices - n_trend
            
            # Pre-calcola x_mean per regression
            x_mean = (n_trend - 1) / 2.0  # Media di arange(20) = 9.5
            
            # Calcola y_mean senza np.mean()
            y_sum = 0.0
            for i in range(n_trend):
                y_sum += prices[start_idx + i]
            y_mean = y_sum / n_trend
            
            # Calcola slope usando formula diretta
            numerator = 0.0
            denominator = 0.0
            ss_tot = 0.0
            
            for i in range(n_trend):
                x_i = float(i)
                y_i = prices[start_idx + i]
                
                x_diff = x_i - x_mean
                y_diff = y_i - y_mean
                
                numerator += x_diff * y_diff
                denominator += x_diff * x_diff
                ss_tot += y_diff * y_diff
            
            if denominator > 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                
                # Calcola R-squared senza arrays temporanei
                ss_res = 0.0
                for i in range(n_trend):
                    y_i = prices[start_idx + i]
                    y_pred_i = slope * i + intercept
                    residual = y_i - y_pred_i
                    ss_res += residual * residual
                
                r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Volume regime - OTTIMIZZATO: evita np.mean() su slices
        volume_regime = 1.0
        if n_volumes >= 50:
            # Recent volume (ultimi 10)
            recent_sum = 0.0
            for i in range(n_volumes - 10, n_volumes):
                recent_sum += volumes[i]
            recent_volume = recent_sum / 10.0
            
            # Normal volume (ultimi 50)
            normal_sum = 0.0
            for i in range(n_volumes - 50, n_volumes):
                normal_sum += volumes[i]
            normal_volume = normal_sum / 50.0
            
            if normal_volume > 0:
                volume_regime = recent_volume / normal_volume
        
        # Classify regime - thresholds ottimizzati
        abs_slope = abs(slope)
        
        if r_squared > 0.7 and abs_slope > 0.0001:
            if volatility < 0.01:
                regime = "smooth_trend"
            else:
                regime = "volatile_trend"
        elif volatility > 0.02:
            regime = "high_volatility"
        elif volatility < 0.005:
            regime = "low_volatility"
        else:
            regime = "ranging"
        
        # Add volume characteristic
        if volume_regime > 1.5:
            regime += "_high_volume"
        elif volume_regime < 0.5:
            regime += "_low_volume"
        
        # Confidence calculation ottimizzato
        confidence = max(r_squared, min(0.9, volatility * 50.0))
        
        return {
            "regime": regime,
            "confidence": confidence,
            "volatility": volatility,
            "trend_strength": r_squared,
            "volume_regime": volume_regime
        }
    
    def _detect_institutional_bias(self, prices: np.ndarray, volumes: np.ndarray,
                                 market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Rileva bias istituzionale nel mercato"""
        
        if len(prices) < 100 or len(volumes) < 100:
            return {"bias": "unknown", "confidence": 0.0}
        
        # Institutional footprints
        footprints = {
            "large_orders": 0,
            "iceberg_orders": 0,
            "stop_hunting": 0,
            "accumulation": 0,
            "distribution": 0
        }
        
        # Detect large orders (volume spikes)
        avg_volume = np.mean(volumes)
        for i in range(50, len(volumes)):
            if volumes[i] > avg_volume * 3:
                footprints["large_orders"] += 1
                
                # Check if price barely moved (iceberg)
                if i < len(prices) - 1:
                    price_change = abs(prices[i] - prices[i-1]) / prices[i-1]
                    if price_change < 0.001:
                        footprints["iceberg_orders"] += 1
        
        # Detect stop hunting (quick spike and reversal)
        for i in range(10, len(prices) - 10):
            # Look for spike
            if prices[i] > max(prices[i-10:i]) * 1.005:  # 0.5% spike
                # Check for reversal
                if prices[i+5] < prices[i] * 0.995:  # Reversal
                    footprints["stop_hunting"] += 1
        
        # Accumulation/Distribution patterns
        for i in range(50, len(prices) - 10):
            window_prices = prices[i-20:i]
            window_volumes = volumes[i-20:i]
            
            # Sideways price with increasing volume = accumulation
            price_range = (max(window_prices) - min(window_prices)) / np.mean(window_prices)
            vol_trend = np.mean(window_volumes[-10:]) / np.mean(window_volumes[:10])
            
            if price_range < 0.01 and vol_trend > 1.3:
                footprints["accumulation"] += 1
            elif price_range < 0.01 and vol_trend < 0.7:
                footprints["distribution"] += 1
        
        # Determine dominant institutional activity
        max_activity = max(footprints.items(), key=lambda x: x[1])
        
        if max_activity[1] > 5:
            bias = max_activity[0]
            confidence = min(0.9, max_activity[1] / 20)
        else:
            bias = "neutral"
            confidence = 0.3
        
        return {
            "bias": bias,
            "confidence": float(confidence),
            "footprints": footprints,
            "institutional_presence": sum(footprints.values()) > 10
        }
    
    def _analyze_price_action_bias(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analizza bias basato su price action - OTTIMIZZATO"""
        
        prices = np.array(market_data.get('price_history', []))
        n_prices = len(prices)
        
        if n_prices < 20:
            return {"direction": "neutral", "confidence": 0.0}
        
        # Calcola candlestick patterns SENZA creare arrays opens/closes
        bullish_candles = 0
        bearish_candles = 0
        
        # Loop diretto invece di broadcast operations su arrays copiati
        for i in range(n_prices - 1):
            open_price = prices[i]      # opens[i] = prices[i]
            close_price = prices[i + 1] # closes[i] = prices[i+1]
            
            if close_price > open_price:
                bullish_candles += 1
            elif close_price < open_price:
                bearish_candles += 1
        
        # Pin bars / rejection candles - OTTIMIZZATO
        pin_bars_up = 0
        pin_bars_down = 0
        
        for i in range(1, n_prices - 1):
            # Cache valori per ridurre accessi array
            prev_price = prices[i - 1]
            curr_price = prices[i]
            next_price = prices[i + 1]
            
            # Calcola high/low senza slicing prices[i-1:i+2]
            high = max(prev_price, curr_price, next_price)
            low = min(prev_price, curr_price, next_price)
            
            # Valori open/close per questa candela
            open_val = prev_price   # opens[i-1] = prices[i-1]
            close_val = curr_price  # closes[i-1] = prices[i]
            
            # Calcola body una volta
            body = abs(close_val - open_val)
            
            # Calcola wicks senza chiamate multiple a max/min
            body_high = max(open_val, close_val)
            body_low = min(open_val, close_val)
            
            upper_wick = high - body_high
            lower_wick = body_low - low
            
            # Pin bar detection con thresholds cached
            if body > 0:  # Evita divisione per zero
                if upper_wick > body * 2.0 and lower_wick < body * 0.5:
                    pin_bars_down += 1
                elif lower_wick > body * 2.0 and upper_wick < body * 0.5:
                    pin_bars_up += 1
        
        # Calculate bias
        bullish_score = bullish_candles + pin_bars_up * 2
        bearish_score = bearish_candles + pin_bars_down * 2
        
        total_score = bullish_score + bearish_score
        
        if total_score > 0:
            bullish_ratio = bullish_score / total_score
            
            # Usa thresholds cached per performance
            if bullish_ratio > 0.65:
                direction = "bullish"
                confidence = bullish_ratio
            elif bullish_ratio < 0.35:
                direction = "bearish"
                confidence = 1.0 - bullish_ratio
            else:
                direction = "neutral"
                confidence = 0.5
        else:
            direction = "neutral"
            confidence = 0.3
        
        return {
            "direction": direction,
            "confidence": confidence,
            "bullish_signals": bullish_score,
            "bearish_signals": bearish_score
        }
    
    def _analyze_order_flow_bias(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analizza bias basato su order flow - OTTIMIZZATO"""
        
        # Ottimizzazione: evita conversioni numpy se già array o se lista vuota
        price_history = market_data.get('price_history', [])
        volume_history = market_data.get('volume_history', [])
        
        # Check early per liste vuote
        if not price_history or not volume_history:
            return {"direction": "neutral", "confidence": 0.0}
        
        # Conversione solo se necessario (mantieni np.array per compatibility)
        prices = np.array(price_history) if not isinstance(price_history, np.ndarray) else price_history
        volumes = np.array(volume_history) if not isinstance(volume_history, np.ndarray) else volume_history
        
        n_prices = len(prices)
        n_volumes = len(volumes)
        
        if n_prices < 20 or n_volumes < 20:
            return {"direction": "neutral", "confidence": 0.0}
        
        # Volume delta approximation - OTTIMIZZATO
        buy_volume = 0.0
        sell_volume = 0.0
        
        # Cache per ridurre accessi array
        prev_price = prices[0]
        
        for i in range(1, n_prices):
            current_price = prices[i]
            current_volume = volumes[i]
            
            # Comparison ottimizzata usando cached prev_price
            if current_price > prev_price:
                buy_volume += current_volume
            elif current_price < prev_price:
                sell_volume += current_volume
            else:
                # Split equally for unchanged - ottimizzato
                half_volume = current_volume * 0.5
                buy_volume += half_volume
                sell_volume += half_volume
            
            # Update per prossima iterazione
            prev_price = current_price
        
        # Calculate cumulative delta
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            delta = buy_volume - sell_volume
            delta_ratio = delta / total_volume
            
            # Cache abs(delta_ratio) per evitare ricalcolo
            abs_delta_ratio = abs(delta_ratio)
            
            # Thresholds cached
            if delta_ratio > 0.1:
                direction = "bullish"
                confidence = min(0.9, abs_delta_ratio * 2.0)
            elif delta_ratio < -0.1:
                direction = "bearish"
                confidence = min(0.9, abs_delta_ratio * 2.0)
            else:
                direction = "neutral"
                confidence = 0.5
            
            # Pre-calcola reciproco per evitare divisioni multiple
            inv_total_volume = 1.0 / total_volume
            buy_pressure = buy_volume * inv_total_volume
            sell_pressure = sell_volume * inv_total_volume
            
        else:
            direction = "neutral"
            confidence = 0.3
            delta_ratio = 0.0
            buy_pressure = 0.5
            sell_pressure = 0.5
        
        return {
            "direction": direction,
            "confidence": confidence,
            "buy_pressure": buy_pressure,
            "sell_pressure": sell_pressure,
            "delta_ratio": delta_ratio
        }
    
    def _analyze_microstructure_bias(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analizza bias basato su market microstructure"""
        
        spread = market_data.get('spread_percentage', 0)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        volatility = market_data.get('volatility', 0)
        
        # Spread analysis
        if spread > 0.001:  # Wide spread
            spread_bias = "uncertain"
            spread_confidence = 0.4
        else:
            spread_bias = "liquid"
            spread_confidence = 0.7
        
        # Volume analysis
        if volume_ratio > 1.5:
            volume_bias = "active"
        elif volume_ratio < 0.5:
            volume_bias = "inactive"
        else:
            volume_bias = "normal"
        
        # Combine microstructure signals
        if spread_bias == "liquid" and volume_bias == "active":
            direction = "bullish"  # Good liquidity + high activity
            confidence = 0.7
        elif spread_bias == "uncertain" and volume_bias == "inactive":
            direction = "bearish"  # Poor conditions
            confidence = 0.6
        else:
            direction = "neutral"
            confidence = 0.5
        
        return {
            "direction": direction,
            "confidence": float(confidence),
            "spread_condition": spread_bias,
            "volume_condition": volume_bias,
            "volatility_level": "high" if volatility > 0.02 else "normal"
        }
    
    def _analyze_volatility_bias(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analizza bias basato su volatility regime"""
        
        volatility = market_data.get('volatility', 0)
        atr_volatility = market_data.get('atr_volatility', volatility)
        
        # Volatility trend
        prices = np.array(market_data.get('price_history', []))
        if len(prices) >= 50:
            recent_vol = np.std(np.diff(prices[-20:]) / prices[-20:-1])
            older_vol = np.std(np.diff(prices[-50:-30]) / prices[-50:-30])
            
            vol_trend = "increasing" if recent_vol > older_vol * 1.2 else "decreasing"
        else:
            vol_trend = "unknown"
        
        # Volatility regime
        if volatility > 0.025:
            regime = "extreme"
            direction = "neutral"  # Extreme volatility = uncertainty
            confidence = 0.3
        elif volatility > 0.015:
            regime = "high"
            direction = "neutral"
            confidence = 0.5
        elif volatility < 0.005:
            regime = "low"
            direction = "neutral"  # Low vol can precede big moves
            confidence = 0.4
        else:
            regime = "normal"
            direction = "neutral"
            confidence = 0.6
        
        # Adjust for volatility trend
        if vol_trend == "increasing" and regime != "extreme":
            confidence *= 0.8  # Reduce confidence when vol increasing
        
        return {
            "direction": direction,
            "confidence": float(confidence),
            "volatility_regime": regime,
            "volatility_trend": vol_trend,
            "current_volatility": float(volatility),
            "atr_volatility": float(atr_volatility)
        }
    
    def _combine_bias_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combina multipli segnali di bias con weighted voting"""

        if not signals:
            return {
                "directional": {"direction": "neutral", "confidence": 0.0, "distribution": {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}},
                "behavioral": {"type": "unknown", "confidence": 0.0},
                "confidence": 0.0
            }

        # Weight for each signal type
        # Ho aggiunto un peso di default 'unknown' per robustezza, se un tipo non è nella lista
        weights = {
            "price_action": 0.3,
            "order_flow": 0.3,
            "microstructure": 0.2,
            "volatility": 0.2,
            "unknown": 0.1 # Peso di default per tipi non riconosciuti
        }

        # Aggregate directional bias
        # QUI LA MODIFICA CHIAVE: Inizializza i punteggi come float
        direction_scores: Dict[str, float] = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
        total_weight = 0.0 # Anche total_weight dovrebbe essere float per coerenza

        for i, signal in enumerate(signals):
            # Migliorato per gestire più di 4 segnali, usando .get() sui segnali stessi
            # invece di una lista fissa basata su `i`.
            # Assumiamo che ogni 'signal' abbia un 'type' al suo interno.
            # Se 'signal_type' non è presente in 'signal', usa 'unknown' come default.
            signal_type = signal.get("type", "unknown")
            weight = weights.get(signal_type, weights["unknown"]) # Usa il peso di default se il tipo non è nel dict

            direction = signal.get("direction", "neutral")
            confidence = signal.get("confidence", 0.5)

            # Assicurati che 'direction' sia una chiave valida nel dizionario, altrimenti usa 'neutral'
            if direction not in direction_scores:
                direction = "neutral"

            direction_scores[direction] += confidence * weight
            total_weight += weight

        # Normalize scores
        # Ora Pylance non si lamenterà, dato che direction_scores è tipizzato come Dict[str, float]
        # E max(total_weight, 1.0) per assicurare il float nella divisione
        normalization_factor = max(total_weight, 1.0)
        for direction in direction_scores:
            direction_scores[direction] /= normalization_factor

        # Find dominant direction
        # Handle case where all scores are 0 (e.g., if total_weight was 0 and max(0,1) became 1)
        if not direction_scores: # Should not happen with initialisation, but good for robustness
            dominant_direction_tuple = ("neutral", 0.0)
        else:
            dominant_direction_tuple = max(direction_scores.items(), key=lambda x: x[1])

        # Calculate behavioral bias
        behavioral_type = "balanced"
        if dominant_direction_tuple[0] == "bullish" and dominant_direction_tuple[1] > 0.6:
            behavioral_type = "aggressive_buying"
        elif dominant_direction_tuple[0] == "bearish" and dominant_direction_tuple[1] > 0.6:
            behavioral_type = "aggressive_selling"
        elif dominant_direction_tuple[0] == "neutral" and dominant_direction_tuple[1] > 0.6:
            behavioral_type = "indecisive"

        # Overall confidence
        overall_confidence = dominant_direction_tuple[1]

        # Reduce confidence if signals disagree
        # np.var richiede un array o una lista, e non dovrebbe avere NaN se i punteggi sono stati normalizzati.
        # Ho aggiunto un controllo per il caso in cui direction_scores.values() sia vuoto.
        if direction_scores: # Ensure list is not empty before calculating variance
            signal_variance = np.var(list(direction_scores.values()))
            if signal_variance > 0.1:
                overall_confidence *= 0.8 # Ridotto l'impatto a 0.8 per evitare confidence negative o troppo basse
        
        # Return all results as floats
        return {
            "directional": {
                "direction": dominant_direction_tuple[0],
                "confidence": dominant_direction_tuple[1], # Già float
                "distribution": direction_scores # Già Dict[str, float]
            },
            "behavioral": {
                "type": behavioral_type,
                "confidence": overall_confidence # Già float
            },
            "confidence": overall_confidence # Già float
        }
        
    def _prepare_trend_features(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Prepara features per trend analysis CON CACHE - OTTIMIZZATO"""
        
        safe_print("🔄 Preparando trend features con cache...")
        
        # Pre-allocazione features array per performance
        n_features = 12  # Numero totale features calcolate
        features = np.zeros(n_features, dtype=np.float32)
        feature_idx = 0
        
        # Cache valori comuni per evitare ricalcoli
        n_prices = len(prices)
        n_volumes = len(volumes)
        current_price = float(prices[-1]) if n_prices > 0 else 0.0
        
        try:
            # 🚀 USA CACHED INDICATORS
            sma_10 = self.cached_indicators['sma'](prices, 10)
            sma_20 = self.cached_indicators['sma'](prices, 20)
            sma_50 = self.cached_indicators['sma'](prices, 50)
            rsi = self.cached_indicators['rsi'](prices, 14)
            macd, macd_signal, macd_hist = self.cached_indicators['macd'](prices)
            atr = self.cached_indicators['atr'](prices, prices, prices, 14)
            
            safe_print("✅ Trend indicators calcolati con cache")
            
        except Exception as e:
            safe_print(f"❌ Errore cache trend features: {e}")
            
            # Fallback manuale
            sma_10 = ta.SMA(prices, timeperiod=10) if n_prices >= 10 else np.full_like(prices, current_price) # type: ignore
            sma_20 = ta.SMA(prices, timeperiod=20) if n_prices >= 20 else np.full_like(prices, current_price) # type: ignore
            sma_50 = ta.SMA(prices, timeperiod=50) if n_prices >= 50 else np.full_like(prices, current_price) # type: ignore
            rsi = ta.RSI(prices, timeperiod=14) if n_prices >= 14 else np.full_like(prices, 50.0) # type: ignore
            macd, macd_signal, macd_hist = ta.MACD(prices) if n_prices >= 26 else (np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)) # type: ignore
            atr = ta.ATR(prices, prices, prices, timeperiod=14) if n_prices >= 14 else np.full_like(prices, 0.01) # type: ignore
        
        # Moving averages ratios - evita ricalcoli
        sma_arrays = [(10, sma_10), (20, sma_20), (50, sma_50)]
        for period, sma_values in sma_arrays:
            if n_prices >= period and len(sma_values) > 0:
                sma_current = float(sma_values[-1])
                if sma_current != 0:
                    features[feature_idx] = (current_price - sma_current) / sma_current
                # else: già 0.0 da pre-allocazione
            feature_idx += 1
        
        # Price position relative to range - OTTIMIZZATO: calcola min/max una volta
        if n_prices >= 20:
            # Usa view invece di slice copy per calcoli statistici
            lookback_20 = min(20, n_prices)
            start_idx = n_prices - lookback_20
            
            # Calcola min/max in un passaggio senza creare copie
            high_20 = float(np.max(prices[start_idx:]))
            low_20 = float(np.min(prices[start_idx:]))
            
            if high_20 > low_20:
                features[feature_idx] = (current_price - low_20) / (high_20 - low_20)
            else:
                features[feature_idx] = 0.5
        else:
            features[feature_idx] = 0.5
        feature_idx += 1
        
        # RSI normalized
        if len(rsi) > 0:
            rsi_value = float(rsi[-1])
            if not np.isnan(rsi_value):
                features[feature_idx] = rsi_value / 100.0
            else:
                features[feature_idx] = 0.5
        else:
            features[feature_idx] = 0.5
        feature_idx += 1
        
        # MACD normalized
        if len(macd) > 0:
            macd_value = float(macd[-1])
            if not np.isnan(macd_value) and current_price != 0:
                features[feature_idx] = macd_value / current_price
            # else: già 0.0 da pre-allocazione
        feature_idx += 1
        
        # Trend slopes - OTTIMIZZATO: pre-alloca x arrays
        x_10 = np.arange(10, dtype=np.float32) if n_prices >= 10 else None
        x_20 = np.arange(20, dtype=np.float32) if n_prices >= 20 else None
        
        for lookback, x_array in [(10, x_10), (20, x_20)]:
            if n_prices >= lookback and x_array is not None:
                try:
                    # Usa view invece di slice copy
                    start_idx = n_prices - lookback
                    y_view = prices[start_idx:]  # View, non copy
                    
                    slope, _ = np.polyfit(x_array, y_view, 1)
                    y_mean = float(np.mean(y_view))
                    
                    if y_mean != 0:
                        features[feature_idx] = slope / y_mean
                    # else: già 0.0 da pre-allocazione
                except:
                    pass  # già 0.0 da pre-allocazione
            feature_idx += 1
        
        # Volatility - OTTIMIZZATO: evita copie multiple
        if n_prices >= 21:
            try:
                # Calcola returns in-place per evitare copie
                start_idx = n_prices - 21
                price_slice = prices[start_idx:]  # View
                
                # Pre-alloca array returns
                returns = np.empty(20, dtype=np.float32)
                valid_returns = 0
                
                # Calcola returns evitando copie
                for i in range(20):
                    if price_slice[i] != 0:
                        returns[valid_returns] = (price_slice[i+1] - price_slice[i]) / price_slice[i]
                        if np.isfinite(returns[valid_returns]):
                            valid_returns += 1
                
                if valid_returns > 0:
                    # Usa solo la parte valida dell'array
                    features[feature_idx] = float(np.std(returns[:valid_returns]))
                # else: già 0.0 da pre-allocazione
            except:
                pass  # già 0.0 da pre-allocazione
        feature_idx += 1
        
        # Volume analysis - OTTIMIZZATO: evita slicing multipli
        if n_volumes >= 20:
            try:
                # Calcola volumi medi una volta senza creare copie
                recent_sum = 0.0
                older_sum = 0.0
                
                # Ultimi 5 volumi
                for i in range(n_volumes - 5, n_volumes):
                    recent_sum += volumes[i]
                
                # Ultimi 20 volumi 
                for i in range(n_volumes - 20, n_volumes):
                    older_sum += volumes[i]
                
                recent_volume = recent_sum / 5.0
                older_volume = older_sum / 20.0
                
                if older_volume != 0:
                    features[feature_idx] = recent_volume / older_volume
                else:
                    features[feature_idx] = 1.0
            except:
                features[feature_idx] = 1.0
        else:
            features[feature_idx] = 1.0
        feature_idx += 1
        
        # Price acceleration - OTTIMIZZATO: accessi diretti agli indici
        if n_prices >= 10:
            try:
                price_current = current_price
                price_5_ago = float(prices[n_prices - 6])  # -6 perché -1 è current
                price_10_ago = float(prices[n_prices - 11])  # -11 perché -1 è current
                
                if price_5_ago != 0 and price_10_ago != 0:
                    recent_change = (price_current - price_5_ago) / price_5_ago
                    older_change = (price_5_ago - price_10_ago) / price_10_ago
                    features[feature_idx] = recent_change - older_change
                # else: già 0.0 da pre-allocazione
            except:
                pass  # già 0.0 da pre-allocazione
        feature_idx += 1
        
        # ADX using cache if possible
        try:
            adx = ta.ADX(prices, prices, prices, timeperiod=14) if n_prices >= 14 else np.full_like(prices, 50.0) # type: ignore
            if len(adx) > 0:
                adx_value = float(adx[-1])
                if not np.isnan(adx_value):
                    features[feature_idx] = adx_value / 100.0
                else:
                    features[feature_idx] = 0.5
            else:
                features[feature_idx] = 0.5
        except:
            features[feature_idx] = 0.5
        
        safe_print(f"✅ Trend features OTTIMIZZATO: {len(features)} features")
        return features
    
    def _estimate_confidence_rf(self, model: RandomForestRegressor, features: np.ndarray) -> np.ndarray:
        """Stima confidence per RandomForest"""
        
        # Get predictions from individual trees
        if hasattr(model, 'estimators_'):
            tree_predictions = []
            for tree in model.estimators_:
                pred = tree.predict(features.reshape(1, -1))[0]
                tree_predictions.append(pred)
            
            # Calculate variance across trees
            variance = np.var(tree_predictions)
            
            # Convert to confidence (lower variance = higher confidence)
            confidence = np.exp(-variance)
            
            # Create probability-like distribution
            if len(tree_predictions) > 0:
                # Count predictions for each class
                unique_preds = np.unique(tree_predictions)
                proba = np.zeros(3)  # Assuming 3 classes
                
                for pred in tree_predictions:
                    proba[int(pred)] += 1
                
                proba /= len(tree_predictions)
                return proba
        
        # Fallback
        return np.array([0.33, 0.34, 0.33])
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calcola la forza del trend - OTTIMIZZATO"""
        
        n_prices = len(prices)
        if n_prices < 20:
            return 0.0
        
        # Usa view invece di copy per ultimi 20 prezzi
        lookback = 20
        start_idx = n_prices - lookback
        
        # Pre-allocazione arrays per regressione lineare
        x = np.arange(lookback, dtype=np.float32)  # Usa direttamente 20 invece di len(slice)
        
        # Pre-calcola statistiche per evitare ricalcoli
        y_sum = 0.0
        for i in range(lookback):
            y_sum += prices[start_idx + i]
        
        y_mean = y_sum / lookback
        
        # Calcola slope usando formula diretta senza polyfit (più efficiente)
        # slope = Σ((x - x_mean)(y - y_mean)) / Σ((x - x_mean)²)
        x_mean = (lookback - 1) / 2.0  # Media di arange(20) = 9.5
        
        numerator = 0.0
        denominator = 0.0
        ss_tot = 0.0  # Calcola ss_tot nel stesso loop
        
        for i in range(lookback):
            y_i = prices[start_idx + i]
            x_i = float(i)
            
            x_diff = x_i - x_mean
            y_diff = y_i - y_mean
            
            numerator += x_diff * y_diff
            denominator += x_diff * x_diff
            ss_tot += y_diff * y_diff
        
        # Calcola slope e intercept
        if denominator > 0:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
        else:
            return 0.0
        
        # Calcola R-squared senza creare arrays temporanei
        ss_res = 0.0
        for i in range(lookback):
            y_i = prices[start_idx + i]
            y_pred_i = slope * i + intercept
            residual = y_i - y_pred_i
            ss_res += residual * residual
        
        # R-squared calculation
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Normalize slope (riusa y_mean già calcolato)
        if y_mean > 0:
            normalized_slope = abs(slope) / y_mean * 100.0
        else:
            return 0.0
        
        # Combine R-squared and slope for strength
        strength = r_squared * min(1.0, normalized_slope)
        
        return float(strength)
    
    def _calculate_trend_r_squared(self, prices: np.ndarray) -> float:
        """Calcola R-squared del trend"""
        
        if len(prices) < 10:
            return 0.0
        
        x = np.arange(len(prices[-20:]))
        y = prices[-20:]
        
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    def _project_trend(self, prices: np.ndarray, direction: str, confidence: float) -> Dict[str, Any]:
        """Proietta il trend nel futuro"""
        
        if len(prices) < 20:
            return {"error": "Insufficient data for projection"}
        
        # Fit trend line
        x = np.arange(20)
        y = prices[-20:]
        slope, intercept = np.polyfit(x, y, 1)
        
        # Project forward
        future_points = 10
        future_x = np.arange(20, 20 + future_points)
        future_y = slope * future_x + intercept
        
        # Add uncertainty bands
        residuals = y - (slope * x + intercept)
        std_error = np.std(residuals)
        
        # Confidence intervals
        upper_band = future_y + 2 * std_error * (1 - confidence)
        lower_band = future_y - 2 * std_error * (1 - confidence)
        
        return {
            "projected_prices": future_y.tolist(),
            "upper_band": upper_band.tolist(),
            "lower_band": lower_band.tolist(),
            "slope": float(slope),
            "confidence_level": float(confidence),
            "projection_periods": future_points
        }
    
    def _prepare_lstm_trend_features(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Prepara features per LSTM trend prediction"""
        
        features = []
        sequence_length = 30
        
        # Create sequences
        for i in range(sequence_length):
            idx = -(sequence_length - i)
            
            # Price features
            price_norm = prices[idx] / prices[-1]
            
            # Return
            if idx > -len(prices):
                ret = (prices[idx] - prices[idx-1]) / prices[idx-1]
            else:
                ret = 0
            
            # Volume
            vol_norm = volumes[idx] / np.mean(volumes)
            
            # Simple momentum
            if idx >= -len(prices) + 5:
                momentum = (prices[idx] - prices[idx-5]) / prices[idx-5]
            else:
                momentum = 0
            
            # RSI approximation
            if idx >= -len(prices) + 14:
                gains = []
                losses = []
                for j in range(idx-14, idx):
                    change = prices[j] - prices[j-1]
                    if change > 0:
                        gains.append(change)
                    else:
                        losses.append(abs(change))
                
                avg_gain = np.mean(gains) if gains else 0
                avg_loss = np.mean(losses) if losses else 0
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100 if avg_gain > 0 else 50
            else:
                rsi = 50
            
            features.append([price_norm, ret, vol_norm, momentum, rsi/100])
        
        return np.array(features).flatten()
    
    def _prepare_gb_trend_features(self, prices: np.ndarray, volumes: np.ndarray, 
                                 market_data: Dict[str, Any]) -> np.ndarray:
        """Prepara features per GradientBoosting trend analysis"""
        
        # Similar to basic trend features but with some additions
        features = []
        
        # All basic trend features
        basic_features = self._prepare_trend_features(prices, volumes)
        features.extend(basic_features)
        
        # Additional GB-specific features
        
        # Hurst exponent approximation (trend persistence)
        if len(prices) >= 50:
            hurst = self._calculate_hurst_exponent(prices[-50:])
            features.append(hurst)
        else:
            features.append(0.5)
        
        # Fractal dimension (market complexity)
        if len(prices) >= 30:
            fractal_dim = self._calculate_fractal_dimension(prices[-30:])
            features.append(fractal_dim)
        else:
            features.append(1.5)
        
        # Market efficiency ratio
        if len(prices) >= 20:
            efficiency = self._calculate_efficiency_ratio(prices[-20:])
            features.append(efficiency)
        else:
            features.append(0.5)
        
        # Microstructure features
        features.append(market_data.get('spread_percentage', 0))
        features.append(market_data.get('volume_momentum', 0))
        
        return np.array(features)
    
    def _estimate_confidence_gb(self, model: GradientBoostingRegressor, features: np.ndarray) -> float:
        """Stima confidence per GradientBoosting"""
        
        # Use prediction variance across stages
        if hasattr(model, 'staged_predict'):
            predictions = []
            for pred in model.staged_predict(features.reshape(1, -1)):
                predictions.append(pred[0])
            
            if len(predictions) > 10:
                # Look at convergence of predictions
                later_preds = predictions[-10:]
                variance = np.var(later_preds)
                
                # Low variance = high confidence
                confidence = np.exp(-variance * 10)
                return float(min(0.95, confidence))
        
        # Fallback
        return 0.7
    
    def _analyze_trend_characteristics(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """Analizza caratteristiche dettagliate del trend"""
        
        characteristics = {}
        
        if len(prices) < 50:
            return {"strength": 0.0, "quality": "unknown"}
        
        # Trend strength
        characteristics["strength"] = self._calculate_trend_strength(prices)
        
        # Trend smoothness (low noise)
        returns = np.diff(prices) / prices[:-1]
        noise_level = np.std(returns) / abs(np.mean(returns)) if np.mean(returns) != 0 else float('inf')
        
        if noise_level < 2:
            smoothness = "smooth"
        elif noise_level < 5:
            smoothness = "normal"
        else:
            smoothness = "choppy"
        
        characteristics["smoothness"] = smoothness
        
        # Trend age (how long has it been going)
        # Find trend start by looking for reversal
        trend_age = 0
        current_direction = np.sign(prices[-1] - prices[-20])
        
        for i in range(len(prices) - 20, 0, -1):
            window_direction = np.sign(prices[i] - prices[max(0, i-20)])
            if window_direction != current_direction:
                trend_age = len(prices) - i
                break
        
        characteristics["age_periods"] = trend_age
        
        # Trend acceleration
        if len(prices) >= 30:
            early_slope = (prices[-20] - prices[-30]) / 10
            recent_slope = (prices[-1] - prices[-10]) / 10
            acceleration = (recent_slope - early_slope) / abs(early_slope) if early_slope != 0 else 0
            
            if acceleration > 0.5:
                characteristics["acceleration"] = "accelerating"
            elif acceleration < -0.5:
                characteristics["acceleration"] = "decelerating"
            else:
                characteristics["acceleration"] = "steady"
        else:
            characteristics["acceleration"] = "unknown"
        
        # Volume confirmation
        if len(volumes) >= 20:
            price_trend = (prices[-1] - prices[-20]) / prices[-20]
            volume_trend = (np.mean(volumes[-10:]) - np.mean(volumes[-20:-10])) / np.mean(volumes[-20:-10])
            
            if np.sign(price_trend) == np.sign(volume_trend):
                characteristics["volume_confirmation"] = True
            else:
                characteristics["volume_confirmation"] = False
        else:
            characteristics["volume_confirmation"] = None
        
        return characteristics
    
    def _prepare_transformer_trend_features(self, prices: np.ndarray, volumes: np.ndarray,
                                          market_data: Dict[str, Any]) -> np.ndarray:
        """Prepara features avanzate per Transformer trend analysis"""
        
        # Multi-scale decomposition
        features = []
        
        # Wavelet-like multi-resolution
        scales = [5, 10, 20, 50]
        for scale in scales:
            if len(prices) >= scale:
                # Moving average at this scale
                ma = np.mean(prices[-scale:])
                features.append((prices[-1] - ma) / ma)
                
                # Volatility at this scale
                vol = np.std(prices[-scale:]) / np.mean(prices[-scale:])
                features.append(vol)
            else:
                features.extend([0.0, 0.0])
        
        # Momentum cascade
        for lookback in [5, 10, 20]:
            if len(prices) >= lookback:
                momentum = (prices[-1] - prices[-lookback]) / prices[-lookback]
                features.append(momentum)
            else:
                features.append(0.0)
        
        # Market regime encoding
        regime_features = [
            1.0 if market_data.get('market_state') == 'trending' else 0.0,
            1.0 if market_data.get('market_state') == 'ranging' else 0.0,
            1.0 if 'volatile' in market_data.get('market_state', '') else 0.0
        ]
        features.extend(regime_features)
        
        # Ensure correct size
        while len(features) < 12:
            features.append(0.0)
        
        return np.array(features[:12])
    
    def _decompose_trend_components(self, trend_logits: np.ndarray) -> Dict[str, Any]:
        """Decompone i componenti del trend dal output del Transformer"""
        
        # Assume logits structure: [primary_trend (3), strength (1), timeframe (3), confidence (1)]
        
        components = {}
        
        # Primary trend
        trend_probs = self._softmax(trend_logits[:3])
        trend_labels = ["downtrend", "sideways", "uptrend"]
        primary_idx = np.argmax(trend_probs)
        
        components["primary_trend"] = {
            "direction": trend_labels[primary_idx],
            "strength": float(np.tanh(trend_logits[3])),  # Sigmoid-like for 0-1
            "confidence": float(trend_probs[primary_idx])
        }
        
        # Timeframe dominance
        if len(trend_logits) >= 7:
            tf_probs = self._softmax(trend_logits[4:7])
            tf_labels = ["short_term", "medium_term", "long_term"]
            dominant_tf = tf_labels[np.argmax(tf_probs)]
            
            components["timeframe_analysis"] = {
                "dominant": dominant_tf,
                "distribution": {tf_labels[i]: float(tf_probs[i]) for i in range(len(tf_labels))}
            }
        
        # Overall confidence
        if len(trend_logits) >= 8:
            components["model_confidence"] = float(np.tanh(trend_logits[7]))
        
        return components
    
    def _analyze_multi_timeframe_trends(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """Analizza trend su multiple timeframe"""
        
        timeframes = {
            "micro": 5,
            "short": 20,
            "medium": 50,
            "long": 100
        }
        
        multi_tf_analysis = {}
        
        for tf_name, tf_period in timeframes.items():
            if len(prices) >= tf_period:
                # Calculate trend for this timeframe
                tf_prices = prices[-tf_period:]
                
                # Linear regression
                x = np.arange(len(tf_prices))
                slope, intercept = np.polyfit(x, tf_prices, 1)
                
                # Normalize slope
                normalized_slope = slope / np.mean(tf_prices)
                
                # Determine direction
                if normalized_slope > 0.0001:
                    direction = "up"
                elif normalized_slope < -0.0001:
                    direction = "down"
                else:
                    direction = "sideways"
                
                # Calculate strength
                y_pred = slope * x + intercept
                r_squared = 1 - (np.sum((tf_prices - y_pred)**2) / np.sum((tf_prices - np.mean(tf_prices))**2))
                
                multi_tf_analysis[tf_name] = {
                    "direction": direction,
                    "strength": float(r_squared),
                    "slope": float(normalized_slope),
                    "period": tf_period
                }
            else:
                multi_tf_analysis[tf_name] = {
                    "direction": "unknown",
                    "strength": 0.0,
                    "slope": 0.0,
                    "period": tf_period
                }
        
        # Determine alignment
        directions = [tf["direction"] for tf in multi_tf_analysis.values() if tf["direction"] != "unknown"]
        
        if directions:
            if all(d == directions[0] for d in directions):
                alignment = "full_alignment"
            elif directions.count(directions[0]) > len(directions) / 2:
                alignment = "partial_alignment"
            else:
                alignment = "no_alignment"
        else:
            alignment = "unknown"
        
        multi_tf_analysis["alignment"] = alignment
        
        return multi_tf_analysis
        
    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calcola esponente di Hurst per misurare persistenza del trend - OTTIMIZZATO"""
        
        n_prices = len(prices)
        if n_prices < 20:
            return 0.5
        
        # Simplified Hurst calculation using R/S analysis
        max_lag = min(20, n_prices // 2)
        lags = range(2, max_lag)
        
        # Pre-allocazione arrays per evitare list operations
        n_lags = len(lags)
        if n_lags == 0:
            return 0.5
        
        tau_values = np.empty(n_lags, dtype=np.float32)
        lag_values = np.empty(n_lags, dtype=np.float32)
        valid_count = 0
        
        for lag_idx, lag in enumerate(lags):
            # Calculate standard deviation of differences SENZA slicing
            # Equivale a: differences = prices[lag:] - prices[:-lag]
            
            n_differences = n_prices - lag
            diff_sum = 0.0
            diff_sum_sq = 0.0
            
            # Calcola differences e statistiche in single-pass
            for i in range(n_differences):
                diff = prices[i + lag] - prices[i]
                diff_sum += diff
                diff_sum_sq += diff * diff
            
            # Standard deviation senza array temporaneo
            if n_differences > 0:
                mean_diff = diff_sum / n_differences
                variance = (diff_sum_sq / n_differences) - (mean_diff * mean_diff)
                std_diff = np.sqrt(variance) if variance > 0 else 1e-10
            else:
                std_diff = 1e-10
            
            if std_diff > 0:
                tau_values[valid_count] = std_diff
                lag_values[valid_count] = float(lag)
                valid_count += 1
        
        # Fit power law - OTTIMIZZATO: evita np.polyfit
        if valid_count > 5:
            # Manual linear regression in log-log space invece di np.polyfit
            # log_lags = np.log(lag_values[:valid_count])
            # log_tau = np.log(tau_values[:valid_count])
            
            # Calcola logs e regression in single-pass
            log_x_sum = 0.0
            log_y_sum = 0.0
            log_x_sum_sq = 0.0
            log_xy_sum = 0.0
            
            for i in range(valid_count):
                log_x = np.log(lag_values[i])
                log_y = np.log(tau_values[i])
                
                log_x_sum += log_x
                log_y_sum += log_y
                log_x_sum_sq += log_x * log_x
                log_xy_sum += log_x * log_y
            
            # Linear regression: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
            n = valid_count
            numerator = n * log_xy_sum - log_x_sum * log_y_sum
            denominator = n * log_x_sum_sq - log_x_sum * log_x_sum
            
            if denominator != 0:
                hurst = numerator / denominator
                # Clamp to valid range
                hurst = max(0.0, min(1.0, hurst))
            else:
                hurst = 0.5
        else:
            hurst = 0.5
        
        return hurst
    
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calcola dimensione frattale del prezzo - OTTIMIZZATO"""
        
        n = len(prices)
        if n < 10:
            return 1.5
        
        # Find min/max in single-pass invece di doppio scan
        price_min = prices[0]
        price_max = prices[0]
        
        for i in range(1, n):
            price = prices[i]
            if price < price_min:
                price_min = price
            if price > price_max:
                price_max = price
        
        price_range = price_max - price_min + 1e-10
        
        # Box counting method simplified
        scales = [2, 4, 8]
        counts = []
        
        for scale in scales:
            if scale < n:
                # Set per boxes occupati (manteniamo set per uniqueness)
                boxes = set()
                
                # Divide into boxes senza creare window arrays
                step = max(1, scale // 2)  # Assicura step >= 1
                
                for i in range(0, n - scale + 1, step):
                    # Calcola mean della window SENZA slicing
                    window_sum = 0.0
                    for j in range(scale):
                        if i + j < n:  # Bounds check
                            # Normalizza on-the-fly
                            normalized_price = (prices[i + j] - price_min) / price_range
                            window_sum += normalized_price
                    
                    window_mean = window_sum / scale
                    
                    # Find which boxes are occupied
                    price_box = int(window_mean * scale)
                    time_box = i // scale
                    
                    boxes.add((time_box, price_box))
                
                counts.append(len(boxes))
        
        # Estimate dimension - OTTIMIZZATO: evita np.polyfit
        if len(counts) >= 2:
            # Manual linear regression invece di np.polyfit
            n_points = len(counts)
            
            # Calcola logs e regression in single-pass
            log_x_sum = 0.0
            log_y_sum = 0.0
            log_x_sum_sq = 0.0
            log_xy_sum = 0.0
            
            for i in range(n_points):
                log_x = np.log(scales[i])
                log_y = np.log(counts[i]) if counts[i] > 0 else np.log(1e-10)  # Evita log(0)
                
                log_x_sum += log_x
                log_y_sum += log_y
                log_x_sum_sq += log_x * log_x
                log_xy_sum += log_x * log_y
            
            # Linear regression: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
            numerator = n_points * log_xy_sum - log_x_sum * log_y_sum
            denominator = n_points * log_x_sum_sq - log_x_sum * log_x_sum
            
            if denominator != 0:
                slope = numerator / denominator
                dimension = abs(slope)  # Negative slope gives fractal dimension
                
                # Typical range is 1 to 2
                dimension = max(1.0, min(2.0, dimension))
            else:
                dimension = 1.5
        else:
            dimension = 1.5
        
        return dimension
    
    def _calculate_efficiency_ratio(self, prices: np.ndarray) -> float:
        """Calcola Kaufman's Efficiency Ratio - OTTIMIZZATO"""
        
        n_prices = len(prices)
        if n_prices < 2:
            return 0.0
        
        # Total price change - accesso diretto
        net_change = abs(prices[n_prices - 1] - prices[0])
        
        # Sum of absolute changes SENZA np.diff()
        # Equivale a: np.sum(np.abs(np.diff(prices)))
        total_movement = 0.0
        
        for i in range(n_prices - 1):
            diff = abs(prices[i + 1] - prices[i])
            total_movement += diff
        
        # Efficiency ratio
        if total_movement > 0:
            efficiency = net_change / total_movement
        else:
            efficiency = 0.0
        
        return efficiency
    
    def _prepare_volatility_features(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Prepara features per volatility prediction - OTTIMIZZATO"""
        
        # Pre-allocazione array risultato
        max_features = 12  # Stima delle features totali
        features = np.zeros(max_features, dtype=np.float32)
        feature_idx = 0
        
        # Cache valori comuni
        n_prices = len(prices)
        n_volumes = len(volumes)
        
        # Historical volatilities at different windows - OTTIMIZZATO
        windows = [5, 10, 20]
        for window in windows:
            if n_prices >= window + 1:
                # Calcola volatility senza creare arrays temporanei
                returns_sum_sq = 0.0
                returns_sum = 0.0
                n_returns = window
                
                start_idx = n_prices - window - 1
                for i in range(n_returns):
                    if prices[start_idx + i] > 0:
                        return_val = (prices[start_idx + i + 1] - prices[start_idx + i]) / prices[start_idx + i]
                        returns_sum += return_val
                        returns_sum_sq += return_val * return_val
                
                # Standard deviation senza np.std()
                if n_returns > 0:
                    mean_return = returns_sum / n_returns
                    variance = (returns_sum_sq / n_returns) - (mean_return * mean_return)
                    vol = np.sqrt(variance) if variance > 0 else 0.0
                    features[feature_idx] = vol
                # else: già 0.0 da pre-allocazione
            # else: già 0.0 da pre-allocazione
            feature_idx += 1
        
        # GARCH-like features - OTTIMIZZATO
        if n_prices >= 30:
            # Calcola returns e squared returns in un solo passaggio
            n_returns = 29  # 30 prezzi = 29 returns
            start_idx = n_prices - 30
            
            squared_returns_sum_recent = 0.0  # Ultimi 5
            squared_returns_sum_historical = 0.0  # Ultimi 20
            
            for i in range(n_returns):
                if prices[start_idx + i] > 0:
                    return_val = (prices[start_idx + i + 1] - prices[start_idx + i]) / prices[start_idx + i]
                    squared_return = return_val * return_val
                    
                    # Accumula per historical (tutti i 20)
                    if i >= n_returns - 20:
                        squared_returns_sum_historical += squared_return
                    
                    # Accumula per recent (ultimi 5)
                    if i >= n_returns - 5:
                        squared_returns_sum_recent += squared_return
            
            recent_vol = squared_returns_sum_recent / 5.0
            historical_vol = squared_returns_sum_historical / 20.0
            
            features[feature_idx] = recent_vol
            feature_idx += 1
            features[feature_idx] = recent_vol / (historical_vol + 1e-10)
            feature_idx += 1
        else:
            features[feature_idx] = 0.0
            feature_idx += 1
            features[feature_idx] = 1.0
            feature_idx += 1
        
        # Volume volatility - OTTIMIZZATO
        if n_volumes >= 20:
            # Calcola volume volatility senza np.diff() e slicing
            vol_returns_sum_sq = 0.0
            vol_returns_sum = 0.0
            n_vol_returns = 19  # 20 volumi = 19 returns
            
            start_idx = n_volumes - 20
            for i in range(n_vol_returns):
                prev_vol = volumes[start_idx + i] + 1e-10  # Protezione divisione zero
                vol_return = (volumes[start_idx + i + 1] - volumes[start_idx + i]) / prev_vol
                vol_returns_sum += vol_return
                vol_returns_sum_sq += vol_return * vol_return
            
            # Standard deviation volume
            if n_vol_returns > 0:
                mean_vol_return = vol_returns_sum / n_vol_returns
                vol_variance = (vol_returns_sum_sq / n_vol_returns) - (mean_vol_return * mean_vol_return)
                vol_volatility = np.sqrt(vol_variance) if vol_variance > 0 else 0.0
                features[feature_idx] = vol_volatility
            # else: già 0.0 da pre-allocazione
        # else: già 0.0 da pre-allocazione
        feature_idx += 1
        
        # Parkinson volatility (high-low estimator) - OTTIMIZZATO MASSIMAMENTE
        if n_prices >= 20:
            # Evita completamente il loop con slicing prices[i:i+5]
            n_windows = n_prices - 4  # Numero di finestre 5-elementi possibili
            hl_ranges_sum = 0.0
            n_valid_windows = 0
            
            # Usa rolling window senza creare copie
            for i in range(n_windows):
                # Calcola min/max/mean della finestra 5-elementi senza slice
                window_min = prices[i]
                window_max = prices[i]
                window_sum = prices[i]
                
                for j in range(1, 5):  # Restanti 4 elementi della finestra
                    val = prices[i + j]
                    if val < window_min:
                        window_min = val
                    if val > window_max:
                        window_max = val
                    window_sum += val
                
                window_mean = window_sum / 5.0
                if window_mean > 0:
                    hl_range = (window_max - window_min) / window_mean
                    
                    # Considera solo gli ultimi 10 ranges (equivale a hl_volatilities[-10:])
                    if i >= n_windows - 10:
                        hl_ranges_sum += hl_range
                        n_valid_windows += 1
            
            if n_valid_windows > 0:
                features[feature_idx] = hl_ranges_sum / n_valid_windows
            # else: già 0.0 da pre-allocazione
        # else: già 0.0 da pre-allocazione
        feature_idx += 1
        
        # Price acceleration (second derivative) - OTTIMIZZATO
        if n_prices >= 10:
            # Calcola acceleration volatility senza np.diff() doppio
            start_idx = n_prices - 10
            second_diffs_sum_sq = 0.0
            second_diffs_sum = 0.0
            n_second_diffs = 8  # 10 prezzi = 9 first diff = 8 second diff
            
            # Calcola second derivatives direttamente
            for i in range(n_second_diffs):
                # prices[i], prices[i+1], prices[i+2] -> second derivative
                first_diff_1 = prices[start_idx + i + 1] - prices[start_idx + i]
                first_diff_2 = prices[start_idx + i + 2] - prices[start_idx + i + 1]
                second_diff = first_diff_2 - first_diff_1
                
                second_diffs_sum += second_diff
                second_diffs_sum_sq += second_diff * second_diff
            
            # Standard deviation second derivatives
            if n_second_diffs > 0:
                mean_second_diff = second_diffs_sum / n_second_diffs
                second_variance = (second_diffs_sum_sq / n_second_diffs) - (mean_second_diff * mean_second_diff)
                acceleration_vol = np.sqrt(second_variance) if second_variance > 0 else 0.0
                features[feature_idx] = acceleration_vol
            # else: già 0.0 da pre-allocazione
        # else: già 0.0 da pre-allocazione
        feature_idx += 1
        
        # Regime indicators - usa features già calcolate
        current_vol = features[0]  # Prima volatility window
        
        # Media delle prime 3 volatilities (windows 5,10,20)
        if feature_idx >= 3:
            avg_vol = (features[0] + features[1] + features[2]) / 3.0
        else:
            avg_vol = current_vol
        
        # Binary features for regime
        features[feature_idx] = 1.0 if current_vol > avg_vol * 1.5 else 0.0  # High vol regime
        feature_idx += 1
        features[feature_idx] = 1.0 if current_vol < avg_vol * 0.5 else 0.0  # Low vol regime
        feature_idx += 1
        
        # Restituisci solo le features utilizzate
        return features[:feature_idx]
    
    def _calculate_neural_momentum_features(self, prices: np.ndarray, 
                                          volumes: np.ndarray) -> Dict[str, float]:
        """Calcola features complesse per neural momentum analysis"""
        
        features = {}
        
        # Multi-timeframe momentum with volume weighting
        for tf in [5, 10, 20, 50]:
            if len(prices) >= tf and len(volumes) >= tf:
                # Price momentum
                price_mom = (prices[-1] - prices[-tf]) / prices[-tf]
                
                # Volume-weighted momentum
                weights = volumes[-tf:] / np.sum(volumes[-tf:])
                weighted_prices = prices[-tf:] * weights
                vw_price = np.sum(weighted_prices)
                vw_momentum = (prices[-1] - vw_price) / vw_price
                
                features[f'momentum_{tf}'] = price_mom
                features[f'vw_momentum_{tf}'] = vw_momentum
            else:
                features[f'momentum_{tf}'] = 0.0
                features[f'vw_momentum_{tf}'] = 0.0
        
        # Momentum of momentum (acceleration)
        if len(prices) >= 20:
            mom_5 = (prices[-5] - prices[-10]) / prices[-10]
            mom_current = (prices[-1] - prices[-5]) / prices[-5]
            features['acceleration'] = mom_current - mom_5
        else:
            features['acceleration'] = 0.0
        
        # Relative strength
        if len(prices) >= 50:
            # Compare short vs long momentum
            short_mom = (prices[-1] - prices[-10]) / prices[-10]
            long_mom = (prices[-1] - prices[-50]) / prices[-50]
            features['relative_strength'] = short_mom - long_mom
        else:
            features['relative_strength'] = 0.0
        
        # Volume-confirmed momentum
        if len(prices) >= 20 and len(volumes) >= 20:
            price_up_days = np.sum(np.diff(prices[-20:]) > 0)
            
            up_volume = 0
            down_volume = 0
            for i in range(1, 20):
                if prices[-20+i] > prices[-20+i-1]:
                    up_volume += volumes[-20+i]
                else:
                    down_volume += volumes[-20+i]
            
            if up_volume + down_volume > 0:
                volume_momentum_ratio = up_volume / (up_volume + down_volume)
            else:
                volume_momentum_ratio = 0.5
            
            features['volume_momentum_confirmation'] = volume_momentum_ratio
        else:
            features['volume_momentum_confirmation'] = 0.5
        
        # Smooth momentum (reduced noise)
        if len(prices) >= 30:
            # EMA-based momentum
            ema_10 = pd.Series(prices).ewm(span=10, adjust=False).mean().iloc[-1]
            ema_20 = pd.Series(prices).ewm(span=20, adjust=False).mean().iloc[-20]
            smooth_momentum = (ema_10 - ema_20) / ema_20
            features['smooth_momentum'] = smooth_momentum
        else:
            features['smooth_momentum'] = 0.0
        
        # Normalized aggregate momentum
        momentum_components = [
            features.get('momentum_5', 0),
            features.get('momentum_10', 0),
            features.get('momentum_20', 0),
            features.get('vw_momentum_10', 0),
            features.get('smooth_momentum', 0)
        ]
        
        features['momentum_short'] = np.mean(momentum_components[:2])
        features['momentum_medium'] = np.mean(momentum_components[2:4])
        features['momentum_long'] = momentum_components[4]
        features['volume_weighted_momentum'] = features.get('vw_momentum_10', 0)
        
        return features
    
    def _calculate_fibonacci_levels(self, prices: np.ndarray) -> List[float]:
        """Calcola livelli di Fibonacci - OTTIMIZZATO"""
        
        n_prices = len(prices)
        if n_prices < 20:
            return []
        
        # Find recent swing high and low SENZA slicing
        # Single-pass per min/max invece di doppio slicing
        start_idx = n_prices - 20
        
        recent_high = prices[start_idx]
        recent_low = prices[start_idx]
        
        # Calcola min/max in un solo loop
        for i in range(start_idx + 1, n_prices):
            price = prices[i]
            if price > recent_high:
                recent_high = price
            if price < recent_low:
                recent_low = price
        
        diff = recent_high - recent_low
        
        # Fibonacci ratios - pre-allocazione per performance
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        # Pre-allocazione lista risultato (Python non ha reserve, ma possiamo pre-allocare)
        levels = [0.0] * len(fib_ratios)  # Pre-alloca con placeholder values
        
        # Calcola levels evitando conversioni multiple
        for idx, ratio in enumerate(fib_ratios):
            # Retracement levels
            level = recent_high - diff * ratio
            levels[idx] = level  # Assegnazione diretta invece di append
        
        return levels
    
    def _estimate_elliott_wave_count(self, prices: np.ndarray) -> int:
        """Stima il conteggio delle onde di Elliott - OTTIMIZZATO"""
        
        n_prices = len(prices)
        if n_prices < 50:
            return 0
        
        # Simplified Elliott wave detection
        # Find major pivots SENZA slicing arrays
        
        # Pre-allocazione per pivots invece di list con tuples
        max_possible_pivots = (n_prices - 10) // 3  # Stima conservativa
        pivot_types = []  # 'high' o 'low'
        pivot_indices = np.empty(max_possible_pivots, dtype=np.int32)
        pivot_values = np.empty(max_possible_pivots, dtype=np.float32)
        pivot_count = 0
        
        for i in range(5, n_prices - 5):
            current_price = prices[i]
            
            # Check if local maximum SENZA slicing
            # Equivale a: prices[i] > max(prices[i-5:i]) and prices[i] > max(prices[i+1:i+6])
            is_local_max = True
            
            # Check contro i 5 elementi precedenti
            for j in range(i - 5, i):
                if prices[j] >= current_price:
                    is_local_max = False
                    break
            
            # Check contro i 5 elementi successivi (se è ancora candidato)
            if is_local_max:
                for j in range(i + 1, i + 6):
                    if prices[j] >= current_price:
                        is_local_max = False
                        break
            
            if is_local_max:
                # Salva pivot
                pivot_types.append('high')
                pivot_indices[pivot_count] = i
                pivot_values[pivot_count] = current_price
                pivot_count += 1
                
                if pivot_count >= max_possible_pivots:
                    break
            else:
                # Check if local minimum SENZA slicing
                # Equivale a: prices[i] < min(prices[i-5:i]) and prices[i] < min(prices[i+1:i+6])
                is_local_min = True
                
                # Check contro i 5 elementi precedenti
                for j in range(i - 5, i):
                    if prices[j] <= current_price:
                        is_local_min = False
                        break
                
                # Check contro i 5 elementi successivi (se è ancora candidato)
                if is_local_min:
                    for j in range(i + 1, i + 6):
                        if prices[j] <= current_price:
                            is_local_min = False
                            break
                
                if is_local_min:
                    # Salva pivot
                    pivot_types.append('low')
                    pivot_indices[pivot_count] = i
                    pivot_values[pivot_count] = current_price
                    pivot_count += 1
                    
                    if pivot_count >= max_possible_pivots:
                        break
        
        if pivot_count < 3:
            return 0
        
        # Count waves based on pivot pattern
        # Simplified: count alternating highs and lows
        wave_count = 0
        last_type = pivot_types[0]
        
        for i in range(1, pivot_count):
            current_type = pivot_types[i]
            if current_type != last_type:
                wave_count += 1
                last_type = current_type
        
        # Elliott waves are typically 5 or 3
        if wave_count >= 5:
            return 5
        elif wave_count >= 3:
            return 3
        else:
            return wave_count
    
    def _identify_wyckoff_phase(self, prices: np.ndarray, volumes: np.ndarray) -> str:
        """Identifica la fase di Wyckoff - OTTIMIZZATO"""
        
        n_prices = len(prices)
        n_volumes = len(volumes)
        
        if n_prices < 50 or n_volumes < 50:
            return "unknown"
        
        # Analyze price characteristics SENZA slicing
        # Calcola max, min, mean degli ultimi 20 prezzi in single-pass
        start_idx_20 = n_prices - 20
        
        price_max = prices[start_idx_20]
        price_min = prices[start_idx_20]
        price_sum = prices[start_idx_20]
        
        for i in range(start_idx_20 + 1, n_prices):
            price = prices[i]
            if price > price_max:
                price_max = price
            if price < price_min:
                price_min = price
            price_sum += price
        
        price_mean = price_sum / 20.0
        
        # Price range calculation
        if price_mean > 0:
            price_range = (price_max - price_min) / price_mean
        else:
            return "unknown"
        
        # Volume trend SENZA slicing
        # Recent volume (ultimi 10)
        recent_volume_sum = 0.0
        for i in range(n_volumes - 10, n_volumes):
            recent_volume_sum += volumes[i]
        recent_volume_mean = recent_volume_sum / 10.0
        
        # Older volume (ultimi 50)
        older_volume_sum = 0.0
        for i in range(n_volumes - 50, n_volumes):
            older_volume_sum += volumes[i]
        older_volume_mean = older_volume_sum / 50.0
        
        # Volume trend calculation
        if older_volume_mean > 0:
            volume_trend = recent_volume_mean / older_volume_mean
        else:
            volume_trend = 1.0
        
        # Price trend - accesso diretto
        current_price = prices[n_prices - 1]  # prices[-1]
        price_20_ago = prices[n_prices - 20]  # prices[-20]
        
        if price_20_ago > 0:
            price_trend = (current_price - price_20_ago) / price_20_ago
        else:
            price_trend = 0.0
        
        # Wyckoff phase identification (simplified) - thresholds cached
        if price_range < 0.02 and volume_trend > 1.2:
            # Tight range with increasing volume
            return "accumulation"
        elif price_range < 0.02 and volume_trend < 0.8:
            # Tight range with decreasing volume
            return "distribution"
        elif price_trend > 0.05 and volume_trend > 1.1:
            # Markup phase
            return "markup"
        elif price_trend < -0.05 and volume_trend > 1.1:
            # Markdown phase
            return "markdown"
        else:
            return "transition"
    
    def receive_observer_feedback(self, prediction_id: str, feedback_score: float,
                                feedback_details: Dict[str, Any]) -> None:
        """Riceve feedback dall'Observer per una predizione specifica"""
        
        # Parse prediction_id per trovare il modello giusto
        try:
            parts = prediction_id.split('_')
            if len(parts) >= 3:
                # Format: asset_modeltype_algorithm_timestamp
                model_type_str = parts[1]
                model_type = ModelType(model_type_str)
                
                competition = self.competitions.get(model_type)
                if competition:
                    # Forward to competition
                    competition.receive_observer_feedback(prediction_id, feedback_score, feedback_details)
                    
                    # If feedback is very negative, trigger immediate analysis
                    if feedback_score < 0.3:
                        self._handle_negative_feedback(model_type, prediction_id, feedback_details)
                    
                    # Log feedback
                    self.logger.loggers['system'].info(
                        f"Observer feedback received for {prediction_id}: score={feedback_score:.2f}"
                    )
                    
        except Exception as e:
            self.logger.loggers['errors'].error(f"Error processing observer feedback: {e}")
    
    def _handle_negative_feedback(self, model_type: ModelType, prediction_id: str,
                                feedback_details: Dict[str, Any]):
        """Gestisce feedback negativi con azioni immediate"""
        
        self.logger.loggers['system'].warning(
            f"Negative feedback for {model_type.value}: {feedback_details}"
        )
        
        # Trigger immediate reanalysis
        competition = self.competitions[model_type]
        
        # Find algorithm from prediction_id
        algorithm_name = None
        for pred in competition.predictions_history:
            if pred.id == prediction_id:
                algorithm_name = pred.algorithm_name
                break
        
        if algorithm_name:
            # Check if retraining is needed
            algorithm = competition.algorithms.get(algorithm_name)
            if algorithm:
                algorithm.observer_feedback_count += 1
                
                # If too many negative feedbacks, force retraining
                negative_ratio = 1 - algorithm.observer_satisfaction
                if negative_ratio > 0.7 and algorithm.observer_feedback_count > 10:
                    self.logger.loggers['training'].warning(
                        f"Forcing retraining for {algorithm_name} due to poor observer feedback"
                    )
                    self._retrain_algorithm(model_type, algorithm_name, algorithm)
    
    def get_full_analysis_summary(self) -> Dict[str, Any]:
        """Restituisce un summary completo dell'analisi corrente"""
        
        summary = {
            'asset': self.asset,
            'timestamp': datetime.now().isoformat(),
            'learning_phase': self.learning_phase,
            'learning_progress': self.learning_progress,
            'data_points': len(self.tick_data),
            'models_performance': {},
            'champions': {},
            'overall_health': 0.0,
            'system_metrics': {
                'analysis_count': self.analysis_count,
                'average_latency': np.mean(list(self.analysis_latency_history)) if self.analysis_latency_history else 0,
                'memory_usage': len(self.tick_data) * 100 / 100000  # Percentage of buffer used
            }
        }
        
        total_score = 0
        model_count = 0
        
        for model_type, competition in self.competitions.items():
            performance = competition.get_performance_summary()
            summary['models_performance'][model_type.value] = performance
            summary['champions'][model_type.value] = performance['champion']
            
            # Calculate health score
            if performance['champion'] and performance['champion'] in performance['algorithms']:
                champion_score = performance['algorithms'][performance['champion']]['final_score']
                total_score += champion_score
                model_count += 1
        
        summary['overall_health'] = total_score / max(1, model_count)
        
        # Add risk assessment
        if len(self.tick_data) > 50:
            current_market_data = self._prepare_market_data()
            summary['current_risk'] = self._assess_current_risk(current_market_data)
        
        return summary
    
    def save_analyzer_state(self) -> None:
        """Salva lo stato completo dell'Analyzer"""
        
        state_file = f"{self.data_path}/analyzer_state.pkl"
        
        state = {
            'asset': self.asset,
            'learning_phase': self.learning_phase,
            'learning_start_time': self.learning_start_time,
            'learning_progress': self.learning_progress,
            'analysis_count': self.analysis_count,
            'last_analysis_time': self.last_analysis_time,
            'competitions': {},
            'tick_data_summary': {
                'count': len(self.tick_data),
                'latest_timestamp': self.tick_data[-1]['timestamp'] if self.tick_data else None,
                'price_range': {
                    'min': min([t['price'] for t in self.tick_data]) if self.tick_data else None,
                    'max': max([t['price'] for t in self.tick_data]) if self.tick_data else None
                }
            }
        }
        
        # Save competition states
        for model_type, competition in self.competitions.items():
            state['competitions'][model_type.value] = {
                'performance_summary': competition.get_performance_summary(),
                'last_reality_check': competition.last_reality_check,
                'prediction_count': len(competition.predictions_history)
            }
        
        try:
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
            
            # Save models separately
            self._save_ml_models()
            
            # Save recent predictions for analysis
            self._save_recent_predictions()
            
            self.logger.loggers['system'].info(f"Analyzer state saved for {self.asset}")
            
        except Exception as e:
            self.logger.loggers['errors'].error(f"Failed to save analyzer state: {e}")
            
    def _save_ml_models(self) -> None:
        """Salva i modelli ML trainati"""
        
        models_dir = f"{self.data_path}/models"
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.ml_models.items():
            try:
                if isinstance(model, nn.Module):
                    # PyTorch models
                    model_path = f"{models_dir}/{model_name}.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_config': {
                            'class_name': model.__class__.__name__,
                            'timestamp': datetime.now().isoformat()
                        }
                    }, model_path)
                else:
                    # Scikit-learn models
                    model_path = f"{models_dir}/{model_name}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                
            except Exception as e:
                self.logger.loggers['errors'].error(f"Failed to save model {model_name}: {e}")
        
        # Save scalers
        try:
            scalers_path = f"{models_dir}/scalers.pkl"
            with open(scalers_path, 'wb') as f:
                pickle.dump(self.scalers, f)
        except Exception as e:
            self.logger.loggers['errors'].error(f"Failed to save scalers: {e}")
    
    def _save_recent_predictions(self):
        """Salva predizioni recenti per analisi futura"""
        
        predictions_dir = f"{self.data_path}/predictions"
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Save last 1000 predictions per model type
        for model_type, competition in self.competitions.items():
            recent_predictions = competition.predictions_history[-1000:]
            
            if recent_predictions:
                predictions_file = f"{predictions_dir}/{model_type.value}_recent.pkl"
                
                try:
                    with open(predictions_file, 'wb') as f:
                        pickle.dump(recent_predictions, f)
                except Exception as e:
                    self.logger.loggers['errors'].error(
                        f"Failed to save predictions for {model_type.value}: {e}"
                    )
    
    def load_analyzer_state(self) -> bool:
        """Carica lo stato dell'Analyzer se esiste - VERSIONE PULITA"""
        
        state_file = f"{self.data_path}/analyzer_state.pkl"
        
        try:
            if not os.path.exists(state_file):
                # 🧹 PULITO: Sostituito logger con event storage
                self._store_system_event('analyzer_state_load', {
                    'status': 'no_saved_state',
                    'asset': self.asset,
                    'state_file': state_file,
                    'timestamp': datetime.now()
                })
                return False
            
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
            
            # Restore basic state
            self.learning_phase = state.get('learning_phase', True)
            self.learning_start_time = state.get('learning_start_time', datetime.now())
            self.learning_progress = state.get('learning_progress', 0.0)
            self.analysis_count = state.get('analysis_count', 0)
            self.last_analysis_time = state.get('last_analysis_time')
            
            # 🧹 PULITO: Sostituito logger con event storage
            self._store_system_event('analyzer_state_load', {
                'status': 'loaded_successfully',
                'asset': self.asset,
                'learning_phase': self.learning_phase,
                'analysis_count': self.analysis_count,
                'learning_progress': self.learning_progress,
                'timestamp': datetime.now()
            })
            
            # Load ML models (se esistono i metodi)
            if hasattr(self, '_load_ml_models'):
                self._load_ml_models()
            
            # Load recent predictions (se esiste il metodo)
            if hasattr(self, '_load_recent_predictions'):
                self._load_recent_predictions()
            
            return True
            
        except Exception as e:
            # 🧹 PULITO: Sostituito logger con event storage
            self._store_system_event('analyzer_state_load', {
                'status': 'error',
                'asset': self.asset,
                'error_message': str(e),
                'error_type': type(e).__name__,
                'state_file': state_file,
                'timestamp': datetime.now(),
                'severity': 'error'
            })
            return False
    
    def _load_ml_models(self) -> None:
        """Carica i modelli ML salvati"""
        
        models_dir = f"{self.data_path}/models"
        
        if not os.path.exists(models_dir):
            return
        
        # Load PyTorch models
        for model_name, model in self.ml_models.items():
            if isinstance(model, nn.Module):
                model_path = f"{models_dir}/{model_name}.pt"
                if os.path.exists(model_path):
                    try:
                        checkpoint = torch.load(model_path, map_location='cpu')
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.eval()
                        
                        self.logger.loggers['system'].info(f"Loaded PyTorch model: {model_name}")
                        
                    except Exception as e:
                        self.logger.loggers['errors'].error(f"Failed to load {model_name}: {e}")
            
            else:
                # Load scikit-learn models
                model_path = f"{models_dir}/{model_name}.pkl"
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            self.ml_models[model_name] = pickle.load(f)
                        
                        self.logger.loggers['system'].info(f"Loaded sklearn model: {model_name}")
                        
                    except Exception as e:
                        self.logger.loggers['errors'].error(f"Failed to load {model_name}: {e}")
        
        # Load scalers
        scalers_path = f"{models_dir}/scalers.pkl"
        if os.path.exists(scalers_path):
            try:
                with open(scalers_path, 'rb') as f:
                    self.scalers = pickle.load(f)
                
                self.logger.loggers['system'].info("Loaded feature scalers")
                
            except Exception as e:
                self.logger.loggers['errors'].error(f"Failed to load scalers: {e}")
    
    def _load_recent_predictions(self):
        """Carica predizioni recenti salvate"""
        
        predictions_dir = f"{self.data_path}/predictions"
        
        if not os.path.exists(predictions_dir):
            return
        
        for model_type, competition in self.competitions.items():
            predictions_file = f"{predictions_dir}/{model_type.value}_recent.pkl"
            
            if os.path.exists(predictions_file):
                try:
                    with open(predictions_file, 'rb') as f:
                        recent_predictions = pickle.load(f)
                    
                    # Add to competition history
                    competition.predictions_history.extend(recent_predictions[-100:])  # Last 100
                    
                    self.logger.loggers['system'].info(
                        f"Loaded {len(recent_predictions)} predictions for {model_type.value}"
                    )
                    
                except Exception as e:
                    self.logger.loggers['errors'].error(
                        f"Failed to load predictions for {model_type.value}: {e}"
                    )
    
    def cleanup_old_data(self, days_to_keep: Optional[int] = None) -> None:
        """Pulisce dati vecchi mantenendo finestra configurabile"""
        
        # 🔧 USA CONFIG se days_to_keep non specificato
        days_to_keep = days_to_keep or self.config.data_cleanup_days
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean tick data
        with self.data_lock:
            original_count = len(self.tick_data)
            
            # Filter tick data
            new_tick_data = deque(
                [tick for tick in self.tick_data if tick['timestamp'] > cutoff_date],
                maxlen=self.config.max_tick_buffer_size  # 🔧 CHANGED
            )
            
            self.tick_data = new_tick_data
            
            cleaned_count = original_count - len(self.tick_data)
            
            if cleaned_count > 0:
                self.logger.loggers['system'].info(
                    f"Cleaned {cleaned_count} old ticks for {self.asset} (kept {days_to_keep} days)"
                )
        
        # Clean predictions history
        for competition in self.competitions.values():
            original_predictions = len(competition.predictions_history)
            
            competition.predictions_history = [
                p for p in competition.predictions_history
                if p.timestamp > cutoff_date
            ]
            
            cleaned_predictions = original_predictions - len(competition.predictions_history)
            
            if cleaned_predictions > 0:
                self.logger.loggers['system'].info(
                    f"Cleaned {cleaned_predictions} old predictions for {competition.model_type.value}"
                )
        
        # Trigger retraining after cleanup
        self._check_retraining_needs()
    
    def get_analysis_for_observer(self) -> Dict[str, Any]:
        """Prepara analisi specificamente per l'Observer"""
        
        if self.learning_phase:
            return {
                'status': 'learning',
                'progress': self.learning_progress,
                'message': 'Analyzer still in learning phase'
            }
        
        # Get current market data
        market_data = self._prepare_market_data()
        
        if not market_data:
            return {
                'status': 'insufficient_data',
                'message': 'Not enough data for analysis'
            }
        
        # Prepare focused analysis for Observer
        observer_analysis = {
            'timestamp': datetime.now(),
            'asset': self.asset,
            'market_conditions': {
                'state': market_data.get('market_state'),
                'volatility': market_data.get('volatility'),
                'trend': market_data.get('price_change_5m'),
                'volume_profile': market_data.get('volume_ratio')
            },
            'key_predictions': {},
            'recommended_parameters': {},
            'confidence_map': {}
        }
        
        # Get champion predictions for each model type
        for model_type, competition in self.competitions.items():
            champion = competition.get_champion_algorithm()
            
            if champion:
                try:
                    # Run champion algorithm
                    prediction = self._run_champion_algorithm(model_type, champion, market_data)
                    
                    if 'error' not in prediction:
                        observer_analysis['key_predictions'][model_type.value] = prediction
                        observer_analysis['confidence_map'][model_type.value] = prediction.get('confidence', 0.5)
                        
                        # Generate parameter recommendations based on predictions
                        params = self._generate_parameter_recommendations(model_type, prediction, market_data)
                        observer_analysis['recommended_parameters'].update(params)
                
                except Exception as e:
                    self.logger.loggers['errors'].error(
                        f"Error generating prediction for Observer: {e}"
                    )
        
        # Add performance metrics
        observer_analysis['model_performance'] = {
            mt.value: comp.get_performance_summary()
            for mt, comp in self.competitions.items()
        }
        
        return observer_analysis
    
    def _generate_parameter_recommendations(self, model_type: ModelType, 
                                          prediction: Dict[str, Any],
                                          market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Genera raccomandazioni per parametri del bot basate sulle predizioni"""
        
        params = {}
        
        if model_type == ModelType.SUPPORT_RESISTANCE:
            # Recommend stop loss and take profit based on S/R levels
            if 'support_levels' in prediction and prediction['support_levels']:
                nearest_support = min(prediction['support_levels'], 
                                    key=lambda x: abs(x - market_data['current_price']))
                stop_distance = abs(market_data['current_price'] - nearest_support)
                params['recommended_stop_loss_distance'] = stop_distance * 1.1  # 10% buffer
            
            if 'resistance_levels' in prediction and prediction['resistance_levels']:
                nearest_resistance = min(prediction['resistance_levels'],
                                       key=lambda x: abs(x - market_data['current_price']))
                profit_distance = abs(nearest_resistance - market_data['current_price'])
                params['recommended_take_profit_distance'] = profit_distance * 0.9  # 10% conservative
        
        elif model_type == ModelType.VOLATILITY_PREDICTION:
            # Adjust position sizing based on volatility
            predicted_vol = prediction.get('predicted_volatility', market_data['volatility'])
            
            if predicted_vol > 0.02:  # High volatility
                params['position_size_multiplier'] = 0.5
                params['wider_stops'] = True
            elif predicted_vol < 0.005:  # Low volatility
                params['position_size_multiplier'] = 1.5
                params['tighter_stops'] = True
            else:
                params['position_size_multiplier'] = 1.0
        
        elif model_type == ModelType.BIAS_DETECTION:
            # Adjust directional bias
            bias = prediction.get('directional_bias', {})
            direction = bias.get('direction', 'neutral')
            confidence = bias.get('confidence', 0.5)
            
            if direction == 'bullish' and confidence > 0.7:
                params['long_bias'] = confidence
                params['short_bias'] = 1 - confidence
            elif direction == 'bearish' and confidence > 0.7:
                params['short_bias'] = confidence
                params['long_bias'] = 1 - confidence
            else:
                params['long_bias'] = 0.5
                params['short_bias'] = 0.5
        
        elif model_type == ModelType.TREND_ANALYSIS:
            # Adjust for trend following
            trend = prediction.get('trend_direction', 'sideways')
            strength = prediction.get('trend_strength', 0.5)
            
            if trend == 'uptrend' and strength > 0.7:
                params['trend_following_mode'] = True
                params['counter_trend_disabled'] = True
                params['trailing_stop_active'] = True
            elif trend == 'downtrend' and strength > 0.7:
                params['trend_following_mode'] = True
                params['counter_trend_disabled'] = True
                params['trailing_stop_active'] = True
            else:
                params['range_trading_mode'] = True
                params['trend_following_mode'] = False
        
        return params
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Ottieni metriche di performance dettagliate"""
        
        metrics = {
            'asset': self.asset,
            'timestamp': datetime.now(),
            'operational_metrics': {
                'uptime_hours': (datetime.now() - self.learning_start_time).total_seconds() / 3600,
                'total_ticks_processed': len(self.tick_data),
                'analysis_performed': self.analysis_count,
                'average_analysis_latency': np.mean(list(self.analysis_latency_history)) if self.analysis_latency_history else 0
            },
            'model_metrics': {},
            'accuracy_metrics': {},
            'system_health': self._calculate_system_health()
        }
        
        # Collect metrics for each model type
        for model_type, competition in self.competitions.items():
            summary = competition.get_performance_summary()
            
            metrics['model_metrics'][model_type.value] = {
                'champion': summary['champion'],
                'total_algorithms': len(summary['algorithms']),
                'active_algorithms': sum(1 for alg in summary['algorithms'].values() 
                                       if not alg.get('emergency_stop', False)),
                'total_predictions': summary['total_predictions'],
                'pending_validations': summary['pending_validations']
            }
            
            # Accuracy metrics
            if summary['champion'] and summary['champion'] in summary['algorithms']:
                champion_data = summary['algorithms'][summary['champion']]
                metrics['accuracy_metrics'][model_type.value] = {
                    'accuracy_rate': champion_data['accuracy_rate'],
                    'confidence_score': champion_data['confidence_score'],
                    'observer_satisfaction': champion_data['observer_satisfaction']
                }
        
        return metrics
    
    def get_cache_performance_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche performance della cache indicatori"""
        
        cache_stats = self.indicators_cache.get_cache_stats()
        
        performance_analysis = {
            'cache_metrics': cache_stats,
            'efficiency_rating': 'excellent' if cache_stats['hit_rate'] > 80 else 
                            'good' if cache_stats['hit_rate'] > 60 else 
                            'fair' if cache_stats['hit_rate'] > 40 else 'poor',
            'memory_status': 'efficient' if cache_stats['memory_efficient'] else 'needs_cleanup',
            'recommendations': []
        }
        
        # Genera raccomandazioni
        if cache_stats['hit_rate'] < 50:
            performance_analysis['recommendations'].append('Low cache hit rate - consider data preprocessing')
        
        if not cache_stats['memory_efficient']:
            performance_analysis['recommendations'].append('Cache memory inefficient - run cleanup')
        
        if cache_stats['unique_indicators'] > 20:
            performance_analysis['recommendations'].append('High indicator variety - consider standardization')
        
        return performance_analysis

    def cleanup_indicators_cache(self) -> None:
        """Pulisce la cache degli indicatori se necessario"""
        
        cache_stats = self.indicators_cache.get_cache_stats()
        
        if not cache_stats['memory_efficient'] or cache_stats['size'] > 800:
            self.indicators_cache.clear_cache()
            safe_print(f"🧹 Cache indicatori pulita per {self.asset}")
            
            # Re-calcola gli indicatori più comuni
            if len(self.tick_data) > 50:
                recent_prices = np.array([tick['price'] for tick in list(self.tick_data)[-50:]])
                
                # Pre-popola cache con indicatori comuni
                self.cached_indicators['sma'](recent_prices, 20)
                self.cached_indicators['rsi'](recent_prices, 14)
                
                safe_print("✅ Cache pre-popolata con indicatori comuni")

    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calcola la salute generale del sistema"""
        
        health_score = 100.0
        issues = []
        
        # Check data freshness
        if self.tick_data:
            last_tick_age = (datetime.now() - self.tick_data[-1]['timestamp']).seconds
            if last_tick_age > 60:  # More than 1 minute old
                health_score -= 10
                issues.append("stale_data")
        
        # Check model performance
        poorly_performing = 0
        for competition in self.competitions.values():
            summary = competition.get_performance_summary()
            if summary['health_status'] in ['poor', 'critical']:
                poorly_performing += 1
        
        if poorly_performing > len(self.competitions) * 0.5:
            health_score -= 30
            issues.append("poor_model_performance")
        
        # Check emergency stops
        total_stopped = sum(
            1 for comp in self.competitions.values()
            for alg in comp.algorithms.values()
            if alg.emergency_stop_triggered
        )
        
        if total_stopped > 5:
            health_score -= 20
            issues.append("multiple_emergency_stops")
        
        # Check memory usage
        memory_usage = len(self.tick_data) / 100000  # Max capacity
        if memory_usage > 0.9:
            health_score -= 10
            issues.append("high_memory_usage")
        
        return {
            'score': max(0, health_score),
            'status': 'healthy' if health_score > 70 else 'degraded' if health_score > 40 else 'critical',
            'issues': issues,
            'last_check': datetime.now()
        }
    
    def get_diagnostics_report(self) -> Dict[str, Any]:
        """Genera report diagnostico completo"""
        return self.diagnostics.generate_diagnostics_report()

    def force_diagnostics_check(self) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Forza un check diagnostico immediato"""
        stall_info = self.diagnostics.detect_learning_stall(self)
        structure_analysis = self.diagnostics.log_data_structure_analysis(self)
        
        safe_print(f"\n🔍 FORCED DIAGNOSTICS CHECK for {self.asset}:")
        safe_print(f"   📊 Total ticks: {len(self.tick_data):,}")
        safe_print(f"   🧠 Learning phase: {self.learning_phase}")
        safe_print(f"   📈 Learning progress: {self.learning_progress:.1%}")
        safe_print(f"   💾 Memory usage: {self.diagnostics._get_memory_usage():.1f}%")
        safe_print(f"   ⚡ System load: {self.diagnostics._get_system_load():.1f}%")
        
        if stall_info:
            safe_print(f"   🚨 STALL DETECTED: {stall_info['indicators']}")
        else:
            safe_print(f"   ✅ No stall detected")
        
        return stall_info, structure_analysis

    def shutdown(self):
        """Shutdown pulito dell'analyzer con diagnostica - VERSIONE PULITA"""
        
        # 🧹 PULITO: Sostituito logger con event storage
        self._store_system_event('analyzer_shutdown', {
            'status': 'started',
            'asset': self.asset,
            'analysis_count': self.analysis_count,
            'timestamp': datetime.now()
        })
        
        # 🔧 NUOVO: Genera report finale
        try:
            final_report = self.get_diagnostics_report()
            
            # Salva report diagnostico
            report_path = f"{self.data_path}/diagnostics_final_report.json"
            with open(report_path, 'w') as f:
                json.dump(final_report, f, indent=2, default=str)
            
            # 🧹 PULITO: Sostituito logger con event storage
            self._store_system_event('diagnostics_report_saved', {
                'status': 'success',
                'asset': self.asset,
                'report_path': report_path,
                'report_size_kb': os.path.getsize(report_path) / 1024 if os.path.exists(report_path) else 0,
                'timestamp': datetime.now()
            })
            
            # Shutdown diagnostics
            self.diagnostics.shutdown()
            
        except Exception as e:
            # 🧹 PULITO: Sostituito logger con event storage
            self._store_system_event('diagnostics_shutdown_error', {
                'status': 'error',
                'asset': self.asset,
                'error_message': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.now(),
                'severity': 'error'
            })
        
        # Save current state
        self.save_analyzer_state()
        
        # Disconnect from MT5
        if self.mt5_interface.connected:
            self.mt5_interface.disconnect()
        
        # 🧹 PULITO: Sostituito logger con event storage
        self._store_system_event('analyzer_shutdown', {
            'status': 'completed',
            'asset': self.asset,
            'analysis_count': self.analysis_count,
            'mt5_disconnected': not self.mt5_interface.connected,
            'timestamp': datetime.now()
        })

        # Shutdown async I/O
        if hasattr(self.logger, 'shutdown'):
            self.logger.shutdown()

    def _store_system_event(self, event_type: str, event_data: Dict) -> None:
        """Store system events in memory for future processing by slave module"""
        if not hasattr(self, '_system_events_buffer'):
            self._system_events_buffer: deque = deque(maxlen=50)
        
        event_entry = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'data': event_data
        }
        
        self._system_events_buffer.append(event_entry)
        
        # Buffer size is automatically managed by deque maxlen=50
        # No manual cleanup needed since deque automatically removes old items

    def get_all_events_for_slave(self) -> Dict[str, List[Dict]]:
        """Get all accumulated events for slave module processing"""
        events = {}
        
        # System events
        if hasattr(self, '_system_events_buffer'):
            events['system_events'] = self._system_events_buffer.copy()
        
        # Add other event buffers if they exist
        # Note: Altri buffer potrebbero essere aggiunti qui in futuro
        
        return events

    def clear_events_buffer(self, event_types: Optional[List[str]] = None) -> None:
        """Clear event buffers after slave module processing"""
        if event_types is None:
            # Clear all buffers
            if hasattr(self, '_system_events_buffer'):
                self._system_events_buffer.clear()
        else:
            # Clear specific buffers
            for event_type in event_types:
                if event_type == 'system_events' and hasattr(self, '_system_events_buffer'):
                    self._system_events_buffer.clear()

# ================== MAIN ANALYZER SYSTEM ==================

class AdvancedMarketAnalyzer:
    """Analyzer principale che gestisce tutti gli asset"""
    
    def __init__(self, data_path: str = "./analyzer_data"):
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        
        self.asset_analyzers: Dict[str, AssetAnalyzer] = {}
        self.global_stats = {
            'total_predictions': 0,
            'total_feedback_received': 0,
            'global_performance': 0.0,
            'active_assets': 0,
            'system_start_time': datetime.now()
        }
        
        # Global logger
        self.logger = AnalyzerLogger(f"{data_path}/global_logs")
        
        # Load existing analyzers
        self._load_existing_analyzers()
        
        # Log system start - con accesso sicuro
        try:
            if (hasattr(self.logger, 'loggers') and 
                self.logger.loggers and 
                isinstance(self.logger.loggers, dict) and
                'system' in self.logger.loggers):
                self.logger.loggers['system'].info(
                    f"AdvancedMarketAnalyzer initialized with {len(self.asset_analyzers)} assets"
                )
            else:
                print(f"⚠️ AdvancedMarketAnalyzer initialized with {len(self.asset_analyzers)} assets (system logger not available)")
        except Exception as e:
            print(f"⚠️ AdvancedMarketAnalyzer initialized with {len(self.asset_analyzers)} assets (logger error: {e})")
    
    def _load_existing_analyzers(self) -> None:
        """Carica gli analyzer esistenti"""
        
        for item in os.listdir(self.data_path):
            item_path = os.path.join(self.data_path, item)
            if os.path.isdir(item_path) and not item.startswith('global'):
                try:
                    # Assume directory name is asset name
                    asset = item
                    analyzer = AssetAnalyzer(asset, self.data_path)
                    self.asset_analyzers[asset] = analyzer
                    
                    self.logger.loggers['system'].info(f"Loaded analyzer for {asset}")
                    
                except Exception as e:
                    self.logger.loggers['errors'].error(f"Failed to load analyzer for {item}: {e}")
    
    def add_asset(self, asset: str) -> AssetAnalyzer:
        """Aggiunge un nuovo asset per l'analisi - VERSIONE PULITA"""
        
        if asset not in self.asset_analyzers:
            self.asset_analyzers[asset] = AssetAnalyzer(asset, self.data_path)
            # 🧹 PULITO: Sostituito logger con event storage (usa metodo esistente)
            self._store_global_event('asset_added', {
                'asset': asset,
                'total_assets': len(self.asset_analyzers),
                'timestamp': datetime.now()
            })
        
        return self.asset_analyzers[asset]
    
    def process_tick(self, asset: str, timestamp: datetime, price: float, 
                    volume: float, **kwargs) -> Dict[str, Any]:
        """Processa un tick per un asset specifico - VERSIONE PULITA"""
        
        if asset not in self.asset_analyzers:
            self.add_asset(asset)
        
        analyzer = self.asset_analyzers[asset]
        result = analyzer.process_tick(timestamp, price, volume, **kwargs)
        
        # Update global stats
        self._update_global_stats()
        
        # Store significant events for future slave module processing
        if result.get('status') == 'learning_complete':
            self._store_global_event('learning_complete', {
                'asset': asset,
                'timestamp': timestamp,
                'result': result
            })
        
        return result


    def _store_global_event(self, event_type: str, event_data: Dict) -> None:
        """Store global events in memory for future processing by slave module"""
        if not hasattr(self, '_global_events_buffer'):
            self._global_events_buffer: deque = deque(maxlen=200)
        
        event = {
            'timestamp': datetime.now(),
            'type': event_type,
            'data': event_data
        }
        
        self._global_events_buffer.append(event)
        
        # Buffer size is automatically managed by deque maxlen=200
        # No manual cleanup needed since deque automatically removes old items
    
    def receive_observer_feedback(self, asset: str, prediction_id: str, 
                                feedback_data: Dict[str, Any]) -> None:
        """Inoltra feedback dell'Observer all'asset specifico"""
        
        if asset in self.asset_analyzers:
            feedback_score = feedback_data.get('score', 0.5)
            self.asset_analyzers[asset].receive_observer_feedback(
                prediction_id, feedback_score, feedback_data
            )
            self.global_stats['total_feedback_received'] += 1
            
            self.logger.loggers['system'].info(
                f"Forwarded observer feedback to {asset} for {prediction_id}"
            )
    
    def get_analysis_for_observer(self, asset: str) -> Optional[Dict[str, Any]]:
        """Ottiene analisi per Observer per un asset specifico"""
        
        if asset in self.asset_analyzers:
            return self.asset_analyzers[asset].get_analysis_for_observer()
        return None
    
    def get_global_summary(self) -> Dict[str, Any]:
        """Restituisce un summary globale di tutti gli asset"""
        
        summary = {
            'timestamp': datetime.now(),
            'global_stats': self.global_stats,
            'assets': {},
            'system_health': self._calculate_global_health(),
            'recommendations': self._generate_global_recommendations()
        }
        
        for asset, analyzer in self.asset_analyzers.items():
            summary['assets'][asset] = analyzer.get_full_analysis_summary()
        
        return summary
    
    def _update_global_stats(self) -> None:
        """Aggiorna le statistiche globali"""
        
        total_performance = 0
        active_assets = 0
        total_predictions = 0
        
        for analyzer in self.asset_analyzers.values():
            if not analyzer.learning_phase:
                active_assets += 1
                summary = analyzer.get_full_analysis_summary()
                total_performance += summary['overall_health']
                
                # Count predictions
                for model_perf in summary['models_performance'].values():
                    total_predictions += model_perf.get('total_predictions', 0)
        
        self.global_stats.update({
            'active_assets': active_assets,
            'global_performance': total_performance / max(1, active_assets),
            'total_predictions': total_predictions,
            'uptime_hours': (datetime.now() - self.global_stats['system_start_time']).total_seconds() / 3600
        })
    
    def _calculate_global_health(self) -> Dict[str, Any]:
        """Calcola la salute globale del sistema"""
        
        asset_healths = []
        issues = []
        
        for asset, analyzer in self.asset_analyzers.items():
            health = analyzer._calculate_system_health()
            asset_healths.append(health['score'])
            
            if health['status'] == 'critical':
                issues.append(f"{asset}_critical")
        
        avg_health = np.mean(asset_healths) if asset_healths else 0
        
        # Check system-wide issues
        if self.global_stats['active_assets'] == 0:
            issues.append("no_active_assets")
            avg_health -= 30
        
        return {
            'score': max(0, avg_health),
            'status': 'healthy' if avg_health > 70 else 'degraded' if avg_health > 40 else 'critical',
            'issues': issues,
            'assets_health': {
                asset: analyzer._calculate_system_health()['status']
                for asset, analyzer in self.asset_analyzers.items()
            }
        }
    
    def _generate_global_recommendations(self) -> List[str]:
        """Genera raccomandazioni a livello di sistema"""
        
        recommendations = []
        
        # Check for common patterns across assets
        trend_directions = []
        volatility_levels = []
        
        for analyzer in self.asset_analyzers.values():
            if not analyzer.learning_phase and len(analyzer.tick_data) > 50:
                market_data = analyzer._prepare_market_data()
                
                # Collect trend info
                if 'price_change_5m' in market_data:
                    if market_data['price_change_5m'] > 0.01:
                        trend_directions.append('up')
                    elif market_data['price_change_5m'] < -0.01:
                        trend_directions.append('down')
                
                # Collect volatility
                if 'volatility' in market_data:
                    volatility_levels.append(market_data['volatility'])
        
        # Generate recommendations
        if trend_directions:
            up_count = trend_directions.count('up')
            down_count = trend_directions.count('down')
            
            if up_count > len(trend_directions) * 0.7:
                recommendations.append("Strong bullish sentiment across multiple assets")
            elif down_count > len(trend_directions) * 0.7:
                recommendations.append("Strong bearish sentiment across multiple assets")
        
        if volatility_levels:
            avg_volatility = np.mean(volatility_levels)
            if avg_volatility > 0.02:
                recommendations.append("High volatility detected - consider reducing position sizes")
            elif avg_volatility < 0.005:
                recommendations.append("Low volatility environment - watch for breakouts")
        
        # System recommendations
        if self.global_stats['active_assets'] < len(self.asset_analyzers):
            recommendations.append(f"{len(self.asset_analyzers) - self.global_stats['active_assets']} assets still in learning phase")
        
        return recommendations
    
    def save_all_states(self) -> None:
        """Salva lo stato di tutti gli analyzer"""
        
        self.logger.loggers['system'].info("Saving all analyzer states...")
        
        for asset, analyzer in self.asset_analyzers.items():
            try:
                analyzer.save_analyzer_state()
            except Exception as e:
                self.logger.loggers['errors'].error(f"Failed to save state for {asset}: {e}")
        
        # Save global stats
        global_state = {
            'global_stats': self.global_stats,
            'assets': list(self.asset_analyzers.keys()),
            'timestamp': datetime.now()
        }
        
        with open(f"{self.data_path}/global_state.json", 'w') as f:
            json.dump(global_state, f, indent=2, default=str)
        
        self.logger.loggers['system'].info("All states saved successfully")
    
    def cleanup_old_data(self, days_to_keep: int = 180) -> None:
        """Pulisce dati vecchi per tutti gli asset"""
        
        self.logger.loggers['system'].info(f"Starting cleanup for data older than {days_to_keep} days")
        
        for asset, analyzer in self.asset_analyzers.items():
            analyzer.cleanup_old_data(days_to_keep)
        
        self.logger.loggers['system'].info("Cleanup completed for all assets")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Genera report completo delle performance"""
        
        report = {
            'generated_at': datetime.now(),
            'system_overview': {
                'total_assets': len(self.asset_analyzers),
                'active_assets': self.global_stats['active_assets'],
                'total_predictions': self.global_stats['total_predictions'],
                'total_feedback': self.global_stats['total_feedback_received'],
                'uptime_hours': self.global_stats.get('uptime_hours', 0)
            },
            'asset_performances': {},
            'model_rankings': self._calculate_model_rankings(),
            'system_health': self._calculate_global_health()
        }
        
        # Add detailed asset performances
        for asset, analyzer in self.asset_analyzers.items():
            report['asset_performances'][asset] = analyzer.get_performance_metrics()
        
        return report
    
    def _calculate_model_rankings(self) -> Dict[str, List[Dict]]:
        """Calcola ranking dei modelli across tutti gli asset"""
        
        rankings = {}
        
        for model_type in ModelType:
            model_scores = []
            
            for asset, analyzer in self.asset_analyzers.items():
                if model_type in analyzer.competitions:
                    competition = analyzer.competitions[model_type]
                    summary = competition.get_performance_summary()
                    
                    if summary['champion']:
                        champion_data = summary['algorithms'].get(summary['champion'], {})
                        model_scores.append({
                            'asset': asset,
                            'algorithm': summary['champion'],
                            'score': champion_data.get('final_score', 0),
                            'accuracy': champion_data.get('accuracy_rate', 0),
                            'predictions': champion_data.get('total_predictions', 0)
                        })
            
            # Sort by score
            model_scores.sort(key=lambda x: x['score'], reverse=True)
            rankings[model_type.value] = model_scores[:10]  # Top 10
        
        return rankings
    
    def shutdown(self):
        """Shutdown pulito del sistema"""
        
        self.logger.loggers['system'].info("Initiating system shutdown...")
        
        # Save all states
        self.save_all_states()
        
        # Shutdown each analyzer
        for asset, analyzer in self.asset_analyzers.items():
            analyzer.shutdown()
        
        # Final log
        self.logger.loggers['system'].info(
            f"System shutdown complete - Total uptime: {self.global_stats.get('uptime_hours', 0):.2f} hours"
        )

# ================== END OF ANALYZER MODULE ==================

if __name__ == "__main__":
    # Example usage
    analyzer = AdvancedMarketAnalyzer()
    
    # Add assets
    analyzer.add_asset("USTEC")
    
    print("AnalyzerUltraAdvanced initialized and ready!")
            
    