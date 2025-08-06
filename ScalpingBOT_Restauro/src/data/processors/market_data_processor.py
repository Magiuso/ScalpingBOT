#!/usr/bin/env python3
"""
MarketDataProcessor - Market Data Processing and Feature Engineering
===================================================================

Sistema di elaborazione dati di mercato con feature engineering avanzato.
ESTRATTO IDENTICO da AssetAnalyzer._prepare_market_data() e metodi correlati.

Features:
- Performance-optimized market data preparation
- Zero-copy data access patterns
- Advanced feature engineering (SMA, RSI, technical indicators)
- Multiple dataset preparation methods (_prepare_*_dataset)
- Thread-safe data processing
- Memory-efficient calculations

Author: ScalpingBOT Team
Version: 1.0.0
"""

import numpy as np
from datetime import datetime
from collections import deque
import threading

# Import configuration for technical indicators
from ScalpingBOT_Restauro.src.config.base.base_config import AnalyzerConfig
from typing import Dict, Any, Optional, List, Tuple, Union

class MarketDataProcessor:
    """Processor per elaborazione dati di mercato con feature engineering"""
    
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """Inizializza il processor con configurazione ottimizzata - BIBBIA COMPLIANT"""
        if config is None:
            raise ValueError("FAIL FAST: MarketDataProcessor requires explicit configuration - no fallback allowed")
        self.config = config
        self.processing_stats = {
            'total_preparations': 0,
            'last_preparation_time': None,
            'average_preparation_time_ms': 0.0,
            'processing_errors': 0
        }
        self.data_lock = threading.RLock()
    
    def prepare_market_data(self, tick_data: deque, min_ticks: int = 20, 
                           window_size: int = 5000) -> Dict[str, Any]:
        """
        Prepara i dati di mercato per l'analisi con feature avanzate
        ESTRATTO IDENTICO da AssetAnalyzer._prepare_market_data()
        
        Args:
            tick_data: Deque con i dati tick
            min_ticks: Numero minimo di tick richiesti
            window_size: Dimensione finestra di analisi
            
        Returns:
            Dict con dati di mercato preparati e feature calcolate
        """
        # 🔧 SOLUZIONE: Ridotta soglia minima da 50 a 20 e aumentata finestra a 5000
        if len(tick_data) < min_ticks:
            return {}
        
        # Performance tracking (minimal, in-memory only)
        processing_start = datetime.now()
        
        try:
            # OTTIMIZZAZIONE CRITICA: Accesso diretto senza copia lista completa
            with self.data_lock:
                total_ticks = len(tick_data)
                # 🔧 SOLUZIONE: Finestra aumentata da 100 a 5000 ticks per calcoli ML accurati
                start_idx = max(0, total_ticks - window_size)
                # Reference diretta + slice range invece di list() copy
                recent_ticks_range = range(start_idx, total_ticks)
                n_ticks = len(recent_ticks_range)
                
                # Performance milestone logging (ogni 100k ticks)
                if total_ticks % 100000 == 0 and total_ticks > 0:
                    pass  # Market data processing milestone reached
            
            # PRE-ALLOCA arrays con dtype esplicito (NO COPIE)
            prices = np.empty(n_ticks, dtype=np.float32)
            volumes = np.empty(n_ticks, dtype=np.float32)
            timestamps = []  # Keep list per timestamp (necessario per output)
            
            # Loop ottimizzato per estrazione dati (accesso diretto tramite indici)
            with self.data_lock:  # Secondo lock minimale per accesso sicuro
                for i, tick_idx in enumerate(recent_ticks_range):
                    tick = tick_data[tick_idx]
                    # BIBBIA COMPLIANT: Use MT5 format with 'last' field - FAIL FAST, no fallbacks
                    if 'last' in tick:
                        prices[i] = tick['last']
                    else:
                        raise ValueError(f"Critical field 'last' missing from tick data at index {i} - MT5 format required")
                    if 'volume' not in tick:
                        raise KeyError(f"Critical field 'volume' missing from tick data at index {i}")
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
            
            # 🔧 BIBBIA COMPLIANT: Price changes con finestra dinamica - single path solution
            price_change_1m = self._calculate_price_change(prices, current_price, 20, n_ticks)
            price_change_5m = self._calculate_price_change(prices, current_price, 100, n_ticks)
            
            # Volume analysis (avg_volume già calcolato)
            volume_ratio = volumes[-1] / max(avg_volume, 1e-10)
            
            # Calcola indicatori tecnici completi utilizzando configurazione centralizzata
            indicators = self._calculate_technical_indicators(prices)
            
            # FAIL-FAST: Verifica che gli indicatori siano stati calcolati correttamente
            if 'sma_20' not in indicators or len(indicators['sma_20']) == 0:
                raise RuntimeError("SMA-20 calculation failed - insufficient data or calculation error")
            if 'sma_50' not in indicators or len(indicators['sma_50']) == 0:
                raise RuntimeError("SMA-50 calculation failed - insufficient data or calculation error")
            if 'rsi' not in indicators or len(indicators['rsi']) == 0:
                raise RuntimeError("RSI calculation failed - insufficient data or calculation error")
                
            sma_20 = indicators['sma_20'][-1]
            sma_50 = indicators['sma_50'][-1]
            rsi = indicators['rsi'][-1]
            
            # FAIL-FAST: Verifica MACD
            if 'macd_line' not in indicators or len(indicators['macd_line']) == 0:
                raise RuntimeError("MACD calculation failed - insufficient data or calculation error")
            if 'macd_signal' not in indicators or len(indicators['macd_signal']) == 0:
                raise RuntimeError("MACD Signal calculation failed - insufficient data or calculation error")
            if 'macd_histogram' not in indicators or len(indicators['macd_histogram']) == 0:
                raise RuntimeError("MACD Histogram calculation failed - insufficient data or calculation error")
                
            macd_line = indicators['macd_line'][-1]
            macd_signal = indicators['macd_signal'][-1]
            macd_histogram = indicators['macd_histogram'][-1]
            
            # FAIL-FAST: Verifica Bollinger Bands
            if 'bb_upper' not in indicators or len(indicators['bb_upper']) == 0:
                raise RuntimeError("Bollinger Bands Upper calculation failed - insufficient data or calculation error")
            if 'bb_lower' not in indicators or len(indicators['bb_lower']) == 0:
                raise RuntimeError("Bollinger Bands Lower calculation failed - insufficient data or calculation error")
                
            bb_upper = indicators['bb_upper'][-1]
            bb_lower = indicators['bb_lower'][-1]
            
            # Calcola returns per ML features
            returns = np.diff(prices, prepend=prices[0]) / np.maximum(prices[:-1], 1e-10)
            returns = np.append(returns, 0)  # Aggiungi ultimo elemento
            log_returns = np.log(np.maximum(prices[1:] / np.maximum(prices[:-1], 1e-10), 1e-10))
            log_returns = np.append(log_returns, 0)  # Aggiungi ultimo elemento
            
            # Update processing stats
            processing_time = (datetime.now() - processing_start).total_seconds() * 1000
            self._update_processing_stats(processing_time)
            
            # Costruisci risultato completo con indicatori tecnici consolidati
            result = {
                'current_price': current_price,
                'price_mean': price_mean,
                'price_std': price_std,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'price_change_1m': price_change_1m,
                'price_change_5m': price_change_5m,
                'price_history': price_history,
                'volume_history': volume_history,
                'timestamps': timestamps,
                'returns': returns,
                'log_returns': log_returns,
                'total_ticks': total_ticks,
                'processed_ticks': n_ticks,
                'processing_time_ms': processing_time,
                
                # === INDICATORI TECNICI CONSOLIDATI ===
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi,
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                
                # === INDICATORI COMPLETI (Arrays) ===
                'indicators': indicators  # Dict completo con tutti gli indicatori calcolati
            }
            
            return result
            
        except Exception as e:
            self.processing_stats['processing_errors'] += 1
            return {'error': str(e), 'processing_failed': True}
    
    def _calculate_price_change(self, prices: np.ndarray, current_price: float, 
                               lookback: int, n_ticks: int) -> float:
        """
        Calcola price change con logica adattiva - BIBBIA COMPLIANT
        ESTRATTO da AssetAnalyzer._prepare_market_data()
        """
        if n_ticks > lookback and prices[-lookback] != 0:
            return (current_price - prices[-lookback]) / prices[-lookback]
        elif n_ticks > 1 and prices[0] != 0:
            return (current_price - prices[0]) / prices[0]
        else:
            return 0.0
    
    def _calculate_technical_indicators(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calcola indicatori tecnici completi utilizzando configurazione centralizzata
        Implementazione consolidata per eliminare duplicazioni nel sistema
        """
        if len(prices) == 0:
            return {}
        
        indicators = {}
        
        # === SMA per tutti i periodi configurati ===
        for period in self.config.sma_periods:
            if len(prices) >= period:
                # Ottimizzato con convoluzione
                sma = np.convolve(prices, np.ones(period)/period, mode='same')
                # Gestisci bordi correttamente
                sma[:period-1] = np.nan
                indicators[f'sma_{period}'] = sma
            else:
                indicators[f'sma_{period}'] = np.full_like(prices, np.nan)
        
        # === RSI con implementazione standard ===
        rsi_period = self.config.rsi_period
        if len(prices) >= rsi_period + 1:
            rsi = self._calculate_rsi(prices, rsi_period)
            indicators['rsi'] = rsi
        else:
            raise ValueError(f"Insufficient price data for RSI calculation: need {rsi_period + 1} prices, got {len(prices)}")
        
        # === EMA per MACD ===
        ema_fast = self._calculate_ema(prices, self.config.macd_fast)
        ema_slow = self._calculate_ema(prices, self.config.macd_slow)
        indicators['ema_fast'] = ema_fast
        indicators['ema_slow'] = ema_slow
        
        # === MACD Line, Signal, Histogram ===
        macd_line = ema_fast - ema_slow
        macd_signal = self._calculate_ema(macd_line, self.config.macd_signal)
        macd_histogram = macd_line - macd_signal
        
        indicators['macd_line'] = macd_line
        indicators['macd_signal'] = macd_signal
        indicators['macd_histogram'] = macd_histogram
        
        # === Bollinger Bands ===
        if len(prices) >= self.config.bollinger_period:
            # FAIL FAST - Require SMA to be pre-calculated
            sma_key = f'sma_{self.config.bollinger_period}'
            if sma_key not in indicators:
                raise KeyError(f"Required SMA indicator '{sma_key}' not found - must be pre-calculated")
            bb_middle = indicators[sma_key]
            bb_std = self._calculate_rolling_std(prices, self.config.bollinger_period)
            
            bb_upper = bb_middle + (self.config.bollinger_std * bb_std)
            bb_lower = bb_middle - (self.config.bollinger_std * bb_std)
            
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
        else:
            indicators['bb_upper'] = np.full_like(prices, np.nan)
            indicators['bb_middle'] = np.full_like(prices, np.nan)
            indicators['bb_lower'] = np.full_like(prices, np.nan)
        
        return indicators
    
    def _calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calcola Simple Moving Average con gestione bordi corretta"""
        if len(prices) < period:
            return np.full_like(prices, np.nan)
        
        sma = np.convolve(prices, np.ones(period)/period, mode='same')
        sma[:period-1] = np.nan  # Imposta NaN per valori insufficienti
        return sma
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calcola Exponential Moving Average"""
        if len(prices) == 0:
            return np.array([])
        
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices, dtype=np.float64)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calcola RSI con gestione robusta degli errori - CONSOLIDATO DA COMPETITION.PY"""
        if len(prices) < period + 1:
            raise ValueError(f"Insufficient price data for RSI calculation: need {period + 1} prices, got {len(prices)}")
        
        # Protezione per valori non validi
        if np.isnan(prices).any():
            raise ValueError("RSI calculation failed: price data contains NaN values")
        if np.isinf(prices).any():
            raise ValueError("RSI calculation failed: price data contains infinite values")
        
        try:
            deltas = np.diff(prices)
            
            # Controllo deltas validi
            if np.isnan(deltas).any():
                raise ValueError("RSI calculation failed: price deltas contain NaN values")
            if np.isinf(deltas).any():
                raise ValueError("RSI calculation failed: price deltas contain infinite values")
            
            seed = deltas[:period+1]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            
            # Divisione per zero più robusta
            if down == 0:
                rs = 100
            elif up == 0:
                rs = 0
            else:
                rs = up / down
                
            rsi = np.zeros_like(prices)  # Initialize with zeros
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
                
                # Protezione divisione per zero
                if down == 0:
                    rs = 100
                elif up == 0:
                    rs = 0
                else:
                    rs = up / down
                    
                # Validation del risultato RSI
                rsi_value = 100 - 100 / (1 + rs)
                if np.isnan(rsi_value) or np.isinf(rsi_value):
                    rsi_value = 50.0
                    
                rsi[i] = np.clip(rsi_value, 0.0, 100.0)
            
            return rsi
            
        except Exception as e:
            raise RuntimeError(f"RSI calculation failed for price data: {e}. Data may be invalid or insufficient.")
    
    def _calculate_rolling_std(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calcola standard deviation rolling per Bollinger Bands"""
        if len(prices) < period:
            return np.full_like(prices, np.nan)
        
        rolling_std = np.full_like(prices, np.nan, dtype=np.float64)
        
        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1:i + 1]
            rolling_std[i] = np.std(window, ddof=0)
        
        return rolling_std
    
    def prepare_lstm_features(self, prices: np.ndarray, volumes: np.ndarray, 
                             market_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Prepara features per modelli LSTM utilizzando processore consolidato
        per evitare duplicazioni con le implementazioni esistenti nel monolite
        
        Args:
            prices: Array dei prezzi
            volumes: Array dei volumi  
            market_data: Dati di mercato aggiuntivi
            
        Returns:
            Array delle features per LSTM
        """
        if len(prices) < 50:  # Minimo per calcoli tecnici
            return np.array([])
        
        # Utilizza il processore consolidato per evitare duplicazioni
        indicators = self._calculate_technical_indicators(prices)
        
        # Price-based features
        returns = np.diff(prices, prepend=prices[0]) / np.maximum(prices[:-1], 1e-10)
        log_returns = np.log(np.maximum(prices[1:] / np.maximum(prices[:-1], 1e-10), 1e-10))
        log_returns = np.append(log_returns, 0)
        
        # Volume features semplificati per evitare duplicazioni
        if len(volumes) == 0:
            raise ValueError("Volume data is empty - cannot calculate volume features")
        volume_mean = np.mean(volumes)
        volume_ratio = volumes / max(volume_mean, 1e-10)
        
        # Volatility features
        if len(returns) < 20:
            raise ValueError(f"Insufficient returns data for volatility calculation: {len(returns)} < 20")
        volatility = np.std(returns[-20:])
        
        # Combine all features usando indicatori consolidati
        features = np.column_stack([
            prices, volumes, returns, log_returns,
            np.nan_to_num(indicators['sma_20']), np.nan_to_num(indicators['sma_50']), 
            np.nan_to_num(indicators['rsi']), np.nan_to_num(volume_ratio),
            np.full_like(prices, volatility)
        ])
        
        return features
    
    def _update_processing_stats(self, processing_time_ms: float) -> None:
        """Aggiorna statistiche di processing"""
        self.processing_stats['total_preparations'] += 1
        self.processing_stats['last_preparation_time'] = datetime.now()
        
        # Calcola media mobile del tempo di processing
        current_avg = self.processing_stats['average_preparation_time_ms']
        total_preps = self.processing_stats['total_preparations']
        
        if total_preps == 1:
            self.processing_stats['average_preparation_time_ms'] = processing_time_ms
        else:
            # Media mobile con peso maggiore sui valori recenti
            alpha = 0.1  # Peso per il nuovo valore
            self.processing_stats['average_preparation_time_ms'] = (
                (1 - alpha) * current_avg + alpha * processing_time_ms
            )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche di processing"""
        return self.processing_stats.copy()

# Factory function per compatibilità
def create_market_data_processor(config: Optional[AnalyzerConfig] = None) -> MarketDataProcessor:
    """Factory function per creare un MarketDataProcessor configurato"""
    return MarketDataProcessor(config)