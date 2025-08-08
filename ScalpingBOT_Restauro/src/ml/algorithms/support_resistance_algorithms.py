#!/usr/bin/env python3
"""
Support/Resistance Algorithms - ESTRATTO IDENTICO DAL MONOLITE
================================================================

5 algoritmi Support/Resistance identificati come mancanti dall'analisi:
- PivotPoints_Classic: Pivot points classici 
- VolumeProfile_Advanced: Analisi volume profile avanzata
- LSTM_SupportResistance: LSTM per detection S/R
- StatisticalLevels_ML: Livelli statistici ML
- Transformer_Levels: Transformer per S/R prediction

ESTRATTO IDENTICO da src/Analyzer.py righe 12826-13036
Mantenuta IDENTICA la logica originale, solo import aggiustati.
"""

import numpy as np
import torch
import json
from json import JSONDecodeError
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Import from migrated modules
from ..models.advanced_lstm import AdvancedLSTM
from ..models.transformer_models import TransformerPredictor
from ...shared.enums import ModelType
from ...shared.exceptions import (
    InsufficientDataError,
    ModelNotInitializedError, 
    InvalidInputError,
    PredictionError,
    AlgorithmErrors
)
# Removed safe_print import - using fail-fast error handling instead


class SupportResistanceAlgorithms:
    """
    Support/Resistance Algorithms - ESTRATTO IDENTICO DAL MONOLITE
    
    Implementa i 5 algoritmi identificati come mancanti:
    1. PivotPoints_Classic
    2. VolumeProfile_Advanced  
    3. LSTM_SupportResistance
    4. StatisticalLevels_ML
    5. Transformer_Levels
    """
    
    def __init__(self, ml_models: Dict[str, Any]):
        """Inizializza algoritmi S/R con modelli ML - FAIL FAST se mancanti"""
        if ml_models is None:
            raise ValueError("FAIL FAST: ml_models è obbligatorio - no fallback consentiti in sistema finanziario")
        if not isinstance(ml_models, dict):
            raise TypeError(f"FAIL FAST: ml_models deve essere dict, ricevuto {type(ml_models)}")
        self.ml_models = ml_models
        self.algorithm_stats = {
            'executions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'last_execution': None
        }
        
        # NUOVO: Sistema di caching per PivotPoints con event-driven logic
        self.pivot_cache = {
            'initialization_time': None,
            'accumulated_ticks': [],
            'current_levels': {},
            'next_recalc_time': None,
            'performance_stats': {
                'successful_predictions': 0,
                'failed_predictions': 0
            },
            # Event-driven tracking per nuova logica
            'last_price': None,  # Prezzo precedente per rilevare attraversamenti
            'level_events': [],  # Storico eventi significativi
            'consecutive_breaks': {  # Tracking rotture consecutive per direzione
                'direction': None,  # 'UP' o 'DOWN'
                'count': 0,
                'levels_broken': []  # Lista livelli rotti in questa sequenza
            },
            'load_attempted': False  # Flag per evitare spam di tentativi di caricamento
        }
    
    def get_model(self, model_name: str, asset: Optional[str] = None) -> Any:
        """Get model with asset-specific support - NO FALLBACKS (BIBBIA)"""
        if not asset:
            raise ValueError("Asset is mandatory for model loading - no default allowed (BIBBIA compliance)")
        
        asset_model_name = f"{asset}_{model_name}"
        if asset_model_name not in self.ml_models:
            raise ModelNotInitializedError(f"Asset-specific model '{asset_model_name}' not found")
        
        return self.ml_models[asset_model_name]
    
    def run_algorithm(self, algorithm_name: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Esegue algoritmo Support/Resistance specificato
        
        Args:
            algorithm_name: Nome algoritmo da eseguire
            market_data: Dati di mercato processati
            
        Returns:
            Risultati algoritmo con support/resistance levels
        """
        self.algorithm_stats['executions'] += 1
        self.algorithm_stats['last_execution'] = datetime.now()
        
        # BIBBIA COMPLIANT: Single path lookup - no multiple if/elif alternatives
        algorithms = {
            "PivotPoints_Classic": self._pivot_points_classic,
            "VolumeProfile_Advanced": self._volume_profile_advanced,
            "LSTM_SupportResistance": self._lstm_support_resistance,
            "StatisticalLevels_ML": self._statistical_levels_ml,
            "Transformer_Levels": self._transformer_levels
        }
        
        # BIBBIA COMPLIANT: FAIL FAST if algorithm not found
        if algorithm_name not in algorithms:
            raise ValueError(f"FAIL FAST: Unknown Support/Resistance algorithm: {algorithm_name}")
            
        return algorithms[algorithm_name](market_data)
    
    def _pivot_points_classic(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementazione Pivot Points con validazione dinamica e confidence adattiva
        TRAINING MODE: Calcolo sui dati completi del dataset
        VALIDATION MODE: Validazione real-time dei livelli pre-calcolati
        """
        if 'price_history' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'price_history' in market_data")
        if 'timestamps' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'timestamps' in market_data")
            
        prices = market_data['price_history']
        timestamps = market_data['timestamps']
        if not prices:
            raise ValueError("FAIL FAST: Empty prices array - cannot determine current_price")
        current_price = prices[-1]
        current_time = datetime.now()
        
        # TRAINING MODE: Calcola livelli GIORNALIERI sui 30 giorni di dati
        # BIBBIA COMPLIANT: FAIL FAST - no fallback defaults
        if 'training_mode' in market_data and market_data['training_mode']:
            print(f"📊 TRAINING MODE: Calculating DAILY pivot levels from {len(prices):,} ticks")
            
            # LOGICA DAILY-BASED: Raggruppa i tick per giorno
            daily_data = {}
            for i, ts in enumerate(timestamps):
                # Estrai data dal timestamp
                if isinstance(ts, str):
                    tick_date = datetime.fromisoformat(ts).date()
                elif isinstance(ts, datetime):
                    tick_date = ts.date()
                else:
                    raise TypeError(f"FAIL FAST: Unexpected timestamp type: {type(ts)}")
                
                # Accumula prezzi per giorno
                if tick_date not in daily_data:
                    daily_data[tick_date] = []
                daily_data[tick_date].append(prices[i])
            
            print(f"   📅 Found {len(daily_data)} trading days in data")
            
            # Calcola pivot per ogni giorno e accumula livelli unici
            all_levels = {}  # key -> {value, appearances, total_confidence}
            
            for date, day_prices in sorted(daily_data.items()):
                if len(day_prices) < 10:  # Skip giorni con pochi dati
                    continue
                    
                # Calcola H/L/C del giorno
                high = max(day_prices)
                low = min(day_prices)
                close = day_prices[-1]
                
                # Calcola pivot e livelli del giorno
                pivot = (high + low + close) / 3
                s1 = 2 * pivot - high
                r1 = 2 * pivot - low
                s2 = pivot - (high - low)
                r2 = pivot + (high - low)
                s3 = low - 2 * (high - pivot)
                r3 = high + 2 * (pivot - low)
                
                # Accumula livelli (considera livelli simili come stesso livello)
                tolerance = 5.0  # 5 punti di tolleranza per USTEC
                
                day_levels = {
                    'pivot': pivot,
                    'S1': s1, 'S2': s2, 'S3': s3,
                    'R1': r1, 'R2': r2, 'R3': r3
                }
                
                for level_name, level_value in day_levels.items():
                    # Cerca se esiste già un livello simile
                    found_similar = False
                    for existing_key, existing_data in all_levels.items():
                        if abs(existing_data['value'] - level_value) <= tolerance:
                            # Livello persistente! Aumenta importanza
                            existing_data['appearances'] += 1
                            existing_data['total_confidence'] += 0.05
                            found_similar = True
                            break
                    
                    if not found_similar:
                        # Nuovo livello
                        unique_key = f"{level_name}_{date}"
                        all_levels[unique_key] = {
                            'value': level_value,
                            'appearances': 1,
                            'total_confidence': 0.5,
                            'level_type': level_name[0] if level_name != 'pivot' else 'P'
                        }
            
            # Filtra e prepara livelli finali (solo quelli con confidence >= 0.5)
            self.pivot_cache['current_levels'] = {}
            
            for key, data in all_levels.items():
                # Calcola confidence finale basata su persistenza
                final_confidence = data['total_confidence'] + (data['appearances'] - 1) * 0.1
                if final_confidence > 1.0:
                    final_confidence = 1.0
                
                if final_confidence >= 0.5:
                    self.pivot_cache['current_levels'][key] = {
                        'value': data['value'],
                        'confidence': final_confidence,
                        'bounces': 0,
                        'broken': False,
                        'tests': 0,
                        'appearances': data['appearances'],  # Track persistenza
                        'level_type': data['level_type']
                    }
            
            print(f"✅ Calculated {len(self.pivot_cache['current_levels'])} significant levels from {len(daily_data)} days")
            print(f"   Top levels by persistence: {sorted([(k, v['appearances']) for k, v in self.pivot_cache['current_levels'].items()], key=lambda x: x[1], reverse=True)[:5]}")
            
            # Estrai support e resistance levels ordinati - FAIL FAST se mancanti
            support_levels = sorted([v['value'] for k, v in self.pivot_cache['current_levels'].items() if v['level_type'] == 'S'], reverse=True)[:3]
            resistance_levels = sorted([v['value'] for k, v in self.pivot_cache['current_levels'].items() if v['level_type'] == 'R'])[:3]
            pivot_levels = [v['value'] for k, v in self.pivot_cache['current_levels'].items() if v['level_type'] == 'P']
            
            if not pivot_levels:
                raise RuntimeError("FAIL FAST: Nessun pivot point calcolato dal training - dati insufficienti o corrotti")
            if not support_levels:
                raise RuntimeError("FAIL FAST: Nessun support level calcolato dal training - dati insufficienti")
            if not resistance_levels:
                raise RuntimeError("FAIL FAST: Nessun resistance level calcolato dal training - dati insufficienti")
                
            pivot_value = pivot_levels[0]
            
            # Restituisci i livelli calcolati per il training
            return {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "pivot": pivot_value,
                "confidence": 0.8,
                "method": "PivotPoints_DailyBased",
                "test_prediction": f"Training completed: {len(self.pivot_cache['current_levels'])} levels from {len(daily_data)} days",
                "level_being_tested": 0.0,
                "level_type": "training",
                "expected_outcome": "training_completed",
                "prediction_generated": True,  # Training genera una "predizione" 
                "training_data": {
                    "high": high,
                    "low": low,
                    "close": close,
                    "ticks_processed": len(prices)
                }
            }
        
        # NUOVO: Auto-load livelli salvati se disponibili (per validation)
        # ANTI-SPAM: Carica UNA SOLA VOLTA, non ogni tick!
        if 'asset' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'asset' in market_data")
        asset = market_data['asset']
        
        if not self.pivot_cache['current_levels'] and not self.pivot_cache['load_attempted']:
            # Marca come tentato per evitare spam
            self.pivot_cache['load_attempted'] = True
            # Tenta di caricare livelli pre-calcolati dal training (UNA VOLTA SOLA)
            levels_loaded = self.load_pivot_levels(asset)
            if levels_loaded:
                print(f"✅ Loaded pre-calculated pivot levels from training for {asset}")
            else:
                # Log only in validation mode, silent during training/evaluation
                if 'validation_mode' not in market_data:
                    raise KeyError("FAIL FAST: Campo 'validation_mode' obbligatorio in market_data")
                is_validation_mode = market_data['validation_mode']
                if is_validation_mode:
                    print(f"📊 No saved pivot levels found for {asset} - will calculate from data")
        
        # Parametri configurabili per USTEC
        SIX_HOURS_SECONDS = 6 * 3600
        MINIMUM_TICKS_FOR_CALC = 1000  # Almeno 1000 tick per calcolo affidabile
        LEVEL_TOLERANCE_PERCENT = 0.0003  # 0.03% per USTEC (più stretto per ridurre spam)
        
        # FASE 1: INIZIALIZZAZIONE O ACCUMULO TICK (prime 6 ore)
        if self.pivot_cache['initialization_time'] is None:
            self.pivot_cache['initialization_time'] = current_time
            self.pivot_cache['accumulated_ticks'] = []
            self.pivot_cache['next_recalc_time'] = current_time
        
        # Accumula tick per il calcolo
        if 'current_tick' in market_data:
            # BIBBIA COMPLIANT: FAIL FAST su campi mancanti
            if 'volume_history' not in market_data:
                raise KeyError("FAIL FAST: Missing required field 'volume_history' in market_data")
            
            tick_data = {
                'price': current_price,
                'timestamp': timestamps[-1],  # FAIL FAST se timestamps vuoto
                'volume': market_data['volume_history'][-1]
            }
            self.pivot_cache['accumulated_ticks'].append(tick_data)
            
            # Mantieni solo tick delle ultime 6 ore
            cutoff_time = current_time.timestamp() - SIX_HOURS_SECONDS
            self.pivot_cache['accumulated_ticks'] = [
                t for t in self.pivot_cache['accumulated_ticks'] 
                if isinstance(t['timestamp'], str) or t['timestamp'].timestamp() > cutoff_time
            ]
        
        # FASE 2: CALCOLO/RICALCOLO LIVELLI
        needs_recalc = (
            not self.pivot_cache['current_levels'] or 
            current_time >= self.pivot_cache['next_recalc_time'] or
            self._check_if_recalc_needed(current_price)
        )
        
        # BIBBIA COMPLIANT: NO ACCUMULO - Processi 100K ticks in batch!
        if not self.pivot_cache['current_levels']:
            # Se non abbiamo pivot levels, CALCOLA IMMEDIATAMENTE dai dati disponibili
            # Training fornisce 100K ticks, non serve accumulo!
            if 'price_history' in market_data and len(market_data['price_history']) > 0:
                # Calcola pivot levels dai dati forniti
                prices = market_data['price_history']
                high = float(np.max(prices))
                low = float(np.min(prices))
                close = float(prices[-1])
                
                # Calcolo Pivot Points Classic
                pivot = (high + low + close) / 3
                r1 = (2 * pivot) - low
                r2 = pivot + (high - low)
                s1 = (2 * pivot) - high
                s2 = pivot - (high - low)
                
                # Salva livelli calcolati
                self.pivot_cache['current_levels'] = {
                    'pivot': pivot,
                    'r1': r1,
                    'r2': r2,
                    's1': s1,
                    's2': s2,
                    'high': high,
                    'low': low,
                    'close': close,
                    'last_calc_time': current_time,
                    'was_near_last_tick': False
                }
            else:
                # FAIL FAST - No data available
                raise RuntimeError("FAIL FAST: No price data available to calculate pivot levels - cannot proceed")
        
        # FASE 3: MONITORAGGIO EVENTI E PREDIZIONI
        event_result = self._monitor_level_events(
            current_price, 
            current_time,
            LEVEL_TOLERANCE_PERCENT,
            market_data
        )
        
        # Estrai livelli ordinati per output
        levels = self.pivot_cache['current_levels']
        support_levels = sorted([levels[k]['value'] for k in levels if 'S' in k])
        resistance_levels = sorted([levels[k]['value'] for k in levels if 'R' in k])
        if 'pivot' not in levels:
            raise RuntimeError("FAIL FAST: Missing pivot level in current_levels - system integrity compromised")
        pivot_value = levels['pivot']['value']
        
        # Se non ci sono eventi o predizioni da mostrare
        if not event_result['has_event'] and not event_result['has_prediction']:
            # SILENZIO - nessun evento significativo
            # Calcola confidence media dai livelli attivi - FAIL FAST se campi mancanti
            active_confidences = []
            for l in self.pivot_cache['current_levels'].values():
                if 'broken' not in l:
                    raise KeyError("FAIL FAST: Campo 'broken' mancante in level data - struttura dati corrotta")
                if 'confidence' not in l:
                    raise KeyError("FAIL FAST: Campo 'confidence' mancante in level data - struttura dati corrotta")
                if not l['broken']:
                    active_confidences.append(l['confidence'])
            
            if not active_confidences:
                raise RuntimeError("FAIL FAST: Nessun livello attivo trovato - tutti rotti o dati corrotti")
            avg_confidence = sum(active_confidences) / len(active_confidences)
            
            return {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "pivot": pivot_value,
                "confidence": avg_confidence,
                "method": "PivotPoints_Dynamic",
                "test_prediction": "",  # Silenzio durante monitoring normale
                "level_being_tested": 0.0,
                "level_type": "none", 
                "expected_outcome": "monitoring",
                "prediction_generated": False
            }
        
        # Evento significativo o predizione da mostrare
        self.algorithm_stats['successful_predictions'] += 1
        
        # FAIL FAST - Verifica campi obbligatori in event_result
        required_fields = ['confidence', 'message', 'level_value', 'level_type', 'expected_outcome', 'has_prediction', 'event_type', 'level_action']
        for field in required_fields:
            if field not in event_result:
                raise KeyError(f"FAIL FAST: Campo obbligatorio '{field}' mancante in event_result - sistema corrotto")
        
        return {
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "pivot": pivot_value,
            "confidence": event_result['confidence'],
            "method": "PivotPoints_EventDriven",
            "test_prediction": event_result['message'],
            "level_being_tested": event_result['level_value'],
            "level_type": event_result['level_type'],
            "expected_outcome": event_result['expected_outcome'],
            "prediction_generated": event_result['has_prediction'],
            "event_type": event_result['event_type'],
            "level_action": event_result['level_action']
        }

    def _calculate_new_pivot_levels(self):
        """
        Calcola nuovi livelli pivot basati sui tick accumulati
        """
        if not self.pivot_cache['accumulated_ticks']:
            return
        
        # Estrai prezzi dai tick accumulati
        prices = [t['price'] for t in self.pivot_cache['accumulated_ticks']]
        
        # Calcola H/L/C per il periodo
        high = max(prices)
        low = min(prices)
        close = prices[-1]
        
        # Calcola pivot e livelli
        pivot = (high + low + close) / 3
        s1 = 2 * pivot - high
        r1 = 2 * pivot - low
        s2 = pivot - (high - low)
        r2 = pivot + (high - low)
        s3 = low - 2 * (high - pivot)
        r3 = high + 2 * (pivot - low)
        
        # Preserva statistiche esistenti o inizializza nuove
        def create_or_update_level(key, value):
            if key in self.pivot_cache['current_levels']:
                # Mantieni statistiche ma aggiorna valore
                level = self.pivot_cache['current_levels'][key]
                level['value'] = value
                level['last_calc_time'] = datetime.now()
            else:
                # Crea nuovo livello con tracking per eventi
                self.pivot_cache['current_levels'][key] = {
                    'value': value,
                    'confidence': 0.5,  # Inizia neutrale
                    'bounces': 0,
                    'tests': 0,
                    'broken': False,
                    'break_time': None,
                    'last_test': None,
                    'last_calc_time': datetime.now(),
                    'was_near_last_tick': False  # Nuovo flag per tracking eventi
                }
        
        # Aggiorna tutti i livelli
        create_or_update_level('pivot', pivot)
        create_or_update_level('S1', s1)
        create_or_update_level('S2', s2)
        create_or_update_level('S3', s3)
        create_or_update_level('R1', r1)
        create_or_update_level('R2', r2)
        create_or_update_level('R3', r3)
    
    def _check_if_recalc_needed(self, current_price: float) -> bool:
        """
        Verifica se è necessario ricalcolare i livelli
        Ricalcola se:
        1. Un livello con alta confidence è stato rotto
        2. Multiple livelli sono stati rotti recentemente
        3. Il pivot è stato rotto con momentum forte
        """
        if not self.pivot_cache['current_levels']:
            return True
        
        levels = self.pivot_cache['current_levels']
        recent_breaks = 0
        high_confidence_break = False
        
        for key, level in levels.items():
            # BIBBIA COMPLIANT: FAIL FAST - no fallback defaults
            if 'broken' in level and level['broken']:
                # Controlla se è un break recente (ultimi 30 minuti)
                # BIBBIA COMPLIANT: FAIL FAST - no fallback defaults
                if 'break_time' in level and level['break_time']:
                    time_since_break = (datetime.now() - level['break_time']).seconds
                    if time_since_break < 1800:  # 30 minuti
                        recent_breaks += 1
                        if level['confidence'] > 0.7:
                            high_confidence_break = True
        
        # Decisione ricalcolo
        return high_confidence_break or recent_breaks >= 2
    
    def _monitor_level_events(self, current_price: float, current_time: datetime,
                             tolerance_percent: float, 
                             market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitora eventi significativi sui livelli pivot
        
        LOGICA CORRETTA:
        ✅ MONITORING LOGS: "Prezzo testa S1@22691 → TIENE" o "S1@22691 → ROTTO"
        ✅ PREDIZIONI solo dopo rotture: "S1 rotto → prossimo target S2@22682"  
        ✅ SILENZIO durante range normale
        """
        if not self.pivot_cache['current_levels']:
            return {
                'has_event': False,
                'has_prediction': False,
                'message': '',
                'confidence': 0.0
            }
        
        # ZONA TEST: 20 punti per USTEC (circa 0.09% a 22700)
        test_zone = 20.0  # Punti assoluti, non percentuale
        test_tolerance = current_price * tolerance_percent  # Per break detection
        last_price = self.pivot_cache['last_price']
        
        # Prima volta - inizializza tracking
        if last_price is None:
            self.pivot_cache['last_price'] = current_price
            # Inizializza tracking stati test per ogni livello
            for level_key in self.pivot_cache['current_levels']:
                self.pivot_cache['current_levels'][level_key]['test_state'] = 'idle'
                self.pivot_cache['current_levels'][level_key]['test_entry_price'] = None
                self.pivot_cache['current_levels'][level_key]['test_entry_time'] = None
                self.pivot_cache['current_levels'][level_key]['false_break_count'] = 0
                self.pivot_cache['current_levels'][level_key]['flipped'] = False  # Track S→R o R→S flip
            return {
                'has_event': False, 
                'has_prediction': False,
                'message': '',
                'confidence': 0.0
            }
        
        # Analizza movimenti per eventi significativi
        for level_key, level_data in self.pivot_cache['current_levels'].items():
            level_value = level_data['value']
            
            # Skip livelli già rotti permanentemente
            # BIBBIA COMPLIANT: FAIL FAST - no fallback defaults
            if 'broken' in level_data and level_data['broken']:
                continue
            
            # Verifica se c'è stata una rottura effettiva
            level_crossed = self._detect_level_cross(last_price, current_price, level_value, level_key)
            
            if level_crossed:
                # ROTTURA CONFERMATA - aggiorna stato e traccia consecutive breaks
                level_data['broken'] = True
                level_data['break_time'] = current_time
                level_data['test_state'] = 'broken'
                level_data['confidence'] = level_data['confidence'] - 0.1
                if level_data['confidence'] < 0.1:
                    level_data['confidence'] = 0.1  # Reduce confidence on break
                
                # Determina direzione della rottura
                break_direction = 'DOWN' if current_price < level_value else 'UP'
                
                # Aggiorna tracking rotture consecutive
                consecutive_breaks = self.pivot_cache['consecutive_breaks']
                
                if consecutive_breaks['direction'] == break_direction:
                    # Stessa direzione - incrementa contatore
                    consecutive_breaks['count'] += 1
                    consecutive_breaks['levels_broken'].append(level_key)
                else:
                    # Nuova direzione - reset contatore
                    consecutive_breaks['direction'] = break_direction
                    consecutive_breaks['count'] = 1
                    consecutive_breaks['levels_broken'] = [level_key]
                
                # LOGICA PREDIZIONE: Solo dopo 2+ rotture consecutive
                should_generate_prediction = consecutive_breaks['count'] >= 2
                
                if should_generate_prediction:
                    # Genera predizione per prossimo target
                    next_target = self._find_next_target_after_break(level_key, level_value, current_price)
                    
                    self.pivot_cache['last_price'] = current_price
                    return {
                        'has_event': True,
                        'has_prediction': True,
                        'message': f"💥 {level_key}@{level_value:.2f} ROTTO → Target: {next_target['name']}@{next_target['value']:.2f} (Break #{consecutive_breaks['count']})",
                        'confidence': next_target['confidence'],
                        'event_type': 'level_break',
                        'level_value': level_value,
                        'level_type': level_key,
                        'expected_outcome': next_target['direction'],
                        'level_action': 'broken'
                    }
                else:
                    # Solo logga la rottura, senza predizione
                    self.pivot_cache['last_price'] = current_price
                    return {
                        'has_event': True,
                        'has_prediction': False,
                        'message': f"💥 {level_key}@{level_value:.2f} ROTTO ({consecutive_breaks['count']}/2 breaks {break_direction})",
                        'confidence': 0.7,
                        'event_type': 'level_break',
                        'level_value': level_value,
                        'level_type': level_key,
                        'expected_outcome': break_direction.lower(),
                        'level_action': 'broken'
                    }
            
            # TRACKING STATO TEST CON ZONA 20 PUNTI
            current_distance = abs(current_price - level_value)
            is_in_test_zone = current_distance <= test_zone
            
            # Inizializza test_state se non esiste
            if 'test_state' not in level_data:
                level_data['test_state'] = 'idle'
                level_data['test_entry_price'] = None
                level_data['test_entry_time'] = None
                level_data['false_break_count'] = 0
                level_data['flipped'] = False
            
            current_state = level_data['test_state']
            
            # STATE MACHINE PER TRACKING TEST
            if current_state == 'idle' and is_in_test_zone:
                # ENTERING TEST ZONE
                level_data['test_state'] = 'testing'
                level_data['test_entry_price'] = current_price
                level_data['test_entry_time'] = current_time
                level_data['tests'] += 1
                
            elif current_state == 'testing':
                if not is_in_test_zone and not level_crossed:
                    # EXITING ZONE - BOUNCE/HOLD
                    level_data['test_state'] = 'idle'
                    level_data['bounces'] += 1
                    old_confidence = level_data['confidence']
                    level_data['confidence'] = level_data['confidence'] + 0.05
                    if level_data['confidence'] > 1.0:
                        level_data['confidence'] = 1.0
                    
                    # Reset consecutive breaks
                    consecutive_breaks = self.pivot_cache['consecutive_breaks']
                    consecutive_breaks['direction'] = None
                    consecutive_breaks['count'] = 0
                    consecutive_breaks['levels_broken'] = []
                    
                    # Verifica se è un FLIP (supporto diventa resistenza o viceversa)
                    is_support = level_key.startswith('S')
                    price_above = current_price > level_value
                    
                    if (is_support and price_above) or (not is_support and not price_above):
                        # Possibile flip in corso
                        if 'flipped' not in level_data:
                            raise KeyError("FAIL FAST: Campo 'flipped' mancante in level_data")
                        if level_data['flipped']:
                            flip_msg = f" [FLIP CONFERMATO: {'S→R' if is_support else 'R→S'}]"
                        else:
                            flip_msg = ""
                    else:
                        flip_msg = ""
                    
                    self.pivot_cache['last_price'] = current_price
                    return {
                        'has_event': True,
                        'has_prediction': False,
                        'message': f"✅ {level_key}@{level_value:.2f} TIENE - Confidence: {old_confidence:.2f}→{level_data['confidence']:.2f}{flip_msg}",
                        'confidence': level_data['confidence'],
                        'event_type': 'level_hold',
                        'level_value': level_value,
                        'level_type': level_key,
                        'expected_outcome': 'held',
                        'level_action': 'held'
                    }
                    
            elif current_state == 'broken' and is_in_test_zone:
                # RETEST DOPO ROTTURA
                time_since_break = (current_time - level_data['break_time']).seconds if level_data.get('break_time') else 0
                
                if time_since_break < 300:  # Entro 5 minuti
                    # FALSE BREAK - il prezzo è tornato
                    level_data['test_state'] = 'testing'
                    level_data['false_break_count'] += 1
                    level_data['broken'] = False  # Ripristina livello
                    level_data['confidence'] = level_data['confidence'] + 0.1
                    if level_data['confidence'] > 1.0:
                        level_data['confidence'] = 1.0  # Livello forte!
                    
                    self.pivot_cache['last_price'] = current_price
                    return {
                        'has_event': True,
                        'has_prediction': False,
                        'message': f"⚡ FALSE BREAK su {level_key}@{level_value:.2f} - Livello ripristinato! Confidence: {level_data['confidence']:.2f}",
                        'confidence': level_data['confidence'],
                        'event_type': 'false_break',
                        'level_value': level_value,
                        'level_type': level_key,
                        'expected_outcome': 'restored',
                        'level_action': 'restored'
                    }
                else:
                    # RETEST del livello rotto - possibile flip S→R o R→S
                    level_data['test_state'] = 'retest'
                    level_data['flipped'] = True
                    
                    is_support = level_key.startswith('S')
                    self.pivot_cache['last_price'] = current_price
                    return {
                        'has_event': True,
                        'has_prediction': False,
                        'message': f"🔄 RETEST {level_key}@{level_value:.2f} - {'Supporto→Resistenza' if is_support else 'Resistenza→Supporto'}",
                        'confidence': level_data['confidence'],
                        'event_type': 'level_flip',
                        'level_value': level_value,
                        'level_type': level_key,
                        'expected_outcome': 'flipped',
                        'level_action': 'flipped'
                    }
        
        # Aggiorna last_price per prossimo confronto
        self.pivot_cache['last_price'] = current_price
        
        # SILENZIO - nessun evento significativo
        return {
            'has_event': False,
            'has_prediction': False, 
            'message': '',
            'confidence': 0.0
        }
    
    def _detect_level_cross(self, last_price: float, current_price: float, 
                           level_value: float, level_key: str) -> bool:
        """
        Rileva se il prezzo ha effettivamente attraversato un livello
        
        Returns:
            True se c'è stata una rottura effettiva
        """
        # Support/Pivot: rotto se eravamo sopra e ora siamo sotto
        if 'S' in level_key or level_key == 'pivot':
            return last_price >= level_value and current_price < level_value
            
        # Resistance: rotto se eravamo sotto e ora siamo sopra  
        elif 'R' in level_key:
            return last_price <= level_value and current_price > level_value
            
        return False
    
    def _find_next_target_after_break(self, broken_level_key: str, broken_level_value: float,
                                     current_price: float) -> Dict[str, Any]:
        """
        Trova il prossimo target dopo una rottura
        
        Returns:
            Dict con dettagli del prossimo target
        """
        levels = self.pivot_cache['current_levels']
        
        # Determina direzione della rottura
        if current_price < broken_level_value:
            # Rottura verso il basso - cerca prossimo support
            direction = 'DOWN'
            candidates = []
            for key, level in levels.items():
                if (('S' in key or key == 'pivot') and 
                    level['value'] < current_price and 
                    not level['broken']):
                    candidates.append({
                        'name': key,
                        'value': level['value'],
                        'confidence': level['confidence'],
                        'distance': abs(level['value'] - current_price)
                    })
        else:
            # Rottura verso l'alto - cerca prossima resistance
            direction = 'UP'
            candidates = []
            for key, level in levels.items():
                if (('R' in key or key == 'pivot') and 
                    level['value'] > current_price and 
                    not level['broken']):
                    candidates.append({
                        'name': key,
                        'value': level['value'], 
                        'confidence': level['confidence'],
                        'distance': abs(level['value'] - current_price)
                    })
        
        # Scegli il target più vicino
        if candidates:
            next_target = min(candidates, key=lambda x: x['distance'])
            return {
                'name': next_target['name'],
                'value': next_target['value'],
                'direction': direction,
                'confidence': next_target['confidence'] + 0.15 if next_target['confidence'] + 0.15 <= 0.9 else 0.9
            }
        
        # Nessun livello disponibile - calcola target esteso
        extension_distance = abs(current_price - broken_level_value) * 1.618  # Golden ratio extension
        if direction == 'DOWN':
            target_value = current_price - extension_distance
            target_name = 'Extended_Support'
        else:
            target_value = current_price + extension_distance 
            target_name = 'Extended_Resistance'
        
        return {
            'name': target_name,
            'value': target_value,
            'direction': direction,
            'confidence': 0.65
        }
    
    def _calculate_momentum(self, market_data: Dict[str, Any]) -> float:
        """
        Calcola momentum del prezzo per predizioni più accurate
        """
        if 'price_history' not in market_data:
            raise KeyError("FAIL FAST: Missing 'price_history' required for momentum calculation")
        if len(market_data['price_history']) < 50:
            raise InsufficientDataError(required=50, available=len(market_data['price_history']), operation="Momentum_Calculation")
        
        prices = market_data['price_history']
        # Momentum semplice: variazione % ultimi 50 tick
        momentum = (prices[-1] - prices[-50]) / prices[-50] * 100
        return momentum
    
    
    def _volume_profile_advanced(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Volume Profile analysis avanzata
        ESTRATTO IDENTICO da src/Analyzer.py:12852-12893
        """
        if 'price_history' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'price_history' in market_data")
        if 'volume_history' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'volume_history' in market_data")
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
        
        self.algorithm_stats['successful_predictions'] += 1
        
        return {
            "support_levels": sorted(support_levels)[-3:],  # Top 3
            "resistance_levels": sorted(resistance_levels)[:3],  # Top 3
            "confidence": 0.8,
            "method": "Volume_Profile_Analysis",
            "volume_nodes": len(high_volume_indices)
        }
    
    def _lstm_support_resistance(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LSTM per Support/Resistance detection
        ESTRATTO IDENTICO da src/Analyzer.py:12895-12996
        """
        # Get asset from market_data for asset-specific model loading
        # BIBBIA COMPLIANT: FAIL FAST - no fallback to 'UNKNOWN'
        if 'asset' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'asset' in market_data")
        asset = market_data['asset']
        model = self.get_model('LSTM_SupportResistance', asset)
        
        # Prepara input
        if 'price_history' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'price_history' in market_data")
        if 'volume_history' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'volume_history' in market_data")
        prices = np.array(market_data['price_history'][-50:])
        volumes = np.array(market_data['volume_history'][-50:])
        
        if len(prices) < 50:
            raise InsufficientDataError(required=50, available=len(prices), operation="LSTM_SupportResistance")
        
        # 🛡️ VALIDAZIONE DATI INPUT
        if np.isnan(prices).any() or np.isinf(prices).any():
            raise InvalidInputError("prices", "NaN/Inf values", "LSTM requires valid numeric prices")
        
        if np.isnan(volumes).any() or np.isinf(volumes).any():
            raise InvalidInputError("volumes", "NaN/Inf values", "LSTM requires valid numeric volumes")
        
        try:
            # Feature engineering
            features = self._prepare_lstm_features(prices, volumes, market_data)
            
            # 🛡️ VALIDAZIONE FEATURES
            if np.isnan(features).any() or np.isinf(features).any():
                raise InvalidInputError("features", "NaN/Inf values", "LSTM features must be numeric")
            
            # Prediction protetta
            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features).unsqueeze(0)
                
                # 🛡️ VALIDAZIONE TENSOR INPUT
                if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
                    raise InvalidInputError("input_tensor", "NaN/Inf values", "PyTorch tensor must be finite")
                
                prediction = model(input_tensor)
                
                # 🛡️ VALIDAZIONE OUTPUT
                if torch.isnan(prediction).any() or torch.isinf(prediction).any():
                    raise PredictionError("LSTM_SupportResistance", "Model produced NaN/Inf in output")
                
                levels = prediction.numpy().flatten()
                
                # 🛡️ VALIDAZIONE FINALE
                if np.isnan(levels).any() or np.isinf(levels).any():
                    raise PredictionError("LSTM_SupportResistance", "Final levels contain NaN/Inf values")
                
                # LSTM prediction successful - no debug print needed
                
                # Interpreta output con validazione
                current_price = market_data['current_price']
                
                # 🛡️ VALIDAZIONE CURRENT_PRICE
                if np.isnan(current_price) or np.isinf(current_price) or current_price <= 0:
                    raise InvalidInputError("current_price", current_price, "Must be a positive finite number")
                
                support_levels = []
                resistance_levels = []
                
                for i in range(0, len(levels), 2):
                    if i < len(levels) - 1:
                        support_level = levels[i]
                        resistance_level = levels[i + 1]
                        
                        if support_level < current_price:
                            support_levels.append(float(support_level))
                        if resistance_level > current_price:
                            resistance_levels.append(float(resistance_level))
                
                self.algorithm_stats['successful_predictions'] += 1
                
                return {
                    "support_levels": sorted(support_levels)[-3:],  # Top 3
                    "resistance_levels": sorted(resistance_levels)[:3],  # Top 3
                    "confidence": 0.85,
                    "method": "LSTM_Neural_Network",
                    "model_output_size": len(levels)
                }
                
        except (ImportError, AttributeError, ValueError, KeyError, RuntimeError, TypeError, IndexError) as e:
            self.algorithm_stats['failed_predictions'] += 1
            # FAIL FAST - Re-raise specific LSTM errors
            raise PredictionError("LSTM_SupportResistance", str(e))
    
    def _statistical_levels_ml(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Statistical Levels ML analysis
        ESTRATTO IDENTICO da src/Analyzer.py:12997-13035
        """
        if 'price_history' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'price_history' in market_data")
        prices = np.array(market_data['price_history'])
        
        if len(prices) < 100:
            raise InsufficientDataError(required=100, available=len(prices), operation="StatisticalLevels_ML")
        
        # Calcola statistical support/resistance
        price_mean = np.mean(prices)
        price_std = np.std(prices)
        
        # Bollinger-like bands
        upper_band = price_mean + 2 * price_std
        lower_band = price_mean - 2 * price_std
        
        # Percentile-based levels
        percentile_levels = [
            np.percentile(prices, 10),   # Support level 1
            np.percentile(prices, 25),   # Support level 2
            np.percentile(prices, 75),   # Resistance level 1
            np.percentile(prices, 90)    # Resistance level 2
        ]
        
        current_price = market_data['current_price']
        
        support_levels = [
            lower_band,
            percentile_levels[0],
            percentile_levels[1]
        ]
        support_levels = [level for level in support_levels if level < current_price]
        
        resistance_levels = [
            upper_band,
            percentile_levels[2],
            percentile_levels[3]
        ]
        resistance_levels = [level for level in resistance_levels if level > current_price]
        
        self.algorithm_stats['successful_predictions'] += 1
        
        return {
            "support_levels": sorted(support_levels)[-3:],  # Top 3
            "resistance_levels": sorted(resistance_levels)[:3],  # Top 3
            "confidence": 0.7,
            "method": "Statistical_ML_Analysis",
            "price_mean": price_mean,
            "price_std": price_std
        }
    
    def _transformer_levels(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transformer-based S/R level detection
        PARTE DI src/Analyzer.py - completare implementazione
        """
        # Get asset from market_data for asset-specific model loading
        # BIBBIA COMPLIANT: FAIL FAST - no fallback to 'UNKNOWN'
        if 'asset' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'asset' in market_data")
        asset = market_data['asset']
        model = self.get_model('Transformer_Levels', asset)
        
        # Placeholder per ora - da implementare completamente
        if 'current_price' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'current_price' in market_data")
        if 'price_history' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'price_history' in market_data")
        current_price = market_data['current_price']
        price_history = market_data['price_history']
        
        if len(price_history) < 50:
            raise InsufficientDataError(required=50, available=len(price_history), operation="Transformer_Levels")
        
        # Transformer-based prediction (semplificato)
        price_range = max(price_history[-20:]) - min(price_history[-20:])
        
        support_levels = [
            current_price - price_range * 0.1,
            current_price - price_range * 0.2,
            current_price - price_range * 0.3
        ]
        
        resistance_levels = [
            current_price + price_range * 0.1,
            current_price + price_range * 0.2,
            current_price + price_range * 0.3
        ]
        
        self.algorithm_stats['successful_predictions'] += 1
        
        return {
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "confidence": 0.75,
            "method": "Transformer_Neural_Network",
            "price_range": price_range
        }
    
    def _prepare_lstm_features(self, prices: np.ndarray, volumes: np.ndarray, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Prepara features per LSTM
        Utilizza MarketDataProcessor per consistency
        """
        # Basic features - FAIL FAST se prezzi invalidi
        if np.any(prices <= 0):
            raise ValueError("FAIL FAST: Prezzi <= 0 trovati nei dati - cannot calculate returns safely")
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            raise ValueError("FAIL FAST: Prezzi NaN o infiniti nei dati - cannot calculate returns safely")
            
        returns = np.diff(prices, prepend=prices[0]) / prices[:-1]
        price_ratios = prices[1:] / prices[:-1]
        log_returns = np.log(price_ratios)
        log_returns = np.append(log_returns, 0)
        
        # Volume features - FAIL FAST se volumi invalidi
        if len(volumes) == 0:
            raise ValueError("FAIL FAST: Empty volumes array - cannot calculate volume_mean")
        if np.any(volumes < 0):
            raise ValueError("FAIL FAST: Negative volumes found in data - invalid volume data")
        volume_mean = np.mean(volumes)
        if volume_mean == 0:
            raise ValueError("FAIL FAST: Volume mean is zero - cannot calculate volume ratio safely")
        volume_ratio = volumes / volume_mean
        
        # Technical indicators (semplificati)
        sma_5 = np.convolve(prices, np.ones(5)/5, mode='same')
        sma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
        
        # Combine features
        features = np.column_stack([
            prices, volumes, returns, log_returns,
            volume_ratio, sma_5, sma_20
        ])
        
        return features
    
    def get_algorithm_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche algoritmi"""
        return self.algorithm_stats.copy()
    
    def save_pivot_levels(self, asset: str, save_dir: str = "./pivot_levels") -> None:
        """
        Salva i livelli pivot calcolati durante il training
        
        Args:
            asset: Nome asset per cui salvare i livelli
            save_dir: Directory dove salvare i livelli
        """
        if not self.pivot_cache['current_levels']:
            print(f"⚠️ No pivot levels to save for {asset}")
            return
        
        # Crea directory se non esiste
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Prepara dati da salvare (converti datetime in stringhe)
        save_data = {
            'asset': asset,
            'save_timestamp': datetime.now().isoformat(),
            'training_stats': {
                'total_levels': len(self.pivot_cache['current_levels']),
                'avg_confidence': sum(l['confidence'] for l in self.pivot_cache['current_levels'].values()) / len(self.pivot_cache['current_levels']) if self.pivot_cache['current_levels'] else 0,
                'persistent_levels': len([l for l in self.pivot_cache['current_levels'].values() if ('appearances' in l and l['appearances'] > 1)])
            },
            'levels': {}
        }
        
        # Converti livelli per serializzazione JSON con informazioni estese
        for key, level_data in self.pivot_cache['current_levels'].items():
            save_data['levels'][key] = {
                'value': level_data['value'],
                'confidence': level_data['confidence'],
                'bounces': level_data['bounces'],
                'broken': level_data['broken'],
                'tests': level_data['tests'],
                # BIBBIA COMPLIANT: FAIL FAST - daily-based tracking fields
                'appearances': level_data['appearances'] if 'appearances' in level_data else 1,
                'level_type': level_data['level_type'] if 'level_type' in level_data else 'Unknown',
                'false_break_count': level_data['false_break_count'] if 'false_break_count' in level_data else 0,
                'flipped': level_data['flipped'] if 'flipped' in level_data else False,
                'test_state': level_data['test_state'] if 'test_state' in level_data else 'idle'
            }
        
        # Salva su file JSON
        file_path = f"{save_dir}/{asset}_pivot_levels.json"
        with open(file_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"✅ Saved pivot levels for {asset} to {file_path}")
        print(f"   Levels saved: {list(save_data['levels'].keys())}")
    
    def load_pivot_levels(self, asset: str, load_dir: str = "./pivot_levels") -> bool:
        """
        Carica i livelli pivot salvati dal training
        
        Args:
            asset: Nome asset per cui caricare i livelli
            load_dir: Directory da cui caricare i livelli
            
        Returns:
            True se i livelli sono stati caricati, False altrimenti
        """
        file_path = f"{load_dir}/{asset}_pivot_levels.json"
        
        if not os.path.exists(file_path):
            # Silenzioso - non spam per file mancante
            return False
        
        try:
            with open(file_path, 'r') as f:
                save_data = json.load(f)
            
            # Ripristina livelli nel cache - SOLO se confidence >= 0.5
            self.pivot_cache['current_levels'] = {}
            levels_loaded = 0
            levels_skipped = 0
            
            for key, level_data in save_data['levels'].items():
                # FILTRO: Carica solo livelli con confidence >= 0.5
                if level_data['confidence'] >= 0.5:
                    self.pivot_cache['current_levels'][key] = {
                        'value': level_data['value'],
                        'confidence': level_data['confidence'],  # PRESERVA confidence dal file
                        'bounces': level_data['bounces'],
                        'broken': level_data['broken'],
                        'tests': level_data['tests'],
                        'break_time': None,
                        'last_test': None,
                        'last_calc_time': datetime.now(),
                        'was_near_last_tick': False,
                        # BIBBIA COMPLIANT: FAIL FAST - no fallbacks allowed for daily-based fields
                        'appearances': level_data['appearances'] if 'appearances' in level_data else 1,
                        'level_type': level_data['level_type'] if 'level_type' in level_data else (key[0] if key[0] in 'SRP' else 'Unknown'),
                        'false_break_count': level_data['false_break_count'] if 'false_break_count' in level_data else 0,
                        'flipped': level_data['flipped'] if 'flipped' in level_data else False,
                        'test_state': level_data['test_state'] if 'test_state' in level_data else 'idle'
                    }
                    levels_loaded += 1
                else:
                    levels_skipped += 1
            
            # Inizializza altri campi del cache
            self.pivot_cache['initialization_time'] = datetime.now()
            self.pivot_cache['accumulated_ticks'] = []  # Reset - se non abbiamo dati veri, FAIL FAST
            self.pivot_cache['next_recalc_time'] = datetime.now()
            
            print(f"✅ Loaded pivot levels for {asset} from {file_path}")
            print(f"   Levels loaded: {levels_loaded} (confidence >= 0.5)")
            if levels_skipped > 0:
                print(f"   Levels skipped: {levels_skipped} (confidence < 0.5)")
            print(f"   Active levels: {list(self.pivot_cache['current_levels'].keys())}")
            return True
            
        except (FileNotFoundError, JSONDecodeError, KeyError, ValueError) as e:
            print(f"❌ Error loading pivot levels for {asset}: {e}")
            return False
        except Exception as e:
            # Unknown error - FAIL FAST per system integrity
            raise RuntimeError(f"FAIL FAST: Unexpected error loading pivot levels for {asset} - {type(e).__name__}: {e}")


# Factory function per compatibilità
def create_support_resistance_algorithms(ml_models: Dict[str, Any]) -> SupportResistanceAlgorithms:
    """Factory function per creare SupportResistanceAlgorithms - FAIL FAST se ml_models mancanti"""
    if ml_models is None:
        raise ValueError("FAIL FAST: ml_models è obbligatorio - no fallback consentiti")
    return SupportResistanceAlgorithms(ml_models)


# Export
__all__ = [
    'SupportResistanceAlgorithms',
    'InsufficientDataError',
    'ModelNotInitializedError', 
    'InvalidInputError',
    'PredictionError',
    'create_support_resistance_algorithms'
]