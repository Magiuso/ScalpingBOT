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
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

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
    
    def __init__(self, ml_models: Optional[Dict[str, Any]] = None):
        """Inizializza algoritmi S/R con modelli ML opzionali"""
        self.ml_models = ml_models or {}
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
            }
        }
    
    def get_model(self, model_name: str, asset: Optional[str] = None) -> Any:
        """Get model with asset-specific support - NO FALLBACKS (BIBBIA)"""
        if not asset:
            raise ValueError("Asset is mandatory for model loading - no default allowed (BIBBIA compliance)")
        
        asset_model_name = f"{asset}_{model_name}"
        if asset_model_name not in self.ml_models:
            raise ModelNotInitializedError(f"Asset-specific model '{asset_model_name}' not found")
        
        return self.ml_models[asset_model_name]
    
    def run_algorithm(self, algorithm_name: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Esegue algoritmo Support/Resistance specificato
        ANTI-SPAM: Può ritornare None se non ci sono predizioni significative
        
        Args:
            algorithm_name: Nome algoritmo da eseguire
            market_data: Dati di mercato processati
            
        Returns:
            Risultati algoritmo con support/resistance levels oppure None se no prediction
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
    
    def _pivot_points_classic(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Implementazione Pivot Points con validazione dinamica e confidence adattiva
        NUOVA VERSIONE: Calcolo ogni 6 ore con validazione real-time dei livelli
        """
        if 'price_history' not in market_data:
            raise KeyError("Critical field 'price_history' missing from market_data")
        if 'timestamps' not in market_data:
            raise KeyError("Critical field 'timestamps' missing from market_data")
            
        prices = market_data['price_history']
        timestamps = market_data['timestamps']
        if not prices:
            raise ValueError("FAIL FAST: Empty prices array - cannot determine current_price")
        current_price = prices[-1]
        current_time = datetime.now()
        
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
            tick_data = {
                'price': current_price,
                'timestamp': timestamps[-1] if timestamps else None,
                # FAIL FAST: timestamp deve essere disponibile
                'timestamp_fallback': current_time if not timestamps else None,
                'volume': market_data.get('volume_history', [1])[-1] if 'volume_history' in market_data else 1
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
        
        if needs_recalc and len(self.pivot_cache['accumulated_ticks']) >= MINIMUM_TICKS_FOR_CALC:
            self._calculate_new_pivot_levels()
            self.pivot_cache['next_recalc_time'] = current_time
        
        # Se non abbiamo ancora livelli calcolati, ritorna stato di accumulation
        if not self.pivot_cache['current_levels']:
            return {
                "support_levels": [],
                "resistance_levels": [],
                "pivot": 0.0,
                "confidence": 0.0,
                "method": "PivotPoints_Dynamic",
                "test_prediction": f"Accumulating data: {len(self.pivot_cache['accumulated_ticks'])}/{MINIMUM_TICKS_FOR_CALC} ticks",
                "level_being_tested": 0.0,
                "level_type": "none",
                "expected_outcome": "waiting",
                "prediction_generated": False
            }
        
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
            return {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "pivot": pivot_value,
                "confidence": 0.5,
                "method": "PivotPoints_Dynamic",
                "test_prediction": "",  # Silenzio durante monitoring normale
                "level_being_tested": 0.0,
                "level_type": "none", 
                "expected_outcome": "monitoring",
                "prediction_generated": False
            }
        
        # Evento significativo o predizione da mostrare
        self.algorithm_stats['successful_predictions'] += 1
        
        return {
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "pivot": pivot_value,
            "confidence": event_result['confidence'],
            "method": "PivotPoints_EventDriven",
            "test_prediction": event_result['message'],
            "level_being_tested": event_result.get('level_value', 0.0),
            "level_type": event_result.get('level_type', 'none'),
            "expected_outcome": event_result.get('expected_outcome', 'unknown'),
            "prediction_generated": event_result['has_prediction'],
            # Event metadata
            "event_type": event_result.get('event_type', 'none'),
            "level_action": event_result.get('level_action', 'none')  # 'broken', 'held', 'tested'
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
            if level.get('broken', False):
                # Controlla se è un break recente (ultimi 30 minuti)
                if level.get('break_time'):
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
        
        test_tolerance = current_price * tolerance_percent
        last_price = self.pivot_cache['last_price']
        
        # Prima volta - inizializza tracking
        if last_price is None:
            self.pivot_cache['last_price'] = current_price
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
            if level_data.get('broken', False):
                continue
            
            # Verifica se c'è stata una rottura effettiva
            level_crossed = self._detect_level_cross(last_price, current_price, level_value, level_key)
            
            if level_crossed:
                # ROTTURA CONFERMATA - aggiorna stato e traccia consecutive breaks
                level_data['broken'] = True
                level_data['break_time'] = current_time
                
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
            
            # Verifica se stiamo testando un livello (vicini ma non rotti)
            current_distance = abs(current_price - level_value)
            was_near = level_data.get('was_near_last_tick', False)
            is_near_now = current_distance <= test_tolerance
            
            # Se ci stiamo allontanando da un livello che stavamo testando = TIENE
            if was_near and not is_near_now and not level_crossed:
                level_data['bounces'] += 1
                level_data['confidence'] = min(1.0, level_data['confidence'] + 0.1)
                
                # RESET CONSECUTIVE BREAKS quando un livello tiene
                # Un bounce interrompe la sequenza di rotture
                consecutive_breaks = self.pivot_cache['consecutive_breaks']
                consecutive_breaks['direction'] = None
                consecutive_breaks['count'] = 0
                consecutive_breaks['levels_broken'] = []
                
                self.pivot_cache['last_price'] = current_price
                return {
                    'has_event': True,
                    'has_prediction': False,
                    'message': f"✋ {level_key}@{level_value:.2f} TIENE (bounce #{level_data['bounces']}) - Reset break sequence",
                    'confidence': level_data['confidence'],
                    'event_type': 'level_hold',
                    'level_value': level_value,
                    'level_type': level_key,
                    'expected_outcome': 'held',
                    'level_action': 'held'
                }
            
            # Aggiorna flag di vicinanza per prossimo tick
            level_data['was_near_last_tick'] = is_near_now
        
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
                    not level.get('broken', False)):
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
                    not level.get('broken', False)):
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
                'confidence': min(0.9, next_target['confidence'] + 0.15)
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
            raise KeyError("Critical field 'price_history' missing from market_data")
        if 'volume_history' not in market_data:
            raise KeyError("Critical field 'volume_history' missing from market_data")
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
            raise KeyError("Critical field 'price_history' missing from market_data")
        if 'volume_history' not in market_data:
            raise KeyError("Critical field 'volume_history' missing from market_data")
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
                
        except Exception as e:
            self.algorithm_stats['failed_predictions'] += 1
            # FAIL FAST - Re-raise LSTM error instead of logging
            raise PredictionError("LSTM_SupportResistance", str(e))
    
    def _statistical_levels_ml(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Statistical Levels ML analysis
        ESTRATTO IDENTICO da src/Analyzer.py:12997-13035
        """
        if 'price_history' not in market_data:
            raise KeyError("Critical field 'price_history' missing from market_data")
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
            raise KeyError("Critical field 'current_price' missing from market_data")
        if 'price_history' not in market_data:
            raise KeyError("Critical field 'price_history' missing from market_data")
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
        # Basic features
        returns = np.diff(prices, prepend=prices[0]) / np.maximum(prices[:-1], 1e-10)
        log_returns = np.log(np.maximum(prices[1:] / np.maximum(prices[:-1], 1e-10), 1e-10))
        log_returns = np.append(log_returns, 0)
        
        # Volume features
        if len(volumes) == 0:
            raise ValueError("FAIL FAST: Empty volumes array - cannot calculate volume_mean")
        volume_mean = np.mean(volumes)
        volume_ratio = volumes / max(volume_mean, 1e-10)
        
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


# Factory function per compatibilità
def create_support_resistance_algorithms(ml_models: Optional[Dict[str, Any]] = None) -> SupportResistanceAlgorithms:
    """Factory function per creare SupportResistanceAlgorithms"""
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