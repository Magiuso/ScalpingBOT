#!/usr/bin/env python3
"""
Algorithm Competition System - Self-Improving Model Selection Framework
========================================================================

Advanced algorithm competition system that manages champion selection among multiple ML models.
ESTRATTO IDENTICO da Analyzer.py (linee 2822-10493).

Features:
- ChampionPreserver: Sistema per preservare i migliori champion
- RealityChecker: Sistema per validare che i pattern appresi siano ancora validi  
- EmergencyStopSystem: Sistema di emergency stop per prevenire perdite catastrofiche
- PostErrorReanalyzer: Sistema per rianalizzare e imparare dagli errori
- AlgorithmCompetition: Sistema di competizione tra algoritmi per ogni modello

The AlgorithmCompetition class:
- Manages champion selection among multiple ML models
- Handles performance validation and champion dethronement  
- Implements emergency stop mechanisms
- Tracks reality checking and model validation
- Manages champion preservation and state

Author: ScalpingBOT Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import json
import pickle
import os
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Deque
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# Import the base models and configuration
from .base_models import Prediction, AlgorithmPerformance
from ...shared.enums import ModelType
from ...config.base.base_config import get_analyzer_config, AnalyzerConfig
from .unified_training_metrics import UnifiedTrainingMetrics
# Removed safe_print import - using fail-fast error handling instead


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
        # BIBBIA COMPLIANT: FAIL FAST - no config fallback
        if config is None:
            raise ValueError("FAIL FAST: RealityChecker requires explicit configuration - no fallback allowed")
        self.config = config
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
            # Analizza volatilità - FAIL FAST
            volatilities = []
            for c in error_conditions:
                if 'volatility' not in c:
                    raise KeyError(f"Critical field 'volatility' missing from market conditions snapshot")
                volatilities.append(c['volatility'])
            avg_error_volatility = np.mean(volatilities)
            
            # Analizza trend - FAIL FAST
            trends = []
            for c in error_conditions:
                if 'price_change_5m' not in c:
                    raise KeyError(f"Critical field 'price_change_5m' missing from market conditions snapshot")
                trends.append(c['price_change_5m'])
            
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
        if key not in self.pattern_validity:
            raise KeyError(f"Pattern validity key '{key}' not found - asset/model combination not initialized")
        if algorithm_name not in self.pattern_validity[key]:
            raise KeyError(f"Algorithm '{algorithm_name}' not found in pattern validity for {key}")
        return self.pattern_validity[key][algorithm_name]


class EmergencyStopSystem:
    """Sistema di emergency stop per prevenire perdite catastrofiche - VERSIONE CORRETTA"""
    
    def __init__(self, logger, config: Optional[AnalyzerConfig] = None):
        self.logger = logger
        # BIBBIA COMPLIANT: FAIL FAST - no config fallback
        if config is None:
            raise ValueError("FAIL FAST: Component requires explicit configuration - no fallback allowed")
        self.config = config
        
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=getattr(self.config, 'performance_window_size', 100) or 100))  # 🔧 FIXED
        
        # 🔧 Usa configurazione per stop triggers invece di hardcoded
        self.stop_triggers = self.config.get_emergency_stop_triggers()
        
        self.stopped_algorithms: Set[str] = set()
        
        # Event buffer per slave module integration - CORRETTO: USARE DEQUE
        self._emergency_events_buffer: deque = deque(maxlen=100)
    
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
        """Attiva l'emergency stop - CORRETTO"""
        algorithm.emergency_stop_triggered = True
        self.stopped_algorithms.add(algorithm_key)
        
        # Store emergency stop event
        self._store_emergency_event('emergency_stop_triggered', {
            'algorithm_key': algorithm_key,
            'triggers': triggers,
            'final_score': algorithm.final_score,
            'algorithm_data': {
                'name': algorithm.name,  # CORRETTO: algorithm.name invece di algorithm.algorithm_name
                'model_type': algorithm.model_type.value if hasattr(algorithm.model_type, 'value') else str(algorithm.model_type)
            }
        })
        
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
        try:
            with open(self.logger.base_path / 'emergency_stops.json', 'a') as f:
                f.write(json.dumps(stop_event) + '\n')
        except Exception as e:
            # Fallback silenzioso se non riesce a scrivere il file
            self._store_emergency_event('file_write_error', {
                'error': str(e),
                'event_type': 'emergency_stop_file_write'
            })
    
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
        """Store emergency system events in memory for future processing by slave module - CORRETTO"""
        
        event_entry = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'data': event_data
        }
        
        # Buffer gestito automaticamente da deque con maxlen=100
        self._emergency_events_buffer.append(event_entry)

    def get_all_events_for_slave(self) -> Dict[str, List[Dict]]:
        """Get all accumulated events for slave module processing"""
        events = {}
        
        # Emergency events - CORRETTO: list() per convertire deque
        events['emergency_events'] = list(self._emergency_events_buffer)
        
        return events

    def clear_events_buffer(self, event_types: Optional[List[str]] = None) -> None:
        """Clear event buffers after slave module processing"""
        if event_types is None:
            # Clear all buffers
            self._emergency_events_buffer.clear()
        else:
            # Clear specific buffers
            for event_type in event_types:
                if event_type == 'emergency_events':
                    self._emergency_events_buffer.clear()
    
    def get_stop_status(self, algorithm_key: str) -> bool:
        """Verifica se un algoritmo è in emergency stop"""
        return algorithm_key in self.stopped_algorithms
    
    def get_stopped_algorithms(self) -> Set[str]:
        """Ottieni tutti gli algoritmi attualmente fermati"""
        return self.stopped_algorithms.copy()
    
    def force_stop_algorithm(self, algorithm_key: str, reason: str = "manual_override"):
        """Forza emergency stop di un algoritmo (per test o override manuale)"""
        self.stopped_algorithms.add(algorithm_key)
        
        self._store_emergency_event('manual_emergency_stop', {
            'algorithm_key': algorithm_key,
            'reason': reason,
            'timestamp': datetime.now(),
            'manual_override': True
        })
    
    def get_emergency_statistics(self) -> Dict[str, Any]:
        """Ottieni statistiche sui emergency stops"""
        
        # Conta eventi per tipo - FAIL FAST
        event_counts = {}
        for event in self._emergency_events_buffer:
            if 'event_type' not in event:
                raise KeyError("Critical field 'event_type' missing from emergency event")
            event_type = event['event_type']
            if event_type not in event_counts:
                event_counts[event_type] = 0
            event_counts[event_type] += 1
        
        # Calcola utilizzo buffer con controllo sicurezza
        buffer_size = len(self._emergency_events_buffer)
        max_size = self._emergency_events_buffer.maxlen
        buffer_utilization = (buffer_size / max_size) * 100 if max_size is not None and max_size > 0 else 0.0
        
        return {
            'total_stopped_algorithms': len(self.stopped_algorithms),
            'stopped_algorithms': list(self.stopped_algorithms),
            'total_events_in_buffer': buffer_size,
            'event_type_counts': event_counts,
            'buffer_utilization': buffer_utilization,
            'buffer_max_size': max_size
        }


class PostErrorReanalyzer:
    """Sistema per rianalizzare e imparare dagli errori"""
    
    def __init__(self, logger):
        self.logger = logger
        self.error_patterns_db: Dict[str, List[Dict]] = defaultdict(list)
        self.lessons_learned: Dict[str, Dict[str, Any]] = {}
        # 🚨 CIRCUIT BREAKER: Ridotto da 1000 a 50 per prevenire loop infiniti
        self.reanalysis_queue: deque = deque(maxlen=50)
        # Tracker per algoritmi permanentemente disabilitati
        self.permanently_disabled_algorithms: Set[str] = set()
        
    def add_to_reanalysis_queue(self, failed_prediction: Prediction, 
                              actual_outcome: Dict[str, Any],
                              market_data_snapshot: Dict[str, Any]):
        """Aggiunge una predizione fallita alla coda di rianalisi - CON CIRCUIT BREAKER"""
        
        algorithm_key = f"{failed_prediction.model_type.value}_{failed_prediction.algorithm_name}"
        
        # 🚨 CHECK SE ALGORITMO È GIÀ DISABILITATO PERMANENTEMENTE
        if algorithm_key in self.permanently_disabled_algorithms:
            # Algorithm permanently disabled - FAIL FAST instead of logging
            raise RuntimeError(f"Algorithm {failed_prediction.algorithm_name} is permanently disabled")
            return
        
        self.reanalysis_queue.append({
            'prediction': failed_prediction,
            'actual_outcome': actual_outcome,
            'market_data': market_data_snapshot,
            'added_timestamp': datetime.now()
        })
        
        # 🔥 CIRCUIT BREAKER LOGIC: Conta errori consecutivi
        recent_errors = [
            item for item in self.reanalysis_queue
            if (item['prediction'].model_type == failed_prediction.model_type and 
                item['prediction'].algorithm_name == failed_prediction.algorithm_name)
        ]
        
        # Se più del 90% della queue è lo stesso algoritmo = DISABLE PERMANENTEMENTE
        if len(recent_errors) > 45:  # 90% di 50
            self.permanently_disabled_algorithms.add(algorithm_key)
            # FAIL FAST - Permanent disable should stop execution
            raise RuntimeError(f"PERMANENT DISABLE: {failed_prediction.algorithm_name} - too many consecutive failures")
            
            # Aggiorna lessons learned con disable permanente
            self.lessons_learned[algorithm_key] = {
                'timestamp': datetime.now(),
                'errors_analyzed': len(recent_errors),
                'status': 'PERMANENTLY_DISABLED',
                'reason': 'excessive_failure_rate',
                'disable_threshold_reached': True
            }
    
    def perform_reanalysis(self, asset: str, model_type: ModelType, 
                         algorithm_name: str) -> Dict[str, Any]:
        """Esegue rianalisi sugli errori recenti - CON CIRCUIT BREAKER CHECK"""
        
        key = f"{asset}_{model_type.value}_{algorithm_name}"
        
        # 🚨 CHECK SE ALGORITMO È DISABILITATO PERMANENTEMENTE
        if key in self.permanently_disabled_algorithms:
            return {
                'status': 'permanently_disabled',
                'errors_analyzed': 0,
                'reason': 'Algorithm disabled due to excessive failures'
            }
        
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
        
        # Analisi condizioni di mercato - FAIL FAST
        market_conditions = []
        for e in errors:
            if 'market_data' not in e:
                continue  # Skip events without market data (optional field)
            market_conditions.append(e['market_data'])
        
        if market_conditions:
            volatilities = []
            for m in market_conditions:
                if 'volatility' not in m:
                    raise KeyError("Critical field 'volatility' missing from market data")
                volatilities.append(m['volatility'])
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
        
        market_states = []
        for e in errors:
            if 'market_data' not in e:
                continue  # Skip events without market data (optional field)
            market_states.append(e['market_data'])
        
        if not market_states:
            return conditions
        
        # Analizza volatilità - FAIL FAST
        volatilities = []
        for m in market_states:
            if 'volatility' not in m:
                raise KeyError("Critical field 'volatility' missing from market data")
            volatilities.append(m['volatility'])
        if volatilities:
            high_vol_errors = sum(1 for v in volatilities if v > 0.015)
            if high_vol_errors > len(errors) * 0.5:
                conditions.append({
                    'condition': 'high_volatility',
                    'threshold': 0.015,
                    'frequency': high_vol_errors / len(errors),
                    'recommendation': 'Reduce confidence during high volatility periods'
                })
        
        # Analizza trend - FAIL FAST
        trends = []
        for m in market_states:
            if 'price_change_5m' not in m:
                raise KeyError("Critical field 'price_change_5m' missing from market data")
            trends.append(m['price_change_5m'])
        if trends:
            strong_trend_errors = sum(1 for t in trends if abs(t) > 0.01)
            if strong_trend_errors > len(errors) * 0.4:
                conditions.append({
                    'condition': 'strong_trend',
                    'threshold': 0.01,
                    'frequency': strong_trend_errors / len(errors),
                    'recommendation': 'Adjust predictions during strong directional moves'
                })
        
        # Analizza volume - FAIL FAST
        volumes = []
        for m in market_states:
            if 'avg_volume' not in m:
                raise KeyError("Critical field 'avg_volume' missing from market data")
            volumes.append(m['avg_volume'])
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
        """Applica le lezioni apprese a una nuova predizione - IMPLEMENTAZIONE COMPLETA"""
        
        key = f"{prediction.model_type.value}_{prediction.algorithm_name}"
        lessons = self.lessons_learned.get(key)
        
        if not lessons:
            return {'adjusted': False, 'confidence_multiplier': 1.0, 'should_skip': False}
        
        confidence_multiplier = 1.0
        adjustments = []
        should_skip = False
        
        # 🚨 CIRCUIT BREAKER: Se troppi errori recenti, SKIPPA l'algoritmo - FAIL FAST
        if 'errors_analyzed' not in lessons:
            raise KeyError(f"Critical field 'errors_analyzed' missing from lessons for {key}")
        if lessons['errors_analyzed'] > 50:
            if 'patterns_found' not in lessons:
                raise KeyError(f"Critical field 'patterns_found' missing from lessons for {key}")
            total_patterns = lessons['patterns_found']
            if isinstance(total_patterns, dict):
                if 'patterns' not in total_patterns:
                    raise KeyError(f"Critical field 'patterns' missing from patterns_found for {key}")
                high_severity_patterns = sum(1 for p in total_patterns['patterns'] 
                                           if 'severity' in p and p['severity'] == 'high')
            else:
                high_severity_patterns = total_patterns
            
            if high_severity_patterns > 3:
                should_skip = True
                adjustments.append('circuit_breaker_triggered')
                # FAIL FAST - Circuit breaker should stop execution
                raise RuntimeError(f"CIRCUIT BREAKER: {prediction.algorithm_name} - too many high severity patterns")
        
        # Applica aggiustamenti basati sulle lezioni - FAIL FAST
        if 'lessons' not in lessons:
            raise KeyError(f"Critical field 'lessons' missing from lessons for {key}")
        for lesson in lessons['lessons']:
            if lesson['type'] == 'temporal_adjustment':
                # Aggiusta per ora del giorno - FAIL FAST
                current_hour = prediction.timestamp.hour
                if 'problematic_hours' not in lesson:
                    raise KeyError(f"Critical field 'problematic_hours' missing from temporal_adjustment lesson")
                problematic_hours = lesson['problematic_hours']
                if current_hour in problematic_hours:
                    confidence_multiplier *= 0.5
                    adjustments.append(f'temporal_adjustment_hour_{current_hour}')
                
            elif lesson['type'] == 'bias_correction':
                # Applica correzione bias - FAIL FAST
                if 'bias_factor' not in lesson:
                    raise KeyError(f"Critical field 'bias_factor' missing from bias_correction lesson")
                bias_factor = lesson['bias_factor']
                if abs(bias_factor) > 0.1:
                    confidence_multiplier *= (1.0 - abs(bias_factor))
                    adjustments.append(f'bias_correction_{bias_factor:.2f}')
            
            elif lesson['type'] == 'condition_awareness':
                # Controlla condizioni di mercato problematiche - FAIL FAST
                if 'volatility' not in market_conditions:
                    raise KeyError("Critical field 'volatility' missing from market conditions")
                current_volatility = market_conditions['volatility']
                if 'problematic_volatility_threshold' not in lesson:
                    raise KeyError("Critical field 'problematic_volatility_threshold' missing from condition_awareness lesson")
                problematic_volatility = lesson['problematic_volatility_threshold']
                
                if current_volatility > problematic_volatility:
                    confidence_multiplier *= 0.3  # Riduce drasticamente la confidence
                    adjustments.append('high_volatility_penalty')
                    
            elif lesson['type'] == 'systematic_failure':
                # Se c'è un fallimento sistematico, skippa completamente - FAIL FAST
                if 'failure_rate' not in lesson:
                    raise KeyError("Critical field 'failure_rate' missing from systematic_failure lesson")
                failure_rate = lesson['failure_rate']
                if failure_rate > 0.8:  # >80% failure rate
                    should_skip = True
                    adjustments.append('systematic_failure_skip')
        
        # 🔥 ADAPTIVE CONFIDENCE REDUCTION basata sui pattern - FAIL FAST
        if 'patterns_found' not in lessons:
            raise KeyError(f"Critical field 'patterns_found' missing from lessons for {key}")
        patterns = lessons['patterns_found']
        if isinstance(patterns, dict) and 'patterns' in patterns:
            for pattern in patterns['patterns']:
                if 'type' not in pattern:
                    raise KeyError("Critical field 'type' missing from pattern")
                if pattern['type'] == 'systematic_bias':
                    confidence_multiplier *= 0.6
                    adjustments.append('systematic_bias_penalty')
                elif 'type' in pattern and pattern['type'] == 'market_condition' and 'severity' in pattern and pattern['severity'] == 'high':
                    confidence_multiplier *= 0.4
                    adjustments.append('market_condition_penalty')
        
        # 🛡️ MINIMUM CONFIDENCE THRESHOLD
        if confidence_multiplier < 0.1:
            should_skip = True
            adjustments.append('minimum_confidence_threshold')
        
        return {
            'adjusted': len(adjustments) > 0,
            'confidence_multiplier': confidence_multiplier,
            'adjustments_applied': adjustments,
            'should_skip': should_skip
        }


class AlgorithmCompetition:
    """Sistema di competizione tra algoritmi per ogni modello con funzionalità avanzate - VERSIONE PULITA"""
    
    def __init__(self, model_type: ModelType, asset: str, logger,
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
        self.performance_window = deque(maxlen=getattr(self.config, 'performance_window_size', 100) or 100)  # 🔧 FIXED
        self.last_reality_check = datetime.now()
        self.reality_check_interval = timedelta(hours=self.config.reality_check_interval_hours)  # 🔧 CHANGED
        
        
    def register_algorithm(self, name: str) -> None:
        """Registra un nuovo algoritmo nella competizione - NO AUTO-CHAMPION"""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Algorithm name must be non-empty string")
        
        if name in self.algorithms:
            raise ValueError(f"Algorithm '{name}' already registered")
        
        self.algorithms[name] = AlgorithmPerformance(
            name=name,
            model_type=self.model_type
        )
        
        # BIBBIA COMPLIANCE: NO auto-champion - deve guadagnarselo
        # Champion rimane None finché un algoritmo non dimostra il suo valore
        
        # 🧹 PULITO: Sostituito logger con event storage
        self._store_system_event('algorithm_registered', {
            'algorithm_name': name,
            'asset': self.asset,
            'model_type': self.model_type.value,
            'has_champion': self.champion is not None,
            'timestamp': datetime.now()
        })
    
    def set_training_performance(self, name: str, training_data: Dict[str, Any], 
                               algorithm_type: Optional[str] = None) -> None:
        """
        Imposta performance dall'addestramento usando UnifiedTrainingMetrics
        
        Args:
            name: Nome algoritmo
            training_data: Dati di training/evaluation
            algorithm_type: Tipo algoritmo ('neural_network', 'classical_ml', 'mathematical', 'auto')
            
        Raises:
            ValueError: Se algoritmo non registrato o training invalido
        """
        if name not in self.algorithms:
            raise ValueError(f"Algorithm '{name}' not registered. Call register_algorithm() first.")
        
        if not isinstance(training_data, dict):
            raise TypeError("training_data must be dict")
        
        try:
            # Auto-detect se non specificato
            if algorithm_type is None or algorithm_type == 'auto':
                normalized_metrics = UnifiedTrainingMetrics.auto_normalize_metrics(training_data, name)
            else:
                # Usa metodo specifico basato sul tipo
                if algorithm_type == 'neural_network':
                    normalized_metrics = UnifiedTrainingMetrics.normalize_neural_network_metrics(training_data)
                elif algorithm_type == 'classical_ml':
                    normalized_metrics = UnifiedTrainingMetrics.normalize_classical_ml_metrics(training_data)
                elif algorithm_type == 'mathematical':
                    normalized_metrics = UnifiedTrainingMetrics.normalize_mathematical_metrics(training_data)
                # ⚠️ pivot_points case RIMOSSO - PivotPoints_Classic ora modulare
                else:
                    raise ValueError(f"Unknown algorithm_type: {algorithm_type}")
            
        except Exception as e:
            # FAIL FAST: Se normalizzazione fallisce, è un errore grave
            raise RuntimeError(f"Failed to normalize training metrics for {name}: {e}")
        
        # Imposta metriche normalizzate sull'algoritmo
        algorithm = self.algorithms[name]
        algorithm.training_score = normalized_metrics['performance_score']
        algorithm.training_confidence = normalized_metrics['confidence']
        algorithm.training_reliability = normalized_metrics['reliability']
        algorithm.training_completed = True
        
        # Store raw metrics per debugging (opzionale)
        if algorithm.raw_training_metrics is None:
            algorithm.raw_training_metrics = {}
        algorithm.raw_training_metrics.update(normalized_metrics)
        
        # Log evento
        self._store_system_event('training_performance_set', {
            'algorithm_name': name,
            'performance_score': normalized_metrics['performance_score'],
            'confidence': normalized_metrics['confidence'],
            'reliability': normalized_metrics['reliability'],
            'algorithm_type': normalized_metrics['algorithm_type'],
            'timestamp': datetime.now()
        })
        
        # Valuta se può diventare champion
        self._update_champion_from_training()
    
    def _update_champion_from_training(self) -> None:
        """Aggiorna champion basandosi su metriche di training - BIBBIA COMPLIANT"""
        if not self.algorithms:
            return
        
        # Trova il migliore algoritmo che ha completato il training
        best_candidate = None
        best_training_score = self.champion_threshold  # Es: 70.0
        
        for name, algorithm in self.algorithms.items():
            if (hasattr(algorithm, 'training_completed') and 
                algorithm.training_completed and
                hasattr(algorithm, 'training_score') and
                algorithm.training_score > best_training_score):
                
                # Verifica reliability minima
                reliability = getattr(algorithm, 'training_reliability', 0.0)
                if reliability >= 0.5:  # Reliability minima 50%
                    best_candidate = name
                    best_training_score = algorithm.training_score
        
        # Aggiorna champion se trovato un candidato migliore
        if best_candidate:
            old_champion = self.champion
            
            # Detronizza vecchio champion se esiste
            if self.champion and self.champion in self.algorithms:
                self.algorithms[self.champion].is_champion = False
            
            # Nuovo champion
            self.champion = best_candidate
            self.algorithms[best_candidate].is_champion = True
            
            # Log cambio champion
            self._store_system_event('champion_selected_from_training', {
                'old_champion': old_champion,
                'new_champion': best_candidate,
                'training_score': best_training_score,
                'confidence': getattr(self.algorithms[best_candidate], 'training_confidence', 0.0),
                'reliability': getattr(self.algorithms[best_candidate], 'training_reliability', 0.0),
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
        
        # 🚨 APPLICA LEZIONI APPRESE - IMPLEMENTAZIONE COMPLETA
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
        
        # 🛡️ CHECK CIRCUIT BREAKER - SE LESSON LEARNED DICE DI SKIPPARE - FAIL FAST
        if 'should_skip' not in lessons_adjustment:
            raise KeyError("Critical field 'should_skip' missing from lessons_adjustment")
        if lessons_adjustment['should_skip']:
            if 'adjustments_applied' not in lessons_adjustment:
                raise KeyError("Critical field 'adjustments_applied' missing from lessons_adjustment")
            skip_reason = ', '.join(lessons_adjustment['adjustments_applied'])
            # FAIL FAST - Skip conditions should raise errors
            raise RuntimeError(f"SKIPPING {algorithm_name} - Lesson Learned: {skip_reason}")
            return f"SKIPPED_BY_LESSONS: {skip_reason}"
        
        # Aggiusta confidence basato sulle lezioni
        adjusted_confidence = confidence * lessons_adjustment['confidence_multiplier']
        
        # 🔒 MINIMUM CONFIDENCE CHECK
        if adjusted_confidence < 0.1:
            # FAIL FAST - Low confidence should raise error
            raise RuntimeError(f"SKIPPING {algorithm_name} - Adjusted confidence too low: {adjusted_confidence:.3f}")
            return f"SKIPPED_LOW_CONFIDENCE: {adjusted_confidence:.3f}"
        
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
        
        # Analisi specifica per tipo di modello - FAIL FAST
        if self.model_type == ModelType.SUPPORT_RESISTANCE:
            if 'support_levels' not in prediction.prediction_data:
                raise KeyError("Critical field 'support_levels' missing from prediction_data for SUPPORT_RESISTANCE model")
            if 'resistance_levels' not in prediction.prediction_data:
                raise KeyError("Critical field 'resistance_levels' missing from prediction_data for SUPPORT_RESISTANCE model")
            predicted_support = prediction.prediction_data['support_levels']
            predicted_resistance = prediction.prediction_data['resistance_levels']
            
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
                if 'pattern' not in prediction.prediction_data:
                    raise KeyError("Critical field 'pattern' missing from prediction_data for PATTERN_RECOGNITION model")
                predicted_pattern = prediction.prediction_data['pattern']
                error_analysis['error_types'].append('pattern_not_materialized')
                error_analysis['patterns'].append(f'false_{predicted_pattern}')
                
                # Analizza se c'era un pattern diverso - FAIL FAST
                if 'actual_pattern' in market_data:
                    error_analysis['specific_failures'].append({
                        'type': 'wrong_pattern',
                        'predicted': predicted_pattern,
                        'actual': market_data['actual_pattern']
                    })
        
        elif self.model_type == ModelType.BIAS_DETECTION:
            if 'directional_bias' not in prediction.prediction_data:
                raise KeyError("Critical field 'directional_bias' missing from prediction_data for BIAS_DETECTION model")
            directional_bias = prediction.prediction_data['directional_bias']
            if 'direction' not in directional_bias:
                raise KeyError("Critical field 'direction' missing from directional_bias for BIAS_DETECTION model")
            predicted_direction = directional_bias['direction']
            if 'actual_direction' in market_data:
                if predicted_direction != market_data['actual_direction']:
                    error_analysis['error_types'].append('wrong_direction')
                    error_analysis['patterns'].append('directional_miss')
        
        # Analizza condizioni di mercato durante l'errore - FAIL FAST
        if prediction.market_conditions_snapshot:
            if 'volatility' not in prediction.market_conditions_snapshot:
                raise KeyError("Critical field 'volatility' missing from market_conditions_snapshot")
            if 'avg_volume' not in prediction.market_conditions_snapshot:
                raise KeyError("Critical field 'avg_volume' missing from market_conditions_snapshot")
            volatility = prediction.market_conditions_snapshot['volatility']
            volume = prediction.market_conditions_snapshot['avg_volume']
            
            # Calculate typical volume reference - FAIL FAST
            if 'typical_volume' not in market_data:
                raise KeyError("Critical field 'typical_volume' missing from market_data for volume comparison")
            typical_volume = market_data['typical_volume']
            
            error_analysis['market_conditions'] = {
                'volatility': volatility,
                'volume': volume,
                'high_volatility': volatility > 0.02,
                'low_volume': volume < typical_volume * 0.5
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
            # Controlla se i livelli predetti sono stati rispettati - FAIL FAST
            if 'support_levels' not in prediction.prediction_data:
                raise KeyError("Critical field 'support_levels' missing from prediction_data")
            if 'resistance_levels' not in prediction.prediction_data:
                raise KeyError("Critical field 'resistance_levels' missing from prediction_data")
            if 'current_price' not in market_data:
                raise KeyError("Critical field 'current_price' missing from market_data")
            predicted_support = prediction.prediction_data['support_levels']
            predicted_resistance = prediction.prediction_data['resistance_levels']
            current_price = market_data['current_price']
            
            outcome['support_accuracy'] = self._check_level_accuracy(
                predicted_support, current_price, market_data, 'support'
            )
            outcome['resistance_accuracy'] = self._check_level_accuracy(
                predicted_resistance, current_price, market_data, 'resistance'
            )
            
        elif self.model_type == ModelType.PATTERN_RECOGNITION:
            # Controlla se il pattern si è manifestato - FAIL FAST
            if 'pattern' not in prediction.prediction_data:
                raise KeyError("Critical field 'pattern' missing from prediction_data")
            predicted_pattern = prediction.prediction_data['pattern']
            if 'direction' not in prediction.prediction_data:
                raise KeyError("Critical field 'direction' missing from prediction_data")
            predicted_direction = prediction.prediction_data['direction']
            
            outcome['pattern_occurred'] = self._check_pattern_occurrence(predicted_pattern, market_data)
            outcome['direction_correct'] = self._check_direction_accuracy(predicted_direction, market_data)
            
        elif self.model_type == ModelType.BIAS_DETECTION:
            # Controlla se il bias era corretto - FAIL FAST
            if 'directional_bias' not in prediction.prediction_data:
                raise KeyError("Critical field 'directional_bias' missing from prediction_data")
            directional_bias = prediction.prediction_data['directional_bias']
            if 'direction' not in directional_bias:
                raise KeyError("Critical field 'direction' missing from directional_bias")
            predicted_bias = directional_bias['direction']
            actual_direction = self._determine_actual_direction(market_data)
            
            outcome['bias_correct'] = predicted_bias == actual_direction
            outcome['actual_direction'] = actual_direction
            
        elif self.model_type == ModelType.TREND_ANALYSIS:
            # Controlla accuratezza del trend - FAIL FAST
            if 'trend_direction' not in prediction.prediction_data:
                raise KeyError("Critical field 'trend_direction' missing from prediction_data")
            predicted_trend = prediction.prediction_data['trend_direction']
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
        
        if 'price_history' not in market_data:
            raise KeyError("Critical field 'price_history' missing from market_data")
        price_history = market_data['price_history']
        accuracy_scores = []
        
        for level in levels:
            # Controlla se il prezzo ha toccato o rispettato il livello
            level_tolerance = level * 0.001  # 0.1% tolerance
            
            touches = 0
            respects = 0
            
            recent_prices = price_history[-20:]  # Ultimi 20 tick
            for i, price in enumerate(recent_prices):
                if abs(price - level) <= level_tolerance:
                    touches += 1
                    
                    # Controlla se il livello ha tenuto (se c'è un tick successivo)
                    if i < len(recent_prices) - 1:  # Verifica che non sia l'ultimo elemento
                        next_price = recent_prices[i + 1]
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
        if 'price_history' not in market_data:
            raise KeyError("Critical field 'price_history' missing from market_data")
        price_history = market_data['price_history']
        
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
        if 'price_history' not in market_data:
            raise KeyError("Critical field 'price_history' missing from market_data")
        price_history = market_data['price_history']
        
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
        if 'price_change_5m' not in market_data:
            raise KeyError("Critical field 'price_change_5m' missing from market_data")
        price_change = market_data['price_change_5m']
        
        if price_change > 0.003:
            return 'bullish'
        elif price_change < -0.003:
            return 'bearish'
        else:
            return 'neutral'
    
    def _determine_actual_trend(self, market_data: Dict[str, Any]) -> str:
        """Determina il trend effettivo"""
        if 'price_history' not in market_data:
            raise KeyError("Critical field 'price_history' missing from market_data")
        prices = market_data['price_history']
        
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
        if 'trend_strength' not in prediction_data:
            raise KeyError("Critical field 'trend_strength' missing from prediction_data")
        predicted_strength = prediction_data['trend_strength']
        
        # Calcola forza effettiva
        if 'price_history' not in market_data:
            raise KeyError("Critical field 'price_history' missing from market_data")
        prices = market_data['price_history']
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
        # FIXED: Better champion selection logic that considers overfitting
        improvement_factor = 1.10  # FIXED: Was 1.20 (20%), now 10% improvement needed
        
        # Check if challenger is better AND not overfitting
        if best_challenger is not None:
            challenger_algorithm = self.algorithms.get(best_challenger)
            is_tree_model = 'RandomForest' in best_challenger or 'GradientBoosting' in best_challenger
            
            # For tree models, check if they have reasonable validation performance and no overfitting
            can_become_champion = (
                challenger_algorithm is not None and
                best_challenger_score > current_champion_score * improvement_factor and
                challenger_algorithm.total_predictions >= self.config.min_predictions_for_champion and
                # Additional check for tree models to prevent overfitting champions
                (not is_tree_model or (
                    challenger_algorithm.accuracy_rate > 0.55 and  # Tree models need decent accuracy
                    not getattr(challenger_algorithm, 'overfitting_detected', False)  # No overfitting detected
                ))
            )
        else:
            can_become_champion = False
        
        if can_become_champion and best_challenger is not None:
            # Preserva il vecchio champion se ha performato bene
            if current_champion_score > 70 and current_champion.total_predictions > 100:
                self._preserve_champion(current_champion)
            
            # Nuovo champion!
            old_champion = self.champion or "None"
            old_score = current_champion_score
            
            current_champion.is_champion = False
            self.algorithms[best_challenger].is_champion = True
            self.champion = best_challenger
            
            # Get the new champion algorithm (we know it exists from can_become_champion check)
            new_champion_algorithm = self.algorithms[best_challenger]
            
            # Calcola ragione del cambio
            reason = self._determine_champion_change_reason(
                current_champion, new_champion_algorithm
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
            events['system_events'] = list(self._system_events_buffer)
        
        # Prediction events
        if hasattr(self, '_prediction_events_buffer'):
            events['prediction_events'] = list(self._prediction_events_buffer)
        
        # Validation metrics
        if hasattr(self, '_validation_metrics_buffer'):
            events['validation_metrics'] = list(self._validation_metrics_buffer)
        
        # Performance metrics
        if hasattr(self, '_performance_metrics_buffer'):
            events['performance_metrics'] = list(self._performance_metrics_buffer)
        
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