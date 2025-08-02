#!/usr/bin/env python3
"""
Base ML Models - Core Model Types and Prediction Structures
===========================================================

Definizioni base per i modelli ML del sistema ScalpingBOT.
ESTRATTO IDENTICO da Analyzer.py (linee 2709-2808).

Features:
- ModelType enum per tutti i tipi di modelli (6 tipi)
- OptimizationProfile per profili di training
- Prediction dataclass per predizioni strutturate
- AlgorithmPerformance per tracking performance con decay

Author: ScalpingBOT Team
Version: 1.0.0
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from typing import Dict, Any, Optional, Deque

# Import shared enums instead of duplicating - CONSOLIDATED
from src.shared.enums import ModelType, OptimizationProfile


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
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        if recent_avg > older_avg * 1.1:
            return "improving"
        elif recent_avg < older_avg * 0.9:
            return "declining"
        else:
            return "stable"


# Factory function per compatibilità
def create_prediction(model_type: ModelType, algorithm_name: str, 
                     prediction_data: Dict[str, Any], confidence: float,
                     validation_criteria: Dict[str, Any]) -> Prediction:
    """Factory function per creare una Prediction"""
    import uuid
    return Prediction(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        model_type=model_type,
        algorithm_name=algorithm_name,
        prediction_data=prediction_data,
        confidence=confidence,
        validation_time=datetime.now(),
        validation_criteria=validation_criteria
    )


def create_algorithm_performance(name: str, model_type: ModelType) -> AlgorithmPerformance:
    """Factory function per creare un AlgorithmPerformance"""
    return AlgorithmPerformance(
        name=name,
        model_type=model_type
    )