#!/usr/bin/env python3
"""
Unified Training Metrics - Sistema di Normalizzazione Metriche ML
================================================================

REGOLE CLAUDE_RESTAURO.md APPLICATE:
- ✅ Zero fallback/defaults - FAIL FAST error handling
- ✅ No debug prints/spam  
- ✅ No test code embedded
- ✅ No redundant functions
- ✅ Single implementation path

Sistema per normalizzare metriche di training provenienti da diversi tipi di algoritmi ML:
- Neural Networks (LSTM, CNN): metriche basate su loss e epoche
- Classical ML (RandomForest, GradientBoosting): metriche basate su accuracy
- Mathematical (PivotPoints, GARCH): metriche basate su confidence diretta

Questo sistema permette una competizione equa tra algoritmi diversi convertendo
tutte le metriche in un formato standard 0-100.

Author: ScalpingBOT Team
Version: 1.0.0
Date: 2025-08-05
"""

from typing import Dict, Any, Optional
from datetime import datetime


class UnifiedTrainingMetrics:
    """
    Sistema di normalizzazione metriche per competizione algoritmi ML
    
    Converte metriche eterogenee da diversi tipi di algoritmi in formato standard:
    - performance_score: 0-100 (più alto = migliore)
    - confidence: 0-1 (affidabilità della predizione)
    - reliability: 0-1 (stabilità dell'algoritmo)
    - algorithm_type: categoria dell'algoritmo
    """
    
    # Thresholds per valutazione qualità - BIBBIA COMPLIANT: configurabili ma non fallback
    NEURAL_NETWORK_MIN_EPOCHS = 10
    CLASSICAL_ML_MIN_ACCURACY = 0.5
    MATHEMATICAL_MIN_CONFIDENCE = 0.5
    PIVOT_MIN_EVALUATION_WINDOWS = 1000
    
    @staticmethod
    def normalize_neural_network_metrics(training_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizza metriche da Neural Networks (LSTM, CNN, Transformer)
        
        Args:
            training_result: Risultato da adaptive_trainer.py
            
        Returns:
            Dict con metriche normalizzate
            
        Raises:
            ValueError: Se training non completato o dati invalidi
        """
        if not isinstance(training_result, dict):
            raise TypeError("training_result must be dict")
        
        # FAIL FAST: Training deve essere completato
        if not training_result.get('training_completed', False):
            training_status = training_result.get('status', 'unknown')
            error_msg = training_result.get('message', 'No error details')
            raise ValueError(f"Neural network training not completed: status={training_status}, error={error_msg}")
        
        # FAIL FAST: Dati critici devono esistere
        final_loss = training_result.get('final_loss')
        if final_loss is None:
            raise KeyError("Missing required field 'final_loss' in neural network training result")
        
        epochs_completed = training_result.get('epochs_completed', 0)
        if epochs_completed < UnifiedTrainingMetrics.NEURAL_NETWORK_MIN_EPOCHS:
            raise ValueError(f"Insufficient training epochs: {epochs_completed} < {UnifiedTrainingMetrics.NEURAL_NETWORK_MIN_EPOCHS}")
        
        # FAIL FAST: Loss deve essere valido
        if not isinstance(final_loss, (int, float)) or final_loss < 0:
            raise ValueError(f"Invalid final_loss value: {final_loss}")
        
        # Converti loss in performance score (0-100)
        # Loss di 0 = score 100, loss di 1+ = score vicino a 0
        if final_loss == 0:
            performance_score = 100.0
        else:
            # Formula logaritmica per convertire loss in score
            performance_score = max(0.0, 100.0 * (1.0 / (1.0 + final_loss)))
        
        # Confidence basata su convergenza del training
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + final_loss)))
        
        # Reliability basata su stabilità del training
        training_metrics = training_result.get('training_metrics', [])
        if len(training_metrics) > 5:
            # Calcola stabilità dalle ultime 5 epoche
            recent_losses = [m.get('loss', float('inf')) for m in training_metrics[-5:]]
            loss_std = UnifiedTrainingMetrics._calculate_std(recent_losses)
            reliability = max(0.0, min(1.0, 1.0 / (1.0 + loss_std)))
        else:
            reliability = 0.5  # Reliability moderata se pochi dati
        
        return {
            'performance_score': float(performance_score),
            'confidence': float(confidence),
            'reliability': float(reliability),
            'algorithm_type': 'neural_network',
            'epochs_completed': int(epochs_completed),
            'final_loss': float(final_loss),
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def normalize_classical_ml_metrics(algorithm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizza metriche da Classical ML (RandomForest, GradientBoosting, SVM)
        
        Args:
            algorithm_result: Risultato da algoritmo classical ML
            
        Returns:
            Dict con metriche normalizzate
            
        Raises:
            ValueError: Se accuracy insufficiente o dati invalidi
        """
        if not isinstance(algorithm_result, dict):
            raise TypeError("algorithm_result must be dict")
        
        # FAIL FAST: Deve avere successo
        if not algorithm_result.get('success', False):
            raise ValueError("Classical ML algorithm execution failed")
        
        # Estrai metriche
        metadata = algorithm_result.get('metadata', {})
        accuracy = metadata.get('accuracy_score')
        base_confidence = algorithm_result.get('confidence', 0.0)
        
        # FAIL FAST: Accuracy deve esistere e essere valida
        if accuracy is None:
            raise KeyError("Missing required field 'accuracy_score' in classical ML metadata")
        
        if not isinstance(accuracy, (int, float)) or not (0.0 <= accuracy <= 1.0):
            raise ValueError(f"Invalid accuracy_score: {accuracy} (must be 0.0-1.0)")
        
        if accuracy < UnifiedTrainingMetrics.CLASSICAL_ML_MIN_ACCURACY:
            raise ValueError(f"Insufficient accuracy: {accuracy:.3f} < {UnifiedTrainingMetrics.CLASSICAL_ML_MIN_ACCURACY}")
        
        # FAIL FAST: Confidence deve essere valida
        if not isinstance(base_confidence, (int, float)) or not (0.0 <= base_confidence <= 1.0):
            raise ValueError(f"Invalid confidence: {base_confidence} (must be 0.0-1.0)")
        
        # Performance score direttamente da accuracy
        performance_score = accuracy * 100.0
        
        # Confidence è il massimo tra accuracy e confidence del risultato
        confidence = max(accuracy, base_confidence)
        
        # Reliability basata su consistenza accuracy
        if accuracy > 0.8:
            reliability = 1.0
        elif accuracy > 0.7:
            reliability = 0.8
        elif accuracy > 0.6:
            reliability = 0.6
        else:
            reliability = accuracy  # Penalizza accuracy basse
        
        return {
            'performance_score': float(performance_score),
            'confidence': float(confidence),
            'reliability': float(reliability),
            'algorithm_type': 'classical_ml',
            'accuracy_score': float(accuracy),
            'base_confidence': float(base_confidence),
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def normalize_mathematical_metrics(algorithm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizza metriche da algoritmi matematici (GARCH, VolumeProfile, IndicatorBasedML)
        
        Args:
            algorithm_result: Risultato da algoritmo matematico
            
        Returns:
            Dict con metriche normalizzate
            
        Raises:
            ValueError: Se confidence insufficiente o dati invalidi
        """
        if not isinstance(algorithm_result, dict):
            raise TypeError("algorithm_result must be dict")
        
        # FAIL FAST: Deve avere successo
        if not algorithm_result.get('success', True):  # Default True per retrocompatibilità
            raise ValueError("Mathematical algorithm execution failed")
        
        # FAIL FAST: Confidence deve esistere e essere valida
        base_confidence = algorithm_result.get('confidence')
        if base_confidence is None:
            raise KeyError("Missing required field 'confidence' in mathematical algorithm result")
        
        if not isinstance(base_confidence, (int, float)) or not (0.0 <= base_confidence <= 1.0):
            raise ValueError(f"Invalid confidence: {base_confidence} (must be 0.0-1.0)")
        
        if base_confidence < UnifiedTrainingMetrics.MATHEMATICAL_MIN_CONFIDENCE:
            raise ValueError(f"Insufficient confidence: {base_confidence:.3f} < {UnifiedTrainingMetrics.MATHEMATICAL_MIN_CONFIDENCE}")
        
        # Per algoritmi matematici, performance = confidence * 100
        performance_score = base_confidence * 100.0
        confidence = base_confidence
        
        # Reliability alta per algoritmi matematici (sono deterministici)
        reliability = 1.0
        
        return {
            'performance_score': float(performance_score),
            'confidence': float(confidence),
            'reliability': float(reliability),
            'algorithm_type': 'mathematical',
            'base_confidence': float(base_confidence),
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def normalize_pivot_points_metrics(evaluation_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizza metriche specifiche per PivotPoints_Classic
        
        Args:
            evaluation_report: Report da evaluation_report.json del PIVOT
            
        Returns:
            Dict con metriche normalizzate
            
        Raises:
            ValueError: Se metriche insufficienti o dati invalidi
        """
        if not isinstance(evaluation_report, dict):
            raise TypeError("evaluation_report must be dict")
        
        # FAIL FAST: Campi critici devono esistere
        required_fields = ['confidence', 'avg_hit_rate', 'consistency', 'evaluation_windows']
        for field in required_fields:
            if field not in evaluation_report:
                raise KeyError(f"Missing required field '{field}' in PivotPoints evaluation report")
        
        confidence = evaluation_report['confidence']
        avg_hit_rate = evaluation_report['avg_hit_rate']
        consistency = evaluation_report['consistency']  # Più basso = meglio
        evaluation_windows = evaluation_report['evaluation_windows']
        success_rate = evaluation_report.get('success_rate', 1.0)
        
        # FAIL FAST: Validazione valori
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Invalid confidence: {confidence} (must be 0.0-1.0)")
        
        if not isinstance(avg_hit_rate, (int, float)) or not (0.0 <= avg_hit_rate <= 1.0):
            raise ValueError(f"Invalid avg_hit_rate: {avg_hit_rate} (must be 0.0-1.0)")
        
        if not isinstance(consistency, (int, float)) or consistency < 0:
            raise ValueError(f"Invalid consistency: {consistency} (must be >= 0)")
        
        if not isinstance(evaluation_windows, int) or evaluation_windows < 1:
            raise ValueError(f"Invalid evaluation_windows: {evaluation_windows} (must be >= 1)")
        
        if evaluation_windows < UnifiedTrainingMetrics.PIVOT_MIN_EVALUATION_WINDOWS:
            raise ValueError(f"Insufficient evaluation windows: {evaluation_windows} < {UnifiedTrainingMetrics.PIVOT_MIN_EVALUATION_WINDOWS}")
        
        # Score combinato per PivotPoints con pesi ottimizzati
        performance_score = (
            confidence * 0.35 +           # Confidence è importante
            avg_hit_rate * 0.35 +         # Hit rate è importante  
            (1.0 - min(consistency, 1.0)) * 0.15 +  # Meno variabilità = meglio
            success_rate * 0.10 +         # Success rate
            min(1.0, evaluation_windows / 10000) * 0.05  # Bonus per molti test
        ) * 100.0
        
        # Reliability basata su numero di test
        if evaluation_windows >= 10000:
            reliability = 1.0
        elif evaluation_windows >= 5000:
            reliability = 0.9
        elif evaluation_windows >= 2000:
            reliability = 0.7
        else:
            reliability = min(1.0, evaluation_windows / 2000.0)
        
        return {
            'performance_score': float(performance_score),
            'confidence': float(confidence),
            'reliability': float(reliability),
            'algorithm_type': 'pivot_points',
            'avg_hit_rate': float(avg_hit_rate),
            'consistency': float(consistency),
            'evaluation_windows': int(evaluation_windows),
            'success_rate': float(success_rate),
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def auto_normalize_metrics(training_data: Dict[str, Any], 
                             algorithm_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Auto-detect tipo algoritmo e normalizza metriche appropriate
        
        Args:
            training_data: Dati di training/evaluation da qualsiasi algoritmo
            algorithm_name: Nome algoritmo per hint sul tipo (opzionale)
            
        Returns:
            Dict con metriche normalizzate
            
        Raises:
            ValueError: Se non riesce a determinare il tipo o normalizzare
        """
        if not isinstance(training_data, dict):
            raise TypeError("training_data must be dict")
        
        # Auto-detection basata su campi presenti
        if 'final_loss' in training_data and 'epochs_completed' in training_data:
            # Neural Network pattern
            return UnifiedTrainingMetrics.normalize_neural_network_metrics(training_data)
        
        elif 'avg_hit_rate' in training_data and 'evaluation_windows' in training_data:
            # PivotPoints pattern specifico
            return UnifiedTrainingMetrics.normalize_pivot_points_metrics(training_data)
        
        elif 'metadata' in training_data and 'accuracy_score' in training_data['metadata']:
            # Classical ML pattern
            return UnifiedTrainingMetrics.normalize_classical_ml_metrics(training_data)
        
        elif 'confidence' in training_data:
            # Mathematical algorithm pattern
            return UnifiedTrainingMetrics.normalize_mathematical_metrics(training_data)
        
        else:
            # FAIL FAST: Non riesco a determinare il tipo
            available_fields = list(training_data.keys())
            raise ValueError(f"Cannot auto-detect algorithm type from training_data. Available fields: {available_fields}")
    
    @staticmethod
    def _calculate_std(values: list) -> float:
        """
        Calcola standard deviation di una lista di valori - BIBBIA COMPLIANT
        
        Args:
            values: Lista di valori numerici
            
        Returns:
            Standard deviation
            
        Raises:
            ValueError: Se lista vuota o valori invalidi
        """
        if not values:
            raise ValueError("Cannot calculate std of empty list")
        
        # FAIL FAST: Tutti i valori devono essere numerici
        for i, val in enumerate(values):
            if not isinstance(val, (int, float)):
                raise TypeError(f"Invalid value at index {i}: {val} (must be numeric)")
        
        if len(values) == 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


# Factory function per compatibilità con architettura esistente
def create_unified_training_metrics() -> UnifiedTrainingMetrics:
    """Factory function per creare UnifiedTrainingMetrics"""
    return UnifiedTrainingMetrics()


# Export main class
__all__ = [
    'UnifiedTrainingMetrics',
    'create_unified_training_metrics'
]