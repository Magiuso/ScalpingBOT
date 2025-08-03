#!/usr/bin/env python3
"""
Algorithm Bridge - INTEGRAZIONE ALGORITMI MIGRATI CON COMPETITION SYSTEM
========================================================================

Bridge che collega gli algoritmi migrati (src/ml/algorithms/) con il sistema
di competition esistente (src/ml/models/competition.py).

Fornisce un'interfaccia unificata per:
- Registrazione algoritmi nelle competition
- Esecuzione algoritmi tramite competition system
- Mappatura risultati algorithm → Prediction objects
- Integration con AssetAnalyzer e AdvancedMarketAnalyzer

MANTIENE la logica identica del monolite ma con architettura modulare.
"""

from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import logging

# Import shared enums
from ...shared.enums import ModelType

# Import migrated algorithms
from ..algorithms.support_resistance_algorithms import (
    SupportResistanceAlgorithms, 
    create_support_resistance_algorithms
)
from ..algorithms.pattern_recognition_algorithms import (
    PatternRecognitionAlgorithms,
    create_pattern_recognition_algorithms
)
from ..algorithms.bias_detection_algorithms import (
    BiasDetectionAlgorithms,
    create_bias_detection_algorithms
)
from ..algorithms.trend_analysis_algorithms import (
    TrendAnalysisAlgorithms,
    create_trend_analysis_algorithms
)
from ..algorithms.volatility_prediction_algorithms import (
    VolatilityPredictionAlgorithms,
    create_volatility_prediction_algorithms
)

# Import competition system
from ..models.competition import AlgorithmCompetition
from ..models.base_models import Prediction, AlgorithmPerformance

# Import exceptions
from ..algorithms.support_resistance_algorithms import (
    InsufficientDataError,
    ModelNotInitializedError,
    InvalidInputError,
    PredictionError
)


class AlgorithmBridge:
    """
    Bridge tra algoritmi migrati e sistema di competition
    
    Fornisce integrazione completa per:
    - 20 algoritmi concreti migrati
    - Competition system esistente
    - Prediction generation
    - Performance tracking
    """
    
    def __init__(self, ml_models: Optional[Dict[str, Any]] = None, 
                 logger: Optional[logging.Logger] = None):
        """
        Inizializza Algorithm Bridge
        
        Args:
            ml_models: Dictionary dei modelli ML (LSTM, CNN, Transformer, etc.)
            logger: Logger per eventi
        """
        self.ml_models = ml_models or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize algorithm engines
        self.sr_algorithms = create_support_resistance_algorithms(ml_models)
        self.pattern_algorithms = create_pattern_recognition_algorithms(ml_models)
        self.bias_algorithms = create_bias_detection_algorithms(ml_models)
        self.trend_algorithms = create_trend_analysis_algorithms(ml_models)
        self.volatility_algorithms = create_volatility_prediction_algorithms(ml_models)
        
        # Algorithm registry: ModelType -> List[algorithm_names]
        self.algorithm_registry = {
            ModelType.SUPPORT_RESISTANCE: [
                "PivotPoints_Classic",
                "VolumeProfile_Advanced", 
                "LSTM_SupportResistance",
                "StatisticalLevels_ML",
                "Transformer_Levels"
            ],
            ModelType.PATTERN_RECOGNITION: [
                "CNN_PatternRecognizer",
                "Classical_Patterns",
                "LSTM_Sequences", 
                "Transformer_Patterns",
                "Ensemble_Patterns"
            ],
            ModelType.BIAS_DETECTION: [
                "Sentiment_LSTM",
                "VolumePrice_Analysis",
                "Momentum_ML",
                "Transformer_Bias", 
                "MultiModal_Bias"
            ],
            ModelType.TREND_ANALYSIS: [
                "RandomForest_Trend",
                "LSTM_TrendPrediction",
                "GradientBoosting_Trend",
                "Transformer_Trend",
                "Ensemble_Trend"
            ],
            ModelType.VOLATILITY_PREDICTION: [
                "GARCH_Volatility",
                "LSTM_Volatility", 
                "Realized_Volatility"
            ]
        }
        
        # Performance tracking
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'execution_times': {},
            'last_execution': None
        }
    
    def get_available_algorithms(self, model_type: ModelType) -> List[str]:
        """
        Restituisce lista algoritmi disponibili per un ModelType
        
        Args:
            model_type: Tipo di modello
            
        Returns:
            Lista nomi algoritmi disponibili
        """
        if model_type not in self.algorithm_registry:
            raise KeyError(f"Unknown model type: {model_type.value}")
        return self.algorithm_registry[model_type]
    
    def execute_algorithm(self, model_type: ModelType, algorithm_name: str, 
                         market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Esegue algoritmo specificato per il ModelType
        
        Args:
            model_type: Tipo di modello (SUPPORT_RESISTANCE, PATTERN_RECOGNITION, etc.)
            algorithm_name: Nome specifico algoritmo da eseguire
            market_data: Dati di mercato processati
            
        Returns:
            Risultati algoritmo in formato standardizzato
            
        Raises:
            ValueError: Se model_type o algorithm_name non supportati
            PredictionError: Se esecuzione algoritmo fallisce
        """
        execution_start = datetime.now()
        self.execution_stats['total_executions'] += 1
        self.execution_stats['last_execution'] = execution_start
        
        try:
            # Route algorithm execution to appropriate engine
            if model_type == ModelType.SUPPORT_RESISTANCE:
                if algorithm_name not in self.algorithm_registry[ModelType.SUPPORT_RESISTANCE]:
                    raise ValueError(f"Algorithm {algorithm_name} not available for {model_type.value}")
                result = self.sr_algorithms.run_algorithm(algorithm_name, market_data)
                
            elif model_type == ModelType.PATTERN_RECOGNITION:
                if algorithm_name not in self.algorithm_registry[ModelType.PATTERN_RECOGNITION]:
                    raise ValueError(f"Algorithm {algorithm_name} not available for {model_type.value}")
                result = self.pattern_algorithms.run_algorithm(algorithm_name, market_data)
                
            elif model_type == ModelType.BIAS_DETECTION:
                if algorithm_name not in self.algorithm_registry[ModelType.BIAS_DETECTION]:
                    raise ValueError(f"Algorithm {algorithm_name} not available for {model_type.value}")
                result = self.bias_algorithms.run_algorithm(algorithm_name, market_data)
                
            elif model_type == ModelType.TREND_ANALYSIS:
                if algorithm_name not in self.algorithm_registry[ModelType.TREND_ANALYSIS]:
                    raise ValueError(f"Algorithm {algorithm_name} not available for {model_type.value}")
                result = self.trend_algorithms.run_algorithm(algorithm_name, market_data)
                
            elif model_type == ModelType.VOLATILITY_PREDICTION:
                if algorithm_name not in self.algorithm_registry[ModelType.VOLATILITY_PREDICTION]:
                    raise ValueError(f"Algorithm {algorithm_name} not available for {model_type.value}")
                result = self.volatility_algorithms.run_algorithm(algorithm_name, market_data)
                
            else:
                raise ValueError(f"Unsupported model type: {model_type.value}")
            
            # Track execution time
            execution_time = (datetime.now() - execution_start).total_seconds()
            if algorithm_name not in self.execution_stats['execution_times']:
                self.execution_stats['execution_times'][algorithm_name] = []
            self.execution_stats['execution_times'][algorithm_name].append(execution_time)
            
            # Add metadata
            result['execution_metadata'] = {
                'model_type': model_type.value,
                'algorithm_name': algorithm_name,
                'execution_time_seconds': execution_time,
                'timestamp': execution_start.isoformat()
            }
            
            self.execution_stats['successful_executions'] += 1
            return result
            
        except Exception as e:
            self.execution_stats['failed_executions'] += 1
            self.logger.error(f"Algorithm execution failed: {model_type.value}/{algorithm_name} - {str(e)}")
            raise PredictionError(f"{model_type.value}_{algorithm_name}", str(e))
    
    def convert_to_prediction(self, algorithm_result: Dict[str, Any], 
                            asset: str, model_type: ModelType) -> Prediction:
        """
        Converte risultato algoritmo in oggetto Prediction standard
        
        Args:
            algorithm_result: Risultato raw dell'algoritmo
            asset: Nome asset
            model_type: Tipo di modello
            
        Returns:
            Oggetto Prediction standardizzato per competition system
        """
        if 'execution_metadata' not in algorithm_result:
            raise KeyError("Missing required field 'execution_metadata' from algorithm result")
        metadata = algorithm_result['execution_metadata']
        
        if 'algorithm_name' not in metadata:
            raise KeyError("Missing required field 'algorithm_name' from execution metadata")
        algorithm_name = metadata['algorithm_name']
        
        # Extract prediction components based on model type
        if model_type == ModelType.SUPPORT_RESISTANCE:
            if 'support_levels' not in algorithm_result:
                raise KeyError("Missing required field 'support_levels' from S/R algorithm result")
            if 'resistance_levels' not in algorithm_result:
                raise KeyError("Missing required field 'resistance_levels' from S/R algorithm result") 
            if 'method' not in algorithm_result:
                raise KeyError("Missing required field 'method' from S/R algorithm result")
            if 'confidence' not in algorithm_result:
                raise KeyError("Missing required field 'confidence' from S/R algorithm result")
            
            prediction_value = {
                'support_levels': algorithm_result['support_levels'],
                'resistance_levels': algorithm_result['resistance_levels'],
                'pivot': algorithm_result['pivot'] if 'pivot' in algorithm_result else None,  # Explicit check for optional field
                'method': algorithm_result['method']
            }
            confidence = algorithm_result['confidence']
            
        elif model_type == ModelType.PATTERN_RECOGNITION:
            if 'detected_patterns' not in algorithm_result:
                raise KeyError("Missing required field 'detected_patterns' from Pattern Recognition algorithm result")
            if 'pattern_strength' not in algorithm_result:
                raise KeyError("Missing required field 'pattern_strength' from Pattern Recognition algorithm result")
            if 'method' not in algorithm_result:
                raise KeyError("Missing required field 'method' from Pattern Recognition algorithm result")
            if 'confidence' not in algorithm_result:
                raise KeyError("Missing required field 'confidence' from Pattern Recognition algorithm result")
            
            prediction_value = {
                'patterns': algorithm_result['detected_patterns'],
                'pattern_strength': algorithm_result['pattern_strength'],
                'method': algorithm_result['method']
            }
            confidence = algorithm_result['confidence']
            
        elif model_type == ModelType.BIAS_DETECTION:
            if 'directional_bias' not in algorithm_result:
                raise KeyError("Missing required field 'directional_bias' from Bias Detection algorithm result")
            if 'method' not in algorithm_result:
                raise KeyError("Missing required field 'method' from Bias Detection algorithm result")
            if 'overall_confidence' not in algorithm_result:
                raise KeyError("Missing required field 'overall_confidence' from Bias Detection algorithm result")
            
            directional_bias = algorithm_result['directional_bias']
            if 'direction' not in directional_bias:
                raise KeyError("Missing required field 'direction' from directional_bias")
            
            prediction_value = {
                'direction': directional_bias['direction'],
                'bias_analysis': directional_bias,
                'method': algorithm_result['method']
            }
            confidence = algorithm_result['overall_confidence']
            
        elif model_type == ModelType.TREND_ANALYSIS:
            if 'trend_direction' not in algorithm_result:
                raise KeyError("Missing required field 'trend_direction' from Trend Analysis algorithm result")
            if 'trend_strength' not in algorithm_result:
                raise KeyError("Missing required field 'trend_strength' from Trend Analysis algorithm result")
            if 'method' not in algorithm_result:
                raise KeyError("Missing required field 'method' from Trend Analysis algorithm result")
            if 'trend_confidence' not in algorithm_result:
                raise KeyError("Missing required field 'trend_confidence' from Trend Analysis algorithm result")
            
            prediction_value = {
                'trend_direction': algorithm_result['trend_direction'],
                'trend_strength': algorithm_result['trend_strength'],
                'method': algorithm_result['method']
            }
            confidence = algorithm_result['trend_confidence']
            
        elif model_type == ModelType.VOLATILITY_PREDICTION:
            if 'volatility_forecast' not in algorithm_result:
                raise KeyError("Missing required field 'volatility_forecast' from Volatility Prediction algorithm result")
            if 'method' not in algorithm_result:
                raise KeyError("Missing required field 'method' from Volatility Prediction algorithm result")
            if 'confidence' not in algorithm_result:
                raise KeyError("Missing required field 'confidence' from Volatility Prediction algorithm result")
            
            prediction_value = {
                'volatility_forecast': algorithm_result['volatility_forecast'],
                'method': algorithm_result['method']
            }
            confidence = algorithm_result['confidence']
            
        else:
            # Unknown model type - fail fast
            raise ValueError(f"Unknown model type for prediction conversion: {model_type.value}")
            
        # Final confidence validation
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            raise ValueError(f"Invalid confidence value: {confidence} - must be float between 0 and 1")
        
        # Create Prediction object with correct parameters
        import uuid
        prediction = Prediction(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            model_type=model_type,
            algorithm_name=algorithm_name,
            prediction_data={
                'asset': asset,
                'prediction_value': prediction_value,
                'raw_result': algorithm_result,
                'execution_metadata': metadata
            },
            confidence=float(confidence),
            validation_time=datetime.now(),
            validation_criteria={
                'method': algorithm_name,
                'model_type': model_type.value,
                'confidence_threshold': 0.5
            }
        )
        
        return prediction
    
    def register_algorithms_in_competition(self, competition: AlgorithmCompetition) -> None:
        """
        Registra tutti gli algoritmi disponibili nella competition
        
        Args:
            competition: AlgorithmCompetition instance per un ModelType specifico
        """
        model_type = competition.model_type
        available_algorithms = self.get_available_algorithms(model_type)
        
        self.logger.info(f"Registering {len(available_algorithms)} algorithms for {model_type.value}")
        
        for algorithm_name in available_algorithms:
            # Create AlgorithmPerformance entry with correct parameters
            performance = AlgorithmPerformance(
                name=algorithm_name,
                model_type=model_type
                # Other parameters have defaults from the dataclass
            )
            
            # Register in competition
            competition.algorithms[algorithm_name] = performance
            
            self.logger.debug(f"Registered algorithm: {algorithm_name} for {model_type.value}")
    
    def create_algorithm_execution_callback(self, model_type: ModelType) -> Callable:
        """
        Crea callback function per esecuzione algoritmi in competition
        
        Args:
            model_type: Tipo di modello per cui creare callback
            
        Returns:
            Callback function che può essere usata dal competition system
        """
        def execute_callback(algorithm_name: str, market_data: Dict[str, Any]) -> Prediction:
            """
            Callback per esecuzione algoritmi dal competition system
            
            Args:
                algorithm_name: Nome algoritmo da eseguire
                market_data: Dati di mercato
                
            Returns:
                Prediction object per competition system
            """
            try:
                # Execute algorithm through bridge
                result = self.execute_algorithm(model_type, algorithm_name, market_data)
                
                # Convert to Prediction
                if 'asset' not in market_data:
                    raise KeyError("Missing required field 'asset' from market_data")
                asset = market_data['asset']
                prediction = self.convert_to_prediction(result, asset, model_type)
                
                return prediction
                
            except Exception as e:
                self.logger.error(f"Algorithm callback failed: {algorithm_name} - {str(e)}")
                # Return minimal prediction to avoid breaking competition
                import uuid
                return Prediction(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    model_type=model_type,
                    algorithm_name=algorithm_name,
                    prediction_data={
                        'asset': market_data['asset'] if 'asset' in market_data else 'error_unknown_asset',
                        'error': str(e),
                        'execution_failed': True
                    },
                    confidence=0.0,
                    validation_time=datetime.now(),
                    validation_criteria={
                        'method': algorithm_name,
                        'model_type': model_type.value,
                        'error_recovery': True
                    }
                )
        
        return execute_callback
    
    def get_bridge_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche bridge"""
        stats = self.execution_stats.copy()
        
        # Add algorithm engine stats
        stats['algorithm_engines'] = {
            'support_resistance': self.sr_algorithms.get_algorithm_stats(),
            'pattern_recognition': self.pattern_algorithms.get_algorithm_stats(),
            'bias_detection': self.bias_algorithms.get_algorithm_stats(),
            'trend_analysis': self.trend_algorithms.get_algorithm_stats(),
            'volatility_prediction': self.volatility_algorithms.get_algorithm_stats()
        }
        
        # Add algorithm counts
        stats['algorithm_counts'] = {
            model_type.value: len(algorithms) 
            for model_type, algorithms in self.algorithm_registry.items()
        }
        
        return stats
    
    def validate_system_integration(self) -> Dict[str, Any]:
        """
        Valida integrazione sistema completo
        
        Returns:
            Report validazione con status e dettagli
        """
        validation_report = {
            'status': 'healthy',
            'issues': [],
            'algorithm_availability': {},
            'model_dependencies': {},
            'integration_tests': {}
        }
        
        # Check algorithm availability
        for model_type, algorithms in self.algorithm_registry.items():
            available_count = len(algorithms)
            validation_report['algorithm_availability'][model_type.value] = {
                'available': available_count,
                'algorithms': algorithms
            }
            
            if available_count == 0:
                validation_report['issues'].append(f"No algorithms available for {model_type.value}")
        
        # Check ML model dependencies
        required_models = set()
        for algorithms in self.algorithm_registry.values():
            for alg in algorithms:
                if 'LSTM' in alg or 'CNN' in alg or 'Transformer' in alg:
                    required_models.add(alg)
        
        validation_report['model_dependencies'] = {
            'required_models': list(required_models),
            'available_models': list(self.ml_models.keys()),
            'missing_models': list(required_models - set(self.ml_models.keys()))
        }
        
        # Set overall status
        if validation_report['issues'] or validation_report['model_dependencies']['missing_models']:
            validation_report['status'] = 'degraded'
        
        if len(validation_report['issues']) > 5:
            validation_report['status'] = 'critical'
        
        return validation_report


# Factory function per compatibility
def create_algorithm_bridge(ml_models: Optional[Dict[str, Any]] = None,
                           logger: Optional[logging.Logger] = None) -> AlgorithmBridge:
    """Factory function per creare AlgorithmBridge"""
    return AlgorithmBridge(ml_models, logger)


# Export
__all__ = [
    'AlgorithmBridge',
    'create_algorithm_bridge'
]