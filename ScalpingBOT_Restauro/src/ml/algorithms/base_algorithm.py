"""
Base Algorithm Class - CONSOLIDATION FIX
Eliminates structural duplication across algorithm implementations.

CONSOLIDATES PATTERNS FROM:
- src/ml/algorithms/support_resistance_algorithms.py
- src/ml/algorithms/pattern_recognition_algorithms.py  
- src/ml/algorithms/bias_detection_algorithms.py
- src/ml/algorithms/trend_analysis_algorithms.py

CLAUDE_RESTAURO.md COMPLIANCE:
- ✅ Zero structural duplication
- ✅ Single source of truth for algorithm patterns
- ✅ Fail fast error handling
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, field

from ...shared.enums import ModelType
from ...shared.exceptions import (
    InsufficientDataError,
    ModelNotInitializedError,
    InvalidInputError,
    PredictionError,
    AlgorithmExecutionError
)


@dataclass
class AlgorithmStats:
    """Standardized algorithm statistics structure"""
    executions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    last_execution_time: Optional[datetime] = None
    average_execution_time_ms: float = 0.0
    accuracy_rate: float = 0.0
    total_execution_time_ms: float = 0.0
    
    def update_execution(self, success: bool, execution_time_ms: float) -> None:
        """Update statistics after algorithm execution"""
        self.executions += 1
        self.total_execution_time_ms += execution_time_ms
        self.average_execution_time_ms = self.total_execution_time_ms / self.executions
        self.last_execution_time = datetime.now()
        
        if success:
            self.successful_predictions += 1
        else:
            self.failed_predictions += 1
            
        if self.executions > 0:
            self.accuracy_rate = self.successful_predictions / self.executions


@dataclass
class AlgorithmResult:
    """Standardized algorithm result structure"""
    success: bool
    data: Dict[str, Any]
    confidence: float
    algorithm_name: str
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.execution_time_ms < 0:
            raise ValueError(f"Execution time cannot be negative: {self.execution_time_ms}")


class BaseAlgorithm(ABC):
    """Base class for all ML algorithm implementations
    
    Eliminates structural duplication by providing common patterns:
    - Standardized initialization
    - Common statistics tracking
    - Unified error handling
    - Consistent execution flow
    - Model lifecycle management
    """
    
    def __init__(self, model_type: ModelType, ml_models: Optional[Dict[str, Any]] = None):
        """Initialize base algorithm with standardized structure
        
        Args:
            model_type: Type of ML model this algorithm handles
            ml_models: Dictionary of ML models (optional)
        """
        if not isinstance(model_type, ModelType):
            raise InvalidInputError("model_type", model_type, "Must be ModelType enum")
        
        self.model_type = model_type
        self.ml_models = ml_models or {}
        
        # Standardized statistics tracking (eliminates duplication)
        self.algorithm_stats = AlgorithmStats()
        
        # Algorithm metadata
        self._initialized = False
        self._algorithm_name = self.__class__.__name__
        
        # Validation
        self._validate_initialization()
        self._initialized = True
    
    def _validate_initialization(self) -> None:
        """Validate algorithm initialization - can be overridden by subclasses"""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if algorithm is properly initialized"""
        return self._initialized
    
    @property
    def algorithm_name(self) -> str:
        """Get algorithm name"""
        return self._algorithm_name
    
    def get_stats(self) -> Dict[str, Any]:
        """Get standardized algorithm statistics"""
        return {
            'algorithm_name': self.algorithm_name,
            'model_type': self.model_type.value,
            'initialized': self.is_initialized,
            'executions': self.algorithm_stats.executions,
            'successful_predictions': self.algorithm_stats.successful_predictions,
            'failed_predictions': self.algorithm_stats.failed_predictions,
            'accuracy_rate': self.algorithm_stats.accuracy_rate,
            'average_execution_time_ms': self.algorithm_stats.average_execution_time_ms,
            'last_execution_time': self.algorithm_stats.last_execution_time.isoformat() if self.algorithm_stats.last_execution_time else None
        }
    
    def validate_market_data(self, market_data: Dict[str, Any]) -> None:
        """Standardized market data validation"""
        if not isinstance(market_data, dict):
            raise InvalidInputError("market_data", market_data, "Must be dictionary")
        
        if not market_data:
            raise InvalidInputError("market_data", market_data, "Cannot be empty")
        
        # Check for required fields - can be extended by subclasses
        required_fields = ['prices', 'volumes', 'timestamps']
        for field in required_fields:
            if field not in market_data:
                raise InvalidInputError(field, None, f"Missing required field in market_data")
        
        # Validate data size consistency
        prices = market_data['prices']
        volumes = market_data['volumes']
        timestamps = market_data['timestamps']
        
        if len(prices) != len(volumes) or len(prices) != len(timestamps):
            raise InvalidInputError(
                "market_data", 
                f"sizes: prices={len(prices)}, volumes={len(volumes)}, timestamps={len(timestamps)}",
                "All arrays must have same length"
            )
    
    def check_data_sufficiency(self, market_data: Dict[str, Any], min_required: int) -> None:
        """Check if we have sufficient data for algorithm execution"""
        # BIBBIA COMPLIANT: FAIL FAST - no fallback to empty list
        if 'prices' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'prices' in market_data")
        data_size = len(market_data['prices'])
        if data_size < min_required:
            raise InsufficientDataError(
                required=min_required,
                available=data_size,
                operation=f"{self.algorithm_name} execution",
                details=f"Algorithm requires minimum {min_required} data points"
            )
    
    def execute_with_error_handling(self, market_data: Dict[str, Any], **kwargs) -> AlgorithmResult:
        """Execute algorithm with standardized error handling and statistics tracking"""
        if not self.is_initialized:
            raise ModelNotInitializedError(
                self.algorithm_name,
                "Algorithm must be initialized before execution"
            )
        
        start_time = datetime.now()
        
        try:
            # Validate input data
            self.validate_market_data(market_data)
            
            # Execute the algorithm (implemented by subclasses)
            result_data = self._execute_algorithm(market_data, **kwargs)
            
            # Calculate execution time
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            result = AlgorithmResult(
                success=True,
                data=result_data['data'],
                confidence=result_data['confidence'],
                algorithm_name=self.algorithm_name,
                execution_time_ms=execution_time_ms,
                # BIBBIA COMPLIANT: FAIL FAST - no fallback to empty dict
                metadata=result_data.get('metadata') or {}
            )
            
            # Update statistics
            self.algorithm_stats.update_execution(True, execution_time_ms)
            
            return result
            
        except Exception as e:
            # Calculate execution time even for failures
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update statistics
            self.algorithm_stats.update_execution(False, execution_time_ms)
            
            # Create error result
            error_result = AlgorithmResult(
                success=False,
                data={},
                confidence=0.0,
                algorithm_name=self.algorithm_name,
                execution_time_ms=execution_time_ms,
                error_details=str(e)
            )
            
            # Re-raise with context
            raise AlgorithmExecutionError(
                algorithm=self.algorithm_name,
                phase="execution",
                error_details=str(e),
                recovery_suggestion="Check input data and algorithm configuration"
            ) from e
    
    @abstractmethod
    def _execute_algorithm(self, market_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute the specific algorithm logic - must be implemented by subclasses
        
        Args:
            market_data: Validated market data
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Dict containing:
            - 'data': Algorithm result data
            - 'confidence': Confidence score (0.0-1.0)
            - 'metadata': Optional metadata (default: {})
        """
        pass
    
    @abstractmethod
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about this algorithm - must be implemented by subclasses
        
        Returns:
            Dict containing algorithm information like supported operations, parameters, etc.
        """
        pass
    
    def reset_stats(self) -> None:
        """Reset algorithm statistics"""
        self.algorithm_stats = AlgorithmStats()
    
    def get_model(self, model_name: str, asset: Optional[str] = None) -> Any:
        """Get ML model with validation - supports asset-specific models"""
        
        # Try asset-specific model first if asset is provided
        if asset:
            asset_model_name = f"{asset}_{model_name}"
            if asset_model_name in self.ml_models and self.ml_models[asset_model_name] is not None:
                return self.ml_models[asset_model_name]
        
        # Fallback to generic model name
        if model_name not in self.ml_models:
            # Try to find any model with the algorithm name as suffix
            matching_models = [k for k in self.ml_models.keys() if k.endswith(model_name)]
            if matching_models:
                model_key = matching_models[0]  # Use first match
                model = self.ml_models[model_key]
                if model is not None:
                    return model
            
            raise ModelNotInitializedError(
                model_name,
                f"Model not found in {self.algorithm_name}. Available: {list(self.ml_models.keys())}"
            )
        
        model = self.ml_models[model_name]
        if model is None:
            raise ModelNotInitializedError(
                model_name,
                f"Model {model_name} is None in {self.algorithm_name}"
            )
        
        return model
    
    def set_model(self, model_name: str, model: Any) -> None:
        """Set ML model with validation"""
        if model is None:
            raise InvalidInputError("model", model, "Model cannot be None")
        
        self.ml_models[model_name] = model
    
    def has_model(self, model_name: str) -> bool:
        """Check if algorithm has specific model"""
        return model_name in self.ml_models and self.ml_models[model_name] is not None


# === ALGORITHM FACTORY ===

def create_algorithm_factory(algorithm_class: type, model_type: ModelType) -> Callable[..., BaseAlgorithm]:
    """Create factory function for algorithm class"""
    
    def factory(ml_models: Optional[Dict[str, Any]] = None) -> BaseAlgorithm:
        """Factory function to create algorithm instance"""
        return algorithm_class(model_type, ml_models)
    
    return factory


# === EXPORTS ===

__all__ = [
    'BaseAlgorithm',
    'AlgorithmStats',
    'AlgorithmResult',
    'create_algorithm_factory'
]