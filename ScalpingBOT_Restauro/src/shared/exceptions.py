"""
Shared Exception Classes - CONSOLIDATION FIX
Consolidates duplicate exception classes from algorithm modules into single source.

MIGRATED FROM:
- src/ml/algorithms/support_resistance_algorithms.py (InsufficientDataError, ModelNotInitializedError, InvalidInputError, PredictionError)
- src/ml/algorithms/pattern_recognition_algorithms.py (imports same exceptions)
- src/ml/algorithms/bias_detection_algorithms.py (imports same exceptions)

CLAUDE_RESTAURO.md COMPLIANCE:
- ✅ Zero duplication
- ✅ Single source of truth
- ✅ Fail fast error handling
"""

from typing import Any, Optional, Dict, List


class InsufficientDataError(Exception):
    """Error for insufficient data to perform operation
    
    Used when algorithms require a minimum amount of data points
    but the available data is below the threshold.
    """
    
    def __init__(self, required: int, available: int, operation: str, details: Optional[str] = None):
        self.required = required
        self.available = available
        self.operation = operation
        self.details = details
        
        message = f"{operation} requires {required} data points, got {available}"
        if details:
            message += f" - {details}"
        super().__init__(message)


class ModelNotInitializedError(Exception):
    """Error for uninitialized ML model
    
    Used when attempting operations on models that haven't been 
    properly initialized or loaded.
    """
    
    def __init__(self, model_name: str, required_action: Optional[str] = None):
        self.model_name = model_name
        self.required_action = required_action
        
        message = f"Model '{model_name}' not initialized"
        if required_action:
            message += f" - {required_action}" 
        super().__init__(message)


class InvalidInputError(Exception):
    """Error for invalid input parameters
    
    Used when input validation fails with specific field information.
    """
    
    def __init__(self, field: str, value: Any, message: str, expected_type: Optional[type] = None):
        self.field = field
        self.value = value
        self.message = message
        self.expected_type = expected_type
        
        error_msg = f"Invalid {field}: {value} - {message}"
        if expected_type:
            error_msg += f" (expected {expected_type.__name__})"
        super().__init__(error_msg)


class PredictionError(Exception):
    """Error for prediction failures
    
    Used when ML algorithms fail to generate valid predictions.
    """
    
    def __init__(self, algorithm: str, message: str, input_data_info: Optional[Dict] = None):
        self.algorithm = algorithm
        self.message = message
        self.input_data_info = input_data_info
        
        error_msg = f"{algorithm} prediction error: {message}"
        if input_data_info:
            error_msg += f" (input: {input_data_info})"
        super().__init__(error_msg)


class ConfigurationError(Exception):
    """Error for configuration issues
    
    Used when system configuration is invalid or missing required parameters.
    """
    
    def __init__(self, component: str, parameter: str, issue: str, suggestion: Optional[str] = None):
        self.component = component
        self.parameter = parameter
        self.issue = issue
        self.suggestion = suggestion
        
        message = f"Configuration error in {component}.{parameter}: {issue}"
        if suggestion:
            message += f" - Suggestion: {suggestion}"
        super().__init__(message)


class DataValidationError(Exception):
    """Error for data validation failures
    
    Used when market data fails validation checks.
    """
    
    def __init__(self, data_type: str, validation_rule: str, details: Optional[str] = None):
        self.data_type = data_type
        self.validation_rule = validation_rule
        self.details = details
        
        message = f"Data validation failed for {data_type}: {validation_rule}"
        if details:
            message += f" - {details}"
        super().__init__(message)


class AlgorithmExecutionError(Exception):
    """Error for algorithm execution failures
    
    Used when algorithms fail during execution with detailed context.
    """
    
    def __init__(self, algorithm: str, phase: str, error_details: str, recovery_suggestion: Optional[str] = None):
        self.algorithm = algorithm
        self.phase = phase
        self.error_details = error_details
        self.recovery_suggestion = recovery_suggestion
        
        message = f"Algorithm '{algorithm}' failed in {phase}: {error_details}"
        if recovery_suggestion:
            message += f" - Recovery: {recovery_suggestion}"
        super().__init__(message)


class CompetitionSystemError(Exception):
    """Error for competition system failures
    
    Used when the champion competition system encounters errors.
    """
    
    def __init__(self, model_type: str, competition_phase: str, error_msg: str):
        self.model_type = model_type
        self.competition_phase = competition_phase
        self.error_msg = error_msg
        
        message = f"Competition system error for {model_type} in {competition_phase}: {error_msg}"
        super().__init__(message)


class MLTrainingError(Exception):
    """Error for ML training failures
    
    Used when ML model training encounters unrecoverable errors.
    """
    
    def __init__(self, model_name: str, training_stage: str, error_details: str, epoch: Optional[int] = None):
        self.model_name = model_name
        self.training_stage = training_stage
        self.error_details = error_details
        self.epoch = epoch
        
        message = f"ML training failed for {model_name} in {training_stage}: {error_details}"
        if epoch is not None:
            message += f" (epoch {epoch})"
        super().__init__(message)


class TensorShapeError(Exception):
    """Error for tensor shape mismatches
    
    Used when tensor operations fail due to shape incompatibilities.
    """
    
    def __init__(self, operation: str, expected_shape: tuple, actual_shape: tuple, context: Optional[str] = None):
        self.operation = operation
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        self.context = context
        
        message = f"Tensor shape error in {operation}: expected {expected_shape}, got {actual_shape}"
        if context:
            message += f" ({context})"
        super().__init__(message)


# === SPECIALIZED ERROR COLLECTIONS ===

class AlgorithmErrors:
    """Collection of algorithm-specific errors"""
    InsufficientData = InsufficientDataError
    ModelNotInitialized = ModelNotInitializedError
    InvalidInput = InvalidInputError
    PredictionFailed = PredictionError
    ExecutionFailed = AlgorithmExecutionError


class SystemErrors:
    """Collection of system-level errors"""
    Configuration = ConfigurationError
    DataValidation = DataValidationError
    Competition = CompetitionSystemError


class MLErrors:
    """Collection of ML-specific errors"""
    Training = MLTrainingError
    TensorShape = TensorShapeError
    ModelNotInitialized = ModelNotInitializedError
    PredictionFailed = PredictionError


# === ERROR HANDLING UTILITIES ===

def create_algorithm_error(algorithm_name: str, error_type: str, message: str, **kwargs) -> Exception:
    """Factory function to create appropriate algorithm errors with fail-fast principles"""
    
    error_types = {
        'insufficient_data': lambda: InsufficientDataError(
            required=kwargs.get('required', 0),
            available=kwargs.get('available', 0), 
            operation=f"{algorithm_name} execution",
            details=message
        ),
        'model_not_initialized': lambda: ModelNotInitializedError(
            model_name=algorithm_name,
            required_action=message
        ),
        'invalid_input': lambda: InvalidInputError(
            field=kwargs.get('field', 'unknown'),
            value=kwargs.get('value', None),
            message=message,
            expected_type=kwargs.get('expected_type')
        ),
        'prediction_failed': lambda: PredictionError(
            algorithm=algorithm_name,
            message=message,
            input_data_info=kwargs.get('input_data_info')
        ),
        'execution_failed': lambda: AlgorithmExecutionError(
            algorithm=algorithm_name,
            phase=kwargs.get('phase', 'unknown'),
            error_details=message,
            recovery_suggestion=kwargs.get('recovery_suggestion')
        )
    }
    
    if error_type not in error_types:
        raise ValueError(f"Unknown error type: {error_type}. Available: {list(error_types.keys())}")
    
    return error_types[error_type]()


# === EXPORTS ===

__all__ = [
    # Individual exceptions
    'InsufficientDataError',
    'ModelNotInitializedError', 
    'InvalidInputError',
    'PredictionError',
    'ConfigurationError',
    'DataValidationError',
    'AlgorithmExecutionError',
    'CompetitionSystemError',
    'MLTrainingError',
    'TensorShapeError',
    
    # Error collections
    'AlgorithmErrors',
    'SystemErrors', 
    'MLErrors',
    
    # Utilities
    'create_algorithm_error'
]