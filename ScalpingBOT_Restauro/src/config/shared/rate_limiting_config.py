"""
Unified Rate Limiting Configuration - CONSOLIDATION FIX
Consolidates all rate limiting configurations from multiple files into single source of truth.

MIGRATED FROM:
- src/config/domain/system_config.py (rate_limits)
- src/config/domain/monitoring_config.py (event_rate_limits + rate_limits duplicate)

CLAUDE_RESTAURO.md COMPLIANCE:
- ✅ Zero duplication
- ✅ Single source of truth
- ✅ Fail fast error handling
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum


class RateLimitCategory(Enum):
    """Categories of rate-limited operations"""
    # Core system operations
    TICK_PROCESSING = "tick_processing"
    PREDICTIONS = "predictions"
    VALIDATIONS = "validations"
    
    # ML Training operations
    MODEL_TRAINING = "model_training"
    LEARNING_PROGRESS = "learning_progress"
    CHAMPION_CHANGES = "champion_changes"
    ALGORITHM_UPDATES = "algorithm_updates"
    
    # Performance monitoring
    PERFORMANCE_METRICS = "performance_metrics"
    MEMORY_USAGE = "memory_usage"
    DIAGNOSTICS = "diagnostics"
    SYSTEM_STATUS = "system_status"
    
    # Debug and validation
    TENSOR_VALIDATION = "tensor_validation"
    GRADIENT_DEBUG = "gradient_debug"
    OVERFITTING_DEBUG = "overfitting_debug"
    INTERNAL_STATE = "internal_state"
    
    # Critical events (never rate limited)
    EMERGENCY_EVENTS = "emergency_events"
    ERROR_EVENTS = "error_events"
    VALIDATION_COMPLETE = "validation_complete"


@dataclass
class RateLimitRule:
    """Single rate limiting rule"""
    category: RateLimitCategory
    events_per_second: float
    description: str
    critical: bool = False  # Critical events bypass rate limiting
    
    def __post_init__(self):
        if self.events_per_second < 0:
            raise ValueError(f"Rate limit cannot be negative: {self.events_per_second}")
        if self.critical and self.events_per_second > 0:
            # Critical events should have 0 rate limit (unlimited)
            self.events_per_second = 0.0


@dataclass  
class UnifiedRateLimitingConfig:
    """Unified rate limiting configuration for all system components
    
    Replaces duplicated rate limiting configurations across:
    - SystemConfig.rate_limits
    - EventFilterSettings.event_rate_limits  
    - PerformanceSettings.rate_limits
    """
    
    # === CORE SYSTEM RATE LIMITS ===
    tick_processing_limit: float = 100.0      # 100 tick events per second max
    predictions_limit: float = 50.0           # 50 prediction events per second max  
    validations_limit: float = 25.0           # 25 validation events per second max
    
    # === ML TRAINING RATE LIMITS ===
    learning_progress_limit: float = 0.1      # 1 every 10 seconds (was causing spam)
    model_training_limit: float = 0.0         # No limit for critical training events
    champion_changes_limit: float = 0.0       # No limit (rare events)
    algorithm_updates_limit: float = 0.1      # 1 every 10 seconds
    
    # === PERFORMANCE MONITORING RATE LIMITS ===
    performance_metrics_limit: float = 0.01   # 1 every 100 seconds  
    memory_usage_limit: float = 0.01          # 1 every 100 seconds
    diagnostics_limit: float = 0.05           # 1 every 20 seconds
    system_status_limit: float = 0.1          # 1 every 10 seconds
    
    # === DEBUG RATE LIMITS ===
    tensor_validation_limit: float = 0.01     # 1 every 100 seconds
    gradient_debug_limit: float = 0.02        # 1 every 50 seconds
    overfitting_debug_limit: float = 0.05     # 1 every 20 seconds
    internal_state_limit: float = 0.02        # 1 every 50 seconds
    
    # === CRITICAL EVENTS (NO RATE LIMITING) ===
    emergency_events_limit: float = 0.0       # No limit (critical)
    error_events_limit: float = 0.0           # No limit (important)
    validation_complete_limit: float = 0.0    # No limit (critical)
    
    def get_rate_limit(self, category: RateLimitCategory) -> float:
        """Get rate limit for specific category with FAIL-FAST validation"""
        
        rate_limit_map = {
            RateLimitCategory.TICK_PROCESSING: self.tick_processing_limit,
            RateLimitCategory.PREDICTIONS: self.predictions_limit,
            RateLimitCategory.VALIDATIONS: self.validations_limit,
            RateLimitCategory.LEARNING_PROGRESS: self.learning_progress_limit,
            RateLimitCategory.MODEL_TRAINING: self.model_training_limit,
            RateLimitCategory.CHAMPION_CHANGES: self.champion_changes_limit,
            RateLimitCategory.ALGORITHM_UPDATES: self.algorithm_updates_limit,
            RateLimitCategory.PERFORMANCE_METRICS: self.performance_metrics_limit,
            RateLimitCategory.MEMORY_USAGE: self.memory_usage_limit,
            RateLimitCategory.DIAGNOSTICS: self.diagnostics_limit,
            RateLimitCategory.SYSTEM_STATUS: self.system_status_limit,
            RateLimitCategory.TENSOR_VALIDATION: self.tensor_validation_limit,
            RateLimitCategory.GRADIENT_DEBUG: self.gradient_debug_limit,
            RateLimitCategory.OVERFITTING_DEBUG: self.overfitting_debug_limit,
            RateLimitCategory.INTERNAL_STATE: self.internal_state_limit,
            RateLimitCategory.EMERGENCY_EVENTS: self.emergency_events_limit,
            RateLimitCategory.ERROR_EVENTS: self.error_events_limit,
            RateLimitCategory.VALIDATION_COMPLETE: self.validation_complete_limit,
        }
        
        if category not in rate_limit_map:
            raise KeyError(f"No rate limit configured for category: {category}")
        
        return rate_limit_map[category]
    
    def get_legacy_rate_limits_dict(self) -> Dict[str, float]:
        """Get rate limits as dictionary for backward compatibility
        
        Returns combined rate limits matching the original format for existing code.
        """
        return {
            # Original system_config.py format (converted from int to float)
            'tick_processing': self.tick_processing_limit,
            'predictions': self.predictions_limit,
            'validations': self.validations_limit,
            'training_events': self.model_training_limit,
            'champion_changes': self.champion_changes_limit,
            'emergency_events': self.emergency_events_limit,
            'diagnostics': self.diagnostics_limit,
            
            # Original monitoring_config.py event_rate_limits format
            'learning_progress': self.learning_progress_limit,
            'champion_change': self.champion_changes_limit,
            'model_training': self.model_training_limit,
            'performance_metrics': self.performance_metrics_limit,
            'prediction_generated': self.predictions_limit,
            'overfitting_debug': self.overfitting_debug_limit,
            'tensor_validation': self.tensor_validation_limit,
            'gradient_debug': self.gradient_debug_limit,
            
            # Original monitoring_config.py rate_limits format
            'validation_complete': self.validation_complete_limit,
            'emergency_stop': self.emergency_events_limit,
            'algorithm_update': self.algorithm_updates_limit,
            'diagnostics_event': self.diagnostics_limit,
            'internal_state': self.internal_state_limit,
            'memory_usage': self.memory_usage_limit,
            'system_status': self.system_status_limit,
            'error_event': self.error_events_limit
        }
    
    def is_critical_event(self, category: RateLimitCategory) -> bool:
        """Check if event category is critical and should bypass rate limiting"""
        critical_categories = {
            RateLimitCategory.EMERGENCY_EVENTS,
            RateLimitCategory.ERROR_EVENTS,
            RateLimitCategory.VALIDATION_COMPLETE,
            RateLimitCategory.MODEL_TRAINING,  # Training events are critical
            RateLimitCategory.CHAMPION_CHANGES  # Champion changes are rare and important
        }
        return category in critical_categories
    
    def validate_configuration(self) -> None:
        """Validate rate limiting configuration with FAIL-FAST principles"""
        
        # Check all rate limits are non-negative
        for category in RateLimitCategory:
            rate_limit = self.get_rate_limit(category)
            if rate_limit < 0:
                raise ValueError(f"Rate limit cannot be negative for {category.value}: {rate_limit}")
        
        # Verify critical events have zero rate limits
        for category in RateLimitCategory:
            if self.is_critical_event(category):
                rate_limit = self.get_rate_limit(category)
                if rate_limit != 0.0:
                    raise ValueError(f"Critical event {category.value} must have zero rate limit, got: {rate_limit}")


# === GLOBAL CONFIGURATION INSTANCE ===

DEFAULT_RATE_LIMITING_CONFIG = UnifiedRateLimitingConfig()

def get_rate_limiting_config() -> UnifiedRateLimitingConfig:
    """Get the global rate limiting configuration"""
    return DEFAULT_RATE_LIMITING_CONFIG

def set_rate_limiting_config(config: UnifiedRateLimitingConfig) -> None:
    """Set a new global rate limiting configuration"""
    global DEFAULT_RATE_LIMITING_CONFIG
    config.validate_configuration()  # Validate before setting
    DEFAULT_RATE_LIMITING_CONFIG = config


# === BACKWARD COMPATIBILITY FUNCTIONS ===

def get_legacy_system_rate_limits() -> Dict[str, int]:
    """Get rate limits in original system_config.py format (int values)"""
    config = get_rate_limiting_config()
    return {
        'tick_processing': int(config.tick_processing_limit),
        'predictions': int(config.predictions_limit),
        'validations': int(config.validations_limit),
        'training_events': int(config.model_training_limit),
        'champion_changes': int(config.champion_changes_limit),
        'emergency_events': int(config.emergency_events_limit),
        'diagnostics': int(config.diagnostics_limit)
    }

def get_legacy_monitoring_rate_limits() -> Dict[str, float]:
    """Get rate limits in original monitoring_config.py format"""
    config = get_rate_limiting_config()
    return config.get_legacy_rate_limits_dict()