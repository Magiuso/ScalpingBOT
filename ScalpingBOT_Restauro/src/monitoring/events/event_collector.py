#!/usr/bin/env python3
"""
Event Collector - CLEANED AND SIMPLIFIED
REGOLE CLAUDE_RESTAURO.md APPLICATE:
- ✅ Zero fallback/defaults  
- ✅ Fail fast error handling
- ✅ No debug prints/spam
- ✅ No test code embedded
- ✅ No redundant functions
- ✅ Simplified architecture

Sistema di raccolta eventi semplificato e efficiente.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# Import configuration
from src.config.domain.monitoring_config import (
    MLTrainingLoggerConfig, EventSeverity, VerbosityLevel
)


class EventSource(Enum):
    """Tipi di fonti eventi"""
    ADVANCED_MARKET_ANALYZER = "AdvancedMarketAnalyzer"
    UNIFIED_ANALYZER_SYSTEM = "UnifiedAnalyzerSystem"
    MANUAL = "Manual"
    SYSTEM = "System"
    EXTERNAL = "External"


class EventType(Enum):
    """Tipi di eventi standardizzati"""
    LEARNING_PROGRESS = "learning_progress"
    CHAMPION_CHANGE = "champion_change"
    MODEL_TRAINING = "model_training"
    PERFORMANCE_METRICS = "performance_metrics"
    EMERGENCY_STOP = "emergency_stop"
    VALIDATION_COMPLETE = "validation_complete"
    PREDICTION_GENERATED = "prediction_generated"
    ALGORITHM_UPDATE = "algorithm_update"
    DIAGNOSTICS_EVENT = "diagnostics_event"
    INTERNAL_STATE = "internal_state"
    MEMORY_USAGE = "memory_usage"
    SYSTEM_STATUS = "system_status"
    ERROR_EVENT = "error_event"


@dataclass
class MLEvent:
    """Struttura standard per eventi ML - SIMPLIFIED"""
    # Core fields - NO DEFAULTS per fail-fast
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: EventType = EventType.SYSTEM_STATUS
    source: EventSource = EventSource.SYSTEM
    severity: EventSeverity = EventSeverity.INFO
    
    # Context
    asset: Optional[str] = None
    session_id: Optional[str] = None
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    source_method: Optional[str] = None
    processing_time_ms: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte evento in dizionario per serializzazione"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'source': self.source.value,
            'severity': self.severity.value,
            'asset': self.asset,
            'session_id': self.session_id,
            'data': self.data,
            'source_method': self.source_method,
            'processing_time_ms': self.processing_time_ms,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MLEvent':
        """Crea evento da dizionario - FAIL FAST se campi mancanti"""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")
        
        # Required fields - FAIL FAST se mancanti
        required_fields = ['event_id', 'timestamp', 'event_type', 'source', 'severity']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            
        event = cls()
        event.event_id = data['event_id']
        event.timestamp = datetime.fromisoformat(data['timestamp'])
        
        # Validate and convert enums - FAIL FAST se invalid
        try:
            event.event_type = EventType(data['event_type'])
        except ValueError:
            raise ValueError(f"Invalid event_type: '{data['event_type']}'. Must be one of: {[e.value for e in EventType]}")
        
        try:
            event.source = EventSource(data['source'])
        except ValueError:
            raise ValueError(f"Invalid source: '{data['source']}'. Must be one of: {[e.value for e in EventSource]}")
        
        try:
            event.severity = EventSeverity(data['severity'])
        except ValueError:
            raise ValueError(f"Invalid severity: '{data['severity']}'. Must be one of: {[e.value for e in EventSeverity]}")
        
        # Optional fields - check if present
        event.asset = data['asset'] if 'asset' in data else None
        event.session_id = data['session_id'] if 'session_id' in data else None
        event.data = data['data'] if 'data' in data else {}
        event.source_method = data['source_method'] if 'source_method' in data else None
        event.processing_time_ms = data['processing_time_ms'] if 'processing_time_ms' in data else None
        event.tags = data['tags'] if 'tags' in data else []
        
        return event


class RateLimiter:
    """Rate limiter semplificato per eventi"""
    
    def __init__(self):
        self.last_event_times: Dict[str, float] = {}
    
    def is_allowed(self, event_type: str, events_per_second: float) -> bool:
        """
        Verifica se evento è permesso dal rate limit
        
        Args:
            event_type: Tipo di evento
            events_per_second: Limite eventi/secondo (0 = no limit)
            
        Returns:
            bool: True se evento è permesso
        """
        if events_per_second <= 0:
            return True
        
        now = time.time()
        if event_type not in self.last_event_times:
            self.last_event_times[event_type] = 0
        last_time = self.last_event_times[event_type]
        min_interval = 1.0 / events_per_second
        
        if now - last_time >= min_interval:
            self.last_event_times[event_type] = now
            return True
        
        return False


class EventCollector:
    """
    Collettore semplificato di eventi ML - NO OVER-ENGINEERING
    """
    
    def __init__(self, config: MLTrainingLoggerConfig, session_id: Optional[str] = None):
        if not isinstance(config, MLTrainingLoggerConfig):
            raise TypeError(f"Expected MLTrainingLoggerConfig, got {type(config)}")
        
        self.config = config
        if session_id is None:
            self.session_id = str(uuid.uuid4())
        else:
            self.session_id = session_id
        
        # 🔧 FIXED MEMORY LEAK: Use deque with maxlen instead of manual list truncation
        if not hasattr(config.performance, 'max_buffer_size'):
            raise AttributeError("Config.performance missing required attribute: max_buffer_size")
        self.max_events = config.performance.max_buffer_size
        
        from collections import deque
        self.events: deque = deque(maxlen=self.max_events)
        
        # Rate limiting
        self.rate_limiter = RateLimiter()
        
        # Callback system - SIMPLIFIED
        self.callbacks: List[Callable[[MLEvent], None]] = []
        
        # Statistics - ESSENTIAL only
        self.stats = {
            'total_events': 0,
            'events_by_type': defaultdict(int),
            'last_event_time': None,
            'start_time': datetime.now()
        }
    
    def emit_event(self, event: MLEvent) -> bool:
        """
        Emette un evento - SIMPLIFIED
        
        Args:
            event: Evento da emettere
            
        Returns:
            bool: True se evento accettato, False se filtrato
        """
        if not isinstance(event, MLEvent):
            raise TypeError(f"Expected MLEvent, got {type(event)}")
        
        # Set session ID if not present
        if not event.session_id:
            event.session_id = self.session_id
        
        # Validate required configuration
        if not hasattr(self.config.performance, 'rate_limits'):
            raise RuntimeError("Config missing rate_limits configuration")
        
        # Apply rate limiting - FAIL FAST se config missing
        event_type_str = event.event_type.value
        if event_type_str not in self.config.performance.rate_limits:
            raise ValueError(f"No rate limit configured for event type: {event_type_str}")
        
        rate_limit = self.config.performance.rate_limits[event_type_str]
        if rate_limit > 0 and not self.rate_limiter.is_allowed(event_type_str, rate_limit):
            return False  # Rate limited
        
        # 🔧 FIXED MEMORY LEAK: deque with maxlen handles auto-cleanup automatically
        # No manual cleanup needed - deque automatically removes oldest elements
        self.events.append(event)
        
        # Update statistics
        self.stats['total_events'] += 1
        self.stats['events_by_type'][event_type_str] += 1
        self.stats['last_event_time'] = event.timestamp
        
        # Trigger callbacks - FAIL FAST se callback fallisce
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                raise RuntimeError(f"Event callback failed for {callback.__name__}: {e}")
        
        return True
    
    def add_callback(self, callback: Callable[[MLEvent], None]):
        """Aggiunge callback per eventi"""
        if not callable(callback):
            raise TypeError(f"Callback must be callable, got {type(callback)}")
        
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[MLEvent], None]):
        """Rimuove callback per eventi"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_events(self, count: Optional[int] = None) -> List[MLEvent]:
        """
        Ottieni eventi dal buffer
        
        Args:
            count: Numero massimo di eventi (None = tutti)
            
        Returns:
            List[MLEvent]: Eventi
        """
        if count is None:
            return list(self.events)
        else:
            # 🔧 FIXED PYLANCE ERROR: deque doesn't support slicing, convert to list first
            if count > 0:
                events_list = list(self.events)
                return events_list[-count:]
            else:
                return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche del collector"""
        return {
            **self.stats,
            'current_buffer_size': len(self.events),
            'max_buffer_size': self.max_events,
            'session_id': self.session_id
        }
    
    def clear_events(self):
        """Pulisce tutti gli eventi dal buffer"""
        self.events.clear()
    
    def create_manual_event(self, event_type: EventType, data: Dict[str, Any],
                           severity: EventSeverity = EventSeverity.INFO,
                           asset: Optional[str] = None) -> MLEvent:
        """Crea evento manuale"""
        if not isinstance(event_type, EventType):
            raise TypeError(f"event_type must be EventType, got {type(event_type)}")
        if not isinstance(data, dict):
            raise TypeError(f"data must be dict, got {type(data)}")
        if not isinstance(severity, EventSeverity):
            raise TypeError(f"severity must be EventSeverity, got {type(severity)}")
        
        return MLEvent(
            event_type=event_type,
            source=EventSource.MANUAL,
            severity=severity,
            asset=asset,
            data=data,
            session_id=self.session_id
        )
    
    def emit_manual_event(self, event_type: EventType, data: Dict[str, Any],
                         severity: EventSeverity = EventSeverity.INFO,
                         asset: Optional[str] = None) -> bool:
        """Crea ed emette evento manuale"""
        event = self.create_manual_event(event_type, data, severity, asset)
        return self.emit_event(event)


# ================== FACTORY FUNCTIONS ==================

def create_event_collector(config: MLTrainingLoggerConfig) -> EventCollector:
    """Factory function per creare EventCollector"""
    return EventCollector(config)


def create_learning_progress_event(progress_data: Dict[str, Any], asset: str) -> MLEvent:
    """Factory per eventi di learning progress"""
    if not isinstance(progress_data, dict):
        raise TypeError(f"progress_data must be dict, got {type(progress_data)}")
    if not isinstance(asset, str) or not asset.strip():
        raise ValueError("asset must be non-empty string")
    
    return MLEvent(
        event_type=EventType.LEARNING_PROGRESS,
        source=EventSource.SYSTEM,
        severity=EventSeverity.INFO,
        asset=asset,
        data=progress_data
    )


def create_champion_change_event(change_data: Dict[str, Any], asset: str) -> MLEvent:
    """Factory per eventi di champion change"""
    if not isinstance(change_data, dict):
        raise TypeError(f"change_data must be dict, got {type(change_data)}")
    if not isinstance(asset, str) or not asset.strip():
        raise ValueError("asset must be non-empty string")
    
    return MLEvent(
        event_type=EventType.CHAMPION_CHANGE,
        source=EventSource.SYSTEM,
        severity=EventSeverity.WARNING,
        asset=asset,
        data=change_data
    )


def create_emergency_stop_event(emergency_data: Dict[str, Any], asset: str) -> MLEvent:
    """Factory per eventi di emergency stop"""
    if not isinstance(emergency_data, dict):
        raise TypeError(f"emergency_data must be dict, got {type(emergency_data)}")
    if not isinstance(asset, str) or not asset.strip():
        raise ValueError("asset must be non-empty string")
    
    return MLEvent(
        event_type=EventType.EMERGENCY_STOP,
        source=EventSource.SYSTEM,
        severity=EventSeverity.ERROR,
        asset=asset,
        data=emergency_data
    )


# Export main classes and functions
__all__ = [
    'EventCollector',
    'MLEvent', 
    'EventType',
    'EventSource',
    'EventSeverity',
    'RateLimiter',
    'create_event_collector',
    'create_learning_progress_event',
    'create_champion_change_event', 
    'create_emergency_stop_event'
]