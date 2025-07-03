#!/usr/bin/env python3
"""
MLTrainingLogger - Event Collector
==================================

Sistema di raccolta eventi da diverse fonti (AdvancedMarketAnalyzer, UnifiedAnalyzerSystem).
Implementa pattern Observer e Hook System per integrazione non-invasiva.

Author: ScalpingBOT Team  
Version: 1.0.0
"""

import asyncio
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import weakref
import inspect

# Import configuration
from .Config_Manager import (
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
    PERFORMANCE_DEBUG = "performance_debug"
    MEMORY_USAGE = "memory_usage"
    SYSTEM_STATUS = "system_status"
    ERROR_EVENT = "error_event"


@dataclass
class MLEvent:
    """
    Struttura standard per eventi ML
    """
    # Core fields
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: Union[EventType, str] = EventType.SYSTEM_STATUS
    source: Union[EventSource, str] = EventSource.SYSTEM
    severity: EventSeverity = EventSeverity.INFO
    
    # Context
    asset: Optional[str] = None
    session_id: Optional[str] = None
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    source_method: Optional[str] = None
    source_object_id: Optional[str] = None
    processing_time_ms: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte evento in dizionario per serializzazione"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value if isinstance(self.event_type, EventType) else str(self.event_type),
            'source': self.source.value if isinstance(self.source, EventSource) else str(self.source),
            'severity': self.severity.value,
            'asset': self.asset,
            'session_id': self.session_id,
            'data': self.data,
            'source_method': self.source_method,
            'source_object_id': self.source_object_id,
            'processing_time_ms': self.processing_time_ms,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MLEvent':
        """Crea evento da dizionario"""
        event = cls()
        event.event_id = data.get('event_id', str(uuid.uuid4()))
        event.timestamp = datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now()
        
        # Handle enums
        event_type_str = data.get('event_type', 'system_status')
        try:
            event.event_type = EventType(event_type_str)
        except ValueError:
            event.event_type = event_type_str
        
        source_str = data.get('source', 'system')
        try:
            event.source = EventSource(source_str)
        except ValueError:
            event.source = source_str
        
        event.severity = EventSeverity(data.get('severity', 'info'))
        event.asset = data.get('asset')
        event.session_id = data.get('session_id')
        event.data = data.get('data', {})
        event.source_method = data.get('source_method')
        event.source_object_id = data.get('source_object_id')
        event.processing_time_ms = data.get('processing_time_ms')
        event.tags = data.get('tags', [])
        
        return event


class RateLimiter:
    """Rate limiter per eventi"""
    
    def __init__(self):
        self.last_event_times: Dict[str, datetime] = {}
        self.event_counts: Dict[str, deque] = defaultdict(lambda: deque())
    
    def is_allowed(self, event_type: str, rate_limit: float) -> bool:
        """
        Verifica se evento è permesso dal rate limit
        
        Args:
            event_type: Tipo di evento
            rate_limit: Limite eventi/secondo (0 = no limit)
            
        Returns:
            bool: True se evento è permesso
        """
        if rate_limit <= 0:
            return True
        
        now = datetime.now()
        
        # Sliding window rate limiting
        window_size = timedelta(seconds=1)
        event_times = self.event_counts[event_type]
        
        # Remove old events outside window
        while event_times and (now - event_times[0]) > window_size:
            event_times.popleft()
        
        # Check if under rate limit
        if len(event_times) < rate_limit:
            event_times.append(now)
            return True
        
        return False


class EventHook:
    """Hook per catturare eventi da oggetti esistenti"""
    
    def __init__(self, target_object: Any, method_name: str, 
                 event_type: EventType, collector: 'EventCollector'):
        self.target_object = weakref.ref(target_object) if target_object else None
        self.method_name = method_name
        self.event_type = event_type
        self.collector = weakref.ref(collector)
        self.original_method = None
        self.is_active = False
    
    def install(self) -> bool:
        """Installa l'hook sul metodo target"""
        try:
            target = self.target_object() if self.target_object else None
            if not target or not hasattr(target, self.method_name):
                print(f"⚠️ Target object or method {self.method_name} not found")  # DEBUG
                return False
            
            self.original_method = getattr(target, self.method_name)
            
            # AGGIUNGI QUESTO CONTROLLO
            if self.original_method is None:
                print(f"⚠️ Method {self.method_name} is None")  # DEBUG
                return False
            
            # Create wrapper function
            def hooked_method(*args, **kwargs):
                start_time = time.time()
                
                # AGGIUNGI QUESTO CONTROLLO
                if self.original_method is None:
                    print(f"❌ original_method is None in {self.method_name}")
                    return None
                
                # Call original method
                result = self.original_method(*args, **kwargs)
                
                # Create event
                processing_time = (time.time() - start_time) * 1000
                self._create_event_from_call(args, kwargs, result, processing_time)
                
                return result
            
            # Replace method
            setattr(target, self.method_name, hooked_method)
            self.is_active = True
            print(f"✅ Hook installed on {target.__class__.__name__}.{self.method_name}")  # DEBUG
            return True
            
        except Exception as e:
            print(f"❌ Failed to install hook on {self.method_name}: {e}")
            return False
    
    def uninstall(self) -> bool:
        """Rimuove l'hook ripristinando il metodo originale"""
        try:
            target = self.target_object() if self.target_object else None
            if not target or not self.original_method:
                return False
            
            setattr(target, self.method_name, self.original_method)
            self.is_active = False
            return True
            
        except Exception as e:
            print(f"Failed to uninstall hook on {self.method_name}: {e}")
            return False
    
    def _create_event_from_call(self, args: tuple, kwargs: dict, 
                              result: Any, processing_time: float):
        """Crea evento dalla chiamata del metodo"""
        collector = self.collector() if self.collector else None
        if not collector:
            return
        
        target = self.target_object() if self.target_object else None
        
        # Extract relevant data from method call
        event_data = {
            'method_args_count': len(args),
            'method_kwargs': list(kwargs.keys()),
            'has_result': result is not None,
            'result_type': type(result).__name__ if result is not None else None
        }
        
        # Add method-specific data extraction
        if self.method_name == 'analyze_tick' and args:
            event_data['tick_price'] = getattr(args[0], 'price', None)
            event_data['tick_volume'] = getattr(args[0], 'volume', None)
        
        elif self.method_name == 'update_champion' and args:
            event_data['new_champion'] = args[0] if args else None
            event_data['model_type'] = args[1] if len(args) > 1 else None
        
        # Create and emit event
        event = MLEvent(
            event_type=self.event_type,
            source=self._get_source_from_target(target),
            data=event_data,
            source_method=self.method_name,
            source_object_id=str(id(target)) if target else None,
            processing_time_ms=processing_time
        )
        
        collector.emit_event(event)
    
    def _get_source_from_target(self, target: Any) -> EventSource:
        """Determina la fonte dall'oggetto target"""
        if not target:
            return EventSource.SYSTEM
        
        class_name = target.__class__.__name__
        if 'AdvancedMarketAnalyzer' in class_name:
            return EventSource.ADVANCED_MARKET_ANALYZER
        elif 'UnifiedAnalyzerSystem' in class_name:
            return EventSource.UNIFIED_ANALYZER_SYSTEM
        else:
            return EventSource.EXTERNAL


class EventBuffer:
    """Buffer thread-safe per eventi"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.overflow_count = 0
    
    def add_event(self, event: MLEvent) -> bool:
        """
        Aggiunge evento al buffer
        
        Returns:
            bool: True se aggiunto, False se buffer pieno
        """
        with self.lock:
            if len(self.buffer) >= self.max_size:
                self.overflow_count += 1
                return False
            
            self.buffer.append(event)
            return True
    
    def get_events(self, count: Optional[int] = None) -> List[MLEvent]:
        """
        Estrae eventi dal buffer
        
        Args:
            count: Numero massimo di eventi (None = tutti)
            
        Returns:
            List[MLEvent]: Eventi estratti
        """
        with self.lock:
            if count is None:
                events = list(self.buffer)
                self.buffer.clear()
            else:
                events = []
                for _ in range(min(count, len(self.buffer))):
                    if self.buffer:
                        events.append(self.buffer.popleft())
            
            return events
    
    def peek_events(self, count: int = 10) -> List[MLEvent]:
        """Visualizza eventi senza rimuoverli"""
        with self.lock:
            return list(self.buffer)[-count:] if count > 0 else list(self.buffer)
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiche del buffer"""
        with self.lock:
            return {
                'current_size': len(self.buffer),
                'max_size': self.max_size,
                'utilization_percent': (len(self.buffer) / self.max_size) * 100,
                'overflow_count': self.overflow_count,
                'is_full': len(self.buffer) >= self.max_size
            }


class EventCollector:
    """
    Collettore principale di eventi ML
    """
    
    def __init__(self, config: MLTrainingLoggerConfig, session_id: Optional[str] = None):
        self.config = config
        self.session_id = session_id or str(uuid.uuid4())
        
        # Event processing
        self.event_buffer = EventBuffer(config.performance.event_queue_size)
        self.rate_limiter = RateLimiter()
        
        # Source management
        self.registered_sources: Dict[str, Any] = {}
        self.active_hooks: List[EventHook] = []
        
        # Callback system
        self.event_callbacks: List[Callable[[MLEvent], None]] = []
        self.source_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'total_events_received': 0,
            'total_events_filtered': 0,
            'total_events_rate_limited': 0,
            'events_by_type': defaultdict(int),
            'events_by_source': defaultdict(int),
            'last_event_time': None,
            'collection_start_time': datetime.now()
        }
        
        # Threading
        self.is_running = False
        self.processing_thread = None
        self.stop_event = threading.Event()
    
    def start(self):
        """Avvia il collettore di eventi"""
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        if self.config.performance.enable_async_processing:
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                name="EventCollector-Processing",
                daemon=True
            )
            self.processing_thread.start()
    
    def stop(self):
        """Ferma il collettore di eventi"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        # Uninstall all hooks
        for hook in self.active_hooks[:]:
            hook.uninstall()
        self.active_hooks.clear()
        
        # Wait for processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
    
    def register_source(self, source_name: str, source_object: Any) -> bool:
        """
        Registra una fonte di eventi
        
        Args:
            source_name: Nome identificativo della fonte
            source_object: Oggetto fonte
            
        Returns:
            bool: True se registrazione riuscita
        """
        try:
            # Store weak reference to avoid circular references
            self.registered_sources[source_name] = weakref.ref(source_object)
            
            # Auto-install hooks based on source type
            self._auto_install_hooks(source_name, source_object)
            
            return True
            
        except Exception as e:
            print(f"Failed to register source {source_name}: {e}")
            return False
    
    def unregister_source(self, source_name: str) -> bool:
        """Rimuove registrazione fonte"""
        if source_name not in self.registered_sources:
            return False
        
        # Remove related hooks
        hooks_to_remove = [h for h in self.active_hooks 
                          if h.target_object and h.target_object() == self.registered_sources[source_name]()]
        
        for hook in hooks_to_remove:
            hook.uninstall()
            self.active_hooks.remove(hook)
        
        del self.registered_sources[source_name]
        return True
    
    def _auto_install_hooks(self, source_name: str, source_object: Any):
        """Installa automaticamente hook basati sul tipo di fonte"""
        
        class_name = source_object.__class__.__name__
        
        if 'AdvancedMarketAnalyzer' in class_name:
            # Hook per AdvancedMarketAnalyzer
            analyzer_hooks = [
                ('analyze_tick', EventType.LEARNING_PROGRESS),
                ('update_champion', EventType.CHAMPION_CHANGE),
                ('train_model', EventType.MODEL_TRAINING),
                ('emergency_stop', EventType.EMERGENCY_STOP),
                ('validate_model', EventType.VALIDATION_COMPLETE)
            ]
            
            for method_name, event_type in analyzer_hooks:
                if hasattr(source_object, method_name):
                    hook = EventHook(source_object, method_name, event_type, self)
                    if hook.install():
                        self.active_hooks.append(hook)
        
        elif 'UnifiedAnalyzerSystem' in class_name:
            # Hook per UnifiedAnalyzerSystem
            unified_hooks = [
                ('process_tick', EventType.PERFORMANCE_METRICS),
                ('get_system_status', EventType.SYSTEM_STATUS),
                ('handle_error', EventType.ERROR_EVENT)
            ]
            
            for method_name, event_type in unified_hooks:
                if hasattr(source_object, method_name):
                    hook = EventHook(source_object, method_name, event_type, self)
                    if hook.install():
                        self.active_hooks.append(hook)
    
    def install_custom_hook(self, target_object: Any, method_name: str, 
                          event_type: EventType) -> bool:
        """
        Installa hook personalizzato
        
        Args:
            target_object: Oggetto target
            method_name: Nome metodo da hookare
            event_type: Tipo di evento da generare
            
        Returns:
            bool: True se installazione riuscita
        """
        hook = EventHook(target_object, method_name, event_type, self)
        if hook.install():
            self.active_hooks.append(hook)
            return True
        return False
    
    def emit_event(self, event: MLEvent) -> bool:
        """
        Emette un evento nel sistema
        
        Args:
            event: Evento da emettere
            
        Returns:
            bool: True se evento accettato
        """
        # Set session ID if not present
        if not event.session_id:
            event.session_id = self.session_id
        
        # Apply filters
        if not self._should_process_event(event):
            self.stats['total_events_filtered'] += 1
            return False
        
        # Apply rate limiting
        event_type_str = event.event_type.value if isinstance(event.event_type, EventType) else str(event.event_type)
        rate_limit = self.config.get_rate_limit(event_type_str)
        
        if not self.rate_limiter.is_allowed(event_type_str, rate_limit):
            self.stats['total_events_rate_limited'] += 1
            return False
        
        # Add to buffer
        if not self.event_buffer.add_event(event):
            # Buffer full - could implement emergency handling here
            return False
        
        # Update statistics
        self.stats['total_events_received'] += 1
        self.stats['events_by_type'][event_type_str] += 1
        
        source_str = event.source.value if isinstance(event.source, EventSource) else str(event.source)
        self.stats['events_by_source'][source_str] += 1
        self.stats['last_event_time'] = event.timestamp
        
        # Trigger callbacks if synchronous processing
        if not self.config.performance.enable_async_processing:
            self._trigger_callbacks(event)
        
        return True
    
    def _should_process_event(self, event: MLEvent) -> bool:
        """Verifica se evento deve essere processato"""
        
        # Check severity filter
        if not self.config.is_event_enabled(
            event.event_type.value if isinstance(event.event_type, EventType) else str(event.event_type),
            event.severity
        ):
            return False
        
        # Check source filter
        source_str = event.source.value if isinstance(event.source, EventSource) else str(event.source)
        if (self.config.event_filter.enabled_sources and 
            source_str not in self.config.event_filter.enabled_sources):
            return False
        
        if source_str in self.config.event_filter.disabled_sources:
            return False
        
        return True
    
    def _processing_loop(self):
        """Loop di processing asincrono degli eventi"""
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Get batch of events
                events = self.event_buffer.get_events(
                    self.config.performance.batch_processing_size
                )
                
                if events:
                    for event in events:
                        self._trigger_callbacks(event)
                
                # Sleep briefly to avoid busy waiting
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error in event processing loop: {e}")
                time.sleep(0.1)
    
    def _trigger_callbacks(self, event: MLEvent):
        """Triggera callback per un evento"""
        
        # Global callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Error in event callback: {e}")
        
        # Source-specific callbacks
        source_str = event.source.value if isinstance(event.source, EventSource) else str(event.source)
        for callback in self.source_callbacks.get(source_str, []):
            try:
                callback(event)
            except Exception as e:
                print(f"Error in source callback for {source_str}: {e}")
    
    def add_callback(self, callback: Callable[[MLEvent], None], 
                    source_filter: Optional[str] = None):
        """
        Aggiunge callback per eventi
        
        Args:
            callback: Funzione callback
            source_filter: Filtro per fonte specifica (opzionale)
        """
        if source_filter:
            self.source_callbacks[source_filter].append(callback)
        else:
            self.event_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[MLEvent], None], 
                       source_filter: Optional[str] = None):
        """Rimuove callback"""
        try:
            if source_filter:
                self.source_callbacks[source_filter].remove(callback)
            else:
                self.event_callbacks.remove(callback)
        except ValueError:
            pass
    
    def get_events(self, count: Optional[int] = None, 
                  event_type: Optional[EventType] = None,
                  source: Optional[EventSource] = None,
                  severity: Optional[EventSeverity] = None) -> List[MLEvent]:
        """
        Recupera eventi dal buffer con filtri opzionali
        
        Args:
            count: Numero massimo di eventi
            event_type: Filtro per tipo evento
            source: Filtro per fonte
            severity: Filtro per severità
            
        Returns:
            List[MLEvent]: Eventi filtrati
        """
        all_events = self.event_buffer.peek_events(count or 1000)
        
        filtered_events = []
        for event in all_events:
            # Apply filters
            if event_type and event.event_type != event_type:
                continue
            if source and event.source != source:
                continue
            if severity and event.severity != severity:
                continue
            
            filtered_events.append(event)
            
            if count and len(filtered_events) >= count:
                break
        
        return filtered_events
    
    def get_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche del collettore"""
        buffer_stats = self.event_buffer.get_stats()
        
        uptime = (datetime.now() - self.stats['collection_start_time']).total_seconds()
        
        return {
            'collector_stats': {
                'is_running': self.is_running,
                'session_id': self.session_id,
                'uptime_seconds': uptime,
                'registered_sources': len(self.registered_sources),
                'active_hooks': len(self.active_hooks),
                'event_callbacks': len(self.event_callbacks)
            },
            'event_stats': self.stats.copy(),
            'buffer_stats': buffer_stats,
            'rate_limiter_stats': {
                'tracked_event_types': len(self.rate_limiter.event_counts),
                'active_windows': sum(len(deque_obj) for deque_obj in self.rate_limiter.event_counts.values())
            }
        }
    
    def create_manual_event(self, event_type: EventType, data: Dict[str, Any],
                          severity: EventSeverity = EventSeverity.INFO,
                          asset: Optional[str] = None) -> MLEvent:
        """
        Crea evento manuale
        
        Args:
            event_type: Tipo di evento
            data: Dati evento
            severity: Severità
            asset: Asset associato
            
        Returns:
            MLEvent: Evento creato
        """
        event = MLEvent(
            event_type=event_type,
            source=EventSource.MANUAL,
            severity=severity,
            asset=asset,
            data=data,
            session_id=self.session_id
        )
        
        return event
    
    def emit_manual_event(self, event_type: EventType, data: Dict[str, Any],
                         severity: EventSeverity = EventSeverity.INFO,
                         asset: Optional[str] = None) -> bool:
        """
        Crea ed emette evento manuale
        
        Returns:
            bool: True se evento emesso con successo
        """
        event = self.create_manual_event(event_type, data, severity, asset)
        return self.emit_event(event)


# Utility functions
def create_learning_progress_event(progress_percent: float, asset: str, 
                                 additional_data: Optional[Dict[str, Any]] = None) -> MLEvent:
    """Crea evento di progresso apprendimento"""
    data = {
        'progress_percent': progress_percent,
        'timestamp': datetime.now().isoformat()
    }
    
    if additional_data:
        data.update(additional_data)
    
    return MLEvent(
        event_type=EventType.LEARNING_PROGRESS,
        source=EventSource.MANUAL,
        severity=EventSeverity.INFO,
        asset=asset,
        data=data
    )


def create_champion_change_event(old_champion: str, new_champion: str, 
                                model_type: str, asset: str) -> MLEvent:
    """Crea evento di cambio campione"""
    return MLEvent(
        event_type=EventType.CHAMPION_CHANGE,
        source=EventSource.MANUAL,
        severity=EventSeverity.INFO,
        asset=asset,
        data={
            'old_champion': old_champion,
            'new_champion': new_champion,
            'model_type': model_type,
            'change_timestamp': datetime.now().isoformat()
        }
    )


def create_emergency_stop_event(reason: str, asset: str, 
                               additional_context: Optional[Dict[str, Any]] = None) -> MLEvent:
    """Crea evento di emergency stop"""
    data = {
        'reason': reason,
        'stop_timestamp': datetime.now().isoformat()
    }
    
    if additional_context:
        data.update(additional_context)
    
    return MLEvent(
        event_type=EventType.EMERGENCY_STOP,
        source=EventSource.MANUAL,
        severity=EventSeverity.CRITICAL,
        asset=asset,
        data=data
    )


# Export main classes
__all__ = [
    'EventCollector',
    'MLEvent', 
    'EventType',
    'EventSource',
    'EventHook',
    'EventBuffer',
    'RateLimiter',
    'create_learning_progress_event',
    'create_champion_change_event', 
    'create_emergency_stop_event'
]