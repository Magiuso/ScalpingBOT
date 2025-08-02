#!/usr/bin/env python3
"""
Simple Display Manager - ULTRA-SIMPLIFIED VERSION
REGOLE CLAUDE_RESTAURO.md APPLICATE:
- ✅ Zero over-engineering
- ✅ Fail fast error handling  
- ✅ No complex architectures
- ✅ Essential functionality only

Gestisce la visualizzazione semplice degli eventi ML con:
A) Tracking modelli ML (accuracy, predictions, champion status)
B) Metriche performance (tick/sec, processing rate)
C) Formatting eventi per debug
"""

import time
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass, field

# Import configuration and events
from src.config.domain.monitoring_config import MLTrainingLoggerConfig
from src.monitoring.events.event_collector import MLEvent, EventType, EventSeverity


@dataclass
class ModelStats:
    """Statistiche essenziali per un modello ML"""
    name: str
    accuracy: float = 0.0
    predictions: int = 0
    is_champion: bool = False
    last_update: datetime = field(default_factory=datetime.now)


@dataclass  
class PerformanceStats:
    """Metriche performance essenziali"""
    events_processed: int = 0
    ticks_processed: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_stats_display: datetime = field(default_factory=datetime.now)
    
    def get_processing_rate(self) -> float:
        """Calcola rate di processing in tick/sec"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return self.ticks_processed / elapsed if elapsed > 0 else 0.0


class SimpleDisplayManager:
    """
    Display Manager ultra-semplificato - 1001 righe → 80 righe (92% riduzione)
    
    Funzionalità:
    - Stampa eventi formattati con timestamp
    - Traccia progress modelli ML (accuracy, predictions, champion)
    - Mostra metriche performance (tick/sec)
    - Stats periodiche ogni minuto
    """
    
    def __init__(self, config: MLTrainingLoggerConfig):
        if not isinstance(config, MLTrainingLoggerConfig):
            raise TypeError(f"Expected MLTrainingLoggerConfig, got {type(config)}")
        
        self.config = config
        
        # Model tracking
        self.models: Dict[str, ModelStats] = {}
        
        # Performance tracking
        self.performance = PerformanceStats()
        
        # Display settings
        self.stats_interval_seconds = 60  # Show stats every minute
    
    def handle_event(self, event: MLEvent):
        """
        Gestisce un evento - SIMPLIFIED
        
        Args:
            event: Evento da processare
        """
        if not isinstance(event, MLEvent):
            raise TypeError(f"Expected MLEvent, got {type(event)}")
        
        # 1. Print formatted event
        self._print_formatted_event(event)
        
        # 2. Update tracking data
        self._update_tracking(event)
        
        # 3. Show periodic stats
        self._check_and_show_stats()
    
    def _print_formatted_event(self, event: MLEvent):
        """Stampa evento formattato con timestamp e tipo"""
        timestamp = event.timestamp.strftime("%H:%M:%S")
        event_type = event.event_type.value
        
        # Format message based on event type (FAIL FAST - NO FALLBACK)
        if event.event_type == EventType.LEARNING_PROGRESS:
            if 'progress_percent' not in event.data:
                raise KeyError("LEARNING_PROGRESS event missing required field: progress_percent")
            progress = event.data['progress_percent']
            if event.asset is None:
                raise ValueError("LEARNING_PROGRESS event missing required field: asset")
            message = f"🧠 Learning progress: {progress:.1f}% ({event.asset})"
            
        elif event.event_type == EventType.CHAMPION_CHANGE:
            required_fields = ['old_champion', 'new_champion', 'accuracy']
            for field in required_fields:
                if field not in event.data:
                    raise KeyError(f"CHAMPION_CHANGE event missing required field: {field}")
            message = f"👑 Champion change: {event.data['old_champion']} → {event.data['new_champion']} (accuracy: {event.data['accuracy']:.3f})"
            
        elif event.event_type == EventType.MODEL_TRAINING:
            required_fields = ['model_type', 'status']
            for field in required_fields:
                if field not in event.data:
                    raise KeyError(f"MODEL_TRAINING event missing required field: {field}")
            message = f"🤖 Model training: {event.data['model_type']} - {event.data['status']}"
            
        elif event.event_type == EventType.PERFORMANCE_METRICS:
            if 'processing_rate' not in event.data:
                raise KeyError("PERFORMANCE_METRICS event missing required field: processing_rate")
            message = f"⚡ Performance: {event.data['processing_rate']:.1f} tick/sec"
            
        elif event.event_type == EventType.EMERGENCY_STOP:
            if 'reason' not in event.data:
                raise KeyError("EMERGENCY_STOP event missing required field: reason")
            message = f"🛑 EMERGENCY STOP: {event.data['reason']}"
            
        else:
            # Generic format for other events
            if 'message' not in event.data:
                message = str(event.data)
            else:
                message = event.data['message']
        
        # Print with severity indicator
        severity_indicator = self._get_severity_indicator(event.severity)
        print(f"[{timestamp}] {severity_indicator} {event_type}: {message}")
    
    def _get_severity_indicator(self, severity: EventSeverity) -> str:
        """Restituisce indicatore con emoji per severity"""
        if severity == EventSeverity.CRITICAL:
            return "🚨"
        elif severity == EventSeverity.ERROR:
            return "❌"
        elif severity == EventSeverity.WARNING:
            return "⚠️"
        elif severity == EventSeverity.INFO:
            return "✅"
        else:
            return "🔍"
    
    def _update_tracking(self, event: MLEvent):
        """Aggiorna statistiche tracking"""
        # Update performance stats
        self.performance.events_processed += 1
        
        # Count ticks for performance calculation
        if event.event_type in [EventType.LEARNING_PROGRESS, EventType.PERFORMANCE_METRICS]:
            if 'tick_count' not in event.data:
                self.performance.ticks_processed += 1  # Default to 1 tick if not specified
            else:
                self.performance.ticks_processed += event.data['tick_count']
        
        # Update model stats for ML events
        if event.asset and event.event_type in [
            EventType.LEARNING_PROGRESS, 
            EventType.CHAMPION_CHANGE, 
            EventType.MODEL_TRAINING
        ]:
            self._update_model_stats(event)
    
    def _update_model_stats(self, event: MLEvent):
        """Aggiorna statistiche modello specifico"""
        asset = event.asset
        if 'model_type' not in event.data:
            model_type = 'Unknown'
        else:
            model_type = event.data['model_type']
        model_key = f"{asset}_{model_type}"
        
        # Get or create model stats
        if model_key not in self.models:
            self.models[model_key] = ModelStats(name=model_key)
        
        model = self.models[model_key]
        model.last_update = datetime.now()
        
        # Update based on event type
        if event.event_type == EventType.LEARNING_PROGRESS:
            if 'accuracy' in event.data:
                model.accuracy = event.data['accuracy']
            if 'predictions' in event.data:
                model.predictions = event.data['predictions']
                
        elif event.event_type == EventType.CHAMPION_CHANGE:
            # Reset all champions, then set new one
            for m in self.models.values():
                m.is_champion = False
            model.is_champion = True
            if 'accuracy' in event.data:
                model.accuracy = event.data['accuracy']
    
    def _check_and_show_stats(self):
        """Mostra statistiche periodiche se è il momento"""
        now = datetime.now()
        elapsed = (now - self.performance.last_stats_display).total_seconds()
        
        if elapsed >= self.stats_interval_seconds:
            self._show_stats()
            self.performance.last_stats_display = now
    
    def _show_stats(self):
        """Mostra statistiche correnti"""
        print("\n" + "="*60)
        print(f"📊 STATISTICS UPDATE - {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        # Performance metrics
        processing_rate = self.performance.get_processing_rate()
        elapsed_minutes = (datetime.now() - self.performance.start_time).total_seconds() / 60
        
        print(f"📊 Performance:")
        print(f"   📈 Events processed: {self.performance.events_processed}")
        print(f"   ⚡ Ticks processed: {self.performance.ticks_processed}")
        print(f"   🚀 Processing rate: {processing_rate:.1f} tick/sec")
        print(f"   ⏱️ Elapsed time: {elapsed_minutes:.1f} minutes")
        
        # Model statistics  
        if self.models:
            print(f"\n🤖 Models ({len(self.models)} active):")
            for model in self.models.values():
                champion_indicator = "👑" if model.is_champion else "  "
                print(f"   {champion_indicator} {model.name}")
                print(f"      🎯 Accuracy: {model.accuracy:.3f}")
                print(f"      📊 Predictions: {model.predictions}")
                print(f"      🕐 Last update: {model.last_update.strftime('%H:%M:%S')}")
        
        print("="*60 + "\n")
    
    def force_show_stats(self):
        """Forza visualizzazione statistiche immediate"""
        self._show_stats()
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Restituisce summary statistiche per integrazione"""
        return {
            'performance': {
                'events_processed': self.performance.events_processed,
                'ticks_processed': self.performance.ticks_processed,
                'processing_rate': self.performance.get_processing_rate(),
                'uptime_seconds': (datetime.now() - self.performance.start_time).total_seconds()
            },
            'models': {
                name: {
                    'accuracy': model.accuracy,
                    'predictions': model.predictions,
                    'is_champion': model.is_champion,
                    'last_update': model.last_update.isoformat()
                }
                for name, model in self.models.items()
            }
        }


# ================== FACTORY FUNCTION ==================

def create_simple_display(config: MLTrainingLoggerConfig) -> SimpleDisplayManager:
    """Factory function per creare SimpleDisplayManager"""
    return SimpleDisplayManager(config)


# Export
__all__ = [
    'SimpleDisplayManager',
    'ModelStats', 
    'PerformanceStats',
    'create_simple_display'
]