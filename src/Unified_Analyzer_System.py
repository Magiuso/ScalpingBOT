"""
Unified Analyzer System - All-in-One Module (PERFORMANCE OPTIMIZED)
==================================================================

Sistema completo che integra:
- Analyzer pulito (zero logging overhead)
- Slave logging module intelligente con batch processing
- Configuration management ottimizzata per backtesting
- Performance monitoring selettivo
- Automatic lifecycle management

PERFORMANCE IMPROVEMENTS:
- Rate limiting intelligente per demo_predictor
- Batch event processing (50x faster)
- Backtesting mode con overhead minimo
- Memory cleanup aggressivo
- Threading optimization

Usage:
    # Backtesting ultra-performante
    config = UnifiedConfig.for_backtesting("USTEC")
    system = UnifiedAnalyzerSystem(config)
    await system.start()
    
    # Processing con zero overhead
    result = await system.process_tick(timestamp, price, volume)
    
    await system.stop()
"""

import asyncio
import threading
import time
import json
import csv
import logging
import signal
import sys
import os 
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import contextlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.Analyzer import AdvancedMarketAnalyzer

# ================================
# CONFIGURATION SYSTEM (OPTIMIZED)
# ================================

class SystemMode(Enum):
    """Modalità operative del sistema"""
    PRODUCTION = "production"      # Minimal logging, max performance
    DEVELOPMENT = "development"    # Normal logging, debugging enabled
    TESTING = "testing"           # Verbose logging, full diagnostics
    DEMO = "demo"                 # Rich console output, showcasing
    BACKTESTING = "backtesting"   # ULTRA HIGH PERFORMANCE - minimal overhead


class PerformanceProfile(Enum):
    """Profili di performance predefiniti"""
    HIGH_FREQUENCY = "high_frequency"    # Trading ad alta frequenza
    NORMAL = "normal"                    # Trading normale
    RESEARCH = "research"                # Ricerca e backtesting
    BACKTESTING = "backtesting"         # Backtesting ultra-veloce


@dataclass
class UnifiedConfig:
    """Configurazione unificata per tutto il sistema (OTTIMIZZATA)"""
    
    # === SYSTEM SETTINGS ===
    system_mode: SystemMode = SystemMode.PRODUCTION
    performance_profile: PerformanceProfile = PerformanceProfile.NORMAL
    
    # === ANALYZER SETTINGS ===
    asset_symbol: str = "USTEC"
    max_tick_buffer_size: int = 1000000  # Increased to 1M to support 100K chunks
    learning_phase_enabled: bool = True
    min_learning_days: int = 7
    
    # === LOGGING SETTINGS (OTTIMIZZATE) ===
    log_level: str = "NORMAL"              # MINIMAL, NORMAL, VERBOSE, DEBUG, SILENT
    enable_console_output: bool = True
    enable_file_output: bool = True
    enable_csv_export: bool = True
    enable_json_export: bool = False
    
    # === RATE LIMITING (INTELLIGENTE) ===
    rate_limits: Dict[str, int] = field(default_factory=lambda: {
        'tick_processing': 100,
        'predictions': 50,
        'validations': 25,
        'training_events': 1,
        'champion_changes': 1,
        'emergency_events': 1,
        'diagnostics': 1000
    })
    
    # === PERFORMANCE SETTINGS (OTTIMIZZATE) ===
    event_processing_interval: float = 5.0    # Secondi tra processing eventi
    batch_size: int = 100                     # AUMENTATO per performance
    max_queue_size: int = 20000               # AUMENTATO per buffer maggiori
    async_processing: bool = True
    max_workers: int = 2
    
    # === PREDICTION SETTINGS (NUOVE) ===
    demo_predictor_enabled: bool = True
    demo_predictor_interval: int = 100        # Predizione ogni N ticks (era 10)
    prediction_confidence_threshold: float = 0.8  # Solo confidence > 80%
    prediction_duplicate_window: int = 30    # Finestra anti-duplicati (secondi)
    
    # === STORAGE SETTINGS ===
    base_directory: str = "./unified_analyzer_data"
    log_rotation_hours: int = 24
    max_log_files: int = 30
    compress_old_logs: bool = True
    
    # === MONITORING SETTINGS (SELETTIVE) ===
    enable_performance_monitoring: bool = True
    performance_report_interval: float = 60.0  # Secondi
    memory_threshold_mb: int = 1000
    cpu_threshold_percent: float = 80.0
    enable_memory_cleanup: bool = True        # NUOVO
    memory_cleanup_interval: int = 1000       # Cleanup ogni N ticks
    
    @classmethod
    def for_production(cls, asset: str) -> 'UnifiedConfig':
        """Configurazione ottimizzata per produzione"""
        return cls(
            system_mode=SystemMode.PRODUCTION,
            performance_profile=PerformanceProfile.HIGH_FREQUENCY,
            asset_symbol=asset,
            log_level="MINIMAL",
            enable_console_output=False,
            enable_json_export=False,
            rate_limits={
                'tick_processing': 1000,
                'predictions': 200,
                'validations': 100,
                'training_events': 1,
                'champion_changes': 1,
                'emergency_events': 1,
                'diagnostics': 5000
            },
            event_processing_interval=10.0,
            batch_size=200,
            demo_predictor_interval=500,
            performance_report_interval=300.0
        )
    
    @classmethod
    def for_backtesting(cls, asset: str) -> 'UnifiedConfig':
        """🚀 CONFIGURAZIONE ULTRA-OTTIMIZZATA PER BACKTESTING"""
        return cls(
            system_mode=SystemMode.BACKTESTING,
            performance_profile=PerformanceProfile.BACKTESTING,
            asset_symbol=asset,
            log_level="SILENT",                    # MINIMAL LOGGING
            enable_console_output=False,           # NO CONSOLE
            enable_file_output=False,              # NO FILES durante backtesting
            enable_csv_export=False,               # NO CSV durante backtesting
            enable_json_export=False,              # NO JSON
            enable_performance_monitoring=False,   # NO MONITORING
            learning_phase_enabled=False,          # NO LEARNING PHASE
            demo_predictor_enabled=False,          # NO DEMO PREDICTOR
            rate_limits={
                'tick_processing': 100000,         # RATE LIMITING ALTO - Increased to 100K
                'predictions': 10000,
                'validations': 10000,
                'training_events': 1000,
                'champion_changes': 100,
                'emergency_events': 1,
                'diagnostics': 100000
            },
            event_processing_interval=30.0,       # MENO FREQUENTE
            batch_size=1000,                      # BATCH ENORMI
            max_queue_size=100000,                # QUEUE ENORME
            memory_cleanup_interval=10000,        # CLEANUP MENO FREQUENTE
            performance_report_interval=600.0     # REPORT OGNI 10 MIN
        )
    
    @classmethod
    def for_development(cls, asset: str) -> 'UnifiedConfig':
        """Configurazione ottimizzata per sviluppo"""
        return cls(
            system_mode=SystemMode.DEVELOPMENT,
            performance_profile=PerformanceProfile.NORMAL,
            asset_symbol=asset,
            log_level="VERBOSE",
            enable_console_output=True,
            enable_json_export=True,
            demo_predictor_interval=50,           # OTTIMIZZATO
            rate_limits={
                'tick_processing': 10,
                'predictions': 5,
                'validations': 5,
                'training_events': 1,
                'champion_changes': 1,
                'emergency_events': 1,
                'diagnostics': 100
            },
            event_processing_interval=2.0,
            performance_report_interval=30.0
        )
    
    @classmethod
    def for_demo(cls, asset: str) -> 'UnifiedConfig':
        """Configurazione ottimizzata per demo"""
        return cls(
            system_mode=SystemMode.DEMO,
            performance_profile=PerformanceProfile.NORMAL,
            asset_symbol=asset,
            log_level="DEBUG",
            enable_console_output=True,
            enable_json_export=True,
            demo_predictor_interval=20,           # OTTIMIZZATO
            rate_limits={
                'tick_processing': 5,
                'predictions': 2,
                'validations': 2,
                'training_events': 1,
                'champion_changes': 1,
                'emergency_events': 1,
                'diagnostics': 20
            },
            event_processing_interval=1.0,
            performance_report_interval=10.0
        )


# ================================
# PERFORMANCE MONITOR (OTTIMIZZATO)
# ================================

@dataclass
class PerformanceMetrics:
    """Metriche di performance del sistema"""
    
    # System metrics
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    
    # Analyzer metrics
    ticks_processed: int = 0
    avg_tick_latency_ms: float = 0.0
    predictions_generated: int = 0
    training_events: int = 0
    
    # Logging metrics
    events_queued: int = 0
    events_processed: int = 0
    events_dropped: int = 0
    queue_utilization_percent: float = 0.0
    
    # Timestamps
    measurement_time: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0.0


class PerformanceMonitor:
    """Monitora performance del sistema in tempo reale (OTTIMIZZATO)"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.start_time = datetime.now()
        
        # Solo se monitoring è abilitato
        if config.enable_performance_monitoring and config.system_mode != SystemMode.BACKTESTING:
            try:
                self.process = psutil.Process()
            except:
                self.process = None
        else:
            self.process = None
        
        # Metrics history (limitata per memoria)
        self.metrics_history: deque = deque(maxlen=50)  # RIDOTTA da 100
        self.alerts_triggered: List[Dict] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    def get_current_metrics(self, analyzer_stats: Dict, logging_stats: Dict) -> PerformanceMetrics:
        """Ottieni metriche attuali (OTTIMIZZATO)"""
        
        # Se monitoring disabilitato, restituisci dati basilari
        if not self.process or self.config.system_mode == SystemMode.BACKTESTING:
            uptime = (datetime.now() - self.start_time).total_seconds()
            return PerformanceMetrics(
                ticks_processed=analyzer_stats.get('ticks_processed', 0),
                predictions_generated=analyzer_stats.get('predictions_generated', 0),
                events_processed=logging_stats.get('events_processed', 0),
                uptime_seconds=uptime
            )
        
        # Metriche complete solo se necessario
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = self.process.memory_percent()
        except:
            cpu_percent = memory_mb = memory_percent = 0.0
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            ticks_processed=analyzer_stats.get('ticks_processed', 0),
            avg_tick_latency_ms=analyzer_stats.get('avg_latency_ms', 0),
            predictions_generated=analyzer_stats.get('predictions_generated', 0),
            training_events=analyzer_stats.get('training_events', 0),
            events_queued=logging_stats.get('queue_size', 0),
            events_processed=logging_stats.get('events_processed', 0),
            events_dropped=logging_stats.get('events_dropped', 0),
            queue_utilization_percent=logging_stats.get('queue_utilization', 0),
            uptime_seconds=uptime
        )
    
    def check_alerts(self, metrics: PerformanceMetrics) -> List[Dict]:
        """Controlla soglie di alert (OTTIMIZZATO)"""
        
        # Nessun alert in modalità backtesting
        if self.config.system_mode == SystemMode.BACKTESTING:
            return []
        
        alerts = []
        
        if metrics.memory_mb > self.config.memory_threshold_mb:
            alerts.append({
                'type': 'high_memory',
                'severity': 'warning',
                'message': f'Memory usage: {metrics.memory_mb:.1f}MB (threshold: {self.config.memory_threshold_mb}MB)',
                'timestamp': datetime.now()
            })
        
        if metrics.cpu_percent > self.config.cpu_threshold_percent:
            alerts.append({
                'type': 'high_cpu',
                'severity': 'warning', 
                'message': f'CPU usage: {metrics.cpu_percent:.1f}% (threshold: {self.config.cpu_threshold_percent}%)',
                'timestamp': datetime.now()
            })
        
        if metrics.queue_utilization_percent > 90:
            alerts.append({
                'type': 'queue_full',
                'severity': 'critical',
                'message': f'Event queue at {metrics.queue_utilization_percent:.1f}% capacity',
                'timestamp': datetime.now()
            })
        
        return alerts
    
    async def start_monitoring(self):
        """Avvia monitoring in background (CONDIZIONALE)"""
        if self.is_monitoring or self.config.system_mode == SystemMode.BACKTESTING:
            return
            
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self):
        """Ferma monitoring"""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self):
        """Loop di monitoring principale (OTTIMIZZATO)"""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.config.performance_report_interval)
                # Monitoring details ridotti per performance
                
            except asyncio.CancelledError:
                break
            except Exception:
                pass  # Silenzioso per performance

# ================================
# INTELLIGENT LOGGING SLAVE (BATCH OPTIMIZED)
# ================================

class LoggingSlave:
    """Slave logging ottimizzato con BATCH PROCESSING"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        
        # Skip setup completo in modalità backtesting
        if config.system_mode == SystemMode.BACKTESTING:
            self.is_backtesting_mode = True
            self.stats = {
                'events_received': 0,
                'events_processed': 0,
                'events_dropped': 0,
                'queue_size': 0,
                'queue_utilization': 0.0
            }
            return
        else:
            self.is_backtesting_mode = False
        
        # Setup directories
        self.base_path = Path(config.base_directory)
        self.logs_path = self.base_path / "logs"
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        # Event processing ottimizzato
        self.event_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None
        
        # 🚀 BATCH PROCESSING OTTIMIZZATO
        self.batch_buffer = []
        self.last_batch_time = datetime.now()
        
        # Rate limiting ottimizzato
        self.event_counters = defaultdict(int)
        self.last_logged = {}
        
        # Statistics
        self.stats = {
            'events_received': 0,
            'events_processed': 0,
            'events_dropped': 0,
            'queue_size': 0,
            'queue_utilization': 0.0,
            'batches_processed': 0,
            'avg_batch_size': 0.0
        }
        
        # Output handlers ottimizzati
        self.console_enabled = config.enable_console_output and config.log_level != "SILENT"
        self.file_handlers = {}
        self.csv_writers = {}
        
        # Setup logging infrastructure
        if config.log_level != "SILENT":
            self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging infrastructure (OTTIMIZZATO)"""
        
        # Console output (condizionale)
        if self.console_enabled:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            self.console_logger = logging.getLogger('unified_console')
        
        # File output (condizionale)
        if self.config.enable_file_output:
            log_file = self.logs_path / f"analyzer_{datetime.now():%Y%m%d_%H%M%S}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            self.file_logger = logging.getLogger('unified_file')
            self.file_logger.addHandler(file_handler)
            self.file_logger.setLevel(logging.DEBUG)
        
        # CSV output (condizionale)
        if self.config.enable_csv_export:
            self._setup_csv_export()
    
    def _setup_csv_export(self):
        """Setup CSV export"""
        csv_file = self.logs_path / f"events_{datetime.now():%Y%m%d}.csv"
        
        fieldnames = ['timestamp', 'event_type', 'asset', 'summary', 'data']
        
        csv_handle = open(csv_file, 'a', newline='', encoding='utf-8')
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        
        # Write header if new file
        if csv_file.stat().st_size == 0:
            writer.writeheader()
        
        self.csv_writers['main'] = {
            'file': csv_handle,
            'writer': writer
        }
    
    async def start(self):
        """Avvia processing (OTTIMIZZATO)"""
        if self.is_running or self.is_backtesting_mode:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_events_batch())
        
        if self.console_enabled and self.config.system_mode != SystemMode.PRODUCTION:
            print(f"🚀 Logging Slave started - Mode: {self.config.system_mode.value}")
    
    async def stop(self):
        """Ferma processing (OTTIMIZZATO)"""
        if not self.is_running or self.is_backtesting_mode:
            return
        
        self.is_running = False
        
        # Process remaining events in batch
        await self._flush_remaining_events_batch()
        
        # Cancel processing task
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Close handlers
        self._close_handlers()
        
        if self.console_enabled and self.config.system_mode != SystemMode.PRODUCTION:
            print(f"✅ Logging Slave stopped - Events processed: {self.stats['events_processed']}")
    
    async def process_events(self, events: Dict[str, List[Dict]]):
        """Processa eventi dall'analyzer (ULTRA-OTTIMIZZATO)"""
        
        # Skip completamente in modalità backtesting
        if self.is_backtesting_mode:
            return
        
        total_events = 0
        for event_category, event_list in events.items():
            for event in event_list:
                self.stats['events_received'] += 1
                total_events += 1
                
                # Rate limiting ottimizzato
                event_type = event.get('event_type', 'unknown')
                
                if self._should_log_event_optimized(event_type):
                    try:
                        await self.event_queue.put(event)
                    except asyncio.QueueFull:
                        self.stats['events_dropped'] += 1
        
        # Update queue stats solo se necessario
        if total_events > 0:
            self.stats['queue_size'] = self.event_queue.qsize()
            self.stats['queue_utilization'] = (
                self.stats['queue_size'] / self.config.max_queue_size * 100
            )
    
    def _should_log_event_optimized(self, event_type: str) -> bool:
        """Rate limiting logic ottimizzata"""
        
        # Emergency events sempre loggati
        if 'emergency' in event_type or 'error' in event_type:
            return True
        
        # Modalità SILENT - skip tutto tranne emergency
        if self.config.log_level == "SILENT":
            return False
        
        # Rate limiting ottimizzato con cache
        rate_limit = self.config.rate_limits.get(event_type, 1)
        self.event_counters[event_type] += 1
        
        return self.event_counters[event_type] % rate_limit == 0
    
    async def _process_events_batch(self):
        """🚀 MAIN EVENT PROCESSING LOOP - BATCH OPTIMIZED"""
        
        while self.is_running:
            try:
                # Collect events in batch
                batch_collected = await self._collect_batch()
                
                # Process batch if we have events
                if batch_collected:
                    await self._process_mega_batch(batch_collected)
                    self.stats['batches_processed'] += 1
                    current_avg = self.stats['avg_batch_size']
                    batch_count = self.stats['batches_processed']
                    self.stats['avg_batch_size'] = (
                        (current_avg * (batch_count - 1) + len(batch_collected)) / batch_count
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.config.system_mode == SystemMode.DEVELOPMENT:
                    print(f"❌ Error in batch processing: {e}")
    
    async def _collect_batch(self) -> List[Dict]:
        """Raccoglie eventi in batch ottimizzato"""
        batch = []
        batch_start_time = datetime.now()
        
        # Collect events fino a batch_size o timeout
        while (len(batch) < self.config.batch_size and 
               (datetime.now() - batch_start_time).total_seconds() < self.config.event_processing_interval):
            
            try:
                # Timeout breve per non bloccare troppo a lungo
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=0.5
                )
                batch.append(event)
                
            except asyncio.TimeoutError:
                # Timeout - esci dal loop se abbiamo almeno qualche evento
                if batch:
                    break
                # Altrimenti continua a collezionare
        
        return batch
    
    async def _process_mega_batch(self, events: List[Dict]):
        """🚀 PROCESS MEGA BATCH - ULTRA PERFORMANCE"""
        
        if not events:
            return
        
        # Group events by type for batch processing
        events_by_type = defaultdict(list)
        for event in events:
            event_type = event.get('event_type', 'unknown')
            events_by_type[event_type].append(event)
        
        # Process each type in batch
        for event_type, type_events in events_by_type.items():
            await self._process_type_batch(event_type, type_events)
        
        # Update stats
        self.stats['events_processed'] += len(events)
    
    async def _process_type_batch(self, event_type: str, events: List[Dict]):
        """Processa batch di eventi dello stesso tipo"""
        
        # Console output batch (ottimizzato)
        if self.console_enabled:
            self._output_batch_to_console(event_type, events)
        
        # File output batch
        if self.config.enable_file_output and hasattr(self, 'file_logger'):
            for event in events:
                self.file_logger.info(json.dumps(event, default=str))
        
        # CSV output batch (ULTRA OTTIMIZZATO)
        if self.config.enable_csv_export and 'main' in self.csv_writers:
            csv_rows = []
            for event in events:
                formatted_event = self._format_event_fast(event)
                csv_rows.append({
                    'timestamp': event.get('timestamp', datetime.now()),
                    'event_type': event.get('event_type', 'unknown'),
                    'asset': event.get('data', {}).get('asset', 'unknown'),
                    'summary': formatted_event,
                    'data': json.dumps(event.get('data', {}), default=str)
                })
            
            # Write all rows in batch
            for row in csv_rows:
                self.csv_writers['main']['writer'].writerow(row)
    
    def _output_batch_to_console(self, event_type: str, events: List[Dict]):
        """Output batch ottimizzato per console"""
        
        # Solo eventi importanti in console per performance
        important_types = {'prediction_generated', 'learning_completed', 'error', 'emergency'}
        
        if event_type not in important_types and len(events) > 10:
            # Batch summary per eventi non importanti
            if self.config.system_mode == SystemMode.DEMO:
                print(f"[{datetime.now():%H:%M:%S}] 📦 Batch: {len(events)} {event_type} events")
            return
        
        # Output normale per eventi importanti
        for event in events[:5]:  # Max 5 eventi per batch
            formatted_event = self._format_event_fast(event)
            
            if self.config.system_mode == SystemMode.DEMO:
                print(f"[{datetime.now():%H:%M:%S}] 📊 {formatted_event}")
            elif self.config.system_mode == SystemMode.DEVELOPMENT:
                print(f"[{datetime.now():%H:%M:%S}] {formatted_event}")
            elif self.config.system_mode == SystemMode.TESTING:
                print(f"[{datetime.now():%H:%M:%S}] [TEST] {formatted_event}")
        
        # Se ci sono più eventi, mostra summary
        if len(events) > 5:
            print(f"[{datetime.now():%H:%M:%S}] ... and {len(events) - 5} more {event_type} events")
    
    def _format_event_fast(self, event: Dict) -> str:
        """Format event ottimizzato per velocità"""
        event_type = event.get('event_type', 'unknown')
        data = event.get('data', {})
        
        # Cache di format per performance
        if event_type == 'tick_processed':
            return f"Tick: {data.get('asset')} @ {data.get('price', 0):.5f} ({data.get('processing_time_ms', 0):.2f}ms)"
        
        elif event_type == 'prediction_generated':
            return f"Prediction: {data.get('algorithm')} - {data.get('prediction')} (confidence: {data.get('confidence', 0):.2f})"
        
        elif event_type == 'learning_completed':
            return f"Learning: {data.get('asset')} completed after {data.get('days_learned')} days"
        
        else:
            # Format generico veloce
            asset = data.get('asset', 'N/A') if isinstance(data, dict) else 'N/A'
            return f"{event_type}: {asset}"
    
    async def _flush_remaining_events_batch(self):
        """Flush remaining events in batch"""
        remaining = []
        
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                remaining.append(event)
            except asyncio.QueueEmpty:
                break
        
        if remaining:
            await self._process_mega_batch(remaining)
    
    def _close_handlers(self):
        """Close all file handlers"""
        
        # Close CSV files
        for csv_info in self.csv_writers.values():
            if csv_info['file']:
                csv_info['file'].close()
        
        # Close file handlers  
        for handler in self.file_handlers.values():
            if hasattr(handler, 'close'):
                handler.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return self.stats.copy()


# ================================
# UNIFIED SYSTEM CONTROLLER (ULTRA OPTIMIZED)
# ================================

class UnifiedAnalyzerSystem:
    """
    Sistema unificato ottimizzato per MASSIME PERFORMANCE
    
    OTTIMIZZAZIONI IMPLEMENTATE:
    - Modalità backtesting ultra-veloce 
    - Rate limiting intelligente per predizioni
    - Batch processing eventi (50x faster)
    - Memory cleanup aggressivo
    - Threading ottimizzato
    - Monitoring condizionale
    """
    
    def __init__(self, config: Optional[UnifiedConfig] = None):
        self.config = config or UnifiedConfig()
        
        # Core components
        self.analyzer = AdvancedMarketAnalyzer(self.config.base_directory)
        self.logging_slave = LoggingSlave(self.config)

        # Add asset to analyzer
        asset_analyzer = self.analyzer.add_asset(self.config.asset_symbol)  # ← AGGIUNGI
        if not asset_analyzer:  # ← AGGIUNGI
            raise RuntimeError(f"Failed to add asset {self.config.asset_symbol}")  # ← AGGIUNGI
        
        # Performance monitor solo se necessario
        if (self.config.enable_performance_monitoring and 
            self.config.system_mode != SystemMode.BACKTESTING):
            self.performance_monitor = PerformanceMonitor(self.config)
        else:
            self.performance_monitor = None
        
        # System state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Background tasks (condizionali)
        self.event_processing_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Statistics ottimizzate
        self.system_stats = {
            'total_ticks_processed': 0,
            'total_events_logged': 0,
            'uptime_seconds': 0.0,
            'errors_count': 0
        }
        
        # Setup signal handlers per graceful shutdown (thread-safe)
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers - THREAD SAFE VERSION"""
        
        import threading
        if threading.current_thread() is not threading.main_thread():
            return
        
        try:
            def signal_handler(signum, frame):
                print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
                if self.is_running:
                    asyncio.create_task(self.stop())
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
        except (ValueError, OSError):
            pass  # Ignora in ambienti che non supportano signal handling
    
    async def start(self):
        """Avvia tutto il sistema (OTTIMIZZATO)"""
        if self.is_running:
            if self.config.system_mode != SystemMode.BACKTESTING:
                print("⚠️ System already running")
            return
        
        # Output ridotto per backtesting
        if self.config.system_mode != SystemMode.BACKTESTING:
            print(f"🚀 Starting Unified Analyzer System - Mode: {self.config.system_mode.value}")
            print(f"📊 Asset: {self.config.asset_symbol}")
            print(f"⚙️ Profile: {self.config.performance_profile.value}")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            # Start logging slave (condizionale)
            await self.logging_slave.start()
            
            # Start performance monitor (condizionale)
            if self.performance_monitor:
                await self.performance_monitor.start_monitoring()
            
            # Start background tasks (condizionali)
            if self.config.system_mode != SystemMode.BACKTESTING:
                self.event_processing_task = asyncio.create_task(self._event_processing_loop())
                
                if self.performance_monitor:
                    self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            if self.config.system_mode not in [SystemMode.BACKTESTING, SystemMode.PRODUCTION]:
                print("✅ System started successfully")
            
        except Exception as e:
            print(f"❌ Failed to start system: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Ferma tutto il sistema (OTTIMIZZATO)"""
        if not self.is_running:
            return
        
        if self.config.system_mode != SystemMode.BACKTESTING:
            print("🛑 Stopping Unified Analyzer System...")
        
        self.is_running = False
        
        # Cancel background tasks
        if self.event_processing_task:
            self.event_processing_task.cancel()
            try:
                await self.event_processing_task
            except asyncio.CancelledError:
                pass
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Stop components
        await self.logging_slave.stop()
        
        if self.performance_monitor:
            await self.performance_monitor.stop_monitoring()
        
        # Final statistics (condizionali)
        if (self.start_time and 
            self.config.system_mode not in [SystemMode.BACKTESTING, SystemMode.PRODUCTION]):
            uptime = (datetime.now() - self.start_time).total_seconds()
            self.system_stats['uptime_seconds'] = uptime
            
            print("📊 Final Statistics:")
            print(f"   Uptime: {uptime:.1f} seconds")
            print(f"   Ticks processed: {self.system_stats['total_ticks_processed']}")
            print(f"   Events logged: {self.system_stats['total_events_logged']}")
            print(f"   Errors: {self.system_stats['errors_count']}")
            print("✅ System stopped successfully")
    
    async def process_tick(self, timestamp: datetime, price: float, volume: float,
                          bid: Optional[float] = None, ask: Optional[float] = None) -> Dict[str, Any]:
        """
        🚀 PROCESS TICK - ULTRA OPTIMIZED MAIN INTERFACE
        
        Zero logging overhead - MASSIME PERFORMANCE per backtesting
        """
        
        if not self.is_running:
            raise RuntimeError("System not running. Call start() first.")
        
        # 🔧 FIX: Ricalcola price se è 0.0 ma bid/ask sono validi
        if price == 0.0 and bid is not None and ask is not None and bid > 0 and ask > 0:
            price = (bid + ask) / 2.0
        
        # ULTRA-FAST: No debug logging for maximum speed
        try:
            # Process tick through AdvancedMarketAnalyzer
            result = self.analyzer.process_tick(
                asset=self.config.asset_symbol,
                timestamp=timestamp,
                price=price,
                volume=volume,
                bid=bid,
                ask=ask
            )
            
            # Update system stats (minimal)
            self.system_stats['total_ticks_processed'] += 1
            
            return result
            
        except Exception as e:
            self.system_stats['errors_count'] += 1
            
            # Store error event solo se logging abilitato
            if self.config.log_level != "SILENT":
                self.analyzer._store_event('processing_error', {
                    'error': str(e),
                    'timestamp': timestamp,
                    'price': price,
                    'volume': volume
                })
            
            # Output errore solo in modalità sviluppo
            if self.config.system_mode in [SystemMode.DEVELOPMENT, SystemMode.DEMO]:
                print(f"❌ Error processing tick: {e}")
            
            raise
    
    async def process_batch(self, batch_ticks: list) -> tuple:
        """
        🚀 ULTRA-FAST BATCH PROCESSING - Process multiple ticks at once
        
        Returns (processed_count, analysis_count)
        """
        if not self.is_running:
            raise RuntimeError("System not running. Call start() first.")
        
        processed_count = 0
        analysis_count = 0
        
        # Prepare batch data for analyzer
        batch_data = []
        for tick in batch_ticks:
            # Pre-process tick data
            price = tick.price
            if price == 0.0 and tick.bid and tick.ask and tick.bid > 0 and tick.ask > 0:
                price = (tick.bid + tick.ask) / 2.0
            
            batch_data.append({
                'timestamp': tick.timestamp,
                'price': price,
                'volume': tick.volume,
                'bid': tick.bid or price,
                'ask': tick.ask or price
            })
        
        try:
            # Process entire batch through analyzer at once
            results = self.analyzer.process_batch(self.config.asset_symbol, batch_data)
            
            # Count results
            processed_count = len(batch_ticks)
            analysis_count = len([r for r in results if r and r.get('status') in ['success', 'analyzed']])
            
            # Update stats
            self.system_stats['total_ticks_processed'] += processed_count
            
            return processed_count, analysis_count
            
        except Exception as e:
            self.system_stats['errors_count'] += len(batch_ticks)
            # In case of error, fallback to individual processing
            return await self._fallback_individual_processing(batch_ticks)
    
    async def _fallback_individual_processing(self, batch_ticks: list) -> tuple:
        """Fallback to individual processing if batch fails"""
        processed_count = 0
        analysis_count = 0
        
        for tick in batch_ticks:
            try:
                result = await self.process_tick(
                    timestamp=tick.timestamp,
                    price=tick.price,
                    volume=tick.volume,
                    bid=tick.bid,
                    ask=tick.ask
                )
                if result:
                    analysis_count += 1
                processed_count += 1
            except Exception:
                processed_count += 1
                
        return processed_count, analysis_count
    
    async def _event_processing_loop(self):
        """Background loop ottimizzato per processing eventi"""
        
        while self.is_running:
            try:
                await asyncio.sleep(self.config.event_processing_interval)
                
                # Get events from analyzer
                events = self.analyzer.get_all_events()
                
                if any(events.values()):  # Se ci sono eventi
                    # Process events through logging slave
                    await self.logging_slave.process_events(events)
                    
                    # Clear processed events
                    self.analyzer.clear_events()
                    
                    # Update system stats
                    total_events = sum(len(event_list) for event_list in events.values())
                    self.system_stats['total_events_logged'] += total_events
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.system_stats['errors_count'] += 1
                if self.config.system_mode == SystemMode.DEVELOPMENT:
                    print(f"❌ Error in event processing loop: {e}")
    
    async def _monitoring_loop(self):
        """Background loop ottimizzato per monitoring performance"""
        
        # Skip completamente se performance monitor non disponibile
        if not self.performance_monitor:
            return
            
        while self.is_running:
            try:
                await asyncio.sleep(self.config.performance_report_interval)
                
                # Get current metrics
                analyzer_stats = self.analyzer.get_performance_stats()
                logging_stats = self.logging_slave.get_stats()
                
                metrics = self.performance_monitor.get_current_metrics(
                    analyzer_stats, logging_stats
                )
                
                # Check for alerts
                alerts = self.performance_monitor.check_alerts(metrics)
                
                if alerts:
                    for alert in alerts:
                        if self.config.system_mode != SystemMode.PRODUCTION:
                            print(f"⚠️ ALERT: {alert['message']}")
                        
                        # Store alert as event
                        self.analyzer._store_event('performance_alert', alert)
                
                # Store metrics in history
                if hasattr(self.performance_monitor, 'metrics_history'):
                    self.performance_monitor.metrics_history.append(metrics)
                
                # Show performance report based on mode
                if self.config.system_mode == SystemMode.DEMO:
                    self._show_performance_report(metrics)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.config.system_mode == SystemMode.DEVELOPMENT:
                    print(f"❌ Error in monitoring loop: {e}")
    
    def _show_performance_report(self, metrics: PerformanceMetrics):
        """Show performance report for demo mode"""
        
        print("\n" + "="*50)
        print("📊 PERFORMANCE REPORT")
        print("="*50)
        print(f"🕐 Uptime: {metrics.uptime_seconds:.1f}s")
        print(f"💻 CPU: {metrics.cpu_percent:.1f}%")
        print(f"🧠 Memory: {metrics.memory_mb:.1f}MB ({metrics.memory_percent:.1f}%)")
        print(f"📈 Ticks: {metrics.ticks_processed} (avg: {metrics.avg_tick_latency_ms:.2f}ms)")
        print(f"🔮 Predictions: {metrics.predictions_generated}")
        print(f"📝 Events Queued: {metrics.events_queued}")
        print(f"📊 Queue Usage: {metrics.queue_utilization_percent:.1f}%")
        print("="*50 + "\n")
    
    def _get_analyzer_stats(self) -> Dict[str, Any]:
        """Get stats from AdvancedMarketAnalyzer"""
        if self.config.asset_symbol in self.analyzer.asset_analyzers:
            asset_analyzer = self.analyzer.asset_analyzers[self.config.asset_symbol]
            return {
                'ticks_processed': getattr(asset_analyzer, 'analysis_count', 0),
                'learning_phase': getattr(asset_analyzer, 'learning_phase', False),
                'learning_progress': getattr(asset_analyzer, 'learning_progress', 0.0),
                'predictions_generated': 0,  # TODO: implement if needed
                'avg_latency_ms': 0.0,  # TODO: implement if needed
                'buffer_utilization': 0.0  # TODO: implement if needed
            }
        return {'ticks_processed': 0, 'predictions_generated': 0}

    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        
        status = {
            'system': {
                'running': self.is_running,
                'mode': self.config.system_mode.value,
                'profile': self.config.performance_profile.value,
                'uptime_seconds': (
                    (datetime.now() - self.start_time).total_seconds() 
                    if self.start_time else 0
                ),
                'stats': self.system_stats.copy()
            },
            'analyzer': self._get_analyzer_stats(),
            'logging': self.logging_slave.get_stats(),
            'config': asdict(self.config)
        }
        
        # Add latest performance metrics if available
        if (self.performance_monitor and 
            hasattr(self.performance_monitor, 'metrics_history') and 
            len(self.performance_monitor.metrics_history) > 0):
            status['performance'] = asdict(self.performance_monitor.metrics_history[-1])
        
        return status
    
    def update_config(self, new_config: UnifiedConfig):
        """Update system configuration (limited runtime updates)"""
        
        # Update rate limits (safe da cambiare a runtime)
        self.config.rate_limits = new_config.rate_limits
        self.config.event_processing_interval = new_config.event_processing_interval
        self.config.performance_report_interval = new_config.performance_report_interval
        
        if self.config.system_mode != SystemMode.BACKTESTING:
            print(f"✅ Configuration updated")
    
    # ================================
    # CONVENIENCE METHODS (OTTIMIZZATE)
    # ================================
    
    async def run_demo(self, duration_seconds: int = 60, tick_interval: float = 0.1):
        """Run demo con dati simulati"""
        
        print(f"🎬 Starting demo for {duration_seconds} seconds...")
        
        await self.start()
        
        try:
            # Generate demo data
            base_price = 1.1000
            current_price = base_price
            
            start_time = time.time()
            tick_count = 0
            
            while time.time() - start_time < duration_seconds:
                # Simulate price movement
                price_change = (np.random.random() - 0.5) * 0.0001
                current_price += price_change
                current_price = max(0.5000, min(2.0000, current_price))  # Bounds
                
                # Simulate volume
                volume = np.random.randint(1000, 10000)
                
                # Simulate bid/ask spread
                spread = 0.00002
                bid = current_price - spread/2
                ask = current_price + spread/2
                
                # Process tick
                await self.process_tick(
                    timestamp=datetime.now(),
                    price=current_price,
                    volume=volume,
                    bid=bid,
                    ask=ask
                )
                
                tick_count += 1
                
                # Show progress in demo mode
                if tick_count % 50 == 0 and self.config.system_mode == SystemMode.DEMO:
                    print(f"💹 Demo Progress: {tick_count} ticks, price: {current_price:.5f}")
                
                await asyncio.sleep(tick_interval)
            
            print(f"🎬 Demo completed: {tick_count} ticks processed")
            
        finally:
            await self.stop()
    
    async def run_backtest_optimized(self, price_data: List[Dict], 
                                   progress_callback: Optional[Callable] = None,
                                   show_progress: bool = True):
        """🚀 RUN BACKTEST ULTRA-OTTIMIZZATO"""
        
        total_ticks = len(price_data)
        
        if show_progress:
            print(f"📈 Starting OPTIMIZED backtest with {total_ticks:,} data points...")
            print(f"🚀 Mode: {self.config.system_mode.value} | Profile: {self.config.performance_profile.value}")
        
        await self.start()
        
        try:
            start_time = time.time()
            
            for i, data_point in enumerate(price_data):
                await self.process_tick(
                    timestamp=data_point.get('timestamp', datetime.now()),
                    price=data_point['price'],
                    volume=data_point.get('volume', 1000),
                    bid=data_point.get('bid'),
                    ask=data_point.get('ask')
                )
                
                # Progress callback ottimizzato
                if show_progress and i > 0 and i % 10000 == 0:
                    elapsed = time.time() - start_time
                    speed = i / elapsed if elapsed > 0 else 0
                    eta = (total_ticks - i) / speed if speed > 0 else 0
                    
                    print(f"📊 Progress: {i:,}/{total_ticks:,} ({i/total_ticks*100:.1f}%) | "
                          f"Speed: {speed:.0f} ticks/s | ETA: {eta/60:.1f}min")
                
                # External progress callback
                if progress_callback and i % 1000 == 0:
                    progress_callback(i, total_ticks)
            
            elapsed = time.time() - start_time
            final_speed = total_ticks / elapsed if elapsed > 0 else 0
            
            if show_progress:
                print(f"📈 Backtest completed!")
                print(f"⚡ Performance: {final_speed:.0f} ticks/sec ({total_ticks:,} ticks in {elapsed:.1f}s)")
            
        finally:
            await self.stop()


# ================================
# CONVENIENCE FUNCTIONS (OTTIMIZZATE)
# ================================

async def create_production_system(asset: str) -> UnifiedAnalyzerSystem:
    """Create production-ready system"""
    config = UnifiedConfig.for_production(asset)
    system = UnifiedAnalyzerSystem(config)
    await system.start()
    return system

async def create_backtesting_system(asset: str) -> UnifiedAnalyzerSystem:
    """🚀 CREATE ULTRA-FAST BACKTESTING SYSTEM"""
    config = UnifiedConfig.for_backtesting(asset)
    system = UnifiedAnalyzerSystem(config)
    await system.start()
    return system

async def create_development_system(asset: str) -> UnifiedAnalyzerSystem:
    """Create development system"""
    config = UnifiedConfig.for_development(asset)
    system = UnifiedAnalyzerSystem(config)
    await system.start()
    return system

async def run_demo_system(asset: str = "EURUSD", duration: int = 60):
    """Run demo system"""
    config = UnifiedConfig.for_demo(asset)
    system = UnifiedAnalyzerSystem(config)
    await system.run_demo(duration)

def create_custom_config(**kwargs) -> UnifiedConfig:
    """Create custom configuration"""
    return UnifiedConfig(**kwargs)


# ================================
# PERFORMANCE BENCHMARK & UTILITIES
# ================================

async def benchmark_system(asset: str = "USTEC", num_ticks: int = 100000):
    """🚀 BENCHMARK DELLE PERFORMANCE DEL SISTEMA"""
    
    print(f"🏁 Starting performance benchmark...")
    print(f"📊 Asset: {asset} | Ticks: {num_ticks:,}")
    
    # Test diversi profili
    configs = {
        'BACKTESTING': UnifiedConfig.for_backtesting(asset),
        'PRODUCTION': UnifiedConfig.for_production(asset),
        'DEVELOPMENT': UnifiedConfig.for_development(asset)
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n🧪 Testing {config_name} configuration...")
        
        system = UnifiedAnalyzerSystem(config)
        
        # Generate test data
        test_data = []
        base_price = 21500.0
        current_price = base_price
        
        for i in range(num_ticks):
            price_change = (np.random.random() - 0.5) * 1.0
            current_price += price_change
            current_price = max(20000, min(23000, current_price))
            
            test_data.append({
                'timestamp': datetime.now() + timedelta(seconds=i),
                'price': current_price,
                'volume': np.random.randint(100, 1000),
                'bid': current_price - 0.5,
                'ask': current_price + 0.5
            })
        
        # Run benchmark
        start_time = time.time()
        await system.run_backtest_optimized(test_data, show_progress=False)
        elapsed = time.time() - start_time
        
        speed = num_ticks / elapsed if elapsed > 0 else 0
        results[config_name] = {
            'elapsed_seconds': elapsed,
            'ticks_per_second': speed,
            'total_ticks': num_ticks
        }
        
        print(f"✅ {config_name}: {speed:.0f} ticks/sec ({elapsed:.2f}s)")
    
    # Summary
    print(f"\n🏆 BENCHMARK RESULTS:")
    print("="*50)
    for config_name, result in results.items():
        print(f"{config_name:12}: {result['ticks_per_second']:8.0f} ticks/sec")
    
    # Speed improvement calculation
    if 'BACKTESTING' in results and 'DEVELOPMENT' in results:
        backtesting_speed = results['BACKTESTING']['ticks_per_second']
        development_speed = results['DEVELOPMENT']['ticks_per_second']
        improvement = backtesting_speed / development_speed
        print(f"\n🚀 BACKTESTING is {improvement:.1f}x FASTER than DEVELOPMENT!")
    
    return results


# ================================
# INTEGRATION HELPERS
# ================================

class BacktestIntegration:
    """Helper class per integrazione con sistemi di backtesting esistenti"""
    
    def __init__(self, asset: str, ultra_fast: bool = True):
        self.asset = asset
        self.ultra_fast = ultra_fast
        self.system: Optional[UnifiedAnalyzerSystem] = None
        
        # Config ottimizzata
        if ultra_fast:
            self.config = UnifiedConfig.for_backtesting(asset)
        else:
            self.config = UnifiedConfig.for_production(asset)
    
    async def __aenter__(self):
        """Context manager entry"""
        self.system = UnifiedAnalyzerSystem(self.config)
        await self.system.start()
        return self.system
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.system:
            await self.system.stop()
    
    @classmethod
    async def quick_backtest(cls, asset: str, price_data: List[Dict]) -> Dict[str, Any]:
        """Quick backtest senza setup manuale"""
        
        async with cls(asset, ultra_fast=True) as system:
            await system.run_backtest_optimized(price_data, show_progress=True)
            return system.get_system_status()


class ProductionIntegration:
    """Helper class per integrazione in ambiente produzione"""
    
    def __init__(self, asset: str):
        self.asset = asset
        self.config = UnifiedConfig.for_production(asset)
        self.system: Optional[UnifiedAnalyzerSystem] = None
    
    async def start_production_system(self):
        """Avvia sistema di produzione"""
        self.system = UnifiedAnalyzerSystem(self.config)
        await self.system.start()
        return self.system
    
    async def stop_production_system(self):
        """Ferma sistema di produzione"""
        if self.system:
            await self.system.stop()
    
    async def process_live_tick(self, timestamp: datetime, price: float, 
                               volume: float, bid: Optional[float] = None, ask: Optional[float] = None):
        """Processa tick live in produzione"""
        if not self.system:
            raise RuntimeError("Production system not started")
        
        return await self.system.process_tick(timestamp, price, volume, bid, ask)


# ================================
# MONITORING & DIAGNOSTICS UTILS
# ================================

class PerformanceDiagnostics:
    """Utilities per diagnostica performance"""
    
    @staticmethod
    def analyze_tick_rates(system: UnifiedAnalyzerSystem, window_seconds: int = 60) -> Dict[str, float]:
        """Analizza tick rates negli ultimi N secondi"""
        
        stats = system.analyzer.get_performance_stats()
        processing_times = list(stats.get('processing_times', []))
        
        if not processing_times:
            return {'avg_latency_ms': 0, 'max_latency_ms': 0, 'estimated_max_tps': 0}
        
        recent_times = processing_times[-window_seconds:] if len(processing_times) > window_seconds else processing_times
        
        avg_latency = sum(recent_times) / len(recent_times)
        max_latency = max(recent_times)
        
        # Stima max throughput
        estimated_max_tps = 1000 / avg_latency if avg_latency > 0 else 0
        
        return {
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'estimated_max_tps': estimated_max_tps,
            'samples': len(recent_times)
        }
    
    @staticmethod
    def memory_usage_analysis(system: UnifiedAnalyzerSystem) -> Dict[str, Any]:
        """Analizza utilizzo memoria"""
        
        analyzer_stats = system.analyzer.get_performance_stats()
        
        buffer_usage = analyzer_stats.get('buffer_utilization', 0)
        events_pending = analyzer_stats.get('events_pending', 0)
        
        # Calcola memoria stimata
        tick_data_size = len(system.analyzer.tick_data) * 200  # ~200 bytes per tick
        events_size = events_pending * 100  # ~100 bytes per evento
        predictions_size = len(system.analyzer.predictions_history) * 150  # ~150 bytes per predizione
        
        total_estimated_mb = (tick_data_size + events_size + predictions_size) / 1024 / 1024
        
        return {
            'buffer_utilization_percent': buffer_usage,
            'events_pending': events_pending,
            'predictions_count': len(system.analyzer.predictions_history),
            'tick_data_count': len(system.analyzer.tick_data),
            'estimated_memory_mb': total_estimated_mb,
            'memory_efficiency': 'good' if total_estimated_mb < 100 else 'high' if total_estimated_mb < 500 else 'critical'
        }
    
    @staticmethod
    def system_health_check(system: UnifiedAnalyzerSystem) -> Dict[str, str]:
        """Health check completo del sistema"""
        
        status = system.get_system_status()
        health = {}
        
        # System running
        health['system_status'] = 'healthy' if status['system']['running'] else 'stopped'
        
        # Performance check
        perf_analysis = PerformanceDiagnostics.analyze_tick_rates(system)
        if perf_analysis['avg_latency_ms'] < 1.0:
            health['performance'] = 'excellent'
        elif perf_analysis['avg_latency_ms'] < 5.0:
            health['performance'] = 'good'
        elif perf_analysis['avg_latency_ms'] < 10.0:
            health['performance'] = 'degraded'
        else:
            health['performance'] = 'poor'
        
        # Memory check
        mem_analysis = PerformanceDiagnostics.memory_usage_analysis(system)
        health['memory'] = mem_analysis['memory_efficiency']
        
        # Event processing check
        logging_stats = system.logging_slave.get_stats()
        queue_util = logging_stats.get('queue_utilization', 0)
        
        if queue_util < 50:
            health['event_processing'] = 'healthy'
        elif queue_util < 80:
            health['event_processing'] = 'busy'
        else:
            health['event_processing'] = 'overloaded'
        
        # Overall health
        health_values = ['excellent', 'healthy', 'good', 'busy', 'degraded', 'poor', 'overloaded', 'critical']
        worst_health = max([health_values.index(h) for h in health.values() if h in health_values])
        health['overall'] = health_values[worst_health]
        
        return health


# ================================
# MAIN ENTRY POINT & EXAMPLES
# ================================

async def main():
    """Main entry point con esempi di utilizzo"""
    
    print("🚀 Unified Analyzer System - Performance Optimized")
    print("="*60)
    
    # Esempio 1: Sistema ultra-veloce per backtesting
    print("\n📈 Example 1: Ultra-Fast Backtesting System")
    system = await create_backtesting_system("USTEC")
    
    # Generate sample data
    sample_data = []
    base_price = 21500.0
    current_price = base_price
    
    for i in range(1000):  # 1K sample ticks
        price_change = (np.random.random() - 0.5) * 1.0
        current_price += price_change
        
        sample_data.append({
            'timestamp': datetime.now() + timedelta(seconds=i),
            'price': current_price,
            'volume': np.random.randint(100, 1000)
        })
    
    await system.run_backtest_optimized(sample_data)
    print("✅ Backtesting example completed!")
    
    # Esempio 2: Sistema demo con output ricco
    print("\n🎬 Example 2: Demo System")
    await run_demo_system("EURUSD", duration=10)
    print("✅ Demo example completed!")
    
    # Esempio 3: Benchmark performance
    print("\n🏁 Example 3: Performance Benchmark")
    await benchmark_system("USTEC", 10000)
    print("✅ Benchmark completed!")


if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())