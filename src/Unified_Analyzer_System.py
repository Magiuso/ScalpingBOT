"""
Unified Analyzer System - All-in-One Module
===========================================

Sistema completo che integra:
- Analyzer pulito (zero logging overhead)
- Slave logging module intelligente  
- Configuration management
- Performance monitoring
- Automatic lifecycle management

Usage:
    system = UnifiedAnalyzerSystem(config)
    await system.start()
    
    # Your analyzer runs with zero logging overhead
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


# ================================
# CONFIGURATION SYSTEM
# ================================

class SystemMode(Enum):
    """Modalità operative del sistema"""
    PRODUCTION = "production"      # Minimal logging, max performance
    DEVELOPMENT = "development"    # Normal logging, debugging enabled
    TESTING = "testing"           # Verbose logging, full diagnostics
    DEMO = "demo"                 # Rich console output, showcasing


class PerformanceProfile(Enum):
    """Profili di performance predefiniti"""
    HIGH_FREQUENCY = "high_frequency"    # Trading ad alta frequenza
    NORMAL = "normal"                    # Trading normale
    RESEARCH = "research"                # Ricerca e backtesting


@dataclass
class UnifiedConfig:
    """Configurazione unificata per tutto il sistema"""
    
    # === SYSTEM SETTINGS ===
    system_mode: SystemMode = SystemMode.PRODUCTION
    performance_profile: PerformanceProfile = PerformanceProfile.NORMAL
    
    # === ANALYZER SETTINGS ===
    asset_symbol: str = "EURUSD"
    max_tick_buffer_size: int = 100000
    learning_phase_enabled: bool = True
    min_learning_days: int = 7
    
    # === LOGGING SETTINGS ===
    log_level: str = "NORMAL"              # MINIMAL, NORMAL, VERBOSE, DEBUG
    enable_console_output: bool = True
    enable_file_output: bool = True
    enable_csv_export: bool = True
    enable_json_export: bool = False
    
    # === RATE LIMITING ===
    rate_limits: Dict[str, int] = field(default_factory=lambda: {
        'tick_processing': 100,
        'predictions': 50,
        'validations': 25,
        'training_events': 1,
        'champion_changes': 1,
        'emergency_events': 1,
        'diagnostics': 1000
    })
    
    # === PERFORMANCE SETTINGS ===
    event_processing_interval: float = 5.0    # Secondi tra processing eventi
    batch_size: int = 50
    max_queue_size: int = 10000
    async_processing: bool = True
    max_workers: int = 2
    
    # === STORAGE SETTINGS ===
    base_directory: str = "./unified_analyzer_data"
    log_rotation_hours: int = 24
    max_log_files: int = 30
    compress_old_logs: bool = True
    
    # === MONITORING SETTINGS ===
    enable_performance_monitoring: bool = True
    performance_report_interval: float = 60.0  # Secondi
    memory_threshold_mb: int = 1000
    cpu_threshold_percent: float = 80.0
    
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
                'predictions': 100,
                'validations': 100,
                'training_events': 1,
                'champion_changes': 1,
                'emergency_events': 1,
                'diagnostics': 5000
            },
            event_processing_interval=10.0,
            performance_report_interval=300.0
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
            rate_limits={
                'tick_processing': 1,
                'predictions': 1,
                'validations': 1,
                'training_events': 1,
                'champion_changes': 1,
                'emergency_events': 1,
                'diagnostics': 10
            },
            event_processing_interval=1.0,
            performance_report_interval=10.0
        )


# ================================
# PERFORMANCE MONITOR
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
    """Monitora performance del sistema in tempo reale"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.start_time = datetime.now()
        self.process = psutil.Process()
        
        # Metrics history
        self.metrics_history: deque = deque(maxlen=100)
        self.alerts_triggered: List[Dict] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    def get_current_metrics(self, analyzer_stats: Dict, logging_stats: Dict) -> PerformanceMetrics:
        """Ottieni metriche attuali"""
        
        # System metrics
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = self.process.memory_percent()
        
        # Uptime
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
        """Controlla soglie di alert"""
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
        """Avvia monitoring in background"""
        if self.is_monitoring:
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
        """Loop di monitoring principale"""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.config.performance_report_interval)
                # Monitoring details would be implemented here
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")


# ================================
# SIMPLIFIED ANALYZER (CLEAN)
# ================================

class CleanAnalyzer:
    """Analyzer completamente pulito - zero logging overhead"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.asset = config.asset_symbol
        
        # Core data structures
        self.tick_data: deque = deque(maxlen=config.max_tick_buffer_size)
        self.predictions_history: List[Dict] = []
        
        # Event buffers for slave logging
        self._event_buffers = {
            'tick_events': deque(maxlen=1000),
            'prediction_events': deque(maxlen=500),
            'training_events': deque(maxlen=200),
            'champion_events': deque(maxlen=100),
            'error_events': deque(maxlen=300),
            'diagnostic_events': deque(maxlen=200)
        }
        
        # Performance tracking
        self.performance_stats = {
            'ticks_processed': 0,
            'predictions_generated': 0,
            'training_events': 0,
            'processing_times': deque(maxlen=100),
            'last_tick_time': None
        }
        
        # Threading
        self.data_lock = threading.RLock()
        
        # Learning state
        self.learning_phase = config.learning_phase_enabled
        self.learning_start_time = datetime.now()
        self.learning_progress = 0.0
    
    def process_tick(self, timestamp: datetime, price: float, volume: float, 
                    bid: Optional[float] = None, ask: Optional[float] = None) -> Dict[str, Any]:
        """Processa tick con zero logging overhead"""
        
        processing_start = time.time()
        
        # Store tick data
        with self.data_lock:
            tick_data = {
                'timestamp': timestamp,
                'price': price,
                'volume': volume,
                'bid': bid or price,
                'ask': ask or price,
                'spread': (ask - bid) if ask and bid else 0
            }
            self.tick_data.append(tick_data)
        
        # Update performance stats
        self.performance_stats['ticks_processed'] += 1
        self.performance_stats['last_tick_time'] = timestamp
        
        # Learning progress
        if self.learning_phase:
            days_learning = (datetime.now() - self.learning_start_time).days
            self.learning_progress = min(1.0, days_learning / self.config.min_learning_days)
            
            if days_learning >= self.config.min_learning_days:
                self.learning_phase = False
                self._store_event('learning_completed', {
                    'asset': self.asset,
                    'days_learned': days_learning,
                    'ticks_collected': len(self.tick_data)
                })
        
        # Generate analysis
        analysis_result = self._generate_analysis()
        
        # Track processing time
        processing_time = (time.time() - processing_start) * 1000  # ms
        self.performance_stats['processing_times'].append(processing_time)
        
        # Store tick event for logging
        self._store_event('tick_processed', {
            'asset': self.asset,
            'price': price,
            'volume': volume,
            'processing_time_ms': processing_time,
            'learning_progress': self.learning_progress if self.learning_phase else 1.0
        })
        
        return analysis_result
    
    def _generate_analysis(self) -> Dict[str, Any]:
        """Genera analisi simulata"""
        if len(self.tick_data) < 10:
            return {'status': 'insufficient_data'}
        
        with self.data_lock:
            recent_prices = [tick['price'] for tick in list(self.tick_data)[-10:]]
        
        current_price = recent_prices[-1]
        price_change = (current_price - recent_prices[0]) / recent_prices[0] * 100
        
        # Simula predizione
        if len(self.tick_data) % 10 == 0:  # Ogni 10 ticks
            prediction = {
                'algorithm': 'demo_predictor',
                'confidence': 0.75 + (hash(str(current_price)) % 100) / 400,  # Random-ish
                'prediction': 'buy' if price_change > 0 else 'sell',
                'timestamp': datetime.now()
            }
            
            self.predictions_history.append(prediction)
            self.performance_stats['predictions_generated'] += 1
            
            self._store_event('prediction_generated', {
                'asset': self.asset,
                'algorithm': prediction['algorithm'],
                'confidence': prediction['confidence'],
                'prediction': prediction['prediction']
            })
        
        return {
            'status': 'success',
            'current_price': current_price,
            'price_change_percent': price_change,
            'trend': 'bullish' if price_change > 0 else 'bearish',
            'confidence': 0.8,
            'tick_count': len(self.tick_data),
            'learning_phase': self.learning_phase,
            'learning_progress': self.learning_progress
        }
    
    def _store_event(self, event_type: str, event_data: Dict):
        """Store event nel buffer appropriato"""
        event = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'data': event_data
        }
        
        # Route to appropriate buffer
        if 'tick' in event_type:
            self._event_buffers['tick_events'].append(event)
        elif 'prediction' in event_type:
            self._event_buffers['prediction_events'].append(event)
        elif 'training' in event_type:
            self._event_buffers['training_events'].append(event)
        elif 'champion' in event_type:
            self._event_buffers['champion_events'].append(event)
        elif 'error' in event_type or 'emergency' in event_type:
            self._event_buffers['error_events'].append(event)
        else:
            self._event_buffers['diagnostic_events'].append(event)
    
    def get_all_events(self) -> Dict[str, List[Dict]]:
        """Ottieni tutti gli eventi per il slave logging"""
        with self.data_lock:
            return {
                buffer_name: list(buffer) 
                for buffer_name, buffer in self._event_buffers.items()
            }
    
    def clear_events(self):
        """Pulisci tutti gli event buffers"""
        with self.data_lock:
            for buffer in self._event_buffers.values():
                buffer.clear()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche performance"""
        with self.data_lock:
            avg_processing_time = (
                sum(self.performance_stats['processing_times']) / 
                len(self.performance_stats['processing_times'])
            ) if self.performance_stats['processing_times'] else 0
            
            return {
                **self.performance_stats,
                'avg_latency_ms': avg_processing_time,
                'buffer_utilization': len(self.tick_data) / self.config.max_tick_buffer_size * 100,
                'events_pending': sum(len(buffer) for buffer in self._event_buffers.values())
            }


# ================================
# INTELLIGENT LOGGING SLAVE  
# ================================

class LoggingSlave:
    """Slave logging ottimizzato e integrato"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        
        # Setup directories
        self.base_path = Path(config.base_directory)
        self.logs_path = self.base_path / "logs"
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        # Event processing
        self.event_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None
        
        # Rate limiting
        self.event_counters = defaultdict(int)
        self.last_logged = {}
        
        # Statistics
        self.stats = {
            'events_received': 0,
            'events_processed': 0,
            'events_dropped': 0,
            'queue_size': 0,
            'queue_utilization': 0.0
        }
        
        # Output handlers
        self.console_enabled = config.enable_console_output
        self.file_handlers = {}
        self.csv_writers = {}
        
        # Setup logging infrastructure
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging infrastructure"""
        
        # Console output
        if self.console_enabled:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            self.console_logger = logging.getLogger('unified_console')
        
        # File output
        if self.config.enable_file_output:
            log_file = self.logs_path / f"analyzer_{datetime.now():%Y%m%d_%H%M%S}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            self.file_logger = logging.getLogger('unified_file')
            self.file_logger.addHandler(file_handler)
            self.file_logger.setLevel(logging.DEBUG)
        
        # CSV output
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
        """Avvia processing"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_events())
        
        if self.console_enabled and self.config.system_mode != SystemMode.PRODUCTION:
            print(f"🚀 Logging Slave started - Mode: {self.config.system_mode.value}")
    
    async def stop(self):
        """Ferma processing"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Process remaining events
        await self._flush_remaining_events()
        
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
        """Processa eventi dall'analyzer"""
        
        for event_category, event_list in events.items():
            for event in event_list:
                self.stats['events_received'] += 1
                
                # Rate limiting check
                event_type = event.get('event_type', 'unknown')
                
                if self._should_log_event(event_type):
                    try:
                        await self.event_queue.put(event)
                    except asyncio.QueueFull:
                        self.stats['events_dropped'] += 1
        
        # Update queue stats
        self.stats['queue_size'] = self.event_queue.qsize()
        self.stats['queue_utilization'] = (
            self.stats['queue_size'] / self.config.max_queue_size * 100
        )
    
    def _should_log_event(self, event_type: str) -> bool:
        """Rate limiting logic"""
        
        # Emergency events always logged
        if 'emergency' in event_type or 'error' in event_type:
            return True
        
        # Apply rate limiting
        rate_limit = self.config.rate_limits.get(event_type, 1)
        self.event_counters[event_type] += 1
        
        return self.event_counters[event_type] % rate_limit == 0
    
    async def _process_events(self):
        """Main event processing loop"""
        batch = []
        last_batch_time = datetime.now()
        
        while self.is_running:
            try:
                # Get events with timeout
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=self.config.event_processing_interval
                    )
                    batch.append(event)
                    
                except asyncio.TimeoutError:
                    pass
                
                # Process batch
                current_time = datetime.now()
                time_since_batch = (current_time - last_batch_time).total_seconds()
                
                if (len(batch) >= self.config.batch_size or 
                    time_since_batch >= self.config.event_processing_interval):
                    
                    if batch:
                        await self._process_batch(batch)
                        batch.clear()
                        last_batch_time = current_time
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.config.system_mode == SystemMode.DEVELOPMENT:
                    print(f"❌ Error processing events: {e}")
    
    async def _process_batch(self, events: List[Dict]):
        """Process batch of events"""
        
        for event in events:
            await self._process_single_event(event)
            self.stats['events_processed'] += 1
    
    async def _process_single_event(self, event: Dict):
        """Process single event"""
        
        # Format for different outputs
        formatted_event = self._format_event(event)
        
        # Console output
        if self.console_enabled:
            self._output_to_console(formatted_event)
        
        # File output
        if self.config.enable_file_output:
            self.file_logger.info(json.dumps(event, default=str))
        
        # CSV output
        if self.config.enable_csv_export and 'main' in self.csv_writers:
            self.csv_writers['main']['writer'].writerow({
                'timestamp': event.get('timestamp', datetime.now()),
                'event_type': event.get('event_type', 'unknown'),
                'asset': event.get('data', {}).get('asset', 'unknown'),
                'summary': formatted_event,
                'data': json.dumps(event.get('data', {}), default=str)
            })
    
    def _format_event(self, event: Dict) -> str:
        """Format event for display"""
        event_type = event.get('event_type', 'unknown')
        data = event.get('data', {})
        
        if event_type == 'tick_processed':
            return f"Tick: {data.get('asset')} @ {data.get('price', 0):.5f} ({data.get('processing_time_ms', 0):.2f}ms)"
        
        elif event_type == 'prediction_generated':
            return f"Prediction: {data.get('algorithm')} - {data.get('prediction')} (confidence: {data.get('confidence', 0):.2f})"
        
        elif event_type == 'learning_completed':
            return f"Learning: {data.get('asset')} completed after {data.get('days_learned')} days"
        
        else:
            return f"{event_type}: {str(data)[:100]}"
    
    def _output_to_console(self, formatted_event: str):
        """Output to console based on mode"""
        
        if self.config.system_mode == SystemMode.DEMO:
            # Rich output for demo
            print(f"[{datetime.now():%H:%M:%S}] 📊 {formatted_event}")
        
        elif self.config.system_mode == SystemMode.DEVELOPMENT:
            # Standard output for development
            print(f"[{datetime.now():%H:%M:%S}] {formatted_event}")
        
        elif self.config.system_mode == SystemMode.TESTING:
            # Verbose output for testing
            print(f"[{datetime.now():%H:%M:%S}] [TEST] {formatted_event}")
        
        # Production mode has console disabled by default
    
    async def _flush_remaining_events(self):
        """Flush remaining events"""
        remaining = []
        
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                remaining.append(event)
            except asyncio.QueueEmpty:
                break
        
        if remaining:
            await self._process_batch(remaining)
    
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
# UNIFIED SYSTEM CONTROLLER
# ================================

class UnifiedAnalyzerSystem:
    """
    Sistema unificato che gestisce tutto:
    - Analyzer pulito
    - Logging slave
    - Performance monitoring 
    - Configuration management
    - Lifecycle management
    """
    
    def __init__(self, config: Optional[UnifiedConfig] = None):
        self.config = config or UnifiedConfig()
        
        # Core components
        self.analyzer = CleanAnalyzer(self.config)
        self.logging_slave = LoggingSlave(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        
        # System state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Background tasks
        self.event_processing_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.system_stats = {
            'total_ticks_processed': 0,
            'total_events_logged': 0,
            'uptime_seconds': 0.0,
            'errors_count': 0
        }
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
            if self.is_running:
                asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self):
        """Avvia tutto il sistema"""
        if self.is_running:
            print("⚠️ System already running")
            return
        
        print(f"🚀 Starting Unified Analyzer System - Mode: {self.config.system_mode.value}")
        print(f"📊 Asset: {self.config.asset_symbol}")
        print(f"⚙️ Profile: {self.config.performance_profile.value}")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            # Start core components
            await self.logging_slave.start()
            
            if self.config.enable_performance_monitoring:
                await self.performance_monitor.start_monitoring()
            
            # Start background tasks
            self.event_processing_task = asyncio.create_task(self._event_processing_loop())
            
            if self.config.enable_performance_monitoring:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            print("✅ System started successfully")
            
        except Exception as e:
            print(f"❌ Failed to start system: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Ferma tutto il sistema"""
        if not self.is_running:
            return
        
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
        
        if self.config.enable_performance_monitoring:
            await self.performance_monitor.stop_monitoring()
        
        # Final statistics
        if self.start_time:
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
        Process a single tick - MAIN INTERFACE
        
        This is the primary method for feeding data to the system.
        Zero logging overhead - all logging handled by slave.
        """
        
        if not self.is_running:
            raise RuntimeError("System not running. Call start() first.")
        
        try:
            # Process tick with zero logging overhead
            result = self.analyzer.process_tick(timestamp, price, volume, bid, ask)
            
            # Update system stats
            self.system_stats['total_ticks_processed'] += 1
            
            return result
            
        except Exception as e:
            self.system_stats['errors_count'] += 1
            
            # Store error event
            self.analyzer._store_event('processing_error', {
                'error': str(e),
                'timestamp': timestamp,
                'price': price,
                'volume': volume
            })
            
            if self.config.system_mode in [SystemMode.DEVELOPMENT, SystemMode.DEMO]:
                print(f"❌ Error processing tick: {e}")
            
            raise
    
    async def _event_processing_loop(self):
        """Background loop per processing eventi"""
        
        while self.is_running:
            try:
                await asyncio.sleep(self.config.event_processing_interval)
                
                # Get events from analyzer
                events = self.analyzer.get_all_events()
                
                if any(events.values()):  # If any events exist
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
        """Background loop per monitoring performance"""
        
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
            'analyzer': self.analyzer.get_performance_stats(),
            'logging': self.logging_slave.get_stats(),
            'config': asdict(self.config)
        }
        
        # Add latest performance metrics if available
        if self.performance_monitor.metrics_history:
            status['performance'] = asdict(self.performance_monitor.metrics_history[-1])
        
        return status
    
    def update_config(self, new_config: UnifiedConfig):
        """Update system configuration (limited runtime updates)"""
        
        # Update rates limits (safe to change at runtime)
        self.config.rate_limits = new_config.rate_limits
        self.config.event_processing_interval = new_config.event_processing_interval
        self.config.performance_report_interval = new_config.performance_report_interval
        
        print(f"✅ Configuration updated")
    
    # Convenience methods for common operations
    
    async def run_demo(self, duration_seconds: int = 60, tick_interval: float = 0.1):
        """Run demo with simulated data"""
        
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
    
    async def run_backtest(self, price_data: List[Dict], progress_callback: Optional[Callable] = None):
        """Run backtest with historical data"""
        
        print(f"📈 Starting backtest with {len(price_data)} data points...")
        
        await self.start()
        
        try:
            for i, data_point in enumerate(price_data):
                await self.process_tick(
                    timestamp=data_point.get('timestamp', datetime.now()),
                    price=data_point['price'],
                    volume=data_point.get('volume', 1000),
                    bid=data_point.get('bid'),
                    ask=data_point.get('ask')
                )
                
                # Progress callback
                if progress_callback and i % 1000 == 0:
                    progress_callback(i, len(price_data))
            
            print(f"📈 Backtest completed")
            
        finally:
            await self.stop()


# ================================
# CONVENIENCE FUNCTIONS
# ================================

async def create_production_system(asset: str) -> UnifiedAnalyzerSystem:
    """Create production-ready system"""
    config = UnifiedConfig.for_production(asset)
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
# EXAMPLE USAGE
# ================================

if __name__ == "__main__":
    
    async def main():
        # Example 1: Production system
        print("Example 1: Production System")
        system = await create_production_system("EURUSD")
        
        # Process some ticks
        for i in range(10):
            await system.process_tick(
                timestamp=datetime.now(),
                price=1.1000 + i * 0.0001,
                volume=1000 + i * 100
            )
            await asyncio.sleep(0.1)
        
        await system.stop()
        print()
        
        # Example 2: Demo system
        print("Example 2: Demo System")
        await run_demo_system("GBPUSD", duration=30)
        print()
        
        # Example 3: Custom configuration
        print("Example 3: Custom Configuration")
        custom_config = create_custom_config(
            system_mode=SystemMode.DEVELOPMENT,
            asset_symbol="USDJPY", 
            log_level="VERBOSE",
            rate_limits={'tick_processing': 5, 'predictions': 1}
        )
        
        system = UnifiedAnalyzerSystem(custom_config)
        await system.start()
        
        # Check status
        status = system.get_system_status()
        print(f"System running: {status['system']['running']}")
        print(f"Mode: {status['system']['mode']}")
        
        await system.stop()
    
    # Run example
    asyncio.run(main())