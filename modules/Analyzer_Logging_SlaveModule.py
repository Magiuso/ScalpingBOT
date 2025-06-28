"""
Analyzer Logging Slave Module
============================

Sistema di logging dedicato che opera indipendentemente dal modulo Analyzer principale.
Gestisce tutti gli eventi accumulati dai buffer systems in modo intelligente e asincrono.
"""

import asyncio
import threading
import time
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor


class LogLevel(Enum):
    """Livelli di logging per il slave module"""
    MINIMAL = "minimal"        # Solo errori critici
    NORMAL = "normal"          # Errori + eventi importanti  
    VERBOSE = "verbose"        # Tutti gli eventi
    DEBUG = "debug"            # Debug completo + diagnostics


class EventPriority(Enum):
    """Priorità degli eventi per processing"""
    CRITICAL = 1    # Emergency stops, errori critici
    HIGH = 2        # Champion changes, training failures
    MEDIUM = 3      # Predictions, validations, training success
    LOW = 4         # Performance metrics, diagnostics routine


@dataclass
class LoggingConfig:
    """Configurazione completa per il logging slave"""
    
    # Livelli di logging
    log_level: LogLevel = LogLevel.NORMAL
    
    # Rate limiting per tipo di evento
    rate_limits: Dict[str, int] = field(default_factory=lambda: {
        'process_tick': 100,           # Log ogni 100 ticks
        'predictions': 50,             # Log ogni 50 predizioni
        'validations': 25,             # Log ogni 25 validazioni
        'training_progress': 10,       # Log ogni 10% progress
        'diagnostics': 1000,           # Log ogni 1000 operazioni
        'champion_changes': 1,         # Log sempre (rare)
        'emergency_events': 1          # Log sempre (critiche)
    })
    
    # Configurazione output
    enable_console_output: bool = True
    enable_file_output: bool = True
    enable_csv_export: bool = True
    enable_json_export: bool = False
    
    # Configurazione aggregazione
    batch_size: int = 50              # Eventi per batch
    batch_interval: float = 5.0       # Secondi tra batch
    max_queue_size: int = 10000       # Max eventi in coda
    
    # Configurazione file
    log_directory: str = "./analyzer_logs_slave"
    log_rotation_hours: int = 24       # Rotazione file ogni 24 ore
    max_log_files: int = 30           # Mantieni 30 file massimo
    
    # Performance settings
    async_processing: bool = True      # Processing asincrono
    max_workers: int = 2              # Thread pool size
    flush_interval: float = 10.0      # Flush forzato ogni 10s


class EventAggregator:
    """Aggregatore intelligente di eventi per ridurre il volume di log"""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.event_counters: Dict[str, int] = defaultdict(int)
        self.last_logged: Dict[str, datetime] = {}
        self.aggregated_metrics: Dict[str, Dict] = defaultdict(dict)
        self.reset_time = datetime.now()
    
    def should_log_event(self, event_type: str, event_data: Dict) -> bool:
        """Determina se un evento deve essere loggato basandosi su rate limiting"""
        
        # Eventi critici sempre loggati
        if event_data.get('priority') == EventPriority.CRITICAL:
            return True
        
        # Rate limiting per tipo di evento
        if event_type in self.config.rate_limits:
            self.event_counters[event_type] += 1
            rate_limit = self.config.rate_limits[event_type]
            
            if self.event_counters[event_type] % rate_limit == 0:
                return True
            else:
                # Accumula metrics per summary
                self._accumulate_metrics(event_type, event_data)
                return False
        
        # Eventi non configurati - log always
        return True
    
    def _accumulate_metrics(self, event_type: str, event_data: Dict):
        """Accumula metriche per eventi non loggati"""
        if event_type not in self.aggregated_metrics:
            self.aggregated_metrics[event_type] = {
                'count': 0,
                'first_seen': datetime.now(),
                'last_seen': datetime.now(),
                'sample_data': event_data
            }
        
        self.aggregated_metrics[event_type]['count'] += 1
        self.aggregated_metrics[event_type]['last_seen'] = datetime.now()
    
    def get_aggregated_summary(self) -> Dict[str, Any]:
        """Ottieni summary degli eventi aggregati"""
        summary = {
            'period_start': self.reset_time,
            'period_end': datetime.now(),
            'aggregated_events': dict(self.aggregated_metrics),
            'total_events_processed': sum(self.event_counters.values()),
            'events_logged': sum(1 for k, v in self.event_counters.items() 
                               if v >= self.config.rate_limits.get(k, 1))
        }
        
        return summary
    
    def reset_aggregation(self):
        """Reset contatori per nuovo periodo"""
        self.event_counters.clear()
        self.aggregated_metrics.clear()
        self.reset_time = datetime.now()


class LogFormatter:
    """Formattatore intelligente per diversi tipi di output"""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
    
    def format_for_console(self, event: Dict) -> str:
        """Formatta evento per output console pulito"""
        timestamp = event.get('timestamp', datetime.now())
        event_type = event.get('event_type', 'unknown')
        
        if self.config.log_level == LogLevel.MINIMAL:
            # Solo info essenziali
            return f"[{timestamp:%H:%M:%S}] {event_type}"
        
        elif self.config.log_level == LogLevel.NORMAL:
            # Info standard
            data = event.get('data', {})
            key_info = self._extract_key_info(event_type, data)
            return f"[{timestamp:%H:%M:%S}] {event_type}: {key_info}"
        
        elif self.config.log_level in [LogLevel.VERBOSE, LogLevel.DEBUG]:
            # Info complete
            return f"[{timestamp:%H:%M:%S}] {event_type}: {json.dumps(event.get('data', {}), indent=2)}"
        
        # Default fallback
        return f"[{timestamp:%H:%M:%S}] {event_type}"
    
    def format_for_file(self, event: Dict) -> str:
        """Formatta evento per file log"""
        timestamp = event.get('timestamp', datetime.now())
        return f"[{timestamp.isoformat()}] {json.dumps(event, default=str)}"
    
    def format_for_csv(self, event: Dict) -> Dict[str, Any]:
        """Formatta evento per export CSV"""
        event_type = event.get('event_type', 'unknown')
        return {
            'timestamp': event.get('timestamp', datetime.now()),
            'event_type': event_type,
            'priority': getattr(event.get('priority'), 'value', 'unknown'),
            'source': event.get('source', 'analyzer'),
            'summary': self._extract_key_info(event_type, event.get('data', {})),
            'data_json': json.dumps(event.get('data', {}), default=str)
        }
    
    def _extract_key_info(self, event_type: str, data: Dict) -> str:
        """Estrae informazioni chiave per display"""
        if event_type == 'champion_changed':
            return f"{data.get('asset', 'unknown')} {data.get('model_type', 'unknown')}: {data.get('old_champion', 'unknown')} → {data.get('new_champion', 'unknown')}"
        
        elif event_type == 'prediction_logged':
            return f"{data.get('asset', 'unknown')} {data.get('algorithm', 'unknown')}: confidence={data.get('confidence', 0):.2f}"
        
        elif event_type == 'training_completed':
            return f"{data.get('algorithm', 'unknown')}: loss={data.get('final_loss', 0):.4f}, improvement={data.get('improvement', 0):.2f}"
        
        elif event_type == 'emergency_events':
            return f"EMERGENCY: {data.get('type', 'unknown')} - {data.get('details', 'no details')}"
        
        else:
            # Generic summary
            summary_data = data.get('summary', str(data)[:100] if data else 'no data')
            return str(summary_data)


class AnalyzerLoggingSlave:
    """
    Modulo slave dedicato per logging Analyzer
    
    Gestisce in modo asincrono e intelligente tutti gli eventi generati
    dal sistema Analyzer pulito, fornendo logging configurabile senza
    impatto sulle performance del sistema principale.
    """
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        self.config = config if config is not None else LoggingConfig()
        
        # Setup directories
        self.log_dir = Path(self.config.log_directory)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.aggregator = EventAggregator(self.config)
        self.formatter = LogFormatter(self.config)
        
        # Event processing
        self.event_queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self.processing_task = None
        self.is_running = False
        
        # Output handlers
        self.file_handlers: Dict[str, Any] = {}
        self.csv_writers: Dict[str, Any] = {}
        
        # Thread pool for sync operations
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Statistics
        self.stats = {
            'events_processed': 0,
            'events_logged': 0,
            'events_dropped': 0,
            'last_flush': datetime.now(),
            'processing_errors': 0
        }
        
        # Setup logging infrastructure
        self._setup_logging_infrastructure()
    
    def _setup_logging_infrastructure(self):
        """Setup file handlers e output infrastructure"""
        
        # Console logger
        if self.config.enable_console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            self.console_logger = logging.getLogger('analyzer_slave_console')
            self.console_logger.addHandler(console_handler)
            self.console_logger.setLevel(logging.INFO)
        
        # File logger
        if self.config.enable_file_output:
            log_file = self.log_dir / f"analyzer_events_{datetime.now():%Y%m%d_%H%M%S}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            self.file_logger = logging.getLogger('analyzer_slave_file')
            self.file_logger.addHandler(file_handler)
            self.file_logger.setLevel(logging.DEBUG)
        
        # CSV setup
        if self.config.enable_csv_export:
            self._setup_csv_writers()
        else:
            self.csv_writers = {}
    
    def _setup_csv_writers(self):
        """Setup CSV writers per diversi tipi di eventi"""
        csv_dir = self.log_dir / 'csv_exports'
        csv_dir.mkdir(exist_ok=True)
        
        csv_file = csv_dir / f"analyzer_events_{datetime.now():%Y%m%d}.csv"
        
        fieldnames = ['timestamp', 'event_type', 'priority', 'source', 'summary', 'data_json']
        
        self.csv_writers['main'] = {
            'file': open(csv_file, 'a', newline='', encoding='utf-8'),
            'writer': csv.DictWriter(open(csv_file, 'a', newline='', encoding='utf-8'), fieldnames=fieldnames)
        }
        
        # Write header if new file
        if csv_file.stat().st_size == 0:
            self.csv_writers['main']['writer'].writeheader()
    
    async def start(self):
        """Avvia il processing degli eventi"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Avvia task di processing
        if self.config.async_processing:
            self.processing_task = asyncio.create_task(self._process_events_async())
        else:
            self.processing_task = asyncio.create_task(self._process_events_sync_wrapper())
        
        # Avvia flush periodico
        asyncio.create_task(self._periodic_flush())
        
        if self.config.enable_console_output and self.config.log_level != LogLevel.MINIMAL:
            print(f"✅ Analyzer Logging Slave started - Level: {self.config.log_level.value}")
    
    async def stop(self):
        """Ferma il processing e flush finale"""
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
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Close file handlers
        self._close_handlers()
        
        if self.config.enable_console_output and self.config.log_level != LogLevel.MINIMAL:
            print(f"✅ Analyzer Logging Slave stopped - Events processed: {self.stats['events_processed']}")
    
    async def process_analyzer_events(self, analyzer_events: Dict[str, List[Dict]]):
        """
        Processa eventi dal sistema Analyzer
        
        Args:
            analyzer_events: Dict con eventi da tutti i buffer dell'Analyzer
        """
        
        for event_category, events in analyzer_events.items():
            for event in events:
                # Aggiungi metadati
                enriched_event = {
                    **event,
                    'source': 'analyzer',
                    'category': event_category,
                    'received_at': datetime.now(),
                    'priority': self._determine_priority(event)
                }
                
                # Queue per processing
                try:
                    await self.event_queue.put(enriched_event)
                except asyncio.QueueFull:
                    self.stats['events_dropped'] += 1
                    # Drop oldest events if queue is full
                    try:
                        await self.event_queue.get_nowait()
                        await self.event_queue.put(enriched_event)
                    except asyncio.QueueEmpty:
                        pass
    
    def _determine_priority(self, event: Dict) -> EventPriority:
        """Determina priorità di un evento"""
        event_type = event.get('event_type', '')
        
        # Critical events
        if any(keyword in event_type.lower() for keyword in ['emergency', 'critical', 'error', 'failed']):
            return EventPriority.CRITICAL
        
        # High priority events
        if any(keyword in event_type.lower() for keyword in ['champion', 'stall', 'corrupted']):
            return EventPriority.HIGH
        
        # Medium priority events
        if any(keyword in event_type.lower() for keyword in ['prediction', 'training', 'validation']):
            return EventPriority.MEDIUM
        
        # Low priority (metrics, diagnostics)
        return EventPriority.LOW
    
    async def _process_events_async(self):
        """Processing asincrono degli eventi"""
        batch = []
        last_batch_time = datetime.now()
        
        while self.is_running:
            try:
                # Get event with timeout
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=self.config.batch_interval
                    )
                    batch.append(event)
                    self.stats['events_processed'] += 1
                    
                except asyncio.TimeoutError:
                    # Process batch even if not full
                    pass
                
                # Process batch when full or interval elapsed
                current_time = datetime.now()
                time_since_batch = (current_time - last_batch_time).total_seconds()
                
                if (len(batch) >= self.config.batch_size or 
                    time_since_batch >= self.config.batch_interval):
                    
                    if batch:
                        await self._process_event_batch(batch)
                        batch.clear()
                        last_batch_time = current_time
                
            except Exception as e:
                self.stats['processing_errors'] += 1
                if self.config.log_level == LogLevel.DEBUG:
                    print(f"❌ Error processing events: {e}")
    
    async def _process_events_sync_wrapper(self):
        """Wrapper per processing sincrono"""
        while self.is_running:
            try:
                event = await self.event_queue.get()
                await self._process_single_event(event)
                self.stats['events_processed'] += 1
                
            except Exception as e:
                self.stats['processing_errors'] += 1
    
    async def _process_event_batch(self, events: List[Dict]):
        """Processa un batch di eventi"""
        
        # Sort by priority
        events.sort(key=lambda e: e.get('priority', EventPriority.LOW).value)
        
        for event in events:
            await self._process_single_event(event)
    
    async def _process_single_event(self, event: Dict):
        """Processa un singolo evento"""
        
        # Check if should log based on aggregation rules
        if not self.aggregator.should_log_event(event.get('event_type', ''), event):
            return
        
        self.stats['events_logged'] += 1
        
        # Format and output
        if self.config.enable_console_output:
            console_msg = self.formatter.format_for_console(event)
            if self.config.log_level != LogLevel.MINIMAL or event.get('priority') == EventPriority.CRITICAL:
                print(console_msg)
        
        if self.config.enable_file_output:
            file_msg = self.formatter.format_for_file(event)
            self.file_logger.info(file_msg)
        
        if self.config.enable_csv_export and 'main' in self.csv_writers:
            csv_row = self.formatter.format_for_csv(event)
            self.csv_writers['main']['writer'].writerow(csv_row)
        
        if self.config.enable_json_export:
            await self._write_json_event(event)
    
    async def _write_json_event(self, event: Dict):
        """Scrivi evento in formato JSON"""
        json_file = self.log_dir / f"events_{datetime.now():%Y%m%d}.jsonl"
        
        def write_json():
            with open(json_file, 'a', encoding='utf-8') as f:
                json.dump(event, f, default=str)
                f.write('\n')
        
        # Use thread pool for file I/O
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, write_json)
    
    async def _periodic_flush(self):
        """Flush periodico e maintenance"""
        while self.is_running:
            await asyncio.sleep(self.config.flush_interval)
            
            # Flush file handlers
            for handler in self.file_handlers.values():
                if hasattr(handler, 'flush'):
                    handler.flush()
            
            # Flush CSV writers
            for csv_info in self.csv_writers.values():
                csv_info['file'].flush()
            
            # Update stats
            self.stats['last_flush'] = datetime.now()
            
            # Log aggregated summary if verbose
            if self.config.log_level in [LogLevel.VERBOSE, LogLevel.DEBUG]:
                summary = self.aggregator.get_aggregated_summary()
                if summary['aggregated_events']:
                    print(f"📊 Aggregated Events Summary: {len(summary['aggregated_events'])} types, "
                          f"{summary['total_events_processed']} total events")
    
    async def _flush_remaining_events(self):
        """Flush eventi rimanenti durante shutdown"""
        remaining_events = []
        
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                remaining_events.append(event)
            except asyncio.QueueEmpty:
                break
        
        if remaining_events:
            await self._process_event_batch(remaining_events)
    
    def _close_handlers(self):
        """Chiudi tutti i file handlers"""
        
        # Close CSV writers
        for csv_info in self.csv_writers.values():
            if csv_info['file']:
                csv_info['file'].close()
        
        # Close file handlers
        for handler in self.file_handlers.values():
            if hasattr(handler, 'close'):
                handler.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Ottieni statistiche del slave module"""
        return {
            **self.stats,
            'queue_size': self.event_queue.qsize(),
            'aggregation_summary': self.aggregator.get_aggregated_summary(),
            'config': {
                'log_level': self.config.log_level.value,
                'rate_limits': self.config.rate_limits,
                'async_processing': self.config.async_processing
            }
        }


# Convenience functions per integration facile

async def create_logging_slave(config: Optional[LoggingConfig] = None) -> AnalyzerLoggingSlave:
    """Crea e avvia logging slave"""
    if config is None:
        config = LoggingConfig()
    slave = AnalyzerLoggingSlave(config)
    await slave.start()
    return slave


async def process_analyzer_data(slave: AnalyzerLoggingSlave, analyzer_instance):
    """
    Processa dati da istanza Analyzer
    
    Args:
        slave: Istanza del logging slave
        analyzer_instance: Istanza dell'Analyzer (AssetAnalyzer o AdvancedMarketAnalyzer)
    """
    
    # Get events from analyzer's logger
    if hasattr(analyzer_instance, 'logger') and hasattr(analyzer_instance.logger, 'get_all_events_for_slave'):
        events = analyzer_instance.logger.get_all_events_for_slave()
        await slave.process_analyzer_events(events)
        
        # Clear processed events
        analyzer_instance.logger.clear_events_buffer()
    
    # Get events from diagnostics if available
    if hasattr(analyzer_instance, 'diagnostics'):
        diagnostic_events = {}
        
        # Check for diagnostic buffers
        for buffer_name in ['_diagnostic_events_buffer', '_emergency_data_buffer', '_performance_metrics_buffer']:
            if hasattr(analyzer_instance.diagnostics, buffer_name):
                buffer = getattr(analyzer_instance.diagnostics, buffer_name)
                if buffer:
                    diagnostic_events[buffer_name.replace('_', '').replace('buffer', '')] = buffer.copy()
                    buffer.clear()
        
        if diagnostic_events:
            await slave.process_analyzer_events(diagnostic_events)


# Example usage integration
"""
# Usage Example:

# 1. Create config
config = LoggingConfig(
    log_level=LogLevel.NORMAL,
    rate_limits={
        'predictions': 100,  # Log every 100 predictions
        'training_completed': 1,  # Log all training completions
    },
    enable_console_output=True,
    enable_csv_export=True
)

# 2. Create and start slave
slave = await create_logging_slave(config)

# 3. In your main analyzer loop:
while analyzer_running:
    # Your analyzer does its work (now with zero logging overhead)
    analyzer.process_tick(timestamp, price, volume)
    
    # Periodically process accumulated events (e.g., every 60 seconds)
    if time_to_process_logs:
        await process_analyzer_data(slave, analyzer)

# 4. Shutdown
await slave.stop()
"""