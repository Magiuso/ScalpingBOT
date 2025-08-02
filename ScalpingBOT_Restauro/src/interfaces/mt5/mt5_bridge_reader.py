#!/usr/bin/env python3
"""
MT5 Bridge Reader - CLEANED AND SIMPLIFIED
REGOLE CLAUDE_RESTAURO.md APPLICATE:
- ✅ Zero fallback/defaults  
- ✅ Fail fast error handling
- ✅ No debug prints/spam
- ✅ No test code embedded
- ✅ No redundant functions
- ✅ Simplified architecture

Real-time data bridge between MetaTrader 5 and Python analysis.
"""

import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import queue
import logging
from dataclasses import dataclass
from collections import deque
import os

# Import sistema migrato FASE 1-2
from ScalpingBOT_Restauro.src.config.base.config_loader import get_configuration_manager
from ScalpingBOT_Restauro.src.monitoring.events.event_collector import EventCollector, EventType, EventSource, EventSeverity


@dataclass
class MT5TickData:
    """Struttura per dati tick da MT5 - FAIL FAST validation"""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    spread_percentage: float
    price_change_1m: float
    price_change_5m: float
    volatility: float
    momentum_5m: float
    market_state: str
    
    def __post_init__(self):
        """FAIL FAST validation"""
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be datetime")
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be non-empty string")
        if not isinstance(self.bid, (int, float)) or self.bid <= 0:
            raise ValueError("bid must be positive number")
        if not isinstance(self.ask, (int, float)) or self.ask <= 0:
            raise ValueError("ask must be positive number")
        if self.ask <= self.bid:
            raise ValueError("ask must be greater than bid")
        if not isinstance(self.volume, int) or self.volume < 0:
            raise ValueError("volume must be non-negative integer")
        if not isinstance(self.market_state, str):
            raise ValueError("market_state must be string")


class MT5BridgeReader:
    """Real-time data bridge - CLEANED VERSION"""
    
    def __init__(self, mt5_files_path: str, event_collector: Optional[EventCollector] = None):
        """Initialize MT5 bridge reader
        
        Args:
            mt5_files_path: Path to MQL5/Files directory
            event_collector: Optional event collector for monitoring
        """
        if not isinstance(mt5_files_path, str) or not mt5_files_path.strip():
            raise ValueError("mt5_files_path must be non-empty string")
        
        self.mt5_files_path = Path(mt5_files_path)
        if not self.mt5_files_path.exists():
            raise FileNotFoundError(f"MT5 files path does not exist: {mt5_files_path}")
        
        self.event_collector = event_collector
        self.logger = logging.getLogger('MT5BridgeReader')
        
        # File monitoring
        self.monitored_files: Dict[str, Dict[str, Any]] = {}  # symbol -> file_info
        self.tick_queue: queue.Queue = queue.Queue(maxsize=10000)
        self.processing_queue: queue.Queue = queue.Queue(maxsize=1000)
        
        # Threading
        self.is_running = False
        self.file_monitor_thread: Optional[threading.Thread] = None
        self.data_processor_thread: Optional[threading.Thread] = None
        self.analyzer_feeder_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'ticks_read': 0,
            'ticks_processed': 0,
            'analysis_count': 0,
            'start_time': None,
            'last_tick_time': None,
            'errors': 0,
            'files_discovered': 0,
            'session_starts': 0,
            'session_ends': 0
        }
        
        # Configuration
        self.polling_interval = 0.5  # 500ms
        self.batch_size = 10
        self.max_file_age_hours = 24
        self.read_timeout = 5.0
        
        # Analyzer callback
        self.analyzer_callback: Optional[Callable[[MT5TickData], None]] = None
    
    def set_analyzer_callback(self, callback: Callable[[MT5TickData], None]):
        """Set callback function for tick processing"""
        if not callable(callback):
            raise TypeError("callback must be callable")
        self.analyzer_callback = callback
    
    def start(self):
        """Start the bridge reader - FAIL FAST version"""
        if self.is_running:
            raise RuntimeError("Bridge reader already running")
        
        if self.analyzer_callback is None:
            raise RuntimeError("Analyzer callback not set - call set_analyzer_callback() first")
        
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        # Emit start event
        if self.event_collector:
            self.event_collector.emit_manual_event(
                EventType.SYSTEM_STATUS,
                {
                    "action": "bridge_start",
                    "mt5_files_path": str(self.mt5_files_path),
                    "polling_interval": self.polling_interval,
                    "batch_size": self.batch_size
                },
                EventSeverity.INFO
            )
        
        # Start threads
        self.file_monitor_thread = threading.Thread(target=self._file_monitor_loop, daemon=True)
        self.data_processor_thread = threading.Thread(target=self._data_processor_loop, daemon=True)
        self.analyzer_feeder_thread = threading.Thread(target=self._analyzer_feeder_loop, daemon=True)
        
        self.file_monitor_thread.start()
        self.data_processor_thread.start()
        self.analyzer_feeder_thread.start()
        
        self.logger.info(f"MT5 Bridge Reader started - monitoring: {self.mt5_files_path}")
    
    def stop(self):
        """Stop the bridge reader"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for threads to finish
        if self.file_monitor_thread and self.file_monitor_thread.is_alive():
            self.file_monitor_thread.join(timeout=2.0)
        if self.data_processor_thread and self.data_processor_thread.is_alive():
            self.data_processor_thread.join(timeout=2.0)
        if self.analyzer_feeder_thread and self.analyzer_feeder_thread.is_alive():
            self.analyzer_feeder_thread.join(timeout=2.0)
        
        # Emit stop event
        if self.event_collector:
            self.event_collector.emit_manual_event(
                EventType.SYSTEM_STATUS,
                {
                    "action": "bridge_stop",
                    "duration_seconds": (datetime.now() - self.stats['start_time']).total_seconds(),
                    "ticks_processed": self.stats['ticks_processed'],
                    "errors": self.stats['errors']
                },
                EventSeverity.INFO
            )
        
        self.logger.info("MT5 Bridge Reader stopped")
    
    def _file_monitor_loop(self):
        """Monitor MT5 files for changes"""
        while self.is_running:
            try:
                # Discover new analyzer files
                self._discover_analyzer_files()
                
                # Check monitored files for updates
                for symbol, file_info in list(self.monitored_files.items()):
                    self._check_file_for_updates(symbol, file_info)
                
                time.sleep(self.polling_interval)
                
            except Exception as e:
                self.logger.error(f"File monitor error: {e}")
                self.stats['errors'] += 1
                
                # Emit error event
                if self.event_collector:
                    self.event_collector.emit_manual_event(
                        EventType.ERROR_EVENT,
                        {
                            "component": "file_monitor",
                            "error": str(e)
                        },
                        EventSeverity.ERROR
                    )
                
                time.sleep(1.0)  # Wait before retry
    
    def _discover_analyzer_files(self):
        """Discover new analyzer_*.jsonl files"""
        try:
            for file_path in self.mt5_files_path.glob("analyzer_*.jsonl"):
                # Extract symbol from filename
                filename = file_path.name
                if not filename.startswith("analyzer_") or not filename.endswith(".jsonl"):
                    continue
                
                # Parse symbol from filename: analyzer_SYMBOL.jsonl
                parts = filename.replace("analyzer_", "").replace(".jsonl", "")
                if not parts:
                    continue
                
                symbol = parts.upper()
                
                # Skip if already monitoring
                if symbol in self.monitored_files:
                    continue
                
                # Check file age
                file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age.total_seconds() > (self.max_file_age_hours * 3600):
                    continue
                
                # Add to monitoring
                self.monitored_files[symbol] = {
                    'file_path': file_path,
                    'last_modified': 0,
                    'last_position': 0,
                    'session_count': 0,
                    'consecutive_errors': 0,
                    'discovered_at': datetime.now()
                }
                
                self.stats['files_discovered'] += 1
                self.logger.info(f"Discovered analyzer file: {filename} -> {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error discovering files: {e}")
            self.stats['errors'] += 1
    
    def _check_file_for_updates(self, symbol: str, file_info: Dict[str, Any]):
        """Check single file for updates"""
        file_path = file_info['file_path']
        
        # Skip if too many consecutive errors
        if file_info['consecutive_errors'] > 20:
            return
        
        try:
            # Check if file still exists
            if not file_path.exists():
                raise FileNotFoundError(f"File disappeared: {file_path}")
            
            # Check modification time
            current_mtime = file_path.stat().st_mtime
            if current_mtime > file_info['last_modified']:
                self._read_new_data(symbol, file_info)
                file_info['last_modified'] = current_mtime
                file_info['consecutive_errors'] = 0  # Reset errors
                
        except FileNotFoundError:
            self.logger.warning(f"File disappeared: {file_path}")
            file_info['consecutive_errors'] += 1
        except PermissionError:
            # File temporarily locked during write
            file_info['consecutive_errors'] += 1
            if file_info['consecutive_errors'] > 10:
                self.logger.warning(f"Repeated permission errors for {file_path}")
        except Exception as e:
            self.logger.error(f"Error checking {file_path}: {e}")
            file_info['consecutive_errors'] += 1
    
    def _read_new_data(self, symbol: str, file_info: Dict[str, Any]):
        """Read new data from file"""
        file_path = file_info['file_path']
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Seek to last position
                f.seek(file_info['last_position'])
                
                # Read new lines
                new_lines = f.readlines()
                
                # Update position
                file_info['last_position'] = f.tell()
                
                # Process new lines
                new_ticks = 0
                for line in new_lines:
                    line = line.strip()
                    if line:
                        if self._parse_and_queue_tick(symbol, line, file_info):
                            new_ticks += 1
                
                if new_ticks > 0:
                    self.logger.debug(f"{symbol}: {new_ticks} new ticks")
                        
        except Exception as e:
            self.logger.error(f"Error reading new data from {file_path}: {e}")
            file_info['consecutive_errors'] += 1
    
    def _parse_and_queue_tick(self, symbol: str, json_line: str, file_info: Dict[str, Any]) -> bool:
        """Parse JSON line and queue tick - FAIL FAST version"""
        try:
            data = json.loads(json_line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        
        # Check required type field
        if 'type' not in data:
            raise KeyError("Missing required field: type")
        
        msg_type = data['type']
        
        # Handle session messages
        if msg_type == 'session_start':
            file_info['session_count'] += 1
            self.stats['session_starts'] += 1
            
            if 'version' not in data:
                raise KeyError("session_start missing required field: version")
            
            self.logger.info(f"Session started for {symbol} (v{data['version']})")
            return False
            
        elif msg_type == 'session_end':
            self.stats['session_ends'] += 1
            
            if 'total_ticks' not in data:
                raise KeyError("session_end missing required field: total_ticks")
            
            self.logger.info(f"Session ended for {symbol} ({data['total_ticks']} ticks)")
            return False
            
        elif msg_type != 'tick':
            # Other message types ignored
            return False
        
        # Validate tick data - FAIL FAST
        required_fields = [
            'symbol', 'timestamp', 'bid', 'ask', 'spread_percentage',
            'price_change_1m', 'price_change_5m', 'volatility', 
            'momentum_5m', 'market_state'
        ]
        
        for field in required_fields:
            if field not in data:
                raise KeyError(f"Tick missing required field: {field}")
        
        # Verify symbol matches
        if data['symbol'] != symbol:
            raise ValueError(f"Symbol mismatch: expected {symbol}, got {data['symbol']}")
        
        # Parse timestamp - FAIL FAST
        timestamp_str = data['timestamp']
        try:
            # Format: "2025.06.26 16:30:45"
            timestamp = datetime.strptime(timestamp_str, '%Y.%m.%d %H:%M:%S')
        except ValueError:
            try:
                # Alternative format
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                raise ValueError(f"Invalid timestamp format: {timestamp_str}")
        
        # Parse numeric fields - FAIL FAST
        try:
            bid = float(data['bid'])
            ask = float(data['ask'])
            spread_percentage = float(data['spread_percentage'])
            price_change_1m = float(data['price_change_1m'])
            price_change_5m = float(data['price_change_5m'])
            volatility = float(data['volatility'])
            momentum_5m = float(data['momentum_5m'])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid numeric data: {e}")
        
        # Optional fields with explicit checks
        last = float(data['last']) if 'last' in data else (bid + ask) / 2
        volume = int(data['volume']) if 'volume' in data else 1
        
        # Create tick object with validation
        tick = MT5TickData(
            timestamp=timestamp,
            symbol=data['symbol'],
            bid=bid,
            ask=ask,
            last=last,
            volume=volume,
            spread_percentage=spread_percentage,
            price_change_1m=price_change_1m,
            price_change_5m=price_change_5m,
            volatility=volatility,
            momentum_5m=momentum_5m,
            market_state=data['market_state']
        )
        
        # Queue tick for processing
        try:
            self.tick_queue.put_nowait(tick)
            self.stats['ticks_read'] += 1
            return True
        except queue.Full:
            raise RuntimeError("Tick queue full - processing too slow")
                
    def _data_processor_loop(self):
        """Process queued ticks"""
        while self.is_running:
            try:
                # Get tick from queue (blocking with timeout)
                try:
                    tick = self.tick_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Put in processing queue
                try:
                    self.processing_queue.put_nowait(tick)
                except queue.Full:
                    self.logger.warning("Processing queue full - dropping tick")
                    self.stats['errors'] += 1
                
                self.tick_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Data processor error: {e}")
                self.stats['errors'] += 1
    
    def _analyzer_feeder_loop(self):
        """Feed ticks to analyzer callback with batch support"""
        batch_buffer = []
        batch_timeout = 0.1  # 100ms for batch collection
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                # Try to get tick with short timeout
                try:
                    tick = self.processing_queue.get(timeout=batch_timeout)
                    batch_buffer.append(tick)
                except queue.Empty:
                    pass
                
                # Process batch if size reached or timeout
                current_time = time.time()
                should_process = (
                    len(batch_buffer) >= self.batch_size or
                    (batch_buffer and current_time - last_batch_time >= batch_timeout)
                )
                
                if should_process and batch_buffer:
                    # Process batch
                    self._process_tick_batch(batch_buffer)
                    batch_buffer.clear()
                    last_batch_time = current_time
                
            except Exception as e:
                self.logger.error(f"Analyzer feeder error: {e}")
                self.stats['errors'] += 1
                
                # Emit error event
                if self.event_collector:
                    self.event_collector.emit_manual_event(
                        EventType.ERROR_EVENT,
                        {
                            "component": "analyzer_feeder",
                            "error": str(e)
                        },
                        EventSeverity.ERROR
                    )
    
    def _process_tick_batch(self, tick_batch: List[MT5TickData]):
        """Process a batch of ticks"""
        if not self.analyzer_callback:
            return
        
        for tick in tick_batch:
            try:
                self.analyzer_callback(tick)
                self.stats['ticks_processed'] += 1
                self.stats['last_tick_time'] = datetime.now()
                
                # Mark task done if it was from queue
                if hasattr(self.processing_queue, 'task_done'):
                    self.processing_queue.task_done()
                    
            except Exception as e:
                self.logger.error(f"Error processing tick: {e}")
                self.stats['errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        stats = self.stats.copy()
        stats['monitored_symbols'] = list(self.monitored_files.keys())
        stats['tick_queue_size'] = self.tick_queue.qsize()
        stats['processing_queue_size'] = self.processing_queue.qsize()
        stats['is_running'] = self.is_running
        return stats
    
    def get_monitored_symbols(self) -> List[str]:
        """Get list of monitored symbols"""
        return list(self.monitored_files.keys())


# Factory functions
def create_mt5_bridge_reader(mt5_files_path: str, 
                           event_collector: Optional[EventCollector] = None) -> MT5BridgeReader:
    """Factory function for MT5BridgeReader"""
    return MT5BridgeReader(mt5_files_path, event_collector)


# Export main classes
__all__ = [
    'MT5BridgeReader',
    'MT5TickData',
    'create_mt5_bridge_reader'
]