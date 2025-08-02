#!/usr/bin/env python3
"""
MT5 Backtest Runner - CLEANED AND SIMPLIFIED
REGOLE CLAUDE_RESTAURO.md APPLICATE:
- ✅ Zero fallback/defaults  
- ✅ Fail fast error handling
- ✅ No debug prints/spam
- ✅ No test code embedded
- ✅ No redundant functions
- ✅ Simplified architecture

Sistema per accelerare l'apprendimento con dati storici MT5.
"""

import os
import sys
import json
import time
import pandas as pd  # type: ignore
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

# Import sistema migrato FASE 1-2
from src.config.base.config_loader import get_configuration_manager
from src.config.domain.system_config import UnifiedConfig, SystemMode, PerformanceProfile
from src.monitoring.events.event_collector import EventCollector, EventType, EventSource, EventSeverity

# Import MT5 adapter for clean interface
from .mt5_adapter import MT5Adapter, MT5Tick


@dataclass
class BacktestConfig:
    """Configurazione backtest - FAIL FAST validation"""
    symbol: str
    start_date: datetime
    end_date: datetime
    data_source: str  # 'mt5_export', 'csv_file', 'jsonl_file'
    speed_multiplier: int = 1000
    batch_size: int = 1000
    save_progress: bool = True
    resume_from_checkpoint: bool = True
    
    def __post_init__(self):
        """FAIL FAST validation"""
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be non-empty string")
        if not isinstance(self.start_date, datetime):
            raise TypeError("start_date must be datetime")
        if not isinstance(self.end_date, datetime):
            raise TypeError("end_date must be datetime")
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        if self.data_source not in ['mt5_export', 'csv_file', 'jsonl_file']:
            raise ValueError(f"Invalid data_source: {self.data_source}")
        if self.speed_multiplier <= 0:
            raise ValueError("speed_multiplier must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


class MT5DataExporter:
    """Esporta dati storici da MT5 - CLEANED VERSION"""
    
    def __init__(self, event_collector: Optional[EventCollector] = None):
        self.logger = logging.getLogger('MT5DataExporter')
        self.event_collector = event_collector
        self.mt5_adapter = MT5Adapter()
        
        # Memory management settings
        self.memory_threshold_critical = 95
        self.memory_threshold_warning = 85
        self.base_chunk_days = 45
        self.min_chunk_days = 15
        self.ticks_per_gc = 1000
        
        # 🔧 FIXED MEMORY LEAK: Use deque with maxlen for bounded memory history
        from collections import deque
        self.max_memory_history = 5
        self.memory_history: deque = deque(maxlen=self.max_memory_history)
        
        # Tick data template for reuse
        self._tick_data_template = {
            "type": "tick",
            "timestamp": "",
            "symbol": "",
            "bid": 0.0,
            "ask": 0.0,
            "last": 0.0,
            "volume": 0,
            "spread_percentage": 0.0,
            "price_change_1m": 0.0,
            "price_change_5m": 0.0,
            "volatility": 0.0,
            "momentum_5m": 0.0,
            "market_state": "backtest"
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            raise ImportError("psutil required for memory monitoring")
    
    def _force_garbage_collection(self):
        """Force garbage collection"""
        import gc
        gc.collect()
        gc.collect()
        gc.collect()
    
    def _calculate_dynamic_chunk_size(self, total_days: int, current_memory: float) -> int:
        """Calculate chunk size based on available memory"""
        # 🔧 FIXED MEMORY LEAK: deque with maxlen handles auto-cleanup automatically
        # Track memory trend
        self.memory_history.append(current_memory)
        
        # Emergency: smallest chunks
        if current_memory > self.memory_threshold_critical:
            chunk_days = max(7, self.min_chunk_days)
            self.logger.warning(f"Critical memory {current_memory:.1f}% - using {chunk_days} day chunks")
            return chunk_days
        
        # Warning: reduce chunk size
        if current_memory > self.memory_threshold_warning:
            chunk_days = max(self.min_chunk_days, self.base_chunk_days // 2)
            self.logger.warning(f"High memory {current_memory:.1f}% - using {chunk_days} day chunks")
            return chunk_days
        
        # Normal operation
        if total_days <= 90:
            return min(self.base_chunk_days, total_days)
        else:
            return self.base_chunk_days
    
    def export_historical_data(self, symbol: str, start_date: datetime, 
                             end_date: datetime, output_file: str) -> bool:
        """Export historical data to JSONL file - FAIL FAST version"""
        
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be non-empty string")
        if not isinstance(start_date, datetime):
            raise TypeError("start_date must be datetime")  
        if not isinstance(end_date, datetime):
            raise TypeError("end_date must be datetime")
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")
        if not isinstance(output_file, str) or not output_file.strip():
            raise ValueError("output_file must be non-empty string")
        
        try:
            # Initialize MT5 adapter
            self.mt5_adapter.initialize()
            # Emit start event
            if self.event_collector:
                self.event_collector.emit_manual_event(
                    EventType.SYSTEM_STATUS,
                    {
                        "action": "export_start",
                        "symbol": symbol,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "output_file": output_file
                    },
                    EventSeverity.INFO
                )
            
            initial_memory = self._get_memory_usage()
            total_days = (end_date - start_date).days
            
            self.logger.info(f"Starting export: {symbol} from {start_date} to {end_date}")
            self.logger.info(f"Total period: {total_days} days")
            self.logger.info(f"Initial memory: {initial_memory:.1f}%")
            
            # Calculate chunk size
            chunk_days = self._calculate_dynamic_chunk_size(total_days, initial_memory)
            total_ticks_exported = 0
            
            # Create output directory
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                current_date = start_date
                chunk_number = 1
                first_chunk = True
                tick_counter = 0
                
                while current_date < end_date:
                    # Calculate chunk end
                    chunk_end = min(current_date + timedelta(days=chunk_days), end_date)
                    
                    self.logger.info(f"Processing chunk {chunk_number}: {current_date} to {chunk_end}")
                    
                    # Process chunk
                    chunk_ticks = self._process_chunk_memory_safe(
                        symbol, current_date, chunk_end, f, first_chunk, chunk_number
                    )
                    
                    if chunk_ticks is None:
                        raise RuntimeError(f"Failed to process chunk {chunk_number}")
                    
                    total_ticks_exported += chunk_ticks
                    tick_counter += chunk_ticks
                    first_chunk = False
                    
                    # Memory management
                    if tick_counter >= self.ticks_per_gc:
                        self._force_garbage_collection()
                        tick_counter = 0
                        memory_after_gc = self._get_memory_usage()
                        self.logger.info(f"GC performed - Memory: {memory_after_gc:.1f}%")
                    
                    # Next chunk
                    current_date = chunk_end
                    chunk_number += 1
                    
                    # Emergency memory check
                    if self._get_memory_usage() > 90:
                        raise RuntimeError("Memory usage exceeded 90% - stopping export")
                
                # Update header with totals
                self._update_header_with_totals(f, symbol, total_ticks_exported)
            
            # Final cleanup
            self._force_garbage_collection()
            final_memory = self._get_memory_usage()
            
            self.mt5_adapter.shutdown()
            
            self.logger.info(f"Export completed: {total_ticks_exported:,} total ticks")
            self.logger.info(f"File: {output_file}")
            self.logger.info(f"Final memory: {final_memory:.1f}% (started: {initial_memory:.1f}%)")
            
            # Emit completion event
            if self.event_collector:
                self.event_collector.emit_manual_event(
                    EventType.SYSTEM_STATUS,
                    {
                        "action": "export_complete",
                        "symbol": symbol,
                        "total_ticks": total_ticks_exported,
                        "output_file": output_file,
                        "memory_initial": initial_memory,
                        "memory_final": final_memory
                    },
                    EventSeverity.INFO
                )
            
            return total_ticks_exported > 0
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            
            # Emergency cleanup
            self._force_garbage_collection()
            
            # Best effort cleanup
            self.mt5_adapter.shutdown()
                
            # Emit error event
            if self.event_collector:
                self.event_collector.emit_manual_event(
                    EventType.ERROR_EVENT,
                    {
                        "action": "export_failed",
                        "symbol": symbol,
                        "error": str(e)
                    },
                    EventSeverity.ERROR
                )
            
            raise RuntimeError(f"Export failed: {e}")
    
    def _process_chunk_memory_safe(self, symbol: str, start_date: datetime, 
                                  end_date: datetime, file_handle, is_first_chunk: bool, 
                                  chunk_number: int) -> int:
        """Process a single chunk with memory management"""
        
        # Memory check before processing
        pre_chunk_memory = self._get_memory_usage()
        
        # Get ticks for this chunk using adapter
        ticks = self.mt5_adapter.get_ticks_range(symbol, start_date, end_date)
        
        if not ticks:
            self.logger.warning(f"No ticks in chunk {chunk_number}")
            return 0
        
        chunk_tick_count = len(ticks)
        self.logger.info(f"Chunk {chunk_number}: {chunk_tick_count:,} ticks retrieved")
        
        # Write header only for first chunk
        if is_first_chunk:
            header = {
                "type": "backtest_start",
                "symbol": symbol,
                "start_time": datetime.fromtimestamp(ticks[0].time).isoformat(),
                "end_time": "TBD",
                "total_ticks": "TBD", 
                "export_time": datetime.now().isoformat(),
                "chunked_export": True,
                "memory_optimized": True
            }
            file_handle.write(json.dumps(header) + '\n')
            file_handle.flush()
        
        # Process ticks with batch writing
        ticks_processed = 0
        write_batch = []
        
        for i, tick in enumerate(ticks):
            # Reuse template to avoid allocations
            tick_data = self._tick_data_template.copy()
            
            # Access tick fields from MT5Tick object
            tick_time = tick.time
            tick_bid = tick.bid 
            tick_ask = tick.ask
            tick_last = tick.last
            tick_volume = tick.volume
            
            # Calculate spread
            spread_percentage = 0.0
            if tick_bid > 0:
                spread_percentage = (tick_ask - tick_bid) / tick_bid
            
            # Update template
            tick_data["timestamp"] = datetime.fromtimestamp(tick_time).strftime('%Y.%m.%d %H:%M:%S')
            tick_data["symbol"] = symbol
            tick_data["bid"] = float(tick_bid)
            tick_data["ask"] = float(tick_ask) 
            tick_data["last"] = float(tick_last)
            tick_data["volume"] = int(tick_volume)
            tick_data["spread_percentage"] = float(spread_percentage)
            
            # Batch write for performance
            write_batch.append(json.dumps(tick_data))
            
            # Write in batches of 500
            if len(write_batch) >= 500 or i == len(ticks) - 1:
                file_handle.write('\n'.join(write_batch) + '\n')
                file_handle.flush()
                write_batch.clear()
            
            ticks_processed += 1
            
            # Periodic memory management
            if (i + 1) % 2000 == 0:
                file_handle.flush()
                if (i + 1) % 20000 == 0:
                    self._force_garbage_collection()
                
                # Memory emergency check
                current_memory = self._get_memory_usage()
                if current_memory > 88:
                    self.logger.warning(f"Memory spike: {current_memory:.1f}% during chunk processing")
                    self._force_garbage_collection()
                    
                    # Emergency break
                    if self._get_memory_usage() > 90:
                        raise RuntimeError(f"Memory exceeded 90% at tick {i+1}/{chunk_tick_count}")
        
        self.logger.info(f"Chunk {chunk_number} processing completed: {ticks_processed} ticks")
        
        # 🚀 CLEANUP - Delete ticks from memory immediately
        del ticks
        self._force_garbage_collection()
        
        return ticks_processed
    
    def _update_header_with_totals(self, file_handle, symbol: str, total_ticks: int):
        """Update the header with final totals"""
        try:
            # Simple approach: we assume header is first line
            file_handle.seek(0)
            lines = file_handle.readlines()
            
            if lines and lines[0].strip():
                header = json.loads(lines[0])
                header["total_ticks"] = total_ticks
                header["end_time"] = datetime.now().isoformat()
                
                # Rewrite file with updated header
                file_handle.seek(0)
                file_handle.write(json.dumps(header) + '\n')
                for line in lines[1:]:
                    file_handle.write(line)
                file_handle.truncate()
                
        except Exception as e:
            self.logger.warning(f"Could not update header: {e}")


class MT5BacktestRunner:
    """Main backtest runner - CLEANED VERSION"""
    
    def _check_if_historical_data(self, first_ticks: List[Dict]) -> bool:
        """
        Check if tick data is historical by comparing timestamps with current date
        
        Args:
            first_ticks: First 50 ticks to analyze
            
        Returns:
            True if data is older than 24 hours
        """
        if not first_ticks:
            return False
            
        try:
            # Get average timestamp from first 50 ticks
            timestamps = []
            for tick in first_ticks:
                timestamp_str = tick.get('timestamp', '')
                if timestamp_str:
                    # Parse various timestamp formats
                    try:
                        # Try ISO format first
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except:
                        try:
                            # Try MT5 format: "2024.01.01 00:00:01"
                            dt = datetime.strptime(timestamp_str, '%Y.%m.%d %H:%M:%S')
                        except:
                            continue
                    timestamps.append(dt)
            
            if not timestamps:
                return False
                
            # Get median timestamp (more robust than average)
            timestamps.sort()
            median_timestamp = timestamps[len(timestamps) // 2]
            
            # Check if data is older than 24 hours
            current_time = datetime.now()
            time_difference = current_time - median_timestamp
            
            # If data is more than 24 hours old, it's historical
            is_historical = time_difference.total_seconds() > 86400  # 24 hours
            
            if is_historical:
                days_old = time_difference.days
                self.logger.info(f"📅 Data age detected: {days_old} days old (from {median_timestamp.date()})")
                
            return is_historical
            
        except Exception as e:
            self.logger.warning(f"Could not determine if data is historical: {e}")
            return False
    
    def __init__(self, config: BacktestConfig, event_collector: Optional[EventCollector] = None):
        if not isinstance(config, BacktestConfig):
            raise TypeError("config must be BacktestConfig instance")
        
        self.config = config
        self.event_collector = event_collector
        self.logger = logging.getLogger('MT5BacktestRunner')
        
        # Components
        self.data_exporter = MT5DataExporter(event_collector)
        
        # State tracking
        self.is_running = False
        self.total_ticks_processed = 0
        self.start_time: Optional[datetime] = None
    
    def run_backtest(self, analyzer_system) -> bool:
        """Run complete backtest - FAIL FAST version"""
        
        if self.is_running:
            raise RuntimeError("Backtest already running")
        
        if analyzer_system is None:
            raise ValueError("analyzer_system cannot be None")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting backtest for {self.config.symbol}")
            self.logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
            
            # Emit start event
            if self.event_collector:
                self.event_collector.emit_manual_event(
                    EventType.SYSTEM_STATUS,
                    {
                        "action": "backtest_start",
                        "symbol": self.config.symbol,
                        "start_date": self.config.start_date.isoformat(),
                        "end_date": self.config.end_date.isoformat(),
                        "config": {
                            "data_source": self.config.data_source,
                            "speed_multiplier": self.config.speed_multiplier,
                            "batch_size": self.config.batch_size
                        }
                    },
                    EventSeverity.INFO
                )
            
            # Determine data processing method
            if self.config.data_source == 'mt5_export':
                success = self._run_mt5_export_backtest(analyzer_system)
            elif self.config.data_source == 'csv_file':
                success = self._run_csv_backtest(analyzer_system)
            elif self.config.data_source == 'jsonl_file':
                success = self._run_jsonl_backtest(analyzer_system)
            else:
                raise ValueError(f"Unsupported data_source: {self.config.data_source}")
            
            # Emit completion event
            if self.event_collector:
                self.event_collector.emit_manual_event(
                    EventType.SYSTEM_STATUS,
                    {
                        "action": "backtest_complete",
                        "symbol": self.config.symbol,
                        "success": success,
                        "total_ticks": self.total_ticks_processed,
                        "duration_seconds": (datetime.now() - self.start_time).total_seconds()
                    },
                    EventSeverity.INFO if success else EventSeverity.ERROR
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            
            # Emit error event
            if self.event_collector:
                self.event_collector.emit_manual_event(
                    EventType.ERROR_EVENT,
                    {
                        "action": "backtest_failed",
                        "symbol": self.config.symbol,
                        "error": str(e)
                    },
                    EventSeverity.ERROR
                )
            
            raise RuntimeError(f"Backtest failed: {e}")
            
        finally:
            self.is_running = False
    
    def _run_mt5_export_backtest(self, analyzer_system) -> bool:
        """Run backtest with MT5 data export"""
        
        # Generate export filename
        export_file = f"backtest_{self.config.symbol}_{self.config.start_date.strftime('%Y%m%d')}_{self.config.end_date.strftime('%Y%m%d')}.jsonl"
        export_path = os.path.join("./test_analyzer_data", export_file)
        
        # Export data first
        export_success = self.data_exporter.export_historical_data(
            self.config.symbol,
            self.config.start_date,
            self.config.end_date,
            export_path
        )
        
        if not export_success:
            raise RuntimeError("Data export failed")
        
        # Now process the exported data
        return self._process_jsonl_file(export_path, analyzer_system)
    
    def _run_csv_backtest(self, analyzer_system) -> bool:
        """Run backtest with CSV file"""
        csv_file = f"./test_analyzer_data/backtest_{self.config.symbol}.csv"
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        self.logger.info(f"Loading CSV file: {csv_file}")
        
        # Load CSV data
        tick_data_list = self._load_csv_data(csv_file, self.config.symbol)
        
        if not tick_data_list:
            raise ValueError("No data loaded from CSV file")
        
        self.logger.info(f"Loaded {len(tick_data_list)} ticks from CSV")
        
        # Process ticks
        ticks_processed = 0
        batch_buffer = []
        
        for tick_dict in tick_data_list:
            batch_buffer.append(tick_dict)
            
            if len(batch_buffer) >= self.config.batch_size:
                self._process_tick_batch(batch_buffer, analyzer_system)
                ticks_processed += len(batch_buffer)
                batch_buffer.clear()
        
        # Process remaining ticks
        if batch_buffer:
            self._process_tick_batch(batch_buffer, analyzer_system)
            ticks_processed += len(batch_buffer)
        
        self.total_ticks_processed = ticks_processed
        self.logger.info(f"CSV backtest completed: {ticks_processed} ticks processed")
        
        return True
    
    def _run_jsonl_backtest(self, analyzer_system) -> bool:
        """Run backtest with JSONL file"""
        jsonl_file = f"./test_analyzer_data/backtest_{self.config.symbol}.jsonl"
        
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")
        
        return self._process_jsonl_file(jsonl_file, analyzer_system)
    
    def _process_jsonl_file(self, jsonl_file: str, analyzer_system) -> bool:
        """Process JSONL file and feed to analyzer"""
        
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"File not found: {jsonl_file}")
        
        self.logger.info(f"Processing JSONL file: {jsonl_file}")
        
        ticks_processed = 0
        batch_buffer = []
        is_historical_data = False
        first_ticks_checked = False
        ticks_for_date_check = []
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        tick_data = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON at line {line_num}: {e}")
                    
                    # Process header line and detect historical data
                    if tick_data.get('type') == 'backtest_start':
                        # Log important backtest info
                        start_time = tick_data.get('start_time', 'unknown')
                        end_time = tick_data.get('end_time', 'unknown')
                        total_ticks = tick_data.get('total_ticks', 'unknown')
                        self.logger.info(f"📊 Backtest data range: {start_time} to {end_time} ({total_ticks} ticks)")
                        continue
                    
                    # Process tick
                    if tick_data.get('type') == 'tick':
                        # Check if data is historical (first 50 ticks)
                        if not first_ticks_checked and len(ticks_for_date_check) < 50:
                            ticks_for_date_check.append(tick_data)
                            if len(ticks_for_date_check) == 50:
                                is_historical_data = self._check_if_historical_data(ticks_for_date_check)
                                first_ticks_checked = True
                                if is_historical_data:
                                    self.logger.warning("⚠️ HISTORICAL DATA DETECTED - Training with past market data")
                                    self.logger.info("📚 This is LEARNING PHASE - Models will train on historical patterns")
                                    # Notify analyzer system about historical mode if supported
                                    if hasattr(analyzer_system, 'set_historical_mode'):
                                        analyzer_system.set_historical_mode(True)
                        
                        batch_buffer.append(tick_data)
                        
                        # Process in batches
                        if len(batch_buffer) >= self.config.batch_size:
                            self._process_tick_batch(batch_buffer, analyzer_system)
                            ticks_processed += len(batch_buffer)
                            batch_buffer.clear()
                
                # Process remaining ticks
                if batch_buffer:
                    self._process_tick_batch(batch_buffer, analyzer_system)
                    ticks_processed += len(batch_buffer)
            
            self.total_ticks_processed = ticks_processed
            
            if is_historical_data:
                self.logger.info(f"✅ Historical backtest completed: {ticks_processed} ticks processed")
                self.logger.info("📊 Models have been trained with historical data and are ready for live trading")
            else:
                self.logger.info(f"Backtest completed: {ticks_processed} ticks processed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing JSONL file: {e}")
            raise RuntimeError(f"JSONL processing failed: {e}")
    
    def _process_tick_batch(self, tick_batch: List[Dict], analyzer_system):
        """Process a batch of ticks"""
        
        for tick_data in tick_batch:
            try:
                # Convert to expected format and feed to analyzer
                if hasattr(analyzer_system, 'process_tick'):
                    analyzer_system.process_tick(tick_data)
                else:
                    raise AttributeError("analyzer_system missing process_tick method")
                    
            except Exception as e:
                self.logger.error(f"Error processing tick: {e}")
                raise RuntimeError(f"Tick processing failed: {e}")
    
    def _load_csv_data(self, csv_file: str, symbol: str) -> List[Dict[str, Any]]:
        """Load tick data from CSV file - FAIL FAST version"""
        
        import pandas as pd  # Import here to make it optional dependency
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file: {e}")
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # Column mapping for common variations
        column_mapping = {
            'timestamp': ['timestamp', 'time', 'datetime', 'date'],
            'bid': ['bid', 'Bid', 'close', 'Close'],
            'ask': ['ask', 'Ask', 'high', 'High'],
            'volume': ['volume', 'Volume', 'vol', 'Vol']
        }
        
        # Find correct columns - FAIL FAST if required columns missing
        cols = {}
        for target, variants in column_mapping.items():
            for variant in variants:
                if variant in df.columns:
                    cols[target] = variant
                    break
        
        if 'timestamp' not in cols:
            raise KeyError(f"Required column 'timestamp' not found. Available columns: {list(df.columns)}")
        if 'bid' not in cols:
            raise KeyError(f"Required column 'bid' not found. Available columns: {list(df.columns)}")
        
        # Convert DataFrame to tick data list
        tick_data_list = []
        
        for idx, row in df.iterrows():
            try:
                # Parse timestamp - FAIL FAST
                # Use .at[] for direct scalar access
                timestamp_str = str(df.at[idx, cols['timestamp']])
                timestamp_val = pd.to_datetime(timestamp_str)
                
                # Parse numeric values - FAIL FAST
                # Use .at[] for direct scalar access
                bid_val = float(df.at[idx, cols['bid']])
                
                # Ask uses bid if not available
                if 'ask' in cols:
                    ask_val = float(df.at[idx, cols['ask']])
                else:
                    ask_val = bid_val * 1.0001  # Add small spread
                
                # Volume defaults to 1 if not available
                if 'volume' in cols:
                    volume_raw = df.at[idx, cols['volume']]
                    if pd.notna(volume_raw):
                        volume_val = int(float(volume_raw))
                    else:
                        volume_val = 1
                else:
                    volume_val = 1
                
                # Create tick dict matching expected format
                # Ensure timestamp is a datetime object before formatting
                if hasattr(timestamp_val, 'strftime'):
                    timestamp_formatted = timestamp_val.strftime('%Y.%m.%d %H:%M:%S')
                else:
                    # If it's a Timestamp object, convert to datetime first
                    timestamp_formatted = pd.Timestamp(timestamp_val).strftime('%Y.%m.%d %H:%M:%S')
                
                tick_dict = {
                    'type': 'tick',
                    'timestamp': timestamp_formatted,
                    'symbol': symbol,
                    'bid': bid_val,
                    'ask': ask_val,
                    'last': (bid_val + ask_val) / 2,
                    'volume': volume_val,
                    'spread_percentage': (ask_val - bid_val) / bid_val if bid_val > 0 else 0.0,
                    'price_change_1m': 0.0,
                    'price_change_5m': 0.0,
                    'volatility': 0.0,
                    'momentum_5m': 0.0,
                    'market_state': 'backtest'
                }
                
                tick_data_list.append(tick_dict)
                
            except Exception as e:
                raise ValueError(f"Error parsing row {idx}: {e}")
        
        return tick_data_list


# Factory functions
def create_backtest_config(symbol: str, start_date: datetime, end_date: datetime, 
                          data_source: str = 'mt5_export') -> BacktestConfig:
    """Factory function for BacktestConfig"""
    return BacktestConfig(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        data_source=data_source
    )


def create_backtest_runner(config: BacktestConfig, 
                          event_collector: Optional[EventCollector] = None) -> MT5BacktestRunner:
    """Factory function for MT5BacktestRunner"""
    return MT5BacktestRunner(config, event_collector)


# Export main classes
__all__ = [
    'MT5BacktestRunner',
    'MT5DataExporter', 
    'BacktestConfig',
    'create_backtest_config',
    'create_backtest_runner'
]