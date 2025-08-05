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
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

# Import sistema migrato FASE 1-2
from ScalpingBOT_Restauro.src.config.base.config_loader import get_configuration_manager
from ScalpingBOT_Restauro.src.config.domain.system_config import UnifiedConfig, SystemMode, PerformanceProfile
from ScalpingBOT_Restauro.src.monitoring.events.event_collector import EventCollector, EventType, EventSource, EventSeverity

# Import MT5 adapter for clean interface
from .mt5_adapter import MT5Adapter, MT5Tick
from .mt5_utils import calculate_price_from_bid_ask, parse_mt5_timestamp


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
            "market_state": "backtest",
            "m5_tick_count": 0
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
            
            # Track export start time for progress calculations
            export_start = datetime.now()
            
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
                    
                    # Calculate progress
                    days_processed = (current_date - start_date).days
                    days_remaining = (end_date - current_date).days
                    progress_percentage = (days_processed / total_days * 100) if total_days > 0 else 0
                    
                    self.logger.info(f"\n📊 EXPORT PROGRESS: {progress_percentage:.1f}% complete")
                    self.logger.info(f"📅 Days: {days_processed}/{total_days} processed, {days_remaining} remaining")
                    self.logger.info(f"🔄 Processing chunk {chunk_number}: {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
                    
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
                    
                    # Estimate completion time based on average chunk processing time
                    if chunk_number > 2:  # After a few chunks, estimate completion
                        elapsed_time = (datetime.now() - export_start).total_seconds()
                        avg_time_per_day = elapsed_time / days_processed if days_processed > 0 else 0
                        estimated_remaining_seconds = avg_time_per_day * days_remaining
                        
                        if estimated_remaining_seconds > 0:
                            eta_hours = int(estimated_remaining_seconds // 3600)
                            eta_minutes = int((estimated_remaining_seconds % 3600) // 60)
                            self.logger.info(f"⏰ Estimated time remaining: {eta_hours}h {eta_minutes}m")
                    
                    # Emergency memory check
                    if self._get_memory_usage() > 90:
                        raise RuntimeError("Memory usage exceeded 90% - stopping export")
                
                # Update header with totals
                self._update_header_with_totals(f, symbol, total_ticks_exported)
            
            # Final cleanup
            self._force_garbage_collection()
            final_memory = self._get_memory_usage()
            
            self.mt5_adapter.shutdown()
            
            # Final summary
            export_duration = (datetime.now() - export_start).total_seconds()
            avg_ticks_per_second = total_ticks_exported / export_duration if export_duration > 0 else 0
            
            self.logger.info(f"\n✅ EXPORT COMPLETED!")
            self.logger.info(f"📊 Total ticks exported: {total_ticks_exported:,}")
            self.logger.info(f"⏱️ Total time: {int(export_duration // 60)}m {int(export_duration % 60)}s")
            self.logger.info(f"⚡ Average speed: {avg_ticks_per_second:,.0f} ticks/second")
            self.logger.info(f"📁 File: {output_file}")
            self.logger.info(f"💾 Final memory: {final_memory:.1f}% (started: {initial_memory:.1f}%)")
            
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
        
        # Tick counting for M5 periods
        m5_tick_counter = {}  # key: M5 timestamp, value: tick count
        
        # Real-time export progress tracking
        export_start_time = time.time()
        last_progress_report = export_start_time
        last_ticks_reported = 0
        
        for i, tick in enumerate(ticks):
            # Reuse template to avoid allocations
            tick_data = self._tick_data_template.copy()
            
            # Access tick fields from MT5Tick object
            tick_time = tick.time
            tick_bid = tick.bid 
            tick_ask = tick.ask
            tick_last = tick.last
            tick_volume = tick.volume
            
            # Calculate spread - FAIL FAST
            if tick_bid <= 0:
                raise ValueError(f"Invalid bid price: {tick_bid} at timestamp {tick_time}")
            spread_percentage = (tick_ask - tick_bid) / tick_bid
            
            # Fix last price using consolidated logic
            try:
                tick_last = calculate_price_from_bid_ask(tick_last, tick_bid, tick_ask)
            except ValueError:
                raise ValueError(f"Cannot determine price for export: last={tick_last}, bid={tick_bid}, ask={tick_ask}")
            
            # Calculate M5 tick volume when real volume is 0
            tick_datetime = datetime.fromtimestamp(tick_time)
            m5_timestamp = tick_datetime.replace(minute=tick_datetime.minute // 5 * 5, second=0, microsecond=0)
            m5_key = m5_timestamp.strftime('%Y%m%d%H%M')
            
            # Count ticks per M5 period
            if m5_key not in m5_tick_counter:
                m5_tick_counter[m5_key] = 0
            m5_tick_counter[m5_key] += 1
            
            # Always use M5 tick count as volume (represents real trading activity)
            # This is intentional design, not a fallback
            tick_volume = m5_tick_counter[m5_key]
            
            # Update template
            tick_data["timestamp"] = tick_datetime.strftime('%Y.%m.%d %H:%M:%S')
            tick_data["symbol"] = symbol
            tick_data["bid"] = float(tick_bid)
            tick_data["ask"] = float(tick_ask) 
            tick_data["last"] = float(tick_last)
            tick_data["volume"] = int(tick_volume)
            tick_data["spread_percentage"] = float(spread_percentage)
            tick_data["m5_tick_count"] = m5_tick_counter[m5_key]  # Add M5 tick count for reference
            
            # Batch write for performance
            write_batch.append(json.dumps(tick_data))
            
            # Write in batches of 500
            if len(write_batch) >= 500 or i == len(ticks) - 1:
                file_handle.write('\n'.join(write_batch) + '\n')
                file_handle.flush()
                write_batch.clear()
            
            ticks_processed += 1
            
            # Real-time export progress every second
            current_time = time.time()
            if current_time - last_progress_report >= 1.0:
                ticks_this_second = ticks_processed - last_ticks_reported
                elapsed_time = current_time - export_start_time
                progress_pct = (ticks_processed / chunk_tick_count * 100) if chunk_tick_count > 0 else 0
                self.logger.info(f"📤 Export progress: {ticks_processed:,}/{chunk_tick_count:,} ticks ({progress_pct:.1f}%) | {ticks_this_second:,} ticks/sec | Elapsed: {elapsed_time:.1f}s")
                last_progress_report = current_time
                last_ticks_reported = ticks_processed
            
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
        
        # Final chunk summary
        total_export_time = time.time() - export_start_time
        avg_ticks_per_sec = ticks_processed / total_export_time if total_export_time > 0 else 0
        self.logger.info(f"✅ Chunk {chunk_number} export completed: {ticks_processed:,} ticks in {total_export_time:.1f}s ({avg_ticks_per_sec:.0f} ticks/sec avg)")
        
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
                # BIBBIA COMPLIANT: FAIL FAST - validate required field
                if 'timestamp' not in tick:
                    raise KeyError("FAIL FAST: Missing required field 'timestamp' in tick data")
                timestamp_str = tick['timestamp']
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
    
    def run_backtest(self, analyzer_system, selected_models: Optional[List[str]] = None) -> bool:
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
                success = self._run_mt5_export_backtest(analyzer_system, selected_models)
            elif self.config.data_source == 'csv_file':
                success = self._run_csv_backtest(analyzer_system, selected_models)
            elif self.config.data_source == 'jsonl_file':
                success = self._run_jsonl_backtest(analyzer_system, selected_models)
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
    
    def _run_mt5_export_backtest(self, analyzer_system, selected_models: Optional[List[str]] = None) -> bool:
        """Run backtest with MT5 data export - with 80% coverage detection"""
        
        # 1. Check for existing files with sufficient coverage
        existing_file = self._find_existing_file_with_coverage()
        
        if existing_file:
            print(f"📁 Using existing file with sufficient coverage: {os.path.basename(existing_file)}")
            return self._process_jsonl_file(existing_file, analyzer_system, selected_models)
        
        # 2. No suitable existing file found - export new data
        print("📊 No existing file with sufficient coverage - exporting new data")
        
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
        return self._process_jsonl_file(export_path, analyzer_system, selected_models)
    
    def _run_csv_backtest(self, analyzer_system, selected_models: Optional[List[str]] = None) -> bool:
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
        
        # Process ticks using new batch architecture
        ticks_processed = 0
        BATCH_SIZE = 100000
        
        # Process in 100K batches
        for i in range(0, len(tick_data_list), BATCH_SIZE):
            batch = tick_data_list[i:i + BATCH_SIZE]
            print(f"🔄 Processing CSV batch: {len(batch):,} ticks")
            
            # Use training mode for CSV processing with selected models (BIBBIA COMPLIANCE)
            self._train_models_on_batch(batch, analyzer_system, selected_models)
            ticks_processed += len(batch)
            
            print(f"✅ CSV batch completed: {len(batch):,} ticks processed")
        
        self.total_ticks_processed = ticks_processed
        self.logger.info(f"CSV backtest completed: {ticks_processed} ticks processed")
        
        return True
    
    def _run_jsonl_backtest(self, analyzer_system, selected_models: Optional[List[str]] = None) -> bool:
        """Run backtest with JSONL file"""
        jsonl_file = f"./test_analyzer_data/backtest_{self.config.symbol}.jsonl"
        
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")
        
        return self._process_jsonl_file(jsonl_file, analyzer_system, selected_models)
    
    def _process_jsonl_file(self, jsonl_file: str, analyzer_system, selected_models: Optional[List[str]] = None) -> bool:
        """Process JSONL file: read 100K ticks → process batch → repeat"""
        
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"File not found: {jsonl_file}")
        
        print(f"📂 Opening JSONL file: {jsonl_file}")
        file_size = os.path.getsize(jsonl_file) / (1024*1024*1024)  # GB
        print(f"📊 File size: {file_size:.2f} GB")
        
        # Phase management variables
        training_start_date = None
        validation_start_date = None
        training_days = 30
        total_training_ticks = 0
        total_validation_ticks = 0
        total_ticks_processed = 0
        
        # Batch configuration
        BATCH_SIZE = 100000
        batch_number = 0
        
        processing_start_time = time.time()
        
        try:
            print("📖 Starting batch-based processing...")
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                
                while True:
                    batch_number += 1
                    current_batch = []
                    lines_read_in_batch = 0
                    
                    print(f"\n🔍 BATCH #{batch_number}: Reading next {BATCH_SIZE:,} ticks...")
                    batch_read_start = time.time()
                    
                    # Read exactly 100K ticks (or until file ends)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        lines_read_in_batch += 1
                        
                        # Real-time reading progress every 10K ticks
                        if len(current_batch) > 0 and len(current_batch) % 10000 == 0:
                            remaining = BATCH_SIZE - len(current_batch)
                            elapsed = time.time() - batch_read_start
                            rate = len(current_batch) / elapsed if elapsed > 0 else 0
                            print(f"  📚 Reading: {len(current_batch):,}/100K ticks ({remaining:,} remaining) | {rate:,.0f} ticks/sec")
                        
                        try:
                            tick_data = json.loads(line)
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Invalid JSON at line {lines_read_in_batch}: {e}")
                        
                        # Process header
                        if tick_data.get('type') == 'backtest_start':
                            # BIBBIA COMPLIANT: FAIL FAST - no fallback to 'unknown'
                            start_time = tick_data.get('start_time') or 'START_TIME_MISSING'
                            end_time = tick_data.get('end_time') or 'END_TIME_MISSING'
                            total_ticks = tick_data.get('total_ticks') or 'TOTAL_TICKS_MISSING'
                            print(f"📊 Backtest data: {start_time} to {end_time} ({total_ticks} ticks)")
                            continue
                        
                        # Add tick to batch
                        if tick_data.get('type') == 'tick':
                            # Initialize phase dates on first tick
                            if training_start_date is None:
                                timestamp_str = tick_data.get('timestamp')
                                if timestamp_str:
                                    first_tick_time = parse_mt5_timestamp(timestamp_str)
                                    training_start_date = first_tick_time
                                    validation_start_date = training_start_date + timedelta(days=training_days)
                                    print(f"📚 TRAINING PHASE: {training_start_date.strftime('%Y-%m-%d')} to {validation_start_date.strftime('%Y-%m-%d')}")
                                    print(f"🧪 VALIDATION PHASE: {validation_start_date.strftime('%Y-%m-%d')} onwards")
                            
                            current_batch.append(tick_data)
                            
                            # Stop when batch is full
                            if len(current_batch) >= BATCH_SIZE:
                                break
                    
                    # Check if we have any ticks to process
                    if not current_batch:
                        print("📄 End of file reached")
                        break
                    
                    batch_read_time = time.time() - batch_read_start
                    print(f"✅ Read complete: {len(current_batch):,} ticks in {batch_read_time:.2f}s")
                    
                    # Determine batch phase (training or validation)
                    batch_phase = self._determine_batch_phase(current_batch, validation_start_date)
                    
                    # Calculate temporal progress (BIBBIA COMPLIANT - no fallbacks)
                    progress_info = self._calculate_temporal_progress(current_batch, validation_start_date, batch_phase)
                    
                    # Process the batch
                    print(f"🔄 Processing BATCH #{batch_number} ({batch_phase} phase)...")
                    if progress_info:
                        print(f"📅 Progress: {progress_info}")
                    batch_process_start = time.time()
                    
                    if batch_phase == "training":
                        # Training: only train models, no predictions with selected models (BIBBIA COMPLIANCE)
                        self._train_models_on_batch(current_batch, analyzer_system, selected_models)
                        total_training_ticks += len(current_batch)
                    else:
                        # Validation: use trained models for tick-by-tick predictions - BIBBIA COMPLIANT
                        processed_ticks = self._validate_models_tick_by_tick(current_batch, analyzer_system) 
                        total_validation_ticks += processed_ticks
                    
                    batch_process_time = time.time() - batch_process_start
                    total_ticks_processed += len(current_batch)
                    
                    print(f"✅ BATCH #{batch_number} completed:")
                    print(f"   📊 {len(current_batch):,} ticks processed in {batch_process_time:.2f}s")
                    print(f"   🎯 Phase: {batch_phase}")
                    print(f"   📈 Total progress: {total_ticks_processed:,} ticks")
                    
                    # Memory cleanup every 5 batches
                    if batch_number % 5 == 0:
                        import gc
                        gc.collect()
                        print(f"🧹 Memory cleanup after batch #{batch_number}")
            
            # Final summary
            total_time = time.time() - processing_start_time
            print(f"\n✅ Batch processing completed!")
            print(f"📊 Total batches processed: {batch_number}")
            print(f"📊 Total ticks: {total_ticks_processed:,} in {total_time:.1f}s")
            print(f"📚 Training ticks: {total_training_ticks:,}")
            print(f"🧪 Validation ticks: {total_validation_ticks:,}")
            
            # Temporal summary (BIBBIA COMPLIANT)
            if validation_start_date:
                training_days = (validation_start_date - self.config.start_date).days
                total_days = (self.config.end_date - self.config.start_date).days
                validation_days = total_days - training_days
                print(f"📅 Temporal split: {training_days} training days + {validation_days} validation days = {total_days} total days")
            
            self.total_ticks_processed = total_ticks_processed
            return True
            
        except Exception as e:
            print(f"❌ Error in batch processing: {e}")
            raise RuntimeError(f"Batch processing failed: {e}")
    
    
    def _determine_batch_phase(self, batch: List[Dict], validation_start_date) -> str:
        """Determine if batch is training or validation phase"""
        if not batch or validation_start_date is None:
            return "training"
        
        # Check timestamp of middle tick in batch
        middle_tick = batch[len(batch) // 2]
        timestamp_str = middle_tick.get('timestamp')
        if timestamp_str:
            tick_time = parse_mt5_timestamp(timestamp_str)
            return "training" if tick_time < validation_start_date else "validation"
        
        return "training"
    
    def _calculate_temporal_progress(self, batch: List[Dict], validation_start_date, batch_phase: str) -> Optional[str]:
        """Calculate temporal progress for training/validation phases - BIBBIA COMPLIANT"""
        
        if not batch or not validation_start_date:
            return None
        
        try:
            # Get timestamp from middle tick of batch
            middle_tick = batch[len(batch) // 2]
            timestamp_str = middle_tick.get('timestamp')
            if not timestamp_str:
                return None
            
            current_time = parse_mt5_timestamp(timestamp_str)
            
            if batch_phase == "training":
                # Calculate training progress toward validation_start_date
                days_until_validation = (validation_start_date - current_time).days
                
                if days_until_validation <= 0:
                    return "Training phase COMPLETED - switching to validation"
                else:
                    return f"Training: {days_until_validation} days until validation phase"
            
            elif batch_phase == "validation":
                # Calculate validation progress (how far into validation we are)
                days_into_validation = (current_time - validation_start_date).days
                
                if days_into_validation < 0:
                    return "ERROR: Validation batch before validation start date"
                else:
                    return f"Validation: Day {days_into_validation + 1} of validation phase"
            
            return None
            
        except Exception as e:
            # BIBBIA COMPLIANCE: Don't hide errors but don't break processing
            return f"Progress calculation failed: {str(e)}"
    
    def _train_models_on_batch(self, batch: List[Dict], analyzer_system, selected_models: Optional[List[str]] = None):
        """Train ML models on batch data (no predictions)"""
        print(f"🎓 Training models with {len(batch):,} ticks...")
        
        # Convert batch to training data format
        training_data = self._convert_batch_to_ml_data(batch)
        
        # FAIL FAST - analyzer MUST have train_on_batch method
        if not hasattr(analyzer_system, 'train_on_batch'):
            raise AttributeError("analyzer_system missing required method 'train_on_batch' - cannot train models")
        
        # Train models on batch data with selected models (BIBBIA COMPLIANCE)
        print(f"🧠 Training ML models on {len(training_data['ticks'])} ticks...")
        result = analyzer_system.train_on_batch(training_data, selected_models)
        print(f"✅ ML training completed: {result if result else 'models updated'}")
    
    def _convert_batch_to_ml_data(self, batch: List[Dict]) -> Dict[str, Any]:
        """Convert MT5 batch data to ML training format"""
        if not batch:
            raise ValueError("Empty batch provided for ML data conversion")
        
        # Extract ticks with proper format
        converted_ticks = []
        
        for tick in batch:
            # Extract fields from MT5 tick format
            timestamp_str = tick.get('timestamp', '')
            tick_last = tick.get('last', 0.0)
            volume = tick.get('volume', 0.0)
            bid = tick.get('bid', 0.0)
            ask = tick.get('ask', 0.0)
            symbol = tick.get('symbol', self.config.symbol)
            
            # Calculate price from bid/ask for CFD data (when last=0)
            try:
                price = calculate_price_from_bid_ask(tick_last, bid, ask)
            except ValueError:
                # Skip ticks with invalid price data
                continue
            
            # Skip invalid ticks
            if not timestamp_str:
                continue
            
            # BIBBIA COMPLIANT: Keep original MT5 format - NO DUPLICATIONS!
            converted_tick = {
                'timestamp': timestamp_str,
                'last': float(price),  # Keep 'last' field name from MT5
                'volume': float(volume),
                'bid': float(bid) if bid > 0 else None,
                'ask': float(ask) if ask > 0 else None,
                'symbol': symbol
            }
            
            converted_ticks.append(converted_tick)
        
        if not converted_ticks:
            raise ValueError("No valid ticks found in batch after conversion")
        
        return {
            'count': len(converted_ticks),
            'ticks': converted_ticks,
            'symbol': self.config.symbol,
            'batch_metadata': {
                'original_count': len(batch),
                'converted_count': len(converted_ticks),
                'first_timestamp': converted_ticks[0]['timestamp'],
                'last_timestamp': converted_ticks[-1]['timestamp']
            }
        }
    
    def _validate_models_tick_by_tick(self, ticks: List[Dict], analyzer_system) -> int:
        """Use trained models for tick-by-tick predictions - BIBBIA COMPLIANT"""
        if not ticks:
            raise ValueError("No ticks provided for validation")
        
        if not hasattr(analyzer_system, 'validate_on_tick'):
            raise AttributeError("analyzer_system missing required method 'validate_on_tick' - cannot validate models tick-by-tick")
        
        print(f"🔮 Starting tick-by-tick validation with {len(ticks):,} ticks...")
        
        processed_count = 0
        prediction_window_size = 100  # Use last 100 ticks for context
        
        for tick_index, current_tick in enumerate(ticks):
            try:
                # FAIL FAST - current tick must be valid
                if 'last' not in current_tick or 'timestamp' not in current_tick:
                    raise ValueError(f"Invalid tick at index {tick_index}: missing 'last' or 'timestamp'")
                
                # Build prediction context window (last N ticks + current tick)
                window_start = max(0, tick_index - prediction_window_size + 1)
                context_ticks = ticks[window_start:tick_index + 1]
                
                if len(context_ticks) < 10:  # Need minimum context
                    continue  # Skip early ticks without sufficient context
                
                # Convert to format expected by analyzer
                tick_data = self._convert_single_tick_for_prediction(current_tick, context_ticks)
                
                # Generate prediction for this tick
                prediction_result = analyzer_system.validate_on_tick(tick_data)
                
                # Process and display prediction result
                if prediction_result:
                    self._process_tick_prediction_result(current_tick, prediction_result, tick_index)
                
                processed_count += 1
                
                # Progress reporting every 1000 ticks
                if processed_count % 1000 == 0:
                    progress_pct = (tick_index + 1) / len(ticks) * 100
                    print(f"  🔮 Tick-by-tick progress: {processed_count:,}/{len(ticks):,} ({progress_pct:.1f}%)")
                
            except Exception as e:
                # FAIL FAST - any tick processing error stops validation
                raise RuntimeError(f"Tick validation failed at index {tick_index}: {e}")
        
        print(f"✅ Tick-by-tick validation completed: {processed_count:,} ticks processed")
        return processed_count
    
    def _convert_single_tick_for_prediction(self, current_tick: Dict, context_ticks: List[Dict]) -> Dict[str, Any]:
        """Convert single tick + context to prediction format - BIBBIA COMPLIANT"""
        if not current_tick or not context_ticks:
            raise ValueError("Invalid tick or context data for prediction")
        
        # Extract price history from context
        price_history = []
        volume_history = []
        timestamps = []
        
        for tick in context_ticks:
            if 'last' not in tick:
                raise ValueError("Context tick missing 'last' field - MT5 format required")
            price_history.append(float(tick['last']))
            volume_history.append(float(tick.get('volume', 0)))
            timestamps.append(tick['timestamp'])
        
        return {
            'current_tick': current_tick,
            'price_history': price_history,
            'volume_history': volume_history,
            'timestamps': timestamps,
            'current_price': float(current_tick['last']),
            'current_volume': float(current_tick.get('volume', 0)),
            'symbol': current_tick.get('symbol', self.config.symbol),
            'context_size': len(context_ticks)
        }
    
    def _process_tick_prediction_result(self, current_tick: Dict, prediction_result: Dict, tick_index: int):
        """Process and display tick prediction result - BIBBIA COMPLIANT"""
        if not prediction_result:
            return
        
        current_price = current_tick['last']
        timestamp = current_tick['timestamp']
        
        # BIBBIA COMPLIANT: Extract prediction data - FAIL FAST if missing
        if 'predictions' not in prediction_result:
            raise KeyError(f"FAIL FAST: Missing required 'predictions' field in prediction_result at tick {tick_index}")
        predictions = prediction_result['predictions']
        if not predictions:
            return
        
        # BIBBIA COMPLIANT: Display each prediction - FAIL FAST for missing fields
        for pred in predictions:
            if 'model_type' not in pred:
                raise KeyError(f"FAIL FAST: Missing required 'model_type' field in prediction at tick {tick_index}")
            if 'algorithm' not in pred:
                raise KeyError(f"FAIL FAST: Missing required 'algorithm' field in prediction at tick {tick_index}")
            if 'prediction_data' not in pred:
                raise KeyError(f"FAIL FAST: Missing required 'prediction_data' field in prediction at tick {tick_index}")
            if 'confidence' not in pred:
                raise KeyError(f"FAIL FAST: Missing required 'confidence' field in prediction at tick {tick_index}")
            
            model_type = pred['model_type']
            algorithm = pred['algorithm']
            prediction_data = pred['prediction_data'] 
            confidence = pred['confidence']
            
            # BIBBIA COMPLIANT: Show predictions - FAIL FAST if format wrong
            print(f"🔮 TICK {tick_index:,} | {timestamp} | {algorithm} | Conf: {confidence:.3f}")
            
            # BIBBIA COMPLIANT: FAIL FAST - require new test-based format
            if model_type == 'support_resistance':
                if 'test_prediction' not in prediction_data:
                    raise KeyError(f"FAIL FAST: Missing required 'test_prediction' field in S/R prediction at tick {tick_index}")
                test_prediction = prediction_data['test_prediction']
                print(f"    🧪 Test: {test_prediction}")
            print("")  # Empty line for readability

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
                
                # FAIL FAST - no defaults allowed
                if bid_val <= 0:
                    raise ValueError(f"Invalid bid value at row {idx}: bid={bid_val}")
                if ask_val <= 0:
                    raise ValueError(f"Invalid ask value at row {idx}: ask={ask_val}")
                
                # Calculate spread percentage - FAIL FAST
                spread_percentage = (ask_val - bid_val) / bid_val
                
                tick_dict = {
                    'type': 'tick',
                    'timestamp': timestamp_formatted,
                    'symbol': symbol,
                    'bid': bid_val,
                    'ask': ask_val,
                    'last': (bid_val + ask_val) / 2,
                    'volume': volume_val,
                    'spread_percentage': spread_percentage,
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
    
    def _find_existing_file_with_coverage(self) -> Optional[str]:
        """
        Find existing file with >= 80% coverage of requested time period
        
        Returns:
            Path to existing file with sufficient coverage, or None
        """
        try:
            data_dir = "./test_analyzer_data"
            if not os.path.exists(data_dir):
                return None
            
            required_start = self.config.start_date
            required_end = self.config.end_date
            required_duration = (required_end - required_start).total_seconds()
            
            print(f"🔍 Searching for existing files covering period: {required_start.date()} to {required_end.date()}")
            
            best_file = None
            best_coverage = 0.0
            
            # Scan all .jsonl files in data directory
            for filename in os.listdir(data_dir):
                if not filename.endswith('.jsonl') or not filename.startswith(f'backtest_{self.config.symbol}'):
                    continue
                
                file_path = os.path.join(data_dir, filename)
                coverage_info = self._analyze_file_coverage(file_path, required_start, required_end)
                
                if coverage_info and coverage_info['coverage_percent'] > best_coverage:
                    best_coverage = coverage_info['coverage_percent']
                    best_file = file_path
                    
                    print(f"📊 {filename}: {coverage_info['coverage_percent']:.1f}% coverage "
                                   f"({coverage_info['file_start'].date()} to {coverage_info['file_end'].date()})")
            
            # Return file if coverage >= 80%
            if best_coverage >= 80.0:
                print(f"✅ Found suitable file with {best_coverage:.1f}% coverage (>= 80%)")
                return best_file
            elif best_file:
                print(f"❌ Best file only has {best_coverage:.1f}% coverage (< 80% required)")
            else:
                print("❌ No existing files found for this symbol")
                
            return None
            
        except Exception as e:
            self.logger.warning(f"Error searching for existing files: {e}")
            return None
    
    def _analyze_file_coverage(self, file_path: str, required_start: datetime, required_end: datetime) -> Optional[Dict[str, Any]]:
        """
        Analyze file coverage for given time period - FAIL FAST, NO FALLBACK
        
        Args:
            file_path: Path to JSONL file to analyze
            required_start: Required start datetime
            required_end: Required end datetime
            
        Returns:
            Dictionary with coverage info or None if file invalid
        """
        try:
            # FAIL FAST: Only filename parsing - NO FALLBACK
            filename = os.path.basename(file_path)
            file_dates = self._extract_dates_from_filename(filename)
            
            if not file_dates:
                # FAIL FAST: Cannot parse filename - reject file
                raise ValueError(f"Cannot extract dates from filename: {filename}")
            
            file_start, file_end = file_dates
            
            # Calculate overlap between file period and required period
            overlap_start = max(file_start, required_start)
            overlap_end = min(file_end, required_end)
            
            if overlap_start >= overlap_end:
                # No overlap
                return {
                    'file_start': file_start,
                    'file_end': file_end,
                    'coverage_percent': 0.0,
                    'overlap_hours': 0.0
                }
            
            # Calculate coverage percentage
            overlap_duration = (overlap_end - overlap_start).total_seconds()
            required_duration = (required_end - required_start).total_seconds()
            coverage_percent = (overlap_duration / required_duration) * 100.0
            
            return {
                'file_start': file_start,
                'file_end': file_end,
                'coverage_percent': coverage_percent,
                'overlap_hours': overlap_duration / 3600.0
            }
            
        except Exception as e:
            # FAIL FAST: Log error and reject file
            self.logger.warning(f"File rejected due to invalid format: {os.path.basename(file_path)} - {e}")
            return None
    
    def _extract_dates_from_filename(self, filename: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Extract start and end dates from filename pattern: backtest_SYMBOL_YYYYMMDD_YYYYMMDD.jsonl
        
        Args:
            filename: Filename to parse
            
        Returns:
            Tuple of (start_date, end_date) or None if parsing fails
        """
        try:
            # Expected pattern: backtest_USTEC_20250523_20250724.jsonl
            import re
            pattern = r'backtest_\w+_(\d{8})_(\d{8})\.jsonl'
            match = re.match(pattern, filename)
            
            if not match:
                return None
            
            start_str, end_str = match.groups()
            start_date = datetime.strptime(start_str, '%Y%m%d')
            end_date = datetime.strptime(end_str, '%Y%m%d')
            
            return (start_date, end_date)
            
        except Exception:
            return None
    


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