#!/usr/bin/env python3
"""
🔄 MT5 Backtest Runner - Versione Unified System
Sistema per accelerare l'apprendimento dell'Analyzer con dati storici
AGGIORNATO per usare UnifiedAnalyzerSystem invece di AdvancedMarketAnalyzer legacy
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
import asyncio
from dataclasses import dataclass

# Import del sistema Unified
from MT5BridgeReader import MT5TickData
from Unified_Analyzer_System import UnifiedAnalyzerSystem, UnifiedConfig, SystemMode, PerformanceProfile, create_custom_config  # type: ignore

@dataclass
class BacktestConfig:
    """Configurazione backtest"""
    symbol: str
    start_date: datetime
    end_date: datetime
    data_source: str  # 'mt5_export', 'csv_file', 'jsonl_file'
    speed_multiplier: int = 1000  # Velocità rispetto al tempo reale
    batch_size: int = 1000
    save_progress: bool = True
    resume_from_checkpoint: bool = True

class MT5DataExporter:
    """Esporta dati storici da MT5 - MEMORY LEAK FIXED + AGGRESSIVE OPTIMIZATION"""
    
    def __init__(self):
        self.logger = logging.getLogger('MT5DataExporter')
        
        # 🚀 Memory management settings - OPTIMIZED FOR 180 DAYS
        self.memory_threshold_critical = 95  # % - Emergency stop
        self.memory_threshold_warning = 85   # % - Reduce chunk size  
        self.base_chunk_days = 45           # Default chunk size
        self.min_chunk_days = 15            # Minimum chunk size
        self.ticks_per_gc = 1000            # Force GC every N ticks
        # Memory trend tracking
        self.memory_history = []
        self.max_memory_history = 5  # Track last 5 measurements
        
        # 🚀 Reusable objects pool to avoid allocations
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
            return 0.0
    
    def _force_garbage_collection(self):
        """Force aggressive garbage collection"""
        import gc
        gc.collect()
        gc.collect()  # Double collection for better cleanup
        gc.collect()  # Triple for maximum effect
        
    def _calculate_dynamic_chunk_size(self, total_days: int, current_memory: float) -> int:
        """Calculate chunk size based on available memory - AGGRESSIVE FOR 180 DAYS"""

        # Track memory trend
        self.memory_history.append(current_memory)
        if len(self.memory_history) > self.max_memory_history:
            self.memory_history.pop(0)

        # If memory is decreasing (GC working well), be more aggressive
        if len(self.memory_history) >= 3:
            memory_trend = self.memory_history[-1] - self.memory_history[-3]
            if memory_trend < -2:  # Memory dropping by 2%+
                self.logger.info(f"📉 Memory trending down ({memory_trend:.1f}%), will be more aggressive")
        
        if current_memory > self.memory_threshold_critical:
            # Emergency: smallest possible chunks
            chunk_days = max(7, self.min_chunk_days)
            self.logger.warning(f"🚨 EMERGENCY: Memory at {current_memory:.1f}%, using {chunk_days}-day chunks")
            
        elif current_memory > self.memory_threshold_warning:
            # Warning: reduced chunks
            chunk_days = max(self.min_chunk_days, self.base_chunk_days // 2)
            self.logger.warning(f"⚠️ HIGH MEMORY: {current_memory:.1f}%, reducing to {chunk_days}-day chunks")
            
        elif current_memory < 50:
            # Plenty of memory: LARGE chunks
            chunk_days = min(60, total_days)
            self.logger.info(f"🚀 LOW MEMORY USAGE: {current_memory:.1f}%, using LARGE {chunk_days}-day chunks")
            
        elif current_memory < 70:
            # Good memory: large chunks
            chunk_days = min(45, total_days)
            self.logger.info(f"📊 GOOD MEMORY: {current_memory:.1f}%, using {chunk_days}-day chunks")
            
        else:
            # Normal memory: standard chunks
            chunk_days = min(self.base_chunk_days, total_days)
            self.logger.info(f"📊 NORMAL MEMORY: {current_memory:.1f}%, using {chunk_days}-day chunks")
            
        return chunk_days
        
    def export_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                             output_file: str) -> bool:
        """Esporta dati storici da MT5 - MEMORY LEAK PROOF VERSION"""
        try:
            # Import MetaTrader5
            try:
                import MetaTrader5 as mt5
            except ImportError:
                self.logger.error("❌ MetaTrader5 package not installed. Install with: pip install MetaTrader5")
                return False
            
            # Initialize MT5 connection
            if not mt5.initialize():  # type: ignore
                self.logger.error("❌ Failed to initialize MT5 connection")
                return False
            
            # 🚀 MEMORY CHECK: Start with clean state
            self._force_garbage_collection()
            initial_memory = self._get_memory_usage()
            
            self.logger.info(f"📊 Exporting {symbol} from {start_date} to {end_date} (MEMORY OPTIMIZED)")
            self.logger.info(f"🧠 Starting memory usage: {initial_memory:.1f}%")
            
            # Calculate period
            total_days = (end_date - start_date).days
            self.logger.info(f"📅 Total period: {total_days} days")
            
            # 🚀 DYNAMIC CHUNK SIZE based on memory
            chunk_days = self._calculate_dynamic_chunk_size(total_days, initial_memory)
            
            # Initialize export variables
            total_ticks_exported = 0
            current_date = start_date
            chunk_number = 1
            first_chunk = True
            tick_counter = 0
            
            # 🚀 STREAMING: Open file once and keep it open for streaming
            try:
                with open(output_file, 'w', encoding='utf-8', buffering=131072) as f:  # 64KB buffer
                    
                    while current_date < end_date:
                        # 🚀 MEMORY CHECK: Monitor before each chunk
                        current_memory = self._get_memory_usage()
                        
                        if current_memory > self.memory_threshold_critical:
                            self.logger.error(f"🚨 CRITICAL MEMORY: {current_memory:.1f}% - EMERGENCY GC")
                            self._force_garbage_collection()
                            current_memory = self._get_memory_usage()
                            
                            if current_memory > self.memory_threshold_critical:
                                self.logger.error(f"🚨 CRITICAL: Still {current_memory:.1f}% after GC - ABORTING")
                                return False
                        
                        # Recalculate chunk size if memory is high
                        if current_memory > self.memory_threshold_warning:
                            chunk_days = max(1, chunk_days // 2)
                            self.logger.warning(f"⚠️ Reducing chunk size to {chunk_days} days due to memory pressure")
                        
                        # Calculate chunk end date
                        chunk_end = min(current_date + timedelta(days=chunk_days), end_date)
                        
                        self.logger.info(f"📊 Chunk {chunk_number}: {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
                        
                        # 🚀 MEMORY-SAFE CHUNK PROCESSING
                        chunk_ticks = self._process_chunk_memory_safe(
                            mt5, symbol, current_date, chunk_end, f, 
                            first_chunk, chunk_number
                        )
                        
                        if chunk_ticks is None:
                            # Error in chunk processing
                            current_date = chunk_end
                            chunk_number += 1
                            continue
                            
                        total_ticks_exported += chunk_ticks
                        tick_counter += chunk_ticks
                        first_chunk = False
                        
                        # 🚀 AGGRESSIVE MEMORY MANAGEMENT
                        if tick_counter >= self.ticks_per_gc:
                            self._force_garbage_collection()
                            tick_counter = 0
                            memory_after_gc = self._get_memory_usage()
                            self.logger.info(f"🧹 GC performed - Memory: {memory_after_gc:.1f}%")
                        
                        # Move to next chunk
                        current_date = chunk_end
                        chunk_number += 1
                        
                        # 🚀 MEMORY PRESSURE: Emergency break
                        if self._get_memory_usage() > 90:
                            self.logger.error("🚨 EMERGENCY: Memory > 90% - Stopping export")
                            break
                
                # 🚀 FINAL CLEANUP
                self.logger.info(f"📝 Updating header with final totals...")
                
            except IOError as e:
                self.logger.error(f"❌ File I/O error: {e}")
                return False
            
            # Update header with totals
            self._update_header_with_totals(output_file, symbol, total_ticks_exported)
            
            # Final cleanup
            self._force_garbage_collection()
            final_memory = self._get_memory_usage()
            
            mt5.shutdown()  # type: ignore
            
            self.logger.info(f"✅ Export completed: {total_ticks_exported:,} total ticks")
            self.logger.info(f"📁 File: {output_file}")
            self.logger.info(f"🧠 Final memory: {final_memory:.1f}% (started: {initial_memory:.1f}%)")
            
            return total_ticks_exported > 0
            
        except Exception as e:
            self.logger.error(f"❌ Export error: {e}")
            import traceback
            self.logger.error(f"❌ Stack trace: {traceback.format_exc()}")
            
            # Emergency cleanup
            self._force_garbage_collection()
            
            try:
                import MetaTrader5 as mt5  # type: ignore
                mt5.shutdown()  # type: ignore
            except:
                pass
            return False
    
    def _process_chunk_memory_safe(self, mt5, symbol: str, start_date: datetime, 
                                end_date: datetime, file_handle, is_first_chunk: bool, 
                                chunk_number: int) -> Optional[int]:
        """Process a single chunk with aggressive memory management + PROGRESS BAR"""
        
        try:
            # 🚀 MEMORY CHECK before processing
            pre_chunk_memory = self._get_memory_usage()
            
            # Get ticks for this chunk
            ticks = mt5.copy_ticks_range(symbol, start_date, end_date, mt5.COPY_TICKS_ALL)  # type: ignore
            
            if ticks is None or len(ticks) == 0:
                self.logger.warning(f"⚠️ No ticks in chunk {chunk_number}")
                return 0
            
            chunk_tick_count = len(ticks)
            self.logger.info(f"✅ Chunk {chunk_number}: {chunk_tick_count:,} ticks retrieved")
            
            # 🚀 WRITE HEADER only for first chunk
            if is_first_chunk:
                header = {
                    "type": "backtest_start",
                    "symbol": symbol,
                    "start_time": datetime.fromtimestamp(ticks[0]['time']).isoformat(),
                    "end_time": "TBD",
                    "total_ticks": "TBD", 
                    "export_time": datetime.now().isoformat(),
                    "chunked_export": True,
                    "memory_optimized": True
                }
                file_handle.write(json.dumps(header) + '\n')
                file_handle.flush()  # Immediate write
            
            # 📊 PROGRESS BAR CONFIGURATION
            if chunk_tick_count > 5_000_000:
                progress_interval = 100_000  # Ogni 100K ticks per chunk enormi (fluido)
            elif chunk_tick_count > 1_000_000:
                progress_interval = 100_000  # Ogni 100K ticks per chunk grandi
            else:
                progress_interval = 25_000   # Ogni 25K ticks per chunk medi
            
            # 🚀 STREAMING TICK PROCESSING - Process one tick at a time
            ticks_processed = 0
            
            # Progress bar iniziale
            print(f"\n📊 Processing Chunk {chunk_number} ({chunk_tick_count:,} ticks):")
            
            for i, tick in enumerate(ticks):
                
                # 🚀 REUSE TEMPLATE to avoid dict creation
                tick_data = self._tick_data_template.copy()
                
                # 🚀 SAFE NUMPY VOID ACCESS
                tick_time = tick['time']
                tick_bid = tick['bid'] 
                tick_ask = tick['ask']
                
                # Safe optional fields
                tick_last = tick['last'] if 'last' in tick.dtype.names else (tick_bid + tick_ask) / 2
                tick_volume = tick['volume'] if 'volume' in tick.dtype.names else 1
                
                # Calculate spread safely
                spread_percentage = 0.0
                if tick_bid > 0:
                    spread_percentage = (tick_ask - tick_bid) / tick_bid
                
                # 🚀 UPDATE TEMPLATE (no new allocations)
                tick_data["timestamp"] = datetime.fromtimestamp(tick_time).strftime('%Y.%m.%d %H:%M:%S')
                tick_data["symbol"] = symbol
                tick_data["bid"] = float(tick_bid)
                tick_data["ask"] = float(tick_ask) 
                tick_data["last"] = float(tick_last)
                tick_data["volume"] = int(tick_volume)
                tick_data["spread_percentage"] = float(spread_percentage)
                
                # 🚀 BATCH WRITE for speed
                if not hasattr(self, '_write_batch'):
                    self._write_batch = []

                self._write_batch.append(json.dumps(tick_data))

                # Write in batches of 100 for speed
                if len(self._write_batch) >= 500 or i == len(ticks) - 1:
                    file_handle.write('\n'.join(self._write_batch) + '\n')
                    file_handle.flush()
                    os.fsync(file_handle.fileno())
                    self._write_batch.clear()

                ticks_processed += 1
                
                # 📊 PROGRESS BAR FLUIDA - Update ogni progress_interval
                if i > 0 and (i % progress_interval == 0 or i == chunk_tick_count - 1):
                    progress_pct = (i / chunk_tick_count) * 100
                    
                    # Progress bar visuale
                    bar_length = 40
                    filled_length = int(bar_length * progress_pct / 100)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    
                    # Calcola velocità
                    if hasattr(self, '_chunk_start_time'):
                        elapsed = datetime.now() - self._chunk_start_time
                        speed = i / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                        eta_seconds = (chunk_tick_count - i) / speed if speed > 0 else 0
                        eta_str = f"ETA: {eta_seconds/60:.1f}min" if eta_seconds < 3600 else f"ETA: {eta_seconds/3600:.1f}h"
                    else:
                        self._chunk_start_time = datetime.now()
                        eta_str = "calculating..."
                        speed = 0
                    
                    # Progress bar con dettagli
                    print(f"\r📊 |{bar}| {progress_pct:5.1f}% " +
                        f"({i:,}/{chunk_tick_count:,}) | " +
                        f"⚡{speed:,.0f} t/s | " +
                        f"{eta_str}", end='', flush=True)
                
                # 🚀 OPTIMIZED GC: Less frequent for speed
                if (i + 1) % 2000 == 0:
                    file_handle.flush()  # Force write to disk
                    if (i + 1) % 20000 == 0:  # Much less frequent GC
                        self._force_garbage_collection()
                    
                    # Memory emergency check
                    current_memory = self._get_memory_usage()
                    if current_memory > 88:
                        self.logger.warning(f"⚠️ Memory spike: {current_memory:.1f}% during chunk processing")
                        self._force_garbage_collection()
                        
                        # Emergency break if still too high
                        if self._get_memory_usage() > 90:
                            self.logger.error(f"🚨 Emergency break at tick {i+1}/{chunk_tick_count}")
                            break
            
            # 📊 PROGRESS BAR FINALE - Complete
            print()  # New line dopo progress bar
            print(f"✅ Chunk {chunk_number} processing completed!")
            
            # Reset timer per prossimo chunk
            if hasattr(self, '_chunk_start_time'):
                delattr(self, '_chunk_start_time')
            
            # 🚀 IMMEDIATE CLEANUP
            del ticks  # Free the numpy array immediately
            self._force_garbage_collection()
            
            post_chunk_memory = self._get_memory_usage()
            memory_delta = post_chunk_memory - pre_chunk_memory
            
            self.logger.info(f"💾 Chunk {chunk_number} processed: {ticks_processed:,} ticks")
            self.logger.info(f"🧠 Memory: {pre_chunk_memory:.1f}% → {post_chunk_memory:.1f}% (Δ{memory_delta:+.1f}%)")
            
            return ticks_processed
            
        except Exception as e:
            # Clear progress bar su errore
            print()
            self.logger.error(f"❌ Error processing chunk {chunk_number}: {e}")
            
            # Emergency cleanup
            if 'ticks' in locals():
                del ticks
            self._force_garbage_collection()
            
            return None
    
    def _update_header_with_totals(self, output_file: str, symbol: str, total_ticks: int) -> None:
        """Update header with final totals - MEMORY SAFE VERSION"""
        try:
            # 🚀 STREAMING READ: Don't load entire file into memory
            temp_file = output_file + '.tmp'
            
            with open(output_file, 'r', encoding='utf-8') as input_file:
                with open(temp_file, 'w', encoding='utf-8') as output_file_handle:
                    
                    # Read and update header (first line only)
                    header_line = input_file.readline().strip()
                    if header_line:
                        header = json.loads(header_line)
                        
                        # Find last tick time by reading file backwards efficiently
                        last_tick_time = self._find_last_tick_time_efficient(output_file)
                        
                        # Update header
                        header['total_ticks'] = total_ticks
                        if last_tick_time:
                            header['end_time'] = last_tick_time
                        
                        # Write updated header
                        output_file_handle.write(json.dumps(header) + '\n')
                        output_file_handle.flush()
                        os.fsync(output_file_handle.fileno())
                    
                    # 🚀 STREAMING COPY: Copy rest of file line by line
                    for line in input_file:
                        output_file_handle.write(line)
                    
                    # Final flush after copying
                    output_file_handle.flush()
                    os.fsync(output_file_handle.fileno())
            
            # Replace original file
            os.replace(temp_file, output_file)
            
            self.logger.info(f"📝 Header updated: {total_ticks:,} total ticks")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not update header: {e}")
            # Clean up temp file if it exists
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
    
    def _find_last_tick_time_efficient(self, file_path: str) -> Optional[str]:
        """Find last tick time without loading entire file"""
        try:
            with open(file_path, 'rb') as f:
                # Seek to end and read backwards
                f.seek(0, 2)  # Go to end
                file_size = f.tell()
                
                # Read last 8KB to find last tick
                read_size = min(8192, file_size)
                f.seek(file_size - read_size)
                tail_data = f.read().decode('utf-8', errors='ignore')
                
                # Find last complete line with tick data
                lines = tail_data.split('\n')
                for line in reversed(lines):
                    if line.strip():
                        try:
                            tick_data = json.loads(line.strip())
                            if tick_data.get('type') == 'tick':
                                return datetime.strptime(
                                    tick_data['timestamp'], 
                                    '%Y.%m.%d %H:%M:%S'
                                ).isoformat()
                        except:
                            continue
            
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not find last tick time: {e}")
            return None
    
    def _log_export_progress(self, chunk_number: int, total_days: int, current_date: datetime, 
                            start_date: datetime, total_ticks_so_far: int) -> None:
        """Log detailed export progress with ETA"""
        
        days_processed = (current_date - start_date).days
        progress_pct = (days_processed / total_days) * 100
        
        # Estimate based on current rate
        if days_processed > 0 and total_ticks_so_far > 0:
            ticks_per_day = total_ticks_so_far / days_processed
            estimated_total_ticks = ticks_per_day * total_days
            eta_days = total_days - days_processed
            
            self.logger.info(f"📈 EXPORT PROGRESS: {progress_pct:.1f}% ({days_processed}/{total_days} days)")
            self.logger.info(f"   ➤ Ticks so far: {total_ticks_so_far:,} | Rate: {ticks_per_day:,.0f}/day")
            self.logger.info(f"   ➤ Estimated total: {estimated_total_ticks:,.0f} ticks")
            self.logger.info(f"   ➤ ETA: {eta_days} days remaining")

    def _save_tick_data(self, df: pd.DataFrame, symbol: str, output_file: str) -> None:
        """DEPRECATED - Legacy method maintained for compatibility"""
        self.logger.warning("⚠️ _save_tick_data called but memory-optimized streaming export is preferred")
        
        # If someone still calls this, at least make it memory-safe
        with open(output_file, 'w', encoding='utf-8', buffering=8192) as f:
            # Header
            header = {
                "type": "backtest_start", 
                "symbol": symbol,
                "start_time": df['time'].iloc[0].isoformat(),
                "end_time": df['time'].iloc[-1].isoformat(),
                "total_ticks": len(df),
                "export_time": datetime.now().isoformat(),
                "chunked_export": False,
                "legacy_method": True
            }
            f.write(json.dumps(header) + '\n')
            f.flush()
            os.fsync(f.fileno())
            
            # Process in small batches to avoid memory issues
            batch_size = 1000
            for start_idx in range(0, len(df), batch_size):
                end_idx = min(start_idx + batch_size, len(df))
                batch = df.iloc[start_idx:end_idx]
                
                for _, row in batch.iterrows():
                    tick_data = {
                        "type": "tick",
                        "timestamp": row['time'].strftime('%Y.%m.%d %H:%M:%S'),
                        "symbol": symbol,
                        "bid": float(row['bid']),
                        "ask": float(row['ask']),
                        "last": float(row.get('last', (float(row['bid']) + float(row['ask'])) / 2)),
                        "volume": int(row.get('volume', 1)),
                        "spread_percentage": float((float(row['ask']) - float(row['bid'])) / float(row['bid'])) if float(row['bid']) > 0 else 0.0,
                        "price_change_1m": 0.0,
                        "price_change_5m": 0.0, 
                        "volatility": 0.0,
                        "momentum_5m": 0.0,
                        "market_state": "backtest"
                    }
                    f.write(json.dumps(tick_data) + '\n')
                
                # Flush every batch to ensure data is written
                f.flush()
                os.fsync(f.fileno())
                
                # Cleanup batch
                del batch
                if start_idx % (batch_size * 10) == 0:
                    self._force_garbage_collection()
class BacktestDataProcessor:
    """Processa dati storici per il backtest"""
    
    def __init__(self):
        self.logger = logging.getLogger('BacktestProcessor')
        
    def load_data_from_csv(self, csv_file: str, symbol: str) -> List[MT5TickData]:
        """Carica dati da file CSV"""
        try:
            df = pd.read_csv(csv_file)
            
            # Mappatura colonne standard
            column_mapping = {
                'timestamp': ['timestamp', 'time', 'datetime', 'date'],
                'bid': ['bid', 'Bid', 'close', 'Close'],
                'ask': ['ask', 'Ask', 'high', 'High'],
                'volume': ['volume', 'Volume', 'vol', 'Vol']
            }
            
            # Trova colonne corrette
            cols: Dict[str, str] = {}
            for target, variants in column_mapping.items():
                for variant in variants:
                    if variant in df.columns:
                        cols[target] = variant
                        break
            
            if 'timestamp' not in cols or 'bid' not in cols:
                raise ValueError(f"Required columns not found in CSV. Available: {list(df.columns)}")
            
            # Converti in MT5TickData con conversioni esplicite
            ticks = []
            for _, row in df.iterrows():
                # Conversioni sicure per evitare errori Pylance
                timestamp_val = row[cols['timestamp']]
                bid_val = float(row[cols['bid']])
                ask_val = float(row[cols.get('ask', cols['bid'])])
                volume_val = int(row[cols['volume']]) if cols.get('volume') and pd.notna(row[cols['volume']]) else 1
                
                tick = MT5TickData(
                    timestamp=pd.to_datetime(timestamp_val).to_pydatetime(),
                    symbol=symbol,
                    bid=bid_val,
                    ask=ask_val,
                    last=ask_val,  # Use ask as last if no separate last price
                    volume=volume_val,
                    spread_percentage=0.0,
                    price_change_1m=0.0,
                    price_change_5m=0.0,
                    volatility=0.0,
                    momentum_5m=0.0,
                    market_state="backtest"
                )
                ticks.append(tick)
            
            self.logger.info(f"✅ Loaded {len(ticks)} ticks from CSV")
            return ticks
            
        except Exception as e:
            self.logger.error(f"❌ Error loading CSV: {e}")
            return []
    
    def load_data_from_jsonl(self, jsonl_file: str) -> List[MT5TickData]:
        """Carica dati da file JSONL - STREAMING VERSION per evitare memory overload"""
        
        try:
            # 🚀 MEMORY-SAFE: Count total lines first
            total_lines = 0
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and ('type":"tick"' in line or '"type": "tick"' in line):
                        total_lines += 1
            
            self.logger.info(f"📊 Found {total_lines:,} tick lines to process")
            
            # 🚀 STREAMING LOAD: Process in memory-safe batches
            batch_size = 100000  # 100K ticks per batch
            ticks = []
            processed_lines = 0
            
            self.logger.info(f"🔄 Loading in batches of {batch_size:,} ticks...")
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                current_batch = []
                
                for line_num, line in enumerate(f, 1):
                    try:
                        line_stripped = line.strip()
                        if not line_stripped:
                            continue
                            
                        data = json.loads(line_stripped)
                        
                        if data.get('type') == 'tick':
                            tick = MT5TickData(
                                timestamp=datetime.strptime(data['timestamp'], '%Y.%m.%d %H:%M:%S'),
                                symbol=str(data['symbol']),
                                bid=float(data['bid']),
                                ask=float(data['ask']),
                                last=float(data.get('last', 0)),
                                volume=int(data.get('volume', 1)),
                                spread_percentage=float(data['spread_percentage']),
                                price_change_1m=float(data['price_change_1m']),
                                price_change_5m=float(data['price_change_5m']),
                                volatility=float(data['volatility']),
                                momentum_5m=float(data['momentum_5m']),
                                market_state=str(data['market_state'])
                            )
                            current_batch.append(tick)
                            processed_lines += 1
                            
                            # 🚀 BATCH PROCESSING: Process and clear every batch_size
                            if len(current_batch) >= batch_size:
                                ticks.extend(current_batch)
                                current_batch.clear()
                                
                                # Memory monitoring + progress
                                progress = (processed_lines / max(1, total_lines)) * 100  # Evita division by zero
                                memory_usage = self._get_memory_usage() if hasattr(self, '_get_memory_usage') else 0
                                
                                self.logger.info(f"📊 Loaded batch: {progress:.1f}% ({processed_lines:,}/{total_lines:,}) | Memory: {memory_usage:.1f}%")
                                
                                # 🚨 MEMORY PRESSURE CHECK
                                if memory_usage > 85:
                                    self.logger.warning(f"⚠️ High memory during loading: {memory_usage:.1f}%")
                                    # Force garbage collection
                                    import gc
                                    gc.collect()
                                    
                                    # Check again after GC
                                    memory_after_gc = self._get_memory_usage() if hasattr(self, '_get_memory_usage') else 0
                                    self.logger.info(f"🧹 After GC: {memory_after_gc:.1f}%")
                                    
                                    if memory_after_gc > 90:
                                        self.logger.error(f"🚨 CRITICAL MEMORY: {memory_after_gc:.1f}% - ABORTING LOAD FOR SAFETY")
                                        self.logger.error(f"📊 Emergency stop: {len(ticks):,} ticks loaded before abort (partial dataset)")
                                        
                                        # Final cleanup before abort
                                        if current_batch:
                                            ticks.extend(current_batch)
                                            current_batch.clear()
                                        
                                        import gc
                                        gc.collect()
                                        
                                        abort_memory = self._get_memory_usage() if hasattr(self, '_get_memory_usage') else 0
                                        self.logger.error(f"💾 Final memory after abort: {abort_memory:.1f}%")
                                        
                                        return ticks  # Return partial dataset for safety
                                
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        if line_num <= 10:  # Log solo primi errori
                            self.logger.warning(f"⚠️ Error parsing line {line_num}: {e}")
                
                # 🚀 FINAL BATCH: Add remaining ticks
                if current_batch:
                    ticks.extend(current_batch)
                    current_batch.clear()
            
            # Final memory cleanup
            import gc
            gc.collect()
            
            final_memory = self._get_memory_usage() if hasattr(self, '_get_memory_usage') else 0
            self.logger.info(f"✅ Loaded {len(ticks):,} ticks from JSONL | Final memory: {final_memory:.1f}%")
            
            return ticks
            
        except Exception as e:
            self.logger.error(f"❌ Error loading JSONL: {e}")
            return []

    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0

class MT5BacktestRunner:
    """Runner principale per backtest - UNIFIED SYSTEM VERSION"""
    
    def __init__(self, analyzer_data_path: str = "./analyzer_data"):
        self.analyzer_data_path = analyzer_data_path
        
        # ✅ NUOVO: Usa UnifiedAnalyzerSystem invece di AdvancedMarketAnalyzer
        self.unified_system: Optional[UnifiedAnalyzerSystem] = None
        self.is_system_running = False
        
        # Utility components
        self.data_processor = BacktestDataProcessor()
        self.exporter = MT5DataExporter()
        
        # Logging setup
        self.logger = logging.getLogger('BacktestRunner')
        self.setup_logging()
        
    def setup_logging(self) -> None:
        """Setup logging per backtest"""
        os.makedirs(self.analyzer_data_path, exist_ok=True)
        
        log_format = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'{self.analyzer_data_path}/backtest.log', encoding='utf-8')
            ]
        )
    
    async def setup_unified_system(self, config: BacktestConfig) -> bool:
        """Setup del sistema unificato per backtest"""
        
        try:
            self.logger.info("🔧 Setting up Unified Analyzer System for backtest...")
            
            # Configurazione ottimizzata per backtest learning
            unified_config = create_custom_config(
                system_mode=SystemMode.TESTING,
                performance_profile=PerformanceProfile.RESEARCH,
                asset_symbol=config.symbol,
                
                # Logging settings per backtest
                log_level="INFO",
                enable_console_output=True,
                enable_file_output=True,
                enable_csv_export=True,
                enable_json_export=False,
                
                # Rate limiting per backtest learning
                rate_limits={
                    'tick_processing': 1000,     # Log ogni 1000 ticks
                    'predictions': 100,          # Log ogni 100 predizioni
                    'validations': 50,           # Log ogni 50 validazioni
                    'training_events': 1,        # Log tutti i training
                    'champion_changes': 1,       # Log tutti i champion changes
                    'emergency_events': 1,       # Log tutte le emergenze
                    'diagnostics': 2000         # Log diagnostici ogni 2000 ops
                },
                
                # Performance settings per backtest
                event_processing_interval=10.0,    # Process eventi ogni 10 secondi
                batch_size=100,                    # Batch grandi per efficienza
                max_queue_size=20000,              # Queue grande per backtest
                
                # Storage per backtest
                base_directory=f"{self.analyzer_data_path}/unified_backtest_{config.symbol}_{datetime.now():%Y%m%d_%H%M%S}",
                
                # Monitoring
                enable_performance_monitoring=True,
                performance_report_interval=60.0,  # Report ogni minuto
                memory_threshold_mb=2000,          # 2GB threshold per backtest
                cpu_threshold_percent=85.0         # 85% CPU threshold
            )
            
            # Crea e avvia sistema unificato
            self.unified_system = UnifiedAnalyzerSystem(unified_config)
            await self.unified_system.start()
            
            self.is_system_running = True
            
            self.logger.info(f"✅ Unified System started for {config.symbol}")
            self.logger.info(f"📁 Logs directory: {unified_config.base_directory}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up unified system: {e}")
            return False
    
    async def cleanup_unified_system(self):
        """Cleanup del sistema unificato"""
        
        if self.unified_system and self.is_system_running:
            try:
                self.logger.info("🧹 Stopping Unified System...")
                await self.unified_system.stop()
                self.is_system_running = False
                self.logger.info("✅ Unified System stopped successfully")
            except Exception as e:
                self.logger.error(f"⚠️ Error stopping unified system: {e}")
    
    def run_backtest(self, config: BacktestConfig) -> bool:
        """Esegue backtest completo - VERSIONE THREAD-SAFE"""
        
        import asyncio
        import threading
        import queue
        
        self.logger.info("🔄 Starting backtest with thread-safe execution")
        
        # Usa sempre un thread separato per evitare conflitti event loop
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def run_async_in_thread():
            try:
                # Crea nuovo event loop nel thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                
                try:
                    # Esegui il backtest asincrono
                    result = new_loop.run_until_complete(self._run_backtest_async(config))
                    result_queue.put(result)
                finally:
                    new_loop.close()
                    
            except Exception as e:
                exception_queue.put(e)
        
        # Esegui in thread separato
        thread = threading.Thread(target=run_async_in_thread, daemon=False)
        thread.start()
        thread.join()
        
        # Controlla errori
        if not exception_queue.empty():
            error = exception_queue.get()
            self.logger.error(f"❌ Error in backtest execution: {error}")
            import traceback
            traceback.print_exc()
            return False
        
        # Ottieni risultato
        if not result_queue.empty():
            result = result_queue.get()
            self.logger.info(f"✅ Backtest completed with result: {result}")
            return result
        else:
            self.logger.error("❌ No result from backtest execution")
            return False
    
    async def _run_backtest_async(self, config: BacktestConfig) -> bool:
        """Esegue backtest completo - VERSIONE ASINCRONA CON DEBUG ESTENSIVO"""
        
        # AGGIUNGI IMPORT ALL'INIZIO
        import traceback
        import os
        
        self.logger.info("="*60)
        self.logger.info("🔄 STARTING MT5 BACKTEST - UNIFIED SYSTEM")
        self.logger.info("="*60)
        self.logger.info(f"Symbol: {config.symbol}")
        self.logger.info(f"Period: {config.start_date} to {config.end_date}")
        self.logger.info(f"Speed: {config.speed_multiplier}x")
        self.logger.info(f"Data source: {config.data_source}")
        
        # DEBUG: Memory iniziale
        initial_memory = self._get_memory_usage()
        self.logger.info(f"🧠 Initial memory usage: {initial_memory:.1f}MB")
        
        try:
            # DEBUG: PHASE 1
            self.logger.info("🔧 PHASE 1: Starting unified system setup...")
            
            # 1. Setup sistema unificato
            if not await self.setup_unified_system(config):
                self.logger.error("❌ PHASE 1 FAILED: Unified system setup failed")
                return False
            
            setup_memory = self._get_memory_usage()
            self.logger.info(f"✅ PHASE 1 COMPLETED: System setup OK, memory: {setup_memory:.1f}MB")
            
            # DEBUG: PHASE 2
            self.logger.info("🔧 PHASE 2: Data export/load starting...")
            
            # 2. Carica/esporta dati
            data_file = f"{self.analyzer_data_path}/backtest_{config.symbol}_{config.start_date.strftime('%Y%m%d')}_{config.end_date.strftime('%Y%m%d')}.jsonl"
            
            if config.data_source == 'mt5_export':
                self.logger.info(f"📊 Starting data export to: {data_file}")
                if not self._export_mt5_data(config, data_file):
                    self.logger.error("❌ PHASE 2 FAILED: Data export failed")
                    return False
                
                export_memory = self._get_memory_usage()
                self.logger.info(f"✅ Data export completed, memory: {export_memory:.1f}MB")
            
            # 3. Carica dati processati
            self.logger.info(f"📂 Loading data from: {data_file}")
            
            # Check file size before loading
            if os.path.exists(data_file):
                file_size_gb = os.path.getsize(data_file) / (1024**3)
                self.logger.info(f"📊 Data file size: {file_size_gb:.2f} GB")
            
            ticks = self._load_backtest_data(config, data_file)
            if not ticks:
                self.logger.error("❌ PHASE 2 FAILED: No data loaded for backtest")
                return False
            
            load_memory = self._get_memory_usage()
            self.logger.info(f"✅ PHASE 2 COMPLETED: {len(ticks):,} ticks loaded, memory: {load_memory:.1f}MB")
            
            # DEBUG: PHASE 3
            self.logger.info("🔧 PHASE 3: ML processing starting...")
            self.logger.info(f"🧠 Ready to process {len(ticks):,} ticks through ML system")
            
            # Memory check before ML processing
            pre_ml_memory = self._get_memory_usage()
            self.logger.info(f"🧠 Pre-ML memory: {pre_ml_memory:.1f}MB")
            
            # Add heartbeat before starting intensive processing
            self.logger.info("💓 HEARTBEAT: Starting ML backtest processing...")
            
            # 4. Esegui backtest con sistema unificato
            success = await self._execute_backtest_unified(config, ticks)
            
            if success:
                final_memory = self._get_memory_usage()
                self.logger.info(f"✅ PHASE 3 COMPLETED: ML processing successful, final memory: {final_memory:.1f}MB")
            else:
                self.logger.error("❌ PHASE 3 FAILED: ML processing failed")
            
            return success
            
        except KeyboardInterrupt:
            self.logger.info("🛑 MANUAL INTERRUPTION: User cancelled backtest")
            return False
            
        except MemoryError as e:
            self.logger.error(f"🧠 MEMORY ERROR: {e}")
            current_memory = self._get_memory_usage()
            self.logger.error(f"🧠 Memory at crash: {current_memory:.1f}MB")
            return False
            
        except ImportError as e:
            self.logger.error(f"📦 IMPORT ERROR: {e}")
            self.logger.error("📦 Missing dependency or module loading issue")
            return False
            
        except Exception as e:
            crash_memory = self._get_memory_usage()
            self.logger.error(f"❌ CRASH DETECTED: {type(e).__name__}: {e}")
            self.logger.error(f"💥 Crash memory: {crash_memory:.1f}MB")
            self.logger.error(f"💥 Crash location: {traceback.format_exc()}")
            
            # Force flush all logs immediately
            try:
                import logging
                logging.shutdown()
            except:
                pass
                
            return False
            
        finally:
            # DEBUG: Cleanup phase
            self.logger.info("🧹 CLEANUP PHASE: Starting system shutdown...")
            
            try:
                cleanup_memory = self._get_memory_usage()
                self.logger.info(f"🧹 Pre-cleanup memory: {cleanup_memory:.1f}MB")
                
                # Cleanup sempre
                await self.cleanup_unified_system()
                
                final_cleanup_memory = self._get_memory_usage()
                self.logger.info(f"✅ CLEANUP COMPLETED: Final memory: {final_cleanup_memory:.1f}MB")
                
            except Exception as cleanup_error:
                self.logger.error(f"⚠️ CLEANUP ERROR: {cleanup_error}")
    
    def _export_mt5_data(self, config: BacktestConfig, output_file: str) -> bool:
        """Esporta dati da MT5 se necessario"""
        if os.path.exists(output_file):
            self.logger.info(f"✅ Using existing data file: {output_file}")
            return True
        
        self.logger.info("📊 Exporting data from MT5...")
        return self.exporter.export_historical_data(
            config.symbol, config.start_date, config.end_date, output_file
        )
    
    def _load_backtest_data(self, config: BacktestConfig, data_file: str) -> List[MT5TickData]:
        """Carica dati per backtest"""
        if config.data_source == 'csv_file':
            return self.data_processor.load_data_from_csv(data_file, config.symbol)
        elif config.data_source in ['mt5_export', 'jsonl_file']:
            return self.data_processor.load_data_from_jsonl(data_file)
        else:
            self.logger.error(f"❌ Unknown data source: {config.data_source}")
            return []
    
    async def _execute_backtest_unified(self, config: BacktestConfig, ticks: List[MT5TickData]) -> bool:
        """Esegue il backtest con sistema unificato - ENHANCED PROGRESS VERSION"""
        
        if not self.unified_system:
            self.logger.error("❌ Unified system not available")
            return False
        
        total_ticks = len(ticks)
        self.logger.info(f"🚀 Starting unified backtest with {total_ticks:,} ticks")
        
        # ENHANCED: Progress configuration
        if total_ticks > 5_000_000:
            # Per volumi molto grandi (>5M): report ogni 250k ticks  
            progress_interval = 250_000
            console_interval = 100_000  # Console più frequente
        elif total_ticks > 1_000_000:
            # Per volumi grandi (>1M): report ogni 100k ticks
            progress_interval = 100_000
            console_interval = 25_000
        elif total_ticks > 100_000:
            # Per volumi medi: report ogni 10k ticks
            progress_interval = 10_000
            console_interval = 5_000
        else:
            # Per volumi piccoli: report ogni 1k ticks
            progress_interval = 1_000
            console_interval = 500
        
        start_time = time.time()
        processed_ticks = 0
        error_count = 0
        last_progress_time = start_time
        
        # ENHANCED: Progress banner iniziale
        print("\n" + "="*90)
        print(f"🚀 UNIFIED BACKTEST STARTING")
        print(f"📊 Symbol: {config.symbol} | Total Ticks: {total_ticks:,} | Speed: {config.speed_multiplier}x")
        print("="*90)
        
        try:
            for i, tick in enumerate(ticks):
                try:
                    # Processa tick con UnifiedAnalyzerSystem
                    result = await self.unified_system.process_tick(
                        timestamp=tick.timestamp,
                        price=(tick.bid + tick.ask) / 2.0,
                        volume=max(1, tick.volume),
                        bid=tick.bid,
                        ask=tick.ask
                    )
                    
                    processed_ticks += 1
                    
                    # Validation checkpoint
                    if i % 1000 == 0 and i > 0:
                        # Verifica che l'analyzer abbia ricevuto i dati
                        analyzer_tick_count = len(self.unified_system.analyzer.tick_data)
                        if analyzer_tick_count != processed_ticks:
                            self.logger.warning(f"⚠️ DATA SYNC ISSUE: Processed {processed_ticks} ticks but analyzer has {analyzer_tick_count}")
                        
                        # Verifica range prezzi nell'analyzer
                        if analyzer_tick_count > 0:
                            recent_prices = [t['price'] for t in self.unified_system.analyzer.tick_data[-10:]]
                            price_range = max(recent_prices) - min(recent_prices)
                            if price_range == 0:
                                self.logger.error(f"❌ PRICE RANGE ZERO: Last 10 prices all identical: {recent_prices[0]}")
                            elif i <= 10000:  # Log only for first 10k ticks
                                self.logger.info(f"✅ Price validation OK: range={price_range:.2f}, sample={recent_prices[:3]}")

                    # Ogni 10K ticks, log memory
                    if i % 10000 == 0:
                        memory_mb = self._get_memory_usage()
                        self.logger.info(f"📊 Memory check: {memory_mb:.1f}MB at tick {i}")
                    
                    # Log heartbeat ogni minuto
                    if i % 60000 == 0:  # Ogni ~60K ticks
                        self.logger.info(f"💓 HEARTBEAT: System alive, processing tick {i}")
                    
                    # Rate limiting per non sovraccaricare
                    if processed_ticks % 100 == 0:
                        await asyncio.sleep(0.001)  # 1ms pause ogni 100 ticks
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 10:  # Log solo primi errori
                        self.logger.warning(f"⚠️ Error processing tick {i}: {e}")
                    
                    if error_count > 100:  # Troppi errori
                        self.logger.error("❌ Too many errors, stopping backtest")
                        return False
                
                # ENHANCED: Console progress (più frequente)
                if i > 0 and (i % console_interval == 0 or i == total_ticks - 1):
                    current_time = time.time()
                    elapsed = current_time - start_time
                    progress = (i / total_ticks) * 100
                    speed = processed_ticks / elapsed if elapsed > 0 else 0
                    eta_seconds = (total_ticks - i) / speed if speed > 0 else 0
                    
                    # Calcola velocità istantanea
                    time_since_last = current_time - last_progress_time
                    ticks_since_last = console_interval if i % console_interval == 0 else (i % console_interval)
                    instant_speed = ticks_since_last / time_since_last if time_since_last > 0 else 0
                    
                    # Progress bar visuale
                    bar_length = 40
                    filled_length = int(bar_length * progress / 100)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    
                    # ENHANCED: Output console dettagliato con colori
                    print(f"\r🔄 {progress:5.1f}% |{bar}| " +
                        f"{processed_ticks:,}/{total_ticks:,} | " +
                        f"⚡{speed:,.0f} t/s | " +
                        f"📊{instant_speed:,.0f} inst | " +
                        f"❌{error_count} err | " +
                        f"⏱️{eta_seconds/60:.1f}min", end='', flush=True)
                    
                    last_progress_time = current_time
                
                # ENHANCED: Detailed log progress (meno frequente ma più dettagliato)
                if i > 0 and (i % progress_interval == 0 or i == total_ticks - 1):
                    current_time = time.time()
                    elapsed = current_time - start_time
                    progress = (i / total_ticks) * 100
                    speed = processed_ticks / elapsed if elapsed > 0 else 0
                    eta_seconds = (total_ticks - i) / speed if speed > 0 else 0
                    
                    # Memory usage (se disponibile)
                    memory_mb = 0
                    try:
                        import psutil
                        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                    except:
                        pass
                    
                    # Log dettagliato
                    self.logger.info(f"📊 PROGRESS MILESTONE")
                    self.logger.info(f"   ➤ Completion: {progress:.1f}% ({processed_ticks:,}/{total_ticks:,} ticks)")
                    self.logger.info(f"   ➤ Speed: {speed:,.0f} ticks/sec (avg) | {instant_speed:,.0f} ticks/sec (inst)")
                    self.logger.info(f"   ➤ Time: {elapsed:.1f}s elapsed | {eta_seconds/60:.1f}min remaining")
                    self.logger.info(f"   ➤ Errors: {error_count} ({(error_count/(processed_ticks+error_count)*100):.2f}%)")
                    if memory_mb > 0:
                        self.logger.info(f"   ➤ Memory: {memory_mb:.1f} MB")
                    
                    # ENHANCED: Performance warnings
                    if speed < 10000 and total_ticks > 1000000:
                        self.logger.warning(f"⚠️ Low processing speed detected: {speed:.0f} ticks/sec")
                    
                    if error_count > processed_ticks * 0.05:  # >5% error rate
                        self.logger.warning(f"⚠️ High error rate: {(error_count/(processed_ticks+error_count)*100):.1f}%")
                    
                # ENHANCED: Checkpoint saves con notifica
                if hasattr(config, 'save_progress') and config.save_progress:
                    checkpoint_interval = max(100_000, total_ticks // 20)  # Ogni 5%
                    if i > 0 and i % checkpoint_interval == 0:
                        try:
                            # Salva stato del sistema unificato
                            if hasattr(self.unified_system, 'get_system_status'):
                                status = self.unified_system.get_system_status()
                                checkpoint_data = {
                                    'progress': progress,
                                    'processed_ticks': processed_ticks,
                                    'timestamp': datetime.now().isoformat(),
                                    'system_status': status
                                }
                                
                                # Salva checkpoint
                                checkpoint_file = f"{self.analyzer_data_path}/checkpoint_{config.symbol}_{i}.json"
                                with open(checkpoint_file, 'w') as f:
                                    json.dump(checkpoint_data, f, indent=2, default=str)
                                
                                print(f"\n💾 Checkpoint saved: {progress:.1f}% -> {checkpoint_file}")
                                self.logger.info(f"💾 Checkpoint saved at {progress:.1f}%: {checkpoint_file}")
                        except Exception as e:
                            self.logger.warning(f"⚠️ Could not save checkpoint: {e}")
                
                # Speed control (opzionale)
                if config.speed_multiplier < 1000:
                    await asyncio.sleep(0.001 / config.speed_multiplier)
            
            # ENHANCED: Final newline dopo la progress bar
            print("\n")
            
            # ENHANCED: Final stats dettagliate
            elapsed = time.time() - start_time
            speed = processed_ticks / elapsed if elapsed > 0 else 0
            success_rate = (processed_ticks/(processed_ticks+error_count))*100 if (processed_ticks+error_count) > 0 else 0
            
            print("=" * 90)
            print("✅ UNIFIED BACKTEST COMPLETED SUCCESSFULLY")
            print("=" * 90)
            print(f"📊 Results Summary:")
            print(f"   ➤ Total processed: {processed_ticks:,} ticks")
            print(f"   ➤ Processing time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
            print(f"   ➤ Average speed: {speed:,.0f} ticks/sec")
            print(f"   ➤ Speed achievement: {speed:.0f}x real-time")
            print(f"   ➤ Error count: {error_count} ({100-success_rate:.2f}%)")
            print(f"   ➤ Success rate: {success_rate:.1f}%")
            print("=" * 90)
            
            # Log finale dettagliato
            self.logger.info("="*60)
            self.logger.info("✅ UNIFIED BACKTEST COMPLETED")
            self.logger.info("="*60)
            self.logger.info(f"📊 FINAL STATISTICS:")
            self.logger.info(f"   Total ticks processed: {processed_ticks:,}")
            self.logger.info(f"   Processing time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
            self.logger.info(f"   Average speed: {speed:,.0f} ticks/sec")
            self.logger.info(f"   Speed multiplier achieved: {speed:.0f}x real-time")
            self.logger.info(f"   Error count: {error_count}")
            self.logger.info(f"   Success rate: {success_rate:.1f}%")
            
            # ENHANCED: System status finale
            if hasattr(self.unified_system, 'get_system_status'):
                try:
                    status = self.unified_system.get_system_status()
                    self.logger.info("📊 FINAL SYSTEM STATUS:")
                    self.logger.info(f"   ➤ System running: {status.get('system', {}).get('running', 'unknown')}")
                    self.logger.info(f"   ➤ Total events logged: {status.get('system', {}).get('stats', {}).get('total_events_logged', 0):,}")
                    self.logger.info(f"   ➤ Uptime: {status.get('system', {}).get('uptime_seconds', 0):.1f}s")
                    
                    if 'analyzer' in status:
                        analyzer_stats = status['analyzer']
                        self.logger.info(f"   ➤ Predictions generated: {analyzer_stats.get('predictions_generated', 0):,}")
                        self.logger.info(f"   ➤ Buffer utilization: {analyzer_stats.get('buffer_utilization', 0):.1f}%")
                        self.logger.info(f"   ➤ Events pending: {analyzer_stats.get('events_pending', 0)}")
                    
                    if 'performance' in status:
                        perf_stats = status['performance']
                        self.logger.info(f"   ➤ Memory usage: {perf_stats.get('memory_mb', 0):.1f}MB")
                        self.logger.info(f"   ➤ CPU usage: {perf_stats.get('cpu_percent', 0):.1f}%")
                    
                    # Salva status finale
                    final_status_file = f"{self.analyzer_data_path}/final_status_{config.symbol}_{datetime.now():%Y%m%d_%H%M%S}.json"
                    with open(final_status_file, 'w') as f:
                        json.dump(status, f, indent=2, default=str)
                    self.logger.info(f"   ➤ Final status saved: {final_status_file}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not get final system status: {e}")
            
            return True
            
        except KeyboardInterrupt:
            print("\n🛑 Backtest interrupted by user")
            self.logger.info("\n🛑 Backtest interrupted by user")
            
            # Salva stato parziale
            try:
                partial_status = {
                    'interrupted_at': datetime.now().isoformat(),
                    'progress': (processed_ticks / total_ticks) * 100,
                    'processed_ticks': processed_ticks,
                    'error_count': error_count,
                    'elapsed_time': time.time() - start_time
                }
                
                partial_file = f"{self.analyzer_data_path}/interrupted_status_{config.symbol}_{datetime.now():%Y%m%d_%H%M%S}.json"
                with open(partial_file, 'w') as f:
                    json.dump(partial_status, f, indent=2, default=str)
                
                print(f"💾 Partial status saved: {partial_file}")
                self.logger.info(f"💾 Partial status saved: {partial_file}")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not save partial status: {e}")
            
            return False
            
        except Exception as e:
            print(f"\n❌ Backtest error: {e}")
            self.logger.error(f"❌ Backtest execution error: {e}")
            import traceback
            traceback.print_exc()
            return False


    # ================================
    # UTILITY: METODO HELPER PER MEMORY MONITORING  
    # ================================

    def _get_memory_usage(self) -> float:
        """Ottieni uso memoria corrente in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except Exception:
            return 0.0


    # ================================
    # ENHANCED: PROGRESS ESTIMATION UTILITY
    # ================================

    def _estimate_completion_time(self, processed: int, total: int, elapsed_time: float) -> str:
        """Stima tempo di completamento con formattazione user-friendly"""
        if processed == 0:
            return "calculating..."
        
        rate = processed / elapsed_time
        remaining = total - processed
        eta_seconds = remaining / rate
        
        if eta_seconds < 60:
            return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            return f"{eta_seconds/60:.1f}min"
        else:
            hours = eta_seconds / 3600
            return f"{hours:.1f}h"

def create_backtest_config(symbol: str, months_back: int = 6) -> BacktestConfig:
    """Crea configurazione con date STABILI per caching intelligente"""
    
    # 🎯 DATE STABILI: Ancora a fine mese per caching
    today = datetime.now().date()
    
    # Ancora alla fine del mese appropriato
    if today.day <= 15:
        # Prima metà mese → usa fine mese precedente
        if today.month == 1:
            end_date = datetime(today.year - 1, 12, 31, 23, 59, 59)
        else:
            # Ultimo giorno del mese precedente
            first_day_current = datetime(today.year, today.month, 1)
            end_date = (first_day_current - timedelta(days=1)).replace(hour=23, minute=59, second=59)
    else:
        # Seconda metà → usa fine mese corrente
        if today.month == 12:
            end_date = datetime(today.year, 12, 31, 23, 59, 59)
        else:
            # Ultimo giorno del mese corrente
            first_day_next = datetime(today.year, today.month + 1, 1)
            end_date = (first_day_next - timedelta(days=1)).replace(hour=23, minute=59, second=59)
    
    start_date = end_date - timedelta(days=months_back * 30)
    
    return BacktestConfig(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        data_source='mt5_export',
        speed_multiplier=1000,  # Massima velocità
        batch_size=1000,
        save_progress=True,
        resume_from_checkpoint=True
    )

async def main_async():
    """Test backtest runner asincrono"""
    
    # Configurazione backtest
    config = create_backtest_config('USTEC', months_back=1)  # 1 mese per test
    
    print("🔄 MT5 Backtest Runner - Unified System")
    print(f"📊 Symbol: {config.symbol}")
    print(f"📅 Period: {config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')}")
    print(f"⚡ Speed: {config.speed_multiplier}x")
    print("🚀 Using: UnifiedAnalyzerSystem")
    
    confirm = input("\n🚀 Start unified backtest? (y/N): ")
    if confirm.lower() != 'y':
        print("Backtest cancelled.")
        return
    
    # Esegui backtest
    runner = MT5BacktestRunner()
    success = await runner._run_backtest_async(config)
    
    if success:
        print("\n✅ Unified backtest completed successfully!")
        print("🧠 UnifiedAnalyzerSystem trained on historical data")
        print("🚀 Ready for real-time analysis with improved performance")
    else:
        print("\n❌ Unified backtest failed")

def main():
    """Test backtest runner"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()