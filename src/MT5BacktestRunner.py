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
    """Esporta dati storici da MT5 - MEMORY SAFE + NUMPY VOID FIXED"""
    
    def __init__(self):
        self.logger = logging.getLogger('MT5DataExporter')
        
    def export_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                             output_file: str) -> bool:
        """Esporta dati storici da MT5 in chunks per evitare memory overflow"""
        try:
            # Prova import MetaTrader5
            try:
                import MetaTrader5 as mt5  # type: ignore
            except ImportError:
                self.logger.error("❌ MetaTrader5 package not installed. Install with: pip install MetaTrader5")
                return False
            
            # Inizializza connessione MT5
            if not mt5.initialize(): # type: ignore
                self.logger.error("❌ Failed to initialize MT5 connection")
                return False
            
            self.logger.info(f"📊 Exporting {symbol} from {start_date} to {end_date} (CHUNKED)")
            
            # CALCOLA NUMERO DI GIORNI
            total_days = (end_date - start_date).days
            self.logger.info(f"📅 Total period: {total_days} days")
            
            # STRATEGIA CHUNKED PER MEMORIA
            if total_days > 60:
                # Per periodi lunghi: chunk di 30 giorni
                chunk_days = 30
                self.logger.info(f"🔄 Using 30-day chunks for memory safety")
            elif total_days > 30:
                # Per periodi medi: chunk di 15 giorni  
                chunk_days = 15
                self.logger.info(f"🔄 Using 15-day chunks")
            else:
                # Per periodi brevi: tutto insieme
                chunk_days = total_days
                self.logger.info(f"🔄 Single chunk (period < 30 days)")
            
            # ESPORTA IN CHUNKS
            total_ticks_exported = 0
            current_date = start_date
            chunk_number = 1
            first_chunk = True
            
            # Apri file per scrittura
            with open(output_file, 'w', encoding='utf-8') as f:
                while current_date < end_date:
                    # Calcola end date del chunk corrente
                    chunk_end = min(current_date + timedelta(days=chunk_days), end_date)
                    
                    self.logger.info(f"📊 Chunk {chunk_number}: {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
                    
                    # EXPORT CHUNK SINGOLO
                    try:
                        # Ottieni dati tick per questo chunk
                        ticks = mt5.copy_ticks_range(symbol, current_date, chunk_end, mt5.COPY_TICKS_ALL) # type: ignore
                        
                        if ticks is None or len(ticks) == 0:
                            self.logger.warning(f"⚠️ No ticks in chunk {chunk_number}")
                            current_date = chunk_end
                            chunk_number += 1
                            continue
                        
                        chunk_tick_count = len(ticks)
                        total_ticks_exported += chunk_tick_count
                        
                        self.logger.info(f"✅ Chunk {chunk_number}: {chunk_tick_count:,} ticks retrieved")
                        
                        # PROCESSA CHUNK NUMPY VOID SAFE
                        if first_chunk:
                            # Scrivi header solo nel primo chunk
                            header = {
                                "type": "backtest_start",
                                "symbol": symbol,
                                "start_time": datetime.fromtimestamp(ticks[0]['time']).isoformat(),
                                "end_time": "TBD",  # Aggiorniamo alla fine
                                "total_ticks": "TBD",  # Aggiorniamo alla fine
                                "export_time": datetime.now().isoformat(),
                                "chunked_export": True,
                                "chunk_days": chunk_days
                            }
                            f.write(json.dumps(header) + '\n')
                            first_chunk = False
                        
                        # PROCESSA TICKS - NUMPY VOID SAFE!
                        for tick in ticks:
                            # ✅ FIX: Accesso diretto alle proprietà numpy.void
                            tick_time = tick['time']
                            tick_bid = tick['bid']
                            tick_ask = tick['ask']
                            
                            # ✅ FIX: Gestione sicura dei campi opzionali
                            tick_last = tick['last'] if 'last' in tick.dtype.names else (tick_bid + tick_ask) / 2
                            tick_volume = tick['volume'] if 'volume' in tick.dtype.names else 1
                            
                            # ✅ FIX: Calcolo spread sicuro
                            spread_percentage = 0.0
                            if tick_bid > 0:
                                spread_percentage = (tick_ask - tick_bid) / tick_bid
                            
                            tick_data = {
                                "type": "tick",
                                "timestamp": datetime.fromtimestamp(tick_time).strftime('%Y.%m.%d %H:%M:%S'),
                                "symbol": symbol,
                                "bid": float(tick_bid),
                                "ask": float(tick_ask),
                                "last": float(tick_last),
                                "volume": int(tick_volume),
                                "spread_percentage": float(spread_percentage),
                                # Calcoli semplificati per backtest
                                "price_change_1m": 0.0,
                                "price_change_5m": 0.0,
                                "volatility": 0.0,
                                "momentum_5m": 0.0,
                                "market_state": "backtest"
                            }
                            f.write(json.dumps(tick_data) + '\n')
                        
                        # LIBERA MEMORIA DEL CHUNK
                        del ticks
                        import gc
                        gc.collect()
                        
                        self.logger.info(f"💾 Chunk {chunk_number} written and memory freed")
                        
                    except Exception as e:
                        self.logger.error(f"❌ Error processing chunk {chunk_number}: {e}")
                        # Log dettagli per debug
                        if 'ticks' in locals() and len(ticks) > 0:
                            self.logger.error(f"🔍 Debug: tick[0] type: {type(ticks[0])}")
                            self.logger.error(f"🔍 Debug: tick[0] dtype: {ticks[0].dtype if hasattr(ticks[0], 'dtype') else 'no dtype'}")
                            if hasattr(ticks[0], 'dtype') and hasattr(ticks[0].dtype, 'names'):
                                self.logger.error(f"🔍 Debug: available fields: {ticks[0].dtype.names}")
                        # Continua con il prossimo chunk invece di fallire completamente
                    
                    # Avanza al prossimo chunk
                    current_date = chunk_end
                    chunk_number += 1
                
                # AGGIORNA HEADER CON TOTALI FINALI
                self.logger.info(f"📝 Updating header with final totals...")
            
            # AGGIORNA HEADER (riscrive il file con header corretto)
            self._update_header_with_totals(output_file, symbol, total_ticks_exported)
            
            mt5.shutdown() # type: ignore
            
            self.logger.info(f"✅ Export completed: {total_ticks_exported:,} total ticks")
            self.logger.info(f"📁 File: {output_file}")
            
            # ✅ CRITICAL: Return False se nessun tick esportato
            if total_ticks_exported == 0:
                self.logger.error("❌ No ticks were successfully exported!")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Export error: {e}")
            import traceback
            self.logger.error(f"❌ Stack trace: {traceback.format_exc()}")
            try:
                import MetaTrader5 as mt5  # type: ignore
                mt5.shutdown() # type: ignore
            except:
                pass
            return False
    
    def _update_header_with_totals(self, output_file: str, symbol: str, total_ticks: int) -> None:
        """Aggiorna l'header con i totali finali"""
        try:
            # Leggi tutto il file
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                self.logger.warning("⚠️ No lines in file to update header")
                return
            
            # Aggiorna la prima riga (header)
            header_line = lines[0].strip()
            header = json.loads(header_line)
            
            # Trova ultimo tick per end_time
            last_tick_time = None
            for line in reversed(lines[1:]):  # Skip header
                if line.strip():
                    try:
                        tick_data = json.loads(line.strip())
                        if tick_data.get('type') == 'tick':
                            last_tick_time = tick_data['timestamp']
                            break
                    except:
                        continue
            
            # Aggiorna header
            header['total_ticks'] = total_ticks
            if last_tick_time:
                header['end_time'] = datetime.strptime(last_tick_time, '%Y.%m.%d %H:%M:%S').isoformat()
            
            # Riscrivi file con header aggiornato
            lines[0] = json.dumps(header) + '\n'
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            self.logger.info(f"📝 Header updated: {total_ticks:,} total ticks")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not update header: {e}")
    
    def _save_tick_data(self, df: pd.DataFrame, symbol: str, output_file: str) -> None:
        """DEPRECATED - Mantienuto per compatibilità ma non più usato"""
        # Questo metodo non viene più chiamato nella versione chunked
        # Mantienuto solo per compatibilità con eventuali chiamate esterne
        self.logger.warning("⚠️ _save_tick_data called but chunked export is preferred")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            header = {
                "type": "backtest_start", 
                "symbol": symbol,
                "start_time": df['time'].iloc[0].isoformat(),
                "end_time": df['time'].iloc[-1].isoformat(),
                "total_ticks": len(df),
                "export_time": datetime.now().isoformat(),
                "chunked_export": False
            }
            f.write(json.dumps(header) + '\n')
            
            # Ticks
            for _, row in df.iterrows():
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
        """Carica dati da file JSONL (formato analyzer)"""
        ticks = []
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
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
                            ticks.append(tick)
                            
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        if line_num <= 10:  # Log solo primi errori
                            self.logger.warning(f"⚠️ Error parsing line {line_num}: {e}")
            
            self.logger.info(f"✅ Loaded {len(ticks)} ticks from JSONL")
            return ticks
            
        except Exception as e:
            self.logger.error(f"❌ Error loading JSONL: {e}")
            return []

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
        """Esegue backtest completo - VERSIONE ASINCRONA"""
        
        self.logger.info("="*60)
        self.logger.info("🔄 STARTING MT5 BACKTEST - UNIFIED SYSTEM")
        self.logger.info("="*60)
        self.logger.info(f"Symbol: {config.symbol}")
        self.logger.info(f"Period: {config.start_date} to {config.end_date}")
        self.logger.info(f"Speed: {config.speed_multiplier}x")
        self.logger.info(f"Data source: {config.data_source}")
        
        try:
            # 1. Setup sistema unificato
            if not await self.setup_unified_system(config):
                return False
            
            # 2. Carica/esporta dati
            data_file = f"{self.analyzer_data_path}/backtest_{config.symbol}_{config.start_date.strftime('%Y%m%d')}_{config.end_date.strftime('%Y%m%d')}.jsonl"
            
            if config.data_source == 'mt5_export':
                if not self._export_mt5_data(config, data_file):
                    return False
            
            # 3. Carica dati processati
            ticks = self._load_backtest_data(config, data_file)
            if not ticks:
                self.logger.error("❌ No data loaded for backtest")
                return False
            
            # 4. Esegui backtest con sistema unificato
            success = await self._execute_backtest_unified(config, ticks)
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Backtest execution error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # Cleanup sempre
            await self.cleanup_unified_system()
    
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
            console_interval = 50_000  # Console più frequente
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
                    checkpoint_interval = max(50_000, total_ticks // 20)  # Ogni 5%
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
    """Crea configurazione backtest per N mesi indietro"""
    end_date = datetime.now()
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