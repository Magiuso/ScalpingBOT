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
    """Esporta dati storici da MT5"""
    
    def __init__(self):
        self.logger = logging.getLogger('MT5DataExporter')
        
    def export_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                             output_file: str) -> bool:
        """Esporta dati storici da MT5"""
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
            
            self.logger.info(f"📊 Exporting {symbol} from {start_date} to {end_date}")
            
            # Ottieni dati tick con type checking
            ticks = mt5.copy_ticks_range(symbol, start_date, end_date, mt5.COPY_TICKS_ALL) # type: ignore
            
            if ticks is None or len(ticks) == 0:
                self.logger.error(f"❌ No tick data found for {symbol}")
                mt5.shutdown() # type: ignore
                return False
            
            self.logger.info(f"✅ Retrieved {len(ticks)} ticks")
            
            # Converti in DataFrame
            df = pd.DataFrame(ticks)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Salva in formato compatibile
            self._save_tick_data(df, symbol, output_file)
            
            mt5.shutdown() # type: ignore
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Export error: {e}")
            try:
                import MetaTrader5 as mt5  # type: ignore
                mt5.shutdown() # type: ignore
            except:
                pass
            return False
    
    def _save_tick_data(self, df: pd.DataFrame, symbol: str, output_file: str) -> None:
        """Salva dati tick in formato JSON Lines"""
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            header = {
                "type": "backtest_start",
                "symbol": symbol,
                "start_time": df['time'].iloc[0].isoformat(),
                "end_time": df['time'].iloc[-1].isoformat(),
                "total_ticks": len(df),
                "export_time": datetime.now().isoformat()
            }
            f.write(json.dumps(header) + '\n')
            
            # Ticks con conversioni esplicite per evitare errori Pylance
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
                    # Calcoli semplificati per backtest
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
    
    def run_backtest_thread_safe(self, config: BacktestConfig) -> bool:
        """Versione thread-safe che evita completamente i problemi di event loop"""
        
        import asyncio
        import threading
        import queue
        
        self.logger.info("🔄 Using thread-safe execution approach")
        
        # Usa sempre un thread separato per evitare conflitti
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def run_async_in_thread():
            try:
                # Nuovo loop nel thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                
                try:
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
        
        # Controlla se ci sono stati errori
        if not exception_queue.empty():
            error = exception_queue.get()
            self.logger.error(f"❌ Error in async execution: {error}")
            return False
        
        # Ottieni risultato
        if not result_queue.empty():
            return result_queue.get()
        else:
            self.logger.error("❌ No result from async execution")
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
        """Esegue il backtest con sistema unificato"""
        
        if not self.unified_system:
            self.logger.error("❌ Unified system not available")
            return False
        
        total_ticks = len(ticks)
        self.logger.info(f"🚀 Starting unified backtest with {total_ticks:,} ticks")
        
        start_time = time.time()
        processed_ticks = 0
        error_count = 0
        
        # Progress reporting
        checkpoint_interval = min(10000, max(1, total_ticks // 100))  # Ogni 1% o 10k ticks
        
        try:
            for i, tick in enumerate(ticks):
                try:
                    # ✅ NUOVO: Usa UnifiedAnalyzerSystem.process_tick()
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
                
                # Progress report
                if i > 0 and i % checkpoint_interval == 0:
                    progress = (i / total_ticks) * 100
                    elapsed = time.time() - start_time
                    speed = processed_ticks / elapsed if elapsed > 0 else 0
                    eta = (total_ticks - i) / speed if speed > 0 else 0
                    
                    self.logger.info(f"📊 Progress: {progress:.1f}% | "
                                   f"Speed: {speed:.0f} ticks/sec | "
                                   f"Processed: {processed_ticks:,} | "
                                   f"Errors: {error_count} | "
                                   f"ETA: {eta/60:.1f}min")
                
                # Speed control (opzionale)
                if config.speed_multiplier < 1000:
                    await asyncio.sleep(0.001 / config.speed_multiplier)
            
            # Final stats
            elapsed = time.time() - start_time
            speed = processed_ticks / elapsed if elapsed > 0 else 0
            
            self.logger.info("="*60)
            self.logger.info("✅ UNIFIED BACKTEST COMPLETED")
            self.logger.info("="*60)
            self.logger.info(f"Total ticks processed: {processed_ticks:,}")
            self.logger.info(f"Processing time: {elapsed:.1f} seconds")
            self.logger.info(f"Average speed: {speed:.0f} ticks/sec")
            self.logger.info(f"Speed multiplier achieved: {speed:.0f}x real-time")
            self.logger.info(f"Error count: {error_count}")
            self.logger.info(f"Success rate: {(processed_ticks/(processed_ticks+error_count))*100:.1f}%")
            
            # System status finale
            if hasattr(self.unified_system, 'get_system_status'):
                try:
                    status = self.unified_system.get_system_status()
                    self.logger.info("📊 FINAL SYSTEM STATUS:")
                    self.logger.info(f"   System running: {status.get('system', {}).get('running', 'unknown')}")
                    self.logger.info(f"   Total events: {status.get('system', {}).get('stats', {}).get('total_events_logged', 0)}")
                    
                    if 'analyzer' in status:
                        analyzer_stats = status['analyzer']
                        self.logger.info(f"   Predictions: {analyzer_stats.get('predictions_generated', 0)}")
                        self.logger.info(f"   Buffer utilization: {analyzer_stats.get('buffer_utilization', 0):.1f}%")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not get final system status: {e}")
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Backtest interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"❌ Backtest execution error: {e}")
            import traceback
            traceback.print_exc()
            return False

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