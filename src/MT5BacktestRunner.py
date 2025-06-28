#!/usr/bin/env python3
"""
🔄 MT5 Backtest Runner - Versione Corretta
Sistema per accelerare l'apprendimento dell'Analyzer con dati storici
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

# Import del nostro sistema
from MT5BridgeReader import MT5TickData
from Analyzer import AdvancedMarketAnalyzer

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
    """Runner principale per backtest"""
    
    def __init__(self, analyzer_data_path: str = "./analyzer_data"):
        self.analyzer_data_path = analyzer_data_path
        self.analyzer = AdvancedMarketAnalyzer(analyzer_data_path)
        self.data_processor = BacktestDataProcessor()
        self.exporter = MT5DataExporter()
        
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
    
    def run_backtest(self, config: BacktestConfig) -> bool:
        """Esegue backtest completo"""
        self.logger.info("="*60)
        self.logger.info("🔄 STARTING MT5 BACKTEST")
        self.logger.info("="*60)
        self.logger.info(f"Symbol: {config.symbol}")
        self.logger.info(f"Period: {config.start_date} to {config.end_date}")
        self.logger.info(f"Speed: {config.speed_multiplier}x")
        self.logger.info(f"Data source: {config.data_source}")
        
        # 1. Carica/esporta dati
        data_file = f"{self.analyzer_data_path}/backtest_{config.symbol}_{config.start_date.strftime('%Y%m%d')}_{config.end_date.strftime('%Y%m%d')}.jsonl"
        
        if config.data_source == 'mt5_export':
            if not self._export_mt5_data(config, data_file):
                return False
        
        # 2. Carica dati processati
        ticks = self._load_backtest_data(config, data_file)
        if not ticks:
            self.logger.error("❌ No data loaded for backtest")
            return False
        
        # 3. Aggiungi asset all'analyzer
        self.analyzer.add_asset(config.symbol)
        
        # 4. Esegui backtest
        return self._execute_backtest(config, ticks)
    
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
    
    def _execute_backtest(self, config: BacktestConfig, ticks: List[MT5TickData]) -> bool:
        """Esegue il backtest vero e proprio"""
        total_ticks = len(ticks)
        self.logger.info(f"🚀 Starting backtest with {total_ticks:,} ticks")
        
        start_time = time.time()
        processed_ticks = 0
        analysis_count = 0
        
        # Checkpoint per progress saving
        checkpoint_interval = min(10000, max(1, total_ticks // 100))  # Ogni 1% o 10k ticks
        last_checkpoint = 0
        
        try:
            for i, tick in enumerate(ticks):
                # Processa tick
                result = self.analyzer.process_tick(
                    asset=tick.symbol,
                    timestamp=tick.timestamp,
                    price=(tick.bid + tick.ask) / 2.0,
                    volume=max(1, tick.volume),
                    bid=tick.bid,
                    ask=tick.ask,
                    additional_data={
                        'spread_percentage': tick.spread_percentage,
                        'price_change_1m': tick.price_change_1m,
                        'price_change_5m': tick.price_change_5m,
                        'volatility': tick.volatility,
                        'momentum_5m': tick.momentum_5m,
                        'market_state': tick.market_state
                    }
                )
                
                processed_ticks += 1
                
                if result and 'status' not in result:
                    analysis_count += 1
                
                # Progress report
                if i > 0 and i % checkpoint_interval == 0:
                    progress = (i / total_ticks) * 100
                    elapsed = time.time() - start_time
                    speed = processed_ticks / elapsed if elapsed > 0 else 0
                    eta = (total_ticks - i) / speed if speed > 0 else 0
                    
                    self.logger.info(f"📊 Progress: {progress:.1f}% | "
                                   f"Speed: {speed:.0f} ticks/sec | "
                                   f"Analysis: {analysis_count} | "
                                   f"ETA: {eta/60:.1f}min")
                    
                    # Save checkpoint
                    if config.save_progress and i - last_checkpoint >= checkpoint_interval:
                        self.analyzer.save_all_states()
                        last_checkpoint = i
                
                # Speed control (se non vogliamo andare alla massima velocità)
                if config.speed_multiplier < 1000:
                    time.sleep(0.001 / config.speed_multiplier)
            
            # Final stats
            elapsed = time.time() - start_time
            speed = processed_ticks / elapsed if elapsed > 0 else 0
            
            self.logger.info("="*60)
            self.logger.info("✅ BACKTEST COMPLETED")
            self.logger.info("="*60)
            self.logger.info(f"Total ticks processed: {processed_ticks:,}")
            self.logger.info(f"Analysis generated: {analysis_count:,}")
            self.logger.info(f"Processing time: {elapsed:.1f} seconds")
            self.logger.info(f"Average speed: {speed:.0f} ticks/sec")
            self.logger.info(f"Speed multiplier achieved: {speed:.0f}x real-time")
            
            # Salva stato finale
            self.analyzer.save_all_states()
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Backtest interrupted by user")
            self.analyzer.save_all_states()
            return False
        except Exception as e:
            self.logger.error(f"❌ Backtest error: {e}")
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

def main():
    """Test backtest runner"""
    
    # Configurazione backtest
    config = create_backtest_config('USTEC', months_back=6)
    
    print("🔄 MT5 Backtest Runner")
    print(f"📊 Symbol: {config.symbol}")
    print(f"📅 Period: {config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')}")
    print(f"⚡ Speed: {config.speed_multiplier}x")
    
    confirm = input("\n🚀 Start backtest? (y/N): ")
    if confirm.lower() != 'y':
        print("Backtest cancelled.")
        return
    
    # Esegui backtest
    runner = MT5BacktestRunner()
    success = runner.run_backtest(config)
    
    if success:
        print("\n✅ Backtest completed successfully!")
        print("🧠 Analyzer is now trained on historical data")
        print("🚀 Ready for real-time analysis with improved performance")
    else:
        print("\n❌ Backtest failed")

if __name__ == "__main__":
    main()