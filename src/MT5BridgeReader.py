#==============================================================================
# MT5BridgeReader.py
# Legge dati dal Bridge MT5 (BUFFER + PERIODIC WRITE) e alimenta l'Analyzer
#==============================================================================

import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import queue
import logging
from dataclasses import dataclass
from collections import deque
import os
import sys

# Aggiungi il path dell'Analyzer se necessario
# sys.path.append('./path/to/analyzer')

# Import dell'Analyzer (assumendo che sia nello stesso progetto)
from Analyzer import AdvancedMarketAnalyzer

@dataclass
class MT5TickData:
    """Struttura per dati tick da MT5"""
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

class MT5BridgeReader:
    """
    Legge dati tick dal Bridge MT5 (BUFFER + PERIODIC WRITE) e li invia all'Analyzer
    """
    
    def __init__(self, mt5_files_path: str, analyzer_data_path: str = "./analyzer_data"):
        """
        Args:
            mt5_files_path: Path alla cartella MQL5/Files di MT5
            analyzer_data_path: Path dove salvare i dati dell'Analyzer
        """
        self.mt5_files_path = Path(mt5_files_path)
        self.analyzer_data_path = analyzer_data_path
        
        # Analyzer
        self.analyzer = AdvancedMarketAnalyzer(analyzer_data_path)
        
        # File monitoring
        self.monitored_files: Dict[str, Dict] = {}  # symbol -> file_info
        self.tick_queue = queue.Queue(maxsize=10000)
        self.processing_queue = queue.Queue(maxsize=1000)
        
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
        self.polling_interval = 0.5  # 500ms - più lento dato che file si aggiorna ogni 10s
        self.batch_size = 10  # Process N ticks at once
        self.max_file_age_hours = 24  # Ignore files older than 24h
        self.read_timeout = 5.0  # Timeout per lettura file
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging system"""
        # Crea directory se non esiste
        os.makedirs(self.analyzer_data_path, exist_ok=True)
        
        log_format = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'{self.analyzer_data_path}/bridge_reader.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger('MT5BridgeReader')
        
    def start(self):
        """Avvia il reader"""
        if self.is_running:
            self.logger.warning("Reader già in esecuzione")
            return
            
        self.logger.info("🚀 Starting MT5 Bridge Reader (BUFFER + PERIODIC WRITE)...")
        
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        # Trova e registra file esistenti
        self._discover_analyzer_files()
        
        # Avvia threads
        self.file_monitor_thread = threading.Thread(target=self._file_monitor_loop, daemon=True)
        self.data_processor_thread = threading.Thread(target=self._data_processor_loop, daemon=True)
        self.analyzer_feeder_thread = threading.Thread(target=self._analyzer_feeder_loop, daemon=True)
        
        self.file_monitor_thread.start()
        self.data_processor_thread.start()
        self.analyzer_feeder_thread.start()
        
        self.logger.info("✅ Reader started successfully")
        
    def stop(self):
        """Ferma il reader"""
        self.logger.info("🛑 Stopping MT5 Bridge Reader...")
        
        self.is_running = False
        
        # Attendi che i thread finiscano
        if self.file_monitor_thread:
            self.file_monitor_thread.join(timeout=5)
        if self.data_processor_thread:
            self.data_processor_thread.join(timeout=5)
        if self.analyzer_feeder_thread:
            self.analyzer_feeder_thread.join(timeout=5)
            
        # Salva stato analyzer
        self.analyzer.save_all_states()
        
        self.logger.info("✅ Reader stopped")
        self._print_final_stats()
        
    def _discover_analyzer_files(self):
        """Scopre file analyzer esistenti nella cartella MT5 - BUFFER + PERIODIC WRITE"""
        if not self.mt5_files_path.exists():
            self.logger.error(f"❌ MT5 files path not found: {self.mt5_files_path}")
            return
            
        # PATTERN CORRETTO: analyzer_SYMBOL.jsonl (SEMPLICE!)
        pattern = "analyzer_*.jsonl"
        files = list(self.mt5_files_path.glob(pattern))
        
        self.logger.info(f"🔍 Scanning {self.mt5_files_path} for pattern: {pattern}")
        self.logger.info(f"🔍 Found {len(files)} potential files: {[f.name for f in files]}")
        
        # NESSUN FILTRO - prendiamo tutti i file analyzer_*.jsonl
        for file_path in files:
            filename = file_path.name
            self.logger.info(f"🔍 Checking file: {filename}")
            
            symbol = self._extract_symbol_from_filename(filename)
            if symbol:
                self.logger.info(f"✅ Valid analyzer file found: {filename} -> symbol: {symbol}")
                self._register_file(symbol, file_path)
            else:
                self.logger.warning(f"⚠️ Could not extract symbol from: {filename}")
                
        self.stats['files_discovered'] = len(self.monitored_files)
        self.logger.info(f"📁 Successfully registered {len(self.monitored_files)} analyzer files")
        
        if len(self.monitored_files) == 0:
            self.logger.warning("⚠️ No analyzer files found! Make sure MT5 EA is running and generating files.")
            self.logger.info(f"🔍 Expected files like: analyzer_USTEC.jsonl, analyzer_EURUSD.jsonl, etc.")
            self.logger.info(f"🔍 In directory: {self.mt5_files_path}")
            
            # Lista tutti i file .jsonl per debug
            all_jsonl = list(self.mt5_files_path.glob("*.jsonl"))
            if all_jsonl:
                self.logger.info(f"🔍 All .jsonl files found: {[f.name for f in all_jsonl]}")
            else:
                self.logger.info("🔍 No .jsonl files found at all!")
        
    def _extract_symbol_from_filename(self, filename: str) -> Optional[str]:
        """Estrae il simbolo dal nome file - FORMATO SEMPLICE: analyzer_SYMBOL.jsonl"""
        try:
            # analyzer_USTEC.jsonl -> USTEC
            if filename.startswith('analyzer_') and filename.endswith('.jsonl'):
                # Rimuovi 'analyzer_' dall'inizio e '.jsonl' dalla fine
                symbol = filename[9:-6]  # len('analyzer_') = 9, len('.jsonl') = 6
                
                # Verifica che il simbolo sia valido (non vuoto e senza caratteri strani)
                if symbol and len(symbol) > 0:
                    self.logger.debug(f"🔍 Extracted symbol '{symbol}' from filename '{filename}'")
                    return symbol
                else:
                    self.logger.warning(f"⚠️ Empty symbol extracted from {filename}")
            else:
                self.logger.debug(f"🔍 Filename {filename} doesn't match analyzer_*.jsonl pattern")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting symbol from {filename}: {e}")
        return None
        
    def _register_file(self, symbol: str, file_path: Path):
        """Registra un file per monitoraggio"""
        if not file_path.exists():
            self.logger.warning(f"⚠️ File not found: {file_path}")
            return
            
        # Aggiungi asset all'analyzer
        self.analyzer.add_asset(symbol)
        
        self.monitored_files[symbol] = {
            'file_path': file_path,
            'last_position': 0,
            'last_modified': file_path.stat().st_mtime,
            'total_lines_read': 0,
            'last_tick_time': None,
            'session_count': 0,
            'last_read_attempt': None,
            'consecutive_errors': 0
        }
        
        # Leggi posizione finale del file per nuovi dati (non da inizio)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.seek(0, 2)  # Vai alla fine del file
                self.monitored_files[symbol]['last_position'] = f.tell()
        except Exception as e:
            self.logger.error(f"❌ Error setting file position for {file_path}: {e}")
            self.monitored_files[symbol]['last_position'] = 0
            
        self.logger.info(f"📈 Registered {symbol} -> {file_path.name}")
        
        # NON leggere dati esistenti per il sistema BUFFER + PERIODIC WRITE
        # Il file viene riscritto periodicamente, quindi leggiamo solo nuovi dati
        self.logger.info(f"🔄 {symbol} ready for real-time monitoring")
        
    def _file_monitor_loop(self):
        """Loop principale per monitorare i file"""
        self.logger.info("🔍 File monitor started")
        
        while self.is_running:
            try:
                for symbol, file_info in self.monitored_files.items():
                    self._check_file_for_updates(symbol, file_info)
                    
                # Auto-discovery di nuovi file ogni 30 secondi
                if hasattr(self, '_last_discovery'):
                    if time.time() - self._last_discovery > 30:
                        self._discover_new_files()
                        self._last_discovery = time.time()
                else:
                    self._last_discovery = time.time()
                    
                time.sleep(self.polling_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Error in file monitor: {e}")
                self.stats['errors'] += 1
                time.sleep(1)
                
    def _discover_new_files(self):
        """Scopre nuovi file analyzer che potrebbero essere stati creati"""
        current_symbols = set(self.monitored_files.keys())
        
        # Ri-scopri tutti i file analyzer
        pattern = "analyzer_*.jsonl"
        files = list(self.mt5_files_path.glob(pattern))
        
        new_files_found = 0
        for file_path in files:
            symbol = self._extract_symbol_from_filename(file_path.name)
            if symbol and symbol not in current_symbols:
                self.logger.info(f"🆕 New analyzer file discovered: {symbol} -> {file_path.name}")
                self._register_file(symbol, file_path)
                new_files_found += 1
                
        if new_files_found > 0:
            self.logger.info(f"✅ Registered {new_files_found} new files")
                
    def _check_file_for_updates(self, symbol: str, file_info: Dict):
        """Controlla se un file ha nuovi dati"""
        file_path = file_info['file_path']
        file_info['last_read_attempt'] = datetime.now()
        
        try:
            current_mtime = file_path.stat().st_mtime
            
            # Se il file è stato modificato
            if current_mtime > file_info['last_modified']:
                self._read_new_data(symbol, file_info)
                file_info['last_modified'] = current_mtime
                file_info['consecutive_errors'] = 0  # Reset errori
                
        except FileNotFoundError:
            self.logger.warning(f"⚠️ File disappeared: {file_path}")
            file_info['consecutive_errors'] += 1
        except PermissionError:
            # File potrebbe essere temporaneamente bloccato durante scrittura
            # Non logghiamo come errore se succede sporadicamente
            file_info['consecutive_errors'] += 1
            if file_info['consecutive_errors'] > 10:
                self.logger.warning(f"⚠️ Repeated permission errors for {file_path}")
        except Exception as e:
            self.logger.error(f"❌ Error checking {file_path}: {e}")
            file_info['consecutive_errors'] += 1
            
    def _read_new_data(self, symbol: str, file_info: Dict):
        """Legge nuovi dati da un file"""
        file_path = file_info['file_path']
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Vai alla posizione dell'ultima lettura
                f.seek(file_info['last_position'])
                
                # Leggi nuove linee
                new_lines = f.readlines()
                
                # Aggiorna posizione
                file_info['last_position'] = f.tell()
                
                # Processa le nuove linee
                new_ticks = 0
                for line in new_lines:
                    line = line.strip()
                    if line:
                        if self._parse_and_queue_tick(symbol, line, file_info):
                            new_ticks += 1
                            
                if new_ticks > 0:
                    self.logger.debug(f"📥 {symbol}: {new_ticks} new ticks")
                        
        except Exception as e:
            self.logger.error(f"❌ Error reading new data from {file_path}: {e}")
            file_info['consecutive_errors'] += 1
            
    def _parse_and_queue_tick(self, symbol: str, json_line: str, file_info: Dict, is_historical: bool = False) -> bool:
        """Parsa una linea JSON e la mette in coda"""
        try:
            data = json.loads(json_line)
            
            # Gestisci diversi tipi di messaggi
            msg_type = data.get('type')
            
            if msg_type == 'session_start':
                file_info['session_count'] += 1
                self.stats['session_starts'] += 1
                self.logger.info(f"🟢 {symbol} session started (v{data.get('version', 'unknown')})")
                return False
                
            elif msg_type == 'session_end':
                self.stats['session_ends'] += 1
                self.logger.info(f"🔴 {symbol} session ended ({data.get('total_ticks', 0)} ticks)")
                return False
                
            elif msg_type != 'tick':
                # Altri tipi di messaggio (latest_data, ecc.)
                return False
            
            # Verifica che sia un tick valido per il simbolo corretto
            if data.get('symbol') != symbol:
                return False
                
            # Parsa timestamp - NUOVO FORMATO
            timestamp_str = data['timestamp']
            try:
                # Formato: "2025.06.26 16:30:45"
                timestamp = datetime.strptime(timestamp_str, '%Y.%m.%d %H:%M:%S')
            except ValueError:
                # Prova formato alternativo
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                
            # Crea oggetto tick
            tick = MT5TickData(
                timestamp=timestamp,
                symbol=data['symbol'],
                bid=float(data['bid']),
                ask=float(data['ask']),
                last=float(data.get('last', 0)),
                volume=int(data.get('volume', 0)),
                spread_percentage=float(data['spread_percentage']),
                price_change_1m=float(data['price_change_1m']),
                price_change_5m=float(data['price_change_5m']),
                volatility=float(data['volatility']),
                momentum_5m=float(data['momentum_5m']),
                market_state=data['market_state']
            )
            
            # Metti in coda per processamento (se non è storico o se la coda non è piena)
            if not self.tick_queue.full():
                self.tick_queue.put(tick)
                self.stats['ticks_read'] += 1
                file_info['total_lines_read'] += 1
                file_info['last_tick_time'] = tick.timestamp
                self.stats['last_tick_time'] = tick.timestamp
                return True
            else:
                if not is_historical:  # Log solo per tick real-time
                    self.logger.warning("⚠️ Tick queue full, dropping tick")
                return False
                
        except json.JSONDecodeError as e:
            self.logger.warning(f"⚠️ Invalid JSON in {symbol}: {json_line[:100]}...")
            return False
        except KeyError as e:
            self.logger.warning(f"⚠️ Missing field {e} in {symbol}: {json_line[:100]}...")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error parsing tick for {symbol}: {e}")
            return False
            
    def _data_processor_loop(self):
        """Loop per processare i tick in batch"""
        self.logger.info("⚙️ Data processor started")
        
        batch = []
        
        while self.is_running:
            try:
                # Raccogli batch di tick
                while len(batch) < self.batch_size and self.is_running:
                    try:
                        tick = self.tick_queue.get(timeout=0.5)
                        batch.append(tick)
                    except queue.Empty:
                        break
                        
                # Processa batch se non vuoto
                if batch:
                    self._process_tick_batch(batch)
                    batch = []
                    
            except Exception as e:
                self.logger.error(f"❌ Error in data processor: {e}")
                self.stats['errors'] += 1
                
    def _process_tick_batch(self, ticks: list):
        """Processa un batch di tick"""
        # Raggruppa per simbolo
        by_symbol = {}
        for tick in ticks:
            if tick.symbol not in by_symbol:
                by_symbol[tick.symbol] = []
            by_symbol[tick.symbol].append(tick)
            
        # Processa ogni simbolo
        for symbol, symbol_ticks in by_symbol.items():
            for tick in symbol_ticks:
                self._send_tick_to_analyzer(tick)
                
    def _send_tick_to_analyzer(self, tick: MT5TickData):
        """Invia un tick all'Analyzer"""
        try:
            # Calcola prezzo mid
            mid_price = (tick.bid + tick.ask) / 2.0
            
            # Invia all'analyzer
            result = self.analyzer.process_tick(
                asset=tick.symbol,
                timestamp=tick.timestamp,
                price=mid_price,
                volume=max(1, tick.volume),  # Volume minimo 1
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
            
            self.stats['ticks_processed'] += 1
            
            # Se l'analyzer ha prodotto un'analisi, mettila in coda
            if result and 'status' not in result:  # Non è un messaggio di stato
                if not self.processing_queue.full():
                    self.processing_queue.put((tick.symbol, result))
                    self.stats['analysis_count'] += 1
                    
        except Exception as e:
            self.logger.error(f"❌ Error sending tick to analyzer: {e}")
            self.stats['errors'] += 1
            
    def _analyzer_feeder_loop(self):
        """Loop per gestire output dell'analyzer"""
        self.logger.info("🧠 Analyzer feeder started")
        
        while self.is_running:
            try:
                # Ottieni analisi dalla coda
                try:
                    symbol, analysis = self.processing_queue.get(timeout=1.0)
                    self._handle_analysis_result(symbol, analysis)
                except queue.Empty:
                    continue
                    
            except Exception as e:
                self.logger.error(f"❌ Error in analyzer feeder: {e}")
                
    def _handle_analysis_result(self, symbol: str, analysis: Dict[str, Any]):
        """Gestisce il risultato di un'analisi"""
        try:
            # Log analisi interessanti
            if analysis.get('recommendations'):
                self.logger.info(f"📊 {symbol} Analysis: {len(analysis['recommendations'])} recommendations")
                
                # Log top recommendation
                top_rec = analysis['recommendations'][0]
                self.logger.info(f"🎯 Top: {top_rec.get('type')} -> {top_rec.get('action')} (conf: {top_rec.get('confidence', 0):.2f})")
                
            # Qui potresti salvare le analisi, inviarle all'Observer, ecc.
            
        except Exception as e:
            self.logger.error(f"❌ Error handling analysis: {e}")
            
    def get_status(self) -> Dict[str, Any]:
        """Ottieni status del reader"""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds() if self.stats['start_time'] else 0
        
        return {
            'running': self.is_running,
            'uptime_seconds': uptime,
            'monitored_files': len(self.monitored_files),
            'stats': self.stats.copy(),
            'queue_sizes': {
                'tick_queue': self.tick_queue.qsize(),
                'processing_queue': self.processing_queue.qsize()
            },
            'assets': {
                symbol: {
                    'file': str(info['file_path'].name),
                    'lines_read': info['total_lines_read'],
                    'last_tick': info['last_tick_time'].isoformat() if info['last_tick_time'] else None,
                    'sessions': info['session_count'],
                    'errors': info['consecutive_errors'],
                    'last_read': info['last_read_attempt'].isoformat() if info['last_read_attempt'] else None
                }
                for symbol, info in self.monitored_files.items()
            }
        }
        
    def print_status(self):
        """Stampa status corrente"""
        status = self.get_status()
        
        print("\n" + "="*60)
        print("📊 MT5 BRIDGE READER STATUS (BUFFER + PERIODIC WRITE)")
        print("="*60)
        print(f"🔄 Running: {'YES' if status['running'] else 'NO'}")
        print(f"⏱️ Uptime: {status['uptime_seconds']:.0f} seconds")
        print(f"📁 Monitored files: {status['monitored_files']}")
        print(f"📈 Ticks read: {status['stats']['ticks_read']:,}")
        print(f"⚙️ Ticks processed: {status['stats']['ticks_processed']:,}")
        print(f"🧠 Analysis count: {status['stats']['analysis_count']:,}")
        print(f"🟢 Session starts: {status['stats']['session_starts']}")
        print(f"🔴 Session ends: {status['stats']['session_ends']}")
        print(f"❌ Errors: {status['stats']['errors']}")
        print(f"📦 Queues: Ticks={status['queue_sizes']['tick_queue']}, Analysis={status['queue_sizes']['processing_queue']}")
        
        if status['stats']['last_tick_time']:
            print(f"🕒 Last tick: {status['stats']['last_tick_time']}")
            
        print("\n📈 ASSETS:")
        for symbol, info in status['assets'].items():
            print(f"  {symbol}: {info['lines_read']:,} ticks, {info['sessions']} sessions, {info['errors']} errors")
            
        print("="*60)
        
    def _print_final_stats(self):
        """Stampa statistiche finali"""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds() if self.stats['start_time'] else 0
        
        print("\n" + "="*50)
        print("📊 FINAL STATISTICS")
        print("="*50)
        print(f"Total uptime: {uptime:.0f} seconds")
        print(f"Files discovered: {self.stats['files_discovered']}")
        print(f"Total ticks read: {self.stats['ticks_read']:,}")
        print(f"Total ticks processed: {self.stats['ticks_processed']:,}")
        print(f"Total analysis: {self.stats['analysis_count']:,}")
        print(f"Processing rate: {self.stats['ticks_processed']/max(1, uptime):.1f} ticks/sec")
        print(f"Session starts/ends: {self.stats['session_starts']}/{self.stats['session_ends']}")
        print(f"Total errors: {self.stats['errors']}")
        print("="*50)


def main():
    """MT5 Bridge Reader - Production Version"""
    
    print("="*60)
    print("🚀 MT5 BRIDGE READER")
    print("="*60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configurazione path
    MT5_FILES_PATH = r"C:\Users\anton\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files"
    ANALYZER_DATA_PATH = "./analyzer_data"
    
    print(f"📁 MT5 Files: {MT5_FILES_PATH}")
    print(f"📁 Analyzer Data: {ANALYZER_DATA_PATH}")
    
    # Verifica prerequisiti
    if not os.path.exists(MT5_FILES_PATH):
        print(f"❌ ERROR: MT5 directory not found!")
        print(f"   Path: {MT5_FILES_PATH}")
        print("   Check your MT5 installation and update the path if needed.")
        return
    
    # Crea directory analyzer
    os.makedirs(ANALYZER_DATA_PATH, exist_ok=True)
    
    # Quick check per file analyzer
    try:
        analyzer_files = [f for f in os.listdir(MT5_FILES_PATH) 
                         if f.startswith('analyzer_') and f.endswith('.jsonl')]
        if analyzer_files:
            print(f"✅ Found {len(analyzer_files)} analyzer files: {analyzer_files}")
        else:
            print("⚠️  No analyzer files found yet. Make sure MT5 EA is running.")
    except Exception as e:
        print(f"⚠️  Could not check files: {e}")
    
    # Inizializza reader
    print("\n🔧 Initializing MT5 Bridge Reader...")
    try:
        reader = MT5BridgeReader(MT5_FILES_PATH, ANALYZER_DATA_PATH)
        print("✅ Reader initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize reader: {e}")
        return
    
    # Avvia reader
    print("\n🚀 Starting reader...")
    try:
        reader.start()
        print("✅ Reader started successfully")
        print("📊 Monitoring MT5 files for real-time data...")
        print("⏹️  Press Ctrl+C to stop")
        print("-" * 60)
        
        # Loop principale con status monitoring
        status_interval = 30  # Status ogni 30 secondi
        detailed_interval = 300  # Status dettagliato ogni 5 minuti
        last_status = time.time()
        last_detailed = time.time()
        
        while True:
            current_time = time.time()
            
            # Status breve ogni 30 secondi
            if current_time - last_status >= status_interval:
                try:
                    status = reader.get_status()
                    uptime = int(status['uptime_seconds'])
                    
                    # Status compatto
                    print(f"📊 {datetime.now().strftime('%H:%M:%S')} | "
                          f"Uptime: {uptime}s | "
                          f"Files: {status['monitored_files']} | "
                          f"Ticks: {status['stats']['ticks_read']:,} | "
                          f"Processed: {status['stats']['ticks_processed']:,} | "
                          f"Analysis: {status['stats']['analysis_count']:,}")
                    
                    # Alert se nessun file monitorato
                    if status['monitored_files'] == 0 and uptime > 60:
                        print("   ⚠️  No files being monitored. Check MT5 EA status.")
                    
                    # Alert se nessun tick da molto tempo
                    if status['stats']['ticks_read'] == 0 and uptime > 120:
                        print("   ⚠️  No ticks received. Check MT5 EA is generating data.")
                    
                except Exception as e:
                    print(f"⚠️  Status error: {e}")
                
                last_status = current_time
            
            # Status dettagliato ogni 5 minuti
            if current_time - last_detailed >= detailed_interval:
                print("\n" + "="*50)
                print("📊 DETAILED STATUS")
                print("="*50)
                try:
                    reader.print_status()
                except Exception as e:
                    print(f"⚠️  Detailed status error: {e}")
                print("="*50)
                last_detailed = current_time
            
            time.sleep(1)  # Check ogni secondo
            
    except KeyboardInterrupt:
        print("\n🛑 User requested shutdown...")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Shutting down...")
        try:
            reader.stop()
            print("✅ Reader stopped successfully")
        except Exception as e:
            print(f"⚠️  Error during shutdown: {e}")
        
        print("👋 MT5 Bridge Reader terminated")
        print(f"📅 Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()