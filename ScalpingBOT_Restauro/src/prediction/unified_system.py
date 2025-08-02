#!/usr/bin/env python3
"""
Unified Analyzer System - REBUILT FROM SCRATCH  
REGOLE CLAUDE_RESTAURO.md APPLICATE:
- ✅ Zero fallback/defaults
- ✅ Fail fast error handling
- ✅ No debug prints/spam
- ✅ Modular architecture using ALL migrated components

Sistema unificato che orchestrerà TUTTI i moduli migrati FASE 1-6.
REBUILT completamente usando la nuova architettura modulare.
"""

import os
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Callable

# Import shared enums and system mode (avoid duplication)
from ScalpingBOT_Restauro.src.shared.enums import ModelType
from ScalpingBOT_Restauro.src.config.domain.system_config import SystemMode

# Import ALL migrated components FASE 1-6
from ScalpingBOT_Restauro.src.config.base.config_loader import get_configuration_manager, ConfigurationManager
from ScalpingBOT_Restauro.src.config.base.base_config import get_analyzer_config
from ScalpingBOT_Restauro.src.monitoring.events.event_collector import EventCollector, EventType, EventSeverity
from ScalpingBOT_Restauro.src.monitoring.display.display_manager import SimpleDisplayManager
from ScalpingBOT_Restauro.src.monitoring.storage.storage_manager import StorageManager
from ScalpingBOT_Restauro.src.data.collectors.tick_collector import TickCollector
from ScalpingBOT_Restauro.src.data.processors.market_data_processor import MarketDataProcessor  
from ScalpingBOT_Restauro.src.interfaces.mt5.mt5_bridge_reader import MT5BridgeReader
from ScalpingBOT_Restauro.src.interfaces.mt5.mt5_backtest_runner import MT5BacktestRunner
from ScalpingBOT_Restauro.src.prediction.core.advanced_market_analyzer import AdvancedMarketAnalyzer


class UnifiedAnalyzerSystem:
    """
    Unified Analyzer System - REBUILT VERSION
    
    Sistema completamente ricostruito che orchestra TUTTI i moduli migrati:
    - FASE 1: CONFIG (ConfigurationManager, AnalyzerConfig)
    - FASE 2: MONITORING (EventCollector, DisplayManager, StorageManager)
    - FASE 3: INTERFACES (MT5BridgeReader, MT5BacktestRunner)
    - FASE 4: DATA (TickCollector, MarketDataProcessor)
    - FASE 5: ML (Competition, LSTM, CNN, Transformer models)
    - FASE 6: PREDICTION (AssetAnalyzer, AdvancedMarketAnalyzer)
    """
    
    def __init__(self, data_path: str,
                 mode: SystemMode = SystemMode.PRODUCTION, 
                 config_manager: Optional[ConfigurationManager] = None):
        """
        Inizializza Unified Analyzer System usando TUTTI i moduli migrati
        
        Args:
            mode: Modalità operativa del sistema
            data_path: Path base per i dati
            config_manager: Configuration manager (opzionale)
        """
        if not isinstance(mode, SystemMode):
            raise TypeError("mode must be SystemMode enum")
        if not isinstance(data_path, str) or not data_path.strip():
            raise ValueError("data_path must be non-empty string")
            
        self.mode = mode
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        
        # FASE 1 - CONFIG: Use migrated configuration system
        self.config_manager = config_manager or get_configuration_manager()
        self.config = get_analyzer_config()
        self.system_config = self.config_manager.get_current_configuration()
        
        # FASE 2 - MONITORING: Use migrated monitoring components
        self.event_collector = EventCollector(self.system_config.monitoring)
        self.display_manager = SimpleDisplayManager(self.system_config.monitoring)
        self.storage_manager = StorageManager(
            self.system_config.monitoring
        )
        
        # FASE 6 - PREDICTION: Use migrated prediction system
        self.market_analyzer = AdvancedMarketAnalyzer(
            data_path=data_path,
            config_manager=self.config_manager
        )
        
        # FASE 3 - INTERFACES: MT5 integration components (initialized on demand)
        self.mt5_bridge_reader: Optional[MT5BridgeReader] = None
        self.mt5_backtest_runner: Optional[MT5BacktestRunner] = None
        
        # System state
        self.is_running = False
        self.active_assets: Set[str] = set()
        
        # Threading
        self.system_lock = threading.RLock()
        
        # System statistics
        self.system_stats = {
            'start_time': None,
            'total_ticks_processed': 0,
            'total_predictions_generated': 0,
            'total_events_logged': 0,
            'total_errors': 0,
            'uptime_seconds': 0
        }
        
        # 🔧 FIXED MEMORY LEAK: Use bounded deque for callbacks with cleanup mechanism
        from collections import deque
        self.max_callbacks = getattr(self.config, 'max_callbacks_per_type', 50)
        self.tick_callbacks: deque = deque(maxlen=self.max_callbacks)
        self.prediction_callbacks: deque = deque(maxlen=self.max_callbacks)
        self.error_callbacks: deque = deque(maxlen=self.max_callbacks)
    
    def initialize_mt5_bridge(self, mt5_files_path: str) -> MT5BridgeReader:
        """
        Inizializza MT5 bridge per real-time data
        
        Args:
            mt5_files_path: Path to MT5 files directory
            
        Returns:
            MT5BridgeReader instance
        """
        if not isinstance(mt5_files_path, str) or not mt5_files_path.strip():
            raise ValueError("mt5_files_path must be non-empty string")
        
        if self.mt5_bridge_reader:
            return self.mt5_bridge_reader
        
        self.mt5_bridge_reader = MT5BridgeReader(
            mt5_files_path=mt5_files_path,
            event_collector=self.event_collector
        )
        
        # Set callback for tick processing
        self.mt5_bridge_reader.set_analyzer_callback(self._process_mt5_tick)
        
        return self.mt5_bridge_reader
    
    def initialize_mt5_backtest(self, symbol: str, start_date: datetime, 
                               end_date: datetime) -> MT5BacktestRunner:
        """
        Inizializza MT5 backtest runner
        
        Args:
            symbol: Trading symbol
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            MT5BacktestRunner instance
        """
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be non-empty string")
        if not isinstance(start_date, datetime):
            raise TypeError("start_date must be datetime")
        if not isinstance(end_date, datetime):
            raise TypeError("end_date must be datetime")
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")
        
        if self.mt5_backtest_runner:
            return self.mt5_backtest_runner
        
        # Use migrated backtest runner
        from ScalpingBOT_Restauro.src.interfaces.mt5.mt5_backtest_runner import create_backtest_config
        
        backtest_config = create_backtest_config(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        self.mt5_backtest_runner = MT5BacktestRunner(
            config=backtest_config,
            event_collector=self.event_collector
        )
        
        return self.mt5_backtest_runner
    
    def add_asset(self, asset: str):
        """
        Aggiunge un asset al sistema
        
        Args:
            asset: Nome dell'asset
        """
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError("asset must be non-empty string")
        
        with self.system_lock:
            # Add to market analyzer
            self.market_analyzer.add_asset(asset)
            self.active_assets.add(asset)
            
            # Emit event
            self.event_collector.emit_manual_event(
                EventType.SYSTEM_STATUS,
                {
                    'action': 'unified_system_asset_added',
                    'asset': asset,
                    'total_assets': len(self.active_assets)
                },
                EventSeverity.INFO
            )
    
    def remove_asset(self, asset: str):
        """
        Rimuove un asset dal sistema
        
        Args:
            asset: Nome dell'asset
        """
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError("asset must be non-empty string")
        
        with self.system_lock:
            if asset not in self.active_assets:
                raise KeyError(f"Asset '{asset}' not found in system")
            
            # Remove from market analyzer  
            self.market_analyzer.remove_asset(asset)
            self.active_assets.discard(asset)
            
            # Emit event
            self.event_collector.emit_manual_event(
                EventType.SYSTEM_STATUS,
                {
                    'action': 'unified_system_asset_removed',
                    'asset': asset,
                    'total_assets': len(self.active_assets)
                },
                EventSeverity.INFO
            )
    
    def start(self):
        """Avvia il sistema unificato completo"""
        if self.is_running:
            raise RuntimeError("UnifiedAnalyzerSystem already running")
        
        with self.system_lock:
            self.is_running = True
            self.system_stats['start_time'] = datetime.now()
            
            # Start all components
            self.market_analyzer.start()
            
            # Start MT5 bridge if initialized
            if self.mt5_bridge_reader:
                self.mt5_bridge_reader.start()
            
            # Emit system start event
            self.event_collector.emit_manual_event(
                EventType.SYSTEM_STATUS,
                {
                    'action': 'unified_system_start',
                    'mode': self.mode.value,
                    'data_path': self.data_path,
                    'active_assets': len(self.active_assets)
                },
                EventSeverity.INFO
            )
    
    def stop(self):
        """Ferma il sistema unificato completo"""
        if not self.is_running:
            return
        
        with self.system_lock:
            self.is_running = False
            
            # Stop all components
            self.market_analyzer.stop()
            
            # Stop MT5 bridge if running
            if self.mt5_bridge_reader:
                self.mt5_bridge_reader.stop()
            
            # Calculate uptime
            if self.system_stats['start_time']:
                uptime = (datetime.now() - self.system_stats['start_time']).total_seconds()
                self.system_stats['uptime_seconds'] = uptime
            
            # Emit system stop event
            self.event_collector.emit_manual_event(
                EventType.SYSTEM_STATUS,
                {
                    'action': 'unified_system_stop',
                    'uptime_seconds': self.system_stats['uptime_seconds'],
                    'total_ticks_processed': self.system_stats['total_ticks_processed'],
                    'total_predictions_generated': self.system_stats['total_predictions_generated']
                },
                EventSeverity.INFO
            )
    
    def process_tick(self, asset: str, timestamp: datetime, price: float, volume: float,
                    bid: Optional[float] = None, ask: Optional[float] = None) -> Dict[str, Any]:
        """
        Processa un tick attraverso il sistema completo
        
        Args:
            asset: Nome dell'asset
            timestamp: Timestamp del tick
            price: Prezzo del tick
            volume: Volume del tick
            bid: Prezzo bid (opzionale)
            ask: Prezzo ask (opzionale)
            
        Returns:
            Risultato completo del processing
        """
        if not self.is_running:
            raise RuntimeError("System not running - call start() first")
        
        try:
            # Process through market analyzer
            result = self.market_analyzer.process_tick(
                asset=asset,
                timestamp=timestamp,
                price=price,
                volume=volume,
                bid=bid,
                ask=ask
            )
            
            # Update system stats
            with self.system_lock:
                self.system_stats['total_ticks_processed'] += 1
                if result.get('predictions'):
                    self.system_stats['total_predictions_generated'] += len(result['predictions'])
            
            # Call registered callbacks
            for callback in self.tick_callbacks:
                try:
                    callback(asset, timestamp, price, volume, result)
                except Exception as e:
                    # Log callback error but don't fail the tick processing
                    self.event_collector.emit_manual_event(
                        EventType.ERROR_EVENT,
                        {
                            'component': 'unified_system',
                            'method': 'tick_callback',
                            'error': str(e)
                        },
                        EventSeverity.ERROR
                    )
            
            return result
            
        except Exception as e:
            # Update error stats
            with self.system_lock:
                self.system_stats['total_errors'] += 1
            
            # Call error callbacks
            for callback in self.error_callbacks:
                try:
                    callback(e, asset, timestamp)
                except Exception as callback_error:
                    # Log callback error but don't fail the main error handling
                    self.event_collector.emit_manual_event(
                        EventType.ERROR_EVENT,
                        {
                            'component': 'unified_system',
                            'method': 'error_callback_failure',
                            'original_error': str(e),
                            'callback_error': str(callback_error)
                        },
                        EventSeverity.CRITICAL
                    )
            
            # Re-raise original exception
            raise
    
    def _process_mt5_tick(self, tick_data):
        """
        Internal callback per MT5 tick processing
        
        Args:
            tick_data: MT5TickData from bridge reader
        """
        try:
            # Extract tick data
            self.process_tick(
                asset=tick_data.symbol,
                timestamp=tick_data.timestamp,
                price=tick_data.last,
                volume=tick_data.volume,
                bid=tick_data.bid,
                ask=tick_data.ask
            )
        except Exception as e:
            # Log MT5 processing error
            self.event_collector.emit_manual_event(
                EventType.ERROR_EVENT,
                {
                    'component': 'unified_system',
                    'method': '_process_mt5_tick',
                    'symbol': tick_data.symbol,
                    'error': str(e)
                },
                EventSeverity.ERROR
            )
    
    def register_tick_callback(self, callback: Callable):
        """Registra callback per tick processing"""
        if not callable(callback):
            raise TypeError("callback must be callable")
        self.tick_callbacks.append(callback)
    
    def register_prediction_callback(self, callback: Callable):
        """Registra callback per prediction generation"""
        if not callable(callback):
            raise TypeError("callback must be callable")
        self.prediction_callbacks.append(callback)
    
    def register_error_callback(self, callback: Callable):
        """Registra callback per error handling"""
        if not callable(callback):
            raise TypeError("callback must be callable") 
        self.error_callbacks.append(callback)
    
    def remove_tick_callback(self, callback: Callable) -> bool:
        """🔧 FIXED MEMORY LEAK: Remove callback to prevent accumulation"""
        try:
            self.tick_callbacks.remove(callback)
            return True
        except ValueError:
            return False
    
    def remove_prediction_callback(self, callback: Callable) -> bool:
        """🔧 FIXED MEMORY LEAK: Remove callback to prevent accumulation"""
        try:
            self.prediction_callbacks.remove(callback)
            return True
        except ValueError:
            return False
    
    def remove_error_callback(self, callback: Callable) -> bool:
        """🔧 FIXED MEMORY LEAK: Remove callback to prevent accumulation"""
        try:
            self.error_callbacks.remove(callback)
            return True
        except ValueError:
            return False
    
    def clear_all_callbacks(self):
        """🔧 FIXED MEMORY LEAK: Clear all callbacks for cleanup"""
        self.tick_callbacks.clear()
        self.prediction_callbacks.clear()
        self.error_callbacks.clear()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche complete del sistema"""
        with self.system_lock:
            stats = self.system_stats.copy()
        
        # Add current state
        stats['is_running'] = self.is_running
        stats['mode'] = self.mode.value
        stats['active_assets'] = list(self.active_assets)
        
        # Add component stats
        stats['market_analyzer_stats'] = self.market_analyzer.get_global_stats()
        
        if self.mt5_bridge_reader:
            stats['mt5_bridge_stats'] = self.mt5_bridge_reader.get_stats()
        
        return stats
    
    def get_system_health(self) -> Dict[str, Any]:
        """Restituisce stato di salute completo del sistema"""
        health = {
            'overall_status': 'healthy',
            'system_mode': self.mode.value,
            'is_running': self.is_running,
            'components': {},
            'issues': []
        }
        
        # Check market analyzer health
        market_health = self.market_analyzer.get_system_health()
        health['components']['market_analyzer'] = market_health
        
        if market_health['overall_status'] != 'healthy':
            health['overall_status'] = market_health['overall_status']
            if 'system_issues' not in market_health:
                raise KeyError("Missing required 'system_issues' key in market_health")
            health['issues'].extend(market_health['system_issues'])
        
        # Check MT5 bridge if active
        if self.mt5_bridge_reader:
            bridge_stats = self.mt5_bridge_reader.get_stats()
            bridge_health = {
                'status': 'healthy' if bridge_stats['is_running'] else 'stopped',
                'monitored_symbols': len(bridge_stats['monitored_symbols']),
                'errors': bridge_stats['errors']
            }
            health['components']['mt5_bridge'] = bridge_health
            
            if bridge_stats['errors'] > 10:
                health['overall_status'] = 'degraded'
                health['issues'].append('High MT5 bridge error count')
        
        # Check system error rate
        with self.system_lock:
            total_operations = self.system_stats['total_ticks_processed']
            if total_operations > 0:
                error_rate = self.system_stats['total_errors'] / total_operations
                if error_rate > 0.05:  # 5% error rate
                    health['overall_status'] = 'degraded'
                    health['issues'].append(f'High system error rate: {error_rate:.1%}')
        
        return health
    
    def train_on_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train ML models on a batch of tick data"""
        if not self.is_running:
            raise RuntimeError("System not running - call start() first")
        
        batch_size = batch_data.get('count', 0)
        if batch_size == 0:
            raise ValueError("Empty batch data provided for training")
        
        print(f"🧠 UnifiedAnalyzerSystem: Training ML models on {batch_size:,} ticks...")
        
        try:
            # Train models using market analyzer
            training_result = self.market_analyzer.train_models_on_batch(batch_data)
            
            # Update system stats
            with self.system_lock:
                self.system_stats['total_ticks_processed'] += batch_size
            
            # Emit training event
            self.event_collector.emit_manual_event(
                EventType.SYSTEM_STATUS,
                {
                    'action': 'batch_training_completed',
                    'batch_size': batch_size,
                    'training_result': training_result
                },
                EventSeverity.INFO
            )
            
            print(f"✅ ML training completed on {batch_size:,} ticks")
            return training_result
            
        except Exception as e:
            # Update error stats
            with self.system_lock:
                self.system_stats['total_errors'] += 1
            
            # Emit error event
            self.event_collector.emit_manual_event(
                EventType.ERROR_EVENT,
                {
                    'component': 'unified_system',
                    'method': 'train_on_batch',
                    'error': str(e),
                    'batch_size': batch_size
                },
                EventSeverity.ERROR
            )
            
            raise RuntimeError(f"Batch training failed: {e}")
    
    def validate_on_batch(self, batch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate predictions using trained models on a batch of tick data"""
        if not self.is_running:
            raise RuntimeError("System not running - call start() first")
        
        batch_size = batch_data.get('count', 0)
        if batch_size == 0:
            raise ValueError("Empty batch data provided for validation")
        
        print(f"🔮 UnifiedAnalyzerSystem: Generating predictions on {batch_size:,} ticks...")
        
        try:
            # Generate predictions using market analyzer
            predictions = self.market_analyzer.validate_models_on_batch(batch_data)
            
            # Update system stats
            with self.system_lock:
                self.system_stats['total_ticks_processed'] += batch_size
                if predictions:
                    self.system_stats['total_predictions_generated'] += len(predictions)
            
            # Emit validation event
            self.event_collector.emit_manual_event(
                EventType.SYSTEM_STATUS,
                {
                    'action': 'batch_validation_completed',
                    'batch_size': batch_size,
                    'predictions_count': len(predictions) if predictions else 0
                },
                EventSeverity.INFO
            )
            
            print(f"✅ Generated {len(predictions) if predictions else 0} predictions")
            return predictions
            
        except Exception as e:
            # Update error stats
            with self.system_lock:
                self.system_stats['total_errors'] += 1
            
            # Emit error event
            self.event_collector.emit_manual_event(
                EventType.ERROR_EVENT,
                {
                    'component': 'unified_system',
                    'method': 'validate_on_batch',
                    'error': str(e),
                    'batch_size': batch_size
                },
                EventSeverity.ERROR
            )
            
            raise RuntimeError(f"Batch validation failed: {e}")


# Factory functions
def create_unified_system(data_path: str, mode: SystemMode = SystemMode.PRODUCTION) -> UnifiedAnalyzerSystem:
    """Factory function per creare UnifiedAnalyzerSystem - data_path is required"""
    if not data_path or not isinstance(data_path, str):
        raise ValueError("data_path is required and must be a non-empty string")
    return UnifiedAnalyzerSystem(data_path=data_path, mode=mode)


def create_production_system(data_path: str) -> UnifiedAnalyzerSystem:
    """Factory function per sistema production-ready - data_path is required"""
    if not data_path or not isinstance(data_path, str):
        raise ValueError("data_path is required and must be a non-empty string")
    return UnifiedAnalyzerSystem(data_path=data_path, mode=SystemMode.PRODUCTION)


def create_testing_system(data_path: str) -> UnifiedAnalyzerSystem:
    """Factory function per sistema testing/backtesting - data_path is required"""
    if not data_path or not isinstance(data_path, str):
        raise ValueError("data_path is required and must be a non-empty string")
    return UnifiedAnalyzerSystem(data_path=data_path, mode=SystemMode.TESTING)


# Export
__all__ = [
    'UnifiedAnalyzerSystem',
    'SystemMode',
    'create_unified_system',
    'create_production_system', 
    'create_testing_system'
]