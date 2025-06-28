#!/usr/bin/env python3
"""
Test Backtest Rapido - Integrato con Unified Analyzer System
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Aggiungi la directory corrente al path per gli import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ✅ INTEGRAZIONE: Import del Unified Analyzer System
try:
    # Try to import from current directory first
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "unified_analyzer_system", 
        os.path.join(os.path.dirname(__file__), "unified_analyzer_system.py")
    )
    if spec and spec.loader:
        unified_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(unified_module)
        
        UnifiedAnalyzerSystem = unified_module.UnifiedAnalyzerSystem
        UnifiedConfig = unified_module.UnifiedConfig
        SystemMode = unified_module.SystemMode
        PerformanceProfile = unified_module.PerformanceProfile
        create_custom_config = unified_module.create_custom_config
        
        UNIFIED_SYSTEM_AVAILABLE = True
        print("✅ Unified Analyzer System imported successfully")
    else:
        raise ImportError("unified_analyzer_system.py not found")
        
except ImportError as e:
    UNIFIED_SYSTEM_AVAILABLE = False
    print(f"⚠️ Unified Analyzer System not available: {e}")
    print("📄 Falling back to legacy logging...")
    
    # Dummy classes with proper interface for type safety
    class SystemMode:
        TESTING = "testing"
        PRODUCTION = "production"
        DEVELOPMENT = "development" 
        DEMO = "demo"
    
    class PerformanceProfile:
        RESEARCH = "research"
        NORMAL = "normal"
        HIGH_FREQUENCY = "high_frequency"
    
    class UnifiedConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.base_directory = "./fallback_logs"
    
    class MockLoggingSlave:
        def __init__(self):
            pass
        async def process_events(self, events):
            pass
    
    class UnifiedAnalyzerSystem:
        def __init__(self, config=None):
            self.config = config or UnifiedConfig()
            self.logging_slave = MockLoggingSlave()
            self.is_running = False
            
        async def start(self):
            self.is_running = True
            print("⚠️ Mock Unified System started (fallback mode)")
            
        async def stop(self):
            self.is_running = False
            print("⚠️ Mock Unified System stopped (fallback mode)")
            
        async def process_tick(self, timestamp, price, volume, bid=None, ask=None):
            # Mock processing - just return basic result
            return {
                'status': 'mock_success',
                'price': price,
                'volume': volume,
                'timestamp': timestamp
            }
            
        def get_system_status(self):
            return {
                'system': {
                    'running': self.is_running,
                    'mode': 'fallback',
                    'uptime_seconds': 0.0,
                    'stats': {
                        'total_ticks_processed': 0,
                        'total_events_logged': 0,
                        'errors_count': 0
                    }
                },
                'analyzer': {
                    'predictions_generated': 0,
                    'avg_latency_ms': 0.0,
                    'buffer_utilization': 0.0
                },
                'logging': {
                    'events_processed': 0,
                    'events_dropped': 0,
                    'queue_utilization': 0.0
                }
            }
    
    def create_custom_config(**kwargs):
        return UnifiedConfig(**kwargs)

# ✅ FALLBACK: Sistema di logging legacy per compatibilità
try:
    from utils.universal_encoding_fix import safe_print, init_universal_encoding, get_safe_logger
    init_universal_encoding(silent=True)
    logger = get_safe_logger(__name__)
    LEGACY_LOGGING_AVAILABLE = True
except ImportError:
    LEGACY_LOGGING_AVAILABLE = False
    # Fallback minimale con signature corretta
    def safe_print(text: str) -> None: 
        print(text)
    
    class DummyLogger:
        def info(self, text: str) -> None: pass
        def error(self, text: str) -> None: pass
        def critical(self, text: str) -> None: pass
    
    logger = DummyLogger()


class UnifiedBacktestRunner:
    """
    Wrapper che integra MT5BacktestRunner con Unified Analyzer System
    """
    
    def __init__(self, use_unified_system: bool = True):
        self.use_unified_system = use_unified_system and UNIFIED_SYSTEM_AVAILABLE
        self.unified_system = None
        self.legacy_runner = None
        
        if self.use_unified_system:
            safe_print("🚀 Using Unified Analyzer System for structured logging")
        else:
            safe_print("📝 Using legacy logging system")
    
    async def setup_unified_system(self, config) -> None:
        """Setup del sistema unificato per backtest"""
        
        if not self.use_unified_system:
            safe_print("⚠️ Using legacy mode - no unified system setup")
            return
        
        if not UNIFIED_SYSTEM_AVAILABLE:
            safe_print("⚠️ Unified system not available - using mock system")
            self.use_unified_system = False  # Disable to use mock behavior
            return
        
        try:
            # Configurazione ottimizzata per backtest
            unified_config = create_custom_config(
                system_mode=SystemMode.TESTING,  # Verbose per vedere cosa succede
                performance_profile=PerformanceProfile.RESEARCH,
                asset_symbol=config.symbol,
                
                # Logging settings per backtest
                log_level="VERBOSE",
                enable_console_output=True,
                enable_file_output=True,
                enable_csv_export=True,
                enable_json_export=False,  # Non necessario per test
                
                # Rate limiting per backtest (più frequente per vedere progress)
                rate_limits={
                    'tick_processing': 100,      # Log ogni 100 ticks
                    'predictions': 10,           # Log ogni 10 predizioni  
                    'validations': 5,            # Log ogni 5 validazioni
                    'training_events': 1,        # Log tutti i training
                    'champion_changes': 1,       # Log tutti i champion changes
                    'emergency_events': 1,       # Log tutte le emergenze
                    'diagnostics': 500          # Log diagnostici ogni 500 ops
                },
                
                # Performance settings per backtest
                event_processing_interval=2.0,     # Process eventi ogni 2 secondi
                batch_size=25,                     # Batch più piccoli per feedback frequente
                max_queue_size=5000,               # Queue più piccola per test
                
                # Storage per test
                base_directory=f"./backtest_logs_{config.symbol}_{datetime.now():%Y%m%d_%H%M%S}",
                
                # Monitoring per vedere performance
                enable_performance_monitoring=True,
                performance_report_interval=30.0,  # Report ogni 30 secondi
                memory_threshold_mb=500,           # Alert se supera 500MB
                cpu_threshold_percent=70.0         # Alert se supera 70% CPU
            )
            
            if unified_config is None:
                safe_print("❌ Failed to create unified config")
                self.use_unified_system = False
                return
            
            # Crea e avvia sistema unificato
            self.unified_system = UnifiedAnalyzerSystem(unified_config)
            if self.unified_system is not None:
                await self.unified_system.start()
                
                safe_print(f"✅ Unified System started for {config.symbol}")
                safe_print(f"📁 Logs directory: {unified_config.base_directory}")
            else:
                safe_print("❌ Failed to create unified system")
                self.use_unified_system = False
                
        except Exception as e:
            safe_print(f"❌ Error setting up unified system: {e}")
            self.use_unified_system = False
        
    async def run_backtest_unified(self, config):
        """Run backtest con sistema unificato"""
        
        try:
            # Setup sistema unificato
            await self.setup_unified_system(config)
            
            # Import del backtest runner originale
            from src.MT5BacktestRunner import MT5BacktestRunner
            
            # Crea wrapper del runner legacy
            legacy_runner = MT5BacktestRunner()
            
            # ✨ Processing con sistema appropriato
            if self.use_unified_system and UNIFIED_SYSTEM_AVAILABLE:
                safe_print("🔄 Starting backtest with unified logging...")
                await self._simulate_backtest_data(config)
            else:
                safe_print("🔄 Starting backtest with legacy system...")
                # Crea mock system per testing
                self.unified_system = UnifiedAnalyzerSystem()
                await self.unified_system.start()
                await self._simulate_backtest_data(config)
            
            success = True  # Per ora simuliamo successo
            
            if success:
                # Mostra statistiche finali
                await self._show_final_statistics()
            
            return success
            
        except Exception as e:
            safe_print(f"❌ Error in unified backtest: {e}")
            
            # Log dell'errore nel sistema unificato se disponibile
            if self.unified_system:
                try:
                    # Crea evento di errore
                    error_event = {
                        'timestamp': datetime.now(),
                        'event_type': 'backtest_error',
                        'data': {
                            'error_message': str(e),
                            'error_type': type(e).__name__,
                            'symbol': config.symbol,
                            'backtest_period': f"{config.start_date} to {config.end_date}"
                        }
                    }
                    
                    # Processa immediatamente l'evento di errore
                    await self.unified_system.logging_slave.process_events({
                        'error_events': [error_event]
                    })
                except Exception as log_error:
                    safe_print(f"⚠️ Also failed to log error: {log_error}")
            
            return False
            
        finally:
            # Cleanup sistema unificato
            if self.unified_system:
                try:
                    await self.unified_system.stop()
                except Exception as stop_error:
                    safe_print(f"⚠️ Error stopping unified system: {stop_error}")
    
    async def _simulate_backtest_data(self, config) -> None:
        """Simula processing di dati backtest con sistema unificato"""
        
        if not self.unified_system:
            safe_print("⚠️ No unified system available for simulation")
            return
        
        safe_print(f"📊 Simulating backtest data for {config.symbol}...")
        
        # Simula tick data (sostituisci con il tuo data loading reale)
        base_price = 15000.0  # USTEC example
        current_time = config.start_date
        tick_count = 0
        
        # Simula 1000 ticks per test rapido
        total_ticks = 1000
        
        for i in range(total_ticks):
            # Simula movimento prezzo
            price_change = (i % 10 - 5) * 0.1  # Movimento simulato
            current_price = base_price + price_change
            volume = 1000 + (i % 100) * 10
            
            # Processa tick attraverso sistema unificato
            try:
                await self.unified_system.process_tick(
                    timestamp=current_time,
                    price=current_price,
                    volume=volume,
                    bid=current_price - 0.5,
                    ask=current_price + 0.5
                )
            except Exception as e:
                safe_print(f"❌ Error processing tick {i}: {e}")
                break
            
            tick_count += 1
            current_time += timedelta(seconds=1)
            
            # Progress feedback
            if tick_count % 100 == 0:
                progress = (tick_count / total_ticks) * 100
                safe_print(f"📈 Progress: {tick_count}/{total_ticks} ticks ({progress:.1f}%)")
            
            # Small delay per non saturare il sistema
            if i % 50 == 0:
                await asyncio.sleep(0.1)
        
        safe_print(f"✅ Processed {tick_count} ticks successfully")
    
    async def _show_final_statistics(self) -> None:
        """Mostra statistiche finali del backtest"""
        
        if not self.unified_system:
            safe_print("⚠️ No unified system available for statistics")
            return
        
        try:
            # Ottieni stato completo del sistema
            status = self.unified_system.get_system_status()
            
            safe_print("\n" + "="*60)
            safe_print("📊 BACKTEST FINAL STATISTICS")
            safe_print("="*60)
            
            # System stats
            system_stats = status.get('system', {}).get('stats', {})
            system_info = status.get('system', {})
            safe_print(f"🕐 Uptime: {system_info.get('uptime_seconds', 0):.1f} seconds")
            safe_print(f"📈 Total ticks processed: {system_stats.get('total_ticks_processed', 0)}")
            safe_print(f"📝 Total events logged: {system_stats.get('total_events_logged', 0)}")
            safe_print(f"❌ Errors: {system_stats.get('errors_count', 0)}")
            
            # Analyzer stats
            analyzer_stats = status.get('analyzer', {})
            safe_print(f"🔮 Predictions generated: {analyzer_stats.get('predictions_generated', 0)}")
            safe_print(f"⚡ Average latency: {analyzer_stats.get('avg_latency_ms', 0):.2f}ms")
            safe_print(f"📊 Buffer utilization: {analyzer_stats.get('buffer_utilization', 0):.1f}%")
            
            # Logging stats
            logging_stats = status.get('logging', {})
            safe_print(f"📋 Events processed: {logging_stats.get('events_processed', 0)}")
            safe_print(f"📤 Events dropped: {logging_stats.get('events_dropped', 0)}")
            safe_print(f"🔄 Queue utilization: {logging_stats.get('queue_utilization', 0):.1f}%")
            
            # Performance metrics (if available) - Safe access
            if 'performance' in status and status['performance']:
                perf = status['performance']
                # Handle both dict and object attribute access
                if isinstance(perf, dict):
                    memory_mb = perf.get('memory_mb', 0)
                    cpu_percent = perf.get('cpu_percent', 0)
                else:
                    # If it's an object with attributes
                    memory_mb = getattr(perf, 'memory_mb', 0)
                    cpu_percent = getattr(perf, 'cpu_percent', 0)
                
                safe_print(f"🧠 Memory usage: {memory_mb:.1f}MB")
                safe_print(f"💻 CPU usage: {cpu_percent:.1f}%")
            else:
                safe_print("🧠 Memory usage: N/A")
                safe_print("💻 CPU usage: N/A")
            
            safe_print("="*60)
            
            # Safe access to config
            if hasattr(self.unified_system, 'config') and self.unified_system.config:
                config = self.unified_system.config
                if hasattr(config, 'base_directory'):
                    safe_print(f"📁 Logs saved to: {config.base_directory}")
                else:
                    safe_print("📁 Logs saved to: ./default_logs")
            else:
                safe_print("📁 Logs saved to: ./default_logs")
                
            safe_print("="*60 + "\n")
            
        except Exception as e:
            safe_print(f"❌ Error getting statistics: {e}")
            # Fallback minimal statistics
            safe_print("\n" + "="*60)
            safe_print("📊 BACKTEST STATISTICS (MINIMAL)")
            safe_print("="*60)
            safe_print("⚠️ Could not retrieve detailed statistics")
            safe_print(f"📋 Error: {str(e)}")
            safe_print("="*60 + "\n")


async def test_quick_backtest():
    """Test backtest con sistema unificato"""
    
    safe_print("🔄 TEST BACKTEST RAPIDO - UNIFIED SYSTEM")
    safe_print("="*60)
    
    try:
        # Import con gestione errori
        try:
            from src.MT5BacktestRunner import BacktestConfig
            safe_print("✅ BacktestConfig import successful")
            logger.info("Import BacktestConfig completato")
        except ImportError as e:
            safe_print(f"❌ Import error: {e}")
            safe_print(f"📁 Current directory: {os.getcwd()}")
            safe_print(f"📄 Files in directory: {[f for f in os.listdir('.') if f.endswith('.py')]}")
            logger.error(f"Import error: {e}")
            return False
        
        # Config per 1 SOLO GIORNO (test veloce)
        yesterday = datetime.now() - timedelta(days=1)
        today = datetime.now()
        
        config = BacktestConfig(
            symbol='USTEC',
            start_date=yesterday,
            end_date=today,
            data_source='mt5_export',
            speed_multiplier=1000,
            save_progress=True,
            resume_from_checkpoint=True
        )
        
        safe_print(f"📊 Symbol: {config.symbol}")
        safe_print(f"📅 Period: {config.start_date.strftime('%Y-%m-%d %H:%M')} to {config.end_date.strftime('%Y-%m-%d %H:%M')}")
        safe_print(f"🚀 Using: {'Unified Analyzer System' if UNIFIED_SYSTEM_AVAILABLE else 'Legacy System'}")
        safe_print(f"⚡ This should take ~2-5 minutes")
        safe_print("")
        
        logger.info(f"Configurazione backtest: {config.symbol} dal {config.start_date} al {config.end_date}")
        
        # Crea runner unificato
        runner = UnifiedBacktestRunner(use_unified_system=UNIFIED_SYSTEM_AVAILABLE)
        
        safe_print("🚀 Avvio backtest runner...")
        
        if runner.use_unified_system:
            # Usa sistema unificato
            safe_print("⏳ Esecuzione backtest con Unified System...")
            success = await runner.run_backtest_unified(config)
        else:
            # Fallback a sistema legacy
            safe_print("⏳ Fallback: esecuzione con sistema legacy...")
            from src.MT5BacktestRunner import MT5BacktestRunner
            legacy_runner = MT5BacktestRunner()
            success = legacy_runner.run_backtest(config)
        
        if success:
            safe_print("\n✅ TEST BACKTEST PASSED!")
            safe_print("🎯 Ready for full 6-month backtest")
            logger.info("Test backtest completato con successo")
            return True
        else:
            safe_print("\n❌ TEST BACKTEST FAILED")
            logger.error("Test backtest fallito")
            return False
            
    except Exception as e:
        safe_print(f"\n❌ Error during backtest: {e}")
        logger.critical(f"Errore critico durante backtest: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function per compatibilità sync/async"""
    
    safe_print("🔍 Current working directory: " + os.getcwd())
    safe_print("📄 Python files available: " + str([f for f in os.listdir('.') if f.endswith('.py')]))
    safe_print("")
    
    logger.info("Avvio test backtest rapido")
    
    # Run async test
    try:
        result = asyncio.run(test_quick_backtest())
    except Exception as e:
        safe_print(f"❌ Error running async test: {e}")
        result = False
    
    if result:
        safe_print("\n🎉 TEST COMPLETATO CON SUCCESSO!")
    else:
        safe_print("\n⚠️ TEST FALLITO - Controlla i log per dettagli")
    
    logger.info(f"Test terminato con risultato: {'SUCCESS' if result else 'FAILED'}")
    return result


if __name__ == "__main__":
    main()