#!/usr/bin/env python3
"""
Microtest per verificare il collegamento ML Training Logger
============================================================

Test sistematici per identificare dove si interrompe la connessione
tra UnifiedSystem, Analyzer e ML Training Logger.

NESSUNA SIMULAZIONE - Solo componenti reali!
"""

import sys
import os
import asyncio
from datetime import datetime
import json

# Setup paths
sys.path.insert(0, r"C:\ScalpingBOT")
sys.path.insert(0, r"C:\ScalpingBOT\src")

# Safe print
def safe_print(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

class MLLoggerConnectionTest:
    """Test progressivi per verificare connessione ML Logger"""
    
    def __init__(self):
        self.symbol = "USTEC"
        self.test_data_path = "./test_analyzer_data"
        self.results = {}
        
    async def test_1_unified_system_init(self):
        """Test 1: Verifica inizializzazione UnifiedSystem"""
        safe_print("\n🧪 TEST 1: UnifiedSystem Initialization")
        safe_print("="*50)
        
        try:
            from src.Unified_Analyzer_System import UnifiedAnalyzerSystem, create_custom_config, SystemMode, PerformanceProfile
            
            # Crea configurazione minima
            config = create_custom_config(
                system_mode=SystemMode.TESTING,
                performance_profile=PerformanceProfile.RESEARCH,
                asset_symbol=self.symbol,
                learning_phase_enabled=True,
                log_level="NORMAL"
            )
            
            # Inizializza sistema
            unified_system = UnifiedAnalyzerSystem(config)
            safe_print("✅ UnifiedAnalyzerSystem created")
            
            # Avvia sistema
            await unified_system.start()
            safe_print("✅ UnifiedAnalyzerSystem started")
            
            # Verifica analyzer
            if hasattr(unified_system, 'analyzer') and unified_system.analyzer:
                safe_print("✅ Analyzer exists in unified_system")
                self.results['unified_init'] = True
                self.results['analyzer_exists'] = True
                return unified_system
            else:
                safe_print("❌ No analyzer in unified_system")
                self.results['unified_init'] = True
                self.results['analyzer_exists'] = False
                return None
                
        except Exception as e:
            safe_print(f"❌ UnifiedSystem init failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['unified_init'] = False
            return None
            
    async def test_2_ml_logger_components(self, unified_system):
        """Test 2: Verifica componenti ML Logger nell'Analyzer"""
        safe_print("\n🧪 TEST 2: ML Logger Components Check")
        safe_print("="*50)
        
        if not unified_system or not unified_system.analyzer:
            safe_print("⚠️ Skipping - no unified system")
            return False
            
        analyzer = unified_system.analyzer
        
        # Check ML logger attributes
        checks = {
            'ml_logger_active': hasattr(analyzer, 'ml_logger_active'),
            'ml_event_collector': hasattr(analyzer, 'ml_event_collector'),
            'ml_display_manager': hasattr(analyzer, 'ml_display_manager'),
            'ml_storage_manager': hasattr(analyzer, 'ml_storage_manager'),
            '_emit_ml_event': hasattr(analyzer, '_emit_ml_event'),
            '_update_ml_display_metrics': hasattr(analyzer, '_update_ml_display_metrics')
        }
        
        for attr, exists in checks.items():
            if exists:
                value = getattr(analyzer, attr, None)
                if callable(value):
                    safe_print(f"✅ {attr}: method exists")
                else:
                    safe_print(f"✅ {attr}: {value is not None}")
            else:
                safe_print(f"❌ {attr}: NOT FOUND")
                
        # Check if ML logger is active
        if analyzer.ml_logger_active:
            safe_print("✅ ML Logger is ACTIVE")
            
            # Check display manager
            if analyzer.ml_display_manager:
                safe_print(f"✅ ML Display Manager: {type(analyzer.ml_display_manager).__name__}")
                
                # Check if display is running
                if hasattr(analyzer.ml_display_manager, '_stop_event'):
                    is_running = not analyzer.ml_display_manager._stop_event.is_set()
                    safe_print(f"✅ Display Manager running: {is_running}")
            
            self.results['ml_logger_active'] = True
            return True
        else:
            safe_print("❌ ML Logger is NOT ACTIVE")
            self.results['ml_logger_active'] = False
            return False
            
    async def test_3_asset_registration(self, unified_system):
        """Test 3: Verifica registrazione asset USTEC"""
        safe_print("\n🧪 TEST 3: Asset Registration Check")
        safe_print("="*50)
        
        if not unified_system or not unified_system.analyzer:
            safe_print("⚠️ Skipping - no unified system")
            return False
            
        analyzer = unified_system.analyzer
        
        # Check asset_analyzers
        if hasattr(analyzer, 'asset_analyzers'):
            safe_print(f"✅ asset_analyzers exists: {type(analyzer.asset_analyzers).__name__}")
            safe_print(f"   Registered assets: {list(analyzer.asset_analyzers.keys())}")
            
            if self.symbol in analyzer.asset_analyzers:
                safe_print(f"✅ {self.symbol} is registered")
                asset_analyzer = analyzer.asset_analyzers[self.symbol]
                
                # Check asset analyzer properties
                safe_print(f"   Asset analyzer type: {type(asset_analyzer).__name__}")
                
                # Check competitions
                if hasattr(asset_analyzer, 'competitions'):
                    safe_print(f"   Competitions: {len(asset_analyzer.competitions)} model types")
                    
                self.results['asset_registered'] = True
                return True
            else:
                safe_print(f"❌ {self.symbol} NOT registered")
                
                # Try to add it
                safe_print(f"🔧 Attempting to add {self.symbol}...")
                try:
                    asset_analyzer = analyzer.add_asset(self.symbol)
                    if asset_analyzer:
                        safe_print(f"✅ Successfully added {self.symbol}")
                        self.results['asset_registered'] = True
                        return True
                except Exception as e:
                    safe_print(f"❌ Failed to add asset: {e}")
                    
        else:
            safe_print("❌ asset_analyzers NOT FOUND")
            
        self.results['asset_registered'] = False
        return False
        
    async def test_4_process_single_tick(self, unified_system):
        """Test 4: Process un singolo tick e verifica eventi ML"""
        safe_print("\n🧪 TEST 4: Single Tick Processing")
        safe_print("="*50)
        
        if not unified_system:
            safe_print("⚠️ Skipping - no unified system")
            return False
            
        # Load one tick from real data
        data_file = f"{self.test_data_path}/backtest_USTEC_20250516_20250715.jsonl"
        
        if not os.path.exists(data_file):
            safe_print(f"❌ Data file not found: {data_file}")
            return False
            
        try:
            # Read first data line (skip header)
            with open(data_file, 'r', encoding='utf-8') as f:
                header = f.readline()  # Skip header
                tick_line = f.readline()  # First data
                
            if not tick_line:
                safe_print("❌ No tick data in file")
                return False
                
            tick_data = json.loads(tick_line.strip())
            safe_print(f"✅ Loaded tick: {tick_data['timestamp']} @ {tick_data['bid']}/{tick_data['ask']}")
            
            # Convert to datetime
            timestamp = datetime.strptime(tick_data['timestamp'], '%Y.%m.%d %H:%M:%S')
            
            # Process tick
            safe_print("🔄 Processing tick...")
            result = await unified_system.process_tick(
                timestamp=timestamp,
                price=(tick_data['bid'] + tick_data['ask']) / 2,
                volume=tick_data.get('volume', 1),
                bid=tick_data['bid'],
                ask=tick_data['ask']
            )
            
            if result:
                safe_print(f"✅ Tick processed: {result.get('status', 'unknown')}")
                
                # Check if ML events were generated
                if unified_system.analyzer and hasattr(unified_system.analyzer, 'get_all_events'):
                    events = unified_system.analyzer.get_all_events()
                    if events:
                        for event_type, event_list in events.items():
                            if event_list:
                                safe_print(f"   📊 {event_type}: {len(event_list)} events")
                                
                # Force display update
                if (unified_system.analyzer and 
                    hasattr(unified_system.analyzer, '_update_ml_display_metrics')):
                    unified_system.analyzer._update_ml_display_metrics(self.symbol)
                    safe_print("✅ Display metrics updated")
                    
                self.results['tick_processed'] = True
                return True
            else:
                safe_print("❌ Tick processing returned None")
                
        except Exception as e:
            safe_print(f"❌ Error processing tick: {e}")
            import traceback
            traceback.print_exc()
            
        self.results['tick_processed'] = False
        return False
        
    async def test_5_batch_processing(self, unified_system):
        """Test 5: Process batch di tick e verifica eventi ML"""
        safe_print("\n🧪 TEST 5: Batch Processing (100 ticks)")
        safe_print("="*50)
        
        if not unified_system:
            safe_print("⚠️ Skipping - no unified system")
            return False
            
        # Load batch of ticks
        data_file = f"{self.test_data_path}/backtest_USTEC_20250516_20250715.jsonl"
        
        try:
            ticks = []
            with open(data_file, 'r', encoding='utf-8') as f:
                f.readline()  # Skip header
                
                # Load 100 ticks
                for i in range(100):
                    line = f.readline()
                    if not line:
                        break
                    tick_data = json.loads(line.strip())
                    ticks.append(tick_data)
                    
            safe_print(f"✅ Loaded {len(ticks)} ticks")
            
            # Convert dicts to objects for process_batch
            class TickObject:
                def __init__(self, data):
                    self.timestamp = datetime.strptime(data['timestamp'], '%Y.%m.%d %H:%M:%S')
                    self.price = data.get('last', (data['bid'] + data['ask']) / 2)
                    self.volume = data.get('volume', 1)
                    self.bid = data.get('bid')
                    self.ask = data.get('ask')
                    
            tick_objects = [TickObject(tick) for tick in ticks]
            
            # Process batch
            safe_print("🔄 Processing batch...")
            processed, analyzed = await unified_system.process_batch(tick_objects)
            
            safe_print(f"✅ Batch processed: {processed} ticks, {analyzed} analyses")
            
            # Check ML display
            if (unified_system.analyzer and 
                unified_system.analyzer.ml_logger_active and
                unified_system.analyzer.ml_display_manager):
                
                # Force metrics update
                unified_system.analyzer._update_ml_display_metrics(self.symbol)
                
                # Emit diagnostic event
                unified_system.analyzer._emit_ml_event('diagnostic', {
                    'event_type': 'test_batch_complete',
                    'symbol': self.symbol,
                    'ticks_processed': processed,
                    'analyses': analyzed
                })
                
                safe_print("✅ ML events emitted")
                
            self.results['batch_processed'] = True
            return True
            
        except Exception as e:
            safe_print(f"❌ Batch processing error: {e}")
            import traceback
            traceback.print_exc()
            
        self.results['batch_processed'] = False
        return False
        
    async def run_all_tests(self):
        """Esegue tutti i test in sequenza"""
        safe_print("\n🚀 Starting ML Logger Connection Tests")
        safe_print("="*70)
        
        # Test 1: Initialize system
        unified_system = await self.test_1_unified_system_init()
        
        if unified_system:
            # Test 2: Check ML logger components
            ml_active = await self.test_2_ml_logger_components(unified_system)
            
            # Test 3: Check asset registration
            asset_ok = await self.test_3_asset_registration(unified_system)
            
            # Test 4: Process single tick
            tick_ok = await self.test_4_process_single_tick(unified_system)
            
            # Test 5: Process batch
            batch_ok = await self.test_5_batch_processing(unified_system)
            
            # Wait a bit for display updates
            await asyncio.sleep(2)
            
            # Stop system
            try:
                await unified_system.stop()
                safe_print("✅ System stopped cleanly")
            except:
                pass
                
        # Summary
        safe_print("\n📊 TEST SUMMARY")
        safe_print("="*70)
        for test, result in self.results.items():
            status = "✅" if result else "❌"
            safe_print(f"{status} {test}: {result}")
            
        # Diagnosis
        safe_print("\n🔍 DIAGNOSIS")
        safe_print("="*70)
        
        if not self.results.get('unified_init'):
            safe_print("❌ PROBLEM: UnifiedSystem initialization failed")
        elif not self.results.get('analyzer_exists'):
            safe_print("❌ PROBLEM: Analyzer not created in UnifiedSystem")
        elif not self.results.get('ml_logger_active'):
            safe_print("❌ PROBLEM: ML Logger not active in Analyzer")
        elif not self.results.get('asset_registered'):
            safe_print("❌ PROBLEM: Asset USTEC not registered in Analyzer")
        elif not self.results.get('tick_processed'):
            safe_print("❌ PROBLEM: Tick processing failed")
        elif not self.results.get('batch_processed'):
            safe_print("❌ PROBLEM: Batch processing failed")
        else:
            safe_print("✅ All systems connected correctly!")
            safe_print("⚠️ If dashboard still shows zeros, check:")
            safe_print("   1. Display Manager event processing")
            safe_print("   2. Rate limiting in event emission")
            safe_print("   3. Dashboard rendering loop")


async def main():
    """Entry point"""
    tester = MLLoggerConnectionTest()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())