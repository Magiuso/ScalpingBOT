#!/usr/bin/env python3
"""
Test Direct Tick Processing - Test diretto 100K tick processing
===============================================================

Test completamente indipendente per verificare dashboard durante processing.
NESSUN collegamento con test_backtest - logica diretta.
"""

import sys
import os
import asyncio
from datetime import datetime
import json
import time

# Setup paths
sys.path.insert(0, r"C:\ScalpingBOT")
sys.path.insert(0, r"C:\ScalpingBOT\src")

def safe_print(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")

class DirectTickProcessingTest:
    """Test diretto per processing tick senza dipendenze"""
    
    def __init__(self):
        self.symbol = "USTEC"
        self.test_data_path = "./test_analyzer_data"
        
    async def test_direct_processing(self):
        """Test processing diretto con 100K ticks"""
        safe_print("🚀 DIRECT TICK PROCESSING TEST")
        safe_print("=" * 70)
        
        try:
            # Apply encoding fixes first
            safe_print("🔧 Applying encoding fixes...")
            try:
                sys.path.insert(0, r"C:\ScalpingBOT\utils")
                from universal_encoding_fix import apply_all_fixes
                apply_all_fixes()
                safe_print("✅ Encoding fixes applied")
            except Exception as e:
                safe_print(f"⚠️ Encoding fix warning: {e}")
            
            # 1. Create UnifiedAnalyzerSystem directly
            safe_print("🔧 Creating UnifiedAnalyzerSystem...")
            from src.Unified_Analyzer_System import UnifiedAnalyzerSystem, create_custom_config, SystemMode, PerformanceProfile
            
            config = create_custom_config(
                system_mode=SystemMode.TESTING,
                performance_profile=PerformanceProfile.RESEARCH,
                asset_symbol=self.symbol,
                learning_phase_enabled=True,
                log_level="NORMAL"
            )
            
            unified_system = UnifiedAnalyzerSystem(config)
            await unified_system.start()
            safe_print("✅ UnifiedAnalyzerSystem started")
            
            # 2. Get analyzer and display manager
            analyzer = unified_system.analyzer
            if not analyzer:
                safe_print("❌ No analyzer available")
                return False
                
            safe_print(f"✅ Analyzer available: {type(analyzer).__name__}")
            
            # Check ML logger status
            ml_active = getattr(analyzer, 'ml_logger_active', False)
            safe_print(f"📊 ML Logger active: {ml_active}")
            
            if not ml_active:
                safe_print("❌ ML Logger not active - cannot test dashboard")
                await unified_system.stop()
                return False
                
            display_manager = getattr(analyzer, 'ml_display_manager', None)
            if not display_manager:
                safe_print("❌ No ML display manager")
                await unified_system.stop()
                return False
                
            safe_print(f"✅ Display Manager: {type(display_manager).__name__}")
            
            # 3. Check initial dashboard state
            safe_print("\n📊 INITIAL DASHBOARD STATE:")
            if hasattr(display_manager, 'current_metrics'):
                metrics = display_manager.current_metrics
                safe_print(f"   Ticks: {getattr(metrics, 'ticks_processed', 0)}")
                safe_print(f"   Rate: {getattr(metrics, 'processing_rate', 'N/A')}")
                safe_print(f"   Metrics object: {type(metrics).__name__}")
            else:
                safe_print("   No current_metrics attribute")
            
            # 4. Load 1K ticks for quick test
            safe_print(f"\n📂 Loading 1K ticks from data file...")
            data_file = f"{self.test_data_path}/backtest_USTEC_20250516_20250715.jsonl"
            
            ticks_data = []
            with open(data_file, 'r', encoding='utf-8') as f:
                f.readline()  # Skip header
                
                for i in range(1000):  # Start with just 1K ticks to see if it completes
                    line = f.readline()
                    if not line:
                        break
                    try:
                        ticks_data.append(json.loads(line.strip()))
                    except:
                        continue
            
            safe_print(f"✅ Loaded {len(ticks_data):,} ticks")
            
            # 5. Convert to tick objects
            safe_print("🔄 Converting to tick objects...")
            
            class TickObject:
                def __init__(self, data):
                    self.timestamp = datetime.strptime(data['timestamp'], '%Y.%m.%d %H:%M:%S')
                    self.price = data.get('last', (data['bid'] + data['ask']) / 2)
                    self.volume = data.get('volume', 1)
                    self.bid = data.get('bid')
                    self.ask = data.get('ask')
            
            tick_objects = [TickObject(tick) for tick in ticks_data]
            safe_print(f"✅ Converted {len(tick_objects):,} tick objects")
            
            # 6. Process ticks and monitor dashboard
            safe_print("\n🚀 STARTING TICK PROCESSING")
            safe_print("=" * 60)
            
            start_time = time.time()
            
            # Check if process_batch exists
            if not hasattr(unified_system, 'process_batch'):
                safe_print("❌ process_batch method not available on unified_system")
                await unified_system.stop()
                return False
            
            # Process the batch
            safe_print(f"🔄 Processing {len(tick_objects):,} ticks...")
            try:
                processed_count, analysis_count = await unified_system.process_batch(tick_objects)
                safe_print(f"✅ Processed: {processed_count:,} ticks, {analysis_count:,} analyses")
                
                # Force display update
                if hasattr(analyzer, '_update_ml_display_metrics'):
                    safe_print("🔄 Calling _update_ml_display_metrics...")
                    analyzer._update_ml_display_metrics(self.symbol)
                    safe_print("✅ Display metrics update called")
                else:
                    safe_print("❌ _update_ml_display_metrics not available")
                
                # Force global stats update
                if hasattr(analyzer, '_update_global_stats'):
                    safe_print("🔄 Calling _update_global_stats...")
                    analyzer._update_global_stats()
                    safe_print("✅ Global stats update called")
                else:
                    safe_print("❌ _update_global_stats not available")
                
            except Exception as e:
                safe_print(f"❌ Error during processing: {e}")
                import traceback
                traceback.print_exc()
            
            # 7. Check dashboard after processing
            safe_print("\n📊 DASHBOARD STATE AFTER PROCESSING:")
            safe_print("=" * 50)
            
            elapsed = time.time() - start_time
            
            if hasattr(display_manager, 'current_metrics'):
                metrics = display_manager.current_metrics
                final_ticks = getattr(metrics, 'ticks_processed', 0)
                final_rate = getattr(metrics, 'processing_rate', 'N/A')
                
                safe_print(f"   Dashboard Ticks: {final_ticks:,}")
                safe_print(f"   Dashboard Rate: {final_rate}")
                safe_print(f"   Processing Time: {elapsed:.1f} seconds")
                
                if final_ticks > 0:
                    safe_print("✅ DASHBOARD SHOWS TICK COUNT!")
                    result = True
                else:
                    safe_print("❌ DASHBOARD STILL SHOWS ZERO TICKS")
                    result = False
                    
                # Debug: show all metrics attributes
                safe_print("\n🔍 ALL DASHBOARD METRICS:")
                for attr in dir(metrics):
                    if not attr.startswith('_'):
                        value = getattr(metrics, attr, 'N/A')
                        safe_print(f"   {attr}: {value}")
            else:
                safe_print("❌ No current_metrics attribute found")
                result = False
            
            # 8. Check where tick count is actually stored
            safe_print("\n🔍 DEBUGGING TICK COUNT SOURCES:")
            
            # Check analyzer performance stats
            if hasattr(analyzer, '_performance_stats'):
                stats = analyzer._performance_stats
                safe_print(f"   Analyzer _performance_stats: {stats}")
            else:
                safe_print("   No _performance_stats in analyzer")
            
            # Check asset analyzer
            if hasattr(analyzer, 'asset_analyzers') and self.symbol in analyzer.asset_analyzers:
                asset_analyzer = analyzer.asset_analyzers[self.symbol]
                if hasattr(asset_analyzer, 'tick_count'):
                    safe_print(f"   Asset analyzer tick_count: {asset_analyzer.tick_count}")
                else:
                    safe_print("   No tick_count in asset analyzer")
            else:
                safe_print("   No asset analyzer found")
            
            # Check unified system status
            if hasattr(unified_system, 'get_system_status'):
                try:
                    status = unified_system.get_system_status()
                    safe_print(f"   Unified system status: {status}")
                except Exception as e:
                    safe_print(f"   Error getting system status: {e}")
            else:
                safe_print("   No get_system_status method")
            
            await unified_system.stop()
            return result
            
        except Exception as e:
            safe_print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Entry point"""
    tester = DirectTickProcessingTest()
    result = await tester.test_direct_processing()
    
    safe_print("\n" + "=" * 70)
    safe_print("📊 TEST CONCLUSION:")
    safe_print("=" * 70)
    
    if result:
        safe_print("✅ DASHBOARD UPDATES CORRECTLY!")
    else:
        safe_print("❌ DASHBOARD DOES NOT UPDATE")
        safe_print("🔍 The issue is in the connection between:")
        safe_print("   - Tick processing (unified_system.process_batch)")
        safe_print("   - Display metrics update (_update_ml_display_metrics)")
        safe_print("   - Dashboard display (_metrics)")


if __name__ == "__main__":
    asyncio.run(main())