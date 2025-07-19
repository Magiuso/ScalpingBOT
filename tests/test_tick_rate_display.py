#!/usr/bin/env python3
"""
Test Tick Rate Display - Verifica aggiornamento in tempo reale
==============================================================

Test per verificare se la riga "Ticks: X | Rate: Y/s" si aggiorna.
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

class TickRateDisplayTest:
    """Test per verificare aggiornamento tick rate"""
    
    def __init__(self):
        self.symbol = "USTEC"
        self.test_data_path = "./test_analyzer_data"
        
    async def test_tick_rate_updates(self):
        """Test aggiornamento tick rate durante processing dei primi 500K"""
        safe_print("\n🧪 TEST: Tick Rate During 500K Processing")
        safe_print("="*50)
        
        try:
            # Importa il test_backtest per usare la stessa logica
            sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
            from test_backtest import MLLearningTestSuite
            
            # Crea test instance
            test_instance = MLLearningTestSuite()
            
            safe_print("🔄 Setting up ML Learning Test (same as test_backtest)...")
            
            # Setup prerequisites
            await test_instance._test_setup_and_prerequisites()
            safe_print("✅ Prerequisites setup complete")
            
            # Start ML training logger integration
            await test_instance._test_ml_training_logger_integration()
            safe_print("✅ ML Training Logger integration complete")
            
            # Get references
            analyzer = test_instance.analyzer
            unified_system = test_instance.unified_system
            
            if not analyzer or not analyzer.ml_display_manager:
                safe_print("❌ No ML display manager available")
                return False
                
            display_manager = analyzer.ml_display_manager
            safe_print("✅ Display Manager found")
            
            # Check initial state
            safe_print("\n📊 Initial State:")
            if hasattr(display_manager, '_metrics'):
                ticks = display_manager._metrics.get('ticks_processed', 0)
                rate = display_manager._metrics.get('processing_rate', 'N/A')
                safe_print(f"   Ticks: {ticks} | Rate: {rate}")
            
            # BYPASS data loading - load 100K ticks directly for processing
            safe_print("\n🔄 BYPASSING memory threshold - loading 100K ticks directly...")
            
            data_file = f"{self.test_data_path}/backtest_USTEC_20250516_20250715.jsonl"
            
            # Load 100K ticks into memory
            ticks_data = []
            with open(data_file, 'r', encoding='utf-8') as f:
                f.readline()  # Skip header
                
                for i in range(100000):  # Load exactly 100K ticks
                    line = f.readline()
                    if not line:
                        break
                    try:
                        ticks_data.append(json.loads(line.strip()))
                    except:
                        continue
            
            safe_print(f"✅ Loaded {len(ticks_data):,} ticks for processing")
            
            # START PROCESSING using test_backtest logic
            safe_print("\n🚀 STARTING PROCESSING - Following test_backtest logic")
            safe_print("=" * 60)
            
            start_time = time.time()
            last_tick_count = 0
            processed_total = 0
            
            # Process in chunks like test_backtest does (500K chunks)
            chunk_size = 500000  # Same as test_backtest
            
            # Convert ticks to objects (same as test_backtest _convert_tick_data)
            tick_objects = []
            for tick_data in ticks_data:
                class TickObject:
                    def __init__(self, data):
                        self.timestamp = datetime.strptime(data['timestamp'], '%Y.%m.%d %H:%M:%S')
                        self.price = data.get('last', (data['bid'] + data['ask']) / 2)
                        self.volume = data.get('volume', 1)
                        self.bid = data.get('bid')
                        self.ask = data.get('ask')
                
                tick_objects.append(TickObject(tick_data))
            
            safe_print(f"✅ Converted {len(tick_objects):,} ticks to objects")
            
            # Process the chunk using unified_system.process_batch (same as test_backtest)
            safe_print(f"\n🔄 Processing chunk of {len(tick_objects):,} ticks...")
            
            try:
                if hasattr(unified_system, 'process_batch'):
                    processed_count, analysis_count = await unified_system.process_batch(tick_objects)
                    processed_total += processed_count
                    
                    safe_print(f"✅ Chunk processed: {processed_count:,} ticks, {analysis_count:,} analyses")
                    
                    # Force update ML display (same as test_backtest)
                    if (analyzer and hasattr(analyzer, '_update_ml_display_metrics')):
                        analyzer._update_ml_display_metrics(self.symbol)
                        safe_print("🔄 Display metrics updated")
                        
                        # Also update global stats (same as test_backtest)
                        if hasattr(analyzer, '_update_global_stats'):
                            analyzer._update_global_stats()
                            safe_print("🔄 Global stats updated")
                    
                    # Emit ML event (same as test_backtest)
                    if (analyzer and hasattr(analyzer, '_emit_ml_event')):
                        analyzer._emit_ml_event('diagnostic', {
                            'event_type': 'chunk_completed',
                            'chunk_number': 1,
                            'processed_ticks': processed_count,
                            'total_processed': processed_total,
                            'symbol': self.symbol,
                            'timestamp': datetime.now()
                        })
                        safe_print("🔄 ML event emitted")
                        
                else:
                    safe_print("❌ process_batch method not available")
                    
            except Exception as e:
                safe_print(f"❌ Error processing chunk: {e}")
                import traceback
                traceback.print_exc()
            
            # Monitor dashboard for a few seconds to see updates
            safe_print("\n📊 MONITORING DASHBOARD AFTER PROCESSING:")
            safe_print("=" * 50)
            
            for i in range(10):  # Monitor for 10 seconds
                # Check display metrics
                if hasattr(display_manager, '_metrics'):
                    metrics = display_manager._metrics
                    ticks = metrics.get('ticks_processed', 0)
                    rate = metrics.get('processing_rate', 'N/A')
                    
                    elapsed = time.time() - start_time
                    safe_print(f"   [{elapsed:6.1f}s] 📊 Dashboard: {ticks:,} ticks | Rate: {rate}")
                    last_tick_count = ticks
                
                await asyncio.sleep(1)
            
            # Final summary
            elapsed_total = time.time() - start_time
            safe_print("\n📊 FINAL SUMMARY:")
            safe_print("=" * 50)
            
            if hasattr(display_manager, '_metrics'):
                metrics = display_manager._metrics
                final_ticks = metrics.get('ticks_processed', 0)
                final_rate = metrics.get('processing_rate', 'N/A')
                safe_print(f"   Total Ticks Processed: {final_ticks:,}")
                safe_print(f"   Final Rate: {final_rate}")
                safe_print(f"   Test Duration: {elapsed_total:.1f} seconds")
                
                if final_ticks > 1000:
                    avg_rate = final_ticks / elapsed_total
                    safe_print(f"   Average Rate: {avg_rate:.1f} ticks/sec")
                    safe_print("✅ DASHBOARD IS UPDATING DURING PROCESSING")
                    return True
                else:
                    safe_print("❌ DASHBOARD NOT UPDATING - Low tick count")
                    return False
            else:
                safe_print("❌ No metrics available")
                return False
            
        except Exception as e:
            safe_print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Entry point"""
    tester = TickRateDisplayTest()
    result = await tester.test_tick_rate_updates()
    
    safe_print("\n" + "="*70)
    safe_print("📊 CONCLUSIONI:")
    safe_print("="*70)
    
    if result:
        safe_print("✅ Test completato")
    else:
        safe_print("❌ Test fallito")
        
    safe_print("\n🔍 Se i tick non si aggiornano nella dashboard:")
    safe_print("   1. Il sistema potrebbe non inviare eventi di aggiornamento tick")
    safe_print("   2. Il display manager potrebbe non ricevere le metriche")
    safe_print("   3. Potrebbe mancare il collegamento tra contatore tick e display")


if __name__ == "__main__":
    asyncio.run(main())