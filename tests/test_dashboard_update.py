#!/usr/bin/env python3
"""
Test Dashboard Update - Debug perché la dashboard mostra tutti zeri
===================================================================

Test mirati per capire perché il ML Training Logger dashboard non si aggiorna.
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

# Safe print
def safe_print(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")

class DashboardUpdateTest:
    """Test per debug dashboard ML Training Logger"""
    
    def __init__(self):
        self.symbol = "USTEC"
        self.test_data_path = "./test_analyzer_data"
        
    async def test_1_direct_event_emission(self):
        """Test 1: Emissione diretta di eventi ML"""
        safe_print("\n🧪 TEST 1: Direct ML Event Emission")
        safe_print("="*50)
        
        try:
            from src.Unified_Analyzer_System import UnifiedAnalyzerSystem, create_custom_config, SystemMode, PerformanceProfile
            
            # Config con rate limits disabilitati per test
            config = create_custom_config(
                system_mode=SystemMode.TESTING,
                performance_profile=PerformanceProfile.RESEARCH,
                asset_symbol=self.symbol,
                learning_phase_enabled=True,
                log_level="NORMAL",
                
                # Disabilita rate limiting per test
                rate_limits={
                    'tick_processing': 10000,
                    'predictions': 10000,
                    'validations': 10000,
                    'training_events': 10000,
                    'champion_changes': 10000,
                    'performance_metrics': 10000
                }
            )
            
            # Initialize system
            unified_system = UnifiedAnalyzerSystem(config)
            await unified_system.start()
            safe_print("✅ System started")
            
            analyzer = unified_system.analyzer
            
            # Verifica ML logger
            safe_print(f"📊 ML Logger active: {analyzer.ml_logger_active}")
            safe_print(f"📊 Display Manager: {analyzer.ml_display_manager is not None}")
            
            # Emit eventi di test direttamente
            if analyzer.ml_logger_active:
                safe_print("\n🔄 Emitting test events...")
                
                # 1. Prediction event
                analyzer._emit_ml_event('prediction', {
                    'model_type': 'support_resistance',
                    'algorithm': 'LSTM_SupportResistance',
                    'symbol': self.symbol,
                    'confidence': 0.85,
                    'predicted_levels': {'support': [21000], 'resistance': [22000]},
                    'timestamp': datetime.now()
                })
                safe_print("✅ Emitted prediction event")
                
                # 2. Training event
                analyzer._emit_ml_event('training', {
                    'model_type': 'support_resistance',
                    'algorithm': 'LSTM_SupportResistance', 
                    'symbol': self.symbol,
                    'accuracy': 0.75,
                    'loss': 0.25,
                    'training_type': 'incremental',
                    'samples': 1000,
                    'timestamp': datetime.now()
                })
                safe_print("✅ Emitted training event")
                
                # 3. Champion change event
                analyzer._emit_ml_event('champion_change', {
                    'model_type': 'support_resistance',
                    'previous_champion': 'Classical_Pivot',
                    'new_champion': 'LSTM_SupportResistance',
                    'improvement': 0.15,
                    'symbol': self.symbol,
                    'timestamp': datetime.now()
                })
                safe_print("✅ Emitted champion change event")
                
                # 4. Performance metric event
                analyzer._emit_ml_event('performance_metric', {
                    'metric_type': 'accuracy',
                    'value': 0.75,
                    'model_type': 'support_resistance',
                    'algorithm': 'LSTM_SupportResistance',
                    'symbol': self.symbol,
                    'timestamp': datetime.now()
                })
                safe_print("✅ Emitted performance metric event")
                
                # Force display update
                await asyncio.sleep(0.5)  # Give time for events to process
                
                safe_print("\n🔄 Forcing display update...")
                analyzer._update_ml_display_metrics(self.symbol)
                safe_print("✅ Display update called")
                
                # Check event buffers
                if hasattr(analyzer, 'ml_event_collector') and analyzer.ml_event_collector:
                    collector = analyzer.ml_event_collector
                    
                    # Check event counts
                    if hasattr(collector, '_event_buffers'):
                        safe_print("\n📊 Event Buffer Status:")
                        for event_type, buffer in collector._event_buffers.items():
                            if hasattr(buffer, '__len__'):
                                safe_print(f"   {event_type}: {len(buffer)} events")
                                
                # Wait for display to update
                await asyncio.sleep(1)
                
                # Get current metrics
                if hasattr(analyzer, 'asset_analyzers') and self.symbol in analyzer.asset_analyzers:
                    asset_analyzer = analyzer.asset_analyzers[self.symbol]
                    
                    safe_print("\n📊 Asset Analyzer Metrics:")
                    
                    # Check champions in detail
                    if hasattr(asset_analyzer, 'competitions'):
                        safe_print("\n📊 Competition Status:")
                        for model_type, competition in asset_analyzer.competitions.items():
                            if hasattr(competition, 'current_champion'):
                                safe_print(f"   {model_type}: Champion = {competition.current_champion}")
                                
                                # Check algorithm performances
                                if hasattr(competition, 'algorithm_performances'):
                                    safe_print(f"      Algorithms registered: {list(competition.algorithm_performances.keys())}")
                                    
                                # Check if in learning phase
                                if hasattr(competition, 'is_learning_phase'):
                                    safe_print(f"      Learning phase: {competition.is_learning_phase}")
                            else:
                                safe_print(f"   {model_type}: No current_champion attribute")
                                
                        champion_count = sum(1 for comp in asset_analyzer.competitions.values() 
                                           if hasattr(comp, 'current_champion') and comp.current_champion)
                        safe_print(f"\n   Total Champions: {champion_count}")
                        
                    # Check health score
                    if hasattr(asset_analyzer, 'get_health_score'):
                        health = asset_analyzer.get_health_score()
                        safe_print(f"   Health Score: {health}%")
                        
                else:
                    safe_print("❌ Asset analyzer not found")
                    
            await unified_system.stop()
            return True
            
        except Exception as e:
            safe_print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    async def test_2_dashboard_rendering(self):
        """Test 2: Verifica rendering dashboard"""
        safe_print("\n🧪 TEST 2: Dashboard Rendering Check")
        safe_print("="*50)
        
        try:
            from src.Unified_Analyzer_System import UnifiedAnalyzerSystem, create_custom_config, SystemMode, PerformanceProfile
            
            config = create_custom_config(
                system_mode=SystemMode.TESTING,
                performance_profile=PerformanceProfile.RESEARCH,
                asset_symbol=self.symbol,
                learning_phase_enabled=True
            )
            
            unified_system = UnifiedAnalyzerSystem(config)
            await unified_system.start()
            
            analyzer = unified_system.analyzer
            display_manager = analyzer.ml_display_manager
            
            if display_manager:
                safe_print("✅ Display Manager found")
                
                # Check display thread
                if hasattr(display_manager, '_display_thread'):
                    thread = display_manager._display_thread
                    safe_print(f"   Display thread alive: {thread.is_alive() if thread else False}")
                    
                # Check metrics
                if hasattr(display_manager, '_metrics'):
                    safe_print(f"   Metrics dict exists: {display_manager._metrics is not None}")
                    if display_manager._metrics:
                        safe_print("   Current metrics:")
                        for key, value in display_manager._metrics.items():
                            safe_print(f"      {key}: {value}")
                            
                # Manually update metrics
                safe_print("\n🔄 Manually updating metrics...")
                display_manager.update_metrics(
                    champions=3,
                    health_score=75,
                    accuracy=0.82,
                    ticks_processed=1000,
                    processing_rate=100.5,
                    current_symbol=self.symbol
                )
                safe_print("✅ Metrics updated manually")
                
                # Wait for render
                await asyncio.sleep(2)
                
                # Check if metrics changed
                if hasattr(display_manager, '_metrics'):
                    safe_print("\n📊 Metrics after update:")
                    for key, value in display_manager._metrics.items():
                        safe_print(f"      {key}: {value}")
                        
                # Check display loop status
                if hasattr(display_manager, '_last_render_time'):
                    safe_print(f"\n⏰ Last render: {display_manager._last_render_time}")
                    
                # Try to get actual dashboard content
                if hasattr(display_manager, '_generate_dashboard_content'):
                    try:
                        content = display_manager._generate_dashboard_content()
                        safe_print("\n📺 Dashboard Content Sample:")
                        # Show first few lines
                        lines = content.split('\n')[:10]
                        for line in lines:
                            if line.strip():
                                safe_print(f"   {line}")
                    except Exception as e:
                        safe_print(f"   Could not generate content: {e}")
                        
            else:
                safe_print("❌ No display manager")
                
            await unified_system.stop()
            return True
            
        except Exception as e:
            safe_print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    async def test_3_learning_phase_metrics(self):
        """Test 3: Metriche durante learning phase"""
        safe_print("\n🧪 TEST 3: Learning Phase Metrics Generation")
        safe_print("="*50)
        
        try:
            from src.Unified_Analyzer_System import UnifiedAnalyzerSystem, create_custom_config, SystemMode, PerformanceProfile
            
            config = create_custom_config(
                system_mode=SystemMode.TESTING,
                performance_profile=PerformanceProfile.RESEARCH,
                asset_symbol=self.symbol,
                learning_phase_enabled=True,
                min_learning_days=0,  # Per test immediato
            )
            
            unified_system = UnifiedAnalyzerSystem(config)
            await unified_system.start()
            
            # Process alcuni tick per attivare learning
            data_file = f"{self.test_data_path}/backtest_USTEC_20250516_20250715.jsonl"
            
            with open(data_file, 'r', encoding='utf-8') as f:
                f.readline()  # Skip header
                
                # Process 20 ticks
                for i in range(20):
                    line = f.readline()
                    if not line:
                        break
                        
                    tick_data = json.loads(line.strip())
                    timestamp = datetime.strptime(tick_data['timestamp'], '%Y.%m.%d %H:%M:%S')
                    
                    result = await unified_system.process_tick(
                        timestamp=timestamp,
                        price=(tick_data['bid'] + tick_data['ask']) / 2,
                        volume=tick_data.get('volume', 1),
                        bid=tick_data['bid'],
                        ask=tick_data['ask']
                    )
                    
                    if i % 5 == 0:
                        safe_print(f"   Processed tick {i+1}: {result.get('status') if result else 'None'}")
                        
            # Check learning metrics
            analyzer = unified_system.analyzer
            if analyzer and hasattr(analyzer, 'asset_analyzers'):
                if self.symbol in analyzer.asset_analyzers:
                    asset_analyzer = analyzer.asset_analyzers[self.symbol]
                    
                    safe_print("\n📊 Learning Phase Status:")
                    
                    # Check learning progress
                    if hasattr(asset_analyzer, 'is_learning_complete'):
                        safe_print(f"   Learning complete: {asset_analyzer.is_learning_complete}")
                        
                    if hasattr(asset_analyzer, 'tick_count'):
                        safe_print(f"   Tick count: {asset_analyzer.tick_count}")
                        
                    if hasattr(asset_analyzer, 'learning_start_time'):
                        safe_print(f"   Learning start: {asset_analyzer.learning_start_time}")
                        
                    # Force metrics calculation
                    if hasattr(asset_analyzer, '_calculate_model_metrics'):
                        metrics = asset_analyzer._calculate_model_metrics()
                        safe_print(f"\n📊 Calculated Metrics:")
                        for key, value in metrics.items():
                            safe_print(f"   {key}: {value}")
                            
            await unified_system.stop()
            return True
            
        except Exception as e:
            safe_print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    async def run_all_tests(self):
        """Esegue tutti i test dashboard"""
        safe_print("\n🚀 Starting Dashboard Update Tests")
        safe_print("="*70)
        
        # Test 1: Direct event emission
        test1_ok = await self.test_1_direct_event_emission()
        
        # Test 2: Dashboard rendering
        test2_ok = await self.test_2_dashboard_rendering()
        
        # Test 3: Learning phase metrics
        test3_ok = await self.test_3_learning_phase_metrics()
        
        # Summary
        safe_print("\n📊 DASHBOARD TEST SUMMARY")
        safe_print("="*70)
        safe_print(f"{'✅' if test1_ok else '❌'} Test 1 - Direct Event Emission: {test1_ok}")
        safe_print(f"{'✅' if test2_ok else '❌'} Test 2 - Dashboard Rendering: {test2_ok}")
        safe_print(f"{'✅' if test3_ok else '❌'} Test 3 - Learning Phase Metrics: {test3_ok}")
        
        # Diagnosis
        safe_print("\n🔍 DIAGNOSIS")
        safe_print("="*70)
        
        if not test1_ok:
            safe_print("❌ Events are not being emitted correctly")
        elif not test2_ok:
            safe_print("❌ Dashboard rendering has issues")
        elif not test3_ok:
            safe_print("❌ Learning phase doesn't generate metrics")
        else:
            safe_print("⚠️ All components work individually")
            safe_print("   The issue might be:")
            safe_print("   1. Rate limiting blocking updates")
            safe_print("   2. Event processing queue overflow")
            safe_print("   3. Display refresh timing")
            safe_print("   4. Metrics calculation during batch processing")


async def main():
    """Entry point"""
    tester = DashboardUpdateTest()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())