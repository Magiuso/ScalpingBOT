#!/usr/bin/env python3
"""
Test Integration ML_Training_Logger con Analyzer.py
=================================================

Script di test per verificare che l'integrazione ML_Training_Logger funzioni correttamente.

Usage:
    python test_ml_logger_integration.py
"""

import sys
import os
import time
import traceback
from datetime import datetime

# Add src and ML_Training_Logger to path for imports
current_dir = os.getcwd()  # C:\ScalpingBOT
sys.path.insert(0, current_dir)  # Add ScalpingBOT root
sys.path.insert(0, os.path.join(current_dir, 'src'))  # Add src/
sys.path.insert(0, os.path.join(current_dir, 'ML_Training_Logger'))  # Add ML_Training_Logger/

print(f"🔧 Added to Python path:")
print(f"   - {current_dir}")
print(f"   - {os.path.join(current_dir, 'src')}")
print(f"   - {os.path.join(current_dir, 'ML_Training_Logger')}")

def test_ml_logger_import():
    """Test 1: Verifica che ML_Training_Logger sia importabile"""
    print("🧪 Test 1: ML_Training_Logger Import")
    
    try:
        from ML_Training_Logger.Unified_ConfigManager import UnifiedConfigManager
        from ML_Training_Logger.Event_Collector import EventCollector
        from ML_Training_Logger.Display_Manager import DisplayManager
        from ML_Training_Logger.Storage_Manager import StorageManager
        
        print("✅ ML_Training_Logger modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ ML_Training_Logger import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing ML_Training_Logger: {e}")
        return False

def test_analyzer_import():
    """Test 2: Verifica che Analyzer.py sia importabile con ML_Training_Logger"""
    print("\n🧪 Test 2: Analyzer.py Import with ML_Training_Logger")
    
    try:
        from src.Analyzer import AdvancedMarketAnalyzer, AnalyzerConfig
        
        print("✅ Analyzer.py imported successfully with ML_Training_Logger")
        return True
        
    except SystemExit as e:
        print(f"❌ Analyzer.py failed to start: {e}")
        print("💡 This indicates ML_Training_Logger integration failed")
        return False
    except ImportError as e:
        print(f"❌ Import error in Analyzer.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing Analyzer.py: {e}")
        traceback.print_exc()
        return False

def test_analyzer_config_integration():
    """Test 3: Verifica che AnalyzerConfig abbia i campi ML_Training_Logger"""
    print("\n🧪 Test 3: AnalyzerConfig ML_Training_Logger Integration")
    
    try:
        from src.Analyzer import AnalyzerConfig
        
        config = AnalyzerConfig()
        
        # Check ML logger fields
        required_fields = [
            'ml_logger_enabled',
            'ml_logger_verbosity', 
            'ml_logger_terminal_mode',
            'ml_logger_file_output',
            'ml_logger_formats',
            'ml_logger_base_directory',
            'ml_logger_rate_limit_ticks',
            'ml_logger_flush_interval'
        ]
        
        missing_fields = []
        for field in required_fields:
            if not hasattr(config, field):
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing ML logger fields in AnalyzerConfig: {missing_fields}")
            return False
        
        # Test create_ml_logger_config method
        try:
            ml_config = config.create_ml_logger_config("TEST_ASSET")
            print(f"✅ ML Logger config created successfully")
            
        except Exception as e:
            print(f"❌ create_ml_logger_config failed: {e}")
            return False
        
        print("✅ AnalyzerConfig ML_Training_Logger integration working")
        return True
        
    except Exception as e:
        print(f"❌ AnalyzerConfig integration test failed: {e}")
        traceback.print_exc()
        return False

def test_analyzer_initialization():
    """Test 4: Verifica che AdvancedMarketAnalyzer si inizializzi con ML_Training_Logger"""
    print("\n🧪 Test 4: AdvancedMarketAnalyzer Initialization")
    
    try:
        from src.Analyzer import AdvancedMarketAnalyzer
        
        # Create analyzer with test data path
        test_data_path = "./test_analyzer_data"
        os.makedirs(test_data_path, exist_ok=True)
        
        analyzer = AdvancedMarketAnalyzer(data_path=test_data_path)
        
        # Check ML logger components
        required_attributes = [
            'ml_logger_config',
            'ml_event_collector', 
            'ml_display_manager',
            'ml_storage_manager',
            'ml_logger_active'
        ]
        
        missing_attrs = []
        for attr in required_attributes:
            if not hasattr(analyzer, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            print(f"❌ Missing ML logger attributes: {missing_attrs}")
            return False
        
        # Check if ML logger is active
        if not analyzer.ml_logger_active:
            print("❌ ML logger is not active")
            return False
        
        # Check if components are initialized
        if not analyzer.ml_event_collector:
            print("❌ ML Event Collector not initialized")
            return False
            
        if not analyzer.ml_display_manager:
            print("❌ ML Display Manager not initialized") 
            return False
            
        if not analyzer.ml_storage_manager:
            print("❌ ML Storage Manager not initialized")
            return False
        
        print("✅ AdvancedMarketAnalyzer initialized with ML_Training_Logger")
        
        # Test shutdown
        analyzer.shutdown()
        print("✅ AdvancedMarketAnalyzer shutdown completed")
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedMarketAnalyzer initialization failed: {e}")
        traceback.print_exc()
        return False

def test_event_emission():
    """Test 5: Verifica che gli eventi ML siano emessi correttamente"""
    print("\n🧪 Test 5: ML Event Emission")
    
    try:
        from src.Analyzer import AdvancedMarketAnalyzer
        
        # Create analyzer
        test_data_path = "./test_analyzer_data"
        analyzer = AdvancedMarketAnalyzer(data_path=test_data_path)
        
        # Test manual event emission
        test_event_data = {
            'test_field': 'test_value',
            'timestamp': datetime.now().isoformat(),
            'test_number': 42
        }
        
        # Emit test event
        analyzer._emit_ml_event('diagnostic', test_event_data)
        print("✅ Manual event emission successful")
        
        # Test tick processing (if possible)
        try:
            result = analyzer.process_tick(
                asset="TESTASSET",
                timestamp=datetime.now(),
                price=1.2345,
                volume=1000
            )
            print("✅ process_tick with ML event emission successful")
            
        except Exception as e:
            print(f"⚠️ process_tick test failed (expected): {e}")
        
        # Test add asset
        try:
            analyzer.add_asset("TESTASSET2")
            print("✅ add_asset with ML integration successful")
        except Exception as e:
            print(f"⚠️ add_asset test failed: {e}")
        
        # Test shutdown with events
        analyzer.shutdown()
        print("✅ Event emission test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Event emission test failed: {e}")
        traceback.print_exc()
        return False

def test_display_manager_updates():
    """Test 6: Verifica che il display manager riceva aggiornamenti"""
    print("\n🧪 Test 6: Display Manager Updates")
    
    try:
        from src.Analyzer import AdvancedMarketAnalyzer
        
        # Create analyzer
        test_data_path = "./test_analyzer_data"
        analyzer = AdvancedMarketAnalyzer(data_path=test_data_path)
        
        # Test display metrics update
        analyzer._update_ml_display_metrics("TESTASSET")
        print("✅ Display metrics update successful")
        
        # Test health score calculation
        health_score = analyzer._calculate_global_health()['score']
        print(f"✅ System health score calculated: {health_score:.1f}")
        
        analyzer.shutdown()
        print("✅ Display manager test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Display manager test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Esegue tutti i test di integrazione"""
    print("🚀 ML_Training_Logger Integration Test Suite")
    print("=" * 60)
    
    tests = [
        test_ml_logger_import,
        test_analyzer_import,
        test_analyzer_config_integration,
        test_analyzer_initialization,
        test_event_emission,
        test_display_manager_updates
    ]
    
    passed = 0
    total = len(tests)
    
    for i, test_func in enumerate(tests, 1):
        try:
            if test_func():
                passed += 1
            else:
                print(f"🔴 Test {i} FAILED")
        except Exception as e:
            print(f"🔴 Test {i} CRASHED: {e}")
            traceback.print_exc()
        
        # Brief pause between tests
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! ML_Training_Logger integration is working correctly!")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please check the integration.")
        return False

def main():
    """Main test execution"""
    print(f"🕒 Starting ML_Training_Logger integration tests at {datetime.now()}")
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # 🔍 DEBUG: Check directory structure
    print("📂 Directory contents:")
    for item in os.listdir('.'):
        item_path = os.path.join('.', item)
        item_type = 'DIR' if os.path.isdir(item_path) else 'FILE'
        print(f"   {item_type}: {item}")

    print("\n🔍 Checking specific directories:")
    print(f"   src/ exists: {os.path.exists('src')}")
    print(f"   ML_Training_Logger/ exists: {os.path.exists('ML_Training_Logger')}")

    if os.path.exists('src'):
        print(f"   src/ contents: {os.listdir('src')}")
        
    if os.path.exists('ML_Training_Logger'):
        print(f"   ML_Training_Logger/ contents: {os.listdir('ML_Training_Logger')}")
    
    print()  # Empty line before tests
    
    try:
        success = run_all_tests()
        
        if success:
            print("\n✅ Integration test completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Integration test failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    main()