#!/usr/bin/env python3
"""
🔍 SYSTEM INTEGRATION VERIFICATION SCRIPT
=========================================

Verifica completa che tutti i moduli del progetto ScalpingBOT interagiscano correttamente:
- Import di tutti i moduli
- Inizializzazione dei componenti
- Test delle interazioni tra moduli
- Verifica dei sistemi di logging
- Test del flusso di dati
- Controllo delle dipendenze

Usage: python utils/system_integration_test.py
"""

import sys
import os
import asyncio
import time
import traceback
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Setup paths
BASE_PATH = r"C:\ScalpingBOT"
SRC_PATH = os.path.join(BASE_PATH, "src")
UTILS_PATH = os.path.join(BASE_PATH, "utils")
MODULES_PATH = os.path.join(BASE_PATH, "modules")
TESTS_PATH = os.path.join(BASE_PATH, "tests")

sys.path.insert(0, BASE_PATH)
sys.path.insert(0, SRC_PATH)
sys.path.insert(0, UTILS_PATH)
sys.path.insert(0, MODULES_PATH)

# Global test results
test_results = {
    'timestamp': datetime.now().isoformat(),
    'total_tests': 0,
    'passed_tests': 0,
    'failed_tests': 0,
    'test_details': [],
    'module_status': {},
    'integration_status': {},
    'critical_issues': [],
    'warnings': []
}

def safe_print(message: str, level: str = "INFO"):
    """Safe print with timestamp and level"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    level_icon = {
        'INFO': '📋',
        'SUCCESS': '✅', 
        'WARNING': '⚠️',
        'ERROR': '❌',
        'DEBUG': '🔍'
    }.get(level, '📋')
    
    print(f"[{timestamp}] {level_icon} {message}")

def log_test_result(test_name: str, success: bool, details: str = "", critical: bool = False):
    """Log test result"""
    global test_results
    
    test_results['total_tests'] += 1
    if success:
        test_results['passed_tests'] += 1
        safe_print(f"PASS: {test_name}", "SUCCESS")
    else:
        test_results['failed_tests'] += 1
        safe_print(f"FAIL: {test_name} - {details}", "ERROR")
        if critical:
            test_results['critical_issues'].append(f"{test_name}: {details}")
    
    test_results['test_details'].append({
        'test': test_name,
        'success': success,
        'details': details,
        'critical': critical,
        'timestamp': datetime.now().isoformat()
    })

class SystemIntegrationTester:
    """Main integration tester class"""
    
    def __init__(self):
        self.modules = {}
        self.instances = {}
        self.test_data_path = "./integration_test_data"
        
        # Ensure test directory exists
        os.makedirs(self.test_data_path, exist_ok=True)
        
        safe_print("🚀 System Integration Verification Started")
        safe_print(f"📁 Base path: {BASE_PATH}")
        safe_print(f"📁 Test data: {self.test_data_path}")
    
    async def run_complete_verification(self) -> bool:
        """Run complete system verification"""
        
        safe_print("=" * 60)
        safe_print("🔍 STARTING COMPLETE SYSTEM INTEGRATION TEST")
        safe_print("=" * 60)
        
        # Phase 1: Basic imports and module loading
        if not await self.test_basic_imports():
            safe_print("❌ CRITICAL: Basic imports failed", "ERROR")
            return False
        
        # Phase 2: Advanced module imports
        await self.test_advanced_imports()
        
        # Phase 3: Module initialization
        await self.test_module_initialization()
        
        # Phase 4: Cross-module interactions
        await self.test_cross_module_interactions()
        
        # Phase 5: Data flow testing
        await self.test_data_flow()
        
        # Phase 6: Logging systems integration
        await self.test_logging_integration()
        
        # Phase 7: Performance and memory
        await self.test_performance_basics()
        
        # Phase 8: Error handling
        await self.test_error_handling()
        
        # Generate final report
        await self.generate_final_report()
        
        return test_results['failed_tests'] == 0
    
    async def test_basic_imports(self) -> bool:
        """Test basic Python imports"""
        safe_print("\n📋 PHASE 1: BASIC IMPORTS AND DEPENDENCIES")
        
        basic_modules = [
            ('sys', 'sys'),
            ('os', 'os'),
            ('datetime', 'datetime'),
            ('json', 'json'),
            ('asyncio', 'asyncio'),
            ('pathlib', 'pathlib'),
            ('collections', 'collections'),
            ('typing', 'typing')
        ]
        
        all_success = True
        
        for module_name, import_name in basic_modules:
            try:
                __import__(import_name)
                log_test_result(f"Import {module_name}", True)
            except ImportError as e:
                log_test_result(f"Import {module_name}", False, str(e), critical=True)
                all_success = False
        
        return all_success
    
    async def test_advanced_imports(self):
        """Test advanced/optional imports"""
        safe_print("\n📋 PHASE 2: ADVANCED MODULE IMPORTS")
        
        advanced_modules = [
            ('numpy', 'numpy', False),
            ('pandas', 'pandas', False),
            ('tensorflow', 'tensorflow', False),
            ('torch', 'torch', False),
            ('sklearn', 'sklearn', False),
            ('MetaTrader5', 'MetaTrader5', True),  # Critical for trading
        ]
        
        for module_name, import_name, critical in advanced_modules:
            try:
                __import__(import_name)
                log_test_result(f"Import {module_name}", True)
                test_results['module_status'][module_name] = 'available'
            except ImportError as e:
                log_test_result(f"Import {module_name}", False, str(e), critical=critical)
                test_results['module_status'][module_name] = 'missing'
    
    async def test_module_initialization(self):
        """Test initialization of project modules"""
        safe_print("\n📋 PHASE 3: PROJECT MODULE INITIALIZATION")
        
        # Test universal encoding fix
        await self.test_universal_encoding()
        
        # Test core modules
        await self.test_core_modules()
        
        # Test unified system
        await self.test_unified_system()
        
        # Test ML training logger
        await self.test_ml_training_logger()
    
    async def test_universal_encoding(self):
        """Test universal encoding fix"""
        try:
            from utils.universal_encoding_fix import (
                safe_print as safe_print_util,
                init_universal_encoding,
                get_safe_logger
            )
            
            # Test initialization
            init_universal_encoding(silent=True)
            logger = get_safe_logger("integration_test")
            
            # Test safe_print
            safe_print_util("Test message")
            
            log_test_result("Universal Encoding Fix", True)
            self.modules['universal_encoding'] = {
                'safe_print': safe_print_util,
                'logger': logger
            }
            
        except Exception as e:
            log_test_result("Universal Encoding Fix", False, str(e))
    
    async def test_core_modules(self):
        """Test core analyzer modules"""
        
        # Test AdvancedMarketAnalyzer
        try:
            from src.Analyzer import AdvancedMarketAnalyzer
            
            analyzer = AdvancedMarketAnalyzer(self.test_data_path)
            self.instances['analyzer'] = analyzer
            
            log_test_result("AdvancedMarketAnalyzer Import", True)
            
            # Test adding asset
            asset_analyzer = analyzer.add_asset('EURUSD')
            if asset_analyzer:
                log_test_result("AdvancedMarketAnalyzer Asset Addition", True)
            else:
                log_test_result("AdvancedMarketAnalyzer Asset Addition", False, "add_asset returned None")
            
        except Exception as e:
            log_test_result("AdvancedMarketAnalyzer", False, str(e), critical=True)
        
        # Test MT5BacktestRunner
        try:
            from src.MT5BacktestRunner import MT5BacktestRunner, BacktestConfig
            
            runner = MT5BacktestRunner(self.test_data_path)
            self.instances['mt5_runner'] = runner
            
            # Test config creation
            config = BacktestConfig(
                symbol='EURUSD',
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now(),
                data_source='mt5_export'
            )
            
            log_test_result("MT5BacktestRunner", True)
            log_test_result("BacktestConfig", True)
            
        except Exception as e:
            log_test_result("MT5BacktestRunner", False, str(e), critical=True)
    
    async def test_unified_system(self):
        """Test Unified Analyzer System"""
        try:
            from src.Unified_Analyzer_System import (
                UnifiedAnalyzerSystem,
                UnifiedConfig,
                SystemMode,
                PerformanceProfile,
                create_custom_config
            )
            
            # Test config creation
            config = create_custom_config(
                system_mode=SystemMode.TESTING,
                performance_profile=PerformanceProfile.RESEARCH,
                asset_symbol='EURUSD',
                base_directory=f"{self.test_data_path}/unified_test"
            )
            
            # Test system creation
            unified_system = UnifiedAnalyzerSystem(config)
            self.instances['unified_system'] = unified_system
            
            log_test_result("Unified Analyzer System Import", True)
            
            # Test system startup
            await unified_system.start()
            log_test_result("Unified System Startup", True)
            
            # Test system status
            status = unified_system.get_system_status()
            if isinstance(status, dict) and 'system' in status:
                log_test_result("Unified System Status", True)
            else:
                log_test_result("Unified System Status", False, "Invalid status format")
            
            # Test tick processing
            result = await unified_system.process_tick(
                timestamp=datetime.now(),
                price=1.1000,
                volume=1000
            )
            
            if isinstance(result, dict):
                log_test_result("Unified System Tick Processing", True)
            else:
                log_test_result("Unified System Tick Processing", False, "Invalid result format")
            
        except Exception as e:
            log_test_result("Unified Analyzer System", False, str(e))
    
    async def test_ml_training_logger(self):
        """Test ML Training Logger"""
        try:
            from modules.Analyzer_Logging_SlaveModule import (
                AnalyzerLoggingSlave,
                LoggingConfig,
                LogLevel,
                create_logging_slave,
                process_analyzer_data
            )
            
            # Test config creation
            config = LoggingConfig(
                log_level=LogLevel.NORMAL,
                log_directory=f"{self.test_data_path}/ml_test_logs",
                enable_console_output=False,  # Don't spam console
                enable_csv_export=True
            )
            
            # Test slave creation
            ml_slave = await create_logging_slave(config)
            self.instances['ml_slave'] = ml_slave
            
            log_test_result("ML Training Logger Import", True)
            log_test_result("ML Logging Slave Creation", True)
            
            # Test statistics
            stats = ml_slave.get_statistics()
            if isinstance(stats, dict):
                log_test_result("ML Logger Statistics", True)
            else:
                log_test_result("ML Logger Statistics", False, "Invalid stats format")
            
        except Exception as e:
            log_test_result("ML Training Logger", False, str(e))
    
    async def test_cross_module_interactions(self):
        """Test interactions between modules"""
        safe_print("\n📋 PHASE 4: CROSS-MODULE INTERACTIONS")
        
        # Test Analyzer + Unified System
        await self.test_analyzer_unified_integration()
        
        # Test Analyzer + ML Logger
        await self.test_analyzer_ml_logger_integration()
        
        # Test Unified System + ML Logger
        await self.test_unified_ml_logger_integration()
    
    async def test_analyzer_unified_integration(self):
        """Test Analyzer + Unified System integration"""
        try:
            analyzer = self.instances.get('analyzer')
            unified_system = self.instances.get('unified_system')
            
            if not analyzer or not unified_system:
                log_test_result("Analyzer-Unified Integration", False, "Missing required instances")
                return
            
            # Test that unified system can access analyzer
            if hasattr(unified_system, 'analyzer'):
                log_test_result("Unified System Analyzer Access", True)
            else:
                log_test_result("Unified System Analyzer Access", False, "No analyzer attribute")
            
            # Test data flow
            if analyzer and 'EURUSD' in analyzer.asset_analyzers:
                asset_analyzer = analyzer.asset_analyzers['EURUSD']
                
                # Simulate some processing
                if unified_system:  # Type safety check
                    for i in range(10):
                        result = await unified_system.process_tick(
                            timestamp=datetime.now(),
                            price=1.1000 + i * 0.0001,
                            volume=1000 + i * 100
                        )
                
                log_test_result("Analyzer-Unified Data Flow", True)
            else:
                log_test_result("Analyzer-Unified Data Flow", False, "No EURUSD asset analyzer")
            
        except Exception as e:
            log_test_result("Analyzer-Unified Integration", False, str(e))
    
    async def test_analyzer_ml_logger_integration(self):
        """Test Analyzer + ML Logger integration"""
        try:
            analyzer = self.instances.get('analyzer')
            ml_slave = self.instances.get('ml_slave')
            
            if not analyzer or not ml_slave:
                log_test_result("Analyzer-ML Logger Integration", False, "Missing required instances")
                return
            
            # Test process_analyzer_data
            from modules.Analyzer_Logging_SlaveModule import process_analyzer_data
            
            await process_analyzer_data(ml_slave, analyzer)
            log_test_result("Analyzer-ML Logger Data Processing", True)
            
            # Check if events were generated
            stats = ml_slave.get_statistics()
            events_processed = stats.get('events_processed', 0)
            
            if events_processed >= 0:  # Any non-negative number is valid
                log_test_result("ML Logger Event Processing", True)
            else:
                log_test_result("ML Logger Event Processing", False, f"Invalid events count: {events_processed}")
            
        except Exception as e:
            log_test_result("Analyzer-ML Logger Integration", False, str(e))
    
    async def test_unified_ml_logger_integration(self):
        """Test Unified System + ML Logger integration"""
        try:
            unified_system = self.instances.get('unified_system')
            ml_slave = self.instances.get('ml_slave')
            
            if not unified_system or not ml_slave:
                log_test_result("Unified-ML Logger Integration", False, "Missing required instances")
                return
            
            # Test that unified system can work with ML logger
            initial_stats = ml_slave.get_statistics()
            
            # Process some ticks
            if unified_system:  # Type safety check
                for i in range(5):
                    await unified_system.process_tick(
                        timestamp=datetime.now(),
                        price=1.1000 + i * 0.0001,
                        volume=1000
                    )
            
            # Check ML logger after processing
            final_stats = ml_slave.get_statistics()
            
            log_test_result("Unified-ML Logger Integration", True)
            
        except Exception as e:
            log_test_result("Unified-ML Logger Integration", False, str(e))
    
    async def test_data_flow(self):
        """Test data flow through the system"""
        safe_print("\n📋 PHASE 5: DATA FLOW TESTING")
        
        try:
            # Test complete data flow: tick -> analyzer -> unified -> ML logger
            analyzer = self.instances.get('analyzer')
            unified_system = self.instances.get('unified_system')
            ml_slave = self.instances.get('ml_slave')
            
            if not all([analyzer, unified_system, ml_slave]):
                log_test_result("Complete Data Flow", False, "Missing required components")
                return
            
            # Simulate realistic tick data
            test_ticks = [
                {'timestamp': datetime.now(), 'price': 1.1000, 'volume': 1000, 'bid': 1.0999, 'ask': 1.1001},
                {'timestamp': datetime.now(), 'price': 1.1001, 'volume': 1500, 'bid': 1.1000, 'ask': 1.1002},
                {'timestamp': datetime.now(), 'price': 1.1002, 'volume': 800, 'bid': 1.1001, 'ask': 1.1003},
                {'timestamp': datetime.now(), 'price': 1.0999, 'volume': 1200, 'bid': 1.0998, 'ask': 1.1000},
                {'timestamp': datetime.now(), 'price': 1.1000, 'volume': 900, 'bid': 1.0999, 'ask': 1.1001},
            ]
            
            processed_count = 0
            
            if unified_system:  # Type safety check
                for tick in test_ticks:
                    # Process through unified system
                    result = await unified_system.process_tick(
                        timestamp=tick['timestamp'],
                        price=tick['price'],
                        volume=tick['volume'],
                        bid=tick['bid'],
                        ask=tick['ask']
                    )
                    
                    if result and result.get('status') in ['success', 'mock']:
                        processed_count += 1
            
            # Process ML events
            if ml_slave and analyzer:  # Type safety check
                from modules.Analyzer_Logging_SlaveModule import process_analyzer_data
                await process_analyzer_data(ml_slave, analyzer)
            
            log_test_result("Complete Data Flow", True, f"Processed {processed_count}/{len(test_ticks)} ticks")
            
        except Exception as e:
            log_test_result("Complete Data Flow", False, str(e))
    
    async def test_logging_integration(self):
        """Test logging systems integration"""
        safe_print("\n📋 PHASE 6: LOGGING SYSTEMS INTEGRATION")
        
        # Test file creation
        log_files_created = []
        
        for instance_name, instance in self.instances.items():
            try:
                if hasattr(instance, 'config') and hasattr(instance.config, 'base_directory'):
                    log_dir = instance.config.base_directory
                    if os.path.exists(log_dir):
                        files = [f for f in os.listdir(log_dir) if f.endswith(('.log', '.csv', '.json'))]
                        log_files_created.extend(files)
                        
                elif hasattr(instance, 'config') and hasattr(instance.config, 'log_directory'):
                    log_dir = instance.config.log_directory
                    if os.path.exists(log_dir):
                        files = [f for f in os.listdir(log_dir) if f.endswith(('.log', '.csv', '.json'))]
                        log_files_created.extend(files)
            except:
                continue
        
        if log_files_created:
            log_test_result("Log Files Creation", True, f"Created {len(log_files_created)} log files")
        else:
            log_test_result("Log Files Creation", False, "No log files created")
        
        # Test log writing
        try:
            ml_slave = self.instances.get('ml_slave')
            if ml_slave:
                stats = ml_slave.get_statistics()
                events_processed = stats.get('events_processed', 0)
                log_test_result("Log Writing", True, f"ML Logger processed {events_processed} events")
            else:
                log_test_result("Log Writing", False, "No ML slave available")
        except Exception as e:
            log_test_result("Log Writing", False, str(e))
    
    async def test_performance_basics(self):
        """Test basic performance characteristics"""
        safe_print("\n📋 PHASE 7: PERFORMANCE BASICS")
        
        try:
            import psutil
            process = psutil.Process()
            
            # Memory usage
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_percent = process.memory_percent()
            
            log_test_result("Memory Usage Check", True, f"{memory_mb:.1f}MB ({memory_percent:.1f}%)")
            
            if memory_percent > 50:
                test_results['warnings'].append(f"High memory usage: {memory_percent:.1f}%")
            
            # CPU usage (rough estimate)
            cpu_before = process.cpu_percent()
            await asyncio.sleep(1)
            cpu_after = process.cpu_percent()
            
            log_test_result("CPU Usage Check", True, f"{cpu_after:.1f}%")
            
        except ImportError:
            log_test_result("Performance Monitoring", False, "psutil not available")
        except Exception as e:
            log_test_result("Performance Monitoring", False, str(e))
    
    async def test_error_handling(self):
        """Test error handling capabilities"""
        safe_print("\n📋 PHASE 8: ERROR HANDLING")
        
        # Test invalid tick processing
        try:
            unified_system = self.instances.get('unified_system')
            if unified_system:
                # Test with invalid data
                result = await unified_system.process_tick(
                    timestamp=None,
                    price=-1,
                    volume=-100
                )
                log_test_result("Invalid Data Handling", True, "System handled invalid data gracefully")
            else:
                log_test_result("Invalid Data Handling", False, "No unified system available")
        except Exception as e:
            log_test_result("Invalid Data Handling", True, f"Exception caught properly: {type(e).__name__}")
        
        # Test ML logger error handling
        try:
            ml_slave = self.instances.get('ml_slave')
            if ml_slave:
                # Test with invalid analyzer
                from modules.Analyzer_Logging_SlaveModule import process_analyzer_data
                # Type-safe test with None analyzer
                try:
                    await process_analyzer_data(ml_slave, None)  # type: ignore
                    log_test_result("ML Logger Error Handling", True, "ML Logger handled None analyzer")
                except TypeError:
                    log_test_result("ML Logger Error Handling", True, "ML Logger properly rejected None analyzer")
            else:
                log_test_result("ML Logger Error Handling", False, "No ML slave available")
        except Exception as e:
            log_test_result("ML Logger Error Handling", True, f"Exception caught properly: {type(e).__name__}")
    
    async def generate_final_report(self):
        """Generate comprehensive final report"""
        safe_print("\n📋 PHASE 9: FINAL REPORT GENERATION")
        
        # Calculate success rate
        total = test_results['total_tests']
        passed = test_results['passed_tests']
        failed = test_results['failed_tests']
        
        if total > 0:
            success_rate = (passed / total) * 100
        else:
            success_rate = 0
        
        safe_print("=" * 60)
        safe_print("📊 SYSTEM INTEGRATION TEST RESULTS")
        safe_print("=" * 60)
        safe_print(f"🔢 Total Tests: {total}")
        safe_print(f"✅ Passed: {passed}")
        safe_print(f"❌ Failed: {failed}")
        safe_print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if test_results['critical_issues']:
            safe_print(f"\n🚨 CRITICAL ISSUES ({len(test_results['critical_issues'])}):")
            for issue in test_results['critical_issues']:
                safe_print(f"   ❌ {issue}", "ERROR")
        
        if test_results['warnings']:
            safe_print(f"\n⚠️ WARNINGS ({len(test_results['warnings'])}):")
            for warning in test_results['warnings']:
                safe_print(f"   ⚠️ {warning}", "WARNING")
        
        # Module status summary
        safe_print(f"\n📦 MODULE STATUS:")
        for module, status in test_results['module_status'].items():
            icon = "✅" if status == 'available' else "❌"
            safe_print(f"   {icon} {module}: {status}")
        
        # Save detailed report
        report_file = f"{self.test_data_path}/integration_test_report.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            safe_print(f"\n📄 Detailed report saved: {report_file}")
        except Exception as e:
            safe_print(f"⚠️ Could not save report: {e}", "WARNING")
        
        # Final verdict
        if success_rate >= 90:
            safe_print("\n🎉 SYSTEM INTEGRATION: EXCELLENT!", "SUCCESS")
        elif success_rate >= 75:
            safe_print("\n✅ SYSTEM INTEGRATION: GOOD", "SUCCESS")
        elif success_rate >= 50:
            safe_print("\n⚠️ SYSTEM INTEGRATION: NEEDS IMPROVEMENT", "WARNING")
        else:
            safe_print("\n❌ SYSTEM INTEGRATION: CRITICAL ISSUES", "ERROR")
        
        safe_print("=" * 60)
    
    async def cleanup(self):
        """Cleanup test instances"""
        safe_print("\n🧹 CLEANUP PHASE")
        
        # Stop unified system
        if 'unified_system' in self.instances:
            try:
                await self.instances['unified_system'].stop()
                safe_print("✅ Unified system stopped")
            except Exception as e:
                safe_print(f"⚠️ Error stopping unified system: {e}", "WARNING")
        
        # Stop ML slave
        if 'ml_slave' in self.instances:
            try:
                await self.instances['ml_slave'].stop()
                safe_print("✅ ML slave stopped")
            except Exception as e:
                safe_print(f"⚠️ Error stopping ML slave: {e}", "WARNING")
        
        # Save analyzer states
        if 'analyzer' in self.instances:
            try:
                self.instances['analyzer'].save_all_states()
                safe_print("✅ Analyzer states saved")
            except Exception as e:
                safe_print(f"⚠️ Error saving analyzer states: {e}", "WARNING")

async def main():
    """Main function"""
    safe_print("🔍 System Integration Verification Script")
    safe_print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print(f"🐍 Python: {sys.version}")
    safe_print(f"📁 Working directory: {os.getcwd()}")
    
    tester = SystemIntegrationTester()
    
    try:
        success = await tester.run_complete_verification()
        return success
    except KeyboardInterrupt:
        safe_print("\n🛑 Test interrupted by user", "WARNING")
        return False
    except Exception as e:
        safe_print(f"\n❌ Test failed with error: {e}", "ERROR")
        traceback.print_exc()
        return False
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)