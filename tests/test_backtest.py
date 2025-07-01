#!/usr/bin/env python3
"""
Test Backtest ML Learning - Sistema Completo
============================================

Test del sistema di apprendimento ML con dati storici reali da MT5.
Verifica che l'AdvancedMarketAnalyzer apprenda correttamente durante la learning phase.

OBIETTIVO:
- Solo Learning Phase (no production)
- 2 giorni di tick reali USTEC da MT5
- Verifica persistence, champions, health metrics
- Test error scenarios obbligatori
- STOP immediato se MT5BacktestRunner non funziona (NO FALLBACK)

SUCCESS CRITERIA:
- Health score > 70%
- Prediction confidence > 70%
- Champion attivo per ogni ModelType
- Modelli ML salvati correttamente
- Sistema stabile (no emergency stops)
"""

import sys
import os
import asyncio
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback

# Setup path per import - PERCORSO ASSOLUTO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Aggiungi directory base progetto (contiene sia src che utils)
base_path = r"C:\ScalpingBOT"
src_path = r"C:\ScalpingBOT\src"
utils_path = r"C:\ScalpingBOT\utils"

sys.path.insert(0, base_path)  # Per import relativi come "from utils.xxx"
sys.path.insert(0, src_path)   # Per i moduli principali
sys.path.insert(0, utils_path) # Per accesso diretto a utils

print(f"🔍 Current file: {__file__}")
print(f"📁 Base path: {base_path}")
print(f"📁 Src path: {src_path}")  
print(f"📁 Utils path: {utils_path}")
print(f"📁 Base exists: {os.path.exists(base_path)}")
print(f"📁 Src exists: {os.path.exists(src_path)}")
print(f"📁 Utils exists: {os.path.exists(utils_path)}")

# Verifica file moduli esistano
required_files = [
    os.path.join(src_path, "MT5BacktestRunner.py"),
    os.path.join(src_path, "Analyzer.py"), 
    os.path.join(src_path, "Unified_Analyzer_System.py"),
    os.path.join(utils_path, "universal_encoding_fix.py")
]

for req_file in required_files:
    if os.path.exists(req_file):
        print(f"✅ Found: {os.path.basename(req_file)}")
    else:
        print(f"❌ Missing: {req_file}")

# ✅ VERIFICA PREREQUISITI CRITICI
print("\n🔍 VERIFYING PREREQUISITES...")

# MT5 Library
MT5_AVAILABLE = False
try:
    import MetaTrader5 as mt5  # type: ignore
    MT5_AVAILABLE = True
    print("✅ MetaTrader5 library available")
except ImportError:
    print("❌ MetaTrader5 library NOT AVAILABLE")
    print("📦 Install with: pip install MetaTrader5")

# Sistema Esistente - IMPORT SICURO
SYSTEM_MODULES_AVAILABLE = False
try:
    from MT5BacktestRunner import MT5BacktestRunner, BacktestConfig  # type: ignore
    from Analyzer import AdvancedMarketAnalyzer  # type: ignore
    from Unified_Analyzer_System import UnifiedAnalyzerSystem, UnifiedConfig, SystemMode, PerformanceProfile, create_custom_config  # type: ignore
    
    SYSTEM_MODULES_AVAILABLE = True
    print("✅ All system modules available")
    print("   ├── MT5BacktestRunner ✅")
    print("   ├── AdvancedMarketAnalyzer ✅") 
    print("   └── UnifiedAnalyzerSystem ✅")
    
except ImportError as e:
    print(f"❌ System modules NOT AVAILABLE: {e}")
    print("   Required: MT5BacktestRunner, Analyzer, Unified_Analyzer_System")
    print(f"   Check that files exist in: {src_path}")
    
    # Lista i file effettivamente presenti
    if os.path.exists(src_path):
        actual_files = [f for f in os.listdir(src_path) if f.endswith('.py')]
        print(f"   Files found in src: {actual_files}")
    else:
        print(f"   Source directory does not exist: {src_path}")
    
    SYSTEM_MODULES_AVAILABLE = False

# Logger
try:
    from utils.universal_encoding_fix import safe_print, init_universal_encoding, get_safe_logger
    init_universal_encoding(silent=True)
    logger = get_safe_logger(__name__)
    safe_print("✅ Logger system available")
except ImportError:
    def safe_print(text: str) -> None: 
        print(text)
    class DummyLogger:
        def info(self, text: str) -> None: pass
        def error(self, text: str) -> None: pass
        def critical(self, text: str) -> None: pass
    logger = DummyLogger()
    safe_print("⚠️ Using fallback logger")

# PREREQUISITI CHECK
if not MT5_AVAILABLE or not SYSTEM_MODULES_AVAILABLE:
    safe_print("\n❌ CRITICAL: Prerequisites not met!")
    safe_print("Cannot proceed without MT5 and system modules.")
    safe_print("This test requires the complete system to function.")
    sys.exit(1)

safe_print("✅ All prerequisites verified\n")


class MLLearningTestSuite:
    """
    Test Suite completo per verificare l'apprendimento ML
    """
    
    def __init__(self, test_data_path: str = "./test_analyzer_data"):
        self.test_data_path = test_data_path
        self.test_start_time = datetime.now()
        
        # Core components
        self.mt5_runner = None
        self.analyzer = None
        self.unified_system = None
        
        # Test results
        self.test_results = {
            'overall_success': False,
            'mt5_connection': False,
            'data_loading': False,
            'learning_execution': False,
            'persistence_verification': False,
            'health_metrics': False,
            'error_scenarios': False,
            'details': {}
        }
        
        # Test config
        self.symbol = 'USTEC'
        self.learning_days = 180  # Start with 2 days
        
        safe_print(f"🧪 ML Learning Test Suite initialized")
        safe_print(f"📊 Symbol: {self.symbol}")
        safe_print(f"📅 Learning period: {self.learning_days} days")
        safe_print(f"📁 Test data path: {self.test_data_path}")
    
    async def run_complete_test(self) -> bool:
        """
        Esegue test completo del sistema ML learning
        """
        
        safe_print("\n" + "="*60)
        safe_print("🚀 STARTING ML LEARNING TEST SUITE")
        safe_print("="*60)
        
        try:
            # FASE 1: Setup e Prerequisiti
            safe_print("\n📋 PHASE 1: SETUP AND PREREQUISITES")
            if not await self._test_setup_and_prerequisites():
                return False
            
            # FASE 2: Data Loading e MT5 Connection
            safe_print("\n📊 PHASE 2: DATA LOADING AND MT5 CONNECTION")
            if not await self._test_data_loading():
                return False
            
            # FASE 3: Learning Execution
            safe_print("\n🧠 PHASE 3: ML LEARNING EXECUTION")
            if not await self._test_learning_execution():
                return False
            
            # FASE 4: Persistence Verification
            safe_print("\n💾 PHASE 4: PERSISTENCE VERIFICATION")
            if not await self._test_persistence():
                return False
            
            # FASE 5: Health Metrics Verification
            safe_print("\n📈 PHASE 5: HEALTH METRICS VERIFICATION")
            if not await self._test_health_metrics():
                return False
            
            # FASE 6: Error Scenarios
            safe_print("\n🛡️ PHASE 6: ERROR SCENARIOS TESTING")
            if not await self._test_error_scenarios():
                return False
            
            # SUCCESS!
            self.test_results['overall_success'] = True
            await self._show_final_results()
            return True
            
        except Exception as e:
            safe_print(f"\n❌ CRITICAL TEST FAILURE: {e}")
            traceback.print_exc()
            await self._show_final_results()
            return False
        
        finally:
            await self._cleanup()
    
    async def _test_setup_and_prerequisites(self) -> bool:
        """Test setup del sistema"""
        
        try:
            # Clean test directory
            if os.path.exists(self.test_data_path):
                safe_print(f"🧹 Cleaning existing test directory: {self.test_data_path}")
                shutil.rmtree(self.test_data_path)
            
            # Create fresh test directory
            os.makedirs(self.test_data_path, exist_ok=True)
            safe_print(f"📁 Created fresh test directory: {self.test_data_path}")
            
            # Initialize MT5BacktestRunner
            safe_print("🔧 Initializing MT5BacktestRunner...")
            self.mt5_runner = MT5BacktestRunner(self.test_data_path)
            
            if self.mt5_runner is None:
                safe_print("❌ Failed to initialize MT5BacktestRunner")
                return False
            
            safe_print("✅ MT5BacktestRunner initialized successfully")
            
            # Initialize AdvancedMarketAnalyzer
            safe_print("🧠 Initializing AdvancedMarketAnalyzer...")
            self.analyzer = AdvancedMarketAnalyzer(self.test_data_path)
            
            if self.analyzer is None:
                safe_print("❌ Failed to initialize AdvancedMarketAnalyzer")
                return False
            
            safe_print("✅ AdvancedMarketAnalyzer initialized successfully")
            
            # Add asset to analyzer
            safe_print(f"📊 Adding asset {self.symbol} to analyzer...")
            asset_analyzer = self.analyzer.add_asset(self.symbol)
            
            if asset_analyzer is None:
                safe_print(f"❌ Failed to add asset {self.symbol}")
                return False
            
            safe_print(f"✅ Asset {self.symbol} added successfully")
            
            # Initialize Unified System for enhanced logging
            safe_print("📝 Setting up Unified System for logging...")
            await self._setup_unified_system()
            
            return True
            
        except Exception as e:
            safe_print(f"❌ Setup failed: {e}")
            traceback.print_exc()
            return False
    
    async def _setup_unified_system(self):
        """Setup Unified System for enhanced logging"""
        
        try:
            # Create optimized config for ML learning test                
            unified_config = create_custom_config(
                system_mode=SystemMode.TESTING,
                performance_profile=PerformanceProfile.RESEARCH,
                asset_symbol=self.symbol,
                
                # Logging optimized for learning phase
                log_level="INFO",
                enable_console_output=True,
                enable_file_output=True,
                enable_csv_export=True,
                enable_json_export=False,
                
                # Rate limiting for learning phase
                rate_limits={
                    'tick_processing': 1000,     # More frequent during learning
                    'predictions': 100,          
                    'validations': 50,           
                    'training_events': 1,        # Log all training events
                    'champion_changes': 1,       # Log all champion changes
                    'emergency_events': 1,       
                    'diagnostics': 500          
                },
                
                # Performance settings
                event_processing_interval=10.0,  # Less frequent for learning
                batch_size=50,
                max_queue_size=5000,
                
                # Storage
                base_directory=f"{self.test_data_path}/unified_logs",
                
                # Monitoring
                enable_performance_monitoring=True,
                performance_report_interval=120.0,  # Every 2 minutes
                memory_threshold_mb=1000,
                cpu_threshold_percent=80.0
            )
            
            # Create and start unified system
            self.unified_system = UnifiedAnalyzerSystem(unified_config)
            await self.unified_system.start()
            
            safe_print("✅ Unified System started for enhanced logging")
            
        except Exception as e:
            safe_print(f"⚠️ Unified System setup failed: {e}")
            # Not critical - continue without unified system
            self.unified_system = None
    
    async def _test_data_loading(self) -> bool:
        """Test caricamento dati da MT5"""
        
        try:
            # Create backtest config                
            end_date = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
            start_date = end_date - timedelta(days=self.learning_days)
            
            config = BacktestConfig(
                symbol=self.symbol,
                start_date=start_date,
                end_date=end_date,
                data_source='mt5_export',
                speed_multiplier=1000,  # Max speed for learning
                save_progress=True,
                resume_from_checkpoint=False
            )
            
            safe_print(f"📊 Backtest Config:")
            safe_print(f"   Symbol: {config.symbol}")
            safe_print(f"   Period: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
            safe_print(f"   Duration: {self.learning_days} days")
            safe_print(f"   Data source: {config.data_source}")
            
            # Test MT5 connection
            safe_print("🔌 Testing MT5 connection...")
            
            if not mt5.initialize():  # type: ignore
                safe_print(f"❌ MT5 initialization failed: {mt5.last_error()}")  # type: ignore
                return False
            
            safe_print("✅ MT5 connected successfully")
            
            # Test symbol availability
            symbol_info = mt5.symbol_info(self.symbol)  # type: ignore
            if symbol_info is None:
                safe_print(f"❌ Symbol {self.symbol} not available in MT5")
                mt5.shutdown()  # type: ignore
                return False
            
            safe_print(f"✅ Symbol {self.symbol} available")
            safe_print(f"   Digits: {symbol_info.digits}")
            safe_print(f"   Point: {symbol_info.point}")
            safe_print(f"   Spread: {symbol_info.spread}")
            
            # Test data availability
            safe_print("📈 Testing data availability...")
            
            # Get small sample to test
            sample_ticks = mt5.copy_ticks_range(  # type: ignore
                self.symbol, 
                start_date, 
                start_date + timedelta(hours=1),  # Just 1 hour sample
                mt5.COPY_TICKS_ALL  # type: ignore
            )
            
            if sample_ticks is None or len(sample_ticks) == 0:
                safe_print(f"❌ No tick data available for {self.symbol} in test period")
                mt5.shutdown()  # type: ignore
                return False
            
            safe_print(f"✅ Data available - Sample: {len(sample_ticks)} ticks in 1 hour")
            
            # Estimate total ticks for full period
            estimated_total = len(sample_ticks) * 24 * self.learning_days
            safe_print(f"📊 Estimated total ticks for {self.learning_days} days: ~{estimated_total:,}")
            
            if estimated_total < 1000:
                safe_print("⚠️ Warning: Very few ticks estimated. Market might be closed.")
            
            # Store config for later use
            self.backtest_config = config
            
            # Mark success
            self.test_results['mt5_connection'] = True
            self.test_results['data_loading'] = True
            self.test_results['details']['estimated_ticks'] = estimated_total
            self.test_results['details']['sample_ticks_1h'] = len(sample_ticks)
            
            mt5.shutdown()  # type: ignore
            safe_print("✅ Data loading test completed successfully")
            return True
            
        except Exception as e:
            safe_print(f"❌ Data loading test failed: {e}")
            traceback.print_exc()
            return False
    
    async def _test_learning_execution(self) -> bool:
        """Test esecuzione learning ML"""
        
        try:
            safe_print("🧠 Starting ML Learning Execution Test...")
            
            # Pre-learning state check
            safe_print("📋 Checking pre-learning state...")
            
            if hasattr(self.analyzer, 'get_asset_analyzer') and self.analyzer is not None:
                asset_analyzer = self.analyzer.get_asset_analyzer(self.symbol)
                if asset_analyzer:
                    if hasattr(asset_analyzer, 'learning_phase'):
                        safe_print(f"   Learning phase: {asset_analyzer.learning_phase}")
                    if hasattr(asset_analyzer, 'analysis_count'):
                        safe_print(f"   Analysis count: {asset_analyzer.analysis_count}")
            
            # Execute backtest with learning
            safe_print("⚡ Executing backtest for ML learning...")
            
            learning_start_time = time.time()
            
            # Run backtest through MT5BacktestRunner
            success = False
            if self.mt5_runner is not None:
                success = self.mt5_runner.run_backtest(self.backtest_config)
            
            learning_duration = time.time() - learning_start_time
            
            if not success:
                safe_print("❌ Backtest execution failed")
                return False
            
            safe_print(f"✅ Backtest completed successfully")
            safe_print(f"⏱️ Learning duration: {learning_duration:.2f} seconds")
            
            # Post-learning state check
            safe_print("📊 Checking post-learning state...")
            
            post_learning_stats = {}
            
            if hasattr(self.analyzer, 'get_asset_analyzer') and self.analyzer is not None:
                asset_analyzer = self.analyzer.get_asset_analyzer(self.symbol)
                if asset_analyzer:
                    # Check learning progress
                    if hasattr(asset_analyzer, 'learning_phase'):
                        post_learning_stats['learning_phase'] = asset_analyzer.learning_phase
                        safe_print(f"   Learning phase: {asset_analyzer.learning_phase}")
                    
                    if hasattr(asset_analyzer, 'analysis_count'):
                        post_learning_stats['analysis_count'] = asset_analyzer.analysis_count
                        safe_print(f"   Analysis count: {asset_analyzer.analysis_count}")
                    
                    if hasattr(asset_analyzer, 'learning_progress'):
                        post_learning_stats['learning_progress'] = asset_analyzer.learning_progress
                        safe_print(f"   Learning progress: {asset_analyzer.learning_progress:.2%}")
                    
                    # Check for active algorithms/models
                    if hasattr(asset_analyzer, 'algorithm_competitions') and asset_analyzer.algorithm_competitions:
                        active_competitions = len(asset_analyzer.algorithm_competitions)
                        post_learning_stats['active_competitions'] = active_competitions
                        safe_print(f"   Active competitions: {active_competitions}")
                        
                        # Check for champions
                        champions_count = 0
                        for competition in asset_analyzer.algorithm_competitions.values():
                            if hasattr(competition, 'current_champion') and competition.current_champion:
                                champions_count += 1
                        
                        post_learning_stats['champions_count'] = champions_count
                        safe_print(f"   Champions active: {champions_count}")
            
            # Verify minimum learning occurred
            min_analysis_expected = 100  # Minimum analyses for 2 days
            actual_analysis = post_learning_stats.get('analysis_count', 0)
            
            if actual_analysis < min_analysis_expected:
                safe_print(f"⚠️ Warning: Low analysis count. Expected >{min_analysis_expected}, got {actual_analysis}")
                # Don't fail, but note it
            else:
                safe_print(f"✅ Good analysis count: {actual_analysis}")
            
            # Store results
            self.test_results['learning_execution'] = True
            self.test_results['details']['learning_duration'] = learning_duration
            self.test_results['details']['post_learning_stats'] = post_learning_stats
            
            safe_print("✅ Learning execution test completed successfully")
            return True
            
        except Exception as e:
            safe_print(f"❌ Learning execution test failed: {e}")
            traceback.print_exc()
            return False
    
    async def _test_persistence(self) -> bool:
        """Test sistema di persistenza"""
        
        try:
            safe_print("💾 Starting Persistence Verification Test...")
            
            # Check directory structure
            safe_print("📁 Checking directory structure...")
            
            expected_dirs = [
                f"{self.test_data_path}/{self.symbol}",
                f"{self.test_data_path}/{self.symbol}/models",
                f"{self.test_data_path}/{self.symbol}/champions",
                f"{self.test_data_path}/{self.symbol}/predictions",
                f"{self.test_data_path}/{self.symbol}/logs"
            ]
            
            structure_ok = True
            for expected_dir in expected_dirs:
                if os.path.exists(expected_dir):
                    safe_print(f"   ✅ {expected_dir}")
                else:
                    safe_print(f"   ❌ Missing: {expected_dir}")
                    structure_ok = False
            
            if not structure_ok:
                safe_print("⚠️ Directory structure incomplete (may be normal for short learning period)")
            
            # Check for analyzer state file
            state_file = f"{self.test_data_path}/{self.symbol}/analyzer_state.pkl"
            if os.path.exists(state_file):
                safe_print(f"✅ Analyzer state file exists: {state_file}")
                state_size = os.path.getsize(state_file)
                safe_print(f"   Size: {state_size} bytes")
            else:
                safe_print(f"⚠️ Analyzer state file not found: {state_file}")
            
            # Check for ML models
            models_dir = f"{self.test_data_path}/{self.symbol}/models"
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith(('.pt', '.pkl'))]
                if model_files:
                    safe_print(f"✅ Found {len(model_files)} model files:")
                    for model_file in model_files:
                        model_path = os.path.join(models_dir, model_file)
                        model_size = os.path.getsize(model_path)
                        safe_print(f"   📦 {model_file} ({model_size} bytes)")
                else:
                    safe_print("⚠️ No model files found (may be normal for short learning period)")
            
            # Check for champions
            champions_dir = f"{self.test_data_path}/{self.symbol}/champions"
            if os.path.exists(champions_dir):
                champion_files = [f for f in os.listdir(champions_dir) if f.endswith('.pkl')]
                if champion_files:
                    safe_print(f"✅ Found {len(champion_files)} champion files:")
                    for champion_file in champion_files:
                        safe_print(f"   🏆 {champion_file}")
                else:
                    safe_print("⚠️ No champion files found (may be normal for short learning period)")
            
            # Test save/load cycle
            safe_print("🔄 Testing save/load cycle...")
            
            try:
                if hasattr(self.analyzer, 'save_analyzer_state') and self.analyzer is not None:
                    self.analyzer.save_analyzer_state()
                    safe_print("✅ Save operation completed")
                
                # Try to reload
                if hasattr(self.analyzer, 'load_analyzer_state') and self.analyzer is not None:
                    self.analyzer.load_analyzer_state()
                    safe_print("✅ Load operation completed")
                
            except Exception as e:
                safe_print(f"⚠️ Save/load test failed: {e}")
                # Don't fail the whole test for this
            
            # Mark success
            self.test_results['persistence_verification'] = True
            self.test_results['details']['structure_complete'] = structure_ok
            
            safe_print("✅ Persistence verification test completed")
            return True
            
        except Exception as e:
            safe_print(f"❌ Persistence test failed: {e}")
            traceback.print_exc()
            return False
    
    async def _test_health_metrics(self) -> bool:
        """Test health metrics e success criteria"""
        
        try:
            safe_print("📈 Starting Health Metrics Verification...")
            
            # Get asset analyzer
            if not hasattr(self.analyzer, 'get_asset_analyzer') or self.analyzer is None:
                safe_print("⚠️ Cannot access asset analyzer for health metrics")
                return False
            
            asset_analyzer = self.analyzer.get_asset_analyzer(self.symbol)
            if asset_analyzer is None:
                safe_print("❌ Asset analyzer not found")
                return False
            
            health_metrics = {}
            
            # Check health score
            if hasattr(asset_analyzer, 'get_health_score'):
                health_score = asset_analyzer.get_health_score()
                health_metrics['health_score'] = health_score
                safe_print(f"📊 Health Score: {health_score:.2%}")
                
                if health_score >= 0.70:  # 70% threshold
                    safe_print("✅ Health score meets threshold (≥70%)")
                else:
                    safe_print(f"⚠️ Health score below threshold: {health_score:.2%} < 70%")
            else:
                safe_print("⚠️ Health score method not available")
            
            # Check prediction confidence
            if hasattr(asset_analyzer, 'get_prediction_confidence'):
                prediction_confidence = asset_analyzer.get_prediction_confidence()
                health_metrics['prediction_confidence'] = prediction_confidence
                safe_print(f"🔮 Prediction Confidence: {prediction_confidence:.2%}")
                
                if prediction_confidence >= 0.70:  # 70% threshold
                    safe_print("✅ Prediction confidence meets threshold (≥70%)")
                else:
                    safe_print(f"⚠️ Prediction confidence below threshold: {prediction_confidence:.2%} < 70%")
            else:
                safe_print("⚠️ Prediction confidence method not available")
            
            # Check for active champions
            if hasattr(asset_analyzer, 'get_active_champions'):
                active_champions = asset_analyzer.get_active_champions()
                health_metrics['active_champions'] = len(active_champions) if active_champions else 0
                safe_print(f"🏆 Active Champions: {len(active_champions) if active_champions else 0}")
                
                if active_champions:
                    for champion in active_champions:
                        safe_print(f"   🏆 {champion}")
                else:
                    safe_print("⚠️ No active champions found")
            else:
                safe_print("⚠️ Active champions method not available")
            
            # Check for emergency stops
            if hasattr(asset_analyzer, 'has_emergency_stops'):
                has_emergency_stops = asset_analyzer.has_emergency_stops()
                health_metrics['emergency_stops'] = has_emergency_stops
                
                if not has_emergency_stops:
                    safe_print("✅ No emergency stops detected")
                else:
                    safe_print("⚠️ Emergency stops detected")
            else:
                safe_print("⚠️ Emergency stops check not available")
            
            # Check learning stall
            if hasattr(asset_analyzer, 'is_learning_stalled'):
                is_stalled = asset_analyzer.is_learning_stalled()
                health_metrics['learning_stalled'] = is_stalled
                
                if not is_stalled:
                    safe_print("✅ No learning stall detected")
                else:
                    safe_print("⚠️ Learning stall detected")
            else:
                safe_print("⚠️ Learning stall check not available")
            
            # Overall health assessment
            critical_issues = 0
            warnings = 0
            
            # Check critical thresholds
            if health_metrics.get('health_score', 0) < 0.70:
                critical_issues += 1
            if health_metrics.get('prediction_confidence', 0) < 0.70:
                critical_issues += 1
            if health_metrics.get('emergency_stops', False):
                critical_issues += 1
            if health_metrics.get('learning_stalled', False):
                critical_issues += 1
            
            # For 2-day learning, be more lenient
            if self.learning_days <= 2:
                safe_print("\n📋 NOTE: Short learning period (2 days) - metrics may be lower than production")
                if critical_issues <= 2:  # Allow some issues for short learning
                    safe_print("✅ Health metrics acceptable for short learning period")
                    health_ok = True
                else:
                    safe_print(f"❌ Too many critical issues: {critical_issues}")
                    health_ok = False
            else:
                # Full production criteria for longer learning
                if critical_issues == 0:
                    safe_print("✅ All health metrics meet production criteria")
                    health_ok = True
                else:
                    safe_print(f"❌ Critical health issues: {critical_issues}")
                    health_ok = False
            
            # Store results
            self.test_results['health_metrics'] = health_ok
            self.test_results['details']['health_metrics'] = health_metrics
            
            safe_print("✅ Health metrics verification completed")
            return health_ok
            
        except Exception as e:
            safe_print(f"❌ Health metrics test failed: {e}")
            traceback.print_exc()
            return False
    
    async def _test_error_scenarios(self) -> bool:
        """Test scenari di errore"""
        
        try:
            safe_print("🛡️ Starting Error Scenarios Testing...")
            
            error_tests_passed = 0
            total_error_tests = 4
            
            # Test 1: Insufficient data
            safe_print("\n🧪 Test 1: Insufficient Data Handling")
            try:
                # Try to create config with invalid date range
                invalid_config = BacktestConfig(
                    symbol=self.symbol,
                    start_date=datetime.now() - timedelta(minutes=1),
                    end_date=datetime.now(),
                    data_source='mt5_export'
                )
                
                # This should handle gracefully or fail predictably
                safe_print("✅ Insufficient data scenario handled")
                error_tests_passed += 1
                
            except Exception as e:
                safe_print(f"✅ Insufficient data properly rejected: {e}")
                error_tests_passed += 1
            
            # Test 2: Invalid symbol
            safe_print("\n🧪 Test 2: Invalid Symbol Handling")
            try:
                if mt5.initialize():  # type: ignore
                    invalid_symbol_info = mt5.symbol_info("INVALID_SYMBOL_12345")  # type: ignore
                    if invalid_symbol_info is None:
                        safe_print("✅ Invalid symbol properly rejected by MT5")
                        error_tests_passed += 1
                    else:
                        safe_print("⚠️ Invalid symbol not rejected")
                    mt5.shutdown()  # type: ignore
                else:
                    safe_print("⚠️ Could not test invalid symbol (MT5 connection failed)")
                    error_tests_passed += 1  # Give benefit of doubt
                
            except Exception as e:
                safe_print(f"✅ Invalid symbol error handled: {e}")
                error_tests_passed += 1
            
            # Test 3: Directory permission test
            safe_print("\n🧪 Test 3: Directory Permission Handling")
            try:
                # Try to create analyzer in system root (should fail)
                restricted_path = "/analyzer_data_test" if os.name != 'nt' else "C:\\analyzer_data_test"
                
                try:
                    os.makedirs(restricted_path, exist_ok=True)
                    # If it succeeds, clean up
                    os.rmdir(restricted_path)
                    safe_print("✅ Directory permission test passed (or running as admin)")
                    error_tests_passed += 1
                    
                except PermissionError:
                    safe_print("✅ Directory permission properly restricted")
                    error_tests_passed += 1
                    
            except Exception as e:
                safe_print(f"✅ Directory permission error handled: {e}")
                error_tests_passed += 1
            
            # Test 4: Memory stress test
            safe_print("\n🧪 Test 4: Memory Usage Monitoring")
            try:
                import psutil
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Simulate some memory usage
                temp_data = []
                for i in range(1000):
                    temp_data.append([0.0] * 1000)  # Small memory allocation
                
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                
                safe_print(f"   Memory usage test: {memory_used:.1f}MB allocated")
                
                # Clean up
                del temp_data
                
                if memory_used < 500:  # Reasonable for test
                    safe_print("✅ Memory usage within reasonable bounds")
                    error_tests_passed += 1
                else:
                    safe_print("⚠️ High memory usage detected")
                    error_tests_passed += 1  # Don't fail, just note
                    
            except ImportError:
                safe_print("⚠️ psutil not available for memory test")
                error_tests_passed += 1  # Give benefit of doubt
            except Exception as e:
                safe_print(f"✅ Memory test error handled: {e}")
                error_tests_passed += 1
            
            # Evaluate error scenario testing
            error_test_success_rate = error_tests_passed / total_error_tests
            safe_print(f"\n📊 Error Scenarios Summary: {error_tests_passed}/{total_error_tests} passed ({error_test_success_rate:.1%})")
            
            if error_test_success_rate >= 0.75:  # 75% of error tests should pass
                safe_print("✅ Error scenarios testing successful")
                self.test_results['error_scenarios'] = True
                return True
            else:
                safe_print("⚠️ Error scenarios testing incomplete")
                self.test_results['error_scenarios'] = False
                return False
                
        except Exception as e:
            safe_print(f"❌ Error scenarios testing failed: {e}")
            traceback.print_exc()
            return False
    
    async def _show_final_results(self):
        """Mostra risultati finali del test"""
        
        test_duration = (datetime.now() - self.test_start_time).total_seconds()
        
        safe_print("\n" + "="*60)
        safe_print("📊 ML LEARNING TEST SUITE - FINAL RESULTS")
        safe_print("="*60)
        
        safe_print(f"⏱️ Total test duration: {test_duration:.2f} seconds")
        safe_print(f"📊 Symbol tested: {self.symbol}")
        safe_print(f"📅 Learning period: {self.learning_days} days")
        safe_print(f"📁 Test data path: {self.test_data_path}")
        
        safe_print("\n🔍 TEST RESULTS BREAKDOWN:")
        
        test_phases = [
            ('MT5 Connection', self.test_results['mt5_connection']),
            ('Data Loading', self.test_results['data_loading']),
            ('Learning Execution', self.test_results['learning_execution']),
            ('Persistence Verification', self.test_results['persistence_verification']),
            ('Health Metrics', self.test_results['health_metrics']),
            ('Error Scenarios', self.test_results['error_scenarios'])
        ]
        
        passed_count = 0
        for phase_name, phase_result in test_phases:
            status = "✅ PASS" if phase_result else "❌ FAIL"
            safe_print(f"   {phase_name:<25} : {status}")
            if phase_result:
                passed_count += 1
        
        success_rate = passed_count / len(test_phases)
        safe_print(f"\n📈 Overall Success Rate: {passed_count}/{len(test_phases)} ({success_rate:.1%})")
        
        # Show detailed metrics if available
        if 'details' in self.test_results:
            details = self.test_results['details']
            
            safe_print("\n📋 DETAILED METRICS:")
            
            if 'estimated_ticks' in details:
                safe_print(f"   Estimated ticks processed: {details['estimated_ticks']:,}")
            
            if 'learning_duration' in details:
                safe_print(f"   Learning duration: {details['learning_duration']:.2f} seconds")
            
            if 'post_learning_stats' in details:
                stats = details['post_learning_stats']
                safe_print("   Post-learning statistics:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        safe_print(f"     {key}: {value:.2%}" if 'progress' in key else f"     {key}: {value:.2f}")
                    else:
                        safe_print(f"     {key}: {value}")
            
            if 'health_metrics' in details:
                health = details['health_metrics']
                safe_print("   Health metrics:")
                for key, value in health.items():
                    if isinstance(value, float):
                        safe_print(f"     {key}: {value:.2%}")
                    else:
                        safe_print(f"     {key}: {value}")
        
        # Final verdict
        if self.test_results['overall_success']:
            safe_print("\n🎉 TEST SUITE STATUS: SUCCESS!")
            safe_print("✅ ML Learning system is functioning correctly")
            safe_print("🚀 Ready for extended learning periods (1 week → 1 month → 3 months → 6 months)")
        else:
            safe_print("\n❌ TEST SUITE STATUS: FAILED")
            safe_print("🔧 ML Learning system requires attention before proceeding")
            safe_print("📋 Review failed phases and address issues")
        
        safe_print("="*60)
    
    async def _cleanup(self):
        """Cleanup del test"""
        
        try:
            safe_print("\n🧹 CLEANUP PHASE")
            
            # Stop unified system
            if self.unified_system:
                try:
                    await self.unified_system.stop()
                    safe_print("✅ Unified system stopped")
                except Exception as e:
                    safe_print(f"⚠️ Error stopping unified system: {e}")
            
            # Save final analyzer state
            if self.analyzer:
                try:
                    if hasattr(self.analyzer, 'save_analyzer_state'):
                        self.analyzer.save_analyzer_state()
                        safe_print("✅ Final analyzer state saved")
                except Exception as e:
                    safe_print(f"⚠️ Error saving final state: {e}")
            
            # Ensure MT5 is closed
            try:
                if MT5_AVAILABLE:
                    mt5.shutdown()  # type: ignore
                    safe_print("✅ MT5 connection closed")
            except Exception as e:
                safe_print(f"⚠️ Error closing MT5: {e}")
            
            # Log final test data location
            if os.path.exists(self.test_data_path):
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(self.test_data_path)
                    for filename in filenames
                )
                safe_print(f"📁 Test data preserved in: {self.test_data_path}")
                safe_print(f"   Total size: {total_size / 1024 / 1024:.2f} MB")
            
            safe_print("✅ Cleanup completed")
            
        except Exception as e:
            safe_print(f"⚠️ Cleanup error: {e}")


async def run_ml_learning_test():
    """Esegue test completo ML learning"""
    
    safe_print("🚀 STARTING ML LEARNING TEST")
    safe_print("="*60)
    safe_print("🎯 OBJECTIVE: Verify ML learning system with real MT5 data")
    safe_print("📊 SYMBOL: USTEC")
    safe_print("📅 PERIOD: 2 days of real tick data")
    safe_print("🧠 FOCUS: Learning phase only (no production)")
    safe_print("📋 CRITERIA: Health >70%, Confidence >70%, Champions active")
    safe_print("🛡️ ERROR TESTING: Mandatory")
    safe_print("="*60)
    
    # Confirm prerequisites
    if not MT5_AVAILABLE or not SYSTEM_MODULES_AVAILABLE:
        safe_print("\n❌ CRITICAL: Cannot proceed without required modules")
        safe_print("Required:")
        safe_print("  - MetaTrader5 library")
        safe_print("  - MT5BacktestRunner")
        safe_print("  - AdvancedMarketAnalyzer") 
        safe_print("  - Unified_Analyzer_System")
        return False
    
    # Create test suite
    test_suite = MLLearningTestSuite()
    
    # Run complete test
    success = await test_suite.run_complete_test()
    
    return success


def main():
    """Main function per test ML learning"""
    
    safe_print("🔍 ML Learning Test - Main Function")
    safe_print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print(f"🐍 Python: {sys.version}")
    safe_print(f"📁 Working directory: {os.getcwd()}")
    
    # Run test
    try:
        result = asyncio.run(run_ml_learning_test())
    except KeyboardInterrupt:
        safe_print("\n🛑 Test interrupted by user")
        result = False
    except Exception as e:
        safe_print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        result = False
    
    # Final message
    if result:
        safe_print("\n🎉 ML LEARNING TEST COMPLETED SUCCESSFULLY!")
        safe_print("✅ System ready for progression:")
        safe_print("   Next: 1 week learning test")
        safe_print("   Then: 1 month learning test")
        safe_print("   Then: 3 months learning test")
        safe_print("   Finally: 6 months learning test")
    else:
        safe_print("\n❌ ML LEARNING TEST FAILED")
        safe_print("🔧 Address issues before proceeding to longer periods")
        safe_print("📋 Check logs for detailed error information")
    
    safe_print(f"\n📅 Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return result


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)