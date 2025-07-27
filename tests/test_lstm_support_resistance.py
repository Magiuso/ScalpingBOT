#!/usr/bin/env python3
"""
Test LSTM Support/Resistance Model
==================================
Test individuale per LSTM_SupportResistance usando dati reali dal backtest
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import asyncio
from datetime import datetime
import torch
import traceback
from pathlib import Path

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities first
from utils.universal_encoding_fix import safe_print, init_universal_encoding

# Import core system modules - IDENTICAL to test_backtest approach
try:
    from src.Analyzer import AdvancedMarketAnalyzer  # type: ignore
    safe_print("✅ Core AdvancedMarketAnalyzer available")
    ANALYZER_AVAILABLE = True
except ImportError as e:
    safe_print(f"❌ Core AdvancedMarketAnalyzer NOT AVAILABLE: {e}")
    ANALYZER_AVAILABLE = False

# Import Unified System - IDENTICAL to test_backtest
from src.Unified_Analyzer_System import UnifiedAnalyzerSystem, create_custom_config, SystemMode, PerformanceProfile

safe_print("✅ UnifiedAnalyzerSystem loaded successfully - REAL SYSTEM ONLY")

# Initialize encoding
init_universal_encoding(silent=True)

class LSTMSupportResistanceTester:
    """Test specifico per LSTM_SupportResistance usando IDENTICAL struttura test_backtest"""
    
    def __init__(self):
        self.test_data_path = "./test_analyzer_data"
        self.symbol = "USTEC"
        self.model_name = "LSTM_SupportResistance"
        self.results = {}
        
        # IDENTICAL to test_backtest
        self.unified_system = None
        self.analyzer = None
        
    def load_real_data(self, limit: int = 100000):
        """Carica dati reali dal file backtest"""
        safe_print(f"📂 Loading real backtest data from {self.test_data_path}")
        
        # Trova il file di backtest più recente
        backtest_files = [f for f in os.listdir(self.test_data_path) 
                         if f.startswith(f'backtest_{self.symbol}') and f.endswith('.jsonl')]
        
        if not backtest_files:
            raise FileNotFoundError(f"No backtest files found for {self.symbol}")
            
        latest_file = sorted(backtest_files)[-1]
        file_path = os.path.join(self.test_data_path, latest_file)
        safe_print(f"📄 Using file: {latest_file}")
        
        # Carica i tick dal file
        ticks = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= limit + 1:  # +1 per skip header
                    break
                    
                try:
                    data = json.loads(line)
                    if data.get('type') == 'tick':
                        ticks.append({
                            'timestamp': data['timestamp'],
                            'bid': data['bid'],
                            'ask': data['ask'],
                            'volume': data.get('volume', 0),
                            'spread': data.get('ask', 0) - data.get('bid', 0)
                        })
                except:
                    continue
                    
        safe_print(f"✅ Loaded {len(ticks)} ticks")
        return ticks
        
    def prepare_training_data(self, ticks):
        """Prepara i dati nel formato richiesto dal modello"""
        safe_print("🔄 Preparing training data...")
        
        # ✅ IDENTICAL CONFIG: Usa ESATTAMENTE la stessa configurazione di test_backtest
        analyzer_config = AnalyzerConfig(
            ml_logger_enabled=True,
            ml_logger_verbosity="verbose",
            ml_logger_terminal_mode="scroll"
        )
        analyzer = AdvancedMarketAnalyzer(data_path=self.test_data_path, config=analyzer_config)
        
        # Simula AssetAnalyzer per preparare i dati
        prices = np.array([t['bid'] for t in ticks])
        volumes = np.array([t['volume'] for t in ticks])
        
        # Calcola indicatori base
        from talib import SMA, RSI, BBANDS, ATR, MACD
        
        # SMA
        sma_20 = SMA(prices, timeperiod=20)
        sma_50 = SMA(prices, timeperiod=50)
        
        # RSI
        rsi = RSI(prices, timeperiod=14)
        
        # Bollinger Bands
        upper, middle, lower = BBANDS(prices, timeperiod=20)
        
        # ATR
        high = prices * 1.0001  # Simula high
        low = prices * 0.9999   # Simula low
        atr = ATR(high, low, prices, timeperiod=14)
        
        # MACD
        macd, signal, hist = MACD(prices)
        
        # Prepara il dizionario dati
        data = {
            'prices': prices,
            'volumes': volumes,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi': rsi,
            'bb_upper': upper,
            'bb_lower': lower,
            'atr': atr,
            'macd': macd,
            'macd_signal': signal
        }
        
        # Usa il metodo dell'analyzer per preparare dataset S/R
        try:
            X, y = analyzer._prepare_sr_dataset(data)
            safe_print(f"✅ Prepared data: X={X.shape}, y={y.shape}")
            return X, y
        except Exception as e:
            safe_print(f"❌ Error preparing dataset: {e}")
            # Fallback: create basic dataset manually
            safe_print("🔄 Using fallback dataset preparation...")
            
            # Basic feature extraction without complex indicators
            window_size = min(50, len(prices) // 20)
            X, y = [], []
            
            for i in range(window_size, len(prices) - 10):
                features = np.concatenate([
                    prices[i-window_size:i],
                    volumes[i-window_size:i] if len(volumes) > i else np.zeros(window_size)
                ])
                
                # Simple S/R targets
                future_prices = prices[i:i+10]
                support = np.min(future_prices)
                resistance = np.max(future_prices)
                
                if len(features) == window_size * 2:  # Basic validation
                    X.append(features)
                    y.append([support, resistance])
                    
            return np.array(X), np.array(y)
    
    def prepare_training_data_with_analyzer(self, analyzer, ticks):
        """Prepara training data usando l'analyzer esistente"""
        safe_print("🔄 Preparing training data with existing analyzer...")
        
        # Prepara i dati base dai tick
        prices = np.array([t['bid'] for t in ticks])
        volumes = np.array([t['volume'] for t in ticks])
        
        # Crea un asset analyzer temporaneo per usare i suoi metodi
        try:
            # Cerca se l'analyzer ha già asset analyzer
            if hasattr(analyzer, 'asset_analyzers') and len(analyzer.asset_analyzers) > 0:
                asset_analyzer = list(analyzer.asset_analyzers.values())[0]
            else:
                # Crea un mock asset per il test
                class MockAsset:
                    def __init__(self):
                        self.symbol = "USTEC"
                        
                # Simula la preparazione del dataset direttamente
                safe_print("🔄 Using direct dataset preparation...")
                return self.prepare_simple_sr_dataset(prices, volumes)
                
        except Exception as e:
            safe_print(f"⚠️ Using fallback preparation: {e}")
            return self.prepare_simple_sr_dataset(prices, volumes)
    
    def prepare_simple_sr_dataset(self, prices, volumes):
        """Prepara un dataset S/R semplificato"""
        window_size = 100
        X, y = [], []
        
        for i in range(window_size, len(prices) - 20):
            if i % 5000 == 0:
                safe_print(f"   Processing sample {i}/{len(prices)-20}...")
                
            # Features: prezzi recenti
            price_features = prices[i-window_size:i]
            
            # Normalizza
            price_mean = price_features.mean()
            price_std = price_features.std()
            if price_std > 0:
                price_features = (price_features - price_mean) / price_std
            
            # Padding to 400 features per compatibilità
            features = np.zeros(400)
            features[:len(price_features)] = price_features
            
            # Target S/R
            future_prices = prices[i:i+20]
            support = np.percentile(future_prices, 20)  # Come nel sistema reale
            resistance = np.percentile(future_prices, 80)
            
            # Normalizza target rispetto al prezzo corrente
            current_price = prices[i]
            support_norm = (support - current_price) / current_price
            resistance_norm = (resistance - current_price) / current_price
            
            X.append(features)
            y.append([support_norm, resistance_norm])
            
        safe_print(f"✅ Simple dataset prepared: X={np.array(X).shape}, y={np.array(y).shape}")
        return np.array(X), np.array(y)
        
    async def setup_unified_system(self):
        """Setup UnifiedAnalyzerSystem IDENTICAL to test_backtest"""
        safe_print("\n🔧 Setting up UnifiedAnalyzerSystem (IDENTICAL to test_backtest)...")
        
        try:
            # Create IDENTICAL config to test_backtest
            unified_config = create_custom_config(
                system_mode=SystemMode.TESTING,
                performance_profile=PerformanceProfile.RESEARCH,
                asset_symbol=self.symbol,
                
                # Learning phase specific settings (IDENTICAL)
                learning_phase_enabled=True,
                max_tick_buffer_size=100000,  # Increased buffer for 100K ticks
                min_learning_days=1,         # Reduced for test
                
                # Logging optimized for learning phase monitoring (IDENTICAL)
                log_level="VERBOSE",
                enable_console_output=True,
                enable_file_output=True,
                enable_csv_export=True,
                enable_json_export=True,
                
                # Rate limiting optimized for learning phase (IDENTICAL)
                rate_limits={
                    'process_tick': 500,
                    'predictions': 25,
                    'validations': 10,
                    'training_events': 1,
                    'champion_changes': 1,
                    'performance_metrics': 5
                }
            )
            
            # Create UnifiedAnalyzerSystem (IDENTICAL)
            self.unified_system = UnifiedAnalyzerSystem(unified_config)
            
            if not self.unified_system:
                raise Exception("Failed to create UnifiedAnalyzerSystem")
            
            # Start the unified system (IDENTICAL to test_backtest)
            await self.unified_system.start()
            safe_print("✅ UnifiedAnalyzerSystem started")
                
            # Get analyzer from unified system (IDENTICAL to test_backtest)
            self.analyzer = self.unified_system.analyzer
            
            if not self.analyzer:
                raise Exception("No analyzer in unified system")
                
            safe_print("✅ UnifiedAnalyzerSystem setup complete")
            safe_print(f"   - Analyzer: {self.analyzer.__class__.__name__}")
            safe_print(f"   - Asset: {self.symbol}")
            
            return True
            
        except Exception as e:
            safe_print(f"❌ UnifiedAnalyzerSystem setup failed: {e}")
            traceback.print_exc()
            return False

    async def _perform_ml_learning_phase(self):
        """IDENTICAL copy from test_backtest"""
        try:
            # Usa l'analyzer dell'unified system (AdvancedMarketAnalyzer)
            if self.unified_system and hasattr(self.unified_system, 'analyzer') and self.unified_system.analyzer:
                analyzer = self.unified_system.analyzer
                asset_symbol = getattr(self.unified_system.config, 'asset_symbol', self.symbol)
                
                safe_print("🔄 Executing intermediate ML learning...")
                # Per ogni asset nell'analyzer, esegui learning phase training
                if hasattr(analyzer, 'asset_analyzers') and analyzer.asset_analyzers and asset_symbol in analyzer.asset_analyzers:
                    asset_analyzer = analyzer.asset_analyzers[asset_symbol]
                    if asset_analyzer and hasattr(asset_analyzer, '_perform_learning_phase_training'):
                        asset_analyzer._perform_learning_phase_training()
                        safe_print(f"✅ Intermediate ML learning completed for {asset_symbol}")
                    else:
                        safe_print("⚠️ Asset analyzer is None or doesn't have _perform_learning_phase_training method")
                else:
                    safe_print(f"⚠️ Asset {asset_symbol} not found in analyzer.asset_analyzers")
            else:
                safe_print("⚠️ No analyzer available for ML learning")
        except Exception as e:
            safe_print(f"⚠️ ML learning phase error: {e}")

    async def _perform_final_training(self):
        """Train ONLY LSTM_SupportResistance model - SPECIFIC to this test"""
        try:
            # Usa l'analyzer dell'unified system (AdvancedMarketAnalyzer)
            if self.unified_system and hasattr(self.unified_system, 'analyzer') and self.unified_system.analyzer:
                analyzer = self.unified_system.analyzer
                asset_symbol = getattr(self.unified_system.config, 'asset_symbol', self.symbol)
                
                safe_print("🎯 Training ONLY LSTM_SupportResistance model...")
                
                # Get the asset analyzer
                if hasattr(analyzer, 'asset_analyzers') and asset_symbol in analyzer.asset_analyzers:
                    asset_analyzer = analyzer.asset_analyzers[asset_symbol]
                    
                    # Train only LSTM_SupportResistance model
                    if hasattr(asset_analyzer, '_retrain_algorithm'):
                        from src.Analyzer import ModelType
                        
                        # Get the algorithm from the competition
                        sr_competition = asset_analyzer.competitions.get(ModelType.SUPPORT_RESISTANCE)
                        if sr_competition and 'LSTM_SupportResistance' in sr_competition.algorithms:
                            algorithm = sr_competition.algorithms['LSTM_SupportResistance']
                            
                            safe_print("🔄 Retraining LSTM_SupportResistance algorithm...")
                            
                            # DIAGNOSTIC: Check training data quality
                            if hasattr(asset_analyzer, 'tick_data') and len(asset_analyzer.tick_data) > 0:
                                safe_print("\n📊 DIAGNOSTIC: Checking training data quality...")
                                prices = [t.get('price', 0) for t in list(asset_analyzer.tick_data)[-1000:]]
                                if prices:
                                    import numpy as np
                                    price_array = np.array(prices)
                                    safe_print(f"   Price Stats: mean={np.mean(price_array):.2f}, std={np.std(price_array):.2f}")
                                    safe_print(f"   Price Range: [{np.min(price_array):.2f}, {np.max(price_array):.2f}]")
                                    safe_print(f"   Unique prices: {len(np.unique(price_array))}")
                            
                            asset_analyzer._retrain_algorithm(ModelType.SUPPORT_RESISTANCE, 'LSTM_SupportResistance', algorithm)
                            
                            safe_print("✅ LSTM_SupportResistance training completed")
                            
                            # Verify model actually learned something
                            safe_print("\n🔍 Verifying model learning...")
                            if hasattr(algorithm, 'final_score'):
                                safe_print(f"   Final Score: {algorithm.final_score}")
                            if hasattr(algorithm, 'accuracy'):
                                safe_print(f"   Accuracy: {algorithm.accuracy:.2%}")
                            if hasattr(algorithm, 'total_predictions'):
                                safe_print(f"   Total Predictions: {algorithm.total_predictions}")
                            
                            # Check if model is in ml_models
                            if hasattr(asset_analyzer, 'ml_models') and 'LSTM_SupportResistance' in asset_analyzer.ml_models:
                                model = asset_analyzer.ml_models['LSTM_SupportResistance']
                                safe_print(f"   Model State: {model.training}")
                                
                                # Test a prediction
                                try:
                                    import torch
                                    test_input = torch.randn(1, 400)  # Random test input
                                    with torch.no_grad():
                                        output = model(test_input)
                                        # OptimizedLSTM returns tuple (output, hidden_states)
                                        if isinstance(output, tuple):
                                            output = output[0]  # Get only the output tensor
                                    safe_print(f"   Test Prediction Shape: {output.shape}")
                                    safe_print(f"   Test Prediction Values: {output.numpy().flatten()}")
                                except Exception as e:
                                    safe_print(f"   ⚠️ Test prediction failed: {e}")
                            
                            return {"status": "success", "message": "LSTM_SupportResistance training completed"}
                        else:
                            safe_print("❌ LSTM_SupportResistance algorithm not found in competition")
                            return {"status": "error", "message": "Algorithm not found"}
                    else:
                        safe_print("❌ Asset analyzer doesn't have _retrain_algorithm method")
                        return {"status": "error", "message": "No _retrain_algorithm method"}
                else:
                    safe_print(f"❌ Asset analyzer not found for {asset_symbol}")
                    return {"status": "error", "message": "Asset analyzer not found"}
            else:
                safe_print("⚠️ No analyzer available for final training")
                return {"status": "error", "message": "No analyzer available"}
        except Exception as e:
            safe_print(f"⚠️ Final training error: {e}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    async def test_model_training(self, ticks):
        """Test model training IDENTICAL to test_backtest approach"""
        safe_print(f"\n🧪 Testing {self.model_name} using test_backtest methodology...")
        
        try:
            # Process some ticks to get data (IDENTICAL to test_backtest)
            safe_print("📊 Processing ticks for ML learning...")
            
            processed_count = 0
            for i, tick_data in enumerate(ticks):
                if i % 10000 == 0:
                    safe_print(f"   Processed {i:,} ticks...")
                
                # Process tick through unified system (IDENTICAL to test_backtest)
                # Convert timestamp string to datetime object
                timestamp_str = tick_data.get('timestamp', datetime.now().isoformat())
                if isinstance(timestamp_str, str):
                    try:
                        # Parse timestamp in format "2025.05.24 00:00:03"
                        timestamp_obj = datetime.strptime(timestamp_str, "%Y.%m.%d %H:%M:%S")
                    except ValueError:
                        # Fallback to current time if parsing fails
                        timestamp_obj = datetime.now()
                else:
                    timestamp_obj = timestamp_str
                
                result = await self.unified_system.process_tick(
                    timestamp=timestamp_obj,
                    price=tick_data.get('bid', 0.0),  # Use bid as price
                    volume=tick_data.get('volume', 0)
                )
                processed_count += 1
                
                # Trigger ML learning every 20K ticks (for 100K total)
                if processed_count % 20000 == 0:
                    safe_print(f"🧠 Triggering ML learning phase...")
                    await self._perform_ml_learning_phase()
                    break  # For this test, one learning phase is enough
                    
            safe_print(f"✅ Processed {processed_count:,} ticks")
            
            # Perform final training (IDENTICAL to test_backtest)
            result = await self._perform_final_training()
            
            # CRITICAL: Force exit from learning phase to enable predictions!
            safe_print("\\n🚀 FORCING EXIT FROM LEARNING PHASE to enable predictions...")
            if self.analyzer and hasattr(self.analyzer, 'asset_analyzers'):
                asset_analyzer = self.analyzer.asset_analyzers.get(self.symbol)
                if asset_analyzer:
                    asset_analyzer.learning_phase = False
                    safe_print("✅ Learning phase set to False - predictions enabled!")
                    
                    # Process additional ticks to generate predictions
                    safe_print("\\n📊 Processing additional ticks to generate predictions...")
                    predictions_count = 0
                    for i in range(min(1000, len(ticks) - processed_count)):
                        tick_data = ticks[processed_count + i]
                        
                        # Convert timestamp
                        timestamp_str = tick_data.get('timestamp', datetime.now().isoformat())
                        if isinstance(timestamp_str, str):
                            try:
                                timestamp_obj = datetime.strptime(timestamp_str, "%Y.%m.%d %H:%M:%S")
                            except ValueError:
                                timestamp_obj = datetime.now()
                        else:
                            timestamp_obj = timestamp_str
                        
                        # Process tick - this should now generate predictions!
                        analysis_result = await self.unified_system.process_tick(
                            timestamp=timestamp_obj,
                            price=tick_data.get('bid', 0.0),
                            volume=tick_data.get('volume', 0)
                        )
                        
                        # Check if predictions were made
                        if 'predictions' in analysis_result and analysis_result['predictions']:
                            predictions_count += 1
                            if predictions_count == 1:
                                safe_print(f"✅ First prediction generated: {list(analysis_result['predictions'].keys())}")
                        
                        if predictions_count >= 10:  # Generate at least 10 predictions
                            break
                    
                    safe_print(f"✅ Generated {predictions_count} predictions after training")
            
            return result
            
        except Exception as e:
            safe_print(f"❌ Model training failed: {e}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    async def check_model_status(self):
        """Check model status IDENTICAL to test_backtest approach"""
        safe_print(f"\n🔍 Checking {self.model_name} status...")
        
        try:
            if (self.unified_system and 
                hasattr(self.unified_system, 'analyzer') and 
                self.unified_system.analyzer and
                hasattr(self.unified_system.analyzer, 'asset_analyzers')):
                
                asset_analyzer = self.unified_system.analyzer.asset_analyzers.get(self.symbol)
                
                if asset_analyzer:
                    safe_print(f"✅ Asset analyzer found for {self.symbol}")
                    
                    # Check if model exists
                    if hasattr(asset_analyzer, 'ml_models') and self.model_name in asset_analyzer.ml_models:
                        model = asset_analyzer.ml_models[self.model_name]
                        safe_print(f"✅ Model found: {model.__class__.__name__}")
                        
                        # Check learning status
                        if hasattr(asset_analyzer, 'learning_phase'):
                            safe_print(f"   Learning phase: {asset_analyzer.learning_phase}")
                        
                        if hasattr(asset_analyzer, 'analysis_count'):
                            safe_print(f"   Analysis count: {asset_analyzer.analysis_count}")
                            
                        return True
                    else:
                        safe_print(f"❌ Model {self.model_name} not found in asset analyzer")
                        if hasattr(asset_analyzer, 'ml_models'):
                            safe_print(f"   Available models: {list(asset_analyzer.ml_models.keys())}")
                        return False
                else:
                    safe_print(f"❌ Asset analyzer not found for {self.symbol}")
                    return False
            else:
                safe_print("❌ No unified system or analyzer available")
                return False
                
        except Exception as e:
            safe_print(f"❌ Error checking model status: {e}")
            traceback.print_exc()
            return False

    async def test_model(self):
        """Testa il modello LSTM_SupportResistance usando IDENTICAL struttura test_backtest"""
        safe_print("\n" + "="*80)
        safe_print(f"🧪 TESTING {self.model_name} - IDENTICAL to test_backtest")
        safe_print("="*80)
        
        try:
            # 1. Setup unified system (IDENTICAL)
            if not await self.setup_unified_system():
                self.results = {"status": "error", "message": "Failed to setup unified system"}
                return self.results
            
            # 2. Load real data (increased to 100K for better learning)
            ticks = self.load_real_data(limit=100000)
            if not ticks:
                self.results = {"status": "error", "message": "No ticks loaded"}
                return self.results
            
            # 3. Test model training (IDENTICAL methodology)
            training_result = await self.test_model_training(ticks)
            if training_result.get('status') != 'success':
                self.results = {"status": "error", "message": f"Model training failed: {training_result.get('message')}"}
                return self.results
            
            # 4. Check model status (IDENTICAL)
            if not await self.check_model_status():
                self.results = {"status": "error", "message": "Model status check failed"}
                return self.results
            
            safe_print(f"\n✅ {self.model_name} test completed successfully!")
            self.results = {
                "status": "success", 
                "message": "Test completed successfully",
                "training_result": training_result
            }
            
        except Exception as e:
            safe_print(f"\n❌ Test failed: {e}")
            traceback.print_exc()
            self.results = {"status": "error", "message": str(e)}
            
        return self.results
        
    def print_summary(self):
        """Stampa un riepilogo dei risultati"""
        safe_print("\n" + "="*80)
        safe_print("📊 TEST SUMMARY")
        safe_print("="*80)
        
        status = self.results.get('status', 'unknown')
        emoji = "✅" if status == "success" else "❌"
        
        safe_print(f"{emoji} Model: {self.model_name}")
        safe_print(f"   Status: {status}")
        
        if status == "error":
            safe_print(f"   Error: {self.results.get('error', 'Unknown error')}")
        elif status == "success":
            training_result = self.results.get('training_result', {})
            safe_print(f"   Final Loss: {training_result.get('final_loss', 'N/A')}")
            safe_print(f"   Epochs Completed: {training_result.get('epochs_completed', 0)}")
            safe_print(f"   Data Shape: {self.results.get('data_shape', {})}")


async def main():
    """Main function IDENTICAL to test_backtest style"""
    safe_print("🚀 Starting LSTM Support/Resistance Model Test")
    safe_print("   Using IDENTICAL structure to test_backtest")
    safe_print(f"📅 Timestamp: {datetime.now()}")
    
    if not ANALYZER_AVAILABLE:
        safe_print("❌ Cannot run test: Analyzer modules not available")
        safe_print("   This is likely due to numpy/talib compatibility issues")
        return
    
    tester = LSTMSupportResistanceTester()
    await tester.test_model()
    tester.print_summary()
    
    safe_print("\n✅ Test completed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())