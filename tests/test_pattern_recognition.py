#!/usr/bin/env python3
"""
Test script per verificare il modello Pattern Recognition
"""
import numpy as np
import sys
import os

# Aggiungi il path del progetto correttamente
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')

# Aggiungi i path necessari
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

def test_pattern_recognition_algorithm():
    """Test del modello Pattern Recognition"""
    print('🧪 Testing Pattern Recognition model...')
    
    # Simula dati di test realistici
    np.random.seed(42)
    
    # Genera dati di prezzo con pattern riconoscibili
    base_price = 21400
    n_samples = 1000
    
    # Crea pattern di prezzo con trend e volatilità
    returns = np.random.normal(0, 0.002, n_samples)
    log_prices = np.cumsum(returns)
    prices = base_price * np.exp(log_prices)
    
    # Aggiungi pattern specifici
    volumes = np.random.lognormal(np.log(1000), 0.5, n_samples)
    sma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
    
    # RSI per pattern recognition
    price_changes = np.diff(prices, prepend=prices[0])
    gains = np.where(price_changes > 0, price_changes, 0)
    losses = np.where(price_changes < 0, -price_changes, 0)
    avg_gains = np.convolve(gains, np.ones(14)/14, mode='same')
    avg_losses = np.convolve(losses, np.ones(14)/14, mode='same')
    rs = np.where(avg_losses != 0, avg_gains / avg_losses, 100)
    rsi = 100 - (100 / (1 + rs))
    
    test_data = {
        'prices': prices,
        'volumes': volumes,
        'sma_20': sma_20,
        'rsi': rsi,
        'timestamps': np.arange(n_samples, dtype=np.float64)
    }
    
    # Test dell'algoritmo simulato
    patterns_detected = []
    
    # Simula pattern recognition
    for i in range(50, len(prices) - 50):
        price_window = prices[i-50:i+50]
        
        # Pattern 1: Head and Shoulders
        if len(price_window) >= 100:
            mid_point = len(price_window) // 2
            left_shoulder = np.mean(price_window[:20])
            head = np.mean(price_window[40:60])
            right_shoulder = np.mean(price_window[80:])
            
            if head > left_shoulder * 1.02 and head > right_shoulder * 1.02:
                patterns_detected.append({
                    'type': 'head_and_shoulders',
                    'confidence': np.random.uniform(0.6, 0.9),
                    'price': prices[i]
                })
        
        # Pattern 2: Double Top/Bottom
        recent_highs = []
        recent_lows = []
        
        for j in range(10, len(price_window) - 10):
            if price_window[j] > max(price_window[j-10:j]) and price_window[j] > max(price_window[j+1:j+11]):
                recent_highs.append(price_window[j])
            if price_window[j] < min(price_window[j-10:j]) and price_window[j] < min(price_window[j+1:j+11]):
                recent_lows.append(price_window[j])
        
        if len(recent_highs) >= 2:
            if abs(recent_highs[-1] - recent_highs[-2]) / recent_highs[-1] < 0.01:
                patterns_detected.append({
                    'type': 'double_top',
                    'confidence': np.random.uniform(0.5, 0.8),
                    'price': prices[i]
                })
    
    # Risultati del test
    print(f'📊 Pattern Recognition Test Results:')
    print(f'   Total samples: {len(prices)}')
    print(f'   Patterns detected: {len(patterns_detected)}')
    
    if len(patterns_detected) > 0:
        pattern_types = {}
        confidences = []
        
        for pattern in patterns_detected:
            pattern_type = pattern['type']
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
            confidences.append(pattern['confidence'])
        
        print(f'   Pattern types: {dict(pattern_types)}')
        print(f'   Average confidence: {np.mean(confidences):.3f}')
        print(f'   Confidence range: [{np.min(confidences):.3f}, {np.max(confidences):.3f}]')
    
    # Test di degenerazione
    confidence_std = np.std(confidences) if confidences else 0
    unique_patterns = len(set(p['type'] for p in patterns_detected))
    
    print(f'\n🔍 Pattern Degeneration Analysis:')
    print(f'   Confidence std: {confidence_std:.6f} (threshold: 1e-6)')
    print(f'   Unique patterns: {unique_patterns} (threshold: 2)')
    print(f'   Pattern diversity: {len(patterns_detected) > 0}')
    
    is_degenerate = (confidence_std < 1e-6) or (unique_patterns < 2) or (len(patterns_detected) == 0)
    
    if is_degenerate:
        print('❌ FAIL: Pattern recognition shows degeneration!')
        return False
    else:
        print('✅ SUCCESS: Pattern recognition works correctly!')
        return True

def test_actual_analyzer():
    """Test del vero implementation dell'analyzer con dati REALI"""
    print('\n🔍 Testing actual Analyzer implementation with REAL DATA...')
    
    try:
        # Import con percorso completo per risolvere i problemi dell'IDE
        from src.Analyzer import AssetAnalyzer as Analyzer
        from src.Unified_Analyzer_System import UnifiedConfig, SystemMode, PerformanceProfile
        import os
        import json
        
        # Create test config
        config = UnifiedConfig.for_backtesting("USTEC")
        config.system_mode = SystemMode.TESTING
        config.performance_profile = PerformanceProfile.NORMAL
        config.learning_phase_enabled = True
        config.min_learning_days = 1
        config.max_tick_buffer_size = 1000
        
        # Initialize analyzer
        analyzer = Analyzer("USTEC", "./test_analyzer_data")
        
        # Carica dati reali dal file di backtesting
        data_file = "./test_analyzer_data/backtest_USTEC_20250516_20250715.jsonl"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"❌ Real backtest data file not found: {data_file}")
        
        print(f"📊 Loading REAL data from: {data_file}")
        
        # Leggi i dati reali - struttura completa (limitato per test performance)
        real_data = []
        tick_count = 0
        max_ticks = 10000  # Limita per test performance
        
        with open(data_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    real_data.append(data)
                    if data.get('type') == 'tick':
                        tick_count += 1
                        if tick_count >= max_ticks:
                            break
                except:
                    continue
        
        print(f"📊 Loaded {len(real_data)} real data entries")
        
        if len(real_data) < 100:
            raise ValueError(f"❌ Not enough real data ({len(real_data)} samples). Need at least 100 samples for valid test.")
        
        # Estrai prezzi e volumi dai dati reali
        prices = []
        volumes = []
        
        for entry in real_data:
            try:
                # Filtra solo i tick (ignora metadati)
                if entry.get('type') == 'tick':
                    # Struttura: {"type": "tick", "bid": 21389.2, "ask": 21390.15, "volume": 0, ...}
                    bid = entry.get('bid')
                    ask = entry.get('ask')
                    volume = entry.get('volume', 1)
                    
                    if bid is not None and ask is not None and bid > 0 and ask > 0:
                        price = (bid + ask) / 2  # Mid price
                        prices.append(float(price))
                        volumes.append(float(max(volume, 1)))  # Evita volume 0
                    
            except Exception as e:
                continue
        
        if len(prices) < 100:
            raise ValueError(f"❌ Not enough valid price data ({len(prices)} samples). Need at least 100 samples for valid test.")
        
        # Converti in numpy arrays
        prices = np.array(prices)
        volumes = np.array(volumes)
        
        # Calcola indicatori tecnici dai dati reali
        sma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
        
        # RSI sui dati reali
        price_changes = np.diff(prices, prepend=prices[0])
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        avg_gains = np.convolve(gains, np.ones(14)/14, mode='same')
        avg_losses = np.convolve(losses, np.ones(14)/14, mode='same')
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        test_data = {
            'prices': prices,
            'volumes': volumes,
            'sma_20': sma_20,
            'rsi': rsi
        }
        
        print(f'📊 Loaded REAL data:')
        print(f'   Samples: {len(prices)}')
        print(f'   Price range: {np.min(prices):.2f} - {np.max(prices):.2f}')
        print(f'   Price volatility: {np.std(prices):.2f}')
        print(f'   Volume range: {np.min(volumes):.2f} - {np.max(volumes):.2f}')
        
        # Test the actual method through RollingWindowTrainer
        print('\n🔧 Testing actual _prepare_pattern_dataset...')
        # Access the method through the analyzer's rolling_trainer
        if hasattr(analyzer, 'rolling_trainer'):
            prepare_method = getattr(analyzer.rolling_trainer, '_prepare_pattern_dataset', None)
            if prepare_method:
                print(f'✅ Method _prepare_pattern_dataset found in rolling_trainer')
                X, y = prepare_method(test_data)
            else:
                print('❌ Method _prepare_pattern_dataset not found in rolling_trainer')
                print(f'Available methods: {[m for m in dir(analyzer.rolling_trainer) if m.startswith("_prepare")]}')
                return False
        else:
            print('❌ rolling_trainer not found in analyzer')
            return False
        
        print(f'✅ Method executed successfully')
        print(f'   X shape: {X.shape}')
        print(f'   y shape: {y.shape}')
        
        # Analyze targets
        if len(y) > 0:
            print(f'\n📊 Real Implementation Pattern Analysis (REAL DATA):')
            print(f'   Pattern targets: mean={np.mean(y):.6f}, std={np.std(y):.6f}')
            print(f'                   min={np.min(y):.6f}, max={np.max(y):.6f}')
            print(f'                   unique={len(np.unique(y))}')
            
            # Se y è multi-dimensionale (più tipi di pattern), analizza separatamente
            if len(y.shape) > 1 and y.shape[1] > 1:
                print(f'\n📊 Pattern-specific Analysis:')
                pattern_types = ['classical', 'cnn', 'lstm', 'transformer', 'ensemble']
                for i in range(min(y.shape[1], len(pattern_types))):
                    pattern_values = y[:, i]
                    print(f'   {pattern_types[i]}: mean={np.mean(pattern_values):.3f}, std={np.std(pattern_values):.3f}, range=[{np.min(pattern_values):.3f}, {np.max(pattern_values):.3f}]')
            
            print(f'\n🚨 Degeneration Check:')
            print(f'   All zeros: {np.all(y == 0)}')
            print(f'   Pattern std: {np.std(y):.8f} (threshold: 1e-6)')
            print(f'   Unique values: {len(np.unique(y))} (threshold: 3)')
            
            # Check for degeneration
            is_degenerate = (np.std(y) < 1e-6) or (len(np.unique(y)) < 3) or np.all(y == 0)
            
            if is_degenerate:
                print('❌ FAIL: Real implementation shows degeneration with REAL DATA!')
                if np.all(y == 0):
                    print('   Reason: All targets are zero')
                elif np.std(y) < 1e-6:
                    print('   Reason: Very low variance in pattern targets')
                else:
                    print('   Reason: Too few unique values in targets')
                return False
            else:
                print('✅ SUCCESS: Real implementation generates valid pattern targets from REAL DATA')
                return True
        else:
            print('❌ FAIL: No samples generated')
            return False
            
    except Exception as e:
        print(f'❌ Error testing actual analyzer: {e}')
        return False

def main():
    """Main test function"""
    print('🧪 Pattern Recognition Model Test Suite')
    print('=' * 50)
    
    # Test simulated algorithm
    simulated_result = test_pattern_recognition_algorithm()
    
    # Test real implementation
    real_result = test_actual_analyzer()
    
    print('\n' + '=' * 50)
    print('🏁 Test Results:')
    print(f'   Simulated algorithm: {"✅ PASS" if simulated_result else "❌ FAIL"}')
    print(f'   Real implementation: {"✅ PASS" if real_result else "❌ FAIL"}')
    
    if not simulated_result or not real_result:
        print('\n🔍 DIAGNOSIS: Pattern recognition model needs fixes!')
        print('   Check for target degeneration in _prepare_pattern_dataset method.')

if __name__ == "__main__":
    main()