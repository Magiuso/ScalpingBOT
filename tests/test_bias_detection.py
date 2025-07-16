#!/usr/bin/env python3
"""
Test script per verificare il modello Bias Detection
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

def test_bias_detection_algorithm():
    """Test del modello Bias Detection"""
    print('🧪 Testing Bias Detection model...')
    
    # Simula dati di test realistici
    np.random.seed(42)
    
    # Genera dati di prezzo con bias direzionali
    base_price = 21400
    n_samples = 1000
    
    # Crea bias bullish e bearish
    bullish_trend = np.linspace(0, 0.05, n_samples // 2)  # 5% trend rialzista
    bearish_trend = np.linspace(0, -0.03, n_samples // 2)  # 3% trend ribassista
    trend_bias = np.concatenate([bullish_trend, bearish_trend])
    
    # Aggiungi rumore
    noise = np.random.normal(0, 0.01, n_samples)
    returns = trend_bias + noise
    
    log_prices = np.cumsum(returns)
    prices = base_price * np.exp(log_prices)
    
    # Volume bias (maggiore volume in trend)
    volumes = np.random.lognormal(np.log(1000), 0.5, n_samples)
    volumes[:n_samples//2] *= 1.3  # Più volume in trend rialzista
    
    # RSI per bias detection
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
        'rsi': rsi,
        'timestamps': np.arange(n_samples, dtype=np.float64)
    }
    
    # Test dell'algoritmo simulato
    bias_detections = []
    
    # Simula bias detection
    for i in range(50, len(prices) - 50):
        price_window = prices[i-50:i+50]
        volume_window = volumes[i-50:i+50]
        rsi_window = rsi[i-50:i+50]
        
        # Bias 1: Volume-Price Analysis
        price_change = (prices[i] - prices[i-20]) / prices[i-20]
        volume_ratio = np.mean(volume_window[30:]) / np.mean(volume_window[:30])
        
        if price_change > 0.02 and volume_ratio > 1.1:
            bias_detections.append({
                'type': 'bullish_volume_bias',
                'strength': min(price_change * volume_ratio, 1.0),
                'price': prices[i]
            })
        elif price_change < -0.02 and volume_ratio > 1.1:
            bias_detections.append({
                'type': 'bearish_volume_bias',
                'strength': min(abs(price_change) * volume_ratio, 1.0),
                'price': prices[i]
            })
        
        # Bias 2: RSI Momentum Bias
        rsi_current = rsi_window[-1]
        rsi_avg = np.mean(rsi_window[:-1])
        
        if rsi_current > 70 and rsi_avg > 60:
            bias_detections.append({
                'type': 'overbought_bias',
                'strength': (rsi_current - 70) / 30,
                'price': prices[i]
            })
        elif rsi_current < 30 and rsi_avg < 40:
            bias_detections.append({
                'type': 'oversold_bias',
                'strength': (30 - rsi_current) / 30,
                'price': prices[i]
            })
        
        # Bias 3: Trend Momentum
        short_ma = np.mean(price_window[30:])
        long_ma = np.mean(price_window[:30])
        
        if short_ma > long_ma * 1.01:
            bias_detections.append({
                'type': 'bullish_trend_bias',
                'strength': (short_ma - long_ma) / long_ma,
                'price': prices[i]
            })
        elif short_ma < long_ma * 0.99:
            bias_detections.append({
                'type': 'bearish_trend_bias',
                'strength': (long_ma - short_ma) / long_ma,
                'price': prices[i]
            })
    
    # Risultati del test
    print(f'📊 Bias Detection Test Results:')
    print(f'   Total samples: {len(prices)}')
    print(f'   Bias detections: {len(bias_detections)}')
    
    if len(bias_detections) > 0:
        bias_types = {}
        strengths = []
        
        for bias in bias_detections:
            bias_type = bias['type']
            bias_types[bias_type] = bias_types.get(bias_type, 0) + 1
            strengths.append(bias['strength'])
        
        print(f'   Bias types: {dict(bias_types)}')
        print(f'   Average strength: {np.mean(strengths):.3f}')
        print(f'   Strength range: [{np.min(strengths):.3f}, {np.max(strengths):.3f}]')
    
    # Test di degenerazione
    strength_std = np.std(strengths) if strengths else 0
    unique_biases = len(set(b['type'] for b in bias_detections))
    
    print(f'\n🔍 Bias Degeneration Analysis:')
    print(f'   Strength std: {strength_std:.6f} (threshold: 1e-6)')
    print(f'   Unique biases: {unique_biases} (threshold: 2)')
    print(f'   Bias diversity: {len(bias_detections) > 0}')
    
    is_degenerate = (strength_std < 1e-6) or (unique_biases < 2) or (len(bias_detections) == 0)
    
    if is_degenerate:
        print('❌ FAIL: Bias detection shows degeneration!')
        return False
    else:
        print('✅ SUCCESS: Bias detection works correctly!')
        return True

def test_actual_analyzer():
    """Test del vero implementation dell'analyzer con dati REALI"""
    print('\n🔍 Testing actual Analyzer implementation with REAL DATA...')
    
    try:
        # Import con percorso completo per risolvere i problemi dell'IDE
        from src.Analyzer import AdvancedMarketAnalyzer as Analyzer
        from src.Unified_Analyzer_System import UnifiedConfig, SystemMode, PerformanceProfile
        import os
        import glob
        import pandas as pd
        
        # Create test config
        config = UnifiedConfig.for_backtesting("USTEC")
        config.system_mode = SystemMode.TESTING
        config.performance_profile = PerformanceProfile.NORMAL
        config.learning_phase_enabled = True
        config.min_learning_days = 1
        config.max_tick_buffer_size = 1000
        
        # Initialize analyzer
        analyzer = Analyzer("./test_analyzer_data")
        
        # Carica dati reali dal file di backtesting
        data_file = "./test_analyzer_data/backtest_USTEC_20250516_20250715.jsonl"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"❌ Real backtest data file not found: {data_file}")
        
        print(f"📊 Loading REAL data from: {data_file}")
        
        # Leggi i dati reali - struttura completa (limitato per test)
        real_data = []
        tick_count = 0
        max_ticks = 10000  # Limita per test performance
        
        with open(data_file, 'r') as f:
            for line in f:
                try:
                    import json
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
        
        # Test the actual method through AdvancedMarketAnalyzer
        print('\n🔧 Testing actual _prepare_bias_dataset...')
        # Access the method directly from the main analyzer instance
        prepare_method = getattr(analyzer, '_prepare_bias_dataset', None)
        if prepare_method:
            print(f'✅ Method _prepare_bias_dataset found in analyzer instance')
            X, y = prepare_method(test_data)
        else:
            print('❌ Method _prepare_bias_dataset not found')
            print(f'Available methods: {[m for m in dir(analyzer) if m.startswith("_prepare")]}')
            return False
        
        print(f'✅ Method executed successfully')
        print(f'   X shape: {X.shape}')
        print(f'   y shape: {y.shape}')
        
        # Analyze targets
        if len(y) > 0:
            print(f'\n📊 Real Implementation Bias Analysis (REAL DATA):')
            print(f'   Bias targets: mean={np.mean(y):.6f}, std={np.std(y):.6f}')
            print(f'                min={np.min(y):.6f}, max={np.max(y):.6f}')
            print(f'                unique={len(np.unique(y))}')
            
            # Analizza i 6 tipi di bias separatamente
            bias_types = ['bullish_trend', 'bearish_trend', 'bullish_volume', 'bearish_volume', 'overbought', 'oversold']
            for i, bias_type in enumerate(bias_types):
                if i < y.shape[1]:
                    bias_values = y[:, i]
                    print(f'   {bias_type}: mean={np.mean(bias_values):.3f}, std={np.std(bias_values):.3f}, range=[{np.min(bias_values):.3f}, {np.max(bias_values):.3f}]')
            
            print(f'\n🚨 Degeneration Check:')
            print(f'   All zeros: {np.all(y == 0)}')
            print(f'   Bias std: {np.std(y):.8f}')
            
            # Check for degeneration
            is_degenerate = (np.std(y) < 1e-6) or (len(np.unique(y)) < 3) or np.all(y == 0)
            
            if is_degenerate:
                print('❌ FAIL: Real implementation shows degeneration!')
                return False
            else:
                print('✅ SUCCESS: Real implementation generates valid bias targets from REAL DATA')
                return True
        else:
            print('❌ FAIL: No samples generated')
            return False
            
    except Exception as e:
        print(f'❌ Error testing actual analyzer: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print('🧪 Bias Detection Model Test Suite')
    print('=' * 50)
    
    # Test simulated algorithm
    simulated_result = test_bias_detection_algorithm()
    
    # Test real implementation
    real_result = test_actual_analyzer()
    
    print('\n' + '=' * 50)
    print('🏁 Test Results:')
    print(f'   Simulated algorithm: {"✅ PASS" if simulated_result else "❌ FAIL"}')
    print(f'   Real implementation: {"✅ PASS" if real_result else "❌ FAIL"}')
    
    if not simulated_result or not real_result:
        print('\n🔍 DIAGNOSIS: Bias detection model needs fixes!')
        print('   Check for target degeneration in _prepare_bias_dataset method.')

if __name__ == "__main__":
    main()