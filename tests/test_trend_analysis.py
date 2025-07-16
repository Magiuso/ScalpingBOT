#!/usr/bin/env python3
"""
Test script per verificare il modello Trend Analysis
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

def test_trend_analysis_algorithm():
    """Test del modello Trend Analysis"""
    print('🧪 Testing Trend Analysis model...')
    
    # Simula dati di test realistici
    np.random.seed(42)
    
    # Genera dati di prezzo con trend chiari
    base_price = 21400
    n_samples = 1000
    
    # Crea trend multipli
    uptrend = np.linspace(0, 0.08, n_samples // 3)     # 8% trend rialzista
    sideways = np.full(n_samples // 3, 0.08)           # Laterale
    downtrend = np.linspace(0.08, 0.02, n_samples // 3) # Trend ribassista
    
    trend_component = np.concatenate([uptrend, sideways, downtrend])
    
    # Aggiungi rumore e cicli
    noise = np.random.normal(0, 0.005, n_samples)
    cycles = 0.02 * np.sin(np.linspace(0, 4*np.pi, n_samples))
    
    returns = trend_component + noise + cycles
    log_prices = np.cumsum(returns)
    prices = base_price * np.exp(log_prices)
    
    # Volume correlato al trend
    volumes = np.random.lognormal(np.log(1000), 0.3, n_samples)
    volumes[:n_samples//3] *= 1.5  # Più volume in uptrend
    
    # Moving averages per trend analysis
    sma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
    sma_50 = np.convolve(prices, np.ones(50)/50, mode='same')
    
    # RSI
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
        'sma_50': sma_50,
        'rsi': rsi,
        'timestamps': np.arange(n_samples, dtype=np.float64)
    }
    
    # Test dell'algoritmo simulato
    trend_detections = []
    
    # Simula trend analysis
    for i in range(100, len(prices) - 50):
        price_window = prices[i-100:i+50]
        sma20_window = sma_20[i-100:i+50]
        sma50_window = sma_50[i-100:i+50]
        volume_window = volumes[i-100:i+50]
        
        # Trend 1: Moving Average Analysis
        sma20_current = sma20_window[-1]
        sma50_current = sma50_window[-1]
        sma20_prev = sma20_window[-20]
        sma50_prev = sma50_window[-20]
        
        if sma20_current > sma50_current and sma20_prev <= sma50_prev:
            trend_detections.append({
                'type': 'uptrend_crossover',
                'strength': (sma20_current - sma50_current) / sma50_current,
                'price': prices[i]
            })
        elif sma20_current < sma50_current and sma20_prev >= sma50_prev:
            trend_detections.append({
                'type': 'downtrend_crossover',
                'strength': (sma50_current - sma20_current) / sma50_current,
                'price': prices[i]
            })
        
        # Trend 2: Price Momentum
        price_change_20 = (prices[i] - prices[i-20]) / prices[i-20]
        price_change_50 = (prices[i] - prices[i-50]) / prices[i-50]
        
        if price_change_20 > 0.03 and price_change_50 > 0.05:
            trend_detections.append({
                'type': 'strong_uptrend',
                'strength': min(price_change_20, 0.1),
                'price': prices[i]
            })
        elif price_change_20 < -0.03 and price_change_50 < -0.05:
            trend_detections.append({
                'type': 'strong_downtrend',
                'strength': min(abs(price_change_20), 0.1),
                'price': prices[i]
            })
        
        # Trend 3: Volume Confirmation
        volume_avg = np.mean(volume_window[:-50])
        volume_recent = np.mean(volume_window[-50:])
        
        if volume_recent > volume_avg * 1.2:
            if price_change_20 > 0.01:
                trend_detections.append({
                    'type': 'volume_confirmed_uptrend',
                    'strength': price_change_20 * (volume_recent / volume_avg),
                    'price': prices[i]
                })
            elif price_change_20 < -0.01:
                trend_detections.append({
                    'type': 'volume_confirmed_downtrend',
                    'strength': abs(price_change_20) * (volume_recent / volume_avg),
                    'price': prices[i]
                })
        
        # Trend 4: Consolidation/Sideways
        price_range = np.max(price_window[-20:]) - np.min(price_window[-20:])
        avg_price = np.mean(price_window[-20:])
        
        if price_range / avg_price < 0.02:  # Range < 2%
            trend_detections.append({
                'type': 'consolidation',
                'strength': 0.02 - (price_range / avg_price),
                'price': prices[i]
            })
    
    # Risultati del test
    print(f'📊 Trend Analysis Test Results:')
    print(f'   Total samples: {len(prices)}')
    print(f'   Trend detections: {len(trend_detections)}')
    
    if len(trend_detections) > 0:
        trend_types = {}
        strengths = []
        
        for trend in trend_detections:
            trend_type = trend['type']
            trend_types[trend_type] = trend_types.get(trend_type, 0) + 1
            strengths.append(trend['strength'])
        
        print(f'   Trend types: {dict(trend_types)}')
        print(f'   Average strength: {np.mean(strengths):.3f}')
        print(f'   Strength range: [{np.min(strengths):.3f}, {np.max(strengths):.3f}]')
    
    # Test di degenerazione
    strength_std = np.std(strengths) if strengths else 0
    unique_trends = len(set(t['type'] for t in trend_detections))
    
    print(f'\n🔍 Trend Degeneration Analysis:')
    print(f'   Strength std: {strength_std:.6f} (threshold: 1e-6)')
    print(f'   Unique trends: {unique_trends} (threshold: 2)')
    print(f'   Trend diversity: {len(trend_detections) > 0}')
    
    is_degenerate = (strength_std < 1e-6) or (unique_trends < 2) or (len(trend_detections) == 0)
    
    if is_degenerate:
        print('❌ FAIL: Trend analysis shows degeneration!')
        return False
    else:
        print('✅ SUCCESS: Trend analysis works correctly!')
        return True

def test_actual_analyzer():
    """Test del vero implementation dell'analyzer"""
    print('\n🔍 Testing actual Analyzer implementation...')
    
    try:
        # Import con percorso completo per risolvere i problemi dell'IDE
        from src.Analyzer import AdvancedMarketAnalyzer as Analyzer
        from src.Unified_Analyzer_System import UnifiedConfig, SystemMode, PerformanceProfile
        
        # Create test config
        config = UnifiedConfig.for_backtesting("USTEC")
        config.system_mode = SystemMode.TESTING
        config.performance_profile = PerformanceProfile.NORMAL
        config.learning_phase_enabled = True
        config.min_learning_days = 1
        config.max_tick_buffer_size = 1000
        
        # Initialize analyzer
        analyzer = Analyzer("./test_analyzer_data")
        
        # Generate realistic test data
        np.random.seed(42)
        n_samples = 500
        base_price = 21400
        
        # Create trending price data
        uptrend = np.linspace(0, 0.06, n_samples // 2)
        downtrend = np.linspace(0.06, 0.02, n_samples // 2)
        trend_component = np.concatenate([uptrend, downtrend])
        
        noise = np.random.normal(0, 0.008, n_samples)
        returns = trend_component + noise
        
        log_prices = np.cumsum(returns)
        prices = base_price * np.exp(log_prices)
        
        # Generate other required data
        volumes = np.random.exponential(1000, n_samples)
        sma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
        
        # Simple RSI
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
        
        print(f'📊 Generated test data:')
        print(f'   Samples: {len(prices)}')
        print(f'   Price range: {np.min(prices):.2f} - {np.max(prices):.2f}')
        print(f'   Price volatility: {np.std(prices):.2f}')
        
        # Test the actual method through AdvancedMarketAnalyzer
        print('\n🔧 Testing actual _prepare_trend_dataset...')
        # Access the method directly from the main analyzer instance
        prepare_method = getattr(analyzer, '_prepare_trend_dataset', None)
        if prepare_method:
            print(f'✅ Method _prepare_trend_dataset found in analyzer instance')
            X, y = prepare_method(test_data)
        else:
            print('❌ Method _prepare_trend_dataset not found')
            print(f'Available methods: {[m for m in dir(analyzer) if m.startswith("_prepare")]}')
            return False
        
        print(f'✅ Method executed successfully')
        print(f'   X shape: {X.shape}')
        print(f'   y shape: {y.shape}')
        
        # Analyze targets
        if len(y) > 0:
            print(f'\n📊 Real Implementation Trend Analysis:')
            print(f'   Trend targets: mean={np.mean(y):.6f}, std={np.std(y):.6f}')
            print(f'                 min={np.min(y):.6f}, max={np.max(y):.6f}')
            print(f'                 unique={len(np.unique(y))}')
            
            print(f'\n🚨 Degeneration Check:')
            print(f'   All zeros: {np.all(y == 0)}')
            print(f'   Trend std: {np.std(y):.8f}')
            
            # Check for degeneration
            is_degenerate = (np.std(y) < 1e-6) or (len(np.unique(y)) < 3) or np.all(y == 0)
            
            if is_degenerate:
                print('❌ FAIL: Real implementation shows degeneration!')
                return False
            else:
                print('✅ SUCCESS: Real implementation generates valid trend targets')
                return True
        else:
            print('❌ FAIL: No samples generated')
            return False
            
    except Exception as e:
        print(f'❌ Error testing actual analyzer: {e}')
        return False

def main():
    """Main test function"""
    print('🧪 Trend Analysis Model Test Suite')
    print('=' * 50)
    
    # Test simulated algorithm
    simulated_result = test_trend_analysis_algorithm()
    
    # Test real implementation
    real_result = test_actual_analyzer()
    
    print('\n' + '=' * 50)
    print('🏁 Test Results:')
    print(f'   Simulated algorithm: {"✅ PASS" if simulated_result else "❌ FAIL"}')
    print(f'   Real implementation: {"✅ PASS" if real_result else "❌ FAIL"}')
    
    if not simulated_result or not real_result:
        print('\n🔍 DIAGNOSIS: Trend analysis model needs fixes!')
        print('   Check for target degeneration in _prepare_trend_dataset method.')

if __name__ == "__main__":
    main()