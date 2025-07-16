#!/usr/bin/env python3
"""
Test script per verificare il modello RSI_Momentum
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

def test_rsi_momentum_algorithm():
    """Test del modello RSI_Momentum"""
    print('🧪 Testing RSI_Momentum model...')
    
    # Simula dati di test realistici
    np.random.seed(42)
    
    # Genera dati di prezzo con momentum patterns
    base_price = 21400
    n_samples = 1000
    
    # Crea momentum patterns con oscillazioni RSI
    momentum_cycle = np.sin(np.linspace(0, 6*np.pi, n_samples)) * 0.02
    trend_component = np.linspace(-0.01, 0.01, n_samples)
    noise = np.random.normal(0, 0.008, n_samples)
    
    returns = momentum_cycle + trend_component + noise
    log_prices = np.cumsum(returns)
    prices = base_price * np.exp(log_prices)
    
    # Volume correlato al momentum
    volumes = np.random.lognormal(np.log(1000), 0.4, n_samples)
    
    # Calcola RSI realistico
    price_changes = np.diff(prices, prepend=prices[0])
    gains = np.where(price_changes > 0, price_changes, 0)
    losses = np.where(price_changes < 0, -price_changes, 0)
    
    # RSI con smoothing
    rsi_values = []
    for i in range(len(prices)):
        if i < 14:
            rsi_values.append(50.0)  # Valore neutro iniziale
        else:
            recent_gains = gains[i-14:i+1]
            recent_losses = losses[i-14:i+1]
            avg_gain = np.mean(recent_gains)
            avg_loss = np.mean(recent_losses)
            
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
    
    rsi = np.array(rsi_values)
    
    # MACD per momentum confirmation
    ema_12 = np.convolve(prices, np.ones(12)/12, mode='same')
    ema_26 = np.convolve(prices, np.ones(26)/26, mode='same')
    macd_line = ema_12 - ema_26
    macd_signal = np.convolve(macd_line, np.ones(9)/9, mode='same')
    
    test_data = {
        'prices': prices,
        'volumes': volumes,
        'rsi': rsi,
        'macd_line': macd_line,
        'macd_signal': macd_signal,
        'timestamps': np.arange(n_samples, dtype=np.float64)
    }
    
    # Test dell'algoritmo simulato
    momentum_signals = []
    
    # Simula RSI momentum analysis
    for i in range(50, len(prices) - 50):
        rsi_window = rsi[i-50:i+50]
        price_window = prices[i-50:i+50]
        volume_window = volumes[i-50:i+50]
        macd_window = macd_line[i-50:i+50]
        
        # Momentum 1: RSI Divergence
        rsi_current = rsi_window[-1]
        rsi_prev = rsi_window[-20]
        price_current = price_window[-1]
        price_prev = price_window[-20]
        
        price_change = (price_current - price_prev) / price_prev
        rsi_change = rsi_current - rsi_prev
        
        # Bullish divergence: price down, RSI up
        if price_change < -0.01 and rsi_change > 5:
            momentum_signals.append({
                'type': 'bullish_divergence',
                'strength': min(abs(price_change) + rsi_change/100, 1.0),
                'rsi': rsi_current,
                'price': price_current
            })
        
        # Bearish divergence: price up, RSI down
        elif price_change > 0.01 and rsi_change < -5:
            momentum_signals.append({
                'type': 'bearish_divergence',
                'strength': min(price_change + abs(rsi_change)/100, 1.0),
                'rsi': rsi_current,
                'price': price_current
            })
        
        # Momentum 2: RSI Overbought/Oversold with Volume
        volume_avg = np.mean(volume_window[:-10])
        volume_current = np.mean(volume_window[-10:])
        volume_ratio = volume_current / volume_avg if volume_avg > 0 else 1.0
        
        if rsi_current > 70 and volume_ratio > 1.2:
            momentum_signals.append({
                'type': 'overbought_momentum',
                'strength': ((rsi_current - 70) / 30) * min(volume_ratio, 2.0),
                'rsi': rsi_current,
                'price': price_current
            })
        elif rsi_current < 30 and volume_ratio > 1.2:
            momentum_signals.append({
                'type': 'oversold_momentum',
                'strength': ((30 - rsi_current) / 30) * min(volume_ratio, 2.0),
                'rsi': rsi_current,
                'price': price_current
            })
        
        # Momentum 3: MACD-RSI Confirmation
        macd_current = macd_window[-1]
        macd_prev = macd_window[-5]
        macd_momentum = macd_current - macd_prev
        
        if macd_momentum > 0 and rsi_current > 50 and rsi_current < 70:
            momentum_signals.append({
                'type': 'bullish_macd_rsi',
                'strength': min(abs(macd_momentum) * (rsi_current - 50) / 50, 1.0),
                'rsi': rsi_current,
                'price': price_current
            })
        elif macd_momentum < 0 and rsi_current < 50 and rsi_current > 30:
            momentum_signals.append({
                'type': 'bearish_macd_rsi',
                'strength': min(abs(macd_momentum) * (50 - rsi_current) / 50, 1.0),
                'rsi': rsi_current,
                'price': price_current
            })
        
        # Momentum 4: RSI Velocity
        rsi_velocity = (rsi_current - rsi_window[-10]) / 10
        if abs(rsi_velocity) > 2:
            momentum_signals.append({
                'type': 'high_rsi_velocity',
                'strength': min(abs(rsi_velocity) / 10, 1.0),
                'rsi': rsi_current,
                'price': price_current
            })
    
    # Risultati del test
    print(f'📊 RSI_Momentum Test Results:')
    print(f'   Total samples: {len(prices)}')
    print(f'   Momentum signals: {len(momentum_signals)}')
    
    if len(momentum_signals) > 0:
        signal_types = {}
        strengths = []
        rsi_values_in_signals = []
        
        for signal in momentum_signals:
            signal_type = signal['type']
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
            strengths.append(signal['strength'])
            rsi_values_in_signals.append(signal['rsi'])
        
        print(f'   Signal types: {dict(signal_types)}')
        print(f'   Average strength: {np.mean(strengths):.3f}')
        print(f'   Strength range: [{np.min(strengths):.3f}, {np.max(strengths):.3f}]')
        print(f'   RSI range in signals: [{np.min(rsi_values_in_signals):.1f}, {np.max(rsi_values_in_signals):.1f}]')
    
    # Test di degenerazione
    strength_std = np.std(strengths) if strengths else 0
    unique_signals = len(set(s['type'] for s in momentum_signals))
    rsi_diversity = len(set(s['rsi'] for s in momentum_signals)) if momentum_signals else 0
    
    print(f'\n🔍 RSI_Momentum Degeneration Analysis:')
    print(f'   Strength std: {strength_std:.6f} (threshold: 1e-6)')
    print(f'   Unique signals: {unique_signals} (threshold: 3)')
    print(f'   RSI diversity: {rsi_diversity} (threshold: 10)')
    print(f'   Signal diversity: {len(momentum_signals) > 0}')
    
    is_degenerate = (strength_std < 1e-6) or (unique_signals < 3) or (rsi_diversity < 10) or (len(momentum_signals) == 0)
    
    if is_degenerate:
        print('❌ FAIL: RSI_Momentum shows degeneration!')
        return False
    else:
        print('✅ SUCCESS: RSI_Momentum works correctly!')
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
        
        # Create price data with momentum patterns
        momentum_cycle = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 0.015
        trend = np.linspace(-0.005, 0.005, n_samples)
        noise = np.random.normal(0, 0.01, n_samples)
        
        returns = momentum_cycle + trend + noise
        log_prices = np.cumsum(returns)
        prices = base_price * np.exp(log_prices)
        
        # Generate other required data
        volumes = np.random.exponential(1000, n_samples)
        sma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
        
        # RSI calculation
        price_changes = np.diff(prices, prepend=prices[0])
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        avg_gains = np.convolve(gains, np.ones(14)/14, mode='same')
        avg_losses = np.convolve(losses, np.ones(14)/14, mode='same')
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = np.convolve(prices, np.ones(12)/12, mode='same')
        ema_26 = np.convolve(prices, np.ones(26)/26, mode='same')
        macd_line = ema_12 - ema_26
        macd_signal = np.convolve(macd_line, np.ones(9)/9, mode='same')
        
        test_data = {
            'prices': prices,
            'volumes': volumes,
            'sma_20': sma_20,
            'rsi': rsi,
            'macd_line': macd_line,
            'macd_signal': macd_signal
        }
        
        print(f'📊 Generated test data:')
        print(f'   Samples: {len(prices)}')
        print(f'   Price range: {np.min(prices):.2f} - {np.max(prices):.2f}')
        print(f'   Price volatility: {np.std(prices):.2f}')
        print(f'   RSI range: {np.min(rsi):.1f} - {np.max(rsi):.1f}')
        
        # Test the actual method through AdvancedMarketAnalyzer
        print('\n🔧 Testing actual _prepare_momentum_dataset...')
        # Access the method directly from the main analyzer instance
        prepare_method = getattr(analyzer, '_prepare_momentum_dataset', None)
        if prepare_method:
            print(f'✅ Method _prepare_momentum_dataset found in analyzer instance')
            X, y = prepare_method(test_data)
        else:
            print('❌ Method _prepare_momentum_dataset not found')
            print(f'Available methods: {[m for m in dir(analyzer) if m.startswith("_prepare")]}')
            return False
        
        print(f'✅ Method executed successfully')
        print(f'   X shape: {X.shape}')
        print(f'   y shape: {y.shape}')
        
        # Analyze targets
        if len(y) > 0:
            print(f'\n📊 Real Implementation RSI_Momentum Analysis:')
            print(f'   Momentum targets: mean={np.mean(y):.6f}, std={np.std(y):.6f}')
            print(f'                    min={np.min(y):.6f}, max={np.max(y):.6f}')
            print(f'                    unique={len(np.unique(y))}')
            
            print(f'\n🚨 Degeneration Check:')
            print(f'   All zeros: {np.all(y == 0)}')
            print(f'   Momentum std: {np.std(y):.8f}')
            
            # Check for degeneration
            is_degenerate = (np.std(y) < 1e-6) or (len(np.unique(y)) < 3) or np.all(y == 0)
            
            if is_degenerate:
                print('❌ FAIL: Real implementation shows degeneration!')
                return False
            else:
                print('✅ SUCCESS: Real implementation generates valid momentum targets')
                return True
        else:
            print('❌ FAIL: No samples generated')
            return False
            
    except Exception as e:
        print(f'❌ Error testing actual analyzer: {e}')
        return False

def main():
    """Main test function"""
    print('🧪 RSI_Momentum Model Test Suite')
    print('=' * 50)
    
    # Test simulated algorithm
    simulated_result = test_rsi_momentum_algorithm()
    
    # Test real implementation
    real_result = test_actual_analyzer()
    
    print('\n' + '=' * 50)
    print('🏁 Test Results:')
    print(f'   Simulated algorithm: {"✅ PASS" if simulated_result else "❌ FAIL"}')
    print(f'   Real implementation: {"✅ PASS" if real_result else "❌ FAIL"}')
    
    if not simulated_result or not real_result:
        print('\n🔍 DIAGNOSIS: RSI_Momentum model needs fixes!')
        print('   Check for target degeneration in _prepare_momentum_dataset method.')

if __name__ == "__main__":
    main()