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
        
        # MACD sui dati reali
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
        
        print(f'📊 Loaded REAL data:')
        print(f'   Samples: {len(prices)}')
        print(f'   Price range: {np.min(prices):.2f} - {np.max(prices):.2f}')
        print(f'   Price volatility: {np.std(prices):.2f}')
        print(f'   Volume range: {np.min(volumes):.2f} - {np.max(volumes):.2f}')
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
            print(f'\n📊 Real Implementation RSI_Momentum Analysis (REAL DATA):')
            print(f'   Momentum targets: mean={np.mean(y):.6f}, std={np.std(y):.6f}')
            print(f'                    min={np.min(y):.6f}, max={np.max(y):.6f}')
            print(f'                    unique={len(np.unique(y))}')
            
            # Analizza le 4 dimensioni di momentum separatamente
            if len(y.shape) > 1 and y.shape[1] >= 4:
                print(f'\n📊 Momentum-specific Analysis:')
                momentum_types = ['rsi_momentum', 'macd_momentum', 'divergence', 'overbought_momentum']
                for i in range(min(y.shape[1], len(momentum_types))):
                    momentum_values = y[:, i]
                    print(f'   {momentum_types[i]}: mean={np.mean(momentum_values):.3f}, std={np.std(momentum_values):.3f}, range=[{np.min(momentum_values):.3f}, {np.max(momentum_values):.3f}]')
            
            print(f'\n🚨 Degeneration Check:')
            print(f'   All zeros: {np.all(y == 0)}')
            print(f'   Momentum std: {np.std(y):.8f} (threshold: 1e-6)')
            print(f'   Unique values: {len(np.unique(y))} (threshold: 10)')
            
            # Check for degeneration
            is_degenerate = (np.std(y) < 1e-6) or (len(np.unique(y)) < 10) or np.all(y == 0)
            
            if is_degenerate:
                print('❌ FAIL: Real implementation shows degeneration with REAL DATA!')
                if np.all(y == 0):
                    print('   Reason: All targets are zero')
                elif np.std(y) < 1e-6:
                    print('   Reason: Very low variance in momentum targets')
                else:
                    print('   Reason: Too few unique values in targets')
                return False
            else:
                print('✅ SUCCESS: Real implementation generates valid momentum targets from REAL DATA')
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
    
    # Test real implementation with REAL DATA only
    real_result = test_actual_analyzer()
    
    print('\n' + '=' * 50)
    print('🏁 Test Results:')
    print(f'   Real implementation: {"✅ PASS" if real_result else "❌ FAIL"}')
    
    if not real_result:
        print('\n🔍 DIAGNOSIS: RSI_Momentum model needs fixes!')
        print('   Check for target degeneration in _prepare_momentum_dataset method.')

if __name__ == "__main__":
    main()