#!/usr/bin/env python3
"""
Test del nuovo LSTM con adapter dinamico
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dynamic_lstm():
    print("🧪 TEST LSTM DINAMICO")
    print("="*50)
    
    try:
        from src.Analyzer import AdvancedMarketAnalyzer
        
        # Inizializza
        analyzer = AdvancedMarketAnalyzer("./analyzer_data")
        analyzer.add_asset('USTEC')
        ustec_analyzer = analyzer.asset_analyzers['USTEC']
        
        # Accesso al modello LSTM
        lstm_model = ustec_analyzer.ml_models['LSTM_SupportResistance']
        
        print(f"✅ LSTM Model: {type(lstm_model)}")
        print(f"   Expected input_size: {lstm_model.expected_input_size}")
        
        # Test sequence con diverse dimensioni
        test_cases = [
            ("2D small", torch.randn(3, 5)),        # 5 features
            ("2D medium", torch.randn(5, 50)),      # 50 features  
            ("2D large", torch.randn(2, 200)),      # 200 features
            ("3D small", torch.randn(4, 10, 8)),    # 8 features
            ("3D medium", torch.randn(3, 20, 150)), # 150 features
            ("3D large", torch.randn(2, 15, 300)),  # 300 features
            ("2D repeat", torch.randn(3, 50)),      # 50 features (già visto)
            ("3D repeat", torch.randn(4, 25, 150)), # 150 features (già visto)
        ]
        
        print(f"\n🔄 Testing multiple input dimensions:")
        
        for test_name, test_input in test_cases:
            try:
                print(f"\n   🧪 {test_name}: {test_input.shape}")
                output = lstm_model(test_input)
                print(f"      ✅ Success! Output: {output.shape}")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
        
        # Mostra statistiche
        print(f"\n📊 RESIZE STATISTICS:")
        stats = lstm_model.get_resize_stats()
        
        print(f"   Total calls: {stats['total_calls']}")
        print(f"   Adapters created: {stats['adapters_created']}")
        print(f"   Unique dimensions: {stats['unique_dimensions_seen']}")
        print(f"   Dimension frequency: {stats['dimension_frequency']}")
        print(f"   Adapter keys: {stats['adapter_keys']}")
        
        # Test reset
        print(f"\n🔄 Testing adapter reset:")
        lstm_model.reset_adapters()
        
        # Test dopo reset
        test_input = torch.randn(2, 100)
        output = lstm_model(test_input)
        print(f"   ✅ Post-reset test successful: {test_input.shape} → {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_dynamic_lstm()