#!/usr/bin/env python3
"""
Esplora struttura AssetAnalyzer per trovare dove sono gli algoritmi
"""

import sys
import os

# Aggiungi path per import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def explore_analyzer_structure():
    print("🔍 ESPLORAZIONE STRUTTURA ANALYZER")
    print("="*60)
    
    try:
        from src.Analyzer import AdvancedMarketAnalyzer
        
        print("✅ Analyzer imported successfully")
        
        # Crea istanza
        analyzer = AdvancedMarketAnalyzer("./analyzer_data")
        analyzer.add_asset('USTEC')
        
        print("✅ Analyzer initialized with USTEC")
        
        if 'USTEC' in analyzer.asset_analyzers:
            ustec_analyzer = analyzer.asset_analyzers['USTEC']
            
            print(f"\n📊 AssetAnalyzer for USTEC:")
            print(f"   Type: {type(ustec_analyzer)}")
            
            # Esplora TUTTI gli attributi
            print(f"\n🔍 ALL ATTRIBUTES:")
            all_attrs = dir(ustec_analyzer)
            
            # Filtra attributi interessanti (non private)
            interesting_attrs = [attr for attr in all_attrs 
                               if not attr.startswith('_') and 
                               not callable(getattr(ustec_analyzer, attr, None))]
            
            print(f"   📋 Attributes ({len(interesting_attrs)}):")
            for attr in interesting_attrs:
                try:
                    value = getattr(ustec_analyzer, attr)
                    attr_type = type(value).__name__
                    
                    if isinstance(value, dict):
                        print(f"      {attr}: dict with {len(value)} items")
                        # Se è un dict, mostra le chiavi
                        if len(value) > 0:
                            keys = list(value.keys())[:5]  # Prime 5 chiavi
                            print(f"         Keys: {keys}...")
                    elif isinstance(value, list):
                        print(f"      {attr}: list with {len(value)} items")
                    elif isinstance(value, (int, float, bool, str)):
                        print(f"      {attr}: {attr_type} = {value}")
                    else:
                        print(f"      {attr}: {attr_type}")
                        
                except Exception as e:
                    print(f"      {attr}: Error accessing - {e}")
            
            # Esplora METODI che potrebbero contenere algoritmi
            print(f"\n🔧 METHODS:")
            methods = [attr for attr in all_attrs 
                      if not attr.startswith('_') and 
                      callable(getattr(ustec_analyzer, attr, None))]
            
            interesting_methods = [m for m in methods if any(keyword in m.lower() 
                                  for keyword in ['algorithm', 'model', 'lstm', 'train', 'support', 'resistance'])]
            
            if interesting_methods:
                print(f"   📋 Interesting methods:")
                for method in interesting_methods:
                    print(f"      {method}()")
            else:
                print(f"   📋 All methods ({len(methods)}):")
                for method in methods[:10]:  # Prime 10
                    print(f"      {method}()")
                if len(methods) > 10:
                    print(f"      ... and {len(methods)-10} more")
            
            # Cerca specificamente LSTM
            print(f"\n🎯 SEARCHING FOR LSTM:")
            
            # Cerca in tutti gli attributi che contengono 'lstm'
            lstm_attrs = [attr for attr in all_attrs if 'lstm' in attr.lower()]
            if lstm_attrs:
                print(f"   Found LSTM-related attributes: {lstm_attrs}")
                for attr in lstm_attrs:
                    try:
                        value = getattr(ustec_analyzer, attr)
                        print(f"      {attr}: {type(value)}")
                    except:
                        print(f"      {attr}: Error accessing")
            
            # Cerca nel contenuto degli attributi dict
            print(f"\n🔍 DEEP SEARCH IN DICT ATTRIBUTES:")
            for attr in interesting_attrs:
                try:
                    value = getattr(ustec_analyzer, attr)
                    if isinstance(value, dict):
                        for key, item in value.items():
                            if 'lstm' in str(key).lower() or 'support' in str(key).lower():
                                print(f"   Found in {attr}['{key}']: {type(item)}")
                                
                                # Se l'item ha attributi, ispezionali
                                if hasattr(item, '__dict__'):
                                    item_attrs = [a for a in dir(item) if not a.startswith('_')]
                                    print(f"      Item attributes: {item_attrs[:5]}...")
                                    
                                    # Cerca input_size specificamente
                                    if hasattr(item, 'input_size'):
                                        print(f"      🎯 input_size: {item.input_size}")
                                    
                except Exception as e:
                    continue
            
        else:
            print("❌ USTEC not found in asset_analyzers")
            print(f"Available assets: {list(analyzer.asset_analyzers.keys())}")
    
    except Exception as e:
        print(f"❌ Error during exploration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    explore_analyzer_structure()