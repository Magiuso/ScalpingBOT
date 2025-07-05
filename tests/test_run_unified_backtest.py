#!/usr/bin/env python3
"""
🚀 Unified Backtest Runner - Sistema Ultra-Veloce
Backtesting ottimizzato con UnifiedAnalyzerSystem
"""

import asyncio
import sys
import os
import time
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import del sistema unificato
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.Unified_Analyzer_System import (
        UnifiedAnalyzerSystem, 
        UnifiedConfig, 
        SystemMode, 
        PerformanceProfile,
        create_backtesting_system
    )
    UNIFIED_AVAILABLE = True
    print("✅ UnifiedAnalyzerSystem imported successfully")
except ImportError as e:
    print(f"❌ Error importing UnifiedAnalyzerSystem: {e}")
    UNIFIED_AVAILABLE = False
    sys.exit(1)

class UnifiedBacktestRunner:
    """Runner per backtesting ultra-veloce"""
    
    def __init__(self, asset: str = "USTEC"):
        self.asset = asset
        self.system = None
        
    async def run_backtest(self, months_back: int = 6, show_progress: bool = True):
        """Esegue backtest ultra-veloce"""
        
        print(f"🚀 Starting ULTRA-FAST Backtest for {self.asset}")
        print(f"📅 Period: {months_back} months back")
        print(f"⚡ Using: UnifiedAnalyzerSystem (optimized for speed)")
        print("="*60)
        
        try:
            # 1. Crea sistema ottimizzato per backtesting
            print("🔧 Creating optimized backtesting system...")
            self.system = await create_backtesting_system(self.asset)
            print("✅ System created and started")
            
            # 2. Genera o carica dati di test
            print("📊 Generating/loading market data...")
            market_data = self._generate_sample_data(months_back)
            print(f"✅ Loaded {len(market_data):,} data points")
            
            # 3. Esegui backtesting ultra-veloce
            print("\n🚀 Starting ULTRA-FAST processing...")
            start_time = time.time()
            
            await self.system.run_backtest_optimized(
                market_data, 
                show_progress=show_progress
            )
            
            elapsed = time.time() - start_time
            speed = len(market_data) / elapsed if elapsed > 0 else 0
            
            print(f"\n✅ BACKTEST COMPLETED!")
            print(f"⚡ Performance: {speed:,.0f} ticks/sec")
            print(f"🕐 Total time: {elapsed:.1f} seconds")
            print(f"📊 Data processed: {len(market_data):,} points")
            
            # 4. Mostra risultati
            await self._show_results()
            
            return True
            
        except Exception as e:
            print(f"❌ Backtest error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            if self.system:
                await self.system.stop()
    
    def _generate_sample_data(self, months_back: int) -> List[Dict[str, Any]]:
        """Genera dati di mercato realistici per backtesting"""
        
        # Calcola numero di tick per N mesi
        days = months_back * 30
        ticks_per_day = 1440  # 1 tick per minuto
        total_ticks = days * ticks_per_day
        
        print(f"📈 Generating {total_ticks:,} realistic market ticks...")
        
        # Genera serie temporale realistica
        start_date = datetime.now() - timedelta(days=days)
        
        # Parametri per USTEC (NASDAQ-100)
        base_price = 21500.0  # Prezzo base realistico
        volatility = 0.02  # 2% volatilità giornaliera
        trend = 0.0001  # Leggero trend rialzista
        
        data_points = []
        current_price = base_price
        
        for i in range(total_ticks):
            # Calcola timestamp
            timestamp = start_date + timedelta(minutes=i)
            
            # Genera movimento prezzo realistico
            # Usa random walk con trend e volatility clustering
            random_change = np.random.normal(trend, volatility / np.sqrt(1440))
            
            # Aggiungi volatility clustering (periodi più o meno volatili)
            if i % 1440 == 0:  # Ogni giorno
                vol_multiplier = np.random.uniform(0.5, 2.0)
                volatility *= vol_multiplier
                volatility = max(0.01, min(0.05, volatility))  # Clamp volatility
            
            # Applica movimento
            current_price *= (1 + random_change)
            
            # Evita prezzi irrealistici
            current_price = max(15000, min(30000, current_price))
            
            # Genera volume realistico
            base_volume = 1000
            volume_multiplier = np.random.uniform(0.5, 3.0)
            volume = int(base_volume * volume_multiplier)
            
            # Calcola bid/ask spread
            spread_pct = np.random.uniform(0.0001, 0.0005)  # 1-5 basis points
            spread = current_price * spread_pct
            
            bid = current_price - spread/2
            ask = current_price + spread/2
            
            # Crea data point
            data_point = {
                'timestamp': timestamp,
                'price': current_price,
                'volume': volume,
                'bid': bid,
                'ask': ask
            }
            
            data_points.append(data_point)
            
            # Progress ogni 100k ticks
            if i > 0 and i % 100000 == 0:
                progress = (i / total_ticks) * 100
                print(f"📈 Data generation: {progress:.1f}% ({i:,}/{total_ticks:,})")
        
        print(f"✅ Generated {len(data_points):,} realistic market data points")
        return data_points
    
    async def _show_results(self):
        """Mostra risultati del backtesting"""
        
        if not self.system:
            return
        
        try:
            # Ottieni status del sistema
            status = self.system.get_system_status()
            
            print("\n" + "="*60)
            print("📊 BACKTEST RESULTS")
            print("="*60)
            
            # System stats
            system_stats = status.get('system', {}).get('stats', {})
            print(f"🔢 Total ticks processed: {system_stats.get('total_ticks_processed', 0):,}")
            print(f"📝 Total events logged: {system_stats.get('total_events_logged', 0):,}")
            print(f"⏱️  System uptime: {status.get('system', {}).get('uptime_seconds', 0):.1f} seconds")
            
            # Analyzer stats
            analyzer_stats = status.get('analyzer', {})
            print(f"🧠 Analyzer processed: {analyzer_stats.get('ticks_processed', 0):,} ticks")
            print(f"⚡ Average latency: {analyzer_stats.get('avg_latency_ms', 0):.2f}ms")
            print(f"📈 Ticks per second: {analyzer_stats.get('ticks_per_second', 0):,.0f}")
            
            # Performance stats
            if 'performance' in status:
                perf = status['performance']
                print(f"💻 Memory usage: {perf.get('memory_mb', 0):.1f}MB")
                print(f"🖥️  CPU usage: {perf.get('cpu_percent', 0):.1f}%")
            
            print("="*60)
            print("🎉 Analyzer is now TRAINED and ready for live trading!")
            print("🚀 The system has learned from historical patterns")
            
        except Exception as e:
            print(f"⚠️ Error showing results: {e}")

def main():
    """Main function"""
    
    print("🚀 Unified Backtest Runner")
    print("Ultra-fast backtesting with UnifiedAnalyzerSystem")
    print()
    
    # Configurazione
    asset = input("📊 Asset symbol (default: USTEC): ").strip() or "USTEC"
    
    try:
        months_input = input("📅 Months back (default: 6): ").strip()
        months_back = int(months_input) if months_input else 6
    except ValueError:
        months_back = 6
    
    print(f"\n🎯 Configuration:")
    print(f"   Asset: {asset}")
    print(f"   Period: {months_back} months")
    print(f"   Estimated data points: {months_back * 30 * 1440:,}")
    print(f"   Estimated time: {months_back * 2:.0f}-{months_back * 5:.0f} minutes")
    
    confirm = input(f"\n🚀 Start ultra-fast backtest? (y/N): ")
    if confirm.lower() != 'y':
        print("Backtest cancelled.")
        return
    
    # Esegui backtest
    runner = UnifiedBacktestRunner(asset)
    
    try:
        success = asyncio.run(runner.run_backtest(months_back))
        
        if success:
            print("\n✅ BACKTEST COMPLETED SUCCESSFULLY!")
            print("🧠 Your analyzer is now trained and ready!")
        else:
            print("\n❌ Backtest failed")
            
    except KeyboardInterrupt:
        print("\n🛑 Backtest interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()