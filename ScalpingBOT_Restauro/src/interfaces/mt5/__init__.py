"""
MT5 integration interfaces - MetaTrader 5 bridge components
"""

from .mt5_backtest_runner import (
    MT5BacktestRunner,
    MT5DataExporter,
    BacktestConfig,
    create_backtest_config,
    create_backtest_runner
)

from .mt5_bridge_reader import (
    MT5BridgeReader,
    MT5TickData,
    create_mt5_bridge_reader
)

# UnifiedAnalyzerSystem removed - will be redesigned in FASE 6 as core orchestrator
# Not an MT5 interface - was incorrectly placed here

from .mt5_adapter import (
    MT5Adapter,
    MT5Tick
)

__all__ = [
    # Backtest runner
    'MT5BacktestRunner',
    'MT5DataExporter', 
    'BacktestConfig',
    'create_backtest_config',
    'create_backtest_runner',
    
    # Bridge reader
    'MT5BridgeReader',
    'MT5TickData',
    'create_mt5_bridge_reader',
    
    # MT5 Adapter
    'MT5Adapter',
    'MT5Tick'
    
    # UnifiedAnalyzerSystem removed - will be redesigned in FASE 6
]