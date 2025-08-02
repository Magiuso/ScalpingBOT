"""
MT5 Adapter - Clean interface for MetaTrader5 integration
Isolates all MT5-specific code and provides type-safe interface
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Protocol, runtime_checkable
from dataclasses import dataclass
import logging
import numpy as np


@dataclass
class MT5Tick:
    """Type-safe tick data structure"""
    time: int  # Unix timestamp
    bid: float
    ask: float
    last: float
    volume: int
    flags: int
    
    @classmethod
    def from_mt5_tick(cls, tick: Any) -> 'MT5Tick':
        """Create from MT5 tick array element"""
        return cls(
            time=int(tick['time']),
            bid=float(tick['bid']),
            ask=float(tick['ask']),
            last=float(tick['last']) if 'last' in tick.dtype.names else float(tick['bid']),
            volume=int(tick['volume']) if 'volume' in tick.dtype.names else 0,
            flags=int(tick['flags']) if 'flags' in tick.dtype.names else 0
        )


@runtime_checkable
class MT5Interface(Protocol):
    """Protocol defining MT5 operations we need"""
    
    def initialize(self) -> bool:
        """Initialize MT5 connection"""
        ...
    
    def shutdown(self) -> None:
        """Shutdown MT5 connection"""
        ...
    
    def copy_ticks_range(self, symbol: str, date_from: datetime, 
                        date_to: datetime, flags: int) -> Optional[Any]:
        """Copy ticks in date range"""
        ...


class MT5Adapter:
    """Clean adapter for MT5 operations"""
    
    def __init__(self):
        self.logger = logging.getLogger('MT5Adapter')
        self._mt5: Optional[Any] = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize MT5 connection - FAIL FAST"""
        if self._initialized:
            return
        
        try:
            # Import MT5 only when needed
            import MetaTrader5 as mt5  # type: ignore
            self._mt5 = mt5
        except ImportError:
            raise ImportError(
                "MetaTrader5 package not installed. "
                "Install with: pip install MetaTrader5"
            )
        
        # Try to initialize
        try:
            success = self._mt5.initialize()  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize MT5: {e}. "
                "Ensure MT5 terminal is installed and accessible."
            )
        
        if not success:
            raise RuntimeError(
                "MT5 initialization returned False. "
                "Check if MT5 terminal is running and accessible."
            )
        
        self._initialized = True
        self.logger.info("MT5 adapter initialized successfully")
    
    def shutdown(self) -> None:
        """Shutdown MT5 connection safely"""
        if not self._initialized or not self._mt5:
            return
        
        try:
            self._mt5.shutdown()  # type: ignore
        except Exception as e:
            self.logger.warning(f"Error during MT5 shutdown: {e}")
        finally:
            self._initialized = False
            self._mt5 = None
    
    def get_ticks_range(self, symbol: str, start_date: datetime, 
                       end_date: datetime) -> List[MT5Tick]:
        """Get ticks in date range - type-safe interface"""
        if not self._initialized:
            raise RuntimeError("MT5 adapter not initialized")
        
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be non-empty string")
        if not isinstance(start_date, datetime):
            raise TypeError("start_date must be datetime")
        if not isinstance(end_date, datetime):
            raise TypeError("end_date must be datetime")
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")
        
        # Get MT5 copy flags constant
        COPY_TICKS_ALL = getattr(self._mt5, 'COPY_TICKS_ALL', 1)  # type: ignore
        
        # Call MT5 function
        try:
            ticks_array = self._mt5.copy_ticks_range(  # type: ignore
                symbol, start_date, end_date, COPY_TICKS_ALL
            )
        except Exception as e:
            raise RuntimeError(f"Failed to get ticks from MT5: {e}")
        
        if ticks_array is None:
            return []
        
        # Convert to type-safe objects
        result = []
        for tick in ticks_array:
            try:
                result.append(MT5Tick.from_mt5_tick(tick))
            except Exception as e:
                self.logger.warning(f"Skipping invalid tick: {e}")
                continue
        
        return result
    
    def __enter__(self):
        """Context manager support"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.shutdown()
        return False