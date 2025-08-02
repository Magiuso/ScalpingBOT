#!/usr/bin/env python3
"""
MT5 Utility Functions - Common logic for MT5 interfaces
"""

from datetime import datetime


def calculate_price_from_bid_ask(last_price: float, bid: float, ask: float) -> float:
    """Calculate price from bid/ask when last price is 0 (CFD data)"""
    if last_price > 0:
        return last_price
    
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    
    # No valid price available
    raise ValueError(f"Cannot calculate price: last={last_price}, bid={bid}, ask={ask}")


def parse_mt5_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string in MT5 format"""
    if not timestamp_str:
        raise ValueError("Timestamp string is empty")
    
    try:
        return datetime.strptime(timestamp_str, '%Y.%m.%d %H:%M:%S')
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format '{timestamp_str}': expected YYYY.MM.DD HH:MM:SS")