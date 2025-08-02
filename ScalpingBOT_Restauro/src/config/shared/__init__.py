"""
Shared configuration modules - CONSOLIDATION FIX
Contains unified configurations that eliminate duplication across the system.

CLAUDE_RESTAURO.md COMPLIANCE:
- ✅ Single source of truth for shared configurations
- ✅ Zero duplication
- ✅ Fail fast error handling
"""

from .rate_limiting_config import (
    UnifiedRateLimitingConfig,
    RateLimitCategory,
    RateLimitRule,
    get_rate_limiting_config,
    set_rate_limiting_config,
    get_legacy_system_rate_limits,
    get_legacy_monitoring_rate_limits,
    DEFAULT_RATE_LIMITING_CONFIG
)

__all__ = [
    'UnifiedRateLimitingConfig',
    'RateLimitCategory', 
    'RateLimitRule',
    'get_rate_limiting_config',
    'set_rate_limiting_config',
    'get_legacy_system_rate_limits',
    'get_legacy_monitoring_rate_limits',
    'DEFAULT_RATE_LIMITING_CONFIG'
]