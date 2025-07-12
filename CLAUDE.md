# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a scalping bot that integrates MetaTrader 5 with Python for automated trading operations. The system combines:
- **MT5**: For order execution (MQL5 code in `MT5/` directory)
- **Python**: For analysis and strategies using machine learning

## Key Architecture Components

### 1. Unified Analyzer System (`src/Unified_Analyzer_System.py`)
- Main entry point for the analysis system
- Handles different operation modes: PRODUCTION, DEVELOPMENT, TESTING, DEMO, BACKTESTING
- Integrates performance optimization for high-frequency trading
- Manages rate limiting and batch processing

### 2. Advanced Market Analyzer (`src/Analyzer.py`)
- Core analysis engine with ML models (LSTM, Random Forest, Gradient Boosting)
- Handles technical indicators calculation (RSI, MACD, Bollinger Bands, etc.)
- Manages learning phase (minimum 90 days) and model competition system
- Champion/challenger model architecture for continuous improvement

### 3. ML Training Logger (`ML_Training_Logger/`)
- Handles event collection and storage
- Manages configuration and display
- Optimized for high-frequency data logging

### 4. MetaTrader 5 Integration
- Bridge files: `AnalyzerBridge.mqh`, `TestBridge.mq5`
- Trading modules in `MT5/11_ScalpingBot_MACD+EMA+Python_In creazione/`
- Implements MACD+EMA strategy with Python integration

## Development Commands

### Environment Setup
```bash
# Create virtual environment (if not exists)
python -m venv analyzer_env

# Activate environment
# Windows:
analyzer_env\Scripts\activate
# Linux/Mac:
source analyzer_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install TA-Lib (Windows - use provided wheel)
pip install TA_Lib-0.4.28-cp310-cp310-win_amd64.whl
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_backtest.py
pytest tests/test_run_unified_backtest.py

# Run with verbose output
pytest -v tests/
```

### Running Backtests
```bash
# Run unified backtest system
python tests/test_run_unified_backtest.py

# Run MT5 backtest runner
python src/MT5BacktestRunner.py
```

### Analysis Tools
```bash
# Analyze backtest data
python src/analyze_file.py test_analyzer_data/backtest_USTEC_20250103_20250703.jsonl

# System integration test
python utils/System_Integration_Test.py

# Debug file access
python utils/debug_file_access.py

# Check library installations
python utils/library_check.py
```

## Configuration Management

The system uses `UnifiedConfig` (in `src/Unified_Analyzer_System.py`) for centralized configuration:
- **SystemMode**: Controls logging and performance profiles
- **PerformanceProfile**: HIGH_FREQUENCY, NORMAL, RESEARCH, BACKTESTING
- **Rate Limiting**: Configurable limits for tick processing, predictions, validations

For backtesting:
```python
config = UnifiedConfig.for_backtesting("USTEC")
system = UnifiedAnalyzerSystem(config)
```

## Important Technical Details

### Learning Phase
- Minimum 90 days of data required before making predictions
- 50,000 ticks threshold for completing learning phase
- Mini-training every 1,000 ticks during learning

### Model Competition System
- Champion model must be 20% better to dethrone current champion
- Minimum 100 predictions required to become champion
- Reality checks every 6 hours
- Emergency stop triggers on 30% accuracy drop or 10 consecutive failures

### Performance Optimization
- Batch processing for events (50x faster)
- Rate limiting for demo predictor
- Memory cleanup and threading optimization
- Backtesting mode with minimal overhead

### Data Storage
- JSONL format for backtest data
- H5/Parquet support for large datasets
- CSV export for analysis
- Automatic data cleanup after 180 days

## MetaTrader 5 Integration

### Expert Advisors

#### 1. MACD+EMA_Modulare.mq5 (`MT5/11_ScalpingBot_MACD+EMA+Python_In creazione/Experts/ScalpingBot/`)
**Main Expert Advisor implementing the MACD+EMA scalping strategy**

**Key Features:**
- **Risk Management**: Auto-lot calculation with configurable risk percentage (default 1.0%)
- **Strategy Parameters**: MACD+EMA combination with M5 timeframe (configurable)
- **Cooldown System**: 30-second minimum between MACD operations
- **Order Limits**: Maximum 5000 orders per asset
- **Magic Number**: 123456 for strategy identification

**Core Strategy Components:**
- **EMA Block**: Trend detection using EMA5 and EMA20 with ADX validation
  - Strong slope threshold: 0.08% for trend validation
  - Minimum EMA distance: 0.1% to avoid noise
  - ADX threshold: 25.0 for trend strength confirmation
- **MACD Block**: Momentum confirmation using 12/26/9 MACD parameters
  - Zero-line proximity detection (0.05% ATR ratio)
  - Cross penalties for counter-trend signals
  - Slope-based acceleration bonuses
- **Body Momentum**: Candle body analysis for entry refinement

#### 2. TestBridge.mq5 (`MT5/11_ScalpingBot_MACD+EMA+Python_In creazione/Experts/ScalpingBot/`)
**Test EA for buffer system and Python integration**

**Features:**
- **Buffer System**: Configurable memory buffer (100,000 characters max, 1,000 lines max)
- **Periodic Writing**: Automatic file writes every 10 seconds
- **Data Collection**: Historical buffer size of 1,000 ticks
- **Timer Integration**: 1-second timer for automated operations

### Key MQL5 Modules

#### 1. AnalyzerBridge.mqh (`Include/ScalpingBot/`)
**Bridge between MT5 and Python analyzer system**

**Core Functionality:**
- **Tick Data Structure**: Comprehensive tick packet with timestamp, OHLC, volume, spread, volatility
- **Buffer Management**: Circular buffer system for historical data
- **Periodic File Writing**: Configurable write intervals with memory buffering
- **Market State Detection**: Real-time market condition analysis
- **Performance Monitoring**: Tick counting and packet statistics

**Data Flow:**
```
MT5 Tick → TickDataPacket → Memory Buffer → Periodic File Write → Python Analyzer
```

#### 2. EMA_MACD_BodyScore.mqh (`Include/ScalpingBot/`)
**Pre-validation module for EMA+MACD+Body Momentum analysis**

**Strategy Orchestration:**
- **Direction Modes**: EMA_MASTER, MAJORITY_VOTE, WEIGHTED_AVG, STRONGEST_SIGNAL
- **Conflict Resolution**: REDUCE_SCORE, BLOCK_SIGNAL, FOLLOW_PRIMARY
- **Score Normalization**: Each block contributes 0-10 points to total score
- **Coherence Detection**: Bonus for aligned signals, penalties for conflicts

**Validation Results:**
- Total score with individual EMA, MACD, Body Momentum contributions
- Direction consensus with conflict detection
- ATR-based dynamic thresholds
- Detailed reasoning for debugging

#### 3. OpenTradeMACDEMA.mqh (`Include/ScalpingBot/`)
**Order execution module with dynamic SL/TP**

**Entry Management:**
- **Entry Score Calculation**: Multi-factor analysis for trade confirmation
- **Dynamic SL/TP**: ATR and RSI-based stop loss and take profit levels
- **Lot Size Calculation**: Risk-based position sizing with account balance consideration
- **Filling Mode Fallback**: Multiple execution modes for order reliability

**Order Flow:**
```
Entry Score → Direction Analysis → SL/TP Calculation → Lot Sizing → Order Execution
```

### Trading Strategy Logic

#### Signal Generation Process:
1. **EMA Trend Analysis**: 
   - EMA5/EMA20 slope calculation and alignment check
   - ADX confirmation for trend strength
   - Convergence/divergence detection
2. **MACD Confirmation**:
   - Histogram slope analysis for momentum
   - Zero-line proximity checks
   - Signal line crossover validation
3. **Body Momentum**:
   - Candle body strength analysis
   - Price action confirmation
4. **Final Scoring**:
   - Weighted combination of all signals
   - Conflict resolution and coherence bonuses
   - Entry threshold validation (configurable)

#### Risk Management:
- **Position Sizing**: Auto-lot calculation based on account risk percentage
- **Stop Loss**: Dynamic SL based on ATR multiples and market volatility
- **Take Profit**: RSI-adjusted TP levels with trend strength consideration
- **Recovery System**: Smart recovery orders for drawdown management

### Configuration Parameters

#### Risk Management:
- `UseAutoLot`: Enable automatic lot calculation (default: true)
- `RiskPercent`: Risk per trade as percentage (default: 1.0%)
- `MaxAutoLot`: Maximum auto-calculated lot size (default: 0.5)

#### Strategy Settings:
- `Timeframe_MACD_EMA`: Strategy timeframe (default: PERIOD_M5)
- `CooldownMACDSeconds`: Minimum time between operations (default: 30s)
- `MagicNumber_MACD`: Strategy identifier (default: 123456)

#### Technical Indicators:
- `MACD_FastPeriod/SlowPeriod/SignalPeriod`: 12/26/9 (standard MACD)
- `MinADXThreshold`: Minimum ADX for trend validation (default: 25.0)
- `StrongSlopePercentThreshold`: Strong trend threshold (default: 0.08%)

### Data Communication

The system uses a sophisticated data flow:
1. **MT5 Side**: Real-time tick collection with technical indicator calculation
2. **Bridge System**: Buffered data transfer with JSON formatting
3. **Python Side**: Machine learning analysis and strategy refinement
4. **Execution**: MT5 receives trading signals and executes orders

**File Structure:**
```
MT5/11_ScalpingBot_MACD+EMA+Python_In creazione/
├── Experts/ScalpingBot/          # Main EAs
├── Include/ScalpingBot/          # Strategy modules
│   ├── AnalyzerBridge.mqh       # Python integration
│   ├── EMA_MACD_BodyScore.mqh   # Signal generation
│   ├── OpenTradeMACDEMA.mqh     # Order execution
│   ├── TrailingStop.mqh         # Position management
│   ├── RecoveryBlock.mqh        # Recovery system
│   └── Utility.mqh              # Common functions
└── Include/Map/                  # Data structures
```

## Complete MQL5 Module Architecture

### Data Structure Modules (`Include/Map/`)

#### 1. MapString.mqh
**Generic template-based map for string-to-struct mappings**
- Linear storage using parallel arrays (`string keys[]`, `T values[]`)
- Methods: `Insert()`, `IsExist()`, `At()`, `Delete()`, `FindIndex()`
- O(n) complexity with dynamic array resizing

#### 2. MapLong_T.mqh  
**Generic template-based map for long integer keys**
- Similar to MapString but optimized for numeric keys
- Identical memory management with `ArrayResize()` and `ArrayRemove()`

#### 3. MapLong_Bool.mqh
**Specialized map for long keys to boolean values**
- Global instance: `CMapLongBool g_boolMap`
- Global wrappers: `MapGetBool()`, `MapSetBool()`, `MapRemoveBool()`
- Used for managing state flags per ticket/identifier

#### 4. MapLong_Int.mqh
**Specialized map with enhanced lifecycle management**
- Constructor/destructor implementation for proper cleanup
- Additional utility: `GetKeys()` method for key extraction
- Used for integer state tracking

#### 5. CMapStringToCampionamentoState.mqh
**Advanced map with LRU eviction for market analysis**
- Complex `CampionamentoState` objects with OHLC, ADX, scoring data
- Automatic memory management with configurable limits
- LRU eviction strategy with 20% eviction percentage
- Timestamp tracking for access patterns

### Core Infrastructure Modules (`Include/ScalpingBot/`)

#### 6. Utility.mqh - System Foundation
**Enterprise-grade caching system with O(1) performance**

**Key Components:**
- **CEMACache**: Multi-period EMA handle management with validation
- **CMACDCache**: Optimized MACD handle caching with error recovery
- **CRSICacheOptimized**: Hash table with bucket chains for 100+ assets
  - FNV-1a hash algorithm for distribution
  - LRU eviction policy with thread-safe operations
  - Performance statistics tracking

**Critical Features:**
- Handle validation using non-blocking buffer tests
- Asynchronous status tracking (READY, PENDING, ERROR, TIMEOUT)
- Universal constants and broker-specific adaptations

#### 7. SafeBuffer.mqh - Data Integrity
**Secure wrapper for MT5 buffer operations**
- `SafeCopyBuffer()`: Validates handles and ensures proper array ordering
- `SafeCopyTime()`: Safe time series copying with validation
- Prevents system crashes from invalid buffer access
- Used universally across all indicator-based modules

#### 8. RollingStats.mqh - Statistical Engine
**O(1) real-time statistical calculations**

**RollingStatsOptimized Structure:**
- Circular buffer with incremental statistics
- Maintains: sum, sumSquares, min/max, mean, variance, stddev
- Robust statistics: median, MAD (Median Absolute Deviation)

**Key Algorithms:**
- O(1) update with sum/variance incremental calculation
- Robust threshold using MAD * 1.4826 (stddev equivalent)
- Quality flags based on sample size reliability

#### 9. SymbolFillingCache.mqh - Execution Optimization
**Broker-specific order filling mode management**
- Automatic detection and caching of optimal execution methods
- Fallback strategy: IOC → RETURN → FOK
- Eliminates order rejections from incorrect filling modes
- Multi-symbol compatibility for portfolio trading

### Signal Generation Modules

#### 10. EntryManager.mqh - Central Orchestrator
**Sophisticated multi-signal aggregation system**

**EntryScoreResult Structure:**
- Individual module scores (MicroTrend, RSI, EMA_MACD_Body, Campionamento, Spike)
- Direction consensus from all modules
- Advanced RSI metrics with confidence and quality assessment
- Spike detection with context bonuses

**Core Logic:**
- Weighted voting system with quality-based adjustments
- Synergy detection (RSI + Spike bonus, contrasting penalties)
- Adaptive thresholding based on market regime
- Module dependencies: MicroTrendScanner, RSIMomentum, EMA_MACD_BodyScore, Campionamento, SpikeDetection

#### 11. MicroTrendScanner.mqh - Trend Analysis
**Multi-component micro-trend detection**
- 7 configurable components: EMA slope, breakout, momentum, ADX, volatility, volume, spread
- Adaptive timeframe scaling (M1: 0.8x, M5: 1.0x, M15: 1.3x)
- Soft/aggressive modes with dynamic thresholds
- Comprehensive spread impact analysis using ATR ratios

#### 12. RSIMomentum.mqh - Advanced RSI Analysis
**Enterprise-grade RSI with auto-direction detection**
- Hash table architecture supporting 100+ assets simultaneously
- Multi-period RSI derivatives and weighted averages
- Market regime detection (trending, ranging, volatile)
- Quality assessment: poor/fair/good/excellent with confidence scores
- Thread-safe operations with TTL-based state caching

#### 13. SpikeDetection.mqh - Price Action Analysis
**Multi-metric spike detection with 13x performance improvement**
- Analysis components: body ratio, range multiplier, volume analysis, close extremity
- Rolling statistics with O(1) performance optimization
- Advanced noise filtering: spread analysis, gap detection, liquidity scoring
- Market context awareness: trading sessions, news timing
- Robust statistics using median/MAD for outlier detection

#### 14. Campionamento.mqh - Statistical Sampling
**Intra-candle sampling with progressive scoring**
- Controlled frequency sampling within candle formation
- Components: velocity, volume, ADX analysis, threshold checks
- Shadow analysis and historical coherence validation
- Autonomous direction detection with confidence metrics
- Per-symbol state tracking with map-based storage

### Position Management Modules

#### 15. TrailingStop.mqh - Dynamic Position Management
**Advanced trailing stop system**
- Peak/worst price tracking for optimal stop placement
- Anti-flood protection with cooldown periods (default 5 seconds)
- Movement-based unlocking on significant price changes
- Broker compliance with minimum distance validation
- Array-based order tracking with index optimization

#### 16. PostTradeEMACheck.mqh - Post-Trade Validation
**EMA acceleration monitoring for reversal detection**
- EMA3 velocity change analysis post-trade
- Recovery signal generation with ATR-based thresholds
- Multi-factor analysis: EMA acceleration + RSI confirmation + market context
- Ranging market detection and avoidance
- Confidence scoring (0-100) with detailed rejection reasons

#### 17. RSI_SLTP_Dinamici.mqh - Dynamic SL/TP
**Adaptive stop loss and take profit calculation**
- Multi-factor analysis: RSI score + derivative analysis + ADX trend classification
- Dynamic multipliers based on trend strength (weak/medium/strong)
- Score-based classification for decision making
- Conservative defaults with threshold-based validation

#### 18. ExtremesPrice.mqh - Price Tracking
**Real-time peak/worst price monitoring**
- Direction-aware logic (BUY: peak=high, SELL: peak=low)
- Automatic position detection integration
- Threshold-based updates for significant movements
- Called on every tick for immediate price tracking

### Recovery System Modules

#### 19. RecoveryTriggerManager.mqh - Trigger Management
**Centralized recovery trigger system**
- ArrayBasedMap for efficient trigger storage
- State tracking: processed/unprocessed triggers
- Ticket association linking original trades with recovery trades
- Duplicate prevention and memory management

#### 20. RecoveryBlock.mqh - Recovery Validation
**Automated recovery execution system**
- Processes pending recovery triggers automatically
- Attempt tracking with configurable limits
- Ban system preventing infinite recovery loops
- Success validation and cleanup logic

#### 21. OpenTradeRecovery.mqh - Recovery Execution
**Recovery order placement and management**
- Dynamic lot calculation with recovery multipliers
- Opposite direction logic with automatic determination
- Multiple filling modes with fallback strategies
- RSI/ADX-based dynamic SL/TP calculation
- Margin validation and concurrent protection

## Module Integration Patterns

### Performance Architecture
- **O(1) Operations**: Hash tables and circular buffers for constant-time access
- **Caching Strategy**: Multi-level caching (EMA, MACD, RSI, Symbol Filling)
- **Memory Management**: LRU eviction, TTL-based cleanup, automatic handle lifecycle
- **Thread Safety**: Lock mechanisms and state management for concurrent access

### Signal Flow Architecture
```
Market Data → SafeBuffer → Individual Modules → EntryManager → Order Execution
                ↓              ↓                    ↓              ↓
            RollingStats → Signal Generation → Score Aggregation → Position Management
                ↓              ↓                    ↓              ↓
            Statistical → Direction Detection → Adaptive Thresholds → Recovery System
```

### Risk Management Layers
1. **Data Validation**: SafeBuffer, symbol/timeframe verification, price validation
2. **Signal Filtering**: Spread analysis, volatility thresholds, market regime detection
3. **Position Sizing**: Risk-based calculation, margin validation, broker compliance
4. **Trade Management**: Dynamic SL/TP, trailing stops, post-trade monitoring
5. **Recovery Protection**: Attempt limits, ban systems, duplicate prevention

### Adaptive Intelligence Features
- **Market Regime Awareness**: Automatic detection and threshold adjustment
- **Quality Assessment**: Signal confidence and reliability scoring
- **Auto-Direction Detection**: No manual direction input required
- **Broker Adaptation**: Learning and caching of broker-specific requirements
- **Statistical Robustness**: Percentile analysis, outlier detection, noise filtering

## Security Notes

- Never commit API keys or passwords
- Configuration files should use environment variables for sensitive data
- The system includes automatic validation for trading parameters