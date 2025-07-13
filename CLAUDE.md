# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with this sophisticated algorithmic trading system.

## 📋 Project Overview

**ScalpingBOT** is an enterprise-grade automated trading system that integrates MetaTrader 5 with Python for advanced scalping operations. The system combines:

- **MT5 Integration**: 21 sophisticated MQL5 modules for order execution and position management
- **Python ML Engine**: 15,211 lines of advanced machine learning analysis with LSTM, CNN, Transformer models
- **Event-Driven Architecture**: Real-time processing with microsecond-level optimizations
- **Champion Competition System**: Self-improving algorithms with adaptive learning

### Key Characteristics
- **Performance**: 100,000+ ticks/second processing capability
- **Architecture**: Event-driven, modular design with unified orchestration  
- **ML Integration**: 6 model types with sophisticated competition framework
- **Fault Tolerance**: Comprehensive error handling with graceful degradation
- **Production Ready**: Enterprise-grade validation and monitoring

---

## 🏗️ Core Architecture Components

### 1. **Unified Analyzer System** (`src/Unified_Analyzer_System.py`)
**Central orchestration hub for all system components**

#### System Modes
- **PRODUCTION**: Minimal logging, maximum performance
- **DEVELOPMENT**: Normal logging with debugging enabled
- **TESTING**: Verbose logging with full diagnostics
- **DEMO**: Rich console output for showcasing
- **BACKTESTING**: Ultra-high performance with minimal overhead

#### Performance Profiles
```python
class PerformanceProfile(Enum):
    HIGH_FREQUENCY = "high_frequency"    # Trading at ultra-high speeds
    NORMAL = "normal"                    # Standard trading operations
    RESEARCH = "research"                # Backtesting and analysis
    BACKTESTING = "backtesting"         # Maximum performance mode
```

#### Rate Limiting Intelligence
```python
rate_limits = {
    'tick_processing': 100,        # Max 100 tick events/sec
    'predictions': 50,             # Max 50 prediction events/sec
    'validations': 25,             # Max 25 validation events/sec
    'training_events': 1,          # Log every training event
    'champion_changes': 1,         # Log every champion change
    'performance_metrics': 10      # Max 10 metric events/sec
}
```

### 2. **Advanced Market Analyzer** (`src/Analyzer.py`)
**15,211 lines of sophisticated ML analysis engine**

#### Machine Learning Models
1. **Support/Resistance Detection** (5 algorithms)
   - Classical Pivot Points with 6-level calculation
   - Volume Profile Analysis (70th percentile high-volume nodes)  
   - LSTM Neural Networks with 5+ validation checkpoints
   - Statistical ML Analysis (frequency-based level detection)
   - Transformer AI (advanced pattern-based prediction)

2. **Pattern Recognition** (4 algorithms)
   - CNN Pattern Recognition (25 classical patterns)
   - LSTM Sequence Analysis (10 sequential patterns)
   - Transformer Advanced AI (16 institutional patterns)
   - Ensemble Consensus (multi-algorithm voting)

3. **Bias Detection** (4 algorithms)
   - LSTM Sentiment Analysis (3-class directional bias)
   - Volume-Price Analysis (professional buying/selling pressure)
   - Momentum ML (multi-timeframe RSI/MACD integration)
   - MultiModal Ensemble (4-component analysis)

4. **Trend Analysis** (5 algorithms)
   - RandomForest, LSTM, GradientBoosting, Transformer, Ensemble

5. **Volatility Prediction** (3 algorithms)
   - GARCH, LSTM, Realized volatility models

6. **Momentum Analysis** (3 algorithms)
   - RSI-based, MACD-based, Neural network momentum

#### Champion Competition Framework
```python
# Algorithm competition with performance tracking
class AlgorithmCompetition:
    champion_threshold: float = 0.20          # 20% improvement to dethrone
    min_predictions_for_champion: int = 100   # Minimum predictions required
    reality_check_interval_hours: int = 6     # Performance validation frequency
    emergency_accuracy_drop: float = 0.30     # 30% drop triggers emergency stop
    emergency_consecutive_failures: int = 10   # 10 failures trigger stop
```

#### Learning Phase Configuration
```python
# Comprehensive learning requirements
learning_config = {
    'min_learning_days': 90,                    # 90 days minimum learning
    'learning_ticks_threshold': 50000,          # 50,000 ticks required
    'learning_mini_training_interval': 1000,    # Train every 1,000 ticks
    'accuracy_threshold': 0.60,                # 60% accuracy requirement
    'max_tick_buffer_size': 100000             # Memory buffer limit
}
```

#### Ultra-Optimized Technical Analysis
**Performance Innovations:**
- **Zero-Copy Memory Management**: In-place operations with pre-allocated arrays
- **Vectorized Calculations**: All indicators use NumPy out= parameter
- **Cache-First Architecture**: Intelligent indicator caching with LRU eviction
- **Manual Mathematical Implementations**: Linear regression without library overhead

### 3. **ML Training Logger** (`ML_Training_Logger/`)
**Comprehensive event collection and storage system**

#### Event Collection (`Event_Collector.py`)
- Non-invasive hook system for event capture
- Intelligent filtering with configurable rate limits
- Batch processing for 50x performance improvement
- Multi-source event aggregation

#### Display Management (`Display_Manager.py`)
- Real-time terminal dashboard with ANSI control
- Progress visualization with ETA calculations
- Scroll mode and dashboard mode support
- Memory-safe display operations

#### Storage Management (`Storage_Manager.py`)
- Automatic file rotation and compression
- Multiple export formats (JSON, CSV, Parquet, H5)
- Memory-safe streaming operations
- Configurable retention policies (180-day default)

#### Configuration Management (`Unified_ConfigManager.py`)
```python
class ConfigVerbosity(Enum):
    MINIMAL = "minimal"      # Only critical events
    STANDARD = "standard"    # Important events + summaries  
    VERBOSE = "verbose"      # All events + diagnostics
    DEBUG = "debug"          # Complete debugging + timing
```

### 4. **MetaTrader 5 Integration**
**21 sophisticated MQL5 modules for professional trading**

#### Core Expert Advisors

##### MACD+EMA_Modulare.mq5 - Main Trading EA
**Advanced scalping strategy implementation:**
- **Risk Management**: Auto-lot calculation with 1.0% default risk
- **Strategy Components**: MACD+EMA combination with M5 timeframe
- **Cooldown System**: 30-second minimum between operations
- **Order Management**: Maximum 5000 orders per asset
- **Magic Number**: 123456 for strategy identification

**Strategy Logic:**
```mql5
// EMA Block: Trend detection
- EMA5 and EMA20 with ADX validation
- Strong slope threshold: 0.08% for trend validation
- Minimum EMA distance: 0.1% to avoid noise
- ADX threshold: 25.0 for trend strength

// MACD Block: Momentum confirmation  
- Standard 12/26/9 MACD parameters
- Zero-line proximity detection (0.05% ATR ratio)
- Cross penalties for counter-trend signals
- Slope-based acceleration bonuses

// Body Momentum: Candle analysis
- Body strength analysis for entry refinement
- Price action confirmation
```

##### TestBridge.mq5 - Python Integration
**Bridge system for real-time data communication:**
- **Buffer System**: 100,000 characters max, 1,000 lines max
- **Periodic Writing**: Automatic file writes every 10 seconds
- **Data Collection**: Historical buffer of 1,000 ticks
- **Timer Integration**: 1-second timer for operations

#### Advanced MQL5 Module Architecture

##### Data Structure Modules (`Include/Map/`)
1. **MapString.mqh**: Generic string-to-struct mappings
2. **MapLong_T.mqh**: Optimized numeric key mappings
3. **MapLong_Bool.mqh**: Boolean state management
4. **MapLong_Int.mqh**: Integer tracking with lifecycle management
5. **CMapStringToCampionamentoState.mqh**: LRU eviction for market analysis

##### Core Infrastructure (`Include/ScalpingBot/`)
6. **Utility.mqh**: Enterprise caching with O(1) performance
   - CEMACache: Multi-period EMA handle management
   - CMACDCache: Optimized MACD handle caching
   - CRSICacheOptimized: Hash table with bucket chains for 100+ assets

7. **SafeBuffer.mqh**: Secure MT5 buffer operations
8. **RollingStats.mqh**: O(1) real-time statistical calculations
9. **SymbolFillingCache.mqh**: Broker-specific execution optimization

##### Signal Generation Modules
10. **EntryManager.mqh**: Multi-signal aggregation orchestrator
11. **MicroTrendScanner.mqh**: 7-component micro-trend detection
12. **RSIMomentum.mqh**: Enterprise RSI with auto-direction detection
13. **SpikeDetection.mqh**: 13x performance improved price action analysis
14. **Campionamento.mqh**: Statistical sampling with progressive scoring

##### Position Management
15. **TrailingStop.mqh**: Dynamic trailing stop system
16. **PostTradeEMACheck.mqh**: Post-trade validation
17. **RSI_SLTP_Dinamici.mqh**: Adaptive SL/TP calculation
18. **ExtremesPrice.mqh**: Real-time peak/worst price monitoring

##### Recovery System
19. **RecoveryTriggerManager.mqh**: Centralized trigger management
20. **RecoveryBlock.mqh**: Automated recovery validation
21. **OpenTradeRecovery.mqh**: Recovery order execution

### 5. **Event-Driven Architecture**

#### Data Flow Pipeline
```
MT5 Tick Data → SafeBuffer → Technical Analysis → ML Models → Algorithm Competition
      ↓              ↓              ↓              ↓              ↓
Event Collection → Rate Limiting → Batch Processing → Champion Selection → Order Execution
      ↓              ↓              ↓              ↓              ↓
Storage Manager → Display Manager → Performance Monitor → Observer Integration → Recovery System
```

#### Event Types and Buffers
```python
# Memory-safe event buffers with automatic size limits
event_buffers = {
    '_prediction_events_buffer': deque(maxlen=500),
    '_champion_events_buffer': deque(maxlen=200),
    '_error_events_buffer': deque(maxlen=300),
    '_training_events_buffer': deque(maxlen=200),
    '_mt5_events_buffer': deque(maxlen=1000),
    '_diagnostic_events_buffer': deque(maxlen=150),
    '_emergency_events_buffer': deque(maxlen=100)
}
```

---

## ⚡ Performance Optimizations

### High-Frequency Trading Features

#### Memory Management Excellence
- **Bounded Buffers**: All event buffers have strict size limits
- **Zero-Copy Operations**: In-place calculations with pre-allocated arrays
- **Intelligent Caching**: LRU-based eviction with usage scoring
- **Aggressive Cleanup**: Automatic garbage collection in backtesting mode

#### Processing Optimizations  
- **Vectorized Calculations**: NumPy-optimized indicator calculations
- **Batch Event Processing**: 50x performance improvement over individual processing
- **Threading Optimization**: Parallel processing for concurrent operations
- **Cache-First Architecture**: Multi-level caching (indicators, handles, shapes)

#### Backtesting Mode Optimizations
```python
# Ultra-high performance configuration
backtesting_config = {
    'minimal_logging_overhead': True,
    'aggressive_memory_cleanup': True,
    'batch_event_processing': True,
    'threading_optimization': True,
    'rate_limiting_intelligence': True,
    'target_performance': '100000+ ticks/second'
}
```

### Neural Network Performance Features

#### Revolutionary Adapter System
- **Dynamic Dimension Handling**: Automatic adaptation to any input size
- **Intelligent Caching**: LRU with usage-based scoring for adapters
- **Memory Optimization**: Auto-cleanup with forced garbage collection
- **Performance Tracking**: Detailed statistics and optimization recommendations

#### Comprehensive NaN Protection
- **Multi-Stage Validation**: Input → Processing → Output protection
- **Graceful Degradation**: Fallback strategies at every processing step
- **Detailed Error Reporting**: Precise NaN/Inf counting and logging
- **Robust Recovery**: Hierarchical fallback mechanisms

---

## 📊 Testing Infrastructure

### Comprehensive Test Framework (`tests/test_backtest.py`)
**3,862 lines of sophisticated ML learning validation**

#### 13-Phase Testing Pipeline
1. **Setup and Prerequisites**: Component initialization
2. **Data Loading and MT5 Connection**: Real data acquisition
3. **ML Learning Execution**: Core learning process validation
4. **Persistence Verification**: State saving validation
5. **Health Metrics Verification**: Performance assessment
6. **Error Scenarios Testing**: Edge case handling
7. **Unified System Events Testing**: Event processing validation
8. **Performance Monitoring**: System metrics validation
9. **Persistence Integration**: Cross-component persistence
10. **ML Learning Progress Tracking**: Learning monitoring
11. **Unified ML Persistence**: Integrated persistence testing
12. **Learning Phase Optimization**: Performance tuning validation
13. **ML Training Logger Events**: Event logging validation

#### Memory-Aware Testing
- **Progressive File Processing**: Memory-safe data loading in chunks
- **Real-Time Memory Monitoring**: 80% threshold with automatic cleanup
- **Batch Processing Integration**: Memory-efficient tick processing
- **Comprehensive Cleanup**: Proper resource management and statistics

### System Integration Testing (`utils/System_Integration_Test.py`)
**8-Phase comprehensive system verification:**
1. Basic imports and dependencies verification
2. Advanced module imports (ML libraries, MT5)
3. Project module initialization and configuration  
4. Cross-module integration testing
5. Data flow validation through complete pipeline
6. Logging systems integration verification
7. Performance and memory usage monitoring
8. Error handling and edge case validation

---

## 🔧 Configuration Management

### Unified Configuration System

#### Configuration Profiles
```python
# Predefined system profiles
class ConfigSystemProfile(Enum):
    PRODUCTION_TRADING = "production_trading"
    DEVELOPMENT_TESTING = "development_testing"  
    RESEARCH_ANALYSIS = "research_analysis"
    DEMO_SHOWCASE = "demo_showcase"
    MONITORING_ONLY = "monitoring_only"
```

#### Performance-Oriented Settings
```python
# Example backtesting configuration
config = UnifiedConfig.for_backtesting("USTEC")
config.update({
    'system_mode': SystemMode.BACKTESTING,
    'performance_profile': PerformanceProfile.HIGH_FREQUENCY,
    'max_tick_buffer_size': 50000,
    'learning_phase_enabled': True,
    'min_learning_days': 1,  # Reduced for testing
    'enable_performance_monitoring': True,
    'async_processing': True,
    'batch_size': 25,
    'max_queue_size': 10000
})
```

#### Technical Indicator Configuration
```python
# Standard technical analysis parameters
indicators_config = {
    'macd': {'fast': 12, 'slow': 26, 'signal': 9},
    'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
    'bollinger_bands': {'period': 20, 'std_dev': 2.0},
    'atr': {'period': 14},
    'ema': {'fast': 5, 'slow': 20},
    'adx': {'period': 14, 'threshold': 25.0}
}
```

---

## 💻 Development Workflow

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv analyzer_env
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
# Comprehensive ML learning test (3,862 lines)
python tests/test_backtest.py

# Unified backtest system
python tests/test_run_unified_backtest.py

# System integration verification
python utils/System_Integration_Test.py

# Run with pytest framework
pytest tests/ -v
```

### Analysis and Debugging
```bash
# Analyze backtest data
python src/analyze_file.py test_analyzer_data/backtest_USTEC_20250103_20250703.jsonl

# Debug file access
python utils/debug_file_access.py

# Check library installations  
python utils/library_check.py

# Code structure analysis
python src/analyze_file.py src/Analyzer.py
```

### Backtesting Operations
```bash
# Quick backtest configuration
python -c "
from src.Unified_Analyzer_System import UnifiedAnalyzerSystem, UnifiedConfig
config = UnifiedConfig.for_backtesting('USTEC')
system = UnifiedAnalyzerSystem(config)
# Start backtesting...
"

# MT5 data export and backtesting
python src/MT5BacktestRunner.py
```

---

## 🔍 Key Integration Patterns

### Signal Flow Architecture
```
Market Data → SafeBuffer → Individual Analysis Modules → EntryManager Aggregation → Order Execution
     ↓              ↓              ↓                    ↓                        ↓
 Data Validation → RollingStats → Signal Generation → Score Aggregation → Position Management
     ↓              ↓              ↓                    ↓                        ↓
Statistical Analysis → Pattern Detection → Direction Detection → Adaptive Thresholds → Recovery System
```

### Data Communication Pattern
```
MT5 Tick Data → AnalyzerBridge.mqh → JSON File Buffer → Python ML Analysis → Trading Signals → MT5 Execution
```

### Risk Management Layers
1. **Data Validation**: SafeBuffer, symbol/timeframe verification, price validation
2. **Signal Filtering**: Spread analysis, volatility thresholds, market regime detection  
3. **Position Sizing**: Risk-based calculation, margin validation, broker compliance
4. **Trade Management**: Dynamic SL/TP, trailing stops, post-trade monitoring
5. **Recovery Protection**: Attempt limits, ban systems, duplicate prevention

### Event Processing Flow
```
Component Events → Event Buffers → Rate Limiting → Batch Processing → Slave Module → Storage → Analysis
```

---

## 📈 Performance Metrics & Monitoring

### Backtesting Performance
- **Target Speed**: 100,000+ ticks/second processing
- **Memory Usage**: Optimized for months of historical data
- **Latency**: Microsecond-level processing for real-time scenarios
- **Throughput**: Configurable based on system resources

### Real-Time Trading Metrics
- **Tick Processing**: Sub-millisecond latency
- **Event Logging**: Intelligent rate limiting prevents bottlenecks
- **Memory Management**: Automatic cleanup and optimization
- **System Monitoring**: Real-time performance tracking

### Health Score Calculation
```python
# Multi-factor system health assessment
health_metrics = {
    'prediction_accuracy': '>= 60%',      # Model performance threshold
    'champion_availability': 'Required',   # Active champions per model type
    'emergency_stops': '== 0',            # No active emergency stops
    'learning_progress': '>= 10%',         # Minimum learning completion
    'memory_usage': '<= 80%',             # Memory utilization limit
    'processing_speed': '<= 10s/tick'     # Maximum processing time
}
```

---

## 🛡️ Security & Risk Management

### Trading Safety Features
- **Position Limits**: Maximum 5000 orders per asset
- **Risk Percentage**: Default 1.0% risk per trade
- **Emergency Stops**: Automatic algorithm disabling on poor performance
- **Margin Validation**: Concurrent position protection
- **Recovery Protection**: Infinite loop prevention with ban systems

### Data Security
- **No API Keys in Code**: Environment variables for sensitive data
- **Automatic Validation**: Trading parameters validation
- **Secure Buffer Operations**: SafeBuffer wrapper prevents crashes
- **State Encryption**: Secure model and state persistence

### Configuration Security
```python
# Secure configuration practices
sensitive_config = {
    'mt5_account': os.getenv('MT5_ACCOUNT'),
    'mt5_password': os.getenv('MT5_PASSWORD'), 
    'mt5_server': os.getenv('MT5_SERVER'),
    'risk_limits': {
        'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '5.0')),
        'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '0.5'))
    }
}
```

---

## 🎯 Success Criteria & Validation

### Learning Phase Success
- **Health Score**: > 70% system health
- **Prediction Confidence**: > 70% model confidence
- **Champion Status**: Active champion for every ModelType
- **Model Persistence**: Successful saving/loading of ML models
- **System Stability**: No emergency stops during learning

### Production Readiness
- **Performance**: Consistent 100,000+ ticks/second processing
- **Memory Efficiency**: < 80% memory utilization under load
- **Error Rate**: < 1% processing errors
- **Recovery Success**: > 95% successful recovery from errors
- **Prediction Accuracy**: > 60% accuracy across all model types

---

## 📚 Documentation & Support

### Code Analysis Tools
- **AST-Based Analysis**: Python code structure analysis with complexity metrics
- **Performance Profiling**: Detailed statistics and optimization recommendations
- **Memory Usage Tracking**: Real-time monitoring with threshold alerts
- **Integration Testing**: Comprehensive validation across all components

### Logging & Monitoring
- **Structured Logging**: Event-driven logging with multiple verbosity levels
- **Performance Metrics**: Real-time system performance tracking
- **Error Analysis**: Comprehensive error tracking with lessons learned
- **Observer Integration**: External feedback integration for continuous improvement

This comprehensive documentation provides Claude Code with complete understanding of this sophisticated algorithmic trading system, enabling effective assistance with development, testing, configuration, and optimization tasks.

---

## 🔮 Future Enhancements

### Architectural Evolution
1. **Microservices Architecture**: Containerization for production deployment
2. **API Gateway**: Standardized API layer for external integrations
3. **Database Integration**: Persistent storage for historical analysis
4. **Real-Time Monitoring**: Comprehensive observability stack
5. **Security Hardening**: Authentication, authorization, and encryption
6. **Horizontal Scaling**: Multi-instance deployment capabilities

### Advanced Features
- **Multi-Asset Portfolio**: Simultaneous trading across multiple instruments
- **Cross-Market Analysis**: Integration of forex, stocks, commodities
- **News Integration**: NLP-based news sentiment analysis
- **Social Trading**: Community-based strategy sharing and validation
- **Mobile Integration**: Real-time monitoring and control via mobile apps

This represents a production-ready, institutional-grade algorithmic trading system suitable for mission-critical financial applications where reliability, performance, and accuracy are paramount.