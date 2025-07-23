# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with this sophisticated algorithmic trading system.

## 📋 Project Overview

**ScalpingBOT** is an enterprise-grade automated trading system that integrates Python-based machine learning analysis with MetaTrader 5 for advanced algorithmic trading operations. The system combines:

- **Advanced ML Engine**: 15,000+ lines of sophisticated machine learning analysis with LSTM, CNN, Transformer, and ensemble models
- **Real-Time Processing**: Event-driven architecture with microsecond-level optimizations
- **Champion Competition System**: Self-improving algorithms with adaptive learning and performance-based selection
- **Comprehensive Testing**: Enterprise-grade validation framework with 13-phase testing pipeline
- **Production-Ready Monitoring**: Real-time event collection, storage, and performance tracking

### Key Characteristics
- **Performance**: 100,000+ ticks/second processing capability
- **Architecture**: Event-driven, modular design with unified orchestration
- **ML Integration**: 6 model types with sophisticated competition framework
- **Fault Tolerance**: Comprehensive error handling with graceful degradation
- **Production Ready**: Enterprise-grade validation, monitoring, and testing

---

## 🏗️ Core Architecture Components

### 1. **Advanced Market Analyzer** (`src/Analyzer.py`)
**15,211 lines of sophisticated ML analysis engine**

#### Machine Learning Models (6 Types):
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
class AlgorithmCompetition:
    champion_threshold: float = 0.20          # 20% improvement to dethrone
    min_predictions_for_champion: int = 100   # Minimum predictions required
    reality_check_interval_hours: int = 6     # Performance validation frequency
    emergency_accuracy_drop: float = 0.30     # 30% drop triggers emergency stop
    emergency_consecutive_failures: int = 10   # 10 failures trigger stop
```

#### Learning Phase Configuration
```python
learning_config = {
    'min_learning_days': 90,                    # 90 days minimum learning
    'learning_ticks_threshold': 50000,          # 50,000 ticks required
    'learning_mini_training_interval': 1000,    # Train every 1,000 ticks
    'accuracy_threshold': 0.60,                # 60% accuracy requirement
    'max_tick_buffer_size': 100000             # Memory buffer limit
}
```

#### Key Classes and Infrastructure:

##### Core Infrastructure:
- **`GradientLogAggregator`**: Intelligent logging rate limiter for gradient issues
- **`LogRateLimiter`**: Global rate limiting system for performance optimization
- **`AnalyzerConfig`**: Configuration management for all analyzer settings
- **`IndicatorsCache`**: High-performance caching with LRU eviction

##### Machine Learning Core:
- **`AdvancedLSTM`**: Sophisticated LSTM with custom architecture
- **`TransformerPredictor`**: Transformer-based prediction model
- **`CNNPatternRecognizer`**: Convolutional neural network for pattern detection
- **`OptimizedLSTMTrainer`**: High-performance LSTM training pipeline

##### Competition System:
- **`AlgorithmCompetition`**: Self-improving system with champion selection
- **`RealityChecker`**: Performance validation and champion dethronement
- **`EmergencyStopSystem`**: Safety mechanism for failing algorithms
- **`ChampionPreserver`**: State management for champion models

##### Asset Management:
- **`AssetAnalyzer`**: Complete analysis engine for individual instruments
  - Manages 6 model types with learning phases
  - Implements adaptive thresholds and performance tracking
  - 90+ day minimum training with continuous validation

##### Master Orchestration:
- **`AdvancedMarketAnalyzer`**: Master coordinator managing all assets
  - Multi-asset portfolio management
  - Event-driven architecture with async processing
  - Integrated logging slave for comprehensive monitoring

### 2. **Unified Analyzer System** (`src/Unified_Analyzer_System.py`)
**Central orchestration hub for performance-optimized operations**

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

#### Key Classes:
- **`UnifiedConfig`**: Comprehensive system configuration with performance tuning
- **`PerformanceMetrics`**: Real-time system performance tracking
- **`PerformanceMonitor`**: Async monitoring with configurable intervals
- **`LoggingSlave`**: Intelligent event processing with batch optimization
- **`UnifiedAnalyzerSystem`**: Main system orchestrator with multiple operational modes

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

### 3. **MT5 Integration Modules**

#### MT5 Backtest Runner (`src/MT5BacktestRunner.py`)
**Historical data processing engine for accelerated learning**

**Key Features:**
- **Dynamic Memory Management**: Adaptive chunk sizing based on available RAM
- **Progressive Processing**: 45-day default chunks, scales down to 15-day minimum
- **Memory Monitoring**: Critical thresholds at 85% warning, 95% emergency stop
- **Integration with UnifiedAnalyzerSystem**: Optimized backtesting performance

#### MT5 Bridge Reader (`src/MT5BridgeReader.py`)
**Real-time data bridge between MetaTrader 5 and Python analysis**

**Architecture:**
- **Multi-threaded**: File monitor, data processor, analyzer feeder threads
- **Queue Management**: Bounded queues (10,000 ticks, 1,000 analysis)
- **File Discovery**: Automatic detection of analyzer_*.jsonl files
- **Session Management**: Start/end session detection and statistics

### 4. **ML Training Logger System** (`ML_Training_Logger/`)
**Comprehensive event collection and monitoring infrastructure**

#### Event Collector (`Event_Collector.py`)
**Advanced event collection and processing system**
- **EventHook System**: Non-invasive method hooking for automatic event capture
- **RateLimiter**: Intelligent event throttling with sliding window algorithm
- **EventBuffer**: Thread-safe event storage with overflow protection
- **Priority Processing**: CRITICAL → HIGH → MEDIUM → LOW event ordering

#### Display Manager (`Display_Manager.py`)
**Real-time terminal visualization with advanced features**
- **Fixed Display Mode**: Real-time updating table with scrollable log area
- **Tree Progress Layout**: Hierarchical display with Unicode symbols
- **ANSI Color Support**: Platform-aware color management
- **Model Progress Tracking**: Individual model status with champion indicators
- **Memory-Safe Display**: Bounded event buffers with automatic cleanup

#### Storage Manager (`Storage_Manager.py`)
**Comprehensive data persistence with multiple formats**
- Automatic file rotation and compression
- Multiple export formats (JSON, CSV, Parquet, H5)
- Configurable retention policies (180-day default)
- Batch processing for performance optimization

#### Unified Configuration Manager (`Unified_ConfigManager.py`)
**Advanced configuration management system**
```python
class ConfigVerbosity(Enum):
    MINIMAL = "minimal"      # Only critical events
    STANDARD = "standard"    # Important events + summaries
    VERBOSE = "verbose"      # All events + diagnostics
    DEBUG = "debug"          # Complete debugging + timing
```

### 5. **Analyzer Logging Slave Module** (`modules/Analyzer_Logging_SlaveModule.py`)
**Independent logging system with intelligent aggregation**

#### Core Features:
- **Event Aggregation**: Rate-limiting with configurable thresholds
- **Priority Processing**: Critical events always logged immediately
- **Asynchronous Processing**: Background event processing with thread pools
- **Multi-format Output**: Console, file, CSV, JSON export simultaneously

#### Intelligent Rate Limiting:
```python
rate_limits = {
    'process_tick': 100,           # Log every 100 ticks
    'predictions': 50,             # Log every 50 predictions
    'validations': 25,             # Log every 25 validations
    'champion_changes': 1,         # Log always (rare)
    'emergency_events': 1          # Log always (critical)
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
backtesting_config = {
    'minimal_logging_overhead': True,
    'aggressive_memory_cleanup': True,
    'batch_event_processing': True,
    'threading_optimization': True,
    'rate_limiting_intelligence': True,
    'target_performance': '100000+ ticks/second'
}
```

---

## 📊 Comprehensive Testing Infrastructure

### Main Test Suite (`tests/test_backtest.py`)
**3,862+ lines of sophisticated ML learning validation**

#### 13-Phase Testing Pipeline:
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

#### Memory-Aware Testing:
- **Progressive File Processing**: Memory-safe data loading in chunks
- **Real-Time Memory Monitoring**: 80% threshold with automatic cleanup
- **Batch Processing Integration**: Memory-efficient tick processing
- **Comprehensive Cleanup**: Proper resource management and statistics

### Specialized Model Tests:
- **`test_bias_detection.py`**: Bias detection model validation
- **`test_pattern_recognition.py`**: Pattern recognition algorithm verification
- **`test_sr_detection.py`**: Support/Resistance detection testing
- **`test_trend_analysis.py`**: Trend analysis model validation
- **`test_volatility_prediction.py`**: Volatility prediction accuracy testing
- **`test_rsi_momentum.py`**: RSI-based momentum analysis verification

### Unified Backtest Runner (`tests/test_run_unified_backtest.py`)
**Ultra-fast backtesting with synthetic data generation**
- Generates realistic market data (tick-by-tick simulation)
- High-performance processing (100,000+ ticks/second target)
- Real-time progress visualization with ETA calculations
- Memory-safe data generation with configurable parameters

### ML Learning Verification (`test_complete_ml_learning_verification.py`)
**Complete ML learning system verification with real market data**
- Target: 100K+ real ticks from test_analyzer_data
- All 6 model types validation (RF, GB, LSTM, CNN, Transformer, Ensemble)
- Learning progress tracking and pattern recognition verification
- Real market data processing with synthetic fallback

---

## 🛠️ Utility Infrastructure

### System Integration Testing (`utils/System_Integration_Test.py`)
**8-Phase comprehensive system verification**
1. Basic imports and dependencies verification
2. Advanced module imports (ML libraries, MT5)
3. Project module initialization and configuration
4. Cross-module integration testing
5. Data flow validation through complete pipeline
6. Logging systems integration verification
7. Performance and memory usage monitoring
8. Error handling and edge case validation

### Universal Encoding Support (`utils/universal_encoding_fix.py`)
**Comprehensive Unicode/Emoji support for cross-platform compatibility**
- **Platform Detection**: Automatic Windows vs Unix handling
- **Console Code Page Management**: Automatic UTF-8 (65001) setup
- **Emoji Fallback System**: 70+ emoji → text mappings
- **Logging Integration**: UTF-8 file handlers with automatic reconfiguration

### Advanced ML Integration (`src/utils/analyzer_ml_integration.py`)
**Optimized training pipeline with comprehensive enhancements**

**Optimization Profiles:**
```python
HIGH_PERFORMANCE:    256 hidden, 2 layers  # Speed-optimized
STABLE_TRAINING:     512 hidden, 4 layers  # Stability-focused
RESEARCH_MODE:       768 hidden, 6 layers  # Maximum capacity
PRODUCTION_READY:    384 hidden, 3 layers  # Balanced approach
```

### Code Analysis Utility (`src/analyze_file.py`)
**AST-based analysis tool for large codebases**
- **Python Structure Analysis**: AST parsing for classes, functions, imports
- **Complexity Metrics**: Line counts, comment density, code quality metrics
- **Refactoring Suggestions**: Intelligent file splitting recommendations
- **Report Generation**: Detailed analysis reports with actionable insights

### Diagnostic Utilities:
- **`utils/diagnose_lstm.py`**: LSTM model diagnostic with dynamic dimension testing
- **`utils/debug_file_access.py`**: File system access diagnostics
- **`utils/library_check.py`**: Comprehensive dependency verification system
- **`utils/explore_analyzer_structure.py`**: Code structure analysis and metrics

---

## 📦 Configuration and Dependencies

### Dependencies (`requirements.txt`)
**Comprehensive library ecosystem for enterprise trading:**

#### Core Data Processing:
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
```

#### Deep Learning Stack:
```
tensorflow==2.15.0
keras==2.15.0
transformers==4.30.2
```

#### Trading Integration:
```
MetaTrader5==5.0.45
TA-Lib==0.4.28
pandas-ta==0.3.14b0
```

#### Performance Optimization:
```
asyncio==3.4.3
numba==0.57.1
joblib==1.3.1
```

#### Storage and Monitoring:
```
h5py==3.9.0
pyarrow==12.0.1
colorlog==6.7.0
python-json-logger==2.0.7
```

#### Development Tools:
```
jupyter==1.0.0
ipython==8.14.0
pytest==7.4.0
```

### Special Dependencies:
- **TA-Lib Windows Wheel**: `TA_Lib-0.4.28-cp310-cp310-win_amd64.whl` (pre-compiled)
- **Virtual Environment**: `analyzer_env/` with complete isolated dependency stack

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
# Comprehensive ML learning test (3,862+ lines)
python tests/test_backtest.py

# Unified backtest system with synthetic data
python tests/test_run_unified_backtest.py

# Complete ML learning verification with real data
python test_complete_ml_learning_verification.py

# System integration verification
python utils/System_Integration_Test.py

# Specialized model tests
python tests/test_bias_detection.py
python tests/test_pattern_recognition.py
python tests/test_sr_detection.py

# Run with pytest framework
pytest tests/ -v
```

### Analysis and Debugging
```bash
# Analyze backtest data
python src/analyze_file.py test_analyzer_data/backtest_USTEC_20250516_20250715.jsonl

# Debug utilities
python utils/debug_file_access.py
python utils/library_check.py
python utils/diagnose_lstm.py

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

# Real-time MT5 bridge
python src/MT5BridgeReader.py
```

---

## 🔍 Key Integration Patterns

### Data Flow Architecture
```
Market Data → MT5 Bridge → Advanced Analyzer → AssetAnalyzer → ML Models → Algorithm Competition → Trading Signals
     ↓              ↓              ↓                    ↓              ↓                    ↓
Event Collection → Rate Limiting → Batch Processing → Champion Selection → Performance Monitoring → Storage
```

### Signal Processing Pipeline
```
Raw Ticks → Technical Analysis → Feature Engineering → ML Prediction → Champion Competition → Signal Generation
     ↓              ↓                    ↓                  ↓                    ↓                  ↓
Data Validation → Pattern Detection → Model Training → Performance Validation → Error Handling → Recovery System
```

### Event Processing Flow
```
Component Events → Event Buffers → Rate Limiting → Batch Processing → ML Training Logger → Storage → Analysis
```

---

## 📈 Performance Metrics & Monitoring

### Backtesting Performance Targets:
- **Processing Speed**: 100,000+ ticks/second
- **Memory Efficiency**: < 80% utilization under load
- **Latency**: Microsecond-level processing for real-time scenarios
- **Throughput**: Configurable based on system resources

### Real-Time Trading Metrics:
- **Tick Processing**: Sub-millisecond latency
- **Event Logging**: Intelligent rate limiting prevents bottlenecks
- **Memory Management**: Automatic cleanup and optimization
- **System Monitoring**: Real-time performance tracking

### Health Score Calculation:
```python
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

### Trading Safety Features:
- **Position Limits**: Configurable maximum positions per asset
- **Risk Percentage**: Configurable risk per trade
- **Emergency Stops**: Automatic algorithm disabling on poor performance
- **Data Validation**: Comprehensive input validation and sanitization
- **State Encryption**: Secure model and state persistence

### Configuration Security:
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

### Learning Phase Success:
- **Health Score**: > 70% system health
- **Prediction Confidence**: > 70% model confidence
- **Champion Status**: Active champion for every ModelType
- **Model Persistence**: Successful saving/loading of ML models
- **System Stability**: No emergency stops during learning

### Production Readiness:
- **Performance**: Consistent 100,000+ ticks/second processing
- **Memory Efficiency**: < 80% memory utilization under load
- **Error Rate**: < 1% processing errors
- **Recovery Success**: > 95% successful recovery from errors
- **Prediction Accuracy**: > 60% accuracy across all model types

---

## 📚 Testing Philosophy and Best Practices

### Comprehensive Test Coverage:
- **Unit Tests**: Individual component verification
- **Integration Tests**: Cross-component interaction validation
- **System Tests**: End-to-end pipeline verification
- **Performance Tests**: Speed and memory usage validation
- **Error Recovery Tests**: Edge case and failure scenario handling

### Real Data Integration:
- **MT5 Data Integration**: Real market data for realistic testing
- **Synthetic Data Generation**: Controlled test scenarios with known outcomes
- **Memory-Safe Processing**: Large dataset handling without memory issues
- **Progressive Loading**: Chunk-based processing for unlimited data sizes

### Enterprise-Grade Quality:
- **Detailed Reporting**: Comprehensive test result documentation
- **Health Scoring**: Multi-factor system health assessment
- **Performance Benchmarking**: Consistent performance measurement
- **Error Classification**: Systematic error categorization and tracking

---

## 🔮 System Capabilities Summary

### Core Features:
1. **Advanced ML Analysis**: 6 model types with champion competition
2. **High-Performance Processing**: 100,000+ ticks/second capability
3. **Real-Time Monitoring**: Comprehensive event collection and visualization
4. **Enterprise Testing**: 13-phase validation with real market data
5. **Production Ready**: Fault tolerance, error handling, and recovery systems

### Key Benefits:
- **Self-Improving**: Champion competition ensures continuous optimization
- **Scalable**: Memory-aware processing handles unlimited data volumes
- **Reliable**: Comprehensive error handling with graceful degradation
- **Maintainable**: Modular architecture with clean separation of concerns
- **Observable**: Rich logging and monitoring for production environments

### Use Cases:
- **Algorithmic Trading**: High-frequency scalping operations
- **Market Research**: Pattern recognition and trend analysis
- **Risk Management**: Volatility prediction and bias detection
- **Strategy Development**: Backtesting and performance validation
- **Production Trading**: Real-time signal generation and execution

This comprehensive documentation provides Claude Code with complete understanding of this sophisticated algorithmic trading system, enabling effective assistance with development, testing, configuration, and optimization tasks.

---

## 🔧 Key Development Guidelines

### Code Quality Standards:
- **Performance First**: Optimizations at every level from memory to algorithms
- **Event-Driven Design**: Async processing with configurable event handling
- **Modular Architecture**: Clean separation of concerns with well-defined interfaces
- **Comprehensive Testing**: Enterprise-grade validation framework
- **Production Ready**: Fault tolerance and graceful degradation

### Architecture Patterns:
1. **Champion Competition**: Self-improving algorithms with performance-based selection
2. **Rate Limiting Intelligence**: Intelligent throttling to prevent system overload
3. **Memory-Aware Processing**: Bounded buffers and automatic cleanup
4. **Batch Event Processing**: 50x performance improvement with event aggregation
5. **Multi-Level Caching**: LRU-based eviction with intelligent cache management

### Integration Best Practices:
- **Non-Invasive Logging**: Hook-based event capture without performance impact
- **Progressive Data Processing**: Memory-safe handling of large datasets
- **Cross-Platform Compatibility**: Universal encoding and emoji support
- **Configurable Verbosity**: Multiple logging levels for different environments
- **Comprehensive Error Handling**: Detailed error tracking with recovery mechanisms

This represents a production-ready, institutional-grade algorithmic trading system suitable for mission-critical financial applications where reliability, performance, and accuracy are paramount.