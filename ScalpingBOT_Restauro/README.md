# 🏗️ ScalpingBOT Restauro - Refactoring Project

## 📁 Project Structure

This is the refactored version of ScalpingBOT with a clean, modular architecture based on 7 macro groups.

### 🎯 7 MACRO GROUPS

#### 1. 📊 **DATA** - Data Management and Preprocessing
- **collectors/** - Data collection from various sources
- **processors/** - Data processing and transformation
- **storage/** - Data persistence and caching

#### 2. 🧠 **ML** - Machine Learning and Models
- **models/** - All ML model implementations
- **trainers/** - Training logic and algorithms
- **features/** - Feature engineering and datasets
- **evaluation/** - Model evaluation and competition

#### 3. 🎯 **PREDICTION** - Prediction Systems
- **predictors/** - 6 prediction types implementation
- **aggregators/** - Multi-model aggregation
- **validators/** - Prediction validation

#### 4. 🏛️ **CORE** - Central Business Logic
- **analyzers/** - Core analyzer classes
- **competition/** - Algorithm competition system
- **state/** - State and session management

#### 5. 📝 **MONITORING** - Logging and Monitoring
- **logging/** - Unified logging system
- **events/** - Event collection and storage
- **display/** - Display and visualization
- **health/** - Health monitoring and diagnostics

#### 6. 🔧 **CONFIG** - Unified Configuration
- **base/** - Base configuration classes
- **domain/** - Domain-specific configs
- **environments/** - Environment configurations

#### 7. 🔌 **INTERFACES** - External Interfaces
- **mt5/** - MetaTrader 5 interfaces
- **apis/** - REST and WebSocket APIs
- **external/** - External service integrations

## 🚀 Migration Status

### Phase 1: Foundation (Week 1-2)
- [ ] CONFIG - Unify 13 configuration classes
- [ ] MONITORING - Consolidate 6 logging systems
- [ ] INTERFACES - Extract 3 MT5 implementations

### Phase 2: Core Extraction (Week 3-6)
- [ ] DATA - Extract data processing logic
- [ ] ML - Separate models and training logic
- [ ] PREDICTION - Isolate prediction systems

### Phase 3: Business Logic (Week 7-10)
- [ ] CORE - Refactor AssetAnalyzer and AdvancedMarketAnalyzer
- [ ] Integration - Connect all modules
- [ ] Testing - Comprehensive testing for each module

## 📋 Key Improvements

1. **No More Monolith** - 20,594 line file split into logical modules
2. **Zero Duplication** - All duplicate functions unified
3. **Clear Architecture** - 7 well-defined macro groups
4. **Maintainable** - Each module has single responsibility
5. **Scalable** - Easy to extend and modify

## 🔄 Duplicate Elimination Plan

### Functions to Unify:
- `process_tick()` → `src/data/processors/tick_processor.py`
- `_prepare_*_dataset()` → `src/ml/features/dataset_builder.py`
- `_prepare_*_features()` → `src/ml/features/feature_engineer.py`
- `train_*_model()` → `src/ml/trainers/` (strategy pattern)
- Logging systems → `src/monitoring/logging/unified_logger.py`

### Configurations to Consolidate:
- 13 config classes → 4 domain configs + base config
- Pattern: Inheritance hierarchy instead of duplication

---

*Refactoring initiated: 2025-01-28*
*Based on complete system analysis of 25,000+ lines of code*