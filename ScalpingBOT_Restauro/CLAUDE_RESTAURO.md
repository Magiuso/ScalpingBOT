# 🏗️ CLAUDE_RESTAURO.md - Regole Ferree del Refactoring

**Data**: 2025-08-02 (AGGIORNATO CON ARCHITETTURA REALE)  
**Progetto**: ScalpingBOT Restauro  
**Scope**: Regole inviolabili per il refactoring del sistema + Documentazione architettura completa

---

## 🎯 ARCHITETTURA **TRULY MULTIASSET** - DYNAMIC SYSTEM

### **🚀 REVOLUTIONARY MULTIASSET SYSTEM**
- **Fully Dynamic**: NO HARDCODED ASSET SYMBOLS - Sistema accetta qualsiasi asset
- **Category-Based Intelligence**: Auto-detection di FOREX, INDICES, COMMODITIES, CRYPTO
- **Pattern Recognition**: Asset classification basata su pattern avanzati  
- **Single Codebase**: Un sistema gestisce infiniti asset senza modifiche al codice

### **🔍 INTELLIGENT ASSET DETECTION**
```python
# Il sistema riconosce automaticamente:
"EURUSD" → FOREX (currency pair detection)
"USTEC"  → INDICES (index pattern detection)  
"XAUUSD" → COMMODITIES (precious metals detection)
"BTCUSD" → CRYPTO (cryptocurrency detection)
"RANDOM" → FOREX (default fallback category)
```

### **📁 DYNAMIC DIRECTORY STRUCTURE**
```
ScalpingBOT_Data/
├── {ANY_ASSET_NAME}/     # ✅ COMPLETELY DYNAMIC - ACCEPTS ANY SYMBOL
│   ├── models/           # Auto-created for any asset
│   ├── logs/             # Isolated logs per asset
│   ├── events/           # ML events per asset
│   ├── config/           # Category-optimized config
│   └── data/             # Historical data per asset
├── EURUSD/              # Example: FOREX category config applied
├── USTEC/               # Example: INDICES category config applied  
├── XAUUSD/              # Example: COMMODITIES category config applied
├── BTCUSD/              # Example: CRYPTO category config applied
└── ANYTHING/            # ✅ ANY SYMBOL WORKS - TRULY MULTIASSET
```

### **⚙️ SMART CONFIGURATION STRATEGY**
- **🌐 GLOBAL CONFIG**: SystemMode, PerformanceProfile, rate limits universali
- **🎯 CATEGORY-SPECIFIC**: Parametri ottimizzati per FOREX/INDICES/COMMODITIES/CRYPTO
- **🔄 RUNTIME DETECTION**: Auto-classification al momento dell'uso
- **📊 OPTIMIZED THRESHOLDS**: Volatility e trading parameters per categoria

### **🏭 PRODUCTION-READY MULTIASSET**
- **♾️ Infinite Scalability**: Aggiungi asset senza toccare codice
- **🚀 Zero Deployment**: Nuovo asset = zero configurazione manuale
- **🧠 Intelligent Adaptation**: Sistema si adatta automaticamente
- **🔒 Isolation Guarantee**: Ogni asset completamente isolato

---

## 🏗️ ARCHITETTURA SISTEMA REALE - AGGIORNATO 2025-08-02

### **📊 ARCHITETTURA 6-FASI IMPLEMENTATA**

Il sistema è stato completamente migrato con questa struttura modulare:

```
ScalpingBOT_Restauro/src/
├── 📁 FASE 1 - CONFIG/           # Sistema di configurazione centralizzato
│   ├── base/                    # Configurazioni core (base_config.py, config_loader.py)
│   ├── domain/                  # Config specifiche (asset_config.py, system_config.py, monitoring_config.py)
│   └── shared/                  # Utilities condivise
├── 📁 FASE 2 - MONITORING/       # Sistema di monitoraggio completo
│   ├── events/                  # Collezione eventi (event_collector.py)
│   ├── display/                 # Gestione display (display_manager.py)
│   ├── storage/                 # Storage eventi (storage_manager.py)
│   └── utils/                   # Utilities monitoraggio
├── 📁 FASE 3 - INTERFACES/       # Interfacce esterne
│   └── mt5/                     # Integrazione MT5 (mt5_adapter.py, mt5_bridge_reader.py, mt5_backtest_runner.py)
├── 📁 FASE 4 - DATA/             # Sistema dati
│   ├── collectors/              # Collezione dati (tick_collector.py)
│   └── processors/              # Elaborazione dati (market_data_processor.py)
├── 📁 FASE 5 - ML/               # Machine Learning completo
│   ├── algorithms/              # 20+ algoritmi ML
│   ├── integration/             # Integrazione ML (algorithm_bridge.py, analyzer_ml_integration.py)
│   ├── models/                  # Modelli neurali (advanced_lstm.py, cnn_models.py, transformer_models.py, competition.py)
│   ├── training/                # Sistema training (adaptive_trainer.py)
│   ├── monitoring/              # Monitoraggio ML (training_monitor.py)
│   └── preprocessing/           # Preprocessing dati (data_preprocessing.py)
├── 📁 FASE 6 - PREDICTION/       # Sistema predizioni
│   ├── core/                    # Core predizioni (advanced_market_analyzer.py, asset_analyzer.py)
│   └── unified_system.py        # Orchestratore principale
└── 📁 SHARED/                    # Componenti condivisi
    ├── enums.py                 # Enumerazioni centralizzate
    └── exceptions.py            # Eccezioni centralizzate
```

### **🎯 SISTEMA ML CON 20+ ALGORITMI IMPLEMENTATI**

#### **Algoritmi per Categoria:**
```python
# Support/Resistance (5 algoritmi)
- PivotPoints_Classic
- VolumeProfile_Advanced  
- LSTM_SupportResistance
- StatisticalLevels_ML
- Transformer_Levels

# Pattern Recognition (5 algoritmi)
- CNN_PatternRecognizer
- Classical_Patterns
- LSTM_Sequences
- Transformer_Patterns
- Ensemble_Patterns

# Bias Detection (5 algoritmi)
- Sentiment_LSTM
- VolumePrice_Analysis
- Momentum_ML
- Transformer_Bias
- MultiModal_Bias

# Trend Analysis (5 algoritmi)
- RandomForest_Trend
- LSTM_TrendPrediction
- GradientBoosting_Trend
- Transformer_Trend
- Ensemble_Trend

# Volatility Prediction (3 algoritmi)
- GARCH_Volatility
- LSTM_Volatility
- Realized_Volatility
```

### **⚙️ SISTEMA DI COMPETIZIONE ML AVANZATO**

#### **Componenti del Sistema:**
- **`AlgorithmCompetition`**: Gestione competizione tra algoritmi
- **`ChampionPreserver`**: Persistenza e recovery dei champion
- **`RealityChecker`**: Validazione performance vs mercato reale
- **`EmergencyStopSystem`**: Stop automatici per algoritmi fallimentari
- **`AlgorithmBridge`**: Integrazione algoritmi nel sistema di competizione

### **🔄 FLUSSO DATI SISTEMA REALE**

```
MT5 Data → TickCollector → MarketDataProcessor → AlgorithmBridge → Competition → Predictions
     ↓           ↓               ↓                    ↓              ↓           ↓
MT5Adapter → collect_tick() → prepare_market_data() → execute_algorithm() → champion_selection → convert_to_prediction()
```

### **📋 FACTORY FUNCTIONS CORRETTE - GUIDA DENOMINAZIONE**

#### **Sistema-Level (Entry Points Principali):**
```python
# SISTEMA UNIFICATO
create_unified_system(data_path, mode=SystemMode.PRODUCTION)
create_production_system(data_path)  
create_testing_system(data_path)

# ANALISI MULTI-ASSET
create_advanced_market_analyzer(data_path, config_manager=None)
create_asset_analyzer(asset, data_path, config_manager=None)
```

#### **ML & Algoritmi:**
```python
# ALGORITHM BRIDGE
create_algorithm_bridge(ml_models=None, logger=None)

# ML TRAINERS
create_enhanced_sr_trainer(input_size, **kwargs)
create_enhanced_pattern_trainer(input_size, **kwargs)  
create_enhanced_bias_trainer(input_size, **kwargs)

# ALGORITMI SPECIFICI
create_support_resistance_algorithms(ml_models=None)
create_pattern_recognition_algorithms(ml_models=None)
create_bias_detection_algorithms(ml_models=None)
create_trend_analysis_algorithms(ml_models=None)
create_volatility_prediction_algorithms(ml_models=None)
```

#### **Configurazione & Monitoring:**
```python
# CONFIGURAZIONE
get_configuration_manager()
load_configuration_for_mode(mode, asset_symbol)
AssetSpecificConfig.for_asset(asset_symbol)

# MONITORING
create_event_collector(config)
create_simple_display(config) 
create_storage_manager(config)
```

#### **Data Processing:**
```python
# DATA PROCESSING
create_tick_collector(max_buffer_size=10000)
create_market_data_processor(config=None)

# MT5 INTEGRATION
create_backtest_config(symbol, start_date, end_date)
create_backtest_runner(config, event_collector=None)
```

### **🚀 PUNTI DI INGRESSO SISTEMA**

#### **Entry Point Principale:**
```python
# UTILIZZO SISTEMA COMPLETO
from src.prediction.unified_system import create_unified_system, SystemMode

system = create_unified_system(data_path="./data", mode=SystemMode.PRODUCTION)
system.add_asset("EURUSD")  # Qualsiasi asset - TRULY MULTIASSET
system.start()

# Processing real-time
result = system.process_tick("EURUSD", datetime.now(), 1.1234, 1000.0)

# Training ML models
training_result = system.train_on_batch(batch_data)

# Predictions con algoritmi trained
predictions = system.validate_on_batch(batch_data)
```

#### **Entry Point Singolo Asset:**
```python
# UTILIZZO SINGOLO ASSET
from src.prediction.core.advanced_market_analyzer import create_advanced_market_analyzer

analyzer = create_advanced_market_analyzer(data_path="./data")
analyzer.add_asset("USTEC")
analyzer.start()

# Training diretto
result = analyzer.train_models_on_batch(batch_data)
```

#### **Entry Point ML Diretto:**
```python
# UTILIZZO ML BRIDGE DIRETTO
from src.ml.integration.algorithm_bridge import create_algorithm_bridge

bridge = create_algorithm_bridge()
result = bridge.execute_algorithm(ModelType.SUPPORT_RESISTANCE, "LSTM_SupportResistance", market_data)
```

### **🔧 METODI CHIAVE E FIRME CORRETTE**

#### **UnifiedAnalyzerSystem (Entry Point Principale):**
```python
# SYSTEM LIFECYCLE
def add_asset(self, asset: str) -> None
def remove_asset(self, asset: str) -> None  
def start(self) -> None
def stop(self) -> None

# DATA PROCESSING
def process_tick(self, asset: str, timestamp: datetime, price: float, volume: float, 
                bid: Optional[float] = None, ask: Optional[float] = None) -> Dict[str, Any]

# ML OPERATIONS
def train_on_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]
def validate_on_batch(self, batch_data: Dict[str, Any]) -> List[Dict[str, Any]]

# MONITORING
def get_system_stats(self) -> Dict[str, Any]
def get_system_health(self) -> Dict[str, Any]
```

#### **AdvancedMarketAnalyzer (Multi-Asset Core):**
```python
# ASSET MANAGEMENT
def add_asset(self, asset: str) -> AssetAnalyzer
def remove_asset(self, asset: str) -> None

# ML TRAINING & VALIDATION
def train_models_on_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]
def validate_models_on_batch(self, batch_data: Dict[str, Any]) -> List[Dict[str, Any]]

# INTERNAL DATA PREPARATION
def _convert_ticks_to_training_data(self, asset_ticks: List[Dict[str, Any]]) -> Dict[str, Any]
def _prepare_prediction_data(self, asset_ticks: List[Dict[str, Any]], champion_algorithm: str, model_type: ModelType) -> Dict[str, Any]
```

#### **AlgorithmBridge (ML Integration Core):**
```python
# ALGORITHM EXECUTION
def execute_algorithm(self, model_type: ModelType, algorithm_name: str, market_data: Dict[str, Any]) -> Dict[str, Any]
def convert_to_prediction(self, algorithm_result: Dict[str, Any], asset: str, model_type: ModelType) -> Prediction

# COMPETITION INTEGRATION
def register_algorithms_in_competition(self, competition: AlgorithmCompetition) -> None
def create_algorithm_execution_callback(self, model_type: ModelType) -> Callable
```

#### **Algorithm Classes (Pattern Standardizzato):**
```python
# TUTTI GLI ALGORITMI SEGUONO QUESTO PATTERN
class [Category]Algorithms:
    def __init__(self, ml_models: Optional[Dict[str, Any]] = None)
    def run_algorithm(self, algorithm_name: str, market_data: Dict[str, Any]) -> Dict[str, Any]
    def get_algorithm_stats(self) -> Dict[str, Any]

# ESEMPI CONCRETI:
- SupportResistanceAlgorithms
- PatternRecognitionAlgorithms  
- BiasDetectionAlgorithms
- TrendAnalysisAlgorithms
- VolatilityPredictionAlgorithms
```

#### **Configuration System:**
```python
# CONFIGURATION MANAGER
def get_configuration_manager() -> ConfigurationManager
def load_configuration_for_mode(mode: SystemMode, asset_symbol: str) -> UnifiedConfig

# ASSET-SPECIFIC CONFIG
AssetSpecificConfig.for_asset(asset_symbol: str) -> AssetSpecificConfig
```

#### **MT5 Integration:**
```python
# BACKTEST RUNNER
def create_backtest_config(symbol: str, start_date: datetime, end_date: datetime) -> BacktestConfig
def run_backtest(self, analyzer_system) -> None

# BRIDGE READER  
def start_monitoring(self, symbols: List[str]) -> None
def stop_monitoring(self) -> None
def set_analyzer_callback(self, callback: Callable) -> None
```

#### **MarketDataProcessor:**
```python
# DATA PREPARATION
def prepare_market_data(self, tick_data: deque, min_ticks: int = 20, window_size: int = 5000) -> Dict[str, Any]
def prepare_lstm_features(self, prices: np.ndarray, volumes: np.ndarray, market_data: Optional[Dict[str, Any]] = None) -> np.ndarray
```

### **⚠️ NAMING CONVENTIONS - REGOLE FERREE**

#### **✅ SEMPRE USARE QUESTI PATTERN:**
```python
# Factory Functions - SEMPRE prefisso create_*
create_unified_system()
create_advanced_market_analyzer()  
create_algorithm_bridge()
create_enhanced_sr_trainer()

# Manager Classes - SEMPRE suffisso *Manager
ConfigurationManager
StorageManager  
DisplayManager

# Config Classes - SEMPRE suffisso *Config
AnalyzerConfig
UnifiedConfig
BacktestConfig

# Algorithm Classes - SEMPRE suffisso *Algorithms
SupportResistanceAlgorithms
PatternRecognitionAlgorithms

# Entry Point Methods - SEMPRE verbi chiari
add_asset(), remove_asset()
start(), stop()
train_on_batch(), validate_on_batch()
```

#### **❌ NON USARE MAI:**
```python
# ❌ EVITARE:
create_system()           # Troppo generico
get_analyzer()           # Ambiguo  
process_data()           # Non specifico
run_model()              # Non chiaro

# ✅ USARE INVECE:
create_unified_system()
create_advanced_market_analyzer()
train_models_on_batch()
execute_algorithm()
```

---

## 💰 REGOLA SUPREMA - TRADING CON DENARO REALE

### **🚨 IL SISTEMA USERÀ SOLDI VERI - ZERO ERRORI TOLLERATI**

Questo sistema effettuerà trading automatico con **DENARO REALE** sui mercati finanziari. Le predizioni del sistema verranno utilizzate per:
- Aprire e chiudere posizioni reali
- Gestire stop loss e take profit  
- Allocare capitale significativo
- Operare 24/5 sui mercati globali

#### **CONSEGUENZE DI ERRORI:**
- **Bug nel codice** = **PERDITE FINANZIARIE REALI**
- **Performance scadenti** = **CAPITALE BRUCIATO**
- **Data leakage** = **FALSI PROFITTI → PERDITE CATASTROFICHE**
- **Crash del sistema** = **POSIZIONI APERTE NON GESTITE**

#### **STANDARD OBBLIGATORI:**
1. **ZERO DATA LEAKAGE**: Nessuna informazione futura nel training
2. **PERFORMANCE O(n)**: Latenza millisecond per tick  
3. **MEMORY BOUNDED**: Nessun memory leak consentito
4. **THREAD SAFE**: Concorrenza gestita correttamente
5. **FAIL SAFE**: Chiusura posizioni in caso di errore
6. **AUDIT TRAIL**: Ogni decisione tracciabile

#### **TESTING RICHIESTO:**
```python
# OGNI funzione critica DEVE avere:
- Unit tests con coverage 100%
- Integration tests con dati reali
- Stress tests con milioni di tick
- Failure scenario tests
- Performance benchmarks < 1ms
```

#### **CODE REVIEW MANDATORY:**
Ogni modifica che tocca:
- Calcolo predizioni
- Gestione ordini
- Risk management  
- Data processing

**DEVE** essere reviewed da almeno 2 sviluppatori senior prima del deploy.

---

## 🧹 REGOLA CLEAN CODE - ZERO CODICE INUTILIZZATO

### **🗑️ PULIZIA TOTALE DA VECCHIE VERSIONI**

Il sistema deve essere **COMPLETAMENTE PULITO** da:
- Codice commentato di vecchie versioni
- Funzioni obsolete non più chiamate
- Import non utilizzati
- Variabili morte
- Logiche alternative abbandonate
- File di backup nel repository

#### **STANDARD DI PULIZIA:**
```python
# ❌ VIETATO:
def calculate_rsi_old(prices):  # Vecchia versione
    pass

# def calculate_rsi_v2(prices):  # Codice commentato
#     return old_logic

# ✅ SOLO CODICE ATTIVO:
def calculate_rsi(prices):
    """Calcola RSI - unica implementazione"""
    return current_implementation
```

#### **VERIFICA PERIODICA:**
```bash
# Tools obbligatori da eseguire:
- pylint --disable=all --enable=unused-import
- vulture src/ --min-confidence 100  
- flake8 --select=F401,F841
- mypy --strict
```

#### **REGOLA DEL SINGOLO PERCORSO:**
- **UNA** implementazione per funzione
- **UNA** versione per algoritmo
- **UNA** strategia per componente
- **ZERO** alternative commentate

---

## 🚨 REGOLE FONDAMENTALI - DA RISPETTARE SEMPRE

### 1. 🔧 **MIGLIORA IL SISTEMA QUANDO NECESSARIO**
- **CONSENTITO** correggere bug e problemi necessari
- **OBBLIGATORIO** rispettare SEMPRE le altre regole (no fallback, fail fast, no dati sintetici)
- **VIETATO** aggiungere funzionalità non richieste

#### Esempio:
```python
# VECCHIO (Analyzer.py - riga 12345)
def calculate_rsi(prices, period=14):
    # Logica complessa di 50 righe
    return rsi_values

# NUOVO (src/ml/features/indicators.py)
def calculate_rsi(prices, period=14):
    # IDENTICHE 50 righe di logica - COPIA ESATTA
    return rsi_values
```

### 2. 🔗 **SOLO AGGIUSTARE CONNESSIONI**
- Modificare **SOLO** gli `import` statements
- Aggiustare **SOLO** i riferimenti tra classi
- Mantenere **IDENTICHE** le firme dei metodi
- Mantenere **IDENTICI** i nomi delle variabili

#### Esempio:
```python
# VECCHIO
from ML_Training_Logger.Event_Collector import EventCollector

# NUOVO  
from src.monitoring.events.event_collector import EventCollector
# Ma la classe EventCollector resta IDENTICA
```

### 3. 🔗 **MODULI INTEGRATI**
Ogni modulo è **PARTE DEL SISTEMA** e non deve avere test embedded:

```python
# I moduli sono pensati per integrazione, NON per esecuzione standalone
# Factory functions per compatibilità e utilizzo da altri moduli
def create_tick_processor(config=None):
    return TickProcessor(config)
```

### 4. ❌ **ZERO FALLBACK - ZERO DATI TEST**

#### **VIETATO ASSOLUTAMENTE**:
```python
# ❌ MAI FARE QUESTO:
if not data:
    data = generate_synthetic_data()  # VIETATO!
    
try:
    connect_to_mt5()
except:
    use_demo_connection()  # VIETATO!

price = data.get('price', 1.0)  # VIETATO IL DEFAULT!
```

#### **SEMPRE E SOLO COSÌ**:
```python
# ✅ FAIL FAST CON ERRORI CHIARI:
if not data:
    raise ValueError("No real data available - cannot proceed")
    
try:
    connect_to_mt5()
except Exception as e:
    raise ConnectionError(f"MT5 connection failed: {e}")

price = data.get('price')
if price is None:
    raise KeyError("Critical field 'price' is missing from tick data")
```

### 5. 💥 **FAIL FAST - ERRORI CHIARI**
- **NIENTE** silent failures
- **NIENTE** log e continua
- **SEMPRE** errori espliciti e descrittivi

```python
# Ogni errore DEVE dire ESATTAMENTE cosa è andato storto:
raise FileNotFoundError(f"Cannot find MT5 data file: {filepath} - Check MT5 is running")
raise ValueError(f"Invalid tick data: missing required fields {missing_fields}")
raise RuntimeError(f"Model not trained for {asset} - Run training first")
```

### 6. 🎯 **UNA SOLA STRADA**
- **Una sola** fonte dati (reale)
- **Una sola** modalità di connessione (reale)
- **Una sola** strategia di processing (niente alternative)
- **Una sola** implementazione per funzione

---

## 📋 CHECKLIST PER OGNI FILE MIGRATO

Prima di considerare completo un file migrato, verificare:

- [ ] **Logica corretta** - Bug fixati quando necessario
- [ ] **Import corretti** - Solo path aggiornati  
- [ ] **Nomi preservati** - Stessi nomi di classi/funzioni/variabili
- [ ] **Modulo integrato** - Nessun test embedded
- [ ] **Zero fallback** - Niente dati test o alternative
- [ ] **Fail fast** - Errori chiari senza recovery
- [ ] **Documentazione** - Commenti originali preservati

---

## 🔄 PROCESSO DI MIGRAZIONE

### Per ogni componente da migrare:

1. **IDENTIFICARE** il codice nel file originale (numero righe)
2. **COPIARE** esattamente il codice (nessuna modifica)
3. **INCOLLARE** nel nuovo file di destinazione
4. **AGGIUSTARE** solo gli import statements
5. **VERIFICARE** che si integri correttamente
6. **DOCUMENTARE** la migrazione nel log

### Esempio log migrazione:
```
[2025-01-28 10:30] MIGRATED: AssetAnalyzer.process_tick()
FROM: src/Analyzer.py (lines 11089-11234)
TO: src/data/processors/tick_processor.py
CHANGES: Only import statements updated
STATUS: Integrated successfully - working with real data
```

---

## ⚠️ VIOLAZIONI COMUNI DA EVITARE

### ❌ **NON FARE**:
1. "Miglioriamo questa funzione mentre la spostiamo"
2. "Aggiungiamo un try-except per sicurezza"
3. "Mettiamo un valore di default per evitare crash"
4. "Creiamo un'interfaccia più pulita"
5. "Ottimizziamo questo loop mentre ci siamo"

### ✅ **FARE SEMPRE**:
1. Copia-incolla esatto del codice
2. Modificare SOLO import e riferimenti
3. Lasciare errori e crash come sono
4. Preservare interfacce esistenti
5. Mantenere anche codice "brutto" se funziona

---

*Questo documento contiene le regole inviolabili per lo sviluppo e manutenzione del sistema ScalpingBOT_Restauro, un sistema di trading algoritmico production-ready che opera con denaro reale sui mercati finanziari.*