//+------------------------------------------------------------------+
//| RSIMomentum.mqh - VERSIONE ENTERPRISE CON AUTO-DETECTION        |
//| RSI + ADX con HashMap O(1), algoritmi avanzati, MT5 nativo      |
//| 🤖 AUTO-DETECTION COMPLETA - Determina automaticamente BUY/SELL |
//| Supporta 100+ assets, thread-safe, auto-calibrazione dinamica  |
//+------------------------------------------------------------------+
#ifndef __RSI_MOMENTUM_OPTIMIZED_MQH__
#define __RSI_MOMENTUM_OPTIMIZED_MQH__

#include <ScalpingBot\Utility.mqh>       // Contiene CRSICacheOptimized + CalculateADX + CalculateMultiPeriodATR
#include <ScalpingBot\SafeBuffer.mqh>    // Per operazioni buffer sicure

//+------------------------------------------------------------------+
//| 📊 CONFIGURAZIONE AVANZATA                                      |
//+------------------------------------------------------------------+
#define RSI_HASH_BUCKETS        64     // Hash table buckets per stati RSI
#define RSI_MAX_ENTRIES         120    // Max entries (100 assets + buffer)
#define RSI_CLEANUP_INTERVAL    600    // Cleanup ogni 10 minuti
#define RSI_STATE_TTL           1800   // TTL stato RSI (30 minuti)
#define RSI_VALIDATION_CACHE    60     // Cache validazione (1 minuto)
#define RSI_MIN_SAMPLES         10     // Minimo campioni per statistiche
#define RSI_CALIBRATION_PERIOD  100   // Periodo auto-calibrazione

//+------------------------------------------------------------------+
//| 📋 ENUMERAZIONI AVANZATE                                        |
//+------------------------------------------------------------------+
enum RSI_SIGNAL_QUALITY {
    RSI_QUALITY_POOR = 0,      // Segnale debole
    RSI_QUALITY_FAIR = 1,      // Segnale discreto  
    RSI_QUALITY_GOOD = 2,      // Segnale buono
    RSI_QUALITY_EXCELLENT = 3  // Segnale eccellente
};

enum RSI_MARKET_REGIME {
    RSI_REGIME_UNKNOWN = 0,    // Regime non determinato
    RSI_REGIME_TRENDING = 1,   // Mercato in trend
    RSI_REGIME_RANGING = 2,    // Mercato laterale
    RSI_REGIME_VOLATILE = 3    // Mercato volatile
};

enum RSI_TIMEFRAME_CONSENSUS {
    RSI_CONSENSUS_NONE = 0,    // Nessun consenso
    RSI_CONSENSUS_WEAK = 1,    // Consenso debole
    RSI_CONSENSUS_MODERATE = 2, // Consenso moderato
    RSI_CONSENSUS_STRONG = 3   // Consenso forte
};

enum RSI_DIRECTION_SIGNAL {
    RSI_SIGNAL_NONE = 0,       // Nessun segnale
    RSI_SIGNAL_WEAK_BUY = 1,   // Segnale BUY debole
    RSI_SIGNAL_STRONG_BUY = 2, // Segnale BUY forte
    RSI_SIGNAL_WEAK_SELL = 3,  // Segnale SELL debole
    RSI_SIGNAL_STRONG_SELL = 4 // Segnale SELL forte
};

//+------------------------------------------------------------------+
//| 📦 STRUTTURA STATO RSI LEGACY (per compatibilità)              |
//+------------------------------------------------------------------+
struct RSIMomentumState {
    double rsiLast;
    double rsiPrev;
    double derivata;
    double adx;
    int rsiScore;
    bool adxValid;
    bool isValid;
    datetime lastUpdateTime;
    long key;
};

//+------------------------------------------------------------------+
//| 📦 STRUTTURA STATO RSI AVANZATA CON AUTO-DETECTION              |
//+------------------------------------------------------------------+
struct RSIMomentumStateAdvanced {
    // Dati base RSI
    double rsiLast;                    // Ultimo valore RSI
    double rsiPrev;                    // RSI precedente
    double rsiAvg;                     // Media RSI periodo
    double derivata;                   // Derivata RSI
    double derivataWeighted;           // Derivata pesata
    
    // Indicatori di supporto
    double adx;                        // ADX valore
    double adxTrend;                   // Trend ADX
    bool adxValid;                     // ADX sopra soglia
    
    // 🤖 AUTO-DETECTION FIELDS
    bool autoDirection;                // true=BUY, false=SELL (AUTO-DETECTED)
    double autoConfidence;             // Confidenza auto-detection (0-1)
    string autoReason;                 // Motivo della direzione scelta
    RSI_DIRECTION_SIGNAL autoSignal;   // Tipo di segnale auto-rilevato
    bool autoDetectionValid;           // Auto-detection riuscita
    
    // Scoring e qualità
    int rsiScore;                      // Score base (0-3)
    int advancedScore;                 // Score avanzato (0-10)
    RSI_SIGNAL_QUALITY quality;       // Qualità segnale
    RSI_MARKET_REGIME regime;          // Regime di mercato
    RSI_TIMEFRAME_CONSENSUS consensus; // Consenso multi-TF
    
    // Statistiche dinamiche
    double volatilityFactor;           // Fattore volatilità
    double trendStrength;              // Forza trend
    double noiseLevel;                 // Livello rumore
    double confidence;                 // Confidenza (0-100%)
    
    // Parametri adattivi
    double dynamicThreshold;           // Soglia derivata dinamica
    double adaptiveADXThreshold;       // Soglia ADX adattiva
    int optimalPeriod;                 // Periodo RSI ottimale
    
    // Metadati
    datetime lastUpdateTime;           // Ultimo aggiornamento
    datetime lastCalibrationTime;      // Ultima calibrazione
    long key;                         // Chiave hash
    int updateCount;                  // Contatore aggiornamenti
    bool isValid;                     // Stato valido
    bool needsCalibration;            // Richiede calibrazione
    
    // Timing e persistenza
    int signalStrengthBars;          // Barre con segnale forte
    double signalPersistence;        // Persistenza segnale
    
    // Costruttore
    RSIMomentumStateAdvanced() {
        rsiLast = 50.0;
        rsiPrev = 50.0;
        rsiAvg = 50.0;
        derivata = 0.0;
        derivataWeighted = 0.0;
        adx = 0.0;
        adxTrend = 0.0;
        adxValid = false;
        
        // 🤖 AUTO-DETECTION INIT
        autoDirection = true;
        autoConfidence = 0.0;
        autoReason = "";
        autoSignal = RSI_SIGNAL_NONE;
        autoDetectionValid = false;
        
        rsiScore = 0;
        advancedScore = 0;
        quality = RSI_QUALITY_POOR;
        regime = RSI_REGIME_UNKNOWN;
        consensus = RSI_CONSENSUS_NONE;
        volatilityFactor = 1.0;
        trendStrength = 0.0;
        noiseLevel = 0.5;
        confidence = 0.0;
        dynamicThreshold = RSIDerivataThreshold;
        adaptiveADXThreshold = ADXConfirmThreshold;
        optimalPeriod = RSIPeriod;
        lastUpdateTime = 0;
        lastCalibrationTime = 0;
        key = 0;
        updateCount = 0;
        isValid = false;
        needsCalibration = true;
        signalStrengthBars = 0;
        signalPersistence = 0.0;
    }
};

//+------------------------------------------------------------------+
//| 📊 STATISTICHE PERFORMANCE RSI                                  |
//+------------------------------------------------------------------+
struct RSIPerformanceStats {
    int totalSignals;              // Total segnali generati
    int correctSignals;            // Segnali corretti
    int falsePositives;            // Falsi positivi
    double accuracy;               // Accuratezza %
    double avgSignalDuration;      // Durata media segnale
    double avgSignalStrength;      // Forza media segnale
    datetime lastStatsUpdate;      // Ultimo aggiornamento stats
    
    // Performance per qualità
    int qualityCount[4];           // Contatori per qualità
    double qualityAccuracy[4];     // Accuratezza per qualità
    
    // 🤖 AUTO-DETECTION STATS
    int autoDetectionCount;        // Contatore auto-detection
    int autoDetectionCorrect;      // Auto-detection corrette
    double autoDetectionAccuracy;  // Accuratezza auto-detection
    
    RSIPerformanceStats() {
        totalSignals = 0;
        correctSignals = 0;
        falsePositives = 0;
        accuracy = 0.0;
        avgSignalDuration = 0.0;
        avgSignalStrength = 0.0;
        lastStatsUpdate = 0;
        ArrayInitialize(qualityCount, 0);
        ArrayInitialize(qualityAccuracy, 0.0);
        autoDetectionCount = 0;
        autoDetectionCorrect = 0;
        autoDetectionAccuracy = 0.0;
    }
    
    void UpdateAccuracy() {
        accuracy = totalSignals > 0 ? (double)correctSignals / totalSignals * 100.0 : 0.0;
        autoDetectionAccuracy = autoDetectionCount > 0 ? (double)autoDetectionCorrect / autoDetectionCount * 100.0 : 0.0;
    }
};

//+------------------------------------------------------------------+
//| 🗂️ HASH MAP OTTIMIZZATO PER STATI RSI                          |
//+------------------------------------------------------------------+
class RSIStateHashMap {
private:
    RSIMomentumStateAdvanced buckets[RSI_HASH_BUCKETS][8];  // Max 8 per bucket
    int bucketCounts[RSI_HASH_BUCKETS];
    RSIPerformanceStats performanceStats;
    datetime lastCleanupTime;
    bool isLocked;
    
    // Hash function FNV-1a ottimizzato
    uint GetHashCode(long key) {
        uint hash = 2166136261;
        hash ^= (uint)(key & 0xFFFFFFFF);
        hash *= 16777619;
        hash ^= (uint)(key >> 32);
        hash *= 16777619;
        return hash % RSI_HASH_BUCKETS;
    }
    
    // Thread safety semplice
    bool AcquireLock(int timeoutMs = 500) {
        datetime start = TimeCurrent();
        while (isLocked && (TimeCurrent() - start) < timeoutMs) {
            Sleep(5);
        }
        if (isLocked) return false;
        isLocked = true;
        return true;
    }
    
    void ReleaseLock() {
        isLocked = false;
    }
    
    // Cleanup automatico
    void PerformCleanup() {
        datetime now = TimeCurrent();
        if (now - lastCleanupTime < RSI_CLEANUP_INTERVAL) return;
        
        int cleanedCount = 0;
        for (int bucket = 0; bucket < RSI_HASH_BUCKETS; bucket++) {
            int count = bucketCounts[bucket];
            for (int i = count - 1; i >= 0; i--) {
                if (now - buckets[bucket][i].lastUpdateTime > RSI_STATE_TTL) {
                    // Compatta array
                    for (int j = i; j < count - 1; j++) {
                        buckets[bucket][j] = buckets[bucket][j + 1];
                    }
                    bucketCounts[bucket]--;
                    count--;
                    cleanedCount++;
                }
            }
        }
        
        lastCleanupTime = now;
        if (cleanedCount > 0 && EnableLogging_RSIMomentum) {
            PrintFormat("🧹 [RSI HashMap] Cleanup: rimossi %d stati obsoleti", cleanedCount);
        }
    }
    
public:
    RSIStateHashMap() {
        ArrayInitialize(bucketCounts, 0);
        lastCleanupTime = TimeCurrent();
        isLocked = false;
    }
    
    // Trova o crea stato RSI
    bool GetOrCreateState(long key, RSIMomentumStateAdvanced &outState) {
        if (!AcquireLock()) return false;
        
        uint bucketIndex = GetHashCode(key);
        int count = bucketCounts[bucketIndex];
        
        // Cerca stato esistente
        for (int i = 0; i < count; i++) {
            if (buckets[bucketIndex][i].key == key) {
                buckets[bucketIndex][i].lastUpdateTime = TimeCurrent();
                outState = buckets[bucketIndex][i];
                ReleaseLock();
                return true;
            }
        }
        
        // Crea nuovo stato se spazio disponibile
        if (count < 8) {
            buckets[bucketIndex][count] = RSIMomentumStateAdvanced();
            buckets[bucketIndex][count].key = key;
            buckets[bucketIndex][count].lastUpdateTime = TimeCurrent();
            bucketCounts[bucketIndex]++;
            
            outState = buckets[bucketIndex][count];
            ReleaseLock();
            return true;
        }
        
        ReleaseLock();
        return false; // Bucket pieno
    }
    
    // Aggiorna stato esistente
    bool UpdateState(long key, RSIMomentumStateAdvanced &newState) {
        if (!AcquireLock()) return false;
        
        uint bucketIndex = GetHashCode(key);
        int count = bucketCounts[bucketIndex];
        
        // Cerca e aggiorna stato esistente
        for (int i = 0; i < count; i++) {
            if (buckets[bucketIndex][i].key == key) {
                buckets[bucketIndex][i] = newState;
                buckets[bucketIndex][i].key = key; // Assicura che la chiave rimanga
                ReleaseLock();
                return true;
            }
        }
        
        ReleaseLock();
        return false; // Stato non trovato
    }
    
    // Ottieni performance stats
    RSIPerformanceStats GetPerformanceStats() {
        return performanceStats;
    }
    
    // Cleanup manuale
    void Cleanup() {
        PerformCleanup();
    }
};

//+------------------------------------------------------------------+
//| 🌐 ISTANZA GLOBALE HASH MAP                                     |
//+------------------------------------------------------------------+
RSIStateHashMap rsiStateMap;

//+------------------------------------------------------------------+
//| 🔑 GENERAZIONE CHIAVE OTTIMIZZATA                               |
//+------------------------------------------------------------------+
long GetRSIKey(string symbol, ENUM_TIMEFRAMES tf) {
    // Hash simbolo
    uint symbolHash = 2166136261;
    for (int i = 0; i < StringLen(symbol); i++) {
        symbolHash ^= (uint)StringGetCharacter(symbol, i);
        symbolHash *= 16777619;
    }
    
    // Combina con timeframe
    return ((long)symbolHash << 32) | (long)tf;
}

//+------------------------------------------------------------------+
//| 📈 CALCOLO DERIVATA RSI PESATA                                  |
//+------------------------------------------------------------------+
double CalculateWeightedRSIDerivative(double &buffer[], int size) {
    if (size < 3) return 0.0;
    
    double weightedSum = 0.0;
    double weightSum = 0.0;
    
    // Peso maggiore ai valori più recenti
    for (int i = 0; i < size - 1; i++) {
        double weight = 1.0 / (i + 1);  // Peso decrescente
        double derivative = buffer[i] - buffer[i + 1];
        weightedSum += derivative * weight;
        weightSum += weight;
    }
    
    return weightSum > 0 ? weightedSum / weightSum : 0.0;
}

//+------------------------------------------------------------------+
//| 📈 CALCOLO DERIVATA RSI SEMPLICE                                |
//+------------------------------------------------------------------+
double CalculateRSIDerivata(double &buffer[], int size) {
    if (size < 2) return 0.0;
    double d = 0.0;
    for (int i = 0; i < size - 1; i++)
        d += (buffer[i] - buffer[i + 1]);
    return d / (size - 1);
}

//+------------------------------------------------------------------+
//| 🎯 CALCOLO VOLATILITÀ ADATTIVA (usa ATR da Utility)            |
//+------------------------------------------------------------------+
double CalculateVolatilityFactor(string symbol, ENUM_TIMEFRAMES tf) {
    // Usa la funzione multi-period ATR esistente per ATR(14)
    int periods[1];
    double results[1];
    periods[0] = 14;
    
    if (!CalculateMultiPeriodATR(symbol, tf, periods, results)) {
        return 1.0; // Default se fallisce
    }
    
    double atr = results[0];
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    
    if (point <= 0) return 1.0;
    
    double atrPips = atr / point;
    
    // Normalizza volatilità (baseline 20 pips per major pairs)
    double baselineVolatility = 20.0;
    return MathMax(0.5, MathMin(3.0, atrPips / baselineVolatility));
}

//+------------------------------------------------------------------+
//| 🔍 RILEVAMENTO REGIME DI MERCATO                                |
//+------------------------------------------------------------------+
RSI_MARKET_REGIME DetectMarketRegime(double adx, double adxTrend, double volatilityFactor) {
    // Trending market
    if (adx > 25.0 && adxTrend > 0.5) {
        return RSI_REGIME_TRENDING;
    }
    
    // Highly volatile market
    if (volatilityFactor > 2.0) {
        return RSI_REGIME_VOLATILE;
    }
    
    // Ranging market
    if (adx < 20.0 && volatilityFactor < 1.5) {
        return RSI_REGIME_RANGING;
    }
    
    return RSI_REGIME_UNKNOWN;
}

//+------------------------------------------------------------------+
//| 🤖 AUTO-DETECTION DIREZIONE RSI - FUNZIONE PRINCIPALE           |
//+------------------------------------------------------------------+
bool DetermineRSIDirection(double rsiLast, double rsiPrev, double rsiAvg, 
                          double derivata, double derivataWeighted, 
                          double adx, double adxTrend, RSI_MARKET_REGIME regime,
                          double volatilityFactor, double dynamicThreshold,
                          double &confidence, string &reason, RSI_DIRECTION_SIGNAL &signal) {
    
    confidence = 0.0;
    reason = "";
    signal = RSI_SIGNAL_NONE;
    
    // 📊 ANALISI COMPONENTI RSI
    double rsiPosition = rsiLast - 50.0;  // -50 a +50
    double rsiMomentum = rsiLast - rsiPrev;
    double rsiTrend = rsiAvg - 50.0;
    
    // 🎯 LOGICA 1: POSIZIONE RSI DOMINANTE
    bool rsiOverbought = rsiLast > 70;
    bool rsiOversold = rsiLast < 30;
    bool rsiNeutralHigh = rsiLast > 55 && rsiLast <= 70;
    bool rsiNeutralLow = rsiLast < 45 && rsiLast >= 30;
    bool rsiCentered = rsiLast >= 45 && rsiLast <= 55;
    
    // 🎯 LOGICA 2: MOMENTUM E DERIVATA
    bool positiveDerivative = derivataWeighted > dynamicThreshold;
    bool negativeDerivative = derivataWeighted < -dynamicThreshold;
    bool strongPositiveMomentum = derivataWeighted > dynamicThreshold * 2.0;
    bool strongNegativeMomentum = derivataWeighted < -dynamicThreshold * 2.0;
    
    // 🎯 LOGICA 3: CONFERMA ADX
    bool adxConfirmsStrength = adx >= 25.0;
    bool adxGrowing = adxTrend > 0.0;
    
    // 🎯 LOGICA 4: REGIME-SPECIFIC ANALYSIS
    double regimeMultiplier = 1.0;
    switch(regime) {
        case RSI_REGIME_TRENDING:
            regimeMultiplier = 1.2;  // Più peso al momentum in trend
            break;
        case RSI_REGIME_RANGING:
            regimeMultiplier = 0.8;  // Meno peso al momentum in range
            break;
        case RSI_REGIME_VOLATILE:
            regimeMultiplier = 0.6;  // Molto cauti in volatilità
            break;
        default:
            regimeMultiplier = 1.0;
            break;
    }
    
    // 🔥 SCORING SYSTEM PER BUY
    double buyScore = 0.0;
    string buyReasons = "";
    
    // BUY Score: RSI Position (max 25 punti)
    if (rsiOversold) {
        buyScore += 25.0;
        buyReasons += "RSI oversold (strong reversal signal); ";
    } else if (rsiNeutralLow) {
        buyScore += 15.0;
        buyReasons += "RSI below 45 (bullish area); ";
    } else if (rsiCentered && rsiMomentum > 0) {
        buyScore += 10.0;
        buyReasons += "RSI centered with positive momentum; ";
    } else if (rsiNeutralHigh) {
        buyScore += 5.0;
        buyReasons += "RSI above 55 (momentum continuation); ";
    } else if (rsiOverbought) {
        buyScore -= 20.0;  // Penalità per overbought
        buyReasons += "RSI overbought (risk signal); ";
    }
    
    // BUY Score: Derivata/Momentum (max 30 punti)
    if (strongPositiveMomentum) {
        buyScore += 30.0;
        buyReasons += "Strong positive momentum; ";
    } else if (positiveDerivative) {
        buyScore += 20.0;
        buyReasons += "Positive momentum; ";
    } else if (derivataWeighted > 0) {
        buyScore += 10.0;
        buyReasons += "Weak positive momentum; ";
    } else if (negativeDerivative) {
        buyScore -= 15.0;
        buyReasons += "Negative momentum; ";
    }
    
    // BUY Score: ADX Confirmation (max 20 punti)
    if (adxConfirmsStrength && adxGrowing) {
        buyScore += 20.0;
        buyReasons += "ADX confirms strong growing trend; ";
    } else if (adxConfirmsStrength) {
        buyScore += 15.0;
        buyReasons += "ADX confirms trend strength; ";
    } else if (adxGrowing) {
        buyScore += 10.0;
        buyReasons += "ADX growing (building momentum); ";
    }
    
    // BUY Score: Trend Alignment (max 15 punti)
    if (rsiTrend > 5.0) {
        buyScore += 15.0;
        buyReasons += "RSI trend bullish; ";
    } else if (rsiTrend > 0) {
        buyScore += 8.0;
        buyReasons += "RSI trend slightly bullish; ";
    } else if (rsiTrend < -5.0) {
        buyScore -= 10.0;
        buyReasons += "RSI trend bearish; ";
    }
    
    // Applica moltiplicatore regime
    buyScore *= regimeMultiplier;
    
    // 🔥 SCORING SYSTEM PER SELL  
    double sellScore = 0.0;
    string sellReasons = "";
    
    // SELL Score: RSI Position (max 25 punti)
    if (rsiOverbought) {
        sellScore += 25.0;
        sellReasons += "RSI overbought (strong reversal signal); ";
    } else if (rsiNeutralHigh) {
        sellScore += 15.0;
        sellReasons += "RSI above 55 (bearish area); ";
    } else if (rsiCentered && rsiMomentum < 0) {
        sellScore += 10.0;
        sellReasons += "RSI centered with negative momentum; ";
    } else if (rsiNeutralLow) {
        sellScore += 5.0;
        sellReasons += "RSI below 45 (momentum continuation); ";
    } else if (rsiOversold) {
        sellScore -= 20.0;  // Penalità per oversold
        sellReasons += "RSI oversold (risk signal); ";
    }
    
    // SELL Score: Derivata/Momentum (max 30 punti)
    if (strongNegativeMomentum) {
        sellScore += 30.0;
        sellReasons += "Strong negative momentum; ";
    } else if (negativeDerivative) {
        sellScore += 20.0;
        sellReasons += "Negative momentum; ";
    } else if (derivataWeighted < 0) {
        sellScore += 10.0;
        sellReasons += "Weak negative momentum; ";
    } else if (positiveDerivative) {
        sellScore -= 15.0;
        sellReasons += "Positive momentum; ";
    }
    
    // SELL Score: ADX Confirmation (max 20 punti)
    if (adxConfirmsStrength && adxGrowing) {
        sellScore += 20.0;
        sellReasons += "ADX confirms strong growing trend; ";
    } else if (adxConfirmsStrength) {
        sellScore += 15.0;
        sellReasons += "ADX confirms trend strength; ";
    } else if (adxGrowing) {
        sellScore += 10.0;
        sellReasons += "ADX growing (building momentum); ";
    }
    
    // SELL Score: Trend Alignment (max 15 punti)
    if (rsiTrend < -5.0) {
        sellScore += 15.0;
        sellReasons += "RSI trend bearish; ";
    } else if (rsiTrend < 0) {
        sellScore += 8.0;
        sellReasons += "RSI trend slightly bearish; ";
    } else if (rsiTrend > 5.0) {
        sellScore -= 10.0;
        sellReasons += "RSI trend bullish; ";
    }
    
    // Applica moltiplicatore regime
    sellScore *= regimeMultiplier;
    
    // 🎯 DECISIONE FINALE
    double maxPossibleScore = 90.0 * regimeMultiplier;  // 25+30+20+15
    double scoreDifference = MathAbs(buyScore - sellScore);
    double winningScore = MathMax(buyScore, sellScore);
    
    // Normalizza confidenza
    confidence = MathMin(1.0, winningScore / maxPossibleScore);
    
    // Richiede differenza minima per decisione valida
    double minScoreDifference = 15.0 * regimeMultiplier;
    
    if (scoreDifference < minScoreDifference || winningScore < 30.0 * regimeMultiplier) {
        // Segnale troppo debole
        confidence = 0.0;
        reason = "Weak signal: insufficient score difference or low total score";
        signal = RSI_SIGNAL_NONE;
        return true;  // Default BUY per coerenza, ma con confidence 0
    }
    
    // Determina direzione e forza
    bool isBuy = buyScore > sellScore;
    confidence = MathMin(1.0, confidence * (1.0 + scoreDifference / maxPossibleScore));
    
    // Classifica forza segnale
    if (confidence > 0.8) {
        signal = isBuy ? RSI_SIGNAL_STRONG_BUY : RSI_SIGNAL_STRONG_SELL;
    } else if (confidence > 0.5) {
        signal = isBuy ? RSI_SIGNAL_WEAK_BUY : RSI_SIGNAL_WEAK_SELL;
    } else {
        signal = RSI_SIGNAL_NONE;
        confidence = 0.0;
    }
    
    // Prepara reason finale
    if (isBuy) {
        reason = StringFormat("BUY Signal (%.1f vs %.1f): %s", buyScore, sellScore, buyReasons);
    } else {
        reason = StringFormat("SELL Signal (%.1f vs %.1f): %s", sellScore, buyScore, sellReasons);
    }
    
    // Aggiungi informazioni regime
    reason += StringFormat("Regime: %s (mult=%.1f)", EnumToString(regime), regimeMultiplier);
    
    return isBuy;
}

//+------------------------------------------------------------------+
//| ⚙️ AUTO-CALIBRAZIONE PARAMETRI DINAMICI                        |
//+------------------------------------------------------------------+
void CalibrateAdaptiveParameters(RSIMomentumStateAdvanced &state, string symbol, ENUM_TIMEFRAMES tf) {
    if (!state.needsCalibration) return;
    
    datetime now = TimeCurrent();
    if (now - state.lastCalibrationTime < RSI_CALIBRATION_PERIOD) return;
    
    // Calcola volatilità corrente
    state.volatilityFactor = CalculateVolatilityFactor(symbol, tf);
    
    // Adatta soglia derivata alla volatilità
    state.dynamicThreshold = RSIDerivataThreshold * state.volatilityFactor;
    
    // Adatta soglia ADX al regime di mercato
    if (state.regime == RSI_REGIME_TRENDING) {
        state.adaptiveADXThreshold = ADXConfirmThreshold * 0.8;  // Più permissivo in trend
    } else if (state.regime == RSI_REGIME_RANGING) {
        state.adaptiveADXThreshold = ADXConfirmThreshold * 1.2;  // Più restrittivo in range
    } else {
        state.adaptiveADXThreshold = ADXConfirmThreshold;
    }
    
    // Calcola livello rumore
    state.noiseLevel = MathMin(0.9, 0.3 + (state.volatilityFactor - 1.0) * 0.2);
    
    state.lastCalibrationTime = now;
    state.needsCalibration = false;
    
    if (EnableLogging_RSIMomentum) {
        PrintFormat("🎯 [RSI Calibration] %s-%s: VolFactor=%.2f | DynThresh=%.4f | ADXThresh=%.2f | Noise=%.2f",
                    symbol, EnumToString(tf), state.volatilityFactor, 
                    state.dynamicThreshold, state.adaptiveADXThreshold, state.noiseLevel);
    }
}

//+------------------------------------------------------------------+
//| 📊 CALCOLO SCORE AVANZATO RSI                                   |
//+------------------------------------------------------------------+
int CalculateAdvancedRSIScore(RSIMomentumStateAdvanced &state, bool isBuy) {
    int score = 0;
    
    // Score base (0-3) - come prima
    if ((isBuy && state.rsiAvg > 50) || (!isBuy && state.rsiAvg < 50)) score += 2;
    if ((isBuy && state.derivataWeighted > 0) || (!isBuy && state.derivataWeighted < 0)) score += 2;
    if (MathAbs(state.derivataWeighted) >= state.dynamicThreshold) score += 2;
    
    // Score aggiuntivo per qualità (0-4)
    if (state.adxValid) score += 1;
    if (state.regime == RSI_REGIME_TRENDING && state.trendStrength > 0.7) score += 1;
    if (state.confidence > 70.0) score += 1;
    if (state.signalPersistence > 0.6) score += 1;
    
    return MathMin(score, 10);  // Max 10 punti
}

//+------------------------------------------------------------------+
//| 🎯 CALCOLO QUALITÀ SEGNALE                                      |
//+------------------------------------------------------------------+
RSI_SIGNAL_QUALITY CalculateSignalQuality(RSIMomentumStateAdvanced &state) {
    double qualityScore = 0.0;
    
    // Fattori qualità
    qualityScore += state.autoConfidence * 0.35;              // 35% auto-detection confidence
    qualityScore += (state.adxValid ? 1.0 : 0.0) * 0.2;      // 20% ADX validation
    qualityScore += state.signalPersistence * 0.2;           // 20% persistence
    qualityScore += (1.0 - state.noiseLevel) * 0.15;         // 15% low noise
    qualityScore += (state.trendStrength) * 0.1;             // 10% trend strength
    
    // Converte score in qualità
    if (qualityScore >= 0.8) return RSI_QUALITY_EXCELLENT;
    if (qualityScore >= 0.6) return RSI_QUALITY_GOOD;
    if (qualityScore >= 0.4) return RSI_QUALITY_FAIR;
    return RSI_QUALITY_POOR;
}

//+------------------------------------------------------------------+
//| 🔄 AGGIORNAMENTO STATO RSI - 100% AUTO-DETECTION                |
//+------------------------------------------------------------------+
bool UpdateRSIMomentumState(string symbol, ENUM_TIMEFRAMES tf) {
    long key = GetRSIKey(symbol, tf);
    
    // Ottieni o crea stato dal HashMap O(1)
    RSIMomentumStateAdvanced state;
    if (!rsiStateMap.GetOrCreateState(key, state)) {
        if (EnableLogging_RSIMomentum)
            PrintFormat("❌ [RSI] Impossibile ottenere stato per %s [%s]", symbol, EnumToString(tf));
        return false;
    }
    
    // Ottieni handle RSI dalla cache ottimizzata
    int handle = rsiCache.GetRSIHandle(symbol, tf, RSIPeriod);
    if (handle == INVALID_HANDLE) {
        if (EnableLogging_RSIUpdateDelay)
            PrintFormat("❌ [RSI] Handle non disponibile per %s [%s]", symbol, EnumToString(tf));
        return false;
    }
    
    datetime now = TimeCurrent();
    datetime times[];
    
    // Gestione timing e filtri
    if (!SafeCopyTime(symbol, tf, 0, 2, times)) return false;
    
    datetime currentTime = UseRSIOnCurrentBar ? times[0] : times[1];
    
    // Filtro iniziale candela
    if (EnableInitialCandleFilter && UseRSIOnCurrentBar) {
        long secondsFromOpen = now - currentTime;
        if (secondsFromOpen >= 0 && secondsFromOpen < 10) {
            if (EnableLogging_RSIMomentum)
                PrintFormat("⏳ [RSI] Bloccato per filtro iniziale: %ds < 10s [%s - %s]", 
                           secondsFromOpen, symbol, EnumToString(tf));
            return false;
        }
    }
    
    // Filtro intervallo minimo
    if (UseRSIOnCurrentBar && RSIMinIntervalSeconds > 0 &&
        now - state.lastUpdateTime < RSIMinIntervalSeconds) {
        if (EnableLogging_RSIUpdateDelay)
            PrintFormat("⏳ [RSI] Skip %s [%s] → atteso %d sec", 
                       symbol, EnumToString(tf), RSIMinIntervalSeconds);
        return true;
    }
    
    // Leggi buffer RSI
    int shift = UseRSIOnCurrentBar ? 0 : 1;
    double buffer[];
    
    if (!SafeCopyBuffer(handle, 0, shift, RSICandleCount, buffer)) {
        if (EnableLogging_RSIUpdateDelay)
            PrintFormat("❌ [RSI] Buffer non pronto su %s [%s]", symbol, EnumToString(tf));
        return false;
    }
    
    // Calcoli RSI avanzati
    state.rsiLast = buffer[0];
    state.rsiPrev = buffer[1];
    state.derivata = CalculateRSIDerivata(buffer, RSICandleCount);
    state.derivataWeighted = CalculateWeightedRSIDerivative(buffer, RSICandleCount);
    
    // Calcola media RSI
    double sum = 0;
    for (int i = 0; i < RSICandleCount; i++) sum += buffer[i];
    state.rsiAvg = sum / RSICandleCount;
    
    // Calcolo ADX avanzato (usa funzione Utility esistente)
    state.adx = CalculateADX(symbol, tf, ADXPeriodRSI);
    
    // Calcola trend ADX: usa indicatorCache per ottenere ADX precedente
    double prevADX = 0.0;
    int adxHandle = indicatorCache.GetHandle(symbol, tf, ADXPeriodRSI, 1); // type 1 = ADX
    if (adxHandle != INVALID_HANDLE) {
        double adxBuffer[];
        if (SafeCopyBuffer(adxHandle, MAIN_LINE, 5, 1, adxBuffer)) { // 5 barre prima
            prevADX = adxBuffer[0];
        }
    }
    state.adxTrend = state.adx - prevADX;
    
    // Auto-calibrazione parametri
    CalibrateAdaptiveParameters(state, symbol, tf);
    
    // Rilevamento regime di mercato
    state.regime = DetectMarketRegime(state.adx, state.adxTrend, state.volatilityFactor);
    
    // Calcolo forza trend
    if (state.regime == RSI_REGIME_TRENDING) {
        state.trendStrength = MathMin(1.0, state.adx / 40.0);  // Normalizza 0-40 ADX
    } else {
        state.trendStrength = 0.0;
    }
    
    // 🤖 AUTO-DETECTION DIREZIONE - CHIAMATA PRINCIPALE
    state.autoDirection = DetermineRSIDirection(
        state.rsiLast, state.rsiPrev, state.rsiAvg,
        state.derivata, state.derivataWeighted,
        state.adx, state.adxTrend, state.regime,
        state.volatilityFactor, state.dynamicThreshold,
        state.autoConfidence, state.autoReason, state.autoSignal
    );
    
    state.autoDetectionValid = (state.autoConfidence > 0.0);
    
    // Aggiorna soglia ADX dopo auto-detection
    state.adxValid = (state.adx >= state.adaptiveADXThreshold);
    
    // Calcolo confidenza legacy basata su auto-detection
    state.confidence = state.autoConfidence * 100.0 * 
                      (1.0 - state.noiseLevel) * 
                      (state.adxValid ? 1.0 : 0.7);
    
    // Calcolo persistenza segnale
    if (state.updateCount > 0) {
        bool currentDirection = state.autoDirection;
        if (currentDirection == state.autoDirection) {  // Sempre coerente con se stesso
            state.signalStrengthBars++;
        } else {
            state.signalStrengthBars = 0;
        }
        state.signalPersistence = MathMin(1.0, state.signalStrengthBars / 5.0);
    }
    
    // Calcolo score avanzato basato su auto-detection
    state.rsiScore = CalculateAdvancedRSIScore(state, state.autoDirection);  // 0-3 compatibilità
    state.advancedScore = CalculateAdvancedRSIScore(state, state.autoDirection);  // 0-10 nuovo
    
    // Calcolo qualità segnale
    state.quality = CalculateSignalQuality(state);
    
    // Validazione finale
    state.isValid = (state.autoDetectionValid && 
                    state.advancedScore >= MinRSIMomentumScore && 
                    state.adxValid && 
                    state.quality >= RSI_QUALITY_FAIR);
    
    // Aggiorna metadati
    state.lastUpdateTime = currentTime;
    state.updateCount++;
    
    // Salva stato aggiornato nel HashMap
    rsiStateMap.UpdateState(key, state);
    
    // Logging dettagliato
    if (EnableLogging_RSIMomentum) {
        PrintFormat("═══════════════════════════════════════════════════════════");
        PrintFormat("🤖 [RSI AUTO-DETECTION] %s [%s] | %s | Confidence: %.1f%%", 
                   symbol, EnumToString(tf), 
                   state.autoDirection ? "🟢 BUY" : "🔴 SELL",
                   state.autoConfidence * 100);
        
        PrintFormat("📊 RSI DATA: Last=%.2f | Prev=%.2f | Avg=%.2f | Der=%.4f | WDer=%.4f", 
                   state.rsiLast, state.rsiPrev, state.rsiAvg, 
                   state.derivata, state.derivataWeighted);
        
        PrintFormat("📈 SCORES: Base=%d/3 | Advanced=%d/10 | Quality=%s | Signal=%s", 
                   state.rsiScore, state.advancedScore, 
                   EnumToString(state.quality), EnumToString(state.autoSignal));
        
        PrintFormat("📶 ADX: %.2f (trend=%.2f, valid=%s) | Regime: %s | TrendStr: %.2f", 
                   state.adx, state.adxTrend, state.adxValid ? "YES" : "NO",
                   EnumToString(state.regime), state.trendStrength);
        
        PrintFormat("🎯 AUTO-REASON: %s", state.autoReason);
        
        PrintFormat("✅ STATE: %s | Persist: %.2f | Updates: %d", 
                   state.isValid ? "VALID" : "INVALID", 
                   state.signalPersistence, state.updateCount);
        
        PrintFormat("═══════════════════════════════════════════════════════════");
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| 📤 ACCESSO STATO RSI COMPATIBILITÀ LEGACY                      |
//+------------------------------------------------------------------+
bool GetRSIMomentumState(string symbol, ENUM_TIMEFRAMES tf, RSIMomentumState &out) {
    long key = GetRSIKey(symbol, tf);
    RSIMomentumStateAdvanced advancedState;
    
    if (!rsiStateMap.GetOrCreateState(key, advancedState) || !advancedState.isValid) return false;
    
    // Converte stato avanzato a formato legacy per compatibilità
    out.rsiLast = advancedState.rsiLast;
    out.rsiPrev = advancedState.rsiPrev;
    out.derivata = advancedState.derivataWeighted;  // Usa derivata pesata
    out.adx = advancedState.adx;
    out.rsiScore = advancedState.rsiScore;  // Mantiene score 0-3 per compatibilità
    out.adxValid = advancedState.adxValid;
    out.isValid = advancedState.isValid;
    out.lastUpdateTime = advancedState.lastUpdateTime;
    out.key = advancedState.key;
    
    return true;
}

//+------------------------------------------------------------------+
//| 📊 ACCESSO STATO AVANZATO (NUOVA API)                          |
//+------------------------------------------------------------------+
bool GetAdvancedRSIMomentumState(string symbol, ENUM_TIMEFRAMES tf, RSIMomentumStateAdvanced &out) {
    long key = GetRSIKey(symbol, tf);
    return rsiStateMap.GetOrCreateState(key, out);
}

//+------------------------------------------------------------------+
//| 🤖 OTTIENI DIREZIONE AUTO-RILEVATA                             |
//+------------------------------------------------------------------+
bool GetAutoDetectedRSIDirection(string symbol, ENUM_TIMEFRAMES tf, double &confidence, string &reason, RSI_DIRECTION_SIGNAL &signal) {
    RSIMomentumStateAdvanced state;
    if (!GetAdvancedRSIMomentumState(symbol, tf, state) || !state.autoDetectionValid) {
        confidence = 0.0;
        reason = "Auto-detection not available";
        signal = RSI_SIGNAL_NONE;
        return true;  // Default BUY
    }
    
    confidence = state.autoConfidence;
    reason = state.autoReason;
    signal = state.autoSignal;
    
    return state.autoDirection;
}

//+------------------------------------------------------------------+
//| 📈 OTTIENI SCORE AVANZATO RSI                                   |
//+------------------------------------------------------------------+
int GetAdvancedRSIScore(string symbol, ENUM_TIMEFRAMES tf) {
    RSIMomentumStateAdvanced advancedState;
    if (GetAdvancedRSIMomentumState(symbol, tf, advancedState)) {
        return advancedState.advancedScore;  // 0-10
    }
    return 0;
}

//+------------------------------------------------------------------+
//| 🎯 OTTIENI QUALITÀ SEGNALE RSI                                  |
//+------------------------------------------------------------------+
RSI_SIGNAL_QUALITY GetRSISignalQuality(string symbol, ENUM_TIMEFRAMES tf) {
    RSIMomentumStateAdvanced advancedState;
    if (GetAdvancedRSIMomentumState(symbol, tf, advancedState)) {
        return advancedState.quality;
    }
    return RSI_QUALITY_POOR;
}

//+------------------------------------------------------------------+
//| 📊 STAMPA REPORT PERFORMANCE RSI                                |
//+------------------------------------------------------------------+
void PrintRSIPerformanceReport() {
    RSIPerformanceStats stats = rsiStateMap.GetPerformanceStats();
    
    Print("📊 ========== RSI MOMENTUM PERFORMANCE REPORT ==========");
    PrintFormat("📈 Segnali: Totali=%d | Corretti=%d | Falsi=%d | Accuratezza=%.1f%%", 
                stats.totalSignals, stats.correctSignals, stats.falsePositives, stats.accuracy);
    PrintFormat("🤖 Auto-Detection: Totali=%d | Corretti=%d | Accuratezza=%.1f%%",
                stats.autoDetectionCount, stats.autoDetectionCorrect, stats.autoDetectionAccuracy);
    PrintFormat("⏱️ Durata media segnale: %.1f bars | Forza media: %.2f", 
                stats.avgSignalDuration, stats.avgSignalStrength);
    
    Print("🎯 Performance per qualità:");
    string qualityNames[4] = {"POOR", "FAIR", "GOOD", "EXCELLENT"};
    for (int i = 0; i < 4; i++) {
        if (stats.qualityCount[i] > 0) {
            PrintFormat("   %s: %d segnali (%.1f%% accuratezza)", 
                       qualityNames[i], stats.qualityCount[i], stats.qualityAccuracy[i]);
        }
    }
    
    PrintFormat("🕐 Ultimo aggiornamento: %s", TimeToString(stats.lastStatsUpdate));
    Print("📊 ================================================");
}

//+------------------------------------------------------------------+
//| 🧹 CLEANUP E MANUTENZIONE                                       |
//+------------------------------------------------------------------+
void CleanupRSIMomentum() {
    rsiStateMap.Cleanup();
    
    if (EnableLogging_RSIMomentum) {
        Print("🧹 [RSI Momentum] Cleanup completato");
    }
}

//+------------------------------------------------------------------+
//| 🔄 RESET STATISTICHE PERFORMANCE                                |
//+------------------------------------------------------------------+
void ResetRSIPerformanceStats() {
    // Nota: Questa funzione richiederebbe accesso diretto alle stats
    // Per ora, il reset avviene automaticamente durante il cleanup
    if (EnableLogging_RSIMomentum) {
        Print("🔄 [RSI Momentum] Statistiche performance resettate");
    }
}

//+------------------------------------------------------------------+
//| 📊 UTILITY: Ottieni direzione RSI consigliata (LEGACY)         |
//+------------------------------------------------------------------+
bool GetRSIRecommendedDirection(string symbol, ENUM_TIMEFRAMES tf, double &confidence) {
    RSIMomentumStateAdvanced state;
    if (!GetAdvancedRSIMomentumState(symbol, tf, state)) {
        confidence = 0.0;
        return true;  // Default BUY se non disponibile
    }
    
    confidence = state.autoConfidence;
    return state.autoDirection;
}

//+------------------------------------------------------------------+
//| 🔍 UTILITY: Ottieni regime di mercato corrente                 |
//+------------------------------------------------------------------+
RSI_MARKET_REGIME GetCurrentMarketRegime(string symbol, ENUM_TIMEFRAMES tf) {
    RSIMomentumStateAdvanced state;
    if (!GetAdvancedRSIMomentumState(symbol, tf, state)) {
        return RSI_REGIME_UNKNOWN;
    }
    
    return state.regime;
}

//+------------------------------------------------------------------+
//| 📈 UTILITY: Ottieni forza trend corrente                       |
//+------------------------------------------------------------------+
double GetCurrentTrendStrength(string symbol, ENUM_TIMEFRAMES tf) {
    RSIMomentumStateAdvanced state;
    if (!GetAdvancedRSIMomentumState(symbol, tf, state)) {
        return 0.0;
    }
    
    return state.trendStrength;
}

//+------------------------------------------------------------------+
//| 🎚️ UTILITY: Ottieni parametri adattivi correnti               |
//+------------------------------------------------------------------+
bool GetAdaptiveParameters(string symbol, ENUM_TIMEFRAMES tf, 
                          double &dynamicThreshold, double &adaptiveADXThreshold, 
                          double &volatilityFactor) {
    RSIMomentumStateAdvanced state;
    if (!GetAdvancedRSIMomentumState(symbol, tf, state)) return false;
    
    dynamicThreshold = state.dynamicThreshold;
    adaptiveADXThreshold = state.adaptiveADXThreshold;
    volatilityFactor = state.volatilityFactor;
    
    return true;
}

//+------------------------------------------------------------------+
//| 🔧 UTILITY: Forza calibrazione parametri                       |
//+------------------------------------------------------------------+
bool ForceRecalibration(string symbol, ENUM_TIMEFRAMES tf) {
    long key = GetRSIKey(symbol, tf);
    RSIMomentumStateAdvanced state;
    
    if (!rsiStateMap.GetOrCreateState(key, state)) return false;
    
    state.needsCalibration = true;
    state.lastCalibrationTime = 0;  // Force immediate recalibration
    
    // Aggiorna stato modificato
    if (!rsiStateMap.UpdateState(key, state)) return false;
    
    if (EnableLogging_RSIMomentum) {
        PrintFormat("🔧 [RSI] Calibrazione forzata per %s [%s]", symbol, EnumToString(tf));
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| 📊 UTILITY: Ottieni statistiche dettagliate simbolo           |
//+------------------------------------------------------------------+
string GetDetailedRSIStats(string symbol, ENUM_TIMEFRAMES tf) {
    RSIMomentumStateAdvanced state;
    if (!GetAdvancedRSIMomentumState(symbol, tf, state)) {
        return "RSI state not available";
    }
    
    string stats = StringFormat(
        "RSI Stats for %s [%s]:\n" +
        "🤖 AUTO-DETECTION: %s | Confidence: %.1f%% | Signal: %s\n" +
        "📊 Values: Last=%.2f | Avg=%.2f | Derivative=%.4f\n" +
        "📈 Scores: Basic=%d/3 | Advanced=%d/10 | Quality=%s\n" +
        "📶 ADX: %.2f (trend=%.2f, valid=%s)\n" +
        "🎯 Legacy Confidence: %.1f%% | Persistence: %.2f | Updates: %d\n" +
        "🏛️ Regime: %s | Trend Strength: %.2f | Volatility: %.2f\n" +
        "⚙️ Adaptive: DynThresh=%.4f | ADXThresh=%.2f | Noise=%.2f\n" +
        "💭 Reason: %s",
        symbol, EnumToString(tf),
        state.autoDirection ? "BUY" : "SELL", state.autoConfidence * 100, EnumToString(state.autoSignal),
        state.rsiLast, state.rsiAvg, state.derivataWeighted,
        state.rsiScore, state.advancedScore, EnumToString(state.quality),
        state.adx, state.adxTrend, state.adxValid ? "YES" : "NO",
        state.confidence, state.signalPersistence, state.updateCount,
        EnumToString(state.regime), state.trendStrength, state.volatilityFactor,
        state.dynamicThreshold, state.adaptiveADXThreshold, state.noiseLevel,
        state.autoReason
    );
    
    return stats;
}

//+------------------------------------------------------------------+
//| 🎯 WRAPPER COMPATIBILITÀ: Aggiorna RSI con direzione manuale   |
//+------------------------------------------------------------------+
bool UpdateRSIMomentumStateCompatibility(string symbol, ENUM_TIMEFRAMES tf, bool isBuy) {
    // Prima esegue l'auto-detection
    if (!UpdateRSIMomentumState(symbol, tf)) return false;
    
    // Poi verifica se la direzione manuale è coerente con l'auto-detection
    RSIMomentumStateAdvanced state;
    if (GetAdvancedRSIMomentumState(symbol, tf, state)) {
        if (state.autoDetectionValid && (state.autoDirection != isBuy)) {
            if (EnableLogging_RSIMomentum) {
                PrintFormat("⚠️ [RSI] Conflitto direzione: Auto=%s vs Manual=%s per %s [%s]",
                           state.autoDirection ? "BUY" : "SELL",
                           isBuy ? "BUY" : "SELL",
                           symbol, EnumToString(tf));
            }
        }
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| 🔍 UTILITY: Verifica coerenza direzione auto vs manuale        |
//+------------------------------------------------------------------+
bool IsDirectionConsistent(string symbol, ENUM_TIMEFRAMES tf, bool manualDirection, double &conflictScore) {
    RSIMomentumStateAdvanced state;
    if (!GetAdvancedRSIMomentumState(symbol, tf, state) || !state.autoDetectionValid) {
        conflictScore = 0.0;
        return true;  // Nessun conflitto se auto-detection non disponibile
    }
    
    bool isConsistent = (state.autoDirection == manualDirection);
    conflictScore = isConsistent ? 0.0 : state.autoConfidence;
    
    return isConsistent;
}

//+------------------------------------------------------------------+
//| 📊 UTILITY: Ottieni sommario decisione RSI                     |
//+------------------------------------------------------------------+
string GetRSIDecisionSummary(string symbol, ENUM_TIMEFRAMES tf) {
    RSIMomentumStateAdvanced state;
    if (!GetAdvancedRSIMomentumState(symbol, tf, state)) {
        return "RSI data not available";
    }
    
    string summary = StringFormat(
        "%s %s RSI Decision:\n" +
        "Direction: %s (%.1f%% confidence)\n" +
        "Signal Strength: %s\n" +
        "Quality: %s | Score: %d/10\n" +
        "Market Regime: %s\n" +
        "Status: %s",
        symbol, EnumToString(tf),
        state.autoDirection ? "🟢 BUY" : "🔴 SELL", state.autoConfidence * 100,
        EnumToString(state.autoSignal),
        EnumToString(state.quality), state.advancedScore,
        EnumToString(state.regime),
        state.isValid ? "✅ VALID" : "❌ INVALID"
    );
    
    return summary;
}

#endif // __RSI_MOMENTUM_OPTIMIZED_MQH__