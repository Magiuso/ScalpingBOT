//+------------------------------------------------------------------+
//|              EMA_MACD_BodyScore - Versione Ottimizzata          |
//|  Modulo di Prevalidazione MACD+EMA e BodyMomentum per BOT 4.0   |
//+------------------------------------------------------------------+

#ifndef __EMA_MACD_BODYSCORE_MQH__
#define __EMA_MACD_BODYSCORE_MQH__

#include <ScalpingBot\Utility.mqh>

//+------------------------------------------------------------------+
//| Enumerazioni per modalità operative                              |
//+------------------------------------------------------------------+
enum ENUM_DIRECTION_MODE {
    DIR_EMA_MASTER,      // EMA determina sempre la direzione
    DIR_MAJORITY_VOTE,   // Voto di maggioranza tra i blocchi
    DIR_WEIGHTED_AVG,    // Media pesata delle indicazioni direzionali
    DIR_STRONGEST_SIGNAL // Segue il blocco con score più alto
};

enum ENUM_CONFLICT_RESOLUTION {
    CONFLICT_REDUCE_SCORE,    // Riduce lo score in caso di conflitto
    CONFLICT_BLOCK_SIGNAL,    // Blocca il segnale se c'è conflitto
    CONFLICT_FOLLOW_PRIMARY   // Segue sempre il blocco primario
};

//+------------------------------------------------------------------+
//| Struttura configurazione orchestratore                           |
//+------------------------------------------------------------------+
struct OrchestratorConfig {
    ENUM_DIRECTION_MODE directionMode;
    ENUM_CONFLICT_RESOLUTION conflictMode;
    double minATRValue;              // ATR minimo per procedere
    double coherenceBonus;           // Bonus quando tutti i segnali concordano
    double conflictPenalty;          // Penalità per segnali contrastanti
    double minScoreThreshold;        // Score minimo per considerare valido un blocco
    bool requireAllBlocksEnabled;    // Richiede tutti i blocchi attivi
    
    // Costruttore con valori di default
    OrchestratorConfig() {
        directionMode = DIR_EMA_MASTER;
        conflictMode = CONFLICT_REDUCE_SCORE;
        minATRValue = 0.00001;
        coherenceBonus = 1.0;
        conflictPenalty = 2.0;
        minScoreThreshold = 2.0;
        requireAllBlocksEnabled = false;
    }
};

//+------------------------------------------------------------------+
//| Struttura estesa per risultati dettagliati                      |
//+------------------------------------------------------------------+
struct PrevalidationResult {
    // Scores
    double scoreTotal;          
    double scoreEMA;            
    double scoreMACD;           
    double scoreBodyMomentum;
    
    // Direzioni
    bool directionEMAMACDBody;  // Direzione finale
    bool emaDirection;          // Direzione rilevata da EMA
    bool macdSuggestedDirection; // Direzione suggerita da MACD se indipendente
    bool bodySuggestedDirection; // Direzione suggerita da Body se indipendente
    
    // Stati
    bool emaTrendDetected;      // EMA ha trovato un trend valido
    bool signalValid;           // Segnale complessivo valido
    bool hasConflict;           // Presenza di conflitti tra blocchi
    int activeBlocks;           // Numero di blocchi attivi
    
    // Dettagli
    string reasons;             
    double coherenceLevel;      // Livello di coerenza tra blocchi (0-1)
    double atrValue;            // Valore ATR utilizzato
    
    void Reset() {
        scoreTotal = 0.0;
        scoreEMA = 0.0;
        scoreMACD = 0.0;
        scoreBodyMomentum = 0.0;
        directionEMAMACDBody = false;
        emaDirection = false;
        macdSuggestedDirection = false;
        bodySuggestedDirection = false;
        emaTrendDetected = false;
        signalValid = false;
        hasConflict = false;
        activeBlocks = 0;
        reasons = "";
        coherenceLevel = 0.0;
        atrValue = 0.0;
    }
};

//+------------------------------------------------------------------+
//| Classe principale orchestratore                                  |
//+------------------------------------------------------------------+
class PrevalidationOrchestrator {
private:
    OrchestratorConfig config;
    
    // Calcola ATR con gestione errori robusta
    bool CalculateATRSafe(string symbol, ENUM_TIMEFRAMES tf, double &atrValue) {
        int atrPeriods[] = {ATRPeriod};
        double atrResults[];
        
        if (!CalculateMultiPeriodATR(symbol, tf, atrPeriods, atrResults)) {
            return false;
        }
        
        atrValue = atrResults[0];
        return (atrValue > config.minATRValue);
    }
    
    // Determina la direzione basata sulla modalità configurata
    bool DetermineDirection(const PrevalidationResult &result) {
        switch(config.directionMode) {
            case DIR_EMA_MASTER:
                return result.emaDirection;
                
            case DIR_MAJORITY_VOTE: {
                int buyVotes = 0;
                if (result.emaDirection && result.emaTrendDetected) buyVotes++;
                if (result.scoreMACD > config.minScoreThreshold) {
                    // Assumiamo che MACD suggerisca BUY se score alto in direzione EMA
                    if (result.emaDirection) buyVotes++;
                }
                if (result.scoreBodyMomentum > config.minScoreThreshold) {
                    // Idem per Body
                    if (result.emaDirection) buyVotes++;
                }
                return (buyVotes >= 2);
            }
            
            case DIR_WEIGHTED_AVG: {
                double buyWeight = 0.0;
                double totalWeight = 0.0;
                
                if (result.emaTrendDetected) {
                    buyWeight += result.emaDirection ? (result.scoreEMA * EMA_Weight) : 0;
                    totalWeight += result.scoreEMA * EMA_Weight;
                }
                
                // Per MACD e Body, assumiamo coerenza con EMA per semplicità
                // In una versione più complessa, questi blocchi potrebbero avere proprie direzioni
                
                return (totalWeight > 0 && buyWeight/totalWeight > 0.5);
            }
            
            case DIR_STRONGEST_SIGNAL: {
                if (result.scoreEMA >= result.scoreMACD && 
                    result.scoreEMA >= result.scoreBodyMomentum) {
                    return result.emaDirection;
                }
                // Altrimenti segui il più forte (per ora assume EMA direction)
                return result.emaDirection;
            }
        }
        
        return false;
    }
    
    // Calcola il livello di coerenza tra i blocchi
    double CalculateCoherence(const PrevalidationResult &result) {
        if (result.activeBlocks < 2) return 1.0; // Nessun conflitto possibile
        
        double coherence = 0.0;
        int comparisons = 0;
        
        // Confronta gli score normalizzati
        if (EnableEMABlock && EnableMACDCheck) {
            double diff = MathAbs(result.scoreEMA - result.scoreMACD) / 10.0;
            coherence += (1.0 - diff);
            comparisons++;
        }
        
        if (EnableEMABlock && EnableBodyMomentumScore) {
            double diff = MathAbs(result.scoreEMA - result.scoreBodyMomentum) / 10.0;
            coherence += (1.0 - diff);
            comparisons++;
        }
        
        if (EnableMACDCheck && EnableBodyMomentumScore) {
            double diff = MathAbs(result.scoreMACD - result.scoreBodyMomentum) / 10.0;
            coherence += (1.0 - diff);
            comparisons++;
        }
        
        return (comparisons > 0) ? coherence / comparisons : 0.0;
    }
    
    // Applica bonus/penalità per coerenza
    void ApplyCoherenceAdjustments(PrevalidationResult &result) {
        result.coherenceLevel = CalculateCoherence(result);
        
        // Bonus per alta coerenza
        if (result.coherenceLevel > 0.8 && result.activeBlocks >= 2) {
            double bonus = config.coherenceBonus * result.coherenceLevel;
            result.scoreTotal += bonus;
            result.reasons += StringFormat("✅ Bonus coerenza +%.2f ", bonus);
        }
        
        // Penalità per conflitti
        if (result.coherenceLevel < 0.5 && result.activeBlocks >= 2) {
            result.hasConflict = true;
            
            switch(config.conflictMode) {
                case CONFLICT_REDUCE_SCORE: {
                    double penalty = config.conflictPenalty * (1.0 - result.coherenceLevel);
                    result.scoreTotal -= penalty;
                    result.reasons += StringFormat("⚠️ Penalità conflitto -%.2f ", penalty);
                    break;
                }
                
                case CONFLICT_BLOCK_SIGNAL:
                    result.signalValid = false;
                    result.reasons += "❌ Segnale bloccato per conflitto ";
                    break;
                    
                case CONFLICT_FOLLOW_PRIMARY:
                    // Nessuna penalità, segue il primario
                    result.reasons += "⚡ Conflitto ignorato (segue primario) ";
                    break;
            }
        }
    }
    
public:
    // Costruttore
    PrevalidationOrchestrator() {
        config = OrchestratorConfig();
    }
    
    PrevalidationOrchestrator(const OrchestratorConfig &cfg) {
        config = cfg;
    }
    
    // Funzione principale di calcolo
    PrevalidationResult Calculate(string symbol, ENUM_TIMEFRAMES tf) {
        PrevalidationResult result;
        result.Reset();
        
        // === 🕐 Filtro candela iniziale ===
        if (EnableInitialCandleFilter) {
            datetime time[];
            if (CopyTime(symbol, tf, 0, 1, time) > 0) {
                long seconds = TimeCurrent() - time[0];
                if (seconds >= 0 && seconds < InitialCandleDelaySec) {
                    result.reasons = StringFormat("⏳ Filtro candela: %ds < %ds", 
                                                seconds, InitialCandleDelaySec);
                    return result;
                }
            }
        }
        
        // === 🔕 Controllo modulo principale ===
        if (!EnableEMAMACDBody) {
            result.reasons = "⚪️ Modulo principale disattivato";
            return result;
        }
        
        // === 📊 Calcolo ATR robusto ===
        if (!CalculateATRSafe(symbol, tf, result.atrValue)) {
            result.reasons = "❌ ATR non disponibile o troppo basso";
            return result;
        }
        
        // === 🔍 Conta blocchi attivi ===
        result.activeBlocks = 0;
        if (EnableEMABlock) result.activeBlocks++;
        if (EnableMACDCheck) result.activeBlocks++;
        if (EnableBodyMomentumScore) result.activeBlocks++;
        
        if (result.activeBlocks == 0) {
            result.reasons = "❌ Nessun blocco di analisi attivo";
            return result;
        }
        
        if (config.requireAllBlocksEnabled && result.activeBlocks < 3) {
            result.reasons = "❌ Richiesti tutti i blocchi attivi";
            return result;
        }
        
        // === 📐 BLOCCO EMA ===
        if (EnableEMABlock) {
            string emaReason = "";
            result.emaTrendDetected = DetectEMACompositeTrend(
                symbol, tf, result.emaDirection, result.scoreEMA, emaReason
            );
            result.reasons += emaReason;
        }
        
        // Determina direzione iniziale
        bool workingDirection = DetermineDirection(result);
        
        // === 📊 BLOCCO MACD ===
        if (EnableMACDCheck) {
            string macdReason = "";
            result.scoreMACD = EvalMACDScore(
                symbol, tf, workingDirection, macdReason, result.atrValue
            );
            result.reasons += macdReason;
        }
        
        // === 💪 BLOCCO BODY MOMENTUM ===
        if (EnableBodyMomentumScore) {
            // Usa la nuova versione con configurazione
            BodyMomentumConfig bodyConfig;
            bodyConfig.volumeConfirmationThreshold = BodyMomentum_VolumeConfirmationThreshold;
            bodyConfig.bodyATRRatioThreshold = BodyMomentum_BodyATRRatioThreshold;
            bodyConfig.stdDevATRRatioThreshold = BodyMomentum_StdDevATRRatioThreshold;
            bodyConfig.weightCoherence = BodyMomentum_WeightCoherence;
            bodyConfig.weightDominance = BodyMomentum_WeightDominance;
            bodyConfig.weightShadow = BodyMomentum_WeightShadow;
            bodyConfig.weightProgression = BodyMomentum_WeightProgression;
            bodyConfig.weightClose = BodyMomentum_WeightClose;
            bodyConfig.weightVolume = BodyMomentum_WeightVolume;
            
            BodyMomentumResult bodyResult;
            result.scoreBodyMomentum = CalcBodyMomentumScoreV2(
                symbol, tf, workingDirection, bodyResult, result.atrValue, bodyConfig
            );
            
            result.reasons += StringFormat("💪 Body=%.2f ", result.scoreBodyMomentum);
        }
        
        // === 🎯 Calcolo score totale ===
        if (NormalizePrevalidationScores) {
            // Modalità pesata: calcola media pesata e scala a 0-30
            double totalWeight = 0.0;
            double weightedSum = 0.0;
            
            if (EnableEMABlock) {
                totalWeight += EMA_Weight;
                weightedSum += result.scoreEMA * EMA_Weight;
            }
            if (EnableMACDCheck) {
                totalWeight += MACD_Weight;
                weightedSum += result.scoreMACD * MACD_Weight;
            }
            if (EnableBodyMomentumScore) {
                totalWeight += BodyMomentum_Weight;
                weightedSum += result.scoreBodyMomentum * BodyMomentum_Weight;
            }
            
            // Media pesata (0-10) scalata a 0-30
            if (totalWeight > 0) {
                double weightedAverage = weightedSum / totalWeight;
                result.scoreTotal = weightedAverage * 3.0;
            } else {
                result.scoreTotal = 0.0;
            }
        } else {
            // Modalità somma diretta: semplicemente somma i punteggi (0-30)
            result.scoreTotal = 0.0;
            
            if (EnableEMABlock) {
                result.scoreTotal += result.scoreEMA;
            }
            if (EnableMACDCheck) {
                result.scoreTotal += result.scoreMACD;
            }
            if (EnableBodyMomentumScore) {
                result.scoreTotal += result.scoreBodyMomentum;
            }
        }
                  
        // === 🔄 Applica aggiustamenti per coerenza ===
        ApplyCoherenceAdjustments(result);
        
        // Assicura che lo score rimanga in range 0-10
        // Se hai MinPrevalidationScore definito come input, usa la riga seguente:
        // result.signalValid = (result.scoreTotal >= MinPrevalidationScore && !result.hasConflict);
        
        // Altrimenti usa questo valore di default:
        result.signalValid = (result.scoreTotal >= 3.0 && !result.hasConflict);
        
        // === 📍 Direzione finale ===
        result.directionEMAMACDBody = workingDirection;
        result.signalValid = (result.scoreTotal >= MinPrevalidationScore && !result.hasConflict);
        
        // === 📋 Log finale ===
        if (EnableLogging_EMA_MACD_Body) {
            LogResults(symbol, tf, result);
        }
        
        return result;
    }
    
private:
    void LogResults(string symbol, ENUM_TIMEFRAMES tf, const PrevalidationResult &result) {
        Print("═══════════════ [Prevalidation Analysis] ═══════════════");
        PrintFormat("📍 %s | %s | ATR: %.5f", symbol, EnumToString(tf), result.atrValue);
        PrintFormat("🔧 Mode: %s | Blocks: %d/3", 
                    EnumToString(config.directionMode), result.activeBlocks);
        
        if (EnableEMABlock) {
            PrintFormat("📐 EMA:   %.2f/10 | Trend: %s | Dir: %s", 
                        result.scoreEMA, 
                        result.emaTrendDetected ? "FOUND" : "NONE",
                        result.emaDirection ? "BUY" : "SELL");
        }
        
        if (EnableMACDCheck) {
            PrintFormat("📊 MACD:  %.2f/10", result.scoreMACD);
        }
        
        if (EnableBodyMomentumScore) {
            PrintFormat("💪 Body:  %.2f/10", result.scoreBodyMomentum);
        }
        
        PrintFormat("🎯 TOTAL: %.2f/10 | Direction: %s | Valid: %s",
                    result.scoreTotal, 
                    result.directionEMAMACDBody ? "BUY" : "SELL",
                    result.signalValid ? "YES" : "NO");
        
        PrintFormat("📊 Coherence: %.1f%% | Conflict: %s",
                    result.coherenceLevel * 100,
                    result.hasConflict ? "YES" : "NO");
        
        if (result.reasons != "") {
            PrintFormat("📝 Details: %s", result.reasons);
        }
        
        Print("═══════════════════════════════════════════════════════");
    }
};

//+------------------------------------------------------------------+
//| Funzione wrapper per compatibilità                               |
//+------------------------------------------------------------------+
PrevalidationResult CalculatePrevalidationScore(string symbol, ENUM_TIMEFRAMES tf)
{
    // Usa configurazione di default o personalizzata
    OrchestratorConfig config;
    // Puoi personalizzare la config qui basandoti su input globali
    // config.directionMode = DirectionModeInput;
    // config.conflictMode = ConflictModeInput;
    
    PrevalidationOrchestrator orchestrator(config);
    return orchestrator.Calculate(symbol, tf);
}

//+--------------------------------------------------------------------------------+
//| BLOCCO 1: 📈 DetectEMACompositeTrend - Versione Ottimizzata con CEMACache     |
//+--------------------------------------------------------------------------------+

//+--------------------------------------------------------------------------------+
//| Strutture di supporto per l'analisi ottimizzata                                |
//+--------------------------------------------------------------------------------+
struct DerivativeAnalysis {
    double d1, d2, d3;
    double avgSlope;
    double avgSlopePercent;
    bool isAccelerating;
    bool isDecelerating;
    double decelerationRate;
    bool allDerivativesCoherent;
    bool isSlowSteady;
    
    // MODIFICATO: Accetta i valori EMA direttamente invece dell'array
    void Calculate(double ema5_0, double ema5_1, double ema5_2, double ema5_3, 
                   double currentPrice, bool isBuy) {
        // Calcolo derivate con i valori passati
        d1 = ema5_0 - ema5_1;
        d2 = ema5_1 - ema5_2;
        d3 = ema5_2 - ema5_3;
        
        // Media delle pendenze
        double slopeSum = d1 + d2 + d3;
        avgSlope = slopeSum / 3.0;
        avgSlopePercent = (currentPrice != 0.0) ? (MathAbs(avgSlope) / currentPrice) * 100.0 : 0.0;
        
        // Analisi accelerazione/decelerazione
        if (isBuy) {
            isAccelerating = (d1 > d2 && d2 > d3);
            isDecelerating = (d1 < d2 && d2 < d3);
            allDerivativesCoherent = (d1 > 0 && d2 > 0 && d3 > 0);
        } else { // SELL
            isAccelerating = (d1 < d2 && d2 < d3);
            isDecelerating = (d1 > d2 && d2 > d3);
            allDerivativesCoherent = (d1 < 0 && d2 < 0 && d3 < 0);
        }
        
        // Calcola tasso di decelerazione
        decelerationRate = 0.0;
        const double MinValidSlope = 0.00001;
        if (MathAbs(d2) > MinValidSlope && MathAbs(d1) < MathAbs(d2)) {
            decelerationRate = 1.0 - (MathAbs(d1) / MathAbs(d2));
            decelerationRate = MathMax(0.0, MathMin(1.0, decelerationRate));
        }
        
        // Trend lento ma costante
        isSlowSteady = allDerivativesCoherent &&
                       MathAbs(d1) < SlowTrendDeltaThreshold && // Assicurati che SlowTrendDeltaThreshold sia definito
                       MathAbs(d2) < SlowTrendDeltaThreshold &&
                       MathAbs(d3) < SlowTrendDeltaThreshold;
    }
};

struct ScoreComponents {
    double baseTrend;       // 0-40% del max score
    double historical;      // 0-20% del max score
    double acceleration;    // 0-20% del max score
    double stability;       // 0-20% del max score
    double penalties;       // Penalità cumulative
    
    double GetTotal() { 
        return MathMax(0.0, baseTrend + historical + acceleration + stability - penalties);
    }
    
    void Reset() {
        baseTrend = 0.0;
        historical = 0.0;
        acceleration = 0.0;
        stability = 0.0;
        penalties = 0.0;
    }
};

//+--------------------------------------------------------------------------------+
//| BLOCCO 1: 📈 DetectEMACompositeTrend - Versione Ottimizzata con CEMACache     |
//+--------------------------------------------------------------------------------+
bool DetectEMACompositeTrend(string symbol, ENUM_TIMEFRAMES tf,
                             bool &trendBuy, double &scoreEMA, string &reasonLog)
{
    // === 🔄 Inizializzazione ===
    scoreEMA = 0.0;
    reasonLog = "";
    trendBuy = false;

    // 🔕 Controllo attivazione blocco
    if (!EnableEMABlock) { // Assicurati che EnableEMABlock sia definito
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog = "⚪️ Blocco EMA disattivato";
        }
        return false;
    }

    // === 📦 Validazione e preparazione dati dalla CEMACache ===
    const int candlesToRead = 4; // Necessario per derivate (d3 richiede 4 punti)

    // Usa CEMACache per ottenere e copiare i dati
    // Assicurati che 'emaCache' sia una variabile globale o passata
    if (!emaCache.GetAndCopyEMADatas(symbol, tf, 5, 20, candlesToRead)) {
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += "❌ ERRORE: Impossibile caricare i dati EMA dalla cache.\n";
        }
        return false;
    }
    
    // Accedi ai dati EMA tramite i metodi della cache
    double ema5_0 = emaCache.GetEMA5(0);
    double ema5_1 = emaCache.GetEMA5(1);
    double ema5_2 = emaCache.GetEMA5(2);
    double ema5_3 = emaCache.GetEMA5(3);

    double ema20_0 = emaCache.GetEMA20(0);
    double ema20_1 = emaCache.GetEMA20(1);

    // Validazione dati
    if (ema5_0 == 0.0 || ema5_1 == 0.0 || ema20_0 == 0.0 || ema20_1 == 0.0) {
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += "⚠️ Dati EMA non validi (valore zero) dalla cache.\n";
        }
        return false;
    }

    // === 📊 Analisi iniziale del trend ===
    double deltaCurrent = ema5_0 - ema5_1;

    // Validazione movimento minimo
    if (MathAbs(deltaCurrent) < Point() * 0.1) {
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += "⚠️ Movimento EMA troppo piccolo per essere significativo\n";
        }
        return false;
    }

    // Determina direzione del trend
    trendBuy = (deltaCurrent > 0);
    double currentPrice = GetNormalizedPrice(symbol, trendBuy); // Assicurati che GetNormalizedPrice sia definito

    // === 📈 Analisi pendenza principale ===
    double slopePercent = (currentPrice != 0.0) ? (MathAbs(deltaCurrent) / currentPrice) * 100.0 : 0.0;
    ScoreComponents scores;
    scores.Reset();

    // Valuta forza del trend
    if (!EvaluateTrendStrength(slopePercent, trendBuy, scores, reasonLog)) { // Questa funzione verrà modificata sotto
        return false;
    }

    // === 📶 Validazione ADX ===
    if (!ValidateADX(symbol, tf, reasonLog)) { // Questa funzione verrà modificata sotto
        return false;
    }

    // === 📏 Analisi distanza EMA ===
    ValidateEMADistance(ema5_0, ema20_0, currentPrice, scores, reasonLog); // Questa funzione verrà modificata sotto

    // === 🧮 Analisi derivate complete ===
    DerivativeAnalysis derivatives;
    derivatives.Calculate(ema5_0, ema5_1, ema5_2, ema5_3, currentPrice, trendBuy);

    // Applica scoring basato sulle derivate
    ApplyDerivativeScoring(derivatives, scores, reasonLog); // Questa funzione verrà modificata sotto

    // === 🚀 Analisi spike e accelerazione ===
    AnalyzeSpikeAndAcceleration(derivatives, ema5_0, ema5_1, ema5_2, ema5_3, 
                                 currentPrice, trendBuy, scores, reasonLog); // Questa funzione verrà modificata sotto

    // === ⚠️ Controllo convergenza ===
    CheckConvergence(ema5_0, ema5_1, ema20_0, ema20_1, scores, reasonLog); // Questa funzione verrà modificata sotto

    // === ✨ Verifiche allineamento e cross ===
    CheckEMAAlignment(ema5_0, ema20_0, trendBuy, scores, reasonLog); // Questa funzione verrà modificata sotto
    CheckEMACross(ema5_0, ema5_1, ema20_0, ema20_1, trendBuy, scores, reasonLog); // Questa funzione verrà modificata sotto

    // === 🎯 Calcolo score finale ===
    double totalScoreRaw = scores.GetTotal();
    scoreEMA = NormalizeScore(totalScoreRaw, EMA_TargetMaxScore); // Assicurati che EMA_TargetMaxScore sia definito

    // === 📋 Logging dettagliato ===
    // Questo blocco Print completo è già condizionato da EnableLogging_EMA
    if (EnableLogging_EMA) {
        LogEMAAnalysis(symbol, tf, trendBuy, slopePercent, derivatives, scores, 
                       totalScoreRaw, scoreEMA, ema5_0, ema5_1, ema5_2, ema5_3, 
                       ema20_0, ema20_1, reasonLog);
    }

    // Riepilogo finale del reasonLog (QUESTA È L'ULTIMA OCCORRENZA DA CONDIZIONARE)
    if (EnableLogging_EMA) { // <-- AGGIUNTA
        reasonLog += "\n--- Riepilogo Score EMA ---\n";
        reasonLog += StringFormat("Base Trend: %.2f | Storico: %.2f | Accel: %.2f | Stabil: %.2f | Penalità: %.2f\n",
                                  scores.baseTrend, scores.historical, scores.acceleration, 
                                  scores.stability, scores.penalties);
        reasonLog += StringFormat("🎯 Score Finale EMA: %.2f/%.2f (Normalizzato: %.2f/10.00)",
                                  totalScoreRaw, EMA_TargetMaxScore, scoreEMA);
    }

    return scoreEMA > 0.0;
}

//+--------------------------------------------------------------------------------+
//| Funzioni di supporto aggiornate per CEMACache                                  |
//+--------------------------------------------------------------------------------+

double GetNormalizedPrice(string symbol, bool isBuy)
{
    double price = isBuy ? SymbolInfoDouble(symbol, SYMBOL_ASK) :
                             SymbolInfoDouble(symbol, SYMBOL_BID);
    if (price <= 0.0) {
        price = SymbolInfoDouble(symbol, SYMBOL_LAST);
        if (price <= 0.0) {
            price = 1.0; // Evita divisioni per zero
        }
    }
    return price;
}

bool EvaluateTrendStrength(double slopePercent, bool isBuy, 
                           ScoreComponents &scores, string &reasonLog)
{
    // Validazione parametri
    // Assicurati che StrongSlopePercentThreshold e SlopePercentThreshold siano definiti
    if (StrongSlopePercentThreshold <= SlopePercentThreshold) {
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += "❌ Configurazione soglie pendenza non valida (Strong <= Base).\n";
        }
        return false;
    }
    
    double maxBaseTrend = EMA_TargetMaxScore * 0.4;
    string trendDir = isBuy ? "BUY" : "SELL";

    if (slopePercent >= StrongSlopePercentThreshold) {
        scores.baseTrend = maxBaseTrend;
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("📈 Trend %s FORTE: pendenza EMA5 = %.2f%% (>= %.2f%%)\n",
                                      trendDir, slopePercent, StrongSlopePercentThreshold);
        }
        return true;
    }
    else if (slopePercent >= SlopePercentThreshold) {
        scores.baseTrend = maxBaseTrend * 0.5;
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("📈 Trend %s MEDIO: pendenza EMA5 = %.2f%% (>= %.2f%%)\n",
                                      trendDir, slopePercent, SlopePercentThreshold);
        }
        return true;
    }

    if (EnableLogging_EMA) { // <-- AGGIUNTA
        reasonLog += StringFormat("❌ Trend %s DEBOLE: pendenza %.2f%% < soglia minima %.2f%%\n",
                                  trendDir, slopePercent, SlopePercentThreshold);
    }
    return false;
}

bool ValidateADX(string symbol, ENUM_TIMEFRAMES tf, string &reasonLog)
{
    // Assicurati che CalculateADX e ADXPeriod siano definiti
    double adx = CalculateADX(symbol, tf, ADXPeriod); 
    
    if (adx <= 0.0) {
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += "❌ Errore o ADX non disponibile (valore <= 0.0).\n";
        }
        return false;
    }
    
    if (adx < MinADXThreshold) { // Assicurati che MinADXThreshold sia definito
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("🛑 ADX troppo basso (%.2f < %.2f). Trend non significativo.\n", 
                                      adx, MinADXThreshold);
        }
        return false;
    }
    
    if (EnableLogging_EMA) { // <-- AGGIUNTA
        reasonLog += StringFormat("📶 ADX = %.2f (OK)\n", adx);
    }
    return true;
}

void ValidateEMADistance(double ema5_0, double ema20_0, double currentPrice,
                           ScoreComponents &scores, string &reasonLog)
{
    double distance = MathAbs(ema5_0 - ema20_0);
    double avgEMA = (ema5_0 + ema20_0) / 2.0;
    double minDist = avgEMA * MinEMADistancePercent / 100.0; // Assicurati che MinEMADistancePercent sia definito
    
    if (distance < minDist) {
        // Assicurati che EMA_DistancePenalty sia definito
        scores.penalties += EMA_DistancePenalty; 
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("⚠️ Distanza EMA insufficiente (%.5f < %.5f) → penalità %.2f\n",
                                      distance, minDist, EMA_DistancePenalty);
        }
    } else { // Aggiunto per un log positivo della distanza se richiesto
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("✅ Distanza EMA sufficiente (%.5f >= %.5f).\n", distance, minDist);
        }
    }
}

// 1. MODIFICA per ApplyDerivativeScoring - Mostra valori accelerazione/decelerazione
void ApplyDerivativeScoring(const DerivativeAnalysis &deriv,
                            ScoreComponents &scores, string &reasonLog)
{
    double maxHistorical = EMA_TargetMaxScore * 0.2;
    double maxStability = EMA_TargetMaxScore * 0.2;
    
    // Bonus pendenza storica
    // Assicurati che StrongSlopePercentThreshold e SlopePercentThreshold siano definiti
    if (deriv.avgSlopePercent >= StrongSlopePercentThreshold) { 
        scores.historical = maxHistorical;
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("✅ Pendenza storica FORTE (%.2f%%) -> bonus %.2f\n",
                                      deriv.avgSlopePercent, maxHistorical);
        }
    }
    else if (deriv.avgSlopePercent >= SlopePercentThreshold) {
        scores.historical = maxHistorical * 0.5;
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("✅ Pendenza storica MEDIA (%.2f%%) -> bonus %.2f\n",
                                      deriv.avgSlopePercent, maxHistorical * 0.5);
        }
    } else { // Penalità pendenza debole
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("❌ Pendenza storica DEBOLE (%.2f%%)\n", deriv.avgSlopePercent);
        }
    }
    
    // Bonus stabilità
    if (deriv.isSlowSteady) {
        scores.stability = maxStability;
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("🧘 Trend stabile e costante -> bonus %.2f\n", maxStability);
        }
    }
    
    // Controllo spike prima della penalità decelerazione
    // Assicurati che SpikeStrengthThreshold sia definito
    bool spikeDetected = (MathAbs(deriv.d2) > SpikeStrengthThreshold || 
                          MathAbs(deriv.d3) > SpikeStrengthThreshold);
    
    if (spikeDetected) {
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("🚀 Spike rilevato (Δ2=%.5f, Δ3=%.5f) - decelerazione post-spike ignorata\n",
                                      deriv.d2, deriv.d3);
        }
    } else if (deriv.decelerationRate > 0.0) {
        double penaltyAmount = scores.baseTrend * deriv.decelerationRate;
        scores.penalties += penaltyAmount;
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("⛔ Decelerazione %.1f%% (Δ1=%.5f < Δ2=%.5f) -> penalità %.2f\n",
                                      deriv.decelerationRate * 100, deriv.d1, deriv.d2, penaltyAmount);
        }
    }
}

// MODIFICA per AnalyzeSpikeAndAcceleration - Mostra valori accelerazione
void AnalyzeSpikeAndAcceleration(const DerivativeAnalysis &deriv, 
                                 double ema5_0, double ema5_1, double ema5_2, double ema5_3,
                                 double currentPrice, bool isBuy,
                                 ScoreComponents &scores, string &reasonLog)
{
    double maxAcceleration = EMA_TargetMaxScore * 0.2;
    
    // Spike detection
    // Assicurati che SpikeStrengthThreshold e SpikeBonusMultiplier siano definiti
    bool spikeDetected = (MathAbs(deriv.d2) > SpikeStrengthThreshold || 
                           MathAbs(deriv.d3) > SpikeStrengthThreshold);
    bool spikeInTrend = (isBuy && (deriv.d2 > 0 || deriv.d3 > 0)) || 
                        (!isBuy && (deriv.d2 < 0 || deriv.d3 < 0));
    
    if (spikeDetected && spikeInTrend) {
        double spikeBonus = maxAcceleration * SpikeBonusMultiplier;
        scores.acceleration += spikeBonus;
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("🚀 Spike %s (Δ2=%.5f, Δ3=%.5f) -> bonus %.2f\n", 
                                      isBuy ? "rialzista" : "ribassista", 
                                      deriv.d2, deriv.d3, spikeBonus);
        }
    }
    
    // Accelerazione generale - MODIFICA: Aggiungi i valori delle derivate
    if (deriv.isAccelerating) {
        double accelBonus = maxAcceleration * 0.5;
        scores.acceleration += accelBonus;
        
        // Log dettagliato con i valori di accelerazione
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("⚡ Accelerazione generale (Δ1=%.5f %s Δ2=%.5f %s Δ3=%.5f) -> bonus %.2f\n", 
                                      MathAbs(deriv.d1), 
                                      (isBuy ? ">" : "<"),
                                      MathAbs(deriv.d2), 
                                      (isBuy ? ">" : "<"),
                                      MathAbs(deriv.d3), 
                                      accelBonus);
        }
    } else if (deriv.isDecelerating) {
        // Aggiungi log per decelerazione
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("🛑 Decelerazione rilevata (Δ1=%.5f %s Δ2=%.5f %s Δ3=%.5f)\n",
                                      MathAbs(deriv.d1),
                                      (isBuy ? "<" : ">"),
                                      MathAbs(deriv.d2),
                                      (isBuy ? "<" : ">"),
                                      MathAbs(deriv.d3));
        }
    } else {
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("〰️ Nessuna accelerazione/decelerazione (Δ1=%.5f, Δ2=%.5f, Δ3=%.5f)\n",
                                      deriv.d1, deriv.d2, deriv.d3);
        }
    }
}

// 2. MODIFICA per CheckConvergence - Mostra valori EMA
void CheckConvergence(double ema5_0, double ema5_1, double ema20_0, double ema20_1,
                      ScoreComponents &scores, string &reasonLog)
{
    // Assicurati che EnableEMAConvergenceCheck e EMA_InversionPenalty siano definiti
    if (!EnableEMAConvergenceCheck) return; 
    
    double diffNow = ema5_0 - ema20_0;
    double diffPrev = ema5_1 - ema20_1;
    
    double deltaNow = MathAbs(diffNow);
    double deltaPrev = MathAbs(diffPrev);
    
    bool hasCrossed = (diffNow * diffPrev < 0);
    
    if (hasCrossed) {
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("🔄 Cross EMA rilevato (EMA5: %.5f→%.5f, EMA20: %.5f→%.5f)\n",
                                      ema5_1, ema5_0, ema20_1, ema20_0);
        }
        return;
    }
    
    if (deltaNow < deltaPrev) {
        scores.penalties += EMA_InversionPenalty;
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("⚠️ Convergenza EMA (Δ ora=%.5f < Δ prima=%.5f) [EMA5=%.5f, EMA20=%.5f] -> Penalità %.2f\n",
                                      deltaNow, deltaPrev, ema5_0, ema20_0, EMA_InversionPenalty);
        }
    } else {
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("✅ Divergenza EMA (Δ ora=%.5f > Δ prima=%.5f) [EMA5=%.5f, EMA20=%.5f]\n", 
                                     deltaNow, deltaPrev, ema5_0, ema20_0);
        }
    }
}

// 3. MODIFICA per CheckEMAAlignment - Mostra valori EMA
void CheckEMAAlignment(double ema5_0, double ema20_0, bool isBuy,
                        ScoreComponents &scores, string &reasonLog)
{
    // Assicurati che EnableEMAAlignmentCheck e EMA_AlignmentBonus siano definiti
    if (!EnableEMAAlignmentCheck) return; 

    bool aligned = (isBuy && ema5_0 > ema20_0) || (!isBuy && ema5_0 < ema20_0);

    if (aligned) {
        scores.baseTrend += EMA_AlignmentBonus;
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("✅ EMA allineate con trend %s (EMA5=%.5f %s EMA20=%.5f) -> Bonus %.2f\n", 
                                      isBuy ? "BUY" : "SELL",
                                      ema5_0, 
                                      (isBuy ? ">" : "<"), 
                                      ema20_0, 
                                      EMA_AlignmentBonus);
        }
    } else {
        if (EnableLogging_EMA) { // <-- AGGIUNTA
            reasonLog += StringFormat("ℹ️ EMA non allineate con trend %s (EMA5=%.5f %s EMA20=%.5f, dovrebbe essere %s)\n",
                                      isBuy ? "BUY" : "SELL",
                                      ema5_0,
                                      (ema5_0 > ema20_0 ? ">" : "<"),
                                      ema20_0,
                                      (isBuy ? ">" : "<"));
        }
    }
}

// BONUS: Modifica anche CheckEMACross per maggiore chiarezza
void CheckEMACross(double ema5_0, double ema5_1, double ema20_0, double ema20_1, bool isBuy,
                    ScoreComponents &scores, string &reasonLog)
{
    // Assicurati che EnableEMACrossCheck, EMA_CrossBonus e EMA_CrossPenalty siano definiti
    if (!EnableEMACrossCheck) return;

    bool crossUp = (ema5_1 <= ema20_1 && ema5_0 > ema20_0);
    bool crossDown = (ema5_1 >= ema20_1 && ema5_0 < ema20_0);

    if (isBuy) {
        if (crossUp) {
            scores.baseTrend += EMA_CrossBonus;
            if (EnableLogging_EMA) { // <-- AGGIUNTA
                reasonLog += StringFormat("✅ Cross EMA rialzista (EMA5: %.5f→%.5f attraversa EMA20: %.5f→%.5f) -> Bonus %.2f\n", 
                                          ema5_1, ema5_0, ema20_1, ema20_0, EMA_CrossBonus);
            }
        } else if (crossDown) {
            scores.penalties += EMA_CrossPenalty;
            if (EnableLogging_EMA) { // <-- AGGIUNTA
                reasonLog += StringFormat("❌ Cross EMA ribassista (EMA5: %.5f→%.5f attraversa EMA20: %.5f→%.5f) -> Penalità %.2f\n", 
                                          ema5_1, ema5_0, ema20_1, ema20_0, EMA_CrossPenalty);
            }
        }
    } else { // SELL
        if (crossDown) {
            scores.baseTrend += EMA_CrossBonus;
            if (EnableLogging_EMA) { // <-- AGGIUNTA
                reasonLog += StringFormat("✅ Cross EMA ribassista (EMA5: %.5f→%.5f attraversa EMA20: %.5f→%.5f) -> Bonus %.2f\n", 
                                          ema5_1, ema5_0, ema20_1, ema20_0, EMA_CrossBonus);
            }
        } else if (crossUp) {
            scores.penalties += EMA_CrossPenalty;
            if (EnableLogging_EMA) { // <-- AGGIUNTA
                reasonLog += StringFormat("❌ Cross EMA rialzista (EMA5: %.5f→%.5f attraversa EMA20: %.5f→%.5f) -> Penalità %.2f\n", 
                                          ema5_1, ema5_0, ema20_1, ema20_0, EMA_CrossPenalty);
            }
        }
    }
}

double NormalizeScore(double rawScore, double maxScore)
{
    if (maxScore <= 0.0) return 0.0;
    
    double normalized = (rawScore / maxScore) * 10.0;
    return MathMin(10.0, MathMax(0.0, normalized));
}

// MODIFICATA: Accetta valori EMA invece di array
void LogEMAAnalysis(string symbol, ENUM_TIMEFRAMES tf, bool isBuy,
                    double slopePercent, const DerivativeAnalysis &deriv,
                    const ScoreComponents &scores, double rawScore, double finalNormalizedScore,
                    double ema5_0, double ema5_1, double ema5_2, double ema5_3,
                    double ema20_0, double ema20_1,
                    string reasonDetails)  
{
    // Questo blocco Print è già condizionato dalla riga 'if (EnableLogging_EMA)' nella funzione chiamante.
    // Non aggiungere qui ulteriori 'if (EnableLogging_EMA)' per non duplicare la condizione.
    Print("═══════════════ [EMA Composite Trend - CEMACache] ═══════════════");
    PrintFormat("📍 %s | %s | Direzione: %s", symbol, EnumToString(tf), isBuy ? "BUY" : "SELL");
    PrintFormat("📊 EMA[0]: EMA5=%.5f, EMA20=%.5f (Δ=%.5f)", ema5_0, ema20_0, ema5_0-ema20_0);
    PrintFormat("📊 EMA[1]: EMA5=%.5f, EMA20=%.5f (Δ=%.5f)", ema5_1, ema20_1, ema5_1-ema20_1);
    PrintFormat("📈 Pendenza attuale: %.2f%%", slopePercent);
    PrintFormat("📐 Pendenza media storica: %.2f%%", deriv.avgSlopePercent);
    PrintFormat("🧮 Derivate: Δ1=%.5f Δ2=%.5f Δ3=%.5f", deriv.d1, deriv.d2, deriv.d3);

    string status = deriv.isAccelerating ? "🚀 ACCEL" :
                    deriv.isDecelerating ? "🛑 DECEL" : "〰️ STAB";
    PrintFormat("📊 Status: %s | Decelerazione: %.1f%%", status, deriv.decelerationRate * 100);

    Print("──── Componenti Score ────");
    PrintFormat("  Base Trend:    %.2f", scores.baseTrend);
    PrintFormat("  Historical:    %.2f", scores.historical);
    PrintFormat("  Acceleration:  %.2f", scores.acceleration);
    PrintFormat("  Stability:     %.2f", scores.stability);
    PrintFormat("  Penalties:   -%.2f", scores.penalties);
    PrintFormat("  TOTALE GREZZO: %.2f → Normalizzato: %.2f/10.00", rawScore, finalNormalizedScore);
    Print("═══════════════════════════════════════════════════════════════");
    
    // Stampa i dettagli accumulati nel reasonLog
    Print(reasonDetails);
}

//+------------------------------------------------------------------+
//| BLOCCO 2: 📊 EvalMACDScore - Supporto dinamico MACD al trend EMA|
//| Calcola punteggio 0-10 basato su:                                |
//| - Direzione del trend MACD (linea MACD vs Signal)                |
//| - Momentum crescente (istogramma MACD)                            |
//| - Pendenza dell'istogramma (accelerazione del momentum)           |
//| - Prossimità/Crossover della linea zero (forza del trend)         |
//| - Nuove logiche per punteggio, accelerazione/decelerazione        |
//| - Normalizzazione dinamica basata su ATR                          |
//| - Penalità/Bonus configurabili                                    |
//| - CALCOLA ISTOGRAMMA MANUALMENTE SE IL BUFFER DÀ ERRORE           |
//| - UTILIZZA CMACDCache per gestione handle robusta                 |
//| **AGGIORNATO:** Logica di Divergenza Prezzo-Istogramma MACD corretta |
//| **AGGIORNATO:** Gestione robusta dei dati con la nuova CMACDCache |
//+------------------------------------------------------------------+

// Costante per la pendenza minima valida dell'istogramma
const double MACD_MinValidHistogramSlope = 0.000001;

//+------------------------------------------------------------------+
//| Strutture di supporto per l'analisi MACD ottimizzata             |
//+------------------------------------------------------------------+
struct MACDData {
    double macdLine[];     // Array dinamico per la linea MACD
    double signalLine[];   // Array dinamico per la linea Signal
    double histogram[];    // Array dinamico per l'Istogramma
    bool histogramCalculated; // Indica se l'istogramma è stato calcolato
    int dataSize;          // Tiene traccia della dimensione corrente degli array

    // Costruttore: Inizializza la struct come vuota
    MACDData() {
        dataSize = 0;
        histogramCalculated = false;
    }
    
    // Resetta la struct liberando la memoria degli array
    void Reset() {
        ArrayFree(macdLine);    // Libera la memoria dell'array macdLine
        ArrayFree(signalLine);  // Libera la memoria dell'array signalLine
        ArrayFree(histogram);   // Libera la memoria dell'array histogram
        dataSize = 0;
        histogramCalculated = false;
    }
    
    // Ridimensiona gli array interni della struct
    void SetSize(int size) {
        if (size <= 0) {
            Reset(); // Se la dimensione è 0 o negativa, resetta tutto
            return;
        }
        // Ridimensiona tutti gli array alla nuova dimensione
        ArrayResize(macdLine, size);
        ArrayResize(signalLine, size);
        ArrayResize(histogram, size);
        dataSize = size; // Aggiorna la dimensione memorizzata
        histogramCalculated = false; // L'istogramma dovrà essere ricalcolato o copiato
    }
    
    // Calcola l'istogramma basandosi su macdLine e signalLine
    void CalculateHistogram() {
        if (dataSize > 0) {
            for(int i = 0; i < dataSize; i++) {
                histogram[i] = macdLine[i] - signalLine[i];
            }
            histogramCalculated = true;
        } else {
            histogramCalculated = false; // Se non ci sono dati, l'istogramma non è calcolato
        }
    }
    
    // Verifica se gli array sono stati allocati e hanno la dimensione corretta
    bool IsValid() const {
        return dataSize > 0 && 
               ArraySize(macdLine) == dataSize && 
               ArraySize(signalLine) == dataSize && 
               ArraySize(histogram) == dataSize;
    }
};

struct MACDScoreComponents {
    double crossScore;      // 0-2 punti
    double coherenceScore;  // 0-2 punti  
    double zeroLineScore;   // 0-2 punti
    double momentumScore;   // 0-2 punti
    double distanceScore;   // 0-2 punti
    double penalties;       // Somma negativa
    
    double GetTotal() const {
        double total = crossScore + coherenceScore + zeroLineScore + 
                       momentumScore + distanceScore - penalties;
        return MathMax(0.0, MathMin(10.0, total));
    }
    
    void Reset() {
        crossScore = 0.0;
        coherenceScore = 0.0;
        zeroLineScore = 0.0;
        momentumScore = 0.0;
        distanceScore = 0.0;
        penalties = 0.0;
    }
};

struct MomentumAnalysis {
    double currentSlope;
    double previousSlope;
    double decelerationFactor;
    bool isAccelerating;
    bool isDecelerating;
    bool slopeInTrendDirection;
    
    void Calculate(const double &hist[], bool isBuy) {
        currentSlope = hist[0] - hist[1];
        previousSlope = hist[1] - hist[2];
        
        // Determina se la pendenza è nella direzione del trend
        slopeInTrendDirection = (isBuy && currentSlope > 0) || (!isBuy && currentSlope < 0);
        bool prevSlopeInTrend = (isBuy && previousSlope > 0) || (!isBuy && previousSlope < 0);
        
        // Reset fattori
        decelerationFactor = 0.0;
        isAccelerating = false;
        isDecelerating = false;
        
        if (!slopeInTrendDirection) {
            // Pendenza contro trend = decelerazione massima
            decelerationFactor = 1.0;
            isDecelerating = true;
        }
        else if (slopeInTrendDirection && prevSlopeInTrend) {
            double absCurrentSlope = MathAbs(currentSlope);
            double absPrevSlope = MathAbs(previousSlope);
            
            if (absPrevSlope > MACD_MinValidHistogramSlope) {
                if (absCurrentSlope > absPrevSlope * 1.1) {
                    isAccelerating = true;
                }
                else if (absCurrentSlope < absPrevSlope * 0.9) {
                    isDecelerating = true;
                    decelerationFactor = 1.0 - (absCurrentSlope / absPrevSlope);
                    decelerationFactor = MathMax(0.0, MathMin(1.0, decelerationFactor));
                }
            }
        }
    }
};

class SwingPointDetector {
private:
    struct SwingPoint {
        int index;
        double value;
    };
    
    // N.B.: Le funzioni FindSwingHighs e FindSwingLows non cambiano la loro firma.
    // Loro accettano un singolo array e la sua dimensione.
    // Sarà la funzione chiamante (CheckBearish/BullishDivergence) a passarle la dimensione corretta.
    void FindSwingHighs(const double &data[], int size, int minBars, SwingPoint &last, SwingPoint &secondLast) {
        last.index = -1;
        secondLast.index = -1;
        
        // Verifica che ci siano abbastanza dati *effettivamente disponibili* per l'analisi dello swing
        if (size < 2 * minBars + 1) { 
             return; 
        }

        for(int i = minBars; i < size - minBars; i++) {
            // Questa è la riga 951 dove hai avuto l'errore
            if (data[i] == 0.0 && i > 0) continue; // Continua solo se non è la barra 0 e il valore è 0
            
            bool isSwingHigh = true;
            for(int j = 1; j <= minBars && isSwingHigh; j++) {
                // Controlla che gli indici siano validi rispetto alla dimensione passata
                if (i - j < 0 || i + j >= size) { 
                    isSwingHigh = false;
                    break; 
                }
                // Controlla anche che i valori vicini non siano 0.0 (non-dati)
                if (data[i-j] == 0.0 || data[i+j] == 0.0) {
                    isSwingHigh = false;
                    break;
                }
                if(data[i] <= data[i-j] || data[i] <= data[i+j]) {
                    isSwingHigh = false;
                }
            }
            
            if(isSwingHigh) {
                // ... (logica di aggiornamento last/secondLast come prima)
                if(last.index == -1) {
                    last.index = i;
                    last.value = data[i];
                }
                else if(i > last.index + minBars) {
                    secondLast = last;
                    last.index = i;
                    last.value = data[i];
                } else if (i < last.index) {
                    if (secondLast.index == -1 || i > secondLast.index) {
                         secondLast.index = i;
                         secondLast.value = data[i];
                    }
                }
            }
        }
        if (last.index != -1 && secondLast.index != -1 && last.index < secondLast.index) {
            SwingPoint temp = last;
            last = secondLast;
            secondLast = temp;
        }
    }
    
    // Trova swing lows (logica speculare a FindSwingHighs)
    void FindSwingLows(const double &data[], int size, int minBars, SwingPoint &last, SwingPoint &secondLast) {
        last.index = -1;
        secondLast.index = -1;
        
        if (size < 2 * minBars + 1) { 
            return;
        }

        for(int i = minBars; i < size - minBars; i++) {
            if (data[i] == 0.0 && i > 0) continue;
            
            bool isSwingLow = true;
            for(int j = 1; j <= minBars && isSwingLow; j++) {
                if (i - j < 0 || i + j >= size) { 
                    isSwingLow = false;
                    break; 
                }
                if (data[i-j] == 0.0 || data[i+j] == 0.0) {
                    isSwingLow = false;
                    break;
                }
                if(data[i] >= data[i-j] || data[i] >= data[i+j]) {
                    isSwingLow = false;
                }
            }
            
            if(isSwingLow) {
                if(last.index == -1) {
                    last.index = i;
                    last.value = data[i];
                }
                else if(i > last.index + minBars) {
                    secondLast = last;
                    last.index = i;
                    last.value = data[i];
                } else if (i < last.index) {
                    if (secondLast.index == -1 || i > secondLast.index) {
                         secondLast.index = i;
                         secondLast.value = data[i];
                    }
                }
            }
        }
        if (last.index != -1 && secondLast.index != -1 && last.index < secondLast.index) {
            SwingPoint temp = last;
            last = secondLast;
            secondLast = temp;
        }
    }
    
public:
    // MODIFICA LA FIRMA: Ora accetta due parametri di dimensione
    bool CheckBearishDivergence(const double &highPrices[], const double &histogram[], 
                                int pricesSize, int histSize, // NUOVI PARAMETRI
                                int minBars, double minRatio) {
        SwingPoint lastPriceHigh, secondLastPriceHigh;
        SwingPoint lastHistHigh, secondLastHistHigh;
        
        // Passa la dimensione corretta a FindSwingHighs
        FindSwingHighs(highPrices, pricesSize, minBars, lastPriceHigh, secondLastPriceHigh);
        FindSwingHighs(histogram, histSize, minBars, lastHistHigh, secondLastHistHigh);
        
        if(lastPriceHigh.index == -1 || secondLastPriceHigh.index == -1 ||
           lastHistHigh.index == -1 || secondLastHistHigh.index == -1) {
            return false;
        }

        if(lastPriceHigh.value > secondLastPriceHigh.value && 
           lastHistHigh.value < secondLastHistHigh.value &&
           lastHistHigh.value > DBL_EPSILON && secondLastHistHigh.value > DBL_EPSILON) 
        {
            // Questi controlli dovrebbero essere meno problematici ora
            bool priceSwingsAreRecent = (lastPriceHigh.index >= 0 && secondLastPriceHigh.index < pricesSize);
            bool histSwingsAreRecent = (lastHistHigh.index >= 0 && secondLastHistHigh.index < histSize);

            if (!priceSwingsAreRecent || !histSwingsAreRecent) return false;
            
            double priceChange = MathAbs(lastPriceHigh.value - secondLastPriceHigh.value);
            double histChange = MathAbs(secondLastHistHigh.value - lastHistHigh.value); 
            
            double priceRatio = (secondLastPriceHigh.value != 0) ? priceChange / MathAbs(secondLastPriceHigh.value) : 0.0;
            double histRatio = (secondLastHistHigh.value != 0) ? histChange / MathAbs(secondLastHistHigh.value) : 0.0;
            
            return (priceRatio > minRatio && histRatio > minRatio);
        }
        
        return false;
    }
    
    // MODIFICA LA FIRMA: Ora accetta due parametri di dimensione
    bool CheckBullishDivergence(const double &lowPrices[], const double &histogram[], 
                                int pricesSize, int histSize, // NUOVI PARAMETRI
                                int minBars, double minRatio) {
        SwingPoint lastPriceLow, secondLastPriceLow;
        SwingPoint lastHistLow, secondLastHistLow;
        
        // Passa la dimensione corretta a FindSwingLows
        FindSwingLows(lowPrices, pricesSize, minBars, lastPriceLow, secondLastPriceLow);
        FindSwingLows(histogram, histSize, minBars, lastHistLow, secondLastHistLow);
        
        if(lastPriceLow.index == -1 || secondLastPriceLow.index == -1 || 
           lastHistLow.index == -1 || secondLastHistLow.index == -1) {
            return false;
        }
        
        if(lastPriceLow.value < secondLastPriceLow.value && 
           lastHistLow.value > secondLastHistLow.value &&
           lastHistLow.value < -DBL_EPSILON && secondLastHistLow.value < -DBL_EPSILON) 
        {
            bool priceSwingsAreRecent = (lastPriceLow.index >= 0 && secondLastPriceLow.index < pricesSize);
            bool histSwingsAreRecent = (lastHistLow.index >= 0 && secondLastHistLow.index < histSize);

            if (!priceSwingsAreRecent || !histSwingsAreRecent) return false;

            double priceChange = MathAbs(secondLastPriceLow.value - lastPriceLow.value);
            double histChange = MathAbs(lastHistLow.value - secondLastHistLow.value); 
            
            double priceRatio = (secondLastPriceLow.value != 0) ? priceChange / MathAbs(secondLastPriceLow.value) : 0.0;
            double histRatio = (secondLastHistLow.value != 0) ? histChange / MathAbs(secondLastHistLow.value) : 0.0;

            return (priceRatio > minRatio && histRatio > minRatio);
        }
        
        return false;
    }
};

//+------------------------------------------------------------------+
//| Funzioni di supporto ottimizzate                                 |
//+------------------------------------------------------------------+
bool LoadMACDData(string symbol, ENUM_TIMEFRAMES tf, int bars, 
                  MACDData &data, double &high[], double &low[], 
                  string &reason, 
                  int &out_availablePricesBars, 
                  int &out_availableIndicatorBars) 
{
    // Inizializzazione e reset della struct MACDData e delle variabili di output
    data.Reset(); 
    out_availablePricesBars = 0; 
    out_availableIndicatorBars = 0; 
    reason = ""; 

    data.SetSize(bars); 
    if (!data.IsValid()) {
        reason = "❌ Errore: Impossibile ridimensionare gli array interni di MACDData. Memoria insufficiente.";
        if (EnableLogging_MACD) Print(reason);
        return false; // Fallimento critico
    }
    
    int macdHandle = macdCache.GetMACDHandle(symbol, tf, MACD_FastPeriod, 
                                             MACD_SlowPeriod, MACD_SignalPeriod);
    
    if (macdHandle == INVALID_HANDLE) {
        reason = StringFormat("❌ Handle MACD non valido dalla cache per %s %s. Errore: %d.",
                               symbol, EnumToString(tf), GetLastError());
        if (EnableLogging_MACD) Print(reason);
        return false; // Fallimento critico
    } else {
        if (EnableLogging_MACD) PrintFormat("DEBUG: Handle MACD %d ottenuto correttamente per %s %s.",
                                            macdHandle, EnumToString(tf));
    }
    
    if (bars <= 0) {
        reason = "❌ Numero di barre richiesto non valido (<= 0).";
        if (EnableLogging_MACD) Print(reason);
        return false; // Fallimento critico
    }

    // --- Copia dei dati MACD/Signal/Hist dai buffer dell'indicatore ---
    int copiedMacd   = CopyBuffer(macdHandle, 0, 0, bars, data.macdLine);
    int copiedSignal = CopyBuffer(macdHandle, 1, 0, bars, data.signalLine);
    int copiedHist   = CopyBuffer(macdHandle, 2, 0, bars, data.histogram); // Mantenuto per informazioni di debug

    // Determina il numero minimo di barre copiate per le linee MACD/Signal.
    // Questo è il criterio CRITICO per il successo della nostra logica.
    out_availableIndicatorBars = MathMin(copiedMacd, copiedSignal); 

    if (EnableLogging_MACD) {
        PrintFormat("DEBUG: CopyBuffer risultati - MACD: %d (Err: %d), Signal: %d (Err: %d), Hist: %d (Err: %d)",
                    copiedMacd, (copiedMacd == -1 ? GetLastError() : 0),
                    copiedSignal, (copiedSignal == -1 ? GetLastError() : 0),
                    copiedHist, (copiedHist == -1 ? GetLastError() : 0));
        PrintFormat("DEBUG: Barre MACD/Signal disponibili (per calcolo Hist): %d (richieste: %d)",
                    out_availableIndicatorBars, bars);
    }

    // --- Controllo Cruciale: I dati MACD e Signal sono disponibili? (Criterio di successo primario) ---
    if (copiedMacd < bars || copiedSignal < bars || copiedMacd == -1 || copiedSignal == -1) { 
        // Se una delle linee MACD o Signal non è stata copiata completamente, o ha fallito del tutto.
        reason = StringFormat("❌ Errore critico: Dati MACD o Signal insufficienti/non copiati. Richiesti: %d. Copiati MACD: %d (Err: %d), Signal: %d (Err: %d).",
                                bars, copiedMacd, (copiedMacd == -1 ? GetLastError() : 0),
                                copiedSignal, (copiedSignal == -1 ? GetLastError() : 0));
        if (EnableLogging_MACD) Print(reason);
        
        // Riempie le barre non copiate con 0.0 per sicurezza, anche se stiamo per uscire.
        // Il loop è sicuro perché out_availableIndicatorBars sarà >= 0 o 0.
        for (int i = out_availableIndicatorBars > 0 ? out_availableIndicatorBars : 0; i < bars; i++) { 
            data.macdLine[i] = 0.0;
            data.signalLine[i] = 0.0;
            data.histogram[i] = 0.0; // L'istogramma verrà ricalcolato comunque
        }
        return false; // Fallimento critico: non si può procedere senza dati MACD/Signal affidabili.
    }
    
    // --- Log dello stato di copia dell'Istogramma (ora non è un fallimento critico) ---
    if (copiedHist == -1) {
        if (EnableLogging_MACD) { // <-- AGGIUNTA
            reason += StringFormat("⚠️ Attenzione: Copia Istogramma fallita (Errore: %d). Sarà calcolato manualmente. ", GetLastError());
        }
    } else if (copiedHist < bars) {
        if (EnableLogging_MACD) { // <-- AGGIUNTA
            reason += StringFormat("⚠️ Attenzione: Solo %d valori Istogramma copiati. Sarà completato manualmente. ", copiedHist);
        }
    }

    data.CalculateHistogram(); 
    if (EnableLogging_MACD) { // <-- AGGIUNTA
        reason += "✅ Istogramma calcolato manualmente (o completato). ";
    }
    
    // --- DEBUG: Copia dei dati prezzi High/Low ---
    int retHigh = CopyHigh(symbol, tf, 0, bars, high);
    int retLow  = CopyLow(symbol, tf, 0, bars, low);
    
    out_availablePricesBars = MathMin(retHigh, retLow);

    if (EnableLogging_MACD) {
        PrintFormat("DEBUG: CopyHigh/Low risultati - High: %d (Err: %d), Low: %d (Err: %d)",
                    retHigh, (retHigh == -1 ? GetLastError() : 0),
                    retLow, (retLow == -1 ? GetLastError() : 0));
        PrintFormat("DEBUG: Barre prezzi disponibili: %d (richieste: %d)",
                    out_availablePricesBars, bars);
    }

    // Gestione dei dati prezzo insufficienti
    if (out_availablePricesBars < bars) {
        if (EnableLogging_MACD) { // <-- AGGIUNTA
            reason += StringFormat("❌ Dati prezzi High/Low insufficienti (%d/%d barre copiate). ", out_availablePricesBars, bars);
        }
        for (int i = out_availablePricesBars; i < bars; i++) {
            high[i] = 0.0;
            low[i] = 0.0;
        }
    }
    
    // Log di riepilogo finale
    // Controllo se 'reason' contiene messaggi (lunghezza > 0) o se i prezzi sono insufficienti
    if (EnableLogging_MACD && (StringLen(reason) > 0 || out_availablePricesBars < bars)) { // Modificato 'StringIsEmpty' a 'StringLen(reason) > 0'
        PrintFormat("DEBUG: LoadMACDData Riepilogo - Motivo: '%s'. MACD/Signal copiati: %d, High/Low copiati: %d (richiesti %d barre)",
                    reason, out_availableIndicatorBars, out_availablePricesBars, bars);
    }
    
    return true; // La funzione ritorna true se le linee MACD/Signal sono state copiate con successo, anche se Hist o i dati prezzo hanno avuto problemi.
}

void AnalyzeMACDCross(const MACDData &data, bool directionEMA, 
                      MACDScoreComponents &scores, string &reason)
{
    // Determina stato corrente e cross
    bool currentMACDBuy = (data.macdLine[0] > data.signalLine[0]);
    bool macdCrossedUp = currentMACDBuy && (data.macdLine[1] <= data.signalLine[1]);
    bool macdCrossedDown = !currentMACDBuy && (data.macdLine[1] >= data.signalLine[1]);
    
    // Score per cross (2 punti max)
    if ((macdCrossedUp && directionEMA) || (macdCrossedDown && !directionEMA)) {
        scores.crossScore = 2.0;
        if (EnableLogging_MACD) { // <-- AGGIUNTA
            reason += directionEMA ? "✅ Cross Up (+2). " : "✅ Cross Down (+2). ";
        }
    }
    else if ((macdCrossedUp && !directionEMA) || (macdCrossedDown && directionEMA)) {
        scores.penalties += 2.0;
        if (EnableLogging_MACD) { // <-- AGGIUNTA
            reason += "❌ Cross contro-trend (-2). ";
        }
    }
    
    // Score per coerenza direzionale (2 punti max)
    bool coherent = (directionEMA && currentMACDBuy) || (!directionEMA && !currentMACDBuy);
    if (coherent) {
        scores.coherenceScore = 2.0;
        if (EnableLogging_MACD) { // <-- AGGIUNTA
            reason += "✅ Direzione coerente (+2). ";
        }
    } else {
        scores.penalties += 1.0;
        if (EnableLogging_MACD) { // <-- AGGIUNTA
            reason += "❌ Direzione incoerente (-1). ";
        }
    }
    
    // Score per posizione zero line (2 punti max)
    bool aboveZero = (data.macdLine[0] > 0 && data.signalLine[0] > 0);
    bool belowZero = (data.macdLine[0] < 0 && data.signalLine[0] < 0);

    if ((directionEMA && aboveZero) || (!directionEMA && belowZero)) {
        scores.zeroLineScore = 2.0;
        if (EnableLogging_MACD) { // <-- AGGIUNTA
            reason += "✅ Zero line OK (+2). ";
        }
    }
    else if ((directionEMA && belowZero) || (!directionEMA && aboveZero)) {
        scores.penalties += 1.0;
        if (EnableLogging_MACD) { // <-- AGGIUNTA
            reason += "❌ Zero line contro trend (-1). ";
        }
    }
}

void ApplyMomentumScoring(const MomentumAnalysis &momentum, 
                          MACDScoreComponents &scores, string &reason)
{
    // Score momentum (2 punti max)
    if (momentum.slopeInTrendDirection) {
        double absSlope = MathAbs(momentum.currentSlope);

        if (absSlope >= MACD_StrongPositiveSlopeThreshold) {
            scores.momentumScore = 2.0;
            if (EnableLogging_MACD) { // <-- AGGIUNTA
                reason += StringFormat("🚀 Momentum forte (+2, slope=%.5f). ", momentum.currentSlope);
            }
        }
        // ... (stessa logica per Momentum medio) ...
        else if (absSlope >= MACD_WeakPositiveSlopeThreshold) {
            scores.momentumScore = 1.0;
            if (EnableLogging_MACD) { // <-- AGGIUNTA
                reason += StringFormat("📈 Momentum medio (+1, slope=%.5f). ", momentum.currentSlope);
            }
        }

        // Bonus/penalità per accelerazione/decelerazione
        if (momentum.isAccelerating) {
            scores.momentumScore = MathMin(2.0, scores.momentumScore + 0.5);
            if (EnableLogging_MACD) { // <-- AGGIUNTA
                reason += "⚡ Accelerazione (+0.5). ";
            }
        }
        else if (momentum.isDecelerating) {
            double penalty = MACD_MaxSlopePenalty * momentum.decelerationFactor;
            scores.penalties += penalty;
            if (EnableLogging_MACD) { // <-- AGGIUNTA
                reason += StringFormat("🛑 Decelerazione (-%0.2f). ", penalty);
            }
        }
    }
    else {
        scores.penalties += 1.0;
        if (EnableLogging_MACD) { // <-- AGGIUNTA
            reason += "❌ Momentum contro trend (-1). ";
        }
    }
}

void AnalyzeSignalDistance(const MACDData &data, 
                           MACDScoreComponents &scores, string &reason)
{
    double distance = MathAbs(data.macdLine[0] - data.signalLine[0]);
    
    // Score distanza (2 punti max)
    if (distance > MACD_MinDistanceForBonus) {
        double normalizedDist = (distance - MACD_MinDistanceForBonus) / 
                               (MACD_MaxDistanceForBonus - MACD_MinDistanceForBonus);
        scores.distanceScore = MathMin(2.0, normalizedDist * 2.0);
        if (EnableLogging_MACD) { // <-- AGGIUNTA
            reason += StringFormat("💪 Distanza MACD-Signal (+%.1f). ", scores.distanceScore);
        }
    }
}

void CheckDivergences(const double &highPrices[], const double &lowPrices[], 
                      const double &histogram[], 
                      int pricesSize, int histSize, // Modifica: Due dimensioni separate
                      bool directionEMA, MACDScoreComponents &scores, string &reason)
{
    // Importante: SwingPointDetector richiede un numero minimo di barre per funzionare.
    // Il MACD_DivergenceLookbackBars è la base per la ricerca delle divergenze.
    // Se non ci sono abbastanza barre nei dati effettivamente disponibili, non possiamo cercare divergenze.
    if (pricesSize < MACD_DivergenceLookbackBars || histSize < MACD_DivergenceLookbackBars) {
        if (EnableLogging_MACD) { // <-- AGGIUNTA
            reason += StringFormat("⚠️ Barre insufficienti per il controllo divergenze (Prezzi:%d, Hist:%d, Richiesto:%d). ",
                                   pricesSize, histSize, MACD_DivergenceLookbackBars);
        }
        return;
    }
    
    SwingPointDetector detector;
    bool divergenceFound = false;
    
    // Per il trend BUY (cercando un segnale di vendita, quindi divergenza ribassista)
    if (directionEMA) { 
        // Cerca divergenza ribassista (prezzo fa Higher High, istogramma fa Lower High)
        // Passa highPrices e la sua dimensione (pricesSize), e histogram con la sua dimensione (histSize)
        divergenceFound = detector.CheckBearishDivergence(highPrices, histogram, 
                                                           pricesSize, histSize, // Passa le dimensioni corrette
                                                           MACD_MinBarsBetweenSwings, 
                                                           MACD_MinDivergenceRatio);
        if (divergenceFound) {
            scores.penalties += MACD_PriceDivergencePenalty;
            if (EnableLogging_MACD) { // <-- AGGIUNTA
                reason += StringFormat("❌ Divergenza ribassista (Prezzo: HH, MACD Hist: LH) (-%.1f). ", MACD_PriceDivergencePenalty);
            }
        }
    }
    // Per il trend SELL (cercando un segnale di acquisto, quindi divergenza rialzista)
    else { 
        // Cerca divergenza rialzista (prezzo fa Lower Low, istogramma fa Higher Low)
        // Passa lowPrices e la sua dimensione (pricesSize), e histogram con la sua dimensione (histSize)
        divergenceFound = detector.CheckBullishDivergence(lowPrices, histogram, 
                                                          pricesSize, histSize, // Passa le dimensioni corrette
                                                          MACD_MinBarsBetweenSwings, 
                                                          MACD_MinDivergenceRatio);
        if (divergenceFound) {
            scores.penalties += MACD_PriceDivergencePenalty;
            if (EnableLogging_MACD) { // <-- AGGIUNTA
                reason += StringFormat("❌ Divergenza rialzista (Prezzo: LL, MACD Hist: HL) (-%.1f). ", MACD_PriceDivergencePenalty);
            }
        }
    }
}

void ApplyFinalChecks(const MACDData &data, double currentATR, 
                      MACDScoreComponents &scores, string &reason)
{
    // Penalità per prossimità alla linea zero (usando ATR per normalizzazione)
    // Assicurati che currentATR sia > 0 per evitare divisioni per zero.
    double normalizedThreshold = 0.0;
    if (currentATR > 0) {
        normalizedThreshold = currentATR * MACD_ZeroLineThresholdRatio;
    }
    
    if (normalizedThreshold > 0 && MathAbs(data.histogram[0]) < normalizedThreshold) {
        scores.penalties += MACD_ZeroLinePenalty;
        if (EnableLogging_MACD) { // <-- AGGIUNTA
            reason += StringFormat("⚠️ MACD vicino a zero (-%.1f). ", MACD_ZeroLinePenalty);
        }
    }
}

void LogMACDAnalysis(string symbol, ENUM_TIMEFRAMES tf, bool directionEMA,
                     const MACDData &data, const MomentumAnalysis &momentum,
                     const MACDScoreComponents &scores)
{
    Print("═══════════════ [MACD Score Evaluation - Optimized] ═══════════════");
    PrintFormat("📍 %s | %s | Direzione desiderata: %s", 
                symbol, EnumToString(tf), directionEMA ? "BUY" : "SELL");
    
    Print("📊 Valori MACD:");
    PrintFormat("  MACD:    %.5f → %.5f → %.5f", data.macdLine[2], data.macdLine[1], data.macdLine[0]);
    PrintFormat("  Signal: %.5f → %.5f → %.5f", data.signalLine[2], data.signalLine[1], data.signalLine[0]);
    PrintFormat("  Hist:    %.5f → %.5f → %.5f %s", 
                data.histogram[2], data.histogram[1], data.histogram[0],
                data.histogramCalculated ? "(calc)" : "(copied)"); // cambiato per chiarezza
    
    Print("📈 Analisi Momentum:");
    PrintFormat("  Pendenza attuale: %.5f | Precedente: %.5f", 
                momentum.currentSlope, momentum.previousSlope);
    PrintFormat("  Status: %s | Decel Factor: %.1f%%",
                momentum.isAccelerating ? "🚀 ACCEL" : 
                momentum.isDecelerating ? "🛑 DECEL" : "〰️ STAB",
                momentum.decelerationFactor * 100);
    
    Print("──── Componenti Score ────");
    PrintFormat("  Cross:      %.1f", scores.crossScore);
    PrintFormat("  Coherence:  %.1f", scores.coherenceScore);
    PrintFormat("  Zero Line:  %.1f", scores.zeroLineScore);
    PrintFormat("  Momentum:   %.1f", scores.momentumScore);
    PrintFormat("  Distance:   %.1f", scores.distanceScore);
    PrintFormat("  Penalties: -%.1f", scores.penalties);
    PrintFormat("  TOTALE:     %.1f/10", scores.GetTotal());
    Print("═══════════════════════════════════════════════════════════════════");
}


//+------------------------------------------------------------------+
//| BLOCCO 2: 📊 EvalMACDScore - Versione Ottimizzata                |
//+------------------------------------------------------------------+
double EvalMACDScore(string symbol, ENUM_TIMEFRAMES tf, bool directionEMA, 
                     string &reasonMACD, double currentATR)
{
    // === 🔄 Inizializzazione ===
    reasonMACD = "";
    MACDScoreComponents scores;
    scores.Reset();
    
    // 🔕 Controllo attivazione
    if (!EnableMACDCheck) {
        reasonMACD = "⚪️ MACD disattivato";
        return 0.0;
    }
    
    // === 📦 Preparazione dati ===
    // Calcola barre necessarie in modo efficiente
    const int barsForAnalysis = 3; // Per l'analisi base servono solo 3 barre (current, prev, prev-prev)
    // MACD_DivergenceLookbackBars viene da un input.
    // Aggiungo un piccolo buffer (2 barre) oltre il lookback per le divergenze
    // Questo è il numero massimo di barre che tenteremo di copiare per prezzi/istogramma
    const int barsForDivergence = MACD_DivergenceLookbackBars + 2; 
    int totalBars = MathMax(barsForAnalysis, barsForDivergence);
    
    MACDData macdData;
    
    double highPrices[], lowPrices[]; 

    if (!ArrayResize(highPrices, totalBars) || !ArrayResize(lowPrices, totalBars)) {
        reasonMACD = "❌ Errore: impossibile ridimensionare gli array dei prezzi.";
        return 0.0;
    }

    int actualPricesBars = 0;    // Variabile per il conteggio effettivo delle barre di prezzo
    int actualIndicatorBars = 0; // Variabile per il conteggio effettivo delle barre dell'indicatore
    
    // Passa le nuove variabili per ottenere i conteggi effettivi
    if (!LoadMACDData(symbol, tf, totalBars, macdData, highPrices, lowPrices, 
                     reasonMACD, actualPricesBars, actualIndicatorBars)) {
        return 0.0;
    }
    
    // === 🎯 Analisi direzione e cross ===
    AnalyzeMACDCross(macdData, directionEMA, scores, reasonMACD);
    
    // Early exit se score base è troppo negativo
    if (scores.GetTotal() <= -2.0) { // Puoi regolare questa soglia
        reasonMACD += "⛔ Score base MACD troppo negativo → analisi interrotta. ";
        return 0.0;
    }
    
    // === 📊 Analisi momentum ===
    MomentumAnalysis momentum;
    momentum.Calculate(macdData.histogram, directionEMA);
    ApplyMomentumScoring(momentum, scores, reasonMACD);
    
    // === 💪 Analisi distanza MACD-Signal ===
    AnalyzeSignalDistance(macdData, scores, reasonMACD);
    
    // === 🔍 Analisi divergenze (solo se necessario e con dati sufficienti) ===
    // Passa i conteggi effettivi delle barre a CheckDivergences
    if (MACD_EnableDivergenceCheck && actualPricesBars >= barsForDivergence && actualIndicatorBars >= barsForDivergence) {
        // MODIFICA QUI: Usa actualPricesBars e actualIndicatorBars come dimensione
        CheckDivergences(highPrices, lowPrices, macdData.histogram, 
                         actualPricesBars, actualIndicatorBars, // Passa i conteggi effettivi!
                         directionEMA, scores, reasonMACD);
    } else if (MACD_EnableDivergenceCheck) {
        reasonMACD += StringFormat("⚠️ Divergenze disattivate: Barre prezzi insufficienti (%d/%d) o Barre indicatore insufficienti (%d/%d) o check disabilitato. ", 
                                  actualPricesBars, barsForDivergence, actualIndicatorBars, barsForDivergence);
    }
    
    // === ⚠️ Controlli finali ===
    ApplyFinalChecks(macdData, currentATR, scores, reasonMACD);
    
    // === 📋 Logging dettagliato ===
    if (EnableLogging_MACD) {
        LogMACDAnalysis(symbol, tf, directionEMA, macdData, momentum, scores);
    }
    
    double finalScore = scores.GetTotal();
    reasonMACD += StringFormat("🎯 Score MACD finale: %.2f/10.00", finalScore);
    
    return finalScore;
}

//+------------------------------------------------------------------+
//| BLOCCO 3: 📊 BodyMomentumScore V2                                |
//| **AGGIORNATO:** Implementate le nuove logiche per l'analisi delle ombre: |
//| - Bonus per candele tipo Marubozu (ombre ridotte)                |
//| - Penalità per ombre di rifiuto contro il trend                  |
//| - Bonus per ombre favorevoli al trend                            |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Strutture di supporto per Body Momentum ottimizzato             |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Strutture di supporto per Body Momentum ottimizzato             |
//+------------------------------------------------------------------+
struct BodyMomentumConfig {
    double volumeConfirmationThreshold;
    double bodyATRRatioThreshold;
    double stdDevATRRatioThreshold;
    double weightCoherence;
    double weightDominance;
    double weightShadow;
    double weightProgression;
    double weightClose;
    double weightVolume;
    
    // Costruttore con valori di default
    BodyMomentumConfig() {
        volumeConfirmationThreshold = 1.2;
        bodyATRRatioThreshold = 0.3;
        stdDevATRRatioThreshold = 0.1;
        weightCoherence = 1.0;
        weightDominance = 1.0;
        weightShadow = 1.0;
        weightProgression = 1.0;
        weightClose = 1.0;
        weightVolume = 1.0;
    }
};

struct BodyMomentumResult {
    double score;
    int rawScore;
    double mediaBody;
    int coherentCandles;
    int dominantBodies;
    int shadowAsymmetryCandles;
    double posInRangeAvg;
    string detailedLog;
    
    void Reset() {
        score = 0.0;
        rawScore = 0;
        mediaBody = 0.0;
        coherentCandles = 0;
        dominantBodies = 0;
        shadowAsymmetryCandles = 0;
        posInRangeAvg = 0.0;
        detailedLog = "";
    }
};

class CandleData {
public:
    double open, close, high, low, volume;
    double body, range, upperShadow, lowerShadow;
    double bodyToRange;
    double closePosition; // Posizione della chiusura nel range (0-1)
    
    // Copia costruttore per CandleData
    CandleData() {}
    CandleData(const CandleData &other) {
        open = other.open;
        close = other.close;
        high = other.high;
        low = other.low;
        volume = other.volume;
        body = other.body;
        range = other.range;
        upperShadow = other.upperShadow;
        lowerShadow = other.lowerShadow;
        bodyToRange = other.bodyToRange;
        closePosition = other.closePosition;
    }
    
    void Calculate() {
        body = MathAbs(close - open);
        range = high - low;
        upperShadow = high - MathMax(open, close);
        lowerShadow = MathMin(open, close) - low;
        bodyToRange = (range > 0) ? body / range : 0.0;
        closePosition = (range > 0) ? (close - low) / range : 0.5;
    }
    
    bool IsBullish() const { return close > open; }
    bool IsBearish() const { return close < open; }
};

class CandleAnalyzer {
private:
    CandleData candles[];
    int candleCount;
    double avgBody;
    double avgRange;
    double avgVolume;
    double stdDevBody;
    double stdDevVolume;
    
public:
    bool LoadData(string symbol, ENUM_TIMEFRAMES tf, int count) {
        candleCount = count;
        ArrayResize(candles, count);
        
        // Carica tutti i dati in un solo passaggio
        for(int i = 0; i < count; i++) {
            int shift = i + 1;
            candles[i].open = iOpen(symbol, tf, shift);
            candles[i].close = iClose(symbol, tf, shift);
            candles[i].high = iHigh(symbol, tf, shift);
            candles[i].low = iLow(symbol, tf, shift);
            candles[i].volume = (double)iVolume(symbol, tf, shift);
            candles[i].Calculate();
        }
        
        CalculateStatistics();
        return true;
    }
    
    void CalculateStatistics() {
        avgBody = 0.0;
        avgRange = 0.0;
        avgVolume = 0.0;
        
        // Calcola medie
        for(int i = 0; i < candleCount; i++) {
            avgBody += candles[i].body;
            avgRange += candles[i].range;
            avgVolume += candles[i].volume;
        }
        avgBody /= candleCount;
        avgRange /= candleCount;
        avgVolume /= candleCount;
        
        // Calcola deviazioni standard
        double sumSqBody = 0.0;
        double sumSqVolume = 0.0;
        for(int i = 0; i < candleCount; i++) {
            sumSqBody += MathPow(candles[i].body - avgBody, 2);
            sumSqVolume += MathPow(candles[i].volume - avgVolume, 2);
        }
        stdDevBody = MathSqrt(sumSqBody / candleCount);
        stdDevVolume = MathSqrt(sumSqVolume / candleCount);
    }
    
    // Getters
    CandleData GetCandle(int index) const { 
        return (index >= 0 && index < candleCount) ? candles[index] : candles[0]; 
    }
    double GetAvgBody() const { return avgBody; }
    double GetAvgRange() const { return avgRange; }
    double GetAvgVolume() const { return avgVolume; }
    double GetStdDevBody() const { return stdDevBody; }
    double GetStdDevVolume() const { return stdDevVolume; }
    int GetCount() const { return candleCount; }
    
    // Analisi direzionale
    int CountCoherentCandles(bool isBuy) const {
        int count = 0;
        for(int i = 0; i < candleCount; i++) {
            bool coherent = (isBuy && candles[i].IsBullish()) || 
                          (!isBuy && candles[i].IsBearish());
            if(coherent) count++;
        }
        return count;
    }
    
    // Analisi progressione
    int CountProgressionCandles(bool isBuy) const {
        int count = 0;
        for(int i = candleCount - 1; i > 0; i--) {
            bool progressing = (isBuy && candles[i].close > candles[i-1].close) ||
                             (!isBuy && candles[i].close < candles[i-1].close);
            if(progressing) count++;
        }
        return count;
    }
    
    // Posizione media chiusura
    double GetAvgClosePosition() const {
        double sum = 0.0;
        for(int i = 0; i < candleCount; i++) {
            sum += candles[i].closePosition;
        }
        return sum / candleCount;
    }
};

class ShadowAnalyzer {
public:
    struct ShadowResult {
        double score;
        string reason;
    };
    
public:
    static ShadowResult AnalyzeShadows(const CandleData &candle, bool isBuy, double currentATR) {
        ShadowResult result;
        result.score = 0.0;
        result.reason = "";
        
        if(candle.range <= 0 || currentATR <= 0) {
            result.reason = "⚠️ Dati ombra non validi. ";
            return result;
        }
        
        double bodyNormalized = candle.body / currentATR;
        double totalShadow = candle.upperShadow + candle.lowerShadow;
        double shadowToRange = totalShadow / candle.range;
        
        // Check Marubozu
        if(IsMarubozu(candle, bodyNormalized, shadowToRange)) {
            result.score += BODY_MarubozuBonus;
            result.reason += StringFormat("✔️ Marubozu (+%.2f, Ombre %.1f%%). ", 
                                        BODY_MarubozuBonus, shadowToRange * 100);
        }
        // Check Rejection Shadow
        else if(HasRejectionShadow(candle, isBuy)) {
            double penalty = BODY_RejectionShadowPenalty;
            result.score -= penalty;
            result.reason += StringFormat("❌ Ombra Rifiuto (-%.2f). ", penalty);
        }
        // Check Favorable Shadow
        else if(HasFavorableShadow(candle, isBuy)) {
            result.score += BODY_FavorableShadowBonus;
            result.reason += StringFormat("✔️ Ombre Favorevoli (+%.2f). ", 
                                        BODY_FavorableShadowBonus);
        }
        
        return result;
    }
    
private:
    static bool IsMarubozu(const CandleData &candle, double bodyNormalized, double shadowToRange) {
        return bodyNormalized >= BODY_MarubozuMinBodyATRRatio && 
               shadowToRange <= BODY_MarubozuShadowRatioThreshold;
    }
    
    static bool HasRejectionShadow(const CandleData &candle, bool isBuy) {
        double rejectionRatio = isBuy ? 
            candle.upperShadow / candle.range : 
            candle.lowerShadow / candle.range;
        return rejectionRatio >= BODY_RejectionShadowRatioThreshold;
    }
    
    static bool HasFavorableShadow(const CandleData &candle, bool isBuy) {
        if(isBuy) {
            return candle.lowerShadow > candle.upperShadow && 
                   candle.upperShadow / candle.range <= BODY_FavorableShadowRatioThreshold;
        } else {
            return candle.upperShadow > candle.lowerShadow && 
                   candle.lowerShadow / candle.range <= BODY_FavorableShadowRatioThreshold;
        }
    }
};

//+------------------------------------------------------------------+
//| Funzione principale ottimizzata                                  |
//+------------------------------------------------------------------+
// Costruttore di default con valori predefiniti
double CalcBodyMomentumScoreV2(string symbol, ENUM_TIMEFRAMES tf, bool isBuy,
                              BodyMomentumResult &result,
                              double currentATR,
                              const BodyMomentumConfig &config)
    {
      
    // Reset risultati
    result.Reset();
    
    // Controllo attivazione
    if (!EnableBodyMomentumScore) {
        result.detailedLog = "⚪️ BodyMomentum disattivato";
        return 0.0;
    }
    
    // === 📊 Carica e analizza dati candele ===
    CandleAnalyzer analyzer;
    if (!analyzer.LoadData(symbol, tf, NumCandlesBodyAnalysis)) {
        result.detailedLog = "❌ Errore caricamento dati candele";
        return 0.0;
    }
    
    // Popola risultati base
    result.mediaBody = analyzer.GetAvgBody();
    result.coherentCandles = analyzer.CountCoherentCandles(isBuy);
    result.posInRangeAvg = analyzer.GetAvgClosePosition();
    
    // === 🎯 Calcolo componenti score ===
    BodyMomentumScoreComponents scores;
    
    // 1. Coerenza direzione (max 2.0 * weight)
    scores.coherence = (2.0 * result.coherentCandles / analyzer.GetCount()) * config.weightCoherence;
    
    // 2. Dominanza corpo (max 2.0 * weight)
    double domRatio = (analyzer.GetAvgRange() > 0) ? 
                     analyzer.GetAvgBody() / analyzer.GetAvgRange() : 0.0;
    scores.dominance = MathMin(2.0, 10.0 * domRatio) * config.weightDominance;
    
    // 3. Progressione chiusure (max 1.0 * weight)
    int progressionCount = analyzer.CountProgressionCandles(isBuy);
    scores.progression = (progressionCount >= 2 ? 1.0 : 0.0) * config.weightProgression;
    
    // 4. Chiusura forte (max 1.0 * weight)
    bool strongClose = (isBuy && result.posInRangeAvg > 0.75) || 
                      (!isBuy && result.posInRangeAvg < 0.25);
    scores.closeStrength = (strongClose ? 1.0 : 0.0) * config.weightClose;
    
    // 5. Volume (max 1.0 * weight)
    scores.volume = CalculateVolumeScore(analyzer, config.volumeConfirmationThreshold) * 
                   config.weightVolume;
    
    // 6. Analisi ombre per candela corrente
    CandleData currentCandle = analyzer.GetCandle(0);
    ShadowAnalyzer::ShadowResult shadowResult = ShadowAnalyzer::AnalyzeShadows(currentCandle, isBuy, currentATR);
    scores.shadow = shadowResult.score;
    
    // === ⚠️ Calcolo penalità ===
    CalculatePenalties(analyzer, currentATR, config, scores);
    
    // === 📊 Score finale ===
    result.score = scores.GetTotal();
    result.rawScore = (int)MathRound(result.score);
    
    // === 📋 Costruisci log dettagliato ===
    if (EnableLogging_BodyMomentum) {
        BuildDetailedLog(symbol, tf, isBuy, analyzer, scores, shadowResult, 
                        currentATR, config, result);
    }
    
    return result.score;
}

//+------------------------------------------------------------------+
//| Funzioni di supporto                                             |
//+------------------------------------------------------------------+
struct BodyMomentumScoreComponents {
    double coherence;
    double dominance;
    double progression;
    double closeStrength;
    double volume;
    double shadow;
    double penalties;
    
    double GetTotal() const {
        double total = coherence + dominance + progression + 
                      closeStrength + volume + shadow - penalties;
        return MathMax(0.0, MathMin(10.0, total));
    }
};

double CalculateVolumeScore(const CandleAnalyzer &analyzer, double threshold) {
    CandleData current = analyzer.GetCandle(0);
    double avgVolume = analyzer.GetAvgVolume();
    
    if (avgVolume <= 0) return 0.0;
    
    if (current.volume > avgVolume * threshold) {
        // Check progressione volume
        bool volumeProgressing = true;
        for(int i = 0; i < MathMin(3, analyzer.GetCount()-1); i++) {
            if(analyzer.GetCandle(i).volume <= analyzer.GetCandle(i+1).volume) {
                volumeProgressing = false;
                break;
            }
        }
        
        if (volumeProgressing || analyzer.GetStdDevVolume() > avgVolume * 0.2) {
            return 1.0;
        } else {
            return 0.5;
        }
    }
    
    return 0.0;
}

void CalculatePenalties(const CandleAnalyzer &analyzer, double currentATR,
                       const BodyMomentumConfig &config, BodyMomentumScoreComponents &scores) {
    scores.penalties = 0.0;
    
    // Penalità corpo medio basso
    if (UseMinAvgBodyCheck && currentATR > 0) {
        double minBody = currentATR * config.bodyATRRatioThreshold;
        if (analyzer.GetAvgBody() < minBody) {
            scores.penalties += 1.0;
        }
    }
    
    // Penalità deviazione standard bassa
    if (UseMinStdDevBodyCheck && currentATR > 0) {
        double minStdDev = currentATR * config.stdDevATRRatioThreshold;
        if (analyzer.GetStdDevBody() < minStdDev) {
            scores.penalties += 1.0;
        }
    }
}

void BuildDetailedLog(string symbol, ENUM_TIMEFRAMES tf, bool isBuy,
                     const CandleAnalyzer &analyzer, const BodyMomentumScoreComponents &scores,
                     const ShadowAnalyzer::ShadowResult &shadowResult,
                     double currentATR, const BodyMomentumConfig &config,
                     BodyMomentumResult &result) {
    Print("═══════════════ [Body Momentum V2 - Optimized] ═══════════════");
    PrintFormat("📍 %s | %s | Direzione: %s", symbol, EnumToString(tf), isBuy ? "BUY" : "SELL");
    PrintFormat("📊 Score Totale: %.2f/10 (Raw: %d)", result.score, result.rawScore);
    
    Print("──── Componenti Score ────");
    PrintFormat("  Coerenza:     %.2f (peso %.1f)", scores.coherence, config.weightCoherence);
    PrintFormat("  Dominanza:    %.2f (peso %.1f)", scores.dominance, config.weightDominance);
    PrintFormat("  Progressione: %.2f (peso %.1f)", scores.progression, config.weightProgression);
    PrintFormat("  Chiusura:     %.2f (peso %.1f)", scores.closeStrength, config.weightClose);
    PrintFormat("  Volume:       %.2f (peso %.1f)", scores.volume, config.weightVolume);
    PrintFormat("  Ombre:        %.2f | %s", scores.shadow, shadowResult.reason);
    PrintFormat("  Penalità:    -%.2f", scores.penalties);
    
    Print("──── Statistiche ────");
    PrintFormat("  Media Corpo: %.5f | StdDev: %.5f", analyzer.GetAvgBody(), analyzer.GetStdDevBody());
    PrintFormat("  Media Range: %.5f | ATR: %.5f", analyzer.GetAvgRange(), currentATR);
    PrintFormat("  Media Volume: %.2f | StdDev: %.2f", analyzer.GetAvgVolume(), analyzer.GetStdDevVolume());
    
    CandleData current = analyzer.GetCandle(0);
    PrintFormat("  Candela Corrente: O:%.5f C:%.5f H:%.5f L:%.5f", 
                current.open, current.close, current.high, current.low);
    Print("═══════════════════════════════════════════════════════════════");
}

//+------------------------------------------------------------------+
//| Wrapper per mantenere compatibilità con vecchia firma            |
//+------------------------------------------------------------------+
double CalcBodyMomentumScoreV2_Legacy(string symbol, ENUM_TIMEFRAMES tf, bool isBuy,
                                     double &mediaBody, int &coherentCandles,
                                     int &dominantBodies, int &shadowAsymmetryCandles,
                                     double &posInRangeAvg, int &rawScore,
                                     double currentATR,
                                     double volumeConfirmationThreshold,
                                     double bodyATRRatioThreshold,
                                     double stdDevATRRatioThreshold,
                                     double weightCoherence,
                                     double weightDominance,
                                     double weightShadow,
                                     double weightProgression,
                                     double weightClose,
                                     double weightVolume)
{
    // Crea configurazione
    BodyMomentumConfig config;
    config.volumeConfirmationThreshold = volumeConfirmationThreshold;
    config.bodyATRRatioThreshold = bodyATRRatioThreshold;
    config.stdDevATRRatioThreshold = stdDevATRRatioThreshold;
    config.weightCoherence = weightCoherence;
    config.weightDominance = weightDominance;
    config.weightShadow = weightShadow;
    config.weightProgression = weightProgression;
    config.weightClose = weightClose;
    config.weightVolume = weightVolume;
    
    // Chiama versione ottimizzata
    BodyMomentumResult result;
    double score = CalcBodyMomentumScoreV2(symbol, tf, isBuy, result, currentATR, config);
    
    // Popola parametri di output
    mediaBody = result.mediaBody;
    coherentCandles = result.coherentCandles;
    dominantBodies = result.dominantBodies;
    shadowAsymmetryCandles = result.shadowAsymmetryCandles;
    posInRangeAvg = result.posInRangeAvg;
    rawScore = result.rawScore;
    
    return score;
}

#endif  // __EMA_MACD_BODYSCORE_MQH__
