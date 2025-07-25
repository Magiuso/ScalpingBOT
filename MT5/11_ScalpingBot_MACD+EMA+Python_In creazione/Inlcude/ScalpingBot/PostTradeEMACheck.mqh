//+------------------------------------------------------------------+
//| OpenTradeMACDEMA.mqh - Recovery Trigger Manager INTEGRATO       |
//| Sostituzione completa di CheckPostTradeConditions()             |
//| Rileva inversioni avanzate, chiude posizioni e crea trigger     |
//| INTEGRAZIONE TOTALE con RecoveryTriggerManager esistente        |
//+------------------------------------------------------------------+
#ifndef __OPENTRADE_MACDEMA_INTEGRATED_MQH__
#define __OPENTRADE_MACDEMA_INTEGRATED_MQH__

#include <ScalpingBot\Utility.mqh>
#include <ScalpingBot\RecoveryTriggerManager.mqh>
#include <Trade\Trade.mqh>
#include <Map\MapLong_Bool.mqh>

//+------------------------------------------------------------------+
//| 📋 ENUMERAZIONI E STRUTTURE                                     |
//+------------------------------------------------------------------+

enum RECOVERY_SIGNAL_TYPE {
    RECOVERY_NONE = 0,           // Nessun segnale
    RECOVERY_BUY = 1,            // Segnale BUY recovery
    RECOVERY_SELL = 2,           // Segnale SELL recovery
    RECOVERY_PENDING = 3         // Segnale in attesa conferma
};

enum RECOVERY_REJECTION_REASON {
    REJECTION_NONE = 0,              // Nessuna reiezione
    REJECTION_EMA3_WEAK = 1,         // EMA3 acceleration insufficiente
    REJECTION_RSI_NO_CONFIRM = 2,    // RSI non conferma
    REJECTION_RANGING_MARKET = 3,    // Mercato in ranging
    REJECTION_LOW_ATR = 4,           // ATR troppo basso
    REJECTION_SIGNAL_LOST = 5,       // Segnale perso durante attesa
    REJECTION_NO_PERSISTENCE = 6,    // Segnale non persistente
    REJECTION_LOW_CONFIDENCE = 7     // Confidence score insufficiente
};

struct RecoverySignalData {
    RECOVERY_SIGNAL_TYPE signalType;        // Tipo segnale
    double ema3Acceleration;                 // Valore acceleration EMA3
    double dynamicThreshold;                 // Soglia dinamica usata
    double rsiValue;                         // Valore RSI corrente
    double rsiDerivative;                    // Derivata RSI
    double atrValue;                         // Valore ATR corrente
    double emaProximity;                     // Prossimità EMA9/EMA21
    bool isRangingMarket;                    // Flag mercato ranging
    double confidenceScore;                  // Score confidenza (0-100)
    RECOVERY_REJECTION_REASON rejectionReason; // Motivo reiezione
    datetime signalTime;                     // Timestamp segnale
    string debugInfo;                        // Info debug
    
    // Costruttore
    RecoverySignalData() {
        signalType = RECOVERY_NONE;
        ema3Acceleration = 0.0;
        dynamicThreshold = 0.0;
        rsiValue = 50.0;
        rsiDerivative = 0.0;
        atrValue = 0.0;
        emaProximity = 0.0;
        isRangingMarket = false;
        confidenceScore = 0.0;
        rejectionReason = REJECTION_NONE;
        signalTime = 0;
        debugInfo = "";
    }
};

struct PendingRecoveryState {
    bool isPending;                          // Flag recovery in attesa
    RECOVERY_SIGNAL_TYPE pendingType;        // Tipo recovery atteso
    datetime pendingTime;                    // Timestamp inizio attesa
    RecoverySignalData originalSignal;       // Segnale originale
    string symbol;                           // Simbolo
    ENUM_TIMEFRAMES timeframe;               // Timeframe
    ulong positionTicket;                    // Ticket posizione monitorata
    
    void Reset() {
        isPending = false;
        pendingType = RECOVERY_NONE;
        pendingTime = 0;
        originalSignal = RecoverySignalData();
        symbol = "";
        timeframe = PERIOD_CURRENT;
        positionTicket = 0;
    }
};

//+------------------------------------------------------------------+
//| 🌐 VARIABILI GLOBALI STATE                                      |
//+------------------------------------------------------------------+
static PendingRecoveryState g_pendingRecovery;

//+------------------------------------------------------------------+
//| 🧮 CALCOLO EMA VELOCITY (VELOCITÀ DI CAMBIO)                    |
//+------------------------------------------------------------------+
double CalculateEMA3Velocity(string symbol, ENUM_TIMEFRAMES tf) {
    // Usa FastEMAPeriod configurabile
    int periods[1];
    periods[0] = FastEMAPeriod;
    
    if (!emaCache.GetMultipleEMAData(symbol, tf, periods, 2)) {  // Solo 2 valori
        if (EnableLogging_PostTradeCheck)
            PrintFormat("❌ [PostTrade] Impossibile ottenere dati EMA%d per %s [%s]", FastEMAPeriod, symbol, EnumToString(tf));
        return 0.0;
    }
    
    if (!emaCache.IsEMAAvailable(FastEMAPeriod)) {
        if (EnableLogging_PostTradeCheck)
            PrintFormat("❌ [PostTrade] EMA%d non disponibile per %s [%s]", FastEMAPeriod, symbol, EnumToString(tf));
        return 0.0;
    }
    
    // Ottieni valori EMA: [0]=corrente, [1]=precedente
    double ema_current = emaCache.GetEMAValue(FastEMAPeriod, 0);
    double ema_prev1 = emaCache.GetEMAValue(FastEMAPeriod, 1);
    
    // Calcola velocità: current - previous
    double velocity = ema_current - ema_prev1;
    
    if (EnableDebugMode) {
        Print("│ 🔍 DEBUG EMA VELOCITY:");
        PrintFormat("│     EMA%d Values: Current=%.5f | Previous=%.5f", 
                   FastEMAPeriod, ema_current, ema_prev1);
        PrintFormat("│     Velocity: %.7f | Direction: %s", velocity, velocity > 0 ? "UP" : "DOWN");
    }
    
    return velocity;
}

//+------------------------------------------------------------------+
//| 📊 CALCOLO RSI E DERIVATA                                       |
//+------------------------------------------------------------------+
bool CalculateRSIData(string symbol, ENUM_TIMEFRAMES tf, double &rsiValue, double &rsiDerivative) {
    // Ottieni handle RSI dalla cache
    int rsiHandle = rsiCache.GetRSIHandle(symbol, tf, RSIPeriod);
    if (rsiHandle == INVALID_HANDLE) {
        if (EnableLogging_PostTradeCheck)
            PrintFormat("❌ [PostTrade] RSI handle non disponibile per %s [%s]", symbol, EnumToString(tf));
        return false;
    }
    
    // Copia buffer RSI (ultimi 3 valori per derivata)
    double rsiBuffer[];
    if (!SafeCopyBuffer(rsiHandle, 0, 0, 3, rsiBuffer)) {
        if (EnableLogging_PostTradeCheck)
            PrintFormat("❌ [PostTrade] Impossibile copiare buffer RSI per %s [%s]", symbol, EnumToString(tf));
        return false;
    }
    
    rsiValue = rsiBuffer[0];  // Valore corrente
    
    // Calcola derivata RSI (semplice: corrente - precedente)
    if (ArraySize(rsiBuffer) >= 2) {
        rsiDerivative = rsiBuffer[1] - rsiBuffer[0];
    } else {
        rsiDerivative = 0.0;
    }
    
    if (EnableDebugMode) {
        Print("│ 🔍 DEBUG RSI ANALYSIS:");
        PrintFormat("│     RSI Value: %.2f | RSI Derivative: %.4f", rsiValue, rsiDerivative);
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| 🎯 CALCOLO SOGLIA DINAMICA                                      |
//+------------------------------------------------------------------+
double CalculateDynamicThreshold(string symbol, ENUM_TIMEFRAMES tf) {
    if (!UseAdaptiveThreshold) {
        return EMA3AccelerationThreshold;  // Usa soglia fissa
    }
    
    // Calcola soglia adattiva basata su ATR
    int atrPeriods[1] = {14};
    double atrResults[];
    
    if (!CalculateMultiPeriodATR(symbol, tf, atrPeriods, atrResults)) {
        if (EnableLogging_PostTradeCheck)
            PrintFormat("⚠️ [PostTrade] ATR non disponibile, uso soglia fissa per %s [%s]", symbol, EnumToString(tf));
        return EMA3AccelerationThreshold;
    }
    
    double atr = atrResults[0];
    double dynamicThreshold = atr * EMA3_ATR_Multiplier;
    
    if (EnableDebugMode) {
        PrintFormat("🔍 [DEBUG] Dynamic Threshold: ATR=%.5f * Multiplier=%.2f = %.7f (vs Fixed=%.7f)", 
                   atr, EMA3_ATR_Multiplier, dynamicThreshold, EMA3AccelerationThreshold);
    }
    
    return dynamicThreshold;
}

//+------------------------------------------------------------------+
//| 🛡️ RILEVAMENTO MERCATO RANGING                                  |
//+------------------------------------------------------------------+
bool IsMarketRanging(string symbol, ENUM_TIMEFRAMES tf, double &atrValue, double &emaProximity) {
    if (!EnableAntiRangingFilter) {
        atrValue = 0.0;
        emaProximity = 0.0;
        return false;  // Filtro disabilitato
    }
    
    // 1. Controllo ATR
    int atrPeriods[1] = {14};
    double atrResults[];
    
    if (CalculateMultiPeriodATR(symbol, tf, atrPeriods, atrResults)) {
        atrValue = atrResults[0];
                 
        // Calcola media ATR degli ultimi 20 periodi
        int atrHandle = iATR(symbol, tf, 14);
        double atrBuffer[20];
        double atrAverage = atrValue; // Fallback
        if (atrHandle != INVALID_HANDLE && SafeCopyBuffer(atrHandle, 0, 0, 20, atrBuffer)) {
            double sum = 0.0;
            for (int i = 0; i < 20; i++) sum += atrBuffer[i];
            atrAverage = sum / 20.0;
        }
        
        double atrMultiplier = atrValue / atrAverage;
        if (atrMultiplier < MinATRThreshold) {
            if (EnableDebugMode) {
                PrintFormat("🔍 [DEBUG] Low ATR detected: %.5f (multiplier: %.2f < %.2f)", 
                           atrValue, atrMultiplier, MinATRThreshold);
            }
            return true;  // ATR troppo basso = ranging
        }
    } else {
        atrValue = 0.0;
    }
    
    // 2. Controllo prossimità EMA9/EMA21
    int emaPeriods[2] = {9, 21};
    
    if (emaCache.GetMultipleEMAData(symbol, tf, emaPeriods, 2)) {
        double ema9 = emaCache.GetEMAValue(9, 0);
        double ema21 = emaCache.GetEMAValue(21, 0);
        
        if (ema21 > 0) {
            double avgPrice = (ema9 + ema21) / 2.0;
            emaProximity = MathAbs(ema9 - ema21) / avgPrice * 100.0;
            
        if (emaProximity < MaxEMAProximity) {
            if (EnableDebugMode) {
                PrintFormat("│ 🔍 EMAs too close: EMA9=%.5f, EMA21=%.5f, Proximity=%.4f%% < %.2f%%", 
                           ema9, ema21, emaProximity, MaxEMAProximity);
            }
            return true;  // EMA troppo vicine = ranging
        }
      }
    } else {
        emaProximity = 0.0;
    }
    
    return false;  // Non in ranging
}

//+------------------------------------------------------------------+
//| ✅ CONFERMA RSI                                                 |
//+------------------------------------------------------------------+
bool ConfirmRSISignal(double rsiValue, double rsiDerivative, bool recoveryIsBuySignal) {
    if (!EnableRSIConfirmation) {
        return true;  // Conferma RSI disabilitata
    }
    
    bool rsiConfirms = false;
    
    if (recoveryIsBuySignal) {
        // Per BUY recovery: RSI in ipervenduto O derivata positiva forte
        rsiConfirms = (rsiValue < RSIBullishLevel) || 
                     (rsiDerivative > RSIDerivativeThreshold);
    } else {
        // Per SELL recovery: RSI in ipercomprato O derivata negativa forte
        rsiConfirms = (rsiValue > RSIBearishLevel) || 
                     (rsiDerivative < -RSIDerivativeThreshold);
    }
    
    if (EnableDebugMode) {
        Print("│ 🔍 DEBUG RSI CONFIRMATION:");
        PrintFormat("│     Direction: %s | RSI: %.2f | Derivative: %.4f | Confirms: %s", 
                   recoveryIsBuySignal ? "BUY" : "SELL", rsiValue, rsiDerivative, rsiConfirms ? "YES" : "NO");
        PrintFormat("│     Logic: %s", 
                   recoveryIsBuySignal ? 
                   StringFormat("RSI<%.0f (%.2f) OR Deriv>%.1f (%.4f)", RSIBullishLevel, rsiValue, RSIDerivativeThreshold, rsiDerivative) :
                   StringFormat("RSI>%.0f (%.2f) OR Deriv<%.1f (%.4f)", RSIBearishLevel, rsiValue, -RSIDerivativeThreshold, rsiDerivative));
    }
    
    return rsiConfirms;
}

//+------------------------------------------------------------------+
//| 📊 CALCOLO SCORE CONFIDENZA (CONFIGURABILE TOTALE)             |
//+------------------------------------------------------------------+
double CalculateConfidenceScore(double ema3Acceleration, double dynamicThreshold,
                               double rsiValue, double rsiDerivative, bool recoveryIsBuySignal,
                               bool isRanging, double atrValue) {
    
    // Se scoring disabilitato, ritorna sempre 100 (passa tutti i test)
    if (!UseConfidenceScoring) {
        return 100.0;
    }
    
    double score = 0.0;
    
    // 1. Score EMA3 Acceleration (peso configurabile)
    if (EMA3_Weight > 0) {
        double accelerationRatio = MathAbs(ema3Acceleration) / dynamicThreshold;
        
        if (UseAdvancedScoring) {
            // Formula avanzata: crescita logaritmica per valori estremi
            double ema3Score = MathMin(EMA3_Weight, MathLog10(1 + accelerationRatio * 9) * EMA3_Weight);
            score += ema3Score;
        } else {
            // Formula semplice: lineare
            double ema3Score = MathMin(EMA3_Weight, accelerationRatio * (EMA3_Weight / 2.0));
            score += ema3Score;
        }
    }
    
    // 2. Score RSI (peso configurabile)
    if (RSI_Weight > 0 && EnableRSIConfirmation) {
        double rsiScore = 0.0;
        
        if (recoveryIsBuySignal) {
            // Più basso RSI = più punti per BUY
            double rsiDistance = MathMax(0.0, RSIBullishLevel - rsiValue);
            rsiScore += (rsiDistance / RSIBullishLevel) * (RSI_Weight * 0.7);
            
            // Derivata positiva = bonus
            if (rsiDerivative > RSIDerivativeThreshold) {
                rsiScore += (RSI_Weight * 0.3);
            }
        } else {
            // Più alto RSI = più punti per SELL
            double rsiDistance = MathMax(0.0, rsiValue - RSIBearishLevel);
            rsiScore += (rsiDistance / (100.0 - RSIBearishLevel)) * (RSI_Weight * 0.7);
            
            // Derivata negativa = bonus
            if (rsiDerivative < -RSIDerivativeThreshold) {
                rsiScore += (RSI_Weight * 0.3);
            }
        }
        
        score += MathMin(RSI_Weight, rsiScore);
    } else if (RSI_Weight > 0 && !EnableRSIConfirmation) {
        // Se RSI disabilitato ma peso > 0, assegna punteggio neutro
        score += (RSI_Weight * 0.5);  // 50% del peso se RSI non richiesto
    }
    
    // 3. Penalità ranging (configurabile)
    if (isRanging && Ranging_Penalty > 0) {
        score -= Ranging_Penalty;
    }
    
    // 4. Bonus ATR (peso configurabile)
    if (ATR_Bonus_Weight > 0 && atrValue > 0) {
        double atrScore = 0.0;
        
        if (UseAdvancedScoring) {
            // Formula avanzata: normalizza ATR su scala 0-1 poi applica peso
            double normalizedATR = MathMin(1.0, atrValue * 5000.0);  // Scala ATR
            atrScore = normalizedATR * ATR_Bonus_Weight;
        } else {
            // Formula semplice
            atrScore = MathMin(ATR_Bonus_Weight, atrValue * 10000.0);
        }
        
        score += atrScore;
    }
    
    // Applica moltiplicatore finale
    score *= ScoreMultiplier;
    
    // Normalizza 0-100
    score = MathMax(0.0, MathMin(100.0, score));
    
    return score;
}

//+------------------------------------------------------------------+
//| 🎯 RILEVAMENTO SEGNALE RECOVERY PRINCIPALE                      |
//+------------------------------------------------------------------+
RecoverySignalData DetectRecoverySignal(string symbol, ENUM_TIMEFRAMES tf, bool currentPositionIsBuy) {
    RecoverySignalData signal;
    signal.signalTime = TimeCurrent();
    
    // 1. Calcola EMA Acceleration (periodo configurabile)
    signal.ema3Acceleration = CalculateEMA3Velocity(symbol, tf);
    signal.dynamicThreshold = CalculateDynamicThreshold(symbol, tf);
    
    // 2. Determina direzione segnale basata su posizione corrente
    bool signalForBuy = !currentPositionIsBuy;  // Recovery opposto alla posizione corrente
    
    // 3. Verifica soglia EMA3 CON DIREZIONE CORRETTA
    bool ema3Triggered = false;
    
    if (signalForBuy) {
        // BUY recovery: acceleration deve essere POSITIVA (trend up)
        ema3Triggered = (signal.ema3Acceleration > 0) && 
                       (MathAbs(signal.ema3Acceleration) >= signal.dynamicThreshold);
    } else {
        // SELL recovery: acceleration deve essere NEGATIVA (trend down)  
        ema3Triggered = (signal.ema3Acceleration < 0) && 
                       (MathAbs(signal.ema3Acceleration) >= signal.dynamicThreshold);
    }
    
    if (EnableDebugMode) {
       PrintFormat("│ 🎯 EMA3 Direction Check: %s recovery needs %s velocity | Got: %.7f | Valid: %s",
                  signalForBuy ? "BUY" : "SELL",
                  signalForBuy ? "POSITIVE" : "NEGATIVE", 
                  signal.ema3Acceleration,
                  ema3Triggered ? "YES" : "NO");
    }
    
    if (!ema3Triggered) {
        signal.rejectionReason = REJECTION_EMA3_WEAK;
        signal.debugInfo = StringFormat("EMA3 direction wrong or weak: %.7f (need %s %.7f)", 
                                       signal.ema3Acceleration,
                                       signalForBuy ? ">" : "<", 
                                       signalForBuy ? signal.dynamicThreshold : -signal.dynamicThreshold);
        return signal;
    }
    
    // 4. Calcola dati RSI
    if (!CalculateRSIData(symbol, tf, signal.rsiValue, signal.rsiDerivative)) {
        signal.rejectionReason = REJECTION_RSI_NO_CONFIRM;
        signal.debugInfo = "RSI data unavailable";
        return signal;
    }
    
    // 5. Conferma RSI
    if (!ConfirmRSISignal(signal.rsiValue, signal.rsiDerivative, signalForBuy)) {
        signal.rejectionReason = REJECTION_RSI_NO_CONFIRM;
        signal.debugInfo = StringFormat("RSI no confirm: RSI=%.2f, Derivative=%.4f", 
                                       signal.rsiValue, signal.rsiDerivative);
        return signal;
    }
    
    // 6. Verifica ranging
    signal.isRangingMarket = IsMarketRanging(symbol, tf, signal.atrValue, signal.emaProximity);
    if (signal.isRangingMarket) {
        signal.rejectionReason = REJECTION_RANGING_MARKET;
        signal.debugInfo = StringFormat("Ranging market: ATR=%.5f, EMA_Prox=%.2f%%", 
                                       signal.atrValue, signal.emaProximity);
        return signal;
    }
    
    // 7. Calcola confidence score
    signal.confidenceScore = CalculateConfidenceScore(signal.ema3Acceleration, signal.dynamicThreshold,
                                                     signal.rsiValue, signal.rsiDerivative, signalForBuy,
                                                     signal.isRangingMarket, signal.atrValue);
    
    // DEBUG: Mostra dettagli confidence score
    if (EnableDebugMode) {
        // Calcola i contributi parziali (semplificato)
        double ema3Ratio = MathAbs(signal.ema3Acceleration) / signal.dynamicThreshold;
        double ema3Contribution = MathMin(EMA3_Weight, ema3Ratio * (EMA3_Weight / 2.0)); // Formula semplificata
        
        // RSI contribution (per SELL - corretto)
        double rsiContribution = 0.0;
        if (!signalForBuy) { // SELL recovery
            // Score per RSI value (70%)
            double rsiDistance = MathMax(0.0, signal.rsiValue - RSIBearishLevel);
            rsiContribution += (rsiDistance / (100.0 - RSIBearishLevel)) * (RSI_Weight * 0.7);
            
            // Bonus per derivata negativa (30%)
            if (signal.rsiDerivative < -RSIDerivativeThreshold) {
                rsiContribution += (RSI_Weight * 0.3);
            }
        }
        rsiContribution = MathMin(RSI_Weight, rsiContribution);
       
        // ATR contribution (approssimativo)
        double atrContribution = MathMin(ATR_Bonus_Weight, signal.atrValue * 10000.0);
        
        // Ranging penalty
        double rangingPenalty = signal.isRangingMarket ? Ranging_Penalty : 0.0;
        
        PrintFormat("│ 🧮 Score Breakdown: EMA3=%.1f/%d | RSI=%.1f/%d | ATR=%.1f/%d | Ranging=-%.1f | Total=%.1f%% (%.1f%% needed)", 
                   ema3Contribution, (int)EMA3_Weight,
                   rsiContribution, (int)RSI_Weight, 
                   atrContribution, (int)ATR_Bonus_Weight,
                   rangingPenalty,
                   signal.confidenceScore, MinConfidenceScore);
    }
    
    // 8. Verifica soglia confidence (solo se scoring abilitato)
    if (UseConfidenceScoring && signal.confidenceScore < MinConfidenceScore) {
        signal.rejectionReason = REJECTION_LOW_CONFIDENCE;
        signal.debugInfo = StringFormat("Low confidence: %.1f%% < %.1f%%", 
                                       signal.confidenceScore, MinConfidenceScore);
        return signal;
    }
    
    // 9. Segnale confermato
    signal.signalType = signalForBuy ? RECOVERY_BUY : RECOVERY_SELL;
    
    if (UseConfidenceScoring) {
        signal.debugInfo = StringFormat("Signal confirmed with %.1f%% confidence (threshold: %.1f%%)", 
                                       signal.confidenceScore, MinConfidenceScore);
    } else {
        signal.debugInfo = "Signal confirmed (scoring disabled)";
    }
    
    return signal;
}

//+------------------------------------------------------------------+
//| ⏰ GESTIONE DELAYED CONFIRMATION                                 |
//+------------------------------------------------------------------+
void StartDelayedConfirmation(RecoverySignalData &signal, string symbol, ENUM_TIMEFRAMES tf, ulong ticket) {
    g_pendingRecovery.isPending = true;
    g_pendingRecovery.pendingType = signal.signalType;
    g_pendingRecovery.pendingTime = TimeCurrent();
    g_pendingRecovery.originalSignal = signal;
    g_pendingRecovery.symbol = symbol;
    g_pendingRecovery.timeframe = tf;
    g_pendingRecovery.positionTicket = ticket;
    
    // Cambia stato segnale a pending
    signal.signalType = RECOVERY_PENDING;
    
    if (EnableLogging_PostTradeCheck) {
        PrintFormat("⏰ [PostTrade] Delayed confirmation started for %s recovery on %s [%s] (Ticket: %d)", 
                   g_pendingRecovery.pendingType == RECOVERY_BUY ? "BUY" : "SELL", 
                   symbol, EnumToString(tf), ticket);
    }
}

//+------------------------------------------------------------------+
//| 🔄 VERIFICA DELAYED CONFIRMATION                                |
//+------------------------------------------------------------------+
RecoverySignalData CheckDelayedConfirmation(string symbol, ENUM_TIMEFRAMES tf, bool currentPositionIsBuy, ulong ticket) {
    RecoverySignalData result;
    
    if (!g_pendingRecovery.isPending || g_pendingRecovery.positionTicket != ticket) {
        return result;  // Nessuna conferma pending per questo ticket
    }
    
    // Verifica se è passata almeno 1 candela
    datetime currentTime = TimeCurrent();
    datetime nextCandleTime = g_pendingRecovery.pendingTime + PeriodSeconds(tf);
    
    if (currentTime < nextCandleTime) {
        result.signalType = RECOVERY_PENDING;
        return result;  // Ancora in attesa
    }
    
    // È passata 1 candela, verifica se segnale è ancora valido
    RecoverySignalData newSignal = DetectRecoverySignal(symbol, tf, currentPositionIsBuy);
    
    if (newSignal.signalType == g_pendingRecovery.pendingType) {
        // Segnale confermato
        if (EnableLogging_PostTradeCheck) {
            PrintFormat("✅ [PostTrade] Delayed confirmation SUCCESS for %s recovery (Ticket: %d)", 
                       newSignal.signalType == RECOVERY_BUY ? "BUY" : "SELL", ticket);
        }
        
        g_pendingRecovery.Reset();
        return newSignal;
    } else {
        // Segnale perso
        if (EnableLogging_PostTradeCheck) {
            PrintFormat("❌ [PostTrade] Delayed confirmation FAILED - signal lost for %s recovery (Ticket: %d)", 
                       g_pendingRecovery.pendingType == RECOVERY_BUY ? "BUY" : "SELL", ticket);
        }
        
        result.rejectionReason = REJECTION_SIGNAL_LOST;
        result.debugInfo = "Signal lost during confirmation wait";
        g_pendingRecovery.Reset();
        return result;
    }
}

//+------------------------------------------------------------------+
//| 🚀 FUNZIONE PRINCIPALE: ANALIZZA RECOVERY TRIGGER               |
//+------------------------------------------------------------------+
RecoverySignalData AnalyzeRecoveryTrigger(string symbol, ENUM_TIMEFRAMES tf, bool currentPositionIsBuy, ulong ticket) {
    // 1. Verifica se c'è una conferma delayed in corso per questo ticket
    if (EnableDelayedConfirmation && g_pendingRecovery.isPending && g_pendingRecovery.positionTicket == ticket) {
        return CheckDelayedConfirmation(symbol, tf, currentPositionIsBuy, ticket);
    }
    
    // 2. Rileva nuovo segnale
    RecoverySignalData signal = DetectRecoverySignal(symbol, tf, currentPositionIsBuy);
    
    // 3. Se segnale valido e delayed confirmation abilitata → start delay
    if (signal.signalType != RECOVERY_NONE && EnableDelayedConfirmation) {
        StartDelayedConfirmation(signal, symbol, tf, ticket);
    }
    
    // 4. Logging con timeframe dinamico
    if (EnableLogging_PostTradeCheck) {
        if (signal.signalType != RECOVERY_NONE) {
            Print("│ 📊 SIGNAL ANALYSIS RESULTS:");
            PrintFormat("│     Signal Type: %s", EnumToString(signal.signalType));
            PrintFormat("│     Confidence: %.1f%% (threshold: %.1f%%)", signal.confidenceScore, MinConfidenceScore);
            PrintFormat("│     EMA%d Velocity: %.7f (|%.7f| vs %.7f)", 
                       FastEMAPeriod, signal.ema3Acceleration, MathAbs(signal.ema3Acceleration), signal.dynamicThreshold);
            PrintFormat("│     RSI: %.2f (derivative: %.4f)", signal.rsiValue, signal.rsiDerivative);
            PrintFormat("│     Info: %s", signal.debugInfo);
            if (emaCache.IsEMAAvailable(9) && emaCache.IsEMAAvailable(21)) {
                double ema9 = emaCache.GetEMAValue(9, 0);
                double ema21 = emaCache.GetEMAValue(21, 0);
                PrintFormat("│     EMA9: %.5f | EMA21: %.5f", ema9, ema21);
            } 
         
            } else if (signal.rejectionReason != REJECTION_NONE) {
                Print("│ ❌ SIGNAL REJECTED:");
                PrintFormat("│     Reason: %s", RecoveryRejectionReasonToString(signal.rejectionReason));
                PrintFormat("│     Details: %s", signal.debugInfo);
                
                // AGGIUNGI QUESTO BLOCCO PER VEDERE EMA ANCHE NEI REJECTION
                if (emaCache.IsEMAAvailable(9) && emaCache.IsEMAAvailable(21)) {
                    double ema9 = emaCache.GetEMAValue(9, 0);
                    double ema21 = emaCache.GetEMAValue(21, 0);
                    PrintFormat("│     EMA9: %.5f | EMA21: %.5f", ema9, ema21);
                }
            }
    }
    
    return signal;
}

//+------------------------------------------------------------------+
//| ❌ CHIUSURA POSIZIONE - Chiude posizione corrente               |
//+------------------------------------------------------------------+
bool CloseCurrentPosition(ulong ticket, string symbol) {
    if (ticket == 0) {
        if (EnableLogging_PostTradeCheck) {
            PrintFormat("❌ [PostTrade] Ticket non valido per chiusura %s", symbol);
        }
        return false;
    }
    
    // Seleziona posizione per ticket
    if (!PositionSelectByTicket(ticket)) {
        if (EnableLogging_PostTradeCheck) {
            PrintFormat("❌ [PostTrade] Impossibile selezionare posizione con ticket %d", ticket);
        }
        return false;
    }
    
    // Usa oggetto CTrade per chiusura
    trade.SetExpertMagicNumber(MagicNumber_MACD);
    
    bool result = trade.PositionClose(ticket);
    
    if (result) {
        if (EnableLogging_PostTradeCheck) {
            PrintFormat("✅ [PostTrade] Posizione chiusa: %s (Ticket: %d)", symbol, ticket);
        }
    } else {
        if (EnableLogging_PostTradeCheck) {
            PrintFormat("❌ [PostTrade] Errore chiusura posizione %s (Ticket: %d): %s", symbol, ticket, trade.ResultComment());
        }
    }
    
    return result;
}

//+------------------------------------------------------------------+
//| 🎯 MONITORAGGIO SINGOLA POSIZIONE                               |
//+------------------------------------------------------------------+
bool MonitorSinglePosition(int idx) {
    // Seleziona posizione per indice usando PositionGetSymbol
    string symbol = PositionGetSymbol(idx);
    if (symbol == "") {
        return false; // Posizione non valida
    }
    
    // Ottieni dati posizione
    ulong ticket = PositionGetInteger(POSITION_TICKET);
    ulong magic = PositionGetInteger(POSITION_MAGIC);
    ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
    double volume = PositionGetDouble(POSITION_VOLUME);
    
    // LOG INIZIO ANALISI POSIZIONE
    if (EnableLogging_PostTradeCheck) {
        Print("┌─────────────────────────────────────────────────────────────────┐");
        PrintFormat("│ 🎯 ANALYZING POSITION #%d", idx + 1);
        PrintFormat("│ 📋 Ticket: %d | Symbol: %s | Type: %s | Volume: %.2f", 
                   ticket, symbol, posType == POSITION_TYPE_BUY ? "BUY" : "SELL", volume);
        Print("├─────────────────────────────────────────────────────────────────┤");
    }
    
    // Filtra solo posizioni del bot MACD
    if (magic != MagicNumber_MACD) {
        if (EnableLogging_PostTradeCheck) {
            PrintFormat("│ ⚠️  SKIP: Wrong magic number (%d ≠ %d)", magic, MagicNumber_MACD);
            Print("└─────────────────────────────────────────────────────────────────┘");
        }
        return false;
    }
    
    // Controlla se ticket è in ban recovery
    bool isBanned = false;
    if (!g_recovery_ban_map.Get(ticket, isBanned)) {
        isBanned = false; // Se non trovato, assume non bannato
    }
    
    if (isBanned) {
        if (EnableLogging_PostTradeCheck) {
            PrintFormat("│ 🚫 SKIP: Ticket %d in recovery ban", ticket);
            Print("└─────────────────────────────────────────────────────────────────┘");
        }
        return false;
    }
    
    // Controlla se già esiste trigger per questo ticket
    if (g_recovery_manager.ContainsTrigger(ticket)) {
        if (EnableLogging_PostTradeCheck) {
            PrintFormat("│ ℹ️  SKIP: Ticket %d already has recovery trigger", ticket);
            Print("└─────────────────────────────────────────────────────────────────┘");
        }
        return false;
    }
    
    bool currentPositionIsBuy = (posType == POSITION_TYPE_BUY);
    
    if (EnableLogging_PostTradeCheck) {
        PrintFormat("│ 🧠 Analyzing %s position for %s recovery...", 
                   currentPositionIsBuy ? "BUY" : "SELL",
                   currentPositionIsBuy ? "SELL" : "BUY");
    }
    
    // Usa il modulo migliorato per detection
    RecoverySignalData recovery = AnalyzeRecoveryTrigger(symbol, Timeframe_MACD_EMA, currentPositionIsBuy, ticket);
    
    if (recovery.signalType == RECOVERY_BUY || recovery.signalType == RECOVERY_SELL) {
        
        if (EnableLogging_PostTradeCheck) {
            Print("│ ✅ RECOVERY SIGNAL CONFIRMED!");
            PrintFormat("│ 🎯 Direction: %s | Confidence: %.1f%%", 
                       EnumToString(recovery.signalType), recovery.confidenceScore);
        }
        
        // Chiudi la posizione corrente
        if (CloseCurrentPosition(ticket, symbol)) {
            
            // Determina tipo originale per recovery
            ENUM_POSITION_TYPE originalType = posType;
            
            // Registra trigger nel sistema recovery
            bool triggerAdded = g_recovery_manager.AddOrUpdateTrigger(
                ticket, 
                symbol, 
                originalType,
                volume
            );
            
            if (triggerAdded) {
                if (EnableLogging_PostTradeCheck) {
                    Print("│ 🔄 RECOVERY TRIGGER CREATED SUCCESSFULLY!");
                    PrintFormat("│ 📤 %s → %s (Vol: %.2f, Conf: %.1f%%)",
                               originalType == POSITION_TYPE_BUY ? "BUY" : "SELL",
                               EnumToString(recovery.signalType), volume, recovery.confidenceScore);
                    Print("└─────────────────────────────────────────────────────────────────┘");
                }
                return true;
            } else {
                if (EnableLogging_PostTradeCheck) {
                    PrintFormat("│ ❌ ERROR: Failed to create recovery trigger for ticket %d", ticket);
                    Print("└─────────────────────────────────────────────────────────────────┘");
                }
            }
            
        } else {
            if (EnableLogging_PostTradeCheck) {
                PrintFormat("│ ❌ ERROR: Failed to close position %s (Ticket: %d)", symbol, ticket);
                Print("└─────────────────────────────────────────────────────────────────┘");
            }
        }
        
    } else if (recovery.signalType == RECOVERY_PENDING) {
        // Segnale in attesa di conferma
        if (EnableLogging_PostTradeCheck) {
            Print("│ ⏳ RECOVERY PENDING - waiting for confirmation");
            Print("└─────────────────────────────────────────────────────────────────┘");
        }
    } else {
        // Segnale rifiutato
        if (EnableLogging_PostTradeCheck) {
            PrintFormat("│ ❌ RECOVERY REJECTED: %s", recovery.debugInfo);
            Print("└─────────────────────────────────────────────────────────────────┘");
        }
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| 🔄 FUNZIONE PRINCIPALE: SOSTITUZIONE CheckPostTradeConditions   |
//+------------------------------------------------------------------+
void CheckPostTradeConditions() {
    // Verifica se modulo è abilitato
    if (!EnablePostTradeEMAExit) {
        return;
    }
    
    static datetime lastCheck = 0;
    
    // Ottimizzazione: controlla solo su nuova candela
    datetime currentBarTime = iTime(_Symbol, Timeframe_MACD_EMA, 0);
    if (currentBarTime == lastCheck) {
        return; // Stessa candela, non ricontrollare
    }
    lastCheck = currentBarTime;
    
    int totalPositions = PositionsTotal();
    int processedPositions = 0;
    int triggersCreated = 0;
    
    if (EnableDebugMode && totalPositions > 0) {
        Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        PrintFormat("🔍 [PostTrade] === CHECKING %d POSITIONS FOR RECOVERY ===", totalPositions);
        Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
    
    // Loop attraverso tutte le posizioni aperte (backwards per sicurezza)
    for (int i = totalPositions - 1; i >= 0; i--) {
        
        if (MonitorSinglePosition(i)) {
            triggersCreated++;
        }
        processedPositions++;
    }
    
    // Log statistiche finali
    if (EnableLogging_PostTradeCheck && processedPositions > 0) {
        Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        PrintFormat("📊 [PostTrade] === SUMMARY: %d positions processed ===", processedPositions);
        PrintFormat("🎯 Triggers created: %d | Total triggers in system: %d", triggersCreated, g_recovery_manager.TotalTriggers());
        Print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

//+------------------------------------------------------------------+
//| 🚨 UTILITY: Get Rejection Reason String                         |
//+------------------------------------------------------------------+
string RecoveryRejectionReasonToString(RECOVERY_REJECTION_REASON reason) {
    switch(reason) {
        case REJECTION_NONE: return "NONE";
        case REJECTION_EMA3_WEAK: return "EMA3_WEAK";
        case REJECTION_RSI_NO_CONFIRM: return "RSI_NO_CONFIRM";
        case REJECTION_RANGING_MARKET: return "RANGING_MARKET";
        case REJECTION_LOW_ATR: return "LOW_ATR";
        case REJECTION_SIGNAL_LOST: return "SIGNAL_LOST";
        case REJECTION_NO_PERSISTENCE: return "NO_PERSISTENCE";
        case REJECTION_LOW_CONFIDENCE: return "LOW_CONFIDENCE";
        default: return "UNKNOWN";
    }
}

//+------------------------------------------------------------------+
//| 🧹 UTILITY: Reset Pending Recovery (per restart EA)             |
//+------------------------------------------------------------------+
void ResetPendingRecoveryState() {
    g_pendingRecovery.Reset();
    
    if (EnableLogging_PostTradeCheck) {
        Print("🔄 [PostTrade] Pending recovery state reset");
    }
}

//+------------------------------------------------------------------+
//| 📊 UTILITY: Statistiche sistema recovery                        |
//+------------------------------------------------------------------+
void PrintRecoverySystemStats() {
    if (!EnableLogging_PostTradeCheck) return;
    
    Print("📊 ========== RECOVERY SYSTEM STATS ==========");
    PrintFormat("🎯 Module Status: %s", EnablePostTradeEMAExit ? "ACTIVE" : "DISABLED");
    PrintFormat("📈 Fast EMA Period: %d", FastEMAPeriod);
    PrintFormat("⏰ Timeframe: %s", EnumToString(Timeframe_MACD_EMA));
    PrintFormat("🎚️ Threshold Mode: %s (%.7f)", 
               UseAdaptiveThreshold ? "Adaptive" : "Fixed",
               UseAdaptiveThreshold ? EMA3_ATR_Multiplier : EMA3AccelerationThreshold);
    PrintFormat("📊 RSI Confirmation: %s", EnableRSIConfirmation ? "ON" : "OFF");
    PrintFormat("🛡️ Anti-Ranging: %s", EnableAntiRangingFilter ? "ON" : "OFF");
    PrintFormat("⏳ Delayed Confirmation: %s", EnableDelayedConfirmation ? "ON" : "OFF");
    PrintFormat("🧠 Confidence Scoring: %s (Min: %.1f%%)", 
               UseConfidenceScoring ? "ON" : "OFF", MinConfidenceScore);
    PrintFormat("🔢 Total Triggers: %d", g_recovery_manager.TotalTriggers());
    PrintFormat("⏸️ Pending Recovery: %s", g_pendingRecovery.isPending ? "YES" : "NO");
    Print("📊 ==========================================");
}

//+------------------------------------------------------------------+
//| 🔧 UTILITY: Forza reset di un pending specifico                 |
//+------------------------------------------------------------------+
void ForceResetPendingIfTicket(ulong ticket) {
    if (g_pendingRecovery.isPending && g_pendingRecovery.positionTicket == ticket) {
        g_pendingRecovery.Reset();
        
        if (EnableLogging_PostTradeCheck) {
            PrintFormat("🔄 [PostTrade] Forced reset pending recovery for ticket %d", ticket);
        }
    }
}

#endif // __OPENTRADE_MACDEMA_INTEGRATED_MQH__