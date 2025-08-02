//+------------------------------------------------------------------+
//|                             Campionamento.mqh                    |
//|      Modulo avanzato per il campionamento intracandela e scoring |
//|------------------------------------------------------------------|
//| Implementa logica per:                                           |
//| - Rilevamento inizio nuova candela                               |
//| - Campionamento a frequenza controllata                          |
//| - Calcolo dello score progressivo (Velocità, Volume, ADX, Soglie)|
//| - Analisi delle ombre e coerenza storica                         |
//| - Rilevamento autonomo della direzione                           |
//+------------------------------------------------------------------+
#ifndef __CAMPIONAMENTO_MQH__
#define __CAMPIONAMENTO_MQH__

#include <ScalpingBot\Utility.mqh>
#include <Map\CMapStringToCampionamentoState.mqh>

// 🗺️ Mappa per gestire lo stato di campionamento per ogni symbol+timeframe
CMapStringToCampionamentoState M_CampionamentoState;

//+------------------------------------------------------------------+
//| 🔄 OnTickCampionamento_Global() - VERSIONE MIGLIORATA           |
//| Chiamata semplificata da OnTick con validazione dati             |
//+------------------------------------------------------------------+
void OnTickCampionamento_Global()
{
    string currentSymbol = Symbol();
    ENUM_TIMEFRAMES currentTF = _Period;
    
    // 🔥 MIGLIORAMENTO #2: Validazione prezzo con fallback BID/ASK
    double currentPrice = NormalizeDouble(SymbolInfoDouble(currentSymbol, SYMBOL_BID), _Digits);
    if (currentPrice <= 0.0)
    {
        currentPrice = NormalizeDouble(SymbolInfoDouble(currentSymbol, SYMBOL_ASK), _Digits);
        if (currentPrice <= 0.0)
        {
            if (EnableLogging_Campionamento)
                PrintFormat("ERROR GlobalTick - Impossibile ottenere prezzo valido per %s. Skip tick.", currentSymbol);
            return; // Skip questo tick se non riusciamo a ottenere un prezzo
        }
        
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG GlobalTick - BID non disponibile, uso ASK: %.5f", currentPrice);
    }
    
    // 🔥 MIGLIORAMENTO #3: Validazione volume
    long currentTickVolume = iVolume(currentSymbol, currentTF, 0);
    if (currentTickVolume < 0)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG GlobalTick - Volume non valido (%d), uso 1 come fallback.", currentTickVolume);
        currentTickVolume = 1; // Fallback volume minimo
    }
    
    // 🔥 MIGLIORAMENTO #4: Validazione simbolo e timeframe
    if (StringLen(currentSymbol) == 0 || currentTF == PERIOD_CURRENT)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("ERROR GlobalTick - Simbolo o timeframe non validi. Symbol='%s', TF=%s", 
                        currentSymbol, EnumToString(currentTF));
        return;
    }
    
    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG GlobalTick - Dati validati: Symbol=%s, TF=%s, Price=%.5f, Volume=%d",
                    currentSymbol, EnumToString(currentTF), currentPrice, currentTickVolume);
    
    // Chiamata alla funzione principale
    OnTickCampionamento(currentSymbol, currentTF, currentPrice, currentTickVolume);
}

//+------------------------------------------------------------------+
//| 📦 Output del modulo Campionamento per EntryManager - MIGLIORATO |
//+------------------------------------------------------------------+
struct CampionamentoResult
{
    double score;               // Punteggio normalizzato 0-10
    bool   direction;           // Direzione determinata (BUY=true, SELL=false)
    string log;                 // Log dettagliato per il debugging
    
    // 🔥 MIGLIORAMENTO #5: Campi aggiuntivi per analisi avanzata
    double rawScore;            // Score raw prima della normalizzazione
    double confidence;          // Livello di confidenza 0-1 (basato su componenti concordi)
    int    ticksProcessed;      // Numero di tick processati per questa candela
    bool   isDataFresh;         // True se i dati sono stati aggiornati di recente
    long lastUpdate;            // Timestamp ultimo aggiornamento
};

//+------------------------------------------------------------------+
//| 🎯 Funzione principale per EntryManager - VERSIONE MIGLIORATA    |
//+------------------------------------------------------------------+
CampionamentoResult GetCampionamentoData(string symbol, ENUM_TIMEFRAMES tf)
{
    CampionamentoResult res;
    
    // 🔥 MIGLIORAMENTO #6: Inizializzazione completa del result
    res.score = 0.0;
    res.direction = false;
    res.log = "⏳ [Campionamento] Modulo disabilitato o dati non disponibili.";
    res.rawScore = 0.0;
    res.confidence = 0.0;
    res.ticksProcessed = 0;
    res.isDataFresh = false;
    res.lastUpdate = 0;
    
    // 🔥 MIGLIORAMENTO #7: Validazione input parametri
    if (StringLen(symbol) == 0)
    {
        res.log = "❌ [Campionamento] ERRORE: Simbolo non valido.";
        if (EnableLogging_Campionamento)
            PrintFormat("ERROR GetCampionamentoData - Simbolo vuoto o non valido.");
        return res;
    }
    
    if (!EnableCampionamentoModule)
    {
        res.log = "⏸️ [Campionamento] Modulo disabilitato.";
        return res;
    }
    
    string key = symbol + EnumToString(tf);
    CampionamentoState tempState;

    // Tentativo di recupero dello stato
    if (!M_CampionamentoState.Get(key, tempState))
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG GetCampionamentoData - Stato non trovato per %s [%s]. Auto-inizializzazione.", 
                        symbol, EnumToString(tf));
        
        // Auto-inizializzazione
        InitializeCampionamentoState(symbol, tf);
        
        // Secondo tentativo di recupero
        if (!M_CampionamentoState.Get(key, tempState))
        {
            res.log = StringFormat("❌ [Campionamento] ERRORE: Impossibile inizializzare stato per %s [%s].", 
                                   symbol, EnumToString(tf));
            if (EnableLogging_Campionamento)
                PrintFormat("ERROR GetCampionamentoData - Fallimento inizializzazione per %s [%s].", symbol, EnumToString(tf));
            return res;
        }
        
        // Stato appena inizializzato
        res.log = StringFormat("🆕 [Campionamento] Stato inizializzato per %s [%s]. Score: %.2f/10", 
                               symbol, EnumToString(tf), tempState.finalNormalizedScore);
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG GetCampionamentoData - Nuovo stato creato per %s [%s].", symbol, EnumToString(tf));
    }
    
    // 🔥 MIGLIORAMENTO #8: Controllo freschezza dati
    long currentTime = GetTickCount();
    long timeSinceUpdate = currentTime - tempState.lastUpdateTime;
    bool dataIsFresh = (timeSinceUpdate < CampionamentoMinUpdateTime * 10); // Fresh se aggiornato negli ultimi 5 secondi
    
    // Popola il result con tutti i dati
    res.score = tempState.finalNormalizedScore;
    res.direction = tempState.finalDirection;
    res.rawScore = tempState.scoreTotalRaw;
    res.ticksProcessed = tempState.consecutiveDirectionTicks;
    res.isDataFresh = dataIsFresh;
    res.lastUpdate = tempState.lastUpdateTime;
    
    // 🔥 MIGLIORAMENTO #9: Calcolo confidence basato su coerenza componenti
    // La confidence è alta quando i vari componenti "concordano" sulla direzione
    double positiveComponents = 0;
    double negativeComponents = 0;
    double totalComponents = 0;
    
    // Analizza i componenti dello score (simulazione basata sui range noti)
    if (tempState.scoreTotalRaw > 5.0) positiveComponents += 1.0;
    if (tempState.scoreTotalRaw < -2.0) negativeComponents += 1.0;
    if (tempState.consecutiveDirectionTicks >= 3) positiveComponents += 0.5; // Movimento consistente
    if (tempState.finalNormalizedScore > 7.0) positiveComponents += 1.0; // Score alto
    if (tempState.finalNormalizedScore < 3.0) negativeComponents += 1.0; // Score basso
    
    totalComponents = positiveComponents + negativeComponents;
    if (totalComponents > 0)
    {
        res.confidence = MathMax(positiveComponents, negativeComponents) / totalComponents;
    }
    else
    {
        res.confidence = 0.5; // Neutral se non ci sono segnali chiari
    }
    
    // 🔥 MIGLIORAMENTO #10: Log migliorato con più informazioni
    res.log = StringFormat("📈 [Campionamento] %s [%s] | Score: %.2f/10 | Dir: %s | Conf: %.1f%%",
                          symbol, EnumToString(tf),
                          tempState.finalNormalizedScore, 
                          tempState.finalDirection ? "BUY 🟢" : "SELL 🔴",
                          res.confidence * 100);
    
    if (!dataIsFresh)
    {
        res.log += StringFormat(" | ⚠️ Dati non aggiornati (%.1fs fa)", (double)timeSinceUpdate / 1000.0);
    }
    
    // Log esteso per debug
    if (EnableLogging_Campionamento)
    {
        res.log += StringFormat("\n  🔍 Raw Score: %.2f | Intracandle: %.2f | Ticks: %d | Fresh: %s",
                               tempState.scoreTotalRaw, 
                               tempState.intracandleScore, 
                               tempState.consecutiveDirectionTicks,
                               dataIsFresh ? "✅" : "❌");
        
        res.log += StringFormat("\n  ⏱️ Last Update: %s | Time Since: %.1fs",
                               TimeToString(tempState.lastUpdateTime, TIME_SECONDS),
                               (double)timeSinceUpdate / 1000.0);
    }
    
    return res;
}

//+------------------------------------------------------------------+
//| 🔄 OnTickCampionamento - VERSIONE CORRETTA
//+------------------------------------------------------------------+
void OnTickCampionamento(string symbol, ENUM_TIMEFRAMES tf, double currentPrice, long currentTickVolume)
{
    string key = symbol + EnumToString(tf);
    
    // Dichiarazione di un oggetto CampionamentoState locale per contenere lo stato
    CampionamentoState state;
    
    // Verifica se lo stato esiste. Se non esiste, lo inizializza e lo inserisce nella mappa.
    if (!M_CampionamentoState.IsExist(key))
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG OnTickCampionamento - Stato Campionamento non trovato per %s [%s]. Inizializzazione.", symbol, EnumToString(tf));
        
        InitializeCampionamentoState(symbol, tf);

        if (!M_CampionamentoState.IsExist(key))
        {
            if (EnableLogging_Campionamento)
                PrintFormat("ERROR OnTickCampionamento - Impossibile inizializzare lo stato Campionamento per %s [%s].", symbol, EnumToString(tf));
            return;
        }
    }
    
    // Recupera lo stato dalla mappa
    if (!M_CampionamentoState.Get(key, state))
    {
        if (EnableLogging_Campionamento)
            PrintFormat("ERROR OnTickCampionamento - Impossibile recuperare lo stato Campionamento per %s [%s].", symbol, EnumToString(tf));
        return;
    }

    // 1. Rileva il cambio di candela
    datetime currentCandleOpenTime = iTime(symbol, tf, 0);

    if (currentCandleOpenTime != state.currentCandleOpenTime)
    {
        state.isClosingPending = true;
        state.currentCandleOpenTime = currentCandleOpenTime;

        if (EnableLogging_Campionamento)
            PrintFormat("🟢 [Campionamento] Nuova candela (%s) iniziata per %s [%s] @ %s",
                        TimeToString(currentCandleOpenTime, TIME_MINUTES), symbol, EnumToString(tf), 
                        TimeToString(currentCandleOpenTime, TIME_DATE|TIME_SECONDS));
    }

    // 2. Gestione della fase di chiusura candela (se pendente)
    if (state.isClosingPending)
    {
        // Cattura i dati finali della candela appena chiusa
        double closedCandleOpen  = state.openPrice;
        double closedCandleHigh  = state.highPrice;
        double closedCandleLow   = state.lowPrice;
        double closedCandleClose = state.lastPrice;
        
        double closedCandleBody = MathAbs(closedCandleClose - closedCandleOpen);
        double closedCandleRange = closedCandleHigh - closedCandleLow;
        double closedBodyRatio = (closedCandleRange > 0) ? closedCandleBody / closedCandleRange : 0.0;
        bool closedCandleDirection = (closedCandleClose > closedCandleOpen);

        bool isReversalCandle = IsReversalCandle(closedCandleOpen, closedCandleHigh, closedCandleLow, 
                                                closedCandleClose, closedCandleDirection, state);
        
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG OnTickCampionamento: Rilevata chiusura candela. Aggiornamento storia.");
            
        // Aggiorna la storia delle candele
        UpdateCandleHistory(state, state.finalNormalizedScore, closedBodyRatio, state.finalDirection,
                           isReversalCandle, closedCandleOpen, closedCandleHigh, closedCandleLow, closedCandleClose);

        // Reset lo stato per la nuova candela
        state.openPrice = currentPrice;
        state.highPrice = currentPrice;
        state.lowPrice = currentPrice;
        state.lastPrice = currentPrice;
        state.prevPrice = currentPrice;

        state.intracandleScore = 0.0;
        state.intracandleDirectionVotes = 0;
        state.consecutiveDirectionTicks = 0;
        state.currentDirection = false;
        state.scoreTotalRaw = 0.0;
        state.finalNormalizedScore = 0.0;
        state.finalDirection = false;

        // 🔥 FIX #1: Aggiorna ADX alla chiusura candela
        state.prevADX = state.currentADX;
        state.currentADX = CalculateADX(symbol, tf, CampionamentoADXPeriod);

        state.isClosingPending = false;

        if (EnableLogging_Campionamento)
            PrintFormat("✅ [Campionamento] Chiusura candela completata. ADX aggiornato: %.2f → %.2f", 
                        state.prevADX, state.currentADX);
    }
    
    // Gestione della frequenza di campionamento
    long currentTimeMs = GetTickCount();
    if (currentTimeMs - state.lastUpdateTime < CampionamentoMinUpdateTime)
    {
        M_CampionamentoState.Insert(key, state);
        return;
    }

    // Aggiorna High/Low intracandela
    state.highPrice = MathMax(state.highPrice, currentPrice);
    state.lowPrice = MathMin(state.lowPrice, currentPrice);
    state.prevPrice = state.lastPrice;
    state.lastPrice = currentPrice;

    // 🔥 FIX #2: Aggiorna ADX anche durante il tick (opzionale per analisi più reattiva)
    // SOLO se è passato abbastanza tempo dall'ultimo aggiornamento ADX
    static long lastADXUpdateTime = 0;
    if (currentTimeMs - lastADXUpdateTime > 5000) // Aggiorna ADX ogni 5 secondi max
    {
        double newADX = CalculateADX(symbol, tf, CampionamentoADXPeriod);
        if (newADX > 0 && MathAbs(newADX - state.currentADX) > 0.1) // Solo se cambia significativamente
        {
            state.prevADX = state.currentADX;
            state.currentADX = newADX;
            lastADXUpdateTime = currentTimeMs;
            
            if (EnableLogging_Campionamento)
                PrintFormat("🔄 [ADX Update] ADX aggiornato durante tick: %.2f → %.2f", state.prevADX, state.currentADX);
        }
    }

    // Calcola i componenti dello score
    double currentPriceSpeedScore = CalculatePriceSpeedScore(symbol, tf, currentPrice, state.prevPrice);
    double currentVolumeScore = CalculateVolumeScore(symbol, tf, currentTickVolume);
    double currentADXDerivativeScore = CalculateADXDerivativeScore(symbol, tf, state);
    double currentThresholdBreakScore = CalculateThresholdBreakScore(symbol, tf, currentPrice);
    double currentShadowScore = CalculateShadowScore(symbol, tf, currentPrice);
    double currentHistoricalAnalysisScore = CalculateHistoricalAnalysisScore(state);
    
    // Aggrega lo score intracandela
    bool tickDirection = (currentPrice > state.prevPrice);
    if (currentPrice != state.prevPrice)
    {
        if (tickDirection == state.currentDirection)
        {
            state.consecutiveDirectionTicks++;
        }
        else
        {
            state.currentDirection = tickDirection;
            state.consecutiveDirectionTicks = 1;
        }
    }
    
    state.intracandleScore = currentPriceSpeedScore;

    // Determina la direzione finale
    state.finalDirection = DetermineOverallDirection(symbol, tf, currentPrice, state);

    // 🔥 FIX #3: Calcolo Score Raw Corretto (può essere negativo)
    state.scoreTotalRaw = 0.0;
    state.scoreTotalRaw += currentPriceSpeedScore;      // 0 to +10
    state.scoreTotalRaw += currentVolumeScore;          // 0 to +5  
    state.scoreTotalRaw += currentADXDerivativeScore;   // -3 to +3
    state.scoreTotalRaw += currentThresholdBreakScore;  // 0 to +5
    state.scoreTotalRaw += currentShadowScore;          // -3 to +3
    state.scoreTotalRaw += currentHistoricalAnalysisScore; // -5 to +5

    // 🔥 FIX #4: Normalizzazione Corretta per Score Negativi
    // Score raw range teorico: da -11 a +31
    // Vogliamo normalizzare a 0-10, quindi:
    // 1. Shifta il range: scoreTotalRaw + 11 = range da 0 a 42
    // 2. Normalizza: (shifted) / 42 * 10 = range da 0 a 10
    
    double shiftedScore = state.scoreTotalRaw + 11.0; // Shifta per rendere tutto positivo
    state.finalNormalizedScore = (shiftedScore / 42.0) * 10.0; // Normalizza a 0-10
    
    // Cappatura di sicurezza
    state.finalNormalizedScore = MathMax(0.0, MathMin(10.0, state.finalNormalizedScore));

    // Aggiorna l'ultimo tempo di campionamento
    state.lastUpdateTime = currentTimeMs;
    
    state.logText = StringFormat("📈 [Campionamento] Score: %.2f/10 | Dir: %s",
                                 state.finalNormalizedScore,
                                 state.finalDirection ? "BUY 🟢" : "SELL 🔴");
    
    // Debug componenti dello score
    if (EnableLogging_Campionamento)
    {
        PrintFormat("💥 [DEBUG Campionamento %s] SCORE COMPONENTI:", EnumToString(tf));
        PrintFormat("  📊 PriceSpeed        = %.2f", currentPriceSpeedScore);
        PrintFormat("  💼 Volume            = %.2f", currentVolumeScore);
        PrintFormat("  📈 ADX Deriv         = %.2f (ADX: %.1f→%.1f)", currentADXDerivativeScore, state.prevADX, state.currentADX);
        PrintFormat("  🚀 Threshold         = %.2f", currentThresholdBreakScore);
        PrintFormat("  🌗 Shadow            = %.2f", currentShadowScore);
        PrintFormat("  🕰️  Historical        = %.2f", currentHistoricalAnalysisScore);
        PrintFormat("  📊 Raw Total         = %.2f (range: -11 a +31)", state.scoreTotalRaw);
        PrintFormat("  📊 Shifted           = %.2f (range: 0 a 42)", state.scoreTotalRaw + 11.0);
        PrintFormat("  🎯 Final Score       = %.2f / 10", state.finalNormalizedScore);
        
        PrintFormat("📈 [Campionamento %s] Score: %.2f/10 (Raw: %.2f) | Dir: %s",
                    EnumToString(tf), state.finalNormalizedScore, state.scoreTotalRaw,
                    state.finalDirection ? "BUY" : "SELL");
    }
    
    // Salva l'oggetto state modificato nella mappa
    M_CampionamentoState.Insert(key, state);
}

//+------------------------------------------------------------------+
//| 🧹 Inizializza lo stato di campionamento - VERSIONE MIGLIORATA  |
//+------------------------------------------------------------------+
void InitializeCampionamentoState(string symbol, ENUM_TIMEFRAMES tf)
{
    // Crea un oggetto locale
    CampionamentoState state;

    // Impostazione campi specifici simbolo/timeframe
    state.symbol = symbol;
    state.timeframe = tf;

    // Recupera il tempo di apertura della candela corrente e i prezzi di mercato
    state.currentCandleOpenTime = iTime(symbol, tf, 0);
    
    double currentPrice = SymbolInfoDouble(symbol, SYMBOL_BID); 
    if (currentPrice <= 0.0) // 🔥 MIGLIORAMENTO: Verifica prezzo valido
    {
        currentPrice = SymbolInfoDouble(symbol, SYMBOL_ASK);
        if (currentPrice <= 0.0)
        {
            if (EnableLogging_Campionamento)
                PrintFormat("⚠️ [Campionamento] ERRORE: Impossibile ottenere prezzo valido per %s. Uso 1.0 come fallback.", symbol);
            currentPrice = 1.0; // Fallback estremo
        }
    }
    
    state.openPrice = currentPrice;
    state.highPrice = currentPrice;
    state.lowPrice = currentPrice;
    state.lastPrice = currentPrice;
    state.prevPrice = currentPrice;

    // 🔥 MIGLIORAMENTO: Calcolo ADX con validazione
    double initialADX = CalculateADX(symbol, tf, CampionamentoADXPeriod);
    if (initialADX <= 0.0 || initialADX > 100.0) // Verifica ADX valido (range tipico 0-100)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("⚠️ [Campionamento] ADX non valido (%.2f) per %s [%s]. Uso 20.0 come default.", 
                        initialADX, symbol, EnumToString(tf));
        initialADX = 20.0; // Valore neutro tipico
    }
    
    state.prevADX = initialADX;
    state.currentADX = initialADX;

    // 🔥 MIGLIORAMENTO: Inizializzazione esplicita di tutti i campi critici
    state.intracandleScore = 0.0;
    state.intracandleDirectionVotes = 0;
    state.consecutiveDirectionTicks = 0;
    state.currentDirection = false;
    state.scoreTotalRaw = 0.0;
    state.finalNormalizedScore = 0.0;
    state.finalDirection = false;
    state.isClosingPending = false;
    state.lastUpdateTime = GetTickCount();
    state.logText = "🔄 [Campionamento] Inizializzato";

    // 🔥 MIGLIORAMENTO: Inizializzazione array storico (evita valori casuali)
    for (int i = 0; i < CampionamentoThresholdLookback; i++)
    {
        state.historicalCandles[i].score = 0.0;
        state.historicalCandles[i].bodyRatio = 0.0;
        state.historicalCandles[i].direction = false;
        state.historicalCandles[i].isReversal = false;
        state.historicalCandles[i].open = 0.0;
        state.historicalCandles[i].high = 0.0;
        state.historicalCandles[i].low = 0.0;
        state.historicalCandles[i].close = 0.0;
    }
    
    // Inserisce l'oggetto nella mappa
    string key = symbol + EnumToString(tf);
    M_CampionamentoState.Insert(key, state);

    if (EnableLogging_Campionamento)
        PrintFormat("⚙️ [Campionamento] Stato inizializzato per %s [%s]: Prezzo=%.5f, ADX=%.2f", 
                    symbol, EnumToString(tf), currentPrice, initialADX);
}

//+------------------------------------------------------------------+
//| ↩️ Resetta lo stato per una nuova candela - VERSIONE MIGLIORATA  |
//+------------------------------------------------------------------+
void ResetCampionamentoState(string symbol, ENUM_TIMEFRAMES tf)
{
    string key = symbol + EnumToString(tf);
    CampionamentoState state; 

    // Recupera lo stato esistente dalla mappa
    if (M_CampionamentoState.Get(key, state))
    {
        // 🔥 MIGLIORAMENTO: Preserva alcuni valori importanti prima del reset
        double preservedCurrentADX = state.currentADX;
        string preservedSymbol = state.symbol;
        ENUM_TIMEFRAMES preservedTF = state.timeframe;
        
        state.currentCandleOpenTime = iTime(symbol, tf, 0);
        double currentMarketPrice = SymbolInfoDouble(symbol, SYMBOL_BID);
        
        // 🔥 MIGLIORAMENTO: Validazione prezzo
        if (currentMarketPrice <= 0.0)
        {
            currentMarketPrice = SymbolInfoDouble(symbol, SYMBOL_ASK);
            if (currentMarketPrice <= 0.0)
            {
                if (EnableLogging_Campionamento)
                    PrintFormat("⚠️ [Campionamento] Prezzo non valido nel reset per %s. Mantengo ultimo prezzo valido.", symbol);
                currentMarketPrice = state.lastPrice > 0 ? state.lastPrice : 1.0;
            }
        }
        
        state.openPrice = currentMarketPrice;
        state.highPrice = currentMarketPrice;
        state.lowPrice = currentMarketPrice;
        state.lastPrice = currentMarketPrice;
        state.prevPrice = currentMarketPrice;

        // Reset score e direzioni
        state.intracandleScore = 0.0;
        state.intracandleDirectionVotes = 0;
        state.consecutiveDirectionTicks = 0;
        state.currentDirection = false;
        state.scoreTotalRaw = 0.0;
        state.finalNormalizedScore = 0.0;
        state.finalDirection = false;
        state.isClosingPending = false;
        state.lastUpdateTime = GetTickCount();
        
        // 🔥 MIGLIORAMENTO: Aggiornamento ADX più intelligente
        double newADX = CalculateADX(symbol, tf, CampionamentoADXPeriod);
        if (newADX > 0.0 && newADX <= 100.0)
        {
            state.prevADX = preservedCurrentADX; // Usa il valore precedente come prev
            state.currentADX = newADX;
        }
        else
        {
            if (EnableLogging_Campionamento)
                PrintFormat("⚠️ [Campionamento] ADX non valido nel reset (%.2f). Mantengo valori precedenti.", newADX);
            // Mantieni i valori ADX esistenti se il nuovo calcolo non è valido
        }

        // 🔥 MIGLIORAMENTO: Preserva symbol e timeframe
        state.symbol = preservedSymbol;
        state.timeframe = preservedTF;
        
        // Salva le modifiche
        M_CampionamentoState.Insert(key, state);
        
        if (EnableLogging_Campionamento)
            PrintFormat("♻️ [Campionamento] Reset completato per %s [%s]: Prezzo=%.5f, ADX=%.2f→%.2f", 
                        symbol, EnumToString(tf), currentMarketPrice, state.prevADX, state.currentADX);
    } 
    else 
    {
        // 🔥 MIGLIORAMENTO: Auto-inizializzazione se stato non esiste
        if (EnableLogging_Campionamento)
            PrintFormat("⚠️ [Campionamento] Stato non esistente per reset %s [%s]. Auto-inizializzazione.", symbol, EnumToString(tf));
        
        InitializeCampionamentoState(symbol, tf); // Crea nuovo stato invece di fallire
    }
}

//+------------------------------------------------------------------+
//| UpdateCandleHistory - VERSIONE OTTIMIZZATA                      |
//+------------------------------------------------------------------+
void UpdateCandleHistory(CampionamentoState &state,
                         double closedCandleScore, double closedBodyRatio,
                         bool closedCandleDirection, bool isReversal,
                         double closedCandleOpen, double closedCandleHigh,
                         double closedCandleLow, double closedCandleClose)
{
    // 🔥 MIGLIORAMENTO: Validazione dei dati in input
    if (closedCandleHigh < closedCandleLow || closedCandleOpen <= 0 || closedCandleClose <= 0)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("⚠️ [UpdateCandleHistory] Dati candela non validi: O=%.5f, H=%.5f, L=%.5f, C=%.5f. Skip aggiornamento.", 
                        closedCandleOpen, closedCandleHigh, closedCandleLow, closedCandleClose);
        return;
    }

    // 🔥 OTTIMIZZAZIONE: Shift array più efficiente
    // Invece di copiare tutti gli elementi uno per uno, usa memmove-like approach
    if (CampionamentoThresholdLookback > 1)
    {
        // Sposta tutti gli elementi di una posizione verso destra
        for (int i = CampionamentoThresholdLookback - 1; i > 0; i--)
        {
            state.historicalCandles[i] = state.historicalCandles[i-1];
        }
    }
    
    // 🔥 MIGLIORAMENTO: Costruzione candela con validazione
    CandelaStorica newCandle;
    newCandle.score = MathMax(0.0, MathMin(10.0, closedCandleScore)); // Assicura range valido
    newCandle.bodyRatio = MathMax(0.0, MathMin(1.0, closedBodyRatio)); // Assicura range 0-1
    newCandle.direction = closedCandleDirection;
    newCandle.isReversal = isReversal;
    newCandle.open = closedCandleOpen;
    newCandle.high = closedCandleHigh;
    newCandle.low = closedCandleLow;
    newCandle.close = closedCandleClose;
    
    // Inserisci la nuova candela all'inizio
    state.historicalCandles[0] = newCandle;
    
    // 🔥 MIGLIORAMENTO: Log più informativo
    if (EnableLogging_Campionamento)
    {
        double candleRange = closedCandleHigh - closedCandleLow;
        double candleBody = MathAbs(closedCandleClose - closedCandleOpen);
        
        PrintFormat("📖 [UpdateCandleHistory] Candela salvata: Score=%.2f, Dir=%s, Body=%.1f%%, Range=%.1f pips, Reversal=%s",
                    closedCandleScore,
                    closedCandleDirection ? "BUY" : "SELL", 
                    (candleRange > 0 ? (candleBody/candleRange)*100 : 0),
                    candleRange * MathPow(10, _Digits-1), // Converti in pips approssimativi
                    isReversal ? "YES" : "NO");
    }

    // 🔥 MIGLIORAMENTO: Statistiche opzionali dell'array storico
    if (EnableLogging_Campionamento)
    {
        int buyCount = 0, sellCount = 0, reversalCount = 0;
        double avgScore = 0.0, validCandles = 0;
        
        for (int i = 0; i < CampionamentoThresholdLookback; i++)
        {
            if (state.historicalCandles[i].score > 0)
            {
                validCandles++;
                avgScore += state.historicalCandles[i].score;
                if (state.historicalCandles[i].direction) buyCount++; else sellCount++;
                if (state.historicalCandles[i].isReversal) reversalCount++;
            }
        }
        
        if (validCandles > 0)
        {
            avgScore /= validCandles;
            PrintFormat("📊 [Storia] Ultimi %.0f candele: BUY=%d, SELL=%d, Reversal=%d, AvgScore=%.2f", 
                        validCandles, buyCount, sellCount, reversalCount, avgScore);
        }
    }
}

//+------------------------------------------------------------------+
//| ⚡ Calcola lo score in base alla velocità del prezzo - VERSIONE MIGLIORATA
//+------------------------------------------------------------------+
double CalculatePriceSpeedScore(string symbol, ENUM_TIMEFRAMES tf, double currentPrice, double prevPrice)
{
    double score = 0.0;

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG PriceSpeed - Input: currentPrice=%.5f, prevPrice=%.5f", currentPrice, prevPrice);

    // Controlli di base
    if (prevPrice == 0.0)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG PriceSpeed - prevPrice è 0.0. Score = 0.0 (primo tick o reset).");
        return 0.0;
    }

    if (currentPrice == prevPrice)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG PriceSpeed - Prezzi identici. Score = 0.0.");
        return 0.0;
    }

    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    if (point == 0.0)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG PriceSpeed - Symbol Point è 0.0. Score = 0.0.");
        return 0.0;
    }

    // 🔥 MIGLIORAMENTO #1: Calcolo Velocità con Direzione
    double priceChange = currentPrice - prevPrice; // NON usare MathAbs per mantenere direzione
    double speed = priceChange / point; // Velocità in punti (positiva o negativa)
    double absSpeed = MathAbs(speed); // Velocità assoluta per i calcoli

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG PriceSpeed - PriceChange: %.5f, Speed: %.2f points, AbsSpeed: %.2f points", priceChange, speed, absSpeed);

    // 🔥 MIGLIORAMENTO #2: Velocità Minima più Intelligente
    if (absSpeed < CampionamentoMinPriceSpeed)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG PriceSpeed - Velocità troppo bassa (%.2f < %.2f). Score = 0.0.", absSpeed, CampionamentoMinPriceSpeed);
        return 0.0;
    }

    // 🔥 MIGLIORAMENTO #3: Sistema di Scoring Progressivo Semplificato
    double baseScore = 0.0;

    // VELOCITÀ ESTREMA (>= MaxSpeed)
    if (absSpeed >= CampionamentoMaxPriceSpeed)
    {
        baseScore = CampionamentoWeightPriceSpeed * 1.5; // Score massimo per movimenti estremi (Weight=10 * 1.5 = 15)
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG PriceSpeed - VELOCITÀ ESTREMA! Speed=%.2f >= Max=%.2f, BaseScore=%.3f", absSpeed, CampionamentoMaxPriceSpeed, baseScore);
    }
    // VELOCITÀ ALTA (tra Min e Max)
    else
    {
        // Score lineare tra MinSpeed e MaxSpeed
        double speedRange = CampionamentoMaxPriceSpeed - CampionamentoMinPriceSpeed;
        if (speedRange > 0)
        {
            double normalizedSpeed = (absSpeed - CampionamentoMinPriceSpeed) / speedRange; // 0.0 - 1.0
            baseScore = CampionamentoWeightPriceSpeed * (0.2 + normalizedSpeed * 0.8); // Score da 20% a 100% del peso
        }
        else
        {
            baseScore = CampionamentoWeightPriceSpeed; // Fallback se Min == Max
        }

        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG PriceSpeed - VELOCITÀ NORMALE: Speed=%.2f, Normalized=%.2f, BaseScore=%.3f", absSpeed, (absSpeed - CampionamentoMinPriceSpeed) / speedRange, baseScore);
    }

    // 🔥 MIGLIORAMENTO #4: Bonus/Penalità per Accelerazione
    // Calcola se il prezzo sta accelerando comparando con movimenti precedenti
    double accelerationBonus = 1.0;
    
    // Ottieni il prezzo di 2 tick fa per calcolare l'accelerazione
    static double prevPrevPrice = 0.0;
    if (prevPrevPrice > 0.0)
    {
        double prevSpeed = MathAbs((prevPrice - prevPrevPrice) / point);
        
        if (prevSpeed > 0)
        {
            double acceleration = absSpeed / prevSpeed; // Rapporto velocità attuale vs precedente
            
            if (acceleration >= 1.5) // Accelerazione significativa
            {
                accelerationBonus += 0.5; // +50% per accelerazione
                if (EnableLogging_Campionamento)
                    PrintFormat("DEBUG PriceSpeed - ACCELERAZIONE! PrevSpeed=%.2f, CurrentSpeed=%.2f, Ratio=%.2fx, Bonus=+50%%", prevSpeed, absSpeed, acceleration);
            }
            else if (acceleration <= 0.7) // Decelerazione
            {
                accelerationBonus -= 0.3; // -30% per decelerazione
                accelerationBonus = MathMax(0.1, accelerationBonus); // Minimo 10%
                if (EnableLogging_Campionamento)
                    PrintFormat("DEBUG PriceSpeed - DECELERAZIONE. PrevSpeed=%.2f, CurrentSpeed=%.2f, Ratio=%.2fx, Penalità=-30%%", prevSpeed, absSpeed, acceleration);
            }
        }
    }
    prevPrevPrice = prevPrice; // Aggiorna per il prossimo calcolo

    // 🔥 MIGLIORAMENTO #5: Bonus per Velocità Sostenuta
    // Usa una variabile statica per tracciare velocità consecutive
    static int consecutiveHighSpeedTicks = 0;
    static double lastHighSpeed = 0.0;
    
    if (absSpeed >= CampionamentoMinPriceSpeed * 1.5) // Velocità "alta"
    {
        if (MathAbs(absSpeed - lastHighSpeed) < CampionamentoMinPriceSpeed * 0.5) // Velocità simile alla precedente
        {
            consecutiveHighSpeedTicks++;
            if (consecutiveHighSpeedTicks >= 3) // 3+ tick consecutivi ad alta velocità
            {
                accelerationBonus += 0.3; // +30% per velocità sostenuta
                if (EnableLogging_Campionamento)
                    PrintFormat("DEBUG PriceSpeed - VELOCITÀ SOSTENUTA! %d tick consecutivi ad alta velocità. Bonus=+30%%", consecutiveHighSpeedTicks);
            }
        }
        else
        {
            consecutiveHighSpeedTicks = 1; // Reset ma conta questo tick
        }
        lastHighSpeed = absSpeed;
    }
    else
    {
        consecutiveHighSpeedTicks = 0; // Reset per velocità basse
        lastHighSpeed = 0.0;
    }

    // 🔥 MIGLIORAMENTO #6: Considerazione Direzione (Opzionale)
    // Questo può essere utilizzato dal chiamante per determinare la direzione complessiva
    // Ma per ora manteniamo lo score sempre positivo
    double directionMultiplier = 1.0;
    
    // Il score finale include la direzione come informazione, ma rimane positivo
    score = baseScore * accelerationBonus * directionMultiplier;
    
    // Cappatura finale
    score = MathMin(score, CampionamentoWeightPriceSpeed * 1.8); // Max 1.8x il peso base (18 punti con Weight=10)
    score = MathMax(score, 0.0); // Minimo 0

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG PriceSpeed - RISULTATO: BaseScore=%.3f, AccelBonus=%.2fx, FinalScore=%.3f, Direction=%s",
                    baseScore, accelerationBonus, score, (speed > 0) ? "UP ⬆️" : "DOWN ⬇️");

    return score;
}

//+------------------------------------------------------------------+
//| 📊 Calcola lo score in base al volume - VERSIONE AVANZATA        |
//+------------------------------------------------------------------+
double CalculateVolumeScore(string symbol, ENUM_TIMEFRAMES tf, long currentTickVolume)
{
    double score = 0.0;

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG VolumeScore - Input: currentTickVolume=%d", currentTickVolume);

    // Controlli di base
    if (currentTickVolume <= 0)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG VolumeScore - currentTickVolume <= 0. Score = 0.0.");
        return 0.0;
    }

    // 🔥 MIGLIORAMENTO #1: Calcolo Volume Medio Dinamico
    const int VOLUME_LOOKBACK = 20; // Analizza ultimi 20 periodi
    long totalVolume = 0;
    long maxVolume = 0;
    long minVolume = LONG_MAX;
    int validPeriods = 0;
    
    // Array per analisi distribuzione volume
    long volumeHistory[20];
    ArrayInitialize(volumeHistory, 0);

    for (int i = 1; i <= VOLUME_LOOKBACK; i++) // Inizia da 1 per escludere il periodo corrente
    {
        long historicalVolume = iVolume(symbol, tf, i);
        if (historicalVolume > 0)
        {
            totalVolume += historicalVolume;
            maxVolume = MathMax(maxVolume, historicalVolume);
            minVolume = MathMin(minVolume, historicalVolume);
            volumeHistory[validPeriods] = historicalVolume;
            validPeriods++;
        }
    }

    if (validPeriods < 5) // Non abbastanza dati storici
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG VolumeScore - Non abbastanza dati volume storici (%d < 5). Uso soglia fissa.", validPeriods);
        
        // Fallback alla logica originale semplificata
        if (currentTickVolume >= CampionamentoVolumeThreshold)
        {
            score = CampionamentoVolumeFactor * MathMin(2.0, (double)currentTickVolume / CampionamentoVolumeThreshold);
        }
        return score;
    }

    double avgVolume = (double)totalVolume / validPeriods;
    double volumeRange = (double)maxVolume - (double)minVolume;

    // 🔥 MIGLIORAMENTO #2: Calcolo Deviazione Standard del Volume
    double volumeVariance = 0.0;
    for (int i = 0; i < validPeriods; i++)
    {
        double diff = volumeHistory[i] - avgVolume;
        volumeVariance += diff * diff;
    }
    double volumeStdDev = MathSqrt(volumeVariance / validPeriods);

    // 🔥 MIGLIORAMENTO #3: Classificazione Volume Multi-Livello
    double volumeRatio = currentTickVolume / avgVolume;
    double volumeZScore = (volumeStdDev > 0) ? (currentTickVolume - avgVolume) / volumeStdDev : 0.0;

    if (EnableLogging_Campionamento)
    {
        PrintFormat("DEBUG VolumeScore - Stats: Avg=%.0f, Max=%d, Min=%d, StdDev=%.1f", avgVolume, maxVolume, minVolume, volumeStdDev);
        PrintFormat("DEBUG VolumeScore - Current: %d, Ratio=%.2fx, Z-Score=%.2f", currentTickVolume, volumeRatio, volumeZScore);
    }

    // 🔥 MIGLIORAMENTO #4: Sistema di Scoring Progressivo
    
    // VOLUME SPIKE ESTREMO (Z-Score > 3.0)
    if (volumeZScore >= 3.0)
    {
        score = CampionamentoVolumeFactor * 3.0; // Score massimo per spike estremi
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG VolumeScore - VOLUME SPIKE ESTREMO! Z-Score=%.2f, Score=%.3f", volumeZScore, score);
    }
    // VOLUME MOLTO ALTO (Z-Score 2.0-3.0)
    else if (volumeZScore >= 2.0)
    {
        score = CampionamentoVolumeFactor * (1.5 + (volumeZScore - 2.0) * 1.5); // Score 1.5x - 3.0x
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG VolumeScore - VOLUME MOLTO ALTO. Z-Score=%.2f, Score=%.3f", volumeZScore, score);
    }
    // VOLUME ALTO (Z-Score 1.0-2.0)
    else if (volumeZScore >= 1.0)
    {
        score = CampionamentoVolumeFactor * (0.5 + volumeZScore * 0.5); // Score 1.0x - 1.5x
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG VolumeScore - VOLUME ALTO. Z-Score=%.2f, Score=%.3f", volumeZScore, score);
    }
    // VOLUME SOPRA MEDIA (Ratio > 1.2)
    else if (volumeRatio >= 1.2)
    {
        score = CampionamentoVolumeFactor * (volumeRatio * 0.4); // Score proporzionale
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG VolumeScore - VOLUME SOPRA MEDIA. Ratio=%.2fx, Score=%.3f", volumeRatio, score);
    }
    // VOLUME NORMALE-BASSO
    else
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG VolumeScore - Volume normale/basso. Ratio=%.2fx, Z-Score=%.2f, Score=0.0", volumeRatio, volumeZScore);
    }

    // 🔥 MIGLIORAMENTO #5: Bonus per Pattern di Volume
    double patternBonus = 1.0;
    
    // Analizza trend di volume delle ultime 3 candele
    if (validPeriods >= 3)
    {
        long vol1 = iVolume(symbol, tf, 1); // Candela precedente
        long vol2 = iVolume(symbol, tf, 2); // 2 candele fa
        long vol3 = iVolume(symbol, tf, 3); // 3 candele fa
        
        // ACCUMULATION PATTERN: Volume crescente progressivamente
        if (currentTickVolume > vol1 && vol1 > vol2 && vol2 > vol3)
        {
            patternBonus += 0.5; // +50% per accumulation
            if (EnableLogging_Campionamento)
                PrintFormat("DEBUG VolumeScore - ACCUMULATION PATTERN rilevato! Bonus=+50%%");
        }
        // VOLUME BREAKOUT: Volume attuale molto superiore alle 3 precedenti
        else if (currentTickVolume > vol1 * 2 && currentTickVolume > vol2 * 2 && currentTickVolume > vol3 * 2)
        {
            patternBonus += 0.7; // +70% per breakout pattern
            if (EnableLogging_Campionamento)
                PrintFormat("DEBUG VolumeScore - VOLUME BREAKOUT PATTERN rilevato! Bonus=+70%%");
        }
    }

    // 🔥 MIGLIORAMENTO #6: Bonus Sessione di Trading
    double sessionBonus = 1.0;
    int hour = GetHour(); // Usa la funzione Utility
    
    // Bonus per sessioni ad alto volume (Londra 8-12, New York 13-17 GMT)
    if ((hour >= 8 && hour <= 12) || (hour >= 13 && hour <= 17))
    {
        sessionBonus += 0.2; // +20% durante sessioni principali
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG VolumeScore - Sessione ad alto volume (ora %d). Bonus=+20%%", hour);
    }
    // Penalità per sessioni a basso volume (22-6 GMT)
    else if (hour >= 22 || hour <= 6)
    {
        sessionBonus -= 0.3; // -30% durante sessioni secondarie
        sessionBonus = MathMax(0.1, sessionBonus); // Minimo 10%
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG VolumeScore - Sessione a basso volume (ora %d). Penalità=-30%%", hour);
    }

    // Applica bonus e cappatura finale per Score Massimo = 5.0
    score = score * patternBonus * sessionBonus;
    score = MathMax(0.0, MathMin(score, 5.0)); // Cap assoluto a 5.0

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG VolumeScore - FINALE: BaseScore=%.3f, PatternBonus=%.2fx, SessionBonus=%.2fx, FinalScore=%.3f", 
                    score / (patternBonus * sessionBonus), patternBonus, sessionBonus, score);

    return score;
}

//+------------------------------------------------------------------+
//| 🚀 Calcola lo score in base alla derivata dell'ADX - VERSIONE CORRETTA +/-2
//+------------------------------------------------------------------+
double CalculateADXDerivativeScore(string symbol, ENUM_TIMEFRAMES tf, const CampionamentoState &state)
{
    double score = 0.0;

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG ADXDerivative - ADX nello stato: currentADX=%.5f, prevADX=%.5f", state.currentADX, state.prevADX);

    // Controlli di validità
    if (CampionamentoADXPeriod <= 0)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ADXDerivative - CampionamentoADXPeriod <= 0. Score = 0.0.");
        return 0.0;
    }

    if (state.prevADX == 0.0 || state.currentADX == 0.0)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ADXDerivative - ADX precedente/attuale è 0.0. Score = 0.0. (In attesa di dati ADX validi).");
        return 0.0;
    }

    // 🔥 MIGLIORAMENTO #1: Calcolo Derivata con Direzione Preservata
    double adxDerivative = state.currentADX - state.prevADX; // MANTIENE il segno +/-
    double absAdxDerivative = MathAbs(adxDerivative);

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG ADXDerivative - adxDerivative: %.5f, absDerivative: %.5f, threshold: %.5f", 
                    adxDerivative, absAdxDerivative, CampionamentoADXDerivativeThreshold);

    // 🔥 MIGLIORAMENTO #2: Soglia di Attivazione
    if (absAdxDerivative < CampionamentoADXDerivativeThreshold)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ADXDerivative - Variazione ADX troppo piccola (%.5f < %.5f). Score = 0.0.", 
                        absAdxDerivative, CampionamentoADXDerivativeThreshold);
        return 0.0;
    }

    // 🔥 MIGLIORAMENTO #3: Score Proporzionale con Direzione (usando CampionamentoADXDerivativeWeight)
    
    // Score base proporzionale alla derivata (0.0 - CampionamentoADXDerivativeWeight)
    double baseScore = MathMin(CampionamentoADXDerivativeWeight, 
                              CampionamentoADXDerivativeWeight * (absAdxDerivative / CampionamentoADXDerivativeThreshold));
    
    // 🔥 MIGLIORAMENTO #4: Applicazione della Direzione
    if (adxDerivative > 0) // ADX crescente = trend si rafforza
    {
        score = baseScore; // Score POSITIVO (+0.1 a +CampionamentoADXDerivativeWeight)
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ADXDerivative - ADX CRESCENTE (trend si rafforza): +%.5f → Score = +%.3f", 
                        adxDerivative, score);
    }
    else // ADX decrescente = trend si indebolisce
    {
        score = -baseScore; // Score NEGATIVO (-0.1 a -CampionamentoADXDerivativeWeight)
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ADXDerivative - ADX DECRESCENTE (trend si indebolisce): %.5f → Score = %.3f", 
                        adxDerivative, score);
    }

    // 🔥 MIGLIORAMENTO #5: Bonus per ADX Estremi
    double adxStrengthBonus = 1.0;
    
    // Se l'ADX attuale è già alto (>25), le variazioni sono più significative
    if (state.currentADX >= 25.0)
    {
        adxStrengthBonus += 0.3; // +30% se ADX è già forte
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ADXDerivative - ADX già forte (%.1f >= 25.0). Bonus=+30%%", state.currentADX);
    }
    // Se l'ADX attuale è molto basso (<15), le variazioni sono meno affidabili
    else if (state.currentADX < 15.0)
    {
        adxStrengthBonus -= 0.4; // -40% se ADX è debole
        adxStrengthBonus = MathMax(0.1, adxStrengthBonus); // Minimo 10%
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ADXDerivative - ADX debole (%.1f < 15.0). Penalità=-40%%", state.currentADX);
    }

    // Applica il bonus mantenendo il segno
    if (score > 0)
        score = score * adxStrengthBonus;
    else if (score < 0)
        score = score * adxStrengthBonus; // Il bonus si applica anche ai negativi

    // 🔥 MIGLIORAMENTO #6: Cappatura Finale +/- CampionamentoADXDerivativeWeight
    score = MathMax(-CampionamentoADXDerivativeWeight, MathMin(CampionamentoADXDerivativeWeight, score));

    // 🔥 MIGLIORAMENTO #7: Gestione Edge Cases per Score Piccolissimi
    if (MathAbs(score) < 0.1)
    {
        score = 0.0; // Azzera score troppo piccoli per evitare rumore
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ADXDerivative - Score troppo piccolo (<0.1). Azzerato.");
    }

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG ADXDerivative - RISULTATO FINALE: ADX %.1f→%.1f (Δ%.3f), StrengthBonus=%.2fx, Score=%.3f", 
                    state.prevADX, state.currentADX, adxDerivative, adxStrengthBonus, score);

    return score;
}

//+------------------------------------------------------------------+
//| 📈 Calcola lo score per la rottura di soglie storiche - VERSIONE MIGLIORATA
//+------------------------------------------------------------------+
double CalculateThresholdBreakScore(string symbol, ENUM_TIMEFRAMES tf, double currentPrice)
{
    double score = 0.0;
    bool isBreakoutBullish = false; // Flag per determinare la direzione del breakout

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG ThresholdBreak - Inizio. Symbol: %s, TF: %s, CurrentPrice: %.5f", symbol, EnumToString(tf), currentPrice);

    // Recupera lo stato
    CampionamentoState state_copy;
    if (!M_CampionamentoState.Get(symbol + EnumToString(tf), state_copy))
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ThresholdBreak - Stato non esistente. Return 0.0.");
        return 0.0;
    }

    if (ArraySize(state_copy.historicalCandles) == 0)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ThresholdBreak - Array historicalCandles vuoto. Return 0.0.");
        return 0.0;
    }

    // 🔥 MIGLIORAMENTO #1: Analisi Multi-Livello dei Supporti/Resistenze
    double highestHigh = 0.0;
    double lowestLow = 999999.0; // Inizializza con valore alto
    double secondHighestHigh = 0.0;
    double secondLowestLow = 999999.0;
    
    bool firstValidHighFound = false;
    bool firstValidLowFound = false;
    int validCandlesCount = 0;
    double avgHigh = 0.0;
    double avgLow = 0.0;

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG ThresholdBreak - Analisi multi-livello su %d candele storiche.", CampionamentoThresholdLookback);

    for (int i = 0; i < CampionamentoThresholdLookback; i++)
    {
        CandelaStorica current_historical_candle;
        if (state_copy.GetHistoricalCandle(i, current_historical_candle))
        {
            if (current_historical_candle.high > 0 && current_historical_candle.low > 0)
            {
                validCandlesCount++;
                avgHigh += current_historical_candle.high;
                avgLow += current_historical_candle.low;

                // Trova highest high e second highest
                if (!firstValidHighFound || current_historical_candle.high > highestHigh)
                {
                    secondHighestHigh = highestHigh; // Il precedente highest diventa second
                    highestHigh = current_historical_candle.high;
                    firstValidHighFound = true;
                }
                else if (current_historical_candle.high > secondHighestHigh && current_historical_candle.high < highestHigh)
                {
                    secondHighestHigh = current_historical_candle.high;
                }

                // Trova lowest low e second lowest
                if (!firstValidLowFound || current_historical_candle.low < lowestLow)
                {
                    secondLowestLow = lowestLow; // Il precedente lowest diventa second
                    lowestLow = current_historical_candle.low;
                    firstValidLowFound = true;
                }
                else if (current_historical_candle.low < secondLowestLow && current_historical_candle.low > lowestLow)
                {
                    secondLowestLow = current_historical_candle.low;
                }

                if (EnableLogging_Campionamento)
                    PrintFormat("DEBUG ThresholdBreak - Candela[%d]: H=%.5f, L=%.5f", i, current_historical_candle.high, current_historical_candle.low);
            }
        }
    }

    if (validCandlesCount == 0)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ThresholdBreak - Nessuna candela valida trovata.");
        return 0.0;
    }

    avgHigh /= validCandlesCount;
    avgLow /= validCandlesCount;

    if (EnableLogging_Campionamento)
    {
        PrintFormat("DEBUG ThresholdBreak - Livelli: Highest=%.5f, 2ndHighest=%.5f, AvgHigh=%.5f", highestHigh, secondHighestHigh, avgHigh);
        PrintFormat("DEBUG ThresholdBreak - Livelli: Lowest=%.5f, 2ndLowest=%.5f, AvgLow=%.5f", lowestLow, secondLowestLow, avgLow);
    }
    
    // 🔥 MIGLIORAMENTO #2: Calcolo Score Proporzionale e Intelligente
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    if (point == 0.0)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ThresholdBreak - Symbol Point è 0.0. Return 0.0.");
        return 0.0;
    }

    // 🔥 MIGLIORAMENTO #3: Analisi Volume del Breakout (se disponibile)
    long currentVolume = iVolume(symbol, tf, 0);
    long avgVolume = 0;
    for (int i = 1; i <= 5; i++) // Media degli ultimi 5 tick volume
        avgVolume += iVolume(symbol, tf, i);
    avgVolume /= 5;
    
    double volumeMultiplier = (avgVolume > 0) ? MathMin(2.0, (double)currentVolume / avgVolume) : 1.0;

    // 🔥 BREAKOUT RIALZISTA (Superiore)
    if (currentPrice > highestHigh && firstValidHighFound)
    {
        isBreakoutBullish = true;
        double breakoutDistance = currentPrice - highestHigh;
        double breakoutPips = breakoutDistance / point;
        
        // Score base proporzionale alla distanza del breakout
        double baseScore = MathMin(CampionamentoThresholdBreakWeight, 
                                  CampionamentoThresholdBreakWeight * (breakoutPips / 50.0)); // Normalizza su 50 pips
        
        // 🔥 BONUS: Breakout "pulito" sopra multiple resistenze
        double cleanBreakoutBonus = 1.0;
        if (currentPrice > secondHighestHigh && secondHighestHigh > 0)
        {
            cleanBreakoutBonus += 0.3; // +30% se supera anche la seconda resistenza
            if (EnableLogging_Campionamento)
                PrintFormat("DEBUG ThresholdBreak - CLEAN BREAKOUT: Supera anche 2nd resistance (%.5f)", secondHighestHigh);
        }
        if (currentPrice > avgHigh)
        {
            cleanBreakoutBonus += 0.2; // +20% se supera la media
            if (EnableLogging_Campionamento)
                PrintFormat("DEBUG ThresholdBreak - Supera anche Average High (%.5f)", avgHigh);
        }
        
        score = baseScore * cleanBreakoutBonus * volumeMultiplier;
        
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ThresholdBreak - BREAKOUT RIALZISTA: Distance=%.1f pips, BaseScore=%.3f, Bonus=%.2f, Volume=%.2f, Final=%.3f",
                        breakoutPips, baseScore, cleanBreakoutBonus, volumeMultiplier, score);
    }
    // 🔥 BREAKOUT RIBASSISTA (Inferiore)
    else if (currentPrice < lowestLow && firstValidLowFound)
    {
        isBreakoutBullish = false;
        double breakoutDistance = lowestLow - currentPrice;
        double breakoutPips = breakoutDistance / point;
        
        // Score base proporzionale alla distanza del breakout
        double baseScore = MathMin(CampionamentoThresholdBreakWeight, 
                                  CampionamentoThresholdBreakWeight * (breakoutPips / 50.0)); // Normalizza su 50 pips
        
        // 🔥 BONUS: Breakout "pulito" sotto multiple supporti
        double cleanBreakoutBonus = 1.0;
        if (currentPrice < secondLowestLow && secondLowestLow < 999999.0)
        {
            cleanBreakoutBonus += 0.3; // +30% se rompe anche il secondo supporto
            if (EnableLogging_Campionamento)
                PrintFormat("DEBUG ThresholdBreak - CLEAN BREAKDOWN: Rompe anche 2nd support (%.5f)", secondLowestLow);
        }
        if (currentPrice < avgLow)
        {
            cleanBreakoutBonus += 0.2; // +20% se rompe la media
            if (EnableLogging_Campionamento)
                PrintFormat("DEBUG ThresholdBreak - Rompe anche Average Low (%.5f)", avgLow);
        }
        
        score = baseScore * cleanBreakoutBonus * volumeMultiplier;
        
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ThresholdBreak - BREAKOUT RIBASSISTA: Distance=%.1f pips, BaseScore=%.3f, Bonus=%.2f, Volume=%.2f, Final=%.3f",
                        breakoutPips, baseScore, cleanBreakoutBonus, volumeMultiplier, score);
    }
    // 🔥 MIGLIORAMENTO #4: Rilevamento "Avvicinamento" alle Soglie
    else
    {
        double distanceToResistance = (firstValidHighFound) ? (highestHigh - currentPrice) / point : 999999.0;
        double distanceToSupport = (firstValidLowFound) ? (currentPrice - lowestLow) / point : 999999.0;
        
        // Se il prezzo si avvicina a una soglia importante (entro 10 pips), assegna un piccolo score
        if (distanceToResistance <= 10.0 && distanceToResistance > 0)
        {
            score = CampionamentoThresholdBreakWeight * 0.1 * (1.0 - distanceToResistance / 10.0); // Score crescente avvicinandosi
            isBreakoutBullish = true;
            if (EnableLogging_Campionamento)
                PrintFormat("DEBUG ThresholdBreak - APPROACHING RESISTANCE: %.1f pips away, Score=%.3f", distanceToResistance, score);
        }
        else if (distanceToSupport <= 10.0 && distanceToSupport > 0)
        {
            score = CampionamentoThresholdBreakWeight * 0.1 * (1.0 - distanceToSupport / 10.0); // Score crescente avvicinandosi
            isBreakoutBullish = false;
            if (EnableLogging_Campionamento)
                PrintFormat("DEBUG ThresholdBreak - APPROACHING SUPPORT: %.1f pips away, Score=%.3f", distanceToSupport, score);
        }
        else
        {
            if (EnableLogging_Campionamento)
                PrintFormat("DEBUG ThresholdBreak - Nessun breakout o avvicinamento significativo. DistRes=%.1f, DistSup=%.1f", 
                            distanceToResistance, distanceToSupport);
        }
    }

    // 🔥 MIGLIORAMENTO #5: Score sempre positivo, direzione gestita separatamente
    // Invece di score negativi, usiamo la logica nel DetermineOverallDirection per interpretare la direzione
    score = MathAbs(score); // Assicura che lo score sia sempre positivo
    
    // Cappatura finale per Score Massimo = 5.0
    score = MathMin(score, 5.0); // Cap assoluto a 5.0

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG ThresholdBreak - RISULTATO FINALE: Score=%.3f, Direzione=%s, VolumeMultiplier=%.2f", 
                    score, isBreakoutBullish ? "BULLISH" : "BEARISH", volumeMultiplier);

    return score;
}

//+------------------------------------------------------------------+
//| 👻 Calcola lo score in base all'analisi delle ombre - VERSIONE COMPLETA +/-3
//+------------------------------------------------------------------+
double CalculateShadowScore(string symbol, ENUM_TIMEFRAMES tf, double currentPrice)
{
    double score = 0.0;

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG ShadowScore - Inizio. Symbol: %s, TF: %s, CurrentPrice: %.5f", symbol, EnumToString(tf), currentPrice);

    // Recupera lo stato
    CampionamentoState state_copy;
    if (!M_CampionamentoState.Get(symbol + EnumToString(tf), state_copy))
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ShadowScore - Stato non esistente. Return 0.0.");
        return 0.0;
    }

    double open = state_copy.openPrice;
    double currentHigh = state_copy.highPrice;
    double currentLow = state_copy.lowPrice;

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG ShadowScore - Candela: O=%.5f, H=%.5f, L=%.5f, C=%.5f", open, currentHigh, currentLow, currentPrice);

    double candleRange = currentHigh - currentLow;
    if (candleRange <= 0)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ShadowScore - CandleRange <= 0 (%.5f). Return 0.0.", candleRange);
        return 0.0;
    }

    // 🔥 MIGLIORAMENTO #1: Calcolo Ombre Standardizzato
    double upperShadow = currentHigh - MathMax(open, currentPrice);
    double lowerShadow = MathMin(open, currentPrice) - currentLow;
    double body = MathAbs(currentPrice - open);
    
    // Ratio delle ombre rispetto al range totale
    double upperShadowRatio = upperShadow / candleRange;
    double lowerShadowRatio = lowerShadow / candleRange;
    double bodyRatio = body / candleRange;
    
    // Direzione della candela
    bool isBullish = (currentPrice >= open);

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG ShadowScore - UpperShadow=%.5f (%.1f%%), LowerShadow=%.5f (%.1f%%), Body=%.5f (%.1f%%), Direction=%s",
                    upperShadow, upperShadowRatio*100, lowerShadow, lowerShadowRatio*100, body, bodyRatio*100, isBullish ? "BULL" : "BEAR");

    // 🔥 MIGLIORAMENTO #2: Analisi Multi-Scenario delle Ombre

    // SCENARIO A: HAMMER/DOJI PATTERN (Score POSITIVO +1 a +3)
    // Ombra inferiore lunga + body piccolo = supporto testato e respinto
    if (lowerShadowRatio >= CampionamentoShadowRatioThreshold && 
        bodyRatio <= 0.3 && // Body piccolo
        upperShadowRatio <= lowerShadowRatio * 0.5) // Ombra superiore molto più piccola
    {
        double hammerScore = CampionamentoShadowWeight * (lowerShadowRatio / CampionamentoShadowRatioThreshold);
        hammerScore = MathMin(hammerScore, CampionamentoShadowWeight); // Cap al peso massimo
        
        score += hammerScore; // Score POSITIVO
        
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ShadowScore - HAMMER/DOJI PATTERN: LowerShadow=%.1f%%, Body=%.1f%%, Score=+%.3f", 
                        lowerShadowRatio*100, bodyRatio*100, hammerScore);
    }
    
    // SCENARIO B: SHOOTING STAR PATTERN (Score NEGATIVO -1 a -3) 
    // Ombra superiore lunga + body piccolo = resistenza testata ma respinta
    else if (upperShadowRatio >= CampionamentoShadowRatioThreshold && 
             bodyRatio <= 0.3 && // Body piccolo
             lowerShadowRatio <= upperShadowRatio * 0.5) // Ombra inferiore molto più piccola
    {
        double shootingStarScore = CampionamentoShadowWeight * (upperShadowRatio / CampionamentoShadowRatioThreshold);
        shootingStarScore = MathMin(shootingStarScore, CampionamentoShadowWeight); // Cap al peso massimo
        
        score -= shootingStarScore; // Score NEGATIVO
        
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ShadowScore - SHOOTING STAR PATTERN: UpperShadow=%.1f%%, Body=%.1f%%, Score=%.3f", 
                        upperShadowRatio*100, bodyRatio*100, -shootingStarScore);
    }
    
    // SCENARIO C: OMBRA CONTRARIA AL MOVIMENTO (Score NEGATIVO -0.5 a -2)
    // Ombra che contradice la direzione del movimento (come nel codice originale)
    else if (isBullish && upperShadowRatio >= CampionamentoShadowRatioThreshold)
    {
        // Candela rialzista con ombra superiore lunga = resistenza incontrata
        double resistanceScore = CampionamentoShadowWeight * 0.7 * (upperShadowRatio / CampionamentoShadowRatioThreshold);
        resistanceScore = MathMin(resistanceScore, CampionamentoShadowWeight * 0.7);
        
        score -= resistanceScore; // Score NEGATIVO
        
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ShadowScore - RESISTENZA su movimento RIALZISTA: UpperShadow=%.1f%%, Score=%.3f", 
                        upperShadowRatio*100, -resistanceScore);
    }
    else if (!isBullish && lowerShadowRatio >= CampionamentoShadowRatioThreshold)
    {
        // Candela ribassista con ombra inferiore lunga = supporto incontrato
        double supportScore = CampionamentoShadowWeight * 0.7 * (lowerShadowRatio / CampionamentoShadowRatioThreshold);
        supportScore = MathMin(supportScore, CampionamentoShadowWeight * 0.7);
        
        score -= supportScore; // Score NEGATIVO (ostacolo al movimento)
        
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ShadowScore - SUPPORTO su movimento RIBASSISTA: LowerShadow=%.1f%%, Score=%.3f", 
                        lowerShadowRatio*100, -supportScore);
    }
    
    // SCENARIO D: OMBRE EQUILIBRATE (Score leggermente POSITIVO +0.1 a +0.5)
    // Entrambe le ombre presenti ma bilanciate = consolidamento
    else if (upperShadowRatio >= 0.15 && lowerShadowRatio >= 0.15 && 
             MathAbs(upperShadowRatio - lowerShadowRatio) <= 0.1) // Ombre simili
    {
        double consolidationScore = CampionamentoShadowWeight * 0.2; // Score basso positivo
        score += consolidationScore;
        
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ShadowScore - CONSOLIDAMENTO: UpperShadow=%.1f%%, LowerShadow=%.1f%%, Score=+%.3f", 
                        upperShadowRatio*100, lowerShadowRatio*100, consolidationScore);
    }

    // 🔥 MIGLIORAMENTO #3: Bonus per Ombre Estreme
    double extremeShadowBonus = 1.0;
    
    if (upperShadowRatio >= 0.7 || lowerShadowRatio >= 0.7) // Ombra >= 70% del range
    {
        extremeShadowBonus += 0.5; // +50% per ombre estreme
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ShadowScore - OMBRA ESTREMA rilevata! Bonus=+50%%");
    }
    
    // Applica bonus mantenendo il segno
    if (score != 0)
        score = score * extremeShadowBonus;

    // 🔥 MIGLIORAMENTO #4: Cappatura Finale +/- CampionamentoShadowWeight (3.0)
    score = MathMax(-CampionamentoShadowWeight, MathMin(CampionamentoShadowWeight, score));

    // 🔥 MIGLIORAMENTO #5: Filtraggio Rumore
    if (MathAbs(score) < 0.1)
    {
        score = 0.0; // Azzera score troppo piccoli
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG ShadowScore - Score troppo piccolo (<0.1). Azzerato.");
    }

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG ShadowScore - RISULTATO FINALE: Pattern analizzato, ExtremeBabonus=%.2fx, Score=%.3f", 
                    extremeShadowBonus, score);

    return score;
}

//+------------------------------------------------------------------+
//| 📜 Calcola lo score per l'analisi storica delle candele - VERSIONE MIGLIORATA
//+------------------------------------------------------------------+
double CalculateHistoricalAnalysisScore(const CampionamentoState &state)
{
    double score = 0.0;

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG HistoricalAnalysis - Inizio.");

    // Inizializzazione variabili di analisi
    double totalHistoricalScore = 0.0;
    int buyVotes = 0;
    int sellVotes = 0;
    int consecutiveTrendCandles = 0;
    int validCandlesProcessed = 0;
    bool currentHistoricalTrendDirection = false;
    double avgBodyRatio = 0.0;
    int reversalCount = 0;

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG HistoricalAnalysis - Analisi su %d candele storiche.", CampionamentoThresholdLookback);

    // 🔥 MIGLIORAMENTO #1: Analisi Storica Semplificata
    for (int i = 0; i < CampionamentoThresholdLookback; i++)
    {
        CandelaStorica current_historical_candle;
        if (!state.GetHistoricalCandle(i, current_historical_candle)) 
        {
            if (EnableLogging_Campionamento)
                PrintFormat("DEBUG HistoricalAnalysis - Candela[%d] non disponibile. Continuo con le altre.", i);
            continue; // Continua invece di interrompere
        }

        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG HistoricalAnalysis - Candela[%d]: Score=%.2f, Dir=%s, BodyRatio=%.3f, Reversal=%s",
                        i, current_historical_candle.score, 
                        current_historical_candle.direction ? "BUY" : "SELL",
                        current_historical_candle.bodyRatio, 
                        current_historical_candle.isReversal ? "TRUE" : "FALSE");

        // Processa solo candele con score significativo
        if (current_historical_candle.score > 0.5) // Soglia minima per considerare la candela
        {
            validCandlesProcessed++;
            totalHistoricalScore += current_historical_candle.score;
            avgBodyRatio += current_historical_candle.bodyRatio;
            
            // Conta voti direzione
            if (current_historical_candle.direction)
                buyVotes++;
            else
                sellVotes++;
            
            // Conta reversal
            if (current_historical_candle.isReversal)
                reversalCount++;
            
            // 🔥 MIGLIORAMENTO #2: Analisi Trend Consecutivi Semplificata
            if (validCandlesProcessed == 1) // Prima candela valida
            {
                currentHistoricalTrendDirection = current_historical_candle.direction;
                consecutiveTrendCandles = 1;
            }
            else
            {
                if (current_historical_candle.direction == currentHistoricalTrendDirection)
                {
                    consecutiveTrendCandles++;
                }
                else
                {
                    // Trend si interrompe, ma continuiamo l'analisi
                    if (EnableLogging_Campionamento)
                        PrintFormat("DEBUG HistoricalAnalysis - Trend interrotto a candela[%d]. Consecutive=%d", i, consecutiveTrendCandles);
                    // Non resettiamo, manteniamo il massimo consecutivo raggiunto
                }
            }

            if (EnableLogging_Campionamento)
                PrintFormat("DEBUG HistoricalAnalysis - Candela[%d] processata. Consecutive=%d, Total Score=%.2f", 
                            i, consecutiveTrendCandles, totalHistoricalScore);
        }
        else
        {
            if (EnableLogging_Campionamento)
                PrintFormat("DEBUG HistoricalAnalysis - Candela[%d] score troppo basso (%.2f <= 0.5). Saltata.", 
                            i, current_historical_candle.score);
        }
    }

    if (validCandlesProcessed == 0)
    {
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG HistoricalAnalysis - Nessuna candela valida trovata. Score = 0.0.");
        return 0.0;
    }

    avgBodyRatio /= validCandlesProcessed;

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG HistoricalAnalysis - Riepilogo: ValidCandles=%d, TotalScore=%.2f, BuyVotes=%d, SellVotes=%d, Consecutive=%d, AvgBodyRatio=%.3f, Reversals=%d", 
                    validCandlesProcessed, totalHistoricalScore, buyVotes, sellVotes, consecutiveTrendCandles, avgBodyRatio, reversalCount);

    // 🔥 MIGLIORAMENTO #3: Sistema di Scoring Bilanciato

    // COMPONENTE A: Bonus per Trend Consolidato (0 a +40% del peso)
    if (consecutiveTrendCandles >= CampionamentoConsecutiveCandlesMin)
    {
        double trendScore = CampionamentoHistoricalAnalysisWeight * 0.4 * MathMin(1.0, (double)consecutiveTrendCandles / (CampionamentoConsecutiveCandlesMin * 2));
        score += trendScore;
        
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG HistoricalAnalysis - TREND CONSOLIDATO: %d candele consecutive → +%.3f", consecutiveTrendCandles, trendScore);
    }

    // COMPONENTE B: Coerenza Direzione (0 a +/-30% del peso)
    double directionCoherence = 0.0;
    if (buyVotes > sellVotes)
    {
        directionCoherence = CampionamentoHistoricalAnalysisWeight * 0.3 * ((double)(buyVotes - sellVotes) / validCandlesProcessed);
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG HistoricalAnalysis - COERENZA BULLISH: %d vs %d → +%.3f", buyVotes, sellVotes, directionCoherence);
    }
    else if (sellVotes > buyVotes)
    {
        directionCoherence = -CampionamentoHistoricalAnalysisWeight * 0.3 * ((double)(sellVotes - buyVotes) / validCandlesProcessed);
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG HistoricalAnalysis - COERENZA BEARISH: %d vs %d → %.3f", sellVotes, buyVotes, directionCoherence);
    }
    score += directionCoherence;

    // COMPONENTE C: Qualità delle Candele (0 a +/-20% del peso)
    double qualityScore = 0.0;
    if (avgBodyRatio >= CampionamentoBodyRangeThreshold)
    {
        qualityScore = CampionamentoHistoricalAnalysisWeight * 0.2 * (avgBodyRatio - CampionamentoBodyRangeThreshold) / (1.0 - CampionamentoBodyRangeThreshold);
        qualityScore = MathMin(CampionamentoHistoricalAnalysisWeight * 0.2, qualityScore);
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG HistoricalAnalysis - ALTA QUALITÀ candele: AvgBodyRatio=%.3f → +%.3f", avgBodyRatio, qualityScore);
    }
    else
    {
        qualityScore = -CampionamentoHistoricalAnalysisWeight * 0.1 * (CampionamentoBodyRangeThreshold - avgBodyRatio) / CampionamentoBodyRangeThreshold;
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG HistoricalAnalysis - BASSA QUALITÀ candele: AvgBodyRatio=%.3f → %.3f", avgBodyRatio, qualityScore);
    }
    score += qualityScore;

    // COMPONENTE D: Penalità per Troppi Reversal (0 a -20% del peso)
    if (reversalCount > validCandlesProcessed * 0.3) // Più del 30% di reversal
    {
        double reversalPenalty = -CampionamentoHistoricalAnalysisWeight * 0.2 * ((double)reversalCount / validCandlesProcessed);
        score += reversalPenalty;
        
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG HistoricalAnalysis - TROPPI REVERSAL: %d/%d (%.1f%%) → %.3f", 
                        reversalCount, validCandlesProcessed, ((double)reversalCount/validCandlesProcessed)*100, reversalPenalty);
    }

    // 🔥 MIGLIORAMENTO #4: Bonus per Score Storico Alto (0 a +10% del peso)
    double avgHistoricalScore = totalHistoricalScore / validCandlesProcessed;
    if (avgHistoricalScore >= 7.0) // Score medio alto
    {
        double strengthBonus = CampionamentoHistoricalAnalysisWeight * 0.1 * (avgHistoricalScore - 7.0) / 3.0; // Bonus fino a +10%
        score += strengthBonus;
        
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG HistoricalAnalysis - FORZA STORICA: AvgScore=%.1f → +%.3f", avgHistoricalScore, strengthBonus);
    }

    // 🔥 MIGLIORAMENTO #5: Cappatura Finale usando CampionamentoHistoricalAnalysisWeight
    score = MathMax(-CampionamentoHistoricalAnalysisWeight, MathMin(CampionamentoHistoricalAnalysisWeight, score));

    // Filtraggio rumore
    if (MathAbs(score) < 0.1)
    {
        score = 0.0;
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG HistoricalAnalysis - Score troppo piccolo (<0.1). Azzerato.");
    }

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG HistoricalAnalysis - RISULTATO FINALE: %d candele valide, Score=%.3f", validCandlesProcessed, score);

    return score;
}

//+------------------------------------------------------------------+
//| 🧭 Determina la direzione complessiva (BUY=true, SELL=false) - VERSIONE CORRETTA
//+------------------------------------------------------------------+
bool DetermineOverallDirection(string symbol, ENUM_TIMEFRAMES tf, double currentPrice, const CampionamentoState &state)
{
    // Debug iniziale della funzione
    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG OverallDirection - Inizio. Symbol: %s, TF: %s, CurrentPrice: %.5f", symbol, EnumToString(tf), currentPrice);

    bool determinedDirection = state.finalDirection; // 🔥 FIX #1: Usa direzione precedente invece di false

    // --- NUOVA LOGICA DI REVERSAL / BREAKOUT (PRIORITARIA CON CONTESTO DI TREND) ---
    MqlRates rates[3]; // 🔥 FIX #2: Usa 3 candele per avere più contesto
    if(CopyRates(symbol, tf, 0, 3, rates) == 3)
    {
        double currentCandleClose = rates[0].close;
        double previousCandleOpen = rates[1].open;
        double previousCandleClose = rates[1].close; // 🔥 FIX #3: Aggiungi chiusura precedente
        double thresholdPriceMovement = CampionamentoReversalBreakThresholdPoints * _Point;

        // 🔥 FIX #4: Usa la direzione della candela precedente, non dello stato
        bool previousCandleDirection = (rates[1].close > rates[1].open); // TRUE = BUY, FALSE = SELL
        
        if (EnableLogging_Campionamento)
            PrintFormat("DEBUG OverallDirection - Candele: Current[0] O=%.5f C=%.5f, Previous[1] O=%.5f C=%.5f, Direction=%s", 
                        rates[0].open, rates[0].close, rates[1].open, rates[1].close, 
                        previousCandleDirection ? "BUY" : "SELL");

        // 🔥 FIX #5: Condizioni di reversal semplificate e più robuste
        
        // LONG REVERSAL: 
        // - Prezzo attuale rompe significativamente sopra la chiusura precedente
        // - E la candela precedente era ribassista (SELL)
        if (currentPrice > previousCandleClose + thresholdPriceMovement && previousCandleDirection == false)
        {
            determinedDirection = true; // Forza BUY
            if (EnableLogging_Campionamento)
                PrintFormat("🟢 LONG REVERSAL! CurrentPrice (%.5f) > PrevClose (%.5f) + Threshold (%.5f) AND PrevDirection=SELL → BUY",
                            currentPrice, previousCandleClose, thresholdPriceMovement);
            return determinedDirection;
        }
        
        // SHORT REVERSAL:
        // - Prezzo attuale rompe significativamente sotto la chiusura precedente  
        // - E la candela precedente era rialzista (BUY)
        else if (currentPrice < previousCandleClose - thresholdPriceMovement && previousCandleDirection == true)
        {
            determinedDirection = false; // Forza SELL
            if (EnableLogging_Campionamento)
                PrintFormat("🔴 SHORT REVERSAL! CurrentPrice (%.5f) < PrevClose (%.5f) - Threshold (%.5f) AND PrevDirection=BUY → SELL",
                            currentPrice, previousCandleClose, thresholdPriceMovement);
            return determinedDirection;
        }
        
        // 🔥 FIX #6: Aggiunge logica di continuazione trend
        // Se non c'è reversal, verifica se continuare il trend esistente
        else
        {
            // Continua trend rialzista se il prezzo è sopra la chiusura precedente
            if (currentPrice > previousCandleClose && previousCandleDirection == true)
            {
                determinedDirection = true;
                if (EnableLogging_Campionamento)
                    PrintFormat("📈 TREND CONTINUATION BUY: CurrentPrice (%.5f) > PrevClose (%.5f) AND PrevDirection=BUY",
                                currentPrice, previousCandleClose);
                return determinedDirection;
            }
            // Continua trend ribassista se il prezzo è sotto la chiusura precedente
            else if (currentPrice < previousCandleClose && previousCandleDirection == false)
            {
                determinedDirection = false;
                if (EnableLogging_Campionamento)
                    PrintFormat("📉 TREND CONTINUATION SELL: CurrentPrice (%.5f) < PrevClose (%.5f) AND PrevDirection=SELL",
                                currentPrice, previousCandleClose);
                return determinedDirection;
            }
        }
    }
    else
    {
        if (EnableLogging_Campionamento)
            Print("DEBUG OverallDirection - Impossibile recuperare dati candele. Continuo con logiche alternative.");
    }

    // 🔥 FIX #7: Logica spike migliorata con soglia più bassa
    if (state.consecutiveDirectionTicks >= MathMax(2, CampionamentoConsecutiveCandlesMin / 2)) // Riduce soglia
    {
        determinedDirection = state.currentDirection;
        if (EnableLogging_Campionamento)
            PrintFormat("🚀 SPIKE DETECTED: %d ticks consecutivi in direzione %s", 
                        state.consecutiveDirectionTicks, determinedDirection ? "BUY" : "SELL");
        return determinedDirection;
    }

    // Se nessuna condizione precedente è soddisfatta, usa la direzione storica
    int historicalBuyVotes = 0;
    int historicalSellVotes = 0;
    
    for (int i = 0; i < CampionamentoThresholdLookback; i++)
    {
        CandelaStorica current_historical_candle;
        if (!state.GetHistoricalCandle(i, current_historical_candle))
            break;
        
        if (current_historical_candle.direction)
            historicalBuyVotes++;
        else
            historicalSellVotes++;
    }
    
    // 🔥 FIX #8: Decisione più decisiva per evitare stallo
    if (historicalBuyVotes > historicalSellVotes)
    {
        determinedDirection = true;
        if (EnableLogging_Campionamento)
            PrintFormat("📊 HISTORICAL BUY: %d vs %d votes", historicalBuyVotes, historicalSellVotes);
    }
    else if (historicalSellVotes > historicalBuyVotes)
    {
        determinedDirection = false;
        if (EnableLogging_Campionamento)
            PrintFormat("📊 HISTORICAL SELL: %d vs %d votes", historicalSellVotes, historicalBuyVotes);
    }
    // Se sono uguali, mantieni la direzione precedente (già impostata all'inizio)
    
    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG OverallDirection - Direzione finale: %s", determinedDirection ? "BUY" : "SELL");

    return determinedDirection;
}

//+--------------------------------------------------------------------------------------------+
//| IsReversalCandle - Determina se una candela è di reversal avanzato - VERSIONE CORRETTA  |
//+--------------------------------------------------------------------------------------------+
bool IsReversalCandle(
    double closedOpen, double closedHigh, double closedLow, double closedClose,
    bool closedCandleDirection,
    const CampionamentoState &state
)
{
    // Parametri di configurazione
    const double REVERSAL_MAX_BODY_RATIO = CampionamentoReversalMaxBodyRatio;        // Es: 0.3 (30%)
    const double REVERSAL_MIN_SHADOW_RATIO = CampionamentoReversalMinShadowRatio;    // Es: 0.5 (50%)
    const int TREND_LOOKBACK = CampionamentoReversalTrendLookback;                   // Es: 5 candele
    const double MIN_TREND_PERCENT = CampionamentoReversalMinTrendPercent;           // Es: 0.6 (60%)

    double closedCandleBody = MathAbs(closedClose - closedOpen);
    double closedCandleRange = closedHigh - closedLow;

    // Calcolo delle ombre
    double upperShadow = closedHigh - MathMax(closedOpen, closedClose);
    double lowerShadow = MathMin(closedOpen, closedClose) - closedLow;

    // Evita divisione per zero
    if (closedCandleRange <= 0) {
        if (EnableLogging_Campionamento)    
            PrintFormat("DEBUG IsReversalCandle: Range della candela <= 0. Non un reversal.");
        return false;
    }

    // --- Parte 1: Verifica del Pattern Base (LOGICA CORRETTA) ---
    bool isSingleCandleReversalPattern = false;
    double currentBodyRatio = closedCandleBody / closedCandleRange;
    double currentUpperShadowRatio = upperShadow / closedCandleRange;
    double currentLowerShadowRatio = lowerShadow / closedCandleRange;

    if (EnableLogging_Campionamento)
        PrintFormat("DEBUG IsReversalCandle: BodyRatio=%.3f, UpperShadow=%.3f, LowerShadow=%.3f", 
                    currentBodyRatio, currentUpperShadowRatio, currentLowerShadowRatio);

    // 🔥 CONDIZIONE 1: Body piccolo (caratteristica comune ai pattern di reversal)
    if (currentBodyRatio <= REVERSAL_MAX_BODY_RATIO)
    {
        // 🔥 HAMMER/DOJI RIALZISTA (Reversal da trend ribassista)
        // - Ombra inferiore lunga (testare il supporto)
        // - Ombra superiore piccola o assente
        // - Direzione della candela NON importante (può essere sia BUY che SELL)
        if (currentLowerShadowRatio >= REVERSAL_MIN_SHADOW_RATIO && 
            currentUpperShadowRatio <= (1.0 - REVERSAL_MIN_SHADOW_RATIO))
        {
            isSingleCandleReversalPattern = true;
            if (EnableLogging_Campionamento)    
                PrintFormat("DEBUG IsReversalCandle: Rilevato pattern HAMMER/DOJI (potenziale reversal rialzista).");
        }
        // 🔥 SHOOTING STAR RIBASSISTA (Reversal da trend rialzista)
        // - Ombra superiore lunga (testare la resistenza)
        // - Ombra inferiore piccola o assente
        // - Direzione della candela NON importante (può essere sia BUY che SELL)
        else if (currentUpperShadowRatio >= REVERSAL_MIN_SHADOW_RATIO && 
                 currentLowerShadowRatio <= (1.0 - REVERSAL_MIN_SHADOW_RATIO))
        {
            isSingleCandleReversalPattern = true;
            if (EnableLogging_Campionamento)    
                PrintFormat("DEBUG IsReversalCandle: Rilevato pattern SHOOTING STAR (potenziale reversal ribassista).");
        }
    }

    if (!isSingleCandleReversalPattern) {
        if (EnableLogging_Campionamento)    
            PrintFormat("DEBUG IsReversalCandle: Nessun pattern base di reversal. BodyRatio=%.3f troppo grande o ombre non significative.", 
                        currentBodyRatio);
        return false;
    }

    // --- Parte 2: Conferma Contestuale del Trend (LOGICA MIGLIORATA) ---
    int historySize = ArraySize(state.historicalCandles);
    
    if (historySize < TREND_LOOKBACK) {
        if (EnableLogging_Campionamento)    
            PrintFormat("DEBUG IsReversalCandle: Non abbastanza candele storiche (%d < %d). Accetto il pattern base.", 
                        historySize, TREND_LOOKBACK);
        return true; // 🔥 Se non abbiamo storia sufficiente, accettiamo il pattern base
    }

    // 🔥 ANALISI DEL TREND PRECEDENTE
    int bullishCandlesCount = 0;
    int bearishCandlesCount = 0;
    double avgHistoricalScore = 0.0;
    int validCandlesCount = 0;

    for (int i = 0; i < TREND_LOOKBACK; i++)
    {
        CandelaStorica current_historical_candle;
        if (!state.GetHistoricalCandle(i, current_historical_candle)) 
        {
            if (EnableLogging_Campionamento)    
                PrintFormat("DEBUG IsReversalCandle: Candela storica[%d] non disponibile. Continuo con le altre.", i);
            continue; // 🔥 Continua invece di interrompere
        }
        
        validCandlesCount++;
        avgHistoricalScore += current_historical_candle.score;
        
        if (current_historical_candle.direction == true)
            bullishCandlesCount++;
        else
            bearishCandlesCount++;
            
        if (EnableLogging_Campionamento)    
            PrintFormat("DEBUG IsReversalCandle: Candela[%d] Dir=%s, Score=%.2f", 
                        i, current_historical_candle.direction ? "BUY" : "SELL", 
                        current_historical_candle.score);
    }

    if (validCandlesCount == 0) {
        if (EnableLogging_Campionamento)    
            PrintFormat("DEBUG IsReversalCandle: Nessuna candela storica valida. Accetto il pattern base.");
        return true;
    }

    avgHistoricalScore /= validCandlesCount;
    double bullishPercent = (double)bullishCandlesCount / validCandlesCount;
    double bearishPercent = (double)bearishCandlesCount / validCandlesCount;

    if (EnableLogging_Campionamento)    
        PrintFormat("DEBUG IsReversalCandle: Trend Analysis - Bull:%.1f%%, Bear:%.1f%%, AvgScore:%.2f", 
                    bullishPercent * 100, bearishPercent * 100, avgHistoricalScore);

    // 🔥 LOGICA DI CONFERMA MIGLIORATA
    bool isValidReversal = false;
    
    // Se abbiamo rilevato un HAMMER/DOJI (reversal rialzista potenziale)
    if (currentLowerShadowRatio >= REVERSAL_MIN_SHADOW_RATIO)
    {
        // Conferma se il trend precedente era prevalentemente ribassista
        if (bearishPercent >= MIN_TREND_PERCENT)
        {
            isValidReversal = true;
            if (EnableLogging_Campionamento)    
                PrintFormat("DEBUG IsReversalCandle: HAMMER confermato! Trend precedente %.1f%% ribassista.", bearishPercent * 100);
        }
    }
    // Se abbiamo rilevato un SHOOTING STAR (reversal ribassista potenziale)
    else if (currentUpperShadowRatio >= REVERSAL_MIN_SHADOW_RATIO)
    {
        // Conferma se il trend precedente era prevalentemente rialzista
        if (bullishPercent >= MIN_TREND_PERCENT)
        {
            isValidReversal = true;
            if (EnableLogging_Campionamento)    
                PrintFormat("DEBUG IsReversalCandle: SHOOTING STAR confermato! Trend precedente %.1f%% rialzista.", bullishPercent * 100);
        }
    }

    // 🔥 BONUS: Considera anche l'intensità del trend (score medio)
    if (isValidReversal && avgHistoricalScore >= 3.0) // Se il trend precedente era forte
    {
        if (EnableLogging_Campionamento)    
            PrintFormat("DEBUG IsReversalCandle: Reversal FORTE confermato! Avg Score precedente: %.2f", avgHistoricalScore);
        return true;
    }
    else if (isValidReversal)
    {
        if (EnableLogging_Campionamento)    
            PrintFormat("DEBUG IsReversalCandle: Reversal DEBOLE confermato. Avg Score precedente: %.2f", avgHistoricalScore);
        return true;
    }
    else
    {
        if (EnableLogging_Campionamento)    
            PrintFormat("DEBUG IsReversalCandle: Pattern non confermato dal trend precedente. Bull:%.1f%%, Bear:%.1f%%", 
                        bullishPercent * 100, bearishPercent * 100);
        return false;
    }
}

//+------------------------------------------------------------------+
//| 🚀 API Pubbliche → da usare in EntryManager - VERSIONE OTTIMIZZATA
//+------------------------------------------------------------------+

// Ritorna lo score attuale del campionamento (0-10)
double GetSamplingScore(string symbol, ENUM_TIMEFRAMES tf)
{
    string key = symbol + EnumToString(tf);
    CampionamentoState state_copy;
    
    if (M_CampionamentoState.Get(key, state_copy))
    {
        return state_copy.finalNormalizedScore;
    }
    
    return 0.0; // Stato non esistente
}

// Ritorna la direzione attuale rilevata dal campionamento
ENUM_ORDER_TYPE GetSamplingDirection(string symbol, ENUM_TIMEFRAMES tf)
{
    string key = symbol + EnumToString(tf);
    CampionamentoState state_copy;
    
    if (M_CampionamentoState.Get(key, state_copy))
    {
        return (state_copy.finalDirection ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
    }
    
    return (ENUM_ORDER_TYPE)-1; // Errore: nessuna direzione valida
}

// Ritorna il log attuale del campionamento
string GetSamplingLog(string symbol, ENUM_TIMEFRAMES tf)
{
    string key = symbol + EnumToString(tf);
    CampionamentoState state_copy;
    
    if (M_CampionamentoState.Get(key, state_copy))
    {
        return state_copy.logText;
    }
    
    return "⚪️ Campionamento non inizializzato";
}

// 🔥 NUOVE API AGGIUNTIVE per funzionalità avanzate

// Ritorna il raw score (prima della normalizzazione)
double GetSamplingRawScore(string symbol, ENUM_TIMEFRAMES tf)
{
    string key = symbol + EnumToString(tf);
    CampionamentoState state_copy;
    
    if (M_CampionamentoState.Get(key, state_copy))
    {
        return state_copy.scoreTotalRaw;
    }
    
    return 0.0;
}

// Ritorna il livello di confidenza (0.0-1.0) basato sulla coerenza dei segnali
double GetSamplingConfidence(string symbol, ENUM_TIMEFRAMES tf)
{
    string key = symbol + EnumToString(tf);
    CampionamentoState state_copy;
    
    if (M_CampionamentoState.Get(key, state_copy))
    {
        // Calcolo confidenza basato su coerenza componenti
        double confidence = 0.5; // Default neutrale
        
        // Fattori che aumentano la confidenza:
        if (state_copy.consecutiveDirectionTicks >= 3) confidence += 0.2; // Movimento consistente
        if (state_copy.finalNormalizedScore > 7.0 || state_copy.finalNormalizedScore < 3.0) confidence += 0.2; // Score estremi
        if (MathAbs(state_copy.scoreTotalRaw) > 5.0) confidence += 0.1; // Raw score significativo
        
        return MathMin(1.0, confidence);
    }
    
    return 0.0;
}

// Verifica se i dati sono aggiornati di recente
bool IsSamplingDataFresh(string symbol, ENUM_TIMEFRAMES tf, int maxAgeMs = 5000)
{
    string key = symbol + EnumToString(tf);
    CampionamentoState state_copy;
    
    if (M_CampionamentoState.Get(key, state_copy))
    {
        long currentTime = GetTickCount();
        long timeSinceUpdate = currentTime - state_copy.lastUpdateTime;
        return (timeSinceUpdate <= maxAgeMs);
    }
    
    return false; // Dati non esistenti = non fresh
}

// Ritorna il numero di tick consecutivi nella direzione attuale
int GetSamplingMomentum(string symbol, ENUM_TIMEFRAMES tf)
{
    string key = symbol + EnumToString(tf);
    CampionamentoState state_copy;
    
    if (M_CampionamentoState.Get(key, state_copy))
    {
        return state_copy.consecutiveDirectionTicks;
    }
    
    return 0;
}

// 🎯 API AVANZATA: Ritorna analisi completa in una chiamata
struct SamplingAnalysis
{
    bool isValid;           // True se i dati sono validi
    double score;           // Score normalizzato 0-10
    double rawScore;        // Score raw
    bool direction;         // True=BUY, False=SELL
    double confidence;      // Confidenza 0-1
    int momentum;           // Tick consecutivi
    bool isFresh;           // Dati recenti
    string status;          // Descrizione stato
};

SamplingAnalysis GetSamplingAnalysis(string symbol, ENUM_TIMEFRAMES tf)
{
    SamplingAnalysis analysis;
    analysis.isValid = false;
    analysis.score = 0.0;
    analysis.rawScore = 0.0;
    analysis.direction = false;
    analysis.confidence = 0.0;
    analysis.momentum = 0;
    analysis.isFresh = false;
    analysis.status = "Data not available";
    
    string key = symbol + EnumToString(tf);
    CampionamentoState state_copy;
    
    if (M_CampionamentoState.Get(key, state_copy))
    {
        analysis.isValid = true;
        analysis.score = state_copy.finalNormalizedScore;
        analysis.rawScore = state_copy.scoreTotalRaw;
        analysis.direction = state_copy.finalDirection;
        analysis.momentum = state_copy.consecutiveDirectionTicks;
        
        // Calcola confidenza
        analysis.confidence = 0.5;
        if (state_copy.consecutiveDirectionTicks >= 3) analysis.confidence += 0.2;
        if (analysis.score > 7.0 || analysis.score < 3.0) analysis.confidence += 0.2;
        if (MathAbs(analysis.rawScore) > 5.0) analysis.confidence += 0.1;
        analysis.confidence = MathMin(1.0, analysis.confidence);
        
        // Verifica freschezza
        long timeSinceUpdate = GetTickCount() - state_copy.lastUpdateTime;
        analysis.isFresh = (timeSinceUpdate <= 5000);
        
        // Status description
        if (!analysis.isFresh) {
            analysis.status = StringFormat("Stale data (%.1fs old)", (double)timeSinceUpdate/1000.0);
        } else if (analysis.confidence > 0.8) {
            analysis.status = StringFormat("High confidence %s signal", analysis.direction ? "BUY" : "SELL");
        } else if (analysis.confidence > 0.6) {
            analysis.status = StringFormat("Medium confidence %s signal", analysis.direction ? "BUY" : "SELL");
        } else {
            analysis.status = "Low confidence signal";
        }
    }
    
    return analysis;
}

#endif // __CAMPIONAMENTO_MQH__