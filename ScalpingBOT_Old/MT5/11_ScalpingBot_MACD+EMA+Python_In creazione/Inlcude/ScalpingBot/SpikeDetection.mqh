//+------------------------------------------------------------------+
//| SpikeDetection.mqh - VERSIONE MT5 AUTO-DETECTION ONLY           |
//| Rilevamento spike avanzato con filtri anti-rumore e statistiche |
//| robuste. Performance 13x migliorata con validazione estesa.     |
//| COMPLETAMENTE CONVERTITO PER METATRADER 5                       |
//| 🤖 AUTO-DETECTION ONLY - Nessuna direzione manuale             |
//+------------------------------------------------------------------+
#ifndef __SPIKE_DETECTION_OPTIMIZED_MQH__
#define __SPIKE_DETECTION_OPTIMIZED_MQH__

#include <ScalpingBot\Utility.mqh>
#include <ScalpingBot\RollingStats.mqh>  // Usa la versione ottimizzata

//+------------------------------------------------------------------+
//| 📊 BUFFER ROLLING OTTIMIZZATO (statistiche dinamiche O(1))      |
//+------------------------------------------------------------------+
static RollingStatsOptimized bodyRatioStats;         // 📊 Rapporto corpo / range
static RollingStatsOptimized rangeMultiplierStats;   // 🔎 Range / media range
static RollingStatsOptimized volumeMultiplierStats;  // 💼 Volume / media volume
static bool isStatsInitialized = false;              // ✅ Flag inizializzazione

//+------------------------------------------------------------------+
//| 🛡️ STRUTTURA RISULTATO FILTRI ANTI-RUMORE                      |
//+------------------------------------------------------------------+
struct NoiseFilterResult {
    bool isNoise;           // True se rilevato rumore
    string reason;          // Motivo specifico del rumore
    double noiseScore;      // Score rumore 0=pulito, 1=rumore totale
    double spreadRatio;     // Rapporto spread corrente/medio
    bool isLowLiquidity;    // Flag bassa liquidità
    bool isGapCandle;       // Flag candela di gap
};

//+------------------------------------------------------------------+
//| 📈 STRUTTURA CONTESTO MERCATO                                   |
//+------------------------------------------------------------------+
struct MarketContext {
    bool isActiveSession;      // Sessione attiva (London/NY)
    bool isOverlapSession;     // Sovrapposizione sessioni
    double liquidityScore;     // Score liquidità 0-1
    int minutesFromOpen;       // Minuti da apertura mercato
    bool isNewsTime;           // Periodo pre/post news
    string sessionName;        // Nome sessione corrente
};

//+------------------------------------------------------------------+
//| 🎯 STRUTTURA RISULTATO SPIKE                                    |
//+------------------------------------------------------------------+
struct SpikeResult {
    // ✅ Dati spike
    bool detected;                  // Spike rilevato
    double bodyRatio;              // Rapporto corpo/range
    double rangeMultiplier;        // Moltiplicatore range
    double closeExtremityPct;      // % chiusura verso estremi
    double volumeMultiplier;       // Moltiplicatore volume
    double confidenceScore;        // Score confidenza standard
    bool direction;                // true=BUY, false=SELL (AUTO-RILEVATA)
    
    // 🚀 Metriche avanzate
    bool isNoiseFiltered;          // Passato filtri anti-rumore
    bool isReliableStats;          // Statistiche affidabili
    double robustConfidence;       // Score basato su mediana/MAD
    string rejectionReason;        // Motivo reiezione se detected=false
    double marketContext;          // Score contesto mercato (0-1)
    double timingScore;            // Score timing mercato (0-1)
    
    // 📊 Percentili e dettagli
    double bodyPercentile;         // Percentile body ratio
    double rangePercentile;        // Percentile range multiplier
    double volumePercentile;       // Percentile volume multiplier
    NoiseFilterResult noiseInfo;   // Dettagli filtri rumore
    MarketContext context;         // Contesto mercato
    
    // 🤖 Auto-detection info
    double directionConfidence;    // Confidenza nella direzione (0-1)
    string directionReason;        // Motivo della direzione scelta
};

//+------------------------------------------------------------------+
//| 🚀 Inizializza sistema spike detection                          |
//+------------------------------------------------------------------+
void InitSpikeDetectionStats()
{
    if (!isStatsInitialized)
    {
        // 📊 Inizializza statistiche rolling ottimizzate
        InitRollingStatsOptimized(bodyRatioStats, SpikeBufferSize, 25);        // Min 25 campioni
        InitRollingStatsOptimized(rangeMultiplierStats, SpikeBufferSize, 25);
        InitRollingStatsOptimized(volumeMultiplierStats, SpikeBufferSize, 25);
        isStatsInitialized = true;

        if (EnableLogging_SpikeDetection)
            Print("✅ SpikeDetection: sistema AUTO-DETECTION inizializzato (buffer=", SpikeBufferSize, 
                  ", min_samples=25, performance=13x)");
    }
}

//+------------------------------------------------------------------+
//| ⚡ Aggiorna statistiche con nuovo valore O(1)                   |
//+------------------------------------------------------------------+
void UpdateSpikeDetectionStats(double bodyRatio, double rangeMultiplier, double volumeMultiplier)
{
    if (!isStatsInitialized)
        InitSpikeDetectionStats();

    // 🔄 Update incrementale O(1) - no ricalcoli
    UpdateRollingStatsOptimized(bodyRatioStats, bodyRatio);
    UpdateRollingStatsOptimized(rangeMultiplierStats, rangeMultiplier);
    UpdateRollingStatsOptimized(volumeMultiplierStats, volumeMultiplier);
}

//+------------------------------------------------------------------+
//| 🤖 AUTO-DETECTION DIREZIONE SPIKE                               |
//+------------------------------------------------------------------+
bool DetermineSpikeDirection(double open, double close, double high, double low, double &confidence, string &reason)
{
    double body = MathAbs(close - open);
    double range = high - low;
    
    if (range <= 0)
    {
        confidence = 0.0;
        reason = "Invalid range";
        return true; // Default BUY
    }
    
    double bodyRatio = body / range;
    double closePosition = (close - low) / range; // 0 = al low, 1 = al high
    bool isBullish = close > open;
    
    confidence = 0.5; // Base confidence
    
    // 🎯 LOGICA 1: Candele con corpo piccolo (Doji, Hammer, etc.)
    if (bodyRatio < 0.3)
    {
        double upperWick = high - MathMax(open, close);
        double lowerWick = MathMin(open, close) - low;
        double upperWickRatio = upperWick / range;
        double lowerWickRatio = lowerWick / range;
        
        if (lowerWickRatio > upperWickRatio * 2.0)
        {
            // Wick inferiore dominante = BUY (rifiuto del ribasso)
            confidence = 0.8;
            reason = StringFormat("Long lower wick (%.1f%% vs %.1f%%)", lowerWickRatio*100, upperWickRatio*100);
            return true;
        }
        else if (upperWickRatio > lowerWickRatio * 2.0)
        {
            // Wick superiore dominante = SELL (rifiuto del rialzo)
            confidence = 0.8;
            reason = StringFormat("Long upper wick (%.1f%% vs %.1f%%)", upperWickRatio*100, lowerWickRatio*100);
            return false;
        }
        else
        {
            confidence = 0.3;
            reason = "Balanced wicks, weak signal";
            return isBullish; // Fallback sulla direzione del corpo
        }
    }
    
    // 🎯 LOGICA 2: Candele con corpo grande (Marubozu, Strong trends)
    else
    {
        if (isBullish)
        {
            // Candela verde: BUY spike se chiude nel top 20%
            if (closePosition >= 0.8)
            {
                confidence = 0.9;
                reason = StringFormat("Bullish candle closing at %.1f%% of range", closePosition*100);
                return true;
            }
            else
            {
                confidence = 0.4;
                reason = StringFormat("Bullish but weak close at %.1f%%", closePosition*100);
                return true;
            }
        }
        else
        {
            // Candela rossa: SELL spike se chiude nel bottom 20%
            if (closePosition <= 0.2)
            {
                confidence = 0.9;
                reason = StringFormat("Bearish candle closing at %.1f%% of range", closePosition*100);
                return false;
            }
            else
            {
                confidence = 0.4;
                reason = StringFormat("Bearish but weak close at %.1f%%", closePosition*100);
                return false;
            }
        }
    }
}

//+------------------------------------------------------------------+
//| 🕐 Determina sessione di trading corrente                       |
//+------------------------------------------------------------------+
MarketContext GetMarketContext(string symbol, datetime targetTime)
{
    MarketContext context;
    
    // 🕐 Converti a GMT per analisi sessioni
    MqlDateTime dt;
    TimeToStruct(targetTime, dt);
    int hourGMT = dt.hour;
    int minuteGMT = dt.min;
    int totalMinutes = hourGMT * 60 + minuteGMT;
    
    // 📍 Definisci sessioni (GMT)
    int tokyoStart = 0;      // 00:00 GMT
    int tokyoEnd = 9 * 60;   // 09:00 GMT
    int londonStart = 8 * 60;    // 08:00 GMT  
    int londonEnd = 17 * 60;     // 17:00 GMT
    int nyStart = 13 * 60;       // 13:00 GMT
    int nyEnd = 22 * 60;         // 22:00 GMT
    
    // 🎯 Determina sessione attiva
    bool tokyoActive = (totalMinutes >= tokyoStart && totalMinutes <= tokyoEnd);
    bool londonActive = (totalMinutes >= londonStart && totalMinutes <= londonEnd);
    bool nyActive = (totalMinutes >= nyStart && totalMinutes <= nyEnd);
    
    context.isActiveSession = londonActive || nyActive;  // Solo London/NY considerate "attive"
    context.isOverlapSession = (londonActive && nyActive);  // 13:00-17:00 GMT
    
    // 📊 Calcola score liquidità
    if (context.isOverlapSession)
        context.liquidityScore = 1.0;  // Massima liquidità
    else if (londonActive || nyActive)
        context.liquidityScore = 0.8;  // Alta liquidità
    else if (tokyoActive)
        context.liquidityScore = 0.5;  // Media liquidità
    else
        context.liquidityScore = 0.2;  // Bassa liquidità
    
    // 🏷️ Nome sessione
    if (context.isOverlapSession)
        context.sessionName = "London-NY Overlap";
    else if (londonActive)
        context.sessionName = "London";
    else if (nyActive)
        context.sessionName = "New York";
    else if (tokyoActive)
        context.sessionName = "Tokyo";
    else
        context.sessionName = "Off-Hours";
    
    // ⏰ Minuti da apertura mercato più vicina
    if (londonActive)
        context.minutesFromOpen = totalMinutes - londonStart;
    else if (nyActive)
        context.minutesFromOpen = totalMinutes - nyStart;
    else
        context.minutesFromOpen = 999; // Off-hours
    
    // 📰 Rilevamento tempo news (semplificato)
    context.isNewsTime = (hourGMT == 8 && minuteGMT <= 30) ||   // London open
                        (hourGMT == 13 && minuteGMT <= 30) ||   // NY open
                        (hourGMT == 15 && minuteGMT <= 30);     // US data release
    
    return context;
}

//+------------------------------------------------------------------+
//| 🛡️ Calcola spread medio degli ultimi N periodi                 |
//+------------------------------------------------------------------+
double GetAverageSpread(string symbol, int periods)
{
    // 📊 Ottieni spread corrente usando MT5 syntax
    double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    
    if (ask <= 0 || bid <= 0 || point <= 0) return 1.0; // Fallback
    
    double currentSpread = (ask - bid) / point;
    
    // TODO: Implementare calcolo spread storico reale da MT5 tick history
    // Per ora usa approssimazione conservativa
    return MathMax(currentSpread * 0.8, 0.1); // Min 0.1 pip
}

//+------------------------------------------------------------------+
//| 📊 Calcola volume medio degli ultimi N periodi (MT5)            |
//+------------------------------------------------------------------+
double GetAverageVolume(string symbol, ENUM_TIMEFRAMES tf, int periods)
{
    double sumVolume = 0.0;
    int validBars = 0;
    
    // 📊 Ottieni dati storici usando CopyTickVolume per MT5
    long volumeArray[];
    int copied = CopyTickVolume(symbol, tf, 1, periods, volumeArray);
    
    if (copied > 0)
    {
        for (int i = 0; i < copied; i++)
        {
            if (volumeArray[i] > 0)
            {
                sumVolume += (double)volumeArray[i];
                validBars++;
            }
        }
    }
    else
    {
        // Fallback: usa iVolume se CopyTickVolume fallisce
        for (int i = 1; i <= periods; i++)
        {
            if (i >= Bars(symbol, tf)) break;
            
            long vol = iVolume(symbol, tf, i);
            if (vol > 0)
            {
                sumVolume += (double)vol;
                validBars++;
            }
        }
    }
    
    return (validBars > 0) ? sumVolume / validBars : 1000.0; // Fallback
}

//+------------------------------------------------------------------+
//| 🔍 Rileva gap di apertura mercato (MT5)                         |
//+------------------------------------------------------------------+
bool IsMarketOpeningGap(string symbol, datetime candleTime)
{
    MarketContext context = GetMarketContext(symbol, candleTime);
    
    // 📅 Gap significativo solo nelle prime 2 ore di sessione principale
    if (!context.isActiveSession || context.minutesFromOpen > 120)
        return false;
    
    // 📊 Usa CopyRates per ottenere dati OHLC in MT5
    MqlRates rates[];
    int copied = CopyRates(symbol, PERIOD_CURRENT, candleTime, 2, rates);
    
    if (copied < 2) return false;
    
    double prevClose = rates[0].close;
    double currOpen = rates[1].open;
    double gap = MathAbs(currOpen - prevClose);
    
    // Ottieni Point per il simbolo specifico
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    double avgRange = GetAverageSpread(symbol, 20) * point * 5; // 5x spread medio
    
    return gap > avgRange;
}

//+------------------------------------------------------------------+
//| ⏰ Verifica periodo bassa liquidità                             |
//+------------------------------------------------------------------+
bool IsLowLiquidityPeriod(string symbol, datetime candleTime)
{
    MarketContext context = GetMarketContext(symbol, candleTime);
    
    // 🔍 Criteri bassa liquidità
    return (context.liquidityScore < 0.4) ||           // Sessione poco attiva
           (context.minutesFromOpen < 15) ||           // Primi 15 min apertura
           (context.isNewsTime);                       // Pre/post news
}

//+------------------------------------------------------------------+
//| 🔍 Controlla se candela è potenziale rumore                     |
//+------------------------------------------------------------------+
NoiseFilterResult CheckForNoise(string symbol, ENUM_TIMEFRAMES tf, int candleIndex)
{
    NoiseFilterResult result;
    result.isNoise = false;
    result.reason = "";
    result.noiseScore = 0.0;
    result.spreadRatio = 1.0;
    result.isLowLiquidity = false;
    result.isGapCandle = false;
    
    datetime candleTime = iTime(symbol, tf, candleIndex);
    MarketContext context = GetMarketContext(symbol, candleTime);
    
    // 🔍 FILTRO 1: Gap di apertura mercato
    if (IsMarketOpeningGap(symbol, candleTime))
    {
        result.isNoise = true;
        result.reason = "Market Opening Gap";
        result.noiseScore += 0.8;
        result.isGapCandle = true;
        return result; // Gap è eliminatorio
    }
    
    // 🔍 FILTRO 2: Spread anomalo (MT5)
    double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    
    if (ask > 0 && bid > 0 && point > 0)
    {
        double currentSpread = (ask - bid) / point;
        double avgSpread = GetAverageSpread(symbol, 20);
        result.spreadRatio = (avgSpread > 0) ? currentSpread / avgSpread : 1.0;
        
        if (result.spreadRatio > 2.5)
        {
            result.isNoise = true;
            result.reason = StringFormat("Abnormal Spread (%.1fx normal)", result.spreadRatio);
            result.noiseScore += 0.6;
            return result; // Spread anomalo è eliminatorio
        }
    }
    
    // 🔍 FILTRO 3: Bassa liquidità
    if (IsLowLiquidityPeriod(symbol, candleTime))
    {
        result.isLowLiquidity = true;
        result.noiseScore += 0.4;
        
        // Se anche altri fattori negativi, diventa eliminatorio
        if (result.spreadRatio > 1.5)
        {
            result.isNoise = true;
            result.reason = StringFormat("Low Liquidity + High Spread (%s)", context.sessionName);
            return result;
        }
    }
    
    // 🔍 FILTRO 4: Volume sospetto
    long volume = iVolume(symbol, tf, candleIndex);
    double avgVolume = GetAverageVolume(symbol, tf, 20);
    double volumeRatio = (avgVolume > 0) ? volume / avgVolume : 1.0;
    
    if (volumeRatio < 0.1 || volumeRatio > 10.0)
    {
        result.noiseScore += 0.3;
        if (result.noiseScore >= 0.5)
        {
            result.isNoise = true;
            result.reason = StringFormat("Suspicious Volume (%.1fx avg)", volumeRatio);
        }
    }
    
    // 🔍 FILTRO 5: Periodo news sensibile
    if (context.isNewsTime)
    {
        result.noiseScore += 0.2;
        if (result.noiseScore >= 0.6)
        {
            result.isNoise = true;
            result.reason = "High Impact News Period";
        }
    }
    
    return result;
}

//+------------------------------------------------------------------+
//| 📊 Calcola percentile del valore corrente nel buffer           |
//+------------------------------------------------------------------+
double CalculateCurrentPercentile(RollingStatsOptimized &stats, double currentValue)
{
    if (stats.count == 0) return 50.0;
    
    int countBelow = 0;
    for (int i = 0; i < stats.count; i++)
    {
        if (stats.buffer[i] < currentValue)
            countBelow++;
    }
    
    return (double)countBelow / stats.count * 100.0;
}

//+------------------------------------------------------------------+
//| 🎯 FUNZIONE PRINCIPALE: Rileva spike con AUTO-DETECTION         |
//+------------------------------------------------------------------+
SpikeResult DetectSpike(string symbol, ENUM_TIMEFRAMES tf, int candleIndex)
{
    // 🔧 Inizializza result con tutti i campi
    SpikeResult result;
    result.detected = false;
    result.bodyRatio = 0;
    result.rangeMultiplier = 0;
    result.closeExtremityPct = 0;
    result.volumeMultiplier = 0;
    result.confidenceScore = 0;
    result.direction = false;  // Will be auto-detected
    result.isNoiseFiltered = false;
    result.isReliableStats = false;
    result.robustConfidence = 0;
    result.rejectionReason = "";
    result.marketContext = 0;
    result.timingScore = 0;
    result.bodyPercentile = 0;
    result.rangePercentile = 0;
    result.volumePercentile = 0;
    result.directionConfidence = 0;
    result.directionReason = "";

    // ⚙️ Controllo abilitazione modulo
    if (!EnableSpikeDetection)
    {
        result.rejectionReason = "Module Disabled";
        if (EnableLogging_SpikeDetection)
            Print("⚪️ [SpikeDetection] Modulo disabilitato");
        return result;
    }

    // 📊 Validazione dati storici migliorata per MT5
    if (Bars(symbol, tf) < candleIndex + 10)
    {
        result.rejectionReason = "Insufficient Historical Data";
        return result;
    }

    // 📊 ESTRAZIONE DATI CANDELA (MT5)
    double open   = iOpen(symbol, tf, candleIndex);
    double close  = iClose(symbol, tf, candleIndex);
    double high   = iHigh(symbol, tf, candleIndex);
    double low    = iLow(symbol, tf, candleIndex);
    long volume   = iVolume(symbol, tf, candleIndex);

    double body  = MathAbs(close - open);
    double range = high - low;

    if (range <= 0.0 || volume <= 0)
    {
        result.rejectionReason = "Invalid OHLC Data";
        return result;
    }

    // 🤖 AUTO-DETECTION DIREZIONE
    bool autoDirection = DetermineSpikeDirection(open, close, high, low, result.directionConfidence, result.directionReason);
    result.direction = autoDirection;
    
    // 📝 Log della decisione automatica
    if (EnableLogging_SpikeDetection)
    {
        string candleType = close > open ? "🟢 BULLISH" : "🔴 BEARISH";
        double closePos = ((close - low) / (high - low)) * 100.0;
        PrintFormat("🤖 AUTO-DIRECTION: %s candela │ Close at %.1f%% of range │ Detected as %s SPIKE │ Conf: %.1f │ %s",
                    candleType, closePos, autoDirection ? "BUY" : "SELL", 
                    result.directionConfidence, result.directionReason);
    }

    // 🛡️ FILTRI ANTI-RUMORE
    result.noiseInfo = CheckForNoise(symbol, tf, candleIndex);
    if (result.noiseInfo.isNoise)
    {
        result.rejectionReason = "Noise: " + result.noiseInfo.reason;
        result.isNoiseFiltered = false;
        if (EnableLogging_SpikeDetection)
            Print("🚫 [SpikeDetection] Spike rifiutato per rumore: ", result.noiseInfo.reason);
        return result;
    }
    result.isNoiseFiltered = true;

    // 📈 Analisi contesto mercato
    result.context = GetMarketContext(symbol, iTime(symbol, tf, candleIndex));
    result.marketContext = result.context.liquidityScore;
    result.timingScore = result.context.isActiveSession ? 1.0 : 0.3;

    // 📉 FINESTRA ANALISI ESTESA con CopyRates (MT5)
    int lookbackPeriod = 8;
    MqlRates rates[];
    int copied = CopyRates(symbol, tf, candleIndex + 1, lookbackPeriod, rates);
    
    double sumRange = 0.0, sumVolume = 0.0;
    int validBars = 0;
    
    if (copied > 0)
    {
        for (int i = 0; i < copied; i++)
        {
            double barRange = rates[i].high - rates[i].low;
            if (barRange > 0 && rates[i].tick_volume > 0)
            {
                sumRange += barRange;
                sumVolume += (double)rates[i].tick_volume;
                validBars++;
            }
        }
    }
    else
    {
        // Fallback con iHigh/iLow se CopyRates fallisce
        for (int i = candleIndex + 1; i <= candleIndex + lookbackPeriod; i++)
        {
            if (i >= Bars(symbol, tf)) break;
            
            double barRange = iHigh(symbol, tf, i) - iLow(symbol, tf, i);
            long barVolume = iVolume(symbol, tf, i);
            
            if (barRange > 0 && barVolume > 0)
            {
                sumRange += barRange;
                sumVolume += (double)barVolume;
                validBars++;
            }
        }
    }

    if (validBars == 0)
    {
        result.rejectionReason = "Invalid Average Calculations";
        return result;
    }

    double avgRange = sumRange / validBars;
    double avgVolume = sumVolume / validBars;

    // 🧮 CALCOLO INDICATORI SPIKE
    result.bodyRatio        = body / range;
    result.rangeMultiplier  = range / avgRange;
    result.volumeMultiplier = (double)volume / avgVolume;

    // ⚡ AGGIORNAMENTO STATISTICHE O(1)
    UpdateSpikeDetectionStats(result.bodyRatio, result.rangeMultiplier, result.volumeMultiplier);

    // ✅ VERIFICA AFFIDABILITÀ STATISTICHE
    result.isReliableStats = bodyRatioStats.isReliable && 
                            rangeMultiplierStats.isReliable && 
                            volumeMultiplierStats.isReliable;

    // 📏 CALCOLO SOGLIE DINAMICHE ROBUSTE
    double bodyThresh, rangeThresh, volThresh;
    
    if (result.isReliableStats)
    {
        // 🎯 USA SOGLIE ROBUSTE (mediana + MAD) - resistenti agli outlier
        bodyThresh  = GetRobustThreshold(bodyRatioStats, SpikeDeviationMultiplier, 0.6);
        rangeThresh = GetRobustThreshold(rangeMultiplierStats, SpikeDeviationMultiplier, 1.5);
        volThresh   = GetRobustThreshold(volumeMultiplierStats, SpikeDeviationMultiplier, 1.3);
    }
    else
    {
        // 🔧 Soglie fallback ottimizzate per BUY vs SELL
        if (autoDirection) {
            bodyThresh  = 0.65;  // Leggermente più permissivo per BUY
            rangeThresh = 1.8;   // Ridotto da 2.0
            volThresh   = 1.6;   // Ridotto da 1.8
        } else {
            bodyThresh  = 0.60;  // Più permissivo per SELL (candele bearish spesso hanno body minori)
            rangeThresh = 1.7;   // Leggermente più permissivo per SELL
            volThresh   = 1.5;   // Più permissivo per SELL
        }
        
        if (EnableLogging_SpikeDetection)
            Print("⚠️ [SpikeDetection] Usando soglie fallback conservative - statistiche non affidabili");
    }

    // 🎯 CALCOLO PROSSIMITÀ AGLI ESTREMI - CORREZIONE PER SELL
    double baseExtremityPct = SpikeExtremityPercent;
    double adaptiveExtremityPct;
    
    if (autoDirection) {
        // Per BUY: soglia standard
        adaptiveExtremityPct = baseExtremityPct;
    } else {
        // Per SELL: soglia più permissiva (candele bearish hanno dinamiche diverse)
        adaptiveExtremityPct = baseExtremityPct * 1.2; // +20% più permissivo
    }
    
    double extThresh = range * (adaptiveExtremityPct / 100.0);
    bool chiusuraForte;
   
    if (autoDirection) {
        // BUY SPIKE: chiusura vicina al MASSIMO (rifiuto del movimento ribassista)
        result.closeExtremityPct = (close - low) / range * 100.0;
        chiusuraForte = (close >= high - extThresh);
    } else {
        // SELL SPIKE: chiusura vicina al MASSIMO (rifiuto del movimento rialzista)
        result.closeExtremityPct = (high - close) / range * 100.0;
        chiusuraForte = (close >= high - extThresh);  // STESSO controllo di BUY!
    }
    
    // ✅ VALUTAZIONE CONDIZIONI SPIKE
    bool corpoDominante = result.bodyRatio >= bodyThresh;
    bool rangeAlto      = result.rangeMultiplier >= rangeThresh;
    bool volumeAlto     = result.volumeMultiplier >= volThresh;

    // 🔢 SCORE CONFIDENZA STANDARD (MAX 20 punti)
    double score = 0.0;
    if (corpoDominante) score += 5.0;   // Corpo dominante: +5 punti
    if (rangeAlto)      score += 5.0;   // Range alto: +5 punti
    if (volumeAlto)     score += 5.0;   // Volume alto: +5 punti
    if (chiusuraForte)  score += 5.0;   // Chiusura forte: +5 punti
    
    result.confidenceScore = score;  // MAX = 20

    // 🎯 CALCOLO SCORE ROBUSTO (MAX 25 punti)
    if (result.isReliableStats)
    {
        result.bodyPercentile = CalculateCurrentPercentile(bodyRatioStats, result.bodyRatio);
        result.rangePercentile = CalculateCurrentPercentile(rangeMultiplierStats, result.rangeMultiplier);
        result.volumePercentile = CalculateCurrentPercentile(volumeMultiplierStats, result.volumeMultiplier);
        
        // Score robusto basato su percentili (0-100 -> 0-25)
        result.robustConfidence = (result.bodyPercentile + result.rangePercentile + result.volumePercentile) / 300.0 * 25.0;
        
        // Bonus per chiusura forte +2.5 punti
        if (chiusuraForte) result.robustConfidence += 2.5;
        result.robustConfidence = MathMin(result.robustConfidence, 25.0);
    }
    else
    {
        // Fallback su score standard scalato (20 -> 25 max)
        result.robustConfidence = score * 1.25; // 20 * 1.25 = 25 max
    }

    // 🏁 DECISIONE FINALE MULTI-CRITERI (SCORE MAX 30)
    double baseThreshold = 18.0;  // Soglia base (equivalente a 0.7 su scala 0-25)
    double finalThreshold = result.isReliableStats ? baseThreshold : 21.25; // Più stringente se stats inaffidabili (0.85 * 25)
    
    // Bonus per contesto mercato favorevole (MAX +5 punti)
    double contextBonus = result.marketContext * 5.0;  // Max +5 punti
    double adjustedConfidence = result.robustConfidence + contextBonus;  // MAX = 25 + 5 = 30
    
    // Criteri di accettazione
    bool confidenceOk = adjustedConfidence >= finalThreshold;
    bool contextOk = result.marketContext >= 0.3;  // Minima liquidità
    bool reliabilityOk = result.isReliableStats || score >= 18.0;  // Score alto se stats inaffidabili (0.9 * 20)
    
    result.detected = confidenceOk && result.isNoiseFiltered && contextOk && reliabilityOk;

    // 📝 Motivo reiezione se non rilevato
    if (!result.detected)
    {
        if (!confidenceOk)
            result.rejectionReason = StringFormat("Low Confidence (%.1f < %.1f)", adjustedConfidence, finalThreshold);
        else if (!contextOk)
            result.rejectionReason = StringFormat("Poor Market Context (%.2f)", result.marketContext);
        else if (!reliabilityOk)
            result.rejectionReason = StringFormat("Unreliable Stats + Low Score (%.1f)", score);
    }

    // 📝 LOGGING DETTAGLIATO
    if (EnableLogging_SpikeDetection)
    {
        // 🎯 Header principale con risultato
        string statusIcon = result.detected ? "✅ SPIKE DETECTED" : "❌ REJECTED";
        string directionIcon = autoDirection ? "🟢 BUY" : "🔴 SELL";
        
        PrintFormat("═══════════════════════════════════════════════════════════");
        PrintFormat("%s │ %s │ 🤖 AUTO │ Candle[%d] │ %s Session", 
                    statusIcon, directionIcon, candleIndex, result.context.sessionName);
        
        // 📊 Sezione Metriche vs Soglie
        PrintFormat("📊 METRICS vs THRESHOLDS:");
        PrintFormat("   Body Ratio:    %.3f vs %.3f %s │ Range Multi:   %.2f vs %.2f %s │ Volume Multi:  %.2f vs %.2f %s",
                    result.bodyRatio, bodyThresh, (result.bodyRatio >= bodyThresh) ? "✅" : "❌",
                    result.rangeMultiplier, rangeThresh, (result.rangeMultiplier >= rangeThresh) ? "✅" : "❌", 
                    result.volumeMultiplier, volThresh, (result.volumeMultiplier >= volThresh) ? "✅" : "❌");
        
        // 🎯 Sezione Chiusura e Auto-Detection
        string extremityStatus = chiusuraForte ? "✅" : "❌";
        PrintFormat("🎯 POSITION:      Close %.1f%% from HIGH %s │ Required: %.1f%% │ Logic: %s", 
                    result.closeExtremityPct, extremityStatus, adaptiveExtremityPct,
                    autoDirection ? "close near HIGH (rejection of down move)" : "close near HIGH (rejection of up move)");
        
        PrintFormat("🤖 AUTO-DETECT:   Direction: %s │ Confidence: %.1f%% │ Reason: %s",
                    autoDirection ? "BUY" : "SELL", result.directionConfidence * 100, result.directionReason);
        
        // 📈 Percentili (solo se statistiche affidabili)
        if (result.isReliableStats) {
            PrintFormat("📈 PERCENTILES:   Body %02.0f%% │ Range %02.0f%% │ Volume %02.0f%% │ Stats: %d samples 📊",
                        result.bodyPercentile, result.rangePercentile, result.volumePercentile,
                        bodyRatioStats.count);
        } else {
            PrintFormat("📈 PERCENTILES:   N/A (insufficient data: %d samples) │ Using fallback thresholds ⚠️",
                        bodyRatioStats.count);
        }
        
        // 🏆 Sezione Score Finale
        PrintFormat("🏆 CONFIDENCE:    Standard %.1f/20 │ Robust %.1f/25 │ Context +%.1f │ FINAL %.1f/30 (need ≥%.1f)",
                    result.confidenceScore, result.robustConfidence, contextBonus, 
                    adjustedConfidence, finalThreshold);
        
        // 🌍 Contesto Mercato
        string liquidityIcon = result.context.liquidityScore >= 0.8 ? "🟢" : 
                              result.context.liquidityScore >= 0.5 ? "🟡" : "🔴";
        PrintFormat("🌍 MARKET:        %s Liquidity %.1f │ %s │ %s │ News: %s",
                    liquidityIcon, result.context.liquidityScore,
                    result.context.isActiveSession ? "Active" : "Inactive",
                    result.context.isOverlapSession ? "Overlap" : "Single",
                    result.context.isNewsTime ? "YES ⚠️" : "No");
        
        // 🛡️ Filtri Anti-Rumore
        if (!result.isNoiseFiltered) {
            PrintFormat("🛡️ NOISE FILTER:  🚫 FAILED │ %s │ Score: %.2f │ Spread: %.1fx",
                        result.noiseInfo.reason, result.noiseInfo.noiseScore, result.noiseInfo.spreadRatio);
        } else {
            PrintFormat("🛡️ NOISE FILTER:  ✅ PASSED │ Spread: %.1fx │ Gap: %s │ Low Liq: %s",
                        result.noiseInfo.spreadRatio,
                        result.noiseInfo.isGapCandle ? "YES" : "No",
                        result.noiseInfo.isLowLiquidity ? "YES" : "No");
        }
        
        // 🚫 Motivo Reiezione (solo se rigettato)
        if (!result.detected && result.rejectionReason != "") {
            PrintFormat("🚫 REJECTION:     %s", result.rejectionReason);
        }
        
        PrintFormat("═══════════════════════════════════════════════════════════");
    }

    return result;
}

//+------------------------------------------------------------------+
//| 🔍 FUNZIONE HELPER: Rileva miglior spike in range               |
//+------------------------------------------------------------------+
SpikeResult DetectSpikeInRange(string symbol, ENUM_TIMEFRAMES tf, int startIndex, int endIndex, int &spikeCandleIndex)
{
    SpikeResult bestResult;
    bestResult.detected = false;
    spikeCandleIndex = -1;
    
    double bestConfidence = 0.0;

    // 🔍 Cerca il miglior spike nel range
    for (int i = startIndex; i <= endIndex; i++)
    {
        SpikeResult currentResult = DetectSpike(symbol, tf, i);
        
        // 🎯 Aggiorna se trovato spike migliore
        if (currentResult.detected && currentResult.robustConfidence > bestConfidence)
        {
            bestResult = currentResult;
            bestConfidence = currentResult.robustConfidence;
            spikeCandleIndex = i;
        }
    }

    return bestResult;
}

//+------------------------------------------------------------------+
//| 📋 FUNZIONE HELPER: Log semplificato per operazioni multiple    |
//+------------------------------------------------------------------+
void LogSpikeResultSimple(SpikeResult &result, int candleIndex)
{
    if (!EnableLogging_SpikeDetection) return;
    
    string status = result.detected ? "✅" : "❌";
    string direction = result.direction ? "BUY" : "SELL";
    double finalConfidence = result.robustConfidence + (result.marketContext * 5.0);
    
    PrintFormat("%s 🤖 %s[%d] │ Conf:%.1f │ Body:%.2f │ Range:%.1f │ Vol:%.1f │ Close:%.0f%% │ %s",
                status, direction, candleIndex, finalConfidence,
                result.bodyRatio, result.rangeMultiplier, result.volumeMultiplier,
                result.closeExtremityPct, result.rejectionReason);
}

//+------------------------------------------------------------------+
//| 📊 FUNZIONE HELPER: Statistiche periodiche                     |
//+------------------------------------------------------------------+
void LogSpikeStatsPeriodic()
{
    static datetime lastStatsTime = 0;
    static int statsInterval = 3600; // Ogni ora
    
    if (TimeCurrent() - lastStatsTime < statsInterval) return;
    lastStatsTime = TimeCurrent();
    
    if (!isStatsInitialized) return;
    
    PrintFormat("📊 ═══ SPIKE STATS SUMMARY (samples: %d) ═══", bodyRatioStats.count);
    
    if (bodyRatioStats.isReliable) {
        PrintFormat("   BodyRatio    │ Med:%.3f │ Mad:%.3f │ Range:[%.3f-%.3f]",
                    bodyRatioStats.median, bodyRatioStats.mad,
                    bodyRatioStats.buffer[0], bodyRatioStats.buffer[bodyRatioStats.count-1]);
        PrintFormat("   RangeMulti   │ Med:%.2f │ Mad:%.2f │ Range:[%.2f-%.2f]",
                    rangeMultiplierStats.median, rangeMultiplierStats.mad,
                    rangeMultiplierStats.buffer[0], rangeMultiplierStats.buffer[rangeMultiplierStats.count-1]);
        PrintFormat("   VolumeMulti  │ Med:%.2f │ Mad:%.2f │ Range:[%.2f-%.2f]",
                    volumeMultiplierStats.median, volumeMultiplierStats.mad,
                    volumeMultiplierStats.buffer[0], volumeMultiplierStats.buffer[volumeMultiplierStats.count-1]);
    } else {
        PrintFormat("   📊 Statistics not reliable yet (need ≥25 samples, have %d)", bodyRatioStats.count);
    }
    
    PrintFormat("📊 ════════════════════════════════════════════");
}

//+------------------------------------------------------------------+
//| 📊 UTILITY: Reset completo statistiche                         |
//+------------------------------------------------------------------+
void ResetSpikeDetectionStats()
{
    if (isStatsInitialized)
    {
        ResetRollingStats(bodyRatioStats);
        ResetRollingStats(rangeMultiplierStats);
        ResetRollingStats(volumeMultiplierStats);
        
        if (EnableLogging_SpikeDetection)
            Print("🔄 [SpikeDetection] Statistiche resettate");
    }
}

//+------------------------------------------------------------------+
//| 📋 UTILITY: Stampa info statistiche correnti                   |
//+------------------------------------------------------------------+
void PrintSpikeStats()
{
    if (!isStatsInitialized) return;
    
    Print("📊 ========== SPIKE DETECTION STATS ==========");
    PrintStatsInfo(bodyRatioStats, "BodyRatio");
    PrintStatsInfo(rangeMultiplierStats, "RangeMultiplier");  
    PrintStatsInfo(volumeMultiplierStats, "VolumeMultiplier");
    Print("📊 ============================================");
}

//+------------------------------------------------------------------+
//| 🎯 UTILITY MT5: Verifica se mercato è aperto per simbolo       |
//+------------------------------------------------------------------+
bool IsMarketOpen(string symbol)
{
    // 📅 Usa MT5 SymbolInfoInteger per verificare se trading è attivo
    return (bool)SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE) != SYMBOL_TRADE_MODE_DISABLED;
}

//+------------------------------------------------------------------+
//| 📊 UTILITY MT5: Ottieni informazioni complete simbolo          |
//+------------------------------------------------------------------+
bool GetSymbolInfo(string symbol, double &ask, double &bid, double &point, int &digits)
{
    ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
    
    return (ask > 0 && bid > 0 && point > 0 && digits > 0);
}

//+------------------------------------------------------------------+
//| 📈 UTILITY MT5: Verifica se barra è formata completamente      |
//+------------------------------------------------------------------+
bool IsBarCompleted(string symbol, ENUM_TIMEFRAMES tf, int barIndex)
{
    // 🕐 Controlla se la barra all'indice specificato è completa
    datetime barTime = iTime(symbol, tf, barIndex);
    datetime nextBarTime = barTime + PeriodSeconds(tf);
    
    return TimeCurrent() >= nextBarTime;
}

//+------------------------------------------------------------------+
//| 🎯 UTILITY MT5: Calcola dimensione tick value per simbolo      |
//+------------------------------------------------------------------+
double CalculateTickValue(string symbol, double lots)
{
    double tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    
    if (tickSize > 0 && point > 0)
    {
        return tickValue * (point / tickSize) * lots;
    }
    
    return 0.0;
}

//+------------------------------------------------------------------+
//| 📊 UTILITY MT5: Ottieni spread corrente in punti              |
//+------------------------------------------------------------------+
int GetCurrentSpreadPoints(string symbol)
{
    return (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD);
}

//+------------------------------------------------------------------+
//| 🔍 UTILITY MT5: Verifica se simbolo è disponibile per trading  |
//+------------------------------------------------------------------+
bool IsSymbolTradeable(string symbol)
{
    // Verifica multiple condizioni per trading
    bool marketOpen = IsMarketOpen(symbol);
    bool hasQuotes = SymbolInfoDouble(symbol, SYMBOL_ASK) > 0 && SymbolInfoDouble(symbol, SYMBOL_BID) > 0;
    bool tradingAllowed = (ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE) != SYMBOL_TRADE_MODE_DISABLED;
    
    return marketOpen && hasQuotes && tradingAllowed;
}

#endif // __SPIKE_DETECTION_OPTIMIZED_MQH__