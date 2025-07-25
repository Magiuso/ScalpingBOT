//+------------------------------------------------------------------+
//|       MicroTrendScanner.mqh                                     |
//| 📊 Modulo micro-trend con 7 componenti configurabili            |
//+------------------------------------------------------------------+
#ifndef __MICROTREND_SCANNER_MQH__
#define __MICROTREND_SCANNER_MQH__

#include <ScalpingBot\Utility.mqh>
#include <ScalpingBot\SafeBuffer.mqh>

//+------------------------------------------------------------------+
//| 🔢 Enum per modalità MicroTrend                                  |
//+------------------------------------------------------------------+
enum ENUM_MicroTrendMode
{
    MICROTREND_SOFT = 0,         // 🎯 Modalità conservativa
    MICROTREND_AGGRESSIVE = 1    // 🚀 Modalità aggressiva
};

//+------------------------------------------------------------------+
//| 🔢 Enum per la Direzione del MicroTrend                         |
//+------------------------------------------------------------------+
enum ENUM_MicroTrendDirection
{
    MICROTREND_NONE = 0,    // Nessuna direzione chiara o punteggio insufficiente
    MICROTREND_BUY  = 1,    // Trend rialzista
    MICROTREND_SELL = 2     // Trend ribassista
};

//+------------------------------------------------------------------+
//| 📊 Struttura per componenti score                               |
//+------------------------------------------------------------------+
struct MicroTrendComponents
{
    double emaScore;
    double breakoutScore;
    double momentumScore;
    double adxScore;
    double volatilityScore;
    double volumeScore;
    double spreadScore;
    
    // Metodo per ottenere il totale
    double GetTotal() 
    { 
        return emaScore + breakoutScore + momentumScore + 
               adxScore + volatilityScore + volumeScore + spreadScore; 
    }
    
    // Reset tutti i valori
    void Reset()
    {
        emaScore = 0.0;
        breakoutScore = 0.0;
        momentumScore = 0.0;
        adxScore = 0.0;
        volatilityScore = 0.0;
        volumeScore = 0.0;
        spreadScore = 0.0;
    }
    
    // Log dettagliato per diagnostica
    void PrintDiagnostics(string label)
    {
        if(!Diagnostic_Mode) return;
        
        PrintFormat("[🔍 MicroTrend Diagnostics %s]", label);
        PrintFormat("   EMA Score:        %.2f %s", emaScore, !Enable_EMASlope ? "(DISABLED)" : "");
        PrintFormat("   Breakout Score:   %.2f %s", breakoutScore, !Enable_Breakout ? "(DISABLED)" : "");
        PrintFormat("   Momentum Score:   %.2f %s", momentumScore, !Enable_Momentum ? "(DISABLED)" : "");
        PrintFormat("   ADX Score:        %.2f %s", adxScore, !Enable_ADX ? "(DISABLED)" : "");
        PrintFormat("   Volatility Score: %.2f %s", volatilityScore, !Enable_Volatility ? "(DISABLED)" : "");
        PrintFormat("   Volume Score:     %.2f %s", volumeScore, !Enable_Volume ? "(DISABLED)" : "");
        PrintFormat("   Spread Score:     %.2f %s", spreadScore, !Enable_SpreadImpact ? "(DISABLED)" : "");
        PrintFormat("   TOTAL:            %.2f / 30.0", GetTotal());
    }
    
};

//+------------------------------------------------------------------+
//| 📈 Struttura risultato MicroTrend con confidence                |
//+------------------------------------------------------------------+
struct MicroTrendResult
{
    ENUM_MicroTrendDirection direction;
    double score;
    double confidence;      // Quanto è chiara la direzione (0-100%)
    double scoreBuy;        // Score componente BUY
    double scoreSell;       // Score componente SELL
    MicroTrendComponents components;
    
    // Metodi utility
    string GetDirectionString()
    {
        switch(direction)
        {
            case MICROTREND_BUY:  return "BUY";
            case MICROTREND_SELL: return "SELL";
            default:              return "NONE";
        }
    }
    
    bool IsValid() { return direction != MICROTREND_NONE && score >= Min_TotalScore; }
    
    void Reset()
    {
        direction = MICROTREND_NONE;
        score = 0.0;
        confidence = 0.0;
        scoreBuy = 0.0;
        scoreSell = 0.0;
        components.Reset();
    }
};

//+------------------------------------------------------------------+
//| 🎯 Ottieni pesi basati sulla modalità                           |
//+------------------------------------------------------------------+
void GetMicroTrendWeights(ENUM_MicroTrendMode mode, 
                          double &wEMA, double &wBreak, double &wMom,
                          double &wADX, double &wVol, double &wVolume, double &wSpread)
{
    if(mode == MICROTREND_SOFT)
    {
        wEMA    = Weight_EMASlope_Soft;
        wBreak  = Weight_Breakout_Soft;
        wMom    = Weight_Momentum_Soft;
        wADX    = Weight_ADX_Soft;
        wVol    = Weight_Volatility_Soft;
        wVolume = Weight_Volume_Soft;
        wSpread = Weight_SpreadImpact_Soft;
    }
    else // AGGRESSIVE
    {
        wEMA    = Weight_EMASlope_Aggr;
        wBreak  = Weight_Breakout_Aggr;
        wMom    = Weight_Momentum_Aggr;
        wADX    = Weight_ADX_Aggr;
        wVol    = Weight_Volatility_Aggr;
        wVolume = Weight_Volume_Aggr;
        wSpread = Weight_SpreadImpact_Aggr;
    }
}

//+------------------------------------------------------------------+
//| 🔧 Adatta parametri per timeframe                               |
//+------------------------------------------------------------------+
int AdaptPeriodForTimeframe(int basePeriod, ENUM_TIMEFRAMES tf)
{
    double multiplier = 1.0;
    
    switch(tf)
    {
        case PERIOD_M1:  multiplier = 0.8; break;
        case PERIOD_M5:  multiplier = 1.0; break;
        case PERIOD_M15: multiplier = 1.3; break;
        case PERIOD_M30: multiplier = 1.6; break;
        case PERIOD_H1:  multiplier = 2.0; break;
        default:         multiplier = 2.5; break;
    }
    
    return (int)MathMax(3, MathRound(basePeriod * multiplier));
}

//+------------------------------------------------------------------+
//| 📈 Calcola EMA Slope Score                                       |
//+------------------------------------------------------------------+
double CalculateEMASlope(string symbol, ENUM_TIMEFRAMES tf, bool directionInput,
                         double &slopePercent, double weight)
{
    if(!Enable_EMASlope) return 0.0;
    
    // Adatta periodo per timeframe
    int emaPeriod = AdaptPeriodForTimeframe(EMA_Period_Base, tf);
    
    // Ottieni handle dalla cache EMA
    int handle = emaCache.GetEMAHandle(symbol, tf, emaPeriod);
    if(handle == INVALID_HANDLE)
    {
        if(EnableLogging_MicroTrend)
            PrintFormat("❌ [MicroTrend] Handle EMA%d non valido per %s [%s]", 
                        emaPeriod, symbol, EnumToString(tf));
        return 0.0;
    }
    
    // Leggi 5 valori EMA
    double ema[5];
    if(!SafeCopyBuffer(handle, 0, 1, 5, ema))
    {
        if(EnableLogging_MicroTrend)
            PrintFormat("❌ [MicroTrend] CopyBuffer fallito per EMA%d", emaPeriod);
        return 0.0;
    }
    
    // Calcola slope medio
    double slopeSum = 0.0;
    for(int i = 1; i < 5; i++)
        slopeSum += (ema[i-1] - ema[i]);
    
    double avgSlope = slopeSum / 4.0;
    
    // Calcola percentuale
    if(ema[4] != 0.0)
        slopePercent = avgSlope / ema[4];
    else
        return 0.0;
    
    // Soglia adattiva per timeframe
    double threshold = tf <= PERIOD_M5 ? 0.0005 : 0.0010;
    
    // Normalizza score (0-1)
    double rawScore = MathMin(MathAbs(slopePercent) / threshold, 1.0);
    
    // Verifica direzione
    if((directionInput && slopePercent > 0) || (!directionInput && slopePercent < 0))
    {
        if(EnableLogging_MicroTrend)
            PrintFormat("✅ [EMA Slope] %.5f%% → Score: %.2f", 
                        slopePercent * 100, weight * rawScore);
        return weight * rawScore;
    }
    
    return 0.0;
}

//+------------------------------------------------------------------+
//| 🚀 Enhanced Breakout Detection                                   |
//+------------------------------------------------------------------+
double CalculateBreakout(string symbol, ENUM_TIMEFRAMES tf, bool directionInput,
                         double weight, bool &breakoutDetected)
{
    if(!Enable_Breakout) return 0.0;
    
    breakoutDetected = false;
    
    // Verifica barre disponibili
    if(Bars(symbol, tf) < Breakout_Lookback + 1)
        return 0.0;
    
    double current = directionInput ? iHigh(symbol, tf, 0) : iLow(symbol, tf, 0);
    double spread = (double)SymbolInfoInteger(symbol, SYMBOL_SPREAD) * SymbolInfoDouble(symbol, SYMBOL_POINT);
    
    // Trova livelli significativi
    int levelsBreached = 0;
    
    for(int i = 1; i <= Breakout_Lookback; i++)
    {
        double level = directionInput ? iHigh(symbol, tf, i) : iLow(symbol, tf, i);
        
        if(directionInput)
        {
            if(current > level + spread) levelsBreached++;
        }
        else
        {
            if(current < level - spread) levelsBreached++;
        }
    }
    
    // Calcola forza del breakout
    double breakoutStrength = (double)levelsBreached / Breakout_Lookback;
    
    // Richiedi almeno 30% dei livelli superati
    if(breakoutStrength >= 0.3)
    {
        breakoutDetected = true;
        double score = weight * breakoutStrength;
        
        if(EnableLogging_MicroTrend)
            PrintFormat("✅ [Breakout %s] Livelli: %d/%d → Score: %.2f",
                        directionInput ? "BUY" : "SELL", levelsBreached, 
                        Breakout_Lookback, score);
        
        return score;
    }
    
    return 0.0;
}

//+------------------------------------------------------------------+
//| 🔥 Momentum Analysis                                             |
//+------------------------------------------------------------------+
double CalculateMomentum(string symbol, ENUM_TIMEFRAMES tf, bool directionInput,
                         double weight, bool &momentumDetected)
{
    if(!Enable_Momentum) return 0.0;
    
    momentumDetected = false;
    
    double open = iOpen(symbol, tf, 0);
    double close = iClose(symbol, tf, 0);
    double high = iHigh(symbol, tf, 0);
    double low = iLow(symbol, tf, 0);
    
    if(open <= 0 || close <= 0 || high <= 0 || low <= 0)
        return 0.0;
    
    double body = MathAbs(close - open);
    double range = high - low;
    
    if(range <= 0) return 0.0;
    
    // Usa cache per ATR
    int atrHandle = indicatorCache.GetHandle(symbol, tf, 14, 0); // 0 = ATR
    if(atrHandle == INVALID_HANDLE) return 0.0;
    
    double atr[];
    if(CopyBuffer(atrHandle, 0, 0, 1, atr) <= 0) return 0.0;
    
    double dynamicThreshold = atr[0] * 0.3;
    
    // Verifica momentum
    if(body / range >= Momentum_BodyRatio && body >= dynamicThreshold)
    {
        // Verifica direzione
        if((directionInput && close > open) || (!directionInput && close < open))
        {
            momentumDetected = true;
            
            // Score basato su forza del momentum
            double momentumStrength = MathMin(body / atr[0], 2.0) / 2.0; // Normalizza 0-1
            double score = weight * momentumStrength;
            
            if(EnableLogging_MicroTrend)
                PrintFormat("✅ [Momentum %s] Body: %.5f, ATR: %.5f → Score: %.2f",
                            directionInput ? "BUY" : "SELL", body, atr[0], score);
            
            return score;
        }
    }
    
    return 0.0;
}

//+------------------------------------------------------------------+
//| 📡 ADX Analysis usando cache                                     |
//+------------------------------------------------------------------+
double CalculateADXScore(string symbol, ENUM_TIMEFRAMES tf, double weight,
                         double &adxValue)
{
    if(!Enable_ADX) return 0.0;
    
    int adxPeriod = AdaptPeriodForTimeframe(ADX_Period_Base, tf);
    
    // Usa cache per ADX
    int adxHandle = indicatorCache.GetHandle(symbol, tf, adxPeriod, 1); // 1 = ADX
    if(adxHandle == INVALID_HANDLE) return 0.0;
    
    double adx[];
    if(CopyBuffer(adxHandle, 0, 0, 1, adx) <= 0) return 0.0;
    
    adxValue = adx[0];
    
    // Soglia adattiva per timeframe
    double threshold = tf <= PERIOD_M5 ? 20.0 : 25.0;
    
    if(adxValue > threshold)
    {
        // Normalizza score basato su forza ADX
        double adxStrength = MathMin((adxValue - threshold) / 30.0, 1.0);
        double score = weight * adxStrength;
        
        if(EnableLogging_MicroTrend)
            PrintFormat("✅ [ADX] Valore: %.2f → Score: %.2f", adxValue, score);
        
        return score;
    }
    
    return 0.0;
}

//+------------------------------------------------------------------+
//| 🌊 Volatility Score (ATR) con cache                             |
//+------------------------------------------------------------------+
double CalculateVolatilityScore(string symbol, ENUM_TIMEFRAMES tf, double weight,
                                double &atrValue, int period1, int period2, int period3)
{
    if(!Enable_Volatility) return 0.0;
    
    // Calcola ATR con 3 periodi usando cache
    double atrSum = 0.0;
    int validCount = 0;
    
    int periods[] = {period1, period2, period3};
    
    for(int i = 0; i < 3; i++)
    {
        int atrHandle = indicatorCache.GetHandle(symbol, tf, periods[i], 0); // 0 = ATR
        if(atrHandle == INVALID_HANDLE) continue;
        
        double atr_buffer[];
        if(CopyBuffer(atrHandle, 0, 0, 1, atr_buffer) > 0)
        {
            atrSum += atr_buffer[0];
            validCount++;
        }
    }
    
    if(validCount == 0) return 0.0;
    
    // Usa media degli ATR
    atrValue = atrSum / validCount;
    
    // Calcola volatilità percentuale
    double price = SymbolInfoDouble(symbol, SYMBOL_BID);
    if (price <= 0) return 0.0;
    double volPercent = (atrValue / price) * 100;
    
    // Soglia minima 0.1% per timeframe bassi
    double minVol = tf <= PERIOD_M5 ? 0.1 : 0.15;
    
    if(volPercent >= minVol)
    {
        // Normalizza score
        double volStrength = MathMin(volPercent / (minVol * 3), 1.0);
        double score = weight * volStrength;
        
        if(EnableLogging_MicroTrend)
            PrintFormat("✅ [Volatility] ATR: %.5f (%.2f%%) → Score: %.2f",
                        atrValue, volPercent, score);
        
        return score;
    }
    
    return 0.0;
}

//+------------------------------------------------------------------+
//| 📊 Volume Analysis                                               |
//+------------------------------------------------------------------+
double CalculateVolumeScore(string symbol, ENUM_TIMEFRAMES tf, double weight)
{
    if(!Enable_Volume) return 0.0;
    
    // Calcola media volume
    long totalVolume = 0;
    int periods = 20;
    
    for(int i = 1; i <= periods; i++)
    {
        long vol = iVolume(symbol, tf, i);
        if(vol <= 0) return 0.0;
        totalVolume += vol;
    }
    
    double avgVolume = (double)totalVolume / periods;
    long currentVolume = iVolume(symbol, tf, 0);
    
    if(avgVolume <= 0 || currentVolume <= 0)
        return 0.0;
    
    // Filtro volume basso
    if(Filter_LowVolume && currentVolume < avgVolume * 0.5)
    {
        if(EnableLogging_MicroTrend)
            Print("⚠️ [Volume] Filtrato per volume troppo basso");
        return 0.0;
    }
    
    double volumeRatio = (double)currentVolume / avgVolume;
    
    // Richiedi volume sopra la media
    if(volumeRatio >= Volume_Threshold)
    {
        // Normalizza score
        double volStrength = MathMin((volumeRatio - 1.0) / 2.0, 1.0);
        double score = weight * volStrength;
        
        if(EnableLogging_MicroTrend)
            PrintFormat("✅ [Volume] Ratio: %.2fx → Score: %.2f",
                        volumeRatio, score);
        
        return score;
    }
    
    return 0.0;
}

//+------------------------------------------------------------------+
//| 💱 Spread Impact Analysis                                       |
//+------------------------------------------------------------------+
// Questa funzione calcola l'impatto dello spread, confrontandolo
// con la volatilità media (ATR) e con un limite massimo assoluto.
// Restituisce uno score basato su quanto lo spread è "favorevole".
//------------------------------------------------------------------+
double CalculateSpreadImpact(string symbol, ENUM_TIMEFRAMES tf, double weight)
{
    // 1. Controllo Abilitazione Modulo
    if(!Enable_SpreadImpact)
    {
        if(EnableLogging_MicroTrend)
            Print("⚪️ [Spread] Modulo disabilitato");
        return 0.0;
    }

    // 2. Recupero Dati Spread
    double spread = (double)SymbolInfoInteger(symbol, SYMBOL_SPREAD); // Spread in "punti" interni del broker (es. 120 per 1.2 pips)
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);           // Valore di un "punto" in valuta di quotazione (es. 0.00001 per EURUSD)
    double spreadPoints = spread * point;                             // Spread in punti prezzo reali dell'asset (es. 1.2 per USTEC)

    // 3. Recupero ATR (come riferimento per la volatilità)
    int atrHandle = indicatorCache.GetHandle(symbol, tf, 14, 0); // Utilizza ATR a 14 periodi. 0 = ATR
    if(atrHandle == INVALID_HANDLE)
    {
        if(EnableLogging_MicroTrend)
            PrintFormat("❌ [Spread] Handle ATR non valido per %s %s", symbol, EnumToString(tf));
        return 0.0;
    }

    double atr_buffer[];
    if(CopyBuffer(atrHandle, 0, 0, 1, atr_buffer) <= 0)
    {
        if(EnableLogging_MicroTrend)
            PrintFormat("❌ [Spread] CopyBuffer ATR fallito per %s %s", symbol, EnumToString(tf));
        return 0.0;
    }

    if(atr_buffer[0] <= 0) // L'ATR deve essere un valore positivo
    {
        if(EnableLogging_MicroTrend)
            PrintFormat("❌ [Spread] Valore ATR non valido (<=0) per %s %s", symbol, EnumToString(tf));
        return 0.0;
    }

    // 4. Calcolo del Rapporto Spread/ATR
    // Questo rapporto indica quanto è "grande" lo spread rispetto alla volatilità media.
    // Un valore più basso è migliore.
    double spreadRatio = spreadPoints / atr_buffer[0];

    // 5. Applicazione dei Filtri per lo Score
    // Condizioni per ottenere uno score positivo:
    // a) Il rapporto spread/ATR deve essere inferiore alla soglia definita dall'utente (Spread_ATR_MaxRatio).
    // b) Lo spread in punti reali (spreadPoints) non deve superare il limite massimo assoluto (Spread_MaxPoints).
    if(spreadRatio < Spread_ATR_MaxRatio && spreadPoints <= Spread_MaxPoints)
    {
        // Calcolo dello Score:
        // Lo score è inversamente proporzionale allo spreadRatio.
        // Se spreadRatio è 0 (spread ideale), il termine (1.0 - spreadRatio / Spread_ATR_MaxRatio) sarà 1.0.
        // Se spreadRatio è vicino a Spread_ATR_MaxRatio, il termine si avvicinerà a 0.
        // Verrà dato il massimo score quando lo spread è molto basso e si ridurrà all'aumentare dello spread.
        double score = weight * (1.0 - spreadRatio / Spread_ATR_MaxRatio);

        // Logging di Successo
        if(EnableLogging_MicroTrend)
            PrintFormat("✅ [Spread] OK - %.2f pts (%.1f%% ATR) → Score: %.2f",
                        spreadPoints, spreadRatio * 100, score);

        return score;
    }
    // 6. Logging dei Fallimenti (Per capire perché non viene dato uno score)
    else
    {
        if(EnableLogging_MicroTrend)
        {
            if (spreadRatio >= Spread_ATR_MaxRatio)
            {
                // Fallimento per rapporto spread/ATR troppo alto
                PrintFormat("⚠️ [Spread] Filtrato: Ratio %.1f%% (limite %.1f%%). Spread %.2f pts, ATR %.2f pts. Raw: %d",
                            spreadRatio * 100, Spread_ATR_MaxRatio * 100, spreadPoints, atr_buffer[0], (int)spread);
            }
            else // Se il rapporto è OK, allora il problema è Spread_MaxPoints
            {
                // Fallimento per spread assoluto troppo alto
                PrintFormat("⚠️ [Spread] Filtrato: Spread %.2f pts (limite %.1f pts). Raw: %d",
                            spreadPoints, Spread_MaxPoints, (int)spread);
            }
        }
        return 0.0; // Restituisce 0.0 se le condizioni non sono soddisfatte
    }
}

//+------------------------------------------------------------------+
//| 🎯 FUNZIONE PRINCIPALE: GetMicroTrendScore (Ottimizzata)        |
//+------------------------------------------------------------------+
MicroTrendResult GetMicroTrendScore(string symbol, ENUM_TIMEFRAMES tf,
                                    double &slopePercent, bool &breakoutDetected,
                                    bool &momentumDetected, double &adxValue, double &atrValue)
{
    MicroTrendResult result;
    result.Reset();

    slopePercent = 0.0;
    breakoutDetected = false;
    momentumDetected = false;
    adxValue = 0.0;
    atrValue = 0.0;

    if(!EnableMicroTrendModule)
    {
        if(EnableLogging_MicroTrend)
            Print("⚪️ [MicroTrend] Modulo disabilitato");
        return result;
    }

    if(EnableInitialCandleFilter)
    {
        datetime time[];
        if(CopyTime(symbol, tf, 0, 1, time) > 0)
        {
            long secondsElapsed = TimeCurrent() - time[0];
            if(secondsElapsed >= 0 && secondsElapsed < 10)
            {
                if(EnableLogging_MicroTrend)
                    PrintFormat("⏳ [MicroTrend] Filtro iniziale: %ds < 10s", secondsElapsed);
                return result;
            }
        }
    }

    int emaPeriod = AdaptPeriodForTimeframe(EMA_Period_Base, tf);
    int handle = emaCache.GetEMAHandle(symbol, tf, emaPeriod);

    if(handle == INVALID_HANDLE)
    {
        if(EnableLogging_MicroTrend)
            PrintFormat("❌ [MicroTrend] Handle EMA non valido");
        return result;
    }

    double ema[5];
    if(!SafeCopyBuffer(handle, 0, 1, 5, ema))
    {
        if(EnableLogging_MicroTrend)
            PrintFormat("❌ [MicroTrend] CopyBuffer EMA fallito");
        return result;
    }

    double slopeSum = 0.0;
    for(int i = 1; i < 5; i++)
        slopeSum += (ema[i] - ema[i-1]);

    double avgSlope = slopeSum / 4.0;

    if(ema[4] != 0.0)
        slopePercent = avgSlope / ema[4];
    else
        return result;

    bool isBuyDirection = (slopePercent > 0);

    double threshold = tf <= PERIOD_M5 ? Min_Slope_Threshold_M5_or_Less : Min_Slope_Threshold_M10_or_More;
    if(MathAbs(slopePercent) < threshold * 0.3)
    {
        if(EnableLogging_MicroTrend)
            PrintFormat("⚠️ [MicroTrend] Slope EMA troppo debole: %.5f%% (min: %.5f%%)",
                        slopePercent * 100, threshold * 100 * 0.3); // Qui ho corretto anche il messaggio di log
                                                                // per mostrare la percentuale della soglia *0.3
        return result;
    }

    double wEMA, wBreak, wMom, wADX, wVol, wVolume, wSpread;
    GetMicroTrendWeights(MicroTrendMode, wEMA, wBreak, wMom, wADX, wVol, wVolume, wSpread);

    // Componenti aggiornati direttamente
    result.components.emaScore       = wEMA   * MathMin(MathAbs(slopePercent) / threshold, 1.0);
    result.components.breakoutScore  = CalculateBreakout(symbol, tf, isBuyDirection, wBreak, breakoutDetected);
    result.components.momentumScore  = CalculateMomentum(symbol, tf, isBuyDirection, wMom, momentumDetected);
    result.components.adxScore       = CalculateADXScore(symbol, tf, wADX, adxValue);
    result.components.volatilityScore = CalculateVolatilityScore(symbol, tf, wVol, atrValue, ATR_Period1, ATR_Period2, ATR_Period3);
    result.components.volumeScore    = CalculateVolumeScore(symbol, tf, wVolume);
    result.components.spreadScore    = CalculateSpreadImpact(symbol, tf, wSpread);

    result.score = result.components.GetTotal();

    bool hasDirectionalSupport = false;
    if(result.components.breakoutScore > 0 || result.components.momentumScore > 0)
        hasDirectionalSupport = true;

    if(!hasDirectionalSupport)
    {
        result.score *= 0.7;
        if(EnableLogging_MicroTrend)
            Print("⚠️ [MicroTrend] Nessun supporto direzionale - score ridotto del 30%");
    }

    if(result.score < Min_TotalScore)
    {
        if(EnableLogging_MicroTrend)
            PrintFormat("⚠️ [MicroTrend] Score %.2f sotto minimo %.2f", result.score, Min_TotalScore);
        result.Reset();
        return result;
    }

    result.direction = isBuyDirection ? MICROTREND_BUY : MICROTREND_SELL;

    double directionalStrength = (result.components.emaScore + 
                                  result.components.breakoutScore + 
                                  result.components.momentumScore) / 
                                 (wEMA + wBreak + wMom);
    result.confidence = directionalStrength * 100;

    if(isBuyDirection)
    {
        result.scoreBuy = result.score;
        result.scoreSell = 0.0;
    }
    else
    {
        result.scoreSell = result.score;
        result.scoreBuy = 0.0;
    }

    if(EnableLogging_MicroTrend)
    {
        PrintFormat("[📊 MicroTrend %s] Score: %.2f/30.0 | Confidence: %.1f%%", 
                    result.GetDirectionString(), result.score, result.confidence);

        PrintFormat("   → Slope EMA: %.5f%% (threshold: %.5f%%)", 
                    slopePercent * 100, threshold * 100);

        PrintFormat("   → EMA: %.2f | Break: %.2f | Mom: %.2f | ADX: %.2f",
                    result.components.emaScore, result.components.breakoutScore, 
                    result.components.momentumScore, result.components.adxScore);

        PrintFormat("   → Vol: %.2f | Volume: %.2f | Spread: %.2f",
                    result.components.volatilityScore, result.components.volumeScore, 
                    result.components.spreadScore);
    }

    if(Diagnostic_Mode)
    {
        result.components.PrintDiagnostics(result.GetDirectionString());
    }

    return result;
}


#endif // __MICROTREND_SCANNER_MQH__