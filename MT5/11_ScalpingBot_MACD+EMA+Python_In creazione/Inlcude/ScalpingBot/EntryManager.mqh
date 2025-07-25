//+------------------------------------------------------------------+
//|                     EntryManager.mqh - VERSIONE MT5             |
//|  Modulo centrale per aggregare punteggi e decidere entry        |
//|  AGGIORNATO per nuovo SpikeDetection con score 0-30             |
//|  INTEGRATO con RSI Momentum Avanzato (score 0-10)              |
//|  COMPLETAMENTE CONVERTITO PER METATRADER 5                      |
//|  Integra MicroTrend, RSI Momentum, EMA_MACD_BodyScore,          |
//|  Campionamento e Spike Detection ottimizzato                    |
//+------------------------------------------------------------------+
#ifndef __ENTRY_MANAGER_MQH__
#define __ENTRY_MANAGER_MQH__

#include <ScalpingBot\MicroTrendScanner.mqh>
#include <ScalpingBot\RSIMomentum.mqh>
#include <ScalpingBot\EMA_MACD_BodyScore.mqh>
#include <ScalpingBot\Campionamento.mqh>
#include <ScalpingBot\SpikeDetection.mqh>  // Versione ottimizzata con score 0-30

//+------------------------------------------------------------------+
//| 📦 Struttura dati aggregata per punteggi ESTESA                 |
//+------------------------------------------------------------------+
struct EntryScoreResult
{
    // Score base
    double scoreTotal;
    double scoreMicroTrend;
    double scoreRSI;
    double scoreEMA_MACD_Body;
    double scoreCampionamento;
    double scoreSpike;
    bool   entryConfirmed;
    string reasonsLog;
    
    // 🧭 Direzioni rilevate dai moduli
    bool directionEMAMACDBody;
    bool directionRSI;
    bool directionMicroTrend;
    bool directionSpike;
    bool directionFinal;  // 🧭 Direzione finale aggregata (BUY=true, SELL=false)

    // 🚀 NUOVI CAMPI PER SPIKE DETECTION AVANZATO
    double spikeConfidenceStandard;    // Score standard 0-20
    double spikeConfidenceRobust;      // Score robusto 0-25
    double spikeContextBonus;          // Bonus contesto 0-5
    double spikeFinalScore;            // Score finale 0-30
    bool   spikeIsReliable;            // Statistiche affidabili
    bool   spikeNoiseFiltered;         // Filtri anti-rumore passati
    string spikeSessionName;           // Nome sessione trading
    string spikeRejectionReason;       // Motivo reiezione se non rilevato
    
    // 📊 Metriche qualità Spike
    double spikeBodyPercentile;        // Percentile body ratio
    double spikeRangePercentile;       // Percentile range multiplier
    double spikeVolumePercentile;      // Percentile volume multiplier
    double spikeMarketContext;         // Score contesto mercato 0-1
    
    // 🆕 NUOVI CAMPI PER RSI MOMENTUM AVANZATO
    double rsiAdvancedScore;           // Score avanzato 0-10
    double rsiConfidence;              // Confidenza segnale RSI 0-100%
    RSI_SIGNAL_QUALITY rsiQuality;     // Qualità segnale RSI
    RSI_MARKET_REGIME rsiMarketRegime; // Regime di mercato rilevato
    double rsiTrendStrength;           // Forza del trend 0-1
    double rsiVolatilityFactor;        // Fattore volatilità
    double rsiSignalPersistence;       // Persistenza segnale 0-1
    double rsiDynamicThreshold;        // Soglia derivata dinamica
    double rsiNoiseLevel;              // Livello rumore 0-1
};

//+------------------------------------------------------------------+
//| 🔍 Funzione principale aggregazione punteggi AGGIORNATA         |
//+------------------------------------------------------------------+
EntryScoreResult CalculateEntryScore(string symbol, ENUM_TIMEFRAMES tf, bool direction)
{
    EntryScoreResult result;
    result.scoreTotal = 0.0;
    result.scoreMicroTrend = 0.0;
    result.scoreRSI = 0.0;
    result.scoreEMA_MACD_Body = 0.0;
    result.scoreCampionamento = 0.0;
    result.scoreSpike = 0.0;
    result.entryConfirmed = false;
    result.reasonsLog = "";
    result.directionFinal = false;
    
    // Inizializza campi Spike avanzati
    result.spikeConfidenceStandard = 0.0;
    result.spikeConfidenceRobust = 0.0;
    result.spikeContextBonus = 0.0;
    result.spikeFinalScore = 0.0;
    result.spikeIsReliable = false;
    result.spikeNoiseFiltered = false;
    result.spikeSessionName = "";
    result.spikeRejectionReason = "";
    result.spikeBodyPercentile = 0.0;
    result.spikeRangePercentile = 0.0;
    result.spikeVolumePercentile = 0.0;
    result.spikeMarketContext = 0.0;
    
    // Inizializza campi RSI avanzati
    result.rsiAdvancedScore = 0.0;
    result.rsiConfidence = 0.0;
    result.rsiQuality = RSI_QUALITY_POOR;
    result.rsiMarketRegime = RSI_REGIME_UNKNOWN;
    result.rsiTrendStrength = 0.0;
    result.rsiVolatilityFactor = 1.0;
    result.rsiSignalPersistence = 0.0;
    result.rsiDynamicThreshold = RSIDerivataThreshold;
    result.rsiNoiseLevel = 0.5;

    if (!EnableEntryManager)
    {
        if (EnableLogging_EntryManager)
            Print("⚪️ [EntryManager] Modulo disabilitato");
        return result;
    }

    // 🔠 Blocchi log separati
    string logMicroTrend = "", logRSI = "", logEMA = "", logCamp = "", logSpike = "";

    // ----------------------------
    // 📊 MicroTrend
    // ----------------------------
    if (EnableMicroTrendModule)
    {
        double slope = 0.0, adx = 0.0, atr = 0.0;
        bool breakout = false, momentum = false;
    
        MicroTrendResult microResult = GetMicroTrendScore(symbol, tf,
                                                          slope, breakout, momentum,
                                                          adx, atr);
        
        result.scoreMicroTrend = microResult.score;
        
        // Normalizza su 10 (lo score massimo ora è 30)
        double normalizedScore = (microResult.score / 30.0) * 10.0;
        result.scoreTotal += normalizedScore * (WeightMicroTrend / 10.0);
    
        result.directionMicroTrend = (microResult.direction == MICROTREND_BUY);
        
        logMicroTrend = StringFormat(
            "📊 [MicroTrend] %.2f / 30 → %.2f / 10 | %s (Conf: %.1f%%)\n"
            "   • Slope = %.2f%% | Breakout = %s | Momentum = %s | ADX = %.2f",
            microResult.score, normalizedScore,
            microResult.GetDirectionString(), microResult.confidence,
            slope * 100.0, breakout ? "✔" : "✘", momentum ? "✔" : "✘", adx);
        
        if(microResult.direction == MICROTREND_NONE)
        {
            logMicroTrend += "\n   ⚠️ Nessuna direzione chiara rilevata";
            result.directionMicroTrend = false;
            result.scoreMicroTrend = 0.0;
        }
    }

    // ----------------------------
    // 📈 RSI Momentum AVANZATO (COMPLETAMENTE RIVISITATO)
    // ----------------------------
    if (EnableRSIMomentumModule)
    {
        // ✅ NUOVA CHIAMATA - Solo symbol e timeframe (auto-detection completa)
        UpdateRSIMomentumState(symbol, tf);

        // Ottieni stato avanzato RSI
        RSIMomentumStateAdvanced rsiAdvanced;
        if (GetAdvancedRSIMomentumState(symbol, tf, rsiAdvanced))
        {
            // 🆕 UTILIZZO SCORE AVANZATO 0-10
            result.rsiAdvancedScore = rsiAdvanced.advancedScore;
            result.scoreRSI = result.rsiAdvancedScore;  // Già su scala 0-10
            result.scoreTotal += result.scoreRSI * (WeightRSIMomentum / 10.0);
            
            // 🆕 SALVA METRICHE AVANZATE
            result.rsiConfidence = rsiAdvanced.confidence;
            result.rsiQuality = rsiAdvanced.quality;
            result.rsiMarketRegime = rsiAdvanced.regime;
            result.rsiTrendStrength = rsiAdvanced.trendStrength;
            result.rsiVolatilityFactor = rsiAdvanced.volatilityFactor;
            result.rsiSignalPersistence = rsiAdvanced.signalPersistence;
            result.rsiDynamicThreshold = rsiAdvanced.dynamicThreshold;
            result.rsiNoiseLevel = rsiAdvanced.noiseLevel;
            
            // 🆕 DIREZIONE RSI BASATA SU AUTO-DETECTION (non più su input direction)
            result.directionRSI = rsiAdvanced.autoDirection;
            
            // 🆕 LOG DETTAGLIATO CON METRICHE AVANZATE E AUTO-DETECTION
            logRSI = StringFormat(
                "📈 [RSI Momentum] Score: %.2f / 10 (Base: %d/3 | Advanced: %d/10)\n"
                "   • RSI: Last=%.2f | Avg=%.2f | Derivata=%.4f | ADX=%.2f\n"
                "   • 🤖 AUTO-DIRECTION: %s (Conf: %.1f%%) | Reason: %s\n"
                "   • 🎯 Qualità: %s | Confidenza: %.1f%% | Persistenza: %.2f\n"
                "   • 📊 Regime: %s | Trend: %.2f | Volatilità: %.2f | Rumore: %.2f\n"
                "   • ⚙️ Soglie dinamiche: Derivata=%.4f | ADX=%.2f",
                result.scoreRSI, rsiAdvanced.rsiScore, rsiAdvanced.advancedScore,
                rsiAdvanced.rsiLast, rsiAdvanced.rsiAvg, rsiAdvanced.derivataWeighted, rsiAdvanced.adx,
                rsiAdvanced.autoDirection ? "BUY 🟢" : "SELL 🔴", rsiAdvanced.autoConfidence * 100, rsiAdvanced.autoReason,
                EnumToString(rsiAdvanced.quality), rsiAdvanced.confidence, rsiAdvanced.signalPersistence,
                EnumToString(rsiAdvanced.regime), rsiAdvanced.trendStrength, 
                rsiAdvanced.volatilityFactor, rsiAdvanced.noiseLevel,
                rsiAdvanced.dynamicThreshold, rsiAdvanced.adaptiveADXThreshold);
            
            // 🆕 BONUS QUALITÀ RSI
            if (rsiAdvanced.quality == RSI_QUALITY_EXCELLENT && rsiAdvanced.confidence > 80.0)
            {
                double qualityBonus = 0.5 * (WeightRSIMomentum / 10.0);
                result.scoreTotal += qualityBonus;
                logRSI += StringFormat("\n   🏆 BONUS QUALITÀ ECCELLENTE: +%.2f", qualityBonus);
            }
            
            // 🆕 PENALITÀ PER RUMORE ELEVATO
            if (rsiAdvanced.noiseLevel > 0.7)
            {
                double noisePenalty = 0.3 * (WeightRSIMomentum / 10.0);
                result.scoreTotal -= noisePenalty;
                logRSI += StringFormat("\n   ⚠️ PENALITÀ RUMORE ELEVATO: -%.2f", noisePenalty);
            }
        }
        else
        {
            // Fallback a stato legacy se necessario
            RSIMomentumState rsiLegacy;
            if (GetRSIMomentumState(symbol, tf, rsiLegacy))
            {
                double normalizedRSI = ((double)rsiLegacy.rsiScore / 3.0) * 10.0;
                result.scoreRSI = normalizedRSI;
                result.scoreTotal += normalizedRSI * (WeightRSIMomentum / 10.0);
                
                // ✅ Per legacy, usa auto-detection se disponibile, altrimenti input direction
                double autoConf;
                string autoReason;
                RSI_DIRECTION_SIGNAL autoSignal;
                bool autoDir = GetAutoDetectedRSIDirection(symbol, tf, autoConf, autoReason, autoSignal);
                result.directionRSI = autoConf > 0.0 ? autoDir : direction;  // Fallback se auto-detection non disponibile
                
                logRSI = StringFormat(
                    "📈 [RSI Momentum Legacy] %.2f / 10\n"
                    "   • RSI Score = %d / 3 | RSI = %.2f | Derivata = %.4f | ADX = %.2f\n"
                    "   • Direction: %s %s",
                    normalizedRSI, rsiLegacy.rsiScore, rsiLegacy.rsiLast, rsiLegacy.derivata, rsiLegacy.adx,
                    result.directionRSI ? "BUY 🟢" : "SELL 🔴",
                    autoConf > 0.0 ? "(Auto-detected)" : "(Fallback)");
            }
            else
            {
                logRSI = "⚠️ [RSI Momentum] Dati non disponibili";
            }
        }
    }

    // ----------------------------
    // 🔍 EMA + MACD + Body
    // ----------------------------
    if (EnableEMAMACDBody)
    {
        PrevalidationResult pre = CalculatePrevalidationScore(symbol, tf);  
        result.directionEMAMACDBody = pre.directionEMAMACDBody;
        
        double scaledScore = pre.scoreTotal * (WeightEMA_MACD_Body / 30.0);
        scaledScore = MathMin(scaledScore, WeightEMA_MACD_Body);
        
        result.scoreEMA_MACD_Body = scaledScore;
        result.scoreTotal += scaledScore;
        
        logEMA = StringFormat(
            "🔍 [EMA+MACD+Body] Score: %.2f/30.0 → %.2f/%.1f\n"
            "   📐 EMA: %.2f | 📊 MACD: %.2f | 💪 Body: %.2f | Dir: %s",
            pre.scoreTotal, scaledScore, WeightEMA_MACD_Body,
            pre.scoreEMA, pre.scoreMACD, pre.scoreBodyMomentum,
            pre.directionEMAMACDBody ? "BUY 🟢" : "SELL 🔴");
    }

    // ----------------------------
    // ⏳ Campionamento
    // ----------------------------
    if (EnableCampionamentoModule)
    {
        double campScore = GetSamplingScore(symbol, tf);
        ENUM_ORDER_TYPE campDir = GetSamplingDirection(symbol, tf);
        string campLog = GetSamplingLog(symbol, tf);
    
        result.scoreCampionamento = campScore;
        result.scoreTotal += campScore * (WeightCampionamento / 10.0);
    
        logCamp = campLog;
    }

    // ----------------------------
    // ⚡ SPIKE DETECTION OTTIMIZZATO
    // ----------------------------
    if (EnableSpikeDetection)
    {
        // 🛡️ Verifica che il simbolo sia valido per MT5
        double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
        double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
        
        if (ask <= 0 || bid <= 0)
        {
            logSpike = "⚡ [Spike Detection] ❌ Simbolo non valido o mercato chiuso";
        }
        else
        {
            // ✅ NUOVO
            SpikeResult spike = DetectSpike(symbol, tf, 1);
            
            // 📊 ESTRAI TUTTI I DATI DAL NUOVO SPIKE RESULT
            result.spikeConfidenceStandard = spike.confidenceScore;      // 0-20
            result.spikeConfidenceRobust = spike.robustConfidence;       // 0-25  
            result.spikeContextBonus = spike.marketContext * 5.0;       // 0-5
            result.spikeFinalScore = spike.robustConfidence + result.spikeContextBonus; // 0-30
            result.spikeIsReliable = spike.isReliableStats;
            result.spikeNoiseFiltered = spike.isNoiseFiltered;
            result.spikeSessionName = spike.context.sessionName;
            result.spikeRejectionReason = spike.rejectionReason;
            result.spikeBodyPercentile = spike.bodyPercentile;
            result.spikeRangePercentile = spike.rangePercentile;
            result.spikeVolumePercentile = spike.volumePercentile;
            result.spikeMarketContext = spike.marketContext;
            
            if (spike.detected)
            {
                // 🎯 SISTEMA DI SCORING A FASCE PER GRANULARITÀ
                double spikeScoreContribution = 0.0;
                
                if (result.spikeFinalScore >= 28.0)      spikeScoreContribution = 10.0;  // Spike eccellente
                else if (result.spikeFinalScore >= 25.0) spikeScoreContribution = 8.5;   // Spike ottimo  
                else if (result.spikeFinalScore >= 22.0) spikeScoreContribution = 7.0;   // Spike buono
                else if (result.spikeFinalScore >= 20.0) spikeScoreContribution = 5.5;   // Spike discreto
                else if (result.spikeFinalScore >= 18.0) spikeScoreContribution = 4.0;   // Spike sufficiente
                else                                     spikeScoreContribution = 2.0;   // Spike debole
                
                // 🎖️ BONUS QUALITÀ
                if (result.spikeIsReliable)      spikeScoreContribution += 0.5;  // +0.5 per stats affidabili
                if (result.spikeNoiseFiltered)   spikeScoreContribution += 0.3;  // +0.3 per filtri anti-rumore
                if (spike.context.isOverlapSession) spikeScoreContribution += 0.7;  // +0.7 per overlap London-NY
                else if (spike.context.isActiveSession) spikeScoreContribution += 0.4;  // +0.4 per sessione attiva
                
                // Limita a 10.0 max
                spikeScoreContribution = MathMin(spikeScoreContribution, 10.0);
                
                result.scoreSpike = spikeScoreContribution;
                result.scoreTotal += spikeScoreContribution * (WeightSpikeDetection / 10.0);
                result.directionSpike = spike.direction;

                // 📝 LOG DETTAGLIATO PER SPIKE RILEVATO
                logSpike = StringFormat(
                    "⚡ [Spike Detection] ✅ RILEVATO - Score: %.1f/10.0\n"
                    "   🎯 Score Finale: %.1f/30 (Standard: %.1f/20 + Robust: %.1f/25 + Context: %.1f/5)\n"
                    "   📊 Percentili: Body=%.0f%% | Range=%.0f%% | Volume=%.0f%%\n"
                    "   🏛️ Sessione: %s (Liquidità: %.1f) | Affidabile: %s | Filtrato: %s\n"
                    "   🧭 Direzione: %s | Qualità: %s",
                    spikeScoreContribution,
                    result.spikeFinalScore, result.spikeConfidenceStandard, result.spikeConfidenceRobust, result.spikeContextBonus,
                    result.spikeBodyPercentile, result.spikeRangePercentile, result.spikeVolumePercentile,
                    result.spikeSessionName, result.spikeMarketContext,
                    result.spikeIsReliable ? "✅" : "❌",
                    result.spikeNoiseFiltered ? "✅" : "❌",
                    spike.direction ? "BUY 🟢" : "SELL 🔴",
                    result.spikeFinalScore >= 25.0 ? "ECCELLENTE" : 
                    result.spikeFinalScore >= 22.0 ? "OTTIMO" : 
                    result.spikeFinalScore >= 20.0 ? "BUONO" : "SUFFICIENTE");
            }
            else
            {
                // 📝 LOG PER SPIKE NON RILEVATO
                logSpike = StringFormat(
                    "⚡ [Spike Detection] ❌ NON RILEVATO\n"
                    "   🚫 Motivo: %s\n"
                    "   📊 Score: Standard=%.1f/20 | Robust=%.1f/25 | Context=%.1f/5 | Finale=%.1f/30\n"
                    "   🏛️ Sessione: %s | Filtrato: %s | Affidabile: %s",
                    result.spikeRejectionReason,
                    result.spikeConfidenceStandard, result.spikeConfidenceRobust, result.spikeContextBonus, result.spikeFinalScore,
                    result.spikeSessionName,
                    result.spikeNoiseFiltered ? "✅" : "❌",
                    result.spikeIsReliable ? "✅" : "❌");
                
                // Nessun contributo al score se spike non rilevato
                result.scoreSpike = 0.0;
                result.directionSpike = false; // Default direction per spike non rilevato
            }
        }
    }

    // ----------------------------
    // 🎯 SINERGIA TRA MODULI (NUOVO)
    // ----------------------------
    double synergyBonus = 0.0;
    string synergySummary = "\n🔗 [Sinergia Moduli]";
    
    // 🆕 Sinergia RSI + Spike Detection
    if (EnableRSIMomentumModule && EnableSpikeDetection && 
        result.scoreRSI > 5.0 && result.scoreSpike > 5.0)
    {
        // Entrambi i moduli rilevano segnale forte nella stessa direzione
        if (result.directionRSI == result.directionSpike)
        {
            synergyBonus += 1.0;
            synergySummary += "\n   • RSI + Spike concordi: +1.0";
            
            // Bonus extra per qualità eccellente
            if (result.rsiQuality >= RSI_QUALITY_GOOD && result.spikeFinalScore >= 25.0)
            {
                synergyBonus += 0.5;
                synergySummary += "\n   • Qualità Premium RSI+Spike: +0.5";
            }
        }
    }
    
    // 🆕 Sinergia RSI Regime + MicroTrend
    if (EnableRSIMomentumModule && EnableMicroTrendModule)
    {
        if (result.rsiMarketRegime == RSI_REGIME_TRENDING && result.scoreMicroTrend > 5.0)
        {
            synergyBonus += 0.5;
            synergySummary += "\n   • RSI Trending + MicroTrend forte: +0.5";
        }
    }
    
    // 🆕 Penalità per segnali contrastanti
    if (EnableRSIMomentumModule && EnableEMAMACDBody)
    {
        if (result.scoreRSI > 5.0 && result.scoreEMA_MACD_Body > 5.0 &&
            result.directionRSI != result.directionEMAMACDBody)
        {
            synergyBonus -= 1.0;
            synergySummary += "\n   • ⚠️ RSI vs EMA/MACD contrasto: -1.0";
        }
    }
    
    // Applica bonus/penalità sinergia
    if (MathAbs(synergyBonus) > 0.01)
    {
        result.scoreTotal += synergyBonus;
        synergySummary += StringFormat("\n   🎯 Totale sinergia: %.2f", synergyBonus);
    }
    else
    {
        synergySummary = "";  // Non mostrare se non c'è sinergia
    }

    // ----------------------------
    // ✅ Esito finale con considerazioni RSI avanzate
    // ----------------------------
    // 🆕 Soglia adattiva basata su regime di mercato RSI
    double adaptiveThreshold = EntryThreshold;
    if (EnableRSIMomentumModule && result.rsiMarketRegime != RSI_REGIME_UNKNOWN)
    {
        if (result.rsiMarketRegime == RSI_REGIME_TRENDING)
        {
            adaptiveThreshold = EntryThreshold * 0.95;  // Più permissivo in trend
        }
        else if (result.rsiMarketRegime == RSI_REGIME_VOLATILE)
        {
            adaptiveThreshold = EntryThreshold * 1.05;  // Più restrittivo in volatilità
        }
    }
    
    result.entryConfirmed = (result.scoreTotal >= adaptiveThreshold);
    
    // ----------------------------
    // 🧭 Calcolo direzione finale (AGGIORNATO CON PESO QUALITÀ)
    // ----------------------------
    double votesBuy = 0.0, votesSell = 0.0;  // 🆕 Voti pesati invece di interi
    string logDirections = "\n🧭 [Voting Direzione Finale - Sistema Pesato]";
    bool directionIsDetermined = false;

    if (EnableEMAMACDBody && result.scoreEMA_MACD_Body > 0.0)
    {
        double weight = 1.0;
        logDirections += StringFormat("\n   • EMA_MACD_Body: %s (peso: %.2f)", 
                                     result.directionEMAMACDBody ? "BUY ✔" : "SELL ✘", weight);
        result.directionEMAMACDBody ? votesBuy += weight : votesSell += weight;
        directionIsDetermined = true;
    }

    if (EnableRSIMomentumModule && result.scoreRSI > 0.0)
    {
        // 🆕 PESO BASATO SU QUALITÀ E CONFIDENZA RSI
        double weight = 1.0;
        if (result.rsiQuality == RSI_QUALITY_EXCELLENT) weight = 2.0;
        else if (result.rsiQuality == RSI_QUALITY_GOOD) weight = 1.5;
        else if (result.rsiQuality == RSI_QUALITY_FAIR) weight = 1.0;
        else weight = 0.5;
        
        // Aggiustamento peso per confidenza
        weight *= (result.rsiConfidence / 100.0);
        
        logDirections += StringFormat("\n   • RSI Momentum : %s (peso: %.2f | Q:%s | C:%.1f%%)", 
                                     result.directionRSI ? "BUY ✔" : "SELL ✘", 
                                     weight, EnumToString(result.rsiQuality), result.rsiConfidence);
        result.directionRSI ? votesBuy += weight : votesSell += weight;
        directionIsDetermined = true;
    }

    if (EnableMicroTrendModule && result.scoreMicroTrend > 0.0)
    {
        double weight = 1.0;
        logDirections += StringFormat("\n   • MicroTrend   : %s (peso: %.2f)", 
                                     result.directionMicroTrend ? "BUY ✔" : "SELL ✘", weight);
        result.directionMicroTrend ? votesBuy += weight : votesSell += weight;
        directionIsDetermined = true;
    }

    if (EnableSpikeDetection && result.scoreSpike > 0.0)
    {
        // 🆕 PESO BASATO SU QUALITÀ SPIKE
        double weight = 1.0;
        if (result.spikeFinalScore >= 25.0) weight = 1.8;
        else if (result.spikeFinalScore >= 20.0) weight = 1.3;
        else weight = 0.8;
        
        logDirections += StringFormat("\n   • Spike Detect : %s (peso: %.2f | Score: %.1f/30)", 
                                     result.directionSpike ? "BUY ✔" : "SELL ✘", 
                                     weight, result.spikeFinalScore);
        result.directionSpike ? votesBuy += weight : votesSell += weight;
        directionIsDetermined = true;
    }
    
    if (EnableCampionamentoModule && result.scoreCampionamento > 0.0)
    {
        ENUM_ORDER_TYPE campDir = GetSamplingDirection(symbol, tf);
        bool campDirection = (campDir == ORDER_TYPE_BUY);
        double weight = 1.0;
        
        logDirections += StringFormat("\n   • Campionamento : %s (peso: %.2f)", 
                                     campDirection ? "BUY ✔" : "SELL ✘", weight);
        campDirection ? votesBuy += weight : votesSell += weight;
        directionIsDetermined = true;
    }
    
    // Debug voting pesato
    if (EnableLogging_EntryManager)
    {
        PrintFormat("🗳️ [DEBUG VOTING FINALE] - BUY=%.2f, SELL=%.2f, Determined=%s", 
                    votesBuy, votesSell, directionIsDetermined ? "TRUE" : "FALSE");
    }

    // Determina direzione finale con sistema pesato
    if (directionIsDetermined)
    {
        if (votesBuy > votesSell)
            result.directionFinal = true; // BUY
        else if (votesSell > votesBuy)
            result.directionFinal = false; // SELL
        // In caso di parità esatta, usa la qualità RSI come tie-breaker
        else if (MathAbs(votesBuy - votesSell) < 0.01 && EnableRSIMomentumModule)
        {
            result.directionFinal = result.directionRSI;
            logDirections += StringFormat("\n   🎯 Parità! Tie-break con RSI: %s", 
                                         result.directionFinal ? "BUY" : "SELL");
        }
        
        logDirections += StringFormat("\n🔚 Direzione finale: %s (BUY=%.2f, SELL=%.2f | Diff=%.2f)",
                                     result.directionFinal ? "BUY 🟢" : "SELL 🔴", 
                                     votesBuy, votesSell, MathAbs(votesBuy - votesSell));
                                     
        // 🆕 Avviso per votazione contesa
        if (MathAbs(votesBuy - votesSell) < 0.5)
        {
            logDirections += "\n   ⚠️ ATTENZIONE: Votazione molto contesa! Considerare attesa.";
        }
    }
    else
    {
        logDirections += "\n🔚 Direzione finale: INDETERMINATA (nessun voto valido)";
    }

    // ----------------------------
    // 📝 LOGGING FINALE DETTAGLIATO
    // ----------------------------
    if (EnableLogging_EntryManager)
    {
        PrintFormat("\n🧠 [EntryManager] Totale = %.2f / 100.00 → %s",
                    result.scoreTotal,
                    result.entryConfirmed ? "✅ APERTURA" : "❌ NO ENTRY");
        
        // 🆕 Mostra soglia adattiva se diversa
        if (MathAbs(adaptiveThreshold - EntryThreshold) > 0.01)
        {
            PrintFormat("   📊 Soglia adattiva: %.2f (base: %.2f) - Regime: %s",
                       adaptiveThreshold, EntryThreshold, 
                       EnumToString(result.rsiMarketRegime));
        }
        
        // Stampa log dettagliati
        if (logMicroTrend != "") Print(logMicroTrend);
        if (logRSI != "")        Print(logRSI);
        if (logEMA != "")        Print(logEMA);
        if (logCamp != "")       Print(logCamp);
        if (logSpike != "")      Print(logSpike);

        // 🆕 Stampa sinergia se presente
        if (synergySummary != "") Print(synergySummary);

        Print(logDirections);
        
        // 📊 SUMMARY RSI AVANZATO se significativo
        if (EnableRSIMomentumModule && result.scoreRSI > 3.0)
        {
            PrintFormat("📈 [RSI Summary] Score=%.1f/10 | Quality=%s | Confidence=%.1f%% | Regime=%s | Persist=%.2f",
                        result.scoreRSI,
                        EnumToString(result.rsiQuality),
                        result.rsiConfidence,
                        EnumToString(result.rsiMarketRegime),
                        result.rsiSignalPersistence);
        }
        
        // 📊 SUMMARY SPIKE AVANZATO se rilevato
        if (EnableSpikeDetection && result.scoreSpike > 0.0)
        {
            PrintFormat("⚡ [Spike Summary] Final=%.1f | Reliable=%s | Session=%s | Quality=%s",
                        result.spikeFinalScore,
                        result.spikeIsReliable ? "✅" : "❌",
                        result.spikeSessionName,
                        result.spikeFinalScore >= 25.0 ? "PREMIUM" : 
                        result.spikeFinalScore >= 20.0 ? "GOOD" : "BASIC");
        }
        
        // 🆕 RACCOMANDAZIONE FINALE BASATA SU QUALITÀ COMPLESSIVA
        if (result.entryConfirmed)
        {
            string recommendation = "";
            double qualityScore = 0.0;
            
            // Calcola quality score complessivo
            if (EnableRSIMomentumModule)
                qualityScore += (result.rsiConfidence / 100.0) * 0.3;
            if (EnableSpikeDetection && result.scoreSpike > 0)
                qualityScore += (result.spikeFinalScore / 30.0) * 0.3;
            if (MathAbs(votesBuy - votesSell) > 1.0)
                qualityScore += 0.2;  // Consenso forte
            if (synergyBonus > 0)
                qualityScore += 0.2;  // Sinergia positiva
                
            if (qualityScore >= 0.8)
                recommendation = "🏆 SEGNALE PREMIUM - Alta confidenza";
            else if (qualityScore >= 0.6)
                recommendation = "✅ SEGNALE BUONO - Procedere con fiducia";
            else if (qualityScore >= 0.4)
                recommendation = "⚠️ SEGNALE DISCRETO - Usare cautela";
            else
                recommendation = "⚡ SEGNALE DEBOLE - Considerare size ridotta";
                
            Print("\n" + recommendation);
        }
    }

    return result;
}

//+------------------------------------------------------------------+
//| 🎯 UTILITY: Ottieni qualità spike come stringa                 |
//+------------------------------------------------------------------+
string GetSpikeQualityString(double finalScore)
{
    if (finalScore >= 28.0)      return "ECCELLENTE";
    else if (finalScore >= 25.0) return "OTTIMO";
    else if (finalScore >= 22.0) return "BUONO";
    else if (finalScore >= 20.0) return "DISCRETO";
    else if (finalScore >= 18.0) return "SUFFICIENTE";
    else                         return "DEBOLE";
}

//+------------------------------------------------------------------+
//| 📊 UTILITY: Verifica se spike è di qualità premium             |
//+------------------------------------------------------------------+
bool IsSpikeQualityPremium(EntryScoreResult &result)
{
    return (result.spikeFinalScore >= 25.0 && 
            result.spikeIsReliable && 
            result.spikeNoiseFiltered &&
            result.spikeMarketContext >= 0.8);
}

//+------------------------------------------------------------------+
//| 🆕 UTILITY: Ottieni qualità RSI come stringa                  |
//+------------------------------------------------------------------+
string GetRSIQualityString(RSI_SIGNAL_QUALITY quality)
{
    switch(quality)
    {
        case RSI_QUALITY_EXCELLENT: return "ECCELLENTE";
        case RSI_QUALITY_GOOD:      return "BUONO";
        case RSI_QUALITY_FAIR:      return "DISCRETO";
        case RSI_QUALITY_POOR:      return "SCARSO";
        default:                    return "SCONOSCIUTO";
    }
}

//+------------------------------------------------------------------+
//| 🆕 UTILITY: Verifica se RSI è di qualità premium              |
//+------------------------------------------------------------------+
bool IsRSIQualityPremium(EntryScoreResult &result)
{
    return (result.rsiQuality >= RSI_QUALITY_GOOD && 
            result.rsiConfidence >= 75.0 && 
            result.rsiSignalPersistence >= 0.6 &&
            result.rsiNoiseLevel <= 0.3);
}

//+------------------------------------------------------------------+
//| 🆕 UTILITY: Ottieni qualità complessiva del segnale           |
//+------------------------------------------------------------------+
double GetOverallSignalQuality(EntryScoreResult &result)
{
    double qualityScore = 0.0;
    int componentsCount = 0;
    
    // Contributo RSI
    if (EnableRSIMomentumModule && result.scoreRSI > 0)
    {
        double rsiContribution = (result.rsiConfidence / 100.0) * 
                                (result.rsiQuality / 3.0) * 
                                result.rsiSignalPersistence;
        qualityScore += rsiContribution;
        componentsCount++;
    }
    
    // Contributo Spike
    if (EnableSpikeDetection && result.scoreSpike > 0)
    {
        double spikeContribution = (result.spikeFinalScore / 30.0) * 
                                  (result.spikeIsReliable ? 1.0 : 0.7) *
                                  (result.spikeNoiseFiltered ? 1.0 : 0.8);
        qualityScore += spikeContribution;
        componentsCount++;
    }
    
    // Contributo MicroTrend
    if (EnableMicroTrendModule && result.scoreMicroTrend > 0)
    {
        double microContribution = result.scoreMicroTrend / 10.0;
        qualityScore += microContribution;
        componentsCount++;
    }
    
    // Media ponderata
    return componentsCount > 0 ? qualityScore / componentsCount : 0.0;
}

//+------------------------------------------------------------------+
//| 🆕 UTILITY: Ottieni raccomandazione position sizing           |
//+------------------------------------------------------------------+
double GetRecommendedPositionSizeMultiplier(EntryScoreResult &result)
{
    double quality = GetOverallSignalQuality(result);
    
    // Sistema di sizing basato su qualità
    if (quality >= 0.9)      return 1.0;   // Size piena per segnali premium
    else if (quality >= 0.7) return 0.8;   // 80% per segnali buoni
    else if (quality >= 0.5) return 0.6;   // 60% per segnali discreti
    else if (quality >= 0.3) return 0.4;   // 40% per segnali deboli
    else                     return 0.25;  // 25% minimo per segnali molto deboli
}

//+------------------------------------------------------------------+
//| 🆕 UTILITY: Verifica coerenza direzione tra moduli           |
//+------------------------------------------------------------------+
bool IsDirectionCoherent(EntryScoreResult &result, double &coherenceScore)
{
    int totalVotes = 0;
    int coherentVotes = 0;
    
    // Raccogli tutti i voti di direzione
    bool directions[];
    ArrayResize(directions, 0);
    
    if (EnableRSIMomentumModule && result.scoreRSI > 0)
    {
        ArrayResize(directions, ArraySize(directions) + 1);
        directions[ArraySize(directions)-1] = result.directionRSI;
    }
    
    if (EnableSpikeDetection && result.scoreSpike > 0)
    {
        ArrayResize(directions, ArraySize(directions) + 1);
        directions[ArraySize(directions)-1] = result.directionSpike;
    }
    
    if (EnableMicroTrendModule && result.scoreMicroTrend > 0)
    {
        ArrayResize(directions, ArraySize(directions) + 1);
        directions[ArraySize(directions)-1] = result.directionMicroTrend;
    }
    
    if (EnableEMAMACDBody && result.scoreEMA_MACD_Body > 0)
    {
        ArrayResize(directions, ArraySize(directions) + 1);
        directions[ArraySize(directions)-1] = result.directionEMAMACDBody;
    }
    
    totalVotes = ArraySize(directions);
    if (totalVotes < 2) 
    {
        coherenceScore = 1.0;  // Un solo modulo = 100% coerente
        return true;
    }
    
    // Conta voti per la direzione finale
    for (int i = 0; i < totalVotes; i++)
    {
        if (directions[i] == result.directionFinal)
            coherentVotes++;
    }
    
    coherenceScore = (double)coherentVotes / totalVotes;
    return coherenceScore >= 0.6;  // Almeno 60% coerenza
}

//+------------------------------------------------------------------+
//| 🆕 UTILITY: Genera report dettagliato del segnale             |
//+------------------------------------------------------------------+
string GenerateSignalReport(EntryScoreResult &result, string symbol, ENUM_TIMEFRAMES tf)
{
    string report = "\n================== SIGNAL REPORT ==================\n";
    report += StringFormat("Symbol: %s | Timeframe: %s\n", symbol, EnumToString(tf));
    report += StringFormat("Total Score: %.2f/100 | Entry: %s\n", 
                          result.scoreTotal, result.entryConfirmed ? "YES" : "NO");
    report += StringFormat("Direction: %s\n", result.directionFinal ? "BUY" : "SELL");
    
    // Qualità complessiva
    double quality = GetOverallSignalQuality(result);
    report += StringFormat("Overall Quality: %.1f%% | Position Size: %.0f%%\n", 
                          quality * 100, GetRecommendedPositionSizeMultiplier(result) * 100);
    
    // Coerenza direzione
    double coherence;
    bool isCoherent = IsDirectionCoherent(result, coherence);
    report += StringFormat("Direction Coherence: %.1f%% %s\n", 
                          coherence * 100, isCoherent ? "✓" : "⚠");
    
    // Dettagli RSI se attivo
    if (EnableRSIMomentumModule && result.scoreRSI > 0)
    {
        report += ("\nRSI Momentum:\n");
        report += StringFormat("  Score: %.1f/10 | Quality: %s\n", 
                             result.scoreRSI, GetRSIQualityString(result.rsiQuality));
        report += StringFormat("  Confidence: %.1f%% | Persistence: %.2f\n",
                             result.rsiConfidence, result.rsiSignalPersistence);
        report += StringFormat("  Market Regime: %s | Trend: %.2f\n",
                             EnumToString(result.rsiMarketRegime), result.rsiTrendStrength);
    }
    
    // Dettagli Spike se attivo
    if (EnableSpikeDetection && result.scoreSpike > 0)
    {
        report += ("\nSpike Detection:\n");
        report += StringFormat("  Score: %.1f/10 | Final: %.1f/30\n",
                             result.scoreSpike, result.spikeFinalScore);
        report += StringFormat("  Quality: %s | Session: %s\n",
                             GetSpikeQualityString(result.spikeFinalScore), 
                             result.spikeSessionName);
        report += StringFormat("  Reliable: %s | Filtered: %s\n",
                             result.spikeIsReliable ? "YES" : "NO",
                             result.spikeNoiseFiltered ? "YES" : "NO");
    }
    
    report += "================================================\n";
    
    return report;
}

#endif // __ENTRY_MANAGER_MQH__