//+------------------------------------------------------------------+
//| 📂 OpenRecoveryTrade.mqh                                         |
//| 🔁 Modulo per apertura di ordini di RECUPERO (da RecoveryBlock) |
//| 📌 Include SL/TP dinamici, moltiplicatore lotto, fallback mode  |
//+------------------------------------------------------------------+
#ifndef __OPEN_RECOVERY_TRADE_MQH__
#define __OPEN_RECOVERY_TRADE_MQH__


#include <ScalpingBot\Utility.mqh>
#include <ScalpingBot\RSI_SLTP_Dinamici.mqh>
#include <ScalpingBot\SymbolFillingCache.mqh>
#include <ScalpingBot\RecoveryTriggerManager.mqh>

//+----------------------------------------------------------------------+
//| ✨ NUOVA FUNZIONE: Apertura ordine di RECUPERO                        |
//| 🎯 Chiamata dal RecoveryBlock.mqh per eseguire un trade di recupero. |
//| 🔹 Il lotto, SL e TP sono gestiti QUI.                               |
//| 💡 original_ticket è il ticket della posizione originale fallita.   |
//+----------------------------------------------------------------------+
bool OpenRecoveryTrade(string symbol, ulong original_ticket)
{
    // 🔒 Evita chiamate multiple simultanee (protezione concorrente)
    static bool isRecoveryOrderOpening = false;
    if (isRecoveryOrderOpening) return false;
    isRecoveryOrderOpening = true;

    if (EnableLogging_OpenTradeRecovery)
        PrintFormat("🔄 [OpenTrade Recovery] Richiesta apertura trade di RECUPERO per ticket originale=%d su %s.",
                    original_ticket, symbol);

    //───────────────────────────────────────────────────────────────
    // 1. Recupera dettagli della posizione originale dal manager
    //───────────────────────────────────────────────────────────────
    RecoveryTriggerInfo trigger;
    if (!g_recovery_manager.GetTriggerInfo(original_ticket, trigger))
    {
        PrintFormat("❌ [OpenTrade Recovery] ERRORE: Nessun trigger trovato per ticket %d", original_ticket);
        isRecoveryOrderOpening = false;
        return false;
    }

    string recoverySymbol          = trigger.Symbol;
    ENUM_POSITION_TYPE originalPositionType = trigger.OriginalPositionType;
    double recoveryLot             = trigger.OriginalLotSize; // Lotto della posizione originale

    if (StringLen(recoverySymbol) == 0 || recoveryLot <= 0.0)
    {
        PrintFormat("❌ [OpenTrade Recovery] ERRORE: Dati trigger non validi (symbol='%s', lot=%.2f)",
                    recoverySymbol, recoveryLot);
        isRecoveryOrderOpening = false;
        return false;
    }

    // ✨ DETERMINA LA DIREZIONE OPPOSTA PER IL TRADE DI RECUPERO ✨
    ENUM_POSITION_TYPE recoveryTradeType;
    if (originalPositionType == POSITION_TYPE_BUY)
    {
        recoveryTradeType = POSITION_TYPE_SELL;
    }
    else if (originalPositionType == POSITION_TYPE_SELL)
    {
        recoveryTradeType = POSITION_TYPE_BUY;
    }
    else
    {
        PrintFormat("❌ [OpenTrade Recovery] ERRORE: Tipo di posizione originale non valido: %s", EnumToString(originalPositionType));
        isRecoveryOrderOpening = false;
        return false;
    }

    if (EnableLogging_OpenTradeRecovery)
        PrintFormat("🔄 [OpenTrade Recovery] Posizione originale: %s. Direzione trade di recupero: %s",
                    EnumToString(originalPositionType), EnumToString(recoveryTradeType));

    //───────────────────────────────────────────────────────────────
    // 3. Ottieni prezzo corrente e info simbolo (SPOSTATO IN ALTO)
    //───────────────────────────────────────────────────────────────
    double entryPrice = (recoveryTradeType == POSITION_TYPE_BUY)
                            ? SymbolInfoDouble(recoverySymbol, SYMBOL_ASK)
                            : SymbolInfoDouble(recoverySymbol, SYMBOL_BID);
    if (entryPrice <= 0)
    {
        if (EnableLogging_OpenTradeRecovery)
            PrintFormat("❌ [OpenTrade Recovery] ERRORE: Prezzo di ingresso non valido (%.5f) per %s. Annullamento.", entryPrice, recoverySymbol);
        isRecoveryOrderOpening = false;
        return false;
    }

    int digits = (int)SymbolInfoInteger(recoverySymbol, SYMBOL_DIGITS);
    double point = SymbolInfoDouble(recoverySymbol, SYMBOL_POINT);

    //───────────────────────────────────────────────────────────────
    // 4. Calcolo SL e TP dinamici per il trade di recupero (SPOSTATO IN ALTO)
    //───────────────────────────────────────────────────────────────
    double sl = 0;
    double tp = 0;

    int atrPeriods[3] = { ATR_Period1, ATR_Period2, ATR_Period3 };
    double atrValues[];

    if (!CalculateMultiPeriodATR(recoverySymbol, Timeframe_MACD_EMA, atrPeriods, atrValues))
    {
        if (EnableLogging_OpenTradeRecovery)
            PrintFormat("❌ [OpenTrade Recovery] ERRORE: ATR non disponibile per %s. Annullamento trade di recupero.", recoverySymbol);
        isRecoveryOrderOpening = false;
        return false;
    }

    double atr = atrValues[1];
    if (atr <= 0.0)
    {
        if (EnableLogging_OpenTradeRecovery)
            PrintFormat("❌ [OpenTrade Recovery] ERRORE: Valore ATR non valido (%.5f) per %s. Annullamento trade di recupero.", atr, recoverySymbol);
        isRecoveryOrderOpening = false;
        return false;
    }

    SLTPMultipliers m = CalcSLTPFromRSIADX(recoverySymbol, Timeframe_MACD_EMA, recoveryTradeType == POSITION_TYPE_BUY);

    sl = (recoveryTradeType == POSITION_TYPE_BUY)
             ? entryPrice - atr * m.slMultiplier
             : entryPrice + atr * m.slMultiplier;

    tp = (recoveryTradeType == POSITION_TYPE_BUY)
             ? entryPrice + atr * m.tpMultiplier
             : entryPrice - atr * m.tpMultiplier; // Correzione per TP SELL

    AdjustStopsIfTooClose(sl, tp, (ENUM_ORDER_TYPE)recoveryTradeType, entryPrice);

    sl = NormalizeDouble(sl, digits);
    tp = NormalizeDouble(tp, digits);

    if (EnableLogging_OpenTradeRecovery)
        PrintFormat("📈 [OpenTrade Recovery] Calcolo dinamico SL: %.5f | TP: %.5f | Entry Price: %.5f (ATR: %.5f, SL_Mult: %.2f, TP_Mult: %.2f)", sl, tp, entryPrice, atr, m.slMultiplier, m.tpMultiplier);

    // Calcola SL in pips (NECESSARIO PER CalculateTradableLot)
    double sl_pips = 0;
    if (recoveryTradeType == POSITION_TYPE_BUY)
    {
        sl_pips = (entryPrice - sl) / point;
    }
    else
    {
        sl_pips = (sl - entryPrice) / point;
    }
    sl_pips = NormalizeDouble(sl_pips, 1); // Normalizza i pips a 1 decimale

    //───────────────────────────────────────────────────────────────
    // 2. Calcolo lotto di recupero E VERIFICA MARGINE (MODIFICATO)
    //───────────────────────────────────────────────────────────────
    double desired_lot_by_multiplier = recoveryLot * RecoveryLotMultiplier;
    
    // ⭐⭐ NUOVA LOGICA: Chiamata a CalculateTradableLot che gestisce tutto ⭐⭐
    double lottoDaUsare = CalculateTradableLot(recoverySymbol, (ENUM_ORDER_TYPE)recoveryTradeType, sl_pips, EnableLogging_OpenTradeRecovery, RecoveryLotMultiplier);
    
    // Se lottoDaUsare è 0 o negativo, significa che non c'è margine o il calcolo è fallito
    if (lottoDaUsare <= 0)
    {
        if (EnableLogging_OpenTradeRecovery)
            PrintFormat("❌ [OpenTrade Recovery] ERRORE: Lotto di recupero calcolato (%.2f) non valido o insufficiente margine per ticket %d. Annullamento.", lottoDaUsare, original_ticket);
        isRecoveryOrderOpening = false;
        return false;
    }
    
    if (EnableLogging_OpenTradeRecovery)
        PrintFormat("💰 [OpenTrade Recovery] Lotto finale calcolato e verificato: %.2f (desiderato: %.2f * %.2f Moltiplicatore).",
                    lottoDaUsare, recoveryLot, RecoveryLotMultiplier);

    //───────────────────────────────────────────────────────────────
    // 5. Prepara richiesta ordine
    //───────────────────────────────────────────────────────────────
    MqlTradeRequest request;
    MqlTradeResult result;
    ZeroMemory(request);
    ZeroMemory(result);

    request.action      = TRADE_ACTION_DEAL;
    request.symbol      = recoverySymbol;
    request.volume      = lottoDaUsare;
    request.type        = (ENUM_ORDER_TYPE)recoveryTradeType;
    request.price       = entryPrice;
    request.sl          = sl;
    request.tp          = tp;
    request.deviation   = 20;
    request.magic       = RecoveryMagicNumber;
    request.comment     = "Recovery Trade";
    request.type_time   = ORDER_TIME_GTC;

    //───────────────────────────────────────────────────────────────
    // 6. Invia ordine con tentativi multipli su filling mode
    //───────────────────────────────────────────────────────────────
    bool sent = false;
    ENUM_ORDER_TYPE_FILLING cachedMode;
    
    if (HasCachedFillingMode(recoverySymbol, cachedMode))
    {
        request.type_filling = cachedMode;
        request.price = (recoveryTradeType == POSITION_TYPE_BUY) ? SymbolInfoDouble(recoverySymbol, SYMBOL_ASK) : SymbolInfoDouble(recoverySymbol, SYMBOL_BID);
        
        if (EnableLogging_OpenTradeRecovery)
            PrintFormat("📤 [OpenTrade Recovery] Tentativo da cache = %s | Prezzo = %.5f | Lotto = %.2f",
                        EnumToString(cachedMode), request.price, request.volume);
        
        bool success_send = trade.OrderSend(request, result); // Usa trade.OrderSend()
        
        if (success_send && result.retcode == TRADE_RETCODE_DONE)
        {
            sent = true;
            if (EnableLogging_OpenTradeRecovery)
                PrintFormat("✅ Ordine di RECUPERO eseguito con filling = %s (da cache)", EnumToString(cachedMode));
        }
        else
        {
            if (EnableLogging_OpenTradeRecovery)
                PrintFormat("❌ Fallito ordine di RECUPERO con filling (da cache): %s | Retcode: %d | Msg: %s",
                            EnumToString(cachedMode), result.retcode, result.comment);
        }
    }
    
    if (!sent)
    {
        // Ho rimosso ORDER_FILLING_RETURN e ORDER_FILLING_FOK dai tentativi
        // basandomi sui tuoi precedenti messaggi di errore "Unsupported filling mode".
        // Se sai che il tuo broker li supporta, ripristinali pure.
        ENUM_ORDER_TYPE_FILLING modes[] = { ORDER_FILLING_IOC }; 
        
        for (int i = 0; i < ArraySize(modes); i++)
        {
            request.type_filling = modes[i];
            request.price = (recoveryTradeType == POSITION_TYPE_BUY) ? SymbolInfoDouble(recoverySymbol, SYMBOL_ASK) : SymbolInfoDouble(recoverySymbol, SYMBOL_BID);
            
            if (EnableLogging_OpenTradeRecovery)
                PrintFormat("📤 [OpenTrade Recovery] Tentativo fallback = %s | Prezzo = %.5f | SL = %.5f | TP = %.5f | Lotto = %.2f",
                            EnumToString(modes[i]), request.price, request.sl, request.tp, request.volume);
            
            bool success_send = trade.OrderSend(request, result); // Usa trade.OrderSend()
            
            if (success_send && result.retcode == TRADE_RETCODE_DONE)
            {
                sent = true;
                CacheFillingMode(recoverySymbol, modes[i]);
                if (EnableLogging_OpenTradeRecovery)
                    PrintFormat("✅ Ordine di RECUPERO eseguito con filling = %s (fallback)", EnumToString(modes[i]));
                break;
            }
            else
            {
                if (EnableLogging_OpenTradeRecovery)
                    PrintFormat("❌ Fallito ordine di RECUPERO con filling = %s | Retcode: %d | Msg: %s",
                                EnumToString(modes[i]), result.retcode, result.comment);
            }
        }
    }
    
    if (!sent)
    {
        if (EnableLogging_OpenTradeRecovery)
            PrintFormat("❌ [OpenTrade Recovery] Ordine di RECUPERO FALLITO TOTALMENTE. Retcode: %d | Msg: %s | Lotto=%.2f | SL=%.5f | TP=%.5f | Filling=%d",
                        result.retcode, result.comment, request.volume, request.sl, request.tp, request.type_filling);
        // Incrementa il conteggio dei tentativi falliti
        g_recovery_attempts_map.Insert(original_ticket, g_recovery_attempts_map.At(original_ticket) + 1);
        isRecoveryOrderOpening = false;
        return false;
    }

    //───────────────────────────────────────────────────────────────
    // 7. Ordine eseguito correttamente → log finale e SALVA IL TICKET DI RECUPERO
    //───────────────────────────────────────────────────────────────
    if (result.deal != 0) // Assicurati che un deal sia stato generato
    {
        g_recovery_manager.SetRecoveryTradeTicket(original_ticket, result.deal);
        if (EnableLogging_OpenTradeRecovery)
            PrintFormat("✅ [OpenTrade Recovery] Salvato ticket di recupero %d per trigger originale %d.", result.deal, original_ticket);
    }
    else
    {
        // Questo è un caso d'errore, ma per completezza:
        if (EnableLogging_OpenTradeRecovery)
            PrintFormat("⚠️ [OpenTrade Recovery] Ordine eseguito ma deal ticket è 0 per trigger originale %d.", original_ticket);
    }

    if (EnableLogging_OpenTradeRecovery)
        PrintFormat("✅ [OpenTrade Recovery] Ordine APERTO: %s | Direzione: %s | Lotto: %.2f | Entry: %.5f | SL: %.5f | TP: %.5f | Filling: %s | Magic: %d",
                    recoverySymbol,
                    EnumToString(recoveryTradeType),
                    lottoDaUsare,
                    result.price,
                    sl,
                    tp,
                    EnumToString(request.type_filling),
                    request.magic);

    isRecoveryOrderOpening = false;
    return true;
}

#endif // __OPEN_RECOVERY_TRADE_MQH__