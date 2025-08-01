//+------------------------------------------------------------------+
//|              OpenTrade.mqh                                       |
//|   📈 Apertura ordine con SL/TP dinamici e filling cache          |
//+------------------------------------------------------------------+
#ifndef __OPENTRADE_MQH__
#define __OPENTRADE_MQH__


#include <ScalpingBot\Utility.mqh>
#include <ScalpingBot\SymbolFillingCache.mqh>
#include <ScalpingBot\EntryManager.mqh>
#include <ScalpingBot\RSI_SLTP_Dinamici.mqh>
#include <ScalpingBot\RecoveryTriggerManager.mqh>

//+------------------------------------------------------------------+
//| 🚀 Funzione principale di apertura ordine (EntryManager)         |
//|                                                                  |
//| 🔹 Analizza direzione BUY/SELL tramite EntryScore                |
//| 🔹 Calcola SL/TP dinamici con ATR e RSI/ADX                      |
//| 🔹 Calcola lotto in base al rischio e SL                        |
//| 🔹 Invia ordine con fallback su 3 filling mode                  |
//+------------------------------------------------------------------+
void OpenTrade(string symbol)
{
    // 🔒 Evita chiamate multiple simultanee (protezione concorrente)
    static bool isOrderOpening = false;
    if (isOrderOpening) return;
    isOrderOpening = true;

    //───────────────────────────────────────────────────────────────
    // 🧠 1. Calcolo punteggio ingresso e direzione finale
    //───────────────────────────────────────────────────────────────
    EntryScoreResult entryResult = CalculateEntryScore(symbol, Timeframe_MACD_EMA, true); // Il 'true' per ora è placeholder
    
    // ❌ Se la strategia non conferma ingresso → esce
    if (!entryResult.entryConfirmed)
    {
        if (EnableLogging_OpenTrade)
            Print("❌ [OpenTrade] Nessun segnale valido → uscita");
    
        isOrderOpening = false;
        return;
    }
    
    // 🧭 Determina direzione finale già calcolata
    ENUM_ORDER_TYPE direction = entryResult.directionFinal ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
    
    // 🔁 Salva direzione globale per trailing o gestione post-trade (assicurati che isBuySignal sia dichiarata globalmente)
    // isBuySignal = (direction == ORDER_TYPE_BUY); 
    // ^ Commentato perché non dichiarata in questo snippet, ma se usata altrove, assicurati che sia globale

    // 📝 Log direzione e punteggio
    if (EnableLogging_OpenTrade)
    {
        PrintFormat("✅ [ENTRY] Direzione finale: %s | Punteggio totale: %.2f / Soglia: %.2f",
                    direction == ORDER_TYPE_BUY ? "BUY 🟢" : "SELL 🔴",
                    entryResult.scoreTotal,
                    EntryThreshold);
    }
    
    //───────────────────────────────────────────────────────────────
    // 📍 2. Ottieni prezzo corrente e info simbolo
    //───────────────────────────────────────────────────────────────

    double entryPrice = (direction == ORDER_TYPE_BUY)
                      ? SymbolInfoDouble(symbol, SYMBOL_ASK)    // 🔼 Prezzo ASK se BUY
                      : SymbolInfoDouble(symbol, SYMBOL_BID);   // 🔽 Prezzo BID se SELL

    // Verifica validità prezzo
    if (entryPrice <= 0)
    {
        if (EnableLogging_OpenTrade)
            PrintFormat("❌ [OpenTrade] ERRORE: Prezzo di ingresso non valido (%.5f) per %s. Annullamento.", entryPrice, symbol);
        isOrderOpening = false;
        return;
    }

    double point   = GetSymbolPoint(symbol);    // 🎯 Valore di un punto per il simbolo
    int digits     = GetSymbolDigits(symbol);   // 🔢 Numero di decimali del simbolo

    //───────────────────────────────────────────────────────────────
    // ⏱️ 3. Verifica cooldown e limite ordini
    //───────────────────────────────────────────────────────────────

    // ⛔ Controlla che sia passato il tempo minimo dall’ultimo ordine
    if ((TimeCurrent() - lastMACDTradeTime) < CooldownMACDSeconds)
    {
        if (EnableLogging_OpenTrade)
            PrintFormat("⏳ [OpenTrade] Cooldown attivo: attendere %d secondi. Ordine annullato.", CooldownMACDSeconds);
        isOrderOpening = false;
        return;
    }

    // ⛔ Evita di aprire troppi ordini contemporanei sullo stesso asset
    if (GetOpenPositionsCount(symbol) >= MaxOrdersPerAsset)
    {
        if (EnableLogging_OpenTrade)
            PrintFormat("🚫 [OpenTrade] Limite massimo ordini per asset (%d) raggiunto. Ordine annullato.", MaxOrdersPerAsset);
        isOrderOpening = false;
        return;
    }

    //───────────────────────────────────────────────────────────────
    // 📊 4. Calcolo SL e TP dinamici con ATR e RSI/ADX
    //───────────────────────────────────────────────────────────────

    // 🧪 Richiedi ATR su 3 periodi configurabili
    int atrPeriods[3] = { ATR_Period1, ATR_Period2, ATR_Period3 };
    double atrValues[];

    // ❌ Se ATR non disponibile → esce
    if (!CalculateMultiPeriodATR(symbol, Timeframe_MACD_EMA, atrPeriods, atrValues))
    {
        if (EnableLogging_OpenTrade)
            PrintFormat("❌ [OpenTrade] ERRORE: ATR non disponibile per %s. Annullamento trade.", symbol);
        isOrderOpening = false;
        return;
    }

    double atr = atrValues[1]; // 🎯 Usa ATR medio come riferimento
    if (atr <= 0.0)
    {
        if (EnableLogging_OpenTrade)
            PrintFormat("❌ [OpenTrade] ERRORE: Valore ATR non valido (%.5f) per %s. Annullamento trade.", atr, symbol);
        isOrderOpening = false;
        return;
    }

    // 📈 Calcola moltiplicatori SL/TP dinamici da RSI + ADX
    SLTPMultipliers m = CalcSLTPFromRSIADX(symbol, Timeframe_MACD_EMA, direction == ORDER_TYPE_BUY);

    // 🛑 Calcolo SL dinamico
    double sl = (direction == ORDER_TYPE_BUY)
              ? entryPrice - atr * m.slMultiplier    // SL sotto il prezzo per BUY
              : entryPrice + atr * m.slMultiplier;  // SL sopra il prezzo per SELL

    // 🎯 Calcolo TP dinamico
    double tp = (direction == ORDER_TYPE_BUY)
              ? entryPrice + atr * m.tpMultiplier
              : entryPrice - atr * m.tpMultiplier; // Correzione per TP SELL

    // 🔧 Adatta SL/TP se troppo vicini (protezioni broker)
    AdjustStopsIfTooClose(sl, tp, direction, entryPrice);

    //───────────────────────────────────────────────────────────────
    // 💰 5. Calcolo lotto in base al rischio e SL (AGGIORNATO)
    //───────────────────────────────────────────────────────────────

    double slPips = GetSLPips(entryPrice, sl, symbol); // 📏 SL in pip per calcolo lotto
    
    // ⭐⭐ NUOVA CHIAMATA ALLA FUNZIONE DI CALCOLO LOTTO UNIFICATA ⭐⭐
    double lottoDaUsare = CalculateTradableLot(symbol, direction, slPips, EnableLogging_OpenTrade, 1.0); // 🧮 Lotto calcolato dinamicamente
    
    // ❌ Verifica se il lotto è valido dopo il calcolo e il controllo margine
    if (lottoDaUsare <= 0.0)
    {
        if (EnableLogging_OpenTrade)
        {
            PrintFormat("❌ [%s] Lotto calcolato (%.2f) non valido o insufficiente margine per aprire l'ordine con SL di %.2f pip. Ordine annullato.", symbol, lottoDaUsare, slPips);
        }
        isOrderOpening = false; // Impedisci l'invio dell'ordine
        return; // Esci dalla funzione
    }
    
    // 📝 Log del lotto calcolato
    if (EnableLogging_OpenTrade)
    {
        PrintFormat("🧮 [%s] Calcolo lotto: SL=%.2f pip → Lotto=%.2f", symbol, slPips, lottoDaUsare);
    }
    
    // 📏 Normalizza SL/TP ai decimali del simbolo
    sl = NormalizeDouble(sl, digits);
    tp = NormalizeDouble(tp, digits);

    //───────────────────────────────────────────────────────────────
    // 📦 6. Prepara richiesta ordine
    //───────────────────────────────────────────────────────────────

    MqlTradeRequest request;        // 📤 Richiesta da inviare
    MqlTradeResult result;          // 📥 Risultato ricevuto
    // MqlTradeCheckResult check;     // 🔎 (non usato qui, ma definito - rimosso perché non necessario)

    ZeroMemory(request);
    ZeroMemory(result);
    // ZeroMemory(check);             // Rimosso

    // 🔧 Setup parametri base richiesta
    request.action      = TRADE_ACTION_DEAL;
    request.symbol      = symbol;
    request.volume      = lottoDaUsare;
    request.type        = direction;
    request.price       = entryPrice;
    request.sl          = sl;
    request.tp          = tp;
    request.deviation   = 20; // Utilizza la tua variabile Deviation se l'hai definita come input globale
    request.magic       = MagicNumber_MACD; // Utilizza il Magic Number specifico per i trade MACD
    request.type_time   = ORDER_TIME_GTC;

    //───────────────────────────────────────────────────────────────
    // 🔁 7. Invia ordine con tentativi multipli su filling mode
    //───────────────────────────────────────────────────────────────    
    bool sent = false;
    ENUM_ORDER_TYPE_FILLING cachedMode;
    
    // 📌 7.1 – Verifica se esiste un filling mode già noto per questo simbolo
    if (HasCachedFillingMode(symbol, cachedMode))
    {
        request.type_filling = cachedMode;
    
        // 🆕 Aggiorna il prezzo al tick corrente
        request.price = (direction == ORDER_TYPE_BUY)
                      ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                      : SymbolInfoDouble(symbol, SYMBOL_BID);
    
        // 📝 Log tentativo da cache
        if (EnableLogging_OpenTrade)
        {
            PrintFormat("📤 [OrderSend] Tentativo da cache = %s | Prezzo = %.5f | Lotto = %.2f",
                        EnumToString(cachedMode), request.price, request.volume);
        }
    
        // 🚀 Invio ordine (assicurati di usare un oggetto CTrade o una funzione globale Trade.OrderSend)
        bool success = trade.OrderSend(request, result); // Assicurati 'trade' sia un oggetto CTrade globale
    
        // ✅ Se ordine eseguito correttamente
        if (success && result.retcode == TRADE_RETCODE_DONE)
        {
            sent = true;
    
            if (EnableLogging_OpenTrade)
                PrintFormat("✅ Ordine eseguito con filling = %s (da cache)", EnumToString(cachedMode));
        }
        else
        {
            if (EnableLogging_OpenTrade)
                PrintFormat("❌ Fallito con filling (da cache): %s | Retcode: %d | Msg: %s",
                            EnumToString(cachedMode), result.retcode, result.comment);
        }
    }
    
    // 🔁 7.2 – Se ordine non inviato dalla cache → ciclo fallback
    if (!sent)
    {
        // Ho lasciato solo IOC per sicurezza, basandomi su precedenti conversazioni su filling mode non supportati.
        // Se sei sicuro che il tuo broker supporti RETURN o FOK, puoi riaggiungerli.
        ENUM_ORDER_TYPE_FILLING modes[] = { ORDER_FILLING_IOC /*, ORDER_FILLING_RETURN, ORDER_FILLING_FOK*/ }; 
    
        for (int i = 0; i < ArraySize(modes); i++)
        {
            request.type_filling = modes[i];
    
            // 🆕 Aggiorna il prezzo corrente (tick real-time)
            request.price = (direction == ORDER_TYPE_BUY)
                          ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                          : SymbolInfoDouble(symbol, SYMBOL_BID);
    
            // 📝 Log tentativo corrente
            if (EnableLogging_OpenTrade)
            {
                PrintFormat("📤 [OrderSend] Tentativo fallback = %s | Prezzo = %.5f | SL = %.5f | TP = %.5f | Lotto = %.2f",
                            EnumToString(modes[i]), request.price, request.sl, request.tp, request.volume);
            }
    
            // 🚀 Invio ordine (assicurati di usare un oggetto CTrade o una funzione globale Trade.OrderSend)
            bool success = trade.OrderSend(request, result); // Assicurati 'trade' sia un oggetto CTrade globale
    
            // ✅ Ordine eseguito → salviamo filling e usciamo
            if (success && result.retcode == TRADE_RETCODE_DONE)
            {
                sent = true;
                CacheFillingMode(symbol, modes[i]); // 💾 Salva il filling funzionante
    
                if (EnableLogging_OpenTrade)
                    PrintFormat("✅ Ordine eseguito con filling = %s (fallback)", EnumToString(modes[i]));
                break;
            }
            else
            {
                if (EnableLogging_OpenTrade)
                    PrintFormat("❌ Fallito con filling = %s | Retcode: %d | Msg: %s",
                                EnumToString(modes[i]), result.retcode, result.comment);
            }
        }
    }
    
    // 🚫 7.3 – Fallimento totale
    if (!sent)
    {
        PrintFormat("❌ [%s] Ordine fallito. Retcode: %d | Msg: %s | Lotto=%.2f | SL=%.5f | TP=%.5f | Filling=%d",
                    symbol, result.retcode, result.comment, request.volume, request.sl, request.tp, request.type_filling);
        isOrderOpening = false;
        return;
    }

    //───────────────────────────────────────────────────────────────
    // ✅ 8. Ordine eseguito correttamente → log finale
    //───────────────────────────────────────────────────────────────
    PrintFormat("✅ [%s] Ordine APERTO: %s | Lotto: %.2f | Entry: %.5f | SL: %.5f | TP: %.5f | Filling: %s | Magic: %d",
                symbol,
                EnumToString(direction),
                lottoDaUsare,
                result.price,
                sl,
                tp,
                EnumToString(request.type_filling),
                request.magic);

    // 🕒 Aggiorna ultimo tempo apertura ordine
    lastMACDTradeTime = TimeCurrent();
    isOrderOpening = false;
}

#endif // __OPENTRADE_MQH__
