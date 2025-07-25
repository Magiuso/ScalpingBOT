//+------------------------------------------------------------------+
//|              TrailingStop.mqh (Versione 3.5)                    |
//|              Modulo di gestione dinamica Trailing Stop           |
//+------------------------------------------------------------------+
#ifndef __TRAILINGSTOP_MQH__
#define __TRAILINGSTOP_MQH__

#include <Trade\Trade.mqh>    // 📦 Gestione ordini (serve per ClosePositionByTicket)

//+------------------------------------------------------------------+
//| 📦 Struttura dati per tracciare gli ordini attivi                |
//+------------------------------------------------------------------+
struct TrackedOrder {
    ulong ticket;                            // 🏷️ ID ordine
    datetime lastModify;                     // 🕓 Ultima modifica SL
    double peakPrice;                        // 📈 PeakPrice raggiunto (BUY) → massimo
    double worstPrice;                       // 📉 WorstPrice raggiunto (SELL) → minimo
    bool failedModify;                       // ❌ Flag antiflood: ultima modifica SL fallita
    bool isTrailingActive;                   // ✅ Trailing attivo per questo ordine
    double lastLoggedDistanceToStart;         // ✅ Ultima distanza log TrailingStart (per cooldown log)
    int lastModifyErrorCode;                  // ✅ Ultimo codice errore modifica SL (per log antiflood)
};

TrackedOrder orders[];              // 📊 Array degli ordini tracciati
int trackedOrders = 0;               // 🔢 Numero ordini tracciati

//+------------------------------------------------------------------+
//| 🔍 Trova indice ordine tracciato                                  |
//+------------------------------------------------------------------+
int FindOrderIndex(ulong ticket)
{
    for (int i = 0; i < trackedOrders; i++)
    {
        if (orders[i].ticket == ticket)
            return i;
    }
    return -1;
}

//+------------------------------------------------------------------+
//| ➕ Aggiunge un nuovo ordine da tracciare                         |
//+------------------------------------------------------------------+
void TrackOrder(ulong ticket)
{
    if (FindOrderIndex(ticket) == -1)
    {
        if (!PositionSelectByTicket(ticket))
            return;

        double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
        ArrayResize(orders, trackedOrders + 1);

        orders[trackedOrders].ticket = ticket;
        orders[trackedOrders].lastModify = 0;
        orders[trackedOrders].peakPrice = entryPrice;
        orders[trackedOrders].worstPrice = entryPrice;
        orders[trackedOrders].failedModify = false;
        orders[trackedOrders].isTrailingActive = false;
        orders[trackedOrders].lastLoggedDistanceToStart = 0;
        orders[trackedOrders].lastModifyErrorCode = 0;

        // ✅ Log StopLevel broker alla creazione ordine
        string symbol = PositionGetString(POSITION_SYMBOL);
        long stopLevelPoints = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
        if (stopLevelPoints == 0)
        {
            stopLevelPoints = 300; // Fallback per tester, assumendo 30 pips per un asset a 5 decimali
            PrintFormat("⚠️ [%s] StopLevel broker = 0 -> fallback impostato manualmente a %d points (modalità tester)", symbol, stopLevelPoints);
        }
        else
        {
            PrintFormat("📝 [%s] StopLevel broker corrente (valore reale broker) = %d points", symbol, stopLevelPoints);
        }

        trackedOrders++;
    }
}

//+------------------------------------------------------------------+
//| 🚀 Gestione Trailing Stop completa                               |
//+------------------------------------------------------------------+
void GestioneTrailingStop()
{
    if (!EnableTrailingStop)
        return;

    // Sincronizza e rimuovi ordini chiusi prima di elaborare
    RefreshTrackedOrders(); 

    for (int i = 0; i < trackedOrders; i++) // Itera solo sugli ordini TRACCIATI
    {
        ulong ticket = orders[i].ticket;
        if (!PositionSelectByTicket(ticket))
        {
            // Questa posizione è stata chiusa o non esiste più, verrà rimossa da RefreshTrackedOrders al prossimo tick
            continue; 
        }

        int idx = i; // Ora idx è già corretto perché stiamo iterando su 'orders'

        string symbol = PositionGetString(POSITION_SYMBOL);
        double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
        ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
        double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
        double currentSL = PositionGetDouble(POSITION_SL);
        double price = (type == POSITION_TYPE_BUY) ? SymbolInfoDouble(symbol, SYMBOL_BID) : SymbolInfoDouble(symbol, SYMBOL_ASK);
        double profitDistance = (type == POSITION_TYPE_BUY) ? (price - entryPrice) : (entryPrice - price);

        // Calcola i valori effettivi di TrailingStart e TrailingStep usando la tua funzione PipsToPoints
        double actualTrailingStartValue = PipsToPoints(symbol, TrailingStart);
        double actualTrailingStepValue  = PipsToPoints(symbol, TrailingStep);

        // Cooldown log TrailingStart
        double distanceToStart = actualTrailingStartValue - profitDistance; 
        if (!orders[idx].isTrailingActive && distanceToStart > 0)
        {
            double delta = MathAbs(distanceToStart - orders[idx].lastLoggedDistanceToStart);
            if (delta >= PipsToPoints(symbol, LogCooldownPipsThreshold)) 
            {
                if (EnableLogging_Trailing)
                PrintFormat("⏸️ [%s] Mancano %.1f pips al TrailingStart", symbol, distanceToStart / GetPipSize(symbol));
                orders[idx].lastLoggedDistanceToStart = distanceToStart;
            }
        }

        // Attivazione Trailing
        if (!orders[idx].isTrailingActive && profitDistance >= actualTrailingStartValue)
        {
            orders[idx].isTrailingActive = true;
            orders[idx].failedModify = false; // Reset antiflood on activation
            if (EnableLogging_Trailing)
            PrintFormat("🟢 [%s] Trailing STARTED! Profit %.1f pips (%.2f USD)", symbol, profitDistance / GetPipSize(symbol), profitDistance);
        }

        if (!orders[idx].isTrailingActive)
            continue;

        // Antiflood attivo - Blocco logico prima della modifica
        if (orders[idx].failedModify)
        {
            if (EnableLogging_Trailing)
            PrintFormat("⛔ [%s] Antiflood attivo → SL bloccato. Ultimo errore: %d", symbol, orders[idx].lastModifyErrorCode);
            // Sblocco antiflood su micro-movimento
            double movement = MathAbs(price - ((type == POSITION_TYPE_BUY) ? orders[idx].peakPrice : orders[idx].worstPrice));
            if (movement >= PipsToPoints(symbol, MinMovementForUnlockPips))
            {
                orders[idx].failedModify = false;
                orders[idx].lastModifyErrorCode = 0; // Reset error code
                if (EnableLogging_Trailing)
                PrintFormat("✅ [%s] Antiflood disattivato: micro-movimento rilevato di %.1f pips", symbol, movement / GetPipSize(symbol));
            }
            else
            {
                continue; // Continua a bloccare se antiflood è attivo e nessun movimento sufficiente
            }
        }
        
        // Aggiorna Peak/Worst
        if (type == POSITION_TYPE_BUY)
        {
            if (price > orders[idx].peakPrice)
            {
                orders[idx].peakPrice = price;
                orders[idx].failedModify = false; // Reset antiflood on new peak
                if (EnableLogging_Trailing)
                PrintFormat("🔼 [%s] PeakPrice aggiornato: %.5f", symbol, price);
            }
        }
        else // POSITION_TYPE_SELL
        {
            if (price < orders[idx].worstPrice)
            {
                orders[idx].worstPrice = price;
                orders[idx].failedModify = false; // Reset antiflood on new worst
                if (EnableLogging_Trailing)
                PrintFormat("🔽 [%s] WorstPrice aggiornato: %.5f", symbol, price);
            }
        }

        // Calcola il nuovo SL target basato sul Peak/Worst Price
        double newSL = (type == POSITION_TYPE_BUY) ? (orders[idx].peakPrice - actualTrailingStepValue)
                                                   : (orders[idx].worstPrice + actualTrailingStepValue);

        // Assicurati che il nuovo SL sia PIÙ FAVOREVOLE o diverso dall'attuale
        bool shouldModify = false;
        if (type == POSITION_TYPE_BUY)
        {
            if (newSL > currentSL) shouldModify = true; 
        }
        else // SELL
        {
            if (newSL < currentSL) shouldModify = true; 
        }
        
        if (shouldModify)
        {
            // Controllo del cooldown tra modifiche per prevenire flooding
            if (TimeCurrent() - orders[idx].lastModify < CooldownModifySec)
            {
                continue;
            }
            
            // Verifica la distanza minima del broker con un buffer di sicurezza
            if (CheckTrailingSLDistance_v2(symbol, price, newSL, type))
            {
                // SL troppo vicino, non procedere con la modifica
                orders[idx].failedModify = true; // Imposta antiflood
                orders[idx].lastModifyErrorCode = 10014;
                if (EnableLogging_Trailing) 
                PrintFormat("❌ [%s] SKIPPING SL modify: %d (Invalid Stops) due to distance from current price. New SL: %.5f, Current Price: %.5f", symbol, orders[idx].lastModifyErrorCode, newSL, price);
                continue;
            }

            double tp = PositionGetDouble(POSITION_TP); // Mantieni il TP esistente
            if (SafeModifySL(symbol, ticket, newSL, tp)) // CHIAMATA: A funzione esterna (es. da Utility)
            {
                orders[idx].lastModify = TimeCurrent();
                orders[idx].failedModify = false;
                orders[idx].lastModifyErrorCode = 0; 
                if (EnableLogging_Trailing)
                PrintFormat("✅ [%s] SL aggiornato a %.5f (TrailingStep %.1f pips)", symbol, newSL, TrailingStep);
            }
            else
            {
                // Errore reale da OrderSend
                int err = GetLastError();
                orders[idx].failedModify = true;
                orders[idx].lastModifyErrorCode = err;
                HandleTradeError(err, "OrderSend SLTP", symbol); // CHIAMATA: Da Utility
            }
        }
    }
}

//+------------------------------------------------------------------+
//| 🔒 Calcola StopLevel con buffer (Interna a TrailingStop.mqh)   |
//| Ritorna lo StopLevel base del broker (senza buffer extra)        |
//+------------------------------------------------------------------+
double GetStopLevelWithBuffer(string symbol)
{
    long stopLevelPoints = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
    if (stopLevelPoints == 0) stopLevelPoints = 10; // Fallback per sicurezza se broker restituisce 0 (1 pip)

    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    return stopLevelPoints * point; // Ritorna lo StopLevel in punti monetari
}

//+------------------------------------------------------------------+
//| 🔍 Controllo distanza minima SL (v2 - Robust)                    |
//| Include buffer di sicurezza e controllo SL vs prezzo corrente    |
//+------------------------------------------------------------------+
bool CheckTrailingSLDistance_v2(string symbol, double currentPrice, double newSL, ENUM_POSITION_TYPE type)
{
    // Usa la funzione interna per ottenere lo StopLevel base
    double baseStopLevel = GetStopLevelWithBuffer(symbol); 
    
    // Aggiungi un piccolo buffer di sicurezza usando l'input MinSLBufferPips
    double safetyBuffer = PipsToPoints(symbol, MinSLBufferPips); // ✅ ORA USA MinSLBufferPips
    
    double requiredMinDistance = baseStopLevel + safetyBuffer;
    
    double distance = (type == POSITION_TYPE_BUY) ? (currentPrice - newSL)
                                                   : (newSL - currentPrice);

    // Se la distanza del nuovo SL dal prezzo corrente è inferiore al minimo richiesto
    if (distance < requiredMinDistance)
    {
        if (EnableLogging_Trailing)
        PrintFormat("❌ [%s] SL troppo vicino. Distanza attuale SL-Prezzo: %.5f. Requisito Minimo Broker: %.5f (StopLevel=%.5f + Buffer=%.5f).", 
                    symbol, distance, requiredMinDistance, baseStopLevel, safetyBuffer);
        return true; 
    }

    // Se lo SL è al diSotto/Sopra del prezzo corrente (già "hit")
    if (type == POSITION_TYPE_BUY && newSL >= currentPrice)
    {
        if (EnableLogging_Trailing)
        PrintFormat("❌ [%s] SL di BUY (%.5f) è maggiore o uguale al prezzo Bid corrente (%.5f).", symbol, newSL, currentPrice);
        return true;
    }
    if (type == POSITION_TYPE_SELL && newSL <= currentPrice)
    {
        if (EnableLogging_Trailing)
        PrintFormat("❌ [%s] SL di SELL (%.5f) è minore o uguale al prezzo Ask corrente (%.5f).", symbol, newSL, currentPrice);
        return true;
    }

    return false;
}

//+------------------------------------------------------------------+
//| 🔄 RefreshTrackedOrders: sincronizza orders[] con MT5 positions |
//| Rimuove posizioni chiuse e aggiunge nuove attive                 |
//+------------------------------------------------------------------+
void RefreshTrackedOrders()
{
    TrackedOrder tempOrders[]; // Array temporaneo per costruire la nuova lista
    int newTrackedCount = 0;

    // Fase 1: Copia solo le posizioni attualmente aperte dall'array orders[] esistente
    for (int i = 0; i < trackedOrders; i++)
    {
        ulong ticket = orders[i].ticket;
        if (PositionSelectByTicket(ticket)) // Se la posizione è ancora aperta
        {
            ArrayResize(tempOrders, newTrackedCount + 1);
            tempOrders[newTrackedCount] = orders[i];
            newTrackedCount++;
        }
    }
    
    // Fase 2: Aggiungi nuove posizioni che non sono ancora tracciate
    int totalPositionsInTerminal = PositionsTotal();
    for (int i = 0; i < totalPositionsInTerminal; i++)
    {
        ulong ticket = PositionGetTicket(i);
        // Verifica se la posizione è già nell'array temporaneo
        bool found = false;
        for (int j = 0; j < newTrackedCount; j++)
        {
            if (tempOrders[j].ticket == ticket)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            // Posizione aperta ma non ancora tracciata, aggiungila
            if (PositionSelectByTicket(ticket)) // Doppia verifica per sicurezza
            {
                ArrayResize(tempOrders, newTrackedCount + 1);
                tempOrders[newTrackedCount].ticket = ticket;
                tempOrders[newTrackedCount].lastModify = 0;
                tempOrders[newTrackedCount].peakPrice = PositionGetDouble(POSITION_PRICE_OPEN); // Inizializza con entry price
                tempOrders[newTrackedCount].worstPrice = PositionGetDouble(POSITION_PRICE_OPEN);
                tempOrders[newTrackedCount].failedModify = false;
                tempOrders[newTrackedCount].isTrailingActive = false;
                tempOrders[newTrackedCount].lastLoggedDistanceToStart = 0;
                tempOrders[newTrackedCount].lastModifyErrorCode = 0;
                
                // Stampa info StopLevel broker anche qui per i nuovi ordini
                string symbol = PositionGetString(POSITION_SYMBOL);
                long stopLevelPoints = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
                if (stopLevelPoints == 0) {
                    PrintFormat("⚠️ [%s] StopLevel broker = 0 -> fallback impostato manualmente a 300 points (RefreshTrackedOrders)", symbol);
                } else {
                    PrintFormat("📝 [%s] StopLevel broker corrente (valore reale broker) = %d points (RefreshTrackedOrders)", symbol, stopLevelPoints);
                }
                newTrackedCount++;
            }
        }
    }

    // Sostituisci l'array orders[] con l'array temporaneo
    ArrayResize(orders, newTrackedCount);
    for (int i = 0; i < newTrackedCount; i++)
    {
        orders[i] = tempOrders[i];
    }
    trackedOrders = newTrackedCount;

    // PrintFormat("ℹ️ RefreshTrackedOrders: Trovate %d posizioni attive.", trackedOrders); // Per debugging
}

//+------------------------------------------------------------------+
//| 🏁 Chiudi una posizione tramite ticket (Interna a TrailingStop.mqh) |
//+------------------------------------------------------------------+
bool ClosePositionByTicket(ulong ticket)
{
    if (!PositionSelectByTicket(ticket))
    {
        PrintFormat("❌ [ClosePosition] Ticket %d non trovato.", ticket);
        return false;
    }

    string symbol = PositionGetString(POSITION_SYMBOL);
    double volume = PositionGetDouble(POSITION_VOLUME);
    ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

    MqlTradeRequest request;
    MqlTradeResult result;
    ZeroMemory(request);
    ZeroMemory(result);

    request.action = TRADE_ACTION_DEAL;
    request.symbol = symbol;
    request.volume = volume;
    request.position = ticket;
    request.type = (type == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
    request.price = (type == POSITION_TYPE_BUY) ? SymbolInfoDouble(symbol, SYMBOL_BID)
                                                 : SymbolInfoDouble(symbol, SYMBOL_ASK);
    request.deviation = 50; // Puoi parametrizarlo se necessario
    request.type_filling = ORDER_FILLING_IOC;
    request.type_time = ORDER_TIME_GTC;

    if (!OrderSend(request, result) || result.retcode != TRADE_RETCODE_DONE)
    {
        // Utilizziamo HandleTradeError come suggerito per consistenza
        HandleTradeError(result.retcode, "ClosePositionByTicket", symbol); // CHIAMATA: Da Utility
        return false;
    }

    PrintFormat("✅ [ClosePosition] Posizione chiusa: Ticket %d.", ticket);
    return true;
}

#endif // __TRAILINGSTOP_MQH__