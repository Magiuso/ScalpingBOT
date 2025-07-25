//+------------------------------------------------------------------+
//|              ExtremesPrice.mqh                                   |
//|              Modulo aggiornamento del prezzo massimo/minimo      |
//+------------------------------------------------------------------+
#ifndef __EXTREMESPRICE_MQH__
#define __EXTREMESPRICE_MQH__


//+------------------------------------------------------------------+
//| 🚀 Funzione per aggiornare PeakPrice e WorstPrice                |
//|    - PeakPrice: miglior prezzo raggiunto (più favorevole)        |
//|    - WorstPrice: peggior prezzo raggiunto (più sfavorevole)      |
//|    ⚠️ Deve essere chiamata ad ogni tick (OnTick)                 |
//+------------------------------------------------------------------+
void AggiornaExtremesPrice()
{
    int total = PositionsTotal();
    for (int i = 0; i < total; i++)
    {
        ulong ticket = PositionGetTicket(i);
        if (!PositionSelectByTicket(ticket))
            continue;

        TrackOrder(ticket);  // 📌 Assicura che l'ordine sia tracciato

        int idx = FindOrderIndex(ticket);
        if (idx == -1)
            continue;

        string symbol = PositionGetString(POSITION_SYMBOL);
        ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
        double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
        double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);

        // 📈 Prezzo attuale rilevante per l’operazione
        double price = (type == POSITION_TYPE_BUY) ? bid : ask;

        //===============================================================
        // ✅ AGGIORNAMENTO PEAKPRICE
        //===============================================================
        if (type == POSITION_TYPE_BUY)
        {
            // 🔼 BUY → aggiorna solo se il prezzo sale
            if (orders[idx].peakPrice == 0.0 || bid > orders[idx].peakPrice)
            {
                orders[idx].peakPrice = bid;
                if (EnableLogging_ExtremesPrice)
                    PrintFormat("🔼 [BUY] PeakPrice aggiornato per %s: %.5f", symbol, bid);
            }
        }
        else if (type == POSITION_TYPE_SELL)
        {
            // 🔽 SELL → aggiorna solo se il prezzo scende
            if (orders[idx].peakPrice == 0.0 || ask < orders[idx].peakPrice)
            {
                orders[idx].peakPrice = ask;
                if (EnableLogging_ExtremesPrice)
                    PrintFormat("🔽 [SELL] PeakPrice aggiornato per %s: %.5f", symbol, ask);
            }
        }

        //===============================================================
        // ⚠️ AGGIORNAMENTO WORSTPRICE
        //===============================================================
        if (type == POSITION_TYPE_BUY)
        {
            // 🔻 BUY → aggiorna solo se il prezzo scende (sfavorevole)
            if (orders[idx].worstPrice == 0.0 || bid < orders[idx].worstPrice)
            {
                orders[idx].worstPrice = bid;
                if (EnableLogging_ExtremesPrice)
                    PrintFormat("🔻 [BUY] WorstPrice aggiornato per %s: %.5f", symbol, bid);
            }
        }
        else if (type == POSITION_TYPE_SELL)
        {
            // 🔺 SELL → aggiorna solo se il prezzo sale (sfavorevole)
            if (orders[idx].worstPrice == 0.0 || ask > orders[idx].worstPrice)
            {
                orders[idx].worstPrice = ask;
                if (EnableLogging_ExtremesPrice)
                    PrintFormat("🔺 [SELL] WorstPrice aggiornato per %s: %.5f", symbol, ask);
            }
        }
    }
}

#endif // __EXTREMESPRICE_MQH__
