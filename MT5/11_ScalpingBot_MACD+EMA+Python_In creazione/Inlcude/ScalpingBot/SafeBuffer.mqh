//+------------------------------------------------------------------+
//| SafeBuffer.mqh - Protezione CopyBuffer e CopyTime               |
//+------------------------------------------------------------------+
#ifndef __SAFEBUFFER_MQH__
#define __SAFEBUFFER_MQH__

//+------------------------------------------------------------------+
//| 📥 Lettura sicura da buffer indicatore                          |
//| Restituisce true se tutti i dati richiesti sono stati copiati   |
//+------------------------------------------------------------------+
bool SafeCopyBuffer(int handle, int bufferIndex, int startShift, int count, double &out[])
{
    // ❌ Handle non valido
    if (handle == INVALID_HANDLE)
        return false;

    // ❌ Grafico non ancora pronto
    if (Bars(_Symbol, _Period) < count + startShift + 1)
        return false;

    // ✅ Imposta ordine temporale normale: [0] = candela attuale
    ArraySetAsSeries(out, true);

    // ✅ Legge buffer
    int copied = CopyBuffer(handle, bufferIndex, startShift, count, out);
    return (copied == count);
}

//+------------------------------------------------------------------+
//| 🕒 Lettura sicura dei tempi                                     |
//+------------------------------------------------------------------+
bool SafeCopyTime(string symbol, ENUM_TIMEFRAMES tf, int shift, int count, datetime &out[])
{
    if (Bars(symbol, tf) < count + shift + 1)
        return false;

    int copied = CopyTime(symbol, tf, shift, count, out);
    return (copied == count);
}

#endif // __SAFEBUFFER_MQH__
