//+------------------------------------------------------------------+
//| SymbolFillingCache.mqh                                           |
//| 🔁 Gestione cache filling mode per simboli                       |
//| ✅ Supporta:                                                     |
//|   • Salvataggio automatico del filling corretto                 |
//|   • Recupero e fallback su IOC / RETURN / FOK                   |
//|   • Versione ottimizzata con chiavi stringa                     |
//| 🧠 Compatibile con sistemi multi-simbolo                         |
//+------------------------------------------------------------------+


#ifndef __SYMBOL_FILLING_CACHE_MQH__
#define __SYMBOL_FILLING_CACHE_MQH__

//+------------------------------------------------------------------+
//| 📦 Mappa semplice: symbol → filling mode                        |
//+------------------------------------------------------------------+

#define MAX_FILLING_CACHE 100

string fillingKeys[MAX_FILLING_CACHE];   // 🧩 Symbol name
int    fillingModes[MAX_FILLING_CACHE];  // 🔧 ENUM_ORDER_TYPE_FILLING as int
int    fillingUsed = 0;

//+------------------------------------------------------------------+
//| ✅ Controlla se un simbolo ha un filling mode noto              |
//+------------------------------------------------------------------+
bool HasCachedFillingMode(string symbol, ENUM_ORDER_TYPE_FILLING &cachedMode)
{
    for (int i = 0; i < fillingUsed; i++)
    {
        if (fillingKeys[i] == symbol)
        {
            cachedMode = (ENUM_ORDER_TYPE_FILLING)fillingModes[i];

            if (EnableLogging_OpenTrade)
                PrintFormat("📦 [FillingCache] Trovato: %s → %s (%d)",
                            symbol, EnumToString(cachedMode), cachedMode);
            return true;
        }
    }

    if (EnableLogging_OpenTrade)
        PrintFormat("📦 [FillingCache] Nessun filling noto per %s", symbol);

    return false;
}

//+------------------------------------------------------------------+
//| 💾 Salva filling mode per simbolo (se non già presente)         |
//+------------------------------------------------------------------+
void CacheFillingMode(string symbol, ENUM_ORDER_TYPE_FILLING mode)
{
    ENUM_ORDER_TYPE_FILLING tmp;
    if (!HasCachedFillingMode(symbol, tmp))
    {
        if (fillingUsed < MAX_FILLING_CACHE)
        {
            fillingKeys[fillingUsed]  = symbol;
            fillingModes[fillingUsed] = (int)mode;
            fillingUsed++;

            if (EnableLogging_OpenTrade)
                PrintFormat("💾 [FillingCache] Aggiunto: %s → %s (%d)",
                            symbol, EnumToString(mode), mode);
        }
        else if (EnableLogging_OpenTrade)
        {
            Print("⚠️ [FillingCache] Cache piena. Nessun salvataggio.");
        }
    }
    else if (EnableLogging_OpenTrade)
    {
        Print("🔁 [FillingCache] Già presente. Nessun aggiornamento.");
    }
}

//+------------------------------------------------------------------+
//| ❌ Rimuove filling mode per simbolo specifico                    |
//+------------------------------------------------------------------+
void ClearFillingMode(string symbol)
{
    for (int i = 0; i < fillingUsed; i++)
    {
        if (fillingKeys[i] == symbol)
        {
            for (int j = i; j < fillingUsed - 1; j++)
            {
                fillingKeys[j]  = fillingKeys[j + 1];
                fillingModes[j] = fillingModes[j + 1];
            }
            fillingUsed--;

            if (EnableLogging_OpenTrade)
                PrintFormat("🧹 [FillingCache] Rimosso %s", symbol);
            return;
        }
    }
}

//+------------------------------------------------------------------+
//| 🧺 Pulisce completamente la cache                                |
//+------------------------------------------------------------------+
void ClearAllFillingModes()
{
    fillingUsed = 0;
    if (EnableLogging_OpenTrade)
        Print("🧺 [FillingCache] Tutti i filling rimossi");
}

#endif // __SYMBOL_FILLING_CACHE_MQH__

