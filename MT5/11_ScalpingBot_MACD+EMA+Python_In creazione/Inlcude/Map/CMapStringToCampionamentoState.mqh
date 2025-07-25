//+------------------------------------------------------------------+
//|                                     CMapStringToCampionamentoState.mqh |
//|                                     Copyright 2024, MetaQuotes Software |
//|                                           https://www.metaquotes.net/ |
//+------------------------------------------------------------------+
#ifndef CMAPSTRINGTOCAMPIONAMENTOSTATE_MQH
#define CMAPSTRINGTOCAMPIONAMENTOSTATE_MQH


//+------------------------------------------------------------------+
//| CandelaStorica - Dati di una candela storica                      |
//+------------------------------------------------------------------+
class CandelaStorica
{
public:
    double score;               // Score assegnato a questa candela (0-10)
    double bodyRatio;           // Body/Range della candela (0-1)
    bool   direction;           // Direzione della candela (true=BUY, false=SELL)
    
    // Prezzi OHLC
    double open;                // Open della candela
    double high;                // High della candela
    double low;                 // Low della candela
    double close;               // Close della candela
    
    bool   isReversal;          // Flag se la candela è stata un reversal significativo

    // Costruttore
    CandelaStorica()
    {
        score = 0.0;
        bodyRatio = 0.0;
        direction = false;
        open = 0.0;
        high = 0.0;
        low = 0.0;
        close = 0.0;
        isReversal = false;
    }
};

//+------------------------------------------------------------------+
//| CampionamentoState - Stato del campionamento per symbol+timeframe |
//+------------------------------------------------------------------+
class CampionamentoState
{
public:
    double score;                       // Score attuale 0-10
    ENUM_ORDER_TYPE direction;          // Direzione BUY / SELL / -1
    string logText;
    
    string              symbol;         
    ENUM_TIMEFRAMES     timeframe;      
    datetime            currentCandleOpenTime;
    long                lastUpdateTime;     
    double              openPrice;          
    double              highPrice;          
    double              lowPrice;           
    double              lastPrice;          
    double              prevPrice;          
    double              currentADX;         
    double              prevADX;            
    double              intracandleScore;   
    int                 intracandleDirectionVotes;
    int                 consecutiveDirectionTicks;
    bool                currentDirection;   

    double              scoreTotalRaw;      
    double              finalNormalizedScore;
    bool                finalDirection;     
    bool                isClosingPending;   

    // Storia delle ultime candele (array di oggetti)
    CandelaStorica historicalCandles[];

    // Costruttore
    CampionamentoState()
    {
        score = 0.0;
        direction = (ENUM_ORDER_TYPE)-1;
        logText = "";
        symbol = "";
        timeframe = PERIOD_CURRENT;
        currentCandleOpenTime = 0;
        lastUpdateTime = 0;
        openPrice = 0.0;
        highPrice = 0.0;
        lowPrice = 0.0;
        lastPrice = 0.0;
        prevPrice = 0.0;
        currentADX = 0.0;
        prevADX = 0.0;
        intracandleScore = 0.0;
        intracandleDirectionVotes = 0;
        consecutiveDirectionTicks = 0;
        currentDirection = false;
        scoreTotalRaw = 0.0;
        finalNormalizedScore = 0.0;
        finalDirection = false;
        isClosingPending = false;

        ArrayResize(historicalCandles, CampionamentoThresholdLookback);
    }

    // Costruttore di copia (essenziale per passare oggetti CampionamentoState)
    CampionamentoState(const CampionamentoState &other)
    {
        score = other.score;
        direction = other.direction;
        logText = other.logText;
        symbol = other.symbol;
        timeframe = other.timeframe;
        currentCandleOpenTime = other.currentCandleOpenTime;
        lastUpdateTime = other.lastUpdateTime;
        openPrice = other.openPrice;
        highPrice = other.highPrice;
        lowPrice = other.lowPrice;
        lastPrice = other.lastPrice;
        prevPrice = other.prevPrice;
        currentADX = other.currentADX;
        prevADX = other.prevADX;
        intracandleScore = other.intracandleScore;
        intracandleDirectionVotes = other.intracandleDirectionVotes;
        consecutiveDirectionTicks = other.consecutiveDirectionTicks;
        currentDirection = other.currentDirection;
        scoreTotalRaw = other.scoreTotalRaw;
        finalNormalizedScore = other.finalNormalizedScore;
        finalDirection = other.finalDirection;
        isClosingPending = other.isClosingPending;
    
        // Copia dell'array dinamico historicalCandles
        ArrayResize(historicalCandles, ArraySize(other.historicalCandles));
        for (int i = 0; i < ArraySize(other.historicalCandles); i++)
        {
            historicalCandles[i] = other.historicalCandles[i];
        }
    }

    // Distruttore
    ~CampionamentoState()
    {
        ArrayResize(historicalCandles, 0); // Libera la memoria dell'array dinamico
    }
    
    // Metodo per accedere a una candela storica (passaggio per riferimento di output)
    bool GetHistoricalCandle(int index, CandelaStorica &outCandela) const
    {
        if (index >= 0 && index < ArraySize(historicalCandles))
        {
            outCandela = historicalCandles[index]; // Copia l'oggetto
            return true;
        }
        return false;
    }
    
    // Metodo per impostare una candela storica
    void SetHistoricalCandle(int index, const CandelaStorica &candela)
    {
        if (index >= 0 && index < ArraySize(historicalCandles))
        {
            historicalCandles[index] = candela;
        }
    }
};

//+------------------------------------------------------------------+
//|                         CMapStringToCampionamentoState           |
//|                    Con gestione automatica della memoria         |
//+------------------------------------------------------------------+
class CMapStringToCampionamentoState
{
private:
    string keys[];
    CampionamentoState values[]; // Array di oggetti CampionamentoState (non puntatori)
    long lastAccessTime[];       // Timestamp dell'ultimo accesso per ogni entry
    int maxEntries;              // Numero massimo di entries consentite
    bool enableAutoEviction;     // Abilita rimozione automatica delle entries più vecchie
    int evictionCount;          // Contatore per statistiche

public:
    CMapStringToCampionamentoState() 
    {
        maxEntries = MAP_MAX_ENTRIES;
        enableAutoEviction = true;
        evictionCount = 0;
    }
    
    ~CMapStringToCampionamentoState()
    {
        ArrayResize(keys, 0);
        ArrayResize(values, 0);
        ArrayResize(lastAccessTime, 0);
    }

    // Imposta il numero massimo di entries
    void SetMaxEntries(int max)
    {
        if (max > 0)
            maxEntries = max;
    }
    
    // Abilita/disabilita l'eviction automatica
    void SetAutoEviction(bool enable)
    {
        enableAutoEviction = enable;
    }
    
    // Ottiene statistiche sull'eviction
    int GetEvictionCount() const
    {
        return evictionCount;
    }

    // Inserisce o aggiorna un elemento nella mappa.
    // Accetta l'oggetto per valore/const reference (verrà copiato).
    bool Insert(string key, const CampionamentoState &value)
    {
        // Prima controlla se dobbiamo fare spazio
        if (enableAutoEviction && maxEntries > 0)
        {
            int currentSize = ArraySize(keys);
            
            // Se siamo al limite, rimuovi le entries più vecchie
            if (currentSize >= maxEntries)
            {
                EvictOldestEntries();
            }
        }
        
        int index = FindIndex(key);
        if (index == -1)
        {
            // Nuova entry
            int old_size = ArraySize(keys);
            ArrayResize(keys, old_size + 1);
            ArrayResize(values, old_size + 1);
            ArrayResize(lastAccessTime, old_size + 1);
            
            keys[old_size] = key;
            values[old_size] = value; // Copia l'oggetto
            lastAccessTime[old_size] = GetTickCount(); // Timestamp corrente
            return true;
        }
        else
        {
            // Aggiorna entry esistente
            values[index] = value; // Aggiorna l'oggetto esistente (copia)
            lastAccessTime[index] = GetTickCount(); // Aggiorna timestamp
            return true;
        }
    }

    // Verifica se una chiave esiste
    bool IsExist(string key) const
    {
        return (FindIndex(key) != -1);
    }

    // Recupera un elemento dalla mappa, copiandolo in 'outValue'.
    // Questo è il modo corretto per ottenere l'oggetto per lavorarci.
    bool Get(string key, CampionamentoState &outValue)
    {
        int index = FindIndex(key);
        if (index != -1)
        {
            outValue = values[index]; // Copia l'oggetto dalla mappa a outValue
            lastAccessTime[index] = GetTickCount(); // Aggiorna timestamp di accesso
            return true;
        }
        return false;
    }

    // Rimuove un elemento dalla mappa
    bool Remove(string key)
    {
        int index = FindIndex(key);
        if (index != -1)
        {
            RemoveElementFromArrayString(keys, index);
            RemoveElementFromArrayObj(values, index);
            RemoveElementFromArrayLong(lastAccessTime, index);
            return true;
        }
        return false;
    }

    // Pulisce la mappa
    void Clear()
    {
        ArrayResize(keys, 0);
        ArrayResize(values, 0);
        ArrayResize(lastAccessTime, 0);
        evictionCount = 0;
    }

    // Restituisce la dimensione della mappa
    int Size() const
    {
        return ArraySize(keys);
    }

    // Ottiene tutte le chiavi (copia in un array esterno)
    void GetKeys(string &outKeys[]) const
    {
        ArrayResize(outKeys, ArraySize(keys));
        for (int i = 0; i < ArraySize(keys); i++)
        {
            outKeys[i] = keys[i];
        }
    }

    // Ottiene chiave e valore per indice (copia il valore in outValue)
    bool GetAt(int index, string &outKey, CampionamentoState &outValue)
    {
        if (index >= 0 && index < ArraySize(keys))
        {
            outKey = keys[index];
            outValue = values[index]; // Copia l'oggetto
            lastAccessTime[index] = GetTickCount(); // Aggiorna timestamp
            return true;
        }
        return false;
    }

    // Ottiene la chiave per indice
    string GetKeyAt(int index) const
    {
        if (index >= 0 && index < ArraySize(keys))
        {
            return keys[index];
        }
        return "";
    }
    
    // Rimuove manualmente le entries più vecchie (può essere chiamata dall'esterno)
    void CleanupOldEntries(int hoursOld = 24)
    {
        long currentTime = GetTickCount();
        long maxAge = hoursOld * 3600 * 1000; // Converti ore in millisecondi
        
        for (int i = ArraySize(keys) - 1; i >= 0; i--)
        {
            if (currentTime - lastAccessTime[i] > maxAge)
            {
                RemoveElementFromArrayString(keys, i);
                RemoveElementFromArrayObj(values, i);
                RemoveElementFromArrayLong(lastAccessTime, i);
                evictionCount++;
            }
        }
    }

private:
    // Trova l'indice di una chiave
    int FindIndex(string key) const
    {
        for (int i = 0; i < ArraySize(keys); i++)
        {
            if (keys[i] == key)
                return i;
        }
        return -1;
    }
    
    // Rimuove le entries più vecchie quando si raggiunge il limite
    void EvictOldestEntries()
    {
        int currentSize = ArraySize(keys);
        if (currentSize == 0) return;
        
        // Calcola quante entries rimuovere (20% del totale)
        int entriesToRemove = MathMax(1, (currentSize * MAP_EVICTION_PERCENT) / 100);
        
        // Trova le entries più vecchie
        for (int removed = 0; removed < entriesToRemove && ArraySize(keys) > 0; removed++)
        {
            int oldestIndex = 0;
            long oldestTime = lastAccessTime[0];
            
            // Trova l'entry più vecchia
            for (int i = 1; i < ArraySize(keys); i++)
            {
                if (lastAccessTime[i] < oldestTime)
                {
                    oldestTime = lastAccessTime[i];
                    oldestIndex = i;
                }
            }
            
            // Rimuovi l'entry più vecchia
            if (EnableLogging_Campionamento)
            {
                PrintFormat("🗑️ [MapEviction] Rimozione entry obsoleta: %s (età: %.1f secondi)", 
                            keys[oldestIndex], 
                            (double)(GetTickCount() - lastAccessTime[oldestIndex]) / 1000.0);
            }
            
            RemoveElementFromArrayString(keys, oldestIndex);
            RemoveElementFromArrayObj(values, oldestIndex);
            RemoveElementFromArrayLong(lastAccessTime, oldestIndex);
            evictionCount++;
        }
        
        if (EnableLogging_Campionamento)
        {
            PrintFormat("🧹 [MapEviction] Rimosse %d entries. Totale evictions: %d, Size attuale: %d/%d", 
                        entriesToRemove, evictionCount, ArraySize(keys), maxEntries);
        }
    }

    // Funzione helper per rimuovere un elemento da un array di stringhe
    void RemoveElementFromArrayString(string &arr[], int index_to_remove)
    {
        int size = ArraySize(arr);
        if (index_to_remove < 0 || index_to_remove >= size)
            return;

        for (int i = index_to_remove; i < size - 1; i++)
        {
            arr[i] = arr[i + 1];
        }

        ArrayResize(arr, size - 1);
    }

    // Funzione helper per rimuovere un elemento da un array di oggetti CampionamentoState
    void RemoveElementFromArrayObj(CampionamentoState &arr[], int index_to_remove)
    {
        int size = ArraySize(arr);
        if (index_to_remove < 0 || index_to_remove >= size)
            return;

        for (int i = index_to_remove; i < size - 1; i++)
        {
            arr[i] = arr[i + 1];
        }

        ArrayResize(arr, size - 1);
    }
    
    // Funzione helper per rimuovere un elemento da un array di long
    void RemoveElementFromArrayLong(long &arr[], int index_to_remove)
    {
        int size = ArraySize(arr);
        if (index_to_remove < 0 || index_to_remove >= size)
            return;

        for (int i = index_to_remove; i < size - 1; i++)
        {
            arr[i] = arr[i + 1];
        }

        ArrayResize(arr, size - 1);
    }
};

#endif // CMAPSTRINGTOCAMPIONAMENTOSTATE_MQH