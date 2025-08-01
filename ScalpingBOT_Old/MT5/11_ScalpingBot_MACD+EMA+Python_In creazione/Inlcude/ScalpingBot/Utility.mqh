//+------------------------------------------------------------------+
//|              Utility.mqh                                         |
//|              Funzioni comuni e condivise tra moduli              |
//+------------------------------------------------------------------+
#ifndef __UTILITY_MQH__
#define __UTILITY_MQH__


//+------------------------------------------------------------------+
//| 📝 BOT 4.0 - Define universali di sistema                       |
//+------------------------------------------------------------------+

// 📐 Precisione di calcolo
#define POINT_HALF    (SymbolInfoDouble(_Symbol, SYMBOL_POINT) / 2.0)

// 🪙 Valori di fallback universali
#define INVALID_PRICE_VALUE    -1.0
#define INVALID_VOLUME_VALUE   -1.0
#define INVALID_TICKET_VALUE   (ulong)(-1)

// 🕰️ Timeframe abbreviati per leggibilità nei moduli multi-timeframe
#define TF_M1      PERIOD_M1
#define TF_M5      PERIOD_M5
#define TF_M15     PERIOD_M15
#define TF_M30     PERIOD_M30
#define TF_H1      PERIOD_H1
#define TF_H4      PERIOD_H4
#define TF_D1      PERIOD_D1

// 🎯 Stringhe standard per logging, csv, observer ecc.
#define BUY_STRING    "BUY"
#define SELL_STRING   "SELL"

// ✅ Codici esito universali del BOT → usabili in qualsiasi modulo
#define BOT_RESULT_SUCCESS    0
#define BOT_RESULT_FAILURE    1
#define BOT_RESULT_RETRY      2

//+------------------------------------------------------------------+
//| 🚨 BOT 4.0 - Define codici errore universali                    |
//+------------------------------------------------------------------+

// 📋 Errori MT5 standard
#define ERR_NO_ERROR                       0
#define ERR_COMMON_ERROR                   2
#define ERR_INVALID_TRADE_PARAMETERS       3
#define ERR_SERVER_BUSY                    4
#define ERR_OLD_CLIENT_VERSION             5
#define ERR_NO_CONNECTION                  6
#define ERR_INVALID_ACCOUNT                8
#define ERR_TRADE_TIMEOUT                  9
#define ERR_ACCOUNT_DISABLED               64
#define ERR_INVALID_ACCOUNT_2              65
#define ERR_TRADE_CONTEXT_BUSY             128
#define ERR_INVALID_PRICE                  129
#define ERR_INVALID_STOPS                  130
#define ERR_INVALID_VOLUME                 131
#define ERR_MARKET_CLOSED                  132
#define ERR_NOT_ENOUGH_MONEY               134
#define ERR_PRICE_CHANGED                  135
#define ERR_OFF_QUOTES                     136
#define ERR_BROKER_BUSY                    137
#define ERR_REQUOTE                        138
#define ERR_ORDER_LOCKED                   139
#define ERR_LONG_POSITIONS_ONLY            140
#define ERR_TOO_MANY_REQUESTS              141
#define ERR_SL_MODIFICATION_REJECTED       4756   // 👈 broker custom (indici etc.)

// 📋 Errori broker estesi (tuoi + broker reali)
#define ERR_NO_CONNECTION_NEW              4060
#define ERR_PRICE_CHANGED_NEW              4106
#define ERR_INVALID_PRICE_NEW              4107
#define ERR_MARKET_CLOSED_NEW              4108
#define ERR_OFF_QUOTES_NEW                 4109
#define ERR_ORDER_SEND_TIMEOUT             10004
#define ERR_NO_PRICES                      10006
#define ERR_BROKER_NOT_CONNECTED           10007
#define ERR_TOO_FREQUENT_REQUESTS          10008

//+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

//-------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| 🧩 BLOCCO 0: 🧠 Mappe Globali                                    |
//+------------------------------------------------------------------+
//-------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| 📦 CEMACache - Versione flessibile per periodi multipli         |
//+------------------------------------------------------------------+
#include <Arrays/ArrayString.mqh>
#include <Arrays/ArrayInt.mqh>

class CEMACache
{
private:
    string keys[];
    int handles[];
    
    // Struttura per memorizzare buffer multipli
    struct EMABufferData {
        int period;
        double buffer[];
        bool isValid;
    };
    
    // Array di buffer per diversi periodi
    EMABufferData m_buffers[];
    
    // Variabili per memorizzare l'ultima richiesta
    string m_last_symbol;
    ENUM_TIMEFRAMES m_last_tf;

    string BuildKey(string symbol, ENUM_TIMEFRAMES tf, int period)
    {
        return symbol + "#" + IntegerToString(tf) + "#" + IntegerToString(period);
    }

    bool IsHandleValid(int handle)
    {
        if (handle == INVALID_HANDLE)
            return false;
        double buffer_check[];
        ArraySetAsSeries(buffer_check, true);
        return (CopyBuffer(handle, 0, 0, 1, buffer_check) >= 0);
    }

    bool WaitForIndicatorReady(int handle, int maxAttempts = 10)
    {
        double buffer_wait[];
        ArraySetAsSeries(buffer_wait, true);
        
        for(int i = 0; i < maxAttempts; i++)
        {
            if(CopyBuffer(handle, 0, 0, 1, buffer_wait) >= 0)
                return true;
                
            if(!MQLInfoInteger(MQL_TESTER))  // Solo se NON in backtest
                Sleep(100);
        }
        return false;
    }

    void DebugLog(string message)
    {
        if (EnableLogging_MicroTrend)
            Print(message);
    }

    // Trova o crea un buffer per un periodo specifico
    int GetBufferIndex(int period)
    {
        for(int i = 0; i < ArraySize(m_buffers); i++)
        {
            if(m_buffers[i].period == period)
                return i;
        }
        
        // Non trovato, crea nuovo
        int newIndex = ArraySize(m_buffers);
        ArrayResize(m_buffers, newIndex + 1);
        m_buffers[newIndex].period = period;
        m_buffers[newIndex].isValid = false;
        ArraySetAsSeries(m_buffers[newIndex].buffer, true);
        
        return newIndex;
    }

public:
    // Costruttore
    CEMACache()
    {
        DebugLog("🚀 [EMA Cache] CEMACache istanziata per periodi multipli.");
    }

    // Distruttore
    ~CEMACache()
    {
        ReleaseAll();
    }

    // NUOVO: Metodo generico per ottenere handle EMA di qualsiasi periodo
    int GetEMAHandle(string symbol, ENUM_TIMEFRAMES tf, int period)
    {
        string key = BuildKey(symbol, tf, period);

        for (int i = 0; i < ArraySize(keys); i++)
        {
            if (keys[i] == key)
            {
                if (IsHandleValid(handles[i]))
                    return handles[i];

                DebugLog("⚠️ [EMA Cache] Handle invalido per " + key + " → reinizializzo");
                handles[i] = iMA(symbol, tf, period, 0, MODE_EMA, PRICE_CLOSE);
                if (!WaitForIndicatorReady(handles[i]))
                    DebugLog("❌ [EMA Cache] Timeout attesa dati per " + key);
                return handles[i];
            }
        }

        // Handle non esistente → creazione
        int handle = iMA(symbol, tf, period, 0, MODE_EMA, PRICE_CLOSE);
        if (handle != INVALID_HANDLE)
        {
            ArrayResize(keys, ArraySize(keys) + 1);
            ArrayResize(handles, ArraySize(handles) + 1);
            keys[ArraySize(keys) - 1] = key;
            handles[ArraySize(handles) - 1] = handle;

            DebugLog("📌 [EMA Cache] Creato handle EMA" + IntegerToString(period) + 
                     " per " + symbol + " " + EnumToString(tf) + " → handle=" + IntegerToString(handle));
            
            if (!WaitForIndicatorReady(handle))
                DebugLog("❌ [EMA Cache] Timeout attesa dati per " + key);
        }
        else
        {
            DebugLog("❌ [EMA Cache] Errore creazione handle EMA per " + key);
        }

        return handle;
    }

    // NUOVO: Metodo per copiare dati di un singolo periodo EMA
    bool CopyEMAData(string symbol, ENUM_TIMEFRAMES tf, int period, 
                     int candlesToRead, double &buffer[])
    {
        int handle = GetEMAHandle(symbol, tf, period);
        if (handle == INVALID_HANDLE)
            return false;
            
        ArrayResize(buffer, candlesToRead);
        ArraySetAsSeries(buffer, true);
        
        int copied = CopyBuffer(handle, 0, 0, candlesToRead, buffer);
        
        if (copied <= 0)
        {
            DebugLog(StringFormat("❌ [EMA Cache] CopyBuffer fallito per EMA%d. Copiati: %d, Err=%d",
                                  period, copied, GetLastError()));
            return false;
        }
        
        return true;
    }

    // NUOVO: Metodo per ottenere e memorizzare dati di periodi multipli
    bool GetMultipleEMAData(string symbol, ENUM_TIMEFRAMES tf, 
                           int &periods[], int candlesToRead)
    {
        m_last_symbol = symbol;
        m_last_tf = tf;
        
        bool allSuccess = true;
        
        for(int i = 0; i < ArraySize(periods); i++)
        {
            int bufferIdx = GetBufferIndex(periods[i]);
            
            if(!CopyEMAData(symbol, tf, periods[i], candlesToRead, m_buffers[bufferIdx].buffer))
            {
                m_buffers[bufferIdx].isValid = false;
                allSuccess = false;
                DebugLog(StringFormat("❌ Fallito caricamento EMA%d", periods[i]));
            }
            else
            {
                m_buffers[bufferIdx].isValid = true;
                DebugLog(StringFormat("✅ Caricato EMA%d con %d candele", 
                                      periods[i], ArraySize(m_buffers[bufferIdx].buffer)));
            }
        }
        
        return allSuccess;
    }

    // NUOVO: Metodo per accedere ai dati EMA per periodo
    double GetEMAValue(int period, int shift)
    {
        for(int i = 0; i < ArraySize(m_buffers); i++)
        {
            if(m_buffers[i].period == period && m_buffers[i].isValid)
            {
                if(shift >= 0 && shift < ArraySize(m_buffers[i].buffer))
                    return m_buffers[i].buffer[shift];
                else
                {
                    DebugLog(StringFormat("❌ Shift %d fuori range per EMA%d", shift, period));
                    return 0.0;
                }
            }
        }
        
        DebugLog(StringFormat("❌ Buffer EMA%d non trovato o non valido", period));
        return 0.0;
    }

    // Metodo compatibile per retrocompatibilità (se ancora necessario)
    bool GetAndCopyEMADatas(string symbol, ENUM_TIMEFRAMES tf,
                           int ema5_period, int ema20_period, int candlesToRead)
    {
        int periods[];
        ArrayResize(periods, 2);
        periods[0] = ema5_period;
        periods[1] = ema20_period;
        
        return GetMultipleEMAData(symbol, tf, periods, candlesToRead);
    }

    // Metodi di accesso specifici per retrocompatibilità
    double GetEMA5(int shift) { return GetEMAValue(5, shift); }
    double GetEMA20(int shift) { return GetEMAValue(20, shift); }

    // Puoi aggiungere metodi specifici per i tuoi periodi comuni
    double GetEMA3(int shift) { return GetEMAValue(3, shift); }
    double GetEMA4(int shift) { return GetEMAValue(4, shift); }
    double GetEMA50(int shift) { return GetEMAValue(50, shift); }

    // Metodo per verificare se un buffer è disponibile
    bool IsEMAAvailable(int period)
    {
        for(int i = 0; i < ArraySize(m_buffers); i++)
        {
            if(m_buffers[i].period == period)
                return m_buffers[i].isValid;
        }
        return false;
    }

    // Info sui parametri dell'ultima richiesta
    string GetLastSymbol() { return m_last_symbol; }
    ENUM_TIMEFRAMES GetLastTimeframe() { return m_last_tf; }

    void ReleaseAll()
    {
        for (int i = 0; i < ArraySize(handles); i++)
        {
            if (handles[i] != INVALID_HANDLE)
                IndicatorRelease(handles[i]);
        }
        ArrayFree(handles);
        ArrayFree(keys);
        
        // Libera tutti i buffer
        for(int i = 0; i < ArraySize(m_buffers); i++)
        {
            ArrayFree(m_buffers[i].buffer);
        }
        ArrayFree(m_buffers);

        DebugLog("🧹 [EMA Cache] Tutti gli handle EMA e i buffer sono stati rilasciati.");
    }
};

// ✅ Istanza globale
CEMACache emaCache;

//+------------------------------------------------------------------+
//| 📦 CMACDCache - Gestione handle MACD multi-asset/multiframe     |
//| ADATTATO PER IGNORARE PROBLEMI DI PRONTEZZA DELL'ISTOGRAMMA     |
//| (Versione più robusta per gestione validità handle)             |
//+------------------------------------------------------------------+
class CMACDCache
{
private:
    string keys[];    // Array per memorizzare le chiavi uniche (simbolo#timeframe#...)
    int handles[];    // Array per memorizzare gli handle degli indicatori MACD

    // Costruisce una chiave unica basata sui parametri del MACD
    string BuildKey(string symbol, ENUM_TIMEFRAMES tf, int fast, int slow, int signal)
    {
        return symbol + "#" + IntegerToString(tf) + "#" + IntegerToString(fast) + "#" + IntegerToString(slow) + "#" + IntegerToString(signal);
    }

    // Controlla se un handle dell'indicatore è valido e non ha errori.
    // Tenta di copiare 0 barre dal buffer 0. Se restituisce >= 0, l'handle è considerato valido.
    bool IsHandleValid(int handle)
    {
        if (handle == INVALID_HANDLE)
            return false;
            
        double temp_buffer[];
        // Un CopyBuffer(handle, 0, 0, 0, ...) può a volte non rilevare handle "morti" immediatamente.
        // Un piccolo CopyBuffer(handle, 0, 0, 1, ...) è più affidabile per verificare se l'indicatore sta lavorando.
        return (CopyBuffer(handle, 0, 0, 1, temp_buffer) >= 0); 
    }

    // Attende che l'indicatore sia "pronto" con dati sufficienti.
    // Utilizza un numero massimo di tentativi invece di un timeout basato sul tempo reale,
    // più adatto per il backtest.
    bool WaitForIndicatorReady(int handle, int bars_needed, int max_attempts = 20, int sleep_ms_between_attempts = 10)
    {
        DebugLog("⏱️ [MACD Cache] Inizio attesa per handle " + IntegerToString(handle) + " (dati sufficienti). Barre necessarie: " + IntegerToString(bars_needed) + ".");

        double temp_macd_buffer[];
        double temp_signal_buffer[];
        double temp_hist_buffer[]; // Manteniamo questo per il tentativo di copia

        for (int attempt = 0; attempt < max_attempts; attempt++)
        {
            // Controlla prima la validità generale dell'handle
            if (!IsHandleValid(handle))
            {
                DebugLog("⚠️ [MACD Cache] Handle " + IntegerToString(handle) + " diventato invalido durante l'attesa (tentativo " + IntegerToString(attempt+1) + "). Interruzione.");
                return false;
            }

            // --- NUOVA LOGICA DI VERIFICA DELLA PRONTEZZA ---
            // Verifichiamo se MACD e Signal sono pronti.
            // Il CopyBuffer per Hist lo eseguiamo ma non blocchiamo l'attesa se fallisce.
            int copied_macd = CopyBuffer(handle, 0, 0, bars_needed, temp_macd_buffer);
            int copied_signal = CopyBuffer(handle, 1, 0, bars_needed, temp_signal_buffer);
            int copied_hist = CopyBuffer(handle, 2, 0, bars_needed, temp_hist_buffer); // Tentativo di copia per debug

            // Logga i risultati del CopyBuffer all'interno dell'attesa
            if (EnableLogging_MACD) {
                PrintFormat("DEBUG CMACDCache (Attesa): CopyBuffer MACD: %d (Err: %d), Signal: %d (Err: %d), Hist: %d (Err: %d) (Tentativo %d)",
                            copied_macd, (copied_macd == -1 ? GetLastError() : 0),
                            copied_signal, (copied_signal == -1 ? GetLastError() : 0),
                            copied_hist, (copied_hist == -1 ? GetLastError() : 0),
                            attempt + 1);
            }

            // La cache è pronta se almeno le linee MACD e Signal sono state copiate completamente.
            // L'istogramma è secondario dato che lo calcoliamo.
            if (copied_macd >= bars_needed && copied_signal >= bars_needed)
            {
                DebugLog("✅ [MACD Cache] Handle " + IntegerToString(handle) + " (MACD/Signal pronti) dopo " + IntegerToString(attempt + 1) + " tentativi.");
                return true;  // MACD e Signal sono pronti
            }
            
            // Attendi brevemente prima di riprovare.
            Sleep(sleep_ms_between_attempts); 
        }
        
        DebugLog("❌ [MACD Cache] Timeout scaduto per handle " + IntegerToString(handle) + " (MACD/Signal non pronti).");
        return false;
    }

    // Funzione di logging di debug. Richiede 'EnableLogging_MACD' come variabile globale o input.
    void DebugLog(string message)
    {
        if (EnableLogging_MACD) 
            Print(message);
    }

public:
    // Distruttore per rilasciare tutti gli handle alla fine della vita dell'EA
    ~CMACDCache() {
        ReleaseAll();
    }

    // Restituisce l'handle dell'indicatore MACD per i parametri specificati.
    // Lo crea e lo cache se non esiste, o lo reinizializza se invalido o non pronto.
    int GetMACDHandle(string symbol, ENUM_TIMEFRAMES tf, int fast, int slow, int signal)
    {
        string key = BuildKey(symbol, tf, fast, slow, signal);
        
        // --- ⚙️ CALCOLO DINAMICO DELLE BARRE MINIME NECESSARIE ---
        int bars_needed_for_full_analysis = MathMax(fast, MathMax(slow, signal)) + 5; 
        bars_needed_for_full_analysis = MathMax(bars_needed_for_full_analysis, MACD_DivergenceLookbackBars + 2); 
        bars_needed_for_full_analysis = MathMax(bars_needed_for_full_analysis, 30); // Assicurati sia almeno un valore ragionevole
        // -----------------------------------------------------------

        // Cerca l'handle nella cache esistente
        for (int i = 0; i < ArraySize(keys); i++)
        {
            if (keys[i] == key)
            {
                // Se l'handle è già valido e "pronto" secondo i criteri della cache (con barre sufficienti), restituiscilo
                if (IsHandleValid(handles[i]) && WaitForIndicatorReady(handles[i], bars_needed_for_full_analysis))
                {
                    DebugLog("✅ [MACD Cache] Handle trovato e pronto per " + key + ". Handle=" + IntegerToString(handles[i]));
                    return handles[i];
                }

                // Se l'handle è invalido o non pronto, reinizializzalo
                DebugLog("⚠️ [MACD Cache] Handle invalido o dati insufficienti per " + key + " → reinizializzo.");
                handles[i] = iMACD(symbol, tf, fast, slow, signal, PRICE_CLOSE);

                if (handles[i] != INVALID_HANDLE)
                {
                    if (!WaitForIndicatorReady(handles[i], bars_needed_for_full_analysis))
                    {
                        DebugLog("❌ [MACD Cache] Timeout attesa dati per " + key + " dopo reinizializzazione.");
                    }
                }
                else
                {
                    DebugLog("❌ [MACD Cache] Errore creazione handle MACD dopo reinizializzazione per " + key + ". GetLastError: " + IntegerToString(GetLastError()));
                }
                return handles[i];
            }
        }

        // ➕ Se l'handle non esiste nella cache, crealo e aggiungilo
        int handle = iMACD(symbol, tf, fast, slow, signal, PRICE_CLOSE);
        if (handle != INVALID_HANDLE)
        {
            int new_size = ArraySize(keys) + 1;
            ArrayResize(keys, new_size);
            ArrayResize(handles, new_size);
            keys[new_size - 1] = key;
            handles[new_size - 1] = handle;

            DebugLog("📌 [MACD Cache] Creato nuovo handle MACD per " + key + ". Handle=" + IntegerToString(handle));

            if (!WaitForIndicatorReady(handle, bars_needed_for_full_analysis))
            {
                DebugLog("❌ [MACD Cache] Timeout attesa dati per " + key + " dopo la creazione.");
            }
        }
        else
        {
            DebugLog("❌ [MACD Cache] Errore creazione nuovo handle MACD per " + key + ". GetLastError: " + IntegerToString(GetLastError()));
        }

        return handle;
    }

    // Rilascia tutti gli handle degli indicatori quando l'EA viene disattivato (OnDeinit)
    void ReleaseAll()
    {
        for (int i = 0; i < ArraySize(handles); i++)
        {
            if (handles[i] != INVALID_HANDLE)
            {
                IndicatorRelease(handles[i]);
                handles[i] = INVALID_HANDLE;    
            }
        }
        ArrayFree(handles); 
        ArrayFree(keys);    

        DebugLog("🧹 [MACD Cache] Tutti gli handle MACD sono stati rilasciati.");
    }
};

// ✅ Istanza globale della cache MACD, accessibile da altre parti del codice
CMACDCache macdCache;

//+------------------------------------------------------------------+
//| CRSICache_Optimized.mqh - VERSIONE ENTERPRISE MT5               |
//| Cache RSI handle ottimizzata per 100+ assets simultanei         |
//| Thread-safe, O(1) lookup, memory-efficient, auto-recovery       |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| 📊 CONFIGURAZIONE E COSTANTI GLOBALI                           |
//+------------------------------------------------------------------+

// Cache RSI configuration
#define CACHE_BUCKET_SIZE       128    // Hash table size
#define CACHE_MAX_ENTRIES       150    // Max 100 assets + buffer
#define MAX_ERROR_COUNT         3      // Max errors before abandon
#define HANDLE_TIMEOUT_MS       2000   // Handle initialization timeout
#define CACHE_CLEANUP_INTERVAL  300    // Cleanup every 5 minutes
#define MAX_HANDLE_AGE          3600   // Handle TTL (1 hour)

//+------------------------------------------------------------------+
//| 📋 ENUMERAZIONI                                                 |
//+------------------------------------------------------------------+
enum RSI_HANDLE_STATUS {
    RSI_READY,           // Handle ready and working
    RSI_PENDING,         // Handle created, waiting for data
    RSI_ERROR,           // Handle in error state
    RSI_TIMEOUT,         // Timeout during initialization
    RSI_ABANDONED        // Handle abandoned after too many errors
};

//+------------------------------------------------------------------+
//| 📦 STRUTTURE DATI                                               |
//+------------------------------------------------------------------+
struct RSICacheEntry {
    string key;
    int handle;
    RSI_HANDLE_STATUS status;
    datetime createdTime;
    datetime lastAccessTime;
    datetime lastValidationTime;
    int errorCount;
    int accessCount;
    bool needsValidation;
    
    void Reset() {
        key = "";
        handle = INVALID_HANDLE;
        status = RSI_ERROR;
        createdTime = TimeCurrent();
        lastAccessTime = TimeCurrent();
        lastValidationTime = 0;
        errorCount = 0;
        accessCount = 0;
        needsValidation = true;
    }
};

struct CacheStatistics {
    int totalRequests;
    int cacheHits;
    int cacheMisses;
    int handleCreations;
    int handleErrors;
    int validationChecks;
    int cleanupOperations;
    datetime lastCleanupTime;
    double avgLookupTimeMs;
    int activeHandles;
    
    void Reset() {
        totalRequests = 0;
        cacheHits = 0;
        cacheMisses = 0;
        handleCreations = 0;
        handleErrors = 0;
        validationChecks = 0;
        cleanupOperations = 0;
        lastCleanupTime = TimeCurrent();
        avgLookupTimeMs = 0.0;
        activeHandles = 0;
    }
    
    double GetHitRate() {
        return totalRequests > 0 ? (double)cacheHits / totalRequests * 100.0 : 0.0;
    }
};

//+------------------------------------------------------------------+
//| 🚀 CLASSE CACHE RSI OTTIMIZZATA (SEMPLIFICATA PER MQL5)        |
//+------------------------------------------------------------------+
class CRSICacheOptimized {
private:
    // Hash table with bucket chains
    RSICacheEntry buckets[CACHE_BUCKET_SIZE][10];
    int bucketCounts[CACHE_BUCKET_SIZE];
    
    // Statistics and monitoring
    CacheStatistics stats;
    
    // Simple thread safety
    bool isLocked;
    datetime lastMaintenanceTime;
    
    //+------------------------------------------------------------------+
    //| 🔢 Hash function (FNV-1a)                                       |
    //+------------------------------------------------------------------+
    uint GetHashCode(string key) {
        uint hash = 2166136261;
        for (int i = 0; i < StringLen(key); i++) {
            hash ^= (uint)StringGetCharacter(key, i);
            hash *= 16777619;
        }
        return hash % CACHE_BUCKET_SIZE;
    }
    
    //+------------------------------------------------------------------+
    //| 🔑 Key generation                                               |
    //+------------------------------------------------------------------+
    string BuildKey(string symbol, ENUM_TIMEFRAMES tf, int period) {
        return symbol + "#" + IntegerToString((int)tf) + "#" + IntegerToString(period);
    }
    
    //+------------------------------------------------------------------+
    //| 🔒 Simple lock mechanism                                        |
    //+------------------------------------------------------------------+
    bool AcquireLock(int timeoutMs = 1000) {
        datetime start = TimeCurrent();
        while (isLocked && (TimeCurrent() - start) < timeoutMs) {
            Sleep(10);
        }
        if (isLocked) return false;
        
        isLocked = true;
        return true;
    }
    
    void ReleaseLock() {
        isLocked = false;
    }
    
    //+------------------------------------------------------------------+
    //| 🔍 Find entry in bucket                                         |
    //+------------------------------------------------------------------+
    int FindEntryIndex(string key, uint bucketIndex) {
        int count = bucketCounts[bucketIndex];
        
        for (int i = 0; i < count; i++) {
            if (buckets[bucketIndex][i].key == key) {
                return i;
            }
        }
        
        return -1;
    }
    
    //+------------------------------------------------------------------+
    //| ➕ Add entry to bucket                                          |
    //+------------------------------------------------------------------+
    int AddEntry(string key, uint bucketIndex) {
        int count = bucketCounts[bucketIndex];
        
        // Check space in bucket
        if (count >= 10) {
            // Bucket overflow: remove oldest entry (LRU)
            EvictLRUFromBucket(bucketIndex);
            count = bucketCounts[bucketIndex];
        }
        
        // Add new entry
        buckets[bucketIndex][count].Reset();
        buckets[bucketIndex][count].key = key;
        buckets[bucketIndex][count].createdTime = TimeCurrent();
        bucketCounts[bucketIndex]++;
        
        return count;
    }
    
    //+------------------------------------------------------------------+
    //| 🗑️ LRU eviction from bucket                                     |
    //+------------------------------------------------------------------+
    void EvictLRUFromBucket(uint bucketIndex) {
        int count = bucketCounts[bucketIndex];
        if (count == 0) return;
        
        // Find entry with oldest lastAccessTime
        int oldestIndex = 0;
        datetime oldestTime = buckets[bucketIndex][0].lastAccessTime;
        
        for (int i = 1; i < count; i++) {
            if (buckets[bucketIndex][i].lastAccessTime < oldestTime) {
                oldestTime = buckets[bucketIndex][i].lastAccessTime;
                oldestIndex = i;
            }
        }
        
        // Release handle if valid
        if (buckets[bucketIndex][oldestIndex].handle != INVALID_HANDLE) {
            IndicatorRelease(buckets[bucketIndex][oldestIndex].handle);
            stats.activeHandles--;
        }
        
        // Compact array removing entry
        for (int i = oldestIndex; i < count - 1; i++) {
            buckets[bucketIndex][i] = buckets[bucketIndex][i + 1];
        }
        
        bucketCounts[bucketIndex]--;
    }
    
    //+------------------------------------------------------------------+
    //| ✅ Non-blocking handle validation                               |
    //+------------------------------------------------------------------+
    bool ValidateHandleAsync(uint bucketIndex, int entryIndex) {
        if (buckets[bucketIndex][entryIndex].handle == INVALID_HANDLE) {
            buckets[bucketIndex][entryIndex].status = RSI_ERROR;
            return false;
        }
        
        // Check if validation is recent
        if (TimeCurrent() - buckets[bucketIndex][entryIndex].lastValidationTime < 30) {
            return buckets[bucketIndex][entryIndex].status == RSI_READY;
        }
        
        // Quick non-blocking test
        double buffer[1];
        int result = CopyBuffer(buckets[bucketIndex][entryIndex].handle, 0, 0, 1, buffer);
        
        buckets[bucketIndex][entryIndex].lastValidationTime = TimeCurrent();
        stats.validationChecks++;
        
        if (result > 0) {
            buckets[bucketIndex][entryIndex].status = RSI_READY;
            buckets[bucketIndex][entryIndex].errorCount = 0; // Reset error count on success
            return true;
        } else {
            buckets[bucketIndex][entryIndex].errorCount++;
            buckets[bucketIndex][entryIndex].status = 
                (buckets[bucketIndex][entryIndex].errorCount >= MAX_ERROR_COUNT) ? 
                RSI_ABANDONED : RSI_ERROR;
            return false;
        }
    }
    
    //+------------------------------------------------------------------+
    //| 🏭 Async handle creation                                        |
    //+------------------------------------------------------------------+
    bool CreateHandleAsync(string symbol, ENUM_TIMEFRAMES tf, int period, 
                          uint bucketIndex, int entryIndex) {
        // Create RSI handle
        int handle = iRSI(symbol, tf, period, PRICE_CLOSE);
        
        if (handle == INVALID_HANDLE) {
            buckets[bucketIndex][entryIndex].status = RSI_ERROR;
            buckets[bucketIndex][entryIndex].errorCount++;
            stats.handleErrors++;
            return false;
        }
        
        buckets[bucketIndex][entryIndex].handle = handle;
        buckets[bucketIndex][entryIndex].status = RSI_PENDING;
        buckets[bucketIndex][entryIndex].createdTime = TimeCurrent();
        stats.handleCreations++;
        stats.activeHandles++;
        
        // Immediate test for data availability (non-blocking)
        if (ValidateHandleAsync(bucketIndex, entryIndex)) {
            buckets[bucketIndex][entryIndex].status = RSI_READY;
            return true;
        }
        
        return true; // Handle created, data pending
    }
    
    //+------------------------------------------------------------------+
    //| 🧹 Periodic cache maintenance                                   |
    //+------------------------------------------------------------------+
    void PerformMaintenance() {
        datetime now = TimeCurrent();
        
        // Execute only if necessary
        if (now - lastMaintenanceTime < CACHE_CLEANUP_INTERVAL) return;
        
        int cleanedCount = 0;
        
        // Scan all buckets
        for (uint bucket = 0; bucket < CACHE_BUCKET_SIZE; bucket++) {
            int count = bucketCounts[bucket];
            
            for (int i = count - 1; i >= 0; i--) { // Backward to avoid shift issues
                bool shouldRemove = false;
                
                // Remove abandoned handles
                if (buckets[bucket][i].status == RSI_ABANDONED) {
                    shouldRemove = true;
                }
                // Remove handles that are too old
                else if (now - buckets[bucket][i].createdTime > MAX_HANDLE_AGE) {
                    shouldRemove = true;
                }
                // Remove permanently errored handles
                else if (buckets[bucket][i].status == RSI_ERROR && 
                        buckets[bucket][i].errorCount >= MAX_ERROR_COUNT) {
                    shouldRemove = true;
                }
                
                if (shouldRemove) {
                    if (buckets[bucket][i].handle != INVALID_HANDLE) {
                        IndicatorRelease(buckets[bucket][i].handle);
                        stats.activeHandles--;
                    }
                    
                    // Compact array
                    for (int j = i; j < count - 1; j++) {
                        buckets[bucket][j] = buckets[bucket][j + 1];
                    }
                    bucketCounts[bucket]--;
                    count--;
                    cleanedCount++;
                }
            }
        }
        
        lastMaintenanceTime = now;
        stats.cleanupOperations++;
        stats.lastCleanupTime = now;
        
        if (cleanedCount > 0) {
            PrintFormat("🧹 [RSI Cache] Cleanup: removed %d obsolete handles", cleanedCount);
        }
    }
    
    //+------------------------------------------------------------------+
    //| 📊 Update access statistics                                     |
    //+------------------------------------------------------------------+
    void UpdateAccessStats(uint bucketIndex, int entryIndex, bool isHit, double lookupTimeMs) {
        stats.totalRequests++;
        
        if (isHit) {
            stats.cacheHits++;
            buckets[bucketIndex][entryIndex].accessCount++;
            buckets[bucketIndex][entryIndex].lastAccessTime = TimeCurrent();
        } else {
            stats.cacheMisses++;
        }
        
        // Update average lookup time (moving average)
        if (stats.totalRequests == 1) {
            stats.avgLookupTimeMs = lookupTimeMs;
        } else {
            stats.avgLookupTimeMs = (stats.avgLookupTimeMs * 0.9) + (lookupTimeMs * 0.1);
        }
    }

public:
    //+------------------------------------------------------------------+
    //| 🏗️ Constructor                                                  |
    //+------------------------------------------------------------------+
    CRSICacheOptimized() {
        // Initialize bucket counts
        ArrayInitialize(bucketCounts, 0);
        
        // Reset lock
        isLocked = false;
        
        // Reset maintenance timer
        lastMaintenanceTime = TimeCurrent();
        
        // Initialize stats
        stats.Reset();
        
        Print("[RSI Cache] Cache initialized for 100+ assets (buckets=" + 
              IntegerToString(CACHE_BUCKET_SIZE) + ")");
    }
    
    //+------------------------------------------------------------------+
    //| 🎯 Main method: GET RSI HANDLE O(1)                             |
    //+------------------------------------------------------------------+
    int GetRSIHandle(string symbol, ENUM_TIMEFRAMES tf, int period) {
        datetime startTime = TimeCurrent();
        
        // Acquire lock thread-safe
        if (!AcquireLock(500)) {
            Print("⚠️ [RSI Cache] Lock timeout for " + symbol);
            return INVALID_HANDLE;
        }
        
        string key = BuildKey(symbol, tf, period);
        uint bucketIndex = GetHashCode(key);
        
        // Search existing entry O(1)
        int entryIndex = FindEntryIndex(key, bucketIndex);
        
        if (entryIndex != -1) {
            // Cache HIT
            double lookupTime = (TimeCurrent() - startTime) * 1000.0;
            UpdateAccessStats(bucketIndex, entryIndex, true, lookupTime);
            
            // Validate handle if necessary
            if (buckets[bucketIndex][entryIndex].needsValidation || 
                buckets[bucketIndex][entryIndex].status != RSI_READY) {
                if (!ValidateHandleAsync(bucketIndex, entryIndex)) {
                    // Invalid handle: recreate
                    if (buckets[bucketIndex][entryIndex].status != RSI_ABANDONED) {
                        CreateHandleAsync(symbol, tf, period, bucketIndex, entryIndex);
                    }
                }
                buckets[bucketIndex][entryIndex].needsValidation = false;
            }
            
            ReleaseLock();
            
            // Periodic maintenance (if needed)
            PerformMaintenance();
            
            return (buckets[bucketIndex][entryIndex].status == RSI_READY) ? 
                   buckets[bucketIndex][entryIndex].handle : INVALID_HANDLE;
        }
        
        // Cache MISS - Create new entry
        entryIndex = AddEntry(key, bucketIndex);
        if (entryIndex == -1) {
            ReleaseLock();
            Print("❌ [RSI Cache] Cannot add entry for " + key);
            return INVALID_HANDLE;
        }
        
        // Create async handle
        CreateHandleAsync(symbol, tf, period, bucketIndex, entryIndex);
        
        double lookupTime = (TimeCurrent() - startTime) * 1000.0;
        UpdateAccessStats(bucketIndex, entryIndex, false, lookupTime);
        
        ReleaseLock();
        
        PrintFormat("📌 [RSI Cache] Created handle for %s (status=%s)", 
                   key, EnumToString(buckets[bucketIndex][entryIndex].status));
        
        return (buckets[bucketIndex][entryIndex].status == RSI_READY) ? 
               buckets[bucketIndex][entryIndex].handle : INVALID_HANDLE;
    }
    
    //+------------------------------------------------------------------+
    //| 📊 Get cache statistics                                         |
    //+------------------------------------------------------------------+
    CacheStatistics GetStatistics() {
        return stats;
    }
    
    //+------------------------------------------------------------------+
    //| 📋 Print detailed report                                        |
    //+------------------------------------------------------------------+
    void PrintDetailedReport() {
        Print("📊 ========== RSI CACHE REPORT ==========");
        PrintFormat("📈 Requests: %d | Hits: %d (%.1f%%) | Misses: %d", 
                    stats.totalRequests, stats.cacheHits, stats.GetHitRate(), stats.cacheMisses);
        PrintFormat("🏭 Handles: Created=%d | Active=%d | Errors=%d", 
                    stats.handleCreations, stats.activeHandles, stats.handleErrors);
        PrintFormat("⚡ Performance: Avg lookup=%.2fms | Validations=%d", 
                    stats.avgLookupTimeMs, stats.validationChecks);
        PrintFormat("🧹 Maintenance: Operations=%d | Last cleanup=%s", 
                    stats.cleanupOperations, TimeToString(stats.lastCleanupTime));
        
        // Bucket distribution
        int usedBuckets = 0;
        int totalEntries = 0;
        for (int i = 0; i < CACHE_BUCKET_SIZE; i++) {
            if (bucketCounts[i] > 0) {
                usedBuckets++;
                totalEntries += bucketCounts[i];
            }
        }
        PrintFormat("🗂️ Buckets: Used=%d/%d | Total entries=%d | Avg per bucket=%.1f", 
                    usedBuckets, CACHE_BUCKET_SIZE, totalEntries, 
                    usedBuckets > 0 ? (double)totalEntries / usedBuckets : 0.0);
        Print("📊 =======================================");
    }
    
    //+------------------------------------------------------------------+
    //| 🗑️ Complete cleanup                                             |
    //+------------------------------------------------------------------+
    void ReleaseAll() {
        if (!AcquireLock(2000)) {
            Print("⚠️ [RSI Cache] Lock timeout during ReleaseAll");
            return;
        }
        
        int releasedCount = 0;
        
        // Release all handles
        for (uint bucket = 0; bucket < CACHE_BUCKET_SIZE; bucket++) {
            for (int i = 0; i < bucketCounts[bucket]; i++) {
                if (buckets[bucket][i].handle != INVALID_HANDLE) {
                    IndicatorRelease(buckets[bucket][i].handle);
                    releasedCount++;
                }
            }
            bucketCounts[bucket] = 0;
        }
        
        // Reset statistics
        stats.Reset();
        
        ReleaseLock();
        
        PrintFormat("🧹 [RSI Cache] Cache completely cleaned: released %d handles", releasedCount);
    }
    
    //+------------------------------------------------------------------+
    //| 🔄 Force refresh specific handle                                |
    //+------------------------------------------------------------------+
    bool RefreshHandle(string symbol, ENUM_TIMEFRAMES tf, int period) {
        if (!AcquireLock()) return false;
        
        string key = BuildKey(symbol, tf, period);
        uint bucketIndex = GetHashCode(key);
        int entryIndex = FindEntryIndex(key, bucketIndex);
        
        if (entryIndex != -1) {
            // Release existing handle
            if (buckets[bucketIndex][entryIndex].handle != INVALID_HANDLE) {
                IndicatorRelease(buckets[bucketIndex][entryIndex].handle);
                stats.activeHandles--;
            }
            
            // Recreate handle
            bool success = CreateHandleAsync(symbol, tf, period, bucketIndex, entryIndex);
            ReleaseLock();
            return success;
        }
        
        ReleaseLock();
        return false;
    }
};

//+------------------------------------------------------------------+
//| 🌐 Global optimized instance                                    |
//+------------------------------------------------------------------+
CRSICacheOptimized rsiCache;

//+------------------------------------------------------------------+
//| 📦 CIndicatorHandleCache - Cache ATR/ADX multi-asset + timeframe|
//+------------------------------------------------------------------+
#include <Arrays/ArrayString.mqh>
#include <Arrays/ArrayInt.mqh>

class CIndicatorHandleCache
{
private:
    string keys[];
    int handles[];

    string BuildKey(string symbol, ENUM_TIMEFRAMES tf, int period, int type)
    {
        return symbol + "#" + IntegerToString(tf) + "#" + IntegerToString(period) + "#" + IntegerToString(type);
    }

    bool IsHandleValid(int handle)
    {
        if (handle == INVALID_HANDLE)
            return false;
        double buffer[];
        return (CopyBuffer(handle, 0, 0, 1, buffer) >= 0);
    }

    bool WaitForIndicatorReady(int handle, int timeoutMs = 1000)
    {
        datetime start = TimeCurrent();
        double buffer[];
        while (TimeCurrent() - start < timeoutMs / 1000.0)
        {
            if (CopyBuffer(handle, 0, 0, 1, buffer) >= 0)
                return true;
            Sleep(100);
        }
        return false;
    }

    void DebugLog(string message)
    {
        if (EnableLogging_MicroTrend)
            Print(message);
    }

public:
    int GetHandle(string symbol, ENUM_TIMEFRAMES tf, int period, int type)
    {
        string key = BuildKey(symbol, tf, period, type);

        for (int i = 0; i < ArraySize(keys); i++)
        {
            if (keys[i] == key)
            {
                if (IsHandleValid(handles[i]))
                    return handles[i];

                DebugLog("⚠️ [Indicator Cache] Handle invalido per " + key + " → reinizializzo");
                handles[i] = (type == 0) ? iATR(symbol, tf, period) : iADX(symbol, tf, period);

                if (!WaitForIndicatorReady(handles[i]))
                    DebugLog("❌ [Indicator Cache] Timeout attesa dati per " + key);

                return handles[i];
            }
        }

        int handle = (type == 0) ? iATR(symbol, tf, period) : iADX(symbol, tf, period);
        if (handle != INVALID_HANDLE)
        {
            ArrayResize(keys, ArraySize(keys) + 1);
            ArrayResize(handles, ArraySize(handles) + 1);
            keys[ArraySize(keys) - 1] = key;
            handles[ArraySize(handles) - 1] = handle;

            DebugLog("📌 [Indicator Cache] Creato handle per " + key + " → handle=" + IntegerToString(handle));

            if (!WaitForIndicatorReady(handle))
                DebugLog("❌ [Indicator Cache] Timeout attesa dati per " + key);
        }
        else
        {
            DebugLog("❌ [Indicator Cache] Errore creazione handle per " + key);
        }

        return handle;
    }

    void ReleaseAll()
    {
        for (int i = 0; i < ArraySize(handles); i++)
        {
            if (handles[i] != INVALID_HANDLE)
                IndicatorRelease(handles[i]);
        }
        ArrayFree(keys);
        ArrayFree(handles);

        DebugLog("🧹 [Indicator Cache] Tutti gli handle ATR/ADX sono stati rilasciati.");
    }
};

// ✅ Istanza globale (da dichiarare in Utility.mqh)
CIndicatorHandleCache indicatorCache;

//+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

//-------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| 🧩 BLOCCO 1 - Symbol & Price Tools                               |
//+------------------------------------------------------------------+
//-------------------------------------------------------------------+

#include <Arrays\ArrayObj.mqh>    // 📦 Necessario per la Symbol Cache

//+------------------------------------------------------------------+
//| 🛢️ Classe per gestire la cache dei simboli                      |
//+------------------------------------------------------------------+
class SymbolCache : public CObject
{
public:
   string name;
   double point;
   int digits;

   // 🎯 Costruttore
   SymbolCache(string symbol)
   {
      name = symbol;
      point = SymbolInfoDouble(symbol, SYMBOL_POINT);
      digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);

      if (EnableLogging_Utility)
         PrintFormat("📦 [Utility] SymbolCache creato: %s | Point=%.10f | Digits=%d",
                     symbol, point, digits);
   }
};

// 📥 Array globale per la cache
CArrayObj symbolCacheArray;

//+------------------------------------------------------------------+
//| 🔍 Recupera SymbolCache oppure la crea                           |
//+------------------------------------------------------------------+
SymbolCache* GetSymbolCache(string symbol)
{
   for (int i = 0; i < symbolCacheArray.Total(); i++)
   {
      SymbolCache *entry = (SymbolCache*)symbolCacheArray.At(i);
      if (entry.name == symbol)
         return entry;
   }

   // 🎉 Se non trovato → crealo
   SymbolCache *newEntry = new SymbolCache(symbol);
   symbolCacheArray.Add(newEntry);
   return newEntry;
}

//+------------------------------------------------------------------+
//| 📍 Ritorna Point del simbolo                                     |
//+------------------------------------------------------------------+
double GetSymbolPoint(string symbol)
{
   SymbolCache *cache = GetSymbolCache(symbol);
   double point = (cache != NULL) ? cache.point : SymbolInfoDouble(symbol, SYMBOL_POINT);

   if (EnableLogging_Utility)
      PrintFormat("📍 [Utility] GetSymbolPoint(%s) = %.10f", symbol, point);

   return point;
}

//+------------------------------------------------------------------+
//| 🧮 Ritorna Digits del simbolo                                    |
//+------------------------------------------------------------------+
int GetSymbolDigits(string symbol)
{
    SymbolCache *cache = GetSymbolCache(symbol);

    int digits;
    if (cache != NULL)
    {
        digits = cache.digits;
        if (EnableLogging_Utility)
            PrintFormat("🧮 [Utility] GetSymbolDigits(%s) = %d (da cache)", symbol, digits);
    }
    else
    {
        long rawDigits = 0;
        if (SymbolInfoInteger(symbol, SYMBOL_DIGITS, rawDigits))
        {
            digits = (int)rawDigits;
            if (EnableLogging_Utility)
                PrintFormat("🧮 [Utility] GetSymbolDigits(%s) = %d (fallback diretta)", symbol, digits);
        }
        else
        {
            digits = 5; // fallback di sicurezza
            if (EnableLogging_Utility)
                PrintFormat("⚠️ [Utility] GetSymbolDigits(%s): errore lettura SYMBOL_DIGITS. Uso fallback = %d", symbol, digits);
        }
    }

    return digits;
}

//+------------------------------------------------------------------+
//| 🎯 Calcola la dimensione pip standard dinamicamente             |
//| - Universale: valuta, indici, metalli, crypto                   |
//+------------------------------------------------------------------+
double GetPipSize(string symbol)
{
    int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    double pipSize;

    // ✅ Caso 1: Forex (es. EURUSD) → 1 pip = 10 * point
    if ((digits == 5 || digits == 3) && bid < 100.0)
    {
        pipSize = point * 10.0;
    }
    // ✅ Caso 2: Indici / Metalli (es. USTEC) → pip = 1.0
    else if (bid > 1000.0 && point <= 0.1)
    {
        pipSize = 1.0;
    }
    // ✅ Caso 3: Altri strumenti → usa point diretto
    else
    {
        pipSize = point;
    }
    if (EnableLogging_Utility)
        PrintFormat("🧪 [DEBUG] Symbol=%s | BID=%.5f | POINT=%.5f | fallback? %s",
                    symbol, bid, point, (bid > 1000.0 ? "NO ✅" : "YES ❌"));

    if (EnableLogging_Utility)
    
    {
        PrintFormat("🎯 [Utility] GetPipSize(%s): Digits=%d | Point=%.5f | Bid=%.5f → PipSize=%.5f",
                    symbol, digits, point, bid, pipSize);
    }

    return pipSize;
}

//+------------------------------------------------------------------+
//| 🔁 Conversione Pips → Points                                      |
//+------------------------------------------------------------------+
double PipsToPoints(string symbol, double pips)
{
   double points = pips * GetPipSize(symbol);
   if (EnableLogging_Utility)
      PrintFormat("🔁 [Utility] PipsToPoints(%s, %.2f pips) = %.5f points", symbol, pips, points);
   return points;
}

//+------------------------------------------------------------------+
//| 🔁 Conversione Points → Pips                                      |
//+------------------------------------------------------------------+
double PointsToPips(string symbol, double points)
{
   double pips = points / GetPipSize(symbol);
   if (EnableLogging_Utility)
      PrintFormat("🔁 [Utility] PointsToPips(%s, %.5f points) = %.2f pips", symbol, points, pips);
   return pips;
}

//+------------------------------------------------------------------+
//| 💰 Ritorna Bid corrente                                          |
//+------------------------------------------------------------------+
double GetCurrentBid(string symbol)
{
   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);

   if (EnableLogging_Utility)
      PrintFormat("💰 [Utility] GetCurrentBid(%s) = %.5f", symbol, bid);

   return bid;
}

//+------------------------------------------------------------------+
//| 💵 Ritorna Ask corrente                                          |
//+------------------------------------------------------------------+
double GetCurrentAsk(string symbol)
{
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);

   if (EnableLogging_Utility)
      PrintFormat("💵 [Utility] GetCurrentAsk(%s) = %.5f", symbol, ask);

   return ask;
}

//+------------------------------------------------------------------+
//| 🔎 Log diagnostico valori SymbolInfoDouble                       |
//+------------------------------------------------------------------+
void LogSymbolInfoDouble(string symbol)
{
    double value;
    Print("🔍 [SymbolInfoDouble] Diagnostica per: ", symbol);

    string keys[] = {
        "SYMBOL_BID",
        "SYMBOL_ASK",
        "SYMBOL_POINT",
        "SYMBOL_TRADE_TICK_SIZE",
        "SYMBOL_TRADE_TICK_VALUE",
        "SYMBOL_TRADE_TICK_VALUE_PROFIT",
        "SYMBOL_TRADE_TICK_VALUE_LOSS",
        "SYMBOL_TRADE_CONTRACT_SIZE",
        "SYMBOL_VOLUME_MIN",
        "SYMBOL_VOLUME_MAX",
        "SYMBOL_VOLUME_STEP"
    };

    ENUM_SYMBOL_INFO_DOUBLE enumKeys[] = {
        SYMBOL_BID,
        SYMBOL_ASK,
        SYMBOL_POINT,
        SYMBOL_TRADE_TICK_SIZE,
        SYMBOL_TRADE_TICK_VALUE,
        SYMBOL_TRADE_TICK_VALUE_PROFIT,
        SYMBOL_TRADE_TICK_VALUE_LOSS,
        SYMBOL_TRADE_CONTRACT_SIZE,
        SYMBOL_VOLUME_MIN,
        SYMBOL_VOLUME_MAX,
        SYMBOL_VOLUME_STEP
    };

    for (int i = 0; i < ArraySize(enumKeys); i++)
    {
        if (SymbolInfoDouble(symbol, enumKeys[i], value))
        {
            PrintFormat("📈 %s = %.10f", keys[i], value);
        }
        else
        {
            PrintFormat("❌ %s → Errore: %d", keys[i], GetLastError());
        }
    }
}

//+------------------------------------------------------------------+
//| 🔍 Log diagnostico valori SymbolInfoInteger                      |
//+------------------------------------------------------------------+
void LogSymbolInfoInteger(string symbol)
{
    long value;
    Print("🔍 [SymbolInfoInteger] Diagnostica per: ", symbol);

    string keys[] = {
        "SYMBOL_DIGITS",
        "SYMBOL_SPREAD",
        "SYMBOL_TRADE_MODE",
        "SYMBOL_TRADE_CALC_MODE",
        "SYMBOL_SWAP_MODE",
        "SYMBOL_SWAP_ROLLOVER3DAYS",
        "SYMBOL_TRADE_EXEMODE",
        "SYMBOL_TRADE_FREEZE_LEVEL",
        "SYMBOL_TRADE_STOPS_LEVEL"
    };

    ENUM_SYMBOL_INFO_INTEGER enumKeys[] = {
        SYMBOL_DIGITS,
        SYMBOL_SPREAD,
        SYMBOL_TRADE_MODE,
        SYMBOL_TRADE_CALC_MODE,
        SYMBOL_SWAP_MODE,
        SYMBOL_SWAP_ROLLOVER3DAYS,
        SYMBOL_TRADE_EXEMODE,
        SYMBOL_TRADE_FREEZE_LEVEL,
        SYMBOL_TRADE_STOPS_LEVEL
    };

    for (int i = 0; i < ArraySize(enumKeys); i++)
    {
        if (SymbolInfoInteger(symbol, enumKeys[i], value))
        {
            PrintFormat("🔢 %s = %d", keys[i], value);
        }
        else
        {
            PrintFormat("❌ %s → Errore: %d", keys[i], GetLastError());
        }
    }
}

//+------------------------------------------------------------------+
//| 📏 Verifica validità distanze SL/TP prima di invio ordine       |
//+------------------------------------------------------------------+
bool CheckSLTPValidity(string symbol, double sl, double tp, ENUM_ORDER_TYPE type, double entryPrice)
{
    double point     = SymbolInfoDouble(symbol, SYMBOL_POINT);
    double pipSize   = GetPipSize(symbol);
    int digits       = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
    double ask       = SymbolInfoDouble(symbol, SYMBOL_ASK);
    double bid       = SymbolInfoDouble(symbol, SYMBOL_BID);
    double stopLevel = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;

    double price     = (type == ORDER_TYPE_BUY) ? ask : bid;

    double slDistance = MathAbs(price - sl);
    double tpDistance = MathAbs(tp - price);

    double slPips = slDistance / pipSize;
    double tpPips = tpDistance / pipSize;

    Print("📏 [Utility] Verifica distanza SL/TP:");
    PrintFormat("   🔹 Prezzo attuale = %.5f", price);
    PrintFormat("   🔻 SL = %.5f → dist = %.5f (%.2f pip)", sl, slDistance, slPips);
    PrintFormat("   🔺 TP = %.5f → dist = %.5f (%.2f pip)", tp, tpDistance, tpPips);
    PrintFormat("   🛡️ StopLevel broker = %.5f punti", stopLevel);

    bool slOk = (slDistance >= stopLevel);
    bool tpOk = (tpDistance >= stopLevel);

    if (!slOk || !tpOk)
    {
        PrintFormat("❌ [Utility] SL/TP troppo vicini! SL ok? %s | TP ok? %s",
                    slOk ? "✅" : "❌", tpOk ? "✅" : "❌");
        return false;
    }

    return true;
}

//+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

//-------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| 🧩 Blocco 2 - Time Tools                                         |
//+------------------------------------------------------------------+
//-------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| 🕒 Ritorna Current Server Time                                    |
//+------------------------------------------------------------------+
datetime GetCurrentTime()
{
    datetime currentTime = TimeCurrent();

    if (EnableLogging_Utility)
        PrintFormat("🕒 [Utility] GetCurrentTime() = %s", TimeToString(currentTime, TIME_DATE | TIME_SECONDS));

    return currentTime;
}

//+------------------------------------------------------------------+
//| 🕑 Ritorna ora locale corrente (0-23)                             |
//+------------------------------------------------------------------+
int GetHour()
{
    datetime t = TimeCurrent();
    MqlDateTime dt;
    TimeToStruct(t, dt);

    if (EnableLogging_Utility)
        PrintFormat("🕑 [Utility] GetHour() = %d", dt.hour);

    return dt.hour;
}

//+------------------------------------------------------------------+
//| 📅 Ritorna giorno corrente del mese (1-31)                        |
//+------------------------------------------------------------------+
int GetDay()
{
    datetime t = TimeCurrent();
    MqlDateTime dt;
    TimeToStruct(t, dt);

    if (EnableLogging_Utility)
        PrintFormat("📅 [Utility] GetDay() = %d", dt.day);

    return dt.day;
}

//+------------------------------------------------------------------+
//| 📆 Ritorna settimana corrente secondo ISO 8601                   |
//+------------------------------------------------------------------+
int GetWeek()
{
    datetime t = TimeCurrent();
    MqlDateTime dt;
    TimeToStruct(t, dt);

    // 🧮 Calcola giorno della settimana (1=lunedì ... 7=domenica)
    int weekday = dt.day_of_week == 0 ? 7 : dt.day_of_week;

    // 📅 Calcola giorno dell’anno
    int dayOfYear = dt.day_of_year;

    // 📝 Calcolo ISO week
    int week = ((dayOfYear - weekday + 10) / 7);

    if (EnableLogging_Utility)
        PrintFormat("📆 [Utility] GetWeek() ISO Week = %d", week);

    return week;
}

//+------------------------------------------------------------------+
//| 🗓️ Ritorna mese corrente (1-12)                                  |
//+------------------------------------------------------------------+
int GetMonth()
{
    datetime t = TimeCurrent();
    MqlDateTime dt;
    TimeToStruct(t, dt);

    if (EnableLogging_Utility)
        PrintFormat("🗓️ [Utility] GetMonth() = %d", dt.mon);

    return dt.mon;
}

//+------------------------------------------------------------------+
//| 📅 Ritorna anno corrente (es: 2025)                              |
//+------------------------------------------------------------------+
int GetYear()
{
    datetime t = TimeCurrent();
    MqlDateTime dt;
    TimeToStruct(t, dt);

    if (EnableLogging_Utility)
        PrintFormat("📅 [Utility] GetYear() = %d", dt.year);

    return dt.year;
}

//+------------------------------------------------------------------+
//| 🕰️ Verifica se nuova candela su un timeframe                     |
//+------------------------------------------------------------------+
bool IsNewBar(string symbol, ENUM_TIMEFRAMES timeframe)
{
    static datetime lastTime = 0;
    datetime currentTime = iTime(symbol, timeframe, 0);

    if (currentTime != lastTime)
    {
        lastTime = currentTime;
        if (EnableLogging_Utility)
            PrintFormat("🕰️ [Utility] Nuova candela rilevata su %s [%s]",
                        symbol, EnumToString(timeframe));
        return true;
    }
    return false;
}

//+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

//-------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| 🧩 Blocco 3 - Pips & SL Tools                                    |
//+------------------------------------------------------------------+
//-------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| 🧮 Calcola distanza tra due prezzi in pips                       |
//+------------------------------------------------------------------+
double CalcPips(string symbol, double price1, double price2)
{
    double pips = (price1 - price2) / GetPipSize(symbol);
    if (EnableLogging_Utility)
        PrintFormat("🧮 [Utility] CalcPips(%s): (%.5f - %.5f) / PipSize = %.2f pips",
                    symbol, price1, price2, pips);
    return pips;
}

//+------------------------------------------------------------------+
//| 🛠️ Normalizza prezzo in base ai Digits                          |
//+------------------------------------------------------------------+
double NormalizePrice(string symbol, double price)
{
    double normalized = NormalizeDouble(price, GetSymbolDigits(symbol));
    if (EnableLogging_Utility)
        PrintFormat("🛠️ [Utility] NormalizePrice(%s, %.10f) = %.10f",
                    symbol, price, normalized);
    return normalized;
}

//+------------------------------------------------------------------+
//| 🔄 Normalizza il volume in base ai parametri del simbolo         |
//+------------------------------------------------------------------+
double NormalizeVolume(double volume, string symbol)
{
    double min_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
    double lot_step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

    // ⛔ Clampa il volume tra min e max consentiti dal broker
    if (volume < min_lot) return min_lot;
    if (volume > max_lot) return max_lot;

    // 📏 Arrotonda il volume al passo del lottaggio del broker
    return round(volume / lot_step) * lot_step;
}

//+------------------------------------------------------------------+
//| 🛡️ Corregge SL e TP se troppo vicini al prezzo d’ingresso       |
//| ➕ Usa stopLevel del broker + buffer extra                       |
//+------------------------------------------------------------------+
void AdjustStopsIfTooClose(double &sl, double &tp, ENUM_ORDER_TYPE orderType, double entryPrice)
{
    // 📏 Punto e cifre del simbolo
    double point       = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    int    digits      = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

    // 🚧 Valore minimo imposto dal broker
    long rawStopLevel  = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
    double stopLevel   = rawStopLevel * point;

    // ➕ Margine extra configurabile
    double buffer       = PipsToPoints(_Symbol, MinSLBufferPips);
    double minDistance  = stopLevel + buffer;

    // 🧮 Calcola distanze attuali
    double distSL = MathAbs(entryPrice - sl);
    double distTP = MathAbs(entryPrice - tp);

    // 🔧 Correggi SL se troppo vicino
    if (distSL < minDistance)
    {
        sl = (orderType == ORDER_TYPE_BUY) ? entryPrice - minDistance
                                           : entryPrice + minDistance;

        if (EnableLogging_OpenTrade)
            PrintFormat("⚠️ [SL] Troppo vicino → Forzato a %.5f | StopLevel=%.5f | Buffer=%.5f",
                        sl, stopLevel, buffer);
    }

    // 🔧 Correggi TP se troppo vicino
    if (distTP < minDistance)
    {
        tp = (orderType == ORDER_TYPE_BUY) ? entryPrice + minDistance
                                           : entryPrice - minDistance;

        if (EnableLogging_OpenTrade)
            PrintFormat("⚠️ [TP] Troppo vicino → Forzato a %.5f | StopLevel=%.5f | Buffer=%.5f",
                        tp, stopLevel, buffer);
    }

    // 🧼 Normalizza per evitare rifiuti da decimali
    sl = NormalizeDouble(sl, digits);
    tp = NormalizeDouble(tp, digits);

    // 📢 Log finale completo
    if (EnableLogging_OpenTrade)
    {
        PrintFormat("✅ [SLTP Adjust] Entry=%.5f | SL=%.5f | TP=%.5f | distSL=%.5f | distTP=%.5f | MinReq=%.5f",
                    entryPrice, sl, tp, MathAbs(entryPrice - sl), MathAbs(entryPrice - tp), minDistance);
    }
}

//+------------------------------------------------------------------+
//| 🚨 Check distanza minima SL (versione migliorata BOT 4.0)        |
//+------------------------------------------------------------------+
bool CheckTrailingSLDistance(string symbol, double price, double newSL, ENUM_POSITION_TYPE type)
{
    double pointSize = SymbolInfoDouble(symbol, SYMBOL_POINT);
    long stopLevelPoints = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
    double stopLevel = stopLevelPoints * pointSize;

    if (stopLevel < pointSize || stopLevel == 0.0)
    {
        int atrHandle = iATR(symbol, PERIOD_M5, 14);
        double atrBuffer[];
        if (CopyBuffer(atrHandle, 0, 0, 1, atrBuffer) > 0)
            stopLevel = atrBuffer[0] * 0.25;

        if (EnableLogging_Utility)
            PrintFormat("⚠️ [Utility] CheckTrailingSLDistance fallback ATR: %.5f", stopLevel);
    }

    double distance = MathAbs(price - newSL);
    if (distance < stopLevel)
    {
        if (EnableLogging_Utility)
            PrintFormat("🚨 [Utility] SL troppo vicino su %s: distanza = %.2f < min = %.2f",
                        symbol, distance, stopLevel);
        return true;
    }

    if (EnableLogging_Utility)
        PrintFormat("✅ [Utility] Distanza SL OK su %s: distanza = %.2f ≥ min = %.2f",
                    symbol, distance, stopLevel);

    return false;
}

//+------------------------------------------------------------------+
//| 📏 Calcola distanza SL in pip standard                          |
//| - entryPrice: prezzo di ingresso                                 |
//| - stopLoss: prezzo dello SL (inferiore o superiore)              |
//| - symbol: strumento (default = _Symbol)                          |
//| Return: distanza SL in pip standard (es: 15 pip = 0.0015 EURUSD) |
//+------------------------------------------------------------------+
double GetSLPips(double entryPrice, double stopLoss, string symbol)
{
    double pipSize = GetPipSize(symbol);
    double slPips = MathAbs(entryPrice - stopLoss) / pipSize;

    if (EnableLogging_Utility)
        PrintFormat("📏 [Utility] GetSLPips(%s): Entry=%.5f | SL=%.5f | PipSize=%.5f → SL=%.2f pip",
                    symbol, entryPrice, stopLoss, pipSize, slPips);

    return slPips;
}

//+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

//-------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| 🧩 Blocco 4 - Lot Management                                     |
//+------------------------------------------------------------------+
//-------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| 🧩 Wrapper unico per selezione metodo di calcolo lotto           |
//+------------------------------------------------------------------+

// Questa funzione ora deve usare i parametri che le vengono passati
// e chiamare la funzione di calcolo lotto appropriata.
double GetFinalLot(double slPips, string symbol, ENUM_ORDER_TYPE direction, bool enableLog = false)
{
    // Chiamiamo la nuova funzione di calcolo del lotto con i parametri corretti
    // Assicurati che 'CalculateTradableLot' sia dichiarata e accessibile in questo file (es. inclusa).
    return CalculateTradableLot(symbol, direction, slPips, enableLog);
}
 
//+------------------------------------------------------------------+
//| 🧮 CalculateTradableLot - Calcolo lotto tradabile con margine    |
//| 🔹 Supporta moltiplicatore (_lotMultiplier) per RecoveryBlock    |
//| 🔹 Fallback decrementale con OrderCalcMargin ✅                  |
//| 🔹 Conferma finale con trade.Check ✅                            |
//| 🔹 Supporta UseAutoLot + MaxAutoLot + LotSize ✅                 |
//+------------------------------------------------------------------+
double CalculateTradableLot(string _symbol, ENUM_ORDER_TYPE _tradeType, double _slPips, bool _enableLog, double _lotMultiplier = 1.0)
{
    //───────────────────────────────────────────────────────────────
    // ⚙️ Modalità calcolo lotto: AUTO o FISSO
    //───────────────────────────────────────────────────────────────
    if (!UseAutoLot)
    {
        if (_enableLog)
            PrintFormat("⚙️ [CalculateTradableLot] UseAutoLot = FALSE → uso Lotto fisso = %.2f", LotSize);
        return LotSize;
    }

    //───────────────────────────────────────────────────────────────
    // 1 ️⃣  Recupera informazioni sul simbolo
    //───────────────────────────────────────────────────────────────
    double point      = SymbolInfoDouble(_symbol, SYMBOL_POINT);
    double tick_value = SymbolInfoDouble(_symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size  = SymbolInfoDouble(_symbol, SYMBOL_TRADE_TICK_SIZE);

    if (point <= 0 || tick_value <= 0 || tick_size <= 0)
    {
        if (_enableLog)
            PrintFormat("❌ [CalculateTradableLot] ERRORE: Impossibile ottenere info simbolo %s (point=%.5f tick_value=%.5f tick_size=%.5f).", _symbol, point, tick_value, tick_size);
        return 0.0;
    }

    //───────────────────────────────────────────────────────────────
    // 2 ️⃣  Calcola valore pip per 1 lotto
    //───────────────────────────────────────────────────────────────
    double pip_value_per_lot = (tick_value / tick_size) * point;

    double loss_per_lot = _slPips * pip_value_per_lot;
    if (loss_per_lot <= 0)
    {
        if (_enableLog)
            PrintFormat("❌ [CalculateTradableLot] ERRORE: SL pips=%.2f o valore pip=%.5f non valido. Annullamento.", _slPips, pip_value_per_lot);
        return 0.0;
    }

    //───────────────────────────────────────────────────────────────
    // 3 ️⃣  Calcolo lotto teorico in base al rischio
    //───────────────────────────────────────────────────────────────
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double risk_amount = balance * (RiskPercent / 100.0);

    double calculated_lot = risk_amount / loss_per_lot;

    // ⭐⭐ Applica moltiplicatore (Recovery usa RecoveryLotMultiplier, bot normale usa 1.0) ⭐⭐
    calculated_lot *= _lotMultiplier;

    // 🚦 Protezione: limite massimo in modalità AUTO
    if (calculated_lot > MaxAutoLot)
    {
        if (_enableLog)
            PrintFormat("⚠️ [CalculateTradableLot] Lotto teorico %.4f eccede MaxAutoLot (%.2f) → ridotto a MaxAutoLot.", calculated_lot, MaxAutoLot);
        calculated_lot = MaxAutoLot;
    }

    if (_enableLog)
        PrintFormat("🔎 [CalculateTradableLot] Lotto teorico=%.4f | Rischio=%.2f | SL=%.2f pips | Multiplier=%.2f", calculated_lot, risk_amount, _slPips, _lotMultiplier);

    //───────────────────────────────────────────────────────────────
    // 4 ️⃣  Normalizzazione lotto (MinLot, MaxLot, LotStep)
    //───────────────────────────────────────────────────────────────
    double minLot  = SymbolInfoDouble(_symbol, SYMBOL_VOLUME_MIN);
    double maxLot  = SymbolInfoDouble(_symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_symbol, SYMBOL_VOLUME_STEP);

    double lot = MathFloor(calculated_lot / lotStep) * lotStep;
    lot = MathMax(minLot, MathMin(lot, maxLot));

    if (_enableLog)
        PrintFormat("🔎 [CalculateTradableLot] Lotto normalizzato=%.4f | MinLot=%.2f | MaxLot=%.2f | LotStep=%.2f", lot, minLot, maxLot, lotStep);

    //───────────────────────────────────────────────────────────────
    // 5 ️⃣  Verifica margine con fallback decrementale (OrderCalcMargin)
    //───────────────────────────────────────────────────────────────
    double current_price = (_tradeType == ORDER_TYPE_BUY) ? SymbolInfoDouble(_symbol, SYMBOL_ASK) : SymbolInfoDouble(_symbol, SYMBOL_BID);
    if (current_price <= 0)
    {
        if (_enableLog)
            PrintFormat("❌ [CalculateTradableLot] ERRORE: Prezzo corrente non valido per %s. Annullamento.", _symbol);
        return 0.0;
    }

    double required_margin = 0;
    double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
    
    bool margin_ok = false;

    for (double test_lot = lot; test_lot >= minLot; test_lot -= lotStep)
    {
        test_lot = NormalizeDouble(test_lot, 2);

        if (OrderCalcMargin(_tradeType, _symbol, test_lot, current_price, required_margin))
        {
            if (required_margin <= free_margin)
            {
                lot = test_lot;
                margin_ok = true;
                break; // ✅ Lotto accettabile trovato
            }
            else
            {
                if (_enableLog)
                    PrintFormat("⚠️ [CalculateTradableLot] Lotto %.2f richiede margine %.2f > disponibile %.2f → provo più basso.", test_lot, required_margin, free_margin);
            }
        }
        else
        {
            if (_enableLog)
                PrintFormat("❌ [CalculateTradableLot] ERRORE OrderCalcMargin fallito per lotto %.2f → Errore: %d", test_lot, GetLastError());
            return 0.0;
        }
    }

    if (!margin_ok)
    {
        if (_enableLog)
            PrintFormat("❌ [CalculateTradableLot] Impossibile trovare un lotto valido per il margine disponibile.");
        return 0.0;
    }

    //───────────────────────────────────────────────────────────────
    // ✅ Lotto finale pronto
    //───────────────────────────────────────────────────────────────
    if (_enableLog)
        PrintFormat("✅ [CalculateTradableLot] Lotto FINALE PRONTO: %.2f | Margine richiesto=%.2f | Margine disponibile=%.2f", lot, required_margin, free_margin);

    return lot;
}

//+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

//-------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| 🧩 Blocco 5 - Safe Modify Tools                                  |
//+------------------------------------------------------------------+
//-------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| 🔐 SafeModifySL v2                                               |
//| Modifica SL/TP con fallback, protezione, normalizzazione, log    |
//| AGGIORNATA: Rimosso AdjustStopsOfPositionIfTooClose              |
//+------------------------------------------------------------------+
bool SafeModifySL(string symbol, ulong ticket, double &sl, double &tp)
{
    // 🧱 Verifica esistenza posizione
    if (!PositionSelectByTicket(ticket))
    {
        if (EnableLogging_Utility)
            PrintFormat("❌ [SafeModifySL] Ticket %d non trovato.", ticket);
        return false;
    }

    // 🎯 Normalizza i prezzi SL/TP
    // ASSICURATI CHE NormalizePrice ESISTA E FUNZIONI CORRETTAMENTE NEL TUO EA
    // Questa funzione è responsabile di arrotondare il prezzo al TickSize corretto.
    sl = NormalizePrice(symbol, sl);
    tp = NormalizePrice(symbol, tp);

    // 🛠️ Prepara richiesta manuale
    MqlTradeRequest request;
    MqlTradeResult  result;
    ZeroMemory(request);
    ZeroMemory(result);

    request.action      = TRADE_ACTION_SLTP;
    request.symbol      = symbol;
    request.position    = ticket;
    request.sl          = sl;
    request.tp          = tp;
    request.type_time   = ORDER_TIME_GTC;
    request.type_filling = ORDER_FILLING_IOC; // Generalmente non necessario per SLTP, ma non dannoso

    // 📢 Logga i valori prima di inviare la richiesta per debugging
    if (EnableLogging_Utility)
        PrintFormat("🔍 [SafeModifySL] Tentativo di modificare SL/TP per ticket %d | SL=%.5f | TP=%.5f", ticket, sl, tp);

    if (!OrderSend(request, result))
    {
        int error = GetLastError();
        // Usa la tua funzione HandleTradeError per loggare il problema
        HandleTradeError(error, "OrderSend SLTP", symbol);
        return false;
    }

    if (result.retcode != TRADE_RETCODE_DONE)
    {
        // Se la richiesta non è stata completata con successo, logga il retcode
        if (EnableLogging_Utility)
            PrintFormat("⚠️ [SafeModifySL] Richiesta fallita. RetCode=%d - %s su ticket %d", result.retcode, result.comment, ticket);
        // Usa la tua funzione HandleTradeError anche per i retcode non DONE
        HandleTradeError(result.retcode, "OrderSend SLTP (RetCode Fail)", symbol);
        return false;
    }

    if (EnableLogging_Utility)
        PrintFormat("✅ [SafeModifySL] SL/TP modificati con successo su ticket %d | SL=%.5f | TP=%.5f", ticket, sl, tp);

    return true;
}

//-----------------------------------------------------------------------+
// --- STRUTTURA E VARIABILE GLOBALE PER IL RECOVERY BLOCK SIGNAL ---   |
//-----------------------------------------------------------------------+
// --- CLASSE PER IL SEGNALE DI RECUPERO ---
struct RecoveryTriggerInfo
{
    string             Symbol;             // Simbolo dell'operazione che ha triggerato il segnale
    ENUM_POSITION_TYPE OriginalPositionType; // Direzione dell'operazione che ha triggerato
    double             OriginalLotSize;    // Lottaggio dell'operazione originale
    bool               IsProcessed;        // Flag: true se il RecoveryBlock ha già tentato di elaborare questo segnale
    datetime           CreationTime;       // Timestamp della creazione del segnale
    ulong              RecoveryTradeTicket; // Nuovo campo: Ticket del trade di recupero associato

    // Costruttore
    RecoveryTriggerInfo(string _symbol = "", ENUM_POSITION_TYPE _type = POSITION_TYPE_BUY, double _lot = 0.0)
        : Symbol(_symbol), OriginalPositionType(_type), OriginalLotSize(_lot),
          IsProcessed(false), CreationTime(TimeCurrent()), RecoveryTradeTicket(0) // Inizializza a 0
    {
    }
};

// --- Implementazione di una Mappa basata su Array (generica per tipi NON-PUNTATORE) ---
template<typename T>
class ArrayBasedMap
{
private:
    long               m_keys[];   // array di chiavi (ticket)
    T                  m_values[]; // array di valori (RecoveryTriggerInfo, CEMACache, ecc.)
    int                m_size;     // numero di elementi nella mappa

public:
    ArrayBasedMap() : m_size(0) {}

    ~ArrayBasedMap()
    {
        Clear();
    }

    // Aggiunge o aggiorna un elemento
    bool Add(long key, const T& value)
    {
        for (int i = 0; i < m_size; i++)
        {
            if (m_keys[i] == key)
            {
                m_values[i] = value; // Aggiorna il valore esistente (copia la struct)
                return true;
            }
        }
        // Aggiunge un nuovo elemento
        ArrayResize(m_keys, m_size + 1);
        ArrayResize(m_values, m_size + 1);
        m_keys[m_size] = key;
        m_values[m_size] = value; // Copia la struct nel nuovo slot
        m_size++;
        return true;
    }

    // Ottiene un elemento (popola 'value' con una copia)
    bool TryGetValue(long key, T& value) const
    {
        for (int i = 0; i < m_size; i++)
        {
            if (m_keys[i] == key)
            {
                value = m_values[i];
                return true;
            }
        }
        return false;
    }

    // Controlla se la chiave esiste
    bool ContainsKey(long key) const
    {
        for (int i = 0; i < m_size; i++)
        {
            if (m_keys[i] == key)
            {
                return true;
            }
        }
        return false;
    }

    // Rimuove un elemento
    bool Remove(long key)
    {
        for (int i = 0; i < m_size; i++)
        {
            if (m_keys[i] == key)
            {
                // Sposta gli elementi successivi indietro
                for (int j = i; j < m_size - 1; j++)
                {
                    m_keys[j] = m_keys[j + 1];
                    m_values[j] = m_values[j + 1];
                }
                m_size--;
                ArrayResize(m_keys, m_size);
                ArrayResize(m_values, m_size);
                return true;
            }
        }
        return false;
    }

    // Ottiene il numero di elementi
    int Size() const { return m_size; }
    
    // Ottiene una copia delle chiavi
    int GetKeys(long& keys_array[]) const
    {
        ArrayResize(keys_array, m_size);
        for(int i = 0; i < m_size; i++)
        {
            keys_array[i] = m_keys[i];
        }
        return m_size;
    }
    
    // Resetta la mappa.
    void Clear()
    {
        m_size = 0;
        ArrayFree(m_keys);
        ArrayFree(m_values);
    }
};

//+------------------------------------------------------------------+
//| Funzione di utilità: EnumToString per ENUM_POSITION_TYPE         |
//+------------------------------------------------------------------+
string EnumToString(ENUM_POSITION_TYPE type)
{
    if (type == POSITION_TYPE_BUY) return "BUY";
    if (type == POSITION_TYPE_SELL) return "SELL";
    return "UNKNOWN";
}

//+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

//-------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| 🧩 Blocco 6 - ATR Calculation + ADX Calculation                  |
//+------------------------------------------------------------------+
//-------------------------------------------------------------------+

//+---------------------------------------------------------------------------+
//| 📊 CalculateMultiPeriodATR() - Utility BOT 4.0                            |
//| Calcola ATR per più periodi indicati, restituisce array valori            |
//|                                                                           |
//| 📥 Input:                                                                 |
//| - symbol: simbolo asset (es. "USTEC")                                     |
//| - timeframe: timeframe su cui calcolare gli ATR                           |
//| - periods[]: array di periodi da calcolare (es. {7,14,21})                |
//|                                                                           |
//| 📤 Output:                                                                |
//| - results[]: array dove verranno salvati i valori ATR                     |
//|                                                                           |
//| ✅ Return: true se tutti gli ATR calcolati correttamente, false se errori|
//+---------------------------------------------------------------------------+
bool CalculateMultiPeriodATR(string symbol, ENUM_TIMEFRAMES timeframe, int &periods[], double &results[])
{
    int count = ArraySize(periods);
    ArrayResize(results, count);

    for (int i = 0; i < count; i++)
    {
        int period = periods[i];
        int atrHandle = indicatorCache.GetHandle(symbol, timeframe, period, 0);  // type 0 = ATR

        if (atrHandle == INVALID_HANDLE)
        {
            if (EnableLogging_Utility)
                PrintFormat("❌ [Utility] ATR handle invalido per %s [%s], periodo %d", symbol, EnumToString(timeframe), period);
            results[i] = 0.0;
            return false;
        }

        double buffer[];
        if (SafeCopyBuffer(atrHandle, 0, 0, 1, buffer))
        {
            results[i] = buffer[0];
            if (EnableLogging_Utility)
                PrintFormat("📊 [Utility] ATR(%d) = %.5f su %s [%s]", period, results[i], symbol, EnumToString(timeframe));
        }
        else
        {
            results[i] = 0.0;
            if (EnableLogging_Utility)
                PrintFormat("❌ [Utility] Errore CopyBuffer ATR(%d) su %s [%s]", period, symbol, EnumToString(timeframe));
            return false;
        }
    }

    return true;
}

//+------------------------------------------------------------------+
//| 📶 CalculateADX() - Utility BOT 4.0 - VERSIONE MIGLIORATA       |
//| Calcola valore ADX corrente su simbolo + timeframe + periodo     |
//+------------------------------------------------------------------+
double CalculateADX(string symbol, ENUM_TIMEFRAMES timeframe, int adxPeriod)
{
    double adxValue = 0.0;
    
    if (EnableLogging_Utility) 
        PrintFormat("DEBUG Utility ADX - Inizio. Symbol=%s, Timeframe=%s, Period=%d", symbol, EnumToString(timeframe), adxPeriod);
    
    // 🔥 MIGLIORAMENTO #1: Validazione parametri input
    if (adxPeriod < 2 || adxPeriod > 100)
    {
        if (EnableLogging_Utility) 
            PrintFormat("DEBUG Utility ADX - ERRORE: Periodo ADX non valido (%d). Deve essere 2-100.", adxPeriod);
        return 0.0;
    }
    
    // 🔥 MIGLIORAMENTO #2: Verifica dati sufficienti
    int availableBars = iBars(symbol, timeframe);
    int requiredBars = adxPeriod * 3; // ADX richiede circa 3x il periodo per essere stabile
    
    if (availableBars < requiredBars)
    {
        if (EnableLogging_Utility) 
            PrintFormat("DEBUG Utility ADX - ATTENZIONE: Dati insufficienti. Disponibili=%d, Richiesti=%d", availableBars, requiredBars);
        // Non returniamo 0, continuiamo comunque ma con warning
    }
    
    int adxHandle = indicatorCache.GetHandle(symbol, timeframe, adxPeriod, 1); // type 1 = ADX
    
    if (adxHandle == INVALID_HANDLE)
    {
        if (EnableLogging_Utility) 
            PrintFormat("DEBUG Utility ADX - ERRORE: indicatorCache.GetHandle ha restituito INVALID_HANDLE. Errore: %d", GetLastError());
        return 0.0;
    }
    
    if (EnableLogging_Utility) 
        PrintFormat("DEBUG Utility ADX - indicatorCache.GetHandle Handle: %d", adxHandle);
    
    // 🔥 MIGLIORAMENTO #3: Attesa calcolo indicatore con retry
    int maxRetries = 5;
    int retryCount = 0;
    bool dataReady = false;
    
    while (retryCount < maxRetries && !dataReady)
    {
        // Verifica se l'indicatore è pronto
        if (BarsCalculated(adxHandle) >= adxPeriod)
        {
            dataReady = true;
            if (EnableLogging_Utility) 
                PrintFormat("DEBUG Utility ADX - Indicatore pronto. BarsCalculated=%d", BarsCalculated(adxHandle));
        }
        else
        {
            retryCount++;
            if (EnableLogging_Utility) 
                PrintFormat("DEBUG Utility ADX - Tentativo %d/%d: BarsCalculated=%d < %d. Attendo...", 
                            retryCount, maxRetries, BarsCalculated(adxHandle), adxPeriod);
            
            Sleep(50); // Attendi 50ms prima del prossimo tentativo
        }
    }
    
    if (!dataReady)
    {
        if (EnableLogging_Utility) 
            PrintFormat("DEBUG Utility ADX - TIMEOUT: Indicatore non pronto dopo %d tentativi.", maxRetries);
        return 0.0;
    }
    
    // 🔥 MIGLIORAMENTO #4: Copia multipli valori per validazione
    double buffer[];
    int numToCopy = MathMin(3, BarsCalculated(adxHandle)); // Copia fino a 3 valori
    
    if (SafeCopyBuffer(adxHandle, MAIN_LINE, 0, numToCopy, buffer)) 
    {
        adxValue = buffer[0]; // Valore più recente
        
        if (EnableLogging_Utility) 
        {
            PrintFormat("DEBUG Utility ADX - CopyBuffer SUCCESS. Valori copiati: %d", numToCopy);
            for (int i = 0; i < numToCopy; i++)
            {
                PrintFormat("DEBUG Utility ADX - ADX[%d]: %.5f", i, buffer[i]);
            }
        }
        
        // 🔥 MIGLIORAMENTO #5: Validazione range ADX
        if (adxValue < 0.0 || adxValue > 100.0)
        {
            if (EnableLogging_Utility) 
                PrintFormat("DEBUG Utility ADX - ATTENZIONE: Valore ADX fuori range normale (%.5f). Range tipico: 0-100.", adxValue);
            
            // Se completamente fuori range, considera non valido
            if (adxValue < -10.0 || adxValue > 150.0)
            {
                if (EnableLogging_Utility) 
                    PrintFormat("DEBUG Utility ADX - ERRORE: Valore ADX estremamente anomalo (%.5f). Ritorno 0.0.", adxValue);
                return 0.0;
            }
            
            // Se leggermente fuori range, normalizza
            adxValue = MathMax(0.0, MathMin(100.0, adxValue));
        }
        
        // 🔥 MIGLIORAMENTO #6: Controllo stabilità (se abbiamo più valori)
        if (numToCopy >= 2)
        {
            double adxPrev = buffer[1];
            double adxVariation = MathAbs(adxValue - adxPrev);
            
            if (adxVariation > 50.0) // Variazione eccessiva tra tick consecutivi
            {
                if (EnableLogging_Utility) 
                    PrintFormat("DEBUG Utility ADX - ATTENZIONE: Variazione ADX eccessiva (%.2f → %.2f, Δ%.2f). Possibile dato anomalo.", 
                                adxPrev, adxValue, adxVariation);
                
                // In caso di variazione estrema, usa la media degli ultimi valori
                if (numToCopy >= 3)
                {
                    double avgADX = (buffer[0] + buffer[1] + buffer[2]) / 3.0;
                    if (EnableLogging_Utility) 
                        PrintFormat("DEBUG Utility ADX - Uso media degli ultimi 3 valori: %.2f", avgADX);
                    adxValue = avgADX;
                }
            }
        }
        
        // 🔥 MIGLIORAMENTO #7: Cache ultimo valore valido (opzionale)
        static double lastValidADX = 0.0;
        static string lastSymbol = "";
        static ENUM_TIMEFRAMES lastTF = PERIOD_CURRENT;
        
        if (adxValue > 0.0)
        {
            lastValidADX = adxValue;
            lastSymbol = symbol;
            lastTF = timeframe;
        }
        else if (lastValidADX > 0.0 && lastSymbol == symbol && lastTF == timeframe)
        {
            // Se il valore attuale non è valido ma abbiamo un ultimo valore valido per stesso symbol/tf
            if (EnableLogging_Utility) 
                PrintFormat("DEBUG Utility ADX - Valore corrente non valido (%.2f). Uso ultimo valore valido: %.2f", adxValue, lastValidADX);
            adxValue = lastValidADX;
        }
    }
    else
    {
        if (EnableLogging_Utility) 
            PrintFormat("DEBUG Utility ADX - ERRORE: SafeCopyBuffer fallito. Errore: %d", GetLastError());
        
        // 🔥 MIGLIORAMENTO #8: Fallback con calcolo manuale semplificato (opzionale)
        // Se il copy buffer fallisce, potremmo provare un calcolo manuale basic
        // ma per ora manteniamo il return 0.0 per sicurezza
        adxValue = 0.0;
    }
    
    if (EnableLogging_Utility) 
        PrintFormat("DEBUG Utility ADX - Uscita funzione. ADX finale: %.5f", adxValue);
    
    return adxValue;
}

//+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

//-------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| 🧩 Blocco 7 - Position & Order Tools                             |
//+------------------------------------------------------------------+
//-------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| ⏳ Verifica se sono passati X secondi dall'inizio della candela |
//+------------------------------------------------------------------+
bool IsInitialCandleDelayPassed(string symbol, ENUM_TIMEFRAMES tf, int delaySec)
{
    datetime time[];
    if (CopyTime(symbol, tf, 0, 1, time) < 1)
        return false;

    datetime candleStart = time[0];
    return (TimeCurrent() - candleStart) >= delaySec;
}

//+------------------------------------------------------------------+
//| 📊 Conta quante posizioni sono aperte su un simbolo              |
//+------------------------------------------------------------------+
int GetOpenPositionsCount(string symbol)
{
    int count = 0;

    for (int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if (PositionSelectByTicket(ticket))
        {
            if (PositionGetString(POSITION_SYMBOL) == symbol)
                count++;
        }
    }

    if (EnableLogging_Utility)
        PrintFormat("📊 [Utility] GetOpenPositionsCount(%s) = %d", symbol, count);

    return count;
}

//+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

//-------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| 🧩 Blocco 8 - Trading Error Handling                             |
//+------------------------------------------------------------------+
//-------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| ❗ CheckLastError()                                               |
//| Legge ultimo errore e logga + reset                              |
//+------------------------------------------------------------------+
void CheckLastError()
{
    int error = GetLastError();
    if (error != 0 && EnableLogging_Utility)
    {
        PrintFormat("❗ [Utility] LastError = %d → %s", error, TradingErrorToString(error));
        ResetLastError();
    }
}

//+------------------------------------------------------------------+
//| ⚠️ Gestione centralizzata degli errori trading                  |
//+------------------------------------------------------------------+
void HandleTradeError(int code, string context, string symbol)
{
    string description = TradingErrorToString(code);
    if (EnableLogging_Utility)
    {
        PrintFormat("❌ [%s] Errore trading su %s → Codice: %d | %s", 
                    context, symbol, code, description);
    }
    ResetLastError();
}

//+------------------------------------------------------------------+
//| 📝 TradingErrorToString()                                        |
//| Restituisce descrizione testo per codice errore                  |
//+------------------------------------------------------------------+
string TradingErrorToString(int error)
{
    switch (error)
    {
        case 0: return "No error";
        case 1: return "No error returned";
        case 2: return "Common error";
        case 3: return "Invalid trade parameters";
        case 4: return "Trade server busy";
        case 5: return "Old version of client";
        case 6: return "No connection with trade server";
        case 8: return "Invalid account";
        case 9: return "Trade timeout";
        case 64: return "Account disabled";
        case 65: return "Invalid account";
        case 128: return "Trade context busy";
        case 129: return "Invalid price";
        case 130: return "Invalid stops";
        case 131: return "Invalid volume";
        case 132: return "Market closed";
        case 133: return "Trade disabled";
        case 134: return "Not enough money";
        case 135: return "Price changed";
        case 136: return "Off quotes";
        case 137: return "Broker busy";
        case 138: return "Requote";
        case 139: return "Order locked";
        case 140: return "Long positions only allowed";
        case 141: return "Too many requests";
        case 4756: return "SL modification rejected (common indices)";
        case 4060: return "No connection (new)";
        case 4106: return "Price changed (new)";
        case 4107: return "Invalid price (new)";
        case 4108: return "Market closed (new)";
        case 4109: return "Off quotes (new)";
        case 10004: return "Order send timeout";
        case 10006: return "No prices";
        case 10007: return "Broker not connected";
        case 10008: return "Too frequent requests";
        // ❓ Fallback
        default: return "Unknown error";
    }
}

//+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

#endif // __UTILITY_MQH__
