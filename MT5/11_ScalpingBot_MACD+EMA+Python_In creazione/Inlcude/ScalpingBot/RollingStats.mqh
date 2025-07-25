//+------------------------------------------------------------------+
//|                     RollingStats.mqh - VERSIONE OTTIMIZZATA     |
//|  Buffer circolare O(1) con statistiche robuste e validazione    |
//+------------------------------------------------------------------+
#ifndef __ROLLING_STATS_OPTIMIZED_MQH__
#define __ROLLING_STATS_OPTIMIZED_MQH__

//+------------------------------------------------------------------+
//| 📊 Struct ottimizzata per statistiche rolling O(1)              |
//+------------------------------------------------------------------+
struct RollingStatsOptimized
{
    // Buffer e indici
    double buffer[];       // Buffer valori circolari
    int maxCount;          // Dimensione massima buffer  
    int index;             // Indice inserimento corrente
    int count;             // Numero valori attuali
    
    // Statistiche incrementali O(1)
    double sum;            // Somma corrente
    double sumSquares;     // Somma quadrati (per varianza)
    double minValue;       // Valore minimo nel buffer
    double maxValue;       // Valore massimo nel buffer
    
    // Statistiche calcolate
    double mean;           // Media aritmetica
    double variance;       // Varianza campionaria
    double stddev;         // Deviazione standard
    
    // Statistiche robuste (per dati non-normali)
    double median;         // Mediana (calcolata on-demand)
    double mad;            // Median Absolute Deviation
    bool needsMedianUpdate; // Flag per ricalcolo mediana
    
    // Validazione e controllo qualità
    bool isReliable;       // True se abbastanza campioni per statistiche affidabili
    datetime lastUpdate;   // Timestamp ultimo aggiornamento
    int minSamplesReliable; // Numero minimo campioni per affidabilità
};

//+------------------------------------------------------------------+
//| 🚀 Inizializza RollingStats ottimizzato                         |
//+------------------------------------------------------------------+
void InitRollingStatsOptimized(RollingStatsOptimized &stats, int maxCount, int minReliable = 20)
{
    // Resize e reset buffer
    ArrayResize(stats.buffer, maxCount);
    ArrayInitialize(stats.buffer, 0.0);
    
    // Reset contatori
    stats.maxCount = maxCount;
    stats.index = 0;
    stats.count = 0;
    stats.minSamplesReliable = MathMax(minReliable, 5); // Minimo assoluto 5
    
    // Reset statistiche incrementali
    stats.sum = 0.0;
    stats.sumSquares = 0.0;
    stats.minValue = DBL_MAX;
    stats.maxValue = -DBL_MAX;
    
    // Reset statistiche calcolate
    stats.mean = 0.0;
    stats.variance = 0.0;
    stats.stddev = 0.0;
    stats.median = 0.0;
    stats.mad = 0.0;
    
    // Reset flag controllo
    stats.needsMedianUpdate = false;
    stats.isReliable = false;
    stats.lastUpdate = TimeCurrent();
}

//+------------------------------------------------------------------+
//| ⚡ Aggiorna statistiche in O(1) - PERFORMANCE CRITICA          |
//+------------------------------------------------------------------+
void UpdateRollingStatsOptimized(RollingStatsOptimized &stats, double newValue)
{
    // 🛡️ Validazione input
    if (!MathIsValidNumber(newValue))
    {
        Print("⚠️ [RollingStats] Valore non valido ignorato: ", newValue);
        return;
    }
    
    // 📥 Recupera valore vecchio che sarà sostituito
    double oldValue = 0.0;
    bool isReplacing = (stats.count == stats.maxCount);
    
    if (isReplacing)
        oldValue = stats.buffer[stats.index];
    
    // 🔄 Aggiorna buffer circolare
    stats.buffer[stats.index] = newValue;
    stats.index = (stats.index + 1) % stats.maxCount;
    
    // 📊 Aggiorna contatori
    if (!isReplacing)
        stats.count++;
    
    // ⚡ Aggiorna statistiche incrementali O(1)
    stats.sum += newValue - oldValue;
    stats.sumSquares += (newValue * newValue) - (oldValue * oldValue);
    
    // 🔍 Aggiorna min/max (solo se necessario)
    if (newValue < stats.minValue)
        stats.minValue = newValue;
    if (newValue > stats.maxValue)
        stats.maxValue = newValue;
    
    // Se rimuoviamo il min/max, dobbiamo ricalcolare (operazione costosa)
    if (isReplacing && (oldValue == stats.minValue || oldValue == stats.maxValue))
    {
        RecalculateMinMax(stats);
    }
    
    // 🧮 Calcola statistiche principali O(1)
    CalculateBasicStats(stats);
    
    // 📈 Marca mediana per ricalcolo
    stats.needsMedianUpdate = true;
    stats.lastUpdate = TimeCurrent();
    
    // ✅ Verifica affidabilità
    stats.isReliable = (stats.count >= stats.minSamplesReliable);
}

//+------------------------------------------------------------------+
//| 🧮 Calcola media, varianza, stddev in O(1)                     |
//+------------------------------------------------------------------+
void CalculateBasicStats(RollingStatsOptimized &stats)
{
    if (stats.count == 0) return;
    
    // Media O(1)
    stats.mean = stats.sum / stats.count;
    
    // Varianza campionaria O(1) - Formula: Var = E[X²] - E[X]²
    if (stats.count > 1)
    {
        double meanSquares = stats.sumSquares / stats.count;
        stats.variance = (meanSquares - stats.mean * stats.mean) * stats.count / (stats.count - 1);
        stats.variance = MathMax(0.0, stats.variance); // Evita negativi per errori numerici
        stats.stddev = MathSqrt(stats.variance);
    }
    else
    {
        stats.variance = 0.0;
        stats.stddev = 0.0;
    }
}

//+------------------------------------------------------------------+
//| 🔍 Ricalcola min/max quando necessario O(n) - chiamata rara    |
//+------------------------------------------------------------------+
void RecalculateMinMax(RollingStatsOptimized &stats)
{
    if (stats.count == 0) return;
    
    stats.minValue = DBL_MAX;
    stats.maxValue = -DBL_MAX;
    
    for (int i = 0; i < stats.count; i++)
    {
        double val = stats.buffer[i];
        if (val < stats.minValue) stats.minValue = val;
        if (val > stats.maxValue) stats.maxValue = val;
    }
}

//+------------------------------------------------------------------+
//| 📈 Calcola mediana on-demand O(n log n) - solo se richiesta    |
//+------------------------------------------------------------------+
double CalculateMedian(RollingStatsOptimized &stats)
{
    if (!stats.needsMedianUpdate && stats.median != 0.0)
        return stats.median;
    
    if (stats.count == 0) return 0.0;
    
    // 📋 Copia e ordina buffer attivo
    double tempArray[];
    ArrayResize(tempArray, stats.count);
    ArrayCopy(tempArray, stats.buffer, 0, 0, stats.count);
    ArraySort(tempArray);
    
    // 🎯 Calcola mediana
    if (stats.count % 2 == 1)
    {
        // Numero dispari - elemento centrale
        stats.median = tempArray[stats.count / 2];
    }
    else
    {
        // Numero pari - media dei due centrali
        int mid = stats.count / 2;
        stats.median = (tempArray[mid - 1] + tempArray[mid]) / 2.0;
    }
    
    stats.needsMedianUpdate = false;
    return stats.median;
}

//+------------------------------------------------------------------+
//| 🛡️ Calcola MAD (Median Absolute Deviation) - statistica robusta|
//+------------------------------------------------------------------+
double CalculateMAD(RollingStatsOptimized &stats)
{
    if (stats.count == 0) return 0.0;
    
    double median = CalculateMedian(stats);
    
    // 📊 Array delle deviazioni assolute dalla mediana
    double deviations[];
    ArrayResize(deviations, stats.count);
    
    for (int i = 0; i < stats.count; i++)
    {
        deviations[i] = MathAbs(stats.buffer[i] - median);
    }
    
    // 🎯 Mediana delle deviazioni = MAD
    ArraySort(deviations);
    
    if (stats.count % 2 == 1)
    {
        stats.mad = deviations[stats.count / 2];
    }
    else
    {
        int mid = stats.count / 2;
        stats.mad = (deviations[mid - 1] + deviations[mid]) / 2.0;
    }
    
    return stats.mad;
}

//+------------------------------------------------------------------+
//| 🎯 Soglia robusta usando mediana + MAD (resistente agli outlier)|
//+------------------------------------------------------------------+
double GetRobustThreshold(RollingStatsOptimized &stats, double multiplier, double fallback)
{
    // 🛡️ Se non abbastanza campioni, usa fallback
    if (!stats.isReliable)
        return fallback;
    
    double median = CalculateMedian(stats);
    double mad = CalculateMAD(stats);
    
    // 📊 Conversione MAD -> equivalente stddev (per distribuzioni normali)
    double robustStddev = mad * 1.4826;
    
    return median + multiplier * robustStddev;
}

//+------------------------------------------------------------------+
//| 📊 Soglia normale con validazione affidabilità                  |
//+------------------------------------------------------------------+
double GetNormalThreshold(RollingStatsOptimized &stats, double multiplier, double fallback)
{
    if (!stats.isReliable)
        return fallback;
    
    return stats.mean + multiplier * stats.stddev;
}

//+------------------------------------------------------------------+
//| 📈 Calcola percentile specifico O(n log n)                     |
//+------------------------------------------------------------------+
double CalculatePercentile(RollingStatsOptimized &stats, double percentile)
{
    if (stats.count == 0 || percentile < 0 || percentile > 100)
        return 0.0;
    
    // 📋 Copia e ordina
    double tempArray[];
    ArrayResize(tempArray, stats.count);
    ArrayCopy(tempArray, stats.buffer, 0, 0, stats.count);
    ArraySort(tempArray);
    
    // 🎯 Calcola indice per percentile
    double index = (percentile / 100.0) * (stats.count - 1);
    int lowerIndex = (int)MathFloor(index);
    int upperIndex = (int)MathCeil(index);
    
    if (lowerIndex == upperIndex)
        return tempArray[lowerIndex];
    
    // 📊 Interpolazione lineare
    double weight = index - lowerIndex;
    return tempArray[lowerIndex] * (1.0 - weight) + tempArray[upperIndex] * weight;
}

//+------------------------------------------------------------------+
//| 🔧 Utility: Reset statistiche                                   |
//+------------------------------------------------------------------+
void ResetRollingStats(RollingStatsOptimized &stats)
{
    InitRollingStatsOptimized(stats, stats.maxCount, stats.minSamplesReliable);
}

//+------------------------------------------------------------------+
//| 📋 Utility: Info debug statistiche                             |
//+------------------------------------------------------------------+
void PrintStatsInfo(RollingStatsOptimized &stats, string label = "RollingStats")
{
    Print("📊 [", label, "] Count=", stats.count, "/", stats.maxCount, 
          " | Reliable=", stats.isReliable ? "✅" : "❌",
          " | Mean=", DoubleToString(stats.mean, 5),
          " | StdDev=", DoubleToString(stats.stddev, 5),
          " | Min=", DoubleToString(stats.minValue, 5),
          " | Max=", DoubleToString(stats.maxValue, 5));
}

//+------------------------------------------------------------------+
//| 🎯 Utility: Verifica se le statistiche sono stabili           |
//+------------------------------------------------------------------+
bool AreStatsStable(RollingStatsOptimized &stats, double maxCvThreshold = 0.1)
{
    if (!stats.isReliable || stats.mean == 0.0)
        return false;
    
    // Coefficiente di variazione (CV) = stddev / mean
    double cv = MathAbs(stats.stddev / stats.mean);
    return cv <= maxCvThreshold;
}

#endif // __ROLLING_STATS_OPTIMIZED_MQH__