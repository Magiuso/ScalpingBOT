//+------------------------------------------------------------------+
//| RSI_SLTP_Dynamic.mqh                                             |
//| Calcola moltiplicatori dinamici SL/TP in base a RSI + ADX        |
//+------------------------------------------------------------------+
#ifndef __RSI_SLTP_DYNAMIC_MQH__
#define __RSI_SLTP_DYNAMIC_MQH__


#include <ScalpingBot\Utility.mqh>

// === 🧩 Struttura dati di output ===
struct SLTPMultipliers
{
   double slMultiplier;     // 🔢 Moltiplicatore per SL
   double tpMultiplier;     // 🔢 Moltiplicatore per TP
   int rsiScore;            // 📊 Punteggio RSI (0–3)
   double adx;              // 📶 Valore ADX
   int trendClass;          // 🚦 1 = debole, 2 = medio, 3 = forte
   int totalScore;          // 🧮 rsiScore + trendClass
};

//+------------------------------------------------------------------+
//| 📌 Funzione principale                                           |
//| Calcola SL e TP dinamici in base a RSI score + forza ADX        |
//+------------------------------------------------------------------+
SLTPMultipliers CalcSLTPFromRSIADX(string symbol, ENUM_TIMEFRAMES tf, bool isBuy)
{
   SLTPMultipliers result;
   result.slMultiplier = SLTP_Medium_SL_Input;
   result.tpMultiplier = SLTP_Medium_TP_Input;
   result.rsiScore = 0;
   result.adx = 0;
   result.trendClass = 1;
   result.totalScore = 0;

   // === 📈 Calcolo RSI e derivata ===
   int rsiHandle = iRSI(symbol, tf, RSIPeriod, PRICE_CLOSE);
   if (rsiHandle == INVALID_HANDLE) return result;

   double rsiBuffer[];
   if (CopyBuffer(rsiHandle, 0, 1, RSICandleCount, rsiBuffer) < RSICandleCount) return result;

   double sum = 0;
   for (int i = 0; i < RSICandleCount; i++) sum += rsiBuffer[i];
   double avgRSI = sum / RSICandleCount;

   double derivata = 0;
   for (int i = 0; i < RSICandleCount - 1; i++)
       derivata += (rsiBuffer[i] - rsiBuffer[i + 1]);
   derivata /= (RSICandleCount - 1);

   // === 🧠 Calcolo punteggio RSI (0–3)
   if ((isBuy && avgRSI > 50) || (!isBuy && avgRSI < 50)) result.rsiScore++;
   if ((isBuy && derivata > 0) || (!isBuy && derivata < 0)) result.rsiScore++;
   if (MathAbs(derivata) >= RSIDerivataThreshold) result.rsiScore++;

   // === 📶 Calcolo ADX
   result.adx = CalculateADX(symbol, tf, ADXPeriodRSI);
   
   if (result.adx >= RSISLTP_ADX_Threshold2)
       result.trendClass = 3;    // 🔴 Forte
   else if (result.adx >= RSISLTP_ADX_Threshold1)
       result.trendClass = 2;    // 🟡 Medio
   else
       result.trendClass = 1;    // 🟢 Debole

   // === 🧮 Calcolo score totale
   result.totalScore = result.rsiScore + result.trendClass;

   // === ⚙️ Assegnazione moltiplicatori SL/TP in base alla forza
   if (result.totalScore <= 2) // Trend debole
   {
      result.slMultiplier = SLTP_Weak_SL_Input;
      result.tpMultiplier = SLTP_Weak_TP_Input;
   }
   else if (result.totalScore <= 4) // Trend medio
   {
      result.slMultiplier = SLTP_Medium_SL_Input;
      result.tpMultiplier = SLTP_Medium_TP_Input;
   }
   else // Trend forte
   {
      result.slMultiplier = SLTP_Strong_SL_Input;
      result.tpMultiplier = SLTP_Strong_TP_Input;
   }

   // === 🪵 Logging dettagliato
   if (EnableLogging_RSIMomentum)
   {
      PrintFormat("📊 [SLTP Dyn] RSI Score = %d | Derivata RSI | ADX = %.2f", result.rsiScore, result.adx);
      PrintFormat("📈 Trend Class = %d → Total Score = %d", result.trendClass, result.totalScore);
      PrintFormat("🎯 Moltiplicatori: SL x%.2f | TP x%.2f", result.slMultiplier, result.tpMultiplier);
   }

   return result;
}

#endif // __RSI_SLTP_DYNAMIC_MQH__
