//+------------------------------------------------------------------------+
//| MACD+EMA con TrailingStop + AutotuneObserver + AutotuneClustering      |
//|                     + Strategy Manager.mq5                             |
//|                                                                        |
//+------------------------------------------------------------------------+
#property copyright "ANTONIO BATTISTA"
#property description "Expert Advisor per lo scalping con gestione avanzata del recupero"
#property strict

#define MAP_MAX_ENTRIES       10     // Numero massimo di entries (può essere sovrascritto)
#define MAP_EVICTION_PERCENT  25     // Percentuale di entries da rimuovere quando si raggiunge il limite

// === 📦 INCLUDES LIBRERIE DI SISTEMA ===
#include <Trade\Trade.mqh>
CTrade trade;                                   // Oggetto globale per invio ordini

// === 📦 INCLUDES DEI MODULI PERSONALIZZATI ===
#include <ScalpingBot\Utility.mqh>                 // Funzioni condivise, gestione lotto, normalizzazione
#include <ScalpingBot\EMA_MACD_BodyScore.mqh>      // Logica EMA+MACD+Body Momentum
#include <ScalpingBot\Campionamento.mqh>           // Campionamento intra-candela
#include <ScalpingBot\TrailingStop.mqh>            // Gestione trailing stop dinamico
#include <ScalpingBot\ExtremesPrice.mqh>           // Aggiornamento prezzo massimo/minimo per trailing avanzato
#include <ScalpingBot\OpenTradeMACDEMA.mqh>        // Apertura ordini per strategia MACD+EAMA
#include <ScalpingBot\PostTradeEMACheck.mqh>       // Chiusura ordini in controtendenza
#include <ScalpingBot\RecoveryTriggerManager.mqh>  // Gestione trigger per apertura ordine recovery
#include <ScalpingBot\RecoveryBlock.mqh>           // Gestione validazione apertura ordini recovery
#include <ScalpingBot\OpenTradeRecovery.mqh>       // Apertura ordini di ricovero


//+------------------------------------------------------------------+
//| === ⚙️ INPUT GLOBALI - GESTIONE LOTTO E RISCHIO                 |
//+------------------------------------------------------------------+
input bool   UseAutoLot              = true;      // LOTTO: 🔁 Attiva il calcolo lotto automatico
input double RiskPercent             = 1.0;       // Lotto: 📊 Rischio per trade in percentuale
input double LotSize                 = 0.1;       // LOTTO: 📦 Lotto fisso (se UseAutoLot = false)
input double RiskPerLotUnit          = 100.0;     // LOTTO: 💶 Rischio stimato per 0.1 lotti
input double MaxAutoLot              = 0.5;       // LOTTO: 🚦 Limite massimo in modalità AUTO

//+------------------------------------------------------------------+
//| === 📈 INPUT STRATEGIA MACD+EMA                                 |
//+------------------------------------------------------------------+
// 📊 TIMEFRAME STRATEGICI
input ENUM_TIMEFRAMES Timeframe_MACD_EMA      = PERIOD_M5; // GENERALE: ⏳ Timeframe strategia MACD+EMA
input int    CooldownMACDSeconds              = 30;        // GENERALE: ⏱️ Tempo minimo tra operazioni MACD
input int    MaxOrdersPerAsset                = 5000;      // GENERALE: 🔄 Numero massimo ordini aperti per simbolo
input uint   MagicNumber_MACD                 = 123456;    // GENERALE: 🪄 Magic number strategia MACD+EMA
input bool EnableInitialCandleFilter          = true;      // GENERALE: ⏳ Abilita filtro anti-errori nei primi secondi di candela
input int InitialCandleDelaySec               = 1;         // GENERALE: ⏱️ Secondi di attesa minimi all'inizio candela

//+------------------------------------------------------------------+
//| === 📐 INPUT PREVALIDAZIONE EMA+MACD+BODY MOMENTUM               |
//+------------------------------------------------------------------+
// 📐 Parametri Trend EMA
input bool   EnableEMABlock                = true;       //EMA: ✅ Abilita blocco rilevamento trend
input double EMA_TargetMaxScore            = 10.0;       //EMA: ⚖️ Peso massimo score EMA (0–10)
input double StrongSlopePercentThreshold   = 0.08;       //EMA: 📈 Soglia pendenza forte per validazione trend(%)
input double SlopePercentThreshold         = 0.05;       //EMA: 📐 Soglia pendenza media (%)
            // 📶 Filtro ADX
input double MinADXThreshold               = 25.0;       //EMA: 📶 Soglia minima ADX per trend valido
            // 📏 Distanza & Penalità
input double MinEMADistancePercent         = 0.1;        //EMA: 📏 Distanza minima EMA5–EMA20 (%)
input double EMA_DistancePenalty           = 2.5;        //EMA: ⚠️ Penalità distanza insufficiente
            // 🧲 Convergenza
input bool   EnableEMAConvergenceCheck     = true;       //EMA: 🧲 Analisi Convergenza attiva
input bool EnableEMACrossCheck             = true;       //EMA: 🧲 Analisi Cross EMA5-EMA20 attiva
input bool EnableEMAAlignmentCheck         = true;       //EMA: 🧲 Analisi Allineamento EMA5-EMA20 attiva
            // 🐢 Trend lento costante
input double SlowTrendDeltaThreshold       = 0.002;      //EMA: 🐢 Delta massimo per trend lento
            // 💥 Spike & Bonus
input double SpikeStrengthThreshold        = 12.0;       //EMA: 💥 Soglia per definire uno spike Δ3/Δ4 (punti)
input double SpikeBonusMultiplier          = 0.2;        //EMA: 💥 Bonus spike (% peso EMA , es. 0.2 = 20%)
            // 🔄 Inversione debole
input double EMA_InversionPenalty          = 5.0;        //EMA: 🔄 Penalità inversione debole
input double InversioneThreshold           = 0.25;       //EMA: ⚠️ Soglia minima per rimbalzo rispetto al trend precedent
input double EMA_CrossPenalty              = 4.0;        //EMA: 🔄 Penalità per cross EMA contrario al trend
input double EMA_CrossBonus                = 1.0;        //EMA: ⚡ Bonus per cross EMA a favore del trend
input double EMA_AlignmentBonus            = 5.0;        //EMA: ⚡ Bonus se EMA5-EMA20 sono allineate con Trend 
            // ⚡ Bonus accelerazione
input double MaxExpectedAccelPercent       = 0.08;       //EMA: ⚡ Max % accelerazione per bonus
input double EMARecentSlopeWeight          = 0.20;       //EMA: ⚡ Bonus extra per Δ1 coerente
// 📉 Parametri MACD
input bool      EnableMACDCheck                 = true;     //MACD: ✅ Attiva il blocco MACD
input double    MACD_MaxScorePerBlock           = 10.0;     //MACD: 🎯 Punteggio massimo per MACD (per normalizzazione interna)
            // === 📈 Parametri tecnici del MACD ===
input int       MACD_FastPeriod                 = 12;       //MACD: 📈 Periodo EMA veloce per MACD
input int       MACD_SlowPeriod                 = 26;       //MACD: 📉 Periodo EMA lento per MACD
input int       MACD_SignalPeriod               = 9;        //MACD: 🔵 Periodo EMA della Signal Line per MACD
            // === 📏 Normalizzazione & soglie dinamiche (basate su ATR) ===
input double    MACD_ZeroLineThresholdRatio     = 0.05;     //MACD: 🟠 Soglia di prossimità linea zero (esaurimento trend) in rapporto all'ATR
            // === ⚠️ Bonus e Penalità principali ===
input double    MACD_CrossPenalty               = 2.0;      //MACD: 🚨 Penalità per incrocio contro-trend
input double    MACD_DirectionPenalty           = 1.0;      //MACD: ➡️ Penalità per direzione MACD incoerente con EMA
input double    MACD_ZeroLinePenalty            = 1.0;      //MACD: 📉 Penalità per MACD troppo vicino alla zero-line
input double    MACD_MinScoreThreshold          = 0.0;      //MACD: ⛔️ Limite minimo al punteggio finale del blocco
            // === 🚀 Bonus per accelerazione (slope istogramma) ===
input double    MACD_StrongPositiveSlopeThreshold = 0.005;  //MACD: 🔼 Soglia per forte pendenza istogramma per bonus massimo
input double    MACD_WeakPositiveSlopeThreshold   = 0.001;  //MACD: 🔽 Soglia per pendenza debole dell'istogramma
input double    MACD_MaxSlopeBonus               = 3.0;     //MACD: ✨ Bonus massimo per slope positiva coerente con trend
input double    MACD_MaxSlopePenalty             = 3.0;     //MACD: 🛑 Penalità massima per decelerazione dell'istogramma
            // === 💪 Bonus per distanza MACD–Signal ===
input double    MACD_MinDistanceForBonus         = 0.0005;  //MACD: 📉 Distanza minima MACD-Signal per attivare bonus
input double    MACD_MaxDistanceForBonus         = 0.0050;  //MACD: 📈 Distanza massima considerata per il bonus
input double    MACD_MaxDistanceBonus            = 2.0;     //MACD: 💪 Bonus massimo per distanza MACD-Signal
            // === 🔀 Divergenza Prezzo–MACD (opzionale, con penalità diretta) ===
input bool      MACD_EnableDivergenceCheck       = true;    // MACD: ✅ Abilita controllo divergenze
input int       MACD_DivergenceLookbackBars      = 30;      // MACD: 🔎 Barre da analizzare per identificare swing di prezzo e MACD
input int       MACD_MinBarsBetweenSwings        = 5;       // MACD: 📏 Minimo numero di barre tra due swing per considerarli validi
input double    MACD_MinDivergenceRatio          = 0.05;    // MACD: ⚠️ Soglia minima di divergenza (es. 0.05 = 5%) tra swing
input double    MACD_PriceDivergencePenalty      = 7.0;     // MACD: ❌ Penalità applicata se viene rilevata divergenza Prezzo–Istogramma
// 📉 Parametri BODY MOMENTUM 
input bool      EnableBodyMomentumScore         = true;     // BODY: ✅ Attiva o disattiva il blocco avanzato BodyMomentum
input int       NumCandlesBodyAnalysis          = 3;        // BODY: 🕯️ Numero di candele da analizzare per il punteggio
            // === 📐 Soglie dinamiche basate su ATR ===
input double    BodyMomentum_VolumeConfirmationThreshold = 1.2;     // BODY: 📦 Volume attuale > 1.2x volume medio ⇒ bonus
input double    BodyMomentum_BodyATRRatioThreshold       = 0.05;    // BODY: 📏 Corpo medio < 5% ATR ⇒ penalità
input double    BodyMomentum_StdDevATRRatioThreshold     = 0.02;    // BODY: 📊 Deviazione std. corpo < 2% ATR ⇒ penalità
            // === ⚖️ Pesi dei criteri interni (somma ≈ 10) ===
input double    BodyMomentum_WeightCoherence    = 2.0;      // BODY: 🤝 Coerenza direzione (candele rialziste/ribassiste)
input double    BodyMomentum_WeightDominance    = 2.0;      // BODY: 💪 Dominanza corpo sul range
input double    BodyMomentum_WeightShadow       = 1.0;      // BODY: ☁️ Ombre asimmetriche favorevoli
input double    BodyMomentum_WeightProgression  = 1.0;      // BODY: 📈 Progressione coerente delle chiusure
input double    BodyMomentum_WeightClose        = 1.0;      // BODY: 🎯 Chiusura nella zona forte del range
input double    BodyMomentum_WeightVolume       = 1.0;      // BODY: 📦 Volume coerente in crescita o anomalo
            // === ⚠️ Penalità fallback (non ATR-based) ===
input bool      UseMinAvgBodyCheck              = true;     // BODY: 🧮 Attiva controllo su corpo medio minimo
input double    MinAvgBodyPoints                = 4.0;      // BODY: 🔻 Corpo medio < 4.0 punti ⇒ penalità
input bool      UseMinStdDevBodyCheck           = true;     // BODY: 📊 Attiva controllo su varianza dei corpi
input double    MinStdDevPercent                = 0.20;     // BODY: 📉 Dev. Std. < 20% ⇒ penalità
            // === 🌗 Parametri per analisi delle ombre  ===
input double    BODY_MarubozuMinBodyATRRatio        = 0.6;   // BODY: ☀️ Corpo > 60% ATR ⇒ candela candidata Marubozu
input double    BODY_MarubozuShadowRatioThreshold   = 0.1;   // BODY: ☀️ Ombre totali < 10% range ⇒ Marubozu
input double    BODY_MarubozuBonus                  = 1.5;   // BODY: ✅ Bonus per candela Marubozu
input double    BODY_RejectionShadowRatioThreshold  = 0.5;   // BODY: ⚠️ Ombra singola > 50% range ⇒ rifiuto
input double    BODY_RejectionShadowPenalty         = 1.5;   // BODY: ❌ Penalità per ombra di rifiuto contro il trend
input double    BODY_FavorableShadowRatioThreshold  = 0.25;  // BODY: 🌙 Ombra sfavorevole < 25% range ⇒ bonus
input double    BODY_FavorableShadowBonus           = 0.75;  // BODY: ✅ Bonus per ombra favorevole asimmetrica
// 📊 Parametri comuni
input int ATRPeriod                                = 14;         // EMA+MACD+BODY: 📏 Periodo ATR per calcolo volatilità (default 14)
input int ADXPeriod                                = 14;         // EMA+MACD+BODY: 🔢 Periodo per il calcolo ADX (default 14))

// 🎛️Pesi blocco EMA_MACD_BodyScore    
input bool   NormalizePrevalidationScores          = true;       // EMA_MACD_BODY: 🔄 Normalizza score su scala pesata (es. max 10.0)
input double EMA_Weight                            = 4.0;        // EMA_MACD_BODY: 📐 Peso massimo blocco EMA
input double MACD_Weight                           = 3.0;        // EMA_MACD_BODY: 📊 Peso massimo blocco MACD
input double BodyMomentum_Weight                   = 3.0;        // EMA_MACD_BODY: 💪 Peso massimo blocco BodyMomentum
input double MinPrevalidationScore                 = 3.0;        // EMA_MACD_BODY: ✅ Score minimo per segnale valido

//+------------------------------------------------------------------+
//| === 📊 INPUT CAMPIONAMENTO INTRA-CANDELA                        |
//+------------------------------------------------------------------+
input int       CampionamentoMinUpdateTime  = 100;             // CAMPIONAMENTO: ⏱️ Frequenza minima di aggiornamento dello score (ms)
            // --- Fattori dello Score Intracandela ---
input double    CampionamentoWeightPriceSpeed       = 10.0;    // CAMPIONAMENTO: ⚡ Peso Score: Velocità del prezzo
input double    CampionamentoMinPriceSpeed          = 2.0;     // CAMPIONAMENTO: ⚡ Soglia minima velocità per bonus (punti)
input double    CampionamentoMaxPriceSpeed          = 20.0;    // CAMPIONAMENTO: ⚡ Soglia massima velocità per bonus (punti)
input double    CampionamentoVolumeFactor           = 2.0;     // CAMPIONAMENTO:  📊 Fattore di influenza del volume
input double    CampionamentoVolumeThreshold        = 100.0;   // CAMPIONAMENTO: 📊 Soglia volume minimo (0 = disabilita)
            // --- ADX ---
input int       CampionamentoADXPeriod              = 14;      // CAMPIONAMENTO:  📶 Periodo ADX per il calcolo
input double    CampionamentoADXDerivativeWeight    = 2.0;     // CAMPIONAMENTO: 🚀 Peso Score: Derivata ADX (bonus/malus max)
input double    CampionamentoADXDerivativeThreshold = 2.0;     // CAMPIONAMENTO:  🚀 Soglia derivata ADX per attivare bonus/malus
            // --- Rottura Soglie Storiche ---
input double    CampionamentoThresholdBreakWeight   = 5.0;     // CAMPIONAMENTO:  📈 Peso Score: Rottura High/Low storici
input int       CampionamentoThresholdLookback      = 20;      // CAMPIONAMENTO:  🕰️ Candele precedenti da considerare per High/Low
            // --- Ombre delle Candele ---
input double    CampionamentoShadowWeight           = 3.0;     // CAMPIONAMENTO:  👻 Peso Score: Analisi delle ombre
input double    CampionamentoShadowRatioThreshold   = 0.4;     // CAMPIONAMENTO:  👻 % Minima di ombra per penalizzare (es. 0.3 = 30% della candela)
            // --- Analisi Storica (Ultime 20 Candele) ---
input double    CampionamentoHistoricalAnalysisWeight = 5.0;   // CAMPIONAMENTO:  📜 Peso Score: Coerenza storica
input int       CampionamentoConsecutiveCandlesMin  = 3;       // CAMPIONAMENTO:  📊 Candele consecutive minime per trend forte
input double    CampionamentoBodyRangeThreshold     = 0.5;     // CAMPIONAMENTO:  📏 Soglia Minima Body/Range per penalizzare candele deboli
            // --- Parametri per logica di inversione
input double CampionamentoReversalMaxBodyRatio      = 0.40;    // CAMPIONAMENTO: 📏 Max body / range (es. 0.30 = 30%) → Candela "piccola"
input double CampionamentoReversalMinShadowRatio    = 0.50;    // CAMPIONAMENTO: 🕯️ Min shadow / range (es. 0.60 = 60%) → Lunga ombra per reversal
input int    CampionamentoReversalTrendLookback     = 5;       // CAMPIONAMENTO: 🔄 Candele da analizzare per trend precedente (es. ultime 3)
input double CampionamentoReversalMinTrendPercent   = 0.60;    // CAMPIONAMENTO: 🎯 Min % di candele in trend coerente (es. 0.70 = 70%)
input double CampionamentoReversalBreakThresholdPoints = 15.0; // CAMPIONAMENTO: ⚡️ Soglia in punti per rilevare reversal/breakout (es. 15.0 punti)
            // --- Scoring Finale ---
input double    CampionamentoScoreNormalizationDivisor = 3.0;  // CAMPIONAMENTO:  🔢 Divisore per normalizzare lo score (es. 3.0 per 0-10)

//+------------------------------------------------------------------+
//| === 📐 INPUT PREVALIDAZIONE RSI MOMENTUM + SL/TP DINAMICI        |
//+------------------------------------------------------------------+
// ⚙️ Attivazione singoli blocchi                             
input bool UseRSIOnCurrentBar             = false;   // RSI: ⏱️ Usa candela attuale (true) o chiusa (false)
input int RSIMinIntervalSeconds           = 0;       // RSI: ⌛ Ritardo minimo aggiornamenti RSI (0 = nessun limite)
// 🧮 Configurazione RSI                                           
input int     RSIPeriod                   = 14;      // RSI: 📈 Periodo RSI
input int     RSICandleCount              = 3;       // RSI: 🔁 Numero candele RSI per media e derivata
input double  RSIDerivataThreshold        = 0.15;    // RSI: 🔼 Soglia minima derivata RSI per conferma trend
input int     MinRSIMomentumScore         = 2;       // RSI: ✅ Punteggio minimo RSI richiesto (max 3)
// 📶 CONFIGURAZIONE ADX                                           
input int     ADXPeriodRSI                = 14;      // RSI: 📶 Periodo ADX per filtro RSI momentum
// 🧠 Soglie classificazione trend per SL/TP   
input double  ADXConfirmThreshold         = 25.0;    // RSI: 🚦 Soglia ADX per trend "debole" (validazione RSI)                 
input double  RSISLTP_ADX_Threshold1      = 30.0;    // RSI: 📊 Soglia ADX per trend "medio"
input double  RSISLTP_ADX_Threshold2      = 35.0;    // RSI: 🚀 Soglia ADX per trend "forte"

// 🎯 Moltiplicatori SL/TP Dinamici                               
input double  SLTP_Weak_SL_Input          = 6.85;    // RSI: 🛑 SL moltiplicatore in trend "debole"
input double  SLTP_Weak_TP_Input          = 4.30;    // RSI: 🎯 TP moltiplicatore in trend "debole"
input double  SLTP_Medium_SL_Input        = 9.55;    // RSI: 🟡 SL moltiplicatore in trend "medio"
input double  SLTP_Medium_TP_Input        = 5.80;    // RSI: 🎯 TP moltiplicatore in trend "medio"
input double  SLTP_Strong_SL_Input        = 2.95;    // RSI: 🔴 SL moltiplicatore in trend "forte"
input double  SLTP_Strong_TP_Input        = 4.75;    // RSI: 🎯 TP moltiplicatore in trend "forte"

//+------------------------------------------------------------------+
//| 📉 INPUT PREVALIDAZIONE MICRO TREND SCANNER                      |
//+------------------------------------------------------------------+
input ENUM_MicroTrendMode MicroTrendMode = MICROTREND_SOFT;  // MICROTREND: 🧭 Modalità MicroTrend (SOFT / AGGRESSIVE)
            // 🔘 Abilitazione componenti
input bool   Enable_EMASlope         = true;   // MICROTREND: 📈 Abilita analisi EMA Slope
input bool   Enable_Breakout         = true;   // MICROTREND: 🔓 Abilita rilevamento Breakout
input bool   Enable_Momentum         = true;   // MICROTREND: ⚡️ Abilita analisi Momentum
input bool   Enable_ADX              = true;   // MICROTREND: 📊 Abilita indicatore ADX
input bool   Enable_Volatility       = true;   // MICROTREND: 🌪️ Abilita analisi Volatilità (ATR)
input bool   Enable_Volume           = true;   // MICROTREND: 📉 Abilita analisi Volume
input bool   Enable_SpreadImpact     = true;   // MICROTREND: 🧮 Abilita valutazione Spread
            // 🎚️ Soglie minime per la pendenza dell'EMA (slopePercent)
input double Min_Slope_Threshold_M5_or_Less  = 0.0005;   // MICROTREND: ⏱️ TF ≤ M5 → Soglia pendenza EMA (0.05%)
input double Min_Slope_Threshold_M10_or_More = 0.0010;   // MICROTREND: ⏱️ TF ≥ M10 → Soglia pendenza EMA (0.10%)

            // 📏 Periodi ATR
input int    ATR_Period1             = 7;      // MICROTREND: 📍 ATR Periodo 1 (corto)
input int    ATR_Period2             = 14;     // MICROTREND: 📍 ATR Periodo 2 (medio)
input int    ATR_Period3             = 21;     // MICROTREND: 📍 ATR Periodo 3 (lungo)
            // ⚖️ Pesi componenti modalità SOFT (Totale Max = 30)
input double Weight_EMASlope_Soft     = 4.0;   // MICROTREND:  📈 Peso EMA Slope (SOFT)
input double Weight_Breakout_Soft     = 5.0;   // MICROTREND:  🔓 Peso Breakout (SOFT)
input double Weight_Momentum_Soft     = 4.0;   // MICROTREND: ⚡️ Peso Momentum (SOFT)
input double Weight_ADX_Soft          = 6.0;   // MICROTREND: 📊 Peso ADX (SOFT)
input double Weight_Volatility_Soft   = 3.0;   // MICROTREND: 🌪️ Peso Volatilità (SOFT)
input double Weight_Volume_Soft       = 4.0;   // MICROTREND: 📉 Peso Volume (SOFT)
input double Weight_SpreadImpact_Soft = 4.0;   // MICROTREND: 🧮 Peso Spread (SOFT)
            // ⚖️ Pesi componenti modalità AGGRESSIVE (Totale Max = 30)
input double Weight_EMASlope_Aggr     = 4.5;   // MICROTREND: 📈 Peso EMA Slope (AGGR)
input double Weight_Breakout_Aggr     = 4.5;   // MICROTREND: 🔓 Peso Breakout (AGGR)
input double Weight_Momentum_Aggr     = 4.5;   // MICROTREND: ⚡️ Peso Momentum (AGGR)
input double Weight_ADX_Aggr          = 4.0;   // MICROTREND: 📊 Peso ADX (AGGR)
input double Weight_Volatility_Aggr   = 4.0;   // MICROTREND: 🌪️ Peso Volatilità (AGGR)
input double Weight_Volume_Aggr       = 4.5;   // MICROTREND: 📉 Peso Volume (AGGR)
input double Weight_SpreadImpact_Aggr = 4.0;   // MICROTREND: 🧮 Peso Spread (AGGR)
            // 🧪 Parametri specifici dei componenti
input int    EMA_Period_Base          = 4;     // MICROTREND: 📈 Periodo EMA base (adattato per TF)
input int    ADX_Period_Base          = 7;     // MICROTREND: 📊 Periodo ADX base
input int    Breakout_Lookback        = 20;    // MICROTREND: 🔓 Barre lookback per breakout
input double Momentum_BodyRatio       = 1.0;   // MICROTREND: ⚡️ Rapporto minimo body/range
input double Volume_Threshold         = 1.5;   // MICROTREND: 📉 Moltiplicatore volume vs media
input double Spread_ATR_MaxRatio      = 0.4;   // MICROTREND: 🔻 Max (Spread / ATR) → Default 0.20 = 20%
input double Spread_MaxPoints         = 5.0;   // MICROTREND: 🧮 Max spread accettabile in punti
            // 🧰 Filtri qualità segnale
input double Min_TotalScore           = 10.0;  // MICROTREND: ✅ Score minimo per segnale valido
input bool   Filter_LowVolume         = true;  // MICROTREND: 🔍 Filtra periodi a basso volume
input bool   Diagnostic_Mode          = true;  // MICROTREND: 🛠️ Modalità diagnostica dettagliata
input double Direction_Tolerance      = 0.15;  // MICROTREND: 🎯 Tolleranza direzione (15%)
input double Min_Confidence           = 30.0;  // MICROTREND: ✅ Confidenza minima richiesta (%)
input double Lateral_Market_ADX       = 15.0;  // MICROTREND: 🔄 Soglia ADX per mercato laterale

//+------------------------------------------------------------------+
//| 📊 INPUT PREVALIDAZIONE SPIKE DETECTION DINAMICO                 |
//+------------------------------------------------------------------+
input int    SpikeBufferSize               = 100;      // SPIKE: 🧠 Dimensione buffer rolling (valori storici)
input double SpikeDeviationMultiplier      = 1.5;      // SPIKE: 📈 Moltiplicatore soglia dinamica (media + X * stddev)
input double SpikeExtremityPercent         = 20.0;     // SPIKE: 🎯 Percentuale distanza chiusura da estremi

//+------------------------------------------------------------------+
//| === 🧠 INPUT ENTRY MANAGER                                       |
//+------------------------------------------------------------------+
// 🎛️ Attivazione moduli principali
input bool EnableEntryManager         = true;     // 🧠 ENTRY MANAGER: Attiva il modulo Entry manager
input bool EnableEMAMACDBody          = true;     // 🧠 ENTRY MANAGER: Attiva il modulo EMA + MACD + BodyMomentum
input bool EnableCampionamentoModule  = true;     // 🧠 ENTRY MANAGER: Attiva il modulo di Campionamento avanzato
input bool EnableMicroTrendModule     = true;     // 🧠 ENTRY MANAGER: Attiva il modulo Microtrend
input bool EnableRSIMomentumModule    = true;     // 🧠 ENTRY MANAGER: Attiva il modulo RSI + ADX
input bool EnableSpikeDetection       = true;     // 🧠 ENTRY MANAGER: Attiva il modulo Spike detection
// ⚖️ Pesi (devono sommare a 100)
input double WeightEMA_MACD_Body      = 20.0;     // ⚖️ ENTRY MANAGER:  Peso EMA+MACD+BodyMomentum
input double WeightCampionamento      = 20.0;     // ⚖️ ENTRY MANAGER:  Peso Campionamento
input double WeightMicroTrend         = 20.0;     // ⚖️ ENTRY MANAGER:  Peso MicroTrend
input double WeightRSIMomentum        = 20.0;     // ⚖️ ENTRY MANAGER:  Peso RSI Momentum
input double WeightSpikeDetection     = 20.0;     // ⚖️ ENTRY MANAGER:  Peso Spike Detection
// 🎯 Soglia finale
input double EntryThreshold           = 60.0;     // 🎯 ENTRY MANAGER: Soglia minima su 100 per confermare l'apertura

//+------------------------------------------------------------------+
//| === 🛡️ INPUT TRAILING STOP DINAMICO                             |
//+------------------------------------------------------------------+
input bool   EnableTrailingStop               = false;       // TRAILING STOP: ✅ Attiva/Disattiva Trailing Stop
input int    MinSLBufferPips                  = 3;           // TRAILING STOP: ➕ Buffer minimo oltre lo StopLevel del broker (in pips)
input double TrailingStart                    = 200;         // TRAILING STOP: 🚀 Distanza(in pips o points) per Trailing start ON
input double TrailingStep                     = 100;         // TRAILING STOP: 📈 Distanza di movimento prezzo per aggiornare SL
input double MinMovementForUnlockPips         = 5;           // TRAILING STOP: ✅ Soglia micro-movimento per sblocco antiflood (in pips)
input double LogCooldownPipsThreshold         = 5;           // TRAILING STOP: ✅ Soglia distanza per log distanza TrailingStart (in pips)
input int    CooldownModifySec                = 10;          // TRAILING STOP: ⏲️ Cooldown tra modifiche SL successive
input int    ATR_FallbackPeriod               = 14;          // TRAILING STOP: 📊 Periodo ATR usato come fallback se StopLevel è nullo
input double ATR_FallbackRatio                = 0.25;        // TRAILING STOP: ⚠️ Percentuale ATR da usare come StopLevel se non definito

//+------------------------------------------------------------------+
//| 📋 INPUT → Modulo PostTrade EMA Reversal Monitor                |
//+------------------------------------------------------------------+
      // 🔧 POST_TRADE: Abilita/disabilita modulo di uscita su reversal
input bool    EnablePostTradeEMAExit       = true;      // POST_TRADE: ⚙️ Attiva modulo recovery post-trade
      // ⚙️ POST_TRADE: Parametri base EMA
input int     FastEMAPeriod                = 3;         // POST_TRADE: 📈 Periodo EMA veloce per calcolo accelerazione
      // 🎯 POST_TRADE: Soglia accelerazione (due modalità)
input bool    UseAdaptiveThreshold         = false;     // POST_TRADE: 🔁 Usa soglia adattiva basata su ATR
input double  EMA3AccelerationThreshold    = 0.2;       // POST_TRADE: 🎯 Soglia fissa di accelerazione EMA3
input double  EMA3_ATR_Multiplier          = 0.75;      // POST_TRADE: 📐 Moltiplicatore ATR per soglia adattiva
      // 📊 POST_TRADE: Sistema conferma RSI
input bool    EnableRSIConfirmation        = true;      // POST_TRADE: ✅ Abilita conferma direzione tramite RSI
input double  RSIBearishLevel              = 65.0;      // POST_TRADE: 📉 RSI sopra questo valore → bias ribassista
input double  RSIBullishLevel              = 35.0;      // POST_TRADE: 📈 RSI sotto questo valore → bias rialzista
input double  RSIDerivativeThreshold       = 0.5;       // POST_TRADE: 🧮 Derivata RSI minima per conferma momentum
      // 🛡️ POST_TRADE: Filtro anti-lateralità
input bool    EnableAntiRangingFilter      = true;      // POST_TRADE: 🛡️ Attiva filtro anti-ranging
input double  MinATRThreshold              = 1.5;       // POST_TRADE: 🔽 ATR minimo (moltiplicatore) per trend valido
input double  MaxEMAProximity              = 0.3;       // POST_TRADE: 📏 Max distanza % tra EMA9 e EMA21 (sotto = ranging)
      // ⏰ POST_TRADE: Sistema conferma ritardata
input bool    EnableDelayedConfirmation    = true;      // POST_TRADE: ⏳ Attendi 1 candela prima di confermare
input bool    RequireSignalPersistence     = true;      // POST_TRADE: 🔁 Richiedi persistenza del segnale nel tempo
      // 🎯 POST_TRADE: Sistema punteggio di confidenza
input bool    UseConfidenceScoring         = true;      // POST_TRADE: 🧠 Abilita scoring di qualità segnale
input double  MinConfidenceScore           = 70.0;      // POST_TRADE: ✅ Soglia minima per considerare valido il punteggio
      // ⚖️ POST_TRADE: Pesi per il calcolo punteggio (somma teorica = 100)
input double  EMA3_Weight                  = 40.0;      // POST_TRADE: 📊 Peso per accelerazione EMA3
input double  RSI_Weight                   = 30.0;      // POST_TRADE: 📊 Peso per RSI e derivata
input double  ATR_Bonus_Weight             = 10.0;      // POST_TRADE: 🎁 Bonus punteggio per ATR elevato
input double  Ranging_Penalty              = 20.0;      // POST_TRADE: ❌ Penalità se il mercato è in ranging
      // 🎛️ POST_TRADE: Modalità calcolo punteggio
input bool    UseAdvancedScoring           = false;     // POST_TRADE: 🔬 Usa formula logaritmica avanzata
input double  ScoreMultiplier              = 1.0;       // POST_TRADE: 🎚️ Moltiplicatore finale del punteggio (0.5–2.0)
      // 📝 POST_TRADE: Logging e debug
input bool    EnableDebugMode              = false;     // POST_TRADE: 🐞 Abilita log dettagliato per debug

//+------------------------------------------------------------------+
//| 📋 INPUT → Modulo RecoveryBlock                                  |
//+------------------------------------------------------------------+
input bool EnableRecoveryBlock                = true;         // RECOVERY: 🔄 Abilita/Disabilita il modulo di recupero (Recovery Block)
input int    MaxRecoveryAttempts              = 5;            // RECOVERY: ❗ Numero massimo di tentativi di recupero per ciascun trigger
input double RecoveryLotMultiplier            = 1.5;          // RECOVERY: 📈 Moltiplicatore del lotto per il trade di recupero
input int  RecoveryMagicNumber                = 7890;         // RECOVERY: 🪄 Magic number per operazioni di recupero
CRecoveryTriggerManager g_recovery_manager;                   // RECOVERY: 🌟 Gestore centrale dei segnali di recupero. Instanza globale unica per tutto l'EA.

//+------------------------------------------------------------------+
//| === 📝 INPUT DI LOG PER OGNI MODULO                              |
//+------------------------------------------------------------------+
input bool EnableLogging_Madre               = true;   // 📋 Log generale codice Madre
input bool EnableLogging_EntryManager        = false;  // 📝 Log dettagliati EntryManager
input bool EnableLogging_EMA_MACD_Body       = false;  // 🔍 Log dettagliati EMA_MACD_BodyScore
input bool EnableLogging_EMA                 = false;  // 🔍 Log dettagliati EMA
input bool EnableLogging_MACD                = false;  // 🔍 Log dettagliati MACD
input bool EnableLogging_BodyMomentum        = true;   // 📝 Log dettagliati BodyMomentum
input bool EnableLogging_Campionamento       = true;   // 📈 Log dettagliati Campionamento intra-candela
input bool EnableLogging_MicroTrend          = false;  // 📊 Log dettagliati MicroTrend
input bool EnableLogging_RSIMomentum         = true;   // 🔍 Log dettagliati RSI Momentum
input bool EnableLogging_RSIUpdateDelay      = false;  // 🔍 Log se RSI salta aggiornamento per filtro temporale
input bool EnableLogging_SpikeDetection      = true;   // 📝 Log dettagliati Spike detection
input bool EnableLogging_Trailing            = true;   // 🔁 Log dettagliati Trailing stop
input bool EnableLogging_PostTradeCheck      = true;   // 🔍 Log dettagliati Post Trade Check
input bool EnableLogging_OpenTrade           = true;   // 🚀 Log dettagliati Apertura trade
input bool EnableLogging_OpenTradeRecovery   = true;   // 📝 Log dettagliati Apertura trade di recupero
input bool EnableLogging_RecoveryBlock       = true;   // 📝 Log dettagliati Recovery Block
input bool EnableLogging_ExtremesPrice       = true;   // 📈 Log dettagliati Aggiornamento Peak/Worse Price
input bool EnableLogging_Utility             = true;   // 🧰 Log funzioni Utility

//+------------------------------------------------------------------+
//| === 🌐 VARIABILI DI STATO GLOBALI                               |
//+------------------------------------------------------------------+
bool isBuySignal                       = true;    // 📍 Direzione segnale corrente (true = BUY, false = SELL)
bool isCampionamentoAttivo             = false;   // 🟡 Campionamento attivo
bool isCampionamentoCompleto           = false;   // 🟢 Campionamento completato
bool isSegnaleCampionamentoValido      = false;   // ✅ Risultato del campionamento

datetime lastMACDTradeTime       = 0;       // 🕓 Timestamp ultimo trade MACD

//---------------------------------------------------------------------------------------------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| ✅ OnInit: inizializzazione EA                                  |
//+------------------------------------------------------------------+
int OnInit()
{
    EventSetTimer(1); // ⏱️ Esecuzione OnTimer ogni secondo   

    if (EnableLogging_Madre)
        Print("✅ ScalpingBot inizializzato correttamente. Timer attivo.");
    
    if (EnableLogging_Madre)
        Print("✅ ScalpingBot inizializzato correttamente. Timer attivo.");
        
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| 🛑 OnDeinit: chiusura EA                                        |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("🔄 OnDeinit → Pulizia handle indicatori...");

    // 🧼 Rilascio cache RSI
    rsiCache.ReleaseAll();
    Print("🗑️ Handle RSI rilasciati");

    // 🧼 Rilascio cache EMA
    emaCache.ReleaseAll();
    Print("🗑️ Handle EMA rilasciati");

    // 🧼 Rilascio cache MACD
    macdCache.ReleaseAll();
    Print("🗑️ Handle MACD rilasciati");

    // 🧼 Rilascio cache ATR + ADX
    indicatorCache.ReleaseAll();
    Print("🗑️ Handle ATR/ADX rilasciati");

    // 🧹 Pulizia RSI HashMap
    CleanupRSIMomentum();  // <-- ✅ Aggiunta importante
    Print("🧹 HashMap RSI pulita");

    // 🧹 Trigger recovery
    g_recovery_manager.Clear();
    Print("🗑️ Trigger Recovery cancellati");
    
    if (EnableLogging_Campionamento)
        Print("🧹 [Campionamento] Cleanup completato - Tutti gli stati deallocati");

    Print("✅ [OnDeinit] Tutti gli handle sono stati rilasciati correttamente.");

    EventKillTimer(); // 🛑 Ferma il timer

    if (EnableLogging_Madre)
        Print("🛑 ScalpingBot terminato correttamente. Timer disattivato.");
}

//+------------------------------------------------------------------+
//| ⏱️ OnTimer: cuore operativo (chiamato ogni secondo)              |
//+------------------------------------------------------------------+
void OnTimer()
{
    // 🎯 2. Invio tentativo ordine se segnale è pronto
    OpenTrade(_Symbol);
    
    // Rimuovi entries non accedute da più di 2 ore
    M_CampionamentoState.CleanupOldEntries(2);
}

//+------------------------------------------------------------------+
//| 📉 OnTick: aggiornamento realtime (solo peakPrice, se usato)     |
//+------------------------------------------------------------------+
void OnTick()
{
    AggiornaExtremesPrice();
    RefreshTrackedOrders();
    OnTickCampionamento_Global();

    if (EnableTrailingStop)
        GestioneTrailingStop();
    
    // 🛠️ Esecuzione dei segnali di recupero
    ExecuteRecoveryTrades();
    
    //Monitoraggio Intra-Trades
    CheckPostTradeConditions();
}
