//+------------------------------------------------------------------+
//|                                                   TestBridge.mq5 |
//|                      EA per sistema BUFFER + PERIODIC WRITE     |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024"
#property version   "4.00"
#property description "EA per testare AnalyzerBridge con sistema BUFFER + PERIODIC WRITE"

#include <ScalpingBot\AnalyzerBridge.mqh>

//+------------------------------------------------------------------+
//| Parametri di input                                              |
//+------------------------------------------------------------------+
input group "=== BUFFER + PERIODIC WRITE CONFIGURATION ==="
input int    WriteInterval = 10;           // Scrivi file ogni N secondi
input int    BufferMaxSize = 100000;       // Dimensione max buffer (caratteri)
input int    BufferMaxLines = 1000;        // Numero max linee nel buffer
input bool   AppendMode = true;            // Modalità append (false = overwrite)

input group "=== GENERAL SETTINGS ==="
input int    SendIntervalMs = 100;         // Intervallo invio tick (millisecondi)
input int    HistoricalBufferSize = 1000;  // Dimensione buffer storico
input bool   DebugMode = true;             // Modalità debug
input bool   EnableTimer = true;           // Abilita timer per scrittura automatica

input group "=== STATUS MONITORING ==="
input int    StatusPrintInterval = 300;    // Stampa status ogni N secondi (0 = disabilitato)
input bool   ForceWriteOnStart = false;    // Forza scrittura all'avvio

//+------------------------------------------------------------------+
//| Variabili globali                                               |
//+------------------------------------------------------------------+
CAnalyzerBridge* bridge;
datetime last_status_print = 0;
int timer_interval = 1;  // Timer ogni 1 secondo per controlli

//+------------------------------------------------------------------+
//| Inizializzazione EA                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== INIZIALIZZAZIONE TestBridge EA v4.00 ===");
   Print("Symbol: ", _Symbol);
   Print("Timeframe: ", EnumToString(_Period));
   
   // Crea istanza bridge
   bridge = new CAnalyzerBridge();
   
   if(bridge == NULL)
   {
      Print("ERRORE CRITICO: Impossibile creare istanza CAnalyzerBridge");
      return INIT_FAILED;
   }
   
   // Configura parametri BUFFER + PERIODIC WRITE prima dell'inizializzazione
   bridge.SetWriteInterval(WriteInterval);
   bridge.SetBufferMaxSize(BufferMaxSize);
   bridge.SetBufferMaxLines(BufferMaxLines);
   bridge.SetAppendMode(AppendMode);
   
   // Configura parametri generali
   bridge.SetSendInterval(SendIntervalMs);
   bridge.SetDebugMode(DebugMode);
   bridge.SetBufferSize(HistoricalBufferSize);
   
   // Inizializza il bridge
   if(!bridge.Initialize(_Symbol, _Period))
   {
      Print("ERRORE: Bridge non inizializzato per ", _Symbol);
      delete bridge;
      bridge = NULL;
      return INIT_FAILED;
   }
   
   // Scrittura iniziale se richiesta
   if(ForceWriteOnStart)
   {
      bridge.ForceWrite();
      Print("✅ Scrittura iniziale eseguita");
   }
   
   // Imposta timer se abilitato
   if(EnableTimer)
   {
      if(!EventSetTimer(timer_interval))
      {
         Print("ATTENZIONE: Impossibile impostare timer. Scrittura automatica disabilitata.");
      }
      else
      {
         Print("⏰ Timer impostato: ", timer_interval, " secondo(i)");
      }
   }
   
   // Stampa configurazione finale
   Print("📋 CONFIGURAZIONE BUFFER + PERIODIC WRITE:");
   Print("   - Scrivi file ogni: ", WriteInterval, " secondi");
   Print("   - Buffer max size: ", BufferMaxSize, " caratteri");
   Print("   - Buffer max lines: ", BufferMaxLines, " linee");
   Print("   - Append mode: ", AppendMode ? "ON" : "OFF");
   Print("   - Send interval: ", SendIntervalMs, " ms");
   Print("   - Historical buffer: ", HistoricalBufferSize, " punti");
   Print("   - Debug mode: ", DebugMode ? "ON" : "OFF");
   Print("   - Timer enabled: ", EnableTimer ? "ON" : "OFF");
   
   // Stampa status iniziale
   last_status_print = TimeCurrent();
   bridge.PrintStatus();
   
   Print("🚀 TestBridge EA inizializzato con successo!");
   Print("📁 FILE DA LEGGERE CON PYTHON: ", bridge.GetOutputFile());
   Print("🎯 Il file è SEMPRE LEGGIBILE tra le scritture!");
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Gestione tick                                                   |
//+------------------------------------------------------------------+
void OnTick()
{
   if(bridge == NULL || !bridge.IsActive())
      return;
   
   // Processa il tick
   bridge.OnTick();
   
   // Stampa status periodico se configurato
   if(StatusPrintInterval > 0)
   {
      datetime now = TimeCurrent();
      if(now - last_status_print >= StatusPrintInterval)
      {
         Print("📊 STATUS PERIODICO:");
         bridge.PrintStatus();
         last_status_print = now;
      }
   }
}

//+------------------------------------------------------------------+
//| Gestione timer per scrittura automatica                         |
//+------------------------------------------------------------------+
void OnTimer()
{
   if(bridge == NULL || !bridge.IsActive())
      return;
   
   // Chiama il timer del bridge per gestire scrittura automatica
   bridge.OnTimer();
   
   // Log periodico leggero ogni 60 secondi
   static datetime last_timer_log = 0;
   datetime now = TimeCurrent();
   
   if(DebugMode && (now - last_timer_log >= 60))
   {
      Print("⏱️ Timer check - Ticks: ", bridge.GetTickCount(), 
            " | Packets: ", bridge.GetPacketsSent(), 
            " | Writes: ", bridge.GetTotalWrites(),
            " | Buffer: ", bridge.GetBufferLines(), " lines");
      last_timer_log = now;
   }
}

//+------------------------------------------------------------------+
//| Deinitializzazione EA                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("=== DEINITIALIZZAZIONE TestBridge EA ===");
   Print("Motivo: ", GetDeinitReasonText(reason));
   
   // Disabilita timer
   if(EnableTimer)
      EventKillTimer();
   
   if(bridge != NULL)
   {
      // Stampa status finale
      Print("📊 STATUS FINALE:");
      bridge.PrintStatus();
      
      // Forza scrittura finale
      bridge.ForceWrite();
      Print("💾 Scrittura finale eseguita");
      
      // Shutdown del bridge
      bridge.Shutdown();
      
      // Elimina istanza
      delete bridge;
      bridge = NULL;
      
      Print("✅ Bridge eliminato correttamente");
   }
   
   Print("👋 TestBridge EA terminato");
}

//+------------------------------------------------------------------+
//| Gestione eventi chart                                           |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long& lparam, const double& dparam, const string& sparam)
{
   if(bridge == NULL)
      return;
   
   // Hotkeys per controllo manuale
   if(id == CHARTEVENT_KEYDOWN)
   {
      switch((int)lparam)
      {
         case 83: // Tasto 'S' - Status
            Print("🔍 STATUS MANUALE (Tasto S):");
            bridge.PrintStatus();
            break;
            
         case 87: // Tasto 'W' - Force Write
            if(bridge.ForceWrite())
            {
               Print("💾 SCRITTURA MANUALE eseguita (Tasto W)");
               Print("📁 File aggiornato: ", bridge.GetOutputFile());
            }
            else
            {
               Print("❌ SCRITTURA MANUALE fallita");
            }
            break;
            
         case 67: // Tasto 'C' - Clear/Reset
            bridge.Reset();
            Print("🔄 RESET MANUALE eseguito (Tasto C)");
            break;
            
         case 72: // Tasto 'H' - Help
            PrintHotkeyHelp();
            break;
      }
   }
}

//+------------------------------------------------------------------+
//| Funzione per ottenere testo motivo deinit                       |
//+------------------------------------------------------------------+
string GetDeinitReasonText(int reason)
{
   switch(reason)
   {
      case REASON_PROGRAM:      return "EA terminato dall'utente";
      case REASON_REMOVE:       return "EA rimosso dal grafico";
      case REASON_RECOMPILE:    return "EA ricompilato";
      case REASON_CHARTCHANGE:  return "Cambio simbolo o timeframe";
      case REASON_CHARTCLOSE:   return "Grafico chiuso";
      case REASON_PARAMETERS:   return "Parametri cambiati";
      case REASON_ACCOUNT:      return "Account cambiato";
      case REASON_TEMPLATE:     return "Template applicato";
      case REASON_INITFAILED:   return "Inizializzazione fallita";
      case REASON_CLOSE:        return "Terminale chiuso";
      default:                  return "Motivo sconosciuto (" + IntegerToString(reason) + ")";
   }
}

//+------------------------------------------------------------------+
//| Funzione per stampare help hotkeys                              |
//+------------------------------------------------------------------+
void PrintHotkeyHelp()
{
   Print("=== HOTKEYS DISPONIBILI ===");
   Print("S - Stampa status corrente");
   Print("W - Forza scrittura buffer su file");
   Print("C - Reset contatori");
   Print("H - Mostra questo help");
   Print("===========================");
   Print("🎯 RICORDA: Il file è SEMPRE leggibile tra le scritture!");
   Print("📁 File output: ", bridge != NULL ? bridge.GetOutputFile() : "N/A");
}
