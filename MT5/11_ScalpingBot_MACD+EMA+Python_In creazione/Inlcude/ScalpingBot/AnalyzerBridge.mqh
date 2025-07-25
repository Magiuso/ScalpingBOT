//+------------------------------------------------------------------+
//|                                               AnalyzerBridge.mqh |
//|                    Bridge BUFFER + PERIODIC WRITE per Python    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024"
#property version   "4.00"

//+------------------------------------------------------------------+
//| Struttura per dati tick                                          |
//+------------------------------------------------------------------+
struct TickDataPacket
{
   datetime timestamp;
   string   symbol;
   double   bid;
   double   ask;
   double   last;
   long     volume;
   double   spread_percentage;
   double   price_change_1m;
   double   price_change_5m;
   double   volatility;
   double   momentum_5m;
   string   market_state;
};

//+------------------------------------------------------------------+
//| Classe Bridge BUFFER + PERIODIC WRITE                           |
//+------------------------------------------------------------------+
class CAnalyzerBridge
{
private:
   // Configurazione base
   string            m_symbol;
   ENUM_TIMEFRAMES   m_timeframe;
   string            m_output_file;
   
   // BUFFER + PERIODIC WRITE SYSTEM
   string            m_memory_buffer;         // Buffer in memoria
   int               m_buffer_max_size;       // Dimensione max buffer (caratteri)
   int               m_buffer_max_lines;      // Numero max linee nel buffer
   int               m_buffer_line_count;     // Linee correnti nel buffer
   datetime          m_last_file_write;       // Ultima scrittura su file
   int               m_write_interval;        // Scrivi file ogni N secondi
   bool              m_append_mode;           // Modalità append (true) o overwrite (false)
   
   // Buffer dati storici per calcoli
   double            m_price_buffer[];
   datetime          m_time_buffer[];
   long              m_volume_buffer[];
   int               m_buffer_size;
   int               m_buffer_index;
   
   // Contatori e controllo
   long              m_tick_count;
   long              m_packets_sent;
   long              m_total_writes;
   datetime          m_last_send_time;
   int               m_send_interval_ms;
   bool              m_is_active;
   bool              m_debug_mode;
   
   // METODI BUFFER + PERIODIC WRITE
   bool              WriteToMemoryBuffer(const string &json_data);
   bool              ShouldWriteToFile(datetime now);
   bool              WriteBufferToFile();
   void              ResetMemoryBuffer();
   string            GetCurrentDateTime();
   
   // Metodi calcolo
   double            CalculateVolatility(int periods = 20);
   double            CalculateMomentum(int periods = 5);
   double            CalculatePriceChange(int minutes);
   string            DetectMarketState();
   void              UpdateBuffer(double price, datetime time, long volume);
   string            CreateJsonPacket(const TickDataPacket &packet);
   
public:
   // Costruttore/Distruttore
                     CAnalyzerBridge(void);
                    ~CAnalyzerBridge(void);
   
   // Inizializzazione
   bool              Initialize(string symbol, ENUM_TIMEFRAMES tf = PERIOD_M1);
   void              Shutdown(void);
   
   // Metodi principali
   bool              SendTick(void);
   void              OnTick(void);
   void              OnTimer(void);
   
   // CONFIGURAZIONE BUFFER + PERIODIC WRITE
   void              SetWriteInterval(int seconds) { m_write_interval = seconds; }
   void              SetBufferMaxSize(int size) { m_buffer_max_size = size; }
   void              SetBufferMaxLines(int lines) { m_buffer_max_lines = lines; }
   void              SetAppendMode(bool append) { m_append_mode = append; }
   
   // Configurazione standard
   void              SetSendInterval(int interval_ms) { m_send_interval_ms = interval_ms; }
   void              SetDebugMode(bool debug) { m_debug_mode = debug; }
   void              SetBufferSize(int size);
   
   // Status e utility
   bool              IsActive(void) { return m_is_active; }
   long              GetTickCount(void) { return m_tick_count; }
   long              GetPacketsSent(void) { return m_packets_sent; }
   long              GetTotalWrites(void) { return m_total_writes; }
   int               GetBufferSize(void) { return StringLen(m_memory_buffer); }
   int               GetBufferLines(void) { return m_buffer_line_count; }
   string            GetOutputFile(void) { return m_output_file; }
   void              PrintStatus(void);
   void              Reset(void);
   bool              ForceWrite(void) { return WriteBufferToFile(); }
};

//+------------------------------------------------------------------+
//| Costruttore BUFFER + PERIODIC WRITE                             |
//+------------------------------------------------------------------+
CAnalyzerBridge::CAnalyzerBridge(void)
{
   m_symbol = "";
   m_timeframe = PERIOD_M1;
   m_output_file = "";
   
   // BUFFER + PERIODIC WRITE CONFIGURATION
   m_memory_buffer = "";
   m_buffer_max_size = 100000;       // 100KB di testo
   m_buffer_max_lines = 1000;        // 1000 linee JSON max
   m_buffer_line_count = 0;
   m_last_file_write = 0;
   m_write_interval = 10;            // Scrivi file ogni 10 secondi
   m_append_mode = true;             // Modalità append di default
   
   // Buffer prezzi per calcoli
   m_buffer_size = 1000;
   m_buffer_index = 0;
   
   // Contatori
   m_tick_count = 0;
   m_packets_sent = 0;
   m_total_writes = 0;
   m_last_send_time = 0;
   m_send_interval_ms = 100;
   m_is_active = false;
   m_debug_mode = true;
   
   // Inizializza buffer prezzi
   ArrayResize(m_price_buffer, m_buffer_size);
   ArrayResize(m_time_buffer, m_buffer_size);
   ArrayResize(m_volume_buffer, m_buffer_size);
   ArrayInitialize(m_price_buffer, 0.0);
   ArrayInitialize(m_time_buffer, 0);
   ArrayInitialize(m_volume_buffer, 0);
}

//+------------------------------------------------------------------+
//| Distruttore                                                      |
//+------------------------------------------------------------------+
CAnalyzerBridge::~CAnalyzerBridge(void)
{
   Shutdown();
}

//+------------------------------------------------------------------+
//| Inizializzazione BUFFER + PERIODIC WRITE                        |
//+------------------------------------------------------------------+
bool CAnalyzerBridge::Initialize(string symbol, ENUM_TIMEFRAMES tf = PERIOD_M1)
{
   if(symbol == "")
   {
      Print("ERROR: Symbol cannot be empty");
      return false;
   }
   
   m_symbol = symbol;
   m_timeframe = tf;
   m_output_file = "analyzer_" + symbol + ".jsonl";
   
   // Test scrittura permessi
   string test_file = "test_permissions.txt";
   int test_handle = FileOpen(test_file, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(test_handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot write to Files directory. Error: ", GetLastError());
      return false;
   }
   FileWriteString(test_handle, "Test OK");
   FileClose(test_handle);
   FileDelete(test_file);
   
   // Inizializza buffer storico per calcoli
   double closes[];
   datetime times[];
   long volumes[];
   
   int copied = CopyClose(m_symbol, m_timeframe, 0, m_buffer_size, closes);
   CopyTime(m_symbol, m_timeframe, 0, m_buffer_size, times);
   CopyTickVolume(m_symbol, m_timeframe, 0, m_buffer_size, volumes);
   
   if(copied > 0)
   {
      int start_index = MathMax(0, copied - m_buffer_size);
      for(int i = start_index; i < copied; i++)
      {
         UpdateBuffer(closes[i], times[i], volumes[i]);
      }
      
      if(m_debug_mode)
         Print("Buffer initialized with ", copied, " historical points");
   }
   
   // Scrivi header iniziale nel file
   if(!m_append_mode || !FileIsExist(m_output_file))
   {
      string header = "{\"type\":\"session_start\",\"timestamp\":\"" + 
                     GetCurrentDateTime() + 
                     "\",\"symbol\":\"" + m_symbol + 
                     "\",\"timeframe\":\"" + EnumToString(m_timeframe) + 
                     "\",\"version\":\"4.00_BUFFER_PERIODIC\"}\n";
      
      int file_handle = FileOpen(m_output_file, FILE_WRITE | FILE_TXT | FILE_ANSI);
      if(file_handle != INVALID_HANDLE)
      {
         FileWriteString(file_handle, header);
         FileClose(file_handle);  // CHIUDI IMMEDIATAMENTE!
         
         if(m_debug_mode)
            Print("📄 File header written: ", m_output_file);
      }
   }
   
   // Inizializza sistema
   m_last_file_write = TimeCurrent();
   ResetMemoryBuffer();
   m_is_active = true;
   
   if(m_debug_mode)
   {
      Print("AnalyzerBridge BUFFER + PERIODIC WRITE initialized for ", m_symbol);
      Print("📁 Output file: ", m_output_file);
      Print("⏱️ Write interval: ", m_write_interval, " seconds");
      Print("📝 Append mode: ", m_append_mode ? "ON" : "OFF");
      Print("🎯 FILE SEMPRE LEGGIBILE tra le scritture!");
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Shutdown BUFFER + PERIODIC WRITE                                |
//+------------------------------------------------------------------+
void CAnalyzerBridge::Shutdown(void)
{
   if(m_is_active)
   {
      // Scrivi buffer finale
      WriteBufferToFile();
      
      // Scrivi footer finale
      string footer = "{\"type\":\"session_end\",\"timestamp\":\"" + 
                     GetCurrentDateTime() + 
                     "\",\"total_ticks\":" + IntegerToString(m_tick_count) + 
                     ",\"total_packets\":" + IntegerToString(m_packets_sent) + 
                     ",\"total_writes\":" + IntegerToString(m_total_writes) + "}\n";
      
      int file_handle = FileOpen(m_output_file, FILE_WRITE | FILE_READ | FILE_TXT | FILE_ANSI);
      if(file_handle != INVALID_HANDLE)
      {
         FileSeek(file_handle, 0, SEEK_END);  // Vai alla fine del file
         FileWriteString(file_handle, footer);
         FileClose(file_handle);  // CHIUDI IMMEDIATAMENTE!
      }
      
      m_is_active = false;
      
      if(m_debug_mode)
         Print("AnalyzerBridge BUFFER + PERIODIC WRITE shutdown - ", 
               m_total_writes, " total writes completed");
   }
}

//+------------------------------------------------------------------+
//| Gestione tick principale                                         |
//+------------------------------------------------------------------+
void CAnalyzerBridge::OnTick(void)
{
   if(!m_is_active)
      return;
   
   m_tick_count++;
   
   // Aggiorna buffer con tick corrente
   double current_price = (SymbolInfoDouble(m_symbol, SYMBOL_BID) + 
                          SymbolInfoDouble(m_symbol, SYMBOL_ASK)) / 2.0;
   datetime current_time = TimeCurrent();
   long current_volume = SymbolInfoInteger(m_symbol, SYMBOL_VOLUME);
   
   UpdateBuffer(current_price, current_time, current_volume);
   
   // Invia dati in base all'intervallo
   if(current_time - m_last_send_time >= m_send_interval_ms / 1000)
   {
      SendTick();
      m_last_send_time = current_time;
   }
}

//+------------------------------------------------------------------+
//| Invia tick - BUFFER + PERIODIC WRITE VERSION                    |
//+------------------------------------------------------------------+
bool CAnalyzerBridge::SendTick(void)
{
   if(!m_is_active)
      return false;
   
   // Crea packet
   TickDataPacket packet;
   
   packet.timestamp = TimeCurrent();
   packet.symbol = m_symbol;
   packet.bid = SymbolInfoDouble(m_symbol, SYMBOL_BID);
   packet.ask = SymbolInfoDouble(m_symbol, SYMBOL_ASK);
   packet.last = SymbolInfoDouble(m_symbol, SYMBOL_LAST);
   packet.volume = SymbolInfoInteger(m_symbol, SYMBOL_VOLUME);
   
   if(packet.bid > 0)
      packet.spread_percentage = (packet.ask - packet.bid) / packet.bid;
   else
      packet.spread_percentage = 0.0;
   
   packet.price_change_1m = CalculatePriceChange(1);
   packet.price_change_5m = CalculatePriceChange(5);
   packet.volatility = CalculateVolatility();
   packet.momentum_5m = CalculateMomentum();
   packet.market_state = DetectMarketState();
   
   // Converti in JSON
   string json_data = CreateJsonPacket(packet);
   
   // SCRIVI NEL MEMORY BUFFER
   bool success = WriteToMemoryBuffer(json_data);
   
   if(success)
   {
      m_packets_sent++;
      
      // Log ogni 100 packets
      if(m_debug_mode && m_packets_sent % 100 == 0)
         Print("BUFFER: ", m_packets_sent, " packets | Buffer: ", GetBufferLines(), 
               " lines | Writes: ", m_total_writes);
   }
   
   return success;
}

//+------------------------------------------------------------------+
//| BUFFER + PERIODIC: Scrivi nel memory buffer                     |
//+------------------------------------------------------------------+
bool CAnalyzerBridge::WriteToMemoryBuffer(const string &json_data)
{
   if(!m_is_active)
      return false;
   
   // Aggiungi al buffer
   m_memory_buffer += json_data;
   m_buffer_line_count++;
   
   datetime now = TimeCurrent();
   
   // CHECK: Scrivi su file se è tempo
   if(ShouldWriteToFile(now))
   {
      WriteBufferToFile();
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Controlla se scrivere su file                                   |
//+------------------------------------------------------------------+
bool CAnalyzerBridge::ShouldWriteToFile(datetime now)
{
   return (now - m_last_file_write >= m_write_interval) ||
          (StringLen(m_memory_buffer) >= m_buffer_max_size) ||
          (m_buffer_line_count >= m_buffer_max_lines);
}

//+------------------------------------------------------------------+
//| Scrivi buffer su file - APRI → SCRIVI → CHIUDI                  |
//+------------------------------------------------------------------+
bool CAnalyzerBridge::WriteBufferToFile()
{
   if(StringLen(m_memory_buffer) == 0)
      return true;
   
   // APRI file in modalità append
   int file_handle = FileOpen(m_output_file, FILE_WRITE | FILE_READ | FILE_TXT | FILE_ANSI);
   if(file_handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot open file for writing: ", m_output_file, " Error: ", GetLastError());
      return false;
   }
   
   // Vai alla fine del file (append)
   FileSeek(file_handle, 0, SEEK_END);
   
   // SCRIVI tutto il buffer
   uint bytes_written = FileWriteString(file_handle, m_memory_buffer);
   
   // CHIUDI IMMEDIATAMENTE - File ora libero per lettura!
   FileClose(file_handle);
   
   m_total_writes++;
   
   if(m_debug_mode)
      Print("💾 WRITE #", m_total_writes, ": ", m_buffer_line_count, 
            " lines (", StringLen(m_memory_buffer), " bytes) → ", m_output_file, 
            " [FILE NOW FREE]");
   
   // Reset buffer
   ResetMemoryBuffer();
   
   return bytes_written > 0;
}

//+------------------------------------------------------------------+
//| Reset memory buffer                                             |
//+------------------------------------------------------------------+
void CAnalyzerBridge::ResetMemoryBuffer()
{
   m_memory_buffer = "";
   m_buffer_line_count = 0;
   m_last_file_write = TimeCurrent();
}

//+------------------------------------------------------------------+
//| OnTimer - Gestisce scrittura periodica                          |
//+------------------------------------------------------------------+
void CAnalyzerBridge::OnTimer(void)
{
   if(!m_is_active)
      return;
   
   datetime now = TimeCurrent();
   
   // Check scrittura periodica
   if(ShouldWriteToFile(now))
   {
      WriteBufferToFile();
   }
}

//+------------------------------------------------------------------+
//| Ottieni data/ora corrente formattata                            |
//+------------------------------------------------------------------+
string CAnalyzerBridge::GetCurrentDateTime()
{
   return TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS);
}

//+------------------------------------------------------------------+
//| Status dettagliato BUFFER + PERIODIC WRITE                      |
//+------------------------------------------------------------------+
void CAnalyzerBridge::PrintStatus(void)
{
   datetime now = TimeCurrent();
   
   int seconds_to_next_write = m_write_interval - (int)(now - m_last_file_write);
   if(seconds_to_next_write < 0) seconds_to_next_write = 0;
   
   Print("=== AnalyzerBridge BUFFER + PERIODIC WRITE Status ===");
   Print("Symbol: ", m_symbol, " | Timeframe: ", EnumToString(m_timeframe));
   Print("Active: ", m_is_active ? "YES" : "NO");
   Print("📁 OUTPUT FILE: ", m_output_file, " <-- PYTHON LEGGE QUESTO!");
   Print("📊 PERFORMANCE:");
   Print("   Ticks processed: ", m_tick_count);
   Print("   Packets sent: ", m_packets_sent);
   Print("   Total writes: ", m_total_writes);
   Print("💾 MEMORY BUFFER:");
   Print("   Lines: ", GetBufferLines(), " | Size: ", GetBufferSize(), " bytes");
   Print("   Next write in: ", seconds_to_next_write, " seconds");
   Print("   Write interval: ", m_write_interval, " seconds");
   Print("   Append mode: ", m_append_mode ? "ON" : "OFF");
   Print("⏰ TIMING:");
   Print("   Last write: ", TimeToString(m_last_file_write, TIME_DATE|TIME_SECONDS));
   Print("   Current time: ", GetCurrentDateTime());
   Print("🎯 FILE STATUS: FREE for reading (writes every ", m_write_interval, "s)");
   Print("====================================================");
}

//+------------------------------------------------------------------+
//| Reset completo                                                  |
//+------------------------------------------------------------------+
void CAnalyzerBridge::Reset(void)
{
   m_tick_count = 0;
   m_packets_sent = 0;
   m_total_writes = 0;
   m_last_send_time = 0;
   
   // Reset buffer prezzi
   ArrayInitialize(m_price_buffer, 0.0);
   ArrayInitialize(m_time_buffer, 0);
   ArrayInitialize(m_volume_buffer, 0);
   m_buffer_index = 0;
   
   // Reset memory buffer
   ResetMemoryBuffer();
   
   if(m_debug_mode)
      Print("AnalyzerBridge BUFFER + PERIODIC WRITE reset complete");
}

//+------------------------------------------------------------------+
//| Aggiorna buffer circolare                                       |
//+------------------------------------------------------------------+
void CAnalyzerBridge::UpdateBuffer(double price, datetime time, long volume)
{
   m_price_buffer[m_buffer_index] = price;
   m_time_buffer[m_buffer_index] = time;
   m_volume_buffer[m_buffer_index] = volume;
   
   m_buffer_index = (m_buffer_index + 1) % m_buffer_size;
}

//+------------------------------------------------------------------+
//| Calcola volatilità                                              |
//+------------------------------------------------------------------+
double CAnalyzerBridge::CalculateVolatility(int periods = 20)
{
   if(periods > m_buffer_size)
      periods = m_buffer_size;
   
   double prices[];
   ArrayResize(prices, periods);
   
   int idx = m_buffer_index;
   for(int i = 0; i < periods; i++)
   {
      idx = (idx - 1 + m_buffer_size) % m_buffer_size;
      prices[i] = m_price_buffer[idx];
   }
   
   double sum_returns = 0.0;
   double sum_sq_returns = 0.0;
   int valid_returns = 0;
   
   for(int i = 1; i < periods; i++)
   {
      if(prices[i] > 0 && prices[i-1] > 0)
      {
         double ret = (prices[i] - prices[i-1]) / prices[i-1];
         sum_returns += ret;
         sum_sq_returns += ret * ret;
         valid_returns++;
      }
   }
   
   if(valid_returns < 2)
      return 0.0;
   
   double mean_return = sum_returns / valid_returns;
   double variance = (sum_sq_returns / valid_returns) - (mean_return * mean_return);
   
   return MathSqrt(MathMax(0, variance));
}

//+------------------------------------------------------------------+
//| Calcola momentum                                                |
//+------------------------------------------------------------------+
double CAnalyzerBridge::CalculateMomentum(int periods = 5)
{
   if(periods > m_buffer_size)
      periods = m_buffer_size;
   
   int current_idx = (m_buffer_index - 1 + m_buffer_size) % m_buffer_size;
   int past_idx = (current_idx - periods + m_buffer_size) % m_buffer_size;
   
   double current_price = m_price_buffer[current_idx];
   double past_price = m_price_buffer[past_idx];
   
   if(past_price <= 0)
      return 0.0;
   
   return (current_price - past_price) / past_price;
}

//+------------------------------------------------------------------+
//| Calcola price change per N minuti                               |
//+------------------------------------------------------------------+
double CAnalyzerBridge::CalculatePriceChange(int minutes)
{
   datetime target_time = TimeCurrent() - minutes * 60;
   
   double past_price = 0.0;
   datetime closest_time = 0;
   
   for(int i = 0; i < m_buffer_size; i++)
   {
      if(m_time_buffer[i] > 0 && MathAbs(m_time_buffer[i] - target_time) < MathAbs(closest_time - target_time))
      {
         closest_time = m_time_buffer[i];
         past_price = m_price_buffer[i];
      }
   }
   
   if(past_price <= 0)
      return 0.0;
   
   double current_price = (SymbolInfoDouble(m_symbol, SYMBOL_BID) + 
                          SymbolInfoDouble(m_symbol, SYMBOL_ASK)) / 2.0;
   
   return (current_price - past_price) / past_price;
}

//+------------------------------------------------------------------+
//| Rileva stato del mercato                                        |
//+------------------------------------------------------------------+
string CAnalyzerBridge::DetectMarketState(void)
{
   double volatility = CalculateVolatility(20);
   double momentum = MathAbs(CalculateMomentum(10));
   
   if(volatility > 0.02)
   {
      if(momentum > 0.01)
         return "volatile_trending";
      else
         return "volatile_ranging";
   }
   else if(volatility < 0.005)
   {
      return "low_activity";
   }
   else
   {
      if(momentum > 0.005)
         return "trending";
      else
         return "ranging";
   }
}

//+------------------------------------------------------------------+
//| Crea JSON packet                                                |
//+------------------------------------------------------------------+
string CAnalyzerBridge::CreateJsonPacket(const TickDataPacket &packet)
{
   string json = "{";
   json += "\"type\":\"tick\",";
   json += "\"timestamp\":\"" + TimeToString(packet.timestamp, TIME_DATE|TIME_SECONDS) + "\",";
   json += "\"symbol\":\"" + packet.symbol + "\",";
   json += "\"bid\":" + DoubleToString(packet.bid, 5) + ",";
   json += "\"ask\":" + DoubleToString(packet.ask, 5) + ",";
   json += "\"last\":" + DoubleToString(packet.last, 5) + ",";
   json += "\"volume\":" + IntegerToString(packet.volume) + ",";
   json += "\"spread_percentage\":" + DoubleToString(packet.spread_percentage, 6) + ",";
   json += "\"price_change_1m\":" + DoubleToString(packet.price_change_1m, 6) + ",";
   json += "\"price_change_5m\":" + DoubleToString(packet.price_change_5m, 6) + ",";
   json += "\"volatility\":" + DoubleToString(packet.volatility, 6) + ",";
   json += "\"momentum_5m\":" + DoubleToString(packet.momentum_5m, 6) + ",";
   json += "\"market_state\":\"" + packet.market_state + "\"";
   json += "}\n";
   
   return json;
}

//+------------------------------------------------------------------+
//| Imposta dimensione buffer                                       |
//+------------------------------------------------------------------+
void CAnalyzerBridge::SetBufferSize(int size)
{
   if(size < 10 || size > 10000)
      return;
   
   m_buffer_size = size;
   ArrayResize(m_price_buffer, m_buffer_size);
   ArrayResize(m_time_buffer, m_buffer_size);
   ArrayResize(m_volume_buffer, m_buffer_size);
}