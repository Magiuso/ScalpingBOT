//+------------------------------------------------------------------+
//| RecoveryTriggerManager.mqh                                      |
//| Classe per la gestione centralizzata dei segnali di recupero.    |
//| Incapsula la mappa dei trigger e la logica di gestione memoria.  |
//| ✅ VERSIONE AGGIORNATA: Utilizza ArrayBasedMap (senza CObject)   |
//+------------------------------------------------------------------+
#ifndef __RECOVERY_TRIGGER_MANAGER_MQH__
#define __RECOVERY_TRIGGER_MANAGER_MQH__

#include <ScalpingBot\Utility.mqh> // Contiene RecoveryTriggerInfo (struct) e ArrayBasedMap

//-------------------------------------------------------+
// --- CLASSE PER LA GESTIONE DEI TRIGGER DI RECUPERO ---|
//-------------------------------------------------------+
class CRecoveryTriggerManager
{
private:
    ArrayBasedMap<RecoveryTriggerInfo> m_triggers_map;

public:
    CRecoveryTriggerManager() {}
    ~CRecoveryTriggerManager() {}

    // Metodo per aggiungere o aggiornare un segnale di recupero
    bool AddOrUpdateTrigger(ulong original_ticket, string symbol, ENUM_POSITION_TYPE original_type, double original_lot)
    {
        RecoveryTriggerInfo existing_trigger_val;

        if (m_triggers_map.TryGetValue(original_ticket, existing_trigger_val) && !existing_trigger_val.IsProcessed)
        {
            return true; // Segnale già presente e in attesa di elaborazione
        }

        RecoveryTriggerInfo new_trigger(symbol, original_type, original_lot);
        return m_triggers_map.Add(original_ticket, new_trigger);
    }

    // Metodo per rimuovere un segnale di recupero
    bool RemoveTrigger(ulong original_ticket)
    {
        return m_triggers_map.Remove(original_ticket);
    }

    // Metodo per ottenere una COPIA del trigger (per lettura)
    bool GetTriggerInfo(ulong original_ticket, RecoveryTriggerInfo& out_info)
    {
        return m_triggers_map.TryGetValue(original_ticket, out_info);
    }
    
    // Metodo per aggiornare lo stato IsProcessed (necessario perché GetTriggerInfo restituisce una copia)
    bool SetTriggerProcessed(ulong original_ticket, bool processed_status)
    {
        RecoveryTriggerInfo current_info;
        if (m_triggers_map.TryGetValue(original_ticket, current_info))
        {
            current_info.IsProcessed = processed_status;
            return m_triggers_map.Add(original_ticket, current_info);
        }
        return false;
    }

    // Nuovo metodo per salvare il ticket del trade di recupero
    bool SetRecoveryTradeTicket(ulong original_ticket, ulong recovery_trade_ticket)
    {
        RecoveryTriggerInfo current_info;
        if (m_triggers_map.TryGetValue(original_ticket, current_info))
        {
            current_info.RecoveryTradeTicket = recovery_trade_ticket;
            return m_triggers_map.Add(original_ticket, current_info); // Aggiorna la mappa
        }
        return false;
    }

    // Metodo per controllare se un trigger esiste
    bool ContainsTrigger(ulong original_ticket)
    {
        return m_triggers_map.ContainsKey(original_ticket);
    }

    // Metodo per ottenere il numero corrente di trigger
    int TotalTriggers() const
    {
        return m_triggers_map.Size();
    }

    // Metodo per ottenere le chiavi (tickets) per l'iterazione
    int GetTriggerTickets(long& keys_array[])
    {
        return m_triggers_map.GetKeys(keys_array);
    }

    // Metodo per la pulizia completa della mappa
    void Clear()
    {
        m_triggers_map.Clear();
    }
};

#endif // __RECOVERY_TRIGGER_MANAGER_MQH__