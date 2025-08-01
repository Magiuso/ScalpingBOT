//+------------------------------------------------------------------+
//| RecoveryBlock.mqh                                                |
//| Modulo per la gestione dei trade di recupero basati su trigger.  |
//| ✅ Usa CMapLongInt per i tentativi per ticket                    |
//+------------------------------------------------------------------+
#ifndef __RECOVERY_BLOCK_MQH__
#define __RECOVERY_BLOCK_MQH__

#include <ScalpingBot\Utility.mqh>
#include <ScalpingBot\RecoveryTriggerManager.mqh>
#include <ScalpingBot\OpenTradeRecovery.mqh> // Assicurati che questo file sia già stato aggiornato come discusso
#include <Map\MapLong_Int.mqh>
#include <Map\MapLong_Bool.mqh>  

CMapLongBool g_recovery_ban_map;      // 🚫 Mappa BAN dedicata
CMapLongInt g_recovery_attempts_map;  // 📊 Mappa dei tentativi per ogni ticket

//+-----------------------------------------------------------------------+
//| 🎯 Funzione principale del Recovery Block → da richiamare in OnTick  |
//+-----------------------------------------------------------------------+
void ExecuteRecoveryTrades()
{
    if (!EnableRecoveryBlock || g_recovery_manager.TotalTriggers() == 0)
        return;

    long trigger_tickets_array[];
    int num_triggers = g_recovery_manager.GetTriggerTickets(trigger_tickets_array);

    for (int i = 0; i < num_triggers; i++)
    {
        ulong originalTradeTicket = (ulong)trigger_tickets_array[i];
        RecoveryTriggerInfo trigger_info;

        if (!g_recovery_manager.GetTriggerInfo(originalTradeTicket, trigger_info))
        {
            if (EnableLogging_RecoveryBlock)
                PrintFormat("❌ [Recovery Block] Trigger ticket %d non trovato → rimosso.", originalTradeTicket);
            g_recovery_manager.RemoveTrigger(originalTradeTicket);
            g_recovery_attempts_map.Delete(originalTradeTicket);
            continue;
        }

        int attempts = 0;
        g_recovery_attempts_map.Get(originalTradeTicket, attempts);

        if (attempts >= MaxRecoveryAttempts)
        {
            if (EnableLogging_RecoveryBlock)
                PrintFormat("⚠️ [Recovery Block] Superato max tentativi (%d) → trigger ticket %d rimosso.", MaxRecoveryAttempts, originalTradeTicket);
            g_recovery_manager.RemoveTrigger(originalTradeTicket);
            g_recovery_attempts_map.Delete(originalTradeTicket);
            
            // → BAN sul ticket per evitare nuovo ciclo di tentativi:
            g_recovery_ban_map.Insert(originalTradeTicket, true);
            continue;
        }

        if (trigger_info.IsProcessed)
        {
            if (EnableLogging_RecoveryBlock)
                PrintFormat("ℹ️ [Recovery Block] Trigger ticket %d già processato (trade di recupero aperto). Attendo chiusura.", originalTradeTicket);
            continue;
        }

        // Tentativo di apertura del trade di recupero
        if (EnableLogging_RecoveryBlock)
            PrintFormat("📤 [Recovery Block] Tentativo #%d → Richiesta trade di recupero (direzione opposta) per %s (ticket originale %d)",
                        attempts + 1, trigger_info.Symbol, originalTradeTicket);

        // Chiamata a OpenRecoveryTrade con la firma aggiornata (due parametri)
        bool success = OpenRecoveryTrade(trigger_info.Symbol, originalTradeTicket);

        if (success)
        {
            // Il trade di recupero è stato richiesto con successo, ora è processato
            g_recovery_manager.SetTriggerProcessed(originalTradeTicket, true);
            g_recovery_attempts_map.Delete(originalTradeTicket); // Pulisci i tentativi
            
            if (EnableLogging_RecoveryBlock)
                PrintFormat("✅ [Recovery Block] Richiesta trade di recupero inviata per ticket originale %d", originalTradeTicket);
        }
        else
        {
            g_recovery_attempts_map.Insert(originalTradeTicket, attempts + 1);

            if (EnableLogging_RecoveryBlock)
                PrintFormat("❌ [Recovery Block] Fallita apertura trade recovery (ticket originale %d). Tentativo %d/%d", originalTradeTicket, attempts + 1, MaxRecoveryAttempts);
        }
    }

    //────────────────────────────────────────────────────────────
    // ♻️ Cleanup dei trigger già processati ma con posizione di RECUPERO chiusa
    //────────────────────────────────────────────────────────────
    long cleanup_tickets_array[];
    int cleanup_total = g_recovery_manager.GetTriggerTickets(cleanup_tickets_array);

    for (int i = 0; i < cleanup_total; i++)
    {
        ulong originalTradeTicket = (ulong)cleanup_tickets_array[i];

        RecoveryTriggerInfo trigger;
        if (!g_recovery_manager.GetTriggerInfo(originalTradeTicket, trigger))
            continue; // Già rimosso o errore, salta

        if (trigger.IsProcessed)
        {
            // Controlla il ticket specifico del trade di recupero salvato
            bool recoveryPositionStillOpen = false;
            if (trigger.RecoveryTradeTicket != 0) // Se il ticket di recupero è stato salvato
            {
                recoveryPositionStillOpen = PositionSelectByTicket(trigger.RecoveryTradeTicket);
            }
            else
            {
                // Fallback: se per qualche motivo il RecoveryTradeTicket non è stato salvato,
                // cerca una posizione generica con il magic number sullo stesso simbolo.
                // Questa parte è meno precisa ma serve da sicurezza.
                for (int j = PositionsTotal() - 1; j >= 0; j--)
                {
                    if (PositionGetInteger(POSITION_MAGIC) == RecoveryMagicNumber &&
                        PositionGetString(POSITION_SYMBOL) == trigger.Symbol)
                    {
                        recoveryPositionStillOpen = true;
                        break; 
                    }
                }
            }

            if (!recoveryPositionStillOpen)
            {
                g_recovery_manager.RemoveTrigger(originalTradeTicket);
                if (EnableLogging_RecoveryBlock)
                    PrintFormat("✅ [Recovery Cleanup] Trigger ticket %d rimosso dopo chiusura del trade di RECUPERO (ticket recupero: %d).", originalTradeTicket, trigger.RecoveryTradeTicket);
            }
            else
            {
                if (EnableLogging_RecoveryBlock)
                    PrintFormat("ℹ️ [Recovery Cleanup] Trade di RECUPERO per trigger ticket %d (ticket recupero: %d) è ancora aperto. Attendo chiusura.", originalTradeTicket, trigger.RecoveryTradeTicket);
            }
        }
    }
}

#endif // __RECOVERY_BLOCK_MQH__
