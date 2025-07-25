//+------------------------------------------------------------------+
//|       MapLongInt.mqh                                            |
//|   Mappa semplificata: chiave long → int                         |
//+------------------------------------------------------------------+
#ifndef __MAPLONGINT_MQH__
#define __MAPLONGINT_MQH__

class CMapLongInt
{
private:
    long keys[];
    int values[];

public:
    // ** AGGIUNGI IL COSTRUTTORE **
    CMapLongInt()
    {
        // Inizializza gli array a dimensione zero
        ArrayResize(keys, 0);
        ArrayResize(values, 0);
    }

    // ** AGGIUNGI IL DISTRUTTORE **
    ~CMapLongInt()
    {
        // Pulisci la mappa e libera la memoria degli array
        Clear();
    }

    void Insert(long key, int value)
    {
        int index = FindIndex(key);
        if (index == -1)
        {
            ArrayResize(keys, ArraySize(keys) + 1);
            ArrayResize(values, ArraySize(values) + 1);
            keys[ArraySize(keys) - 1] = key;
            values[ArraySize(values) - 1] = value;
        }
        else
        {
            values[index] = value;
        }
    }

    bool IsExist(long key)
    {
        return (FindIndex(key) != -1);
    }

    bool Get(long key, int &value)
    {
        int index = FindIndex(key);
        if (index != -1)
        {
            value = values[index];
            return true;
        }
        return false;
    }

    int At(long key)
    {
        int index = FindIndex(key);
        return (index != -1) ? values[index] : -1;
    }

    void Delete(long key)
    {
        int index = FindIndex(key);
        if (index != -1)
        {
            ArrayRemove(keys, index);
            ArrayRemove(values, index);
        }
    }

    void Clear()
    {
        ArrayResize(keys, 0); // Rimuovi tutti gli elementi
        ArrayResize(values, 0);
        // Oppure, più esplicitamente: ArrayFree(keys); ArrayFree(values);
        // ma ArrayResize(..., 0) ha lo stesso effetto di liberare la memoria.
    }

    void GetKeys(long &outKeys[])
    {
        ArrayCopy(outKeys, keys);
    }

private:
    int FindIndex(long key)
    {
        for (int i = 0; i < ArraySize(keys); i++)
        {
            if (keys[i] == key)
                return i;
        }
        return -1;
    }
};

#endif // __MAPLONGINT_MQH__