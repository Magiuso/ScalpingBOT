//+------------------------------------------------------------------+
//| MapLong_Bool.mqh                                                 |
//| Mappa semplificata long → bool per gestione stati per ticket     |
//+------------------------------------------------------------------+
#ifndef __MAPLONG_BOOL_MQH__
#define __MAPLONG_BOOL_MQH__

class CMapLongBool
{
private:
    long keys[];
    bool values[];

    int FindIndex(long key)
    {
        for (int i = 0; i < ArraySize(keys); i++)
        {
            if (keys[i] == key)
                return i;
        }
        return -1;
    }

public:
    void Insert(long key, bool value)
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

    bool Get(long key, bool &value)
    {
        int index = FindIndex(key);
        if (index != -1)
        {
            value = values[index];
            return true;
        }
        return false;
    }

    bool GetDefault(long key, bool defaultValue = false)
    {
        bool val;
        if (Get(key, val))
            return val;
        return defaultValue;
    }

    void Remove(long key)
    {
        int index = FindIndex(key);
        if (index != -1)
        {
            for (int i = index; i < ArraySize(keys) - 1; i++)
            {
                keys[i] = keys[i + 1];
                values[i] = values[i + 1];
            }
            ArrayResize(keys, ArraySize(keys) - 1);
            ArrayResize(values, ArraySize(values) - 1);
        }
    }

    void Clear()
    {
        ArrayResize(keys, 0);
        ArrayResize(values, 0);
    }
};

// === 🌐 Oggetto globale ===
CMapLongBool g_boolMap;

// === 🔁 Wrapper funzione globale (per leggibilità) ===
bool MapGetBool(long key)
{
    return g_boolMap.GetDefault(key, false);
}

void MapSetBool(long key, bool value)
{
    g_boolMap.Insert(key, value);
}

void MapRemoveBool(long key)
{
    g_boolMap.Remove(key);
}

#endif // __MAPLONG_BOOL_MQH__
