//+------------------------------------------------------------------+
//|                 Map.mqh                                          |
//|     Mappa semplificata chiave string → struct                    |
//+------------------------------------------------------------------+
#ifndef __MAP_STRING_MQH__
#define __MAP_STRING_MQH__

template<typename T>
class CMapStringToStruct
{
private:
    string keys[];
    T values[];

public:
    void Insert(string key, const T &value)
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

    bool IsExist(string key)
    {
        return (FindIndex(key) != -1);
    }

    T At(string key)
    {
        int index = FindIndex(key);
        if (index != -1)
            return values[index];

        static T dummy;  // ✅ local static, allowed
        return dummy;
    }

    void Delete(string key)
    {
        int index = FindIndex(key);
        if (index != -1)
        {
            ArrayRemove(keys, index);
            ArrayRemove(values, index);
        }
    }

private:
    int FindIndex(string key)
    {
        for (int i = 0; i < ArraySize(keys); i++)
        {
            if (keys[i] == key)
                return i;
        }
        return -1;
    }
};

#endif // __MAP_STRING_MQH__
