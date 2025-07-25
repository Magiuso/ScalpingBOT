//+------------------------------------------------------------------+
//|                 MapLongT.mqh                                     |
//|     Mappa semplificata chiave long → struct                      |
//+------------------------------------------------------------------+

#ifndef __MAPLONGT_MQH__
#define __MAPLONGT_MQH__

template<typename T>
class CMapLongToStruct
{
private:
    long keys[];
    T values[];

public:
    void Insert(long key, const T &value)
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

    T At(long key)
    {
        int index = FindIndex(key);
        if (index != -1)
            return values[index];

        static T dummy;
        return dummy;
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

#endif // __MAPLONGT_MQH__

