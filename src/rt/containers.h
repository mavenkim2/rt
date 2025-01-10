namespace rt
{
template <typename ElementType>
struct CheckedIterator
{
    CheckedIterator(ElementType *inPtr, i32 &inNum)
        : ptr(inPtr), initialNum(inNum), currentNum(inNum)
    {
    }
    FORCEINLINE ElementType &operator*() const { return *ptr; }
    FORCEINLINE CheckedIterator &operator++()
    {
        ++ptr;
        return *this;
    }
    FORCEINLINE bool operator!=(const CheckedIterator &rhs) const
    {
        Assert(initialNum == currentNum);
        return rhs.ptr != ptr;
    }

    i32 initialNum;
    i32 &currentNum;
    ElementType *ptr;
};

template <typename ElementType>
struct ReversedCheckedIterator
{
    ReversedCheckedIterator(ElementType *inPtr, i32 &inNum)
        : ptr(inPtr), initialNum(inNum), currentNum(inNum)
    {
    }
    FORCEINLINE ElementType &operator*() const { return *(ptr - 1); }
    FORCEINLINE ReversedCheckedIterator &operator++()
    {
        --ptr;
        return *this;
    }
    FORCEINLINE bool operator!=(const ReversedCheckedIterator &rhs) const
    {
        Assert(initialNum == currentNum);
        return rhs.ptr != ptr;
    }

    i32 initialNum;
    i32 &currentNum;
    ElementType *ptr;
};

template <typename T>
struct StaticArray
{
    StaticArray() : size(0), capacity(0), data(0) {}

    StaticArray(Arena *arena, i32 inCap) : size(0), capacity(inCap)
    {
        data = (T *)PushArray(arena, u8, sizeof(T) * capacity);
    }

    FORCEINLINE void Init(Arena *arena, i32 inCap)
    {
        size     = 0;
        capacity = inCap;
        data     = (T *)PushArray(arena, u8, sizeof(T) * capacity);
    }

    FORCEINLINE void Push(T element)
    {
        Assert(size < capacity);
        data[size] = element;
        size++;
    }

    FORCEINLINE T Pop()
    {
        Assert(size > 0);
        T result = std::move(data[size - 1]);
        size--;
        return result;
    }

    FORCEINLINE u32 Length() { return size; }

    FORCEINLINE b8 Empty() { return size != 0; }

    FORCEINLINE u32 Clear() { size = 0; }

    FORCEINLINE T &operator[](i32 index)
    {
        Assert(index >= 0 && index < size);
        return data[index];
    }

    FORCEINLINE const T &operator[](i32 index) const
    {
        Assert(index >= 0 && index < size);
        return data[index];
    }

    CheckedIterator<T> begin() { return CheckedIterator<T>(data, size); }
    CheckedIterator<T> begin() const { return CheckedIterator<const T>(data, size); }
    CheckedIterator<T> end() { return CheckedIterator<T>(data + size, size); }
    CheckedIterator<T> end() const { return CheckedIterator<const T>(data + size, size); }
    ReversedCheckedIterator<T> rbegin()
    {
        return ReversedCheckedIterator<T>(data + size, size);
    }
    ReversedCheckedIterator<T> rbegin() const
    {
        return ReversedCheckedIterator<const T>(data + size, size);
    }
    ReversedCheckedIterator<T> rend() { return ReversedCheckedIterator<T>(data, size); }
    ReversedCheckedIterator<T> rend() const
    {
        return ReversedCheckedIterator<const T>(data, size);
    }

    i32 size;
    i32 capacity;
    T *data;
};

template <typename T>
void Copy(StaticArray<T> &to, StaticArray<T> &from)
{
    Assert(to.capacity >= from.capacity && to.capacity >= from.size);
    MemoryCopy(to.data, from.data, sizeof(T) * from.size);
    to.size = from.size;
}

template <typename T, u32 capacity>
struct FixedArray
{
    FixedArray() : size(0) {}

    FORCEINLINE void Push(T element)
    {
        Assert(size < capacity);
        data[size] = element;
        size++;
    }

    FORCEINLINE void Add(T element) { Push(element); }

    FORCEINLINE void Emplace()
    {
        Assert(size < capacity);
        size += 1;
    }

    FORCEINLINE T &Back() { return data[size - 1]; }

    FORCEINLINE const T &Back() const { return data[size - 1]; }

    FORCEINLINE T Pop()
    {
        Assert(size > 0);
        T result = std::move(data[size - 1]);
        size--;
        return
    }

    FORCEINLINE u32 Length() const { return size; }

    FORCEINLINE b8 Empty() const { return size != 0; }

    FORCEINLINE u32 Clear() { size = 0; }

    FORCEINLINE u32 Capacity() const { return capacity; }

    FORCEINLINE void Resize(u32 inSize)
    {
        Assert(inSize <= capacity);
        size = inSize;
    }

    FORCEINLINE const T &operator[](i32 index) const
    {
        Assert(index >= 0 && index < size);
        return data[index];
    }

    FORCEINLINE T &operator[](i32 index)
    {
        Assert(index >= 0 && index < size);
        return data[index];
    }

    CheckedIterator<T> begin() { return CheckedIterator<T>(data, size); }
    CheckedIterator<T> begin() const { return CheckedIterator<const T>(data, size); }
    CheckedIterator<T> end() { return CheckedIterator<T>(data + size, size); }
    CheckedIterator<T> end() const { return CheckedIterator<const T>(data + size, size); }
    ReversedCheckedIterator<T> rbegin()
    {
        return ReversedCheckedIterator<T>(data + size, size);
    }
    ReversedCheckedIterator<T> rbegin() const
    {
        return ReversedCheckedIterator<const T>(data + size, size);
    }
    ReversedCheckedIterator<T> rend() { return ReversedCheckedIterator<T>(data, size); }
    ReversedCheckedIterator<T> rend() const
    {
        return ReversedCheckedIterator<const T>(data, size);
    }

    i32 size;
    T data[capacity];
};

// Basic dynamic array implementation
template <typename T, i32 tag = 0>
struct Array
{
    i32 size;
    i32 capacity;
    T *data;
    Arena *arena;

    // Array()
    // {
    //     size     = 0;
    //     capacity = 4;
    //     data     = (T *)Memory::Malloc(sizeof(T) * capacity);
    // }
    // Array(u32 initialCapacity) : capacity(initialCapacity)
    // {
    //     size = 0;
    //     data = (T *)Memory::Malloc(sizeof(T) * capacity);
    // }
    Array() {}
    Array(Arena *inArena) : arena(inArena)
    {
        arena    = inArena;
        size     = 0;
        capacity = 4;
#if TRACK_MEMORY
        data = (T *)PushArrayTagged(arena, u8, sizeof(T) * capacity, tag);
#else
        data = (T *)PushArray(arena, u8, sizeof(T) * capacity);
#endif
    }

    Array(Arena *inArena, u32 initialCapacity) : arena(inArena)
    {
        size     = 0;
        capacity = initialCapacity;
#if TRACK_MEMORY
        data = (T *)PushArrayTagged(arena, u8, sizeof(T) * capacity, tag);
#else
        data = (T *)PushArray(arena, u8, sizeof(T) * capacity);
#endif
    }

    inline void RangeCheck(i32 index) { Assert(index >= 0 && index < size); }

    inline T &operator[](i32 index)
    {
        RangeCheck(index);
        return data[index];
    }

    inline const T &operator[](i32 index) const
    {
        RangeCheck(index);
        return data[index];
    }

    inline void Reserve(const u32 num)
    {
        capacity = num;
#if TRACK_MEMORY
        T *newData = (T *)PushArrayTagged(arena, u8, sizeof(T) * capacity, tag);
#else
        T *newData = (T *)PushArray(arena, u8, sizeof(T) * capacity);
#endif
        MemoryCopy(newData, data, sizeof(T) * size);
        data = newData;
    }

    inline void Resize(const u32 num) { AddOrGrow(num - size); }

    u32 Length() const { return size; }

    b8 Empty() const { return size == 0; }

    void Emplace() { AddOrGrow(1); }

    T &Back() { return data[size - 1]; }

    const T &Back() const { return data[size - 1]; }

    void Add(T &&element)
    {
        AddOrGrow(1);
        data[size - 1] = std::move(element);
    }

    void Add(T &element)
    {
        AddOrGrow(1);
        data[size - 1] = element;
    }

    T &AddBack()
    {
        Emplace();
        return Back();
    }

    void Append(Array<T> &other)
    {
        AddOrGrow(other.size);
        MemoryCopy(data + size - 1, other.data, sizeof(T) * other.size);
    }

    void Push(T &element) { Add(element); }

    void Push(T &&element) { Add(std::move(element)); }

    void Remove(u32 index)
    {
        RangeCheck(index);
        (index != size - 1)
            ? MemoryCopy(data + index + 1, data + index, sizeof(T) * (size - index))
            : 0;
        size--;
    }

    void RemoveSwapBack(u32 index)
    {
        RangeCheck(index);
        (index != size - 1) ? data[index] = std::move(data[size - 1]) : 0;
        size--;
    }

    void Clear() { size = 0; }

    CheckedIterator<T> begin() { return CheckedIterator<T>(data, size); }
    CheckedIterator<T> end() { return CheckedIterator<T>(data + size, size); }
    CheckedIterator<T> rbegin() { return CheckedIterator<T, true>(data + size, size); }
    CheckedIterator<T> rend() { return CheckedIterator<T, true>(data, size); }

private:
    inline void AddOrGrow(i32 num)
    {
        if (size + num > capacity)
        {
            capacity = (size + num) * 2;
#if TRACK_MEMORY
            T *newData = (T *)PushArrayTagged(arena, u8, sizeof(T) * capacity, tag);
#else
            T *newData = (T *)PushArray(arena, u8, sizeof(T) * capacity);
#endif
            MemoryCopy(newData, data, sizeof(T) * size);
            data = newData;
        }
        size += num;
    }
};

struct HashIndex
{
    Arena *arena;
    i32 *hash;
    i32 *indexChain;
    i32 hashSize;
    i32 indexChainSize;
    i32 hashMask;

    HashIndex(Arena *arena, i32 inHashSize, i32 inChainSize) : arena(arena)
    {
        hashSize = inHashSize;
        Assert((hashSize & (hashSize - 1)) == 0); // pow 2
        indexChainSize = inChainSize;
        hashMask       = inHashSize - 1;
        hash           = PushArray(arena, i32, inHashSize);
        MemorySet(hash, 0xff, sizeof(hash[0]) * hashSize);
        indexChain = PushArray(arena, i32, inChainSize);
        MemorySet(indexChain, 0xff, sizeof(indexChain[0]) * indexChainSize);
    }

    HashIndex(Arena *arena) : HashIndex(arena, 1024, 1024) {}

    i32 FirstInHash(i32 inHash)
    {
        i32 result = hash[inHash & hashMask];
        return result;
    }

    i32 NextInHash(i32 in)
    {
        i32 result = indexChain[in];
        return result;
    }

    template <typename T>
    inline void AddInHash(T obj, i32 index)
    {
        i32 inHash = (i32)Hash<T>(obj);
        AddInHash(inHash, index);
    }

    void AddInHash(i32 key, i32 index)
    {
        i32 slot = key & hashMask;
        if (index > indexChainSize)
        {
            i32 *newIndexChain = PushArray(arena, i32, indexChainSize * 2);
            MemoryCopy(newIndexChain, indexChain, sizeof(i32) * indexChainSize);
            indexChainSize *= 2;
            indexChain = newIndexChain;
        }
        else
        {
            indexChain[index] = hash[slot];
            hash[slot]        = index;
        }
    }

    b32 RemoveFromHash(i32 key, i32 index)
    {
        b32 result = 0;
        i32 slot   = key & hashMask;
        if (hash[slot] == index)
        {
            hash[slot] = -1;
            result     = 1;
        }
        else
        {
            for (i32 i = hash[slot]; i != -1; i = indexChain[i])
            {
                if (indexChain[i] == index)
                {
                    indexChain[i] = indexChain[index];
                    result        = 1;
                    break;
                }
            }
        }
        indexChain[index] = -1;
        return result;
    }
};

template <typename Key, typename Value>
struct HashExt
{
    Key *keys;
    Key invalidKey;
    Value *values;
    u32 num;
    u64 mask;

    HashExt() {}
    HashExt(Arena *arena, u32 inNum, Key invalidKey) : invalidKey(invalidKey)
    {
        num = (1 << (Bsr(inNum) + 1));
        Assert(IsPow2(num) && num > inNum);
        mask = num - 1;
        keys = (Key *)PushArrayNoZero(arena, u8, sizeof(Key) * num);
        for (u32 i = 0; i < num; i++)
        {
            keys[i] = invalidKey;
        }
        values = (Value *)PushArrayNoZero(arena, u8, sizeof(Value) * num);
    }
    void Create(const Key &key, const Value &value)
    {
        u64 hash       = Hash(key);
        u64 index      = hash & mask;
        u64 startIndex = index;
        do
        {
            Assert(keys[index] != key);
            if (keys[index] == invalidKey)
            {
                keys[index]   = key;
                values[index] = value;
                return;
            }
            index = (index + 1) & mask;
        } while (index != startIndex);

        Assert(!"hash table overflowed");
    }
    bool Find(const Key &key, Value &out) const
    {
        u64 hash       = Hash(key);
        u64 index      = hash & mask;
        u64 startIndex = index;
        do
        {
            if (keys[index] == invalidKey) return false;
            if (keys[index] == key)
            {
                out = values[index];
                return true;
            }
            index = (index + 1) & mask;
        } while (index != startIndex);

        return false;
    }
};

template <typename T, i32 numSlots, i32 chunkSize, i32 numStripes, i32 tag>
struct HashSet
{
    StaticAssert(IsPow2(numSlots), CachePow2N);
    struct ChunkNode
    {
        T values[chunkSize];
        u32 count;
        ChunkNode *next;
    };
    ChunkNode *nodes;
    Mutex *mutexes;

    HashSet() {}
    HashSet(Arena *arena)
    {
        nodes   = PushArrayTagged(arena, ChunkNode, numSlots, tag);
        mutexes = PushArrayTagged(arena, Mutex, numStripes, tag);
    }
    const T *GetOrCreate(Arena *arena, T value);
};

template <typename T, i32 numSlots, i32 chunkSize, i32 numStripes, i32 tag>
const T *HashSet<T, numSlots, chunkSize, numStripes, tag>::GetOrCreate(Arena *arena, T value)
{
    u64 hash        = Hash<T>(value);
    ChunkNode *node = &nodes[hash & (numSlots - 1)];
    ChunkNode *prev = 0;

    u32 stripe = hash & (numStripes - 1);
    BeginRMutex(&mutexes[stripe]);
    while (node)
    {
        for (u32 i = 0; i < node->count; i++)
        {
            if (node->values[i] == value)
            {
                EndRMutex(&mutexes[stripe]);
                return &node->values[i];
            }
        }
        prev = node;
        node = node->next;
    }
    EndRMutex(&mutexes[stripe]);

    T *out = 0;
    BeginWMutex(&mutexes[stripe]);
    if (prev->count == ArrayLength(prev->values))
    {
        node       = PushStructTagged(arena, ChunkNode, tag);
        prev->next = node;
        prev       = node;
    }
    prev->values[prev->count] = value;
    out                       = &prev->values[prev->count++];
    EndWMutex(&mutexes[stripe]);
    return out;
}

template <typename T, i32 numPerChunk, i32 memoryTag = 0>
struct ChunkedLinkedList
{
    Arena *arena;
    struct ChunkNode
    {
        T values[numPerChunk];
        u32 count;
        ChunkNode *next;
    };
    ChunkNode *first;
    ChunkNode *last;
    u32 totalCount;

    // struct Iterator
    // {
    //     ChunkNode *node;
    //     u32 localIndex;
    //     u32 numRemaining;
    //
    //     bool End() { return numRemaining == 0; }
    //
    //     void Next()
    //     {
    //         localIndex++;
    //         numRemaining--;
    //         if (localIndex = numPerChunk)
    //         {
    //             node = node->next;
    //         }
    //     }
    //     T *Get() { return node->values[localIndex]; }
    // };
    //
    // Iterator Itr(u32 start = 0, u32 end = totalCount)
    // {
    //     Iterator itr;
    //     itr.node         = first;
    //     itr.numRemaining = end - start;
    //     u32 index        = start;
    //     while (index > numPerChunk)
    //     {
    //         itr.node = itr.node->next;
    //         index -= numPerChunk;
    //     }
    //     itr.localIndex = index;
    //     return itr;
    // }

    ChunkedLinkedList(Arena *arena) : arena(arena), first(0), last(0), totalCount(0)
    {
        AddNode();
    }
    T &AddBack()
    {
        if (last->count >= numPerChunk)
        {
            AddNode();
        }
        T &result = last->values[last->count++];
        totalCount++;
        return result;
    }
    T &operator[](u32 i)
    {
        Assert(i < totalCount);
        ChunkNode *node = first;
        for (;;)
        {
            Assert(node);
            if (i >= node->count)
            {
                i -= node->count;
                node = node->next;
            }
            else
            {
                return node->values[i];
            }
        }
    }
    inline void Push(const T &val) { AddBack() = val; }
    inline void Push(const T &&val) { AddBack() = std::move(val); }
    inline const T &Last() const { return last->values[last->count - 1]; }
    inline T &Last() { return last->values[last->count - 1]; }

    inline void AddNode()
    {
        ChunkNode *newNode = PushStructTagged(arena, ChunkNode, memoryTag);
        QueuePush(first, last, newNode);
    }
    inline u32 Length() const { return totalCount; }
    inline void Flatten(T *out)
    {
        T *ptr = out;
        for (ChunkNode *node = first; node != 0; node = node->next)
        {
            MemoryCopy(ptr, node->values, node->count * sizeof(T));
            ptr += node->count;
        }
    }
    inline void Flatten(StaticArray<T> &array)
    {
        u32 runningCount = 0;
        for (ChunkNode *node = first; node != 0; node = node->next)
        {
            Assert(runningCount + node->count < (u32)array.capacity);
            MemoryCopy(array.data + runningCount, node->values, node->count * sizeof(T));
            runningCount += node->count;
        }
    }
    inline void Merge(ChunkedLinkedList<T, numPerChunk, memoryTag> *list)
    {
        if (list->totalCount)
        {
            Assert(list->first && last);
            last->next = list->first;
            last       = list->last;
            Assert(last->next == 0);
            totalCount += list->totalCount;
        }
    }
};

template <typename T>
struct HashMap
{
    struct HashNode
    {
        u32 hash;
        T value;

        HashNode *next;
    };

    struct HashList
    {
        HashNode *first;
        HashNode *last;
    };

    HashList *map;
    u32 count;
    u32 hashMask;

    HashMap(Arena *arena, u32 count) : count(count)
    {
        map = PushArray(arena, HashList, count);
        Assert(IsPow2(count));
        u32 hashMask = count - 1;
    }

    T &Get(string name)
    {
        u32 hash       = Hash(name);
        HashList &list = map[hash & hashMask];
        HashNode *node = list.first;
        while (node)
        {
            if (node->hash == hash)
            {
                if (name == node->value) return node->value;
            }
            node = node->next;
        }
        if (!node)
        {
            Error(0, "Name not found in hashmap\n");
        }
        return 0;
    }

    void Add(Arena *arena, T &val)
    {
        u32 hash            = val.Hash(); // Hash(packet->name);
        SceneHashList &list = map[hash & hashMask];
        SceneHashNode *node = list.first;
        while (node)
        {
            if (node->hash == hash)
            {
                if (node->value == val)
                {
                    Error(0, "Error: Using a duplicate name.\n");
                }
                else
                {
                    Error(0, "Hash collision\n");
                }
            }
            node = node->next;
        }
        Assert(!node);

        node       = PushStruct(arena, SceneHashNode);
        node->hash = hash;
        node->val  = val;

        QueuePush(list.first, list.last, node);
    }

    void Merge(HashMap<T> &from)
    {
        Assert(from.count == count);
        for (u32 i = 0; i < size; i++)
        {
            HashList *toList   = &map[i];
            HashList *fromList = &from.map[i];
            if (toList->first == 0)
            {
                toList->first = fromList->first;
            }
            else
            {
                Assert(toList->last);
                toList->last->next = fromList->first;
                if (fromList->last) toList->last = fromList->last;
            }
        }
    }
};

} // namespace rt
