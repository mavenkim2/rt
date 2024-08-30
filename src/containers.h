template <typename ElementType>
struct CheckedIterator
{
    CheckedIterator(ElementType *inPtr, i32 &inNum) : ptr(inPtr), initialNum(inNum), currentNum(inNum) {}
    FORCEINLINE ElementType &operator*() const
    {
        return *ptr;
    }
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
    ReversedCheckedIterator(ElementType *inPtr, i32 &inNum) : ptr(inPtr), initialNum(inNum), currentNum(inNum) {}
    FORCEINLINE ElementType &operator*() const
    {
        return *(ptr - 1);
    }
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
    StaticArray() : size(0), capacity(0), data(0)
    {
    }

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

    FORCEINLINE u32 Length()
    {
        return size;
    }

    FORCEINLINE b8 Empty()
    {
        return size != 0;
    }

    FORCEINLINE u32 Clear()
    {
        size = 0;
    }

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
    ReversedCheckedIterator<T> rbegin() { return ReversedCheckedIterator<T>(data + size, size); }
    ReversedCheckedIterator<T> rbegin() const { return ReversedCheckedIterator<const T>(data + size, size); }
    ReversedCheckedIterator<T> rend() { return ReversedCheckedIterator<T>(data, size); }
    ReversedCheckedIterator<T> rend() const { return ReversedCheckedIterator<const T>(data, size); }

    i32 size;
    i32 capacity;
    T *data;
};

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

    FORCEINLINE void Add(T element)
    {
        Push(element);
    }

    FORCEINLINE void Emplace()
    {
        Assert(size < capacity);
        size += 1;
    }

    FORCEINLINE T &Back()
    {
        return data[size - 1];
    }

    FORCEINLINE const T &Back() const
    {
        return data[size - 1];
    }

    FORCEINLINE T Pop()
    {
        Assert(size > 0);
        T result = std::move(data[size - 1]);
        size--;
        return
    }

    FORCEINLINE u32 Length() const
    {
        return size;
    }

    FORCEINLINE b8 Empty() const
    {
        return size != 0;
    }

    FORCEINLINE u32 Clear()
    {
        size = 0;
    }

    FORCEINLINE u32 Capacity() const
    {
        return capacity;
    }

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
    ReversedCheckedIterator<T> rbegin() { return ReversedCheckedIterator<T>(data + size, size); }
    ReversedCheckedIterator<T> rbegin() const { return ReversedCheckedIterator<const T>(data + size, size); }
    ReversedCheckedIterator<T> rend() { return ReversedCheckedIterator<T>(data, size); }
    ReversedCheckedIterator<T> rend() const { return ReversedCheckedIterator<const T>(data, size); }

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
        data       = (T *)PushArray(arena, u8, sizeof(T) * capacity);
#endif
    }

    Array(Arena *inArena, u32 initialCapacity) : arena(inArena)
    {
        size     = 0;
        capacity = initialCapacity;
#if TRACK_MEMORY
        data = (T *)PushArrayTagged(arena, u8, sizeof(T) * capacity, tag);
#else
        data       = (T *)PushArray(arena, u8, sizeof(T) * capacity);
#endif
    }

    inline void RangeCheck(i32 index)
    {
        Assert(index >= 0 && index < size);
    }

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

    inline void Resize(const u32 num)
    {
        AddOrGrow(num - size);
    }

    const u32 Length()
    {
        return size;
    }

    const b8 Empty()
    {
        return size == 0;
    }

    void Emplace()
    {
        AddOrGrow(1);
    }

    T &Back()
    {
        return data[size - 1];
    }

    const T &Back() const
    {
        return data[size - 1];
    }

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

    void Push(T &element)
    {
        Add(element);
    }

    void Push(T &&element)
    {
        Add(std::move(element));
    }

    void Remove(u32 index)
    {
        RangeCheck(index);
        (index != size - 1) ? MemoryCopy(data + index + 1, data + index, sizeof(T) * (size - index)) : 0;
        size--;
    }

    void RemoveSwapBack(u32 index)
    {
        RangeCheck(index);
        (index != size - 1) ? data[index] = std::move(data[size - 1]) : 0;
        size--;
    }

    void Clear()
    {
        size = 0;
    }

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

#if 0 
template <u32 hashSize, u32 indexSize>
struct AtomicFixedHashTable
{
    std::atomic<u8> locks[hashSize]; // TODO: maybe could have "stripes", which would save memory but some buckets
                                     // would be locked even if they are unused
    u32 hash[hashSize];
    u32 nextIndex[hashSize];

    AtomicFixedHashTable();
    void Clear();
    void BeginLock(u32 lockIndex);
    void EndLock(u32 lockIndex);
    u32 First(u32 key) const;
    u32 FirstConcurrent(u32 key);
    u32 Next(u32 index) const;
    b8 IsValid(u32 index) const;
    b8 IsValidConcurrent(u32 key, u32 index);

    void Add(u32 key, u32 index);
    void AddConcurrent(u32 key, u32 index);
    void RemoveConcurrent(u32 key, u32 index);

    u32 Find(const u32 inHash, const u32 *array) const;

    template <typename T>
    u32 Find(const u32 inHash, const T *array, const T element) const;
};

template <u32 hashSize, u32 indexSize>
inline AtomicFixedHashTable<hashSize, indexSize>::AtomicFixedHashTable()
{
    StaticAssert(indexSize < 0xffffffff, IndexIsNotU32Max);
    StaticAssert((hashSize & (hashSize - 1)) == 0, HashIsPowerOfTwo);
    Clear();
}

template <u32 hashSize, u32 indexSize>
inline b8 AtomicFixedHashTable<hashSize, indexSize>::IsValid(u32 index) const
{
    return index != 0xffffffff;
}

template <u32 hashSize, u32 indexSize>
inline b8 AtomicFixedHashTable<hashSize, indexSize>::IsValidConcurrent(u32 key, u32 index)
{
    if (index == 0xffffffff)
    {
        key &= (hashSize - 1);
        EndLock(key);
    }
    return index != 0xffffffff;
}

template <u32 hashSize, u32 indexSize>
inline void AtomicFixedHashTable<hashSize, indexSize>::Clear()
{
    MemorySet(hash, 0xff, sizeof(hash));
    MemorySet(locks, 0, sizeof(locks));
}

template <u32 hashSize, u32 indexSize>
inline void AtomicFixedHashTable<hashSize, indexSize>::BeginLock(u32 lockIndex)
{
    u8 val = 0;
    while (!locks[lockIndex].compare_exchange_weak(val, val + 1))
    {
        _mm_pause();
    }
}

template <u32 hashSize, u32 indexSize>
inline void AtomicFixedHashTable<hashSize, indexSize>::EndLock(u32 lockIndex)
{
    locks[lockIndex].store(0);
}

template <u32 hashSize, u32 indexSize>
inline u32 AtomicFixedHashTable<hashSize, indexSize>::FirstConcurrent(u32 key)
{
    key &= (hashSize - 1);
    BeginLock(key);
    return hash[key];
}

template <u32 hashSize, u32 indexSize>
inline u32 AtomicFixedHashTable<hashSize, indexSize>::First(u32 key) const
{
    key &= (hashSize - 1);
    return hash[key];
}

template <u32 hashSize, u32 indexSize>
inline u32 AtomicFixedHashTable<hashSize, indexSize>::Next(u32 index) const
{
    return nextIndex[index];
}

template <u32 hashSize, u32 indexSize>
inline void AtomicFixedHashTable<hashSize, indexSize>::AddConcurrent(u32 key, u32 index)
{
    Assert(index < indexSize);
    key &= (hashSize - 1);
    BeginLock(key);
    nextIndex[index] = hash[key];
    hash[key]        = index;
    EndLock(key);
}

template <u32 hashSize, u32 indexSize>
inline void AtomicFixedHashTable<hashSize, indexSize>::Add(u32 key, u32 index)
{
    Assert(index < indexSize);
    key &= (hashSize - 1);
    nextIndex[index] = hash[key];
    hash[key]        = index;
}

template <u32 hashSize, u32 indexSize>
inline void AtomicFixedHashTable<hashSize, indexSize>::RemoveConcurrent(u32 key, u32 index)
{
    key &= (hashSize - 1);
    BeginLock(key);
    if (hash[key] == index)
    {
        hash[key] = nextIndex[index];
    }
    else
    {
        for (u32 i = hash[key]; IsValidLock(key, i); i = Next(i))
        {
            if (nextIndex[i] == index)
            {
                nextIndex[i] = nextIndex[index];
                break;
            }
        }
    }
}

template <u32 hashSize, u32 indexSize>
inline u32 AtomicFixedHashTable<hashSize, indexSize>::Find(const u32 inHash, const u32 *array) const
{
    u32 index = 0xffffffff;
    u32 key   = inHash & (hashSize - 1);
    for (u32 i = First(key); IsValid(i); i = Next(i))
    {
        if (array[i] == inHash)
        {
            index = i;
            break;
        }
    }
    return index;
}

template <u32 hashSize, u32 indexSize>
template <typename T>
inline u32 AtomicFixedHashTable<hashSize, indexSize>::Find(const u32 inHash, const T *array, const T element) const
{
    u32 index = 0xffffffff;
    u32 key   = inHash & (hashSize - 1);
    for (u32 i = First(key); IsValid(i); i = Next(i))
    {
        if (array[i] == element)
        {
            index = i;
            break;
        }
    }
    return index;
}
#endif

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
