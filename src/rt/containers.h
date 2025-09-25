#ifndef CONTAINERS_H_
#define CONTAINERS_H_

#include <initializer_list>
#include <array>

#include "atomics.h"
#include "base.h"
#include "math/basemath.h"
#include "memory.h"
#include "string.h"
#include "thread_statistics.h"

namespace rt
{
template <typename ElementType>
struct CheckedIterator
{
    CheckedIterator(ElementType *inPtr) : ptr(inPtr) {}
    __forceinline ElementType &operator*() const { return *ptr; }
    __forceinline CheckedIterator &operator++()
    {
        ++ptr;
        return *this;
    }
    __forceinline bool operator!=(const CheckedIterator &rhs) const { return rhs.ptr != ptr; }

    ElementType *ptr;
};

template <typename ElementType>
struct ReversedCheckedIterator
{
    ReversedCheckedIterator(ElementType *inPtr, i32 &inNum)
        : ptr(inPtr), initialNum(inNum), currentNum(inNum)
    {
    }
    __forceinline ElementType &operator*() const { return *(ptr - 1); }
    __forceinline ReversedCheckedIterator &operator++()
    {
        --ptr;
        return *this;
    }
    __forceinline bool operator!=(const ReversedCheckedIterator &rhs) const
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
    StaticArray() : size_(0), capacity(0), data(0) {}

    StaticArray(Arena *arena, i32 inCap) : size_(0), capacity(inCap)
    {
        data = (T *)PushArray(arena, u8, sizeof(T) * capacity);
    }
    StaticArray(Arena *arena, i32 inCap, i32 size) : size_(size), capacity(inCap)
    {
        data = (T *)PushArray(arena, u8, sizeof(T) * capacity);
    }
    StaticArray(Arena *arena, std::vector<T> &vector)
        : size_((int)vector.size()), capacity((int)vector.size())
    {
        data = (T *)PushArrayNoZero(arena, u8, sizeof(T) * size_);
        MemoryCopy(data, vector.data(), sizeof(T) * size_);
    }
    StaticArray(T *buffer, u32 size) : data(buffer), capacity(size), size_(size) {}

    __forceinline void Init(Arena *arena, i32 inCap)
    {
        size_    = 0;
        capacity = inCap;
        data     = (T *)PushArray(arena, u8, sizeof(T) * capacity);
    }

    __forceinline void Push(T element)
    {
        ErrorExit(size_ < capacity, "%u %u\n", size_, capacity);
        data[size_] = element;
        size_++;
    }

    __forceinline bool PushUnique(T element)
    {
        for (int i = 0; i < size_; i++)
        {
            if (data[i] == element) return false;
        }
        Push(element);
        return true;
    }

    __forceinline void push_back(T element) { Push(element); }

    __forceinline T Pop()
    {
        Assert(size_ > 0);
        T result = std::move(data[size_ - 1]);
        size_--;
        return result;
    }

    __forceinline T &Last() { return data[size_ - 1]; }

    __forceinline u32 Length() const { return size_; }
    __forceinline u32 size() const { return Length(); }
    __forceinline i32 &size() { return size_; }

    __forceinline b8 Empty() { return size_ != 0; }

    __forceinline void Clear() { size_ = 0; }

    __forceinline void Resize(int s) { size_ = s; }

    __forceinline T &operator[](i32 index)
    {
        ErrorExit(index >= 0 && index < size_, "index: %u, size %u\n", index, size_);
        return data[index];
    }

    __forceinline const T &operator[](i32 index) const
    {
        Assert(index >= 0 && index < size_);
        return data[index];
    }

    CheckedIterator<T> begin() { return CheckedIterator<T>(data); }
    CheckedIterator<const T> begin() const { return CheckedIterator<const T>(data); }
    CheckedIterator<T> end() { return CheckedIterator<T>(data + size_); }
    CheckedIterator<const T> end() const { return CheckedIterator<const T>(data + size_); }
    ReversedCheckedIterator<T> rbegin()
    {
        return ReversedCheckedIterator<T>(data + size_, size_);
    }
    ReversedCheckedIterator<T> rbegin() const
    {
        return ReversedCheckedIterator<const T>(data + size_, size_);
    }
    ReversedCheckedIterator<T> rend() { return ReversedCheckedIterator<T>(data, size_); }
    ReversedCheckedIterator<T> rend() const
    {
        return ReversedCheckedIterator<const T>(data, size_);
    }

    i32 size_;
    i32 capacity;
    T *data;
};

template <typename T>
void Copy(StaticArray<T> &to, StaticArray<T> &from)
{
    Assert(to.capacity >= from.size_);
    MemoryCopy(to.data, from.data, sizeof(T) * from.size_);
    to.size_ = from.size_;
}

template <typename T>
void Copy(StaticArray<T> &to, std::vector<T> &from)
{
    Assert(to.capacity >= from.size());
    MemoryCopy(to.data, from.data(), sizeof(T) * from.size());
    to.size_ = (int)from.size();
}

template <typename T, u32 capacity>
struct FixedArray
{
    i32 size;
    T data[capacity];
    FixedArray() : size(0) {}
    FixedArray(int count) : size(count) { Assert(count <= capacity); }
    FixedArray(std::initializer_list<T> list) : size((int)list.size())
    {
        Assert(list.size() <= capacity);
        std::array<T, capacity> ar;
        std::copy(list.begin(), list.end(), ar.begin());
        MemoryCopy(data, ar.data(), sizeof(T) * capacity);
    }

    __forceinline void Push(T element)
    {
        Assert(size < capacity);
        data[size] = element;
        size++;
    }

    __forceinline void Add(T element) { Push(element); }

    __forceinline void Emplace()
    {
        Assert(size < capacity);
        size += 1;
    }

    __forceinline T &Back() { return data[size - 1]; }

    __forceinline const T &Back() const { return data[size - 1]; }

    __forceinline T Pop()
    {
        Assert(size > 0);
        T result = std::move(data[size - 1]);
        size--;
        return result;
    }

    __forceinline u32 Length() const { return size; }

    __forceinline b8 Empty() const { return size != 0; }

    __forceinline void Clear() { size = 0; }

    __forceinline u32 Capacity() const { return capacity; }

    __forceinline void Resize(u32 inSize)
    {
        Assert(inSize <= capacity);
        size = inSize;
    }

    __forceinline const T &operator[](i32 index) const
    {
        Assert(index >= 0 && index < size);
        return data[index];
    }

    __forceinline T &operator[](i32 index)
    {
        Assert(index >= 0 && index < size);
        return data[index];
    }

    CheckedIterator<T> begin() { return CheckedIterator<T>(data); }
    CheckedIterator<T> begin() const { return CheckedIterator<const T>(data); }
    CheckedIterator<T> end() { return CheckedIterator<T>(data + size); }
    CheckedIterator<T> end() const { return CheckedIterator<const T>(data + size); }
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

    inline void RangeCheck(i32 index) const { Assert(index >= 0 && index < size); }

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

    bool AddUnique(T element)
    {
        for (int i = 0; i < size; i++)
        {
            if (data[i] == element)
            {
                return false;
            }
        }
        Add(element);
        return true;
    }

    void Append(Array<T> &other)
    {
        AddOrGrow(other.size);
        MemoryCopy(data + size - 1, other.data, sizeof(T) * other.size);
    }

    void Push(T &element) { Add(element); }
    __forceinline T Pop()
    {
        Assert(size > 0);
        T result = std::move(data[size - 1]);
        size--;
        return result;
    }

    __forceinline bool PushUnique(T &element)
    {
        for (int i = 0; i < size; i++)
        {
            if (data[i] == element) return false;
        }
        Push(element);
        return true;
    }

    __forceinline bool PushUnique(T &&element)
    {
        for (int i = 0; i < size; i++)
        {
            if (data[i] == element) return false;
        }
        Push(element);
        return true;
    }

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

    CheckedIterator<T> begin() { return CheckedIterator<T>(data); }
    CheckedIterator<T> end() { return CheckedIterator<T>(data + size); }

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

    void AddConcurrent(int key, int index)
    {
        Assert(index < indexChainSize);
        key &= hashMask;

        indexChain[index] = AtomicExchange(&hash[key], index);
    }

    b32 RemoveFromHash(i32 key, i32 index)
    {
        i32 slot = key & hashMask;
        if (hash[slot] == index)
        {
            hash[slot] = indexChain[index];
            return true;
        }
        else
        {
            for (i32 i = hash[slot]; i != -1; i = indexChain[i])
            {
                if (indexChain[i] == index)
                {
                    indexChain[i] = indexChain[index];
                    return true;
                }
            }
        }
        return false;
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

template <typename T>
struct SimpleHashSet
{
    struct Node
    {
        T value;
        Node *next;
    };
    Node *nodes;
    u32 numSlots;
    u32 totalCount;

    SimpleHashSet() {}
    SimpleHashSet(Arena *arena, u32 numSlots) : numSlots(numSlots), totalCount(0)
    {
        Assert(IsPow2(numSlots));
        nodes = PushArray(arena, Node, numSlots);
    }
    const bool AddUnique(Arena *arena, u32 hash, T value);
};

template <typename T>
const bool SimpleHashSet<T>::AddUnique(Arena *arena, u32 hash, T value)
{
    Node *node = &nodes[hash & (numSlots - 1)];
    Node *prev = 0;

    while (node)
    {
        if (node->value == value)
        {
            return false;
        }
        prev = node;
        node = node->next;
    }

    node        = PushStruct(arena, Node);
    prev->value = value;
    prev->next  = node;
    totalCount++;

    return true;
}

template <typename T, i32 memoryTag = 0>
struct ChunkedLinkedList
{
    Arena *arena;
    struct ChunkNode
    {
        T *values;
        u32 count;
        u32 cap;
        ChunkNode *next;

        T &operator[](int index)
        {
            Assert(index < count);
            return values[index];
        }
        const T &operator[](int index) const
        {
            Assert(index < count);
            return values[index];
        }
    };
    ChunkNode *first;
    ChunkNode *last;
    u32 totalCount;
    u32 numPerChunk;

    ChunkedLinkedList() {}
    ChunkedLinkedList(Arena *arena)
        : arena(arena), first(0), last(0), totalCount(0), numPerChunk(512)
    {
    }
    ChunkedLinkedList(Arena *arena, u32 numPerChunk)
        : arena(arena), first(0), last(0), totalCount(0), numPerChunk(numPerChunk)
    {
    }
    T &AddBack()
    {
        if (!last || last->count >= last->cap)
        {
            AddNode();
        }
        T &result = last->values[last->count++];
        totalCount++;
        return result;
    }
    void AddBack(T &&val)
    {
        if (!last || last->count >= last->cap)
        {
            AddNode();
        }
        Assert(last && last->values);
        Assert(last->count < last->cap);
        last->values[last->count++] = std::move(val);
        totalCount++;
    }

    inline void Push(const T &val) { AddBack() = val; }
    inline void Push(const T &&val) { AddBack() = std::move(val); }

    inline void Pop(T *result)
    {
        if (last && last->count)
        {
            *result = last->values[--last->count];
            totalCount--;
        }
        else if (first != last)
        {
            ChunkNode *prev = first;
            ChunkNode *node = first;
            while (node != last)
            {
                prev = node;
                node = node->next;
            }
            last = prev;
            Pop(result);
        }
        else
        {
            *result = {};
        }
    }
    inline const T &Last() const { return last->values[last->count - 1]; }
    inline T &Last() { return last->values[last->count - 1]; }

    inline void AddNode()
    {
        if (last && last->next)
        {
            last = last->next;
        }
        else
        {
            ChunkNode *newNode = PushStructTagged(arena, ChunkNode, memoryTag);
            newNode->values =
                (T *)PushArrayNoZeroTagged(arena, u8, sizeof(T) * numPerChunk, memoryTag);
            newNode->cap = numPerChunk;
            QueuePush(first, last, newNode);
        }
    }

    inline ChunkNode *AddNode(int count)
    {
        ChunkNode *newNode = PushStructTagged(arena, ChunkNode, memoryTag);
        newNode->values    = (T *)PushArrayTagged(arena, u8, sizeof(T) * count, memoryTag);
        newNode->count     = count;
        newNode->cap       = count;
        if (last && last->next)
        {
            newNode->next = last->next;
            last->next    = newNode;
            last          = newNode;
        }
        else
        {
            QueuePush(first, last, newNode);
        }
        totalCount += count;
        return newNode;
    }

    inline u32 Length() const { return totalCount; }
    inline void Flatten(T *out) const
    {
        T *ptr = out;
        for (ChunkNode *node = first; node != 0; node = node->next)
        {
            MemoryCopy(ptr, node->values, node->count * sizeof(T));
            ptr += node->count;
        }
    }
    inline void Flatten(StaticArray<T> &array) const
    {
        u32 runningCount = 0;
        for (ChunkNode *node = first; node != 0; node = node->next)
        {
            Assert(runningCount + node->count <= (u32)array.capacity);
            MemoryCopy(array.data + runningCount, node->values, node->count * sizeof(T));
            runningCount += node->count;
        }
        array.size = runningCount;
    }
    inline void Merge(ChunkedLinkedList<T, memoryTag> *list)
    {
        if (list->totalCount)
        {
            if (!first)
            {
                Assert(!last);
                Assert(totalCount == 0);
                first = list->first;
                last  = list->last;
                totalCount += list->totalCount;
            }
            else
            {
                Assert(list->first && last);
                last->next = list->first;
                last       = list->last;
                Assert(last->next == 0);
                totalCount += list->totalCount;
            }
        }
    }
    void Clear()
    {
        totalCount = 0;
        for (auto *node = first; node != 0; node = node->next)
        {
            node->count = 0;
        }
        last = first;
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
        hashMask = count - 1;
    }

    const T *Get(string name)
    {
        u32 hash       = Hash(name);
        HashList &list = map[hash & hashMask];
        HashNode *node = list.first;
        while (node)
        {
            if (node->hash == hash)
            {
                if (node->value == name) return &node->value;
            }
            node = node->next;
        }
        if (!node)
        {
            ErrorExit(0, "Name not found in hashmap: %S\n", name);
        }
        return 0;
    }

    T *Find(const T &value)
    {
        u32 hash       = value.Hash();
        HashList &list = map[hash & hashMask];
        HashNode *node = list.first;
        while (node)
        {
            if (node->hash == hash)
            {
                if (node->value == value)
                {
                    return &node->value;
                }
            }
            node = node->next;
        }
        Assert(!node);
        return 0;
    }

    void Add(Arena *arena, const T &val, bool errorDuplicate = true)
    {
        u32 hash       = val.Hash();
        HashList &list = map[hash & hashMask];
        HashNode *node = list.first;
        while (node)
        {
            if (node->hash == hash)
            {
                if (node->value == val)
                {
                    ErrorExit(errorDuplicate, "Error: Using a duplicate name.\n");
                    return;
                }
            }
            node = node->next;
        }
        Assert(!node);

        node        = PushStruct(arena, HashNode);
        node->hash  = hash;
        node->value = val;

        QueuePush(list.first, list.last, node);
    }

    void Merge(HashMap<T> &from)
    {
        Assert(from.count == count);
        for (u32 i = 0; i < count; i++)
        {
            HashList *toList   = &map[i];
            HashList *fromList = &from.map[i];
            if (fromList->first == 0) continue;

            if (toList->first == 0)
            {
                toList->first = fromList->first;
                toList->last  = fromList->last;
            }
            else
            {
                toList->last->next = fromList->first;
                toList->last       = fromList->last;
            }
        }
    }
};

struct AtomicHashIndex
{
    static const u32 invalidIndex = 0xffffffff;
    std::atomic<u8> *locks;
    int *hash;
    int *nextIndex;
    int hashCount;
    int indexCount;

    AtomicHashIndex(Arena *arena, int hashSize);

    void Clear();
    void BeginLock(int lockIndex);
    void EndLock(int lockIndex);

    int First(int key) const;
    int Next(int index) const;

    void Add(int key, int index);
    void AddConcurrent(int key, int index);

    template <typename Predicate>
    inline int Find(int inHash, Predicate &predicate) const;

    template <typename Predicate>
    inline int FindConcurrent(int hash, Predicate &predicate) const;
};

inline AtomicHashIndex::AtomicHashIndex(Arena *arena, int hashSize)
{
    hashCount  = NextPowerOfTwo(hashSize);
    indexCount = hashCount;

    hash      = PushArrayNoZero(arena, int, hashCount);
    nextIndex = PushArrayNoZero(arena, int, indexCount);
    locks     = PushArray(arena, std::atomic<u8>, hashCount);
    Clear();
}

inline void AtomicHashIndex::Clear() { MemorySet(hash, 0xff, sizeof(int) * hashCount); }

inline void AtomicHashIndex::BeginLock(int lockIndex)
{
    u8 val = 0;
    while (!locks[lockIndex].compare_exchange_weak(val, 1, std::memory_order_acquire))
    {
        val = 0;
        _mm_pause();
    }
}

inline void AtomicHashIndex::EndLock(int lockIndex)
{
    locks[lockIndex].store(0, std::memory_order_release);
}

inline int AtomicHashIndex::First(int key) const
{
    key &= (hashCount - 1);
    return hash[key];
}

inline int AtomicHashIndex::Next(int index) const { return nextIndex[index]; }

inline void AtomicHashIndex::AddConcurrent(int key, int index)
{
    Assert(index < indexCount);
    key &= (hashCount - 1);
    BeginLock(key);
    nextIndex[index] = hash[key];
    hash[key]        = index;
    EndLock(key);
}

inline void AtomicHashIndex::Add(int key, int index)
{
    Assert(index < indexCount);
    key &= (hashCount - 1);
    nextIndex[index] = hash[key];
    hash[key]        = index;
}

template <typename Predicate>
inline int AtomicHashIndex::Find(int inHash, Predicate &predicate) const
{
    int key = inHash & (hashSize - 1);
    for (int i = First(key); i != invalidIndex; i = Next(i))
    {
        if (predicate(i))
        {
            return i;
        }
    }
    return invalidIndex;
}

template <typename Predicate>
inline int AtomicHashIndex::FindConcurrent(int inHash, Predicate &predicate) const
{
    int key   = inHash & (hashSize - 1);
    int index = invalidIndex;

    BeginLock(key);
    for (int i = First(key); i != invalidIndex; i = Next(i))
    {
        if (predicate(i))
        {
            index = i;
            break;
        }
    }
    EndLock(key);
    return index;
}

template <typename T>
struct Graph
{
    u32 *offsets;
    T *data;

    Graph() : offsets(0), data(0) {}

    template <typename F>
    u32 InitializeStatic(Arena *arena, u32 itrCount, u32 count, F &&func)
    {
        offsets       = PushArray(arena, u32, count + 1);
        u32 *offsets1 = &offsets[1];
        u32 total     = 0;
        for (u32 index = 0; index < itrCount; index++)
        {
            total += func(index, offsets1);
        }

        u32 out = total;

        data = PushArrayNoZero(arena, T, total);

        total = 0;
        for (u32 index = 0; index < count; index++)
        {
            u32 num         = offsets1[index];
            offsets1[index] = total;
            total += num;
        }

        for (u32 index = 0; index < itrCount; index++)
        {
            func(index, offsets1, data);
        }

        return total;
    }

    template <typename F>
    u32 InitializeStatic(Arena *arena, u32 count, F &&func)
    {
        return InitializeStatic(arena, count, count, func);
    }
};

template <typename T>
struct ArrayView
{
    T *data;
    u32 num;

    ArrayView() : data(0), num(0) {}
    ArrayView(Array<T> &array, u32 offset, u32 num) : num(num)
    {
        Assert(offset + num <= array.Length());
        data = array.data + offset;
    }
    ArrayView(Array<T> &array) : num(array.Length()) { data = array.data; }
    ArrayView(StaticArray<T> &array, u32 offset, u32 num) : num(num)
    {
        Assert(offset + num <= array.Length());
        data = array.data + offset;
    }
    ArrayView(StaticArray<T> &array) : num(array.Length()) { data = array.data; }
    ArrayView(ArrayView<T> &view, u32 offset, u32 num) : num(num)
    {
        Assert(offset + num <= view.num);
        data = view.data + offset;
    }
    ArrayView(T *data, u32 num) : data(data), num(num) {}
    T &operator[](u32 index)
    {
        Assert(index < num);
        return data[index];
    }
    const T &operator[](u32 index) const
    {
        Assert(index < num);
        return data[index];
    }
    u32 Length() const { return num; }

    void Copy(StaticArray<T> &array)
    {
        Assert(array.capacity >= num);
        MemoryCopy(array.data, data, sizeof(T) * num);
        array.size() = num;
    }

    CheckedIterator<T> begin() { return CheckedIterator<T>(data); }
    CheckedIterator<T> begin() const { return CheckedIterator<const T>(data); }
    CheckedIterator<T> end() { return CheckedIterator<T>(data + num); }
    CheckedIterator<T> end() const { return CheckedIterator<const T>(data + num); }
};

} // namespace rt
#endif
