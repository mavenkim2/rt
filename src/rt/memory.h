#ifndef MEMORY_H
#define MEMORY_H

#define ARENA_HEADER_SIZE              128
#define ARENA_COMMIT_SIZE              kilobytes(64)
#define ARENA_RESERVE_SIZE             megabytes(64)
#define ARENA_RESERVE_SIZE_LARGE_PAGES megabytes(8)
#define ARENA_COMMIT_SIZE_LARGE_PAGES  megabytes(2)

namespace rt
{

struct Arena
{
    struct Arena *prev;
    struct Arena *current;
    u64 basePos;
    u64 pos;
    u64 cmt;
    u64 res;
    u64 align;
    b32 largePagesEnabled;
    b8 grow;
};

struct TempArena
{
    Arena *arena;
    u64 pos;
};

Arena *ArenaAlloc(u64 resSize, u64 cmtSize, u64 align, b32 largePages = 0);
Arena *ArenaAlloc(u64 align);
Arena *ArenaAlloc();
void *ArenaPushNoZero(Arena *arena, u64 size);
void *ArenaPush(Arena *arena, u64 size);
u64 ArenaPos(Arena *arena);
void ArenaPopTo(Arena *arena, u64 pos);
void ArenaPopToZero(Arena *arena, u64 pos);
TempArena TempBegin(Arena *arena);
void TempEnd(TempArena temp);
b32 CheckZero(u32 size, u8 *instance);
void ArenaRelease(Arena *arena);
void ArenaClear(Arena *arena);

#define PushArrayNoZero(arena, type, count) (type *)ArenaPushNoZero(arena, sizeof(type) * (count))
#define PushStructNoZero(arena, type)       PushArrayNoZero(arena, type, 1)
#define PushArray(arena, type, count)       (type *)ArenaPush(arena, sizeof(type) * (count))
#define PushStruct(arena, type)             PushArray(arena, type, 1)

enum MemoryType
{
    MemoryType_File,
    MemoryType_Shape,
    MemoryType_Material,
    MemoryType_Texture,
    MemoryType_Light,
    MemoryType_Instance,
    MemoryType_Transform,
    MemoryType_String,
    MemoryType_BVH,
    MemoryType_Record,
    MemoryType_Other,
};

template <typename T>
__forceinline T *PushArrayDefault(Arena *arena, u32 count)
{
    T *out = (T *)PushArray(arena, u8, sizeof(T) * count);
    for (u32 i = 0; i < count; i++)
    {
        out[i] = T();
    }
    return out;
}

#if TRACK_MEMORY
#define PushArrayTagged(arena, type, count, tag) \
    (((u64 *)(&threadMemoryStatistics[GetThreadIndex()]))[tag] += sizeof(type) * count, PushArray(arena, type, count))
#define PushArrayNoZeroTagged(arena, type, count, tag) \
    (((u64 *)(&threadMemoryStatistics[GetThreadIndex()]))[tag] += sizeof(type) * count, PushArrayNoZero(arena, type, count))

#define PushStructTagged(arena, type, tag) \
    (((u64 *)(&threadMemoryStatistics[GetThreadIndex()]))[tag] += sizeof(type), PushStruct(arena, type))

#define PushStructNoZeroTagged(arena, type, tag) \
    (((u64 *)(&threadMemoryStatistics[GetThreadIndex()]))[tag] += sizeof(type), PushStructNoZero(arena, type))

template <typename T>
__forceinline T *PushArrayDefaultTagged(Arena *arena, u32 count, MemoryType tag)
{
    ((u64 *)(&threadMemoryStatistics[GetThreadIndex()]))[tag] += sizeof(T);
    T *out = (T *)PushArray(arena, u8, sizeof(T) * count);
    for (u32 i = 0; i < count; i++)
    {
        out[i] = T();
    }
    return out;
}
#else
#define PushArrayTagged(arena, type, count, tag)       PushArray(arena, type, count)
#define PushArrayNoZeroTagged(arena, type, count, tag) PushArrayNoZero(arena, type, count)
#define PushStructTagged(arena, type, tag)             PushStruct(arena, type)
#define PushStructNoZeroTagged(arena, type, tag)       PushStructNoZero(arena, type)

template <typename T>
__forceinline T *PushArrayDefaultTagged(Arena *arena, u32 count, MemoryType type)
{
    return PushArrayDefault<T>(arena, count);
}
#endif

#define PushStructConstruct(arena, Type) new (PushStructNoZero(arena, Type)) Type

#define ScratchEnd(temp) TempEnd(temp)

#define IsZero(instance) CheckZero(sizeof(instance), (u8 *)(&instance))
} // namespace rt

#endif
