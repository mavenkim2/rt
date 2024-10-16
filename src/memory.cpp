namespace rt
{
Arena *ArenaAlloc(u64 resSize, u64 cmtSize, u64 align, b32 largePages)
{
    void *memory          = 0;
    b32 largePagesEnabled = OS_LargePagesEnabled();
    b32 largePagesFailed  = 0;
    if (largePagesEnabled && largePages)
    {
#ifdef _WIN32

        size_t largePageSize = OS_GetLargePageSize();
        resSize              = AlignPow2(resSize, largePageSize);
        memory               = OS_ReserveLarge(resSize);
        largePagesFailed     = memory == 0;
#else
#error os not supported
#endif
    }
    if (!(largePagesEnabled && largePages) || largePagesFailed)
    {
        largePagesEnabled = 0;
        u64 pageSize      = OS_PageSize();
        resSize           = AlignPow2(resSize, pageSize);
        cmtSize           = AlignPow2(cmtSize, pageSize);
        memory            = OS_Reserve(resSize);
        if (!OS_Commit(memory, cmtSize))
        {
            memory = 0;
            OS_Release(memory);
        }
    }

    Assert(memory);
    Arena *arena = (Arena *)memory;
    if (arena)
    {
        arena->prev              = 0;
        arena->current           = arena;
        arena->basePos           = 0;
        arena->pos               = ARENA_HEADER_SIZE;
        arena->cmt               = cmtSize;
        arena->res               = resSize;
        arena->align             = align;
        arena->grow              = 1;
        arena->largePagesEnabled = largePagesEnabled;
    }

    return arena;
}

Arena *ArenaAlloc(u64 align)
{
    Arena *result = ArenaAlloc(ARENA_RESERVE_SIZE, ARENA_COMMIT_SIZE, align);
    return result;
}

Arena *ArenaAlloc()
{
    Arena *result = ArenaAlloc(ARENA_RESERVE_SIZE, ARENA_COMMIT_SIZE, 8);
    return result;
}

Arena *ArenaAllocLargePages()
{
    Arena *result = ArenaAlloc(ARENA_RESERVE_SIZE_LARGE_PAGES, megabytes(2), 8, 1);
    return result;
}

void *ArenaPushNoZero(Arena *arena, u64 size)
{
    Arena *current      = arena->current;
    u64 currentAlignPos = AlignPow2(current->pos, current->align);
    u64 newPos          = currentAlignPos + size;
    if (current->res < newPos && current->grow)
    {
        Arena *newArena = 0;
        if (current->largePagesEnabled)
        {
            if (size < ARENA_RESERVE_SIZE_LARGE_PAGES / 2 + 1)
            {
                newArena = ArenaAllocLargePages();
            }
            else
            {
                u64 newBlockSize = size + ARENA_HEADER_SIZE;
                newArena         = ArenaAlloc(newBlockSize, ARENA_COMMIT_SIZE_LARGE_PAGES, arena->align, 1);
            }
        }
        else
        {
            if (size < ARENA_RESERVE_SIZE / 2 + 1)
            {
                newArena = ArenaAlloc();
            }
            else
            {
                u64 newBlockSize = size + ARENA_HEADER_SIZE;
                newArena         = ArenaAlloc(newBlockSize, ARENA_COMMIT_SIZE, arena->align);
            }
        }
        if (newArena)
        {
            newArena->basePos = current->basePos + current->res;
            newArena->prev    = current;
            newArena->align   = current->align;
            arena->current    = newArena;
            current           = newArena;
            currentAlignPos   = AlignPow2(current->pos, current->align);
            newPos            = currentAlignPos + size;
        }
        else
        {
            assert(!"Arena alloc failed");
        }
    }
    if (current->cmt < newPos)
    {
        if (!current->largePagesEnabled)
        {
            u64 cmtAligned = AlignPow2(newPos, ARENA_COMMIT_SIZE);
            cmtAligned     = Min(cmtAligned, current->res);
            u64 cmtSize    = cmtAligned - current->cmt;
            b8 result      = OS_Commit((u8 *)current + current->cmt, cmtSize);
            assert(result);
            current->cmt = cmtAligned;
        }
        else
        {
            u64 cmtAligned = AlignPow2(newPos, ARENA_COMMIT_SIZE_LARGE_PAGES);
            cmtAligned     = Min(cmtAligned, current->res);
            u64 cmtSize    = cmtAligned - current->cmt;
            current->cmt   = cmtAligned;
        }
    }
    void *result = 0;
    if (current->cmt >= newPos)
    {
        result       = (u8 *)current + currentAlignPos;
        current->pos = newPos;
    }
    if (result == 0)
    {
        assert(!"Allocation failed");
    }
    return result;
}

void *ArenaPush(Arena *arena, u64 size)
{
    void *result = ArenaPushNoZero(arena, size);
    MemoryZero(result, size);
    return result;
}

void ArenaPopTo(Arena *arena, u64 pos)
{
    pos            = std::max<u64>(ARENA_HEADER_SIZE, pos);
    Arena *current = arena->current;
    for (Arena *prev = 0; current->basePos >= pos; current = prev)
    {
        prev = current->prev;
        OS_Release(current);
    }
    assert(current);
    arena->current = current;
    u64 newPos     = pos - current->basePos;
    assert(newPos <= current->pos);
    current->pos = newPos;
}

void ArenaPopToZero(Arena *arena, u64 pos)
{
    pos            = std::max<u64>(ARENA_HEADER_SIZE, pos);
    Arena *current = arena->current;
    for (Arena *prev = 0; current->basePos >= pos; current = prev)
    {
        prev = current->prev;
        OS_Release(current);
    }
    assert(current);
    u64 newPos = pos - current->basePos;
    assert(newPos <= current->pos);
    current->pos = newPos;
}

u64 ArenaPos(Arena *arena)
{
    Arena *current = arena->current;
    u64 pos        = current->basePos + current->pos;
    return pos;
}

TempArena TempBegin(Arena *arena)
{
    u64 pos        = ArenaPos(arena);
    TempArena temp = {arena, pos};
    return temp;
}

void TempEnd(TempArena temp)
{
    ArenaPopTo(temp.arena, temp.pos);
}

b32 CheckZero(u32 size, u8 *instance)
{
    b32 result = true;
    while (size-- > 0)
    {
        if (*instance++)
        {
            result = false;
            break;
        }
    }
    return result;
}

void ArenaClear(Arena *arena)
{
    ArenaPopTo(arena, ARENA_HEADER_SIZE);
}

void ArenaRelease(Arena *arena)
{
    for (Arena *a = arena->current, *prev = 0; a != 0; a = prev)
    {
        prev = a->prev;
        OS_Release(a);
    }
}
} // namespace rt
