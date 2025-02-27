#include "thread_context.h"

namespace rt
{
thread_local ThreadContext *tLocalContext;
static const u64 threadArenaAlign = 32;

void InitThreadContext(Arena *arena, const char *name, b32 isMainThread)
{
    ThreadContext *tctx = PushStruct(arena, ThreadContext);
    if (isMainThread)
    {
        tctx->index = 0;
    }
    InitThreadContext(tctx, 1);
    SetThreadName(name);
}

void InitThreadContext(ThreadContext *t, b32 isMainThread)
{
    for (u32 i = 0; i < t->numScratchArenas; i++)
    {
        t->arenas[i] = ArenaAlloc(threadArenaAlign);
    }
    t->isMainThread = isMainThread;
    tLocalContext   = t;
}

void ReleaseThreadContext()
{
    ThreadContext *t = GetThreadContext();
    for (u32 i = 0; i < t->numScratchArenas; i++)
    {
        ArenaRelease(t->arenas[i]);
    }
}

ThreadContext *GetThreadContext() { return tLocalContext; }

void SetThreadContext(ThreadContext *tctx) { tLocalContext = tctx; }

Arena *GetThreadContextScratch(Arena **conflicts, u32 count)
{
    ThreadContext *t = GetThreadContext();
    Arena *result    = 0;
    for (u32 i = 0; i < t->numScratchArenas; i++)
    {
        Arena *arenaPtr = t->arenas[i];
        b32 hasConflict = 0;
        for (u32 j = 0; j < count; j++)
        {
            Arena *conflictPtr = conflicts[j];
            if (arenaPtr == conflictPtr)
            {
                hasConflict = 1;
                break;
            }
        }
        if (!hasConflict)
        {
            result = arenaPtr;
            return result;
        }
    }
    return result;
}

void SetThreadName(string name)
{
    ThreadContext *context = GetThreadContext();
    context->threadNameSize =
        name.size < sizeof(context->threadName) ? name.size : sizeof(context->threadName);
    MemoryCopy(context->threadName, name.str, context->threadNameSize);
    OS_SetThreadName(name);
}

void SetThreadIndex(u32 index)
{
    ThreadContext *context = GetThreadContext();
    context->index         = index;
}

u32 GetThreadIndex()
{
    ThreadContext *context = GetThreadContext();
    u32 result             = context->index;
    return result;
}

void BaseThreadEntry(OS_ThreadFunction *func, void *params)
{
    ThreadContext tContext_ = {};
    InitThreadContext(&tContext_);
    func(params, &tContext_);
    ReleaseThreadContext();
}
} // namespace rt
