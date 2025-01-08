#ifndef THREAD_CONTEXT_H
#define THREAD_CONTEXT_H
namespace rt
{

struct ThreadContext
{
    Arena *arenas[2];
    u8 threadName[64];
    u64 threadNameSize;

    b32 isMainThread;
    u32 index;
};

void InitThreadContext(Arena *arena, const char *name, b32 isMainThread = 0);
void InitThreadContext(ThreadContext *t, b32 isMainThread = 0);
void ReleaseThreadContext();
ThreadContext *GetThreadContext();
void SetThreadContext(ThreadContext *tctx);
Arena *GetThreadContextScratch(Arena **conflicts, u32 count);
void SetThreadName(string name);
void SetThreadIndex(u32 index);
u32 GetThreadIndex();
void BaseThreadEntry(OS_ThreadFunction *func, void *params);

#define ScratchStart(conflicts, count) TempBegin(GetThreadContextScratch((conflicts), (count)))

struct ScratchArena
{
    TempArena temp;
    ScratchArena() { temp = ScratchStart(0, 0); }
    ~ScratchArena() { ScratchEnd(temp); }
};

} // namespace rt

#endif
