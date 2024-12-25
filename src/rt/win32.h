namespace rt
{
struct ThreadContext;
#define THREAD_ENTRY_POINT(name) void name(void *parameter, ThreadContext *ctx)
typedef THREAD_ENTRY_POINT(OS_ThreadFunction);

struct OS_Handle
{
    u64 handle;
};

struct Win32Thread
{
    OS_ThreadFunction *func;
    void *ptr;

    HANDLE handle;
    Win32Thread *next;
};

struct PerformanceCounter
{
    u64 counter;
};

// struct OS_FileProperties
// {
//     string name;
//     u64 size;
//     u64 lastModified;
//     b32 isDirectory;
// };
//
// typedef u32 OS_FileIterFlags;
// enum
// {
//     OS_FileIterFlag_SkipDirectories = (1 << 0),
//     OS_FileIterFlag_SkipFiles       = (1 << 1),
//     OS_FileIterFlag_SkipHiddenFiles = (1 << 2),
//     OS_FileIterFlag_Complete        = (1 << 31),
// };
//
// struct OS_FileIter
// {
//     OS_FileIterFlags flags;
//     u8 memory[600];
// };

PerformanceCounter OS_StartCounter();
f32 OS_GetMilliseconds(PerformanceCounter counter);
u64 OS_GetCounts(PerformanceCounter counter);
u32 OS_NumProcessors();
void *OS_Reserve(u64 size, void *ptr = 0);
b8 OS_Commit(void *ptr, u64 size);
void OS_Release(void *memory);
void *OS_Alloc(u64 size, void *ptr = 0);
u64 OS_PageSize();
void OS_CreateWorkThread(OS_ThreadFunction func, void *parameter);
DWORD Win32ThreadProc(void *parameter);
Win32Thread *Win32GetFreeThread();
inline void Win32FreeThread(Win32Thread *thread);
void OS_SetThreadName(string name);
string OS_ReadFile(Arena *arena, string filename, u64 offset = 0);
b32 OS_WriteFile(string filename, void *fileMemory, u64 fileSize);
b32 OS_WriteFile(string filename, string buffer);
string OS_MapFileRead(string filename);
void OS_UnmapFile(void *ptr);
OS_Handle GetMainThreadHandle();
void OS_SetThreadAffinity(OS_Handle input, i32 index);
OS_Handle OS_ThreadStart(OS_ThreadFunction *func, void *ptr);
OS_Handle OS_CreateSemaphore(u32 maxCount);
void OS_ReleaseSemaphore(OS_Handle input);
void OS_ReleaseSemaphores(OS_Handle input, u32 count);
void OS_ThreadJoin(OS_Handle handle);
b32 OS_SignalWait(OS_Handle input);
void OS_Init();
} // namespace rt
