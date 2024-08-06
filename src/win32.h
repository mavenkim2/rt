struct ThreadContext;
#define THREAD_ENTRY_POINT(name) void name(void *parameter, ThreadContext *ctx)
typedef THREAD_ENTRY_POINT(OS_ThreadFunction);

struct Win32Thread
{
    OS_ThreadFunction *func;
    void *ptr;

    Win32Thread *next;
};

u32 OS_NumProcessors();
void *OS_Reserve(u64 size);
b8 OS_Commit(void *ptr, u64 size);
void OS_Release(void *memory);
u64 OS_PageSize();
void OS_CreateWorkThread(OS_ThreadFunction func, void *parameter);
DWORD Win32ThreadProc(void *parameter);
Win32Thread *Win32GetFreeThread();
void OS_SetThreadName(char *name, u32 size);
void OS_Init();
