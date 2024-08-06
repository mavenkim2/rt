#ifdef _WIN32

#include <windows.h>

static Arena *win32Arena;
static Win32Thread *win32FreeThread;

u32 OS_NumProcessors()
{
    SYSTEM_INFO systemInfo;
    GetSystemInfo(&systemInfo);
    return systemInfo.dwNumberOfProcessors;
}

void *OS_Reserve(u64 size)
{
    void *ptr = VirtualAlloc(0, size, MEM_RESERVE, PAGE_READWRITE);
    return ptr;
}

b8 OS_Commit(void *ptr, u64 size)
{
    b8 result = (VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE) != 0);
    return result;
}

void OS_Release(void *memory)
{
    VirtualFree(memory, 0, MEM_RELEASE);
}

u64 OS_PageSize()
{
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwPageSize;
}

DWORD Win32ThreadProc(void *parameter)
{
    Win32Thread *thread = (Win32Thread *)parameter;
    BaseThreadEntry(thread->func, thread->ptr);
    return 0;
}

Win32Thread *Win32GetFreeThread()
{
    Win32Thread *thread = win32FreeThread;
    if (!thread)
    {
        thread = PushStruct(win32Arena, Win32Thread);
    }
    else
    {
        win32FreeThread = win32FreeThread->next;
    }
    return thread;
}

void OS_CreateWorkThread(OS_ThreadFunction func, void *parameter)
{
    DWORD threadID;
    Win32Thread *thread = Win32GetFreeThread();
    thread->func        = func;
    thread->ptr         = parameter;

    HANDLE threadHandle = CreateThread(0, 0, Win32ThreadProc, (void *)(thread), 0, &threadID);
    CloseHandle(threadHandle);
}

void OS_SetThreadName(char *name, u32 size)
{
    TempArena scratch = ScratchStart(0, 0);

    u32 resultSize     = (u32)(size);
    wchar_t *result    = (wchar_t *)PushArray(scratch.arena, u8, resultSize + 1);
    resultSize         = MultiByteToWideChar(CP_UTF8, 0, name, (i32)size, result, (i32)resultSize);
    result[resultSize] = 0;
    SetThreadDescription(GetCurrentThread(), result);

    ScratchEnd(scratch);
}

void OS_Init()
{
    win32Arena = ArenaAlloc();
}

inline u64 InterlockedAdd(u64 volatile *addend, u64 value)
{
    return InterlockedExchangeAdd64((volatile LONG64 *)addend, value);
}

#endif
