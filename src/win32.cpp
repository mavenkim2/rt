#ifdef _WIN32
/**/
#include <windows.h>
namespace rt
{

static Arena *win32Arena;
static Win32Thread *win32FreeThread;
static u64 osPerformanceFrequency;
static b32 win32LargePagesEnabled;

PerformanceCounter OS_StartCounter()
{
    PerformanceCounter counter;
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    counter.counter = c.QuadPart;
    return counter;
}

f32 OS_GetMilliseconds(PerformanceCounter counter)
{
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    f32 result = (f32)(1000.f * (c.QuadPart - counter.counter)) / (osPerformanceFrequency);
    return result;
}

u64 OS_GetCounts(PerformanceCounter counter)
{
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    u64 result = 1000 * (c.QuadPart - counter.counter);
    return result;
}

f32 OS_GetMilliseconds(u64 count)
{
    return (f32)count / osPerformanceFrequency;
}

u32 OS_NumProcessors()
{
    SYSTEM_INFO systemInfo;
    GetSystemInfo(&systemInfo);
    return systemInfo.dwNumberOfProcessors;
}

void *OS_Reserve(u64 size, void *ptr)
{
    void *out = VirtualAlloc(ptr, size, MEM_RESERVE, PAGE_READWRITE);
    return out;
}

void *OS_ReserveLarge(u64 size)
{
    void *out = VirtualAlloc(0, size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
    return out;
}

b32 OS_SetLargePages()
{
    b32 is_ok = 0;
    HANDLE token;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token))
    {
        LUID luid;
        if (LookupPrivilegeValue(0, SE_LOCK_MEMORY_NAME, &luid))
        {
            TOKEN_PRIVILEGES priv;
            priv.PrivilegeCount           = 1;
            priv.Privileges[0].Luid       = luid;
            priv.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
            if (AdjustTokenPrivileges(token, 0, &priv, sizeof(priv), 0, 0))
            {
                win32LargePagesEnabled = 1;
                is_ok                  = 1;
            }
        }
        CloseHandle(token);
    }
    return is_ok;
}

size_t OS_GetLargePageSize()
{
    return GetLargePageMinimum();
}

b32 OS_LargePagesEnabled()
{
    return win32LargePagesEnabled;
}

b8 OS_Commit(void *ptr, u64 size)
{
    b8 result = (VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE) != 0);
    return result;
}

void *OS_Alloc(u64 size, void *ptr)
{
    void *out = VirtualAlloc(ptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    return out;
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

inline void Win32FreeThread(Win32Thread *thread)
{
    StackPush(win32FreeThread, thread);
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

void OS_SetThreadName(string name)
{
    TempArena scratch = ScratchStart(0, 0);

    u32 resultSize     = (u32)(name.size);
    wchar_t *result    = (wchar_t *)PushArray(scratch.arena, u8, resultSize + 1);
    resultSize         = MultiByteToWideChar(CP_UTF8, 0, (char *)name.str, (i32)(name.size), result, (i32)resultSize);
    result[resultSize] = 0;
    SetThreadDescription(GetCurrentThread(), result);

    ScratchEnd(scratch);
}

string OS_ReadFile(Arena *arena, string filename)
{
    HANDLE file = CreateFileA((char *)filename.str, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    Error(file != INVALID_HANDLE_VALUE, "Could not open file: %S\n", filename);
    u64 size;
    GetFileSizeEx(file, (LARGE_INTEGER *)&size);
    string result;
    result.str  = PushArrayTagged(arena, u8, size, MemoryType_File);
    result.size = size;

    u64 totalReadSize = 0;
    for (; totalReadSize < size;)
    {
        u64 readAmount = size - totalReadSize;
        u32 sizeToRead = (readAmount > 0xffffffff) ? 0xffffffff : (u32)readAmount;
        DWORD readSize = 0;
        if (!ReadFile(file, (u8 *)result.str + totalReadSize, sizeToRead, &readSize, 0)) break;
        totalReadSize += readSize;
        if (readSize != sizeToRead) break;
    }
    Assert(totalReadSize == size);
    return result;
}

b32 OS_WriteFile(string filename, void *fileMemory, u64 fileSize)
{
    HANDLE fileHandle = CreateFileA((char *)filename.str, GENERIC_WRITE, 0, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    u64 totalWritten  = 0;
    if (fileHandle != INVALID_HANDLE_VALUE)
    {
        for (; totalWritten < fileSize;)
        {
            DWORD bytesToWrite = (DWORD)Min(fileSize - totalWritten, 0xffffffffull);
            DWORD bytesWritten;
            if (!WriteFile(fileHandle, (u8 *)fileMemory, bytesToWrite, &bytesWritten, NULL)) break;
            totalWritten += bytesWritten;
        }
        CloseHandle(fileHandle);
    }
    return totalWritten == fileSize;
}

b32 OS_WriteFile(string filename, string buffer)
{
    return OS_WriteFile(filename, buffer.str, (u32)buffer.size);
}

string OS_MapFileRead(string filename)
{
    HANDLE file = CreateFileA((char *)filename.str, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    Error(file != INVALID_HANDLE_VALUE, "Could not open file: %S\n", filename);
    u64 size;
    GetFileSizeEx(file, (LARGE_INTEGER *)&size);

    HANDLE mapping = CreateFileMapping(file, 0, PAGE_READONLY, 0, 0, 0);
    CloseHandle(file);
    Error(mapping != INVALID_HANDLE_VALUE, "Could not map file: %S\n", filename);

    LPVOID ptr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);

    string result;
    result.str  = (u8 *)ptr;
    result.size = size;
    return result;
}

void OS_UnmapFile(void *ptr)
{
    UnmapViewOfFile(ptr);
}

OS_Handle GetMainThreadHandle()
{
    OS_Handle out = {(u64)GetCurrentThread()};
    return out;
}

void OS_SetThreadAffinity(OS_Handle input, i32 index)
{
    HANDLE handle  = (HANDLE)input.handle;
    DWORD_PTR mask = 1ull << index;
    SetThreadAffinityMask(handle, mask);
}

OS_Handle OS_ThreadStart(OS_ThreadFunction *func, void *ptr)
{
    Win32Thread *thread = Win32GetFreeThread();
    thread->func        = func;
    thread->ptr         = ptr;
    thread->handle      = CreateThread(0, 0, Win32ThreadProc, thread, 0, 0);
    OS_Handle handle    = {(u64)thread};
    return handle;
}

OS_Handle OS_CreateSemaphore(u32 maxCount)
{
    HANDLE handle    = CreateSemaphoreEx(0, 0, maxCount, 0, 0, SEMAPHORE_ALL_ACCESS);
    OS_Handle result = {(u64)handle};
    return result;
}

void OS_ReleaseSemaphore(OS_Handle input)
{
    HANDLE handle = (HANDLE)input.handle;
    ReleaseSemaphore(handle, 1, 0);
}

void OS_ReleaseSemaphores(OS_Handle input, u32 count)
{
    HANDLE handle = (HANDLE)input.handle;
    ReleaseSemaphore(handle, count, 0);
}

void OS_ThreadJoin(OS_Handle handle)
{
    Win32Thread *thread = (Win32Thread *)handle.handle;
    if (thread && thread->handle && thread->handle != INVALID_HANDLE_VALUE)
    {
        WaitForSingleObject(thread->handle, INFINITE);
        CloseHandle(thread->handle);
    }
    Win32FreeThread(thread);
}

b32 OS_SignalWait(OS_Handle input)
{
    HANDLE handle = (HANDLE)input.handle;
    DWORD result  = WaitForSingleObject(handle, U32Max);
    Assert(result == WAIT_OBJECT_0 || result == WAIT_TIMEOUT);
    return (result == WAIT_OBJECT_0);
}

void OS_Yield()
{
    SwitchToThread();
}

void OS_Init()
{
    win32Arena = ArenaAlloc();
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    osPerformanceFrequency = frequency.QuadPart;
}

inline u64 InterlockedAdd(u64 volatile *addend, u64 value)
{
    return InterlockedExchangeAdd64((volatile LONG64 *)addend, value);
}

} // namespace rt
#endif
