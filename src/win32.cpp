#ifdef _WIN32

#include <windows.h>

static Arena *win32Arena;
static Win32Thread *win32FreeThread;
static u64 osPerformanceFrequency;

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

string OS_ReadFile(Arena *arena, string filename)
{
    HANDLE file = CreateFileA((char *)filename.str, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    Error(file != INVALID_HANDLE_VALUE, "Could not open file: %s\n", (char *)filename.str);
    u64 size;
    GetFileSizeEx(file, (LARGE_INTEGER *)&size);
    string result;
    result.str  = PushArray(arena, u8, size);
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

#endif
