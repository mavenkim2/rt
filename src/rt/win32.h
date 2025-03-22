#ifndef RT_WIN32_H
#define RT_WIN32_H

#include "base.h"
#include "memory.h"
#include "string.h"
#include "containers.h"

namespace rt
{

struct Win32Thread
{
    OS_ThreadFunction *func;
    void *ptr;

    HANDLE handle;
    Win32Thread *next;
};

PerformanceCounter OS_StartCounter();
f32 OS_GetMilliseconds(PerformanceCounter counter);
u64 OS_GetCounts(PerformanceCounter counter);
u32 OS_NumProcessors();
void *OS_Reserve(u64 size, void *ptr = 0);
void *OS_ReserveLarge(u64 size);
b8 OS_Commit(void *ptr, u64 size);
void OS_Release(void *memory);
void *OS_Alloc(u64 size, void *ptr = 0);
u64 OS_PageSize();
size_t OS_GetLargePageSize();
void OS_CreateWorkThread(OS_ThreadFunction func, void *parameter);
DWORD Win32ThreadProc(void *parameter);
Win32Thread *Win32GetFreeThread();
inline void Win32FreeThread(Win32Thread *thread);
void OS_SetThreadName(string name);
OS_Handle OS_CreateFile(string filename);
bool OS_CloseFile(OS_Handle handle);
u64 OS_GetFileSize(string filename);
u64 OS_GetFileSize2(string filename);
bool OS_ReadFile(OS_Handle handle, void *buffer, size_t size, u64 offset = 0);
string OS_ReadFile(Arena *arena, string filename, u64 offset = 0);
b32 OS_WriteFile(string filename, void *fileMemory, u64 fileSize);
b32 OS_WriteFile(string filename, string buffer);
bool OS_DeleteFile(string filename);
string OS_MapFileRead(string filename);
u8 *OS_MapFileWrite(string filename, u64 size);
u8 *OS_MapFileAppend(string filename, u64 size);
bool OS_UnmapFile(void *ptr);
void OS_ResizeFile(string filename, u64 size);
void OS_FlushMappedFile(void *ptr, size_t size);
bool OS_DirectoryExists(string filename);
bool OS_FileExists(string filename);
bool OS_CreateDirectory(string filename);
OS_Handle GetMainThreadHandle();
void OS_SetThreadAffinity(OS_Handle input, i32 index);
OS_Handle OS_ThreadStart(OS_ThreadFunction *func, void *ptr);
OS_Handle OS_CreateSemaphore(u32 maxCount);
void OS_ReleaseSemaphore(OS_Handle input);
void OS_ReleaseSemaphores(OS_Handle input, u32 count);
void OS_ThreadJoin(OS_Handle handle);
b32 OS_SignalWait(OS_Handle input);
void OS_Init();
f32 OS_NowSeconds();
void OS_Sleep(u32 ms);
void OS_GetMousePos(OS_Handle handle, u32 &x, u32 &y);

ChunkedLinkedList<OS_Event> &OS_GetEvents();

LRESULT Win32_Callback(HWND window, UINT message, WPARAM wParam, LPARAM lParam);
OS_Handle OS_WindowInit(int width = 0, int height = 0);
} // namespace rt

#endif
