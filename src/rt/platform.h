#ifndef PLATFORM_H
#define PLATFORM_H

#include <cstring>
#include "base.h"

namespace rt
{
struct ThreadContext;
#define THREAD_ENTRY_POINT(name) void name(void *parameter, ThreadContext *ctx)
typedef THREAD_ENTRY_POINT(OS_ThreadFunction);

struct OS_Handle
{
    u64 handle;
};
struct PerformanceCounter
{
    u64 counter;
};

enum OS_Key
{
    OS_Mouse_L,
    OS_Mouse_R,
    OS_Key_A,
    OS_Key_B,
    OS_Key_C,
    OS_Key_D,
    OS_Key_E,
    OS_Key_F,
    OS_Key_G,
    OS_Key_H,
    OS_Key_I,
    OS_Key_J,
    OS_Key_K,
    OS_Key_L,
    OS_Key_M,
    OS_Key_N,
    OS_Key_O,
    OS_Key_P,
    OS_Key_Q,
    OS_Key_R,
    OS_Key_S,
    OS_Key_T,
    OS_Key_U,
    OS_Key_V,
    OS_Key_W,
    OS_Key_X,
    OS_Key_Y,
    OS_Key_Z,
    OS_Key_Space,
    OS_Key_Shift,
    OS_Key_F1,
    OS_Key_F2,
    OS_Key_F3,
    OS_Key_F4,
    OS_Key_F5,
    OS_Key_F6,
    OS_Key_F7,
    OS_Key_F8,
    OS_Key_F9,
    OS_Key_F10,
    OS_Key_F11,
    OS_Key_F12,
    OS_Key_Count,
};

enum OS_EventType
{
    OS_EventType_Quit,
    OS_EventType_KeyPressed,
    OS_EventType_KeyReleased,
    OS_EventType_LoseFocus,
};

struct OS_Event
{
    OS_EventType type;

    // key info
    OS_Key key;
    b32 transition;

    // mouse click info
    f32 posX;
    f32 posY;
};

struct OS_Events
{
    OS_Event *events;
    u32 numEvents;
};

struct OS_EventChunk
{
    OS_EventChunk *next;
    OS_Event *events;

    u32 count;
    u32 cap;
};

struct OS_EventList
{
    OS_EventChunk *first;
    OS_EventChunk *last;
    u32 numEvents;
};

} // namespace rt

#if _WIN32
#include <windows.h>
#include "win32.h"
#else
#error sorry, os not supported
#endif

// #include <stdio.h>
// #include <string.h>
// #include <cstdlib>

#endif
