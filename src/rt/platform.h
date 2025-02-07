#ifndef PLATFORM_H
#define PLATFORM_H

#include <cstring>

namespace rt
{
struct ThreadContext;
#define THREAD_ENTRY_POINT(name) void name(void *parameter, ThreadContext *ctx)
typedef THREAD_ENTRY_POINT(OS_ThreadFunction);

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
