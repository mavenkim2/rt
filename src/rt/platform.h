#ifndef PLATFORM_H
#define PLATFORM_H

namespace rt
{
struct ThreadContext;
#define THREAD_ENTRY_POINT(name) void name(void *parameter, ThreadContext *ctx)
typedef THREAD_ENTRY_POINT(OS_ThreadFunction);

#if _WIN32
#include "win32.h"
#else 
#error sorry, os not supported
#endif
} // namespace rt

#endif
