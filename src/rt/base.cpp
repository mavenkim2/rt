#include "string.h"
#include "../third_party/stb_sprintf.h"

namespace rt
{
void Print(const char *fmt, va_list args)
{
    char printBuffer[1024];
    stbsp_vsprintf(printBuffer, fmt, args);
#if _WIN32
    OutputDebugStringA(printBuffer);
#else
    printf(printBuffer);
#endif
}

void Print(const char *fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    char printBuffer[1024];
    stbsp_vsprintf(printBuffer, fmt, va);
    va_end(va);
#if _WIN32
    OutputDebugStringA(printBuffer);
#else
    printf(printBuffer);
#endif
}
} // namespace rt
