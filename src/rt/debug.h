#ifndef DEBUG_H
#define DEBUG_H

#include "platform.h"
#include "graphics/vulkan.h"

namespace rt
{
struct Event
{
    u32 offset;
    PerformanceCounter counter;
    Event(u32 offset);
    ~Event();
};

#define TIMED_FUNCTION(arg) Event event_##__LINE__##arg(OffsetOf(ThreadStatistics, arg) / 8)

struct CommandBuffer;

struct Record
{
    f64 totalTimes[20];
    u32 numInvocations[20];
    u64 count;
    char *functionName;
};

struct Range
{
    char *filename;
    char *function;
    PerformanceCounter counter;
    f64 timeElapsed;
    u32 lineNumber;

    // GPU
    CommandBuffer *cmd;
    i32 gpuBeginIndex;
    i32 gpuEndIndex;

    bool IsGPURange();
};

struct DebugSlotNode
{
    u32 sid;
    u32 recordIndex;
    DebugSlotNode *next;
};

struct DebugSlot
{
    Mutex mutex;
    DebugSlotNode *first;
    DebugSlotNode *last;
};

struct DebugState
{
    static const u32 totalNumSlots = 1024;

    Arena *arena;

    // Timing
    QueryPool timestampPool;
    Record records[totalNumSlots];
    Range ranges[device->numBuffers][1024];
    u32 numRecords;
    DebugSlot *slots;
    std::atomic<u32> currentRangeIndex[device->numBuffers];
    GPUBuffer queryResultBuffer[device->numBuffers];
    std::atomic<u32> queryIndex;
    // AS_Handle debugFont;

    // Pipeline statistics
    QueryPool pipelineStatisticsPool;
    u64 pipelineStatistics[device->numBuffers][2];

    b8 initialized = 0;

    void BeginFrame(CommandBuffer *cmd);
    void EndFrame(CommandBuffer *cmd);
    // void BeginTriangleCount(CommandBuffer *cmd);
    // void EndTriangleCount(CommandBuffer *cmd);
    u32 BeginRange(char *filename, char *functionName, u32 lineNum, CommandBuffer *cmd = 0);
    void EndRange(u32 rangeIndex);
    Record *GetRecord(char *name);
    void PrintDebugRecords();
};

struct DebugEvent
{
    u32 rangeIndex;
    DebugEvent(char *filename, char *functionName, u32 lineNum, CommandBuffer *cmdList = {});
    ~DebugEvent();
};

#define TIMED_GPU(cmd)                   DebugEvent(__FILE__, FUNCTION_NAME, __LINE__, cmd);
#define TIMED_GPU_RANGE_BEGIN(cmd, name) debugState.BeginRange(__FILE__, name, __LINE__, cmd)
#define TIMED_RANGE_END(index)           debugState.EndRange(index)
#define TIMED_CPU()                      DebugEvent(__FILE__, FUNCTION_NAME, __LINE__)
#define TIMED_CPU_RANGE_BEGIN()          debugState.BeginRange(__FILE__, FUNCTION_NAME, __LINE__)

#define TIMED_CPU_RANGE_NAME_BEGIN(name) debugState.BeginRange(__FILE__, name, __LINE__)

extern DebugState debugState;

} // namespace rt
#endif
