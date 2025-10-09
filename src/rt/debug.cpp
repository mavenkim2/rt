#include "base.h"
#include "string.h"
#include "hash.h"
#include "debug.h"
#include "thread_context.h"
#include "thread_statistics.h"
#include "win32.h"

namespace rt
{
DebugState debugState;

Event::Event(u32 offset) : offset(offset) { counter = OS_StartCounter(); }
Event::~Event()
{
    f32 time = OS_GetMilliseconds(counter);
    ((f64 *)(&threadLocalStatistics[GetThreadIndex()]))[offset] += (f64)time;
}

bool Range::IsGPURange() { return (bool)cmd; }

void DebugState::BeginFrame(CommandBuffer *cmd)
{
    if (!initialized)
    {
        initialized = 1;

        arena = ArenaAlloc();
        slots = PushArray(arena, DebugSlot, totalNumSlots);

        device->CreateQueryPool(&timestampPool, QueryType_Timestamp, 1024);
        device->CreateQueryPool(&pipelineStatisticsPool, QueryType_PipelineStatistics, 4);

        u32 bufferSize =
            (u32)(sizeof(u64) * (timestampPool.count + pipelineStatisticsPool.count));

        for (u32 i = 0; i < ArrayLength(queryResultBuffer); i++)
        {
            queryResultBuffer[i] = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                        bufferSize, MemoryUsage::GPU_TO_CPU);
        }
    }

    u32 currentBuffer = device->GetCurrentBuffer();
    u32 numRanges     = currentRangeIndex[currentBuffer].load();

    for (u32 recordIndex = 0; recordIndex < numRecords; recordIndex++)
    {
        Record *record                = &records[recordIndex];
        u32 index                     = record->count++ % (ArrayLength(record->totalTimes));
        record->numInvocations[index] = 0;
        record->totalTimes[index]     = 0;
    }

    u64 *mappedData = (u64 *)queryResultBuffer[currentBuffer].mappedPtr;
    for (u32 rangeIndex = 0; rangeIndex < numRanges; rangeIndex++)
    {
        Range *range   = &ranges[currentBuffer][rangeIndex];
        Record *record = debugState.GetRecord(range->function);
        if (range->IsGPURange())
        {
            f64 timestampPeriod = device->GetTimestampPeriod() * 1000;
            range->timeElapsed =
                (f64)(mappedData[range->gpuEndIndex] - mappedData[range->gpuBeginIndex]) *
                timestampPeriod;
        }
        u32 index = (record->count - 1) % (ArrayLength(record->totalTimes));
        record->numInvocations[index]++;
        record->totalTimes[index] += range->timeElapsed;
    }

    u32 lastQuery                        = queryIndex.load();
    pipelineStatistics[currentBuffer][0] = mappedData[lastQuery + 0];
    pipelineStatistics[currentBuffer][1] = mappedData[lastQuery + 1];

    cmd->ResetQuery(&timestampPool, 0, timestampPool.count);
    cmd->ResetQuery(&pipelineStatisticsPool, 0, pipelineStatisticsPool.count);
    currentRangeIndex[currentBuffer].store(0);
    queryIndex.store(0);
}

void DebugState::EndFrame(CommandBuffer *cmd)
{
    u32 currentBuffer = device->GetCurrentBuffer();

    u32 numTimestampQueries = queryIndex.load();
    {
        cmd->Barrier(&queryResultBuffer[currentBuffer], VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_TRANSFER_READ_BIT);
        cmd->FlushBarriers();
    }
    cmd->ResolveQuery(&timestampPool, &queryResultBuffer[currentBuffer], 0,
                      numTimestampQueries, 0);
    // cmd->ResolveQuery(&pipelineStatisticsPool, &queryResultBuffer[currentBuffer], 0, 1,
    //                   sizeof(u64) * numTimestampQueries);
}

Record *DebugState::GetRecord(char *name)
{
    string str      = Str8C(name);
    u32 sid         = Hash(str);
    DebugSlot *slot = &slots[sid % (totalNumSlots - 1)];
    Record *record  = 0;
    for (DebugSlotNode *node = slot->first; node != 0; node = node->next)
    {
        if (node->sid == sid)
        {
            record = &records[node->recordIndex];
            break;
        }
    }
    if (!record)
    {
        u32 recordIndex     = numRecords++;
        DebugSlotNode *node = PushStruct(arena, DebugSlotNode);
        node->recordIndex   = recordIndex;
        node->sid           = sid;
        QueuePush(slot->first, slot->last, node);

        record               = &records[recordIndex];
        record->functionName = name;
        Assert(record->count == 0);
    }
    return record;
}

// void DebugState::BeginTriangleCount(CommandBuffer *cmd)
// {
//     cmd->BeginQuery(&pipelineStatisticsPool, 0);
// }
// void DebugState::EndTriangleCount(CommandBuffer *cmd)
// {
//     cmd->EndQuery(&pipelineStatisticsPool, 0);
// }

u32 DebugState::BeginRange(char *filename, char *functionName, u32 lineNum, CommandBuffer *cmd)
{
    u32 currentBuffer = device->GetCurrentBuffer();
    u32 rangeIndex    = debugState.currentRangeIndex[currentBuffer].fetch_add(1);
    Assert(rangeIndex < ArrayLength(debugState.ranges[0]));

    Range *range      = &debugState.ranges[currentBuffer][rangeIndex];
    range->filename   = filename;
    range->function   = functionName;
    range->lineNumber = lineNum;
    range->cmd        = cmd;

    if (range->IsGPURange())
    {
        range->gpuBeginIndex = debugState.queryIndex.fetch_add(1);
        range->cmd->EndQuery(&debugState.timestampPool, range->gpuBeginIndex);
    }
    else
    {
        range->counter = OS_StartCounter();
    }
    return rangeIndex;
}

void DebugState::EndRange(u32 rangeIndex)
{
    u32 currentBuffer = device->GetCurrentBuffer();
    Range *range      = &debugState.ranges[currentBuffer][rangeIndex];
    if (range->IsGPURange())
    {
        range->gpuEndIndex = debugState.queryIndex.fetch_add(1);
        range->cmd->EndQuery(&debugState.timestampPool, range->gpuEndIndex);
    }
    else
    {
        range->timeElapsed = OS_GetMilliseconds(range->counter);
    }
}

void DebugState::PrintDebugRecords()
{
    for (u32 recordIndex = 0; recordIndex < numRecords; recordIndex++)
    {
        Record *record     = &records[recordIndex];
        u64 size           = ArrayLength(record->totalTimes);
        f64 avg            = 0;
        f32 avgInvocations = 0;
        u64 num            = Min(size, record->count);
        f32 time           = record->totalTimes[(record->count - 1) % size];
        for (u32 timeIndex = 0; timeIndex < num; timeIndex++)
        {
            avg += record->totalTimes[timeIndex];
            avgInvocations += record->numInvocations[timeIndex];
        }
        avg /= num;
        avgInvocations /= num;
        Print("%s | Total Time: %f ms | Avg Total Time: %f ms | Avg Invocations: %f\n",
              record->functionName, time, avg, avgInvocations);
    }

    // u64 triangleCount      = 0;
    // u64 clippingPrimitives = 0;
    // u32 length             = device->numBuffers;
    // for (u32 i = 0; i < length; i++)
    // {
    //     triangleCount += pipelineStatistics[i][0];
    //     clippingPrimitives += pipelineStatistics[i][1];
    // }
    // Print("Triangle counts: %u\n", triangleCount / length);
    // Print("Clipping primitives: %u\n\n", clippingPrimitives / length);
}

DebugEvent::DebugEvent(char *filename, char *functionName, u32 lineNum, CommandBuffer *cmd)
{
    rangeIndex = debugState.BeginRange(filename, functionName, lineNum, cmd);
}

DebugEvent::~DebugEvent() { debugState.EndRange(rangeIndex); }

} // namespace rt
