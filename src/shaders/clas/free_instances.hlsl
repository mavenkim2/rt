#include "../../rt/shader_interop/as_shaderinterop.h"

RWStructuredBuffer<GPUInstance> instances : register(u0);
StructuredBuffer<uint2> freedPartitions : register(t1);
RWStructuredBuffer<uint> globals : register(u2);
RWStructuredBuffer<uint> instanceFreeList : register(u3);

[numthreads(64, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint instanceIndex = dtID.x;
#if 0
    if (dtID.x == 0)
    {
        globals[GLOBALS_VISIBLE_PARTITION_INDIRECT_X] = (globals[GLOBALS_VISIBLE_PARTITION_COUNT] + 31) / 32;
    }
#endif

    if (instanceIndex >= 1u << 21u) return;

    GPUInstance instance = instances[instanceIndex];
    for (uint i = 0; i < globals[GLOBALS_FREED_PARTITION_COUNT]; i++)
    {
        uint partition = freedPartitions[i].x;
        uint flags = freedPartitions[i].y;
        if (instance.partitionIndex == partition && ((instance.flags & GPU_INSTANCE_FLAG_FREED) == 0)
            && (instance.flags & flags))
        {
            uint instanceFreeListIndex;
            InterlockedAdd(instanceFreeList[0], 1, instanceFreeListIndex);
            instanceFreeList[instanceFreeListIndex + 1] = instanceIndex;
            instances[instanceIndex].flags |= GPU_INSTANCE_FLAG_FREED;
            return;
        }
    }
}
