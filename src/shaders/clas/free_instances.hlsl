#include "../../rt/shader_interop/as_shaderinterop.h"

RWStructuredBuffer<GPUInstance> instances : register(u0);
StructuredBuffer<uint2> freedPartitions : register(t1);
RWStructuredBuffer<uint> globals : register(u2);
RWStructuredBuffer<uint> instanceFreeList : register(u3);
RWStructuredBuffer<uint> freedInstances : register(u4);

[[vk::push_constant]] NumPushConstant pc;

[numthreads(128, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint instanceIndex = dtID.x;
    if (instanceIndex >= pc.num) return;

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

            uint freedIndexIndex;
            InterlockedAdd(globals[GLOBALS_FREED_INSTANCE_COUNT_INDEX], 1, freedIndexIndex);
            freedInstances[freedIndexIndex] = instanceIndex;
            return;
        }
    }
}
