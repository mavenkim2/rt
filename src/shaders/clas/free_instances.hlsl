#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<GPUInstance> instances : register(t0);
StructuredBuffer<uint> freedPartitions : register(t1);
StructuredBuffer<uint> globals : register(t2);
RWStructuredBuffer<uint> instanceFreeList : register(u3);

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint instanceIndex = dtID.x;
    if (instanceIndex >= 1u << 21u) return;

    for (uint i = 0; i < globals[GLOBALS_FREED_PARTITION_COUNT]; i++)
    {
        uint partition = freedPartitions[i];
        if (instances[instanceIndex].partitionIndex == partition)
        {
            uint instanceFreeListIndex;
            InterlockedAdd(instanceFreeList[0], 1, instanceFreeListIndex);
            instanceFreeList[instanceFreeListIndex + 1] = instanceIndex;
            return;
        }
    }
}
