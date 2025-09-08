#include "../../rt/shader_interop/as_shaderinterop.h"

RWStructuredBuffer<GPUInstance> gpuInstances : register(u0);
StructuredBuffer<GPUInstance> newInstances : register(t1);
StructuredBuffer<uint> evictedPartitions : register(t2);
RWStructuredBuffer<uint> globals : register(u3);

[[vk::push_constant]] InstanceStreamingPushConstant pc;

[numthreads(32, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 dtID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    // TODO: if max instances goes above 2^21 this doesn't work
    uint instanceIndex = dtID.x;
        
    if (gpuInstances[instanceIndex].partitionIndex == ~0u)
    {
        uint descriptorIndex;
        InterlockedAdd(globals[GLOBALS_NEW_INSTANCE_COUNT], 1, descriptorIndex);
        if (descriptorIndex < pc.numNewInstances)
        {
            gpuInstances[instanceIndex] = newInstances[descriptorIndex];
        }
        return;
    }

    for (uint index = 0; index < pc.numEvictedPartitions; index++)
    {
        uint evictedPartition = evictedPartitions[index];
        if (gpuInstances[instanceIndex].partitionIndex == evictedPartition)
        {
            uint descriptorIndex;
            InterlockedAdd(globals[GLOBALS_NEW_INSTANCE_COUNT], 1, descriptorIndex);
            if (descriptorIndex < pc.numNewInstances)
            {
                gpuInstances[instanceIndex] = newInstances[descriptorIndex];
            }
            else 
            {
                // Disable instance
                gpuInstances[instanceIndex].partitionIndex = ~0u;
            }
            break;
        }
    }
}
