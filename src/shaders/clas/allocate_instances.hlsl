#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<uint> visiblePartitions : register(t0);
RWStructuredBuffer<uint> globals : register(u1);
StructuredBuffer<GPUTransform> instanceTransforms : register(t2);
RWStructuredBuffer<PartitionInfo> partitionInfos: register(u3);
RWStructuredBuffer<int> instanceFreeList : register(u4);
RWStructuredBuffer<GPUInstance> gpuInstances : register(u5);
StructuredBuffer<Resource> resources : register(t6);
StructuredBuffer<uint> partitionResourceIDs : register(t7);
RWStructuredBuffer<uint> allocatedInstancesBuffer : register(u8);
StructuredBuffer<uint> instanceBitVector : register(t9);

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint visiblePartitionIndex = dtID.x;
    if (visiblePartitionIndex >= globals[GLOBALS_VISIBLE_PARTITION_COUNT]) return;

    uint partition = visiblePartitions[visiblePartitionIndex];
    PartitionInfo info = partitionInfos[partition];

    if (0)//info.flags & PARTITION_FLAG_PROXY_RENDERED)
    {
        //TODO
        int instanceFreeListIndex;
        InterlockedAdd(instanceFreeList[0], -1, instanceFreeListIndex);
        if (instanceFreeListIndex < 1)
        {
            return;
        }
        uint instanceIndex = instanceFreeList[instanceFreeListIndex];

        gpuInstances[instanceIndex].transformIndex = ~0u;
        gpuInstances[instanceIndex].resourceID = 0;
        gpuInstances[instanceIndex].partitionIndex = partition;
        gpuInstances[instanceIndex].flags = GPU_INSTANCE_FLAG_MERGED;
        return;
    }

    int numTransforms = (int)info.transformCount; 

    for (int i = 0; i < numTransforms; i++)
    {
        uint index = info.transformOffset + i;
        uint bit = instanceBitVector[index >> 5u] & (1u << (index & 31u));
        if (bit) continue;

        int instanceFreeListIndex;
        InterlockedAdd(instanceFreeList[0], -1, instanceFreeListIndex);
        if (instanceFreeListIndex < 1)
        { 
            return;
        }
        uint instanceIndex = instanceFreeList[instanceFreeListIndex];

        uint allocatedInstanceIndex;
        InterlockedAdd(globals[GLOBALS_ALLOCATED_INSTANCE_COUNT_INDEX], 1, allocatedInstanceIndex);

        allocatedInstancesBuffer[allocatedInstanceIndex] = instanceIndex;

        uint resourceID = partitionResourceIDs[info.transformOffset + (uint)i];
        gpuInstances[instanceIndex].transformIndex = info.transformOffset + (uint)i;
        gpuInstances[instanceIndex].resourceID = resourceID;
        gpuInstances[instanceIndex].partitionIndex = partition;
        gpuInstances[instanceIndex].flags = GPU_INSTANCE_FLAG_INDIV;
    }
}
