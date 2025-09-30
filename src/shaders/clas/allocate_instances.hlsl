#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<uint> visiblePartitions : register(t0);
RWStructuredBuffer<uint> globals : register(u1);
StructuredBuffer<GPUTransform> instanceTransforms : register(t2);
StructuredBuffer<PartitionInfo> partitionInfos: register(t3);
RWStructuredBuffer<int> instanceFreeList : register(u4);
RWStructuredBuffer<GPUInstance> gpuInstances : register(u5);
StructuredBuffer<Resource> resources : register(t6);
StructuredBuffer<uint> partitionResourceIDs : register(t7);
RWStructuredBuffer<uint> allocatedInstancesBuffer : register(u8);

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint visiblePartitionIndex = dtID.x;
    if (visiblePartitionIndex >= globals[GLOBALS_VISIBLE_PARTITION_COUNT]) return;

    uint partition = visiblePartitions[visiblePartitionIndex];
    PartitionInfo info = partitionInfos[partition];

    if (info.flags & PARTITION_FLAG_PROXY_RENDERED)
    {
        int instanceFreeListIndex;
        InterlockedAdd(instanceFreeList[0], -1, instanceFreeListIndex);
        if (instanceFreeListIndex < 1)
        {
            instanceFreeList[0] = 0;
            return;
        }
        uint instanceIndex = instanceFreeList[instanceFreeListIndex];

        gpuInstances[instanceIndex].transformIndex = ~0u;
        gpuInstances[instanceIndex].resourceID = 0;
        gpuInstances[instanceIndex].partitionIndex = partition;
        gpuInstances[instanceIndex].flags = GPU_INSTANCE_FLAG_MERGED;
        return;
    }

    int instanceFreeListOffset;
    int numTransforms = (int)info.transformCount; 
    InterlockedAdd(instanceFreeList[0], -numTransforms, instanceFreeListOffset);

    if (instanceFreeListOffset - numTransforms < 1)
    {
        instanceFreeList[0] = 0;
        return;
    }

    uint allocatedInstanceIndex;
    InterlockedAdd(globals[GLOBALS_ALLOCATED_INSTANCE_COUNT_INDEX], numTransforms, allocatedInstanceIndex);

    for (int i = 0; i < numTransforms; i++)
    {
        int instanceFreeListIndex = instanceFreeListOffset - i;
        uint instanceIndex = instanceFreeList[instanceFreeListIndex];
        uint resourceID = partitionResourceIDs[info.transformOffset + (uint)i];

        allocatedInstancesBuffer[allocatedInstanceIndex + i] = instanceIndex;

        gpuInstances[instanceIndex].transformIndex = info.transformOffset + (uint)i;
        gpuInstances[instanceIndex].resourceID = resourceID;
        gpuInstances[instanceIndex].partitionIndex = partition;
        gpuInstances[instanceIndex].flags = GPU_INSTANCE_FLAG_INDIV;
    }
}
