#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<uint> visiblePartitions : register(t0);
StructuredBuffer<uint> globals : register(t1);
StructuredBuffer<GPUTransform> instanceTransforms : register(t2);
RWStructuredBuffer<PartitionInfo> partitionInfos: register(u3);
RWStructuredBuffer<int> instanceFreeList : register(u4);
RWStructuredBuffer<GPUInstance> gpuInstances : register(u5);

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
        if (instanceFreeListIndex < 1) return;
        uint instanceIndex = instanceFreeList[instanceFreeListIndex];

        partitionInfos[partition].proxyInstanceIndex = instanceIndex;
        gpuInstances[instanceIndex].transformIndex = ~0u;
        gpuInstances[instanceIndex].globalRootNodeOffset = 0;
        gpuInstances[instanceIndex].resourceID = 0;
        gpuInstances[instanceIndex].partitionIndex = partition;
        gpuInstances[instanceIndex].voxelAddressOffset = 0;
        gpuInstances[instanceIndex].clusterLookupTableOffset = dtID.x;
        gpuInstances[instanceIndex].flags = GPU_INSTANCE_FLAG_MERGED;
        return;
    }

    int instanceFreeListOffset;
    int numTransforms = info.transformCount; 
    InterlockedAdd(instanceFreeList[0], -numTransforms, instanceFreeListOffset);

    for (int i = 0; i < numTransforms; i++)
    {
        int instanceFreeListIndex = instanceFreeListOffset - i;
        if (instanceFreeListIndex < 1) return;
        uint instanceIndex = instanceFreeList[instanceFreeListIndex];

        // TODO: proper resource id
        gpuInstances[instanceIndex].transformIndex = info.transformOffset + i;
        gpuInstances[instanceIndex].globalRootNodeOffset = 0;
        gpuInstances[instanceIndex].resourceID = 0;
        gpuInstances[instanceIndex].partitionIndex = partition;
        gpuInstances[instanceIndex].voxelAddressOffset = 0;
        gpuInstances[instanceIndex].clusterLookupTableOffset = 0;
        gpuInstances[instanceIndex].flags = 0;
    }
}
