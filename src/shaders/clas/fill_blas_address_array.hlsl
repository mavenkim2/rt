#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"

StructuredBuffer<uint> globals : register(t0);
StructuredBuffer<VisibleCluster> visibleClusters : register(t1);
RWStructuredBuffer<BLASData> blasDatas : register(u2);

StructuredBuffer<uint64_t> inputAddressArray : register(t3);
RWStructuredBuffer<uint64_t> blasAddressArray : register(u4);

StructuredBuffer<CLASPageInfo> clasPageInfos : register(t5);
StructuredBuffer<uint64_t> blasVoxelAddressTable : register(t6);
RWStructuredBuffer<BLASVoxelInfo> blasVoxelInfos : register(u7);

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    if (dtID.x >= globals[GLOBALS_VISIBLE_CLUSTER_COUNT_INDEX]) return;

    VisibleCluster visibleCluster = visibleClusters[dtID.x];
    CLASPageInfo clasPageInfo = clasPageInfos[visibleCluster.pageIndex];

    // ideas: 
    // 1. hash table... i feel like this doesn't scale.
    // 2. allocate an offset into an array (aka some virtual system?)
    // 3. yeah i think it just has to be a virtual table 
    // 4. it's a virtual table per instance. 
    // 5. actualyl it's not that simple 
    // 6. because the allocation has to be done on the gpu ....
    // 7. so we would need a free list or something during the ptlas write phase
    // 8. and then in the update unused we append to the free list...
    // 9. it could work?

    uint blasIndex = visibleCluster.blasIndex;
    if (visibleCluster.instanceID != ~0u)
    {
        uint destIndex;
        InterlockedAdd(blasDatas[blasIndex].voxelClusterCount, 1, destIndex);
        destIndex += blasDatas[blasIndex].voxelClusterStartIndex;

        uint addressIndex = visibleCluster.pageIndex * MAX_CLUSTERS_PER_PAGE + visibleCluster.clusterIndex;
        uint64_t address = blasVoxelAddressTable[addressIndex];

        BLASVoxelInfo info;
        info.address = address;
        info.clusterID = (visibleCluster.pageIndex << MAX_CLUSTERS_PER_PAGE_BITS) | visibleCluster.clusterIndex;
        info.instanceIndex = visibleCluster.instanceID;
        blasVoxelInfos[destIndex] = info;
    }
    else 
    {
        uint destIndex;
        InterlockedAdd(blasDatas[blasIndex].clusterCount, 1, destIndex);

        destIndex += blasDatas[blasIndex].clusterStartIndex;

        uint addressIndex = clasPageInfo.addressStartIndex + visibleCluster.clusterIndex;
        blasAddressArray[destIndex] = inputAddressArray[addressIndex];
    }

}
