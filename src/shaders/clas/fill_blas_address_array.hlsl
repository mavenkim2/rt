#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"

StructuredBuffer<uint> globals : register(t0);
StructuredBuffer<VisibleCluster> visibleClusters : register(t1);
RWStructuredBuffer<BLASData> blasDatas : register(u2);

StructuredBuffer<uint64_t> inputAddressArray : register(t3);
RWStructuredBuffer<uint64_t> blasAddressArray : register(u4);

StructuredBuffer<CLASPageInfo> clasPageInfos : register(t5);

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    if (dtID.x >= globals[GLOBALS_VISIBLE_CLUSTER_COUNT_INDEX]) return;

    VisibleCluster visibleCluster = visibleClusters[dtID.x];
    CLASPageInfo clasPageInfo = clasPageInfos[visibleCluster.pageIndex];

    uint blasIndex = visibleCluster.blasIndex;
    uint destIndex;
    InterlockedAdd(blasDatas[blasIndex].clusterCount, 1, destIndex);

    destIndex += blasDatas[blasIndex].clusterStartIndex;
    
    uint addressIndex = clasPageInfo.addressStartIndex + visibleCluster.clusterIndex;
    blasAddressArray[destIndex] = inputAddressArray[addressIndex];
}
