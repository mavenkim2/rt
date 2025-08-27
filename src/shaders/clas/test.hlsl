#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../wave_intrinsics.hlsli"
#include "../dense_geometry.hlsli"

RWStructuredBuffer<VisibleCluster> visibleClusters : register(u0);
RWStructuredBuffer<uint> globals : register(u1);

groupshared uint clusterStartIndex;

[numthreads(MAX_CLUSTERS_PER_PAGE, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
#if 0
    uint pageIndex = groupID.x;
    uint basePageAddress = GetClusterPageBaseAddress(pageIndex);
    uint numClusters = GetNumClustersInPage(basePageAddress);

    if (groupIndex == 0)
    {
        InterlockedAdd(globals[GLOBALS_CLAS_COUNT_INDEX], numClusters, clusterStartIndex);
    }
    GroupMemoryBarrierWithGroupSync();

    if (groupIndex >= numClusters) return;

    uint clusterIndex = groupIndex;
    uint descriptorIndex = clusterStartIndex + groupIndex;

    VisibleCluster cluster;
    cluster.pageIndex = pageIndex;
    cluster.clusterIndex = clusterIndex;
    //cluster.instanceID = 0;//instanceID;
    cluster.blasIndex = 0;//blasIndex;

    visibleClusters[descriptorIndex] = cluster;
#endif
}
