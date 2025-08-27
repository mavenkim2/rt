#include "../../rt/shader_interop/as_shaderinterop.h"

RWStructuredBuffer<BLASData> blasDatas : register(u0);
RWStructuredBuffer<uint> globals : register(u1);
RWStructuredBuffer<uint2> offsetsAndCounts : register(u2);

[numthreads(32, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint blasIndex = DTid.x;
    if (blasIndex >= globals[GLOBALS_BLAS_COUNT_INDEX]) return;

    if (blasDatas[blasIndex].clusterCount == 0 && blasDatas[blasIndex].voxelClusterCount == 0) return;

    uint clasStartIndex;
    InterlockedAdd(globals[GLOBALS_BLAS_CLAS_COUNT_INDEX], blasDatas[blasIndex].clusterCount, clasStartIndex);

    BLASData blasData = blasDatas[blasIndex];
    uint hasClusters = blasData.clusterCount ? 1 : 0;
    if (hasClusters + blasData.voxelClusterCount > 1)
    {
        uint instanceIndex;
        InterlockedAdd(offsetsAndCounts[0].x, 1, instanceIndex);
        uint numInstances = blasData.voxelClusterCount + hasClusters;
        uint instanceOffset;
        InterlockedAdd(globals[GLOBALS_VISIBLE_INSTANCE_COUNT], numInstances, instanceOffset);
        offsetsAndCounts[instanceIndex + 1] = uint2(instanceOffset, 0);
        blasDatas[blasIndex].tlasIndex = instanceIndex;
    }
    else 
    {
        blasDatas[blasIndex].tlasIndex = ~0u;
    }

    blasDatas[blasIndex].clusterStartIndex = clasStartIndex;
    blasDatas[blasIndex].clusterCount = 0;
}
