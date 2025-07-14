#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<CLASPageInfo> clasPageInfos : register(t0);
StructuredBuffer<uint64_t> srcAddresses : register(t1);
StructuredBuffer<uint32_t> srcSizes : register(t2);

RWStructuredBuffer<uint64_t> dstAddresses : register(u3);
RWStructuredBuffer<uint32_t> dstSizes : register(u4);

#define THREADS_PER_GROUP 32
[numthreads(THREADS_PER_GROUP, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    uint pageIndex = groupID.x;
    CLASPageInfo clasPageInfo = clasPageInfos[pageIndex];

    if (clasPageInfo.tempClusterOffset == ~0u) return;

    for (uint clusterIndex = groupIndex; clusterIndex < clasPageInfo.clasCount; clusterIndex += THREADS_PER_GROUP)
    {
        uint srcIndex = clasPageInfo.tempClusterOffset + clusterIndex;
        uint dstIndex = clasPageInfo.addressStartIndex + clusterIndex;
        dstAddresses[dstIndex] = srcAddresses[srcIndex];
        dstSizes[dstIndex] = srcSizes[srcIndex];
    }
}
