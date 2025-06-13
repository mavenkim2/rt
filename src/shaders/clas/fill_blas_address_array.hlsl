#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<ClusterPageData> clusterPageDatas : register(t0);
RWStructuredBuffer<BLASData> blasDatas : register(u1);

StructuredBuffer<uint64_t> inputAddressArray : register(t2);
RWStructuredBuffer<uint64_t> blasAddressArray : register(u3);

[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    ClusterPageData clusterPage = clusterPageDatas[dispatchThreadID.x];
    uint destIndex;
    InterlockedAdd(blasDatas[clusterPage.blasIndex].clusterCount, 1, destIndex);

    destIndex += blasDatas[clusterPage.blasIndex].clusterStartIndex;
    blasAddressArray[destIndex] = inputAddressArray[dispatchThreadID.x];
}
