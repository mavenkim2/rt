#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<DecodeClusterData> decodeClusterDatas : register(t0);
RWStructuredBuffer<BLASData> blasDatas : register(u1);

StructuredBuffer<uint64_t> inputAddressArray : register(t2);
RWStructuredBuffer<uint64_t> blasAddressArray : register(u3);

[[vk::push_constant]] NumPushConstant pc;

[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x >= pc.num) return;

    DecodeClusterData decodeClusterData = decodeClusterDatas[dispatchThreadID.x];

    uint destIndex;
    InterlockedAdd(blasDatas[decodeClusterData.blasIndex].clusterCount, 1, destIndex);

    destIndex += blasDatas[decodeClusterData.blasIndex].clusterStartIndex;
    blasAddressArray[destIndex] = inputAddressArray[dispatchThreadID.x];
}
