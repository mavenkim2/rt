#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<BLASData> blasDatas : register(t0);
RWStructuredBuffer<BUILD_CLUSTERS_BOTTOM_LEVEL_INFO> buildClusterBottomLevelInfos : register(u1);
StructuredBuffer<uint> globals : register(t2);

[[vk::push_constant]] AddressPushConstant pc;

[numthreads(FILL_CLUSTER_BOTTOM_LEVEL_INFO_GROUPSIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint blasIndex = DTid.x;
    if (blasIndex >= globals[GLOBALS_BLAS_COUNT_INDEX]) return;

    BLASData blas = blasDatas[blasIndex];

    if (blas.clusterCount == 0) return;

    uint64_t clasBaseAddress = ((uint64_t(pc.addressHighBits) << 32) | (pc.addressLowBits));

    buildClusterBottomLevelInfos[blasIndex].clusterReferencesCount = blas.clusterCount;
    buildClusterBottomLevelInfos[blasIndex].clusterReferencesStride = RAY_TRACING_ADDRESS_STRIDE;
    buildClusterBottomLevelInfos[blasIndex].clusterReferences = clasBaseAddress + RAY_TRACING_ADDRESS_STRIDE * blas.clusterStartIndex;
}
