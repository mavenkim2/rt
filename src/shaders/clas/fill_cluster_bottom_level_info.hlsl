#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<BLASData> blasDatas : register(t0);
RWStructuredBuffer<BUILD_CLUSTERS_BOTTOM_LEVEL_INFO> buildClusterBottomLevelInfos : register(u1);
RWStructuredBuffer<uint> globals : register(u2);

[[vk::push_constant]] FillClusterBottomLevelInfoPushConstant pc;

[numthreads(FILL_CLUSTER_BOTTOM_LEVEL_INFO_GROUPSIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint blasIndex = DTid.x;
    if (blasIndex >= globals[GLOBALS_BLAS_COUNT_INDEX]) return;

    BLASData blas = blasDatas[blasIndex];

    if (blas.clusterCount == 0) return;

    InterlockedAdd(globals[GLOBALS_BLAS_COUNT_INDEX], 1);
    uint64_t clasBaseAddress = ((uint64_t(pc.arrayBaseAddressHighBits) << 32) | (pc.arrayBaseAddressLowBits));

    buildClusterBottomLevelInfos[blasIndex].clusterReferencesCount = blas.clusterCount;
    buildClusterBottomLevelInfos[blasIndex].clusterReferencesStride = RAY_TRACING_ADDRESS_STRIDE;
    buildClusterBottomLevelInfos[blasIndex].clusterReferences = clasBaseAddress + RAY_TRACING_ADDRESS_STRIDE * blas.clusterStartIndex;
}
