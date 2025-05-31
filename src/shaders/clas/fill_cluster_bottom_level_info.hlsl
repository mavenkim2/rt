#include "../../rt/shader_interop/as_shaderinterop.h"

// Input
RWStructuredBuffer<BUILD_CLUSTERS_BOTTOM_LEVEL_INFO> blasData : register(u0);
RWStructuredBuffer<uint> globals : register(u1);

[[vk::push_constant]] FillClusterBottomLevelInfoPushConstant pc;

[numthreads(FILL_CLUSTER_BOTTOM_LEVEL_INFO_GROUPSIZE, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID)
{
    uint blasIndex = DTid.x;
    if (blasIndex >= pc.blasCount) return;

    BUILD_CLUSTERS_BOTTOM_LEVEL_INFO blas = blasData[blasIndex];

    if (blas.clusterReferencesCount == 0) return;

    uint clasAddressStart;
    InterlockedAdd(globals[GLOBALS_CLAS_COUNT_INDEX], blas.clusterReferencesCount, clasAddressStart);

    uint64_t clasBaseAddress;

    blasData[blasIndex].clusterReferencesStride = 0;
    blasData[blasIndex].clusterReferences = clasBaseAddress + RAY_TRACING_ADDRESS_STRIDE * clasAddressStart;
}
