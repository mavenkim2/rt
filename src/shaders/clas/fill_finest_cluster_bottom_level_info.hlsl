#include "../../rt/shader_interop/as_shaderinterop.h"

RWStructuredBuffer<BLASData> blasDatas : register(u0);
RWStructuredBuffer<BUILD_CLUSTERS_BOTTOM_LEVEL_INFO> buildClusterBottomLevelInfos : register(u1);
RWStructuredBuffer<uint> globals : register(u2);
StructuredBuffer<GPUInstance> gpuInstances : register(t3);
RWStructuredBuffer<Resource> resources : register(u4);
RWStructuredBuffer<uint> resourceBitVector : register(u5);

[[vk::push_constant]] AddressPushConstant pc;

[numthreads(FILL_CLUSTER_BOTTOM_LEVEL_INFO_GROUPSIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint blasIndex = DTid.x;
    if (blasIndex >= globals[GLOBALS_BLAS_COUNT_INDEX]) return;

    BLASData blas = blasDatas[blasIndex];

    if (blas.clusterCount == 0) return;

    uint resourceID = gpuInstances[blas.instanceID].resourceID;

    if (resources[resourceID].maxClusters == blas.clusterCount)
    {
        uint wasSet;
        uint bit = 1u << (resourceID & 31u);
        InterlockedOr(resourceBitVector[resourceID >> 5u], bit, wasSet);

        if (wasSet & bit)
        {
            return;
        }


        uint descriptorIndex;
        InterlockedAdd(globals[GLOBALS_BLAS_FINAL_COUNT_INDEX], 1, descriptorIndex);

        blasDatas[blasIndex].addressIndex = descriptorIndex;
        resources[resourceID].finestAddressIndex = descriptorIndex;

        uint64_t clasBaseAddress = ((uint64_t(pc.addressHighBits) << 32) | (pc.addressLowBits));

        buildClusterBottomLevelInfos[descriptorIndex].clusterReferencesCount = blas.clusterCount;
        buildClusterBottomLevelInfos[descriptorIndex].clusterReferencesStride = RAY_TRACING_ADDRESS_STRIDE;
        buildClusterBottomLevelInfos[descriptorIndex].clusterReferences = clasBaseAddress + RAY_TRACING_ADDRESS_STRIDE * blas.clusterStartIndex;
    }
}
