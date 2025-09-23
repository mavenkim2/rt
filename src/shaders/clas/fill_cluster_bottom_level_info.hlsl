#include "../../rt/shader_interop/as_shaderinterop.h"

RWStructuredBuffer<BLASData> blasDatas : register(u0);
StructuredBuffer<uint> globals : register(t1);
StructuredBuffer<GPUInstance> gpuInstances : register(t2);
StructuredBuffer<Resource> resources : register(t3);

#if 0
[[vk::push_constant]] AddressPushConstant pc;
#endif

[numthreads(FILL_CLUSTER_BOTTOM_LEVEL_INFO_GROUPSIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint blasIndex = DTid.x;
    if (blasIndex >= globals[GLOBALS_BLAS_COUNT_INDEX]) return;

    BLASData blas = blasDatas[blasIndex];
    uint flags = blas.addressIndex;
    uint resourceID = gpuInstances[blas.instanceID].resourceID;
    Resource resource = resources[resourceID];

    if (flags & GPU_INSTANCE_FLAG_MERGED_INSTANCE)
    {
        blasDatas[blasIndex].addressIndex = resources[resourceID].mergedAddressIndex;
    }
    else if (flags & GPU_INSTANCE_FLAG_SHARED_INSTANCE)
    {
        blasDatas[blasIndex].addressIndex = resources[resourceID].sharedAddressIndex;
    }

#if 0
    uint descriptorIndex;
    InterlockedAdd(globals[GLOBALS_BLAS_FINAL_COUNT_INDEX], 1, descriptorIndex);

    uint64_t clasBaseAddress = ((uint64_t(pc.addressHighBits) << 32) | (pc.addressLowBits));

    buildClusterBottomLevelInfos[descriptorIndex].clusterReferencesCount = blas.clusterCount;
    buildClusterBottomLevelInfos[descriptorIndex].clusterReferencesStride = RAY_TRACING_ADDRESS_STRIDE;
    buildClusterBottomLevelInfos[descriptorIndex].clusterReferences = clasBaseAddress + RAY_TRACING_ADDRESS_STRIDE * blas.clusterStartIndex;
#endif
}
