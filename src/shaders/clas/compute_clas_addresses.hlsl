#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../dense_geometry.hlsli"

StructuredBuffer<uint> pageIndices : register(t0);
RWStructuredBuffer<uint64_t> clasBuildAddresses : register(u1);
StructuredBuffer<uint> clasBuildSizes : register(t2);
RWStructuredBuffer<uint64_t> clasTemplateAddresses : register(u3);
StructuredBuffer<uint> clasTemplateSizes : register(t4);
RWStructuredBuffer<uint> globals : register(u5);
RWStructuredBuffer<CLASPageInfo> clasPageInfos : register(u6);
StructuredBuffer<DecodeClusterData> decodeClusterDatas : register(t7);

groupshared uint pageClusterAccelSize;
groupshared uint pageClusterAccelOffset;
groupshared uint descriptorStartIndex;

[[vk::push_constant]] AddressPushConstant pc;

[numthreads(MAX_CLUSTERS_PER_PAGE, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    if (groupIndex == 0)
    {
        pageClusterAccelSize = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint pageIndex = pageIndices[groupID.x];
    uint basePageAddress = GetClusterPageBaseAddress(pageIndex);
    uint numClusters = GetNumClustersInPage(basePageAddress);

    if (groupIndex >= numClusters) return;

    uint decodeDataIndex = clasPageInfos[pageIndex].decodeStartIndex + groupIndex;
    DecodeClusterData decodeClusterData = decodeClusterDatas[decodeDataIndex];

    uint addressIndex = decodeClusterData.addressIndex;
    uint clasSize = 0;

    if (addressIndex >> 31u)
    {
        uint index = addressIndex & 0x7fffffffu;
        clasSize = clasTemplateSizes[index];
    }
    else 
    {
        clasSize = clasBuildSizes[addressIndex];
    }

    uint clasOldPageDataByteOffset = globals[GLOBALS_OLD_PAGE_DATA_BYTES];

    // TODO: perform prefix sum to order CLAS within a page
    uint clusterInPageOffset;
    InterlockedAdd(pageClusterAccelSize, clasSize, clusterInPageOffset);
    GroupMemoryBarrierWithGroupSync();

    // Reuse page cluster accel size as page byte offset
    if (groupIndex == 0)
    {
        InterlockedAdd(globals[GLOBALS_NEW_PAGE_DATA_BYTES], pageClusterAccelSize, pageClusterAccelOffset);
    }
    GroupMemoryBarrierWithGroupSync();

    uint64_t clusterBaseAddress = ((uint64_t)pc.addressHighBits << 32u) | (uint64_t)pc.addressLowBits;
    uint64_t clasAddress = clusterBaseAddress + clasOldPageDataByteOffset + pageClusterAccelOffset + clusterInPageOffset;

    if (addressIndex >> 31u)
    {
        clasTemplateAddresses[addressIndex] = clasAddress;
    }
    else 
    {
        clasBuildAddresses[addressIndex] = clasAddress;
    }

    if (groupIndex == 0)
    {
        clasPageInfos[pageIndex].accelByteOffset = clasOldPageDataByteOffset + pageClusterAccelOffset;
        clasPageInfos[pageIndex].clasSize = pageClusterAccelSize;
        clasPageInfos[pageIndex].clasCount = numClusters;
    }
}
