#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../dense_geometry.hlsli"

#if 0
StructuredBuffer<uint> pageIndices : register(t0);
RWStructuredBuffer<uint64_t> clasAddresses : register(u1);
StructuredBuffer<uint> clasSizes : register(t2);
RWStructuredBuffer<uint> globals : register(u3);
RWStructuredBuffer<CLASPageInfo> clasPageInfos : register(u4);
StructuredBuffer<DecodeClusterData> decodeClusterDatas : register(t5);

#else
RWStructuredBuffer<uint64_t> clasAddresses : register(u0);
StructuredBuffer<uint> clasSizes : register(t1);
RWStructuredBuffer<uint> globals : register(u2);
StructuredBuffer<DecodeClusterData> decodeClusterDatas : register(t3);
#endif

#if 0
groupshared uint pageClusterAccelSize;
groupshared uint pageClusterAccelOffset;
groupshared uint descriptorStartIndex;
#endif

[[vk::push_constant]] AddressPushConstant pc;

//[numthreads(MAX_CLUSTERS_PER_PAGE, 1, 1)]
[numthreads(32, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 dispatchThreadID : SV_DispatchThreadID)
{
#if 0
    if (groupIndex == 0)
    {
        pageClusterAccelSize = 0;
        descriptorStartIndex = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint pageIndex = pageIndices[groupID.x];
    uint numTriangleClusters = clasPageInfos[pageIndex].numTriangleClusters;

    if (groupIndex >= numTriangleClusters) return;

    DecodeClusterData clusterData = decodeClusterDatas[clasPageInfos[pageIndex].tempClusterOffset + groupIndex];
    uint clusterID = clusterData.clusterIndex;

    uint descriptorIndex = clasPageInfos[pageIndex].addressStartIndex + clusterID;
    uint clasOldPageDataByteOffset = globals[GLOBALS_OLD_PAGE_DATA_BYTES];
    uint clasSize = clasSizes[descriptorIndex];

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
    clasAddresses[descriptorIndex] = clasAddress;

    if (groupIndex == 0)
    {
        clasPageInfos[pageIndex].accelByteOffset = clasOldPageDataByteOffset + pageClusterAccelOffset;
        clasPageInfos[pageIndex].clasSize = pageClusterAccelSize;
        clasPageInfos[pageIndex].clasCount = numTriangleClusters;
    }
#else
    uint numClusters = globals[GLOBALS_CLAS_COUNT_INDEX];
    uint clusterID = dispatchThreadID.x;
    if (clusterID >= numClusters) return;

    DecodeClusterData clusterData = decodeClusterDatas[clusterID];

    uint clasOldPageDataByteOffset = globals[GLOBALS_OLD_PAGE_DATA_BYTES];
    uint clasSize = clasSizes[clusterID];

    // Reuse page cluster accel size as page byte offset
    uint clasOffset;
    InterlockedAdd(globals[GLOBALS_NEW_PAGE_DATA_BYTES], clasSize, clasOffset);

    uint64_t clusterBaseAddress = ((uint64_t)pc.addressHighBits << 32u) | (uint64_t)pc.addressLowBits;

    uint64_t clasAddress = clusterBaseAddress + clasOldPageDataByteOffset + clasOffset;
    clasAddresses[clusterID] = clasAddress;
#endif
}
