#include "../bit_twiddling.hlsli"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"

StructuredBuffer<uint> globals : register(t0);
StructuredBuffer<VisibleCluster> visibleClusters : register(t1);
RWStructuredBuffer<BLASData> blasDatas : register(u2);

StructuredBuffer<uint64_t> inputAddressArray : register(t3);
RWStructuredBuffer<uint64_t> blasAddressArray : register(u4);

StructuredBuffer<CLASPageInfo> clasPageInfos : register(t5);
StructuredBuffer<uint64_t> blasVoxelAddressTable : register(t6);
RWStructuredBuffer<uint2> offsetsAndCounts : register(u7);
RWStructuredBuffer<AccelerationStructureInstance> instanceDescriptors : register(u8);

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    if (dtID.x >= globals[GLOBALS_VISIBLE_CLUSTER_COUNT_INDEX]) return;

    VisibleCluster visibleCluster = visibleClusters[dtID.x];
    uint blasIndex = visibleCluster.blasIndex;
    uint clusterIndex = BitFieldExtractU32(visibleCluster.isVoxel_pageIndex_clusterIndex, MAX_CLUSTERS_PER_PAGE_BITS, 0);
    uint pageIndex = BitFieldExtractU32(visibleCluster.isVoxel_pageIndex_clusterIndex, 12, MAX_CLUSTERS_PER_PAGE_BITS);
    uint isVoxel = BitFieldExtractU32(visibleCluster.isVoxel_pageIndex_clusterIndex, 1, 12 + MAX_CLUSTERS_PER_PAGE_BITS);

    if (isVoxel)
    {
        uint addressIndex = pageIndex * MAX_CLUSTERS_PER_PAGE + clusterIndex;
        uint64_t address = blasVoxelAddressTable[addressIndex];

        if (blasDatas[blasIndex].tlasIndex == ~0u)
        {
            blasDatas[blasIndex].addressIndex = addressIndex;
        }
        else 
        {
            float3x4 transform = float3x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
            AccelerationStructureInstance instanceDescriptor;
            instanceDescriptor.transform = transform;
            instanceDescriptor.instanceID = addressIndex;
            instanceDescriptor.instanceMask = 0xff;
            instanceDescriptor.instanceContributionToHitGroupIndex = 0;
            instanceDescriptor.flags = 0;
            instanceDescriptor.blasDeviceAddress = address;

            uint descriptorIndex;
            uint tlasIndex = blasDatas[blasIndex].tlasIndex;
            InterlockedAdd(offsetsAndCounts[tlasIndex].y, 1, descriptorIndex);
            descriptorIndex += offsetsAndCounts[tlasIndex].x;
            instanceDescriptors[descriptorIndex] = instanceDescriptor;
        }
    }
    else 
    {
        CLASPageInfo clasPageInfo = clasPageInfos[pageIndex];
        uint destIndex;
        InterlockedAdd(blasDatas[blasIndex].clusterCount, 1, destIndex);

        destIndex += blasDatas[blasIndex].clusterStartIndex;

        uint addressIndex = clasPageInfo.addressStartIndex + clusterIndex;
        blasAddressArray[destIndex] = inputAddressArray[addressIndex];
    }

}
