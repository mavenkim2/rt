#include "../bit_twiddling.hlsli"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../dense_geometry.hlsli"
#include "ptlas_write_instances.hlsli"

StructuredBuffer<VisibleCluster> visibleClusters : register(t7);
RWStructuredBuffer<BLASData> blasDatas : register(u9);

StructuredBuffer<uint64_t> inputAddressArray : register(t10);
RWStructuredBuffer<uint64_t> blasAddressArray : register(u11);

StructuredBuffer<CLASPageInfo> clasPageInfos : register(t12);
StructuredBuffer<uint64_t> blasVoxelAddressTable : register(t13);
StructuredBuffer<GPUInstance> gpuInstances : register(t14);
StructuredBuffer<AABB> aabbs : register(t15);

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
        GPUInstance instance = gpuInstances[blasDatas[blasIndex].instanceID];

        uint basePageAddress = GetClusterPageBaseAddress(pageIndex);
        uint numClusters = GetNumClustersInPage(basePageAddress);
        DenseGeometry header = GetDenseGeometryHeader(basePageAddress, numClusters, clusterIndex);
        uint virtualInstanceID = instance.virtualInstanceIDOffset + 1 + header.id;

        //AABB aabb = aabbs[instance.resourceID];
        AABB aabb;
#if 0
        aabb.minX = header.boundsMin.x;
        aabb.minY = header.boundsMin.y;
        aabb.minZ = header.boundsMin.z;
        aabb.maxX = header.boundsMax.x;
        aabb.maxY = header.boundsMax.y;
        aabb.maxZ = header.boundsMax.z;
#endif

        //printf("%f %f %f %f %f %f\n", aabb.minX, aabb.minY, aabb.minZ, aabb.maxX, aabb.maxY, aabb.maxZ);

        WritePTLASDescriptors(instance, address, virtualInstanceID, addressIndex, aabb, false, 0u);
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
