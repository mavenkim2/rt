#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../wave_intrinsics.hlsli"
#include "../dense_geometry.hlsli"

StructuredBuffer<uint> pageIndices : register(t0);
RWStructuredBuffer<BUILD_CLUSTERS_TRIANGLE_INFO> buildClusterTriangleInfos : register(u1);
RWStructuredBuffer<DecodeClusterData> decodeClusterDatas : register(u2);
RWStructuredBuffer<uint> globals : register(u3);
RWStructuredBuffer<CLASPageInfo> clasPageInfos : register(u4);

StructuredBuffer<uint64_t> templateAddresses : register(t5);
RWStructuredBuffer<INSTANTIATE_CLUSTER_TEMPLATE_INFO> instantiateInfos : register(u6);

[[vk::push_constant]] FillClusterTriangleInfoPushConstant pc;

groupshared uint clusterStartIndex;
groupshared uint numClusterAddresses;
groupshared uint numVoxelClusterAddresses;
groupshared uint voxelClusterStartIndex;

[numthreads(MAX_CLUSTERS_PER_PAGE, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID: SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    if (dispatchThreadID.x == 0)
    {
        globals[GLOBALS_DECODE_INDIRECT_Y] = 1;
        globals[GLOBALS_DECODE_INDIRECT_Z] = 1;
    }

    uint pageIndex = pageIndices[groupID.x];
    uint basePageAddress = GetClusterPageBaseAddress(pageIndex);
    uint numClusters = GetNumClustersInPage(basePageAddress);

    if (groupIndex == 0)
    {
        numClusterAddresses = 0;
        numVoxelClusterAddresses = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    if (groupIndex >= numClusters) return;

    uint64_t indexBufferBaseAddress = ((uint64_t(pc.indexBufferBaseAddressHighBits) << 32) | (pc.indexBufferBaseAddressLowBits));
    uint64_t vertexBufferBaseAddress = ((uint64_t(pc.vertexBufferBaseAddressHighBits) << 32) | (pc.vertexBufferBaseAddressLowBits));

    uint clusterID = groupIndex;

    DenseGeometry header = GetDenseGeometryHeader(basePageAddress, numClusters, clusterID);

    uint addressOffset = 0;
    uint voxelAddressOffset = 0;
    if (header.numTriangles)
    {
        InterlockedAdd(numClusterAddresses, 1, addressOffset);
        InterlockedAdd(globals[GLOBALS_TRIANGLE_CLUSTER_COUNT], 1);
    }
    else 
    {
        uint numAddresses = (header.numBricks + MAX_BRICKS_PER_TEMPLATE - 1) / MAX_BRICKS_PER_TEMPLATE;
        InterlockedAdd(numClusterAddresses, numAddresses, addressOffset);
        InterlockedAdd(numVoxelClusterAddresses, numAddresses, voxelAddressOffset);
    }
    GroupMemoryBarrierWithGroupSync();

    if (groupIndex == 0)
    {
        InterlockedAdd(globals[GLOBALS_CLAS_COUNT_INDEX], numClusterAddresses, clusterStartIndex);
        InterlockedAdd(globals[GLOBALS_TEMPLATE_CLUSTER_COUNT], numVoxelClusterAddresses, voxelClusterStartIndex);

        clasPageInfos[pageIndex].addressStartIndex = pc.clusterOffset + clusterStartIndex;
        clasPageInfos[pageIndex].tempClusterOffset = clusterStartIndex;
        clasPageInfos[pageIndex].clasSize = 0;
        //clasPageInfos[pageIndex].numTriangleClusters = numTriangleClusters;
    }
    GroupMemoryBarrierWithGroupSync();

    if (header.numTriangles)
    {
        uint vertexBufferOffset, indexBufferOffset;

        uint decodeIndex = clusterStartIndex + addressOffset;
        // TODO: this is wrong
        uint buildIndex = clusterStartIndex + addressOffset;
        //uint instantiateIndex = voxelClusterStartIndex + voxelAddressOffset;

        InterlockedAdd(globals[GLOBALS_VERTEX_BUFFER_OFFSET_INDEX], header.numVertices, vertexBufferOffset);
        InterlockedAdd(globals[GLOBALS_INDEX_BUFFER_OFFSET_INDEX], header.numTriangles * 3, indexBufferOffset);

        BUILD_CLUSTERS_TRIANGLE_INFO desc = (BUILD_CLUSTERS_TRIANGLE_INFO)0;
        desc.clusterId = pageIndex * MAX_CLUSTERS_PER_PAGE + clusterID;
        desc.clusterFlags = 0;
        desc.triangleCount = header.numTriangles;
        desc.vertexCount = header.numVertices;
        desc.positionTruncateBitCount = 0;
        desc.indexFormat = 1; // VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV
        desc.opacityMicromapIndexFormat = 0;
        desc.baseGeometryIndexAndFlags = 0;
        desc.indexBufferStride = 0; // tightly packed
        desc.vertexBufferStride = 0;
        desc.geometryIndexAndFlagsBufferStride = 0;
        desc.opacityMicromapIndexBufferStride = 0;
        desc.indexBuffer = indexBufferBaseAddress + indexBufferOffset * 1;
        desc.vertexBuffer = vertexBufferBaseAddress + vertexBufferOffset * 12;
        desc.geometryIndexAndFlagsBuffer = 0;
        desc.opacityMicromapArray = 0;
        desc.opacityMicromapIndexBuffer = 0;

        buildClusterTriangleInfos[buildIndex] = desc;

        DecodeClusterData clusterData;
        clusterData.pageIndex = pageIndex;
        clusterData.clusterIndex = clusterID;
        clusterData.indexBufferOffset = indexBufferOffset;
        clusterData.vertexBufferOffset = vertexBufferOffset;

        decodeClusterDatas[decodeIndex] = clusterData;
    }
    else
    {
        uint decodeIndex = clusterStartIndex + addressOffset;
        uint instantiateIndex = voxelClusterStartIndex + voxelAddressOffset;

        for (uint i = 0; i < header.numBricks; i+= MAX_BRICKS_PER_TEMPLATE)
        {
            uint numBricksInTemplate = min(header.numBricks - i, MAX_BRICKS_PER_TEMPLATE);
            uint vertexBufferOffset;

            InterlockedAdd(globals[GLOBALS_VERTEX_BUFFER_OFFSET_INDEX], numBricksInTemplate * 8, vertexBufferOffset);

            INSTANTIATE_CLUSTER_TEMPLATE_INFO instantiateInfo = (INSTANTIATE_CLUSTER_TEMPLATE_INFO)0;
            instantiateInfo.clusterIdOffset =  pageIndex * MAX_CLUSTERS_PER_PAGE + clusterID;
            instantiateInfo.clusterTemplateAddress = templateAddresses[numBricksInTemplate - 1];
            instantiateInfo.vertexBuffer.startAddress = vertexBufferBaseAddress + vertexBufferOffset * 12;
            instantiateInfo.vertexBuffer.strideInBytes = 12;

            instantiateInfos[instantiateIndex++] = instantiateInfo;

            DecodeClusterData clusterData;
            clusterData.pageIndex = pageIndex;
            clusterData.clusterIndex = clusterID;
            clusterData.indexBufferOffset = i;
            clusterData.vertexBufferOffset = vertexBufferOffset;

            decodeClusterDatas[decodeIndex++] = clusterData;
        }
    }
}
