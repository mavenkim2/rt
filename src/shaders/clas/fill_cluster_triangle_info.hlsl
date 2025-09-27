#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../wave_intrinsics.hlsli"
#include "../dense_geometry.hlsli"

#if 0
StructuredBuffer<uint> pageIndices : register(t0);
RWStructuredBuffer<BUILD_CLUSTERS_TRIANGLE_INFO> buildClusterTriangleInfos : register(u1);
RWStructuredBuffer<DecodeClusterData> decodeClusterDatas : register(u2);
RWStructuredBuffer<uint> globals : register(u3);
RWStructuredBuffer<CLASPageInfo> clasPageInfos : register(u4);
#else
RWStructuredBuffer<BUILD_CLUSTERS_TRIANGLE_INFO> buildClusterTriangleInfos : register(u0);
RWStructuredBuffer<DecodeClusterData> decodeClusterDatas : register(u1);
RWStructuredBuffer<uint> globals : register(u2);
#endif

[[vk::push_constant]] FillClusterTriangleInfoPushConstant pc;

#if 0
groupshared uint clusterStartIndex;
groupshared uint numTriangleClusters;
#endif

//[numthreads(MAX_CLUSTERS_PER_PAGE, 1, 1)]
[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID: SV_GroupID, uint groupIndex : SV_GroupIndex)
{
#if 0
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
        numTriangleClusters = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    if (groupIndex >= numClusters) return;

    uint64_t indexBufferBaseAddress = ((uint64_t(pc.indexBufferBaseAddressHighBits) << 32) | (pc.indexBufferBaseAddressLowBits));
    uint64_t vertexBufferBaseAddress = ((uint64_t(pc.vertexBufferBaseAddressHighBits) << 32) | (pc.vertexBufferBaseAddressLowBits));

    uint clusterID = groupIndex;

    DenseGeometry header = GetDenseGeometryHeader(basePageAddress, numClusters, clusterID);

    uint addressOffset = 0;
    if (header.numTriangles)
    {
        InterlockedAdd(numTriangleClusters, 1, addressOffset);
    }
    GroupMemoryBarrierWithGroupSync();

    if (groupIndex == 0)
    {
        InterlockedAdd(globals[GLOBALS_CLAS_COUNT_INDEX], numTriangleClusters, clusterStartIndex);

        clasPageInfos[pageIndex].addressStartIndex = pc.clusterOffset + clusterStartIndex;
        clasPageInfos[pageIndex].tempClusterOffset = clusterStartIndex;
        clasPageInfos[pageIndex].clasSize = 0;
        clasPageInfos[pageIndex].numTriangleClusters = numTriangleClusters;
    }
    GroupMemoryBarrierWithGroupSync();

    if (header.numTriangles)
    {
        uint vertexBufferOffset, indexBufferOffset;

        uint descriptorIndex = clusterStartIndex + addressOffset;

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

        buildClusterTriangleInfos[descriptorIndex] = desc;

        uint decodeIndex = clusterStartIndex + addressOffset;
        DecodeClusterData clusterData;
        clusterData.pageIndex = pageIndex;
        clusterData.clusterIndex = clusterID;
        clusterData.indexBufferOffset = indexBufferOffset;
        clusterData.vertexBufferOffset = vertexBufferOffset;

        decodeClusterDatas[decodeIndex] = clusterData;
    }
#else

    uint numClusters = pc.numClusters;
    uint baseAddress = pc.baseAddress;
    if (dispatchThreadID.x == 0)
    {
        globals[GLOBALS_CLAS_COUNT_INDEX] = numClusters;
        globals[GLOBALS_DECODE_INDIRECT_Y] = 1;
        globals[GLOBALS_DECODE_INDIRECT_Z] = 1;
    }

    uint clusterIndex = dispatchThreadID.x;
    if (clusterIndex >= numClusters) return;

    DenseGeometry header = GetDenseGeometryHeader2(clusterIndex, baseAddress);

    uint64_t indexBufferBaseAddress = ((uint64_t(pc.indexBufferBaseAddressHighBits) << 32) | (pc.indexBufferBaseAddressLowBits));
    uint64_t vertexBufferBaseAddress = ((uint64_t(pc.vertexBufferBaseAddressHighBits) << 32) | (pc.vertexBufferBaseAddressLowBits));

    uint addressOffset = 0;

    uint vertexBufferOffset, indexBufferOffset;

    uint descriptorIndex = clusterIndex;

    InterlockedAdd(globals[GLOBALS_VERTEX_BUFFER_OFFSET_INDEX], header.numVertices, vertexBufferOffset);
    InterlockedAdd(globals[GLOBALS_INDEX_BUFFER_OFFSET_INDEX], header.numTriangles * 3, indexBufferOffset);

    BUILD_CLUSTERS_TRIANGLE_INFO desc = (BUILD_CLUSTERS_TRIANGLE_INFO)0;
    desc.clusterId = clusterIndex;
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

    buildClusterTriangleInfos[descriptorIndex] = desc;

    DecodeClusterData clusterData;
    clusterData.pageIndex = 0;//pageIndex;
    clusterData.clusterIndex = clusterIndex;
    clusterData.indexBufferOffset = indexBufferOffset;
    clusterData.vertexBufferOffset = vertexBufferOffset;

    decodeClusterDatas[descriptorIndex] = clusterData;
#endif
}
