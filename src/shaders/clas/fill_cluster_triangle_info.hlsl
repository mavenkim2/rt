#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../wave_intrinsics.hlsli"
#include "../dense_geometry.hlsli"

StructuredBuffer<ClusterPageData> clusterPages : register(t0);
RWStructuredBuffer<BUILD_CLUSTERS_TRIANGLE_INFO> buildClusterTriangleInfos : register(u1);
RWStructuredBuffer<BLASData> blasDatas : register(u2);
RWStructuredBuffer<DecodeClusterData> decodeClusterDatas : register(u3);
RWStructuredBuffer<uint> globals : register(u4);

[[vk::push_constant]] FillClusterTriangleInfoPushConstant pc;

[numthreads(MAX_CLUSTERS_PER_PAGE, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID: SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    ClusterPageData page = clusterPages[groupID.x];
    if (groupIndex >= page.clusterCount) return;

    InterlockedAdd(globals[GLOBALS_CLAS_COUNT_INDEX], 1);

    uint64_t indexBufferBaseAddress = ((uint64_t(pc.indexBufferBaseAddressHighBits) << 32) | (pc.indexBufferBaseAddressLowBits));
    uint64_t vertexBufferBaseAddress = ((uint64_t(pc.vertexBufferBaseAddressHighBits) << 32) | (pc.vertexBufferBaseAddressLowBits));

    uint clusterID = page.clusterStart + groupIndex;
    PackedDenseGeometryHeader packedHeader = denseGeometryHeaders[clusterID];
    DenseGeometry header = GetDenseGeometryHeader(packedHeader, clusterID);

    uint vertexBufferOffset, indexBufferOffset;
    WaveInterlockedAdd(globals[GLOBALS_VERTEX_BUFFER_OFFSET_INDEX], header.numVertices, vertexBufferOffset);
    WaveInterlockedAdd(globals[GLOBALS_INDEX_BUFFER_OFFSET_INDEX], header.numTriangles * 3, indexBufferOffset);

    if (groupIndex == 0)
    {
        InterlockedAdd(blasDatas[page.blasIndex].clusterCount, page.clusterCount);
    }

    BUILD_CLUSTERS_TRIANGLE_INFO desc = (BUILD_CLUSTERS_TRIANGLE_INFO)0;
    desc.clusterId = clusterID;
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

    buildClusterTriangleInfos[clusterID] = desc;

    DecodeClusterData clusterData;
    clusterData.indexBufferOffset = indexBufferOffset;
    clusterData.vertexBufferOffset = vertexBufferOffset;
    clusterData.blasIndex = page.blasIndex;

    decodeClusterDatas[clusterID] = clusterData;
}
