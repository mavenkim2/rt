#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../dense_geometry.hlsli"

[[vk::push_constant]] DecodePushConstant pc;

RWByteAddressBuffer indexBuffer : register(u0);
RWStructuredBuffer<float3> decodeVertexBuffer : register(u1);
RWStructuredBuffer<BUILD_CLUSTER_TRIANGLE_INFO> buildClasDescs : register(u2);
RWStructuredBuffer<uint> globals : register(u3);


[numthreads(32, 1, 1)] 
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 dtID : SV_DispatchThreadID)
{
    uint headerIndex = groupID.x;

    //if (all(dtID == 0))
    //{
        //globals[GLOBALS_BLAS_COUNT_INDEX] = 1;
    //}

    if (headerIndex >= pc.numHeaders) return;

    PackedDenseGeometryHeader denseHeader = denseGeometryHeaders[groupID.x];

    DenseGeometry header = GetDenseGeometryHeader(denseHeader, groupID.x);

    // Decode triangle indices
    uint waveNumActiveLanes = min(32, WaveGetLaneCount());

    // Decode indices
    uint indexBufferOffset = 0;
    if (WaveIsFirstLane())
    {
        InterlockedAdd(globals[GLOBALS_INDEX_BUFFER_OFFSET_INDEX], header.numTriangles * 3, indexBufferOffset);
    }

    for (uint triangleIndex = groupIndex; triangleIndex < header.numTriangles; triangleIndex += waveNumActiveLanes)
    {
        // write u8 indices
        uint3 triangleIndices = header.DecodeTriangle(triangleIndex);

        uint indexBits = triangleIndices[0] | (triangleIndices[1] << 8u) | (triangleIndices[2] << 16u);
        uint2 offset = GetAlignedAddressAndBitOffset(indexBufferOffset + triangleIndex * 3, 0);

        // Write the uint8 indices atomically
        if (indexBits + offset[1] >= 32)
        {
            uint mask = ~0u << ((indexBits + offset[1]) & 31u);
            indexBuffer.InterlockedAnd(offset[0] + 4, mask);
            indexBuffer.InterlockedOr(offset[0] + 4, indexBits >> (32 - offset[1]));
        }
        uint mask = ~(((1u << 24u) - 1u) << offset[1]);
        indexBuffer.InterlockedAnd(offset[0], mask);
        indexBuffer.InterlockedOr(offset[0], indexBits << offset[1]);
    }

    // Decode vertices
    uint vertexBufferOffset = 0;
    if (WaveIsFirstLane())
    {
        InterlockedAdd(globals[GLOBALS_VERTEX_BUFFER_OFFSET_INDEX], header.numVertices, vertexBufferOffset);
    }

    for (uint vertexIndex = groupIndex; vertexIndex < header.numVertices; vertexIndex += waveNumActiveLanes)
    {
        float3 position = header.DecodePosition(vertexIndex);
        decodeVertexBuffer[vertexBufferOffset + vertexIndex] = position;
    }


    // TODO: maybe this goes to a separate pass if culling/tessellation is done
    // Write CLAS build infos
    if (WaveIsFirstLane())
    {
        InterlockedAdd(globals[GLOBALS_CLAS_COUNT_INDEX], 1);

        uint64_t indexBufferBaseAddress = ((pc.indexBufferBaseAddressHighBits << 32) | (pc.indexBufferBaseAddressLowBits));
        uint64_t vertexBufferBaseAddress = ((pc.vertexBufferBaseAddressHighBits << 32) | (pc.vertexBufferBaseAddressLowBits));

        BUILD_CLUSTER_TRIANGLE_INFO desc = (BUILD_CLUSTER_TRIANGLE_INFO)0;
        desc.clusterId = groupID.x;
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

        buildClasDescs[groupID.x] = desc;
    }
}
