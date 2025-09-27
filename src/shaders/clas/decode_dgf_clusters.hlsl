#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../dense_geometry.hlsli"

RWByteAddressBuffer indexBuffer : register(u0);
RWStructuredBuffer<float3> decodeVertexBuffer : register(u1);

StructuredBuffer<DecodeClusterData> decodeClusterDatas : register(t2);
StructuredBuffer<uint> globals : register(t3);

#define THREAD_GROUP_SIZE 32
[numthreads(THREAD_GROUP_SIZE, 1, 1)] 
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 dispatchThreadID : SV_DispatchThreadID)
{
#if 0
    uint clusterID = groupID.x;

    if (clusterID >= globals[GLOBALS_CLAS_COUNT_INDEX]) return;

    DecodeClusterData decodeClusterData = decodeClusterDatas[clusterID];
    uint pageIndex = decodeClusterData.pageIndex;
    uint clusterIndex = decodeClusterData.clusterIndex;
    uint indexBufferOffset = decodeClusterData.indexBufferOffset;
    uint vertexBufferOffset = decodeClusterData.vertexBufferOffset;

    uint basePageAddress = GetClusterPageBaseAddress(pageIndex);
    uint numClusters = GetNumClustersInPage(basePageAddress);
    DenseGeometry header = GetDenseGeometryHeader(basePageAddress, numClusters, clusterIndex);

    if (header.numTriangles)
    {
        // Decode triangle indices
        for (uint triangleIndex = groupIndex; triangleIndex < header.numTriangles; triangleIndex += THREAD_GROUP_SIZE)
        {
            // write u8 indices
            uint3 triangleIndices = header.DecodeTriangle(triangleIndex);

            uint indexBits = triangleIndices[0] | (triangleIndices[1] << 8u) | (triangleIndices[2] << 16u);
            uint2 offset = GetAlignedAddressAndBitOffset(indexBufferOffset + triangleIndex * 3, 0);

            const uint writeBitSize = 24;

            // Write the uint8 indices atomically
            if (offset[1] + writeBitSize >= 32)
            {
                uint mask = ~0u << ((writeBitSize + offset[1]) & 31u);
                indexBuffer.InterlockedAnd(offset[0] + 4, mask);
                indexBuffer.InterlockedOr(offset[0] + 4, indexBits >> (32 - offset[1]));
            }
            uint mask = ~(((1u << writeBitSize) - 1u) << offset[1]);
            indexBuffer.InterlockedAnd(offset[0], mask);
            indexBuffer.InterlockedOr(offset[0], indexBits << offset[1]);
        }

        // Decode vertices
        for (uint vertexIndex = groupIndex; vertexIndex < header.numVertices; vertexIndex += THREAD_GROUP_SIZE)
        {
            float3 position = header.DecodePosition(vertexIndex);
            decodeVertexBuffer[vertexBufferOffset + vertexIndex] = position;
        }
    }
#else
    uint id = dispatchThreadID.x;
    // TODO
    uint baseAddress = 0;

    if (id >= globals[GLOBALS_CLAS_COUNT_INDEX]) return;

    DecodeClusterData decodeClusterData = decodeClusterDatas[id];
    //uint pageIndex = decodeClusterData.pageIndex;
    uint clusterIndex = decodeClusterData.clusterIndex;
    uint indexBufferOffset = decodeClusterData.indexBufferOffset;
    uint vertexBufferOffset = decodeClusterData.vertexBufferOffset;

    DenseGeometry header = GetDenseGeometryHeader2(clusterIndex, baseAddress);

    // Decode triangle indices
    for (uint triangleIndex = 0; triangleIndex < header.numTriangles; triangleIndex++)
    {
        // write u8 indices
        uint3 triangleIndices = header.DecodeTriangle(triangleIndex);

        uint indexBits = triangleIndices[0] | (triangleIndices[1] << 8u) | (triangleIndices[2] << 16u);
        uint2 offset = GetAlignedAddressAndBitOffset(indexBufferOffset + triangleIndex * 3, 0);

        const uint writeBitSize = 24;

        // Write the uint8 indices atomically
        if (offset[1] + writeBitSize >= 32)
        {
            uint mask = ~0u << ((writeBitSize + offset[1]) & 31u);
            indexBuffer.InterlockedAnd(offset[0] + 4, mask);
            indexBuffer.InterlockedOr(offset[0] + 4, indexBits >> (32 - offset[1]));
        }
        uint mask = ~(((1u << writeBitSize) - 1u) << offset[1]);
        indexBuffer.InterlockedAnd(offset[0], mask);
        indexBuffer.InterlockedOr(offset[0], indexBits << offset[1]);
    }

    // Decode vertices
    for (uint vertexIndex = 0; vertexIndex < header.numVertices; vertexIndex++)
    {
        float3 position = header.DecodePosition(vertexIndex);
        decodeVertexBuffer[vertexBufferOffset + vertexIndex] = position;
    }
#endif
}
