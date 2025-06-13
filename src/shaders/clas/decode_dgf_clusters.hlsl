#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../dense_geometry.hlsli"

[[vk::push_constant]] FillClusterTriangleInfoPushConstant pc;

RWByteAddressBuffer indexBuffer : register(u0);
RWStructuredBuffer<float3> decodeVertexBuffer : register(u1);

StructuredBuffer<DecodeClusterData> decodeClusterData : register(t2);

[numthreads(32, 1, 1)] 
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 dtID : SV_DispatchThreadID)
{
    uint headerIndex = groupID.x;

    //if (all(dtID == 0))
    //{
        //globals[GLOBALS_BLAS_COUNT_INDEX] = 1;
    //}

    if (headerIndex >= pc.numClusters) return;

    uint clusterID = groupID.x;

    uint indexBufferOffset = decodeClusterData[clusterID].indexBufferOffset;
    uint vertexBufferOffset = decodeClusterData[clusterID].vertexBufferOffset;

    PackedDenseGeometryHeader denseHeader = denseGeometryHeaders[clusterID];
    DenseGeometry header = GetDenseGeometryHeader(denseHeader, clusterID);

    //if (!header.numTriangles == buildClusterTriangleInfos[clusterID].numTriangles)
    //{
        //printf("bad\n");
    //}

    // Decode triangle indices
    uint waveNumActiveLanes = min(32, WaveGetLaneCount());

    // Decode indices
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
    for (uint vertexIndex = groupIndex; vertexIndex < header.numVertices; vertexIndex += waveNumActiveLanes)
    {
        float3 position = header.DecodePosition(vertexIndex);
        decodeVertexBuffer[vertexBufferOffset + vertexIndex] = position;
    }
}
