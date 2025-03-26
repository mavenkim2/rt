#include "dense_geometry.hlsli"

struct DecodePushConstant
{
    uint numHeaders;
};

struct DecodeGlobals 
{
};

[[vk::push_constant] DecodePushConstant pc;
ByteAddressBuffer denseGeometryData : register(t0);

RWByteAddressBuffer indexBuffer : register(u1);
RWStructuredBuffer<float3> decodeVertexBuffer : register(u2);
RWStructuredBuffer<BuildClasDesc> buildClasDescs : register(u3);
RWStructuredBuffer<DecodeGlobals> : register();

[numthreads(32, 1, 1)] 
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    uint headerIndex = groupID.x;
    if (headerIndex >= pc.numHeaders) return;

    // TODO: this isn't right yet
    DenseGeometry header = GetDenseGeometryHeader(headerIndex);
    
    // Decode triangle indices
    uint waveNumActiveLanes = min(32, WaveGetLaneCount());

    // Decode indices
    uint indexBufferOffset = 0;
    if (WaveIsFirstLane())
    {
        InterlockedAdd(globals.indexBufferOffset, header.numTriangles * 3, indexBufferOffset);
    }

    for (uint triangleIndex = groupIndex; triangleIndex < header.numTriangles; triangleIndex += waveNumActiveLanes)
    {
        // write u8 indices
        uint3 triangleIndices = ?;

        uint indexBits = triangleIndices[0] | (triangleIndices[1] << 8u) | (triangleIndices[2] << 16u);
        uint2 offset = GetAlignedAddressAndBitOffset(indexBufferOffset + triangleIndex * 3, 0);

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
        InterlockedAdd(globals.vertexBufferOffset, header.numVertices, vertexBufferOffset);
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
        BuildClasDesc desc = (BuildClasDesc)0;
        desc.clusterId = groupID.x;
        desc.clusterFlags = 0;
        desc.triangleCount = header.numTriangles;
        desc.vertexCount = header.numVertices;
        desc.positionTruncateBitCount = 0;
        desc.indexFormat = 1; // VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV
        desc.opacityMicromapIndexFormat = 0;
        desc.baseGeometryIndexAndFlags = ?;
        desc.indexBufferStride = 0; // tightly packed
        desc.vertexBufferStride = 0;
        desc.geometryIndexAndFlagBufferStride = ?;
        desc.opacityMicromapBufferStride = 0;
        desc.indexBuffer = indexBufferBaseAddress + indexBufferOffset * 1;
        desc.vertexBuffer = vertexBufferBaseAddress + vertexBufferOffset * 12;
        desc.geomemtryIndexAndFlagsBuffer = ?;
        desc.opacityMicromapArray = 0;
        desc.opacityMicromapIndexBuffer = 0;

        buildClasDescs[groupID.x] = desc;
    }
}
