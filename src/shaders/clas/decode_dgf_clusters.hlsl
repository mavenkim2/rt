#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../dense_geometry.hlsli"

RWByteAddressBuffer indexBuffer : register(u0);
RWStructuredBuffer<float3> decodeVertexBuffer : register(u1);

StructuredBuffer<DecodeClusterData> decodeClusterDatas : register(t2);
StructuredBuffer<uint> globals : register(t3);

#define THREAD_GROUP_SIZE 32
[numthreads(THREAD_GROUP_SIZE, 1, 1)] 
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
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

    // Decode triangle indices
    if (header.numBricks)
    {
        float voxelSize = header.lodError;
        for (uint brickIndex = groupIndex; brickIndex < header.numBricks; brickIndex += THREAD_GROUP_SIZE)
        {
            Brick brick = header.DecodeBrick(brickIndex);
            float3 position = header.DecodePosition(brick.vertexOffset);

            // Get the brick bounds
            uint3 maxP;
            GetBrickMax(brick.bitMask, maxP);
            
            float3 aabbMin = position;
            float3 aabbMax = position + float3(maxP) * voxelSize;

            for (int z = 0; z < 2; z++)
            {
                for (int y = 0; y < 2; y++)
                {
                    for (int x = 0; x < 2; x++)
                    {
                        float3 position = float3(x ? aabbMax.x : aabbMin.x, y ? aabbMax.y : aabbMin.y, z ? aabbMax.z : aabbMin.z);
                        uint i = 4 * z + 2 * y + x;
                        decodeVertexBuffer[vertexBufferOffset + brickIndex * 8u + i] = position;
                    }
                }
            }
        }
    }
    else 
    {
        // Decode indices
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
}
