#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../dense_geometry.hlsli"

RWByteAddressBuffer indexBuffer : register(u0);
RWStructuredBuffer<float3> decodeVertexBuffer : register(u1);

StructuredBuffer<DecodeClusterData> decodeClusterDatas : register(t2);
StructuredBuffer<uint> globals : register(t3);
RWStructuredBuffer<AABB> aabbs : register(u4);

#define THREAD_GROUP_SIZE 32
[numthreads(THREAD_GROUP_SIZE, 1, 1)] 
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    uint clusterID = groupID.x;

    if (clusterID >= globals[GLOBALS_ALL_CLUSTER_COUNT_INDEX]) return;

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
    else if (header.numBricks)
    {
        for (uint brickIndex = groupIndex; brickIndex < header.numBricks; brickIndex += THREAD_GROUP_SIZE)
        {
            Brick brick = header.DecodeBrick(brickIndex);
            float3 position = header.DecodePosition(brickIndex);

            uint3 maxP;
            GetBrickMax(brick.bitMask, maxP);

#ifndef TRACE_BRICKS
            uint aabbOffset = header.aabbOffset + brickIndex * 64;
            for (uint i = 0; i < 64; i++)
            {
                uint bit = i;
                if ((brick.bitMask >> i) & 1)
                {
                    uint x         = bit & 3u;
                    uint y         = (bit >> 2u) & 3u;
                    uint z         = bit >> 4u;
                    float3 aabbMin = position + float3(x, y, z) * header.lodError;
                    float3 aabbMax = aabbMin + header.lodError;

                    AABB aabb;
                    aabb.minX = aabbMin.x;
                    aabb.minY = aabbMin.y;
                    aabb.minZ = aabbMin.z;
                    aabb.maxX = aabbMax.x;
                    aabb.maxY = aabbMax.y;
                    aabb.maxZ = aabbMax.z;

                    aabbs[aabbOffset++] = aabb;
                }
                else
                {
                    AABB aabb = (AABB)0;
                    aabb.minX = 0.f/0.f;
                    aabbs[aabbOffset++] = aabb;
                }
            }

#else    
            float3 aabbMin = position;
            float3 aabbMax = position + Vec3f(maxP) * lodError;

            AABB aabb;
            aabb.minX = aabbMin.x;
            aabb.minY = aabbMin.y;
            aabb.minZ = aabbMin.z;
            aabb.maxX = aabbMax.x;
            aabb.maxY = aabbMax.y;
            aabb.maxZ = aabbMax.z;

            aabbs[aabbOffset++] = aabb;
#endif
        }
    }
}
