#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../dense_geometry.hlsli"

StructuredBuffer<uint> globals : register(t0);
RWStructuredBuffer<AABB> aabbs : register(u1);
StructuredBuffer<VoxelPageDecodeData> decodeDatas : register(t2);

#define THREAD_GROUP_SIZE 32
[numthreads(THREAD_GROUP_SIZE, 1, 1)] 
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    uint clusterID = groupID.x;

    //if (clusterID >= globals[GLOBALS_ALL_CLUSTER_COUNT_INDEX]) return;

    VoxelPageDecodeData data = decodeDatas[clusterID];
    uint pageIndex = data.pageIndex;
    uint basePageAddress = GetClusterPageBaseAddress(pageIndex);
    uint numClusters = GetNumClustersInPage(basePageAddress);

    if (groupID.x >= numClusters) return;

    for (uint clusterIndex = groupIndex; clusterIndex < numClusters; clusterIndex++)
    {
        DenseGeometry header = GetDenseGeometryHeader(basePageAddress, numClusters, clusterIndex);

        for (uint brickIndex = 0; brickIndex < header.numBricks; brickIndex++)
        {
            Brick brick = header.DecodeBrick(brickIndex);
            float3 position = header.DecodePosition(brickIndex);

            uint3 maxP;
            GetBrickMax(brick.bitMask, maxP);

#ifndef TRACE_BRICKS
            uint aabbOffset = data.offset + header.aabbOffset + brickIndex * 64;
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
