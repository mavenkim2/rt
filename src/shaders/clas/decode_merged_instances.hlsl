#include "../common.hlsli"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"

StructuredBuffer<GPUTransform> instanceTransforms : register(t0);
StructuredBuffer<AABB> resourceAABBs : register(t1);
RWStructuredBuffer<AABB> aabbs : register(u2);
RWStructuredBuffer<PartitionInfo> partitionInfos: register(u3);
StructuredBuffer<GPUTruncatedEllipsoid> truncatedEllipsoids : register(t4);
StructuredBuffer<uint2> partitionsAndOffset : register(t5);
StructuredBuffer<uint> resourceIDs : register(t6);

[numthreads(32, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 dtID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    uint instanceGroupIndex = partitionsAndOffset[groupID.x].x;
    uint aabbOffset = partitionsAndOffset[groupID.x].y;
    PartitionInfo info = partitionInfos[instanceGroupIndex];

    for (uint index = groupIndex; index < info.transformCount; index += 32)
    {
        uint transformIndex = info.transformOffset + index;
        uint aabbIndex = aabbOffset + index;
        //GPUInstance instance = gpuInstances[instanceIndex];

        GPUTruncatedEllipsoid ellipsoid = truncatedEllipsoids[resourceIDs[transformIndex]];
        float3 aabbMin = ellipsoid.sphere.xyz - ellipsoid.sphere.w;
        float3 aabbMax = ellipsoid.sphere.xyz + ellipsoid.sphere.w;

        float3x4 worldFromObject = ConvertGPUMatrix(instanceTransforms[transformIndex], info.base, info.scale);
        float3x4 ellipsoidFromObject = ellipsoid.transform;
        float3x4 objectFromEllipsoid_ = Inverse(ellipsoidFromObject);
        float4x4 objectFromEllipsoid = float4x4(
                        objectFromEllipsoid_[0][0], objectFromEllipsoid_[0][1], objectFromEllipsoid_[0][2], objectFromEllipsoid_[0][3], 
                        objectFromEllipsoid_[1][0], objectFromEllipsoid_[1][1], objectFromEllipsoid_[1][2], objectFromEllipsoid_[1][3], 
                        objectFromEllipsoid_[2][0], objectFromEllipsoid_[2][1], objectFromEllipsoid_[2][2], objectFromEllipsoid_[2][3], 
                        0, 0, 0, 1.f);
        float3x4 worldFromEllipsoid = mul(worldFromObject, objectFromEllipsoid);

        float3 minP = float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 maxP = float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (int z = 0; z < 2; z++)
        {
            for (int y = 0; y < 2; y++)
            {
                for (int x = 0; x < 2; x++)
                {
                    float3 p = float3(x ? aabbMax.x : aabbMin.x, y ? aabbMax.y : aabbMin.y, z ? aabbMax.z : aabbMin.z);
                    float3 pos = mul(worldFromEllipsoid, float4(p, 1.f));
                    minP = min(minP, pos);
                    maxP = max(maxP, pos);
                }
            }
        }

        AABB transformedAABB;
        transformedAABB.minX = minP.x;
        transformedAABB.minY = minP.y;
        transformedAABB.minZ = minP.z;
        transformedAABB.maxX = maxP.x;
        transformedAABB.maxY = maxP.y;
        transformedAABB.maxZ = maxP.z;
        aabbs[aabbIndex] = transformedAABB;
    }
}
