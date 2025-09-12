#include "../common.hlsli"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"

StructuredBuffer<row_major float3x4> instanceTransforms : register(t0);
StructuredBuffer<AABB> resourceAABBs : register(t1);
RWStructuredBuffer<AABB> aabbs : register(u2);
RWStructuredBuffer<uint> partitionErrorThresholds: register(u3);
StructuredBuffer<uint> instanceGroupTransformOffsets: register(t4);

[numthreads(32, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 dtID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    uint instanceGroupIndex = groupID.x;
    uint offset = instanceGroupTransformOffsets[instanceGroupIndex];
    uint end = instanceGroupTransformOffsets[instanceGroupIndex + 1];

    for (uint transformIndex = offset + groupIndex; transformIndex < end; transformIndex += 32)
    {
        float3x4 worldFromObject = instanceTransforms[transformIndex];
        //GPUInstance instance = gpuInstances[instanceIndex];

        // TODO: 
        AABB aabb = resourceAABBs[0];

        float3 minP = float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 maxP = float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (int z = 0; z < 2; z++)
        {
            for (int y = 0; y < 2; y++)
            {
                for (int x = 0; x < 2; x++)
                {
                    float3 p = float3(x ? aabb.maxX : aabb.minX, y ? aabb.maxY : aabb.minY, z ? aabb.maxZ : aabb.minZ);
                    float3 pos = mul(worldFromObject, float4(p, 1.f));
                    minP = min(minP, pos);
                    maxP = max(maxP, pos);
                }
            }
        }

#if 0
        float error = 0.f;
        float maxScale = 0.f;
        float instanceError = error * maxScale;

        uint uintInstanceError = asuint(instanceError);
        InterlockedMax(partitionErrorThresholds[instance.groupIndex], uintInstanceError);
#endif

        AABB transformedAABB;
        transformedAABB.minX = minP.x;//- 100.f;
        transformedAABB.minY = minP.y;//- 100.f;
        transformedAABB.minZ = minP.z;//- 100.f;
        transformedAABB.maxX = maxP.x;//+ 100.f;
        transformedAABB.maxY = maxP.y;//+ 100.f;
        transformedAABB.maxZ = maxP.z;//+ 100.f;
        aabbs[transformIndex] = transformedAABB;
    }
}
