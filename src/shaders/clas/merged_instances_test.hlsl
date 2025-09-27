#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "lod_error_test.hlsli"

RWStructuredBuffer<PartitionInfo> partitionInfos : register(u0);
RWStructuredBuffer<uint> globals : register(u1);

RWStructuredBuffer<uint> visiblePartitions : register(u2);
RWStructuredBuffer<uint2> freedPartitions : register(u3);
RWStructuredBuffer<uint> instanceFreeList : register(u4);
ConstantBuffer<GPUScene> gpuScene : register(b5);
Texture2D<float> depthPyramid : register(t6);

#define ENABLE_OCCLUSION
#include "cull.hlsli"

[[vk::push_constant]] MergedInstancesPushConstant pc;

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    if (dtID.x == 0)
    {
        globals[GLOBALS_VISIBLE_PARTITION_INDIRECT_Y] = 1;
        globals[GLOBALS_VISIBLE_PARTITION_INDIRECT_Z] = 1;
    }

    uint partitionIndex = dtID.x;
    if (partitionIndex >= pc.num) return;
    PartitionInfo info = partitionInfos[partitionIndex];

    uint flags = info.flags;
    bool useProxies = false;
    float3x4 renderFromObject = float3x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    float error = info.lodError;
    float4 lodBounds = info.lodBounds;

    if ((flags & PARTITION_FLAG_HAS_PROXIES))
    {
        float3 minP = lodBounds.xyz - lodBounds.w;
        float3 maxP = lodBounds.xyz + lodBounds.w;

        float4 aabb;
        float maxZ;
        float test;
        bool cull = FrustumCull(gpuScene.clipFromRender, renderFromObject, 
                minP, maxP, gpuScene.p22, gpuScene.p23, aabb, maxZ, test);

        useProxies = cull;

#if 0
        if (!cull && !pc.firstFrame)
        {
            uint lod;
            bool occluded = HZBOcclusionTest(aabb, maxZ, int2(gpuScene.width, gpuScene.height), lod);
            useProxies = occluded;
            partitionInfos[partitionIndex].debug0 = aabb;
            partitionInfos[partitionIndex].debug1 = lod;
        }
#endif
    }
#if 0
    else if ((flags & PARTITION_FLAG_HAS_PROXIES) && pc.firstFrame)
    {
        useProxies = true;
    }

    Translate(renderFromObject, -gpuScene.cameraP);

    if (!useProxies && (flags & PARTITION_FLAG_HAS_PROXIES))
    {
        float test;
        float2 edgeScales = TestNode(renderFromObject, gpuScene.cameraFromRender, lodBounds, 1.f, test, false);
        useProxies = error * gpuScene.lodScale < edgeScales.x;
    }
#endif
    //useProxies = false;

    partitionInfos[partitionIndex].test = useProxies;
    if (useProxies)
    {
        if (info.flags & PARTITION_FLAG_INSTANCES_RENDERED)
        {
            partitionInfos[partitionIndex].flags &= ~PARTITION_FLAG_INSTANCES_RENDERED;
            uint descriptorIndex;
            InterlockedAdd(globals[GLOBALS_FREED_PARTITION_COUNT], 1, descriptorIndex);
            freedPartitions[descriptorIndex] = uint2(partitionIndex, GPU_INSTANCE_FLAG_INDIV);
        }
#if 0
        if ((info.flags & PARTITION_FLAG_PROXY_RENDERED) == 0)
        {
            partitionInfos[partitionIndex].flags |= PARTITION_FLAG_PROXY_RENDERED;
            uint descriptorIndex;
            InterlockedAdd(globals[GLOBALS_VISIBLE_PARTITION_COUNT], 1, descriptorIndex);
            visiblePartitions[descriptorIndex] = partitionIndex;
        }
#endif
    }
    else 
    {
#if 0
        if (info.flags & PARTITION_FLAG_PROXY_RENDERED)
        {
            partitionInfos[partitionIndex].flags &= ~PARTITION_FLAG_PROXY_RENDERED; 
            uint descriptorIndex;
            InterlockedAdd(globals[GLOBALS_FREED_PARTITION_COUNT], 1, descriptorIndex);
            freedPartitions[descriptorIndex] = uint2(partitionIndex, GPU_INSTANCE_FLAG_MERGED);
        }
#endif
        if ((info.flags & PARTITION_FLAG_INSTANCES_RENDERED) == 0)
        {
            partitionInfos[partitionIndex].flags |= PARTITION_FLAG_INSTANCES_RENDERED;
            uint descriptorIndex;
            InterlockedAdd(globals[GLOBALS_VISIBLE_PARTITION_COUNT], 1, descriptorIndex);
            visiblePartitions[descriptorIndex] = partitionIndex;
        }
    }
}
