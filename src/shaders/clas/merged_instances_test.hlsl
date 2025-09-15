#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "lod_error_test.hlsli"
#include "cull.hlsli"

RWStructuredBuffer<PartitionInfo> partitionInfos : register(u0);
RWStructuredBuffer<uint> globals : register(u1);
StructuredBuffer<uint64_t> mergedPartitionDeviceAddresses : register(t2);

RWStructuredBuffer<uint> visiblePartitions : register(u3);
RWStructuredBuffer<uint2> freedPartitions : register(u4);
RWStructuredBuffer<uint> instanceFreeList : register(u5);
ConstantBuffer<GPUScene> gpuScene : register(b6);

[[vk::push_constant]] NumPushConstant pc;

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

    float error = info.lodError;
    float4 lodBounds = info.lodBounds;

    float3x4 renderFromObject = float3x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    float3 minP = lodBounds.xyz - lodBounds.w;
    float3 maxP = lodBounds.xyz + lodBounds.w;
    bool cull = FrustumCull(gpuScene.clipFromRender, renderFromObject, 
            minP, maxP, gpuScene.p22, gpuScene.p23);

    Translate(renderFromObject, -gpuScene.cameraP);

    float test;
    float2 edgeScales = TestNode(renderFromObject, gpuScene.cameraFromRender, lodBounds, 1.f, test, cull);

    if (cull || error * gpuScene.lodScale < edgeScales.x)
    {
        if (info.flags & PARTITION_FLAG_INSTANCES_RENDERED)
        {
            partitionInfos[partitionIndex].flags &= ~PARTITION_FLAG_INSTANCES_RENDERED;
            uint descriptorIndex;
            InterlockedAdd(globals[GLOBALS_FREED_PARTITION_COUNT], 1, descriptorIndex);
            freedPartitions[descriptorIndex] = uint2(partitionIndex, GPU_INSTANCE_FLAG_INDIV);
        }
        if ((info.flags & PARTITION_FLAG_PROXY_RENDERED) == 0)
        {
            partitionInfos[partitionIndex].flags |= PARTITION_FLAG_PROXY_RENDERED;
            uint descriptorIndex;
            InterlockedAdd(globals[GLOBALS_VISIBLE_PARTITION_COUNT], 1, descriptorIndex);
            visiblePartitions[descriptorIndex] = partitionIndex;
        }
    }
    else 
    {
        if (info.flags & PARTITION_FLAG_PROXY_RENDERED)
        {
            partitionInfos[partitionIndex].flags &= ~PARTITION_FLAG_PROXY_RENDERED; 
            uint descriptorIndex;
            InterlockedAdd(globals[GLOBALS_FREED_PARTITION_COUNT], 1, descriptorIndex);
            freedPartitions[descriptorIndex] = uint2(partitionIndex, GPU_INSTANCE_FLAG_MERGED);
        }
        if ((info.flags & PARTITION_FLAG_INSTANCES_RENDERED) == 0)
        {
            partitionInfos[partitionIndex].flags |= PARTITION_FLAG_INSTANCES_RENDERED;
            uint descriptorIndex;
            InterlockedAdd(globals[GLOBALS_VISIBLE_PARTITION_COUNT], 1, descriptorIndex);
            visiblePartitions[descriptorIndex] = partitionIndex;
        }
    }
}
