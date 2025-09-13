#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "lod_error_test.hlsli"

RWStructuredBuffer<PartitionInfo> partitionInfos : register(u0);
RWStructuredBuffer<uint> globals : register(u2);
ConstantBuffer<GPUScene> gpuScene : register(b3);
StructuredBuffer<uint64_t> mergedPartitionDeviceAddresses : register(t4);

RWStructuredBuffer<uint> visiblePartitions : register(u5);
RWStructuredBuffer<uint> freedPartitions : register(u6);
RWStructuredBuffer<uint> instanceFreeList : register(u7);

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint partitionIndex = dtID.x;
    PartitionInfo info = partitionInfos[partitionIndex];

    float error = asfloat(info.lodError);
    float4 lodBounds = info.lodBounds;

    float3x4 worldFromObject = float3x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);

    float test;
    bool culled = false;
    float2 edgeScales = TestNode(worldFromObject, gpuScene.cameraFromRender, lodBounds, 1.f, test, culled);

    if (error * gpuScene.lodScale < edgeScales.x)
    {
        if (info.flags & PARTITION_FLAG_INSTANCES_RENDERED)
        {
            partitionInfos[partitionIndex].flags &= ~PARTITION_FLAG_INSTANCES_RENDERED;
            uint descriptorIndex;
            InterlockedAdd(globals[GLOBALS_FREED_PARTITION_COUNT], 1, descriptorIndex);
            freedPartitions[descriptorIndex] = partitionIndex;
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
            uint proxyInstanceIndex = info.proxyInstanceIndex;
            partitionInfos[partitionIndex].flags &= ~PARTITION_FLAG_PROXY_RENDERED;
            partitionInfos[partitionIndex].proxyInstanceIndex = ~0u;
            uint instanceFreeListIndex;
            InterlockedAdd(instanceFreeList[0], 1, instanceFreeListIndex);
            instanceFreeList[instanceFreeListIndex + 1] = proxyInstanceIndex;
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
