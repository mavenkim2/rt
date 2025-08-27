#include "../common.hlsli"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"

StructuredBuffer<uint64_t> blasAddresses : register(t0);
RWStructuredBuffer<uint> globals : register(u1);
StructuredBuffer<BLASData> blasDatas : register(t2);
StructuredBuffer<GPUInstance> gpuInstances : register(t3);
RWStructuredBuffer<AccelerationStructureInstance> instanceDescriptors : register(u4);

StructuredBuffer<BLASVoxelInfo> blasVoxelInfos : register(t5);
ConstantBuffer<GPUScene> gpuScene : register(b6);

[numthreads(32, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
#if 0
    uint blasIndex = DTid.x;
    if (blasIndex >= globals[GLOBALS_BLAS_COUNT_INDEX]) return;

    BLASData blasData = blasDatas[blasIndex];
    if (blasData.clusterCount == 0 && blasData.voxelClusterCount == 0) return;

    GPUInstance instance = gpuInstances[blasData.instanceID];

    float3x4 renderFromObject = instance.worldFromObject;

    AccelerationStructureInstance instanceDescriptor;
    instanceDescriptor.transform = renderFromObject;
    instanceDescriptor.instanceID = blasData.instanceID;
    instanceDescriptor.instanceMask = 0xff;
    instanceDescriptor.instanceContributionToHitGroupIndex = 0;
    instanceDescriptor.flags = 0;
    instanceDescriptor.blasDeviceAddress = blasAddresses[blasData.addressIndex];

    instanceDescriptors[blasData.addressIndex] = instanceDescriptor;

    for (int i = 0; i < blasData.voxelClusterCount; i++)
    {
        uint descriptorIndex;
        InterlockedAdd(globals[GLOBALS_BLAS_FINAL_COUNT_INDEX], 1, descriptorIndex);

        BLASVoxelInfo info = blasVoxelInfos[blasData.voxelClusterStartIndex + i];

        instanceDescriptor.transform = renderFromObject;
        instanceDescriptor.instanceID = info.clusterID;
        instanceDescriptor.instanceMask = 0xff;
        instanceDescriptor.instanceContributionToHitGroupIndex = 0;
        instanceDescriptor.flags = 0;
        instanceDescriptor.blasDeviceAddress = info.address;

        instanceDescriptors[descriptorIndex] = instanceDescriptor;
    }
#endif
}
