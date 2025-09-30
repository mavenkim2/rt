#include "../common.hlsli"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../bit_twiddling.hlsli"

RWStructuredBuffer<uint> globals : register(u0);
RWStructuredBuffer<PTLAS_WRITE_INSTANCE_INFO> ptlasInstanceWriteInfos : register(u1);
RWStructuredBuffer<PTLAS_UPDATE_INSTANCE_INFO> ptlasInstanceUpdateInfos : register(u2);
StructuredBuffer<uint64_t> blasAddresses : register(t3);
StructuredBuffer<uint> allocatedInstances : register(t4);
RWStructuredBuffer<GPUInstance> gpuInstances : register(u5);
StructuredBuffer<AABB> aabbs : register(t6);
StructuredBuffer<GPUTransform> instanceTransforms : register(t7);
StructuredBuffer<PartitionInfo> partitionInfos : register(t8);

#include "ptlas_write_instances.hlsli"

[numthreads(32, 1, 1)] 
void main(uint3 dtID : SV_DispatchThreadID)
{
    if (dtID.x >= globals[GLOBALS_ALLOCATED_INSTANCE_COUNT_INDEX]) return;

    uint instanceIndex = allocatedInstances[dtID.x];
    GPUInstance instance = gpuInstances[instanceIndex];

    uint64_t address = 0;
    float3x4 worldFromObject;

    PartitionInfo info = partitionInfos[instance.partitionIndex];
    bool update = true;
    uint flags = 0x10u;
    AABB aabb;

    if (0)//instance.flags & GPU_INSTANCE_FLAG_MERGED)
    {
        address = info.mergedProxyDeviceAddress;
        worldFromObject = float3x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
        update = false;
        flags = 0u;
    }
    else 
    {
        //address = resources[instance.resourceID].blasDeviceAddress;//blasAddresses[blasData.addressIndex];
        address = blasAddresses[instance.resourceID];
        worldFromObject = ConvertGPUMatrix(instanceTransforms[instance.transformIndex], info.base, info.scale);
        aabb = aabbs[instance.resourceID];
    }
    WritePTLASDescriptors(worldFromObject, address, instanceIndex, instance.resourceID, aabb, update, flags);
}
