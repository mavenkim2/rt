#include "../common.hlsli"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../bit_twiddling.hlsli"

RWStructuredBuffer<uint> globals : register(u0);
RWStructuredBuffer<PTLAS_WRITE_INSTANCE_INFO> ptlasInstanceWriteInfos : register(u1);
RWStructuredBuffer<PTLAS_UPDATE_INSTANCE_INFO> ptlasInstanceUpdateInfos : register(u2);
StructuredBuffer<uint64_t> blasAddresses : register(t3);
StructuredBuffer<BLASData> blasDatas : register(t4);
RWStructuredBuffer<GPUInstance> gpuInstances : register(u5);
StructuredBuffer<AABB> aabbs : register(t6);
StructuredBuffer<VoxelAddressTableEntry> voxelAddressTable : register(t7);
StructuredBuffer<uint> instanceBitmasks : register(t8);
StructuredBuffer<GPUTransform> instanceTransforms : register(t9);
StructuredBuffer<PartitionInfo> partitionInfos : register(t10);

#include "ptlas_write_instances.hlsli"

[numthreads(32, 1, 1)] 
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint blasIndex = dtID.x;
    if (blasIndex >= globals[GLOBALS_BLAS_COUNT_INDEX]) return;

    BLASData blasData = blasDatas[blasIndex];
    uint instanceIndex = blasData.instanceID;
    uint bitMask = instanceBitmasks[blasData.instanceID];
    GPUInstance instance = gpuInstances[blasData.instanceID];
    if (blasData.clusterCount == 0)
    {
#if 0
    if ((instance.flags & GPU_INSTANCE_FLAG_FREED) && (instance.flags & GPU_INSTANCE_FLAG_WAS_RENDERED))
    {
        if ((instance.flags & GPU_INSTANCE_FLAG_MERGED) == 0)
        {
            uint descriptorIndex;

            InterlockedAdd(globals[GLOBALS_PTLAS_UPDATE_COUNT_INDEX], 1, descriptorIndex);

            gpuInstances[instanceIndex].flags &= ~GPU_INSTANCE_FLAG_WAS_RENDERED;
            PTLAS_UPDATE_INSTANCE_INFO instanceInfo;
            instanceInfo.instanceIndex = instanceIndex;
            instanceInfo.instanceContributionToHitGroupIndex = 0;
            instanceInfo.accelerationStructure = 0;
            ptlasInstanceUpdateInfos[descriptorIndex] = instanceInfo;
        }
        else 
        {
            uint descriptorIndex;

            InterlockedAdd(globals[GLOBALS_PTLAS_WRITE_COUNT_INDEX], 1, descriptorIndex);

            gpuInstances[instanceIndex].flags &= ~GPU_INSTANCE_FLAG_WAS_RENDERED;
            PTLAS_WRITE_INSTANCE_INFO instanceInfo = (PTLAS_WRITE_INSTANCE_INFO)0;
            instanceInfo.instanceIndex = instanceIndex;
            instanceInfo.partitionIndex = instance.partitionIndex;
            ptlasInstanceWriteInfos[descriptorIndex] = instanceInfo;
        }
    }
#endif
        return;
    }


    uint64_t address = 0;
    uint tableOffset = 0;
    // TODO: this needs to remove the bottom n bits, where n is the number of voxel-free levels
    if (bitMask & ~1)
    {
        while (bitMask)
        {
            uint offset = firstbitlow(bitMask);
            VoxelAddressTableEntry entry = voxelAddressTable[instance.voxelAddressOffset + offset];
            address = entry.address;
            tableOffset = entry.tableOffset;
            if (address != 0)
            {
                gpuInstances[blasData.instanceID].clusterLookupTableOffset = tableOffset;
                break;
            }
            bitMask &= bitMask - 1;
        }
    }
    else 
    {
        address = blasAddresses[blasData.addressIndex];
    }

    if (address == 0) return;

    AABB aabb = aabbs[instance.resourceID];

    PartitionInfo info = partitionInfos[instance.partitionIndex];
    float3x4 worldFromObject = ConvertGPUMatrix(instanceTransforms[instance.transformIndex], info.base, info.scale);
    WritePTLASDescriptors(worldFromObject, address, blasData.instanceID, blasData.instanceID, aabb, true, 0x10u);
}
