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
    if (blasData.clusterCount == 0 && bitMask == 0)
    {
        return;
    }


    uint64_t address = 0;
    uint tableOffset = 0;
    // TODO: this needs to remove the bottom n bits, where n is the number of voxel-free levels
    if (bitMask & ~3)
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
