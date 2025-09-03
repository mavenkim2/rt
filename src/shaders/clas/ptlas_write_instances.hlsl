#include "../common.hlsli"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../bit_twiddling.hlsli"

RWStructuredBuffer<uint> renderedBitVector : register(u0);
RWStructuredBuffer<uint> thisFrameBitVector : register(u1);
RWStructuredBuffer<uint> globals : register(u2);
RWStructuredBuffer<PTLAS_WRITE_INSTANCE_INFO> ptlasInstanceWriteInfos : register(u3);
RWStructuredBuffer<PTLAS_UPDATE_INSTANCE_INFO> ptlasInstanceUpdateInfos : register(u4);
StructuredBuffer<uint64_t> blasAddresses : register(t5);
StructuredBuffer<BLASData> blasDatas : register(t6);
RWStructuredBuffer<GPUInstance> gpuInstances : register(u7);
StructuredBuffer<AABB> aabbs : register(t8);
StructuredBuffer<VoxelAddressTableEntry> voxelAddressTable : register(t9);
StructuredBuffer<uint> instanceBitmasks : register(t10);

#include "ptlas_write_instances.hlsli"

[numthreads(32, 1, 1)] 
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint blasIndex = dtID.x;
    if (blasIndex >= globals[GLOBALS_BLAS_COUNT_INDEX]) return;

    BLASData blasData = blasDatas[blasIndex];
    uint bitMask = instanceBitmasks[blasData.instanceID];
    if (blasData.clusterCount == 0 && bitMask == 0) return;

    GPUInstance instance = gpuInstances[blasData.instanceID];

    uint64_t address = 0;
    uint tableOffset = 0;
    if (bitMask)
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
    WritePTLASDescriptors(instance, address, blasData.instanceID, blasData.instanceID, aabb, true, 0x10u);
}
