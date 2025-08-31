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

void WritePTLASDescriptors(GPUInstance instance, uint64_t address, uint instanceIndex, uint instanceID, AABB aabb, bool update, uint flags)
{
    uint partition = instance.partitionIndex;

    uint2 offsets = uint2(instanceIndex >> 5u, instanceIndex & 31u);

    bool wasRendered = renderedBitVector[offsets.x] & (1u << offsets.y);

    const int maxInstances = 1u << 21u;
    const int numPartitions = 16;
    const int numInstancesPerPartition = maxInstances / numPartitions;

    if (!wasRendered)
    {
        uint descriptorIndex;
        InterlockedAdd(globals[GLOBALS_PTLAS_WRITE_COUNT_INDEX], 1, descriptorIndex);

        PTLAS_WRITE_INSTANCE_INFO instanceInfo = (PTLAS_WRITE_INSTANCE_INFO)0;
        if (flags & 0x10u)
        {
            float3 minP = float3(FLT_MAX, FLT_MAX, FLT_MAX);
            float3 maxP = float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
            for (int z = 0; z < 2; z++)
            {
                for (int y = 0; y < 2; y++)
                {
                    for (int x = 0; x < 2; x++)
                    {
                        float3 p = float3(x ? aabb.maxX : aabb.minX, y ? aabb.maxY : aabb.minY, z ? aabb.maxZ : aabb.minZ);
                        float3 pos = mul(instance.worldFromObject, float4(p, 1.f));
                        minP = min(minP, pos);
                        maxP = max(maxP, pos);
                    }
                }
            }
            for (int i = 0; i < 3; i++)
            {
                instanceInfo.explicitAABB[i] = minP[i];
                instanceInfo.explicitAABB[3 + i] = maxP[i];
            }
        }
        instanceInfo.instanceFlags |= flags;

        instanceInfo.transform = instance.worldFromObject;
        instanceInfo.instanceID = instanceID;
        instanceInfo.instanceMask = 0xff;
        instanceInfo.instanceContributionToHitGroupIndex = 0;
        instanceInfo.instanceIndex = instanceIndex;
        instanceInfo.partitionIndex = partition;
        instanceInfo.accelerationStructure = address;

        ptlasInstanceWriteInfos[descriptorIndex] = instanceInfo;
        InterlockedOr(renderedBitVector[offsets.x], (1u << offsets.y));
    }
    else if (update)
    {
        uint descriptorIndex;
        InterlockedAdd(globals[GLOBALS_PTLAS_UPDATE_COUNT_INDEX], 1, descriptorIndex);

        PTLAS_UPDATE_INSTANCE_INFO instanceInfo;
        instanceInfo.instanceIndex = instanceIndex;
        instanceInfo.instanceContributionToHitGroupIndex = 0;
        instanceInfo.accelerationStructure = address;

        ptlasInstanceUpdateInfos[descriptorIndex] = instanceInfo;
    }

    InterlockedOr(thisFrameBitVector[offsets.x], 1u << offsets.y);
}

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
    WritePTLASDescriptors(instance, address, blasData.instanceID, tableOffset, aabb, true, 0x10u);
}
