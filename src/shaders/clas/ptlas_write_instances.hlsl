#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<uint64_t> blasAddresses : register(t0);
StructuredBuffer<uint> globals : register(t1);
StructuredBuffer<BLASData> blasDatas : register(t2);
StructuredBuffer<GPUInstance> gpuInstances : register(t3);
RWStructuredBuffer<PTLAS_WRITE_INSTANCE_INFO> ptlasInstanceInfos : register(u4);
RWStructuredBuffer<PTLAS_INDIRECT_COMMAND> ptlasIndirectCommands : register(u5);
StructuredBuffer<AABB> aabbs : register(t6);
RWByteAddressBuffer instanceBitVector : register(t7);

//RWStructuredBuffer<AccelerationStructureInstance> instanceDescriptors : register(u4);
//RWStructuredBuffer<BUILD_RANGE_INFO> buildRangeInfos : register(u5);

[[vk::push_constant]] AddressPushConstant pc;

[numthreads(32, 1, 1)] 
void main(uint3 dtID : SV_DispatchThreadID)
{
    if (dtID.x == 0)
    {
        ptlasIndirectCommands[0].opType = PTLAS_TYPE_WRITE_INSTANCE;
        ptlasIndirectCommands[0].startAddress = 
        ptlasIndirectCommands[0].strideInBytes = PTLAS_WRITE_INSTANCE_INFO_STRIDE;
    }

    uint blasIndex = DTid.x;
    if (blasIndex >= globals[GLOBALS_BLAS_COUNT_INDEX]) return;

    BLASData blasData = blasDatas[blasIndex];
    GPUInstance instance = gpuInstances[blasData.instanceIndex];

    uint bit = blasData.instanceIndex & 31u;
    uint byteAddress = (blasData.instanceIndex >> 5u) << 2u;
    uint wasWritten = instanceBitVector.Load(byteAddress & (1u << bit));

    if (wasWritten)
    {
        uint index;
        InterlockedAdd(ptlasIndirectCommands[PTLAS_TYPE_UPDATE_INSTANCE].argCount, 1, index);

        PTLAS_UPDATE_INSTANCE_INFO instanceInfo;
        instanceInfo.instanceIndex = blasData.instanceIndex;
        instanceInfo.instanceContributionToHitGroupIndex = 0;
        instanceInfo.accelerationStructure = blasData.clusterCount == 0 ? 0 : blasAddresses[blasData.addressIndex]; 

        ptlasInstanceInfos[index] = instanceInfo;
    }
    else 
    {
        uint index;
        InterlockedAdd(ptlasIndirectCommands[PTLAS_TYPE_WRITE_INSTANCE].argCount, 1, index);

        // TODO: when instance merging, 
        PTLAS_WRITE_INSTANCE_INFO instanceInfo;
        instanceInfo.transform = instance.renderFromObject;
        instanceInfo.explicitAABB[0] = ?;
        instanceInfo.instanceID = blasData.instanceIndex;
        instanceInfo.instanceMask = 0xff;
        instanceInfo.instanceContributionToHitGroupIndex = 0;
        instanceInfo.instanceFlags = 1;
        instanceInfo.instanceIndex = ?;
        instanceInfo.partitionIndex = 0;//?;
        instanceInfo.accelerationStructure = blasAddresses[blasData.addressIndex]; 

        ptlasInstanceInfos[index] = instanceInfo;

        instanceBitVector.InterlockedOr(byteAddress, 1u << bit);
    }
}
