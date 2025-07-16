#include "../common.hlsli"
#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<uint64_t> blasAddresses : register(t0);
RWStructuredBuffer<uint> globals : register(u1);
StructuredBuffer<BLASData> blasDatas : register(t2);
StructuredBuffer<GPUInstance> gpuInstances : register(t3);

RWStructuredBuffer<PTLAS_WRITE_INSTANCE_INFO> ptlasInstanceWriteInfos : register(u4);
RWStructuredBuffer<PTLAS_UPDATE_INSTANCE_INFO> ptlasInstanceUpdateInfos : register(u5);
RWStructuredBuffer<PTLAS_INDIRECT_COMMAND> ptlasIndirectCommands : register(u6);
StructuredBuffer<InstanceRef> instanceRefs : register(t7);
RWByteAddressBuffer instanceBitVector : register(u8);

[[vk::push_constant]] PtlasPushConstant pc;

[numthreads(32, 1, 1)] 
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint blasIndex = dtID.x;
    if (blasIndex >= globals[GLOBALS_BLAS_COUNT_INDEX]) return;

    BLASData blasData = blasDatas[blasIndex];
    InstanceRef instanceRef = instanceRefs[blasData.instanceRefIndex];
    GPUInstance instance = gpuInstances[instanceRef.instanceID];

    uint bit = blasData.instanceRefIndex & 31u;
    uint byteAddress = (blasData.instanceRefIndex >> 5u) << 2u;
    uint wasWritten = instanceBitVector.Load(byteAddress & (1u << bit));

    if (wasWritten)
    {
        uint index;
        InterlockedAdd(ptlasIndirectCommands[PTLAS_TYPE_UPDATE_INSTANCE].argCount, 1, index);

        if (index == 0)
        {
            ptlasIndirectCommands[PTLAS_TYPE_WRITE_INSTANCE].opType = PTLAS_TYPE_WRITE_INSTANCE;
            ptlasIndirectCommands[PTLAS_TYPE_WRITE_INSTANCE].startAddress = pc.updateAddress;
            ptlasIndirectCommands[PTLAS_TYPE_WRITE_INSTANCE].strideInBytes = PTLAS_WRITE_INSTANCE_INFO_STRIDE;
            InterlockedAdd(globals[GLOBALS_PTLAS_OP_TYPE_COUNT_INDEX], 1);
        }

        PTLAS_UPDATE_INSTANCE_INFO instanceInfo;
        instanceInfo.instanceIndex = blasData.instanceRefIndex;
        instanceInfo.instanceContributionToHitGroupIndex = 0;
        instanceInfo.accelerationStructure = blasData.clusterCount == 0 ? 0 : blasAddresses[blasData.addressIndex]; 

        ptlasInstanceUpdateInfos[index] = instanceInfo;
    }
    else if (blasData.clusterCount != 0)
    {
        uint index;
        InterlockedAdd(ptlasIndirectCommands[PTLAS_TYPE_WRITE_INSTANCE].argCount, 1, index);

        if (index == 0)
        {
            ptlasIndirectCommands[PTLAS_TYPE_WRITE_INSTANCE].opType = PTLAS_TYPE_UPDATE_INSTANCE;
            ptlasIndirectCommands[PTLAS_TYPE_WRITE_INSTANCE].startAddress = pc.updateAddress;
            ptlasIndirectCommands[PTLAS_TYPE_WRITE_INSTANCE].strideInBytes = PTLAS_UPDATE_INSTANCE_INFO_STRIDE;
            InterlockedAdd(globals[GLOBALS_PTLAS_OP_TYPE_COUNT_INDEX], 1);
        }

        PTLAS_WRITE_INSTANCE_INFO instanceInfo;
        instanceInfo.transform = instance.renderFromObject;

        float3 minP = float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 maxP = float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (int z = 2; z <= 5; z += 3)
        {
            for (int y = 1; y <= 4; y += 3)
            {
                for (int x = 0; x <= 3; x += 3)
                {
                    float3 pos = float3(instanceRef.bounds[x], instanceRef.bounds[y], instanceRef.bounds[z]);
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

        instanceInfo.instanceID = instanceRef.instanceID;
        instanceInfo.instanceMask = 0xff;
        instanceInfo.instanceContributionToHitGroupIndex = 0;
        instanceInfo.instanceFlags = (1u << 0u) | (1u << 4u);
        instanceInfo.instanceIndex = blasData.instanceRefIndex;
        instanceInfo.partitionIndex = 0;
        instanceInfo.accelerationStructure = blasAddresses[blasData.addressIndex]; 

        ptlasInstanceWriteInfos[index] = instanceInfo;

        instanceBitVector.InterlockedOr(byteAddress, 1u << bit);
    }
}
