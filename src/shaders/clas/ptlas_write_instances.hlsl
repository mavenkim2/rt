#include "../common.hlsli"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../bit_twiddling.hlslI"

StructuredBuffer<uint64_t> blasAddresses : register(t0);
globallycoherent RWStructuredBuffer<uint> globals : register(u1);
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

    uint2 offsets = GetAlignedAddressAndBitOffset(0, blasData.instanceRefIndex);
    uint wasWritten = instanceBitVector.Load(offsets[0]) & (1u << offsets.y);

    if (wasWritten)
    {
        uint flags;
        uint flag = 1u << PTLAS_TYPE_UPDATE_INSTANCE;
        InterlockedOr(globals[GLOBALS_PTLAS_OP_TYPE_FLAGS], flag, flags);

        if ((flags & flag) == 0)
        {
            uint opIndex;
            InterlockedAdd(globals[GLOBALS_PTLAS_OP_TYPE_COUNT_INDEX], 1, opIndex);
            globals[GLOBALS_PTLAS_UPDATE_INSTANCE_INDEX] = (1u << 31u) | opIndex;
            DeviceMemoryBarrier();
        }

        uint opIndex = globals[GLOBALS_PTLAS_UPDATE_INSTANCE_INDEX];
        while (opIndex == 0)
        {
            opIndex = globals[GLOBALS_PTLAS_UPDATE_INSTANCE_INDEX];
        }

        opIndex &= 0x7fffffff;
        uint index;
        InterlockedAdd(ptlasIndirectCommands[opIndex].argCount, 1, index);

        if (index == 0)
        {
            ptlasIndirectCommands[opIndex].opType = PTLAS_TYPE_UPDATE_INSTANCE;
            ptlasIndirectCommands[opIndex].startAddress = pc.updateAddress;
            ptlasIndirectCommands[opIndex].strideInBytes = PTLAS_UPDATE_INSTANCE_INFO_STRIDE;
        }

        PTLAS_UPDATE_INSTANCE_INFO instanceInfo;
        instanceInfo.instanceIndex = blasData.instanceRefIndex;
        instanceInfo.instanceContributionToHitGroupIndex = 0;
        instanceInfo.accelerationStructure = blasData.clusterCount == 0 ? 0 : blasAddresses[blasData.addressIndex]; 

        ptlasInstanceUpdateInfos[index] = instanceInfo;
    }
    else if (blasData.clusterCount != 0)
    {
        uint flags;
        uint flag = 1u << PTLAS_TYPE_WRITE_INSTANCE;
        InterlockedOr(globals[GLOBALS_PTLAS_OP_TYPE_FLAGS], flag, flags);

        if ((flags & flag) == 0)
        {
            uint opIndex;
            InterlockedAdd(globals[GLOBALS_PTLAS_OP_TYPE_COUNT_INDEX], 1, opIndex);
            globals[GLOBALS_PTLAS_WRITE_INSTANCE_INDEX] = (1u << 31u) | opIndex;
            DeviceMemoryBarrier();
        }

        uint opIndex = globals[GLOBALS_PTLAS_WRITE_INSTANCE_INDEX];
        while (opIndex == 0)
        {
            opIndex = globals[GLOBALS_PTLAS_WRITE_INSTANCE_INDEX];
        }
        opIndex &= 0x7fffffff;

        uint index;
        InterlockedAdd(ptlasIndirectCommands[opIndex].argCount, 1, index);

        if (index == 0)
        {
            ptlasIndirectCommands[opIndex].opType = PTLAS_TYPE_WRITE_INSTANCE;
            ptlasIndirectCommands[opIndex].startAddress = pc.writeAddress;
            ptlasIndirectCommands[opIndex].strideInBytes = PTLAS_WRITE_INSTANCE_INFO_STRIDE;
        }

        PTLAS_WRITE_INSTANCE_INFO instanceInfo = (PTLAS_WRITE_INSTANCE_INFO)0;
        instanceInfo.transform = instance.renderFromObject;

#if 0
        float3 minP = float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 maxP = float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (int z = 2; z <= 5; z += 3)
        {
            for (int y = 1; y <= 4; y += 3)
            {
                for (int x = 0; x <= 3; x += 3)
                {
                    float3 pos = mul(instance.renderFromObject, float4(instanceRef.bounds[x], instanceRef.bounds[y], instanceRef.bounds[z], 1.f));
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
#endif
        for (int i = 0; i < 3; i++)
        {
            instanceInfo.explicitAABB[i] = instanceRef.bounds[i];
            instanceInfo.explicitAABB[3 + i] = instanceRef.bounds[3 + i];
        }

        instanceInfo.instanceID = instanceRef.partitionIndex;//instanceRef.instanceID;
        instanceInfo.instanceMask = 0xff;
        instanceInfo.instanceContributionToHitGroupIndex = 0;
        instanceInfo.instanceFlags = (1u << 0u) | (1u << 4u);
        instanceInfo.instanceIndex = blasData.instanceRefIndex;
        instanceInfo.partitionIndex = instanceRef.partitionIndex;
        instanceInfo.accelerationStructure = blasAddresses[blasData.addressIndex]; 

        ptlasInstanceWriteInfos[index] = instanceInfo;

        instanceBitVector.InterlockedOr(offsets.x, 1u << offsets.y);
    }
    else 
    {
        //InterlockedAdd(globals[GLOBALS_DEBUG], 1);
    }
}
