#ifndef PTLAS_WRITE_INSTANCES_HLSLI_
#define PTLAS_WRITE_INSTANCES_HLSLI_

#include "../../rt/shader_interop/as_shaderinterop.h"

RWStructuredBuffer<uint> renderedBitVector : register(u0);
RWStructuredBuffer<uint> thisFrameBitVector : register(u1);
RWStructuredBuffer<uint> globals : register(u2);
RWStructuredBuffer<PTLAS_WRITE_INSTANCE_INFO> ptlasInstanceWriteInfos : register(u3);
RWStructuredBuffer<PTLAS_UPDATE_INSTANCE_INFO> ptlasInstanceUpdateInfos : register(u4);

void WritePTLASDescriptors(GPUInstance instance, uint64_t address, uint instanceIndex, uint instanceID, AABB aabb, bool update)
{
    uint partition = instance.partitionIndex;
    uint2 offsets = uint2(instanceIndex >> 5u, instanceIndex & 31u);
    bool wasRendered = renderedBitVector[offsets.x] & (1u << offsets.y);

    if (!wasRendered)
    {
        uint descriptorIndex;
        InterlockedAdd(globals[GLOBALS_PTLAS_WRITE_COUNT_INDEX], 1, descriptorIndex);

        PTLAS_WRITE_INSTANCE_INFO instanceInfo = (PTLAS_WRITE_INSTANCE_INFO)0;
        if (update)
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
            instanceInfo.instanceFlags |= 0x10u;
        }

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

#endif
