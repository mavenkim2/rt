#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<uint> lastFrameBitVector : register(t0);
StructuredBuffer<uint> thisFrameBitVector : register(t1);
RWStructuredBuffer<uint> globals : register(u2);
RWStructuredBuffer<PTLAS_UPDATE_INSTANCE_INFO> ptlasInstanceUpdateInfos : register(u3);
RWStructuredBuffer<uint> virtualInstanceTable : register(u4);
RWStructuredBuffer<int> instanceIDFreeList : register(u5);
RWStructuredBuffer<PTLAS_WRITE_INSTANCE_INFO> ptlasInstanceWriteInfos : register(u6);

[numthreads(32, 1, 1)]
void main(uint dtID : SV_DispatchThreadID)
{
    for (;;)
    {
        uint index;
        InterlockedAdd(globals[GLOBALS_UNUSED_CHECK_INDEX], 1, index);

        const int maxInstances = 1u << 21u;
        const int numPartitions = 16;
        const int numInstancesPerPartition = maxInstances / numPartitions;
        const int maxIndex = (1u << 24u) >> 3u;

        if (index >= maxIndex) break;

        uint frameBits = thisFrameBitVector[index];
        uint lastFrameBits = lastFrameBitVector[index];

        uint unusedMask = lastFrameBits & ~frameBits;
        while (unusedMask)
        {
            uint virtualInstanceIndex = 32 * index + firstbitlow(unusedMask);

            uint instanceIndex = virtualInstanceTable[virtualInstanceIndex];
            bool update = instanceIndex >> 31u;
            instanceIndex &= 0x7fffffffu;

            uint partition = instanceIndex / numInstancesPerPartition;
            uint descriptorIndex;

            if (update)
            {
                InterlockedAdd(globals[GLOBALS_PTLAS_UPDATE_COUNT_INDEX], 1, descriptorIndex);

                PTLAS_UPDATE_INSTANCE_INFO instanceInfo;
                instanceInfo.instanceIndex = instanceIndex;
                instanceInfo.instanceContributionToHitGroupIndex = 0;
                instanceInfo.accelerationStructure = 0;
                ptlasInstanceUpdateInfos[descriptorIndex] = instanceInfo;
            }
            else 
            {
                PTLAS_WRITE_INSTANCE_INFO instanceInfo = (PTLAS_WRITE_INSTANCE_INFO)0;
                instanceInfo.instanceMask = 0xff;
                instanceInfo.instanceIndex = instanceIndex;

                InterlockedAdd(globals[GLOBALS_PTLAS_WRITE_COUNT_INDEX], 1, descriptorIndex);
                ptlasInstanceWriteInfos[descriptorIndex] = instanceInfo;
            }

            virtualInstanceTable[virtualInstanceIndex] = ~0u;

            instanceIndex = instanceIndex % numInstancesPerPartition;
            uint freeListCountIndex = partition * (numInstancesPerPartition + 1u);
            int freeListIndex;
            InterlockedAdd(instanceIDFreeList[freeListCountIndex], 1, freeListIndex);
            freeListIndex += freeListCountIndex;
            instanceIDFreeList[freeListIndex + 1] = instanceIndex;

            unusedMask &= unusedMask - 1;
        }
    }
}
