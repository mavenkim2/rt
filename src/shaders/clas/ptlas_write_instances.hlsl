#include "../common.hlsli"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../bit_twiddling.hlslI"

StructuredBuffer<uint64_t> blasAddresses : register(t0);
RWStructuredBuffer<uint> globals : register(u1);
StructuredBuffer<BLASData> blasDatas : register(t2);
RWStructuredBuffer<GPUInstance> gpuInstances : register(u3);

RWStructuredBuffer<PTLAS_WRITE_INSTANCE_INFO> ptlasInstanceWriteInfos : register(u4);
RWStructuredBuffer<PTLAS_UPDATE_INSTANCE_INFO> ptlasInstanceUpdateInfos : register(u5);

StructuredBuffer<BLASVoxelInfo> blasVoxelInfos : register(t6);
StructuredBuffer<CLASPageInfo> clasPageInfos : register(t7);
RWStructuredBuffer<uint> instanceRenderedBitVector : register(u8);
StructuredBuffer<uint> lastFrameBitVector : register(t9);
RWStructuredBuffer<uint> thisFrameBitVector : register(u10);

void WritePTLASDescriptors(GPUInstance instance, uint64_t address, uint instanceID, uint instanceIndex) 
{
    uint2 offsets = uint2(instanceIndex >> 5u, instanceIndex & 31u);
    uint wasRendered = instanceRenderedBitVector[offsets.x] & (1u << offsets.y);
    uint wasRenderedLastFrame = lastFrameBitVector[offsets.x] & (1u << offsets.y);

    if (!wasRendered)
    {
        InterlockedOr(instanceRenderedBitVector[offsets.x], 1u << offsets.y);

        uint descriptorIndex;
        InterlockedAdd(globals[GLOBALS_PTLAS_WRITE_COUNT_INDEX], 1, descriptorIndex);

        PTLAS_WRITE_INSTANCE_INFO instanceInfo = (PTLAS_WRITE_INSTANCE_INFO)0;
        instanceInfo.transform = instance.worldFromObject;
        instanceInfo.instanceID = instanceID;
        instanceInfo.instanceMask = 0xff;
        instanceInfo.instanceContributionToHitGroupIndex = 0;
        instanceInfo.instanceFlags = (1u << 0u); //| (1u << 4u);
        instanceInfo.instanceIndex = instanceIndex;
        instanceInfo.partitionIndex = 0;
        instanceInfo.accelerationStructure = address;

        ptlasInstanceWriteInfos[descriptorIndex] = instanceInfo;
    }
    else if (wasRendered && !wasRenderedLastFrame)
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
    GPUInstance instance = gpuInstances[blasData.instanceID];

    if (blasData.voxelClusterCount) 
    {
        for (int index = 0; index < blasData.voxelClusterCount; index++)
        {
            BLASVoxelInfo info = blasVoxelInfos[blasData.voxelClusterStartIndex + index];
            uint pageIndex = info.clusterID >> MAX_CLUSTERS_PER_PAGE_BITS;
            uint clusterIndex = info.clusterID & (MAX_CLUSTERS_PER_PAGE - 1);

            CLASPageInfo pageInfo = clasPageInfos[pageIndex];
            uint instanceIndex = instance.instanceIDStart + 1 + pageInfo.voxelClusterOffset + clusterIndex;

            WritePTLASDescriptors(instance, info.address, info.clusterID, instanceIndex);
        }
    }

    if (blasData.clusterCount) 
    {
        uint64_t address = blasAddresses[blasData.addressIndex];
        WritePTLASDescriptors(instance, address, instance.instanceIDStart, instance.instanceIDStart);
    }
}
