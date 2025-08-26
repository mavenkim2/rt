#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"

StructuredBuffer<BLASVoxelInfo> blasVoxelInfos : register(t0);
StructuredBuffer<BLASData> blasDatas : register(t1);
StructuredBuffer<uint64_t> blasAddresses : register(t2);
RWStructuredBuffer<AccelerationStructureInstance> instanceDescriptors : register(u3);
RWStructuredBuffer<uint> globals : register(u4);
RWStructuredBuffer<uint2> offsetAndCount : register(u5);

[numthreads(32, 1, 1)]
void main(uint dtID : SV_DispatchThreadID) 
{
    uint blasIndex = dtID.x;
    if (blasIndex >= globals[GLOBALS_BLAS_COUNT_INDEX]) return;
    if (blasDatas[blasIndex].clusterCount == 0 && blasDatas[blasIndex].voxelClusterCount == 0) return;

    BLASData blasData = blasDatas[blasIndex];

    uint numInstances = blasData.voxelClusterCount + (blasData.clusterCount ? 1 : 0);
    uint instanceOffset;
    InterlockedAdd(globals[GLOBALS_VISIBLE_INSTANCE_COUNT], numInstances, instanceOffset);

    float3x4 transform = float3x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
    if (blasData.clusterCount)
    {
        AccelerationStructureInstance instanceDescriptor;
        instanceDescriptor.transform = transform;
        instanceDescriptor.instanceID = 0;
        instanceDescriptor.instanceMask = 0xff;
        instanceDescriptor.instanceContributionToHitGroupIndex = 0;
        instanceDescriptor.flags = 0;
        instanceDescriptor.blasDeviceAddress = blasAddresses[blasData.addressIndex];

        instanceDescriptors[instanceOffset++] = instanceDescriptor;
    }
    for (uint i = 0; i < blasData.voxelClusterCount; i++)
    {
        BLASVoxelInfo info = blasVoxelInfos[blasData.voxelClusterStartIndex + i];

        AccelerationStructureInstance instanceDescriptor;
        instanceDescriptor.transform = transform;
        instanceDescriptor.instanceID = info.clusterID;
        instanceDescriptor.instanceMask = 0xff;
        instanceDescriptor.instanceContributionToHitGroupIndex = 0;
        instanceDescriptor.flags = 0;
        instanceDescriptor.blasDeviceAddress = info.address;

        instanceDescriptors[instanceOffset++] = instanceDescriptor;
    }
    uint instanceIndex;
    InterlockedAdd(offsetAndCount[0].x, 1, instanceIndex);
    offsetAndCount[instanceIndex + 1] = uint2(instanceOffset, numInstances);
}
