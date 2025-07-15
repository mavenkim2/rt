#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<uint64_t> blasAddresses : register(t0);
StructuredBuffer<uint> globals : register(t1);
StructuredBuffer<BLASData> blasDatas : register(t2);
StructuredBuffer<GPUInstance> gpuInstances : register(t3);
RWStructuredBuffer<AccelerationStructureInstance> instanceDescriptors : register(u4);
RWStructuredBuffer<BUILD_RANGE_INFO> buildRangeInfos : register(u5);

[numthreads(32, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
#if 0
    uint blasIndex = DTid.x;
    if (blasIndex >= globals[GLOBALS_BLAS_COUNT_INDEX]) return;

    BLASData blasData = blasDatas[blasIndex];
    if (blasData.clusterCount == 0) return;

    GPUInstance instance = gpuInstances[blasData.instanceIndex];

    uint index;
    InterlockedAdd(buildRangeInfos[0].primitiveCount, 1, index);

    AccelerationStructureInstance instanceDescriptor;
    instanceDescriptor.transform = instance.renderFromObject;
    instanceDescriptor.instanceID = blasData.instanceIndex;
    instanceDescriptor.instanceMask = 0xff;
    instanceDescriptor.instanceContributionToHitGroupIndex = 0;
    instanceDescriptor.flags = 0;
    instanceDescriptor.blasDeviceAddress = blasAddresses[blasData.addressIndex];

    instanceDescriptors[index] = instanceDescriptor;
#endif
}
