#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<uint64_t> blasAddresses : register(t0);
StructuredBuffer<uint> globals : register(t1);
StructuredBuffer<BLASData> blasDatas : register(t2);
StructuredBuffer<GPUInstance> gpuInstances : register(t3);
RWStructuredBuffer<AccelerationStructureInstance> instanceDescriptors : register(u4);

[numthreads(32, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint index = DTid.x;
    if (index >= globals[GLOBALS_BLAS_COUNT_INDEX]) return;

    BLASData blasData = blasDatas[index];
    GPUInstance instance = gpuInstances[blasData.instanceIndex];

    AccelerationStructureInstance instanceDescriptor;
    instanceDescriptor.transform = instance.renderFromObject;
    instanceDescriptor.instanceID = blasData.instanceIndex; // TODO: do I need this?
    instanceDescriptor.instanceMask = 0xff;
    instanceDescriptor.instanceContributionToHitGroupIndex = 0;
    instanceDescriptor.flags = 0;
    instanceDescriptor.blasDeviceAddress = blasAddresses[index];

    instanceDescriptors[index] = instanceDescriptor;
}
