#include "../../rt/shader_interop/as_shaderinterop.h"

[[vk::push_constant]] NumPushConstant pc;

StructuredBuffer<uint64_t> blasAddresses : register(t0);
RWStructuredBuffer<AccelerationStructureInstance> instances : register(u1);

[numthreads(32, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint index = DTid.x;
    if (index >= pc.num) return;

    AccelerationStructureInstance instance = instances[index];
    instance.blasDeviceAddress = blasAddresses[instance.instanceID];
    instances[index] = instance;
}
