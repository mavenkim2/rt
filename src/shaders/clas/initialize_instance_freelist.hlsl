#include "../../rt/shader_interop/as_shaderinterop.h"
RWStructuredBuffer<int> instanceIDFreeList : register(u0);
RWStructuredBuffer<GPUInstance> gpuInstances : register(u1);

[numthreads(1, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    int maxInstances = 1u << 22u;
    instanceIDFreeList[0] = maxInstances;
    for (int i = 0; i < maxInstances; i++)
    {
        instanceIDFreeList[i + 1] = maxInstances - i - 1;
        GPUInstance instance = (GPUInstance)0;
        instance.flags = GPU_INSTANCE_FLAG_FREED;
        instance.partitionIndex = ~0u;
        gpuInstances[i] = instance;
    }
}
