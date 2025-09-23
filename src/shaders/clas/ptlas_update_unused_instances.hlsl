#include "../../rt/shader_interop/as_shaderinterop.h"

RWStructuredBuffer<GPUInstance> gpuInstances : register(u0);
RWStructuredBuffer<uint> globals : register(u1);
RWStructuredBuffer<PTLAS_WRITE_INSTANCE_INFO> ptlasWriteInstanceInfos : register(u2);

[numthreads(64, 1, 1)]
void main(uint dtID : SV_DispatchThreadID)
{
    uint instanceIndex = dtID.x;
    if (instanceIndex >= 1u << 21u) return;

    GPUInstance instance = gpuInstances[instanceIndex];
    if ((instance.flags & GPU_INSTANCE_FLAG_FREED) && (instance.flags & GPU_INSTANCE_FLAG_WAS_RENDERED))
    {
        uint descriptorIndex;

        InterlockedAdd(globals[GLOBALS_PTLAS_WRITE_COUNT_INDEX], 1, descriptorIndex);

        gpuInstances[instanceIndex].flags &= ~GPU_INSTANCE_FLAG_WAS_RENDERED;
        PTLAS_WRITE_INSTANCE_INFO instanceInfo = (PTLAS_WRITE_INSTANCE_INFO)0;
        instanceInfo.instanceIndex = instanceIndex;
        instanceInfo.partitionIndex = instance.partitionIndex;
        ptlasWriteInstanceInfos[descriptorIndex] = instanceInfo;
    }
}
