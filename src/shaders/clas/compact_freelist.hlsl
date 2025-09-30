#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<uint> lastFreeList : register(t0);
RWStructuredBuffer<uint> newFreeList : register(u1);
StructuredBuffer<GPUInstance> instances : register(t2);

[numthreads(128, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    if (dtID.x >= lastFreeList[0]) return;

    uint instanceIndex = lastFreeList[dtID.x + 1];
    if (instances[instanceIndex].flags & GPU_INSTANCE_FLAG_FREED)
    {
        uint freeListIndex;
        InterlockedAdd(newFreeList[0], 1, freeListIndex);
        newFreeList[freeListIndex + 1] = instanceIndex;
    }
}
