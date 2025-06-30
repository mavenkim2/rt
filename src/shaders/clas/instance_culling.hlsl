#include "../../rt/shader_interop/gpu_shaderinterop.h"
#include "../../rt/shader_interop/as_shaderinterop.h"

// TODO: hierarchical instance culling
StructuredBuffer<GPUInstance> gpuInstances : register(t0);
RWStructuredBuffer<uint> globals : register(u1);

[[vk::push_constant]] NumPushConstants pc;

[numthreads(64, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint instanceIndex = dtID.x;
    if (instanceIndex >= pc.num) return;

    GPUInstance instance = gpuInstances[instanceIndex];

    uint blasIndex;
    InterlockedAdd(globals[GLOBALS_BLAS_COUNT_INDEX], 1, blasIndex);
    
    uint4 node;
    node.x = instanceIndex;
    node.y = ?;
    node.z = blasIndex;
    node.w = 0;
}
