#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"

// TODO: hierarchical instance culling
StructuredBuffer<GPUInstance> gpuInstances : register(t0);
RWStructuredBuffer<uint> globals : register(u1);
RWStructuredBuffer<CandidateNode> nodeQueue : register(u2);

[[vk::push_constant]] NumPushConstant pc;

[numthreads(64, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint instanceIndex = dtID.x;
    if (instanceIndex >= pc.num) return;

    GPUInstance instance = gpuInstances[instanceIndex];

    uint blasIndex;
    InterlockedAdd(globals[GLOBALS_BLAS_COUNT_INDEX], 1, blasIndex);
    
    CandidateNode candidateNode;
    candidateNode.instanceID = instanceIndex;
    candidateNode.nodeOffset = 0;
    candidateNode.blasIndex = blasIndex;
    candidateNode.pad = 0;

    nodeQueue[blasIndex] = candidateNode;
}
