#include "../common.hlsli"
#include "cull.hlsli"
#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"

// TODO: hierarchical instance culling
StructuredBuffer<GPUInstance> gpuInstances : register(t0);
RWStructuredBuffer<uint> globals : register(u1);
RWStructuredBuffer<CandidateNode> nodeQueue : register(u2);
RWStructuredBuffer<Queue> queue : register(u3);
RWStructuredBuffer<BLASData> blasDatas : register(u4);
StructuredBuffer<InstanceRef> instanceRefs : register(t5);
ConstantBuffer<GPUScene> gpuScene : register(b6);

[[vk::push_constant]] NumPushConstant pc;

[numthreads(64, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint instanceRefIndex = dtID.x;
    if (instanceRefIndex >= pc.num) return;

    InstanceRef ref = instanceRefs[instanceRefIndex];
    GPUInstance instance = gpuInstances[ref.instanceID];

    bool cull = FrustumCull(gpuScene.clipFromRender, instance.renderFromObject, 
        float3(ref.bounds[0], ref.bounds[1], ref.bounds[2]), 
        float3(ref.bounds[3], ref.bounds[5], ref.bounds[5]), gpuScene.p22, gpuScene.p23);
    if (cull) return;

    uint blasIndex;
    InterlockedAdd(globals[GLOBALS_BLAS_COUNT_INDEX], 1, blasIndex);
    
    CandidateNode candidateNode;
    candidateNode.instanceID = ref.instanceID;
    candidateNode.nodeOffset = ref.nodeOffset;
    candidateNode.blasIndex = blasIndex;
    candidateNode.pad = 0;

    nodeQueue[blasIndex] = candidateNode;

    blasDatas[blasIndex].instanceRefIndex = instanceRefIndex;

    InterlockedAdd(queue[0].nodeWriteOffset, 1);
    InterlockedAdd(queue[0].numNodes, 1);
}
