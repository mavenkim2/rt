#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"

RWStructuredBuffer<uint> globals : register(u0);
RWStructuredBuffer<BLASData> blasDatas : register(u1);
StructuredBuffer<GPUInstance> instances : register(t2);
StructuredBuffer<uint> maxMinLodLevel : register(t3);
RWStructuredBuffer<CandidateNode> nodeQueue : register(u4);
RWStructuredBuffer<Queue> queue : register(u5);
RWStructuredBuffer<uint> resourceBitVector : register(u6);

[numthreads(64, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID) 
{
    uint instanceIndex = dtID.x;
    if (instanceIndex >= 1u << 21u) return;

    GPUInstance instance = instances[instanceIndex];
    if (instance.flags & (GPU_INSTANCE_FLAG_FREED | GPU_INSTANCE_FLAG_MERGED)) return;

    // Share
    BLASData blasData = (BLASData)0;
    blasData.instanceID = instanceIndex;
    uint blasIndex;
    InterlockedAdd(globals[GLOBALS_BLAS_COUNT_INDEX], 1, blasIndex);
    uint sharedInstance = maxMinLodLevel[instance.resourceID] & ((1u << 27u) - 1u);

    if (instanceIndex == sharedInstance)
    {
        blasData.addressIndex = GPU_INSTANCE_FLAG_SHARED_INSTANCE;

        CandidateNode candidateNode;
        candidateNode.instanceID = instanceIndex;
        candidateNode.nodeOffset = 0;
        candidateNode.blasIndex = blasIndex;
        candidateNode.flags = 0;

        uint nodeIndex;
        InterlockedAdd(queue[0].nodeWriteOffset, 1, nodeIndex);
        InterlockedAdd(queue[0].numNodes, 1);

        nodeQueue[nodeIndex] = candidateNode;
    }
    // TODO: don't hardcode this
    else if (instance.minLodLevel >= 3)
    {
        blasData.addressIndex = GPU_INSTANCE_FLAG_SHARED_INSTANCE;
    }
    // Merge
    else 
    {
        uint wasSet;
        uint bit = 1u << (instance.resourceID & 31u);
        InterlockedOr(resourceBitVector[instance.resourceID >> 5u], bit, wasSet);

        blasData.addressIndex = GPU_INSTANCE_FLAG_MERGED_INSTANCE;
        // TODO: for the selected merged instance, probably have to do hierarchy traversal twice
        uint flags = (wasSet & bit) ? CANDIDATE_NODE_FLAG_STREAMING_ONLY : CANDIDATE_NODE_FLAG_HIGHEST_DETAIL;

        CandidateNode candidateNode;
        candidateNode.instanceID = instanceIndex;
        candidateNode.nodeOffset = 0;
        candidateNode.blasIndex = blasIndex;
        candidateNode.flags = flags;

        uint nodeIndex;
        InterlockedAdd(queue[0].nodeWriteOffset, 1, nodeIndex);
        InterlockedAdd(queue[0].numNodes, 1);

        nodeQueue[nodeIndex] = candidateNode;
    }
    blasDatas[blasIndex] = blasData;
}
