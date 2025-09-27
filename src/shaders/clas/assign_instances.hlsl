#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"

RWStructuredBuffer<uint> globals : register(u0);
RWStructuredBuffer<BLASData> blasDatas : register(u1);
StructuredBuffer<GPUInstance> instances : register(t2);
StructuredBuffer<uint2> maxMinLodLevel : register(t3);
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

    //if (instance.resourceID > 8 || instance.resourceID == 2) return;
    if (instance.resourceID != 0) return;// && instance.resourceID != 18 && instance.resourceID != 19) return;

    // Share
    BLASData blasData = (BLASData)0;
    blasData.instanceID = instanceIndex;
    uint blasIndex;
    InterlockedAdd(globals[GLOBALS_BLAS_COUNT_INDEX], 1, blasIndex);
    uint sharedInstance = maxMinLodLevel[instance.resourceID].x & ((1u << 27u) - 1u);
    uint minMaxLodLevel = maxMinLodLevel[instance.resourceID].y;

    if (0)//instanceIndex == sharedInstance)
    {
        blasData.addressIndex = GPU_INSTANCE_FLAG_SHARED_INSTANCE;

        CandidateNode candidateNode;
        candidateNode.nodeOffset = 0u;
        candidateNode.blasIndex = blasIndex;

        uint nodeIndex;
        InterlockedAdd(queue[0].nodeWriteOffset, 1, nodeIndex);
        InterlockedAdd(queue[0].numNodes, 1);

        nodeQueue[nodeIndex] = candidateNode;
    }
    // TODO: don't hardcode
    else if (0)//instance.minLodLevel >= 3)//max(3, minMaxLodLevel))
    {
        blasData.addressIndex = GPU_INSTANCE_FLAG_SHARED_INSTANCE;
    }
    // Merge
    else
    {
        blasData.addressIndex = GPU_INSTANCE_FLAG_MERGED_INSTANCE;

        //if (instance.minLodLevel < minMaxLodLevel)
        {
            uint wasSet;
            uint bit = 1u << (instance.resourceID & 31u);
            InterlockedOr(resourceBitVector[instance.resourceID >> 5u], bit, wasSet);

            if ((wasSet & bit) == 0)
            {
                // TODO: for the selected merged instance, probably have to do hierarchy traversal twice
                uint flags = (wasSet & bit) ? CANDIDATE_NODE_FLAG_STREAMING_ONLY : CANDIDATE_NODE_FLAG_HIGHEST_DETAIL;

                CandidateNode candidateNode;
                candidateNode.nodeOffset = 0u;//flags << 16u;
                candidateNode.blasIndex = blasIndex;

                uint nodeIndex;
                InterlockedAdd(queue[0].nodeWriteOffset, 1, nodeIndex);
                InterlockedAdd(queue[0].numNodes, 1);

                nodeQueue[nodeIndex] = candidateNode;
            }
        }
    }
    blasDatas[blasIndex] = blasData;
}
