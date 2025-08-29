#include "../common.hlsli"
#include "cull.hlsli"
#include "../bit_twiddling.hlsli"
#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"

// TODO: hierarchical instance culling
RWStructuredBuffer<GPUInstance> gpuInstances : register(u0);
RWStructuredBuffer<uint> globals : register(u1);
RWStructuredBuffer<CandidateNode> nodeQueue : register(u2);
RWStructuredBuffer<Queue> queue : register(u3);
RWStructuredBuffer<BLASData> blasDatas : register(u4);
ConstantBuffer<GPUScene> gpuScene : register(b5);
StructuredBuffer<AABB> aabbs : register(t6);

[[vk::push_constant]] NumPushConstant pc;

[numthreads(64, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint instanceIndex = dtID.x;
    if (instanceIndex >= pc.num) return;

    GPUInstance instance = gpuInstances[instanceIndex];

    AABB aabb = aabbs[instance.resourceID];
    bool cull = FrustumCull(gpuScene.clipFromRender, instance.worldFromObject, 
            float3(aabb.minX, aabb.minY, aabb.minZ),
            float3(aabb.maxX, aabb.maxY, aabb.maxZ), gpuScene.p22, gpuScene.p23);

    instance.cull = cull;

#if 0

    if (cull) 
    {
        //globals[GLOBALS_DEBUG] += 100000;
        uint2 offsets = GetAlignedAddressAndBitOffset(0, instanceRefIndex);
        uint wasWritten = instanceBitVector.Load(offsets[0]) & (1u << offsets.y);
        InterlockedAdd(globals[GLOBALS_DEBUG], 1000 + instanceRefIndex);

        if (wasWritten)
        {

        uint flags;
        uint flag = 1u << PTLAS_TYPE_UPDATE_INSTANCE;
        InterlockedOr(globals[GLOBALS_PTLAS_OP_TYPE_FLAGS], flag, flags);

        if ((flags & flag) == 0)
        {
            uint opIndex;
            InterlockedAdd(globals[GLOBALS_PTLAS_OP_TYPE_COUNT_INDEX], 1, opIndex);
            globals[GLOBALS_PTLAS_UPDATE_INSTANCE_INDEX] = (1u << 31u) | opIndex;
            DeviceMemoryBarrier();
        }

        uint opIndex = globals[GLOBALS_PTLAS_UPDATE_INSTANCE_INDEX];
        while (opIndex == 0)
        {
            opIndex = globals[GLOBALS_PTLAS_UPDATE_INSTANCE_INDEX];
        }

        opIndex &= 0x7fffffff;
        uint index;
        InterlockedAdd(ptlasIndirectCommands[opIndex].argCount, 1, index);

        PTLAS_UPDATE_INSTANCE_INFO instanceInfo;
        instanceInfo.instanceIndex = instanceRefIndex;
        instanceInfo.instanceContributionToHitGroupIndex = 0;
        instanceInfo.accelerationStructure = 0;
        ptlasInstanceUpdateInfos[index] = instanceInfo;

        }

        return;
    }
#endif

    uint blasIndex;
    InterlockedAdd(globals[GLOBALS_BLAS_COUNT_INDEX], 1, blasIndex);
    
    CandidateNode candidateNode;
    candidateNode.instanceID = instanceIndex;
    candidateNode.nodeOffset = 0;
    candidateNode.blasIndex = blasIndex;
    candidateNode.pad = 0;

    nodeQueue[blasIndex] = candidateNode;

    blasDatas[blasIndex].instanceID = instanceIndex;

    InterlockedAdd(queue[0].nodeWriteOffset, 1);
    InterlockedAdd(queue[0].numNodes, 1);
}
