#include "../common.hlsli"
#include "cull.hlsli"
#include "../bit_twiddling.hlsli"
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
RWStructuredBuffer<PTLAS_INDIRECT_COMMAND> ptlasIndirectCommands : register(u7);
RWStructuredBuffer<PTLAS_UPDATE_INSTANCE_INFO> ptlasInstanceUpdateInfos : register(u8);
ByteAddressBuffer instanceBitVector : register(t9);

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
        float3(ref.bounds[3], ref.bounds[4], ref.bounds[5]), gpuScene.p22, gpuScene.p23);

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
    candidateNode.instanceID = ref.instanceID;
    candidateNode.nodeOffset = ref.nodeOffset;
    candidateNode.blasIndex = blasIndex;
    candidateNode.pad = 0;

    nodeQueue[blasIndex] = candidateNode;

    blasDatas[blasIndex].instanceRefIndex = instanceRefIndex;

    InterlockedAdd(queue[0].nodeWriteOffset, 1);
    InterlockedAdd(queue[0].numNodes, 1);
}
