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
StructuredBuffer<AABB> aabbs : register(t5);
ConstantBuffer<GPUScene> gpuScene : register(b6);

RWStructuredBuffer<uint> renderedBitVector : register(u7);
RWStructuredBuffer<uint> thisFrameBitVector : register(u8);
RWStructuredBuffer<PTLAS_WRITE_INSTANCE_INFO> ptlasInstanceWriteInfos : register(u9);
RWStructuredBuffer<PTLAS_UPDATE_INSTANCE_INFO> ptlasInstanceUpdateInfos : register(u10);
StructuredBuffer<uint64_t> mergedPartitionDeviceAddresses : register(t11);

#include "ptlas_write_instances.hlsli"

[[vk::push_constant]] InstanceCullingPushConstant pc;

[numthreads(64, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint instanceIndex = dtID.x;
    if (instanceIndex >= pc.num) return;

    GPUInstance instance = gpuInstances[instanceIndex];

    if (instance.partitionIndex == ~0u) return;

    // Proxy
    //if (instance.resourceID == ~0u)
    if (instance.flags & GPU_INSTANCE_FLAG_MERGED)
    {
#if 0
        if (pc.oneBlasAddress != 0)
        {
            uint proxyIndex = instance.partitionIndex;
            AABB aabb;
            aabb.minX = -1;
            aabb.minY = -1;
            aabb.minZ = -1;
            aabb.maxX = 1;
            aabb.maxY = 1;
            aabb.maxZ = 1;
            WritePTLASDescriptors(instance, pc.oneBlasAddress, instanceIndex, instanceIndex,
                                  //aabb, true, 0x10u);
                                  aabb, false, 0x0u, 0);
        }
#endif
        AABB aabb;
        uint64_t address = mergedPartitionDeviceAddresses[instance.groupIndex];
        instance.worldFromObject = float3x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
        WritePTLASDescriptors(instance, address, instanceIndex, instanceIndex,
                              aabb, false, 0x0u);
        return;
    }

    AABB aabb = aabbs[instance.resourceID];
    bool cull = FrustumCull(gpuScene.clipFromRender, instance.worldFromObject, 
            float3(aabb.minX, aabb.minY, aabb.minZ),
            float3(aabb.maxX, aabb.maxY, aabb.maxZ), gpuScene.p22, gpuScene.p23);

    instance.flags |= (cull << 0u);

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
