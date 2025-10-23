#define RayTracingClusterAccelerationStructureNV 5437

[[vk::ext_capability(RayTracingClusterAccelerationStructureNV)]]
[[vk::ext_extension("SPV_NV_cluster_acceleration_structure")]]

#define OpRayQueryGetIntersectionClusterIdNV 5345
#define OpHitObjectGetClusterIdNV 5346

#define RayQueryCandidateIntersectionKHR 0
#define RayQueryCommittedIntersectionKHR 1

[[vk::ext_instruction(OpRayQueryGetIntersectionClusterIdNV)]]
uint GetClusterIDNV([[vk::ext_reference]] RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> query, uint intersection);

[[vk::ext_instruction(OpRayQueryGetIntersectionClusterIdNV)]]
uint GetClusterIDNV([[vk::ext_reference]] RayQuery<RAY_FLAG_NONE | RAY_FLAG_FORCE_OPAQUE> query, uint intersection);

#include "wavefront_helper.hlsli"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"
#include "../../rt/shader_interop/wavefront_shaderinterop.h"

RaytracingAccelerationStructure accel : register(t0);
RWStructuredBuffer<WavefrontQueue> queues : register(u1);
ConstantBuffer<WavefrontDescriptors> descriptors : register(b2);

[shader("raygeneration")]
void main()
{
    uint3 dtID = DispatchRaysIndex();
    uint3 rayDims = DispatchRaysDimensions();
    uint threadIndex = dtID.x;

    uint start = queues[WAVEFRONT_RAY_QUEUE_INDEX].readOffset;
    uint end = queues[WAVEFRONT_RAY_QUEUE_INDEX].writeOffset;
    uint queueIndex = start + threadIndex;

    if (queueIndex >= end) return;
    queueIndex %= WAVEFRONT_QUEUE_SIZE;

    float3 pos = GetFloat3(descriptors.rayQueuePosIndex, queueIndex);
    float3 dir = GetFloat3(descriptors.rayQueueDirIndex, queueIndex);

    RayDesc desc;
    desc.Origin = pos;
    desc.Direction = dir;
    desc.TMin = 0;
    desc.TMax = FLT_MAX;

    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(accel, RAY_FLAG_NONE, 0xff, desc);

    query.Proceed();

    if (query.CommittedStatus() == COMMITTED_NOTHING)
    {
        uint outIndex;
        InterlockedAdd(queues[WAVEFRONT_MISS_QUEUE_INDEX].writeOffset, 1, outIndex);

        outIndex %= WAVEFRONT_QUEUE_SIZE;

        uint pixelIndex = GetUint(descriptors.rayQueuePixelIndex, queueIndex);
        StoreUint(pixelIndex, descriptors.missQueuePixelIndex, outIndex);
        StoreFloat3(dir, descriptors.missQueueDirIndex, outIndex);
    }
    else 
    {
        uint outIndex;
        InterlockedAdd(queues[WAVEFRONT_SHADE_QUEUE_INDEX].writeOffset, 1, outIndex);

        outIndex %= WAVEFRONT_QUEUE_SIZE;

        uint clusterID = GetClusterIDNV(query, RayQueryCommittedIntersectionKHR);
        uint triangleIndex = query.CommittedPrimitiveIndex();
        uint instanceID = query.CommittedInstanceID();
        float2 bary = query.CommittedTriangleBarycentrics();

        uint pixelIndex = GetUint(descriptors.rayQueuePixelIndex, queueIndex);
        uint clusterID_triangleIndex = (clusterID << MAX_CLUSTER_TRIANGLES_BIT) | triangleIndex;
        StoreUint(clusterID_triangleIndex, descriptors.hitShadingQueueClusterIDIndex, outIndex);
        StoreUint(instanceID, descriptors.hitShadingQueueInstanceIDIndex, outIndex);
        StoreFloat2(bary, descriptors.hitShadingQueueBaryIndex, outIndex);
        StoreUint(pixelIndex, descriptors.hitShadingQueuePixelIndex, outIndex);
        StoreFloat(query.CommittedRayT(), descriptors.hitShadingQueueRayTIndex, outIndex);
        StoreFloat3(dir, descriptors.hitShadingQueueDirIndex, outIndex);
    }
}
