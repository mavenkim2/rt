#include "wavefront_helper.hlsli"
#include "nvidia/clas.hlsli"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"

RaytracingAccelerationStructure accel : register(t0);
RWStructuredBuffer<WavefrontQueue> queues : register(u1);

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint start = queues[rayKernelQueueIndex].readOffset;
    uint end = queues[rayKernelQueueIndex].writeOffset;
    uint queueIndex = start + dtID.x;
    if (queueIndex >= end) return;
    queueIndex %= WAVEFRONT_QUEUE_SIZE;

    float3 pos = GetFloat3(rayQueuePosIndex, queueIndex);
    float3 dir = GetFloat3(rayQueuePosIndex, queueIndex);

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
    }
    else 
    {
        uint outIndex;
        uint hitQueueIndex;
        InterlockedAdd(queues[hitQueueIndex].writeOffset, 1, outIndex);

        outIndex %= size;

        uint clusterID = GetClusterIDNV(query, RayQueryCommittedIntersectionKHR);
        uint triangleIndex = query.CommittedPrimitiveIndex();
        uint instanceID = query.CommittedInstanceID();
        float2 bary = query.CommittedTriangleBarycentrics();

        uint clusterID_triangleIndex = (clusterID << MAX_CLUSTER_TRIANGLES_BIT) | triangleIndex;
        StoreUint(clusterID_triangleIndex, hitShadingQueueClusterIDIndex, outIndex);
        StoreUint(instanceID, hitShadingQueueInstanceIDIndex, outIndex);
        StoreFloat2(bary, hitShadingQueueBaryIndex, outIndex);
    }
}
