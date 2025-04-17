#include "../dgf_intersect.hlsli"

bool VisibilityRay(RaytracingAccelerationStructure rts, float3 origin, float3 dir, float tMax) 
{
    RayDesc desc;
    desc.Origin = origin;
    desc.Direction = dir;
    desc.TMin = 0;
    desc.TMax = tMax;

    RayQuery<RAY_FLAG_SKIP_TRIANGLES | RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(rts, RAY_FLAG_NONE, 0xff, desc);

    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
            uint primitiveIndex = query.CandidatePrimitiveIndex();
            uint instanceID = query.CandidateInstanceID();
            float3 o = query.CandidateObjectRayOrigin();
            float3 d = query.CandidateObjectRayDirection();

            float tHit = 0;
            uint kind = 0;
            float2 tempBary = 0;

            bool result = IntersectCluster(instanceID, primitiveIndex, o, d, query.RayTMin(), 
                                           query.CommittedRayT(), tHit, kind, tempBary);

            if (result) return true;
        }
    }
    return false;
}
