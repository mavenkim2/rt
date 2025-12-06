#ifndef INTERSECT_RAY_AABB_HLSLI_
#define INTERSECT_RAY_AABB_HLSLI_

bool IntersectRayAABB(float3 boundsMin, float3 boundsMax, float3 o, float3 d, out float tEntry, out float tLeave)
{
    float3 invDir = 1.f / d;
    float3 tIntersectMin = (boundsMin - o) * invDir;
    float3 tIntersectMax = (boundsMax - o) * invDir;

    float3 tMin = min(tIntersectMin, tIntersectMax);
    float3 tMax = max(tIntersectMin, tIntersectMax);

    tEntry = max(tMin.x, max(tMin.y, max(tMin.z, 0.f)));
    tLeave = min(tMax.x, min(tMax.y, tMax.z));

    return tEntry <= tLeave;
}

#endif
