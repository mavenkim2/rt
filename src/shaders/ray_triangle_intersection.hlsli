#ifndef RAY_TRIANGLE_INTERSECTION_HLSLI_
#define RAY_TRIANGLE_INTERSECTION_HLSLI_
bool RayTriangleIntersectionMollerTrumbore(float3 o, float3 d, float3 v0, float3 v1, float3 v2,
                                           out float intersectionT, out float2 barycentrics)
{
    intersectionT = 0;
    barycentrics = 0;
    const float epsilon = 1e-9f;

    const float3 e1 = v0 - v1;
    const float3 e2 = v2 - v0;
    const float3 ng = cross(e2, e1);
    const float3 c  = v0 - o;

    const float det = dot(d, ng);
    if (det > -epsilon && det < epsilon) return false;
    const float invDet = rcp(det);

    const float3 dxt = cross(c, d);

    const float u = dot(dxt, e2) * invDet;
    const float v = dot(dxt, e1) * invDet;

    barycentrics = float2(u, v);

    if (u < 0 || v < 0 || u + v > 1) return false;

    const float t = dot(ng, c) * invDet;
    intersectionT = t;
    barycentrics.x = u;
    barycentrics.y = v;

    return true;
}
#endif
