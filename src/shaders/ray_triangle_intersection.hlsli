#pragma warning(disable:G5793C645)

bool RayTriangleIntersectionMollerTrumbore(float3 o, float3 d, float3 v0, float3 v1, float3 v2,
                                           float tFar, out float intersectionT,
                                           out float2 barycentrics)
{
    const float epsilon = 1e-9;

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

    if (u < 0 || u > 1 || v < 0 || v > 1) return false;

    const float t = dot(ng, c);
    if (t <= 0 || t > abs(det) * tFar) return false;

    tFar           = t;
    barycentrics.x = u;
    barycentrics.y = v;
    return true;
}
