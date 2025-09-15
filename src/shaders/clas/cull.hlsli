#include "../common.hlsli"

bool FrustumCull(float4x4 clipFromRender, float3x4 renderFromObject, float3 minP, float3 maxP, float p22, float p23)
{
    float4x4 renderFromObject_ = float4x4(
        renderFromObject[0][0], renderFromObject[0][1], renderFromObject[0][2], renderFromObject[0][3], 
        renderFromObject[1][0], renderFromObject[1][1], renderFromObject[1][2], renderFromObject[1][3], 
        renderFromObject[2][0], renderFromObject[2][1], renderFromObject[2][2], renderFromObject[2][3], 
        0, 0, 0, 1.f);
    float4x4 mvp = mul(clipFromRender, renderFromObject_);
    float4 sX = mul(mvp, float4(maxP.x - minP.x, 0, 0, 0));
    float4 sY = mul(mvp, float4(0, maxP.y - minP.y, 0, 0));
    float4 sZ = mul(mvp, float4(0, 0, maxP.z - minP.z, 0));

    float4 planesMin = 1.f;
    // If the min of p.x - p.w > 0, then all points are past the right clip plane.
    // If the min of -p.x - p.w > 0, then -p.x > p.w -> p.x < -p.w for all points.
    float minW = FLT_MAX;
    float maxW = -FLT_MAX;
    float4 aabb;
    aabb.xy = float2(1.f, 1.f);
    aabb.zw = float2(-1.f, -1.f);

#define PLANEMIN(a, b) planesMin = min(planesMin, min(float4(a.xy, -a.xy) - a.w, float4(b.xy, -b.xy) - b.w))
#define PROCESS(a, b) \
{ \
    float2 pa = a.xy/a.w; \
    float2 pb = b.xy/b.w; \
    minW = min(minW, min(a.w, b.w)); \
    maxW = max(maxW, max(a.w, b.w)); \
    aabb.xy = min(aabb.xy, min(pa, pb)); \
    aabb.zw = max(aabb.zw, max(pa, pb)); \
}
    
    float4 p0 = mul(mvp, float4(minP.x, minP.y, minP.z, 1.0));
    float4 p1 = p0 + sZ;

    float4 p2 = p0 + sX;
    float4 p3 = p1 + sX;

    float4 p4 = p2 + sY;
    float4 p5 = p3 + sY;

    float4 p6 = p4 - sX;
    float4 p7 = p5 - sX;

    bool visible = true;

    if (1)
    {
        PLANEMIN(p0, p1);
        PLANEMIN(p2, p3);
        PLANEMIN(p4, p5);
        PLANEMIN(p6, p7);
        visible = !(any(planesMin > 0.f));
    }
    PROCESS(p0, p1);
    PROCESS(p2, p3);
    PROCESS(p4, p5);
    PROCESS(p6, p7);

    // NOTE: w = -z in view space. Min(z) = -(Max(-z)) and Max(z) = -(Min(-z))
    float maxZ = -maxW * p22 + p23;
    float minZ = -minW * p22 + p23;

    // partially = at least partially
    bool test = maxZ > minZ;//minW > 0;//maxZ > minZ;
    bool isPartiallyInsideNearPlane = maxZ > 0;
    bool isPartiallyOutsideNearPlane = minZ <= 0;
    bool isPartiallyInsideFarPlane = minZ < minW;
    //bool isPartiallyOutsideFarPlane = maxZ >= minW;

    visible = visible && isPartiallyInsideFarPlane && isPartiallyInsideNearPlane;
    
#if 0
    results.aabb = aabb;
    results.minZ = saturate(minZ / minW);
    results.isVisible = visible;
    results.crossesNearPlane = isPartiallyOutsideNearPlane;
#endif

    return !visible;
}
