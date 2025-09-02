#include "../common.hlsli"

bool FrustumCull(float4x4 clipFromRender, float3x4 renderFromObject, float3 minP, float3 maxP, float p22, float p23)
{
    // Frustum culling
    float4x4 renderFromObject_ = float4x4(
        renderFromObject[0][0], renderFromObject[0][1], renderFromObject[0][2], renderFromObject[0][3], 
        renderFromObject[1][0], renderFromObject[1][1], renderFromObject[1][2], renderFromObject[1][3], 
        renderFromObject[2][0], renderFromObject[2][1], renderFromObject[2][2], renderFromObject[2][3], 
        0, 0, 0, 1.f);

    float4x4 mvp = mul(clipFromRender, renderFromObject_);

    float4 sides[3];
    sides[0] = mul(mvp, float4(maxP[0] - minP[0], 0.f, 0.f, 0.f));
    sides[1] = mul(mvp, float4(maxP[1] - minP[1], 0.f, 0.f, 0.f));
    sides[2] = mul(mvp, float4(maxP[2] - minP[2], 0.f, 0.f, 0.f));

    float4 planeMin = 1.f;
    float4 p0 = mul(mvp, float4(minP, 1.f));

    float minW = FLT_MAX;
    float maxW = -FLT_MAX;

    for (int i = 0; i < 8; i++)
    {
        float4 p = p0;
        if (i & 1) p += sides[0];
        if (i & 2) p += sides[1];
        if (i & 4) p += sides[2];
        planeMin = min(float4(p.xy, -p.xy) - p.w, planeMin);

        minW = min(minW, p.w);
        maxW = max(maxW, p.w);
    }

    bool cull = any(planeMin > 0.f);

    float z0 = -minW * p22 + p23;
    float z1 = -maxW * p22 + p23;

    float minZ = min(z0, z1);
    float maxZ = max(z0, z1);

    cull |= maxZ < 0.f;
    return cull;
}
