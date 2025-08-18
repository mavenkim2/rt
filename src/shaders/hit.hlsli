#include "common.hlsli"
#include "dense_geometry.hlsli"

struct HitInfo 
{
    float3 hitP;
    float2 uv;

    float3 ss;
    float3 ts;
    float3 n;

    float3 gn;
};

HitInfo CalculateTriangleHitInfo(float3 p0, float3 p1, float3 p2, float3 n0, float3 n1, float3 n2, 
                                 float2 uv0, float2 uv1, float2 uv2, float2 bary) 
{

    float2 duv10 = uv1 - uv0;
    float2 duv20 = uv2 - uv0;
    float det = mad(duv10.x, duv20.y, -duv10.y * duv20.x);

    float3 dpdu, dpdv, dndu, dndv = 0;
    float3 dp10 = p1 - p0;
    float3 dp20 = p2 - p0;
    float3 dn10 = n1 - n0;
    float3 dn20 = n2 - n0;

    float3 gn = normalize(cross(p0 - p2, p1 - p2));
    float3 origin = p0 + dp10 * bary.x + dp20 * bary.y;
    float2 uv = uv0 + duv10 * bary.x + duv20 * bary.y;

    if (abs(det) < 1e-9f)
    {
        float2x3 tb = BuildOrthonormalBasis(gn);
        dpdu = tb[0];
        dpdv = tb[1];
    }
    else 
    {
        float invDet = rcp(det);

        dpdu = mad(duv20.y, dp10, -duv10.y * dp20) * invDet;
        dpdv = mad(-duv20.x, dp10, duv10.x * dp20) * invDet;
        
        dndu = mad(duv20.y, dn10, -duv10.y * dn20) * invDet;
        dndv = mad(-duv20.x, dn10, duv10.x * dn20) * invDet;
    }

    float3 n = normalize(n0 + dn10 * bary[0] + dn20 * bary[1]);

    float3 ss = dpdu;
    float3 ts = cross(n, ss);
    if (dot(ts, ts) > 0)
    {
        ss = cross(ts, n);
    }
    else
    {
        float2x3 tb = BuildOrthonormalBasis(n);
        ss = tb[0];
        ts = tb[1];
    }

    ss = normalize(ss);
    ts = cross(n, ss);

    HitInfo hitInfo;
    hitInfo.hitP = origin;
    hitInfo.uv = uv;
    hitInfo.ss = ss;
    hitInfo.ts = ts;
    hitInfo.n = n;
    hitInfo.gn = gn;

    return hitInfo;
}
