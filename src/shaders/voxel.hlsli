#ifndef VOXEL_HLSLI
#define VOXEL_HLSLI

#include "common.hlsli"
#include "sampling.hlsli"

float3 SampleUniformSphere(float2 u)
{
    float z   = 1 - 2 * u[0];
    float r   = sqrt(1 - z * z);
    float phi = 2 * PI * u[1];
    return float3(r * cos(phi), r * sin(phi), z);
}

struct SGGX 
{
    float nxx;
    float nxy;
    float nxz;

    float nyy;
    float nyz;

    float nzz;

    float ProjectedArea(float3 wi)
    {
        float3 temp;
        temp.x = nxy * wi.y;
        temp.y = nyz * wi.z;
        temp.z = nxz * wi.x;

        temp *= 2.f;

        temp.x += nxx * wi.x;
        temp.y += nyy * wi.y;
        temp.z += nzz * wi.z;
        
        return dot(temp, wi);
    }

    float3 SampleSGGX(float3 w, float3 wk, float3 wj, float2 u) 
    {
        float3x3 S = float3x3(nxx, nxy, nxz, nxy, nyy, nyz, nxz, nyz, nzz);

        float skk = ProjectedArea(wk);
        float sjj = ProjectedArea(wj);
        float sii = ProjectedArea(w);

        float3 sw = mul(S, w);

        float ski = dot(wk, sw);
        float skj = dot(wk, mul(S, wj));
        float sji = dot(wj, sw);
        
        float div = rsqrt(sii);

        float sqrtDet = sqrt(abs(skk * sjj * sii - skj * skj * sii - ski * ski * sjj - sji * sji * skk + 2 * skj * skj * sji));
        float test = sjj * sii - sji * sji;
        float val = sqrt(max(0.f, sjj * sii - sji * sji));
        float mjxDiv = 1.f / val;

        float3 mk = float3(sqrtDet * mjxDiv, 0, 0);
        float3 mj = div * float3(-(ski * sji - skj * sii) * mjxDiv, val, 0.f);
        float3 mi = div * float3(ski, sji, sii);

        //printf("%f \n %f %f %f\n%f %f %f\n%f %f %f\n%f %f %f", sqrtDet, mk.x, mk.y, mk.z, mj.x, mj.y, mj.z, mi.x, mi.y, mi.z, sjj, sii, sji);

        //float3 p = SampleUniformSphere(u);
        float3 p;
        float r = sqrt(u.x);
        float phi = 2.f * PI * u.y;
        p.x = r * cos(phi);
        p.y = r * sin(phi);
        p.z = sqrt(1 - p.x * p.x - p.y * p.y);
        float3 wm = normalize(p.x * mk + p.y * mj + p.z * mi);

        wm = normalize(wm.x * wk + wm.y * wj + wm.z * w);

        return wm;
    }

    float PhaseFunction(float3 wm, float3 wi)
    {
        return InvPi * dot(wm, wi);
    }
};

#endif
