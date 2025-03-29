#ifndef BSDF_HLSLI_
#define BSDF_HLSLI_

#include "../common.hlsli"
#include "bxdf.hlsli"
#include "../sampling.hlsli"

float3 SampleDielectric(float3 wo, float3 n, float eta, float2 rand, inout float3 throughput) 
{
    float u = rand.x;
    float R = FrDielectric(dot(wo, n), eta);

    float T = 1 - R;
    float pr = R, pt = T;

    float3 dir;
    if (u < pr / (pr + pt))
    {
        dir = Reflect(wo, n);
    }
    else
    {
        float etap;
        bool valid = Refract(wo, n, eta, etap, dir);
        if (!valid)
        {
            return 0;
        }
    
        throughput /= etap * etap;
    }

    return normalize(dir);
}

float3 SampleDiffuse(float3 wo, float3 n, float2 u, inout float3 throughput)
{
    float3 wi = SampleCosineHemisphere(u);
    wi.z = wo.z < 0 ? -wi.z : wi.z;
    float pdf = CosineHemispherePDF(abs(wi.z));

    // TODO: temp
    float3 R = float3(0.1, 0.5, 0.3);
    throughput *= R * InvPi * abs(wi.z) * rcp(pdf);

    return wi;
}

#endif
