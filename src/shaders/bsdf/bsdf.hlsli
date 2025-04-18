#ifndef BSDF_HLSLI_
#define BSDF_HLSLI_

#include "../common.hlsli"
#include "../bit_twiddling.hlsli"
#include "../tex/filter.hlsli"
#include "bxdf.hlsli"
#include "../sampling.hlsli"

float3 SampleDielectric(float3 wo, float eta, float2 rand, inout float3 throughput) 
{
    float u = rand.x;
    float R = FrDielectric(wo.z, eta);

    float T = 1 - R;
    float pr = R, pt = T;

    float3 dir;
    if (u < pr / (pr + pt))
    {
        dir = float3(-wo.x, -wo.y, wo.z);
    }
    else
    {
        float etap;
        bool valid = Refract(wo, float3(0, 0, 1), eta, etap, dir);
        if (!valid)
        {
            return 0;
        }
    
        throughput /= etap * etap;
    }

    return normalize(dir);
}

float3 SampleDiffuse(float3 R, float3 wo, float2 u, inout float3 throughput, bool debug = false)
{
    float3 wi = SampleCosineHemisphere(u);
    wi.z = wo.z < 0 ? -wi.z : wi.z;
    float pdf = CosineHemispherePDF(abs(wi.z));

    if (0) 
    {
        printf("R: %f %f %f\n", R.x, R.y, R.z);
    }
    throughput *= R * InvPi * abs(wi.z) * rcp(pdf);

    return wi;
}

#endif
