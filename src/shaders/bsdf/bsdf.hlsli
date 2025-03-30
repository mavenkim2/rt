#ifndef BSDF_HLSLI_
#define BSDF_HLSLI_

#include "../common.hlsli"
#include "../bit_twiddling.hlsli"
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

float3 SampleDiffuse(int reflectanceDescriptor, int faceID, float3 wo, float2 u, inout float3 throughput, bool debug = false)
{
    float3 wi = SampleCosineHemisphere(u);
    wi.z = wo.z < 0 ? -wi.z : wi.z;
    float pdf = CosineHemispherePDF(abs(wi.z));

    uint2 offsets = GetAlignedAddressAndBitOffset(3 * faceID, 0);
    uint2 result = bindlessBuffer[reflectanceDescriptor].Load2(offsets[0]);
    uint data = BitAlignU32(result.y, result.x, offsets[1]);
    float3 R;
    R.x = BitFieldExtractU32(data, 8, 0);
    R.y = BitFieldExtractU32(data, 8, 8);
    R.z = BitFieldExtractU32(data, 8, 16);

    R = pow(max(R, 0) / 255.f, 2.2);

    if (debug) 
    {
        printf("faceID: %u %u\nR: %f %f %f\n", faceID, reflectanceDescriptor, R.x, R.y, R.z);
    }
    throughput *= R * InvPi * abs(wi.z) * rcp(pdf);

    return wi;
}

#endif
