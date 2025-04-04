#pragma dxc diagnostic push
#pragma dxc diagnostic ignored "-Wambig-lit-shift"
#pragma dxc diagnostic ignored "-Wunused-value"

#include "../common.hlsli"
#include "../ptex/filter.hlsli"
#define USE_CMPMSC
#define ASPM_HLSL
#define ASPM_GPU
#include "bcn_common_kernel.h"

#pragma dxc diagnostic pop

// BC1
Texture2D input : register(t0);
RWTexture2D<uint2> output : register(u1);

// NOTE: number of threads in a thread group
[numthreads(8, 8, 1)]
void main (uint3 DTid : SV_DispatchThreadID)
{
    uint2 dimensions;
    input.GetDimensions(dimensions.x, dimensions.y);
    uint2 blockDim = (dimensions + 4 - 1) / 4.f;

    [branch] // NOTE: hint to compiler to evaluate 
    if (any(DTid.xy >= blockDim))
    {
        return;
    }

    float2 rcpDim = rcp(dimensions);
    float2 uv = float2(DTid.xy * 4 + 0.5) * rcpDim;

    float3 block[16];

    // Get the 4x4 region of texels
    // Gather returns in this order
    // W Z 
    // X Y

    for (uint y = 0; y < 2; y++)
    {
        for (uint x = 0; x < 2; x++)
        {
            float4 red = input.GatherRed(samplerLinearClamp, uv, int2(2 * x, 2 * y));
            float4 green = input.GatherGreen(samplerLinearClamp, uv, int2(2 * x, 2 * y));
            float4 blue = input.GatherBlue(samplerLinearClamp, uv, int2(2 * x, 2 * y));

            block[8 * y + 2 * x] = GammaToLinear(float3(red[3], green[3], blue[3]));
            block[8 * y + 2 * x + 1] = GammaToLinear(float3(red[2], green[2], blue[2]));
            block[8 * y + 2 * x + 4] = GammaToLinear(float3(red[0], green[0], blue[0]));
            block[8 * y + 2 * x + 5] = GammaToLinear(float3(red[1], green[1], blue[1]));
        }
    }

    output[DTid.xy] = CompressBlockBC1_UNORM(block, CMP_QUALITY2, false);
}

