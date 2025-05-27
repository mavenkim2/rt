#pragma dxc diagnostic push
#pragma dxc diagnostic ignored "-Wambig-lit-shift"
#pragma dxc diagnostic ignored "-Wunused-value"

#include "../common.hlsli"
#define USE_CMPMSC
#define ASPM_HLSL
#define ASPM_GPU
#include "../bcn_common_kernel.h"

#pragma dxc diagnostic pop

static const uint blockLUT = {0, 1, 4, 5};
groupshared float4 rgba[16][16]; 

Texture2D input : register(t0);

#define PTEX

#ifdef PTEX
float3 Load(RWTexture2D<float4> tex, uint2 loc)
{
    float4 v = tex[loc];
    // Gamma 2.2
    v = pow(v, 2.2);
    return v.xyz;
}
#else 
#error
#endif

[numthreads(16, 16, 1)]
void main (uint3 groupID : SV_GroupID, uint groupThreadIndex : SV_GroupIndex)
{
    //  0  1  2  3  16  17  18  19 
    //  4  5  6  7  20  21  22  23
    //  8  9 10 11  24  25  26  27
    // 12 13 14 15  28  29  30  31
    // 32 33 34 35  48  49  50  51 
    // 36 37 38 39  52  53  54  55
    // 40 41 42 43  56  57  58  59
    // 44 45 46 47  60  61  62  63

    int bindlessBaseIndex, bindlessBaseRWTexture;

    // Get swizzled thread ID
    uint localGroupThreadIndex = groupThreadIndex & 63;
    uint2 swizzledThreadID = 0;

    swizzledThreadID.x += localGroupThreadIndex & 0x3;
    swizzledThreadID.x += 4 * ((localGroupThreadIndex >> 4) & 0x1);
    swizzledThreadID.y += (localGroupThreadIndex >> 2) & 0x3;
    swizzledThreadID.y += 4 * ((localGroupThreadIndex >> 5) & 0x1);

    swizzledThreadID.x += 8 * ((localGroupThreadIndex >> 6) & 0x1);
    swizzledThreadID.y += 8 * (localGroupThreadIndex >> 7);

    uint x = swizzledThreadID.x;
    uint y = swizzledThreadID.y;

    float4 vals[4];

    RWTexture2D<float4> baseTex = bindlessRWTexture[bindlessBaseIndex];
    RWTexture2D<uint2> baseOutput = bindlessRWTextureUint[bindlessBaseRWTexture];

    uint width, uint height;
    baseTex.GetDimensions(width, height);

    // There are 16x16 threads. Each thread covers 2x2 texels in a subregion (at the base level).
    // There are 4 subregions. Each group handles in total a 64x64 region.

    // Mip 0: 64x64 region
    float3 block[16];
    for (int i = 0; i < 4; i++)
    {
        uint2 loadCoords = groupID.xy * 64 + swizzledThreadID * 4 + 2 * uint2(i & 1, i >> 1);

        float3 v0 = Load(baseTex, loadCoords);
        float3 v1 = Load(baseTex, loadCoords + uint2(1, 0));
        float3 v2 = Load(baseTex, loadCoords + uint2(0, 1));
        float3 v3 = Load(baseTex, loadCoords + uint2(1, 1));

        uint blockStartIndex = 2 * blockLUT[i];

        block[blockStartIndex + 0] = v0;
        block[blockStartIndex + 1] = v1;
        block[blockStartIndex + 4] = v2;
        block[blockStartIndex + 5] = v3;


        // Downsample
        vals[i] = (v0 + v1 + v2 + v3) * 0.25f;
    }

    uint2 loadCoords = groupID.xy * 16 + swizzledThreadID;
    baseOutput[loadCoords] = CompressBlockBC1_UNORM(block, CMP_QUALITY2, false);

    // Block compress
    if ((localGroupThreadIndex & 3) == 0)
    {
        uint baseLaneIndex = WaveGetLaneIndex() & ~0x3;
        for (int threadIndex = 0; threadIndex < 4; threadIndex++)
        {
            uint indexStart = 2 * blockLUT[threadIndex];
            uint waveIndex = baseLaneIndex + blockLUT[threadIndex];
            for (int texelIndex = 0; texelIndex < 4; texelIndex++)
            {
                uint neighbor = blockLUT[texelIndex];
                block[indexStart + neighbor] = WaveReadLaneAt(v[texelIndex], waveIndex).rgb;
            }
        }
    }

    // Mip 1: 32x32 region
    // Block compress and write to mip 1 output texture
    if ((localGroupThreadIndex & 0xf) == 0)
    {
        float3 block[16];
        uint laneIndex = WaveGetLaneIndex() & ~0xf;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                uint blockIndex = 8 * ((j >> 2) & 1);
                blockIndex += 4 * ((j >> 1) & 1);
                blockIndex += 2 * ((j >> 3) & 1);
                blockIndex += j & 1;

                block[blockIndex] = float4 v = WaveReadLaneAt(v[i], laneIndex + j);
            }
            uint2 blockCoords = texCoords[i] >> 2;
            baseOutput[blockCoords] = CompressBlockBC1_UNORM(block, CMP_QUALITY2, false);
        }
    }

    // Store second mip to groupshared
    if ((localGroupThreadIndex & 3) == 0)
    {
        for (int i = 0; i < 4; i++)
        {
            uint laneIndex = WaveGetLaneIndex() & ~0x3;
            float4 v0 = v[i];
            float4 v1 = WaveReadLaneAt(v[i], laneIndex | 1);
            float4 v2 = WaveReadLaneAt(v[i], laneIndex | 2);
            float4 v3 = WaveReadLaneAt(v[i], laneIndex | 3);
            float4 vResult = (v0 + v1 + v2 + v3) * 0.25f;

            uint2 index = (texCoords[i] >> 1) & 0xf;
            rgba[index.x][index.y] = vResult;
        }
    }

    // Block compress second mip
    float4 v = rgba[x][y];
    if ((localGroupThreadIndex & 0xf) == 0)
    {
        float3 block[16];
        uint laneIndex = WaveGetLaneIndex() & ~0xf;
        for (int i = 0; i < 16; i++)
        {
            float4 v0 = WaveReadLaneAt(v, laneIndex + i);
            block[?] = v0;
        }
    }
}
