#pragma dxc diagnostic push
#pragma dxc diagnostic ignored "-Wambig-lit-shift"
#pragma dxc diagnostic ignored "-Wunused-value"

#include "../common.hlsli"
#define USE_CMPMSC
#define ASPM_HLSL
#define ASPM_GPU
#include "../bcn_common_kernel.h"

#pragma dxc diagnostic pop

groupshared float4 rgba[16][16]; 

Texture2D input : register(t0);

[numthreads(8, 8, 1)]
void main (uint3 groupID : SV_GroupID, uint groupThreadIndex : SV_GroupIndex)
{
    // 0 1 8   9 16 17
    // 2 3 10 11 18 19
    // 4 5 12 13 20 21
    // 6 7 14 15 22 23

    int bindlessBaseIndex, bindlessBaseRWTexture;

    uint localGroupThreadIndex = groupThreadIndex & 63;
    uint2 swizzledThreadID = 8 * groupID;

    swizzledThreadID.x += 4 * ((localGroupThreadIndex >> 4) & 1);
    swizzledThreadID.x += 2 * ((localGroupThreadIndex >> 3) & 1);
    swizzledThreadID.x += localGroupThreadIndex & 1;

    swizzledThreadID.y += 4 * ((localGroupThreadIndex >> 5) & 1);
    swizzledThreadID.y += 2 * ((localGroupThreadIndex >> 2) & 1)
    swizzledThreadID.y += (localGroupThreadIndex >> 2) & 1;

    swizzledThreadID.y = (groupThreadID.y >> 2) << 1

    float4 vals[4];
    uint2 texCoords[4];

    RWTexture2D<float4> baseTex = bindlessRWTexture[bindlessBaseIndex];
    RWTexture2D<float4> texMip1 = bindlessRWTexture[bindlessBaseIndex + 1];
    RWTexture2D<float4> texMip2 = bindlessRWTexture[bindlessBaseIndex + 1];

    RWTexture2D<uint2> baseOutput = bindlessRWTextureUint[bindlessBaseRWTexture];

    uint width, uint height;
    baseTex.GetDimensions(width, height);

    // First mip level handled using wave intrinsics
    for (int i = 0; i < 4; i++)
    {
        uint2 loadCoords = groupID.xy * 64 + swizzledThreadID * 2 + 32 * uint2(i & 1, i >> 1);
        float4 v0 = baseTex[loadCoords];
        float4 v1 = baseTex[loadCoords + uint2(1, 0)];
        float4 v2 = baseTex[loadCoords + uint2(0, 1)];
        float4 v3 = baseTex[loadCoords + uint2(1, 1)];

        // Block compress
        if ((localGroupThreadIndex & 3) == 0)
        {
            float3 block[16];

            uint baseLaneIndex = WaveGetLaneIndex() & ~0x3;
            for (int i = 0; i < 4; i++)
            {
                uint indexStart = (i << 2) | ((i & 1) << 1);
                block[indexStart + 0] = WaveReadLaneAt(v0, baseLaneIndex | i).rgb;
                block[indexStart + 1] = WaveReadLaneAt(v1, baseLaneIndex | i).rgb;
                block[indexStart + 4] = WaveReadLaneAt(v2, baseLaneIndex | i).rgb;
                block[indexStart + 5] = WaveReadLaneAt(v3, baseLaneIndex | i).rgb;
            }
            baseOutput[loadCoords >> 2] = CompressBlockBC1_UNORM(block, CMP_QUALITY2, false);
        }

        // Then create next mip
        texCoords[i] = loadCoords >> 1;
        vals[i] = (v0 + v1 + v2 + v3) * 0.25f;
    }

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

            rgba[(texCoords[i] >> 2) & 0xf] = vResult;
        }
    }

    // Block compress second mip
    {
    }
}
