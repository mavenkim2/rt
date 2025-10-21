#ifndef VIRTUAL_TEXTURES_HLSLI_
#define VIRTUAL_TEXTURES_HLSLI_

#include "../common.hlsli"
#include "../../rt/shader_interop/virtual_textures_shaderinterop.h"

StructuredBuffer<uint4> pageHashTable : register(t5);

namespace VirtualTexture
{
    bool SampleHelper(uint textureIndex, uint tileIndex, uint faceID, uint mip, float2 uv, uint2 texSize, out float4 result)
    {
        uint64_t bitsToHash = uint64_t(textureIndex) | (uint64_t(tileIndex) << 16) | 
                              (uint64_t(faceID) << 32) | (uint64_t(mip) << 60);
        uint64_t hash = MixBits(bitsToHash);

        uint maxHashSize = (1u << 24u);
        uint mask = maxHashSize - 1;
        uint hashIndex = uint(hash) & mask;

        for (;;)
        {
            uint4 hashValue = pageHashTable[hashIndex];
            if (hashValue.x != ~0u)
            {
                TextureHashTableEntry pageTableEntry = UnpackPageTableEntry(hashValue);
                if (pageTableEntry.faceID == faceID && pageTableEntry.textureIndex == textureIndex && pageTableEntry.mip == mip && pageTableEntry.tileIndex == tileIndex)
                {
                    uint width, height;
                    bindlessFloat4Textures[NonUniformResourceIndex(pageTableEntry.bindlessIndex)].GetDimensions(width, height);

                    float2 texel = float2(pageTableEntry.offset) + uv * texSize;
                    texel = clamp(texel, 0, float2(width, height));
                    float2 texCoord = texel / float2(width, height);
                    result = bindlessFloat4Textures[NonUniformResourceIndex(pageTableEntry.bindlessIndex)].SampleLevel(samplerNearestClamp, texCoord, 0.f);
                    return true;
                }
            }
            else 
            {
                return false;
            }
            hashIndex = (hashIndex + 1) % maxHashSize;
        }
    }

    float4 Sample(uint textureIndex, uint tileIndex, uint faceID, uint mip, float2 uv, uint2 texSize, uint maxMip)
    {
        float4 result = float4(0, 0, 1, 1);
        if (!SampleHelper(textureIndex, tileIndex, faceID, mip, uv, texSize, result))
        {
            SampleHelper(textureIndex, 0, faceID, maxMip, float2(0.5, 0.5), float2(1, 1), result);
        }
        return result;
    }
}

#endif
