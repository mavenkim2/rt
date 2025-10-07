#ifndef VIRTUAL_TEXTURES_HLSLI_
#define VIRTUAL_TEXTURES_HLSLI_

#include "../common.hlsli"
#include "./../rt/shader_interop/virtual_textures_shaderinterop.h"

StructuredBuffer<uint4> pageHashTable : register(t5);

namespace VirtualTexture
{
    float4 Sample(uint textureIndex, uint tileIndex, uint faceID, uint mip, float2 uv, uint2 texSize)
    {
        uint64_t bitsToHash = uint64_t(textureIndex) | (uint64_t(tileIndex) << 16) | 
                              (uint64_t(faceID) << 32) | (uint64_t(mip) << 60);
        uint64_t hash = MixBits(bitsToHash);

        uint maxHashSize = (1u << 24u);
        uint hashIndex = uint(hash % maxHashSize);

        for (;;)
        {
            uint4 hashValue = pageHashTable[hashIndex];
            if (hashValue.x != ~0u)
            {
                TextureHashTableEntry pageTableEntry = UnpackPageTableEntry(hashValue);
                if (pageTableEntry.faceID == faceID && pageTableEntry.textureIndex == textureIndex && pageTableEntry.mip == mip)
                {
                    uint width, height, layers;
                    bindlessFloat4ArrayTextures[NonUniformResourceIndex(pageTableEntry.bindlessIndex)].GetDimensions(width, height, layers);

                    float2 texel = float2(pageTableEntry.offset) + uv * uint2(width, height);
                    float3 texCoord = float3(texel / float2(width, height), pageTableEntry.layer);
                    float4 result = bindlessFloat4ArrayTextures[NonUniformResourceIndex(pageTableEntry.bindlessIndex)].SampleLevel(samplerNearestClamp, texCoord, 0.f);

                    //float4 result = 0;

                    return result;
                }
            }
            else 
            {
                return 0;
                //break;
            }
            hashIndex = (hashIndex + 1) % maxHashSize;
        }

// TODO fallback
#if 0
        {
            uint maxMip = max(firstbithigh(texSize.x), firstbithigh(texSize.y)) + 1;
            uint64_t bitsToHash = uint64_t(textureIndex) | (uint64_t(maxMip) << 16) | (uint64_t(faceID) << 32u);
            uint64_t hash = MixBits(bitsToHash);

            uint hashIndex = uint(hash % maxHashSize);

            for (;;)
            {
                uint3 hashValue = pageHashTable[hashIndex];
                if (hashValue.x != ~0u)
                {
                    uint testTextureIndex = BitFieldExtractU32(hashValue.x, 16, 0);
                    uint testMip = BitFieldExtractU32(hashValue.x, 4, 16);
                    if (hashValue.y == faceID && testTextureIndex == textureIndex && maxMip == testMip)
                    {
                        uint layer;
                        uint4 pageTableEntry = UnpackPageTableEntry(hashValue.z, layer);
                        uint pageX = pageTableEntry.x;
                        uint pageY = pageTableEntry.y;
                        uint offsetInPageX = pageTableEntry.z;
                        uint offsetInPageY = pageTableEntry.w;

                        uint2 texel = uint2(pageX * PAGE_WIDTH + offsetInPageX, pageY * PAGE_WIDTH + offsetInPageY);
                        float3 texCoord = float3(float2(texel) / float2(poolWidth, poolHeight), layer);

                        return texCoord;
                    }
                }
                else 
                {
                    return 0;
                }
                hashIndex = (hashIndex + 1) % maxHashSize;
            }
        }

#endif
    }
}

#endif
