#ifndef VIRTUAL_TEXTURES_HLSLI_
#define VIRTUAL_TEXTURES_HLSLI_

#include "../common.hlsli"
#include "./../rt/shader_interop/virtual_textures_shaderinterop.h"

StructuredBuffer<uint4> pageHashTable : register(t5);
//Texture2DArray physicalPages: register(t6);

namespace VirtualTexture
{
#if 0
    uint ClampMipLevel(uint2 texSize, uint mipLevel)
    {
        uint log2Width = firstbithigh(texSize.x);
        uint log2Height = firstbithigh(texSize.y);
        mipLevel = clamp(mipLevel, 0, min(log2Width, log2Height));
        return mipLevel;
    }
    uint2 CalculateVirtualPage(uint2 baseOffset, float2 uv, uint2 texSize, uint mipLevel)
    {
        mipLevel = ClampMipLevel(texSize, mipLevel);

        const float2 faceTexelOffset = uv * float2(texSize);
        const uint2 virtualAddress = (uint2(baseOffset + (float2)faceTexelOffset) >> mipLevel);
        const uint2 virtualPage = virtualAddress >> PAGE_SHIFT;
        return virtualPage;
    }

    static float3 GetPhysicalUV(uint2 baseOffset, float2 uv, uint2 texSize, uint mipLevel, bool debug = false)
    {
        uint poolWidth, poolHeight, poolNumLayers;
        physicalPages.GetDimensions(poolWidth, poolHeight, poolNumLayers);

        uint log2Width = firstbithigh(texSize.x);
        uint log2Height = firstbithigh(texSize.y);
        uint maxMipLevel = min(log2Width, log2Height);
        mipLevel = clamp(mipLevel, 0, maxMipLevel);

        uint2 virtualPage = CalculateVirtualPage(baseOffset, uv, texSize, mipLevel);
        uint packed = pageTable.Load(float3(virtualPage.x, virtualPage.y, mipLevel));

#if 1
        uint iters = 0;
        while (packed == ~0u && mipLevel < maxMipLevel)
        {
            iters++;
            mipLevel += 1;
            virtualPage = CalculateVirtualPage(baseOffset, uv, texSize, mipLevel);
            packed = pageTable.Load(float3(virtualPage.x, virtualPage.y, mipLevel));
        }
#endif

        uint4 pageTableEntry = UnpackPageTableEntry(packed);
        uint pageX = pageTableEntry.x;
        uint pageY = pageTableEntry.y;
        uint newMipLevel = pageTableEntry.z;
        uint layer = pageTableEntry.w;

        if (0)
        {
            printf("mip: %u %u virtualPage: %u %u\n phys page: %u %u, layer: %u iters: %u", mipLevel, newMipLevel, virtualPage.x, virtualPage.y, 
                    pageX, pageY, layer, iters);
        }

        const float2 faceTexelOffset = uv * float2(texSize);
        const float2 physicalPageCoord = float2(pageX, pageY) * PAGE_WIDTH;
        const float2 mipFaceTexelOffset = faceTexelOffset / float(1u << newMipLevel);
        const uint2 offsetInPage = (baseOffset >> newMipLevel) & (PAGE_WIDTH - 1);

        const float2 texCoord = (physicalPageCoord + offsetInPage + (mipFaceTexelOffset % PAGE_WIDTH)) / float2(poolWidth, poolHeight);
        float3 result = float3(texCoord.x, texCoord.y, layer);

        return result;
    }
#endif

    float4 Sample(uint textureIndex, uint faceID, uint mip, float2 uv, uint2 texSize)//, SamplerState sampler) 
    {
        uint64_t bitsToHash = uint64_t(textureIndex) | (uint64_t(mip) << 16) | (uint64_t(faceID) << 32u);
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
