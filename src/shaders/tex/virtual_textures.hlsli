#ifndef VIRTUAL_TEXTURES_HLSLI_
#define VIRTUAL_TEXTURES_HLSLI_

#include "./../rt/shader_interop/virtual_textures_shaderinterop.h"

Texture2D<uint> pageTable : register(t5);
Texture2DArray physicalPages: register(t6);

namespace VirtualTexture
{
    static float3 GetPhysicalUV(uint2 baseOffset, float2 uv, uint2 texSize, uint mipLevel)
    {
        uint poolWidth, poolHeight, poolNumLayers;
        physicalPages.GetDimensions(poolWidth, poolHeight, poolNumLayers);

        uint log2Width = firstbithigh(texSize.x);
        uint log2Height = firstbithigh(texSize.y);
        mipLevel = clamp(mipLevel, 0, min(log2Width, log2Height) - 2);

        const float2 faceTexelOffset = uv * float2(texSize);
        const uint2 virtualAddress = ((baseOffset + (uint2)faceTexelOffset) >> mipLevel);
        const uint2 virtualPage = virtualAddress >> PAGE_SHIFT;
        const uint packed = pageTable.Load(float3(virtualPage.x, virtualPage.y, mipLevel));

        uint pageX = BitFieldExtractU32(packed, 8, 0);
        uint pageY = BitFieldExtractU32(packed, 8, 8);
        uint layer = BitFieldExtractU32(packed, 2, 16);
        const float2 physicalPageCoord = float2(pageX, pageY) * PAGE_WIDTH;
        const float2 offsetInPage = float2(virtualAddress & (PAGE_WIDTH - 1));

        const float2 texCoord = (physicalPageCoord + offsetInPage + (faceTexelOffset % PAGE_WIDTH)) / float2(poolWidth, poolHeight);
        const float3 result = float3(texCoord.x, texCoord.y, layer);

        return result;
    }

#if 0
    static float3 GetPhysicalUV(StructuredBuffer<uint2> pageTable,
                                uint basePageOffset,
                                uint faceID,
                                float2 uv, uint mipLevel = 0, bool debug = false)
    {
        // page information.x contains the faceID
        uint faceIndex = basePageOffset + faceID;
        uint2 physicalPageInfo = pageTable.Load(faceIndex);

        uint x = BitFieldExtractU32(physicalPageInfo.x, 15, 0);
        uint y = BitFieldExtractU32(physicalPageInfo.x, 15, 15);
        uint layerIndex = BitFieldExtractU32(physicalPageInfo.x, 2, 30);

        int log2Width = BitFieldExtractU32(physicalPageInfo.y, 4, 0);
        int log2Height = BitFieldExtractU32(physicalPageInfo.y, 4, 4);
        uint basePhysicalLevel = BitFieldExtractU32(physicalPageInfo.y, 4, 8);
        int rotate = BitFieldExtractU32(physicalPageInfo.y, 1, 12);

        log2Width = max(0, log2Width - basePhysicalLevel);
        log2Height = max(0, log2Height - basePhysicalLevel);
        uint numLevels = max(log2Width, log2Height) + 1u;

        uint2 offset = uint2(x, y);
        //((1u << numLevels) - 1u) & ~((1u << (numLevels - mipLevel - basePhysicalLevel + 1)) - 1u);

        // Packing pattern: subsequent mips zigzag right and down. once one of the dimensions 
        // becomes 1, keep appending to the right.
        // |            ||      |
        // |      0     ||  1   |
        // |            ||      |
        // |            ||      |
        // |            ||2 ||3|
        // |            ||  |
        // |            |
        // |            |

        for (int levelIndex = 0; levelIndex < (int)mipLevel - (int)basePhysicalLevel; levelIndex++)
        {
            uint index = min(log2Width, log2Height) == 0 ? 0 : (levelIndex & 1);
            uint faceOffset = CalculateFaceSize(log2Width, log2Height)[index];
            faceOffset = (faceOffset + 3) & ~3u;
            offset[index] += faceOffset;
            log2Width = max(log2Width - 1, 0);
            log2Height = max(log2Height - 1, 0);
        }
        
        uv = rotate ? float2(1 - uv.y, uv.x) : uv;
        uint borderSize = GetBorderSize(log2Width, log2Height);
        offset += uv * float2(1u << log2Width, 1u << log2Height) + borderSize;

        if (0)
        {
            printf("miplevel: %u, faceID: %u, offset: %u %u, base x y: %u %u, log: %u %u, uv: %f %f\n", 
                    mipLevel, faceID, offset.x, offset.y, x, y, log2Width, log2Height, uv.x, uv.y);
        }

        uint width, height, layers;
        physicalPages.GetDimensions(width, height, layers);
        float3 physicalUv = float3((float)offset.x / width, (float)offset.y / height, layerIndex);
        return physicalUv;
    }
#endif
}

#endif
