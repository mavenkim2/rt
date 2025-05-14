#ifndef VIRTUAL_TEXTURES_HLSLI_
#define VIRTUAL_TEXTURES_HLSLI_

#include "./../rt/shader_interop/virtual_textures_shaderinterop.h"

Texture2D<uint> pageTable : register(t5);
Texture2DArray physicalPages: register(t6);

namespace VirtualTexture
{
    static float3 GetPhysicalUV(uint2 baseOffset, float2 uv, uint2 texSize, uint mipLevel, bool debug = false)
    {
        uint poolWidth, poolHeight, poolNumLayers;
        physicalPages.GetDimensions(poolWidth, poolHeight, poolNumLayers);

        uint log2Width = firstbithigh(texSize.x);
        uint log2Height = firstbithigh(texSize.y);
        mipLevel = clamp(mipLevel, 0, min(log2Width, log2Height));

        const float2 faceTexelOffset = uv * float2(texSize);
        const uint2 virtualAddress = ((baseOffset + (uint2)faceTexelOffset) >> mipLevel);
        const uint2 virtualPage = virtualAddress >> PAGE_SHIFT;
        const uint packed = pageTable.Load(float3(virtualPage.x, virtualPage.y, mipLevel));

        uint pageX = BitFieldExtractU32(packed, 8, 0);
        uint pageY = BitFieldExtractU32(packed, 8, 8);
        uint layer = BitFieldExtractU32(packed, 2, 16);
        const float2 physicalPageCoord = float2(pageX, pageY) * PAGE_WIDTH;
        const float2 offsetInPage = float2(virtualAddress & (PAGE_WIDTH - 1));

        const float2 mipFaceTexelOffset = float2(asfloat(asuint(faceTexelOffset.x) - (mipLevel << 23)), asfloat(asuint(faceTexelOffset.y) - (mipLevel << 23)));

        if (debug)
        {
            printf("virtual address: %u %u\n", virtualAddress.x, virtualAddress.y);
        }

        const float2 texCoord = (physicalPageCoord + offsetInPage + (mipFaceTexelOffset % PAGE_WIDTH)) / float2(poolWidth, poolHeight);
        const float3 result = float3(texCoord.x, texCoord.y, layer);

        return result;
    }
}

#endif
