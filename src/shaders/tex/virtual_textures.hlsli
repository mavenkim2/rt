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

        uint pageX = BitFieldExtractU32(packed, 7, 0);
        uint pageY = BitFieldExtractU32(packed, 7, 7);
        uint layer = BitFieldExtractU32(packed, 4, 14);
        const float2 physicalPageCoord = float2(pageX, pageY) * PAGE_WIDTH;

        const float2 mipFaceTexelOffset = faceTexelOffset / float(1u << mipLevel);

        const uint2 offsetInPage = (baseOffset >> mipLevel) & (PAGE_WIDTH - 1);

        const float2 texCoord = (physicalPageCoord + offsetInPage + (mipFaceTexelOffset % PAGE_WIDTH)) / float2(poolWidth, poolHeight);

        if (debug)
        {
            printf("virtual address: %u %u, phys page: %f %f, offset: %u %u, face texel offset: %f %f, mip: %u, in uv: %f %fout uv: %f %f, %u \n", 
                    virtualAddress.x, virtualAddress.y, physicalPageCoord.x, physicalPageCoord.y, 
                    offsetInPage.x, offsetInPage.y, mipFaceTexelOffset.x, mipFaceTexelOffset.y, mipLevel, 
                    uv.x, uv.y, texCoord.x, texCoord.y, layer);
        }

        const float3 result = float3(texCoord.x, texCoord.y, layer);

        return result;
    }
}

#endif
