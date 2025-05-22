#ifndef VIRTUAL_TEXTURES_HLSLI_
#define VIRTUAL_TEXTURES_HLSLI_

#include "./../rt/shader_interop/virtual_textures_shaderinterop.h"

Texture2D<uint> pageTable : register(t5);
Texture2DArray physicalPages: register(t6);

namespace VirtualTexture
{
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
        const uint2 virtualAddress = ((baseOffset + (uint2)faceTexelOffset) >> mipLevel);
        const uint2 virtualPage = virtualAddress >> PAGE_SHIFT;
        return virtualPage;
    }

    static float3 GetPhysicalUV(uint2 baseOffset, float2 uv, uint2 texSize, uint mipLevel, bool debug = false)
    {
        uint poolWidth, poolHeight, poolNumLayers;
        physicalPages.GetDimensions(poolWidth, poolHeight, poolNumLayers);

        uint2 virtualPage = CalculateVirtualPage(baseOffset, uv, texSize, mipLevel);
        uint packed = pageTable.Load(float3(virtualPage.x, virtualPage.y, mipLevel));

#if 0
        while (packed == ~0u)
        {
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
            printf("mip: %u %u virtualPage: %u %u\n phys page: %u %u", mipLevel, newMipLevel, virtualPage.x, virtualPage.y, 
                    pageX, pageY);
        }

        const float2 faceTexelOffset = uv * float2(texSize);
        const float2 physicalPageCoord = float2(pageX, pageY) * PAGE_WIDTH;
        const float2 mipFaceTexelOffset = faceTexelOffset / float(1u << newMipLevel);
        const uint2 offsetInPage = (baseOffset >> newMipLevel) & (PAGE_WIDTH - 1);

        const float2 texCoord = (physicalPageCoord + offsetInPage + (mipFaceTexelOffset % PAGE_WIDTH)) / float2(poolWidth, poolHeight);
        const float3 result = float3(texCoord.x, texCoord.y, layer);

        return result;
    }
}

#endif
