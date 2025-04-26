#ifndef VIRTUAL_TEXTURES_HLSLI_
#define VIRTUAL_TEXTURES_HLSLI_

#include "./../rt/shader_interop/virtual_textures_shaderinterop.h"

Texture1D<uint2> pageTable : register(t5);
Texture2DArray physicalPages: register(t6);

struct VirtualTexture 
{
    uint pageWidthPerPool;

    float3 GetPhysicalUV(Texture1D<uint2> pageTable,
                         uint basePageOffset,
                         uint faceID,
                         float2 uv, uint mipLevel = 0, bool debug = false)
    {
        // page information.x contains the faceID
        uint faceIndex = basePageOffset + faceID;
        uint2 physicalPageInfo = pageTable.Load(faceIndex);

        uint x = physicalPageInfo.x & ((1u << 15u) - 1u);
        uint y = physicalPageInfo.x >> 15u;
        uint layerIndex = physicalPageInfo.x >> 30u;

        uint basePhysicalLevel = BitFieldExtractU32(physicalPageInfo.y, 4, 8);

        int log2Width = BitFieldExtractU32(physicalPageInfo.y, 4, 0);
        int log2Height = BitFieldExtractU32(physicalPageInfo.y, 4, 4);
        log2Width = max(1, log2Width - basePhysicalLevel);
        log2Height = max(1, log2Height - basePhysicalLevel);

        uint2 offset = uint2(x, y);
        for (int levelIndex = basePhysicalLevel; levelIndex < mipLevel; levelIndex++)
        {
            offset += CalculateFaceSize(log2Width, log2Height);
        }

        uint width, height, layers;
        physicalPages.GetDimensions(width, height, layers);
        float3 physicalUv = float3((float)offset.x / width, (float)offset.y / height, layerIndex);
        return physicalUv;
    }

#if 0
    float3 GetPhysicalUV(Texture1D<uint> pageTable,
                         uint basePageOffset,
                         uint3 pageInformation,
                         float2 uv, uint mipLevel = 0, bool debug = false)
    {
        const uint pageBorder = GetBorderSize(mipLevel);
        const uint2 faceSize = (1u << pageInformation.yz);
        const uint2 numPages = max((int2)pageInformation.yz - 7, 1);

        uint pageOffsetX = floor(uv.x * numPages.x);
        uint pageOffsetY = floor(uv.y * numPages.y);
        uint pageOffset = pageOffsetY * numPages.x + pageOffsetX;

        uint mipShift = 2 * mipLevel;
        uint virtualPage = basePageOffset + pageInformation.x + pageOffset;
        uint physicalPage = pageTable.Load(int2(virtualPage, mipLevel));
    
        uint log2PhysicalPoolPageWidth = firstbithigh(pageWidthPerPool);
        uint numPagesPerPool = pageWidthPerPool * pageWidthPerPool;
        uint log2PhysicalPoolNumPages = firstbithigh(numPagesPerPool);

        uint physicalPageLayer = physicalPage >> log2PhysicalPoolNumPages;
        uint physicalPageInPool = physicalPage & (numPagesPerPool - 1u);
    
        uint physicalPageInPoolY = physicalPageInPool >> log2PhysicalPoolPageWidth;
        uint physicalPageInPoolX = physicalPageInPool & (pageWidthPerPool - 1);
    
        const uint texelWidthPerPage = GetTileTexelWidth(mipLevel);
        const uint texelWidthPerPool = texelWidthPerPage * pageWidthPerPool;
        const uint subTileDim = texelWidthPerPage >> mipLevel;

        uint2 pageStart = uint2(physicalPageInPoolX, physicalPageInPoolY) * texelWidthPerPage;

        const uint baseTexelWidthPerPage = max((uint)BASE_TEXEL_WIDTH_PER_PAGE >> mipLevel, 1);
        uint texelOffsetX = frac(uv.x * numPages.x) * min(baseTexelWidthPerPage, faceSize.x);
        uint texelOffsetY = frac(uv.y * numPages.y) * min(baseTexelWidthPerPage, faceSize.y);
        pageStart += uint2(texelOffsetX, texelOffsetY) + pageBorder;

        // Offset into subtile
        uint offsetInPage = virtualPage & ((1u << mipShift) - 1u);
        uint mipDimMask = ((1u << mipLevel) - 1u);
        uint offsetInPageX = offsetInPage & mipDimMask;
        uint offsetInPageY = (offsetInPage >> mipLevel) & mipDimMask;

        pageStart += uint2(offsetInPageX * subTileDim, offsetInPageY * subTileDim);
        float2 physicalUv = pageStart / (float)texelWidthPerPool;

        if (0)
        {
            printf("physical uv: %f %f\nphysical page: %u", physicalUv.x, physicalUv.y, physicalPage);
        }
        return float3(physicalUv, physicalPageLayer);
    }
#endif
};
#endif
