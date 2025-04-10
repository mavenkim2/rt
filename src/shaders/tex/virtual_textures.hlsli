#ifndef VIRTUAL_TEXTURES_HLSLI_
#define VIRTUAL_TEXTURES_HLSLI_

#define PAGE_WIDTH 128
#define BORDER_SIZE 4

#include "../../rt/shader_interop/hit_shaderinterop.h"

StructuredBuffer<uint> pageTable : register(t5);
Texture2DArray physicalPages: register(t6);

struct VirtualTexture 
{
    uint pageWidthPerPool;
    uint texelWidthPerPage;

    float3 GetPhysicalUV(StructuredBuffer<uint> pageTable, 
                         GPUMaterial material,
                         uint3 pageInformation,
                         float2 uv)
    {
        const uint pageBorder = 4;
        const uint2 faceSize = (1u << pageInformation.yz);
        const uint2 numPages = max(faceSize >> 7, 1);

        uint pageOffsetX = floor(uv.x * numPages.x);
        uint pageOffsetY = floor(uv.y * numPages.y);
        uint pageOffset = floor(uv.y * numPages.y * numPages.x) + pageOffsetX;

        uint virtualPage = material.pageOffset + pageInformation.x;
        uint physicalPage = pageTable[virtualPage];
    
        uint log2PhysicalPoolPageWidth = firstbithigh(pageWidthPerPool);
        uint numPagesPerPool = pageWidthPerPool * pageWidthPerPool;
        uint log2PhysicalPoolNumPages = firstbithigh(numPagesPerPool);

        uint physicalPageLayer = physicalPage >> log2PhysicalPoolNumPages;
        uint physicalPageInPool = physicalPage & (numPagesPerPool - 1u);
    
        uint physicalPageInPoolY = physicalPageInPool >> log2PhysicalPoolPageWidth;
        uint physicalPageInPoolX = physicalPageInPool & (pageWidthPerPool - 1);
    
        const float scaleST = (float)(PAGE_WIDTH + 2 * BORDER_SIZE) / texelWidthPerPage;

        float fracX = (uv.x - (float)pageOffsetX / numPages.x) * numPages.x;
        float fracY = (uv.y - (float)pageOffsetY / numPages.y) * numPages.y;
    
        float2 physicalUv = scaleST * uint2(physicalPageInPoolX, physicalPageInPoolY);
        physicalUv += float2(fracX, fracY) * scaleST;
        return float3(physicalUv, physicalPageLayer);
    }
};
#endif
