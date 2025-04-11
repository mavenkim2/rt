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
                         float2 uv, bool debug = false)
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

        uint physicalPageLayer = physicalPage >> (2 * log2PhysicalPoolNumPages);
        uint physicalPageInPool = physicalPage & (numPagesPerPool - 1u);
    
        uint physicalPageInPoolY = physicalPageInPool >> log2PhysicalPoolPageWidth;
        uint physicalPageInPoolX = physicalPageInPool & (pageWidthPerPool - 1);
    
        const uint texelWidthPerPool = texelWidthPerPage * pageWidthPerPool;

        uint2 pageStart = uint2(physicalPageInPoolX, physicalPageInPoolY) * texelWidthPerPage;
        uint texelOffsetX = (uv.x * numPages.x - pageOffsetX) * texelWidthPerPage;
        uint texelOffsetY = (uv.y * numPages.y - pageOffsetY) * texelWidthPerPage;
        pageStart += uint2(texelOffsetX + BORDER_SIZE, texelOffsetY + BORDER_SIZE);
        float2 physicalUv = pageStart / (float)texelWidthPerPool;

        if (0)
        {
            printf("physical uv: %f %f\n, physical page: %u", physicalUv.x, physicalUv.y, physicalPage);
        }
        return float3(physicalUv, physicalPageLayer);
    }
};
#endif
