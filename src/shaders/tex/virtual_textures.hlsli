#ifndef VIRTUAL_TEXTURES_HLSLI_
#define VIRTUAL_TEXTURES_HLSLI_

struct VirtualTexture 
{
    uint physicalPoolPageWidth;

    void Something(StructuredBuffer<uint> pageTable, Texture2DArray physicalTextures, GPUMaterial material, uint faceOffset, uint2 numPages, float2 uv)
    {
        uint pageOffsetX = floor(uv.x * numPages.x);
        uint pageOffsetY = floor(uv.y * numPages.y);
        uint pageOffset = floor(uv.y * numPages.y * numPages.x) + pageOffsetX;
    
        uint virtualPage = material.baseVirtualAddress + faceOffset + pageOffset;
        uint physicalPage = pageTable[virtualPage];
    
        uint log2PhysicalPoolPageWidth = firstbithigh(physicalPoolPageWidth);
        uint numPagesPerPool = physicalPoolPageWidth * physicalPoolPageWidth;
        uint log2PhysicalPoolNumPages = firstbithigh(numPagesPerPool);

        uint physicalPageLayer = physicalPage >> log2PhysicalPoolNumPages;
        uint physicalPageInPool = physicalPage & (numPagesPerPool - 1u);
    
        uint physicalPageInPoolY = physicalPageInPool >> log2PhysicalPoolPageWidth;
        uint physicalPageInPoolX = physicalPageInPool & (physicalPoolPageWidth - 1);
    
        const float scaleST = (float)pageWidth / poolWidth;
        float fracX = (uv.x - (float)pageOffsetX / numPages.x) * numPages.x;
        float fracY = (uv.y - (float)pageOffsetY / numPages.y) * numPages.y;
    
        float2 physicalUv = scaleST * uint2(physicalPageInPoolX, physicalPageInPoolY);
        physicalUv += float2(fracX, fracY) * scaleST;
    }
};

void Func() 
{
    // constants
    const int pageWidth = 128; // 128 x 128 texels per page
    const int pageBorder = 4; // 4-texel inset border
    const int pageWidthWithBorder = 128 + 2 * pageBorder;

    const int physPagesWide = ?;//32; // 4096 x 4096 texels per physical texture
    const int virtPagesWide = 1024; // 120K x 120K texels per virtual texture

    const float pageFracScale = ( pageWidth + 2.0f * pageBorder ) / ( pageWidth * physPagesWide );

    unsigned int physX, physY; // coordinates of the physical page (in whole pages)
    unsighed int physMipLevel; // mip level of the virtual page stored in the physical page
    unsigned int virtX, virtY; // coordinates of the virtual page (in whole pages)

    float physTexelsWide = physPagesWide * pageWidth;
    float virtLevelPagesWide = virtPagesWide >> physMipLevel;

    // The scale and bias that can be used to translate a virtual address to a physical address.
    float scaleST = virtLevelPagesWide * pageWidth / physTexelsWide;
    float biasS = ( physX * pageWidth + pageBorder ) / physTexelsWide - scaleST * virtX / virtLevelPagesWide;
    float biasT = ( physY * pageWidth + pageBorder ) / physTexelsWide - scaleST * virtY / virtLevelPagesWide;
    // Scale factor applied to the virtual texture coordinate derivatives
    // used for anisotropic texture lookups.
    float derivativeScale = pageFracScale * virtLevelPagesWide;
    // Physical page coordinates converted to 8-bit values that can be used directly in the fragment
    // program to lookup the scale and bias from a mapping texture with one texel per physical page.
    unsigned char texelPhysX = (unsigned char)( ( physX / ( physPagesWide - 1.0f ) ) * 255.0f + 0.01f );
    unsigned char texelPhysY = (unsigned char)( ( physY / ( physPagesWide - 1.0f ) ) * 255.0f + 0.01f );
}
