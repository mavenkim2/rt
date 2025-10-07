#ifndef VIRTUAL_TEXTURES_SHADERINTEROP_H_
#define VIRTUAL_TEXTURES_SHADERINTEROP_H_

#include "hlsl_cpp_compat.h"
#include "bit_twiddling_shaderinterop.h"

#ifdef __cplusplus
namespace rt
{
#endif

#define PAGE_WIDTH 128
#define PAGE_SHIFT 7

#define MAX_COMPRESSED_LEVEL      6
#define MAX_LEVEL                 16
#define BASE_TEXEL_WIDTH_PER_PAGE 128

#define BLOCK_WIDTH      4
#define LOG2_BLOCK_WIDTH 2

#define PHYSICAL_POOL_NUM_PAGES_WIDE_BITS 7
#define PHYSICAL_POOL_NUM_PAGES_WIDE      (1u << PHYSICAL_POOL_NUM_PAGES_WIDE_BITS)

struct PageTableUpdatePushConstant
{
    uint numRequests;
    uint bindlessPageTableStartIndex;
};

struct PageTableUpdateRequest
{
    // uint2 virtualPage;
    //  uint faceID;
    //  uint mipLevel;

    uint hashIndex;
    uint4 packed;
};

struct GPUFaceData
{
    uint4 neighborFaces;
    uint2 faceOffset;
    uint2 log2Dim;
    uint rotate;
};

struct TextureHashTableEntry
{
    uint2 offset;
    uint mip;
    uint layer;
    uint bindlessIndex;

    uint textureIndex;
    uint tileIndex;
    uint faceID;
};

#ifdef __cplusplus
inline TextureHashTableEntry UnpackPageTableEntry(uint4 packedPageTableEntry)
#else
inline TextureHashTableEntry UnpackPageTableEntry(uint4 packedPageTableEntry)
#endif
{
    TextureHashTableEntry result;

    uint textureIndex  = BitFieldExtractU32(packedPageTableEntry.x, 16, 0);
    uint tileIndex     = BitFieldExtractU32(packedPageTableEntry.x, 16, 16);
    uint faceID        = BitFieldExtractU32(packedPageTableEntry.y, 28, 0);
    uint mipLevel      = BitFieldExtractU32(packedPageTableEntry.y, 28, 0);
    uint bindlessIndex = packedPageTableEntry.z;
    uint layerIndex    = BitFieldExtractU32(packedPageTableEntry.w, 12, 0);
    uint offsetX       = BitFieldExtractU32(packedPageTableEntry.w, 4, 12);
    uint offsetY       = BitFieldExtractU32(packedPageTableEntry.w, 4, 16);

    result.offset        = uint2(offsetX, offsetY);
    result.mip           = mipLevel;
    result.layer         = layerIndex;
    result.bindlessIndex = bindlessIndex;
    result.textureIndex  = textureIndex;
    result.tileIndex     = tileIndex;
    result.faceID        = faceID;

    return result;
}

inline uint PackPageTableEntry(uint pageLocationX, uint pageLocationY, uint offsetInPageX,
                               uint offsetInPageY, uint layer)
{
    uint packed       = 0;
    uint packedOffset = 0;
    packed            = BitFieldPackU32(packed, pageLocationX, packedOffset,
                                        PHYSICAL_POOL_NUM_PAGES_WIDE_BITS);
    packed            = BitFieldPackU32(packed, pageLocationY, packedOffset,
                                        PHYSICAL_POOL_NUM_PAGES_WIDE_BITS);
    packed            = BitFieldPackU32(packed, offsetInPageX, packedOffset, PAGE_SHIFT);
    packed            = BitFieldPackU32(packed, offsetInPageY, packedOffset, PAGE_SHIFT);
    packed            = BitFieldPackU32(packed, layer, packedOffset, 4);

    return packed;
}

inline uint2 PackFeedbackEntry(uint virtualPageX, uint virtualPageY, uint textureIndex,
                               uint mipLevel)
{
    uint2 result;
    result.x = (virtualPageY << 16) | virtualPageX;
    result.y = (mipLevel << 24) | textureIndex;
    return result;
}

#ifdef __cplusplus
}
#endif
#endif
