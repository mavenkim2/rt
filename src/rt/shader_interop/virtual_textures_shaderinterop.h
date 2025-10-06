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
    uint3 packed;
};

struct GPUFaceData
{
    uint4 neighborFaces;
    uint2 faceOffset;
    uint2 log2Dim;
    uint rotate;
};

#ifdef __cplusplus
inline uint4 UnpackPageTableEntry(uint packedPageTableEntry, uint &outLayer)
#else
inline uint4 UnpackPageTableEntry(uint packedPageTableEntry, out uint outLayer)
#endif
{
    uint pageX =
        BitFieldExtractU32(packedPageTableEntry, PHYSICAL_POOL_NUM_PAGES_WIDE_BITS, 0);
    uint pageY = BitFieldExtractU32(packedPageTableEntry, PHYSICAL_POOL_NUM_PAGES_WIDE_BITS,
                                    PHYSICAL_POOL_NUM_PAGES_WIDE_BITS);
    uint offsetInPageX = BitFieldExtractU32(packedPageTableEntry, PAGE_SHIFT,
                                            2 * PHYSICAL_POOL_NUM_PAGES_WIDE_BITS);
    uint offsetInPageY = BitFieldExtractU32(
        packedPageTableEntry, PAGE_SHIFT, 2 * PHYSICAL_POOL_NUM_PAGES_WIDE_BITS + PAGE_SHIFT);

    uint layer = BitFieldExtractU32(packedPageTableEntry, 4,
                                    2 * PHYSICAL_POOL_NUM_PAGES_WIDE_BITS + 2 * PAGE_SHIFT);
    outLayer   = layer;
    return uint4(pageX, pageY, offsetInPageX, offsetInPageY);
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
