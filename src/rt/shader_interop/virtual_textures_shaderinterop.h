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
#define PHYSICAL_POOL_NUM_PAGES_WIDE      (1u << PHYSICAL_PAGE_NUM_PAGES_WIDE_BITS)

struct PageTableUpdatePushConstant
{
    uint numRequests;
};

struct PageTableUpdateRequest
{
    uint2 virtualPage;
    uint packed;
};

struct GPUFaceData
{
    uint4 neighborFaces;
    uint2 faceOffset;
    uint2 log2Dim;
    uint rotate;
};

inline uint4 UnpackPageTableEntry(uint packedPageTableEntry)
{
    uint pageX =
        BitFieldExtractU32(packedPageTableEntry, PHYSICAL_POOL_NUM_PAGES_WIDE_BITS, 0);
    uint pageY = BitFieldExtractU32(packedPageTableEntry, PHYSICAL_POOL_NUM_PAGES_WIDE_BITS,
                                    PHYSICAL_POOL_NUM_PAGES_WIDE);
    uint mip =
        BitFieldExtractU32(packedPageTableEntry, 4, 2 * PHYSICAL_POOL_NUM_PAGES_WIDE_BITS);
    uint layer =
        BitFieldExtractU32(packedPageTableEntry, 4, 2 * PHYSICAL_POOL_NUM_PAGES_WIDE_BITS + 4);
    return uint4(pageX, pageY, mip, layer);
}

inline uint PackPageTableEntry(uint pageLocationX, uint pageLocationY, uint mip, uint layer)
{
    uint packed       = 0;
    uint packedOffset = 0;
    packed            = BitFieldPackU32(packed, pageLocation.x, packedOffset,
                                        PHYSICAL_POOL_NUM_PAGES_WIDE_BITS);
    packed            = BitFieldPackU32(packed, pageLocation.y, packedOffset,
                                        PHYSICAL_POOL_NUM_PAGES_WIDE_BITS);
    packed            = BitFieldPackU32(packed, mip, packedOffset, 4);
    packed            = BitFieldPackU32(packed, layer, packedOffset, 4);
    return packed;
}

#ifdef __cplusplus
}
#endif
#endif
