#ifndef VIRTUAL_TEXTURES_SHADERINTEROP_H_
#define VIRTUAL_TEXTURES_SHADERINTEROP_H_

#include "hlsl_cpp_compat.h"

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

struct PageTableUpdatePushConstant
{
    uint numRequests;
};

struct PageTableUpdateRequest
{
    uint2 virtualPage;
    uint packed;
};

inline uint GetBorderSize(int log2Width, int log2Height)
{
#ifdef __cplusplus
    return Min(log2Width, log2Height) < 2 ? 1 : 4;
#else
    return min(log2Width, log2Height) < 2 ? 1 : 4;
#endif
}

inline int2 CalculateFaceSize(int log2Width, int log2Height)
{
    int borderSize = GetBorderSize(log2Width, log2Height);
    int2 allocationSize =
        int2((1u << log2Width) + 2 * borderSize, (1u << log2Height) + 2 * borderSize);
    return allocationSize;
}

inline uint GetBorderSize(uint levelIndex)
{
    return levelIndex < MAX_COMPRESSED_LEVEL ? 4 : 1;
}

inline uint GetTileTexelWidth(uint levelIndex)
{
    uint mipBorderSize = GetBorderSize(levelIndex);
    return BASE_TEXEL_WIDTH_PER_PAGE + ((2 * mipBorderSize) << levelIndex);
}

#ifdef __cplusplus
}
#endif
#endif
