#ifndef VIRTUAL_TEXTURES_SHADERINTEROP_H_
#define VIRTUAL_TEXTURES_SHADERINTEROP_H_

#include "hlsl_cpp_compat.h"

#ifdef __cplusplus
namespace rt
{
#endif

#define MAX_COMPRESSED_LEVEL      6
#define BASE_TEXEL_WIDTH_PER_PAGE 128

#define BLOCK_WIDTH      4
#define LOG2_BLOCK_WIDTH 2

struct PageTableUpdatePushConstant
{
    uint numRequests;
};

struct PageTableUpdateRequest
{
    uint faceIndex;
    uint virtualPage;
    uint physicalPage;
};

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
