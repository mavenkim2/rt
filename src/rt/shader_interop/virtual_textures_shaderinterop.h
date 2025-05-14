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

#ifdef __cplusplus
}
#endif
#endif
