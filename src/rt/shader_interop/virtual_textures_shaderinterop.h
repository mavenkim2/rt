#ifndef VIRTUAL_TEXTURES_SHADERINTEROP_H_
#define VIRTUAL_TEXTURES_SHADERINTEROP_H_

#include "hlsl_cpp_compat.h"

#ifdef __cplusplus
namespace rt
{
#endif

struct PageTableUpdatePushConstant
{
    uint numRequests;
};

struct PageTableUpdateRequest
{
    uint virtualPage;
    uint physicalPage;
};

#ifdef __cplusplus
}
#endif
#endif
