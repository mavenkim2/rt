#ifndef SHADER_DEBUG_INFO_H_
#define SHADER_DEBUG_INFO_H_

#include "hlsl_cpp_compat.h"

#ifdef __cplusplus
namespace rt
{
#endif

struct ShaderDebugInfo
{
    uint2 mousePos;
};

#ifdef __cplusplus
}
#endif
#endif
