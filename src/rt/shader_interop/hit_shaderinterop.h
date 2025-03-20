#ifndef HIT_SHADERINTEROP_H_
#define HIT_SHADERINTEROP_H_

#include "hlsl_cpp_compat.h"

#ifdef __cplusplus
namespace rt
{
#endif

struct RTBindingData
{
    uint materialIndex;
};

struct GPUMaterial
{
    float eta;
};

#ifdef __cplusplus
}
#endif

#endif
