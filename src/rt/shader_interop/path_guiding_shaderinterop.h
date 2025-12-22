#ifndef PATH_GUIDING_SHADERINTEROP_H_
#define PATH_GUIDING_SHADERINTEROP_H_

#include "hlsl_cpp_compat.h"

#ifdef __cplusplus
namespace rt
{
#endif

#define PATH_GUIDING_GROUP_SIZE 32
#define MAX_COMPONENTS          32

struct VMM
{
    float kappas[MAX_COMPONENTS];
    float3 directions[MAX_COMPONENTS];
    float weights[MAX_COMPONENTS];

    uint numComponents;
};

struct PathGuidingSample
{
    float3 pos;
    float3 dir;
    float3 radiance;
    float pdf;
    float weight;

    uint vmmIndex;
};

#ifdef __cplusplus
}
#endif

#endif
