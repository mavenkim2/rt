#ifndef RAY_SHADERINTEROP_H_
#define RAY_SHADERINTEROP_H_

#include "hlsl_cpp_compat.h"

#define PATH_TRACE_NUM_THREADS_X 8u
#define PATH_TRACE_NUM_THREADS_Y 8u

#ifdef __cplusplus
namespace rt
{
#endif

struct RayPushConstant
{
    uint envMap;
    uint frameNum;
    uint width;
    uint height;
};

static const uint blockShift   = 7;
static const uint triangleMask = 0x7f;

// struct DenseGeometryHeader
// {
//     uint3 anchor;
// };

#ifdef __cplusplus
}
#endif

#endif
