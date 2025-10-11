#ifndef RAY_SHADERINTEROP_H_
#define RAY_SHADERINTEROP_H_

#include "hlsl_cpp_compat.h"

#define PATH_TRACE_NUM_THREADS_X 8u
#define PATH_TRACE_NUM_THREADS_Y 8u

#define PATH_TRACE_TILE_WIDTH 16u
#define PATH_TRACE_LOG2_TILE_WIDTH 4u

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
    float filterIntegral;
};

#ifdef __cplusplus
}
#endif

#endif
