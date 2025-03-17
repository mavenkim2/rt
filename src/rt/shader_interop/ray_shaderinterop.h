#ifndef RAY_SHADERINTEROP_H_
#define RAY_SHADERINTEROP_H_

#define PATH_TRACE_NUM_THREADS_X 8u 
#define PATH_TRACE_NUM_THREADS_Y 8u 

#ifdef __cplusplus
namespace rt
{
typedef unsigned int uint;
#endif

struct RayPushConstant
{
    uint envMap;
    uint frameNum;
    uint width;
    uint height;
};

// struct DenseGeometryHeader 
// {
//     uint3 anchor;
// };

#ifdef __cplusplus
}
#endif

#endif
