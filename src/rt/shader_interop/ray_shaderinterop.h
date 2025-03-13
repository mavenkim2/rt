#ifndef RAY_SHADERINTEROP_H_
#define RAY_SHADERINTEROP_H_

#ifdef __cplusplus
namespace rt
{
typedef unsigned int uint;
#endif

struct RayPushConstant
{
    uint envMap;
    uint bindingTable;
    uint width;
    uint height;
};

#ifdef __cplusplus
}
#endif

#endif
