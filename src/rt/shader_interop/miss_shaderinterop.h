#ifndef MISS_SHADERINTEROP_H_
#define MISS_SHADERINTEROP_H_

#ifdef __cplusplus
namespace rt 
{
typedef unsigned int uint;
#endif

struct MissPushConstant 
{
    uint envMap;
    uint width;
    uint height;
};

#ifdef __cplusplus
}
#endif

#endif
