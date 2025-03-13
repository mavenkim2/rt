#ifndef HIT_SHADERINTEROP_H_
#define HIT_SHADERINTEROP_H_

#ifdef __cplusplus
namespace rt
{
typedef unsigned int uint;
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
