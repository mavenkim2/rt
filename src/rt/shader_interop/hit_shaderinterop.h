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

enum class GPUMaterialType
{
    Diffuse,
    Dielectric,
};

struct GPUMaterial
{
    GPUMaterialType type;

    // Virtual texture info
    uint2 baseVirtualPage;
    uint textureIndex;
    uint minLog2Dim;
    uint numVirtualOffsetBits;
    uint numFaceDimBits;
    uint numFaceIDBits;
    uint faceDataOffset;

    // Shading info
    float eta;
};

#ifdef __cplusplus
}
#endif

#endif
