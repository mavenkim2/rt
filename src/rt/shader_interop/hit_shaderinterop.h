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
    Disney,
};

struct GPUMaterial
{
    GPUMaterialType type;

    // Virtual texture info
    int textureIndex;
    uint minLog2Dim;
    uint numVirtualOffsetBits;
    uint numFaceDimBits;
    uint numFaceIDBits;
    uint faceDataOffset;

    float diffTrans;
    float4 baseColor;
    float specTrans;
    float clearcoatGloss;
    float3 scatterDistance;
    float clearcoat;
    float specularTint;
    float ior;
    float metallic;
    float flatness;
    float sheen;
    float sheenTint;
    float anisotropic;
    float alpha;
    float roughness;
    bool thin;
};

#ifdef __cplusplus
}
#endif

#endif
