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
    int faceOffset;

    // uint numVirtualOffsetBits;
    // uint numFaceDimBits;
    // uint numFaceIDBits;
    // uint faceDataOffset;

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

enum class GPULightType
{
    Area,
    Directional,
    Envmap,
};

// why this way?
// 1. if i want to switch to per type arrays, i feel like this would be the least painful way
// 2. it lets me just have every light/medium in one array

struct GPULight
{
    GPULightType lightType;

    float3x4 transform;

    // Constant or Image
    float3 color;
    int bindlessIndex;

    // Area light info
    float2 dim;

    // Directional light info
    float3 dir;
};

enum class GPUMediumType
{
    Homogeneous,
    Nanovdb,
};

struct GPUMedium
{
    GPUMediumType mediumType;

    // Nanovdb parameters
    int bindlessOctreeIndex;
    int bindlessNanovdbBufferIndex;
};

#ifdef __cplusplus
}
#endif

#endif
