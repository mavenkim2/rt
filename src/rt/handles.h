#ifndef HANDLES_H
#define HANDLES_H
#include "template.h"
namespace rt
{

#define CREATE_ENUM_AND_TYPE_PACK(packName, enumName, ...)                                    \
    using packName = TypePack<COMMA_SEPARATED_LIST(__VA_ARGS__)>;                             \
    enum class enumName                                                                       \
    {                                                                                         \
        COMMA_SEPARATED_LIST(__VA_ARGS__),                                                    \
        Max,                                                                                  \
    };                                                                                        \
    ENUM_CLASS_FLAGS(enumName)

enum class GeometryType
{
    QuadMesh,
    TriangleMesh,
    Instance,
    Max,
};

template <typename BxDFShader, typename NormalShader>
struct Material2;

template <i32 K>
struct VecBase;

template <>
struct VecBase<1>
{
    using Type = LaneNF32;
};

template <>
struct VecBase<3>
{
    using Type = Vec3lfn;
};

template <i32 K>
using Veclfn = typename VecBase<K>::Type;

#if 0
struct NullShader;
struct ConstantTexture;
struct ConstantSpectrumTexture;
struct PtexTexture;
template <typename TextureShader>
struct BumpMap;
template <typename RflShader>
struct DiffuseMaterial;
template <typename RflShader, typename TrmShader>
struct DiffuseTransmissionMaterial;
template <typename RghShader, typename Spectrum>
struct DielectricMaterial;
template <typename RghShader, typename RflShader, typename AlbedoShader, typename Spectrum>
struct CoatedDiffuseMaterial;
struct MSDielectricMaterial;

// TODO: automate this :)
using BumpMapPtex                     = BumpMap<PtexTexture>;
using DiffuseMaterialPtex             = DiffuseMaterial<PtexTexture>;
using DiffuseTransmissionMaterialPtex = DiffuseTransmissionMaterial<PtexTexture, PtexTexture>;

// NOTE: isotropic roughness, constant ior
using DielectricMaterialConstant = DielectricMaterial<ConstantTexture, ConstantSpectrum>;

// Material types
using DiffuseMaterialBumpMapPtex = Material2<DiffuseMaterialPtex, BumpMapPtex>;
using DiffuseMaterialBase        = Material2<DiffuseMaterialPtex, NullShader>;
using DiffuseTransmissionMaterialBumpMapPtex =
    Material2<DiffuseTransmissionMaterialPtex, BumpMapPtex>;
using DielectricMaterialBumpMapPtex = Material2<DielectricMaterialConstant, BumpMapPtex>;

using CoatedDiffuseMaterialPtex =
    CoatedDiffuseMaterial<ConstantTexture, PtexTexture, ConstantSpectrumTexture,
                          ConstantSpectrum>;
using CoatedDiffuseMaterialBase =
    CoatedDiffuseMaterial<ConstantTexture, ConstantSpectrumTexture, ConstantTexture,
                          ConstantSpectrum>;

using CoatedDiffuseMaterial1 = Material2<CoatedDiffuseMaterialPtex, NullShader>;
using CoatedDiffuseMaterial2 = Material2<CoatedDiffuseMaterialBase, NullShader>;
using DielectricMaterialBase = Material2<DielectricMaterialConstant, NullShader>;

using MSDielectricMaterial1 = Material2<MSDielectricMaterial, NullShader>;
CREATE_ENUM_AND_TYPE_PACK(MaterialTypes, MaterialType, DielectricMaterialBase,
        CoatedDiffuseMaterial1, CoatedDiffuseMaterial2,
        DiffuseMaterialBase, MSDielectricMaterial1);
#endif

struct VolumeHandle
{
    u32 index;
    VolumeHandle() {}
    VolumeHandle(u32 index) : index(index) {}
};

struct DiffuseAreaLight;
struct DistantLight;
struct UniformInfiniteLight;
struct ImageInfiniteLight;
CREATE_ENUM_AND_TYPE_PACK(LightTypes, LightClass, DiffuseAreaLight, DistantLight,
                          UniformInfiniteLight, ImageInfiniteLight);
using InfiniteLightTypes = TypePack<UniformInfiniteLight, ImageInfiniteLight>;

// enum LightClass
// {
//     LightClass_Area,    // diffuse area light
//     LightClass_Distant, // dirac delta direction
//     LightClass_InfUnf,  // uniform infinite light
//     LightClass_InfImg,  // environment map
//     LightClass_Count,
// };

struct LightHandle
{
    u32 data;
    LightHandle() : data(0xffffffff) {}
    LightHandle(u32 a) : data(a) {}
    LightHandle(LightClass type, u32 index) { data = (type << 28) | (index & 0x0fffffff); }
    LightClass GetType() const { return LightClass(data >> 28); }
    u32 GetIndex() const { return data & 0x0fffffff; }
    __forceinline operator bool() { return data != 0xffffffff; }
};

enum class MaterialTypes
{
    Interface,
    Diffuse,
    DiffuseTransmission,
    CoatedDiffuse,
    Dielectric,
    Max,
};

struct MaterialHandle
{
    static_assert((u32)MaterialTypes::Max < 16, "too many material types");
    u32 data;
    MaterialHandle() : data(0xffffffff) {}
    explicit MaterialHandle(u32 a) : data(a) {}
    MaterialHandle(MaterialTypes type, u32 index)
    {
        data = ((u32)type << 28) | (index & 0x0fffffff);
    }
    MaterialTypes GetType() const { return MaterialTypes(data >> 28); }
    u32 GetIndex() const { return data & 0x0fffffff; }
    __forceinline operator bool() { return data != 0xffffffff; }
};

} // namespace rt
#endif
