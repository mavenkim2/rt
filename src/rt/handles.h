#ifndef HANDLES_H
#define HANDLES_H
#include "template.h"
namespace rt
{

#define CREATE_ENUM_AND_TYPE_PACK(packName, enumName, ...)                                    \
    using packName = TypePack<COMMA_SEPARATED_LIST(__VA_ARGS__)>;                             \
    enum class enumName                                                                       \
    {                                                                                         \
        COMMA_SEPARATED_LIST(__VA_ARGS__)                                                     \
    };                                                                                        \
    ENUM_CLASS_FLAGS(enumName)

#define COMMA_SEPARATED_LIST(...)                                                             \
    COMMA_SEPARATED_LIST_HELPER(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)
#define COMMA_SEPARATED_LIST_HELPER(x, ...) EXPAND(CONCAT(RECURSE__, x)(EXPAND, __VA_ARGS__))

#define COUNT_ARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15,     \
                        _16, _17, _18, _19, _20, N, ...)                                      \
    N
#define COUNT_ARGS(...)                                                                       \
    EXPAND(COUNT_ARGS_IMPL(__VA_ARGS__, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7,  \
                           6, 5, 4, 3, 2, 1))

struct QuadMesh;
struct TriangleMesh;
struct Instance;

CREATE_ENUM_AND_TYPE_PACK(PrimitiveTypes, GeometryType, QuadMesh, TriangleMesh, Instance);

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

struct NullShader;
template <i32 nc>
struct ConstantTexture;
template <i32 nc>
struct PtexTexture;
template <typename TextureType, typename RGBSpectrum>
struct ImageTextureShader;
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

// TODO: automate this :)
template <i32 K>
using PtexShader = ImageTextureShader<PtexTexture<K>, RGBAlbedoSpectrum>;

using BumpMapPtex         = BumpMap<PtexShader<1>>;
using DiffuseMaterialPtex = DiffuseMaterial<PtexShader<3>>;
using DiffuseTransmissionMaterialPtex =
    DiffuseTransmissionMaterial<PtexShader<3>, PtexShader<3>>;

// NOTE: isotropic roughness, constant ior
using DielectricMaterialConstant = DielectricMaterial<ConstantTexture<1>, ConstantSpectrum>;

// Material types
using DiffuseMaterialBumpMapPtex = Material2<DiffuseMaterialPtex, BumpMapPtex>;
using DiffuseTransmissionMaterialBumpMapPtex =
    Material2<DiffuseTransmissionMaterialPtex, BumpMapPtex>;
using DielectricMaterialBumpMapPtex = Material2<DielectricMaterialConstant, BumpMapPtex>;

using CoatedDiffuseMaterialPtex = CoatedDiffuseMaterial<ConstantTexture<1>, PtexShader<3>,
                                                        ConstantTexture<1>, ConstantSpectrum>;

using CoatedDiffuseMaterial1 = Material2<CoatedDiffuseMaterialPtex, NullShader>;
using DielectricMaterialBase = Material2<DielectricMaterialConstant, NullShader>;

CREATE_ENUM_AND_TYPE_PACK(MaterialTypes, MaterialType, DielectricMaterialBase,
                          CoatedDiffuseMaterial1);

struct MaterialHandle
{
    u32 data;
    MaterialHandle() : data(0xffffffff) {}
    MaterialHandle(u32 data) : data(data) {}

    MaterialHandle(MaterialType type, u32 index)
    {
        data = (type << 28) | (index & 0x0fffffff);
    }
    static MaterialType GetType(u32 in) { return MaterialType(in >> 28); }
    MaterialType GetType() const { return GetType(data); }
    static u32 GetIndex(u32 in) { return in & 0x0fffffff; }
    u32 GetIndex() const { return GetIndex(data); }
    __forceinline operator bool() { return data != 0xffffffff; }
};

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
} // namespace rt
#endif
