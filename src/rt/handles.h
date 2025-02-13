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
    CatmullClark,
    Max,
};

string ConvertGeometryTypeToString(GeometryType type)
{
    switch (type)
    {
        case GeometryType::QuadMesh: return "quad";
        case GeometryType::TriangleMesh: return "tri";
        case GeometryType::Instance: return "inst";
        case GeometryType::CatmullClark: return "catclark";
        default: return "invalid";
    }
}

GeometryType ConvertStringIDToGeometryType(StringId id)
{
}

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
    MaterialHandle() : data(0x0fffffff) {}
    explicit MaterialHandle(u32 a) : data(a) {}
    MaterialHandle(MaterialTypes type, u32 index)
    {
        data = ((u32)type << 28) | (index & 0x0fffffff);
    }
    MaterialTypes GetType() const { return MaterialTypes(data >> 28); }
    u32 GetIndex() const { return data & 0x0fffffff; }
    __forceinline operator bool() { return data != 0x0fffffff; }
};

} // namespace rt
#endif
