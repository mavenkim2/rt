#ifndef HANDLES_H
#define HANDLES_H
namespace rt
{

// TODO: temporary
enum MaterialType
{
    MT_DielectricMaterial,
};

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

enum LightClass
{
    LightClass_Area,    // diffuse area light
    LightClass_Distant, // dirac delta direction
    LightClass_InfUnf,  // uniform infinite light
    LightClass_InfImg,  // environment map
    LightClass_Count,
};

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
