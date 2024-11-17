#ifndef LIGHTS_H
#define LIGHTS_H

namespace rt
{
// NOTE: rectangle area light, invisible, not two sided
// quad:
// p1 ---- p0
// |        |
// |        |
// |        |
// p2 ---- p3
enum class LightType : u32
{
    DeltaPosition,
    DeltaDirection,
    Area,
    Infinite,
};
enum LightClass
{
    LightClass_Area,    // diffuse area light
    LightClass_InfImg,  // environment map
    LightClass_Distant, // dirac delta direction
    LightClass_Count,
};

struct LightSample
{
    SampledSpectrum L;
    Vec3IF32 samplePoint;
    Vec3IF32 n;
    LaneIF32 pdf;
    // TODO: simd
    LightType lightType;

    LightSample() {}
};

struct LightHandle
{
    u32 data;
    LightHandle() {}
    LightHandle(LightClass type, u32 index)
    {
        data = (type << 28) | (index & 0x0fffffff);
    }
    LightClass GetType() const
    {
        return LightClass(data >> 28);
    }
    u32 GetIndex() const
    {
        return data & 0x0fffffff;
    }
    __forceinline operator bool() { return data != 0; }
};

bool IsDeltaLight(LightType type)
{
    return type == LightType::DeltaPosition || type == LightType::DeltaDirection;
}

struct DistantLight
{
    Vec3f d;
    const DenselySampledSpectrum *Lemit;
    f32 sceneRadius;
    f32 scale;
};

struct DiffuseAreaLight
{
    Vec3f *p;
    f32 scale = 1.f;
    const AffineSpace *renderFromLight;
    const DenselySampledSpectrum *Lemit;
    static constexpr f32 MinSphericalArea = 1e-4;
    f32 area;

    DiffuseAreaLight(Vec3f *p, f32 scale, DenselySampledSpectrum *Lemit) : p(p), scale(scale), Lemit(Lemit)
    {
        area = Length(Cross(p[1] - p[0], p[3] - p[0]));
    }

    LightType GetLightType() const { return LightType::Area; }
};
} // namespace rt
#endif
