#ifndef INTEGRATE_H
#define INTEGRATE_H

namespace rt
{
// lane width for integration
#define IntN 1
#if IntN == 1
typedef bool MaskF32;
#else
typedef LaneF32<IntN> MaskF32;
#endif
typedef LaneF32<IntN> LaneIF32;
typedef LaneU32<IntN> LaneIU32;
typedef Vec2<LaneIF32> Vec2IF32;
typedef Vec3<LaneIF32> Vec3IF32;
typedef Vec4<LaneIF32> Vec4IF32;

struct SurfaceInteraction
{
    Vec3IF32 p;
    Vec3IF32 n;
    struct
    {
        Vec3IF32 n;
    } shading;
    LaneIU32 lightIndices;
};

struct LightSample
{
    Vec3IF32 samplePoint;
    Vec3IF32 n;
    LaneIF32 pdf;
};

// NOTE: rectangle area light, invisible, not two sided
// quad:
// p1 ---- p0
// |        |
// |        |
// |        |
// p2 ---- p3
enum class LightType
{
    DeltaPosition,
    DeltaDirection,
    Area,
    Infinite,
};

bool IsDeltaLight(LightType type)
{
    return type == LightType::DeltaPosition || type == LightType::DeltaDirection;
}

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

    SampledSpectrum Le(const Vec3f &n, const Vec3f &w, const SampledWavelengths &lambda) const
    {
        if (Dot(n, w) < 0) return SampledSpectrum(0.f);
        return scale * Lemit->Sample(lambda);
    }
};

struct RayDifferential
{
    Vec3IF32 o;
    Vec3IF32 d;
    LaneIF32 t;
    Vec3IF32 rxOrigin, ryOrigin;
    Vec3IF32 rxDir, ryDir;
};

// RayDifferential ComputeRayDifferentials(const RayDifferential &ray)
// {
// }

} // namespace rt
#endif
