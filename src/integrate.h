#ifndef INTEGRATE_H
#define INTEGRATE_H

namespace rt
{
// lane width for integration
#define IntN 1
#if IntN == 1
typedef f32 LaneIF32;
#else
typedef LaneF32<IntN> LaneIF32;
#endif
typedef Vec2<LaneIF32> Vec2IF32;
typedef Vec3<LaneIF32> Vec3IF32;

template <u32 N>
struct SurfaceInteraction
{
    Vec3IF32 p;
    Vec3IF32 n;
    struct
    {
        Vec3IF32 n;
    } shading;
};

struct LightSample
{
    Vec3f samplePoint;
    f32 pdf;
};

// NOTE: rectangle area light, invisible, not two sided
// quad:
// p1 ---- p0
// |        |
// |        |
// |        |
// p2 ---- p3
struct DiffuseAreaLight
{
    Vec3f *p;
    f32 scale = 1.f;
    const AffineSpace *renderFromLight;
    const DenselySampledSpectrum *Lemit;
    static const f32 MinSphericalArea = 1e-4;
    f32 area;

    DiffuseAreaLight(Vec3f *p, f32 scale, DenselySampledSpectrum *Lemit) : p(p), scale(scale), Lemit(Lemit)
    {
        area = Length(Cross(p[1] - p[0], p[3] - p[0]));
    }

    SampledSpectrum L(const Vec3f &n, const Vec3f &w, const SampledWavelengths &lambda) const
    {
        if (Dot(n, w) < 0) return SampledSpectrum(0.f);
        return scale * Lemit->Sample(lambda);
    }
    void PDF_Li()
    {
    }
};

struct RayDifferential
{
    Vec3f rxOrigin, ryOrigin;
    Vec3f rxDir, ryDir;
};

RayDifferential ComputeRayDifferentials(const RayDifferential &ray)
{
}

} // namespace rt
#endif
