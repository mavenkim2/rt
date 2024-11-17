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
    LightClass_Distant, // dirac delta direction
    LightClass_InfUnf,  // uniform infinite light
    LightClass_InfImg,  // environment map
    LightClass_Count,
};

struct LightSample
{
    SampledSpectrum L;
    Vec3IF32 samplePoint;
    Vec3IF32 wi;
    LaneIF32 pdf;
    // TODO: simd
    LightType lightType;

    LightSample() {}
    LightSample(SampledSpectrum L, Vec3IF32 samplePoint, Vec3IF32 wi, LaneIF32 pdf, LightType lightType)
        : L(L), samplePoint(samplePoint), wi(wi), pdf(pdf), lightType(lightType) {}
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

// clang-format off
#define SAMPLE_LI_HEADER() static LightSample SAMPLE_LI_BODY
#define SAMPLE_LI(type)    LightSample type##::SAMPLE_LI_BODY
#define SAMPLE_LI_BODY                                                                         \
    SampleLi(const Scene2 *scene, const LaneIU32 lightIndices, const SurfaceInteraction &intr, \
             const SampledWavelengths &lambda, const Vec2IF32 &u, bool allowIncompletePDF)

#define PDF_LI_HEADER() static LaneIF32 PDF_LI_BODY
#define PDF_LI(type)    LaneIF32 type##::PDF_LI_BODY 
#define PDF_LI_BODY \
    PDF_Li(const Scene2 *scene, const LaneIU32 lightIndices, \
           const Vec3IF32 &prevIntrP, const SurfaceInteraction &intr, bool allowIncompletePDF)

#define LE_HEADER(type) static SampledSpectrum LE_BODY(type)
#define LE(type)    SampledSpectrum type##::LE_BODY(type)
#define LE_BODY(type) Le(type *light, const Vec3f &n, const Vec3f &w, const SampledWavelengths &lambda)

// clang-format on

#define LightFunctions(type) \
    SAMPLE_LI_HEADER();      \
    PDF_LI_HEADER();         \
    LE_HEADER(type);

#define LightFunctionsDirac(type)    \
    SAMPLE_LI_HEADER();              \
    PDF_LI_HEADER()                  \
    {                                \
        return 0.f;                  \
    }                                \
    LE_HEADER(type)                  \
    {                                \
        return SampledSpectrum(0.f); \
    }

const DenselySampledSpectrum *LookupSpectrum(Spectrum s)
{
    return 0;
}

struct DiffuseAreaLight
{
    Vec3f *p;
    f32 scale = 1.f;
    const AffineSpace *renderFromLight;
    const DenselySampledSpectrum *Lemit;
    static constexpr f32 MinSphericalArea = 1e-4;
    f32 area;

    DiffuseAreaLight(Vec3f *p, f32 scale, Spectrum Lemit) : p(p), scale(scale), Lemit(LookupSpectrum(Lemit))
    {
        area = Length(Cross(p[1] - p[0], p[3] - p[0]));
    }

    LightType GetLightType() const { return LightType::Area; }
    LightFunctions(DiffuseAreaLight);
};

// TODO: loop over all of these after the scene is fullly instantiated and add the scene radius
struct DistantLight
{
    Vec3f d;
    const DenselySampledSpectrum *Lemit;
    f32 sceneRadius;
    f32 scale;

    DistantLight(Vec3f d, Spectrum *Lemit, f32 scale = 1.f) : d(d), Lemit(LookupSpectrum(Lemit)), scale(scale) {}
    LightFunctionsDirac(DistantLight);
};

// TODO: render from light?
struct UniformInfiniteLight
{
    const DenselySampledSpectrum *Lemit;
    f32 scale;
    f32 sceneRadius;

    UniformInfiniteLight(Spectrum *Lemit, f32 scale = 1.f) : Lemit(LookupSpectrum(Lemit)), scale(scale) {}
    LightFunctions(UniformInfiniteLight);
};

struct ImageInfiniteLight
{
    // Image image;
    const RGBColorSpace *imageColorSpace;
    f32 scale;
    f32 sceneRadius;
    // PiecewiseConstant2D distribution;
    // PiecewiseConstant2D compensatedDistribution;
    LightFunctions(ImageInfiniteLight);
};

} // namespace rt
#endif
