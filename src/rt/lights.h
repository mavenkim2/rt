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

struct LightSample
{
    SampledSpectrum L;
    Vec3lfn samplePoint;
    Vec3lfn wi;
    LaneNF32 pdf;
    // TODO: simd
    LightType lightType;

    LightSample() {}
    LightSample(SampledSpectrum L, Vec3lfn samplePoint, Vec3lfn wi, LaneNF32 pdf,
                LightType lightType)
        : L(L), samplePoint(samplePoint), wi(wi), pdf(pdf), lightType(lightType)
    {
    }
};

bool IsDeltaLight(LightType type)
{
    return type == LightType::DeltaPosition || type == LightType::DeltaDirection;
}

struct Scene;

// clang-format off
#define SAMPLE_LI_HEADER() static LightSample SAMPLE_LI_BODY
#define SAMPLE_LI(type)    LightSample type::SAMPLE_LI_BODY
#define SAMPLE_LI_BODY                                                                         \
    SampleLi(const Scene *scene, const LaneNU32 lightIndices, const SurfaceInteractionsN &intr, \
             const SampledWavelengths &lambda, const Vec2lfn &u, bool allowIncompletePDF)

#define PDF_LI_HEADER() static LaneNF32 PDF_LI_BODY
#define PDF_LI(type)    LaneNF32 type::PDF_LI_BODY 
#define PDF_LI_BODY \
    PDF_Li(const Scene *scene, const LaneNU32 lightIndices, \
           const Vec3lfn &prevIntrP, const SurfaceInteractionsN &intr, bool allowIncompletePDF)

#define PDF_LI_INF_HEADER(type) static LaneNF32 PDF_LI_INF_BODY(type)
#define PDF_LI_INF(type)    LaneNF32 type::PDF_LI_INF_BODY(type)
#define PDF_LI_INF_BODY(type) \
    PDF_Li(type *light, Vec3f &w, bool allowIncompletePDF)

#define LE_HEADER(type) static SampledSpectrum LE_BODY(type)
#define LE(type)    SampledSpectrum type::LE_BODY(type)
#define LE_BODY(type) Le(const type *light, const Vec3f &n, const Vec3f &w, const SampledWavelengths &lambda)

#define LE_INF_HEADER(type) static SampledSpectrum LE_INF_BODY(type)
#define LE_INF(type)    SampledSpectrum type::LE_INF_BODY(type)
#define LE_INF_BODY(type) Le(type *light, const Vec3f &w, const SampledWavelengths &lambda)

// clang-format on

#define LightFunctions(type)                                                                  \
    SAMPLE_LI_HEADER();                                                                       \
    PDF_LI_HEADER();                                                                          \
    LE_HEADER(type);

#define LightFunctionsDirac(type)                                                             \
    SAMPLE_LI_HEADER();                                                                       \
    PDF_LI_HEADER() { return 0.f; }                                                           \
    LE_HEADER(type) { return SampledSpectrum(0.f); }

#define LightFunctionsInf(type)                                                               \
    SAMPLE_LI_HEADER();                                                                       \
    PDF_LI_INF_HEADER(type);                                                                  \
    LE_INF_HEADER(type);

const DenselySampledSpectrum *LookupSpectrum(Spectrum s) { return 0; }

struct DiffuseAreaLight
{
    Vec3f *p;
    f32 scale = 1.f;
    const AffineSpace *renderFromLight;
    const DenselySampledSpectrum *Lemit;
    static constexpr f32 MinSphericalArea = 1e-4;
    f32 area;

    DiffuseAreaLight(Vec3f *p, f32 scale, Spectrum Lemit)
        : p(p), scale(scale), Lemit(LookupSpectrum(Lemit))
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

    DistantLight(Vec3f d, Spectrum Lemit, f32 scale = 1.f)
        : d(d), Lemit(LookupSpectrum(Lemit)), scale(scale)
    {
    }
    LightFunctionsDirac(DistantLight);
};

// TODO: render from light?
struct UniformInfiniteLight
{
    const DenselySampledSpectrum *Lemit;
    f32 scale;
    f32 sceneRadius;

    UniformInfiniteLight(Spectrum Lemit, f32 scale = 1.f)
        : Lemit(LookupSpectrum(Lemit)), scale(scale)
    {
    }
    LightFunctionsInf(UniformInfiniteLight);
};

struct PiecewiseConstant1D
{
    f32 *cdf;
    const f32 *func;
    u32 num;
    f32 funcInt;
    f32 minD, maxD;
    PiecewiseConstant1D() {}
    PiecewiseConstant1D(Arena *arena, const f32 *values, u32 numValues, f32 minD, f32 maxD)
        : num(numValues), func(values), minD(minD), maxD(maxD)
    {
        num       = numValues;
        cdf       = PushArrayNoZero(arena, f32, numValues + 1);
        f32 total = 0.f;
        cdf[0]    = 0.f;
        for (u32 i = 1; i <= numValues; i++)
        {
            total += Abs(values[i - 1]);
            cdf[i] = total;
        }

        Assert(total != 0.f);
        Assert(total == total);
        funcInt = total * (maxD - minD) / numValues;
        for (u32 i = 1; i <= numValues; i++)
        {
            cdf[i] /= total;
        }
    }

    f32 Integral() const { return funcInt; }
    f32 Sample(f32 u, f32 *pdf = 0, u32 *offset = 0) const
    {
        u32 index = FindInterval(num + 1, [&](u32 index) { return cdf[index] <= u; });
        if (offset) *offset = index;
        if (pdf) *pdf = func[index] / funcInt;
        f32 cdfRange = cdf[index + 1] - cdf[index];
        f32 du       = (u - cdf[index]) * (cdfRange > 0.f ? 1.f / cdfRange : 0.f);
        f32 t        = (index + du) / f32(num);
        Assert(t < 1.f);
        return Lerp(t, minD, maxD);
    }
};

struct PiecewiseConstant2D
{
    PiecewiseConstant1D marginal;
    PiecewiseConstant1D *conditional;
    Vec2f minD, maxD;

    PiecewiseConstant2D() {}
    PiecewiseConstant2D(Arena *arena, const f32 *values, u32 nu, u32 nv, Vec2f minD,
                        Vec2f maxD)
        : minD(minD), maxD(maxD)
    {
        conditional = PushArrayNoZero(arena, PiecewiseConstant1D, nv);
        for (u32 v = 0; v < nv; v++)
        {
            conditional[v] = PiecewiseConstant1D(arena, values + v * nu, nu, minD[0], maxD[0]);
        }
        f32 *marginalFunc = PushArrayNoZero(arena, f32, nv);
        for (u32 v = 0; v < nv; v++)
        {
            marginalFunc[v] = conditional[v].Integral();
        }
        marginal = PiecewiseConstant1D(arena, marginalFunc, nv, minD[1], maxD[1]);
    }
    Vec2f Sample(Vec2f u, f32 *pdf = 0, Vec2u *offset = 0) const
    {
        f32 pdfs[2];
        Vec2u p;
        f32 d1 = marginal.Sample(u[1], &pdfs[1], &p[1]);
        f32 d0 = conditional[p[1]].Sample(u[0], &pdfs[0], &p[0]);
        if (pdf) *pdf = pdfs[0] * pdfs[1];
        if (offset) *offset = p;
        return Vec2f(d0, d1);
    }
    f32 PDF(Vec2f u) const
    {
        Assert(maxD != minD);
        u32 sizeU = conditional[0].num;
        u32 sizeV = marginal.num;
        u         = (u - minD) / (maxD - minD);
        Vec2u p   = Clamp(Vec2u(u * Vec2f(f32(sizeU), f32(sizeV))), Vec2u(0),
                          Vec2u(sizeU - 1, sizeV - 1));
        return conditional[p[1]].func[p[0]] / marginal.Integral();
    }
};

struct ImageInfiniteLight
{
    Image image;
    const AffineSpace *renderFromLight;
    const RGBColorSpace *imageColorSpace;
    f32 scale;
    f32 sceneRadius;
    PiecewiseConstant2D distribution;
    PiecewiseConstant2D compensatedDistribution;

    ImageInfiniteLight(Arena *arena, Image image, const AffineSpace *renderFromLight,
                       const RGBColorSpace *imageColorSpace, f32 sceneRadius, f32 scale = 1.f);
    LightFunctionsInf(ImageInfiniteLight);
    SampledSpectrum ImageLe(Vec2f uv, const SampledWavelengths &lambda) const;
};

} // namespace rt
#endif
