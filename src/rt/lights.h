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

struct ShapeSample;
struct LightSample
{
    SampledSpectrum L;
    Vec3f samplePoint;
    Vec3f wi;
    f32 pdf;
    LightType lightType;

    LightSample() {}
    LightSample(const ShapeSample &sample, const SampledSpectrum &L, LightType type);

    LightSample(SampledSpectrum L, Vec3f samplePoint, Vec3f wi, f32 pdf, LightType lightType)
        : L(L), samplePoint(samplePoint), wi(wi), pdf(pdf), lightType(lightType)
    {
    }
};

bool IsDeltaLight(LightType type)
{
    return type == LightType::DeltaPosition || type == LightType::DeltaDirection;
}

struct Scene;

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

struct Light
{
    virtual LightSample SampleLi(SurfaceInteraction &intr, Vec2f &u,
                                 SampledWavelengths &lambda, bool allowIncompletePDF = 0) = 0;

    virtual f32 PDF_Li(const Vec3f &prevIntrP, const SurfaceInteraction &intr,
                       bool allowIncompletePDF = 0)              = 0;
    virtual SampledSpectrum Le(const Vec3f &n, const Vec3f &w,
                               const SampledWavelengths &lambda) = 0;
};

struct InfiniteLight : Light
{
    virtual f32 PDF_Li(const Vec3f &w, bool allowIncompletePDF)                  = 0;
    virtual SampledSpectrum Le(const Vec3f &w, const SampledWavelengths &lambda) = 0;

    f32 PDF_Li(const Vec3f &prevIntrP, const SurfaceInteraction &intr, bool allowIncompletePdf)
    {
        return PDF_Li(Normalize(intr.p - prevIntrP), allowIncompletePdf);
    }
    SampledSpectrum Le(const Vec3f &n, const Vec3f &w, const SampledWavelengths &lambda)
    {
        return Le(w, lambda);
    }
};

struct DiffuseAreaLight : Light
{
    f32 scale = 1.f;
    AffineSpace *renderFromLight;

    int geomID, sceneID;
    const DenselySampledSpectrum *Lemit;

    LightType type;

    DiffuseAreaLight() {}
    virtual LightSample SampleLi(SurfaceInteraction &intr, Vec2f &u,
                                 SampledWavelengths &lambda, bool allowIncompletePDF) override;
    virtual f32 PDF_Li(const Vec3f &prevIntrP, const SurfaceInteraction &intr,
                       bool allowIncompletePDF) override;
    virtual SampledSpectrum Le(const Vec3f &n, const Vec3f &w,
                               const SampledWavelengths &lambda) override;
    // DiffuseAreaLight(Vec3f *p, f32 scale, Spectrum Lemit)
    //     : p(p), scale(scale), Lemit(LookupSpectrum(Lemit))
    // {
    //     area = Length(Cross(p[1] - p[0], p[3] - p[0]));
    // }
};

// TODO: loop over all of these after the scene is fully instantiated and add the scene radius
// struct DistantLight
// {
//     Vec3f d;
//     const DenselySampledSpectrum *Lemit;
//     f32 sceneRadius;
//     f32 scale;
//
//     DistantLight(Vec3f d, Spectrum Lemit, f32 scale = 1.f)
//         : d(d), Lemit(LookupSpectrum(Lemit)), scale(scale)
//     {
//     }
//     LightFunctionsDirac(DistantLight);
// };

// TODO: render from light?
// struct UniformInfiniteLight
// {
//     const DenselySampledSpectrum Lemit;
//     f32 scale;
//     f32 sceneRadius;
//
//     UniformInfiniteLight(Spectrum Lemit, f32 scale = 1.f)
//         : Lemit(DenselySampledSpectrum(Lemit)), scale(scale)
//     {
//     }
//     LightFunctionsInf(UniformInfiniteLight);
// };

#if 0
struct AliasTable
{
    struct AliasEntry
    {
        f32 threshold;
        f32 p;
        int alias;
    };
    AliasEntry *entries;
    u32 numValues;
    AliasTable() {}
    // https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=f65bcde1fcf82e05388b31de80cba10bf65acc07
    AliasTable(Arena *arena, const f32 *values, u32 numValues) : numValues(numValues)
    {
        TempArena temp = ScratchStart(0, 0);
        entries        = PushArray(arena, AliasEntry, numValues);
        f32 total      = 0;
        for (u32 i = 0; i < numValues; i++)
        {
            total += values[i];
        }
        Assert(total);
        total = 1.f / total;
        for (u32 i = 0; i < numValues; i++)
        {
            entries[i].p = values[i] * total;
        }

        int *under     = PushArray(temp.arena, int, numValues);
        int *over      = PushArray(temp.arena, int, numValues);
        int *indices[] = {
            under,
            over,
        };

        // i have 3 open loops
        //     1. this
        //     2. simd quasi monte carlo sample generation
        //     3. adaptive tessellation choosing better edge rates when off screen

        int counts[2] = {};

        // Divide indices into two bins based on probability
        f32 threshold = 1.f / numValues;
        for (u32 i = 0; i < numValues; i++)
        {
            int choice                        = values[i] > threshold;
            indices[choice][counts[choice]++] = i;
        }

        while (counts[0] != 0 && counts[1] != 0)
        {
            int s = counts[0]--;
            int l = counts[1]--;

            int smallIndex = indices[0][s];
            int largeIndex = indices[1][l];

            f32 probSmall     = numValues * probs[smallIndex];
            AliasEntry *entry = &entries[smallIndex];
            entry->alias      = largeIndex;
            entry->threshold  = probSmall;

            entries[largeIndex].p = probs[largeIndex] += (probs[smallIndex] - threshold);

            int choice                        = entries[largeIndex].p > threshold;
            indices[choice][counts[choice]++] = largeIndex;
        }

        while (counts[0] > 0)
        {
            counts[0]--;
            probs[indices[0][counts[0]]] = 1;
        }
        while (counts[1] > 0)
        {
            counts[1]--;
            probs[indices[1][counts[1]]] = 1;
        }

        ScratchEnd(temp);
    }

    int Sample(f32 u, f32 *pdf, f32 *du) const
    {
        f32 index               = u * numValues;
        u32 flooredIndex        = (u32)Floor(index);
        const AliasEntry *entry = &entries[Min(flooredIndex, numValues - 1)];

        f32 q = Min(index - flooredIndex, oneMinusEpsilon);
        if (q < entry->threshold)
        {
            *pdf = entry->p;
            *du  = Min((index - flooredIndex) / triplet->threshold, oneMinusEpsilon);
            return flooredIndex;
        }
        *pdf = entries[entry->alias].p;
        *du  = Min((q - entry->threshold) / (1.f - entry->threshold), oneMinusEpsilon);
        return entry->alias;
    }
};
#endif

// struct AliasTable2D
// {
//     AliasTable marginal;
//     AliasTable *conditional;
//     AliasTable2D() {}
//     AliasTable2D(Arena *arena, const f32 *values, u32 nu, u32 nv)
//     {
//         conditional = PushArrayNoZero(arena, AliasTable, nv);
//         for (u32 v = 0; v < nv; v++)
//         {
//             conditional[v] = AliasTable(arena, values + v * nu, nu);
//         }
//         f32 *marginalFunc = PushArrayNoZero(arena, f32, nv);
//         for (u32 v = 0; v < nv; v++)
//         {
//             // marginalFunc[v] = ? ;
//         }
//         marginal = AliasTable(arena, marginalFunc, nv);
//     }
//     Vec2f Sample(Vec2f u, f32 *pdf = 0, Vec2u *offset = 0) const
//     {
//         f32 pdfs[2];
//         Vec2u p;
//         f32 d1, d0;
//         p[1] = marginal.Sample(u[1], &pdfs[1], &d1);
//         // d1 *= marginal.numValues;
//         p[0] = conditional[p[1]].Sample(u[0], &pdfs[0], &d0);
//         if (pdf) *pdf = pdfs[0] * pdfs[1];
//         if (offset) *offset = p;
//         return Vec2f(d0, d1);
//     }
// };

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

struct ImageInfiniteLight : InfiniteLight
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
    virtual LightSample SampleLi(SurfaceInteraction &intr, Vec2f &u,
                                 SampledWavelengths &lambda, bool allowIncompletePDF) override;
    virtual f32 PDF_Li(const Vec3f &w, bool allowIncompletePDF) override;
    virtual SampledSpectrum Le(const Vec3f &w, const SampledWavelengths &lambda) override;
    SampledSpectrum ImageLe(Vec2f uv, const SampledWavelengths &lambda) const;
};

} // namespace rt
#endif
