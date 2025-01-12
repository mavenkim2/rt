#ifndef INTEGRATE_H
#define INTEGRATE_H

#include "bxdf.h"
#include "color.h"
#include "math/simd_include.h"
#include "spectrum.h"
#include <Ptexture.h>

namespace rt
{
static const f32 tMinEpsilon = 0.0001f;

#if 0
template <typename T, i32 i>
struct Dual;

template <typename T>
struct Dual<T, 1>
{
    T a; // real
    T d; // infinitesimal
    Dual() {}
    Dual(T a, T d = T(1.f)) : a(a), d(d) {}
};

template <typename T>
struct Dual<T, 2>
{
    T a;    // real
    T d[2]; // infinitesimal
    Dual() {}
    Dual(T a, T d0 = T(1.f), T d1 = T(1.f)) : a(a), d{d0, d1} {}
};

template <typename T>
__forceinline Dual<T, 1> operator+(const Dual<T, 1> &a, const Dual<T, 1> &b)
{
    return Dual<T, 1>(a.a + b.a, a.d + b.d);
}

template <typename T>
__forceinline Dual<T, 1> operator*(const Dual<T, 1> &a, const Dual<T, 1> &b)
{
    return Dual<T, 1>(a.a * b.a, a.a * b.b + a.b * b.a);
}

template <typename T>
__forceinline Dual<T, 2> operator+(const Dual<T, 2> &a, const Dual<T, 2> &b)
{
    return Dual<T, 2>(a.a + b.a, a.d[0] + b.d[0], a.d[1] + b.d[1]);
}

template <typename T>
__forceinline Dual<T, 2> operator*(const Dual<T, 2> &a, const Dual<T, 2> &b)
{
    return Dual<T, 2>(a.a * b.a, a.a * b.d[0] + a.d[0] * b.a, a.a * b.d[1] + a.d[1] * b.a);
}

template <typename T>
T ReflectTest(T &wo, T &n)
{
    return -wo + 2 * Dot(wo, n) * n;
}
#endif

struct SortKey
{
    u32 value;
    u32 index;
};

template <typename BxDF>
struct BSDFBase;

template <i32 K>
struct SurfaceInteractions
{
    using LaneKF32 = LaneF32<K>;
    using LaneKU32 = LaneU32<K>;
    Vec3lf<K> p;
    Vec3lf<K> pError;
    Vec3lf<K> n;
    Vec2lf<K> uv;
    struct
    {
        Vec3lf<K> n;
        Vec3lf<K> dpdu;
        Vec3lf<K> dpdv;
        Vec3lf<K> dndu;
        Vec3lf<K> dndv;
    } shading;
    LaneKF32 tHit;
    LaneKU32 lightIndices;
    LaneKU32 materialIDs;
    LaneKU32 faceIndices;
    LaneKU32 rayStateHandles;
    // LaneIU32 volumeIndices;

    SurfaceInteractions() {}
    SurfaceInteractions(const Vec3lf<K> &p, const Vec3lf<K> &n, const Vec2lf<K> &uv)
        : p(p), n(n), uv(uv)
    {
    }
    // SurfaceInteraction(const Vec3f &p, const Vec3f &n, Vec2f u, f32 tHit) : p(p), n(n),
    // uv(u), tHit(tHit) {}

    u32 GenerateKey() { return {}; }
};

typedef SurfaceInteractions<1> SurfaceInteraction;
typedef SurfaceInteractions<IntN> SurfaceInteractionsN;

static Ptex::PtexCache *cache;
struct : public PtexErrorHandler
{
    void reportError(const char *error) override { Error(0, "%s", error); }
} errorHandler;

enum class ColorEncoding
{
    Linear,
    Gamma,
    SRGB,
};

#if 0
struct ConstantTexture
{
    f32 c;

    ConstantTexture() {}
    ConstantTexture(const f32 &t) : c(t) {}
    static LaneNF32 Evaluate(SurfaceInteractionsN &, ConstantTexture **textures, Vec4lfn &,
                             SampledWavelengthsN &)
    {
        LaneNF32 result;
        for (u32 i = 0; i < IntN; i++)
        {
            Set(result, i) = textures[i]->c;
        }
        return result;
    }
    __forceinline static LaneNF32 EvaluateFloat(SurfaceInteractionsN &intr,
                                                ConstantTexture **textures, Vec4lfn &v,
                                                SampledWavelengthsN &lambda)
    {
        return Evaluate(intr, textures, v, lambda);
    }

    __forceinline static LaneNF32 EvaluateAlbedo(SurfaceInteractionsN &intr,
                                                 ConstantTexture **textures, Vec4lfn &v,
                                                 SampledWavelengthsN &lambda)
    {
        return Evaluate(intr, textures, v, lambda);
    }
};

struct ConstantSpectrumTexture
{
    RGBAlbedoSpectrum c;
    ConstantSpectrumTexture() {}
    ConstantSpectrumTexture(RGBAlbedoSpectrum c) : c(c) {}
    static SampledSpectrumN EvaluateAlbedo(SurfaceInteractionsN &,
                                           ConstantSpectrumTexture **textures, Vec4lfn &,
                                           SampledWavelengthsN &lambda)
    {
        SampledSpectrumN result;
        for (u32 i = 0; i < IntN; i++)
        {
            ConstantSpectrumTexture *tex = textures[i];
            SampledSpectrum out          = tex->c.Sample(lambda);
            for (u32 j = 0; j < NSampledWavelengths; j++)
            {
                Set(result.values[j], i) = out.values[j];
            }
        }
        return result;
    }
};

struct PtexTexture
{
    string filename;
    ColorEncoding encoding;
    f32 scale;
    PtexTexture() {}
    PtexTexture(string filename, ColorEncoding encoding = ColorEncoding::SRGB, f32 scale = 1.f)
        : filename(filename), encoding(encoding), scale(scale)
    {
    }

    template <i32 nc>
    static void EvaluateBase(SurfaceInteractionsN &intrs, PtexTexture **textures,
                             const Vec4lfn &filterWidths, f32 results[nc * IntN])
    {
        for (u32 index = 0; index < IntN; index++)
        {
            Vec4f filterWidth = Get(filterWidths, index);
            Vec2f uv          = Get(intrs.uv, index);
            u32 faceIndex     = Get(intrs.faceIndices, index);

            PtexTexture *tex = textures[index];
            Assert(cache);
            Ptex::String error;
            Ptex::PtexTexture *texture = cache->get((char *)tex->filename.str, error);
            Assert(texture);
            u32 numFaces = texture->getInfo().numFaces;
            Assert(faceIndex < numFaces);
            Ptex::PtexFilter::Options opts(Ptex::PtexFilter::FilterType::f_bspline);
            Ptex::PtexFilter *filter = Ptex::PtexFilter::getFilter(texture, opts);
            Assert(texture->numChannels() == nc);

            // TODO: ray differentials
            // Vec2f uv(0.5f, 0.5f);

            f32 out[nc];
            filter->eval(out, 0, nc, faceIndex, uv[0], uv[1], filterWidth[0], filterWidth[1],
                         filterWidth[2], filterWidth[3]);

            // if (dfduOut && dfdvOut)
            // {
            //     // Finite differencing
            //     f32 du = .5f * (Abs(filterWidth[0]) + Abs(filterWidth[2]));
            //     du     = Select(du == 0.f, 0.0005f, du);
            //     f32 dv = .5f * (Abs(filterWidth[1]) + Abs(filterWidth[3]));
            //     dv     = Select(dv == 0.f, 0.0005f, dv);
            //
            //     f32 *dfdu;
            //     f32 *dfdv;
            //     if constexpr (nc == 1)
            //     {
            //         dfdu = dfduOut;
            //         dfdv = dfdvOut;
            //     }
            //     else
            //     {
            //         dfdu = dfduOut->e;
            //         dfdv = dfdvOut->e;
            //     }
            //     filter->eval(dfdu, 0, nc, faceIndex, uv[0] + du, uv[1], filterWidth[0],
            //                  filterWidth[1], filterWidth[2], filterWidth[3]);
            //     filter->eval(dfdv, 0, nc, faceIndex, uv[0], uv[1] + dv, filterWidth[0],
            //                  filterWidth[1], filterWidth[2], filterWidth[3]);
            // }

            texture->release();
            filter->release();

            // Convert to srgb
            if constexpr (nc == 1) results[index] = out[0];
            else
            {
                if (tex->encoding == ColorEncoding::SRGB)
                {
                    u8 rgb[3];
                    for (i32 i = 0; i < nc; i++)
                    {
                        rgb[i] = u8(Clamp(out[i] * 255.f + 0.5f, 0.f, 255.f));
                    }
                    Vec3f rgbF = SRGBToLinear(rgb);
                    out[0]     = rgbF.x;
                    out[1]     = rgbF.y;
                    out[2]     = rgbF.z;
                }
                else
                {
                    out[0] = Pow(out[0], 2.2f);
                    out[1] = Pow(out[1], 2.2f);
                    out[2] = Pow(out[2], 2.2f);
                }
                for (i32 i = 0; i < nc; i++)
                {
                    out[i] *= tex->scale;
                }

                results[index * 3]     = out[0];
                results[index * 3 + 1] = out[1];
                results[index * 3 + 2] = out[2];
            }
        }
    }
    static LaneNF32 EvaluateFloat(SurfaceInteractionsN &intrs, PtexTexture **textures,
                                  const Vec4lfn &filterWidths)
    {
        LaneNF32 result;
        EvaluateBase<1>(intrs, textures, filterWidths, &result);
        return result;
    }
    static SampledSpectrumN EvaluateAlbedo(SurfaceInteractionsN &intrs, PtexTexture **textures,
                                           const Vec4lfn &filterWidths,
                                           SampledWavelengthsN &lambda)
    {
        Vec3f results[IntN];
        EvaluateBase<3>(intrs, textures, filterWidths, results[0].e);
        return RGBAlbedoSpectrum::Sample(*RGBColorSpace::sRGB, results, lambda);
    }
};
#endif

// template <typename Texture>
struct NormalMap
{
    template <i32 K>
    void Evaluate(SurfaceInteractions<K> &intrs)
    {
        Vec3f ns(2 * normalMap.BilerpChannel(uv, wrap), -1);
        ns = Normalize(ns);

        f32 dpduLength    = Length(dpdu);
        f32 dpdvLength    = Length(dpdv);
        dpdu              = dpdu / length;
        LinearSpace frame = LinearSpace::FromXZ(
            dpdu, intrs.shading.ns); // Cross(ns, intrs.shading.dpdu), intrs.shading.ns);
        // Transform to world space
        ns   = TransformV(frame, ns);
        dpdu = Normalize(dpdu - Dot(dpdu, ns) * ns) * dpduLength;
        dpdv = Normalize(Cross(ns, dpdu)) * dpdvLength;
    }
};

#if 0
template <typename TextureType, typename RGBSpectrum>
struct ImageTextureShader
{
    static const u32 numChannels = TextureType::numChannels;
    using Vec                    = std::conditional_t<numChannels == 3, Vec3f, f32>;
    TextureType texture;
    ImageTextureShader() {}
    ImageTextureShader(TextureType texture) : texture(texture) {}

    static auto Evaluate(SurfaceInteractionsN &intrs, ImageTextureShader **textures,
                         Vec4lfn &filterWidths, SampledWavelengthsN &lambda)
    {
        Veclfn<numChannels> results;
        for (u32 i = 0; i < IntN; i++)
        {
            Set(results, i) = textures[i]->texture.Evaluate(intrs, filterWidths, i);
        }
        // Convert to spectra
        if constexpr (numChannels == 1)
        {
            return results;
        }
        else
        {
            static_assert(numChannels == 3, "Num channels must be 1 or 3");
            Vec3lfn coeffs;
            if constexpr (std::is_same_v<RGBSpectrum, RGBAlbedoSpectrum>)
            {
                return RGBAlbedoSpectrum::Sample(*RGBColorSpace::sRGB, results, lambda);
            }
            else if constexpr (std::is_same_v<RGBSpectrum, RGBUnboundedSpectrum>)
            {
                return RGBUnboundedSpectrum::Sample(*RGBColorSpace::sRGB, results, lambda);
            }
            else
            {
                Error(0, "RGBSpectrum must be RGBAlbedoSpectrum or RGBUnboundedSpectrum");
            }
        }
    }

    static void Evaluate(SurfaceInteractionsN &intrs, Vec4lfn &filterWidths,
                         Veclfn<numChannels> &dfdu, Veclfn<numChannels> &dfdv,
                         const ImageTextureShader **textures, Veclfn<numChannels> &results)
    {
        // Finite differencing
        // LaneF32<K> du = .5f * Abs(filterWidths[0], filterWidths[2]);
        // du            = Select(du == 0.f, 0.0005f, du);
        // LaneF32<K> dv = .5f * Abs(filterWidths[1], filterWidths[3]);
        // dv            = Select(dv == 0.f, 0.0005f, dv);

        for (u32 i = 0; i < IntN; i++)
        {
            Vec out_dfdu, out_dfdv;
            Set(results, i) =
                textures[i]->texture.Evaluate(intrs, filterWidths, i, &out_dfdu, &out_dfdv);
            Set(dfdu, i) = out_dfdu;
            Set(dfdv, i) = out_dfdv;
            // TODO: this won't work for 3 channel partial derivatives
            // dfdu[i]         = textures[i]->texture.Evaluate(uv + Vec2f(du[i], 0.f),
            // filterWidth, Get(intrs.faceIndices, i)); dfdv[i]         =
            // textures[i]->texture.Evaluate(uv + Vec2f(0.f, dv[i]), filterWidth,
            // Get(intrs.faceIndices, i));
        }
    }
};

template <typename TextureShader>
struct BumpMap
{
    // p' = p + d * n, d is displacement, estimate shading normal by computing dp'du and dp'dv
    // (using chain rule)
    TextureShader displacementShader;
    template <i32 width>
    static void Evaluate(SurfaceInteractions<width> &intrs, Vec4lfn &filterWidths,
                         const BumpMap<TextureShader> **bumpMaps)
    {
        const TextureShader *displacementShaders[width];
        for (u32 i = 0; i < width; i++)
        {
            displacementShaders[i] = &bumpMaps[i]->displacementShader;
        }

        LaneF32<width> dddu, dddv;
        LaneF32<width> displacement;
        TextureShader::Evaluate(intrs, filterWidths, dddu, dddv, displacementShaders,
                                displacement);

        Vec3lf<width> dpdu =
            intrs.shading.dpdu + dddu * intrs.shading.n + displacement * intrs.shading.dndu;
        Vec3lf<width> dpdv =
            intrs.shading.dpdv + dddv * intrs.shading.n + displacement * intrs.shading.dndv;

        intrs.shading.n    = Cross(dpdu, dpdv);
        intrs.shading.dpdu = dpdu;
        intrs.shading.dpdv = dpdv;
    }
};

#define MaterialHeaders(materialName)                                                         \
    materialName() = default;                                                                 \
    static BxDF GetBxDF(SurfaceInteractionsN &intr, materialName **materials,                 \
                        Vec4lfn &filterWidths, SampledWavelengthsN &lambda);                  \
    BxDF GetBxDF(SurfaceInteraction &intr, Vec4lfn &filterWidths, SampledWavelengthsN &lambda);

template <typename RflShader>
struct DiffuseMaterial
{
    using BxDF = DiffuseBxDF;
    RflShader rflShader;

    MaterialHeaders(DiffuseMaterial);
    DiffuseMaterial(RflShader rflShader) : rflShader(rflShader) {}
};

template <typename RflShader, typename TrmShader>
struct DiffuseTransmissionMaterial
{
    using BxDF = DiffuseTransmissionBxDF;
    RflShader rflShader;
    TrmShader trmShader;

    MaterialHeaders(DiffuseTransmissionMaterial);
    DiffuseTransmissionMaterial(RflShader rflShader, TrmShader trmShader)
        : rflShader(rflShader), trmShader(trmShader)
    {
    }
};

template <typename RghShader, typename Spectrum>
struct DielectricMaterial
{
    using BxDF = DielectricBxDF;
    RghShader rghShader;
    Spectrum ior;

    MaterialHeaders(DielectricMaterial);
    DielectricMaterial(RghShader rghShader, Spectrum ior) : rghShader(rghShader), ior(ior) {}
};

struct MSDielectricMaterial
{
    using BxDF = MSDielectricBxDF;
    // RghShader rghShader;
    f32 alphaX, alphaY;
    ConstantSpectrum ior;

    MaterialHeaders(MSDielectricMaterial);
    MSDielectricMaterial(f32 alphaX, f32 alphaY, ConstantSpectrum ior)
        : alphaX(alphaX), alphaY(alphaY), ior(ior)
    {
    }
};

// NOTE: dielectric interface above a diffuse
template <typename RghShader, typename RflShader, typename AlbedoShader, typename Spectrum>
struct CoatedDiffuseMaterial
{
    using BxDF = CoatedDiffuseBxDF;
    RghShader rghShader;
    RflShader rflShader;
    AlbedoShader albShader;

    Spectrum ior;
    f32 thickness, g;
    u32 maxDepth, nSamples;

    MaterialHeaders(CoatedDiffuseMaterial);
    CoatedDiffuseMaterial(RghShader rghShader, RflShader rflShader, AlbedoShader albShader,
                          Spectrum ior, f32 thickness = .01, f32 g = 0, u32 maxDepth = 10,
                          u32 nSamples = 1)
        : rghShader(rghShader), rflShader(rflShader), albShader(albShader),
          thickness(thickness), g(g), ior(ior), maxDepth(maxDepth), nSamples(nSamples)
    {
    }
};

struct NullShader
{
    NullShader() {}
    static void Evaluate(SurfaceInteractionsN, Vec4lfn, const NullShader **) {}
};

template <typename BxDFShader, typename NormalShader>
struct Material2
{
    using BxDF = typename BxDFShader::BxDF;
    BxDFShader bxdfShader;
    NormalShader normalShader;
    Material2() = default;
    Material2(BxDFShader bxdfShader, NormalShader normalShader)
        : bxdfShader(bxdfShader), normalShader(normalShader)
    {
    }
    template <typename BxDFOut>
    static void Evaluate(Arena *arena, SurfaceInteractionsN &intr, SampledWavelengthsN &lambda,
                         BSDFBase<BxDFOut> *result)
    {
        Assert(0);
        // BxDFShader *bxdfs[IntN];
        // const NormalShader *normalShaders[IntN];
        //
        // using MaterialType = Material2<BxDFShader, NormalShader>;
        // // void *materials    = GetScene()->materials.Get<MaterialType>();
        // for (u32 i = 0; i < IntN; i++)
        // {
        //     MaterialHandle handle = Get(intr.materialIDs, i);
        //     u32 index             = handle.GetIndex();
        //     Assert(index < GetScene()->materials.GetCount<MaterialType>());
        //     auto &material   = materials[index];
        //     bxdfs[i]         = &material.bxdfShader;
        //     normalShaders[i] = &material.normalShader;
        // }
        // auto bxdf = BxDFShader::GetBxDF(intr, bxdfs);
        BxDF *bxdf = PushStruct(arena, BxDF);
        // Vec4lfn filterWidths(zero);
        // *bxdf = BxDFShader::GetBxDF(intr, bxdfs, filterWidths, lambda);
        // NormalShader::Evaluate(intr, filterWidths, normalShaders);

        new (result) BSDFBase<BxDFOut>(bxdf, intr.shading.dpdu, intr.shading.n);
    }
};
#endif

static const u32 invalidVolume = 0xffffffff;
struct Ray2
{
    Vec3f o;
    Vec3f d;
    f32 tFar;
    u32 volumeIndex = invalidVolume;

    Ray2() {}
    Ray2(const Vec3f &o, const Vec3f &d) : o(o), d(d) {}
    Ray2(const Vec3f &o, const Vec3f &d, f32 tFar) : o(o), d(d), tFar(tFar) {}
    Vec3f operator()(f32 t) const { return o + t * d; }
};

Ray2 Transform(const Mat4 &m, const Ray2 &r)
{
    Ray2 newRay = r;
    newRay.o    = TransformP(m, r.o);
    newRay.d    = TransformV(m, r.d);
    return newRay;
}

Ray2 Transform(const AffineSpace &m, const Ray2 &r)
{
    Ray2 newRay = r;
    newRay.o    = TransformP(m, r.o);
    newRay.d    = TransformV(m, r.d);
    return newRay;
}

struct RayDifferential
{
    Vec3lfn o;
    Vec3lfn d;
    LaneNF32 t;
    Vec3lfn rxOrigin, ryOrigin;
    Vec3lfn rxDir, ryDir;
};

struct OctreeNode
{
    // OctreeNode *children[8];
    // Bounds bounds;
    VolumeHandle volumeHandles[4];
    u32 numVolumes;
    OctreeNode *children;
    f32 extinctionMin;
    f32 extinctionMax;
};

struct RaySegment
{
    f32 tMin;
    f32 tMax;
    SampledSpectrum cMaj;
    SampledSpectrum cMin;
    VolumeHandle handles[4];
    RaySegment() {}
    RaySegment(f32 tMin, f32 tMax, f32 min, f32 max, SampledSpectrum spec,
               VolumeHandle *handles)
        : tMin(tMin), tMax(tMax), cMaj(spec * min), cMin(spec * max),
          handles{handles[0], handles[1], handles[2], handles[3]}
    {
    }
};

struct VolumeAggregate
{
    Bounds volumeBounds;
    OctreeNode *root;

    struct Iterator
    {
        static constexpr u32 MAX_VOLUMES = 8;
        const Ray2 *ray;
        Lane8F32 invRayDx;
        Lane8F32 invRayDy;
        Lane8F32 invRayDz;
        SampledSpectrum cExtinct;
        f32 tMax;

        struct StackEntry
        {
            OctreeNode *node;
            Bounds b;
            f32 tMin, tMax;
            StackEntry() {}
            StackEntry(OctreeNode *node, Bounds &b, f32 tMin, f32 tMax)
                : node(node), b(b), tMin(tMin), tMax(tMax)
            {
            }
        };
        StackEntry entries[128];
        u32 stackPtr;

        Iterator() {}
        Iterator(const Ray2 *ray, const SampledSpectrum &cExtinct, f32 tMax,
                 VolumeAggregate *agg)
            : ray(ray), cExtinct(cExtinct), tMax(tMax)
        {
            entries[stackPtr++] = StackEntry(agg->root, agg->volumeBounds, tMinEpsilon, tMax);
            invRayDx            = Rcp(ray->d[0] == -0.f ? 0.f : ray->d[0]);
            invRayDy            = Rcp(ray->d[1] == -0.f ? 0.f : ray->d[1]);
            invRayDz            = Rcp(ray->d[2] == -0.f ? 0.f : ray->d[2]);
        }

        bool Next(RaySegment &segment);
    };

    VolumeAggregate() {}
    Iterator CreateIterator(const Ray2 *ray, const SampledSpectrum &cExtinct, f32 tMax)
    {
        return Iterator(ray, cExtinct, tMax, this);
    }
    void Build(Arena *arena);
};

struct NEESample
{
    SampledSpectrum L_beta_tray;
    SampledSpectrum p_l;
    SampledSpectrum p_u;
    bool delta;
};

struct RenderParams2
{
    Mat4 cameraFromRaster;
    Mat4 renderFromCamera;
    u32 width;
    u32 height;
    Vec2u pixelMin = Vec2u(0, 0);
    Vec2u pixelMax = Vec2u(0, 0);
    Vec2f filterRadius;
    u32 spp;
    u32 maxDepth;
    f32 lensRadius  = 0.f;
    f32 focalLength = 0.f;
};

NEESample VolumetricSampleEmitter(const SurfaceInteraction &intr, Ray2 &ray, Sampler sampler,
                                  SampledSpectrum beta, const SampledSpectrum &p,
                                  const SampledWavelengths &lambda, Vec3f &wi);
static SampledWavelengths SampleVisible(f32 u);

f32 VisibleWavelengthsPDF(f32 lambda)
{
    if (lambda < LambdaMin || lambda > LambdaMax)
    {
        return 0;
    }
    return 0.0039398042f / Sqr(std::cosh(0.0072f * (lambda - 538)));
}

f32 SampleVisibleWavelengths(f32 u)
{
    return 538 - 138.888889f * std::atanh(0.85691062f - 1.82750197f * u);
}

// Importance sampling the
static SampledWavelengths SampleVisible(f32 u)
{
    SampledWavelengths swl;
    for (i32 i = 0; i < NSampledWavelengths; i++)
    {
        f32 up = u + f32(i) / NSampledWavelengths;
        if (up > 1) up -= 1;
        swl.lambda[i] = SampleVisibleWavelengths(up);
        swl.pdf[i]    = VisibleWavelengthsPDF(swl.lambda[i]);
    }
    return swl;
}

template <u32 N>
__forceinline void Transpose(const Lane4F32 lanes[N], Vec3lf<N> &out)
{
    if constexpr (N == 1) out = ToVec3f(lanes[0]);
    else if constexpr (N == 4)
        Transpose4x3(lanes[0], lanes[1], lanes[2], lanes[3], out.x, out.y, out.z);
    else if constexpr (N == 8)
        Transpose8x3(lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6],
                     lanes[7], out.x, out.y, out.z);
    else Assert(0);
}

} // namespace rt
#endif
