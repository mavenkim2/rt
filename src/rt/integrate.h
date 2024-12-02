#ifndef INTEGRATE_H
#define INTEGRATE_H

#include <Ptexture.h>

namespace rt
{
static const f32 tMinEpsilon = 0.00001f;

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
    Vec3lf<K> n;
    Vec2lf<K> uv;
    struct
    {
        Vec3lf<K> n;
        Vec3lf<K> dpdu;
        Vec3lf<K> dpdv;
    } shading;
    LaneKF32 tHit;
    LaneKU32 lightIndices;
    LaneKU32 materialIDs;
    LaneKU32 faceIndices;
    LaneKU32 rayStateHandles;
    // LaneIU32 volumeIndices;

    SurfaceInteractions() {}
    SurfaceInteractions(const Vec3lf<K> &p, const Vec3lf<K> &n, const Vec2lf<K> &uv) : p(p), n(n), uv(uv) {}
    // SurfaceInteraction(const Vec3f &p, const Vec3f &n, Vec2f u, f32 tHit) : p(p), n(n), uv(u), tHit(tHit) {}
    bool ComputeShading(BSDFBase<BxDF> &bsdf);

    u32 GenerateKey()
    {
        return {};
    }
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
    SRGB,
};

template <i32 nc>
struct PtexTexture
{
    static const u32 numChannels = nc;
    string filename;
    ColorEncoding encoding;
    f32 scale;
    PtexTexture(string filename, ColorEncoding encoding = ColorEncoding::SRGB, f32 scale = 1.f)
        : filename(filename), encoding(encoding), scale(scale) {}

    auto Evaluate(const Vec2f &uv, const Vec4f &filterWidth, u32 faceIndex)
    {
        Assert(cache);
        Ptex::String error;
        Ptex::PtexTexture *texture = cache->get((char *)filename.str, error);
        Assert(texture);
        Ptex::PtexFilter::Options opts(Ptex::PtexFilter::FilterType::f_bspline);
        Ptex::PtexFilter *filter = Ptex::PtexFilter::getFilter(texture, opts);
        Assert(nc == texture->numChannels());

        // TODO: ray differentials
        // f32 filterWidth = 0.75f;
        // Vec2f uv(0.5f, 0.5f);

        f32 out[numChannels];
        filter->eval(out, 0, nc, faceIndex, uv[0], uv[1], filterWidth[0], filterWidth[1], filterWidth[2], filterWidth[3]);
        texture->release();
        filter->release();

        // Convert to srgb
        if constexpr (numChannels == 1) return out[0];

        if (encoding == ColorEncoding::SRGB)
        {
            for (i32 i = 0; i < nc; i++)
            {
                out[i] = ExactLinearToSRGB(out[i]);
            }
        }
        for (i32 i = 0; i < nc; i++)
        {
            out[i] *= scale;
        }

        Assert(numChannels == 3);
        return Vec3f(out[0], out[1], out[2]);
    }
};

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
        LinearSpace frame = LinearSpace::FromXZ(dpdu, intrs.shading.ns); // Cross(ns, intrs.shading.dpdu), intrs.shading.ns);
        // Transform to world space
        ns   = TransformV(frame, ns);
        dpdu = Normalize(dpdu - Dot(dpdu, ns) * ns) * dpduLength;
        dpdv = Normalize(Cross(ns, dpdu)) * dpdvLength;
    }
};

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

template <typename TextureType>
struct ImageTextureShader
{
    static const u32 numChannels = TextureType::numChannels;
    TextureType texture;
    ImageTextureShader() {}

    static Veclfn<numChannels> Evaluate(SurfaceInteractionsN &intrs, Vec4lfn &filterWidths,
                                        LaneNF32 &dfdu, LaneNF32 &dfdv, const ImageTextureShader<TextureType> **textures,
                                        Veclfn<numChannels> &out)
    {
        Veclfn<numChannels> results;
        // Finite differencing
        LaneF32<K> du = .5f * Abs(filterWidths[0], filterWidths[2]);
        du            = Select(du == 0.f, 0.0005f, du);
        LaneF32<K> dv = .5f * Abs(filterWidths[1], filterWidths[3]);
        dv            = Select(dv == 0.f, 0.0005f, dv);

        for (u32 i = 0; i < IntN; i++)
        {
            Vec2f uv(intrs.uv[0][i], intrs.uv[1][i]);
            Vec4f filterWidth(filterWidths[0][i], filterWidths[1][i], filterWidths[2][i], filterWidths[3][i]);

            Set(results, i) = textures[i]->texture.Evaluate(uv, filterWidth, intrs.faceIndex[i]);
            dfdu[i]         = textures[i]->texture.Evaluate(uv + Vec2f(du[i], 0.f), filterWidth, intrs.faceIndex[i]);
            dfdv[i]         = textures[i]->texture.Evaluate(uv + Vec2f(0.f, dv[i]), filterWidth, intrs.faceIndex[i]);
        }
        return results;
    }
};

template <typename TextureShader>
struct BumpMap
{
    // p' = p + d * n, d is displacement, estimate shading normal by computing dp'du and dp'dv (using chain rule)
    TextureShader displacementShader;
    template <i32 width>
    static void Evaluate(SurfaceInteractions<width> &intrs, const BumpMap<TextureShader> **bumpMaps)
    {
        TextureShader *displacementShaders[width];
        for (u32 i = 0; i < width; i++)
        {
            displacementShaders[i] = &bumpMaps[i]->displacementShader;
        }

        LaneF32<width> dddu, dddv;
        LaneF32<width> displacement = TextureShader::Evaluate(intrs, dpdu, dpdv, displacementShaders);

        Vec3lf<width> dpdu = intrs.shading.dpdu + dddu * intrs.shading.n + displacement * intrs.shading.dndu;
        Vec3lf<width> dpdv = intrs.shading.dpdv + dddv * intrs.shading.n + displacement * intrs.shading.dndv;

        intrs.shading.n    = Cross(dpdu, dpdv);
        intrs.shading.dpdu = dpdu;
        intrs.shading.dpdv = dpdv;
    }
};

template <typename TextureShader>
struct DiffuseMaterial
{
    using BxDF = DiffuseBxDF;
    TextureShader reflectanceShader;

    static BxDF GetBxDF(SurfaceInteractionsN &intr);
    BxDF GetBxDF(SurfaceInteraction &intr);
};

template <typename TextureShaderReflectance, typename TextureShaderTransmission>
struct DiffuseTransmissionMaterialBase
{
    using BxDF = DiffuseTransmissionBxDF;
    TextureShaderReflectance reflectanceShader;
    TextureShaderTransmission transmissionShader;

    static BxDF GetBxDF(SurfaceInteractionsN &intr);
    BxDF GetBxDF(SurfaceInteraction &intr);
};

template <typename TextureShaderReflectance, typename TextureShaderTransmission>
using DiffuseTransmissionMaterial = DiffuseTransmissionMaterialBase<TextureShaderReflectance, TextureShaderTransmission>;

template <typename BxDFShader, typename NormalShader>
struct Material2
{
    using BxDF = typename BxDFShader::BxDF;
    BxDFShader bxdfShader;
    NormalShader normalShader;
    static BSDFBase<BxDF> Evaluate(SurfaceInteractionsN &intr)
    {
        BxDF *bxdfs[IntN];
        NormalShader *normalShaders[K];

        Materiall2 *materials = scene->materials.Get<Material2>();
        for (u32 i = 0; i < IntN; i++)
        {
            Material2 &material = materials[intrs.materialIDs[i]];
            bxdfs[i]            = material.bxdfShader;
            normalShaders[i]    = material.normalShader;
        }
        auto bxdf = BxDFShader::GetBxDF(intr, bxdfs);
        NormalShader::Evaluate(intrs, normalShaders);

        return BSDFBase<BxDF>(bxdf, intr.shading.dpdu, intr.shading.n);
    }
};

// TODO: automate this :)
template <i32 K>
using PtexShader = ImageTextureShader<PtexTexture<K>>;
template <>
using BumpMapPtex = BumpMap<PtexShader<1>>;
template <>
using DiffuseMaterialPtex = DiffuseMaterial<PtexShader<3>>;
template <>
using DiffuseMaterialPtex = DiffuseTransmissionMaterial<PtexShader<3>, PtexShader<3>>;
template <>
using DielectricMaterialPtex = DielectricMaterial<PtexShader<3>, PtexShader<3>>;

template <>
using DiffuseMaterialBumpMapPtex = Material2<DiffuseMaterialPtex, BumpMapPtex>;
template <>
using DiffuseTransmissionMaterialBumpMapPtex = Material2<DiffuseTransmissionMaterialPtex, BumpMapPtex>;
template <>
using DielectricMaterialBumpMapPtex = Material2<DielectricMaterialPtex, BumpMapPtex>;

// struct

static const u32 invalidVolume = 0xffffffff;
struct Ray2
{
    Vec3f o;
    Vec3f d;
    f32 tMax;
    u32 volumeIndex = invalidVolume;

    Ray2() {}
    Ray2(const Vec3f &o, const Vec3f &d) : o(o), d(d) {}
    Vec3f operator()(f32 t) const
    {
        return o + t * d;
    }
};

Ray2 Transform(const Mat4 &m, const Ray2 &r)
{
    return Ray2(TransformP(m, r.o), TransformV(m, r.d));
}

Ray2 Transform(const AffineSpace &m, const Ray2 &r)
{
    return Ray2(TransformP(m, r.o), TransformV(m, r.d));
}

struct RayDifferential
{
    Vec3lfn o;
    Vec3lfn d;
    LaneNF32 t;
    Vec3lfn rxOrigin, ryOrigin;
    Vec3lfn rxDir, ryDir;
};

struct VolumeHandle
{
    u32 index;
    VolumeHandle() {}
    VolumeHandle(u32 index) : index(index) {}
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
    RaySegment(f32 tMin, f32 tMax, f32 min, f32 max, SampledSpectrum spec, VolumeHandle *handles)
        : tMin(tMin), tMax(tMax), cMaj(spec * min), cMin(spec * max), handles{handles[0], handles[1], handles[2], handles[3]} {}
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
                : node(node), b(b), tMin(tMin), tMax(tMax) {}
        };
        StackEntry entries[128];
        u32 stackPtr;

        Iterator() {}
        Iterator(const Ray2 *ray, const SampledSpectrum &cExtinct, f32 tMax, VolumeAggregate *agg)
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

NEESample VolumetricSampleEmitter(const SurfaceInteraction &intr, Ray2 &ray, Sampler sampler,
                                  SampledSpectrum beta, const SampledSpectrum &p, const SampledWavelengths &lambda, Vec3f &wi);

} // namespace rt
#endif
