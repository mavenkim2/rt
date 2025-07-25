#ifndef SCENE_H
#define SCENE_H

#include "bvh/bvh_types.h"
#include "bxdf.h"
#ifdef USE_GPU
#include "gpu_scene.h"
#else
#include "cpu_scene.h"
#endif
#include "handles.h"
#include "lights.h"
#include "parallel.h"
#include "mesh.h"
#include "scene_load.h"
#include "subdivision.h"
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/SampleFromVoxels.h>
#include <Ptexture.h>

// #include "lights.h"
namespace rt
{
struct Image;
struct RenderParams2;
struct GPUMaterial;

enum class IndexType
{
    u8,
    u16,
    u32,
};

#ifdef USE_GPU
typedef GPUMesh MeshType;
#else
typedef Mesh MeshType;
#endif

struct Volume
{
    u32 shapeIndex;
    f32 Extinction(const Vec3f &p, f32 time, f32 filterWidth) const;
    void QueryExtinction(const Bounds &bounds, f32 &cMin, f32 &cMaj) const;
    // PhaseFunction PhaseFunction() const;
};

struct NanoVDBBuffer
{
    TempArena arena;
    u64 allocSize;
    u8 *ptr;
    // NOTE: kind of messy, but the buffer owns the arena
    NanoVDBBuffer() = default;
    NanoVDBBuffer(u64 size, Arena *arena) : arena(TempBegin(arena)) { init(size); }
    u64 size() const { return allocSize; }
    const u8 *data() const { return ptr; }
    u8 *data() { return ptr; }

    void init(u64 size)
    {
        if (size == allocSize) return;
        if (allocSize > 0) clear();
        if (size == 0) return;
        allocSize = size;
        ptr       = PushArrayNoZero(arena.arena, u8, allocSize);
    }
    static NanoVDBBuffer create(u64 size, const NanoVDBBuffer *context = 0)
    {
        return NanoVDBBuffer(size, context ? context->arena.arena : ArenaAlloc());
    }
    void clear()
    {
        TempEnd(arena);
        allocSize = 0;
        ptr       = 0;
    }
};

inline f32 HenyeyGreenstein(f32 cosTheta, f32 g)
{
    g         = Clamp(g, -.99f, .99f);
    f32 denom = 1 + Sqr(g) + 2 * g * cosTheta;
    return Inv4Pi * (1 - Sqr(g)) / (denom * SafeSqrt(denom));
}

inline f32 HenyeyGreenstein(Vec3f wo, Vec3f wi, f32 g)
{
    return HenyeyGreenstein(Dot(wo, wi), g);
}

inline Vec3f SampleHenyeyGreenstein(const Vec3f &wo, f32 g, Vec2f u, f32 *pdf = 0)
{
    f32 cosTheta;
    if (Abs(g) < 1e-3f) cosTheta = 1 - 2 * u[0];
    else cosTheta = -1 / (2 * g) * (1 + Sqr(g) - Sqr((1 - Sqr(g)) / (1 + g - 2 * g * u[0])));

    f32 sinTheta = SafeSqrt(1 - Sqr(cosTheta));
    f32 phi      = TwoPi * u[1];

    // TODO: implement FromZ
    Vec3f wi;
    // Frame wFrame = Frame::FromZ(wo);
    // Vector3f wi  = wFrame.FromLocal(Vec3f(sinTheta * Cos(phi), sinTheta * Sin(phi),
    // cosTheta));

    if (pdf) *pdf = HenyeyGreenstein(cosTheta, g);
    return wi;
}

struct PhaseFunctionSample
{
    Vec3f wi;
    f32 p;
    f32 pdf = 0.f;
    PhaseFunctionSample() {}
    PhaseFunctionSample(const Vec3f &wi, f32 p, f32 pdf = 0.f) : wi(wi), p(p), pdf(pdf) {}
};

struct PhaseFunction
{
    f32 g;
    PhaseFunction() {}
    PhaseFunction(f32 g) : g(g) {}
    // NOTE: HG phase function is perfectly importance sampled, so the value of the
    // phasefunction = the pdf
    SampledSpectrum EvaluateSample(Vec3f wo, Vec3f wi, f32 *pdf) const
    {
        Assert(pdf);
        f32 p = HenyeyGreenstein(wo, wi, g);
        *pdf  = p;
        return SampledSpectrum(p);
    }
    PhaseFunctionSample GenerateSample(Vec3f wo, Vec2f u) const
    {
        f32 pdf;
        Vec3f wi = SampleHenyeyGreenstein(wo, g, u, &pdf);
        return PhaseFunctionSample{wi, pdf, pdf};
    }
    f32 PDF(Vec3f wo, Vec3f wi) const { return HenyeyGreenstein(wo, wi, g); }
};

struct NanoVDBVolume
{
    const AffineSpace *renderFromMedium;
    const AffineSpace mediumFromRender;
    static nanovdb::GridHandle<NanoVDBBuffer> ReadGrid(string str, string type)
    {
        nanovdb::GridHandle<NanoVDBBuffer> handle;
        try
        {
            handle = nanovdb::io::readGrid<NanoVDBBuffer>(
                std::string((const char *)str.str, str.size),
                std::string((const char *)type.str, type.size));
        } catch (std::exception)
        {
            ErrorExit(0, "Could not read file: %S\n", str);
        }
        return handle;
    }

    nanovdb::GridHandle<NanoVDBBuffer> densityGrid;
    nanovdb::GridHandle<NanoVDBBuffer> temperatureGrid;
    DenselySampledSpectrum cAbs;
    DenselySampledSpectrum cScatter;
    const nanovdb::FloatGrid *densityFloatGrid     = 0;
    const nanovdb::FloatGrid *temperatureFloatGrid = 0;
    // NOTE: world space bounds
    Bounds bounds;
    f32 LeScale, temperatureOffset, temperatureScale, cScale;
    PhaseFunction phaseFunction;

    NanoVDBVolume() {}
    NanoVDBVolume(string filename, const AffineSpace *renderFromMedium, Spectrum *cAbs,
                  Spectrum *cScatter, f32 g, f32 cScale, f32 LeScale = 1.f,
                  f32 temperatureOffset = 0.f, f32 temperatureScale = 1.f)
        : mediumFromRender(Inverse(*renderFromMedium)), cAbs(DenselySampledSpectrum(cAbs)),
          cScatter(DenselySampledSpectrum(cScatter)), phaseFunction(g), cScale(cScale),
          LeScale(LeScale), temperatureOffset(temperatureOffset),
          temperatureScale(temperatureScale)
    {
        densityGrid          = ReadGrid(filename, "density");
        temperatureGrid      = ReadGrid(filename, "temperature");
        densityFloatGrid     = densityGrid.grid<f32>();
        temperatureFloatGrid = temperatureGrid.grid<f32>();

        nanovdb::BBox<nanovdb::Vec3R> bbox = densityFloatGrid->worldBBox();
        bounds                             = Transform(
            *renderFromMedium,
            Bounds(Vec3f((f32)bbox.min()[0], (f32)bbox.min()[1], (f32)bbox.min()[2]),
                                               Vec3f((f32)bbox.max()[0], (f32)bbox.max()[1], (f32)bbox.max()[2])));

        nanovdb::BBox<nanovdb::Vec3R> bbox2 = temperatureFloatGrid->worldBBox();
        bounds.Extend(Transform(
            *renderFromMedium,
            Bounds(Vec3f((f32)bbox2.min()[0], (f32)bbox2.min()[1], (f32)bbox2.min()[2]),
                   Vec3f((f32)bbox2.max()[0], (f32)bbox2.max()[1], (f32)bbox2.max()[2]))));
    }

    SampledSpectrum Le(Vec3f p, const SampledWavelengths &lambda) const
    {
        // p = Transform(*mediumFromRender, p);
        if (!temperatureFloatGrid) return SampledSpectrum(0.f);
        nanovdb::Vec3<f32> pIndex =
            temperatureFloatGrid->worldToIndexF(nanovdb::Vec3<f32>(p.x, p.y, p.z));
        using TreeSampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
        f32 temp          = TreeSampler(temperatureFloatGrid->tree())(pIndex);
        temp              = (temp - temperatureOffset) * temperatureScale;
        if (temp <= 100.f) return SampledSpectrum(0.f);
        return LeScale * BlackbodySpectrum(temp).Sample(lambda);
    }
    void Extinction(Vec3f p, const SampledWavelengths &lambda, SampledSpectrum &outAbs,
                    SampledSpectrum &outScatter,
                    SampledSpectrum &le) const //, f32, f32) const
    {
        // p = ApplyInverse(*renderFromMedium, p);
        p = TransformP(mediumFromRender, p);
        nanovdb::Vec3<f32> pIndex =
            densityFloatGrid->worldToIndexF(nanovdb::Vec3<f32>(p.x, p.y, p.z));
        using TreeSampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
        f32 density       = TreeSampler(densityFloatGrid->tree())(pIndex);

        outAbs     = cAbs.Sample(lambda) * density;
        outScatter = cScatter.Sample(lambda) * density;
        le         = Le(p, lambda);
    }
    const PhaseFunction &PhaseFunction() const { return phaseFunction; }
    void QueryExtinction(Bounds inBounds, f32 &cMin, f32 &cMaj) const
    {
        inBounds = Transform(mediumFromRender, inBounds);

        if (!Intersects(bounds, inBounds))
        {
            cMin = 0.f;
            cMaj = 0.f;
            return;
        }

        nanovdb::Vec3<f32> i0 = densityFloatGrid->worldToIndexF(
            nanovdb::Vec3<f32>(inBounds.minP[0], inBounds.minP[1], inBounds.minP[2]));
        nanovdb::Vec3<f32> i1 = densityFloatGrid->worldToIndexF(
            nanovdb::Vec3<f32>(inBounds.maxP[0], inBounds.maxP[1], inBounds.maxP[2]));

        struct MediumData
        {
            f32 cMin, cMaj;
        };

        Vec3i begin((i32)i0[0] - 1, (i32)i0[1] - 1, (i32)i0[2] - 1);
        Vec3i end((i32)i1[1] + 1, (i32)i1[1] + 1, (i32)i1[2] + 1);

        Vec3i width = end - begin;

        MediumData datum;
        ParallelReduce(
            &datum, 0, width.x * width.y * width.z, PARALLEL_THRESHOLD,
            [&](MediumData &data, u32 jobID, u32 start, u32 count) {
                auto accessor = densityFloatGrid->getAccessor();
                f32 cMin      = pos_inf;
                f32 cMax      = neg_inf;

                // TODO: see if loop carried dependency, or index computation, is
                // significant overhead vs access time
                for (u32 i = start; i < count; i++)
                {
                    i32 nx    = begin[0] + (i % width[0]);
                    i32 ny    = begin[1] + ((i / width[0]) % width[1]);
                    i32 nz    = begin[2] + (i / (width[0] * width[1]));
                    f32 value = accessor.getValue({nx, ny, nz});
                    cMin      = Min(cMin, value);
                    cMax      = Max(cMax, value);
                }
                datum.cMin = cMin;
                datum.cMaj = cMax;
            },
            [&](MediumData &left, const MediumData &right) {
                left.cMin = Min(left.cMin, right.cMin);
                left.cMaj = Max(left.cMaj, right.cMaj);
            });
        cMin = datum.cMin;
        cMaj = datum.cMaj;
    }
};

struct Instance
{
    // TODO: materials
    u32 id;
    // GeometryID geomID;
    u32 transformIndex;
};

////////////////////////////////////////////////////////

enum class AttributeType
{
    Float,
    // Spectrum,
    RGB,
    String,
    Int,
    Bool,
};

struct Texture
{
    virtual void Start(struct ShadingThreadState *);
    virtual void Stop() {}
    virtual f32 EvaluateFloat(SurfaceInteraction &si, const Vec4f &filterWidths)
    {
        ErrorExit(0, "EvaluateFloat is not defined for sub class \n");
        return 0.f;
    }
    virtual SampledSpectrum EvaluateAlbedo(SurfaceInteraction &si, SampledWavelengths &lambda,
                                           const Vec4f &filterWidths)
    {
        ErrorExit(0, "EvaluateAlbedo is not defined for sub class\n");
        return {};
    }
    SampledSpectrum EvaluateAlbedo(const Vec3f &color, SampledWavelengths &lambda)
    {
        if (color == Vec3f(0.f)) return SampledSpectrum(0.f);
        Assert(!IsNaN(color[0]) && !IsNaN(color[1]) && !IsNaN(color[2]));
        return RGBAlbedoSpectrum(*RGBColorSpace::sRGB, Clamp(color, Vec3f(0.f), Vec3f(1.f)))
            .Sample(lambda);
    }
};

struct Material
{
    // TODO: actually displacement map
    struct Texture *displacement;
    int ptexReflectanceIndex;

    virtual BxDF *Evaluate(Arena *arena, SurfaceInteraction &si, SampledWavelengths &lambda,
                           const Vec4f &filterWidths) = 0;

    virtual f32 GetIOR() { return 1.f; }
    virtual bool IsTransmissive() { return false; }
    virtual MaterialTypes GetType() { return MaterialTypes::Interface; }

    // Used in SIMD mode, loads and caches data that may be used across multiple calls
    virtual void Start(struct ShadingThreadState *state) {}
    virtual void Stop() {}

    virtual GPUMaterial ConvertToGPU();
};

struct NullMaterial : Material
{
    typedef BxDF BxDFType;
    NullMaterial() {}
    BxDF *Evaluate(Arena *arena, SurfaceInteraction &si, SampledWavelengths &lambda,
                   const Vec4f &filterWidths) override
    {
        return 0;
    }
};

struct DiffuseMaterial : Material
{
    typedef DiffuseBxDF BxDFType;
    Texture *reflectance;
    DiffuseMaterial(Texture *reflectance) : reflectance(reflectance) {}
    BxDF *Evaluate(Arena *arena, SurfaceInteraction &si, SampledWavelengths &lambda,
                   const Vec4f &filterWidths) override
    {
        DiffuseBxDF *bxdf = PushStructConstruct(arena, DiffuseBxDF)();
        *bxdf             = EvaluateHelper(si, lambda, filterWidths);
        return bxdf;
    }
    DiffuseBxDF EvaluateHelper(SurfaceInteraction &si, SampledWavelengths &lambda,
                               const Vec4f &filterWidths)

    {
        SampledSpectrum s = reflectance->EvaluateAlbedo(si, lambda, filterWidths);

        return DiffuseBxDF(s);
    }
    virtual void Start(ShadingThreadState *state) override { reflectance->Start(state); }
    virtual void Stop() override { reflectance->Stop(); }
    virtual MaterialTypes GetType() override { return MaterialTypes::Diffuse; }
    virtual GPUMaterial ConvertToGPU() override;
};

struct DiffuseTransmissionMaterial : Material
{
    typedef DiffuseTransmissionBxDF BxDFType;
    Texture *reflectance;
    Texture *transmittance;
    f32 scale;
    DiffuseTransmissionMaterial(Texture *reflectance, Texture *transmittance, f32 scale)
        : reflectance(reflectance), transmittance(transmittance), scale(scale)
    {
    }
    BxDF *Evaluate(Arena *arena, SurfaceInteraction &si, SampledWavelengths &lambda,
                   const Vec4f &filterWidths) override
    {
        DiffuseTransmissionBxDF *bxdf = PushStructConstruct(arena, DiffuseTransmissionBxDF)();
        *bxdf                         = EvaluateHelper(si, lambda, filterWidths);
        return bxdf;
    }

    DiffuseTransmissionBxDF EvaluateHelper(SurfaceInteraction &si, SampledWavelengths &lambda,
                                           const Vec4f &filterWidths)
    {
        SampledSpectrum r = reflectance->EvaluateAlbedo(si, lambda, filterWidths);
        SampledSpectrum t = transmittance->EvaluateAlbedo(si, lambda, filterWidths);

        return DiffuseTransmissionBxDF(r, t);
    }
    virtual void Start(ShadingThreadState *state) override
    {
        reflectance->Start(state);
        transmittance->Start(state);
    }
    virtual void Stop() override
    {
        reflectance->Stop();
        transmittance->Stop();
    }
    virtual MaterialTypes GetType() override { return MaterialTypes::DiffuseTransmission; }
};

struct DielectricMaterial : Material
{
    typedef DielectricBxDF BxDFType;

    Texture *uRoughnessTexture;
    Texture *vRoughnessTexture;
    f32 eta;

    DielectricMaterial() = default;
    DielectricMaterial(Texture *u, Texture *v, f32 eta)
        : uRoughnessTexture(u), vRoughnessTexture(v), eta(eta)
    {
    }

    virtual f32 GetIOR() override { return eta; }
    BxDF *Evaluate(Arena *arena, SurfaceInteraction &si, SampledWavelengths &lambda,
                   const Vec4f &filterWidths) override
    {
        DielectricBxDF *bxdf = PushStructConstruct(arena, DielectricBxDF)();
        *bxdf                = EvaluateHelper(si, lambda, filterWidths);
        return bxdf;
    }

    DielectricBxDF EvaluateHelper(SurfaceInteraction &si, SampledWavelengths &lambda,
                                  const Vec4f &filterWidths)
    {
        f32 uRoughness, vRoughness;
        if (uRoughnessTexture == vRoughnessTexture)
        {
            uRoughness = vRoughness = uRoughnessTexture->EvaluateFloat(si, filterWidths);
        }
        else
        {
            uRoughness = uRoughnessTexture->EvaluateFloat(si, filterWidths);

            vRoughness = vRoughnessTexture->EvaluateFloat(si, filterWidths);
        }

        uRoughness = TrowbridgeReitzDistribution::RoughnessToAlpha(uRoughness);
        vRoughness = TrowbridgeReitzDistribution::RoughnessToAlpha(vRoughness);

        // if (eta.TypeIndex<ConstantSpectrum>() == eta.GetTag())
        // {
        //     lambda.TerminateSecondary();
        // }
        f32 ior = eta; // eta(lambda[0]);
        return DielectricBxDF(ior, TrowbridgeReitzDistribution(uRoughness, vRoughness));
    }
    virtual bool IsTransmissive() override { return true; }
    virtual void Start(ShadingThreadState *state) override
    {
        uRoughnessTexture->Start(state);
        if (uRoughnessTexture != vRoughnessTexture) vRoughnessTexture->Start(state);
    }
    virtual void Stop() override
    {
        uRoughnessTexture->Stop();
        if (uRoughnessTexture != vRoughnessTexture) vRoughnessTexture->Stop();
    }
    virtual MaterialTypes GetType() override { return MaterialTypes::Dielectric; }
    virtual GPUMaterial ConvertToGPU() override;
};

struct CoatedDiffuseMaterial : Material
{
    typedef CoatedDiffuseBxDF BxDFType;

    DielectricMaterial dielectric;
    DiffuseMaterial diffuse;
    Texture *albedo;
    Texture *g;
    i32 maxDepth;
    i32 nSamples;
    f32 thickness;

    CoatedDiffuseMaterial(DielectricMaterial die, DiffuseMaterial diff, Texture *albedo,
                          Texture *g, i32 maxDepth, i32 nSamples, f32 thickness)
        : dielectric(die), diffuse(diff), albedo(albedo), g(g), maxDepth(maxDepth),
          nSamples(nSamples), thickness(thickness)
    {
    }

    BxDF *Evaluate(Arena *arena, SurfaceInteraction &si, SampledWavelengths &lambda,
                   const Vec4f &filterWidths) override
    {
        CoatedDiffuseBxDF *bxdf = PushStructConstruct(arena, CoatedDiffuseBxDF)();
        *bxdf                   = EvaluateHelper(si, lambda, filterWidths);
        return bxdf;
    }

    CoatedDiffuseBxDF EvaluateHelper(SurfaceInteraction &si, SampledWavelengths &lambda,
                                     const Vec4f &filterWidths)

    {
        SampledSpectrum albedoValue = albedo->EvaluateAlbedo(si, lambda, filterWidths);

        f32 gValue = g->EvaluateFloat(si, filterWidths);

        return CoatedDiffuseBxDF(dielectric.EvaluateHelper(si, lambda, filterWidths),
                                 diffuse.EvaluateHelper(si, lambda, filterWidths), albedoValue,
                                 gValue, thickness, maxDepth, nSamples);
    }
    virtual void Start(ShadingThreadState *state) override
    {
        dielectric.Start(state);
        diffuse.Start(state);
    }
    virtual void Stop() override
    {
        dielectric.Stop();
        diffuse.Stop();
    }
    virtual MaterialTypes GetType() override { return MaterialTypes::CoatedDiffuse; }
};

struct PrimitiveIndices
{
    // TODO: these are actaully ids (type + index)
    LightHandle lightID;
    // u32 volumeIndex;
    MaterialHandle materialID;
    Texture *alphaTexture;

    PrimitiveIndices() {}
    PrimitiveIndices(LightHandle lightID, MaterialHandle materialID)
        : lightID(lightID), materialID(materialID), alphaTexture(0)
    {
    }
    PrimitiveIndices(LightHandle lightID, MaterialHandle materialID, Texture *alpha)
        : lightID(lightID), materialID(materialID), alphaTexture(alpha)
    {
    }
};

struct Ray2;
template <i32 K>
struct SurfaceInteractions;

enum class ColorEncoding
{
    None,
    Linear,
    Gamma,
    SRGB,
};

enum FilterType
{
    Bspline,
    CatmullRom,
};

struct PtexTexture : Texture
{
    string filename;
    f32 scale = 1.f;
    ColorEncoding encoding;
    FilterType filterType;

    PtexTexture(string filename, FilterType filterType = FilterType::Bspline,
                ColorEncoding encoding = ColorEncoding::Gamma, f32 scale = 1.f);

    f32 EvaluateFloat(SurfaceInteraction &si, const Vec4f &filterWidths) override;

    SampledSpectrum EvaluateAlbedo(SurfaceInteraction &si, SampledWavelengths &lambda,
                                   const Vec4f &filterWidths) override;

    virtual void Start(ShadingThreadState *state) override;

    virtual void Stop() override;

    template <i32 c>
    void EvaluateHelper(Ptex::PtexTexture *texture, SurfaceInteraction &intr,
                        const Vec4f &filterWidths, f32 *result) const;
    template <i32 c>
    void EvaluateHelper(SurfaceInteraction &intr, const Vec4f &filterWidths, f32 *result);
};

struct SceneDebug
{
    string filename;
    u32 geomID;
    struct ScenePrimitives *scene;
    Vec3f color;
    Vec4f filterWidths;

    Vec2i pixel;
    OpenSubdivMesh *mesh;

    PtexTexture *texture;
    void *queue;

    // CatmullClarkPatch *patch;
    // u32 num;
    // u32 primID;
    u32 index;
    // u32 sampleNum;
    // std::atomic<u32> *numTiles;
    // u32 tileCount;
};

static thread_local SceneDebug debug_;
inline SceneDebug *GetDebug() { return &debug_; }

struct TessellationParams
{
    Bounds bounds;
    AffineSpace transform;
    f64 currentMinDistance;
    Mutex mutex;
};

struct ShapeSample
{
    Vec3f p;
    Vec3f n;
    Vec3f w;
    f32 pdf;
};

template <int N>
struct StackEntry;

struct ScenePrimitives
{
    static const int maxSceneDepth = 4;
#ifndef USE_GPU
    typedef bool (*IntersectFunc)(ScenePrimitives *, StackEntry<DefaultN>, Ray2 &,
                                  SurfaceInteractions<1> &);
    typedef bool (*OccludedFunc)(ScenePrimitives *, StackEntry<DefaultN>, Ray2 &);
#endif

    string filename;

    GeometryType geometryType;

    Vec3f boundsMin;
    Vec3f boundsMax;
    BVHNodeN nodePtr;

#ifdef USE_GPU
    GPUAccelerationStructure gpuBVH;
    Semaphore semaphore;
#endif

    // NOTE: is one of PrimitiveType
    void *primitives;
    int bvhPrimSize;

    // NOTE: only set if not a leaf node in the scene hierarchy
    union
    {
        ScenePrimitives **childScenes;
        TessellationParams *tessellationParams;
    };
    u32 numChildScenes;
    AffineSpace *affineTransforms;
    PrimitiveIndices *primIndices;

    std::atomic<int> depth;
    u32 numTransforms;
#ifndef USE_GPU
    IntersectFunc intersectFunc;
    OccludedFunc occludedFunc;
#endif
    u32 numPrimitives, numFaces;

    int sceneIndex;
    int gpuInstanceID;

    ScenePrimitives() {}
    Bounds GetBounds() const { return Bounds(Lane4F32(boundsMin), Lane4F32(boundsMax)); }
    void SetBounds(const Bounds &inBounds)
    {
        boundsMin = ToVec3f(inBounds.minP);
        boundsMax = ToVec3f(inBounds.maxP);
    }

    ShapeSample SampleQuad(SurfaceInteraction &intr, Vec2f &u, AffineSpace *transform,
                           int geomID);
    ShapeSample Sample(SurfaceInteraction &intr, AffineSpace *space, Vec2f &u, int geomID);
};

struct Scene
{
    ScenePrimitives scene;

    // ArrayTuple<LightTypes> lights;

    // StaticArray<Light *> lights;
    // StaticArray<InfiniteLight *> infiniteLights;

    // TODO: use my own allocators?
    std::vector<Light *> lights;
    std::vector<InfiniteLight *> infiniteLights;

    StaticArray<Material *> materials;
    std::vector<PtexTexture> ptexTextures;
    StaticArray<Texture *> textures;

    std::vector<Mesh> causticCasters;

    // u32 numLights;

    Bounds BuildBVH(Arena **arenas, BuildSettings &settings);
    Material *GetMaterial(SurfaceInteraction &si);
    // DiffuseAreaLight *GetAreaLights() { return lights.Get<DiffuseAreaLight>(); }
    // const DiffuseAreaLight *GetAreaLights() const { return lights.Get<DiffuseAreaLight>(); }
};

extern Scene *scene_;
inline Scene *GetScene() { return scene_; }

extern ScenePrimitives **scenes_;
inline ScenePrimitives **GetScenes() { return scenes_; }
inline void SetScenes(ScenePrimitives **scenes) { scenes_ = scenes; }

inline Mesh *GetMesh(int sceneID, int geomID)
{
    Assert(scenes_);
    ScenePrimitives *s = scenes_[sceneID];
    Assert(s->geometryType != GeometryType::Instance);
    return (Mesh *)s->primitives + geomID;
}

struct MaterialNode
{
    string str;
    MaterialHandle handle;

    u32 Hash() const { return rt::Hash(str); }
    bool operator==(const MaterialNode &m) const { return str == m.str; }
    bool operator==(string s) const { return s == str; }
};

typedef HashMap<MaterialNode> MaterialHashMap;

#ifndef USE_GPU
void BuildTLASBVH(Arena **arenas, ScenePrimitives *scene);

template <GeometryType type>
void BuildBVH(Arena **arenas, ScenePrimitives *scene);
void BuildQuadBVH(Arena **arenas, ScenePrimitives *scene);
void BuildTriangleBVH(Arena **arenas, ScenePrimitives *scene);
void BuildCatClarkBVH(Arena **arenas, ScenePrimitives *scene);
template <GeometryType type>
void ComputeTessellationParams(Mesh *meshes, TessellationParams *params, u32 start, u32 count);

void BuildSceneBVHs(Arena **arenas, ScenePrimitives *scene, const Mat4 &NDCFromCamera,
                    const Mat4 &cameraFromRender, int screenHeight);
#endif
Bounds GetSceneBounds(ScenePrimitives *scene);
void Render(RenderParams2 *params, int numScenes, Image *envMap);

int LoadScene(RenderParams2 *params, Arena **tempArenas, string directory, string filename);
DiffuseAreaLight *ParseAreaLight(Arena *arena, Tokenizer *tokenizer, AffineSpace *space,
                                 int sceneID, int geomID);
Texture *ParseTexture(Arena *arena, Tokenizer *tokenizer, string directory, int *index = 0,
                      FilterType type        = FilterType::CatmullRom,
                      ColorEncoding encoding = ColorEncoding::None);

} // namespace rt
#endif
