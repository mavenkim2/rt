#ifndef SCENE_H
#define SCENE_H

#include "bvh/bvh_types.h"
#include "handles.h"
#include "math/lane4f32.h"
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/SampleFromVoxels.h>

// #include "lights.h"
namespace rt
{

#if 0
class Sphere
{
public:
    Sphere() {}
    Sphere(Vec3f c, f32 r, Material *m) : center(c), radius(fmax(0.f, r)), material(m)
    {
        centerVec = Vec3f(0, 0, 0);
    }
    Sphere(Vec3f c1, Vec3f c2, f32 r, Material *m)
        : center(c1), radius(fmax(0.f, r)), material(m)
    {
        centerVec = c2 - c1;
    }
    bool Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const;
    Vec3f Center(f32 time) const;
    AABB GetAABB() const;
    static void GetUV(f32 &u, f32 &v, const Vec3f &p)
    {
        f32 zenith  = acos(-p.y);
        f32 azimuth = atan2(-p.z, p.x) + PI;

        u = azimuth / (2 * PI);
        v = zenith / PI;
    }
    f32 PdfValue(const Vec3f &origin, const Vec3f &direction) const;
    Vec3f Random(const Vec3f &origin, Vec2f u) const;

    Vec3f center;
    f32 radius;

private:
    Material *material;
    Vec3f centerVec;
};

typedef u32 PrimitiveType;
enum
{
    PrimitiveType_Sphere,
    PrimitiveType_Quad,
    PrimitiveType_Box,
    PrimitiveType_Triangle,
    PrimitiveType_Curve,
    PrimitiveType_Subdiv,

    // NOTE: triangle mesh instances only (for now)
    PrimitiveType_Instance     = 16,
    PrimitiveType_InstanceMask = PrimitiveType_Instance - 1,

    PrimitiveType_Count,
};

__forceinline b32 IsInstanced(u32 i) { return ~PrimitiveType_InstanceMask & i; }
__forceinline PrimitiveType GetBaseType(u32 i)
{
    return PrimitiveType(i & PrimitiveType_InstanceMask);
}

struct Disk
{
    f32 radius, height;
    const AffineSpace *objectFromRender;

    Disk(const AffineSpace *t, f32 radius = 1.f, f32 height = 0.f)
        : objectFromRender(t), radius(radius), height(height)
    {
    }
    bool Intersect(const Ray2 &r, SurfaceInteraction &intr, f32 tMax = pos_inf)
    {
        Vec3f oi = TransformP(*objectFromRender, Vec3f(r.o));
        Vec3f di = TransformV(*objectFromRender, Vec3f(r.d));

        // Compute plane intersection for disk
        // Reject disk intersections for rays parallel to the disk's plane
        if (f32(di.z) == 0) return false;

        f32 tShapeHit = (height - f32(oi.z)) / f32(di.z);
        if (tShapeHit <= 0 || tShapeHit >= tMax) return false;

        // See if hit point is inside disk radii and $\phimax$
        Vec3f pHit = Vec3f(oi) + (f32)tShapeHit * Vec3f(di);
        f32 dist2  = Sqr(pHit.x) + Sqr(pHit.y);
        if (dist2 > Sqr(radius)) return false;

        f32 phi = std::atan2(pHit.y, pHit.x);
        if (phi < 0) phi += 2 * PI;

        const f32 phiMax = 2 * PI;
        // Find parametric representation of disk hit
        f32 u    = phi / phiMax;
        f32 rHit = Sqrt(dist2);
        f32 v    = (radius - rHit) / radius; //(radius - innerRadius);

        Vec3f dpdu(-phiMax * pHit.y, phiMax * pHit.x, 0);
        Vec3f dpdv = Vec3f(pHit.x, pHit.y, 0) * (-radius) / rHit;
        // Vec3f dndu(0, 0, 0), dndv(0, 0, 0);

        // Refine disk intersection point
        pHit.z = height;

        // Return _SurfaceInteraction_ for quadric intersection
        intr = SurfaceInteraction(pHit, Normalize(Cross(dpdu, dpdv)), Vec2f(u, v));
        return true;

        // Return _QuadricIntersection_ for disk intersection
        // return QuadricIntersection{tShapeHit, pHit, phi};
    }
};

struct ConstantMedium
{
    f32 negInvDensity;
    Material *phaseFunction;

    ConstantMedium(f32 density, Material *material)
        : negInvDensity(-1 / density), phaseFunction(material)
    {
    }

    template <typename T>
    bool Hit(const T &primitive, const Ray &r, const f32 tMin, const f32 tMax,
             HitRecord &record) const
    {
        HitRecord rec1, rec2;
        if (!primitive.Hit(r, -infinity, infinity, rec1)) return false;
        // std::clog << "\n1. tMin=" << rec1.t;
        if (!primitive.Hit(r, rec1.t + 0.0001f, infinity, rec2))
        {
            // std::clog << "\ntMin=" << rec1.t << '\n'; //", tMax=" << rec2.t << '\n';
            return false;
        }

        if (rec1.t < tMin) rec1.t = tMin;
        if (rec2.t > tMax) rec2.t = tMax;

        if (rec1.t >= rec2.t)
        {
            // std::clog << "\ntMin=" << rec1.t << ", tMax=" << rec2.t;
            return false;
        }
        // std::clog << "\n2.";

        if (rec1.t < 0) rec1.t = 0;

        f32 rayLength              = Length(r.d);
        f32 distanceInsideBoundary = (rec2.t - rec1.t) * rayLength;
        f32 hitDistance            = negInvDensity * log(RandomFloat());

        if (hitDistance > distanceInsideBoundary) return false;

        record.t           = rec1.t + hitDistance / rayLength;
        record.p           = r.at(record.t);
        record.normal      = Vec3f(1, 0, 0);
        record.isFrontFace = true;
        record.material    = phaseFunction;
        return true;
    }
};
#endif

enum class IndexType
{
    u8,
    u16,
    u32,
};

struct Mesh
{
    Vec3f *p;
    Vec3f *n;
    Vec2f *uv;
    u32 *indices;
    int *faceIDs;
    u32 numIndices;
    u32 numVertices;
    u32 numFaces;

    u32 GetNumFaces() const { return numFaces; }

    __forceinline const Vec3f &GetIndexedVertex(u32 index) const
    {
        if (indices)
        {
            Assert(index < numIndices);
            return p[indices[index]];
        }
        Assert(index < numVertices);
        return p[index];
    }
};

template <GeometryType type, typename PrimRefType = PrimRef>
struct GenerateMeshRefsHelper
{
    Vec3f *p;
    u32 *indices;

    __forceinline void operator()(PrimRefType *refs, u32 offset, u32 geomID, u32 start,
                                  u32 count, RecordAOSSplits &record)
    {
        static_assert(type == GeometryType::TriangleMesh || type == GeometryType::QuadMesh);
        constexpr u32 numVerticesPerFace = type == GeometryType::QuadMesh ? 4 : 3;
        Bounds geomBounds;
        Bounds centBounds;
        for (u32 i = start; i < start + count; i++, offset++)
        {
            PrimRefType *prim = &refs[offset];
            Vec3f v[numVerticesPerFace];
            if (indices)
            {
                for (u32 j = 0; j < numVerticesPerFace; j++)
                    v[j] = p[indices[numVerticesPerFace * i + j]];
            }
            else
            {
                for (u32 j = 0; j < numVerticesPerFace; j++)
                    v[j] = p[numVerticesPerFace * i + j];
            }

            Vec3f min(pos_inf);
            Vec3f max(neg_inf);
            for (u32 j = 0; j < numVerticesPerFace; j++)
            {
                min = Min(min, v[j]);
                max = Max(max, v[j]);
            }

            Lane4F32 mins = Lane4F32(min.x, min.y, min.z, 0);
            Lane4F32 maxs = Lane4F32(max.x, max.y, max.z, 0);
            Lane4F32::StoreU(prim->min, -mins);

            if constexpr (!std::is_same_v<PrimRefType, PrimRefCompressed>)
                prim->geomID = geomID;
            prim->maxX   = max.x;
            prim->maxY   = max.y;
            prim->maxZ   = max.z;
            prim->primID = i;

            geomBounds.Extend(mins, maxs);
            centBounds.Extend(maxs + mins);
        }
        record.geomBounds = Lane8F32(-geomBounds.minP, geomBounds.maxP);
        record.centBounds = Lane8F32(-centBounds.minP, centBounds.maxP);
    }
    // Computes bounds only
    __forceinline Bounds operator()(u32 start, u32 count)
    {
        static_assert(type == GeometryType::TriangleMesh || type == GeometryType::QuadMesh);
        constexpr u32 numVerticesPerFace = type == GeometryType::QuadMesh ? 4 : 3;
        Bounds geomBounds;
        for (u32 i = start; i < start + count; i++)
        {
            Vec3f v[numVerticesPerFace];
            if (indices)
            {
                for (u32 j = 0; j < numVerticesPerFace; j++)
                    v[j] = p[indices[numVerticesPerFace * i + j]];
            }
            else
            {
                for (u32 j = 0; j < numVerticesPerFace; j++)
                    v[j] = p[numVerticesPerFace * i + j];
            }

            Vec3f min(pos_inf);
            Vec3f max(neg_inf);
            for (u32 j = 0; j < numVerticesPerFace; j++)
            {
                min = Min(min, v[j]);
                max = Max(max, v[j]);
            }

            Lane4F32 mins = Lane4F32(min.x, min.y, min.z, 0);
            Lane4F32 maxs = Lane4F32(max.x, max.y, max.z, 0);

            geomBounds.Extend(mins, maxs);
        }
        return geomBounds;
    }
};

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

f32 HenyeyGreenstein(f32 cosTheta, f32 g)
{
    g         = Clamp(g, -.99f, .99f);
    f32 denom = 1 + Sqr(g) + 2 * g * cosTheta;
    return Inv4Pi * (1 - Sqr(g)) / (denom * SafeSqrt(denom));
}

f32 HenyeyGreenstein(Vec3f wo, Vec3f wi, f32 g) { return HenyeyGreenstein(Dot(wo, wi), g); }

Vec3f SampleHenyeyGreenstein(const Vec3f &wo, f32 g, Vec2f u, f32 *pdf = 0)
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
    NanoVDBVolume(string filename, const AffineSpace *renderFromMedium, Spectrum cAbs,
                  Spectrum cScatter, f32 g, f32 cScale, f32 LeScale = 1.f,
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

    virtual BxDF Evaluate(Arena *arena, SurfaceInteraction &si, SampledWavelengths &lambda,
                          const Vec4f &filterWidths) = 0;

    // Used in SIMD mode, loads and caches data that may be used across multiple calls
    virtual void Start(struct ShadingThreadState *state) {}
    virtual void Stop() {}
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
struct PtexTexture;

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

thread_local SceneDebug debug_;
static SceneDebug *GetDebug() { return &debug_; }

struct TessellationParams
{
    Bounds bounds;
    AffineSpace transform;
    f32 currentMinDistance;
    Mutex mutex;
    AtomicHashIndex edgeIndexMap;
    struct EdgeKeyValue
    {
        u64 edgeID;
        int edgeRate;
    };
    EdgeKeyValue *edgeKeyValues;
};

struct ShapeSample
{
    Vec3f p;
    Vec3f n;
    Vec3f w;
    f32 pdf;
};

struct ScenePrimitives
{
    static const int maxSceneDepth = 4;
    typedef bool (*IntersectFunc)(ScenePrimitives *, BVHNodeN, Ray2 &,
                                  SurfaceInteractions<1> &);
    typedef bool (*OccludedFunc)(ScenePrimitives *, BVHNodeN, Ray2 &);

    string filename;

    GeometryType geometryType;

    // temp
    Tokenizer dataTokenizer;

    Vec3f boundsMin;
    Vec3f boundsMax;
    BVHNodeN nodePtr;

    // NOTE: is one of PrimitiveType
    void *primitives;

    // NOTE: only set if not a leaf node in the scene hierarchy
    union
    {
        ScenePrimitives **childScenes;
        TessellationParams *tessellationParams;
    };
    u32 numChildScenes;
    AffineSpace *affineTransforms;
    PrimitiveIndices *primIndices;

    Scheduler::Counter counter = {};
    u32 numTransforms;
    IntersectFunc intersectFunc;
    OccludedFunc occludedFunc;
    u32 numPrimitives, numFaces;

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
    StaticArray<Texture *> textures;

    // u32 numLights;

    Bounds BuildBVH(Arena **arenas, BuildSettings &settings);
    // DiffuseAreaLight *GetAreaLights() { return lights.Get<DiffuseAreaLight>(); }
    // const DiffuseAreaLight *GetAreaLights() const { return lights.Get<DiffuseAreaLight>(); }
};

struct Scene *scene_;
Scene *GetScene() { return scene_; }

ScenePrimitives **scenes_;
ScenePrimitives **GetScenes() { return scenes_; }
void SetScenes(ScenePrimitives **scenes) { scenes_ = scenes; }

// AttributeTable *GetMaterialTable(u32 tableIndex)
// {
//     Scene *scene = GetScene();
//     return &scene->materialTables[tableIndex];
// }

void BuildTLASBVH(Arena **arenas, ScenePrimitives *scene);

template <GeometryType type>
void BuildBVH(Arena **arenas, ScenePrimitives *scene);
void BuildQuadBVH(Arena **arenas, ScenePrimitives *scene);
void BuildTriangleBVH(Arena **arenas, ScenePrimitives *scene);
void BuildCatClarkBVH(Arena **arenas, ScenePrimitives *scene);
template <GeometryType type>
void ComputeTessellationParams(Mesh *meshes, TessellationParams *params, u32 start, u32 count);

} // namespace rt
#endif
