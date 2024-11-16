#ifndef INTEGRATE_H
#define INTEGRATE_H

namespace rt
{
static const f32 tMinEpsilon = 0.00001f;
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
    LaneIF32 tHit;
    LaneIU32 lightIndices;
    LaneIU32 volumeIndices;
};

struct LightSample
{
    Vec3IF32 samplePoint;
    Vec3IF32 n;
    LaneIF32 pdf;
    // TODO: simd
    SampledSpectrum L;
    bool isDelta;

    LightSample() {}
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
enum LightClass
{
    LightClass_Area,
    LightClass_InfImg,
    LightClass_Count,
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

// struct 

struct Ray2
{
    Vec3f o;
    Vec3f d;
    f32 tMax;
    u32 volumeIndex;

    Vec3f operator()(f32 t)
    {
        return o + t * d;
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

struct PhaseFunctionSample
{
    Vec3f wi;
    f32 p;
    f32 pdf = 0.f;
    PhaseFunctionSample() {}
};

struct PhaseFunction
{
    SampledSpectrum EvaluateSample(Vec3f wo, Vec3f wi, f32 *pdf) const
    {
        Assert(pdf);
        *pdf = 0.f;
        return SampledSpectrum(0.f);
    }
    PhaseFunctionSample GenerateSample(Vec3f wo, Vec2f u) const
    {
        return PhaseFunctionSample();
    }
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
    RaySegment() {}
    RaySegment(f32 tMin, f32 tMax, f32 min, f32 max, SampledSpectrum spec)
        : tMin(tMin), tMax(tMax), cMaj(spec * min), cMin(spec * max) {}
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

    Iterator CreateIterator(const Ray2 *ray, const SampledSpectrum &cExtinct, f32 tMax)
    {
        return Iterator(ray, cExtinct, tMax, this);
    }
    void Build(Arena *arena, struct Scene2 *scene);
};

} // namespace rt
#endif
