#ifndef INTEGRATE_H
#define INTEGRATE_H

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
