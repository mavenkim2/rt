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
    Vec3lf<K> dpdu, dpdv;
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
    f32 curvature;
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
    void reportError(const char *error) override { ErrorExit(0, "%s", error); }
} errorHandler;

enum class ColorEncoding
{
    Linear,
    Gamma,
    SRGB,
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
        LinearSpace frame = LinearSpace::FromXZ(
            dpdu, intrs.shading.ns); // Cross(ns, intrs.shading.dpdu), intrs.shading.ns);
        // Transform to world space
        ns   = TransformV(frame, ns);
        dpdu = Normalize(dpdu - Dot(dpdu, ns) * ns) * dpduLength;
        dpdv = Normalize(Cross(ns, dpdu)) * dpdvLength;
    }
};

static const u32 invalidVolume = 0xffffffff;
struct Ray2
{
    Vec3f o;
    Vec3f d;
    f32 tFar;
    // f32 spread;
    // f32 radius;
    Vec3f pxOffset, pyOffset;
    Vec3f dxOffset, dyOffset;

    u32 volumeIndex = invalidVolume;

    Ray2() {}
    Ray2(const Vec3f &o, const Vec3f &d) : o(o), d(d) {}
    Ray2(const Vec3f &o, const Vec3f &d, f32 tFar) : o(o), d(d), tFar(tFar) {}
    Vec3f operator()(f32 t) const { return o + t * d; }
};

Ray2 Transform(const Mat4 &m, const Ray2 &r)
{
    Ray2 newRay     = r;
    newRay.o        = TransformP(m, r.o);
    newRay.pxOffset = TransformP(m, r.pxOffset);
    newRay.pyOffset = TransformP(m, r.pyOffset);

    newRay.d        = TransformV(m, r.d);
    newRay.dxOffset = TransformV(m, r.dxOffset);
    newRay.dyOffset = TransformV(m, r.dyOffset);

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

f32 PowerHeuristic(u32 numA, f32 pdfA, u32 numB, f32 pdfB);
void DefocusBlur(const Vec3f &dIn, const Vec2f &pLens, const f32 focalLength, Vec3f &o,
                 Vec3f &d);
Vec3f IntersectRayPlane(const Vec3f &planeN, const Vec3f &planeP, const Vec3f &rayP,
                        const Vec3f &rayD);

struct CameraDifferentials
{
    Vec3f minPosX;
    Vec3f minPosY;
    Vec3f minDirX;
    Vec3f minDirY;

    CameraDifferentials()
        : minPosX(pos_inf), minPosY(pos_inf), minDirX(pos_inf), minDirY(pos_inf)
    {
    }

    void Merge(const CameraDifferentials &other)
    {
        minPosX = Min(minPosX, other.minPosX);
        minPosY = Min(minPosY, other.minPosY);
        minDirX = Min(minDirX, other.minDirX);
        minDirY = Min(minDirY, other.minDirY);
    }
};

struct Camera
{
    Mat4 cameraFromRaster;
    Mat4 renderFromCamera;
    Vec3f dxCamera;
    Vec3f dyCamera;
    f32 focalLength;
    f32 lensRadius;

    CameraDifferentials diff;
    f32 sppScale;

    Camera() {}
    Camera(const Mat4 &cameraFromRaster, const Mat4 &renderFromCamera, const Vec3f &dxCamera,
           const Vec3f &dyCamera, f32 focalLength, f32 lensRadius, u32 spp)
        : cameraFromRaster(cameraFromRaster), renderFromCamera(renderFromCamera),
          dxCamera(dxCamera), dyCamera(dyCamera), focalLength(focalLength),
          lensRadius(lensRadius)
    {
        sppScale = Max(.125f, 1.f / Sqrt((f32)spp));
    }

    Ray2 GenerateRayDifferentials(const Vec2f &pFilm, Vec2f pLens)
    {
        Vec3f pCamera = TransformP(cameraFromRaster, Vec3f(pFilm, 0.f));
        Ray2 ray(Vec3f(0.f, 0.f, 0.f), Normalize(pCamera), pos_inf);
        if (lensRadius > 0.f)
        {
            pLens = lensRadius * SampleUniformDiskConcentric(pLens);

            DefocusBlur(ray.d, pLens, focalLength, ray.o, ray.d);
            DefocusBlur(Normalize(pCamera + dxCamera), pLens, focalLength, ray.pxOffset,
                        ray.dxOffset);
            DefocusBlur(Normalize(pCamera + dyCamera), pLens, focalLength, ray.pyOffset,
                        ray.dyOffset);
        }
        ray = Transform(renderFromCamera, ray);
        return ray;
    }
};

} // namespace rt
#endif
