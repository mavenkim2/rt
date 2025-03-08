#ifndef INTEGRATE_H
#define INTEGRATE_H

#include "bxdf.h"
#include "color.h"
#include "handles.h"
#include "math/simd_include.h"
#include "platform.h"
#include "sampler.h"
#include "spectrum.h"
#include "surface_interaction.h"
#include <Ptexture.h>

namespace rt
{
static const f32 tMinEpsilon = 0.0001f;

template <typename BxDF>
struct BSDFBase;

extern Ptex::PtexCache *cache;
struct : public PtexErrorHandler
{
    void reportError(const char *error) override { ErrorExit(0, "%s", error); }
} errorHandler;

struct PtexHandle
{
    i64 offset;
    OS_Handle osHandle;

    i64 bufferOffset;
    i64 fileSize;
    static const i64 bufferSize = 8192;
    void *buffer;
};

struct : public PtexInputHandler
{

public:
    /** Open a file in read mode.
        Returns null if there was an error.
        If an error occurs, the error string is available via lastError().
    */
    virtual Handle open(const char *path) override
    {
        // TODO: free list pool this
        PtexHandle *handle = (PtexHandle *)malloc(sizeof(PtexHandle) + PtexHandle::bufferSize);
        handle->offset     = 0;
        handle->osHandle   = OS_CreateFile(Str8C(path));
        handle->buffer     = (void *)(handle + 1);
        handle->bufferOffset = -1;
        handle->fileSize     = OS_GetFileSize2(Str8C(path));
        return handle;
    }

    /** Seek to an absolute byte position in the input stream. */
    // virtual void seek(Handle handle, int64_t pos) override
    // {
    //     u8 **ptr = (u8 **)handle;
    //     // Assert(pos < (int64_t)str.size);
    //     *ptr = str.str + pos;
    // }
    virtual void seek(Handle handle, int64_t pos) override
    {
        PtexHandle *offset = (PtexHandle *)handle;
        offset->offset     = pos;
    }

    /** Read a number of bytes from the file.
        Returns the number of bytes successfully read.
        If less than the requested number of bytes is read, the error string
        is available via lastError().
    */
    virtual size_t read(void *buffer, size_t size, Handle handle) override
    {
        // u8 **ptr = (u8 **)handle;
        // Assert(size_t(*ptr - str.str) + size < (size_t)str.size);
        PtexHandle *ptexHandle = (PtexHandle *)handle;
        u64 offset             = ptexHandle->offset;
        Assert(ptexHandle->osHandle.handle);

        size_t result = 0;
        if (ptexHandle->buffer && ptexHandle->bufferOffset != -1 &&
            ptexHandle->offset >= ptexHandle->bufferOffset &&
            ptexHandle->offset + size <=
                Min(ptexHandle->bufferOffset + ptexHandle->bufferSize, ptexHandle->fileSize))
        {
            MemoryCopy(buffer,
                       (u8 *)ptexHandle->buffer +
                           (ptexHandle->offset - ptexHandle->bufferOffset),
                       size);
            result = size;
            ptexHandle->offset += size;
        }
        else
        {
            if (size <= ptexHandle->bufferSize)
            {
                i64 readSize =
                    Min(ptexHandle->bufferSize, ptexHandle->fileSize - ptexHandle->offset);
                result =
                    OS_ReadFile(ptexHandle->osHandle, ptexHandle->buffer, readSize, offset)
                        ? size
                        : 0;
                MemoryCopy(buffer, ptexHandle->buffer, size);
            }
            else
            {
                result = OS_ReadFile(ptexHandle->osHandle, buffer, size, offset) ? size : 0;
            }
            ptexHandle->bufferOffset = offset;
            ptexHandle->offset += result;
        }
        Assert(result);
        return result;
    }

    /** Close a file.  Returns false if an error occurs, and the error
        string is available via lastError().  */
    virtual bool close(Handle handle) override
    {
        PtexHandle *h = (PtexHandle *)handle;
        bool result   = OS_CloseFile(h->osHandle);
        free((PtexHandle *)handle);
        return result;
    }
    virtual const char *lastError() override { return 0; }

    /** Return the last error message encountered. */
} ptexInputHandler;

// template <typename Texture>
// struct NormalMap
// {
//     template <i32 K>
//     void Evaluate(SurfaceInteractions<K> &intrs)
//     {
//         Vec3f ns(2 * normalMap.BilerpChannel(uv, wrap), -1);
//         ns = Normalize(ns);
//
//         f32 dpduLength    = Length(dpdu);
//         f32 dpdvLength    = Length(dpdv);
//         dpdu              = dpdu / length;
//         LinearSpace frame = LinearSpace::FromXZ(
//             dpdu, intrs.shading.ns); // Cross(ns, intrs.shading.dpdu), intrs.shading.ns);
//         // Transform to world space
//         ns   = TransformV(frame, ns);
//         dpdu = Normalize(dpdu - Dot(dpdu, ns) * ns) * dpduLength;
//         dpdv = Normalize(Cross(ns, dpdu)) * dpdvLength;
//     }
// };

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

inline Ray2 Transform(const Mat4 &m, const Ray2 &r)
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

inline Ray2 Transform(const AffineSpace &m, const Ray2 &r)
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

// NEESample VolumetricSampleEmitter(const SurfaceInteraction &intr, Ray2 &ray, Sampler sampler,
//                                   SampledSpectrum beta, const SampledSpectrum &p,
//                                   const SampledWavelengths &lambda, Vec3f &wi);
SampledWavelengths SampleVisible(f32 u);

inline f32 VisibleWavelengthsPDF(f32 lambda)
{
    if (lambda < LambdaMin || lambda > LambdaMax)
    {
        return 0;
    }
    return 0.0039398042f / Sqr(std::cosh(0.0072f * (lambda - 538)));
}

inline f32 SampleVisibleWavelengths(f32 u)
{
    return 538 - 138.888889f * std::atanh(0.85691062f - 1.82750197f * u);
}

// Importance sampling the
inline SampledWavelengths SampleVisible(f32 u)
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

void InitializePtex();
Vec3f ConvertRadianceToRGB(const SampledSpectrum &Lin, const SampledWavelengths &lambda,
                           u32 maxComponentValue = 10);
void GenerateMinimumDifferentials(Camera &camera, RenderParams2 &params, u32 width, u32 height,
                                  u32 taskCount, u32 tileCountX, u32 tileWidth, u32 tileHeight,
                                  u32 pixelWidth, u32 pixelHeight);
struct Scene;
bool Intersect(Scene *scene, Ray2 &ray, SurfaceInteraction &si);
bool Occluded(Scene *scene, Ray2 &ray);
void Render(Arena *arena, RenderParams2 &params);

void CalculateFilterWidths(const Ray2 &ray, const Camera &camera, const Vec3f &p,
                           const Vec3f &n, const Vec3f &dpdu, const Vec3f &dpdv, Vec3f &dpdx,
                           Vec3f &dpdy, f32 &dudx, f32 &dvdx, f32 &dudy, f32 &dvdy);
Vec3f OffsetRayOrigin(const Vec3f &p, const Vec3f &err, const Vec3f &n, const Vec3f &wi);
void UpdateRayDifferentials(Ray2 &ray, const Vec3f &wi, const Vec3f &p, Vec3f n,
                            const Vec3f &dndu, const Vec3f &dndv, const Vec3f &dpdx,
                            const Vec3f &dpdy, const f32 dudx, const f32 dvdx, const f32 dudy,
                            const f32 dvdy, f32 eta, u32 flags);

struct LightSample;
bool OccludedByOpaqueSurface(Scene *scene, Ray2 &r, SurfaceInteraction &si, LightSample &ls);

} // namespace rt
#endif
