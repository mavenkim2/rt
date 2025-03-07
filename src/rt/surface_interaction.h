#ifndef SURFACE_INTERACTION_H_
#define SURFACE_INTERACTION_H_

#include "math/simd_include.h"
#include "math/vec3.h"

namespace rt
{
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

    LaneKU32 sceneID;
    LaneKU32 geomID;

    LaneKU32 rayStateHandles;
    f32 curvature;
    // LaneIU32 volumeIndices;

    SurfaceInteractions() {}
    SurfaceInteractions(const Vec3lf<K> &p, const Vec3lf<K> &n, const Vec2lf<K> &uv)
        : p(p), n(n), uv(uv)
    {
    }
};

typedef SurfaceInteractions<1> SurfaceInteraction;
typedef SurfaceInteractions<IntN> SurfaceInteractionsN;
} // namespace rt
#endif
