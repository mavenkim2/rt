#ifndef RAY_H_
#define RAY_H_

#include "vec3.h"
#include "matx.h"

namespace rt
{

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
    Ray2 newRay     = r;
    newRay.o        = TransformP(m, r.o);
    newRay.pxOffset = TransformP(m, r.pxOffset);
    newRay.pyOffset = TransformP(m, r.pyOffset);

    newRay.d        = TransformV(m, r.d);
    newRay.dxOffset = TransformV(m, r.dxOffset);
    newRay.dyOffset = TransformV(m, r.dyOffset);

    return newRay;
}

} // namespace rt
#endif
