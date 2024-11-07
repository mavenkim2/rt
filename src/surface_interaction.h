#ifndef SURFACE_INTERACTION_H
#define SURFACE_INTERACTION_H

namespace rt
{
struct SurfaceInteraction
{
    Vec3f p;
    Vec3f n;
};

struct RayDifferential
{
    Vec3f rxOrigin, ryOrigin;
    Vec3f rxDir, ryDir;
};

RayDifferential ComputeRayDifferentials(const RayDifferential &ray)
{
}

} // namespace rt
#endif
