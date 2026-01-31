#include "bvh.h"
#include "../gpu/cuda/cuda_device.h"

namespace rt
{

struct OptixBVH : BVH
{
};

struct OptixDevice
{
};

static BVH *BuildBVH(Arena *arena) { OptixBVH *bvh = PushStructConstruct(arena, OptixBVH); }

static bool IntersectClosestBVH(BVH *bvh, Intersection intersect) {}

} // namespace rt
