// 1. Different BSDF lobes will just be a massive switch statement unless proven
//    that a different queue per BSDF is faster
// 2. Need a small BVH abstraction. probably using embree/optix
// 3. No subdivision/tessellation for now
// 4. cpu and gpu path guiding
// 5. start with integrator???

namespace rt
{

struct Intersection
{
};

struct
{
};

void IntegratorTest()
{
    BuildBVH();

    // intersect bvh
    BVH bvh;
    Intersection intersect;
    bool intersected = IntersectClosestBVH(&bvh, intersect);
    if (intersected)
    {
    }
}

} // namespace rt
