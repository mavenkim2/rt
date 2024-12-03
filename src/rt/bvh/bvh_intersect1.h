#ifndef BVH_INTERSECT1_H
#define BVH_INTERSECT1_H
#include "bvh_types.h"
#include "scene.h"

namespace rt
{
enum BVHType
{
    BVHType_Quantized;
};

template <i32 types, typename Intersector>
bool Intersect(Ray2 &ray, BVHNode4 bvhNode, SurfaceInteraction &itr)
{
    typedef typename Intersector::Primitive Primitive;
    // NodeType *node = bvhNode.;bvhNode->GetQuantizedNode();
    NodeType *node;

    Vec3lf4 invRayD(Select(ray.d.x == -0, pos_inf, 1 / ray.d.x),
                    Select(ray.d.y == -0, pos_inf, 1 / ray.d.y),
                    Select(ray.d.z == -0, pos_inf, 1 / ray.d.z));

    struct StackEntry
    {
        QuantizedNode4 *node;
    };
    StackEntry stack[64];
    for (;;)
    {
        StackEntry
            Vec3lf4 mins,
            maxs;
        node->GetBounds(&mins, &maxs);

        Vec3lf4 tMins = (mins - ray.o) * invRayD;
        Vec3lf4 tMaxs = (maxs - ray.o) * invRayD;

        Vec3lf4 tEntries = Min(tMaxs, tMins);
        Vec3lf4 tLeave   = Max(tMins, tMaxs);

        Lane4F32 tEntry    = Max(tEntry[0], Max(tEntry[1], Max(tEntry[2], tMinEpsilon)));
        Lane4F32 tLeaveRaw = Min(tLeave[0], Max(tLeave[1], Max(tLeave[2], pos_inf)));

        const Lane4F32 tLeave = Min(tLeaveRaw, tLaneClosest);

        const Lane4F32 intersectMask = tEntry <= tLeave;
        const i32 intersectFlags     = Movemask(intersectMask);

        Lane4F32 t_dcba = Select(intersectMask, tLeaveRaw, pos_inf);
    }
}

template <i32 N, i32 types>
struct InstanceIntersector<BVHType_Quantized>
{
    using Primitive = Instance;

    bool Intersect(Ray2 &ray, Instance &instance)
    {
        AffineSpace &t = scene->affineTransforms[instance.transformIndex];
        Ray2 r         = Transform(t, ray);
        Assert(instance.geomID.GetType() == GeometryID::quadMeshType);
        BVHNodeType node = scenes[instance.geomID.GetIndex()].nodePtr;
        return Intersect<>(r, node);
    }
};

struct TriangleIntersector
{
    // https://www.graphics.cornell.edu/pubs/1997/MT97.pdf
    bool Intersect(Ray2 &ray, const Vec3lf<K> &v0, const Vec3lf<K> &v1, const Vec3lf<K> &v2)
    {
        Vec3lf<K> e1 = v1 - v0;
        Vec3lf<K> e2 = v2 - v0;
        Vec3lf<K> ng = Cross(e1, e2);
        Vec3lf<K> c  = v0 - ray.o;

        LaneF32<K> det    = Dot(-ray.d, ng);
        LaneF32<K> absDet = Abs(det);
        Vec3lf<K> dxt     = Cross(ray.d, c);

        LaneF32<K> u = Dot(dxt, e2);
        LaneF32<K> v = Dot(-dxt, e1);

        Mask<LaneF32<K>> mask = (den != LaneF32<K>(0)) & (u >= 0) & (v >= 0) & (u + v <= absDet);

        if (None(mask)) return false;

        LaneF32<K> t = Dot(ng, c);

        mask &=
    }
};

struct QuadIntersector
{
    using Primitve = Quad;

    template <i32 K>
    bool Intersect(Ray2 &ray, const Vec3lf<K> &v0, const Vec3lf<K> &v1, const Vec3lf<K> &v2, const Veclf<K> &v3)
    {
        v0, v1, v2
    }
};

} // namespace rt
