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

template <i32 types, i32 K>
void GetBounds(BVHNode4 node, Vec3lf4 &mins, Vec3lf4 &maxs);

template <i32 types, i32 K, typename Intersector>
bool Intersect(Ray2 &ray, BVHNode<K> bvhNode, SurfaceInteraction &itr);

// Single primitive intersector
template <i32 types, typename Intersector>
bool Intersect<types, 4, Intersector>(Ray2 &ray, BVHNode4 bvhNode, SurfaceInteraction &itr)
{
    typedef typename Intersector::Primitive Primitive;

    Vec3lf4 invRayD(Select(ray.d.x == -0, pos_inf, 1 / ray.d.x),
                    Select(ray.d.y == -0, pos_inf, 1 / ray.d.y),
                    Select(ray.d.z == -0, pos_inf, 1 / ray.d.z));

    u32 closestHitPrimitiveType;
    u32 closestHitPrimitiveIndex;

    struct StackEntry
    {
        BVHNode4 ptr;
        f32 dist;
    };
    StackEntry stack[64];
    i32 stackPtr = 1;
    stack[0]     = {bvhNode, pos_inf};
    while (stackPtr > 0)
    {
        StackEntry &entry = stack[--stackPtr];
        if (entry.dist > ray.tFar) continue;

        if (entry.ptr.IsLeaf())
        {
            Intersector::Intersect(ray, stack.ptr, itr);
            continue;
        }

        Vec3lf4 mins, maxs;
        // Get bounds from the node
        GetBounds<types, 4>(node, &mins, &maxs);

        // Intersect the bounds
        Vec3lf4 tMins = (mins - ray.o) * invRayD;
        Vec3lf4 tMaxs = (maxs - ray.o) * invRayD;

        Vec3lf4 tEntries = Min(tMaxs, tMins);
        Vec3lf4 tLeave   = Max(tMins, tMaxs);

        Lane4F32 tEntry    = Max(tEntry[0], Max(tEntry[1], Max(tEntry[2], tMinEpsilon)));
        Lane4F32 tLeaveRaw = Min(tLeave[0], Min(tLeave[1], Min(tLeave[2], pos_inf)));

        const Lane4F32 tLeave = Min(tLeaveRaw, tLaneClosest);

        const Lane4F32 intersectMask = tEntry <= tLeave;

        const u32 childType0   = node->children[0].GetType();
        const u32 childType1   = node->children[1].GetType();
        const u32 childType2   = node->children[2].GetType();
        const u32 childType3   = node->children[3].GetType();
        Lane4F32 validNodeMask = Lane4U32(childType0, childType1, childType, childType3) != Lane4U32(BVHNode::tyEmpty);

        Lane4F32 mask            = validNodeMask & intersectMask;
        const i32 intersectFlags = Movemask(mask);

        Lane4F32 t_dcba    = Select(mask, tLeaveRaw, pos_inf);
        const u32 numNodes = PopCount(intersectFlags);

        if (numNodes <= 1)
        {
            // If numNodes <= 1, then numNode will be 0, 1, 2, 4, or 8. x/2 - x/8 maps to
            // 0, 0, 1, 2, 3
            stack[stackPtr] = node.childIndex[(intersectFlags >> 1) - (intersectFlags >> 3)];
            stackPtr += numNodes;
        }
        else
        {
            // Branchless adding leaf nodes
            const Lane4F32 abac = ShuffleReverse<0, 1, 0, 2>(t_dcba);
            const Lane4F32 adcd = ShuffleReverse<0, 3, 2, 3>(t_dcba);

            const u32 da_cb_ba_ac = Movemask(t_dcba < abac) & 0xe;
            const u32 aa_db_ca_dc = Movemask(adcd < abac);

            u32 da_cb_ba_db_ca_dc = da_cb_ba_ac * 4 + aa_db_ca_dc;

            u32 indexA = PopCount(da_cb_ba_db_ca_dc & 0x2a);
            u32 indexB = PopCount((da_cb_ba_db_ca_dc ^ 0x08) & 0x1c);
            u32 indexC = PopCount((da_cb_ba_db_ca_dc ^ 0x12) & 0x13);
            u32 indexD = PopCount((~da_cb_ba_db_ca_dc) & 0x25);

            stack[stackPtr + ((numNodes - 1 - indexA) & 3)] = node.childIndex[0];
            stack[stackPtr + ((numNodes - 1 - indexB) & 3)] = node.childIndex[1];
            stack[stackPtr + ((numNodes - 1 - indexC) & 3)] = node.childIndex[2];
            stack[stackPtr + ((numNodes - 1 - indexD) & 3)] = node.childIndex[3];

            stackPtr += numNodes;
        }
    }
}

template <i32 K>
void GetBounds<BVHType_Quantized, K>(BVHNode4 node, Vec3lf<K> &mins, Vec3lf<K> &maxs)
{
    QuantizedNode<K> *qNode = node.GetQuantizedNode();
    node->GetBounds(mins, maxs);
}

template <i32 N, i32 types>
struct InstanceIntersector<BVHType_Quantized>
{
    using Primitive = Instance;

    bool Intersect(Ray2 &ray, BVHNode<N> ptr, SurfaceInteraction &itr)
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
    template <i32 K>
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
