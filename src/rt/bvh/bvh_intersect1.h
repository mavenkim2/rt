#ifndef BVH_INTERSECT1_H
#define BVH_INTERSECT1_H
#include "bvh_types.h"
#include "../scene.h"

namespace rt
{

typedef u32 BVHNodeType;
enum
{
    BVHNodeType_Quantized       = 1 << 0,
    BVHNodeType_QuantizedLeaves = 1 << 1,

    BVH_QN   = BVHNodeType_Quantized,
    BVH_QNLF = BVHNodeType_QuantizedLeaves,
    BVH_AQ   = BVHNodeType_Quantized | BVHNodeType_QuantizedLeaves,
};

template <u32 K, u32 types>
auto GetNode(BVHNode4 node);

template <u32 N>
struct StackEntry
{
    BVHNode<N> ptr;
    f32 dist;
};

template <u32 K, u32 types>
struct BVHTraverser;

template <u32 types>
struct BVHTraverser<4, types>
{
    using StackEntry = StackEntry<4>;
    static void Traverse(StackEntry entry, StackEntry *stack, u32 &stackPtr, Ray2 &ray)
    {
        Vec3lf4 mins, maxs;
        // Get bounds from the node
        auto node = GetNode<4, types>(entry.ptr);
        node->GetBounds(mins.e, maxs.e);

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
            stack[stackPtr] = {node->children[(intersectFlags >> 1) - (intersectFlags >> 3)], t_dcba[0]};
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

            stack[stackPtr + ((numNodes - 1 - indexA) & 3)] = {node->children[0], t_dcba[0]};
            stack[stackPtr + ((numNodes - 1 - indexB) & 3)] = {node->children[1], t_dcba[1]};
            stack[stackPtr + ((numNodes - 1 - indexC) & 3)] = {node->children[2], t_dcba[2]};
            stack[stackPtr + ((numNodes - 1 - indexD) & 3)] = {node->children[3], t_dcba[3]};

            stackPtr += numNodes;
        }
    }
};

template <u32 K, u32 types, typename Intersector>
struct BVHIntersector;

template <u32 types, typename Intersector>
struct BVHIntersector<4, types, Intersector>
{
    using Intersection = typename Intersector::Intersection;
    BVHIntersector() {}
    static bool Intersect(Ray2 &ray, BVHNode4 bvhNode, SurfaceInteraction &itr)
    {
        // typedef typename Intersector::Primitive Primitive;

        Vec3lf4 invRayD(Select(ray.d.x == -0, pos_inf, 1 / ray.d.x),
                        Select(ray.d.y == -0, pos_inf, 1 / ray.d.y),
                        Select(ray.d.z == -0, pos_inf, 1 / ray.d.z));

        Intersection intersection;

        StackEntry stack[256];
        i32 stackPtr = 1;
        stack[0]     = {bvhNode, ray.tFar};
        while (stackPtr > 0)
        {
            Assert(stackPtr <= ArrayLength(stack));
            StackEntry entry = stack[--stackPtr];
            if (entry.dist > ray.tFar) continue;

            if (entry.ptr.IsLeaf())
            {
                Intersector::Intersect(ray, entry.ptr, intersection);
                continue;
            }

            BVHTraverser<4, types>::Traverse(entry, stack, stackPtr, ray);
        }
    }
};

template <>
auto GetNode<4, BVH_QN>(BVHNode<4> node)
{
    QuantizedNode<4> *qNode = node.GetQuantizedNode();
    return qNode;
}

template <>
auto GetNode<4, BVH_QNLF>(BVHNode<4> node)
{
    CompressedLeafNode<4> *leaf = node.GetCompressedLeaf();
    return leaf;
}

// using IntersectorTypes = TypePack<QuadIntersector, TriangleIntersector>;

#define DispatchHelp(x, ...)                                  \
    template <typename F, DispatchTmplHelper(x, __VA_ARGS__)> \
    auto Dispatch(F &&func, u32 index)                        \
    {                                                         \
        Assert(index >= 0 && index < x);                      \
        switch (index)                                        \
        {                                                     \
            DispatchSwitchHelper(x, __VA_ARGS__)              \
        }                                                     \
    }

#define COMMA                      ,
#define DispatchTmplHelper(x, ...) EXPAND(CONCAT(RECURSE__, x)(TMPL, __VA_ARGS__))
#define TMPL(x, ...)               typename CONCAT(T, x)

#define DispatchSwitchHelper(x, ...) CASES(x, __VA_ARGS__)
#define CASE(x) \
    case x: return func(CONCAT(T, x)());
#define CASES(n, ...)                EXPAND(CONCAT(RECURSE_, n)(CASE, __VA_ARGS__))
#define RECURSE_1(macro, first)      macro(first)
#define RECURSE_2(macro, first, ...) macro(first) EXPAND(RECURSE_1(macro, __VA_ARGS__))
#define RECURSE_3(macro, first, ...) macro(first) EXPAND(RECURSE_2(macro, __VA_ARGS__))
#define RECURSE_4(macro, first, ...) macro(first) EXPAND(RECURSE_3(macro, __VA_ARGS__))
#define RECURSE_5(macro, first, ...) macro(first) EXPAND(RECURSE_4(macro, __VA_ARGS__))
#define RECURSE_6(macro, first, ...) macro(first) EXPAND(RECURSE_5(macro, __VA_ARGS__))
#define RECURSE_7(macro, first, ...) macro(first) EXPAND(RECURSE_6(macro, __VA_ARGS__))

#define RECURSE__1(macro, first)      macro(first)
#define RECURSE__2(macro, first, ...) macro(first), EXPAND(RECURSE__1(macro, __VA_ARGS__))
#define RECURSE__3(macro, first, ...) macro(first), EXPAND(RECURSE__2(macro, __VA_ARGS__))
#define RECURSE__4(macro, first, ...) macro(first), EXPAND(RECURSE__3(macro, __VA_ARGS__))
#define RECURSE__5(macro, first, ...) macro(first), EXPAND(RECURSE__4(macro, __VA_ARGS__))
#define RECURSE__6(macro, first, ...) macro(first), EXPAND(RECURSE__5(macro, __VA_ARGS__))
#define RECURSE__7(macro, first, ...) macro(first), EXPAND(RECURSE__6(macro, __VA_ARGS__))

#define EXPAND(x)    x
#define CONCAT(a, b) a##b

template <typename F, typename T0>
auto Dispatch(F &&func, u32 index)
{
    Assert(index == 0);
    return func(T0());
}

DispatchHelp(2, 0, 1);
DispatchHelp(3, 0, 1, 2);
DispatchHelp(4, 0, 1, 2, 3);
DispatchHelp(5, 0, 1, 2, 3, 4);
DispatchHelp(6, 0, 1, 2, 3, 4, 5);
DispatchHelp(7, 0, 1, 2, 3, 4, 5, 6);
// DispatchHelp(4, 0, 1, 2, 3);

// template <typename F, typename T0, typename T1>
// auto Dispatch(F &&func, u32 index)
// {
//     Assert(index < 2 && index >= 0);
//     switch (index)
//     {
//         case 0: return func(T0());
//         default: return func(T1());
//     }
// }

template <u32 N, u32 types>
struct InstanceIntersector;

template <u32 K>
struct TriangleIntersection
{
    LaneF32<K> u, v, t;
    LaneU32<K> geomIDs, primIDs;
    TriangleIntersection() {}
};

template <u32 N>
struct TriangleIntersector
{
    using Intersection = TriangleIntersection<N>;
    using Primitive    = Triangle<N>;
    // https://www.graphics.cornell.edu/pubs/1997/MT97.pdf
    static Mask<LaneF32<N>> Intersect(Ray2 &ray, const Vec3lf<N> &v0, const Vec3lf<N> &v1, const Vec3lf<N> &v2, Intersection &itr)
    {
        Vec3lf<N> e1 = v1 - v0;
        Vec3lf<N> e2 = v2 - v0;
        Vec3lf<N> ng = Cross(e1, e2);
        Vec3lf<N> c  = v0 - ray.o;

        LaneF32<N> det    = Dot(ray.d, ng);
        LaneF32<N> absDet = Abs(det);
        LaneF32<N> sgnDet = Signmask(det);
        Vec3lf<N> dxt     = Cross(c, ray.d);

        LaneF32<N> u = Dot(dxt, e2) ^ sgnDet;
        LaneF32<N> v = Dot(-dxt, e1) ^ sgnDet;

        Mask<LaneF32<N>> mask = (den != LaneF32<N>(0)) & (u >= 0) & (v >= 0) & (u + v <= absDet);

        if (None(mask)) return false;

        LaneF32<N> t = Dot(ng, c) ^ sgnDet;

        mask &= (absDet * tMinEpsilon < t) & (t <= absDen * ray.tFar);

        if (None(mask)) return false;

        const LaneF32<N> rcpAbsDet = Rcp(absDet);
        itr.u                      = Min(u * rcpAbsDet, 1.f);
        itr.v                      = Min(v * rcpAbsDet, 1.f);
        itr.t                      = t * rcpAbsDet;
        return mask;
    }
    static bool Intersect(Ray2 &ray, BVHNode<N> ptr, Intersection &itr)
    {
        Assert(ptr.IsLeaf());
        Primitive *primitives = (Primitive *)ptr.GetPtr();
        u32 num               = ptr.GetNum();

        Vec3lf<N> triV0, triV1, triV2;
        Lane4F32 v0[N];
        Lane4F32 v1[N];
        Lane4F32 v2[N];
        alignas(4 * N) u32 geomIDs[N];
        alignas(4 * N) u32 primIDs[N];
        for (u32 i = 0; i < num; i++)
        {
            Primitive &prim    = primitives[i];
            TriangleMesh *mesh = &scene->Get<TriangleMesh>()[prim.geomIDs[i]];
            v0[i]              = Lane4F32::LoadU((f32 *)(mesh->p + 3 * prim.primIDs[i] + 0));
            v1[i]              = Lane4F32::LoadU((f32 *)(mesh->p + 3 * prim.primIDs[i] + 1));
            v2[i]              = Lane4F32::LoadU((f32 *)(mesh->p + 3 * prim.primIDs[i] + 2));
        }
        Transpose(v0, triV0);
        Transpose(v1, triV1);
        Transpose(v2, triV2);

        auto mask   = Intersect(ray, triV0, triV1, triV2, itr);
        itr.geomIDs = Select(mask, LaneU32<N>::LoadU(geomIDs), itr.geomIDs);
        itr.primIDs = Select(mask, LaneU32<N>::LoadU(primIDs), itr.primIDs);
    }
    static void FinalizeIntersection(Ray2 &r, SurfaceInteraction &si, Intersection &itr) { return; }
};

// struct QuadIntersector
// {
//     using Primitve = Quad;
//
//     template <i32 K>
//     bool Intersect(Ray2 &ray, const Vec3lf<K> &v0, const Vec3lf<K> &v1, const Vec3lf<K> &v2, const Veclf<K> &v3)
//     {
//         v0, v1, v2
//     }
// };

template <typename... Ts>
struct DispatchTypes
{
    template <typename F>
    __forceinline static auto Dispatch(F &&closure, u32 index)
    {
        return Dispatch<Ts...>(closure, index);
    }
};

template <u32 N>
struct InstanceIntersector<N, BVH_AQ>
{
    using IntersectorTypes = DispatchTypes<TriangleIntersector<N>>;
    using Primitive        = TLASLeaf<N>;

    InstanceIntersector() {}
    static bool Intersect(Ray2 &ray, BVHNode<N> ptr, SurfaceInteraction &itr)
    {
        // intersect the bounds
        // get the leaves to intersect
        // transform the ray
        // intersect the bottom level hierarchy
        Primitive *leaves = (Primitive *)ptr.GetType();
        u32 num           = ptr.GetNum();
        bool result       = false;
        for (u32 i = 0; i < num; i++)
        {
            Primitive &prim    = leaves[i];
            Instance &instance = scene->instances[prim.index];

            AffineSpace &t  = scene->affineTransforms[instance.transformIndex];
            BVHNode<N> node = scenes[instance.geomID.GetIndex()].nodePtr;
            Vec3f rayO      = ray.o;
            Vec3f rayD      = ray.d;
            ray             = Transform(t, ray);
            auto closure    = [&](auto type) {
                using Intersector = std::decay_t<decltype(ptr)>;
                return BVHIntersector<N, types, Intersector>::Intersect(r, node, itr);
            };
            result |= IntersectorTypes::Dispatch(closure, instance.geomID.GetType());
            ray.o = rayO;
            ray.d = rayD;
        }
    }
};

template <u32 N, typename Intersector>
struct CompressedLeafIntersector
{
    CompressedLeafIntersector() {}
    static bool Intersect(Ray2 &ray, BVHNode<N> ptr, SurfaceInteraction &itr)
    {
        bool result = false;
        switch (ptr.GetType())
        {
            case BVHNode<N>::tyCompressedLeaf:
            {
                StackEntry stack[N];
                i32 stackPtr = 0;
                BVHTraverser<N, BVH_QNLF>::Traverse({ptr, ray.tFar}, stack, stackPtr, ray);

                stackPtr--;
                for (; stackPtr >= 0; stackPtr--)
                {
                    result |= Intersector::Intersect(ray, stack[stackPtr].ptr, itr);
                }
            }
            break;
            default: result |= Intersector::Intersect(ray, ptr, itr);
        }
    }
};

template <u32 K, u32 types>
using BVHTriangleIntersector = BVHIntersector<K, types, TriangleIntersector<K>>;
typedef BVHTriangleIntersector<4, BVH_AQ> BVHTriangleIntersector4;

} // namespace rt
#endif
