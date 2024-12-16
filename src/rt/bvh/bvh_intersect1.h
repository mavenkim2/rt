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

template <i32 K, i32 types>
auto GetNode(BVHNode4 node);
template <>
auto GetNode<4, BVH_QN>(BVHNode<4> node)
{
    QuantizedNode<4> *qNode = node.GetQuantizedNode();
    return qNode;
}

template <>
auto GetNode<4, BVH_QNLF>(BVHNode<4> node)
{
    CompressedLeafNode<4> *qNode = node.GetCompressedLeaf();
    return qNode;
}

template <>
auto GetNode<4, BVH_AQ>(BVHNode<4> node)
{
    QuantizedNode<4> *qNode = node.GetQuantizedNode();
    return qNode;
}

template <typename T>
struct StackEntry
{
    T ptr;
    f32 dist;
};

template <i32 K>
struct TravRay
{
    Vec3lf<K> o;
    Vec3lf<K> invRayD;
    LaneF32<K> tFar;
    TravRay(Ray2 &r)
        : o(Vec3lf<K>(r.o)),
          invRayD(Vec3lf4(Select(r.d.x == 0, 0, 1 / r.d.x), Select(r.d.y == 0, 0, 1 / r.d.y),
                          Select(r.d.z == 0, 0, 1 / r.d.z))),
          tFar(LaneF32<K>(r.tFar))
    {
    }
};

template <i32 K, i32 types>
struct BVHTraverser;

template <i32 types>
struct BVHTraverser<4, types>
{
    using StackEntry = StackEntry<BVHNode4>;
    template <typename Node>
    static void Intersect(const Node *node, const TravRay<4> &ray, Lane4F32 &tEntryOut,
                          Lane4F32 &mask)
    {
        Vec3lf4 mins, maxs;
        // Get bounds from the node
        node->GetBounds(mins.e, maxs.e);

        // Intersect the bounds
        Vec3lf4 tMins = (mins - ray.o) * ray.invRayD;
        Vec3lf4 tMaxs = (maxs - ray.o) * ray.invRayD;

        Vec3lf4 tEntries = Min(tMaxs, tMins);
        Vec3lf4 tLeaves  = Max(tMins, tMaxs);

        Lane4F32 tEntry = Max(tEntries[0], Max(tEntries[1], Max(tEntries[2], 0)));
        Lane4F32 tLeave = Min(tLeaves[0], Min(tLeaves[1], Min(tLeaves[2], ray.tFar)));
        tEntry *= (1 - gamma(3));
        tLeave *= (1 + gamma(3));

        const Lane4F32 intersectMask = tEntry <= tLeave;

        const u32 childType0   = node->GetType(0);
        const u32 childType1   = node->GetType(1);
        const u32 childType2   = node->GetType(2);
        const u32 childType3   = node->GetType(3);
        Lane4F32 validNodeMask = Lane4U32(childType0, childType1, childType2, childType3) !=
                                 Lane4U32(BVHNode4::tyEmpty);

        tEntryOut = tEntry;
        mask      = validNodeMask & intersectMask;
    }
    static void Traverse(StackEntry entry, StackEntry *stack, i32 &stackPtr, TravRay<4> &ray)
    {
        Lane4F32 tEntry, mask;
        auto node = GetNode<4, types>(entry.ptr);
        Intersect(node, ray, tEntry, mask);

        const i32 intersectFlags = Movemask(mask);

        Lane4F32 t_dcba    = Select(mask, tEntry, pos_inf);
        const u32 numNodes = PopCount(intersectFlags);

        if (numNodes <= 1)
        {
            // If numNodes <= 1, then numNode will be 0, 1, 2, 4, or 8. x/2 - x/8 maps to
            // 0, 0, 1, 2, 3
            u32 nodeIndex   = (intersectFlags >> 1) - (intersectFlags >> 3);
            stack[stackPtr] = {node->Child(nodeIndex), t_dcba[nodeIndex]};
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

            stack[stackPtr + ((numNodes - 1 - indexA) & 3)] = {node->Child(0), t_dcba[0]};
            stack[stackPtr + ((numNodes - 1 - indexB) & 3)] = {node->Child(1), t_dcba[1]};
            stack[stackPtr + ((numNodes - 1 - indexC) & 3)] = {node->Child(2), t_dcba[2]};
            stack[stackPtr + ((numNodes - 1 - indexD) & 3)] = {node->Child(3), t_dcba[3]};

            stackPtr += numNodes;
        }
    }
    static void TraverseAny(BVHNode4 entry, BVHNode4 *stack, i32 &stackPtr, TravRay<4> &ray)
    {
        auto node = GetNode<4, types>(entry);
        Lane4F32 tEntry, mask;
        Intersect(node, ray, tEntry, mask);
        i32 intersectFlags = Movemask(mask);
        u32 oldPtr         = stackPtr;
        stackPtr += PopCount(intersectFlags);
        if (intersectFlags == 0) return;
        for (;;)
        {
            i32 bit = Bsf(intersectFlags);
            intersectFlags &= intersectFlags - 1;
            stack[oldPtr++] = node->Child(bit);
            if (intersectFlags == 0) return;
        }
    }
};

template <i32 K, i32 types, typename Intersector>
struct BVHIntersector;

template <i32 types, typename Intersector>
struct BVHIntersector<4, types, Intersector>
{
    using StackEntry = StackEntry<BVHNode4>;

    BVHIntersector() {}
    static bool Intersect(Ray2 &ray, BVHNode4 bvhNode, SurfaceInteraction &itr)
    {
        // typedef typename Intersector::Primitive Primitive;

        TravRay<4> r(ray);

        StackEntry stack[256];
        i32 stackPtr = 1;
        stack[0]     = {bvhNode, ray.tFar};
        bool result  = false;
        Intersector intersector;
        while (stackPtr > 0)
        {
            Assert(stackPtr <= ArrayLength(stack));
            StackEntry entry = stack[--stackPtr];
            Assert(entry.ptr.data);
            if (entry.dist > ray.tFar) continue;

            if (entry.ptr.IsLeaf())
            {
                result |= intersector.Intersect(ray, entry.ptr, itr, r);
                r.tFar = ray.tFar;
                continue;
            }

            BVHTraverser<4, types>::Traverse(entry, stack, stackPtr, r);
        }
        return result;
    }
    static bool Occluded(Ray2 &ray, BVHNode4 bvhNode)
    {
        TravRay<4> r(ray);

        BVHNode4 stack[256];
        i32 stackPtr = 1;
        stack[0]     = bvhNode;
        bool result  = false;
        Intersector intersector;
        while (stackPtr > 0)
        {
            Assert(stackPtr <= ArrayLength(stack));
            BVHNode4 entry = stack[--stackPtr];

            if (entry.IsLeaf())
            {
                SurfaceInteraction itr;
                // TODO: implement Occluded for intersectors
                if (intersector.Intersect(ray, entry, itr, r)) return true;
                continue;
            }

            BVHTraverser<4, types>::TraverseAny(entry, stack, stackPtr, r);
        }
        return false;
    }
};

template <i32 N>
struct InstanceIntersector;

template <i32 K>
struct TriangleIntersection
{
    LaneF32<K> u, v, t;
    LaneU32<K> geomIDs = {};
    LaneU32<K> primIDs = {};
    TriangleIntersection() {}
};

template <i32 N, typename T>
struct TriangleIntersectorBase;

template <i32 N, template <i32> class Prim>
struct TriangleIntersectorBase<N, Prim<N>>
{
    using Primitive           = Prim<N>;
    TriangleIntersectorBase() = default;
    // https://www.graphics.cornell.edu/pubs/1997/MT97.pdf
    static Mask<LaneF32<N>> Intersect(const Ray2 &ray, const LaneF32<N> &tFar,
                                      const Vec3lf<N> &v0, const Vec3lf<N> &v1,
                                      const Vec3lf<N> &v2, TriangleIntersection<N> &itr)
    {
        Vec3lf<N> o  = Vec3lf<N>(ray.o);
        Vec3lf<N> d  = Vec3lf<N>(ray.d);
        Vec3lf<N> e1 = v0 - v1;
        Vec3lf<N> e2 = v2 - v0;
        Vec3lf<N> ng = Cross(e2, e1); // 4
        Vec3lf<N> c  = v0 - o;

        LaneF32<N> det    = Dot(d, ng); // 7
        LaneF32<N> absDet = Abs(det);
        LaneF32<N> sgnDet;
        if constexpr (N == 1) sgnDet = det > 0 ? 1.f : -1.f;
        else sgnDet = Signmask(det);

        Vec3lf<N> dxt = Cross(c, d); // 3

        LaneF32<N> u = Dot(dxt, e2); // 3 1 -> 7
        LaneF32<N> v = Dot(dxt, e1); // 3 1 -> 7
        if constexpr (N == 1)
        {
            u *= sgnDet;
            v *= sgnDet;
        }
        else
        {
            u ^= sgnDet;
            v ^= sgnDet;
        }

        Mask<LaneF32<N>> mask =
            (det != LaneF32<N>(zero)) & (u >= 0) & (v >= 0) & (u + v <= absDet);

        if (None(mask)) return false;

        LaneF32<N> t = Dot(ng, c); // 7

        if constexpr (N == 1) t *= sgnDet;
        else t ^= sgnDet;
        mask &= (0 < t) & (t <= absDet * tFar);

        if (None(mask)) return false;

#if 1
        LaneF32<N> maxX = Max(Abs(e1.x), Abs(e2.x));
        LaneF32<N> maxY = Max(Abs(e1.y), Abs(e2.y));
        LaneF32<N> maxZ = Max(Abs(e1.z), Abs(e2.z));

        LaneF32<N> errorX = maxX * gamma(1);
        LaneF32<N> errorY = maxY * gamma(1);
        LaneF32<N> errorZ = maxZ * gamma(1);
        LaneF32<N> deltaX = 2 * (gamma(2) * maxY * maxZ + maxY * errorZ + maxZ * errorY);
        LaneF32<N> deltaY = 2 * (gamma(2) * maxZ * maxX + maxZ * errorX + maxX * errorZ);
        LaneF32<N> deltaZ = 2 * (gamma(2) * maxX * maxY + maxX * errorY + maxY * errorX);

        LaneF32<N> deltaT =
            absDet * (gamma(3) * t + deltaX * c.x + deltaY * c.y * deltaZ * c.z +
                      gamma(1) * (ng.x + ng.y + ng.z)); // + ng.x * c.x * gamma(1);

        mask &= t > deltaT;

        if (None(mask)) return false;
#endif

        const LaneF32<N> rcpAbsDet = Rcp(absDet);
        itr.u                      = Min(u * rcpAbsDet, 1.f);
        itr.v                      = Min(v * rcpAbsDet, 1.f);
        itr.t                      = t * rcpAbsDet;
        return mask;
    }
    static bool Intersect(Ray2 &ray, Primitive *primitives, u32 num, SurfaceInteraction &si,
                          const Mask<LaneF32<N>> &validMask = Mask<LaneF32<N>>(true))
    {
        Vec3lf<N> triV0, triV1, triV2;
        Lane4F32 v0[N];
        Lane4F32 v1[N];
        Lane4F32 v2[N];
        alignas(4 * N) u32 geomIDs[N];
        alignas(4 * N) u32 primIDs[N];

        Mask<LaneF32<N>> outMask(false);

        TriangleIntersection<N> itr;
        itr.t = LaneF32<N>(ray.tFar);
        for (u32 i = 0; i < num; i++)
        {
            TriangleIntersection<N> triItr;
            Primitive &prim = primitives[i];
            prim.GetData(v0, v1, v2, geomIDs, primIDs);
            // prim.geomID
            Transpose<N>(v0, triV0);
            Transpose<N>(v1, triV1);
            Transpose<N>(v2, triV2);

            Mask<LaneF32<N>> mask = Intersect(ray, itr.t, triV0, triV1, triV2, triItr);
            itr.u                 = Select(mask, triItr.u, itr.u);
            itr.v                 = Select(mask, triItr.v, itr.v);
            itr.t                 = Select(mask, triItr.t, itr.t);
            itr.geomIDs           = AsUInt(
                Select(mask, AsFloat(MemSimdU32<N>::LoadU(geomIDs)), AsFloat(itr.geomIDs)));
            itr.primIDs = AsUInt(
                Select(mask, AsFloat(MemSimdU32<N>::LoadU(primIDs)), AsFloat(itr.primIDs)));
            outMask |= mask;
        }
        outMask &= validMask;
        if (Any(outMask))
        {
            u32 maskBits = Movemask(outMask);
            f32 tFar     = ReduceMin(itr.t);
            ray.tFar     = tFar;
            u32 index;
            for (;;)
            {
                if (maskBits == 0) return false;
                u32 i = Bsf(maskBits);
                maskBits &= maskBits - 1;
                if (tFar == Get(itr.t, i))
                {
                    index = i;
                    break;
                }
            }
            f32 u = Get(itr.u, index);
            f32 v = Get(itr.v, index);
            f32 w = 1 - u - v;
            // Assert(u >= 0 && u <= 1 && v >= 0 && v <= 1 && w >= 0 && w <= 1);
            // TODO: calculate partial derivatives, shading normal, uv etcs from below
            Scene2 *scene = GetScene();

            TriangleMesh *mesh = &scene->triangleMeshes[Get(itr.geomIDs, index)];
            u32 primID         = Get(itr.primIDs, index);

            u32 vertexIndices[3];
            if (mesh->indices)
            {
                vertexIndices[0] = mesh->indices[3 * primID + 0];
                vertexIndices[1] = mesh->indices[3 * primID + 1];
                vertexIndices[2] = mesh->indices[3 * primID + 2];
            }
            else
            {
                vertexIndices[0] = 3 * primID + 0;
                vertexIndices[1] = 3 * primID + 1;
                vertexIndices[2] = 3 * primID + 2;
            }
            Vec2f uv[3];
            if (mesh->uv)
            {
                uv[0] = mesh->uv[vertexIndices[0]];
                uv[1] = mesh->uv[vertexIndices[1]];
                uv[2] = mesh->uv[vertexIndices[2]];
            }
            else
            {
                uv[0] = Vec2f(0, 0);
                uv[1] = Vec2f(1, 0);
                uv[2] = Vec2f(1, 1);
            }
            Vec3f p[3] = {mesh->p[vertexIndices[0]], mesh->p[vertexIndices[1]],
                          mesh->p[vertexIndices[2]]};

            si.p = w * p[0] + u * p[1] + v * p[2];
            // TODO: still iffy about this
            si.pError = gamma(6) * (Abs(w * p[0]) + Abs(u * p[1]) + Abs(v * p[2]));

            Vec3f dp02  = p[0] - p[2];
            Vec3f dp12  = p[1] - p[2];
            si.n        = Normalize(Cross(dp02, dp12)); // p[1] - p[0], p[2] - p[0]));
            Vec2f duv02 = uv[0] - uv[2];
            Vec2f duv12 = uv[1] - uv[2];
            f32 det     = FMS(duv02[0], duv12[1], duv02[1] * duv12[0]);
            Vec3f dpdu, dpdv;
            if (det < 1e-9f)
            {
                CoordinateSystem(si.n, &dpdu, &dpdv);
            }
            else
            {
                f32 invDet = 1 / det;
                dpdu       = FMS(Vec3f(duv12[1]), dp02, duv02[1] * dp12) * invDet;
                dpdv       = FMS(Vec3f(duv02[0]), dp12, duv12[0] * dp02) * invDet;
            }

            if (mesh->n)
            {
                si.shading.n = w * mesh->n[vertexIndices[0]] + u * mesh->n[vertexIndices[1]] +
                               v * mesh->n[vertexIndices[2]];
                si.shading.n =
                    LengthSquared(si.shading.n) > 0 ? Normalize(si.shading.n) : si.n;
            }
            else
            {
                si.shading.n = si.n;
            }
            Vec3f ss = dpdu;
            Vec3f ts = Cross(si.shading.n, ss);
            if (LengthSquared(ts) > 0)
            {
                ss = Cross(ts, si.shading.n);
            }
            else
            {
                CoordinateSystem(si.shading.n, &ss, &ts);
            }

            si.shading.dpdu = ss;
            si.shading.dpdv = ts;
            si.uv           = w * uv[0] + u * uv[1] + v * uv[2];
            si.tHit         = tFar;
            // TODO: get index based on type
            const Scene2::PrimitiveIndices *indices =
                &scene->primIndices[Get(itr.geomIDs, index)];
            si.materialIDs = indices->materialID.data;
            // TODO: properly obtain the light handle
            si.lightIndices = 0;
            // TODO: only for moana is this true
            si.faceIndices = primID / 2;
            return true;
        }
        return false;
    }
    template <i32 K>
    static bool Intersect(Ray2 &ray, BVHNode<K> ptr, SurfaceInteraction &si, TravRay<K> &)
    {
        Assert(ptr.IsLeaf());
        Primitive *primitives = (Primitive *)ptr.GetPtr();
        u32 num               = ptr.GetNum();
        return Intersect(ray, primitives, num, si);
    }
};

// struct QuadIntersector
// {
//     using Primitve = Quad;
//
//     static bool Intersect(Ray2 &ray, Primitive *primitives, u32 num, SurfaceInteraction &si)
//     {
//         Vec3lf<N> triV0, triV1, triV2, triV3;
//         Lane4F32 v0[N];
//         Lane4F32 v1[N];
//         Lane4F32 v2[N];
//         Lane4F32 v3[N];
//         alignas(4 * N) u32 geomIDs[N];
//         alignas(4 * N) u32 primIDs[N];
//
//         LaneF32<N> t(ray.tFar);
//         Mask<LaneF32<N>> outMask(false);
//
//         TriangleIntersection<N> itr;
//         for (u32 i = 0; i < num; i++)
//         {
//             Primitive &prim = primitives[i];
//             prim.GetData(v0, v1, v2, v3, geomIDs, primIDs);
//             // prim.geomID
//             Transpose<N>(v0, triV0);
//             Transpose<N>(v1, triV1);
//             Transpose<N>(v2, triV2);
//             Transpose<N>(v3, triV3);
//
//             Mask<LaneF32<N>> mask = Intersect(ray, t, triV0, triV1, triV2, itr);
//             itr.geomIDs           = Select(mask, MemSimdU32<N>::LoadU(geomIDs),
//             itr.geomIDs); itr.primIDs           = Select(mask,
//             MemSimdU32<N>::LoadU(primIDs), itr.primIDs); t                     =
//             Select(mask, itr.t, t); outMask |= mask;
//         }
//     }
//
//     template <i32 K>
//     static bool Intersect(Ray2 &ray, BVHNode<K> ptr, SurfaceInteraction &si, TravRay<K> &)
//     {
//         Assert(ptr.IsLeaf());
//         Primitive *primitives = (Primitive *)ptr.GetPtr();
//         u32 num               = ptr.GetNum();
//         return Intersect(ray, primitives, num, si);
//     }
// };

template <i32 N>
using TriangleIntersector = TriangleIntersectorBase<N, Triangle<N>>;

template <i32 N>
using TriangleIntersectorCmp = TriangleIntersectorBase<N, TriangleCompressed<N>>;

template <typename... Ts>
struct DispatchTypes
{
    template <typename F>
    __forceinline static auto Dispatch(F &&closure, u32 index)
    {
        return Dispatch<Ts...>(closure, index);
    }
};

template <typename F, typename... Ts>
auto Dispatch(F &&closure, TypePack<Ts...>, u32 index)
{
    Dispatch<F, Ts...>(std::move(closure), index);
}

template <i32 N>
struct InstanceIntersector
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
        Primitive *leaves = (Primitive *)ptr.GetPtr();
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

template <i32 N, typename Intersector>
struct CompressedLeafIntersector
{
    using StackEntry = StackEntry<BVHNode<N>>;
    using Primitive  = typename Intersector::Primitive;
    Intersector intersector;
    CompressedLeafIntersector() {}
    bool Intersect(Ray2 &ray, BVHNode<N> ptr, SurfaceInteraction &itr, TravRay<N> &tray)
    {
        bool result = false;
        switch (ptr.GetType())
        {
            case BVHNode<N>::tyCompressedLeaf:
            {
                StackEntry stack[N];
                i32 stackPtr = 0;
                BVHTraverser<N, BVH_QNLF>::Traverse({ptr, ray.tFar}, stack, stackPtr, tray);

                stackPtr--;
                CompressedLeafNode<N> *node = ptr.GetCompressedLeaf();
                for (; stackPtr >= 0; stackPtr--)
                {
                    // TODO: this is kind of hacky
                    StackEntry entry = stack[stackPtr];
                    if (entry.dist > ray.tFar) continue;
                    uintptr_t child       = entry.ptr.data;
                    u32 start             = (child == 0 ? 0 : node->offsets[child - 1]);
                    Primitive *primitives = (Primitive *)(node + 1) + start;
                    u32 num               = node->offsets[child] - start;

                    result |= intersector.Intersect(ray, primitives, num, itr);
                }
            }
            break;
            default: result |= intersector.Intersect(ray, ptr, itr, tray);
        }
        return result;
    }
};

template <typename T, typename U>
struct QueueIntersector;

template <typename Intersector, template <i32> class Prim>
struct QueueIntersector<Intersector, Prim<1>>
{
    using Primitive  = Prim<1>;
    using PrimitiveN = Prim<8>;

    QueueIntersector() {}
    bool Intersect(Ray2 &ray, Primitive *primitives, u32 num, SurfaceInteraction &si)
    {
        Assert(num < 8);
        bool result = false;

        PrimitiveN prim;
        prim.Fill(primitives, num);
        result |= Intersector::Intersect(ray, &prim, 1, si, Lane8F32::Mask((1u << num) - 1u));
        return result;
    }
    template <i32 N>
    bool Intersect(Ray2 &ray, BVHNode<N> ptr, SurfaceInteraction &si, TravRay<N> &tray)
    {
        Assert(ptr.IsLeaf());
        Primitive *primitives = (Primitive *)ptr.GetPtr();
        u32 num               = ptr.GetNum();
        return Intersect(ray, primitives, num, si);
    }
};

typedef QueueIntersector<TriangleIntersector<8>, Triangle<1>> TriangleQueueIntersector;
typedef QueueIntersector<TriangleIntersectorCmp<8>, TriangleCompressed<1>>
    TriangleCmpQueueIntersector;

template <i32 K, i32 N, i32 types>
using BVHTriangleIntersector = BVHIntersector<K, types, TriangleIntersector<N>>;
typedef BVHTriangleIntersector<4, 1, BVH_AQ> BVH4TriangleIntersector1;

template <i32 K, i32 N, i32 types>
using BVHTriangleIntersectorCmp = BVHIntersector<K, types, TriangleIntersectorCmp<N>>;
typedef BVHTriangleIntersectorCmp<4, 1, BVH_AQ> BVH4TriangleIntersectorCmp1;

// NOTE: K is for BVH branching factor, N is for Primitive
template <i32 K, i32 N, i32 types>
using BVHTriangleCLIntersector =
    BVHIntersector<K, types, CompressedLeafIntersector<K, TriangleIntersector<N>>>;
typedef BVHTriangleCLIntersector<4, 1, BVH_AQ> BVH4TriangleCLIntersector1;

template <i32 K, i32 N, i32 types>
using BVHTriangleCLIntersectorCmp =
    BVHIntersector<K, types, CompressedLeafIntersector<K, TriangleIntersectorCmp<N>>>;
typedef BVHTriangleCLIntersectorCmp<4, 1, BVH_AQ> BVH4TriangleCLIntersectorCmp1;

// Queue intersectors
template <i32 K, i32 types>
using BVHTriangleCLQueueIntersectorCmp =
    BVHIntersector<K, types, CompressedLeafIntersector<K, TriangleCmpQueueIntersector>>;
typedef BVHTriangleCLQueueIntersectorCmp<4, BVH_AQ> BVH4TriangleCLQueueIntersectorCmp8;

template <i32 K, i32 types>
using BVHTriangleCLQueueIntersector =
    BVHIntersector<K, types, CompressedLeafIntersector<K, TriangleQueueIntersector>>;
typedef BVHTriangleCLQueueIntersector<4, BVH_AQ> BVH4TriangleCLQueueIntersector8;

} // namespace rt
#endif
