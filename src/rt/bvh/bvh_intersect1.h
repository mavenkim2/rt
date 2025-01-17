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
auto GetNode(BVHNode<K> node);
template <>
auto GetNode<4, BVH_QN>(BVHNode<4> node)
{
    Assert(node.IsQuantizedNode());
    QuantizedNode<4> *qNode = node.GetQuantizedNode();
    return qNode;
}

template <>
auto GetNode<4, BVH_QNLF>(BVHNode<4> node)
{
    Assert(node.IsCompressedLeaf());
    CompressedLeafNode<4> *qNode = node.GetCompressedLeaf();
    return qNode;
}

template <>
auto GetNode<4, BVH_AQ>(BVHNode<4> node)
{
    Assert(node.IsQuantizedNode());
    QuantizedNode<4> *qNode = node.GetQuantizedNode();
    return qNode;
}

template <>
auto GetNode<8, BVH_QN>(BVHNode<8> node)
{
    Assert(node.IsQuantizedNode());
    QuantizedNode<8> *qNode = node.GetQuantizedNode();
    return qNode;
}

template <>
auto GetNode<8, BVH_QNLF>(BVHNode<8> node)
{
    Assert(node.IsCompressedLeaf());
    CompressedLeafNode<8> *qNode = node.GetCompressedLeaf();
    return qNode;
}

template <>
auto GetNode<8, BVH_AQ>(BVHNode<8> node)
{
    Assert(node.IsQuantizedNode());
    QuantizedNode<8> *qNode = node.GetQuantizedNode();
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

template <i32 types>
struct BVHTraverser<8, types>
{
    using StackEntry = StackEntry<BVHNode8>;
    template <typename Node>
    static void Intersect(const Node *node, const TravRay<8> &ray, Lane8F32 &tEntryOut,
                          Lane8F32 &mask)
    {
        Vec3lf8 mins, maxs;
        // Get bounds from the node
        node->GetBounds(mins.e, maxs.e);

        // Intersect the bounds
        Vec3lf8 tMins = (mins - ray.o) * ray.invRayD;
        Vec3lf8 tMaxs = (maxs - ray.o) * ray.invRayD;

        Vec3lf8 tEntries = Min(tMaxs, tMins);
        Vec3lf8 tLeaves  = Max(tMins, tMaxs);

        Lane8F32 tEntry = Max(tEntries[0], Max(tEntries[1], Max(tEntries[2], 0)));
        Lane8F32 tLeave = Min(tLeaves[0], Min(tLeaves[1], Min(tLeaves[2], ray.tFar)));
        tEntry *= (1 - gamma(3));
        tLeave *= (1 + gamma(3));

        const Lane8F32 intersectMask = tEntry <= tLeave;

        const u32 childType0 = node->GetType(0);
        const u32 childType1 = node->GetType(1);
        const u32 childType2 = node->GetType(2);
        const u32 childType3 = node->GetType(3);
        const u32 childType4 = node->GetType(4);
        const u32 childType5 = node->GetType(5);
        const u32 childType6 = node->GetType(6);
        const u32 childType7 = node->GetType(7);
        Lane8F32 validNodeMask =
            Lane8U32(childType0, childType1, childType2, childType3, childType4, childType5,
                     childType6, childType7) != Lane8U32(BVHNode8::tyEmpty);

        tEntryOut = tEntry;
        mask      = validNodeMask & intersectMask;
    }
    static void Traverse(StackEntry entry, StackEntry *stack, i32 &stackPtr, TravRay<8> &ray)
    {
        Lane8F32 tEntry, mask;
        auto node = GetNode<8, types>(entry.ptr);
        Intersect(node, ray, tEntry, mask);

        const i32 intersectFlags = Movemask(mask);

        Lane8F32 t_hgfedcba = Select(mask, tEntry, pos_inf);
        const u32 numNodes  = PopCount(intersectFlags);

        if (numNodes <= 1)
        {
            // If numNodes <= 1, then numNode will be 0, 1, 2, 4, or 8. x/2 - x/8 maps to
            // 0, 0, 1, 2, 3
            u32 nodeIndex   = (intersectFlags >> 1) - (intersectFlags >> 3);
            stack[stackPtr] = {node->Child(nodeIndex), t_hgfedcba[nodeIndex]};
            stackPtr += numNodes;
        }
        else
        {
            // Branchless adding leaf nodes
            Lane8F32 t_aaaaaaaa = Shuffle<0>(t_hgfedcba);
            Lane8F32 t_edbcbbca = ShuffleReverse<4, 3, 1, 2, 1, 1, 2, 0>(t_hgfedcba);
            Lane8F32 t_gfcfeddb = ShuffleReverse<6, 5, 2, 5, 4, 3, 3, 1>(t_hgfedcba);
            Lane8F32 t_hhhgfgeh = ShuffleReverse<7, 7, 7, 6, 5, 6, 4, 7>(t_hgfedcba);

            const u32 mask0 = Movemask(t_aaaaaaaa < t_gfcfeddb);
            const u32 mask1 = Movemask(t_edbcbbca < t_gfcfeddb);
            const u32 mask2 = Movemask(t_edbcbbca < t_hhhgfgeh);
            const u32 mask3 = Movemask(t_gfcfeddb < t_hhhgfgeh);

            const u32 nodeMask = mask0 | (mask1 << 8) | (mask2 << 16) | (mask3 << 24);

            u32 indexA = PopCount(~nodeMask & 0x000100ed);
            u32 indexB = PopCount((nodeMask ^ 0x002c2c00) & 0x002c2d00);
            u32 indexC = PopCount((nodeMask ^ 0x20121200) & 0x20123220);
            u32 indexD = PopCount((nodeMask ^ 0x06404000) & 0x06404602);
            u32 indexE = PopCount((nodeMask ^ 0x08808000) & 0x0a828808);
            u32 indexF = PopCount((nodeMask ^ 0x50000000) & 0x58085010);
            u32 indexG = PopCount((nodeMask ^ 0x80000000) & 0x94148080);
            u32 indexH = PopCount(nodeMask & 0xe0e10000);

            stack[stackPtr + ((numNodes - 1 - indexA) & 7)] = {node->Child(0), t_hgfedcba[0]};
            stack[stackPtr + ((numNodes - 1 - indexB) & 7)] = {node->Child(1), t_hgfedcba[1]};
            stack[stackPtr + ((numNodes - 1 - indexC) & 7)] = {node->Child(2), t_hgfedcba[2]};
            stack[stackPtr + ((numNodes - 1 - indexD) & 7)] = {node->Child(3), t_hgfedcba[3]};

            stack[stackPtr + ((numNodes - 1 - indexA) & 7)] = {node->Child(4), t_hgfedcba[4]};
            stack[stackPtr + ((numNodes - 1 - indexB) & 7)] = {node->Child(5), t_hgfedcba[5]};
            stack[stackPtr + ((numNodes - 1 - indexC) & 7)] = {node->Child(6), t_hgfedcba[6]};
            stack[stackPtr + ((numNodes - 1 - indexD) & 7)] = {node->Child(7), t_hgfedcba[7]};

            stackPtr += numNodes;
        }
    }
    static void TraverseAny(BVHNode8 entry, BVHNode8 *stack, i32 &stackPtr, TravRay<8> &ray)
    {
        auto node = GetNode<8, types>(entry);
        Lane8F32 tEntry, mask;
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

template <i32 K, i32 types, typename Intersector>
struct BVHIntersector
{
    using StackEntry = StackEntry<BVHNode<K>>;

    BVHIntersector() {}
    static bool Intersect(ScenePrimitives *scene, BVHNode<K> nodePtr, Ray2 &ray,
                          SurfaceInteraction &itr)
    {
        // typedef typename Intersector::Primitive Primitive;
        TravRay<K> r(ray);

        StackEntry stack[K == 4 ? 256 : 512];
        i32 stackPtr = 1;
        stack[0]     = {nodePtr, ray.tFar};
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
                result |= intersector.Intersect(scene, ray, entry.ptr, itr, r);
                r.tFar = ray.tFar;
                continue;
            }

            BVHTraverser<K, types>::Traverse(entry, stack, stackPtr, r);
        }
        return result;
    }
    static bool Occluded(ScenePrimitives *scene, BVHNode<K> nodePtr, Ray2 &ray)
    {
        TravRay<K> r(ray);

        BVHNode<K> stack[K == 4 ? 256 : 512];
        i32 stackPtr = 1;
        stack[0]     = nodePtr;
        bool result  = false;
        Intersector intersector;
        while (stackPtr > 0)
        {
            Assert(stackPtr <= ArrayLength(stack));
            BVHNode<K> entry = stack[--stackPtr];

            if (entry.IsLeaf())
            {
                SurfaceInteraction itr;
                // TODO: implement Occluded for intersectors
                if (intersector.Intersect(scene, ray, entry, itr, r)) return true;
                continue;
            }

            BVHTraverser<K, types>::TraverseAny(entry, stack, stackPtr, r);
        }
        return false;
    }
};

template <i32 K>
struct TriangleIntersection
{
    LaneF32<K> u, v, t;
    LaneU32<K> geomIDs = {};
    LaneU32<K> primIDs = {};
    TriangleIntersection() {}
};

Vec4f BsplineBasis(f32 u)
{
    f32 u2 = u * u;
    f32 u3 = u2 * u;
    return Vec4f(1.f / 6.f * (-u3 + 3.f * u2 - 3.f * u + 1),
                 1.f / 6.f * (3.f * u3 - 6.f * u2 + 4.f),
                 1.f / 6.f * (-3 * u3 + 3 * u2 + 3 * u + 1), 1.f / 6.f * u3);
}

Vec4f BsplineDerivativeBasis(f32 u)
{
    f32 u2 = u * u;
    return Vec4f(1.f / 6.f * (-3.f * u2 + 6.f * u - 3.f), 1.f / 6.f * (9.f * u2 - 12.f * u),
                 1.f / 6.f * (-9.f * u2 + 6.f * u + 3.f), 0.5f * u2);
}

// f32 CalculateCatmullClarkCurvature(const Vec3f *points, const Vec2f &uv)
f32 CalculateCurvature(const Vec3f &dpdu, const Vec3f &dpdv, const Vec3f &dndu,
                       const Vec3f &dndv)
{
    f32 E = Dot(dpdu, dpdu);
    f32 F = Dot(dpdu, dpdv);
    f32 G = Dot(dpdv, dpdv);

    f32 e = Dot(-dndu, dpdu);
    f32 f = Dot(-dndv, dpdu);
    f32 g = Dot(-dndv, dpdv);

    f32 curvature = E * g - 2.f * F * f + G * e / (2.f * (E * G - Sqr(F)));
    return curvature;
}

template <i32 N>
static Mask<LaneF32<N>> TriangleIntersect(const Ray2 &ray, const LaneF32<N> &tFar,
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

    LaneF32<N> deltaT = absDet * (gamma(3) * t + deltaX * c.x + deltaY * c.y * deltaZ * c.z +
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

static void SurfaceInteractionFromTriangleIntersection(ScenePrimitives *scene,
                                                       const u32 geomID, const u32 primID,
                                                       const u32 ids[3], f32 u, f32 v, f32 w,
                                                       SurfaceInteraction &si,
                                                       bool isSecondTri = false)
{
    Mesh *mesh = (Mesh *)scene->primitives + geomID;
    u32 vertexIndices[3];
    if (mesh->indices)
    {
        vertexIndices[0] = mesh->indices[ids[0]];
        vertexIndices[1] = mesh->indices[ids[1]];
        vertexIndices[2] = mesh->indices[ids[2]];
    }
    else
    {
        vertexIndices[0] = ids[0];
        vertexIndices[1] = ids[1];
        vertexIndices[2] = ids[2];
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
        if (isSecondTri)
        {
            uv[0] = Vec2f(0, 0);
            uv[1] = Vec2f(1, 1);
            uv[2] = Vec2f(0, 1);
        }
        else
        {
            uv[0] = Vec2f(0, 0);
            uv[1] = Vec2f(1, 0);
            uv[2] = Vec2f(1, 1);
        }
    }
    Vec3f p[3] = {mesh->p[vertexIndices[0]], mesh->p[vertexIndices[1]],
                  mesh->p[vertexIndices[2]]};

    si.p      = u * p[1] + v * p[2] + w * p[0];
    si.pError = gamma(6) * (Abs(u * p[1]) + Abs(v * p[2]) + Abs(w * p[0]));

    Vec3f dp02 = p[0] - p[2];
    Vec3f dp12 = p[1] - p[2];
    si.n       = Normalize(Cross(dp02, dp12));
    Vec3f dn02 = {};
    Vec3f dn12 = {};

    if (mesh->n)
    {
        si.shading.n = u * mesh->n[vertexIndices[1]] + v * mesh->n[vertexIndices[2]] +
                       w * mesh->n[vertexIndices[0]];
        si.shading.n = LengthSquared(si.shading.n) > 0 ? Normalize(si.shading.n) : si.n;

        dn02 = mesh->n[vertexIndices[0]] - mesh->n[vertexIndices[2]];
        dn12 = mesh->n[vertexIndices[1]] - mesh->n[vertexIndices[2]];
    }
    else
    {
        si.shading.n = si.n;
    }

    Vec2f duv02 = uv[0] - uv[2];
    Vec2f duv12 = uv[1] - uv[2];
    f32 det     = FMS(duv02[0], duv12[1], duv02[1] * duv12[0]);
    Vec3f dpdu, dpdv, dndu, dndv;
    if (det < 1e-9f)
    {
        CoordinateSystem(si.n, &dpdu, &dpdv);
    }
    else
    {
        f32 invDet = 1 / det;
        dpdu       = FMS(Vec3f(duv12[1]), dp02, duv02[1] * dp12) * invDet;
        dpdv       = FMS(Vec3f(duv02[0]), dp12, duv12[0] * dp02) * invDet;

        dndu = FMS(Vec3f(duv12[1]), dn02, duv02[1] * dn12) * invDet;
        dndv = FMS(Vec3f(duv02[0]), dn12, duv12[0] * dn02) * invDet;
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

    si.dpdu         = dpdu;
    si.dpdv         = dpdv;
    si.shading.dpdu = ss;
    si.shading.dpdv = ts;
    si.shading.dndu = dndu;
    si.shading.dndv = dndv;
    si.uv           = u * uv[1] + v * uv[2] + w * uv[0];
    Assert(geomID < scene->numPrimitives);
    const PrimitiveIndices *indices = scene->primIndices + geomID;
    si.materialIDs                  = indices->materialIndex;

    // TODO: properly obtain the light handle
    si.lightIndices = 0;
    si.faceIndices  = primID;
    si.curvature =
        CalculateCurvature(si.shading.dpdu, si.shading.dpdv, si.shading.dndu, si.shading.dndv);
}

template <i32 N, typename T>
struct TriangleIntersectorBase;

template <i32 N, template <i32> class Prim>
struct TriangleIntersectorBase<N, Prim<N>>
{
    using Primitive           = Prim<N>;
    TriangleIntersectorBase() = default;
    // https://www.graphics.cornell.edu/pubs/1997/MT97.pdf
    static bool Intersect(ScenePrimitives *scene, Ray2 &ray, Primitive *primitives, u32 num,
                          SurfaceInteraction &si)
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
            prim.GetData(scene, v0, v1, v2, geomIDs, primIDs);
            // prim.geomID
            Transpose<N>(v0, triV0);
            Transpose<N>(v1, triV1);
            Transpose<N>(v2, triV2);

            Mask<LaneF32<N>> mask = TriangleIntersect(ray, itr.t, triV0, triV1, triV2, triItr);
            itr.u                 = Select(mask, triItr.u, itr.u);
            itr.v                 = Select(mask, triItr.v, itr.v);
            itr.t                 = Select(mask, triItr.t, itr.t);
            itr.geomIDs           = AsUInt(
                Select(mask, AsFloat(MemSimdU32<N>::LoadU(geomIDs)), AsFloat(itr.geomIDs)));
            itr.primIDs = AsUInt(
                Select(mask, AsFloat(MemSimdU32<N>::LoadU(primIDs)), AsFloat(itr.primIDs)));
            outMask |= mask;
        }
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
            f32 u   = Get(itr.u, index);
            f32 v   = Get(itr.v, index);
            f32 w   = 1 - u - v;
            si.tHit = tFar;
            // Assert(u >= 0 && u <= 1 && v >= 0 && v <= 1 && w >= 0 && w <= 1);

            u32 primID = Get(itr.primIDs, index);
            u32 ids[]  = {
                3 * primID + 0,
                3 * primID + 1,
                3 * primID + 2,
            };
            SurfaceInteractionFromTriangleIntersection(scene, Get(itr.geomIDs, index), primID,
                                                       ids, u, v, w, si);
            return true;
        }
        return false;
    }
    template <i32 K>
    static bool Intersect(ScenePrimitives *scene, Ray2 &ray, BVHNode<K> ptr,
                          SurfaceInteraction &si, TravRay<K> &)
    {
        Assert(ptr.IsLeaf());
        Primitive *primitives = (Primitive *)ptr.GetPtr();
        u32 num               = ptr.GetNum();
        return Intersect(scene, ray, primitives, num, si);
    }
};

template <i32 N, typename T>
struct QuadIntersectorBase;

template <i32 N, template <i32> class Prim>
struct QuadIntersectorBase<N, Prim<N>>
{
    using Primitive       = Prim<N>;
    QuadIntersectorBase() = default;

    static bool Intersect(ScenePrimitives *scene, Ray2 &ray, Primitive *primitives, u32 num,
                          SurfaceInteraction &si)

    {
        auto trueMask  = Mask<LaneF32<N>>(TrueTy());
        auto falseMask = Mask<LaneF32<N>>(FalseTy());
        Vec3lf<N> triV0, triV1, triV2, triV3;
        Lane4F32 v0[N];
        Lane4F32 v1[N];
        Lane4F32 v2[N];
        Lane4F32 v3[N];
        alignas(4 * N) u32 geomIDs[N];
        alignas(4 * N) u32 primIDs[N];

        LaneF32<N> t(ray.tFar);
        Mask<LaneF32<N>> outMask(false);
        Mask<LaneF32<N>> triMask(false);

        TriangleIntersection<N> itr;
        itr.t = LaneF32<N>(ray.tFar);
        for (u32 i = 0; i < num; i++)
        {
            TriangleIntersection<N> triItr;
            Primitive &prim = primitives[i];
            prim.GetData(scene, v0, v1, v2, v3, geomIDs, primIDs);
            // prim.geomID
            Transpose<N>(v0, triV0);
            Transpose<N>(v1, triV1);
            Transpose<N>(v2, triV2);
            Transpose<N>(v3, triV3);

            Mask<LaneF32<N>> mask = TriangleIntersect(ray, itr.t, triV0, triV1, triV2, triItr);
            triMask               = Select(mask, falseMask, triMask);
            itr.u                 = Select(mask, triItr.u, itr.u);
            itr.v                 = Select(mask, triItr.v, itr.v);
            itr.t                 = Select(mask, triItr.t, itr.t);
            itr.geomIDs           = AsUInt(
                Select(mask, AsFloat(MemSimdU32<N>::LoadU(geomIDs)), AsFloat(itr.geomIDs)));
            itr.primIDs = AsUInt(
                Select(mask, AsFloat(MemSimdU32<N>::LoadU(primIDs)), AsFloat(itr.primIDs)));
            outMask |= mask;

            mask        = TriangleIntersect(ray, itr.t, triV0, triV2, triV3, triItr);
            triMask     = Select(mask, trueMask, triMask);
            itr.u       = Select(mask, triItr.u, itr.u);
            itr.v       = Select(mask, triItr.v, itr.v);
            itr.t       = Select(mask, triItr.t, itr.t);
            itr.geomIDs = AsUInt(
                Select(mask, AsFloat(MemSimdU32<N>::LoadU(geomIDs)), AsFloat(itr.geomIDs)));
            itr.primIDs = AsUInt(
                Select(mask, AsFloat(MemSimdU32<N>::LoadU(primIDs)), AsFloat(itr.primIDs)));
            outMask |= mask;
        }

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
            u32 isSecondTri = Movemask(triMask) & (1 << index);
            si.tHit         = tFar;
            f32 u           = Get(itr.u, index);
            f32 v           = Get(itr.v, index);
            f32 w           = 1 - u - v;
            // Assert(u >= 0 && u <= 1 && v >= 0 && v <= 1 && w >= 0 && w <= 1);

            u32 ids[3];
            u32 primID = Get(itr.primIDs, index);
            if (isSecondTri)
            {
                ids[0] = 4 * primID + 0;
                ids[1] = 4 * primID + 2;
                ids[2] = 4 * primID + 3;
            }
            else
            {
                ids[0] = 4 * primID + 0;
                ids[1] = 4 * primID + 1;
                ids[2] = 4 * primID + 2;
            }
            SurfaceInteractionFromTriangleIntersection(
                scene, Get(itr.geomIDs, index), primID / 2, ids, u, v, w, si, isSecondTri);

#if DEBUG
            GetDebug()->filename = scene->filename;
            GetDebug()->geomID   = Get(itr.geomIDs, index);
            GetDebug()->scene    = scene;
#endif

            return true;
        }
        return false;
    }
    template <i32 K>
    static bool Intersect(ScenePrimitives *scene, Ray2 &ray, BVHNode<K> ptr,
                          SurfaceInteraction &si, TravRay<K> &)
    {
        Assert(ptr.IsLeaf());
        Primitive *primitives = (Primitive *)ptr.GetPtr();
        u32 num               = ptr.GetNum();
        return Intersect(scene, ray, primitives, num, si);
    }
};

template <i32 N>
using TriangleIntersector = TriangleIntersectorBase<N, Triangle<N>>;

template <i32 N>
using TriangleIntersectorCmp = TriangleIntersectorBase<N, TriangleCompressed<N>>;

template <i32 N>
using QuadIntersector = QuadIntersectorBase<N, Quad<N>>;

template <i32 N>
using QuadIntersectorCmp = QuadIntersectorBase<N, QuadCompressed<N>>;

struct InstanceIntersector
{
    using Primitive = TLASLeaf;

    InstanceIntersector() {}
    static bool Intersect(ScenePrimitives *scene, Ray2 &ray, Primitive *primitives, u32 num,
                          SurfaceInteraction &si)
    {
        // intersect the bounds
        // get the leaves to intersect
        // transform the ray
        // intersect the bottom level hierarchy
        bool result               = false;
        const Instance *instances = (const Instance *)scene->primitives;
        for (u32 i = 0; i < num; i++)
        {
            Primitive &prim = primitives[i];

            AffineSpace *t;
            ScenePrimitives *childScene;

            prim.GetData(scene, t, childScene);
            Assert(childScene && t);
            Vec3f rayO = ray.o;
            Vec3f rayD = ray.d;

            AffineSpace inv = Inverse(*t);
            Mat3 invTp      = Transpose(Mat3(inv.c0, inv.c1, inv.c2));
            ray.o           = TransformP(inv, ray.o);
            ray.d           = TransformV(inv, ray.d);
            if (childScene->intersectFunc(childScene, prim.nodePtr, ray, si))
            {
                si.p            = TransformP(*t, si.p, si.pError);
                si.n            = Normalize(invTp * si.n);
                si.shading.n    = Normalize(invTp * si.shading.n);
                si.shading.dpdu = TransformV(*t, si.shading.dpdu);

                result = true;
            }
            ray.o = rayO;
            ray.d = rayD;
        }
        return result;
    }
    template <i32 K>
    static bool Intersect(ScenePrimitives *scene, Ray2 &ray, BVHNode<K> ptr,
                          SurfaceInteraction &si, TravRay<K> &)
    {
        Assert(ptr.IsLeaf());
        Primitive *primitives = (Primitive *)ptr.GetPtr();
        u32 num               = ptr.GetNum();
        return Intersect(scene, ray, primitives, num, si);
    }
};

template <i32 N, typename Intersector>
struct CompressedLeafIntersector
{
    using StackEntry = StackEntry<BVHNode<N>>;
    using Primitive  = typename Intersector::Primitive;
    Intersector intersector;
    CompressedLeafIntersector() {}
    template <typename Scene>
    bool Intersect(Scene *scene, Ray2 &ray, BVHNode<N> ptr, SurfaceInteraction &itr,
                   TravRay<N> &tray)
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

                    result |= intersector.Intersect(scene, ray, primitives, num, itr);
                }
            }
            break;
            default: result |= intersector.Intersect(scene, ray, ptr, itr, tray);
        }
        return result;
    }
};

template <i32 K, i32 N, i32 types>
using BVHTriangleIntersector = BVHIntersector<K, types, TriangleIntersector<N>>;
typedef BVHTriangleIntersector<4, 1, BVH_AQ> BVH4TriangleIntersector1;
typedef BVHTriangleIntersector<4, 8, BVH_AQ> BVH4TriangleIntersector8;
typedef BVHTriangleIntersector<8, 8, BVH_AQ> BVH8TriangleIntersector8;

template <i32 K, i32 N, i32 types>
using BVHTriangleIntersectorCmp = BVHIntersector<K, types, TriangleIntersectorCmp<N>>;
typedef BVHTriangleIntersectorCmp<4, 1, BVH_AQ> BVH4TriangleIntersectorCmp1;
typedef BVHTriangleIntersectorCmp<4, 8, BVH_AQ> BVH4TriangleIntersectorCmp8;
typedef BVHTriangleIntersectorCmp<8, 8, BVH_AQ> BVH8TriangleIntersectorCmp8;

template <i32 K, i32 N, i32 types>
using BVHQuadIntersector = BVHIntersector<K, types, QuadIntersector<N>>;
typedef BVHQuadIntersector<4, 1, BVH_AQ> BVH4QuadIntersector1;
typedef BVHQuadIntersector<4, 8, BVH_AQ> BVH4QuadIntersector8;
typedef BVHQuadIntersector<8, 8, BVH_AQ> BVH8QuadIntersector8;

template <i32 K, i32 N, i32 types>
using BVHQuadIntersectorCmp = BVHIntersector<K, types, QuadIntersectorCmp<N>>;
typedef BVHQuadIntersectorCmp<4, 1, BVH_AQ> BVH4QuadIntersectorCmp1;
typedef BVHQuadIntersectorCmp<4, 8, BVH_AQ> BVH4QuadIntersectorCmp8;
typedef BVHQuadIntersectorCmp<8, 8, BVH_AQ> BVH8QuadIntersectorCmp8;

// NOTE: K is for BVH branching factor, N is for Primitive
template <i32 K, i32 N, i32 types>
using BVHTriangleCLIntersector =
    BVHIntersector<K, types, CompressedLeafIntersector<K, TriangleIntersector<N>>>;
typedef BVHTriangleCLIntersector<4, 1, BVH_AQ> BVH4TriangleCLIntersector1;
typedef BVHTriangleCLIntersector<4, 8, BVH_AQ> BVH4TriangleCLIntersector8;
typedef BVHTriangleCLIntersector<8, 8, BVH_AQ> BVH8TriangleCLIntersector8;

template <i32 K, i32 N, i32 types>
using BVHTriangleCLIntersectorCmp =
    BVHIntersector<K, types, CompressedLeafIntersector<K, TriangleIntersectorCmp<N>>>;
typedef BVHTriangleCLIntersectorCmp<4, 1, BVH_AQ> BVH4TriangleCLIntersectorCmp1;
typedef BVHTriangleCLIntersectorCmp<4, 8, BVH_AQ> BVH4TriangleCLIntersectorCmp8;
typedef BVHTriangleCLIntersectorCmp<8, 8, BVH_AQ> BVH8TriangleCLIntersectorCmp8;

template <i32 K, i32 N, i32 types>
using BVHQuadCLIntersector =
    BVHIntersector<K, types, CompressedLeafIntersector<K, QuadIntersector<N>>>;
typedef BVHQuadCLIntersector<4, 1, BVH_AQ> BVH4QuadCLIntersector1;
typedef BVHQuadCLIntersector<4, 8, BVH_AQ> BVH4QuadCLIntersector8;
typedef BVHQuadCLIntersector<8, 8, BVH_AQ> BVH8QuadCLIntersector8;

template <i32 K, i32 N, i32 types>
using BVHQuadCLIntersectorCmp =
    BVHIntersector<K, types, CompressedLeafIntersector<K, QuadIntersectorCmp<N>>>;
typedef BVHQuadCLIntersectorCmp<4, 1, BVH_AQ> BVH4QuadCLIntersectorCmp1;
typedef BVHQuadCLIntersectorCmp<4, 8, BVH_AQ> BVH4QuadCLIntersectorCmp8;
typedef BVHQuadCLIntersectorCmp<8, 8, BVH_AQ> BVH8QuadCLIntersectorCmp8;

template <i32 K, i32 types>
using BVHInstanceCLIntersector =
    BVHIntersector<K, types, CompressedLeafIntersector<K, InstanceIntersector>>;
typedef BVHInstanceCLIntersector<4, BVH_AQ> BVH4InstanceCLIntersector;
typedef BVHInstanceCLIntersector<8, BVH_AQ> BVH8InstanceCLIntersector;

// Helpers
template <i32 N, GeometryType type, typename PrimRefType>
struct IntersectorHelperBase;

template <>
struct IntersectorHelperBase<4, GeometryType::TriangleMesh, PrimRefCompressed>
{
    using IntersectorType = BVH4TriangleCLIntersectorCmp8;
};

template <>
struct IntersectorHelperBase<4, GeometryType::TriangleMesh, PrimRef>
{
    using IntersectorType = BVH4TriangleCLIntersector8;
};

template <>
struct IntersectorHelperBase<4, GeometryType::QuadMesh, PrimRefCompressed>
{
    using IntersectorType = BVH4QuadCLIntersectorCmp8;
};

template <>
struct IntersectorHelperBase<4, GeometryType::QuadMesh, PrimRef>
{
    using IntersectorType = BVH4QuadCLIntersector8;
};

template <>
struct IntersectorHelperBase<4, GeometryType::Instance, BRef>
{
    using IntersectorType = BVH4InstanceCLIntersector;
};

template <>
struct IntersectorHelperBase<8, GeometryType::TriangleMesh, PrimRefCompressed>
{
    using IntersectorType = BVH8TriangleCLIntersectorCmp8;
};

template <>
struct IntersectorHelperBase<8, GeometryType::TriangleMesh, PrimRef>
{
    using IntersectorType = BVH8TriangleCLIntersector8;
};

template <>
struct IntersectorHelperBase<8, GeometryType::QuadMesh, PrimRefCompressed>
{
    using IntersectorType = BVH8QuadCLIntersectorCmp8;
};

template <>
struct IntersectorHelperBase<8, GeometryType::QuadMesh, PrimRef>
{
    using IntersectorType = BVH8QuadCLIntersector8;
};

template <>
struct IntersectorHelperBase<8, GeometryType::Instance, BRef>
{
    using IntersectorType = BVH8InstanceCLIntersector;
};

#ifdef USE_BVH4
template <GeometryType type, typename PrimRefType>
using IntersectorHelper = IntersectorHelperBase<4, type, PrimRefType>;
#elif defined(USE_BVH8)
template <GeometryType type, typename PrimRefType>
using IntersectorHelper = IntersectorHelperBase<8, type, PrimRefType>;
#endif

} // namespace rt
#endif
