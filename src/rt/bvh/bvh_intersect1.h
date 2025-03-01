#ifndef BVH_INTERSECT1_H
#define BVH_INTERSECT1_H
#include "bvh_types.h"
#include "../scene.h"
#include <immintrin.h>
#include <xmmintrin.h>

// #define USE_AFRA_TRAVERSAL
// https://afra.dev/publications/Afra2013Incoherent.pdf (pg. 5)

namespace rt
{

typedef u32 BVHNodeType;
enum
{
    BVHNodeType_Quantized       = 1 << 0,
    BVHNodeType_QuantizedLeaves = 1 << 1,

    BVH_QN   = BVHNodeType_Quantized,
    BVH_QNLF = BVHNodeType_QuantizedLeaves,
};

template <i32 K, i32 types, typename StackEntry>
struct GetNode;

template <int N>
struct StackEntry;

#ifdef USE_QUANTIZE_COMPRESS

template <int N>
struct StackEntry
{
    using EntryType = BVHNode<N>;

    union
    {
        struct
        {
            BVHNode<N> ptr;
            u32 type;
            u32 dist;
            // f32 dist;
        };
        Lane4U32 data;
    };

    StackEntry() {}
    StackEntry(const BVHNode<N> &ptr, u32 type, u32 dist)
        : data(_mm_set_epi64x(((size_t)dist << 32) | ((size_t)type), (size_t)ptr.data))
    {
    }
    // : ptr(ptr), type(type), dist(dist) {}
    StackEntry(const BVHNode<N> &ptr, f32 dist)
        : ptr((uintptr_t)ptr.GetPtr()), type(ptr.GetType()), dist(dist)
    {
    }

    StackEntry(const StackEntry &other) : data(other.data) {}

    __forceinline StackEntry &operator=(const StackEntry &other)
    {
        data = other.data;
        return *this;
    }

    template <typename Prim, typename Node>
    __forceinline static StackEntry<N> Create(Node *node, int index, u32 d = 0.f)
    {
        return StackEntry<N>(node->Child(index, (int)sizeof(Prim)), node->GetType(index), d);
    }

    bool IsLeaf() const { return type >= BVHNode<N>::tyLeaf; }
    void *GetPtr() const { return (void *)ptr.data; }
    u32 GetType() const { return type; }
    u32 GetNum() const
    {
        Assert(IsLeaf());
        return type - BVHNode<N>::tyLeaf;
    }
};

template <i32 K, typename StackEntry>
struct GetNode<K, BVH_QN, StackEntry>
{
    GetNode() {}
    auto operator()(const StackEntry &entry)
    {
        Assert(entry.type == StackEntry::EntryType::tyQuantizedNode);
        QuantizedCompressedNode<K> *qNode = (QuantizedCompressedNode<K> *)entry.ptr.data;
        return qNode;
    }
};

template <i32 N>
void BVHPrefetch(const StackEntry<N> &entry, u32 types)
{
    if (types == BVH_QN)
    {
        _mm_prefetch((char *)entry.ptr.data, _MM_HINT_T2);
        _mm_prefetch((char *)entry.ptr.data + 64, _MM_HINT_T2);
    }
    else
    {
        Assert(0);
    }
}

#else

template <int N>
struct StackEntry
{
    BVHNode<N> ptr;
    f32 dist;

    StackEntry() = default;
    StackEntry(const BVHNode<N> &ptr, f32 dist) : ptr(ptr), dist(dist) {}

    template <typename Prim, typename Node>
    static StackEntry<N> Create(Node *node, int index, f32 d = 0.f)
    {
        return StackEntry<N>{node->Child(index), d};
    }

    bool IsLeaf() const { return ptr.IsLeaf(); }
    void *GetPtr() const { return ptr.GetPtr(); }
    u32 GetNum() const { return ptr.GetNum(); }
    u32 GetType() const { return ptr.GetType(); }
};

template <i32 K, typename StackEntry>
struct GetNode<K, BVH_QN, StackEntry>
{
    GetNode() {}
    auto operator()(const StackEntry &entry)
    {
        Assert(entry.ptr.IsQuantizedNode());
        QuantizedNode<K> *qNode = entry.ptr.GetQuantizedNode();
        return qNode;
    }
};

template <i32 K, typename StackEntry>
struct GetNode<K, BVH_QNLF, StackEntry>
{
    GetNode() {}
    auto operator()(const StackEntry &entry)
    {
        Assert(entry.ptr.IsCompressedLeaf());
        CompressedLeafNode<K> *qNode = entry.ptr.GetCompressedLeaf();
        return qNode;
    }
};
#endif

template <i32 K>
struct TravRay
{
    Vec3lf<K> o;
    Vec3lf<K> d;
    Vec3lf<K> invRayD;
    LaneF32<K> tFar;
    TravRay(Ray2 &r)
        : o(Vec3lf<K>(r.o)), d(Vec3lf<K>(r.d)),
          invRayD(Vec3lf<K>(Select(r.d.x == 0, 0, 1 / r.d.x), Select(r.d.y == 0, 0, 1 / r.d.y),
                            Select(r.d.z == 0, 0, 1 / r.d.z))),
          tFar(LaneF32<K>(r.tFar))
    {
    }
};

template <i32 K, i32 types, typename Prim>
struct BVHTraverser;

template <i32 types, typename Prim>
struct BVHTraverser<4, types, Prim>
{
    template <typename Node>
    static void Intersect(const Node *node, const TravRay<4> &ray, Lane4F32 &tEntryOut,
                          Lane4F32 &mask)
    {
        Vec3lf4 mins, maxs;
        // Get bounds from the node
        node->GetBounds(mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2]);

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

        Lane4F32 validNodeMask = node->GetValid();

        tEntryOut = tEntry;
        mask      = validNodeMask & intersectMask;
    }
    template <typename StackEntry>
    static void Traverse(StackEntry entry, StackEntry *stack, i32 &stackPtr, TravRay<4> &ray)
    {
        Lane4F32 tEntry, mask;
        auto node = GetNode<4, types>{}(entry);
        Intersect(node, ray, tEntry, mask);

        const i32 intersectFlags = Movemask(mask);

        Lane4F32 t_dcba    = Select(mask, tEntry, pos_inf);
        const u32 numNodes = PopCount(intersectFlags);

        if (numNodes <= 1)
        {
            // If numNodes <= 1, then numNode will be 0, 1, 2, 4, or 8. x/2 - x/8 maps to
            // 0, 0, 1, 2, 3
            u32 nodeIndex = (intersectFlags >> 1) - (intersectFlags >> 3);
            stack[stackPtr] =
                StackEntry::template Create<Prim>(node, nodeIndex, t_dcba[nodeIndex]);
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

            stack[stackPtr + ((numNodes - 1 - indexA) & 3)] =
                StackEntry::template Create<Prim>(node, 0, t_dcba[0]);
            stack[stackPtr + ((numNodes - 1 - indexB) & 3)] =
                StackEntry::template Create<Prim>(node, 1, t_dcba[1]);
            stack[stackPtr + ((numNodes - 1 - indexC) & 3)] =
                StackEntry::template Create<Prim>(node, 2, t_dcba[2]);
            stack[stackPtr + ((numNodes - 1 - indexD) & 3)] =
                StackEntry::template Create<Prim>(node, 3, t_dcba[3]);

            stackPtr += numNodes;
        }
    }
    template <typename StackEntry>
    static void TraverseAny(StackEntry &entry, StackEntry *stack, i32 &stackPtr,
                            TravRay<4> &ray)
    {
        auto node = GetNode<4, types>{}(entry);
        Lane4F32 tEntry, mask;
        Intersect(node, ray, tEntry, mask);
        i32 intersectFlags = Movemask(mask);
        u32 oldPtr         = stackPtr;
        stackPtr += PopCount(intersectFlags);
        while (intersectFlags)
        {
            i32 bit = Bsf(intersectFlags);
            intersectFlags &= intersectFlags - 1;
            stack[oldPtr++] = StackEntry::template Create<Prim>(node, bit);
        }
    }
};

// embree kernels
__forceinline static void cmp_xchg(Lane4U32 &a, Lane4U32 &b)
{
    const Lane4F32 mask0 = b < a;
    const Lane4F32 mask(Shuffle<3, 3, 3, 3>(mask0));

    const Lane4U32 c = Select(mask, b, a);
    const Lane4U32 d = Select(mask, a, b);
    a                = c;
    b                = d;
}

/*! Sort 3 stack items. */
__forceinline static void sort3(Lane4U32 &s1, Lane4U32 &s2, Lane4U32 &s3)
{
    cmp_xchg(s2, s1);
    cmp_xchg(s3, s2);
    cmp_xchg(s2, s1);
}

/*! Sort 4 stack items. */
__forceinline static void sort4(Lane4U32 &s1, Lane4U32 &s2, Lane4U32 &s3, Lane4U32 &s4)
{
    cmp_xchg(s2, s1);
    cmp_xchg(s4, s3);
    cmp_xchg(s3, s1);
    cmp_xchg(s4, s2);
    cmp_xchg(s3, s2);
}

template <int N>
__forceinline void sort(StackEntry<N> *begin, StackEntry<N> *end)
{
    for (StackEntry<N> *i = begin + 1; i != end; ++i)
    {
        const Lane4F32 item = Lane4F32::Load((float *)i);
        const u32 dist      = i->dist;
        StackEntry<N> *j    = i;

        while ((j != begin) && ((j - 1)->dist < dist))
        {
            Lane4F32::Store(j, Lane4F32::Load((float *)(j - 1)));
            --j;
        }

        Lane4F32::Store(j, item);
    }
}

template <i32 types, typename Prim>
struct BVHTraverser<8, types, Prim>
{
    template <typename Node>
    static void Intersect(const Node *node, const TravRay<8> &ray, Lane8F32 &tEntryOut,
                          Lane8F32 &mask)
    {
        Vec3lf8 mins, maxs;
        // Get bounds from the node
        node->GetBounds(mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2]);

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

        Lane8F32 validNodeMask = node->GetValid();

        tEntryOut = tEntry;
        mask      = validNodeMask & intersectMask;
    }
    template <typename StackEntry>
    static bool Traverse(StackEntry &entry, StackEntry *stack, i32 &stackPtr, TravRay<8> &ray)
    {
#ifndef USE_AFRA_TRAVERSAL
        Lane8F32 tEntry, mask;
        auto node = GetNode<8, types, StackEntry>{}(entry);
        Intersect(node, ray, tEntry, mask);

        const i32 intersectFlags = Movemask(mask);

        Lane8F32 t_hgfedcba = Select(mask, tEntry, pos_inf);
        const u32 numNodes  = PopCount(intersectFlags);

        if (numNodes == 0) return true;
        else if (numNodes == 1)
        {
            u32 nodeIndex = Bsf(intersectFlags);
            stack[stackPtr] =
                StackEntry::template Create<Prim>(node, nodeIndex, t_hgfedcba[nodeIndex]);
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

            // TODO: revisit this, see if there's anything faster
            // NOTE: in avx 512, this can be replaced with: 
            // _mm256_popcnt_epi32
            u32 indexA = PopCount(~nodeMask & 0x000100ed);
            u32 indexB = PopCount((nodeMask ^ 0x002c2c00) & 0x002c2d00);
            u32 indexC = PopCount((nodeMask ^ 0x20121200) & 0x20123220);
            u32 indexD = PopCount((nodeMask ^ 0x06404000) & 0x06404602);
            u32 indexE = PopCount((nodeMask ^ 0x08808000) & 0x0a828808);
            u32 indexF = PopCount((nodeMask ^ 0x50000000) & 0x58085010);
            u32 indexG = PopCount((nodeMask ^ 0x80000000) & 0x94148080);
            u32 indexH = PopCount(nodeMask & 0xe0e10000);

            stack[stackPtr + ((numNodes - 1 - indexA) & 7)] =
                StackEntry::template Create<Prim>(node, 0, t_hgfedcba[0]);
            stack[stackPtr + ((numNodes - 1 - indexB) & 7)] =
                StackEntry::template Create<Prim>(node, 1, t_hgfedcba[1]);
            stack[stackPtr + ((numNodes - 1 - indexC) & 7)] =
                StackEntry::template Create<Prim>(node, 2, t_hgfedcba[2]);
            stack[stackPtr + ((numNodes - 1 - indexD) & 7)] =
                StackEntry::template Create<Prim>(node, 3, t_hgfedcba[3]);

            stack[stackPtr + ((numNodes - 1 - indexE) & 7)] =
                StackEntry::template Create<Prim>(node, 4, t_hgfedcba[4]);
            stack[stackPtr + ((numNodes - 1 - indexF) & 7)] =
                StackEntry::template Create<Prim>(node, 5, t_hgfedcba[5]);
            stack[stackPtr + ((numNodes - 1 - indexG) & 7)] =
                StackEntry::template Create<Prim>(node, 6, t_hgfedcba[6]);
            stack[stackPtr + ((numNodes - 1 - indexH) & 7)] =
                StackEntry::template Create<Prim>(node, 7, t_hgfedcba[7]);

            stackPtr += numNodes;
        }
        return true;
#else
        {
            Lane8F32 tEntry, laneMask;
            auto node = GetNode<8, types, StackEntry>{}(entry);
            Intersect(node, ray, tEntry, laneMask);

            u32 mask = Movemask(laneMask);
            if (mask == 0) return true;
            Lane8F32 tNear = Select(laneMask, tEntry, pos_inf);

            Assert(mask != 0);

            /*! one child is hit, continue with that child */
            u32 r = Bscf(mask);

            StackEntry c0 = StackEntry::template Create<Prim>(node, r, tNear[r]);
            BVHPrefetch(c0, types);
            if (mask == 0)
            {
                entry = c0;
                return false;
            }

            /*! two children are hit, push far child, and continue with closer child */
            r             = Bscf(mask);
            StackEntry c1 = StackEntry::template Create<Prim>(node, r, tNear[r]);
            BVHPrefetch(c1, types);
            const u32 d1 = (u32)tNear[r];

            if (mask == 0)
            {
                if (c0.dist < c1.dist)
                {
                    Lane4U32::Store(&stack[stackPtr], c1.data);
                    entry = c0;
                    stackPtr++;
                    return false;
                }
                else
                {
                    Lane4U32::Store(&stack[stackPtr], c0.data);
                    entry = c1;
                    stackPtr++;
                    return false;
                }
            }

            r             = Bscf(mask);
            StackEntry c2 = StackEntry::template Create<Prim>(node, r, tNear[r]);
            BVHPrefetch(c2, types);
            /* 3 hits */
            if (mask == 0)
            {
                sort3(c0.data, c1.data, c2.data);
                Lane4U32::Store(&stack[stackPtr], c0.data);
                Lane4U32::Store(&stack[stackPtr + 1], c1.data);
                entry = c2;
                stackPtr += 2;
                return false;
            }
            r = Bscf(mask);

            StackEntry c3 = StackEntry::template Create<Prim>(node, r, tNear[r]);
            BVHPrefetch(c3, types);
            /* 4 hits */
            if (mask == 0)
            {
                sort4(c0.data, c1.data, c2.data, c3.data);
                Lane4U32::Store(&stack[stackPtr], c0.data);
                Lane4U32::Store(&stack[stackPtr + 1], c1.data);
                Lane4U32::Store(&stack[stackPtr + 2], c2.data);
                entry = c3;
                stackPtr += 3;
                return false;
            }
            Lane4U32::Store(&stack[stackPtr], c0.data);
            Lane4U32::Store(&stack[stackPtr + 1], c1.data);
            Lane4U32::Store(&stack[stackPtr + 2], c2.data);
            Lane4U32::Store(&stack[stackPtr + 3], c3.data);
            /*! fallback case if more than 4 children are hit */
            StackEntry *stackFirst = stack + stackPtr;
            stackPtr += 4;
            while (mask)
            {
                r            = Bscf(mask);
                StackEntry c = StackEntry::template Create<Prim>(node, r, tNear[r]);
                BVHPrefetch(c, types);
                Lane4U32::Store(&stack[stackPtr++], c.data);
            }
            sort(stackFirst, &stack[stackPtr]);
            return true;
        }
#endif
    }

    template <typename StackEntry>
    static void TraverseAny(StackEntry &entry, StackEntry *stack, i32 &stackPtr,
                            TravRay<8> &ray)
    {
        auto node = GetNode<8, types, StackEntry>{}(entry);
        Lane8F32 tEntry, mask;
        Intersect(node, ray, tEntry, mask);
        i32 intersectFlags = Movemask(mask);
        u32 oldPtr         = stackPtr;
        stackPtr += PopCount(intersectFlags);
        while (intersectFlags)
        {
            i32 bit = Bsf(intersectFlags);
            intersectFlags &= intersectFlags - 1;
            stack[oldPtr++] = StackEntry::template Create<Prim>(node, bit);
        }
    }
};

template <i32 K, i32 types, typename Intersector>
struct BVHIntersector
{
    using StackEntry = StackEntry<K>;
    using Prim       = typename Intersector::Primitive;

    BVHIntersector() {}
    static bool Intersect(ScenePrimitives *scene, StackEntry stackEntry, Ray2 &ray,
                          SurfaceInteraction &itr)
    {
        // typedef typename Intersector::Primitive Primitive;
        TravRay<K> r(ray);

        StackEntry stack[K == 4 ? 256 : 512];
        i32 stackPtr = 1;
        stack[0]     = stackEntry;
        bool result  = false;
        Intersector intersector;

    top:
        while (stackPtr > 0)
        {
            Assert(stackPtr <= ArrayLength(stack));
            StackEntry entry = stack[--stackPtr];
            Assert(entry.ptr.data && entry.ptr.data != BVHNode<K>::tyEmpty);
            if (entry.dist > ray.tFar) continue;

            // if (entry.IsLeaf())
            while (!entry.IsLeaf())
            {
                bool entryNotReplaced =
                    BVHTraverser<K, types, Prim>::Traverse(entry, stack, stackPtr, r);
                Assert(entry.ptr.data && entry.ptr.data != BVHNode<K>::tyEmpty);
                if (entryNotReplaced || entry.dist > ray.tFar) goto top;
            }
            result |= intersector.Intersect(scene, ray, entry, itr, r);
            r.tFar = ray.tFar;
        }
        return result;
    }
    static bool Occluded(ScenePrimitives *scene, StackEntry stackEntry, Ray2 &ray)
    {
        TravRay<K> r(ray);

        // BVHNode<K> stack[K == 4 ? 256 : 512];
        StackEntry stack[K == 4 ? 256 : 512];
        i32 stackPtr = 1;
        stack[0]     = stackEntry;
        bool result  = false;
        Intersector intersector;
        while (stackPtr > 0)
        {
            Assert(stackPtr <= ArrayLength(stack));
            // BVHNode<K> entry = stack[--stackPtr];
            StackEntry entry = stack[--stackPtr];

            if (entry.IsLeaf())
            {
                SurfaceInteraction itr;
                // TODO: implement Occluded for intersectors
                if (intersector.Intersect(scene, ray, entry, itr, r)) return true;
                continue;
            }
            BVHTraverser<K, types, Prim>::TraverseAny(entry, stack, stackPtr, r);
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
static Mask<LaneF32<N>> TriangleIntersect(const TravRay<N> &ray, const LaneF32<N> &tFar,
                                          const Vec3lf<N> &v0, const Vec3lf<N> &v1,
                                          const Vec3lf<N> &v2, TriangleIntersection<N> &itr)
{
    Vec3lf<N> o  = ray.o;
    Vec3lf<N> d  = ray.d;
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

static bool SurfaceInteractionFromTriangleIntersection(ScenePrimitives *scene,
                                                       const u32 geomID, const u32 primID,
                                                       const u32 ids[3], f32 u, f32 v, f32 w,
                                                       SurfaceInteraction &si,
                                                       bool isSecondTri = false)
{
    Mesh *mesh                      = (Mesh *)scene->primitives + geomID;
    const PrimitiveIndices *indices = scene->primIndices + geomID;

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

    Vec3f dpdu, dpdv, dndu, dndv;

    Vec2f duv02 = uv[0] - uv[2];
    Vec2f duv12 = uv[1] - uv[2];
    f32 det     = FMS(duv02[0], duv12[1], duv02[1] * duv12[0]);
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
    si.materialIDs = indices->materialID.data;

    si.sceneID = scene->sceneIndex;
    si.geomID  = geomID;

    // TODO: properly obtain the light handle
    si.lightIndices = indices->lightID.data;
    si.faceIndices  = mesh->faceIDs ? mesh->faceIDs[primID] : primID;
    si.curvature =
        CalculateCurvature(si.shading.dpdu, si.shading.dpdv, si.shading.dndu, si.shading.dndv);

    // TODO: actual stochastic alpha testing
    if (indices->alphaTexture &&
        indices->alphaTexture->EvaluateFloat(si, Vec4f{0.f, 0.f, 0.f, 0.f}) == 0.f)
    {
        return false;
    }

    return true;
}

template <i32 N, typename T>
struct TriangleIntersectorBase;

template <i32 N, template <i32> class Prim>
struct TriangleIntersectorBase<N, Prim<N>>
{
    using Primitive           = Prim<N>;
    TriangleIntersectorBase() = default;
    // https://www.graphics.cornell.edu/pubs/1997/MT97.pdf
    template <i32 K>
    static bool Intersect(ScenePrimitives *scene, Ray2 &ray, Primitive *primitives, u32 num,
                          SurfaceInteraction &si, TravRay<K> &r)
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

            Mask<LaneF32<N>> mask = TriangleIntersect(r, itr.t, triV0, triV1, triV2, triItr);
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
            for (;;)
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
                // NOTE: if alpha testing fails, then need to reevaluate
                SurfaceInteraction siitr;
                bool result = SurfaceInteractionFromTriangleIntersection(
                    scene, Get(itr.geomIDs, index), primID, ids, u, v, w, siitr);
                if (result)
                {
                    si                = siitr;
                    GetDebug()->scene = scene;
                    return true;
                }
                else
                {
                    outMask &= !LaneF32<N>::Mask(1u << index);
                    itr.t[index] = pos_inf;
                }
            }
        }
        return false;
    }
    template <typename StackEntry, i32 K>
    static bool Intersect(ScenePrimitives *scene, Ray2 &ray, StackEntry entry,
                          SurfaceInteraction &si, TravRay<K> &r)
    {
        Primitive *primitives = (Primitive *)entry.GetPtr();
        u32 num               = entry.GetNum();
        return Intersect(scene, ray, primitives, num, si, r);
    }
};

template <i32 N, typename T>
struct QuadIntersectorBase;

// TODO: verify that the lights are actually in the right place
template <i32 N, template <i32> class Prim>
struct QuadIntersectorBase<N, Prim<N>>
{
    using Primitive       = Prim<N>;
    QuadIntersectorBase() = default;

    template <i32 K>
    static bool Intersect(ScenePrimitives *scene, Ray2 &ray, Primitive *primitives, u32 num,
                          SurfaceInteraction &si, TravRay<K> &r)

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

            Mask<LaneF32<N>> mask = TriangleIntersect(r, itr.t, triV0, triV1, triV2, triItr);
            triMask               = Select(mask, falseMask, triMask);
            itr.u                 = Select(mask, triItr.u, itr.u);
            itr.v                 = Select(mask, triItr.v, itr.v);
            itr.t                 = Select(mask, triItr.t, itr.t);
            itr.geomIDs           = AsUInt(
                Select(mask, AsFloat(MemSimdU32<N>::LoadU(geomIDs)), AsFloat(itr.geomIDs)));
            itr.primIDs = AsUInt(
                Select(mask, AsFloat(MemSimdU32<N>::LoadU(primIDs)), AsFloat(itr.primIDs)));
            outMask |= mask;

            mask        = TriangleIntersect(r, itr.t, triV0, triV2, triV3, triItr);
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
            for (;;)
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
                SurfaceInteraction siitr;
                bool result = SurfaceInteractionFromTriangleIntersection(
                    scene, Get(itr.geomIDs, index), primID / 2, ids, u, v, w, siitr,
                    isSecondTri);

                if (result)
                {
                    si = siitr;
                    return true;
                }
                else
                {
                    outMask &= !LaneF32<N>::Mask(1u << index);
                    itr.t[index] = pos_inf;
                }
            }
        }
        return false;
    }
    template <typename StackEntry, i32 K>
    static bool Intersect(ScenePrimitives *scene, Ray2 &ray, StackEntry entry,
                          SurfaceInteraction &si, TravRay<K> &r)
    {
        Primitive *primitives = (Primitive *)entry.GetPtr();
        u32 num               = entry.GetNum();
        return Intersect(scene, ray, primitives, num, si, r);
    }
};

struct CatClarkPatchIntersector
{
    typedef CatmullClarkPatch Primitive;
    static const int N = 8;
    template <i32 K>
    static bool Intersect(ScenePrimitives *scene, Ray2 &ray, Primitive *primitives, u32 num,
                          SurfaceInteraction &si, TravRay<K> &r)

    {
        auto trueMask  = Mask<Lane8F32>(TrueTy());
        auto falseMask = Mask<Lane8F32>(FalseTy());

        alignas(4 * N) u32 patchIDs[N];
        alignas(4 * N) u32 geomIDs[N];
        alignas(4 * N) u32 primIDs[N];

        int queueCount = 0;

        Lane8F32 t(ray.tFar);

        Lane8F32 outMask = _mm256_setzero_ps();

        TriangleIntersection<N> itr;
        Lane8U32 patchID;

        itr.t = LaneF32<N>(ray.tFar);

        OpenSubdivMesh *meshes = (OpenSubdivMesh *)scene->primitives;

        for (u32 i = 0; i < num; i++)
        {
            alignas(4 * N) u32 triIndices[3][N] = {};
            auto EmptyQueue                     = [&](f32 *vertices) {
                Vec3lf8 triV0, triV1, triV2;
                Lane8U32 v0Indices = Lane8U32::Load(triIndices[0]);
                Lane8U32 v1Indices = Lane8U32::Load(triIndices[1]);
                Lane8U32 v2Indices = Lane8U32::Load(triIndices[2]);
                triV0.x            = GatherFloat(vertices, v0Indices);
                triV0.y            = GatherFloat(vertices, v0Indices + 1);
                triV0.z            = GatherFloat(vertices, v0Indices + 2);

                triV1.x = GatherFloat(vertices, v1Indices);
                triV1.y = GatherFloat(vertices, v1Indices + 1);
                triV1.z = GatherFloat(vertices, v1Indices + 2);

                triV2.x = GatherFloat(vertices, v2Indices);
                triV2.y = GatherFloat(vertices, v2Indices + 1);
                triV2.z = GatherFloat(vertices, v2Indices + 2);

                TriangleIntersection<N> triItr;

                Mask<LaneF32<N>> mask =
                    TriangleIntersect(r, itr.t, triV0, triV1, triV2, triItr);
                mask &= Lane8F32::Mask((1u << queueCount) - 1);
                outMask |= mask;

                itr.u       = Select(mask, triItr.u, itr.u);
                itr.v       = Select(mask, triItr.v, itr.v);
                itr.t       = Select(mask, triItr.t, itr.t);
                itr.geomIDs = AsUInt(Select(mask, AsFloat(MemSimdU32<N>::LoadU(geomIDs)),
                                                                AsFloat(itr.geomIDs)));
                itr.primIDs = AsUInt(Select(mask, AsFloat(MemSimdU32<N>::LoadU(primIDs)),
                                                                AsFloat(itr.primIDs)));
                patchID     = AsUInt(
                    Select(mask, AsFloat(MemSimdU32<N>::LoadU(patchIDs)), AsFloat(patchID)));

                queueCount = 0;
            };
            Primitive &prim = primitives[i];
            u32 primID      = prim.primID;
            u32 bits        = prim.geomID;

            CatClarkTriangleType type = GetPatchType(bits);
            u32 geomID                = GetTriangleIndex(bits);
            int edgeIndex             = GetMeta(bits);

            Assert(geomID < scene->numPrimitives);
            OpenSubdivMesh *mesh = meshes + geomID;
            const auto &indices  = mesh->stitchingIndices;
            const auto &vertices = mesh->vertices;

            auto PushQueue = [&](CatClarkTriangleType type, int id, int id0, int id1, int id2,
                                 int meta = 0) {
                triIndices[0][queueCount] = 3 * id0;
                triIndices[1][queueCount] = 3 * id1;
                triIndices[2][queueCount] = 3 * id2;
                patchIDs[queueCount]      = CreatePatchID(type, meta, id);
                geomIDs[queueCount]       = geomID;
                primIDs[queueCount]       = primID;
                queueCount++;

                if (queueCount == N) EmptyQueue((f32 *)vertices.data);
            };

            switch (type)
            {
                case CatClarkTriangleType::Untess:
                {
                    const UntessellatedPatch *patch = &mesh->untessellatedPatches[primID];

                    int indexStart = 4 * primID;

                    PushQueue(CatClarkTriangleType::Untess, 0, indices[indexStart + 0],
                              indices[indexStart + 1], indices[indexStart + 2]);
                    PushQueue(CatClarkTriangleType::Untess, 1, indices[indexStart + 0],
                              indices[indexStart + 2], indices[indexStart + 3]);
                }
                break;
                case CatClarkTriangleType::TessStitching:
                {
                    const BVHEdge *edge          = &mesh->bvhEdges[primID];
                    const OpenSubdivPatch *patch = &mesh->patches[edge->patchIndex];
                    auto iterator                = patch->CreateIterator(edgeIndex);
                    iterator.StepForward(edge->steps);
                    int count = 0;
                    for (; iterator.steps < 8 && iterator.Next();)
                    {
                        PushQueue(CatClarkTriangleType::TessStitching, count++,
                                  iterator.indices[0], iterator.indices[1],
                                  iterator.indices[2], edgeIndex);
                    }
                }
                break;
                case CatClarkTriangleType::TessGrid:
                {
                    // Step inner grid
                    const BVHPatch *bvhPatch     = &mesh->bvhPatches[primID];
                    const OpenSubdivPatch *patch = &mesh->patches[bvhPatch->patchIndex];

                    Vec2i uvStart, uvEnd;
                    bvhPatch->grid.Decompress(uvStart, uvEnd);

                    for (int v = uvStart[1]; v < uvEnd[1]; v++)
                    {
                        for (int u = uvStart[0]; u < uvEnd[0]; u++)
                        {
                            const int id00 = patch->GetGridIndex(u, v);
                            const int id10 = patch->GetGridIndex(u + 1, v);
                            const int id11 = patch->GetGridIndex(u + 1, v + 1);
                            const int id01 = patch->GetGridIndex(u, v + 1);

                            Assert(u < 0xffff && v < 0xffff);

                            u32 count = ((u32)v << 16u) | (u32)u;

                            PushQueue(CatClarkTriangleType::TessGrid, count, id00, id10, id11);
                            PushQueue(CatClarkTriangleType::TessGrid, count, id00, id11, id01,
                                      1);
                        }
                    }
                }
                break;
                default: Assert(0);
            }
            if (queueCount) EmptyQueue((f32 *)vertices.data);
        }

        if (Any(outMask))
        {
            // TODO: add alpha testing
            u32 maskBits = Movemask(outMask);
            f32 tFar     = ReduceMin(itr.t);
            ray.tFar     = tFar;
            int index    = -1;
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

            Assert(index != -1);
            u32 patchIndex            = patchID[index];
            u32 primID                = itr.primIDs[index];
            u32 geomID                = itr.geomIDs[index];
            CatClarkTriangleType type = GetPatchType(patchIndex);
            int id                    = GetTriangleIndex(patchIndex);

            Assert(geomID < scene->numPrimitives);
            OpenSubdivMesh *mesh = meshes + geomID;

            int faceID = -1;

            Vec2f uvs[3];
            u32 vertexIndices[3];

            switch (type)
            {
                case CatClarkTriangleType::Untess:
                {
                    UntessellatedPatch *patch = &mesh->untessellatedPatches[primID];
                    faceID                    = patch->faceID;
                    int start                 = 4 * primID;

                    if (id == 0)
                    {
                        vertexIndices[0] = mesh->stitchingIndices[start + 0];
                        vertexIndices[1] = mesh->stitchingIndices[start + 1];
                        vertexIndices[2] = mesh->stitchingIndices[start + 2];

                        uvs[0] = Vec2f(0, 0);
                        uvs[1] = Vec2f(1, 0);
                        uvs[2] = Vec2f(1, 1);
                    }
                    else if (id == 1)
                    {
                        vertexIndices[0] = mesh->stitchingIndices[start + 0];
                        vertexIndices[1] = mesh->stitchingIndices[start + 2];
                        vertexIndices[2] = mesh->stitchingIndices[start + 3];

                        uvs[0] = Vec2f(0, 0);
                        uvs[1] = Vec2f(1, 1);
                        uvs[2] = Vec2f(0, 1);
                    }
                }
                break;
                case CatClarkTriangleType::TessGrid:
                {
                    const BVHPatch *bvhPatch     = &mesh->bvhPatches[primID];
                    const OpenSubdivPatch *patch = &mesh->patches[bvhPatch->patchIndex];
                    faceID                       = patch->faceID;

                    int edgeU = patch->GetMaxEdgeFactorU();
                    int edgeV = patch->GetMaxEdgeFactorV();

                    Vec2f uvStep(1.f / edgeU, 1.f / edgeV);

                    int meta = GetMeta(patchIndex);
                    int u    = id & 0xffff;
                    int v    = (id >> 16) & 0x7fff;

                    // Second triangle
                    if (meta)
                    {
                        vertexIndices[0] = patch->GetGridIndex(u, v);
                        vertexIndices[1] = patch->GetGridIndex(u + 1, v + 1);
                        vertexIndices[2] = patch->GetGridIndex(u, v + 1);
                        uvs[0]           = uvStep * Vec2f((f32)(u + 1), (f32)(v + 1));
                        uvs[1]           = uvStep * Vec2f((f32)(u + 2), (f32)(v + 2));
                        uvs[2]           = uvStep * Vec2f((f32)(u + 1), (f32)(v + 2));
                    }
                    else
                    {
                        vertexIndices[0] = patch->GetGridIndex(u, v);
                        vertexIndices[1] = patch->GetGridIndex(u + 1, v);
                        vertexIndices[2] = patch->GetGridIndex(u + 1, v + 1);
                        uvs[0]           = uvStep * Vec2f((f32)(u + 1), (f32)(v + 1));
                        uvs[1]           = uvStep * Vec2f((f32)(u + 2), (f32)(v + 1));
                        uvs[2]           = uvStep * Vec2f((f32)(u + 2), (f32)(v + 2));
                    }
                }
                break;
                case CatClarkTriangleType::TessStitching:
                {
                    const BVHEdge *bvhEdge       = &mesh->bvhEdges[primID];
                    const OpenSubdivPatch *patch = &mesh->patches[bvhEdge->patchIndex];
                    faceID                       = patch->faceID;

                    // Reconstruct uvs
                    int edgeIndex = GetMeta(patchIndex);
                    auto iterator = patch->CreateIterator(edgeIndex);
                    iterator.StepForward(bvhEdge->steps);
                    iterator.GetUV(id, uvs);
                    vertexIndices[0] = iterator.indices[0];
                    vertexIndices[1] = iterator.indices[1];
                    vertexIndices[2] = iterator.indices[2];
                }
                break;
                default: ErrorExit(0, "type :%u id : %u\n", (u32)type, id);
            }

            // TODO: compress this into SurfaceInteractionFromTriangleIntersection
            // Recalculate uv of
            f32 u   = Get(itr.u, index);
            f32 v   = Get(itr.v, index);
            f32 w   = 1 - u - v;
            si.tHit = tFar;

            Vec3f p[3] = {
                mesh->vertices[vertexIndices[0]],
                mesh->vertices[vertexIndices[1]],
                mesh->vertices[vertexIndices[2]],
            };
            Vec3f n[3] = {
                mesh->normals[vertexIndices[0]],
                mesh->normals[vertexIndices[1]],
                mesh->normals[vertexIndices[2]],
            };

            si.p = u * p[1] + v * p[2] + w * p[0];

            si.pError = gamma(6) * (Abs(u * mesh->vertices[vertexIndices[1]]) +
                                    Abs(v * mesh->vertices[vertexIndices[2]]) +
                                    Abs(w * mesh->vertices[vertexIndices[0]]));

            si.uv = u * uvs[1] + v * uvs[2] + w * uvs[0];

            const PrimitiveIndices *indices = scene->primIndices + geomID;
            si.materialIDs                  = indices->materialID.data;
            si.lightIndices                 = 0;
            si.faceIndices                  = faceID;

            Vec3f dp02 = p[0] - p[2];
            Vec3f dp12 = p[1] - p[2];
            si.n       = Normalize(Cross(dp02, dp12));

            si.shading.n = u * n[1] + v * n[2] + w * n[0];
            si.shading.n = LengthSquared(si.shading.n) > 0 ? Normalize(si.shading.n) : si.n;

#if 1
            Vec3f dpdu, dpdv, dndu, dndv;

            Vec3f dn02 = n[0] - n[2];
            Vec3f dn12 = n[1] - n[2];

            Vec2f duv02 = uvs[0] - uvs[2];
            Vec2f duv12 = uvs[1] - uvs[2];
            f32 det     = FMS(duv02[0], duv12[1], duv02[1] * duv12[0]);
            if (det < 1e-9f)
            {
                CoordinateSystem(si.n, &dpdu, &dpdv);
            }
            else
            {
                f32 invDet = 1 / det;
                si.dpdu    = FMS(Vec3f(duv12[1]), dp02, duv02[1] * dp12) * invDet;
                si.dpdv    = FMS(Vec3f(duv02[0]), dp12, duv12[0] * dp02) * invDet;

                si.shading.dndu = FMS(Vec3f(duv12[1]), dn02, duv02[1] * dn12) * invDet;
                si.shading.dndv = FMS(Vec3f(duv02[0]), dn12, duv12[0] * dn02) * invDet;
            }

            Vec3f ss = si.dpdu;
            Vec3f ts = Cross(si.shading.n, ss);
            if (LengthSquared(ts) > 0)
            {
                ss = Cross(ts, si.shading.n);
            }
            else
            {
                CoordinateSystem(si.shading.n, &ss, &ts);
            }
#endif
            si.shading.dpdu   = ss;
            si.shading.dpdv   = ts;
            si.sceneID        = scene->sceneIndex;
            si.geomID         = geomID;
            GetDebug()->scene = scene;
            return true;
        }

        return false;
    }
    template <typename StackEntry, i32 K>
    static bool Intersect(ScenePrimitives *scene, Ray2 &ray, StackEntry entry,
                          SurfaceInteraction &si, TravRay<K> &r)
    {
        Primitive *primitives = (Primitive *)entry.GetPtr();
        u32 num               = entry.GetNum();
        return Intersect(scene, ray, primitives, num, si, r);
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
    template <i32 K>
    static bool Intersect(ScenePrimitives *scene, Ray2 &ray, Primitive *primitives, u32 num,
                          SurfaceInteraction &si, TravRay<K> &)
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
#ifdef USE_QUANTIZE_COMPRESS
            StackEntry entry(prim.nodePtr, prim.GetType(), ray.tFar);
#else
            StackEntry entry(prim.nodePtr, ray.tFar);
#endif

            Assert(childScene && t);
            Vec3f rayO = ray.o;
            Vec3f rayD = ray.d;

            AffineSpace inv = Inverse(*t);
            Mat3 invTp      = Transpose(Mat3(inv.c0, inv.c1, inv.c2));
            ray.o           = TransformP(inv, ray.o);
            ray.d           = TransformV(inv, ray.d);
            if (childScene->intersectFunc(childScene, entry, ray, si))
            {
                si.p            = TransformP(*t, si.p, si.pError);
                si.n            = Normalize(invTp * si.n);
                si.shading.n    = Normalize(invTp * si.shading.n);
                si.shading.dndu = invTp * si.shading.dndu;
                si.shading.dndv = invTp * si.shading.dndv;
                si.dpdu         = TransformV(*t, si.dpdu);
                si.dpdv         = TransformV(*t, si.dpdv);
                si.shading.dpdu = TransformV(*t, si.shading.dpdu);
                si.shading.dpdv = TransformV(*t, si.shading.dpdv);
                // TODO: ?
                si.shading.n = FaceForward(si.shading.n, si.n);

                result = true;
            }
            ray.o = rayO;
            ray.d = rayD;
        }
        return result;
    }

    template <i32 K>
    static bool Occluded(ScenePrimitives *scene, Ray2 &ray, Primitive *primitives, u32 num,
                         SurfaceInteraction &si, TravRay<K> &)
    {
        const Instance *instances = (const Instance *)scene->primitives;
        for (u32 i = 0; i < num; i++)
        {
            Primitive &prim = primitives[i];

            AffineSpace *t;
            ScenePrimitives *childScene;
            prim.GetData(scene, t, childScene);
#ifdef USE_QUANTIZE_COMPRESS
            StackEntry entry(prim.nodePtr, prim.GetType(), ray.tFar);
#else
            StackEntry entry(prim.nodePtr, ray.tFar);
#endif

            Assert(childScene && t);
            Vec3f rayO = ray.o;
            Vec3f rayD = ray.d;

            AffineSpace inv = Inverse(*t);
            Mat3 invTp      = Transpose(Mat3(inv.c0, inv.c1, inv.c2));
            ray.o           = TransformP(inv, ray.o);
            ray.d           = TransformV(inv, ray.d);
            bool result     = childScene->occludedFunc(childScene, entry, ray);
            ray.o           = rayO;
            ray.d           = rayD;
            if (result) return true;
        }
        return false;
    }
    template <typename StackEntry, i32 K>
    static bool Intersect(ScenePrimitives *scene, Ray2 &ray, StackEntry entry,
                          SurfaceInteraction &si, TravRay<K> &r)
    {
        Primitive *primitives = (Primitive *)entry.GetPtr();
        u32 num               = entry.GetNum();
        return Intersect(scene, ray, primitives, num, si, r);
    }

    template <typename StackEntry, i32 K>
    static bool Occluded(ScenePrimitives *scene, Ray2 &ray, StackEntry entry,
                         SurfaceInteraction &si, TravRay<K> &r)
    {
        Primitive *primitives = (Primitive *)entry.GetPtr();
        u32 num               = entry.GetNum();
        return Occluded(scene, ray, primitives, num, si, r);
    }
};

template <i32 N, typename Intersector>
struct CompressedLeafIntersector
{
    using StackEntry = StackEntry<N>;
    using Primitive  = typename Intersector::Primitive;
    Intersector intersector;
    CompressedLeafIntersector() {}
    template <typename Scene>
    bool Intersect(Scene *scene, Ray2 &ray, StackEntry entry, SurfaceInteraction &itr,
                   TravRay<N> &tray)
    {
        bool result = false;

        switch (entry.GetType())
        {
            case BVHNode<N>::tyCompressedLeaf:
            {
                StackEntry stack[N];
                i32 stackPtr = 0;
                BVHTraverser<N, BVH_QNLF, Primitive>::Traverse(entry, stack, stackPtr, tray);

                stackPtr--;
                CompressedLeafNode<N> *node = entry.ptr.GetCompressedLeaf();
                for (; stackPtr >= 0; stackPtr--)
                {
                    // TODO: this is kind of hacky
                    StackEntry e = stack[stackPtr];
                    if (e.dist > ray.tFar) continue;
                    uintptr_t child       = e.ptr.data;
                    u32 start             = (child == 0 ? 0 : node->offsets[child - 1]);
                    Primitive *primitives = (Primitive *)(node + 1) + start;
                    u32 num               = node->offsets[child] - start;

                    result |= intersector.Intersect(scene, ray, primitives, num, itr, tray);
                }
            }
            break;
            default: result |= intersector.Intersect(scene, ray, entry, itr, tray);
        }
        return result;
    }
};

#ifdef USE_QUANTIZE_COMPRESS

template <i32 K, i32 N, i32 types>
using BVHTriangleIntersector = BVHIntersector<K, types, TriangleIntersector<N>>;
typedef BVHTriangleIntersector<4, 1, BVH_QN> BVH4TriangleIntersector1;
typedef BVHTriangleIntersector<4, 8, BVH_QN> BVH4TriangleIntersector8;
typedef BVHTriangleIntersector<8, 8, BVH_QN> BVH8TriangleIntersector8;

template <i32 K, i32 N, i32 types>
using BVHTriangleIntersectorCmp = BVHIntersector<K, types, TriangleIntersectorCmp<N>>;
typedef BVHTriangleIntersectorCmp<4, 1, BVH_QN> BVH4TriangleIntersectorCmp1;
typedef BVHTriangleIntersectorCmp<4, 8, BVH_QN> BVH4TriangleIntersectorCmp8;
typedef BVHTriangleIntersectorCmp<8, 8, BVH_QN> BVH8TriangleIntersectorCmp8;

template <i32 K, i32 N, i32 types>
using BVHQuadIntersector = BVHIntersector<K, types, QuadIntersector<N>>;
typedef BVHQuadIntersector<4, 1, BVH_QN> BVH4QuadIntersector1;
typedef BVHQuadIntersector<4, 8, BVH_QN> BVH4QuadIntersector8;
typedef BVHQuadIntersector<8, 8, BVH_QN> BVH8QuadIntersector8;

template <i32 K, i32 N, i32 types>
using BVHQuadIntersectorCmp = BVHIntersector<K, types, QuadIntersectorCmp<N>>;
typedef BVHQuadIntersectorCmp<4, 1, BVH_QN> BVH4QuadIntersectorCmp1;
typedef BVHQuadIntersectorCmp<4, 8, BVH_QN> BVH4QuadIntersectorCmp8;
typedef BVHQuadIntersectorCmp<8, 8, BVH_QN> BVH8QuadIntersectorCmp8;

template <i32 K, i32 types>
using BVHInstanceIntersector = BVHIntersector<K, types, InstanceIntersector>;
typedef BVHInstanceIntersector<4, BVH_QN> BVH4InstanceIntersector;
typedef BVHInstanceIntersector<8, BVH_QN> BVH8InstanceIntersector;

template <i32 K, i32 types>
using CatClarkIntersector = BVHIntersector<K, types, CatClarkPatchIntersector>;
typedef CatClarkIntersector<8, BVH_QN> BVH8PatchIntersector;

#else

// NOTE: K is for BVH branching factor, N is for Primitive
template <i32 K, i32 N, i32 types>
using BVHTriangleIntersector =
    BVHIntersector<K, types, CompressedLeafIntersector<K, TriangleIntersector<N>>>;
typedef BVHTriangleIntersector<4, 1, BVH_QN> BVH4TriangleIntersector1;
typedef BVHTriangleIntersector<4, 8, BVH_QN> BVH4TriangleIntersector8;
typedef BVHTriangleIntersector<8, 8, BVH_QN> BVH8TriangleIntersector8;

template <i32 K, i32 N, i32 types>
using BVHTriangleIntersectorCmp =
    BVHIntersector<K, types, CompressedLeafIntersector<K, TriangleIntersectorCmp<N>>>;
typedef BVHTriangleIntersectorCmp<4, 1, BVH_QN> BVH4TriangleIntersectorCmp1;
typedef BVHTriangleIntersectorCmp<4, 8, BVH_QN> BVH4TriangleIntersectorCmp8;
typedef BVHTriangleIntersectorCmp<8, 8, BVH_QN> BVH8TriangleIntersectorCmp8;

template <i32 K, i32 N, i32 types>
using BVHQuadIntersector =
    BVHIntersector<K, types, CompressedLeafIntersector<K, QuadIntersector<N>>>;
typedef BVHQuadIntersector<4, 1, BVH_QN> BVH4QuadIntersector1;
typedef BVHQuadIntersector<4, 8, BVH_QN> BVH4QuadIntersector8;
typedef BVHQuadIntersector<8, 8, BVH_QN> BVH8QuadIntersector8;

template <i32 K, i32 N, i32 types>
using BVHQuadIntersectorCmp =
    BVHIntersector<K, types, CompressedLeafIntersector<K, QuadIntersectorCmp<N>>>;
typedef BVHQuadIntersectorCmp<4, 1, BVH_QN> BVH4QuadIntersectorCmp1;
typedef BVHQuadIntersectorCmp<4, 8, BVH_QN> BVH4QuadIntersectorCmp8;
typedef BVHQuadIntersectorCmp<8, 8, BVH_QN> BVH8QuadIntersectorCmp8;

template <i32 K, i32 types>
using BVHInstanceIntersector =
    BVHIntersector<K, types, CompressedLeafIntersector<K, InstanceIntersector>>;
typedef BVHInstanceIntersector<4, BVH_QN> BVH4InstanceIntersector;
typedef BVHInstanceIntersector<8, BVH_QN> BVH8InstanceIntersector;

template <i32 K, i32 types>
using CatClarkIntersector =
    BVHIntersector<K, types, CompressedLeafIntersector<K, CatClarkPatchIntersector>>;
typedef CatClarkIntersector<8, BVH_QN> BVH8PatchIntersector;
#endif

// Helpers
template <i32 N, GeometryType type, typename PrimRefType>
struct IntersectorHelperBase;

template <>
struct IntersectorHelperBase<4, GeometryType::TriangleMesh, PrimRefCompressed>
{
    using IntersectorType = BVH4TriangleIntersectorCmp8;
};

template <>
struct IntersectorHelperBase<4, GeometryType::TriangleMesh, PrimRef>
{
    using IntersectorType = BVH4TriangleIntersector8;
};

template <>
struct IntersectorHelperBase<4, GeometryType::QuadMesh, PrimRefCompressed>
{
    using IntersectorType = BVH4QuadIntersectorCmp8;
};

template <>
struct IntersectorHelperBase<4, GeometryType::QuadMesh, PrimRef>
{
    using IntersectorType = BVH4QuadIntersector8;
};

template <>
struct IntersectorHelperBase<4, GeometryType::Instance, BRef>
{
    using IntersectorType = BVH4InstanceIntersector;
};

template <>
struct IntersectorHelperBase<8, GeometryType::TriangleMesh, PrimRefCompressed>
{
    using IntersectorType = BVH8TriangleIntersectorCmp8;
};

template <>
struct IntersectorHelperBase<8, GeometryType::TriangleMesh, PrimRef>
{
    using IntersectorType = BVH8TriangleIntersector8;
};

template <>
struct IntersectorHelperBase<8, GeometryType::QuadMesh, PrimRefCompressed>
{
    using IntersectorType = BVH8QuadIntersectorCmp8;
};

template <>
struct IntersectorHelperBase<8, GeometryType::QuadMesh, PrimRef>
{
    using IntersectorType = BVH8QuadIntersector8;
};

template <>
struct IntersectorHelperBase<8, GeometryType::Instance, BRef>
{
    using IntersectorType = BVH8InstanceIntersector;
};

template <>
struct IntersectorHelperBase<8, GeometryType::CatmullClark, PrimRef>
{
    using IntersectorType = BVH8PatchIntersector;
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
