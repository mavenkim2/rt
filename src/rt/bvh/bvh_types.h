#ifndef BVH_TYPES_H
#define BVH_TYPES_H
#include <immintrin.h>
namespace rt
{
static const u32 LANE_WIDTH         = 8;
static const i32 LANE_WIDTHi        = 8;
static const u32 GROW_AMOUNT        = 2;         // 1.2f;
static const u32 PARALLEL_THRESHOLD = 32 * 1024; // 32 * 1024; // 64 * 1024;
                                                 //
#define EXPONENTIAL_QUANTIZE
#define USE_QUANTIZE_COMPRESS

struct BuildSettings
{
    u32 maxLeafSize  = 7;
    u32 logBlockSize = 0;
    i32 maxDepth     = 32;
    f32 intCost      = 1.f;
    f32 travCost     = 1.f;

    u32 blockAdd;
};

struct PartitionPayload
{
    u32 *lOffsets;
    u32 *rOffsets;
    u32 *lCounts;
    u32 *rCounts;
    u32 count;
    u32 groupSize;
    PartitionPayload() {}
    PartitionPayload(u32 *lOffsets, u32 *rOffsets, u32 *lCounts, u32 *rCounts, u32 count,
                     u32 groupSize)
        : lOffsets(lOffsets), rOffsets(rOffsets), lCounts(lCounts), rCounts(rCounts),
          count(count), groupSize(groupSize)
    {
    }
};

struct Split
{
    enum Type
    {
        Object,
        Spatial,
    };
    Type type;
    f32 bestSAH;
    u32 bestPos;
    u32 bestDim;
    f32 bestValue;

    // TODO: this is maybe a bit jank
    void *ptr;
    struct SplitPayload
    {
        u32 *splitOffsets = 0;
        u32 num           = 0;
    };
    SplitPayload payload;
    u32 numLeft;
    u32 numRight;

    u64 allocPos;

    Split() {}
    Split(f32 sah, u32 pos, u32 dim, f32 val)
        : bestSAH(sah), bestPos(pos), bestDim(dim), bestValue(val)
    {
    }
};

struct PrimRef
{
    union
    {
        __m256 m256;
        struct
        {
            f32 minX, minY, minZ;
            u32 geomID;
            f32 maxX, maxY, maxZ;
            u32 primID;
        };
        struct
        {
            f32 min[3];
            u32 geomID_;
            f32 max[3];
            u32 primID_;
        };
    };
    PrimRef() {}
    PrimRef(const Lane8F32 &l) { MemoryCopy(this, &l, sizeof(PrimRef)); }
    __forceinline Lane8F32 Load() const { return Lane8F32::Load(&m256); }
};

// NOTE: if BVH is built over only one quad mesh. must make sure to pad with an extra entry
// when allocating
struct PrimRefCompressed
{
    union
    {
        struct
        {
            f32 minX, minY, minZ;
            u32 primID;
            f32 maxX, maxY, maxZ;
        };
        struct
        {
            f32 min[3];
            u32 primID_;
            f32 max[3];
        };
    };
    PrimRefCompressed() {}
    PrimRefCompressed(const Lane8F32 &l) { MemoryCopy(this, &l, sizeof(PrimRef)); }
    __forceinline Lane8F32 Load() const { return Lane8F32::LoadU(this); }
};

struct ExtRange
{
    u32 start;
    u32 count;
    u32 extEnd;

    ExtRange() {}
    __forceinline ExtRange(u32 start, u32 count, u32 extEnd)
        : start(start), count(count), extEnd(extEnd)
    {
        Assert(extEnd >= start + count);
    }
    __forceinline u32 End() const { return start + count; }
    __forceinline u32 Size() const { return count; }
    __forceinline u32 ExtSize() const { return extEnd - (start + count); }
    __forceinline u32 TotalSize() const { return extEnd - start; }
};

struct alignas(CACHE_LINE_SIZE) RecordAOSSplits
{
    using PrimitiveData = PrimRef;
    union
    {
        struct
        {
            Lane8F32 geomBounds;
            Lane8F32 centBounds;
        };
        struct
        {
            f32 geomMin[3];
            u32 start;
            f32 geomMax[3];
            u32 count;
            f32 centMin[3];
            u32 extEnd;
            f32 centMax[3];
            u32 pad_;
        };
    };

    RecordAOSSplits() {}
    RecordAOSSplits(NegInfTy) : geomBounds(neg_inf), centBounds(neg_inf) {}
    __forceinline RecordAOSSplits &operator=(const RecordAOSSplits &other)
    {
        geomBounds = other.geomBounds;
        centBounds = other.centBounds;
        return *this;
    }
    u32 Start() const { return start; }
    u32 Count() const { return count; }
    u32 End() const { return start + count; }
    u32 ExtEnd() const { return extEnd; }
    u32 ExtSize() const { return extEnd - (start + count); }
    void SetRange(u32 inStart, u32 inCount)
    {
        start = inStart;
        count = inCount;
    }
    void SetRange(u32 inStart, u32 inCount, u32 inExtEnd)
    {
        start  = inStart;
        count  = inCount;
        extEnd = inExtEnd;
    }
    void Merge(const RecordAOSSplits &other)
    {
        geomBounds = Max(geomBounds, other.geomBounds);
        centBounds = Max(centBounds, other.centBounds);
    }
    void SafeMerge(const RecordAOSSplits &other)
    {
        u32 s = start;
        u32 c = count;
        u32 e = extEnd;
        Merge(other);
        start  = s;
        count  = c;
        extEnd = e;
    }
};

//////////////////////////////
// BVH Representation
//
template <i32 N>
struct QuantizedNode;

template <i32 N>
struct QuantizedCompressedNode;

template <i32 N>
struct CompressedLeafNode;

template <template <i32> class Node, i32 N>
void GetBounds(const Node<N> *node, LaneF32<N> *outMin, LaneF32<N> *outMax)
{
    Lane4U32 lX, lY, lZ;
    Lane4U32 uX, uY, uZ;

    if constexpr (N == 4)
    {
        lX = _mm_cvtsi32_si128(*(u32 *)node->lowerX);
        lY = _mm_cvtsi32_si128(*(u32 *)node->lowerY);
        lZ = _mm_cvtsi32_si128(*(u32 *)node->lowerZ);

        uX = _mm_cvtsi32_si128(*(u32 *)node->upperX);
        uY = _mm_cvtsi32_si128(*(u32 *)node->upperY);
        uZ = _mm_cvtsi32_si128(*(u32 *)node->upperZ);
    }
    else
    {
        lX = _mm_set_epi64x(0, *(u64 *)node->lowerX);
        lY = _mm_set_epi64x(0, *(u64 *)node->lowerY);
        lZ = _mm_set_epi64x(0, *(u64 *)node->lowerZ);

        uX = _mm_set_epi64x(0, *(u64 *)node->upperX);
        uY = _mm_set_epi64x(0, *(u64 *)node->upperY);
        uZ = _mm_set_epi64x(0, *(u64 *)node->upperZ);
    }

    LaneU32<N> lExpandedMinX;
    LaneU32<N> lExpandedMinY;
    LaneU32<N> lExpandedMinZ;

    LaneU32<N> lExpandedMaxX;
    LaneU32<N> lExpandedMaxY;
    LaneU32<N> lExpandedMaxZ;
    if constexpr (N == 4)
    {
        lExpandedMinX = _mm_cvtepu16_epi32(_mm_cvtepu8_epi16(lX));
        lExpandedMinY = _mm_cvtepu16_epi32(_mm_cvtepu8_epi16(lY));
        lExpandedMinZ = _mm_cvtepu16_epi32(_mm_cvtepu8_epi16(lZ));

        lExpandedMaxX = _mm_cvtepu16_epi32(_mm_cvtepu8_epi16(uX));
        lExpandedMaxY = _mm_cvtepu16_epi32(_mm_cvtepu8_epi16(uY));
        lExpandedMaxZ = _mm_cvtepu16_epi32(_mm_cvtepu8_epi16(uZ));
    }
    else
    {
        lExpandedMinX = _mm256_cvtepu8_epi32(lX);
        lExpandedMinY = _mm256_cvtepu8_epi32(lY);
        lExpandedMinZ = _mm256_cvtepu8_epi32(lZ);
        lExpandedMaxX = _mm256_cvtepu8_epi32(uX);
        lExpandedMaxY = _mm256_cvtepu8_epi32(uY);
        lExpandedMaxZ = _mm256_cvtepu8_epi32(uZ);
    }
    LaneF32<N> minX(node->minP.x);
    LaneF32<N> minY(node->minP.y);
    LaneF32<N> minZ(node->minP.z);

#ifdef EXPONENTIAL_QUANTIZE
    LaneF32<N> scaleX = AsFloat(LaneU32<N>(node->scale[0] << 23));
    LaneF32<N> scaleY = AsFloat(LaneU32<N>(node->scale[1] << 23));
    LaneF32<N> scaleZ = AsFloat(LaneU32<N>(node->scale[2] << 23));
#else
    LaneF32<N> scaleX = LaneF32<N>(node->scale[0]);
    LaneF32<N> scaleY = LaneF32<N>(node->scale[1]);
    LaneF32<N> scaleZ = LaneF32<N>(node->scale[2]);
#endif

    outMin[0] = FMA(LaneF32<N>(lExpandedMinX), scaleX, minX);
    outMin[1] = FMA(LaneF32<N>(lExpandedMinY), scaleY, minY);
    outMin[2] = FMA(LaneF32<N>(lExpandedMinZ), scaleZ, minZ);
    outMax[0] = FMA(LaneF32<N>(lExpandedMaxX), scaleX, minX);
    outMax[1] = FMA(LaneF32<N>(lExpandedMaxY), scaleY, minY);
    outMax[2] = FMA(LaneF32<N>(lExpandedMaxZ), scaleZ, minZ);
}

template <i32 N>
struct BVHNode
{
    static const size_t alignment = 16;
    static const size_t alignMask = alignment - 1;
    StaticAssert(IsPow2(alignment), Pow2Alignment);
    static const size_t tyQuantizedNode  = 0;
    static const size_t tyEmpty          = 7;
    static const size_t tyCompressedLeaf = 8;
    static const size_t tyLeaf           = 8;
    // NOTE: the leaf count is count - tyLeaf

    uintptr_t data;

    BVHNode() {}
    BVHNode(uintptr_t data) : data(data) {}
    static void CheckAlignment(void *ptr);
    static BVHNode<N> EncodeNode(QuantizedNode<N> *node);
    static BVHNode<N> EncodeQuantizedCompressedNode(QuantizedCompressedNode<N> *node);
    static BVHNode<N> EncodeCompressedNode(CompressedLeafNode<N> *node);
    static BVHNode<N> EncodeLeaf(void *leaf, u32 num);
    static BVHNode<N> EncodeEmpty() { return BVHNode<N>(tyEmpty); }
    QuantizedNode<N> *GetQuantizedNode() const;
    QuantizedCompressedNode<N> *GetQuantizedCompressedNode() const;
    CompressedLeafNode<N> *GetCompressedLeaf() const;
    void *GetPtr() const { return (void *)(data & ~alignMask); }
    u32 GetNum() const { return GetType() - BVHNode<N>::tyLeaf; }
    u32 GetType() const { return u32(data & alignMask); }
    bool IsLeaf() const { return GetType() >= tyLeaf; }
    bool IsEmpty() const { return data == tyEmpty; }
    bool IsQuantizedNode() const { return GetType() == tyQuantizedNode; }
    bool IsCompressedLeaf() const { return GetType() == tyCompressedLeaf; }
};

template <i32 N>
void BVHNode<N>::CheckAlignment(void *ptr)
{
    Assert(!((size_t)ptr & alignMask));
}

template <i32 N>
BVHNode<N> BVHNode<N>::EncodeNode(QuantizedNode<N> *node)
{
    CheckAlignment(node);
    return BVHNode((size_t)node | tyQuantizedNode);
}

template <i32 N>
BVHNode<N> BVHNode<N>::EncodeQuantizedCompressedNode(QuantizedCompressedNode<N> *node)
{
    CheckAlignment(node);
    return BVHNode((size_t)node | tyQuantizedNode);
}

template <i32 N>
BVHNode<N> BVHNode<N>::EncodeCompressedNode(CompressedLeafNode<N> *node)
{
    CheckAlignment(node);
    return BVHNode((size_t)node | tyCompressedLeaf);
}

template <i32 N>
BVHNode<N> BVHNode<N>::EncodeLeaf(void *leaf, u32 num)
{
    CheckAlignment(leaf);
    Assert(num >= 1 && num <= 7);
    return BVHNode((size_t)leaf | (tyLeaf + num));
}

template <i32 N>
QuantizedNode<N> *BVHNode<N>::GetQuantizedNode() const
{
    Assert(IsQuantizedNode());
    return (QuantizedNode<N> *)(data & ~(0xf));
}

template <i32 N>
QuantizedCompressedNode<N> *BVHNode<N>::GetQuantizedCompressedNode() const
{
    Assert(IsQuantizedNode());
    return (QuantizedCompressedNode<N> *)(data & ~(0xf));
}

template <i32 N>
CompressedLeafNode<N> *BVHNode<N>::GetCompressedLeaf() const
{
    Assert(IsCompressedLeaf());
    return (CompressedLeafNode<N> *)(data & ~alignMask);
}

typedef BVHNode<4> BVHNode4;
typedef BVHNode<8> BVHNode8;

template <i32 N>
struct BuildRef
{
    f32 min[3];
    // u32 objectID;
    u32 instanceID;
    f32 max[3];
    u32 numPrims;
    // uintptr_t ptr;
    BVHNode<N> nodePtr;
#ifdef USE_QUANTIZE_COMPRESS
    u32 type;
#endif

    __forceinline Lane8F32 Load() const { return Lane8F32::LoadU(min); }
    __forceinline void StoreBounds(const Bounds &b)
    {
        Lane4F32::StoreU(min, -b.minP);
        Lane4F32::StoreU(max, b.maxP);
    }
    __forceinline void SafeStoreBounds(const Bounds &b)
    {
        u32 i = instanceID;
        u32 n = numPrims;
        Lane4F32::StoreU(min, -b.minP);
        Lane4F32::StoreU(max, b.maxP);
        instanceID = i;
        numPrims   = n;
    }
};

typedef BuildRef<4> BuildRef4;
typedef BuildRef<8> BuildRef8;

template <i32 N>
struct QuantizedNode
{
    StaticAssert(N == 4 || N == 8, NMustBe4Or8);

    BVHNode<N> children[N];
#ifdef EXPONENTIAL_QUANTIZE
    u8 scale[3];
#else
    Vec3f scale;
#endif
    // u8 meta;

    u8 lowerX[N];
    u8 lowerY[N];
    u8 lowerZ[N];

    // NOTE: upperX = 255 when node is invalid
    u8 upperX[N];
    u8 upperY[N];
    u8 upperZ[N];

    Vec3f minP;

    // NOTE: nodes + leaves
    u32 GetNumChildren() const
    {
        int count = 0;
        for (int i = 0; i < N; i++)
        {
            count += (children[i].GetType() != BVHNode<N>::tyEmpty);
        }
        return count;
    }

    BVHNode<N> Child(u32 i) const { return children[i]; }

    u32 GetType(u32 childIndex) const { return children[childIndex].GetType(); }

    void GetBounds(LaneF32<N> *outMin, LaneF32<N> *outMax) const
    {
        ::GetBounds(this, outMin, outMax);
    }
};

template <i32 N>
struct QuantizedCompressedNode
{
    StaticAssert(N == 4 || N == 8, NMustBe4Or8);

    static const int EmptyNodeValue = 0xff;

    u8 meta[N];
    BVHNode<N> basePtr;
#ifdef EXPONENTIAL_QUANTIZE
    u8 scale[3];
#else
    Vec3f scale;
#endif

    u8 baseMeta;
    u8 lowerX[N];
    u8 lowerY[N];
    u8 lowerZ[N];

    // NOTE: upperX = 255 when node is invalid
    u8 upperX[N];
    u8 upperY[N];
    u8 upperZ[N];

    Vec3f minP;

    BVHNode<N> Child(u32 i, int size) const
    {
        if (meta[i] == EmptyNodeValue) return BVHNode<N>::EncodeEmpty();
        // u32 type = baseMeta & (1 << i);
        bool type = (baseMeta >> i) & 1;

        // Calculates the index
        if (type)
        {
            int index = PopCount(((u32)baseMeta) & ((1 << i) - 1));
            Assert(index < N);
            Assert(!basePtr.IsEmpty());
            auto *node = basePtr.GetQuantizedCompressedNode() + index;
            return BVHNode<N>((uintptr_t)node);
        }

        int numNodes = PopCount(baseMeta);
        int start    = i == 0 ? 0 : meta[i - 1];
        Assert(!basePtr.IsEmpty());
        auto *ptr = basePtr.GetQuantizedCompressedNode() + numNodes;
        return BVHNode<N>((uintptr_t)((u8 *)ptr + size * start));
    }

    void GetChildren(BVHNode<N> children[N], int size) const
    {
        int nodeMask  = (int)baseMeta;
        uintptr_t ptr = (uintptr_t)basePtr.GetPtr();
        int leafStart = PopCount(baseMeta) * sizeof(QuantizedCompressedNode<N>);
        int nodeCount = 0;
        for (int i = 0; i < N; i++)
        {
            bool type       = (nodeMask >> i) & 1;
            int leafOffset  = leafStart + (i == 0 ? 0 : meta[i - 1] * size);
            int childOffset = nodeCount * (int)sizeof(QuantizedCompressedNode<N>);
            nodeCount += type;
            int offset  = leafOffset ^ ((leafOffset ^ childOffset) & (0 - type));
            children[i] = BVHNode<N>(ptr + offset);
        }
    }

    // NOTE: nodes + leaves
    u32 GetNumChildren() const
    {
        int count = 0;
        for (int i = 0; i < N; i++)
        {
            count += meta[i] != EmptyNodeValue;
        }
        return count;
    }

    __forceinline u32 GetType(u32 childIndex) const
    {
        if (meta[childIndex] == EmptyNodeValue) return BVHNode<N>::tyEmpty;
        u32 type = baseMeta & (1 << childIndex);
        if (type)
        {
            return BVHNode<N>::tyQuantizedNode;
        }
        else
        {
            int start = childIndex == 0 ? 0 : meta[childIndex - 1];
            ErrorExit(meta[childIndex] > start, "meta: %i, start: %i\n", meta[childIndex],
                      start);
            int count = meta[childIndex] - start;
            ErrorExit(count && count <= 7, "count: %i\n", count);
            return BVHNode<N>::tyLeaf + count;
        }
    }

    LaneF32<N> GetValid() const
    {
        if constexpr (N == 4)
        {
            Lane4U32 lane = _mm_cvtsi32_si128(*(u32 *)meta);
            return lane != Lane4U32(EmptyNodeValue);
        }
        else
        {
            Lane8U32 lane = _mm256_cvtepu8_epi32(_mm_set_epi64x(0, *(u64 *)meta));
            return lane != Lane8U32(EmptyNodeValue);
        }
    }

    void GetBounds(LaneF32<N> *outMin, LaneF32<N> *outMax) const
    {
        ::GetBounds(this, outMin, outMax);
    }
};

template <i32 N>
using QuantizedNode4 = QuantizedNode<4>;
template <i32 N>
using QuantizedNode8 = QuantizedNode<8>;

template <i32 N>
struct CompressedLeafNode
{
    StaticAssert(N == 4 || N == 8, NMustBe4Or8);

#ifdef EXPONENTIAL_QUANTIZE
    u8 scale[3];
#else
    Vec3f scale;
#endif

    u8 offsets[N];
    u8 lowerX[N];
    u8 lowerY[N];
    u8 lowerZ[N];

    u8 upperX[N];
    u8 upperY[N];
    u8 upperZ[N];

    Vec3f minP;
    void GetBounds(LaneF32<N> *outMin, LaneF32<N> *outMax) const
    {
        ::GetBounds(this, outMin, outMax);
    }
    BVHNode<N> Child(u32 index) const { return BVHNode<N>(index); }
    u32 GetType(u32 childIndex) const
    {
        return offsets[childIndex] == 0 ? BVHNode<N>::tyEmpty : 0;
    }
};

// #define USE_BVH4
#define USE_BVH8

#if !defined(USE_BVH4) && !defined(USE_BVH8)
#define USE_BVH4
#endif

#if defined(USE_BVH4) && defined(USE_BVH8)
#undef USE_BVH4
#endif

#ifdef USE_BVH4
typedef BVHNode<4> BVHNodeN;
typedef BuildRef<4> BRef;
typedef QuantizedNode<4> QNode;
typedef QuantizedCompressedNode<4> QCNode;
static const u32 DefaultN    = 4;
static const u32 DefaultLogN = 2;
#elif defined(USE_BVH8)
typedef BVHNode<8> BVHNodeN;
typedef BuildRef<8> BRef;
typedef QuantizedNode<8> QNode;
typedef QuantizedCompressedNode<8> QCNode;
static const u32 DefaultN    = 8;
static const u32 DefaultLogN = 3;
#endif

struct ScenePrimitives;
template <i32 N>
struct LeafPrim
{
    u32 geomIDs[N];
    u32 primIDs[N];
    LeafPrim() {}
    __forceinline void Fill(const ScenePrimitives *, PrimRef *refs, u32 &begin, u32 end);
    __forceinline void Fill(LeafPrim<1> *prims, u32 num)
    {
        Assert(num);
        u32 last;
        for (u32 i = 0; i < N; i++)
        {
            if (i < num)
            {
                geomIDs[i] = prims[i].geomIDs[0];
                primIDs[i] = prims[i].primIDs[0];
                last       = i;
            }
            else
            {
                geomIDs[i] = prims[last].geomIDs[0];
                primIDs[i] = prims[last].primIDs[0];
            }
        }
    }
};

template <i32 N>
struct LeafPrimCompressed
{
    u32 primIDs[N];
    LeafPrimCompressed() {}
    __forceinline void Fill(const ScenePrimitives *, PrimRefCompressed *refs, u32 &begin,
                            u32 end);
    __forceinline void Fill(LeafPrimCompressed<1> *prims, u32 num)
    {
        Assert(num);
        u32 last;
        for (u32 i = 0; i < N; i++)
        {
            if (i < num)
            {
                primIDs[i] = prims[i].primIDs[0];
                last       = i;
            }
            else
            {
                primIDs[i] = prims[last].primIDs[0];
            }
        }
    }
};

template <i32 N>
struct Triangle : LeafPrim<N>
{
    Triangle() {}

    void GetData(const ScenePrimitives *scene, Lane4F32 v0[N], Lane4F32 v1[N], Lane4F32 v2[N],
                 u32 outGeomIDs[N], u32 outPrimIDs[N]) const;
};

template <i32 N>
struct TriangleCompressed : LeafPrimCompressed<N>
{
    TriangleCompressed() {}

    void GetData(const ScenePrimitives *scene, Lane4F32 v0[N], Lane4F32 v1[N], Lane4F32 v2[N],
                 u32 outGeomIDs[N], u32 outPrimIDs[N]) const;
};

template <i32 N>
struct Quad : LeafPrim<N>
{
    Quad() {}

    void GetData(const ScenePrimitives *scene, Lane4F32 v0[N], Lane4F32 v1[N], Lane4F32 v2[N],
                 Lane4F32 v3[N], u32 outGeomIDs[N], u32 outPrimIDs[N]) const;
};

template <i32 N>
struct QuadCompressed : LeafPrimCompressed<N>
{
    QuadCompressed() {}

    void GetData(const ScenePrimitives *scene, Lane4F32 v0[N], Lane4F32 v1[N], Lane4F32 v2[N],
                 Lane4F32 v3[N], u32 outGeomIDs[N], u32 outPrimIDs[N]) const;
};

struct CatmullClarkPatch
{
    u32 geomID;
    u32 primID;

    CatmullClarkPatch() {}
    __forceinline void Fill(const ScenePrimitives *scene, PrimRef *ref, u32 &begin, u32 end)
    {
        Assert(end > begin);

        geomID = ref[begin].geomID;
        primID = ref[begin].primID;
        begin++;
    }
};

struct TLASLeaf
{
    static const u32 sceneIndexMask = 0x0fffffff;
    static const u32 typeShift      = 28;
    BVHNodeN nodePtr;
    u32 sceneIndex;
    u32 transformIndex;
    TLASLeaf() {}
    template <i32 N>
    __forceinline void Fill(const ScenePrimitives *scene, BuildRef<N> *refs, u32 &begin,
                            u32 end)
    {
        Assert(end > begin);
        const Instance *instances = (const Instance *)scene->primitives;

        BuildRef<N> *ref  = &refs[begin];
        u32 instanceIndex = ref->instanceID;
        nodePtr           = ref->nodePtr;
        Assert(instanceIndex < scene->numPrimitives);
        sceneIndex = instances[instanceIndex].id;
        Assert(sceneIndex < scene->numChildScenes);
#ifdef USE_QUANTIZE_COMPRESS
        Assert(sceneIndex <= sceneIndexMask);
        Assert(ref->type <= 0xf);
        sceneIndex |= (ref->type << typeShift);
#endif
        transformIndex = instances[instanceIndex].transformIndex;
        ErrorExit(transformIndex < scene->numTransforms, "transformIndex: %i\n",
                  transformIndex);
        begin++;
    }
#ifdef USE_QUANTIZE_COMPRESS
    u32 GetType() const { return sceneIndex >> typeShift; }
#endif

    u32 GetSceneIndex() const
    {
#ifdef USE_QUANTIZE_COMPRESS
        return sceneIndex & sceneIndexMask;
#else
        return sceneIndex;
#endif
    }
    void GetData(const ScenePrimitives *scene, AffineSpace *&t, ScenePrimitives *&childScene);
};

} // namespace rt
#endif
