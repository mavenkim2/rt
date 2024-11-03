#ifndef BVH_TYPES_H
#define BVH_TYPES_H
namespace rt
{
static const u32 LANE_WIDTH         = 8;
static const i32 LANE_WIDTHi        = 8;
static const u32 GROW_AMOUNT        = 2;         // 1.2f;
static const u32 PARALLEL_THRESHOLD = 32 * 1024; // 32 * 1024; // 64 * 1024;

struct BuildSettings
{
    u32 maxLeafSize = 7;
    i32 maxDepth    = 32;
    f32 intCost     = 1.f;
    f32 travCost    = 1.f;
    bool twoLevel   = true;
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
    PartitionPayload(u32 *lOffsets, u32 *rOffsets, u32 *lCounts, u32 *rCounts, u32 count, u32 groupSize)
        : lOffsets(lOffsets), rOffsets(rOffsets), lCounts(lCounts), rCounts(rCounts), count(count), groupSize(groupSize) {}
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
    Split(f32 sah, u32 pos, u32 dim, f32 val) : bestSAH(sah), bestPos(pos), bestDim(dim), bestValue(val) {}
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
    PrimRef(const Lane8F32 &l)
    {
        MemoryCopy(this, &l, sizeof(PrimRef));
    }
    __forceinline Lane8F32 Load() const
    {
        return Lane8F32::Load(&m256);
    }
};

// NOTE: if BVH is built over only one quad mesh. must make sure to pad with an extra entry when allocating
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
    PrimRefCompressed(const Lane8F32 &l)
    {
        MemoryCopy(this, &l, sizeof(PrimRef));
    }
    __forceinline Lane8F32 Load() const
    {
        return Lane8F32::LoadU(this);
    }
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

struct ScalarBounds
{
    f32 minX;
    f32 minY;
    f32 minZ;
    f32 maxX;
    f32 maxY;
    f32 maxZ;

    Bounds ToBounds() const
    {
        Bounds result;
        result.minP = Lane4F32::LoadU(&minX);
        result.maxP = Lane4F32::LoadU(&maxX);
        return result;
    }
    // NOTE: geomBounds must be set first, then cent bounds, and then the range in RecordSOASplits must be updated
    void FromBounds(const Bounds &b)
    {
        Lane4F32::StoreU(&minX, b.minP);
        Lane4F32::StoreU(&maxX, b.maxP);
    }
    f32 HalfArea() const
    {
        f32 diffX = maxX - minX;
        f32 diffY = maxY - minY;
        f32 diffZ = maxZ - minZ;
        return FMA(diffX, diffY + diffZ, diffY * diffZ);
    }
};

f32 HalfArea(const ScalarBounds &b)
{
    return b.HalfArea();
}

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
struct CompressedLeafNode
{
    StaticAssert(N == 4 || N == 8, NMustBe4Or8);

    u8 scale[3];
    u8 meta;

    u8 lowerX[N];
    u8 lowerY[N];
    u8 lowerZ[N];

    u8 upperX[N];
    u8 upperY[N];
    u8 upperZ[N];

    Vec3f minP;
};

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
    static BVHNode<N> EncodeCompressedNode(CompressedLeafNode<N> *node);
    static BVHNode<N> EncodeLeaf(void *leaf, u32 num);
    QuantizedNode<N> *GetQuantizedNode() const;
    size_t GetType() const { return data & alignMask; }
    bool IsLeaf() const { return GetType() >= tyLeaf; }
    bool IsQuantizedNode() const { return GetType() == tyQuantizedNode; }
};

template <i32 N>
void BVHNode<N>::CheckAlignment(void *ptr)
{
    Assert(!((size_t)ptr & alignMask));
    assert(!((size_t)ptr & alignMask));
}

template <i32 N>
BVHNode<N> BVHNode<N>::EncodeNode(QuantizedNode<N> *node)
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
    Assert(num >= 1);
    return BVHNode((size_t)leaf | (tyLeaf + num));
}

template <i32 N>
QuantizedNode<N> *BVHNode<N>::GetQuantizedNode() const
{
    Assert(IsQuantizedNode());
    return (QuantizedNode<N> *)(data & ~(0xf));
}

typedef BVHNode<4> BVHNode4;
typedef BVHNode<8> BVHNode8;

template <u32 N>
struct BuildRef
{
    f32 min[3];
    // u32 objectID;
    u32 instanceID;
    f32 max[3];
    u32 numPrims;
    BVHNode<N> nodePtr;

    __forceinline Lane8F32 Load() const
    {
        return Lane8F32::LoadU(min);
    }
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
    u8 scale[3];
    u8 meta;

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
        return PopCount(meta);
    }
    const BVHNode<N> &Child(u32 i) const
    {
        return children[i];
    }

    void GetBounds(LaneF32<N> *outMin, LaneF32<N> *outMax) // f32 *outMinX, f32 *outMinY, f32 *outMinZ, f32 *outMaxX, f32 *outMaxY, f32 *outMaxZ) const
    {
        LaneU32<N> lX = LaneU32<N>(*(u32 *)lowerX);
        LaneU32<N> lY = LaneU32<N>(*(u32 *)lowerY);
        LaneU32<N> lZ = LaneU32<N>(*(u32 *)lowerZ);

        LaneU32<N> uX = LaneU32<N>(*(u32 *)upperX);
        LaneU32<N> uY = LaneU32<N>(*(u32 *)upperY);
        LaneU32<N> uZ = LaneU32<N>(*(u32 *)upperZ);

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
        LaneF32<N> minX(minP.x);
        LaneF32<N> minY(minP.y);
        LaneF32<N> minZ(minP.z);

        LaneF32<N> scaleX = AsFloat(LaneU32<N>(scale[0] << 23));
        LaneF32<N> scaleY = AsFloat(LaneU32<N>(scale[1] << 23));
        LaneF32<N> scaleZ = AsFloat(LaneU32<N>(scale[2] << 23));

        //     LaneF32<N>::Store(outMinX, minX + LaneF32<N>(lExpandedMinX) * scaleX);
        //     LaneF32<N>::Store(outMinY, minY + LaneF32<N>(lExpandedMinY) * scaleY);
        //     LaneF32<N>::Store(outMinZ, minZ + LaneF32<N>(lExpandedMinZ) * scaleZ);
        //     LaneF32<N>::Store(outMaxX, minX + LaneF32<N>(lExpandedMaxX) * scaleX);
        //     LaneF32<N>::Store(outMaxY, minY + LaneF32<N>(lExpandedMaxY) * scaleY);
        //     LaneF32<N>::Store(outMaxZ, minZ + LaneF32<N>(lExpandedMaxZ) * scaleZ);
        outMin[0] = FMA(LaneF32<N>(lExpandedMinX), scaleX, minX);
        outMin[1] = FMA(LaneF32<N>(lExpandedMinY), scaleY, minY);
        outMin[2] = FMA(LaneF32<N>(lExpandedMinZ), scaleZ, minZ);
        outMax[0] = FMA(LaneF32<N>(lExpandedMaxX), scaleX, minX);
        outMax[1] = FMA(LaneF32<N>(lExpandedMaxY), scaleY, minY);
        outMax[2] = FMA(LaneF32<N>(lExpandedMaxZ), scaleZ, minZ);
    }
    // }
    QuantizedNode<N> GetBaseChildPtr() const
    {
        return (QuantizedNode<N> *)(internalOffset & ~(0xf));
    }
};

template <i32 N>
using QuantizedNode4 = QuantizedNode<4>;
template <i32 N>
using QuantizedNode8 = QuantizedNode<8>;

#define USE_BVH4
// #define USE_BVH8

#if defined(USE_BVH4) && defined(USE_BVH8)
#undef USE_BVH4
#endif

#ifdef USE_BVH4
typedef BVHNode<4> BVHNodeType;
typedef BuildRef<4> BRef;
typedef QuantizedNode<4> QNode;
static const u32 DefaultN = 4;
#elif defined(USE_BVH8)
typedef BVHNode<8> BVHNodeType;
typedef BuildRef<8> BRef;
typedef QuantizedNode<8> QNode;
static const u32 DefaultN = 8;
#endif

template <i32 N>
struct Triangle
{
    u32 geomIDs[N];
    u32 primIDs[N];
    Triangle() {}

    __forceinline static Triangle<N> Fill(PrimRef *refs)
    {
        Triangle<N> tri;
        for (u32 i = 0; i < N; i++)
        {
            PrimRef *ref   = &refs[i];
            tri.geomIDs[i] = ref->geomID;
            tri.primIDs[i] = ref->primID;
        }
        return tri;
    }
};

template <i32 N>
struct TriangleCompressed
{
    u32 primIDs[N];
    TriangleCompressed() {}

    __forceinline static TriangleCompressed<N> Fill(PrimRefCompressed *refs)
    {
        TriangleCompressed<N> tri;
        for (u32 i = 0; i < N; i++)
        {
            PrimRefCompressed *ref = &refs[i];
            tri.primIDs[i]         = ref->primID;
        }
        return tri;
    }
};

template <i32 N>
struct TLASLeaf
{
    BVHNode<N> nodePtr;
    TLASLeaf() {}
    __forceinline static TLASLeaf<N> Fill(BuildRef<N> *ref)
    {
        TLASLeaf<N> result;
        result.nodePtr = ref->nodePtr;
        return result;
    }
};

} // namespace rt
#endif
