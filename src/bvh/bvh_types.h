#ifndef BVH_TYPES_H
#define BVH_TYPES_H
namespace rt
{
static const u32 LANE_WIDTH         = 8;
static const i32 LANE_WIDTHi        = 8;
static const f32 GROW_AMOUNT        = 1.2f;
static const u32 PARALLEL_THRESHOLD = 32 * 1024;

static const Lane8F32 signFlipMask(-0.f, -0.f, -0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
static const u32 LUTAxis[] = {1, 2, 0};

struct BuildSettings
{
    u32 maxLeafSize = 3;
    i32 maxDepth    = 32;
    f32 intCost     = 1.f;
    f32 travCost    = 1.f;
    bool twoLevel   = true;
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
    u64 allocPos;

    Split() {}
    Split(f32 sah, u32 pos, u32 dim, f32 val) : bestSAH(sah), bestPos(pos), bestDim(dim), bestValue(val) {}
};
struct PrimData
{
    union
    {
        struct
        {
            // NOTE: contains the geomID
            Lane4F32 minP;
            // NOTE: contains the primID
            Lane4F32 maxP;
        };
        Lane8F32 m256;
    };
    PrimData(const PrimData &other)
    {
        minP = other.minP;
        maxP = other.maxP;
    }
    PrimData &operator=(const PrimData &other)
    {
        minP = other.minP;
        maxP = other.maxP;
        return *this;
    }

    __forceinline GeometryID GeomID() const
    {
        return GeometryID(minP.u);
    }

    __forceinline u32 PrimID() const
    {
        return maxP.u;
    }

    __forceinline void SetGeomID(GeometryID geomID)
    {
        minP.u = geomID.id;
    }

    __forceinline void SetPrimID(u32 primID)
    {
        maxP.u = primID;
    }

    __forceinline Bounds GetBounds() const
    {
        Bounds out;
        out.minP = minP;
        out.maxP = maxP;
        return out;
    }
};

struct Record
{
    PrimData *data;
    Bounds geomBounds;
    Bounds centBounds;
    struct Range
    {
        u32 start;
        u32 end;
        Range() {}
        Range(u32 start, u32 end) : start(start), end(end) {}
        u32 End() const { return end; }
        u32 Size() const { return end - start; }
    };
    Range range;

    Record() {}
    Record(PrimData *data, const Bounds &gBounds, const Bounds &cBounds, const u32 start, const u32 end)
        : data(data), geomBounds(gBounds), centBounds(cBounds), range(start, end) {}
    u32 Size() const { return range.Size(); }
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
};
struct PrimDataSOA
{
    union
    {
        struct
        {
            f32 *minX;
            f32 *minY;
            f32 *minZ;
            u32 *geomIDs;
            f32 *maxX;
            f32 *maxY;
            f32 *maxZ;
            u32 *primIDs;
        };
        f32 *arr[8];
    };
    __forceinline void Set(const PrimDataSOA &other, const u32 lIndex, const u32 rIndex)
    {
        minX[lIndex]    = other.minX[rIndex];
        minY[lIndex]    = other.minY[rIndex];
        minZ[lIndex]    = other.minZ[rIndex];
        maxX[lIndex]    = other.maxX[rIndex];
        maxY[lIndex]    = other.maxY[rIndex];
        maxZ[lIndex]    = other.maxZ[rIndex];
        geomIDs[lIndex] = other.geomIDs[rIndex];
        primIDs[lIndex] = other.primIDs[rIndex];
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


struct RecordSOASplits
{
    using PrimitiveData = PrimDataSOA;
    // Bounds geomBounds;
    // Bounds centBounds;
    ScalarBounds geomBounds;
    ScalarBounds centBounds;
    ExtRange range;
};

struct Bounds8F32
{
    Lane8F32 minU;
    Lane8F32 minV;
    Lane8F32 minW;

    Lane8F32 maxU;
    Lane8F32 maxV;
    Lane8F32 maxW;

    Bounds8F32() : minU(pos_inf), minV(pos_inf), minW(pos_inf), maxU(neg_inf), maxV(neg_inf), maxW(neg_inf) {}
    Bounds8F32(NegInfTy) : minU(neg_inf), minV(neg_inf), minW(neg_inf), maxU(neg_inf), maxV(neg_inf), maxW(neg_inf) {}

    __forceinline Bounds ToBounds()
    {
        f32 sMinU = ReduceMin(minU);
        f32 sMinV = ReduceMin(minV);
        f32 sMinW = ReduceMin(minW);

        f32 sMaxU = ReduceMax(maxU);
        f32 sMaxV = ReduceMax(maxV);
        f32 sMaxW = ReduceMax(maxW);

        return Bounds(Lane4F32(sMinU, sMinV, sMinW, 0.f), Lane4F32(sMaxU, sMaxV, sMaxW, 0.f));
    }

    __forceinline Bounds ToBoundsNegMin()
    {
        f32 sMinU = ReduceMax(minU);
        f32 sMinV = ReduceMax(minV);
        f32 sMinW = ReduceMax(minW);

        f32 sMaxU = ReduceMax(maxU);
        f32 sMaxV = ReduceMax(maxV);
        f32 sMaxW = ReduceMax(maxW);

        return Bounds(Lane4F32(-sMinU, -sMinV, -sMinW, 0.f), Lane4F32(sMaxU, sMaxV, sMaxW, 0.f));
    }
    __forceinline void Extend(const Bounds8F32 &other)
    {
        minU = Min(minU, other.minU);
        minV = Min(minV, other.minV);
        minW = Min(minW, other.minW);

        maxU = Max(maxU, other.maxU);
        maxV = Max(maxV, other.maxV);
        maxW = Max(maxW, other.maxW);
    }
    __forceinline void Extend(const Lane8F32 &x, const Lane8F32 &y, const Lane8F32 &z)
    {
        minU = Min(minU, x);
        minV = Min(minV, y);
        minW = Min(minW, z);

        maxU = Max(maxU, x);
        maxV = Max(maxV, y);
        maxW = Max(maxW, z);
    }
    __forceinline void ExtendNegativeMin(const Lane8F32 &minX, const Lane8F32 &minY, const Lane8F32 &minZ,
                                         const Lane8F32 &maxX, const Lane8F32 &maxY, const Lane8F32 &maxZ)
    {
        minU = Max(minU, minX);
        minV = Max(minV, minY);
        minW = Max(minW, minZ);

        maxU = Max(maxU, maxX);
        maxV = Max(maxV, maxY);
        maxW = Max(maxW, maxZ);
    }
    __forceinline void ExtendNegativeMin(const Bounds8F32 &other)
    {
        minU = Max(minU, other.minU);
        minV = Max(minV, other.minV);
        minW = Max(minW, other.minW);

        maxU = Max(maxU, other.maxU);
        maxV = Max(maxV, other.maxV);
        maxW = Max(maxW, other.maxW);
    }

    __forceinline void MaskExtendL(const Lane8F32 &mask, const Lane8F32 &u, const Lane8F32 &v, const Lane8F32 &w)
    {
        minU = MaskMin(mask, minU, u);
        minV = MaskMin(mask, minV, v);
        minW = MaskMin(mask, minW, w);

        maxV = MaskMax(mask, maxV, v);
        maxW = MaskMax(mask, maxW, w);
    }

    __forceinline void MaskExtendR(const Lane8F32 &mask, const Lane8F32 &u,
                                   const Lane8F32 &v, const Lane8F32 &w)
    {

#if 1
        minV = MaskMin(mask, minV, v);
        minW = MaskMin(mask, minW, w);

        maxU = MaskMax(mask, maxU, u);
        maxV = MaskMax(mask, maxV, v);
        maxW = MaskMax(mask, maxW, w);
#else
        minV = Select(mask, minV, Min(minV, v));
        minW = Select(mask, minW, Min(minW, w));

        maxU = Select(mask, maxU, Max(maxU, u));
        maxV = Select(mask, maxV, Max(maxV, v));
        maxW = Select(mask, maxW, Max(maxW, w));
#endif
    }

    __forceinline void MaskExtendVW(const Lane8F32 &mask, const Lane8F32 &v, const Lane8F32 &w)
    {
        minV = MaskMin(mask, minV, v);
        maxV = MaskMax(mask, maxV, v);

        minW = MaskMin(mask, minW, w);
        maxW = MaskMax(mask, maxW, w);
    }
};
} // namespace rt
#endif