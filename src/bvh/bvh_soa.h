#ifndef BVH_SOA_H
#define BVH_SOA_H
namespace rt
{
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
};

struct ExtRangeSOA
{
    PrimDataSOA *data;
    u32 start;
    u32 count;
    u32 extEnd;

    __forceinline ExtRangeSOA(PrimDataSOA *data, u32 start, u32 count, u32 extEnd)
        : data(data), start(start), count(count), extEnd(extEnd) {}
    __forceinline u32 End() const { return start + count; }
    __forceinline u32 ExtSize() const { return extEnd - (start + count); }
    __forceinline u32 TotalSize() const { return extEnd - start; }
};

// ways of doing this
// what I have now: data is SOA. find the bin index min and max in each dimension. increment the count of the bin and place the
// face index in the bin. when the count reaches the max, start working on the 8 triangle/plane test. at the end,
// process the remaining triangles and flush the bins.

// 1. test this with AoS? (the 8-wide prim data version).
//       slightly slower. however, it could still be worth using this if partitioning soa is a pain.
// 2. testing triangle vs multiple planes (4 triangles vs 4 planes, 1 triangle vs multiple planes)

// NOTE: stores -minX -minY -minZ _ maxX maxY maxZ, so union is one max inst, intersect is a min

struct Bounds8
{
    Lane8F32 v;

    Bounds8() : v(neg_inf) {}
    Bounds8(EmptyTy) : v(neg_inf) {}
    Bounds8(PosInfTy) : v(pos_inf) {}
    __forceinline Bounds8(Lane8F32 in)
    {
        v = in ^ signFlipMask;
    }

    __forceinline operator const __m256 &() const { return v; }
    __forceinline operator __m256 &() { return v; }

    __forceinline explicit operator Bounds() const
    {
        Bounds out;
        out.minP = FlipSign(Extract4<0>(v));
        out.maxP = Extract4<1>(v);
        return out;
    }

    __forceinline void Extend(const Bounds8 &other)
    {
        v = Max(v, other.v);
    }
    __forceinline void Intersect(const Bounds8 &other)
    {
        v = Min(v, other.v);
    }
};

struct Triangle8
{
    union
    {
        struct
        {
            Lane8F32 v0u;
            Lane8F32 v0v;
            Lane8F32 v0w;

            Lane8F32 v1u;
            Lane8F32 v1v;
            Lane8F32 v1w;

            Lane8F32 v2u;
            Lane8F32 v2v;
            Lane8F32 v2w;
        };
        Lane8F32 v[9];
    };
    Triangle8() {}
    Triangle8(const Triangle8 &other) : v0u(other.v0u), v0v(other.v0v), v0w(other.v0w),
                                        v1u(other.v1u), v1v(other.v1v), v1w(other.v1w),
                                        v2u(other.v2u), v2v(other.v2v), v2w(other.v2w) {}

    __forceinline const Lane8F32 &operator[](i32 i) const
    {
        Assert(i < 9);
        return v[i];
    }
    __forceinline Lane8F32 &operator[](i32 i)
    {
        Assert(i < 9);
        return v[i];
    }

    static Triangle8 Load(TriangleMesh *mesh, const u32 dim, const u32 faceIndices[8])
    {
        u32 faceIndexA = faceIndices[0];
        u32 faceIndexB = faceIndices[1];
        u32 faceIndexC = faceIndices[2];
        u32 faceIndexD = faceIndices[3];

        Lane4F32 v0a = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3]]));
        Lane4F32 v1a = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3 + 1]]));
        Lane4F32 v2a = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3 + 2]]));

        Lane4F32 v0b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3]]));
        Lane4F32 v1b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3 + 1]]));
        Lane4F32 v2b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3 + 2]]));

        Lane4F32 v0c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3]]));
        Lane4F32 v1c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3 + 1]]));
        Lane4F32 v2c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3 + 2]]));

        Lane4F32 v0d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3]]));
        Lane4F32 v1d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3 + 1]]));
        Lane4F32 v2d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3 + 2]]));

        Vec3lf4 p0;
        Vec3lf4 p1;
        Vec3lf4 p2;

        Transpose4x3(v0a, v0b, v0c, v0d, p0.x, p0.y, p0.z);
        Transpose4x3(v1a, v1b, v1c, v1d, p1.x, p1.y, p1.z);
        Transpose4x3(v2a, v2b, v2c, v2d, p2.x, p2.y, p2.z);

        faceIndexA = faceIndices[4];
        faceIndexB = faceIndices[5];
        faceIndexC = faceIndices[6];
        faceIndexD = faceIndices[7];
        v0a        = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3]]));
        v1a        = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3 + 1]]));
        v2a        = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3 + 2]]));

        v0b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3]]));
        v1b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3 + 1]]));
        v2b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3 + 2]]));

        v0c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3]]));
        v1c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3 + 1]]));
        v2c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3 + 2]]));

        v0d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3]]));
        v1d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3 + 1]]));
        v2d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3 + 2]]));

        Vec3lf4 p3;
        Vec3lf4 p4;
        Vec3lf4 p5;

        Transpose4x3(v0a, v0b, v0c, v0d, p3.x, p3.y, p3.z);
        Transpose4x3(v1a, v1b, v1c, v1d, p4.x, p4.y, p4.z);
        Transpose4x3(v2a, v2b, v2c, v2d, p5.x, p5.y, p5.z);

        u32 v = LUTAxis[dim];
        u32 w = LUTAxis[v];

        Triangle8 out;
        out.v0u = Lane8F32(p0[dim], p3[dim]);
        out.v1u = Lane8F32(p1[dim], p4[dim]);
        out.v2u = Lane8F32(p2[dim], p5[dim]);

        out.v0v = Lane8F32(p0[v], p3[v]);
        out.v1v = Lane8F32(p1[v], p4[v]);
        out.v2v = Lane8F32(p2[v], p5[v]);

        out.v0w = Lane8F32(p0[w], p3[w]);
        out.v1w = Lane8F32(p1[w], p4[w]);
        out.v2w = Lane8F32(p2[w], p5[w]);
        return out;
    }
};

// NOTE: l contains bounds to intersect against
template <bool isPartition = false>
__forceinline void ClipTriangle(const TriangleMesh *mesh, const u32 dim, const Triangle8 &tri, const Lane8F32 &splitPos,
                                Bounds8 *l, Bounds8 *r)
{
    threadLocalStatistics->misc += 1;
    static const u32 LUTX[] = {0, 2, 1};
    static const u32 LUTY[] = {1, 0, 2};
    static const u32 LUTZ[] = {2, 1, 0};

    Lane8F32 clipMasks[] = {
        tri.v0u < splitPos,
        tri.v1u < splitPos,
        tri.v2u < splitPos,
    };

    Bounds8F32 left;
    Bounds8F32 right;

    u32 first = 0;
    u32 next  = LUTAxis[first];
    for (u32 edgeIndex = 0; edgeIndex < 3; edgeIndex++)
    {
        const u32 v0IndexStart = 3 * first;
        const u32 v1IndexStart = 3 * next;

        const Lane8F32 &v0u = tri[v0IndexStart];
        const Lane8F32 &v1u = tri[v1IndexStart];

        const Lane8F32 &v0v = tri[v0IndexStart + 1];
        const Lane8F32 &v1v = tri[v1IndexStart + 1];

        const Lane8F32 &v0w = tri[v0IndexStart + 2];
        const Lane8F32 &v1w = tri[v1IndexStart + 2];

        const Lane8F32 &clipMask = clipMasks[first];
        left.MaskExtendL(clipMask, splitPos, v0u, v0v, v0w);
        right.MaskExtendR(clipMask, splitPos, v0u, v0v, v0w);

        const Lane8F32 div = Select(v1u == v0u, Lane8F32(zero), Rcp(v1u - v0u));
        const Lane8F32 t   = (splitPos - v0u) * div;

        const Lane8F32 subV = v1v - v0v;
        const Lane8F32 subW = v1w - v0w;

        const Lane8F32 clippedV = FMA(t, subV, v0v);
        const Lane8F32 clippedW = FMA(t, subW, v0w);

        const Lane8F32 edgeIsClipped = clipMask ^ clipMasks[next];

        left.MaskExtendVW(edgeIsClipped, clippedV, clippedW);
        right.MaskExtendVW(edgeIsClipped, clippedV, clippedW);

        first = next;
        next  = LUTAxis[next];
    }

    const Lane8F32 posInf(pos_inf);

    Lane8F32 lOut[8];
    Lane8F32 rOut[8];

    Lane8F32 *leftMinX = (Lane8F32 *)(&left) + LUTX[dim];
    Lane8F32 *leftMinY = (Lane8F32 *)(&left) + LUTY[dim];
    Lane8F32 *leftMinZ = (Lane8F32 *)(&left) + LUTZ[dim];

    Lane8F32 *leftMaxX = (Lane8F32 *)(&left) + 3 + LUTX[dim];
    Lane8F32 *leftMaxY = (Lane8F32 *)(&left) + 3 + LUTY[dim];
    Lane8F32 *leftMaxZ = (Lane8F32 *)(&left) + 3 + LUTZ[dim];

    Lane8F32 *rightMinX = (Lane8F32 *)(&right) + LUTX[dim];
    Lane8F32 *rightMinY = (Lane8F32 *)(&right) + LUTY[dim];
    Lane8F32 *rightMinZ = (Lane8F32 *)(&right) + LUTZ[dim];

    Lane8F32 *rightMaxX = (Lane8F32 *)(&right) + 3 + LUTX[dim];
    Lane8F32 *rightMaxY = (Lane8F32 *)(&right) + 3 + LUTY[dim];
    Lane8F32 *rightMaxZ = (Lane8F32 *)(&right) + 3 + LUTZ[dim];

    Transpose8x8(*leftMinX, *leftMinY, *leftMinZ, posInf, *leftMaxX, *leftMaxY, *leftMaxZ, posInf,
                 lOut[0], lOut[1], lOut[2], lOut[3], lOut[4], lOut[5], lOut[6], lOut[7]);
    Transpose8x8(*rightMinX, *rightMinY, *rightMinZ, posInf, *rightMaxX, *rightMaxY, *rightMaxZ, posInf,
                 rOut[0], rOut[1], rOut[2], rOut[3], rOut[4], rOut[5], rOut[6], rOut[7]);

    for (u32 i = 0; i < 8; i++)
    {
        Bounds8 lBounds8(lOut[i]);
        Bounds8 rBounds8(rOut[i]);

        if constexpr (!isPartition)
        {
            lBounds8.Intersect(l[i]);
            rBounds8.Intersect(l[i]);
        }

        l[i] = lBounds8;
        r[i] = rBounds8;
    }
}

__forceinline void ClipTrianglePartition(const TriangleMesh *mesh, const u32 dim, const Triangle8 &tri, const Lane8F32 &splitPos,
                                         Bounds8 *l, Bounds8 *r)
{
    ClipTriangle<true>(mesh, dim, tri, splitPos, l, r);
}

__forceinline void ClipTriangleTest(const TriangleMesh *mesh, const u32 faceIndices[8], const u32 dim,
                                    const f32 leftBound, const f32 rightBound, Bounds8F32 &out)
{
    // threadLocalStatistics->misc += 1;

    Assert(leftBound < rightBound);
    Lane8F32 clipLeft(leftBound);
    Lane8F32 clipRight(rightBound);

    // clock_t start  = clock();
    u32 faceIndexA = faceIndices[0];
    u32 faceIndexB = faceIndices[1];
    u32 faceIndexC = faceIndices[2];
    u32 faceIndexD = faceIndices[3];

    Lane4F32 v0a = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3]]));
    Lane4F32 v1a = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3 + 1]]));
    Lane4F32 v2a = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3 + 2]]));

    Lane4F32 v0b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3]]));
    Lane4F32 v1b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3 + 1]]));
    Lane4F32 v2b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3 + 2]]));

    Lane4F32 v0c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3]]));
    Lane4F32 v1c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3 + 1]]));
    Lane4F32 v2c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3 + 2]]));

    Lane4F32 v0d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3]]));
    Lane4F32 v1d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3 + 1]]));
    Lane4F32 v2d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3 + 2]]));

    Vec3lf4 p0;
    Vec3lf4 p1;
    Vec3lf4 p2;

    Transpose4x3(v0a, v0b, v0c, v0d, p0.x, p0.y, p0.z);
    Transpose4x3(v1a, v1b, v1c, v1d, p1.x, p1.y, p1.z);
    Transpose4x3(v2a, v2b, v2c, v2d, p2.x, p2.y, p2.z);

    faceIndexA = faceIndices[4];
    faceIndexB = faceIndices[5];
    faceIndexC = faceIndices[6];
    faceIndexD = faceIndices[7];
    v0a        = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3]]));
    v1a        = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3 + 1]]));
    v2a        = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3 + 2]]));

    v0b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3]]));
    v1b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3 + 1]]));
    v2b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3 + 2]]));

    v0c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3]]));
    v1c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3 + 1]]));
    v2c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3 + 2]]));

    v0d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3]]));
    v1d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3 + 1]]));
    v2d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3 + 2]]));

    Vec3lf4 p3;
    Vec3lf4 p4;
    Vec3lf4 p5;

    Transpose4x3(v0a, v0b, v0c, v0d, p3.x, p3.y, p3.z);
    Transpose4x3(v1a, v1b, v1c, v1d, p4.x, p4.y, p4.z);
    Transpose4x3(v2a, v2b, v2c, v2d, p5.x, p5.y, p5.z);

    u32 v = LUTAxis[dim];
    u32 w = LUTAxis[v];
    Lane8F32 v0u(p0[dim], p3[dim]);
    Lane8F32 v1u(p1[dim], p4[dim]);
    Lane8F32 v2u(p2[dim], p5[dim]);

    Lane8F32 v0v(p0[v], p3[v]);
    Lane8F32 v1v(p1[v], p4[v]);
    Lane8F32 v2v(p2[v], p5[v]);

    Lane8F32 v0w(p0[w], p3[w]);
    Lane8F32 v1w(p1[w], p4[w]);
    Lane8F32 v2w(p2[w], p5[w]);

    // clock_t end = clock();
    // threadLocalStatistics->misc += (u64)(end - start);

    Lane8F32 minU = Min(v0u, Min(v1u, v2u));
    out.minU      = Min(out.minU, Max(minU, clipLeft));

    Lane8F32 maxU = Max(v0u, Max(v1u, v2u));
    out.maxU      = Max(out.maxU, Min(maxU, clipRight));

    Lane8F32 v0uClipLeftMask = v0u >= clipLeft;
    Lane8F32 v1uClipLeftMask = v1u >= clipLeft;
    Lane8F32 v2uClipLeftMask = v2u >= clipLeft;

    Lane8F32 v0uClipRightMask = v0u <= clipRight;
    Lane8F32 v1uClipRightMask = v1u <= clipRight;
    Lane8F32 v2uClipRightMask = v2u <= clipRight;

    out.MaskExtendVW(v0uClipLeftMask & v0uClipRightMask, v0v, v0w);
    out.MaskExtendVW(v1uClipLeftMask & v1uClipRightMask, v1v, v1w);
    out.MaskExtendVW(v2uClipLeftMask & v2uClipRightMask, v2v, v2w);

    // Edge 0: v0 to v1
    Lane8F32 div0    = Select(v1u == v0u, Lane8F32(zero), Rcp(v1u - v0u));
    Lane8F32 t0Left  = (clipLeft - v0u) * div0;
    Lane8F32 t0Right = (clipRight - v0u) * div0;

    Lane8F32 sub0v              = v1v - v0v;
    Lane8F32 sub0w              = v1w - v0w;
    Lane8F32 edge0ClippedVLeft  = FMA(t0Left, sub0v, v0v);
    Lane8F32 edge0ClippedVRight = FMA(t0Right, sub0v, v0v);

    Lane8F32 edge0ClippedWLeft  = FMA(t0Left, sub0w, v0w);
    Lane8F32 edge0ClippedWRight = FMA(t0Right, sub0w, v0w);

    Lane8F32 edgeIsClippedLeftMask0  = v0uClipLeftMask ^ v1uClipLeftMask;
    Lane8F32 edgeIsClippedRightMask0 = v0uClipRightMask ^ v1uClipRightMask;

    out.MaskExtendVW(edgeIsClippedLeftMask0, edge0ClippedVLeft, edge0ClippedWLeft);
    out.MaskExtendVW(edgeIsClippedRightMask0, edge0ClippedVRight, edge0ClippedWRight);

    // Edge 1 : v1 to v2
    Lane8F32 div1    = Select(v2u == v1u, Lane8F32(zero), Rcp(v2u - v1u));
    Lane8F32 t1Left  = (clipLeft - v1u) * div1;
    Lane8F32 t1Right = (clipRight - v1u) * div1;

    Lane8F32 sub1v              = v2v - v1v;
    Lane8F32 sub1w              = v2w - v1w;
    Lane8F32 edge1ClippedVLeft  = FMA(t1Left, sub1v, v1v);
    Lane8F32 edge1ClippedVRight = FMA(t1Right, sub1v, v1v);

    Lane8F32 edge1ClippedWLeft  = FMA(t1Left, sub1w, v1w);
    Lane8F32 edge1ClippedWRight = FMA(t1Right, sub1w, v1w);

    Lane8F32 edgeIsClippedLeftMask1  = v2uClipLeftMask ^ v1uClipLeftMask;
    Lane8F32 edgeIsClippedRightMask1 = v2uClipRightMask ^ v1uClipRightMask;

    out.MaskExtendVW(edgeIsClippedLeftMask1, edge1ClippedVLeft, edge1ClippedWLeft);
    out.MaskExtendVW(edgeIsClippedRightMask1, edge1ClippedVRight, edge1ClippedWRight);

    // Edge 2 : v2 to v0
    Lane8F32 div2    = Select(v0u == v2u, Lane8F32(zero), Rcp(v0u - v2u));
    Lane8F32 t2Left  = (clipLeft - v2u) * div2;
    Lane8F32 t2Right = (clipRight - v2u) * div2;

    Lane8F32 sub2v              = v0v - v2v;
    Lane8F32 sub2w              = v0w - v2w;
    Lane8F32 edge2ClippedVLeft  = FMA(t2Left, sub2v, v2v);
    Lane8F32 edge2ClippedVRight = FMA(t2Right, sub2v, v2v);

    Lane8F32 edge2ClippedWLeft  = FMA(t2Left, sub2w, v2w);
    Lane8F32 edge2ClippedWRight = FMA(t2Right, sub2w, v2w);

    Lane8F32 edgeIsClippedLeftMask2  = v0uClipLeftMask ^ v2uClipLeftMask;
    Lane8F32 edgeIsClippedRightMask2 = v0uClipRightMask ^ v2uClipRightMask;

    out.MaskExtendVW(edgeIsClippedLeftMask2, edge2ClippedVLeft, edge2ClippedWLeft);
    out.MaskExtendVW(edgeIsClippedRightMask2, edge2ClippedVRight, edge2ClippedWRight);
}

template <i32 numBins = 16>
struct HeuristicSOASplitBinning
{
    Lane8F32 baseX;
    Lane8F32 baseY;
    Lane8F32 baseZ;

    Lane8F32 scaleX;
    Lane8F32 scaleY;
    Lane8F32 scaleZ;

    Lane8F32 base;
    Lane8F32 scale;

    static const u32 LANE_WIDTH = 8;

    // temp storage
    u32 faceIndices[3][numBins][LANE_WIDTH];
    u32 binCounts[3][numBins];

    // result data
    Lane4U32 entryCounts[numBins];
    Lane4U32 exitCounts[numBins];

    Bounds finalBounds[3][numBins];

    // used in soa 8 triangle 1 plane binning
    f32 splitPositions[3][numBins + 1];
    Bounds8F32 bins[3][numBins];

    // used in soa 8 triangle multi plane diff binning
    Bounds8 bins8[3][numBins];
    Lane8F32 invScaleX;
    Lane8F32 invScaleY;
    Lane8F32 invScaleZ;

    HeuristicSOASplitBinning(Bounds &bounds)
    {
        const Lane4F32 eps = 1e-34f;

        Lane8F32 minP(bounds.minP);
        baseX = Shuffle<0>(minP);
        baseY = Shuffle<1>(minP);
        baseZ = Shuffle<2>(minP);

        base = Lane8F32(bounds.minP, bounds.minP);

        const Lane4F32 diag = Max(bounds.maxP - bounds.minP, eps);

        Lane4F32 scale4 = Select(diag > eps, Lane4F32((f32)numBins) / diag, Lane4F32(0.f));

        Lane8F32 scale8(scale4);
        scaleX = Shuffle<0>(scale8);
        scaleY = Shuffle<1>(scale8);
        scaleZ = Shuffle<2>(scale8);

        scale = Lane8F32(scale4, scale4);

        // test
        Lane4F32 invScale4 = Select(scale4 == 0.f, 0.f, 1.f / scale4);
        Lane8F32 invScale8(invScale4);
        invScaleX = Shuffle<0>(invScale8);
        invScaleY = Shuffle<1>(invScale8);
        invScaleZ = Shuffle<2>(invScale8);

        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 i = 0; i < numBins; i++)
            {
                binCounts[dim][i]      = 0;
                splitPositions[dim][i] = bounds.minP[dim] + (i * diag[dim] / (f32)numBins);
            }
            splitPositions[dim][numBins] = bounds.maxP[dim];
        }
    }

    void BinDiffTest(TriangleMesh *mesh, PrimDataSOA *soa, u32 start, u32 count)
    {
        Lane8U32 z = Lane8U32(0);
        Lane8U32 e = Lane8U32(numBins - 1);

        u32 faceIDPrev[8];

        u32 indexMinPrev[8];
        u32 indexMaxPrev[8];

        u32 binIndexStart[3][numBins][8];

        Lane8F32 baseArr[]     = {baseX, baseY, baseZ};
        Lane8F32 scaleArr[]    = {scaleX, scaleY, scaleZ};
        Lane8F32 invScaleArr[] = {invScaleX, invScaleY, invScaleZ};

        u32 i            = start;
        u32 alignedCount = count - count % LANE_WIDTH;
        f32 totalTime    = 0.f;
        for (; i < start + count; i += LANE_WIDTH)
        {
            u32 num          = Min(LANE_WIDTH, start + count - i);
            Lane8U32 faceIds = Lane8U32::LoadU(soa->primIDs + i);

            Lane8F32 prevMin[] = {
                Lane8F32::LoadU(soa->minX + i),
                Lane8F32::LoadU(soa->minY + i),
                Lane8F32::LoadU(soa->minZ + i),
            };

            Lane8F32 prevMax[] = {
                Lane8F32::LoadU(soa->maxX + i),
                Lane8F32::LoadU(soa->maxY + i),
                Lane8F32::LoadU(soa->maxZ + i),
            };

            for (u32 dim = 0; dim < 3; dim++)
            {
                Lane8F32 baseD     = baseArr[dim];
                Lane8F32 scaleD    = scaleArr[dim];
                Lane8F32 invScaleD = invScaleArr[dim];

                Lane8U32 binIndexMin = Clamp(z, e, Flooru((prevMin[dim] - baseD) * scaleD));
                Lane8U32 binIndexMax = Clamp(z, e, Flooru((prevMax[dim] - baseD) * scaleD));

                Lane8U32::Store(indexMinPrev, binIndexMin);
                Lane8U32::Store(indexMaxPrev, binIndexMax);
                Lane8U32::Store(faceIDPrev, faceIds);

                for (u32 prevIndex = 0; prevIndex < num; prevIndex++)
                {
                    u32 faceID   = faceIDPrev[prevIndex];
                    u32 indexMin = indexMinPrev[prevIndex];
                    u32 indexMax = indexMaxPrev[prevIndex];

                    entryCounts[indexMin][dim] += 1;
                    exitCounts[indexMax][dim] += 1;

                    Assert(indexMax >= indexMin);
                    u32 diff = indexMax - indexMin;

                    // Fast path, no splitting
                    if (diff == 0)
                    {
                        // PerformanceCounter counter = OS_StartCounter();
                        Lane8F32 primBounds(prevMin[0][prevIndex], prevMin[1][prevIndex], prevMin[2][prevIndex], pos_inf,
                                            prevMax[0][prevIndex], prevMax[1][prevIndex], prevMax[2][prevIndex], pos_inf);
                        bins8[dim][indexMin].Extend(primBounds);
                        continue;
                    }

                    faceIndices[dim][diff][binCounts[dim][diff]]   = faceID;
                    binIndexStart[dim][diff][binCounts[dim][diff]] = indexMin;
                    binCounts[dim][diff]++;
                    if (binCounts[dim][diff] == LANE_WIDTH)
                    {
                        Triangle8 tri     = Triangle8::Load(mesh, dim, faceIndices[dim][diff]);
                        Lane8U32 startBin = Lane8U32::Load(binIndexStart[dim][diff]);

                        Bounds8 bounds[2][LANE_WIDTH];
                        for (u32 boundIndex = 0; boundIndex < LANE_WIDTH; boundIndex++)
                        {
                            bounds[0][boundIndex] = Bounds8(pos_inf);
                        }
                        u32 binIndices[LANE_WIDTH];

                        u32 current = 0;
                        for (u32 d = 0; d < diff; d++)
                        {
                            Lane8U32::Store(binIndices, startBin);
                            startBin += 1u;
                            Lane8F32 splitPos = Lane8F32(startBin) * invScaleD + baseD;
                            ClipTriangle(mesh, dim, tri, splitPos, bounds[current], bounds[!current]);

                            for (u32 b = 0; b < LANE_WIDTH; b++)
                            {
                                bins8[dim][binIndices[b]].Extend(bounds[current][b]);
                            }
                            current = !current;
                        }
                        binCounts[dim][diff] = 0;
                    }
                }
            }
        }

        // Empty the bins
        Lane8F32 posInf(pos_inf);
        Lane8F32 negInf(neg_inf);
        for (u32 dim = 0; dim < 3; dim++)
        {
            Lane8F32 invScaleD = invScaleArr[dim];
            Lane8F32 baseD     = baseArr[dim];
            for (u32 diff = 1; diff < numBins; diff++)
            {
                u32 remainingCount = binCounts[dim][diff];

                Triangle8 tri     = Triangle8::Load(mesh, dim, faceIndices[dim][diff]);
                Lane8U32 startBin = Lane8U32::Load(binIndexStart[dim][diff]);

                Bounds8 bounds[2][8];
                u32 binIndices[8];

                u32 current = 0;
                for (u32 d = 0; d < diff; d++)
                {
                    Lane8U32::Store(binIndices, startBin);
                    startBin += 1u;
                    Lane8F32 splitPos = Lane8F32(startBin) * invScaleD + baseD;
                    ClipTriangle(mesh, dim, tri, splitPos, bounds[current], bounds[!current]);

                    for (u32 b = 0; b < remainingCount; b++)
                    {
                        bins8[dim][binIndices[b]].Extend(bounds[current][b]);
                    }
                    current = !current;
                }
            }
        }

        for (u32 dim = 0; dim < 3; dim++)
        {
            for (i = 0; i < numBins; i++)
            {
                finalBounds[dim][i] = Bounds(bins8[dim][i]);
            }
        }
    }

    void BinTest(TriangleMesh *mesh, PrimRef *refs, u32 start, u32 count)
    {
        Lane8U32 z = Lane8U32(0);
        Lane8U32 e = Lane8U32(numBins - 1);

        u32 binIndices[8];

        for (u32 i = start; i < start + count; i++)
        {
            PrimRef *ref      = &refs[i];
            Lane8U32 binIndex = Clamp(z, e, Flooru((ref->m256 - base) * scale));

            Lane8U32::Store(binIndices, binIndex);

            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 indexMin = binIndices[dim];
                u32 indexMax = binIndices[4 + dim];

                entryCounts[indexMin][dim] += 1;
                exitCounts[indexMax][dim] += 1;

                u32 faceID = ref->primID;

                for (u32 index = indexMin; index <= indexMax; index++)
                {
                    faceIndices[dim][index][binCounts[dim][index]++] = faceID;
                    if (binCounts[dim][index] == LANE_WIDTH)
                    {
                        Bounds8F32 out;
                        ClipTriangleTest(mesh, faceIndices[dim][index], dim, splitPositions[dim][index], splitPositions[dim][index + 1], out);
                        binCounts[dim][index] = 0;
                        bins[dim][index].Extend(out);
                    }
                }
            }
        }

        // Empty the bins
        Lane8F32 posInf(pos_inf);
        Lane8F32 negInf(neg_inf);
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 binIndex = 0; binIndex < numBins; binIndex++)
            {
                u32 remainingCount = binCounts[dim][binIndex];
                u32 bitMask        = (1 << remainingCount) - 1;
                Lane8F32 mask      = Lane8F32::Mask(bitMask);
                Bounds8F32 out;
                ClipTriangleTest(mesh, faceIndices[dim][binIndex], dim,
                                 splitPositions[dim][binIndex], splitPositions[dim][binIndex + 1], out);

                out.minU = Select(mask, out.minU, posInf);
                out.minV = Select(mask, out.minV, posInf);
                out.minW = Select(mask, out.minW, posInf);
                out.maxU = Select(mask, out.maxU, negInf);
                out.maxV = Select(mask, out.maxV, negInf);
                out.maxW = Select(mask, out.maxW, negInf);

                bins[dim][binIndex].Extend(out);
            }
        }

        for (u32 i = 0; i < numBins; i++)
        {
            Bounds8F32 &bX = bins[0][i];
            f32 bXMinX     = ReduceMin(bX.minU);
            f32 bXMinY     = ReduceMin(bX.minV);
            f32 bXMinZ     = ReduceMin(bX.minW);

            f32 bXMaxX = ReduceMax(bX.maxU);
            f32 bXMaxY = ReduceMax(bX.maxV);
            f32 bXMaxZ = ReduceMax(bX.maxW);

            Lane4F32 xMinP(bXMinX, bXMinY, bXMinZ, 0.f);
            Lane4F32 xMaxP(bXMaxX, bXMaxY, bXMaxZ, 0.f);

            finalBounds[0][i] = Bounds(xMinP, xMaxP);

            Bounds8F32 &bY = bins[1][i];
            f32 bYMinX     = ReduceMin(bY.minW);
            f32 bYMinY     = ReduceMin(bY.minU);
            f32 bYMinZ     = ReduceMin(bY.minV);

            f32 bYMaxX = ReduceMax(bY.maxW);
            f32 bYMaxY = ReduceMax(bY.maxU);
            f32 bYMaxZ = ReduceMax(bY.maxV);

            Lane4F32 yMinP(bYMinX, bYMinY, bYMinZ, 0.f);
            Lane4F32 yMaxP(bYMaxX, bYMaxY, bYMaxZ, 0.f);

            finalBounds[1][i] = Bounds(yMinP, yMaxP);

            Bounds8F32 &bZ = bins[2][i];
            f32 bZMinX     = ReduceMin(bZ.minV);
            f32 bZMinY     = ReduceMin(bZ.minW);
            f32 bZMinZ     = ReduceMin(bZ.minU);

            f32 bZMaxX = ReduceMax(bZ.maxV);
            f32 bZMaxY = ReduceMax(bZ.maxW);
            f32 bZMaxZ = ReduceMax(bZ.maxU);

            Lane4F32 zMinP(bZMinX, bZMinY, bZMinZ, 0.f);
            Lane4F32 zMaxP(bZMaxX, bZMaxY, bZMaxZ, 0.f);

            finalBounds[2][i] = Bounds(zMinP, zMaxP);
        }
    }

    void Split(Arena *arena, TriangleMesh *mesh, ExtRangeSOA range, Split split)
    {
        // partitioning
        u32 dim                = split.bestDim;
        f32 *minStream         = range.data->arr[dim];
        f32 *maxStream         = range.data->arr[dim + 4];
        u32 alignedCount       = range.count - range.count % LANE_WIDTH;
        Lane8F32 invScaleArr[] = {invScaleX, invScaleY, invScaleZ};
        Lane8F32 invScaleD     = invScaleArr[dim];
        Lane8F32 baseArr[]     = {baseX, baseY, baseZ};
        Lane8F32 baseD         = baseArr[dim];

        u32 count = 0;

        // u32 faceIDQueue[16];
        u32 refIDQueue[16];
        // mask compress
        Bounds8 bounds[2][8];

        u32 splitCount      = 0;
        u32 totalSplitCount = 0;
        u32 splitBegin      = range.End();
        u32 splitMax        = range.ExtSize();
        PrimDataSOA out;
        out.minX    = range.data->minX + splitBegin;
        out.minY    = range.data->minY + splitBegin;
        out.minZ    = range.data->minZ + splitBegin;
        out.geomIDs = range.data->geomIDs + splitBegin;
        out.maxX    = range.data->maxX + splitBegin;
        out.maxY    = range.data->maxY + splitBegin;
        out.maxZ    = range.data->maxZ + splitBegin;
        out.primIDs = range.data->primIDs + splitBegin;
        f32 test    = 0.f;
        for (u32 i = range.start; i < range.start + alignedCount; i += LANE_WIDTH)
        {
            Lane8F32 min = Lane8F32::LoadU(minStream + i);
            Lane8F32 max = Lane8F32::LoadU(maxStream + i);
            // Lane8U32 faceIDs   = Lane8U32::LoadU(range.data->primIDs + i);
            Lane8U32 refIDs    = Lane8U32::Step(i);
            Lane8F32 splitMask = (min <= split.bestValue & max > split.bestValue);
            u32 mask           = Movemask(splitMask);
            Lane8U32::StoreU(refIDQueue + count, MaskCompress(mask, refIDs));
            count += PopCount(mask);

            if (count >= LANE_WIDTH)
            {
                count -= LANE_WIDTH;
                u32 faceIDQueue[8];
                Lane8U32::Store((int *)faceIDQueue, _mm256_i32gather_epi32((int *)range.data->primIDs,
                                                                           Lane8U32::LoadU(refIDQueue + count), 4));
                // for (u32 refIDIndex = 0; refIDIndex < 8; refIDIndex++)
                // {
                //     faceIDQueue[refIDIndex] = range.data->primIDs[refIDQueue[refIDIndex + count]];
                // }
                splitCount += LANE_WIDTH;
                totalSplitCount += LANE_WIDTH;
                if (splitCount > splitMax)
                {
                    const u32 streamSize = u32(sizeof(u32) * range.count * GROW_AMOUNT);
                    u8 *alloc            = PushArray(arena, u8, u32(streamSize * LANE_WIDTH));
                    splitCount           = 8;
                    out.minX             = (f32 *)(alloc + streamSize * 0);
                    out.minY             = (f32 *)(alloc + streamSize * 1);
                    out.minZ             = (f32 *)(alloc + streamSize * 2);
                    out.geomIDs          = (u32 *)(alloc + streamSize * 3);
                    out.maxX             = (f32 *)(alloc + streamSize * 4);
                    out.maxY             = (f32 *)(alloc + streamSize * 5);
                    out.maxZ             = (f32 *)(alloc + streamSize * 6);
                    out.primIDs          = (u32 *)(alloc + streamSize * 7);
                }
                for (u32 queueIndex = 0; queueIndex < 8; queueIndex++)
                {
                    const u32 refID = refIDQueue[count + queueIndex];
                    // const u32 refID = faceIDQueue[count + queueIndex];

                    range.data->minX[refID] = -bounds[0][queueIndex].v[0];
                    range.data->minY[refID] = -bounds[0][queueIndex].v[1];
                    range.data->minZ[refID] = -bounds[0][queueIndex].v[2];
                    range.data->maxX[refID] = bounds[0][queueIndex].v[4];
                    range.data->maxY[refID] = bounds[0][queueIndex].v[5];
                    range.data->maxZ[refID] = bounds[0][queueIndex].v[6];

                    out.minX[splitCount - 8 + queueIndex] = -bounds[1][queueIndex].v[0];
                    out.minY[splitCount - 8 + queueIndex] = -bounds[1][queueIndex].v[1];
                    out.minZ[splitCount - 8 + queueIndex] = -bounds[1][queueIndex].v[2];
                    out.maxX[splitCount - 8 + queueIndex] = bounds[1][queueIndex].v[4];
                    out.maxY[splitCount - 8 + queueIndex] = bounds[1][queueIndex].v[5];
                    out.maxZ[splitCount - 8 + queueIndex] = bounds[1][queueIndex].v[6];
                }

                Triangle8 tri = Triangle8::Load(mesh, dim, faceIDQueue);
                ClipTrianglePartition(mesh, dim, tri, split.bestValue, bounds[0], bounds[1]);
            }
            // PerformanceCounter counter = OS_StartCounter();
            // Lane8U32::StoreU(faceIDQueue + count, MaskCompress(mask, faceIDs));
            // test += OS_GetMilliseconds(counter);
        }
        // printf("test: %fms\n", test);
        // for (; i < range.End(); i++)
        // {
        //     f32 min        = minStream[i];
        //     f32 max        = maxStream[i];
        //     u32 faceID     = range.data.primIDs[i];
        //     faceIDs[count] = faceID;
        //     count += (min <= split.bestPos && max > split.bestPos);
        // }
    }

    void Bin(TriangleMesh *mesh, PrimDataSOA *soa, u32 start, u32 count)
    {
        u32 faceIDPrev[8];

        u32 indexMinPrev[8];
        u32 indexMaxPrev[8];

        Lane8U32 z = Lane8U32(0);
        Lane8U32 e = Lane8U32(numBins - 1);

        u32 i = start;

        Lane8F32 baseArr[]  = {baseX, baseY, baseZ};
        Lane8F32 scaleArr[] = {scaleX, scaleY, scaleZ};

        u32 alignedCount = count - count % 8;
        u32 end          = start + count;
        for (; i < start + alignedCount; i += 8)
        {
            Lane8U32 faceIds = Lane8U32::LoadU(soa->primIDs + i);

            Lane8F32 prevMin[] = {
                Lane8F32::LoadU(soa->minX + i),
                Lane8F32::LoadU(soa->minY + i),
                Lane8F32::LoadU(soa->minZ + i),
            };

            Lane8F32 prevMax[] = {
                Lane8F32::LoadU(soa->maxX + i),
                Lane8F32::LoadU(soa->maxY + i),
                Lane8F32::LoadU(soa->maxZ + i),
            };

            for (u32 dim = 0; dim < 3; dim++)
            {
                Lane8F32 baseD  = baseArr[dim];
                Lane8F32 scaleD = scaleArr[dim];

                Lane8U32 binIndexMin = Clamp(z, e, Flooru((prevMin[dim] - baseD) * scaleD));
                Lane8U32 binIndexMax = Clamp(z, e, Flooru((prevMax[dim] - baseD) * scaleD));

                Lane8U32::Store(indexMinPrev, binIndexMin);
                Lane8U32::Store(indexMaxPrev, binIndexMax);
                Lane8U32::Store(faceIDPrev, faceIds);

                for (u32 prevIndex = 0; prevIndex < 8; prevIndex++)
                {
                    u32 faceID   = faceIDPrev[prevIndex];
                    u32 indexMin = indexMinPrev[prevIndex];
                    u32 indexMax = indexMaxPrev[prevIndex];

                    entryCounts[indexMin][dim] += 1;
                    exitCounts[indexMax][dim] += 1;

                    // if (indexMin == indexMax)
                    // {
                    //     bins[dim][indexMin].minU = Min(bins[dim][indexMin],
                    //
                    //     continue;
                    // }

                    for (u32 index = indexMin; index <= indexMax; index++)
                    {
                        faceIndices[dim][index][binCounts[dim][index]++] = faceID;
                        if (binCounts[dim][index] == LANE_WIDTH)
                        {
                            Bounds8F32 out;
                            ClipTriangleTest(mesh, faceIndices[dim][index], dim,
                                             splitPositions[dim][index], splitPositions[dim][index + 1], out);
                            binCounts[dim][index] = 0;
                            bins[dim][index].Extend(out);
                        }
                    }
                }
            }
        }
        f32 baseSArr[] = {
            baseX[0],
            baseY[0],
            baseZ[0],
        };
        f32 scaleSArr[] = {
            scaleX[0],
            scaleY[0],
            scaleZ[0],
        };

        // Add the remaining triangles
        for (; i < end; i++)
        {
            f32 prevMin[] = {
                soa->minX[i],
                soa->minY[i],
                soa->minZ[i],
            };

            f32 prevMax[] = {
                soa->maxX[i],
                soa->maxY[i],
                soa->maxZ[i],
            };

            u32 faceID = soa->primIDs[i];

            for (u32 dim = 0; dim < 3; dim++)
            {
                f32 baseS    = baseSArr[dim];
                f32 scaleS   = scaleSArr[dim];
                u32 indexMin = Clamp(0u, numBins - 1u, (u32)Floor((prevMin[dim] - baseS) * scaleS));
                u32 indexMax = Clamp(0u, numBins - 1u, (u32)Floor((prevMax[dim] - baseS) * scaleS));
                entryCounts[indexMin][dim] += 1;
                exitCounts[indexMax][dim] += 1;

                for (u32 index = indexMin; index <= indexMax; index++)
                {
                    faceIndices[dim][index][binCounts[dim][index]++] = faceID;
                    if (binCounts[dim][index] == LANE_WIDTH)
                    {
                        Bounds8F32 out;
                        ClipTriangleTest(mesh, faceIndices[dim][index], dim,
                                         splitPositions[dim][index], splitPositions[dim][index + 1], out);
                        binCounts[dim][index] = 0;
                        bins[dim][index].Extend(out);
                    }
                }
            }
        }

        // Empty the bins
        Lane8F32 posInf(pos_inf);
        Lane8F32 negInf(neg_inf);
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 binIndex = 0; binIndex < numBins; binIndex++)
            {
                u32 remainingCount = binCounts[dim][binIndex];
                u32 bitMask        = (1 << remainingCount) - 1;
                Lane8F32 mask      = Lane8F32::Mask(bitMask);
                Bounds8F32 out;
                ClipTriangleTest(mesh, faceIndices[dim][binIndex], dim,
                                 splitPositions[dim][binIndex], splitPositions[dim][binIndex + 1], out);

                out.minU = Select(mask, out.minU, posInf);
                out.minV = Select(mask, out.minV, posInf);
                out.minW = Select(mask, out.minW, posInf);
                out.maxU = Select(mask, out.maxU, negInf);
                out.maxV = Select(mask, out.maxV, negInf);
                out.maxW = Select(mask, out.maxW, negInf);

                bins[dim][binIndex].Extend(out);
            }
        }

        for (i = 0; i < numBins; i++)
        {
            Bounds8F32 &bX = bins[0][i];
            f32 bXMinX     = ReduceMin(bX.minU);
            f32 bXMinY     = ReduceMin(bX.minV);
            f32 bXMinZ     = ReduceMin(bX.minW);

            f32 bXMaxX = ReduceMax(bX.maxU);
            f32 bXMaxY = ReduceMax(bX.maxV);
            f32 bXMaxZ = ReduceMax(bX.maxW);

            Lane4F32 xMinP(bXMinX, bXMinY, bXMinZ, 0.f);
            Lane4F32 xMaxP(bXMaxX, bXMaxY, bXMaxZ, 0.f);

            finalBounds[0][i] = Bounds(xMinP, xMaxP);

            Bounds8F32 &bY = bins[1][i];
            f32 bYMinX     = ReduceMin(bY.minW);
            f32 bYMinY     = ReduceMin(bY.minU);
            f32 bYMinZ     = ReduceMin(bY.minV);

            f32 bYMaxX = ReduceMax(bY.maxW);
            f32 bYMaxY = ReduceMax(bY.maxU);
            f32 bYMaxZ = ReduceMax(bY.maxV);

            Lane4F32 yMinP(bYMinX, bYMinY, bYMinZ, 0.f);
            Lane4F32 yMaxP(bYMaxX, bYMaxY, bYMaxZ, 0.f);

            finalBounds[1][i] = Bounds(yMinP, yMaxP);

            Bounds8F32 &bZ = bins[2][i];
            f32 bZMinX     = ReduceMin(bZ.minV);
            f32 bZMinY     = ReduceMin(bZ.minW);
            f32 bZMinZ     = ReduceMin(bZ.minU);

            f32 bZMaxX = ReduceMax(bZ.maxV);
            f32 bZMaxY = ReduceMax(bZ.maxW);
            f32 bZMaxZ = ReduceMax(bZ.maxU);

            Lane4F32 zMinP(bZMinX, bZMinY, bZMinZ, 0.f);
            Lane4F32 zMaxP(bZMaxX, bZMaxY, bZMaxZ, 0.f);

            finalBounds[2][i] = Bounds(zMinP, zMaxP);
        }
    }
}; // namespace rt

// NOTE: this is embree's implementation of split binning for SBVH
template <i32 numBins = 16>
struct TestSplitBinningBase
{
    Lane4F32 base;
    Lane4F32 scale;
    Lane4F32 invScale;

    Lane4U32 numEnd[numBins];
    Lane4U32 numBegin[numBins];
    Bounds bins[3][numBins];

    TestSplitBinningBase(Bounds &bounds)
    {
        const Lane4F32 eps = 1e-34f;

        base = bounds.minP;

        const Lane4F32 diag = Max(bounds.maxP - bounds.minP, eps);
        scale               = Select(diag > eps, Lane4F32((f32)numBins) / diag, Lane4F32(0.f));
        invScale            = (diag / (f32)numBins);

        for (u32 i = 0; i < numBins; i++)
        {
            numEnd[i]   = 0;
            numBegin[i] = 0;
        }
    }
    __forceinline void Add(const u32 dim, const u32 beginID, const u32 endID, const u32 binID, const Bounds &b, const u32 n = 1)
    {
        numEnd[endID][dim] += n;
        numBegin[beginID][dim] += n;
        bins[dim][binID].Extend(b);
    }
    __forceinline Lane4U32 GetBin(const Lane4F32 &l)
    {
        return Clamp(Lane4U32(0), Lane4U32(numBins - 1), Flooru((l - base) * scale));
    }
    __forceinline f32 GetPos(u32 bin, u32 dim)
    {
        return FMA((f32)bin, invScale[dim], base[dim]);
    }
    __forceinline void Bin(TriangleMesh *mesh, const PrimData *source, u32 start, u32 count)
    {
        for (u32 i = start; i < start + count; i++)
        {
            const PrimData &prim = source[i];

            // if (unlikely(splits <= 1))
            // {
            //     const vint4 bin = mapping.bin(center(prim.bounds()));
            //     for (size_t dim = 0; dim < 3; dim++)
            //     {
            //         assert(bin[dim] >= (int)0 && bin[dim] < (int)BINS);
            //         add(dim, bin[dim], bin[dim], bin[dim], prim.bounds());
            //     }
            // }
            const Lane4U32 bin0 = GetBin(prim.minP);
            const Lane4U32 bin1 = GetBin(prim.maxP);

            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 bin;
                u32 l = bin0[dim];
                u32 r = bin1[dim];

                // same bin optimization
                if (l == r)
                {
                    Add(dim, l, l, l, prim.GetBounds());
                    continue;
                }
                u32 bin_start = bin0[dim];
                u32 bin_end   = bin1[dim];
                Bounds rest   = prim.GetBounds();

                /* assure that split position always overlaps the primitive bounds */
                while (bin_start < bin_end && GetPos(bin_start + 1, dim) <= rest.minP[dim]) bin_start++;
                while (bin_start < bin_end && GetPos(bin_end, dim) >= rest.maxP[dim]) bin_end--;

                for (bin = bin_start; bin < bin_end; bin++)
                {
                    const float pos = GetPos(bin + 1, dim);
                    Bounds left, right;
                    ClipTriangleSimple(mesh, rest, prim.PrimID(), dim, pos, left, right);

                    if (left.Empty()) l++;
                    bins[dim][bin].Extend(left);
                    rest = right;
                }
                if (rest.Empty()) r--;
                Add(dim, l, r, bin, rest);
            }
        }
    }
    __forceinline void Split(TriangleMesh *mesh, PrimData *prims, ExtRange range, Split split)
    {
        u32 dim                         = split.bestDim;
        const size_t max_ext_range_size = range.ExtSize();
        const size_t ext_range_start    = range.End();

        /* atomic counter for number of primref splits */
        // std::atomic<size_t> ext_elements;
        // ext_elements.store(0);

        const float fpos = (split.bestPos * invScale[dim]) + base[dim];

        u32 ID = 0;

        for (size_t i = range.start; i < range.End(); i++)
        {
            // const unsigned int splits = prims[i].GeomID();

            const u32 bin0 = (u32)Floor((prims[i].minP[dim] - base[dim]) * scale[dim]);
            const u32 bin1 = (u32)Floor((prims[i].maxP[dim] - base[dim]) * scale[dim]);
            if (bin0 < split.bestPos && bin1 >= split.bestPos)
            {
                Bounds left, right;
                Bounds primBounds;
                primBounds.minP = prims[i].minP;
                primBounds.maxP = prims[i].maxP;
                ClipTriangleSimple(mesh, primBounds, prims[i].PrimID(), dim, fpos, left, right);

                // no empty splits
                // if (left.Empty() || right.Empty()) continue;

                /* break if the number of subdivided elements are greater than the maximum allowed size */
                if (ID++ >= max_ext_range_size) break;

                /* only write within the correct bounds */
                prims[i].minP                    = left.minP;
                prims[i].maxP                    = left.maxP;
                prims[ext_range_start + ID].minP = right.minP;
                prims[ext_range_start + ID].maxP = right.maxP;
            }
        }

        // const size_t numExtElements = min(max_ext_range_size, ext_elements.load());
        // set._end += numExtElements;
    }
};

template <i32 numBins = 16>
Split SpatialSplitBest(const Bounds bounds[3][numBins],
                       const Lane4U32 *entryCounts,
                       const Lane4U32 *exitCounts,
                       const u32 blockShift = 0)
{
    Bounds boundsDimX;
    Bounds boundsDimY;
    Bounds boundsDimZ;

    Lane4U32 count = 0;
    Lane4U32 lCounts[numBins];
    Lane4F32 area[numBins] = {};

    const u32 blockAdd = (1 << blockShift) - 1;

    for (u32 i = 0; i < numBins - 1; i++)
    {
        count += entryCounts[i];
        lCounts[i] = count;

        boundsDimX.Extend(bounds[0][i]);
        boundsDimY.Extend(bounds[1][i]);
        boundsDimZ.Extend(bounds[2][i]);

        Lane4F32 minX, minY, minZ;
        Lane4F32 maxX, maxY, maxZ;
        Transpose3x3(boundsDimX.minP, boundsDimY.minP, boundsDimZ.minP, minX, minY, minZ);
        Transpose3x3(boundsDimX.maxP, boundsDimY.maxP, boundsDimZ.maxP, maxX, maxY, maxZ);

        Lane4F32 extentX = maxX - minX;
        Lane4F32 extentY = maxY - minY;
        Lane4F32 extentZ = maxZ - minZ;

        area[i] = FMA(extentX, extentY + extentZ, extentY * extentZ);
    }

    count = 0;

    boundsDimX        = Bounds();
    boundsDimY        = Bounds();
    boundsDimZ        = Bounds();
    Lane4F32 lBestSAH = pos_inf;
    Lane4U32 lBestPos = 0;
    for (u32 i = numBins - 1; i >= 1; i--)
    {
        count += exitCounts[i];

        boundsDimX.Extend(bounds[0][i]);
        boundsDimY.Extend(bounds[1][i]);
        boundsDimZ.Extend(bounds[2][i]);

        Lane4F32 minX, minY, minZ;
        Lane4F32 maxX, maxY, maxZ;
        Transpose3x3(boundsDimX.minP, boundsDimY.minP, boundsDimZ.minP, minX, minY, minZ);
        Transpose3x3(boundsDimX.maxP, boundsDimY.maxP, boundsDimZ.maxP, maxX, maxY, maxZ);

        Lane4F32 extentX = maxX - minX;
        Lane4F32 extentY = maxY - minY;
        Lane4F32 extentZ = maxZ - minZ;

        const Lane4F32 rArea = FMA(extentX, extentY + extentZ, extentY * extentZ);

        const Lane4U32 lCount = (lCounts[i - 1] + blockAdd) >> blockShift;
        const Lane4U32 rCount = (count + blockAdd) >> blockShift;
        const Lane4F32 lArea  = area[i - 1];

        // TODO: consider increasing the cost of having empty children/leaves
        // (lCount & (blockAdd - 1));
        const Lane4F32 sah = FMA(rArea, Lane4F32(rCount), lArea * Lane4F32(lCount));

        lBestPos = Select(sah < lBestSAH, Lane4U32(i), lBestPos);
        lBestSAH = Select(sah < lBestSAH, sah, lBestSAH);
    }

    f32 bestArea = pos_inf;
    u32 bestPos  = 0;
    u32 bestDim  = 0;
    for (u32 dim = 0; dim < 3; dim++)
    {
        // if (scale[dim] == 0.f) continue;

        if (lBestSAH[dim] < bestArea)
        {
            bestArea = lBestSAH[dim];
            bestPos  = lBestPos[dim];
            bestDim  = dim;
        }
    }
    // f32 bestValue = splitPositions[bestDim][bestPos + 1];
    return Split(bestArea, bestPos, bestDim, 0.f); // bestValue);
}

} // namespace rt

#endif
