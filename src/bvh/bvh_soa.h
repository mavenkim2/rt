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

    __forceinline void Extend(const Lane8F32 &other)
    {
        v = Max(v, other.v);
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
void ClipTriangle(const TriangleMesh *mesh, const u32 dim, const Triangle8 &tri, const Lane8F32 &splitPos,
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

    Transpose6x8(*leftMinX, *leftMinY, *leftMinZ, *leftMaxX, *leftMaxY, *leftMaxZ,
                 lOut[0], lOut[1], lOut[2], lOut[3], lOut[4], lOut[5], lOut[6], lOut[7]);
    Transpose6x8(*rightMinX, *rightMinY, *rightMinZ, *rightMaxX, *rightMaxY, *rightMaxZ,
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

template <i32 numBins = 16>
struct HeuristicSOASplitBinning
{
    Lane8F32 baseX;
    Lane8F32 baseY;
    Lane8F32 baseZ;

    Lane8F32 scaleX;
    Lane8F32 scaleY;
    Lane8F32 scaleZ;

    static const u32 LANE_WIDTH = 8;

    // temp storage
    u32 faceIndices[3][numBins][LANE_WIDTH];
    u32 binCounts[3][numBins];

    // result data
    Lane4U32 entryCounts[numBins];
    Lane4U32 exitCounts[numBins];

    Bounds finalBounds[3][numBins];

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

        const Lane4F32 diag = Max(bounds.maxP - bounds.minP, eps);

        Lane4F32 scale4 = Select(diag > eps, Lane4F32((f32)numBins) / diag, Lane4F32(0.f));

        Lane8F32 scale8(scale4);
        scaleX = Shuffle<0>(scale8);
        scaleY = Shuffle<1>(scale8);
        scaleZ = Shuffle<2>(scale8);

        // test
        Lane4F32 invScale4 = Select(scale4 == 0.f, 0.f, 1.f / scale4);
        Lane8F32 invScale8(invScale4);
        invScaleX = Shuffle<0>(invScale8);
        invScaleY = Shuffle<1>(invScale8);
        invScaleZ = Shuffle<2>(invScale8);

        for (u32 i = 0; i < numBins; i++)
        {
            entryCounts[i] = 0;
            exitCounts[i]  = 0;
            for (u32 dim = 0; dim < 3; dim++)
            {
                binCounts[dim][i] = 0;
                for (u32 j = 0; j < 8; j++)
                {
                    faceIndices[dim][i][j] = 0;
                }
            }
        }
    }

    void BinDiffTest(TriangleMesh *mesh, PrimDataSOA *soa, u32 start, u32 count)
    {
        Lane8U32 z = Lane8U32(0);
        Lane8U32 e = Lane8U32(numBins - 1);

        alignas(32) u32 faceIDPrev[8];

        alignas(32) u32 indexMinPrev[8];
        alignas(32) u32 indexMaxPrev[8];

        alignas(32) u32 binIndexStart[3][numBins][8];

        Lane8F32 baseArr[]     = {baseX, baseY, baseZ};
        Lane8F32 scaleArr[]    = {scaleX, scaleY, scaleZ};
        Lane8F32 scaleNegArr[] = {FlipSign(scaleX), FlipSign(scaleY), FlipSign(scaleZ)};
        Lane8F32 invScaleArr[] = {invScaleX, invScaleY, invScaleZ};

        u32 i            = start;
        u32 alignedCount = count - count % LANE_WIDTH;
        f32 totalTime    = 0.f;

        Lane8F32 lanes[8];

        for (; i < start + count; i += LANE_WIDTH)
        {
            bool transposed  = false;
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

            Lane8U32::Store(faceIDPrev, faceIds);
            for (u32 dim = 0; dim < 3; dim++)
            {
                const Lane8F32 &baseD     = baseArr[dim];
                const Lane8F32 &scaleD    = scaleArr[dim];
                const Lane8F32 &scaleDNeg = scaleNegArr[dim];
                const Lane8F32 &invScaleD = invScaleArr[dim];

                Lane8U32 binIndexMin = Clamp(z, e, Flooru((baseD + prevMin[dim]) * scaleDNeg));
                Lane8U32 binIndexMax = Clamp(z, e, Flooru((prevMax[dim] - baseD) * scaleD));

                Lane8U32::Store(indexMinPrev, binIndexMin);
                Lane8U32::Store(indexMaxPrev, binIndexMax);

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
                        if (!transposed)
                        {
                            Transpose6x8(prevMin[0], prevMin[1], prevMin[2], prevMax[0], prevMax[1], prevMax[2],
                                         lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7]);
                            transposed = true;
                        }
                        bins8[dim][indexMin].Extend(lanes[prevIndex]);
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
                        alignas(32) u32 binIndices[LANE_WIDTH];

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
                                faceIndices[dim][diff][b] = 0;
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
                if (remainingCount == 0) continue;

                Triangle8 tri     = Triangle8::Load(mesh, dim, faceIndices[dim][diff]);
                Lane8U32 startBin = Lane8U32::Load(binIndexStart[dim][diff]);

                Bounds8 bounds[2][8];
                alignas(32) u32 binIndices[8];

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

        u32 refIDQueue[16];
        Bounds8 bounds[2][8];

        u32 splitCount      = 0;
        u32 totalSplitCount = 0;
        u32 splitBegin      = range.End();
        u32 splitMax        = range.ExtSize();
        PrimDataSOA out;
        out.minX         = range.data->minX + splitBegin;
        out.minY         = range.data->minY + splitBegin;
        out.minZ         = range.data->minZ + splitBegin;
        out.geomIDs      = range.data->geomIDs + splitBegin;
        out.maxX         = range.data->maxX + splitBegin;
        out.maxY         = range.data->maxY + splitBegin;
        out.maxZ         = range.data->maxZ + splitBegin;
        out.primIDs      = range.data->primIDs + splitBegin;
        f32 test         = 0.f;
        f32 negBestValue = -split.bestValue;
        for (u32 i = range.start; i < range.start + alignedCount; i += LANE_WIDTH)
        {
            Lane8F32 min = Lane8F32::LoadU(minStream + i);
            Lane8F32 max = Lane8F32::LoadU(maxStream + i);
            // Lane8U32 faceIDs   = Lane8U32::LoadU(range.data->primIDs + i);
            Lane8U32 refIDs    = Lane8U32::Step(i);
            Lane8F32 splitMask = (min > negBestValue & max > split.bestValue);
            u32 mask           = Movemask(splitMask);
            Lane8U32::StoreU(refIDQueue + count, MaskCompress(mask, refIDs));
            count += PopCount(mask);

            if (count >= LANE_WIDTH)
            {
                count -= LANE_WIDTH;
                alignas(32) u32 faceIDQueue[8];
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

                    range.data->minX[refID] = bounds[0][queueIndex].v[0];
                    range.data->minY[refID] = bounds[0][queueIndex].v[1];
                    range.data->minZ[refID] = bounds[0][queueIndex].v[2];
                    range.data->maxX[refID] = bounds[0][queueIndex].v[4];
                    range.data->maxY[refID] = bounds[0][queueIndex].v[5];
                    range.data->maxZ[refID] = bounds[0][queueIndex].v[6];

                    out.minX[splitCount - 8 + queueIndex] = bounds[1][queueIndex].v[0];
                    out.minY[splitCount - 8 + queueIndex] = bounds[1][queueIndex].v[1];
                    out.minZ[splitCount - 8 + queueIndex] = bounds[1][queueIndex].v[2];
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
};

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
