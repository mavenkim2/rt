#ifndef BVH_SOA_H
#define BVH_SOA_H
namespace rt
{
// TODO:
// unfortunately, partitioning in place with SOA scales really really badly. possible solutions:
// 1. AoS, one avx register for each prim data.
// 2. AoSoA
// 3. use an index buffer and partition that (would require gathers on the binning though, as well as more memory)

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
        minX[lIndex] = other.minX[rIndex];
        minY[lIndex] = other.minY[rIndex];
        minZ[lIndex] = other.minZ[rIndex];
        maxX[lIndex] = other.maxX[rIndex];
        maxY[lIndex] = other.maxY[rIndex];
        maxZ[lIndex] = other.maxZ[rIndex];
    }
    __forceinline void Swap(u32 lIndex, u32 rIndex)
    {
        ::rt::Swap(minX[lIndex], minX[rIndex]);
        ::rt::Swap(minY[lIndex], minY[rIndex]);
        ::rt::Swap(minZ[lIndex], minZ[rIndex]);
        ::rt::Swap(geomIDs[lIndex], geomIDs[rIndex]);
        ::rt::Swap(maxX[lIndex], maxX[rIndex]);
        ::rt::Swap(maxY[lIndex], maxY[rIndex]);
        ::rt::Swap(maxZ[lIndex], maxZ[rIndex]);
        ::rt::Swap(primIDs[lIndex], primIDs[rIndex]);
    }
};

template <u32 width = 8>
struct PrimDataAOSOA
{
    StaticAssert(IsPow2(width), pow2);
    static const u32 shift = BsfConst(width);
    static const u32 mask  = width - 1;
    f32 minX[width];
    f32 minY[width];
    f32 minZ[width];
    u32 geomIDs[width];
    f32 maxX[width];
    f32 maxY[width];
    f32 maxZ[width];
    u32 primIDs[width];

    f32 *operator[](i32 dim)
    {
        return (f32 *)this + dim * width;
    }
};

__forceinline void Swap(PrimDataAOSOA<8> *data, u32 lIndex, u32 rIndex)
{
    u32 lBlockIndex   = lIndex >> PrimDataAOSOA<8>::shift;
    u32 lIndexInBlock = lIndex & PrimDataAOSOA<8>::mask;

    u32 rBlockIndex   = rIndex >> PrimDataAOSOA<8>::shift;
    u32 rIndexInBlock = rIndex & PrimDataAOSOA<8>::mask;
    Swap(data[lBlockIndex].minX[lIndexInBlock], data[rBlockIndex].minX[rIndexInBlock]);
    Swap(data[lBlockIndex].minY[lIndexInBlock], data[rBlockIndex].minY[rIndexInBlock]);
    Swap(data[lBlockIndex].minZ[lIndexInBlock], data[rBlockIndex].minZ[rIndexInBlock]);
    Swap(data[lBlockIndex].maxX[lIndexInBlock], data[rBlockIndex].maxX[rIndexInBlock]);
    Swap(data[lBlockIndex].maxY[lIndexInBlock], data[rBlockIndex].maxY[rIndexInBlock]);
    Swap(data[lBlockIndex].maxZ[lIndexInBlock], data[rBlockIndex].maxZ[rIndexInBlock]);
}

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

struct ExtRangeAOSOA
{
    PrimDataAOSOA<8> *data;
    u32 start;
    u32 count;
    u32 extEnd;

    __forceinline ExtRangeAOSOA(PrimDataAOSOA<8> *data, u32 start, u32 count, u32 extEnd)
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
    static const u32 LUTX[] = {0, 2, 1};
    static const u32 LUTY[] = {1, 0, 2};
    static const u32 LUTZ[] = {2, 1, 0};

    Lane8F32 clipMasks[] = {
        tri.v0u <= splitPos,
        tri.v1u <= splitPos,
        tri.v2u <= splitPos,
    };

    Lane8F32 clipMasksR[] = {
        tri.v0u >= splitPos,
        tri.v1u >= splitPos,
        tri.v2u >= splitPos,
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
        right.MaskExtendR(clipMasksR[first], splitPos, v0u, v0v, v0w);

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
struct alignas(32) HeuristicSOASplitBinning
{
    Bounds8 bins8[3][numBins];
    Lane8F32 base[3];
    Lane8F32 invScale[3];
    Lane8F32 scale[3];

    static const u32 LANE_WIDTH = 8;

    // temp storage
    u32 faceIndices[3][numBins][2 * LANE_WIDTH - 1];
    u32 binCounts[3][numBins];

    // result data
    Lane4U32 entryCounts[numBins];
    Lane4U32 exitCounts[numBins];

    Bounds finalBounds[3][numBins];

    // used in soa 8 triangle multi plane diff binning

    HeuristicSOASplitBinning(Bounds &bounds)
    {
        const Lane4F32 eps = 1e-34f;

        Lane8F32 minP(bounds.minP);
        base[0] = Shuffle<0>(minP);
        base[1] = Shuffle<1>(minP);
        base[2] = Shuffle<2>(minP);

        const Lane4F32 diag = Max(bounds.maxP - bounds.minP, eps);

        Lane4F32 scale4 = Select(diag > eps, Lane4F32((f32)numBins) / diag, Lane4F32(0.f));

        Lane8F32 scale8(scale4);
        scale[0] = Shuffle<0>(scale8);
        scale[1] = Shuffle<1>(scale8);
        scale[2] = Shuffle<2>(scale8);

        // test
        Lane4F32 invScale4 = Select(scale4 == 0.f, 0.f, 1.f / scale4);
        Lane8F32 invScale8(invScale4);
        invScale[0] = Shuffle<0>(invScale8);
        invScale[1] = Shuffle<1>(invScale8);
        invScale[2] = Shuffle<2>(invScale8);

        for (u32 i = 0; i < numBins; i++)
        {
            entryCounts[i] = 0;
            exitCounts[i]  = 0;
            for (u32 dim = 0; dim < 3; dim++)
            {
                binCounts[dim][i] = 0;
                for (u32 j = 0; j < ArrayLength(faceIndices[0][0]); j++)
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

        // alignas(32) u32 faceIDPrev[8];

        // alignas(32) u32 indexMinPrev[8];
        // alignas(32) u32 indexMaxPrev[8];
        alignas(32) u32 binIndexStart[3][numBins][8];

        Lane8F32 scaleNegArr[] = {FlipSign(scale[0]), FlipSign(scale[1]), FlipSign(scale[2])};

        u32 i            = start;
        u32 alignedCount = count - count % LANE_WIDTH;
        f32 totalTime    = 0.f;

        Lane8F32 lanes[8];

        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            u32 *faceIDs = soa->primIDs + i;

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

            // Lane8U32::Store(faceIDPrev, faceIds);

            Lane8U32 indexMinArr[3] = {
                Clamp(z, e, Flooru((base[0] + prevMin[0]) * scaleNegArr[0])),
                Clamp(z, e, Flooru((base[1] + prevMin[1]) * scaleNegArr[1])),
                Clamp(z, e, Flooru((base[2] + prevMin[2]) * scaleNegArr[2])),
            };
            Lane8U32 indexMaxArr[3] = {
                Clamp(z, e, Flooru((prevMax[0] - base[0]) * scale[0])),
                Clamp(z, e, Flooru((prevMax[1] - base[1]) * scale[1])),
                Clamp(z, e, Flooru((prevMax[2] - base[2]) * scale[2])),
            };

            Lane8U32 binDiffX = indexMaxArr[0] - indexMinArr[0];
            Lane8U32 binDiffY = indexMaxArr[1] - indexMinArr[1];
            Lane8U32 binDiffZ = indexMaxArr[2] - indexMinArr[2];
            // x0 x1 x2 x3 y0 y1 y2 y3 x4 x5 x6 x7 y4 y5 y6 y7
            Lane8U32 out0 = _mm256_packus_epi32(binDiffX, binDiffY);
            // x0 x1 x2 x3 |  y0 y1 y2 y3 |  z0 00 z1 00 |  z2 00 z3 00 |  x4 x5 x6 x7 |  y4 y5 y6 y7 |  z4 00 z5 00 | z6 00 z7 00

            alignas(32) u8 bytes[32];
            Lane8U32 out1 = _mm256_packus_epi16(out0, binDiffZ);
            Lane8U32::Store(bytes, out1);

            u32 bitMask[3] = {};

            static const u32 order[3][LANE_WIDTH] = {
                {0, 1, 2, 3, 16, 17, 18, 19},
                {4, 5, 6, 7, 20, 21, 22, 23},
                {8, 10, 12, 14, 24, 26, 28, 30},
            };
            Transpose6x8(prevMin[0], prevMin[1], prevMin[2], prevMax[0], prevMax[1], prevMax[2],
                         lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7]);
            for (u32 dim = 0; dim < 3; dim++)
            {
                alignas(32) u32 indexMins[8];
                Lane8U32::Store(indexMins, indexMinArr[dim]);
                for (u32 b = 0; b < LANE_WIDTH; b++)
                {
                    u32 diff     = bytes[order[dim][b]];
                    u32 indexMin = indexMins[b];
                    entryCounts[indexMin][dim] += 1;
                    exitCounts[indexMin + diff][dim] += 1;
                    switch (diff)
                    {
                        case 0:
                        {
                            bins8[dim][indexMin].Extend(lanes[b]);
                        }
                        break;
                        default:
                        {
                            bitMask[dim] |= (1 << diff);
                            faceIndices[dim][diff][binCounts[dim][diff]]   = faceIDs[b];
                            binIndexStart[dim][diff][binCounts[dim][diff]] = indexMin;
                            binCounts[dim][diff]++;
                        }
                    }
                }
            }
            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 numIters = PopCount(bitMask[dim]);
                // while (bitMask[dim])
                for (u32 iter = 0; iter < numIters; iter++)
                {
                    u32 bin = Bsf(bitMask[dim]);
                    if (binCounts[dim][bin] >= LANE_WIDTH)
                    {
                        binCounts[dim][bin] -= LANE_WIDTH;
                        u32 binCount = binCounts[dim][bin];

                        Triangle8 tri     = Triangle8::Load(mesh, dim, faceIndices[dim][bin] + binCount);
                        Lane8U32 startBin = Lane8U32::LoadU(binIndexStart[dim][bin] + binCount);

                        Bounds8 bounds[2][LANE_WIDTH];
                        for (u32 boundIndex = 0; boundIndex < LANE_WIDTH; boundIndex++)
                        {
                            bounds[0][boundIndex] = Bounds8(pos_inf);
                        }
                        u32 binIndices[LANE_WIDTH];

                        u32 current = 0;
                        for (u32 d = 0; d < bin; d++)
                        {
                            Lane8U32::Store(binIndices, startBin);
                            startBin += 1u;
                            Lane8F32 splitPos = Lane8F32(startBin) * invScale[dim] + base[dim];
                            ClipTriangle(mesh, dim, tri, splitPos, bounds[current], bounds[!current]);

                            for (u32 b = 0; b < LANE_WIDTH; b++)
                            {
                                bins8[dim][binIndices[b]].Extend(bounds[current][b]);
                            }
                            current = !current;
                        }
                        binCounts[dim][bin] = 0;
                    }
                    bitMask[dim] &= bitMask[dim] - 1;
                }
            }
        }
        // Finish the remaining primitives
        f32 scaleArr[3]          = {scale[0][0], scale[1][0], scale[2][0]};
        f32 baseArr[3]           = {base[0][0], base[1][0], base[2][0]};
        f32 scaleNegScalarArr[3] = {scaleNegArr[0][0], scaleNegArr[1][0], scaleNegArr[2][0]};
        for (; i < start + count; i++)
        {
            f32 min[3] = {
                soa->minX[i],
                soa->minY[i],
                soa->minZ[i],
            };

            f32 max[3] = {
                soa->maxX[i],
                soa->maxY[i],
                soa->maxZ[i],
            };
            u32 faceID = soa->primIDs[i];

            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 binIndexMin = Clamp(0u, numBins - 1u, (u32)Floor((baseArr[dim] + min[dim]) * scaleNegScalarArr[dim]));
                u32 binIndexMax = Clamp(0u, numBins - 1u, (u32)Floor((max[dim] - baseArr[dim]) * scaleArr[dim]));
                u32 diff        = binIndexMax - binIndexMin;
                entryCounts[binIndexMin][dim] += 1;
                exitCounts[binIndexMax][dim] += 1;
                switch (diff)
                {
                    case 0:
                    {
                        Lane8F32 prim(min[0], min[1], min[2], pos_inf, max[0], max[1], max[2], pos_inf);
                        bins8[dim][binIndexMin].Extend(prim);
                    }
                    break;
                    default:
                    {
                        faceIndices[dim][diff][binCounts[dim][diff]]   = faceID;
                        binIndexStart[dim][diff][binCounts[dim][diff]] = binIndexMin;
                        binCounts[dim][diff]++;
                    }
                }
            }
        }
        // Empty the bins
        Lane8F32 posInf(pos_inf);
        Lane8F32 negInf(neg_inf);
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 diff = 1; diff < numBins; diff++)
            {
                u32 remainingCount = binCounts[dim][diff];

                const u32 numIters = ((remainingCount + 7) >> 3);
                for (u32 remaining = 0; remaining < numIters; remaining++)
                {
                    u32 numPrims      = Min(remainingCount, 8u);
                    Triangle8 tri     = Triangle8::Load(mesh, dim, faceIndices[dim][diff] + remaining * LANE_WIDTH);
                    Lane8U32 startBin = Lane8U32::Load(binIndexStart[dim][diff] + remaining * LANE_WIDTH);

                    Bounds8 bounds[2][8];
                    alignas(32) u32 binIndices[8];

                    u32 current = 0;
                    for (u32 d = 0; d < diff; d++)
                    {
                        Lane8U32::Store(binIndices, startBin);
                        startBin += 1u;
                        Lane8F32 splitPos = Lane8F32(startBin) * invScale[dim] + base[dim];
                        ClipTriangle(mesh, dim, tri, splitPos, bounds[current], bounds[!current]);

                        for (u32 b = 0; b < numPrims; b++)
                        {
                            bins8[dim][binIndices[b]].Extend(bounds[current][b]);
                        }
                        current = !current;
                    }
                    remainingCount -= 8;
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

    u32 SplitSOA(Arena *arena, TriangleMesh *mesh, ExtRangeSOA range, Split split, Bounds &outLeft, Bounds &outRight)
    {
        // partitioning
        u32 dim          = split.bestDim;
        f32 *minStream   = range.data->arr[dim];
        f32 *maxStream   = range.data->arr[dim + 4];
        u32 alignedCount = range.count - range.count % LANE_WIDTH;

        u32 count = 0;

        u32 refIDQueue[LANE_WIDTH * 2 - 1];
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
        u32 i            = range.start;
        for (; i < range.start + alignedCount; i += LANE_WIDTH)
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
                Triangle8 tri = Triangle8::Load(mesh, dim, faceIDQueue);
                ClipTrianglePartition(mesh, dim, tri, split.bestValue, bounds[0], bounds[1]);
                for (u32 queueIndex = 0; queueIndex < LANE_WIDTH; queueIndex++)
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
            }
        }
        for (; i < range.End(); i++)
        {
            f32 min           = minStream[i];
            f32 max           = maxStream[i];
            refIDQueue[count] = i;
            count += (min <= split.bestPos && max > split.bestPos);
        }
        if (splitCount + count > splitMax)
        {
            const u32 streamSize = u32(sizeof(u32) * range.count * GROW_AMOUNT);
            u8 *alloc            = PushArray(arena, u8, u32(streamSize * LANE_WIDTH));
            splitCount           = 0;
            out.minX             = (f32 *)(alloc + streamSize * 0);
            out.minY             = (f32 *)(alloc + streamSize * 1);
            out.minZ             = (f32 *)(alloc + streamSize * 2);
            out.geomIDs          = (u32 *)(alloc + streamSize * 3);
            out.maxX             = (f32 *)(alloc + streamSize * 4);
            out.maxY             = (f32 *)(alloc + streamSize * 5);
            out.maxZ             = (f32 *)(alloc + streamSize * 6);
            out.primIDs          = (u32 *)(alloc + streamSize * 7);
        }
        // Flush the queue
        for (u32 queueIndex = 0; queueIndex < count; queueIndex++)
        {
            u32 refID = refIDQueue[queueIndex];
            Bounds inBounds(Lane4F32(range.data->minX[refID], range.data->minY[refID], range.data->minZ[refID], 0.f),
                            Lane4F32(range.data->maxX[refID], range.data->maxY[refID], range.data->maxZ[refID], 0.f));
            Bounds l;
            Bounds r;
            ClipTriangleSimple(mesh, inBounds, range.data->primIDs[refID], dim, split.bestValue, l, r);
            range.data->minX[refID] = l.minP[0];
            range.data->minY[refID] = l.minP[1];
            range.data->minZ[refID] = l.minP[2];
            range.data->maxX[refID] = l.maxP[0];
            range.data->maxY[refID] = l.maxP[1];
            range.data->maxZ[refID] = l.maxP[2];

            out.minX[splitCount]    = l.minP[0];
            out.minY[splitCount]    = l.minP[1];
            out.minZ[splitCount]    = l.minP[2];
            out.geomIDs[splitCount] = range.data->geomIDs[refID];
            out.maxX[splitCount]    = l.maxP[0];
            out.maxY[splitCount]    = l.maxP[1];
            out.maxZ[splitCount]    = l.maxP[2];
            out.primIDs[splitCount] = range.data->primIDs[refID];
            splitCount++;
            totalSplitCount++;
        }
        if (totalSplitCount > range.ExtSize())
        {
            // Partition by moving elements to the right of the split to the new allocation
            PrimDataSOA *soa[2] = {range.data, &out};
            u32 counts[2]       = {};
            for (i = range.start; i < range.start + alignedCount; i += LANE_WIDTH)
            {
                Lane8F32 min       = Lane8F32::LoadU(minStream + i);
                Lane8F32 max       = Lane8F32::LoadU(maxStream + i);
                Lane8F32 centroid  = (max - min) * 0.5f;
                Lane8F32 splitMask = centroid >= split.bestValue;
                u32 mask           = Movemask(splitMask);

                for (u32 b = 0; b < LANE_WIDTH; b++)
                {
                    const u32 choice = (mask >> b) & 1;
                    soa[choice]->Set(*range.data, counts[choice], i + b);
                    counts[choice]++;
                }
            }
            Assert(false);
            return counts[0];
        }
        else
        {
            // Normal partitioning

            // Bounds8F32 left(neg_inf);
            // Bounds8F32 right(neg_inf);

            u32 v      = LUTAxis[dim];
            u32 w      = LUTAxis[v];
            f32 *minVs = range.data->arr[v];
            f32 *minWs = range.data->arr[w];

            f32 *maxVs = range.data->arr[v + 4];
            f32 *maxWs = range.data->arr[w + 4];

            const u32 numPerCacheLine   = CACHE_LINE_SIZE / sizeof(f32);
            const u32 blockSize         = numPerCacheLine * 2;
            const u32 blockMask         = blockSize - 1;
            const u32 blockShift        = Bsf(blockSize);
            const u32 numJobs           = OS_NumProcessors();
            const u32 numBlocksPerChunk = numJobs;
            const u32 chunkSize         = blockSize * numJobs;

            // NOTE: these should be aligned on a cache line
            u32 align = numPerCacheLine;
            Assert(IsPow2(align));
            u32 lStartAligned   = AlignPow2(range.start, align);
            u32 rEndAligned     = range.End() & ~(align - 1);
            alignedCount        = rEndAligned - lStartAligned;
            const u32 numChunks = (alignedCount + chunkSize - 1) / chunkSize;

            TempArena temp    = ScratchStart(0, 0);
            u32 *blockIndices = PushArray(temp.arena, u32, numChunks);
            u32 *outMid       = PushArray(temp.arena, u32, numJobs);
            for (u32 chunkIndex = 0; chunkIndex < numChunks; chunkIndex++)
            {
                blockIndices[chunkIndex] = RandomInt(0, numBlocksPerChunk);
            }

            // NOTE: it is very important to ensure that none of the processors touch the same cache lines when
            // reading/writing
            Scheduler::Counter counter = {};
            scheduler.Schedule(&counter, numJobs, 1, [&](u32 jobID) {
                const u32 queueSize      = LANE_WIDTH * 2 - 1;
                const u32 rightQueueSize = queueSize + LANE_WIDTH;
                const u32 group          = jobID;
                auto GetIndex            = [&](u32 index) {
                    Assert((index & (LANE_WIDTH - 1)) == 0);
                    const u32 chunkIndex   = index >> blockShift;
                    const u32 blockIndex   = (blockIndices[chunkIndex] + group) & (numBlocksPerChunk - 1);
                    const u32 indexInBlock = index & blockMask;

                    u32 outIndex = lStartAligned + chunkIndex * chunkSize + (blockIndex << blockShift) + indexInBlock;
                    return outIndex;
                };
                u32 leftQueue[queueSize];
                u32 leftCount = 0;

                u32 rightQueue[rightQueueSize];
                u32 rightCount = 0;

                u32 l          = 0;
                u32 r          = (numChunks << blockShift);
                u32 lastRIndex = GetIndex(r);
                Assert(!(lastRIndex >= rEndAligned && (lastRIndex - rEndAligned) == LANE_WIDTH));
                r = lastRIndex > rEndAligned ? AlignPow2(r, blockSize) - blockSize - LANE_WIDTH : r - LANE_WIDTH;
                for (;;)
                {
                    while (l <= r && leftCount < LANE_WIDTH)
                    {
                        u32 lIndex    = GetIndex(l);
                        Lane8F32 lMin = Lane8F32::Load(minStream + lIndex);
                        Lane8F32 lMax = Lane8F32::Load(maxStream + lIndex);
                        Lane8F32 minV = Lane8F32::Load(minVs + lIndex);
                        Lane8F32 minW = Lane8F32::Load(minWs + lIndex);
                        Lane8F32 maxV = Lane8F32::Load(maxVs + lIndex);
                        Lane8F32 maxW = Lane8F32::Load(maxWs + lIndex);

                        Lane8F32 lCentroid   = (lMax - lMin) * 0.5f;
                        Lane8F32 notLeftMask = lCentroid >= split.bestValue;
                        Lane8F32 leftMask    = lCentroid < split.bestValue;
                        u32 mask             = Movemask(notLeftMask);
                        Lane8U32 leftRefId   = Lane8U32::Step(lIndex);
                        Lane8U32::StoreU(leftQueue + leftCount, MaskCompress(mask, leftRefId));

                        // left.MaskExtendNegativeMin(leftMask, lMin, minV, minW, lMax, maxV, maxW);
                        // right.MaskExtendNegativeMin(notLeftMask, lMin, minV, minW, lMax, maxV, maxW);
                        l += LANE_WIDTH;
                        leftCount += PopCount(mask);
                    }
                    while (l <= r && rightCount < LANE_WIDTH)
                    {
                        u32 rIndex    = GetIndex(r);
                        Lane8F32 min  = Lane8F32::Load(minStream + rIndex);
                        Lane8F32 max  = Lane8F32::Load(maxStream + rIndex);
                        Lane8F32 minV = Lane8F32::Load(minVs + rIndex);
                        Lane8F32 minW = Lane8F32::Load(minWs + rIndex);
                        Lane8F32 maxV = Lane8F32::Load(maxVs + rIndex);
                        Lane8F32 maxW = Lane8F32::Load(maxWs + rIndex);

                        Lane8F32 centroid     = (max - min) * 0.5f;
                        Lane8F32 notRightMask = centroid < split.bestValue;
                        Lane8F32 rightMask    = centroid >= split.bestValue;
                        const u32 mask        = Movemask(notRightMask);

                        u32 notRightCount  = PopCount(mask);
                        Lane8F32 storeMask = Lane8F32::Mask((1 << notRightCount) - 1u);

                        Lane8U32 refID = Lane8U32::Step(rIndex);
                        // Store so that entries are sorted from smallest to largest
                        rightCount += notRightCount;
                        Lane8U32::StoreU(storeMask, rightQueue + queueSize - rightCount, MaskCompress(mask, refID));
                        // left.MaskExtendNegativeMin(notRightMask, min, minV, minW, max, maxV, maxW);
                        // right.MaskExtendNegativeMin(rightMask, min, minV, minW, max, maxV, maxW);
                        r -= LANE_WIDTH;
                    }
                    // TODO: finish properly
                    if (l > r)
                    {
                        outMid[group] = GetIndex(l);
                        break;
                    }
                    Assert(leftCount >= LANE_WIDTH && leftCount <= queueSize);
                    Assert(rightCount >= LANE_WIDTH && rightCount <= queueSize);

                    u32 minCount = Min(leftCount, rightCount);

                    for (u32 i = 0; i < minCount; i++)
                    {
                        u32 leftIndex  = leftQueue[i];
                        u32 rightIndex = rightQueue[queueSize - 1 - i];
                        range.data->Swap(leftIndex, rightIndex);
                    }
                    leftCount -= minCount;
                    rightCount -= minCount;
                    for (u32 i = 0; i < leftCount; i++)
                    {
                        leftQueue[i] = leftQueue[i + minCount];
                    }
                    for (u32 i = 0; i < rightCount; i++)
                    {
                        rightQueue[queueSize - 1 - i] = rightQueue[queueSize - 1 - i - minCount];
                    }
                }
                threadLocalStatistics[GetThreadIndex()].misc += 1; // OS_GetMilliseconds(perfCounter);
            });
            scheduler.Wait(&counter);
            u32 mid = pos_inf;
            for (u32 chunkIndex = 0; chunkIndex < numJobs; chunkIndex++)
            {
                mid = Min(outMid[chunkIndex], mid);
            }

#if 0
            for (;;)

            {
                while (l <= r && leftCount < LANE_WIDTH)
                {
                    Lane8F32 lMin = Lane8F32::LoadU(minStream + l);
                    Lane8F32 lMax = Lane8F32::LoadU(maxStream + l);
                    Lane8F32 minV = Lane8F32::LoadU(minVs + l);
                    Lane8F32 minW = Lane8F32::LoadU(minWs + l);
                    Lane8F32 maxV = Lane8F32::LoadU(maxVs + l);
                    Lane8F32 maxW = Lane8F32::LoadU(maxWs + l);

                    Lane8F32 lCentroid   = (lMax - lMin) * 0.5f;
                    Lane8F32 notLeftMask = lCentroid >= split.bestValue;
                    Lane8F32 leftMask    = lCentroid < split.bestValue;
                    u32 mask             = Movemask(notLeftMask);
                    Lane8U32 leftRefId   = Lane8U32::Step(l);
                    Lane8U32::StoreU(leftQueue + leftCount, MaskCompress(mask, leftRefId));

                    left.MaskExtendNegativeMin(leftMask, lMin, minV, minW, lMax, maxV, maxW);
                    right.MaskExtendNegativeMin(notLeftMask, lMin, minV, minW, lMax, maxV, maxW);
                    l += LANE_WIDTH;
                    leftCount += PopCount(mask);
                }
                while (l <= r && rightCount < LANE_WIDTH)
                {
                    Lane8F32 min  = Lane8F32::LoadU(minStream + r);
                    Lane8F32 max  = Lane8F32::LoadU(maxStream + r);
                    Lane8F32 minV = Lane8F32::LoadU(minVs + r);
                    Lane8F32 minW = Lane8F32::LoadU(minWs + r);
                    Lane8F32 maxV = Lane8F32::LoadU(maxVs + r);
                    Lane8F32 maxW = Lane8F32::LoadU(maxWs + r);

                    Lane8F32 centroid     = (max - min) * 0.5f;
                    Lane8F32 notRightMask = centroid < split.bestValue;
                    Lane8F32 rightMask    = centroid >= split.bestValue;
                    const u32 mask        = Movemask(notRightMask);

                    u32 notRightCount  = PopCount(mask);
                    Lane8F32 storeMask = Lane8F32::Mask((1 << notRightCount) - 1u);

                    Lane8U32 refID = Lane8U32::Step(r);
                    // Store so that entries are sorted from smallest to largest
                    rightCount += notRightCount;
                    Lane8U32::StoreU(storeMask, rightQueue + queueSize - rightCount, MaskCompress(mask, refID));
                    left.MaskExtendNegativeMin(notRightMask, min, minV, minW, max, maxV, maxW);
                    right.MaskExtendNegativeMin(rightMask, min, minV, minW, max, maxV, maxW);
                    r -= LANE_WIDTH;
                }
                if (l > r)
                {
                    u32 minCount = Min(leftCount, rightCount);
                    for (i = 0; i < minCount; i++)
                    {
                        range.data->Swap(leftQueue[i], rightQueue[queueSize - 1 - i]);
                    }
                    if (leftCount != minCount)
                    {
                        l = leftQueue[minCount];
                        r += LANE_WIDTH;
                    }
                    else
                    {
                        r = rightQueue[queueSize - 1 - minCount];
                        l -= LANE_WIDTH;
                    }
                    for (;;)
                    {
                        while (l <= r)
                        {
                            f32 min      = minStream[l];
                            f32 max      = maxStream[l];
                            f32 centroid = (max - min) * 0.5f;
                            if (centroid >= split.bestValue) break;
                            f32 minV  = minVs[l];
                            f32 minW  = minWs[l];
                            f32 maxV  = maxVs[l];
                            f32 maxW  = maxWs[l];
                            left.minU = Max(left.minU, min);
                            left.minV = Max(left.minV, minV);
                            left.minW = Max(left.minW, minW);
                            left.maxU = Max(left.maxU, max);
                            left.maxV = Max(left.maxV, maxV);
                            left.maxW = Max(left.maxW, maxW);

                            l++;
                        }
                        while (l <= r)
                        {
                            f32 min      = minStream[r];
                            f32 max      = maxStream[r];
                            f32 centroid = (max - min) * 0.5f;
                            if (centroid < split.bestValue) break;
                            f32 minV   = minVs[r];
                            f32 minW   = minWs[r];
                            f32 maxV   = maxVs[r];
                            f32 maxW   = maxWs[r];
                            right.minU = Max(right.minU, min);
                            right.minV = Max(right.minV, minV);
                            right.minW = Max(right.minW, minW);
                            right.maxU = Max(right.maxU, max);
                            right.maxV = Max(right.maxV, maxV);
                            right.maxW = Max(right.maxW, maxW);
                            r--;
                        }
                        if (l > r) break;

                        f32 lMinU  = minStream[l];
                        f32 lMinV  = minVs[l];
                        f32 lMinW  = minWs[l];
                        f32 lMaxU  = maxStream[l];
                        f32 lMaxV  = maxVs[l];
                        f32 lMaxW  = maxWs[l];
                        right.minU = Max(right.minU, lMinU);
                        right.minV = Max(right.minV, lMinV);
                        right.minW = Max(right.minW, lMinW);
                        right.maxU = Max(right.maxU, lMaxU);
                        right.maxV = Max(right.maxV, lMaxV);
                        right.maxW = Max(right.maxW, lMaxW);

                        f32 rMinU = minStream[r];
                        f32 rMinV = minVs[r];
                        f32 rMinW = minWs[r];
                        f32 rMaxU = maxStream[r];
                        f32 rMaxV = maxVs[r];
                        f32 rMaxW = maxWs[r];
                        left.minU = Max(left.minU, rMinU);
                        left.minV = Max(left.minV, rMinV);
                        left.minW = Max(left.minW, rMinW);
                        left.maxU = Max(left.maxU, rMaxU);
                        left.maxV = Max(left.maxV, rMaxV);
                        left.maxW = Max(left.maxW, rMaxW);
                        range.data->Swap(l, r);
                        l++;
                        r--;
                    }
                    break;
                }
                Assert(leftCount >= LANE_WIDTH);
                Assert(rightCount >= LANE_WIDTH);

                u32 minCount = Min(leftCount, rightCount);

                for (i = 0; i < minCount; i++)
                {
                    u32 leftIndex  = leftQueue[i];
                    u32 rightIndex = rightQueue[queueSize - 1 - i];
                    range.data->Swap(leftIndex, rightIndex);
                }
                leftCount -= minCount;
                rightCount -= minCount;
                for (i = 0; i < leftCount; i++)
                {
                    leftQueue[i] = leftQueue[i + minCount];
                }
                for (i = 0; i < rightCount; i++)
                {
                    rightQueue[queueSize - 1 - i] = rightQueue[queueSize - 1 - i - minCount];
                }
            }
            f32 minScalars[6] = {
                ReduceMax(left.minU),
                ReduceMax(left.minV),
                ReduceMax(left.minW),
                ReduceMax(right.minU),
                ReduceMax(right.minV),
                ReduceMax(right.minW),
            };
            f32 maxScalars[6] = {
                ReduceMax(left.maxU),
                ReduceMax(left.maxV),
                ReduceMax(left.maxW),
                ReduceMax(right.maxU),
                ReduceMax(right.maxV),
                ReduceMax(right.maxW),
            };

            outLeft.minP[dim]  = minScalars[0];
            outLeft.minP[v]    = minScalars[1];
            outLeft.minP[w]    = minScalars[2];
            outRight.minP[dim] = minScalars[3];
            outRight.minP[v]   = minScalars[4];
            outRight.minP[w]   = minScalars[5];
            outLeft.maxP[dim]  = maxScalars[0];
            outLeft.maxP[v]    = maxScalars[1];
            outLeft.maxP[w]    = maxScalars[2];
            outRight.maxP[dim] = maxScalars[3];
            outRight.maxP[v]   = maxScalars[4];
            outRight.maxP[w]   = maxScalars[5];

#endif
            return mid;
        }
    }

    void Split(Arena *arena, TriangleMesh *mesh, ExtRangeAOSOA range, struct Split split, Bounds &outLeft, Bounds &outRight)
    {
        u32 dim = split.bestDim;
        Lane8U32 shufMask(dim);

        static constexpr u32 dataShift = BsfConst(LANE_WIDTH);
        static constexpr u32 dataMask  = LANE_WIDTH - 1;

        u32 refIDQueue[LANE_WIDTH * 2 - 1];
        u32 faceIDQueue[LANE_WIDTH * 2 - 1];
        Bounds8 bounds[2][8];

        PrimDataAOSOA<8> *data = range.data;
        f32 negBestValue       = -split.bestValue;
        u32 queueCount         = 0;
        u32 splitCount         = 0;

        u32 splitBegin = range.End();
        u32 splitMax   = range.ExtSize();

        // TODO: this is not correct. the start could be in the middle of a block, and so can the end
        u32 end = range.End() >> dataShift;
        for (u32 i = range.start >> dataShift; i < end; i++)
        {
            PrimDataAOSOA<8> *prims = &data[i];
            Lane8F32 min            = Lane8F32::LoadU((*prims)[dim]);
            Lane8F32 max            = Lane8F32::LoadU((*prims)[dim + 4]);
            Lane8U32 faceIDs        = Lane8U32::LoadU(prims->primIDs);
            Lane8U32 refIDs         = Lane8U32::Step(i * LANE_WIDTH);
            Lane8F32 splitMask      = (min > negBestValue & max > split.bestValue);
            u32 mask                = Movemask(splitMask);
            Lane8U32::StoreU(refIDQueue + queueCount, MaskCompress(mask, refIDs));
            Lane8U32::StoreU(faceIDQueue + queueCount, MaskCompress(mask, faceIDs));
            queueCount += PopCount(mask);

            if (queueCount >= LANE_WIDTH)
            {
                queueCount -= LANE_WIDTH;
                splitCount += LANE_WIDTH;
                if (splitCount > splitMax) break;
                Triangle8 tri = Triangle8::Load(mesh, dim, faceIDQueue + queueCount);
                ClipTrianglePartition(mesh, dim, tri, split.bestValue, bounds[0], bounds[1]);
                for (u32 queueIndex = 0; queueIndex < LANE_WIDTH; queueIndex++)
                {
                    const u32 refID        = refIDQueue[queueCount + queueIndex];
                    const u32 blockIndex   = refID >> dataShift;
                    const u32 indexInBlock = refID & dataMask;

                    const u32 rightBlockIndex   = (splitBegin + splitCount) >> dataShift;
                    const u32 rightIndexInBlock = (splitBegin + splitCount) & dataMask;

                    data[blockIndex].minX[indexInBlock] = bounds[0][queueIndex].v[0];
                    data[blockIndex].minY[indexInBlock] = bounds[0][queueIndex].v[1];
                    data[blockIndex].minZ[indexInBlock] = bounds[0][queueIndex].v[2];
                    data[blockIndex].maxX[indexInBlock] = bounds[0][queueIndex].v[4];
                    data[blockIndex].maxY[indexInBlock] = bounds[0][queueIndex].v[5];
                    data[blockIndex].maxZ[indexInBlock] = bounds[0][queueIndex].v[6];

                    data[rightBlockIndex].minX[indexInBlock] = bounds[1][queueIndex].v[0];
                    data[rightBlockIndex].minY[indexInBlock] = bounds[1][queueIndex].v[1];
                    data[rightBlockIndex].minZ[indexInBlock] = bounds[1][queueIndex].v[2];
                    data[rightBlockIndex].maxX[indexInBlock] = bounds[1][queueIndex].v[4];
                    data[rightBlockIndex].maxY[indexInBlock] = bounds[1][queueIndex].v[5];
                    data[rightBlockIndex].maxZ[indexInBlock] = bounds[1][queueIndex].v[6];
                }
            }
        }
        {
            // Normal partitioning
            const u32 queueSize      = LANE_WIDTH * 2 - 1;
            const u32 rightQueueSize = queueSize + LANE_WIDTH;
            u32 leftQueue[queueSize];
            u32 leftCount = 0;

            u32 rightQueue[rightQueueSize];
            u32 rightCount = 0;

            i32 l = (i32)range.start;
            i32 r = (i32)range.End() - 8;

            Bounds8F32 left(neg_inf);
            Bounds8F32 right(neg_inf);

            u32 v = LUTAxis[dim];
            u32 w = LUTAxis[v];
            for (;;)
            {
                while (l <= r && leftCount < LANE_WIDTH)
                {
                    PrimDataAOSOA<8> *prim = &data[l >> dataShift];
                    Lane8F32 lMin          = Lane8F32::LoadU((*prim)[dim]);
                    Lane8F32 lMax          = Lane8F32::LoadU((*prim)[v]);
                    Lane8F32 minV          = Lane8F32::LoadU((*prim)[w]);
                    Lane8F32 minW          = Lane8F32::LoadU((*prim)[dim]);
                    Lane8F32 maxV          = Lane8F32::LoadU((*prim)[v]);
                    Lane8F32 maxW          = Lane8F32::LoadU((*prim)[w]);

                    Lane8F32 lCentroid   = (lMax - lMin) * 0.5f;
                    Lane8F32 notLeftMask = lCentroid >= split.bestValue;
                    Lane8F32 leftMask    = lCentroid < split.bestValue;
                    u32 mask             = Movemask(notLeftMask);
                    Lane8U32 leftRefId   = Lane8U32::Step(l);
                    Lane8U32::StoreU(leftQueue + leftCount, MaskCompress(mask, leftRefId));

                    left.MaskExtendNegativeMin(leftMask, lMin, minV, minW, lMax, maxV, maxW);
                    right.MaskExtendNegativeMin(notLeftMask, lMin, minV, minW, lMax, maxV, maxW);
                    l += LANE_WIDTH;
                    leftCount += PopCount(mask);
                }
                while (l <= r && rightCount < LANE_WIDTH)
                {
                    PrimDataAOSOA<8> *prim = &data[r >> dataShift];
                    Lane8F32 min           = Lane8F32::LoadU((*prim)[dim]);
                    Lane8F32 max           = Lane8F32::LoadU((*prim)[v]);
                    Lane8F32 minV          = Lane8F32::LoadU((*prim)[w]);
                    Lane8F32 minW          = Lane8F32::LoadU((*prim)[dim]);
                    Lane8F32 maxV          = Lane8F32::LoadU((*prim)[v]);
                    Lane8F32 maxW          = Lane8F32::LoadU((*prim)[w]);

                    Lane8F32 centroid     = (max - min) * 0.5f;
                    Lane8F32 notRightMask = centroid < split.bestValue;
                    Lane8F32 rightMask    = centroid >= split.bestValue;
                    const u32 mask        = Movemask(notRightMask);

                    u32 notRightCount  = PopCount(mask);
                    Lane8F32 storeMask = Lane8F32::Mask((1 << notRightCount) - 1u);

                    Lane8U32 refID = Lane8U32::Step(r);
                    // Store so that entries are sorted from smallest to largest
                    rightCount += notRightCount;
                    Lane8U32::StoreU(storeMask, rightQueue + queueSize - rightCount, MaskCompress(mask, refID));
                    left.MaskExtendNegativeMin(notRightMask, min, minV, minW, max, maxV, maxW);
                    right.MaskExtendNegativeMin(rightMask, min, minV, minW, max, maxV, maxW);
                    r -= LANE_WIDTH;
                }
                if (l > r) break;
                // {
                //     u32 minCount = Min(leftCount, rightCount);
                //     for (i = 0; i < minCount; i++)
                //     {
                //         Swap(data, leftQueue[i], rightQueue[queueSize - 1 - i]);
                //     }
                //     if (leftCount != minCount)
                //     {
                //         l = leftQueue[minCount];
                //         r += LANE_WIDTH;
                //     }
                //     else
                //     {
                //         r = rightQueue[queueSize - 1 - minCount];
                //         l -= LANE_WIDTH;
                //     }
                //     for (;;)
                //     {
                //         while (l <= r)
                //         {
                //             f32 min      = minStream[l];
                //             f32 max      = maxStream[l];
                //             f32 centroid = (max - min) * 0.5f;
                //             if (centroid >= split.bestValue) break;
                //             f32 minV  = minVs[l];
                //             f32 minW  = minWs[l];
                //             f32 maxV  = maxVs[l];
                //             f32 maxW  = maxWs[l];
                //             left.minU = Max(left.minU, min);
                //             left.minV = Max(left.minV, minV);
                //             left.minW = Max(left.minW, minW);
                //             left.maxU = Max(left.maxU, max);
                //             left.maxV = Max(left.maxV, maxV);
                //             left.maxW = Max(left.maxW, maxW);
                //
                //             l++;
                //         }
                //         while (l <= r)
                //         {
                //             f32 min      = minStream[r];
                //             f32 max      = maxStream[r];
                //             f32 centroid = (max - min) * 0.5f;
                //             if (centroid < split.bestValue) break;
                //             f32 minV   = minVs[r];
                //             f32 minW   = minWs[r];
                //             f32 maxV   = maxVs[r];
                //             f32 maxW   = maxWs[r];
                //             right.minU = Max(right.minU, min);
                //             right.minV = Max(right.minV, minV);
                //             right.minW = Max(right.minW, minW);
                //             right.maxU = Max(right.maxU, max);
                //             right.maxV = Max(right.maxV, maxV);
                //             right.maxW = Max(right.maxW, maxW);
                //             r--;
                //         }
                //         if (l > r) break;
                //
                //         f32 lMinU  = minStream[l];
                //         f32 lMinV  = minVs[l];
                //         f32 lMinW  = minWs[l];
                //         f32 lMaxU  = maxStream[l];
                //         f32 lMaxV  = maxVs[l];
                //         f32 lMaxW  = maxWs[l];
                //         right.minU = Max(right.minU, lMinU);
                //         right.minV = Max(right.minV, lMinV);
                //         right.minW = Max(right.minW, lMinW);
                //         right.maxU = Max(right.maxU, lMaxU);
                //         right.maxV = Max(right.maxV, lMaxV);
                //         right.maxW = Max(right.maxW, lMaxW);
                //
                //         f32 rMinU = minStream[r];
                //         f32 rMinV = minVs[r];
                //         f32 rMinW = minWs[r];
                //         f32 rMaxU = maxStream[r];
                //         f32 rMaxV = maxVs[r];
                //         f32 rMaxW = maxWs[r];
                //         left.minU = Max(left.minU, rMinU);
                //         left.minV = Max(left.minV, rMinV);
                //         left.minW = Max(left.minW, rMinW);
                //         left.maxU = Max(left.maxU, rMaxU);
                //         left.maxV = Max(left.maxV, rMaxV);
                //         left.maxW = Max(left.maxW, rMaxW);
                //         range.data->Swap(l, r);
                //         l++;
                //         r--;
                //     }
                //     break;
                // }
                Assert(leftCount >= LANE_WIDTH);
                Assert(rightCount >= LANE_WIDTH);

                u32 minCount = Min(leftCount, rightCount);

                for (u32 i = 0; i < minCount; i++)
                {
                    u32 leftIndex  = leftQueue[i];
                    u32 rightIndex = rightQueue[queueSize - 1 - i];
                    Swap(data, leftIndex, rightIndex);
                }
                leftCount -= minCount;
                rightCount -= minCount;
                for (u32 i = 0; i < leftCount; i++)
                {
                    leftQueue[i] = leftQueue[i + minCount];
                }
                for (u32 i = 0; i < rightCount; i++)
                {
                    rightQueue[queueSize - 1 - i] = rightQueue[queueSize - 1 - i - minCount];
                }
            }
            f32 minScalars[6] = {
                ReduceMax(left.minU),
                ReduceMax(left.minV),
                ReduceMax(left.minW),
                ReduceMax(right.minU),
                ReduceMax(right.minV),
                ReduceMax(right.minW),
            };
            f32 maxScalars[6] = {
                ReduceMax(left.maxU),
                ReduceMax(left.maxV),
                ReduceMax(left.maxW),
                ReduceMax(right.maxU),
                ReduceMax(right.maxV),
                ReduceMax(right.maxW),
            };

            outLeft.minP[dim]  = minScalars[0];
            outLeft.minP[v]    = minScalars[1];
            outLeft.minP[w]    = minScalars[2];
            outRight.minP[dim] = minScalars[3];
            outRight.minP[v]   = minScalars[4];
            outRight.minP[w]   = minScalars[5];
            outLeft.maxP[dim]  = maxScalars[0];
            outLeft.maxP[v]    = maxScalars[1];
            outLeft.maxP[w]    = maxScalars[2];
            outRight.maxP[dim] = maxScalars[3];
            outRight.maxP[v]   = maxScalars[4];
            outRight.maxP[w]   = maxScalars[5];

            // return l;
        }
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
    __forceinline u32 Split(TriangleMesh *mesh, PrimData *prims, ExtRange range, Split split, Bounds &outLeft, Bounds &outRight)
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

        PartitionResult result;
        PartitionParallel(split, prims, range.start, range.End(), &result);
#if 0
        u32 l = range.start;
        u32 r = range.End() - 1;
        for (;;)
        {
            b32 isLeft = true;
            Lane4F32 lCentroid;
            Lane4F32 rCentroid;
            PrimData *lPrim;
            PrimData *rPrim;
            do
            {
                lPrim     = &prims[l];
                lCentroid = (lPrim->minP + lPrim->maxP) * 0.5f;
                if (lCentroid[split.bestDim] >= split.bestValue) break;
                outLeft.Extend(lPrim->minP, lPrim->maxP);
                l++;
            } while (l <= r);
            do
            {
                rPrim     = &prims[r];
                rCentroid = (rPrim->minP + rPrim->maxP) * 0.5f;
                if (rCentroid[split.bestDim] < split.bestValue) break;
                outRight.Extend(rPrim->minP, rPrim->maxP);
                r--;
            } while (l <= r);
            if (l > r) break;

            Swap(lPrim->minP, rPrim->minP);
            Swap(lPrim->maxP, rPrim->maxP);
            outLeft.Extend(lPrim->minP, lPrim->maxP);
            outRight.Extend(rPrim->minP, rPrim->maxP);

            l++;
            r--;
        }

        return l;
#endif
        return result.mid;

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
    return Split(bestArea, bestPos, bestDim, 0.f); // bestValue);
}

} // namespace rt

#endif
