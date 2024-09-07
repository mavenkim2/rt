#ifndef BVH_SAH_H
#define BVH_SAH_H
#include "../math/simd_base.h"
namespace rt
{

// TODO: make sure to pad to max lane width, set min and maxes to pos_inf and neg_inf and count to zero for padded entries
struct PrimDataSOA
{
    f32 *minX;
    f32 *minY;
    f32 *minZ;
    f32 *maxX;
    f32 *maxY;
    f32 *maxZ;

    u32 *counts;
    u32 total;
};

struct Split
{
    f32 bestSAH;
    u32 bestPos;
    u32 bestDim;

    Split(f32 sah, u32 pos, u32 dim) : bestSAH(sah), bestPos(pos), bestDim(dim) {}
};

template <i32 numBins>
struct HeuristicSAHBinned
{
    StaticAssert(numBins >= 4, MoreThan4Bins);

    static const u32 indexMask   = MAX_LANE_WIDTH - 1;
    static const u32 pow2NumBins = NextPowerOfTwo(numBins);
    static const u32 numBinsX    = pow2NumBins < MAX_LANE_WIDTH ? pow2NumBins : MAX_LANE_WIDTH;
    static const u32 numBinsY    = (numBins + numBinsX - 1) / numBinsX; // MAX_LANE_WIDTH - 1) / MAX_LANE_WIDTH;

    LaneF32<numBinsX> binMinX[numBinsY][3];
    LaneF32<numBinsX> binMinY[numBinsY][3];
    LaneF32<numBinsX> binMinZ[numBinsY][3];
    LaneF32<numBinsX> binMaxX[numBinsY][3];
    LaneF32<numBinsX> binMaxY[numBinsY][3];
    LaneF32<numBinsX> binMaxZ[numBinsY][3];

    LaneU32<numBinsX> binCounts[numBinsY][3];

    AABB base;
    LaneVec3f<MAX_LANE_WIDTH> minP;
    LaneVec3f<MAX_LANE_WIDTH> scale;

    HeuristicSAHBinned() {}
    // TODO IMPORTANT: base is the bounds over the centroids
    HeuristicSAHBinned(const AABB &base)
    {
        const LaneF32<MAX_LANE_WIDTH> eps    = 1e-34f;
        minP                                 = base.minP;
        const LaneVec3f<MAX_LANE_WIDTH> diag = Max(base.maxP - base.minP, Vec3f(0.f));
        scale                                = Select(range > eps, LaneVec3f<MAX_LANE_WIDTH>(Vec3f(1.f)) / diag, 0.f);
    }

    __forceinline void Bin(PrimDataSOA *primData)
    {
        for (u32 i = 0; i < primData->total; i += numBinsX)
        {
            // Load
            LaneF32<numBinsX> minX = LaneF32<numBinsX>::Load(primData->minX + i);
            LaneF32<numBinsX> minY = LaneF32<numBinsX>::Load(primData->minY + i);
            LaneF32<numBinsX> minZ = LaneF32<numBinsX>::Load(primData->minZ + i);

            LaneF32<numBinsX> maxX = LaneF32<numBinsX>::Load(primData->maxX + i);
            LaneF32<numBinsX> maxY = LaneF32<numBinsX>::Load(primData->maxY + i);
            LaneF32<numBinsX> maxZ = LaneF32<numBinsX>::Load(primData->maxZ + i);

            LaneU32<numBinsX> counts = LaneU32<MAX_LANE_WIDTH>::Load(primData->counts + i);

            // Find the bin index based on the centroid
            LaneVec3f<MAX_LANE_WIDTH> centroid =
                LaneVec3f<MAX_LANE_WIDTH>((minX + maxX) * 0.5f, (minY + maxY) * 0.5f, (minZ + maxZ) * 0.5f);

            LaneVec3u<MAX_LANE_WIDTH> binIndices = Flooru((centroid - minP) * scale * N);

            for (u32 j = 0; j < numBinsY; j++)
            {
                for (u32 dim = 0; dim < 3; dim++)
                {
                    LaneU32 maskDim     = (binIndices[dim] < numBinsX * (j + 1));
                    LaneU32 permutation = (binIndices[dim] & indexMask);
                    binMinX[j][dim]     = MaskMin(maskDim, binMinX[j], Permute(minX, permutation));
                    binMaxX[j][dim]     = MaskMax(maskDim, binMaxX[j], Permute(maxX, permutation));

                    binMinY[j][dim] = MaskMin(maskDim, binMinY[j], Permute(minY, permutation));
                    binMaxY[j][dim] = MaskMax(maskDim, binMaxY[j], Permute(maxY, permutation));

                    binMinZ[j][dim] = MaskMin(maskDim, binMinZ[j], Permute(minZ, permutation));
                    binMaxZ[j][dim] = MaskMax(maskDim, binMaxZ[j], Permute(maxZ, permutation));

                    binCounts[j][dim] = MaskAdd(maskDim, binCounts[j][dim], Permute(counts, permutation));
                }
            }
        }
    }

    template <bool isMin>
    __forceinline void PrefixForward(const LaneF32<numBinsX> &binIn,
                                     LaneF32<numBinsX> &prev,
                                     LaneF32<numBinsX> &out)
    {
#if MAX_LANE_WIDTH == 4
        if constexpr (isMin)
        {
            out = Min(ShiftUp<1, pos_inf>(binIn), binIn);
            out = Min(ShiftUp<2, pos_inf>(out, out));
            out = Min(prev, out);
        }
        else
        {
            out = Max(ShiftUp<1, neg_inf>(binIn), binIn);
            out = Max(ShiftUp<2, neg_inf>(out), out);
            out = Max(prev, out);
        }

#elif MAX_LANE_WIDTH == 8
        if constexpr (isMin)
        {
            out = Min(ShiftUp<1, pos_inf>(binIn), binIn);
            out = Min(ShiftUp<2, pos_inf>(out), out);
            out = Min(ShiftUp<4, pos_inf>(out), out);
            out = Min(prev, out);
        }
        else
        {
            out = Max(ShiftUp<1, neg_inf>(binIn), binIn);
            out = Max(ShiftUp<2, neg_inf>(out), out);
            out = Max(ShiftUp<4, neg_inf>(out), out);
            out = Max(prev, out);
        }

#elif MAX_LANE_WIDTH == 16
        if constexpr (isMin)
        {
            out = Min(ShiftUp<1, pos_inf>(binIn), binIn);
            out = Min(ShiftUp<2, pos_inf>(out), out);
            out = Min(ShiftUp<4, pos_inf>(out), out);
            out = Min(ShiftUp<8, pos_inf>(out), out);
            out = Min(prev, out);
        }
        else
        {
            out = Max(ShiftUp<1, neg_inf>(binIn), binIn);
            out = Max(ShiftUp<2, neg_inf>(out), out);
            out = Max(ShiftUp<4, neg_inf>(out), out);
            out = Max(ShiftUp<8, neg_inf>(out), out);
            out = Max(prev, out);
        }
#endif
        prev = Shuffle<numBinsX - 1>(out);
    }

    template <bool isMin>
    __forceinline void PrefixBackward(const LaneF32<numBinsX> &binIn,
                                      LaneF32<numBinsX> &prev,
                                      LaneF32<numBinsX> &out)
    {
#if MAX_LANE_WIDTH == 4
        if constexpr (isMin)
        {
            out = ShiftDown<1, pos_inf>(binIn);
            out = Min(ShiftDown<2, pos_inf>(out, out));
            out = Min(prev, out);
        }
        else
        {
            out = ShiftDown<1, neg_inf>(binIn);
            out = Max(ShiftDown<2, neg_inf>(out), out);
            out = Max(prev, out);
        }

#elif MAX_LANE_WIDTH == 8
        if constexpr (isMin)
        {
            out = ShiftDown<1, pos_inf>(binIn);
            out = Min(ShiftDown<2, pos_inf>(out), out);
            out = Min(ShiftDown<4, pos_inf>(out), out);
            out = Min(prev, out);
        }
        else
        {
            out = ShiftDown<1, neg_inf>(binIn), binIn);
            out = Max(ShiftDown<2, neg_inf>(out), out);
            out = Max(ShiftDown<4, neg_inf>(out), out);
            out = Max(prev, out);
        }

#elif MAX_LANE_WIDTH == 16
        if constexpr (isMin)
        {
            out = ShiftDown<1, pos_inf>(binIn);
            out = Min(ShiftDown<2, pos_inf>(out), out);
            out = Min(ShiftDown<4, pos_inf>(out), out);
            out = Min(ShiftDown<8, pos_inf>(out), out);
            out = Min(prev, out);
        }
        else
        {
            out = ShiftDown<1, neg_inf>(binIn);
            out = Max(ShiftDown<2, neg_inf>(out), out);
            out = Max(ShiftDown<4, neg_inf>(out), out);
            out = Max(ShiftDown<8, neg_inf>(out), out);
            out = Max(prev, out);
        }
#endif
        prev = Shuffle<0>(out);
    }

    // NOTE: 1 << blockShift is the block size (e.g. intersecting groups of 4 triangles -> blockShift = 2)
    __forceinline Split Best(const u32 blockShift)
    {
        LaneF32<numBinsX> binSumMinX[numBinsY][3];
        LaneF32<numBinsX> binSumMinY[numBinsY][3];
        LaneF32<numBinsX> binSumMinZ[numBinsY][3];
        LaneF32<numBinsX> binSumMaxX[numBinsY][3];
        LaneF32<numBinsX> binSumMaxY[numBinsY][3];
        LaneF32<numBinsX> binSumMaxZ[numBinsY][3];

        LaneF32<numBinsX> binPrevMinX[3];
        LaneF32<numBinsX> binPrevMinY[3];
        LaneF32<numBinsX> binPrevMinZ[3];
        LaneF32<numBinsX> binPrevMaxX[3];
        LaneF32<numBinsX> binPrevMaxY[3];
        LaneF32<numBinsX> binPrevMaxZ[3];
        LaneF32<numBinsX> prevCount[3];

        LaneF32<numBinsX> counts[numBinsY][3];
        LaneF32<numBinsX> area[numBinsY][3];

        for (u32 dim = 0; dim < 3; dim++)
        {
            binPrevMinX[dim] = pos_inf;
            binPrevMinY[dim] = pos_inf;
            binPrevMinZ[dim] = pos_inf;

            binPrevMaxX[dim] = neg_inf;
            binPrevMaxY[dim] = neg_inf;
            binPrevMaxZ[dim] = neg_inf;
            prevCount[dim]   = 0;
            for (u32 i = 0; i < numBinsY; i++)
            {
                counts[i][dim] = 0;
                area[i][dim]   = 0;
            }
        }

        for (u32 j = 0; j < numBinsY; j++)
        {
            for (u32 dim = 0; dim < 3; dim++)
            {
                PrefixForward<true>(binMinX[j][dim], binPrevMinX[dim], binSumMinX[j][dim]);
                PrefixForward<true>(binMinY[j][dim], binPrevMinY[dim], binSumMinY[j][dim]);
                PrefixForward<true>(binMinZ[j][dim], binPrevMinZ[dim], binSumMinZ[j][dim]);

                PrefixForward<false>(binMaxX[j][dim], binPrevMaxX[dim], binSumMaxX[j][dim]);
                PrefixForward<false>(binMaxY[j][dim], binPrevMaxY[dim], binSumMaxY[j][dim]);
                PrefixForward<false>(binMaxZ[j][dim], binPrevMaxZ[dim], binSumMaxZ[j][dim]);

#if MAX_LANE_WIDTH == 4
                counts[j][dim] = ShiftUp<1>(binCounts[j][dim]) + binCounts[j][dim];
                counts[j][dim] += ShiftUp<2>(counts[j][dim]) + counts[j][dim];
                counts[j][dim] += prevCounts;
#elif MAX_LANE_WIDTH == 8
                counts[j][dim] = ShiftUp<1>(binCounts[j][dim]) + binCounts[j][dim];
                counts[j][dim] += ShiftUp<2>(counts[j][dim]) + counts[j][dim];
                counts[j][dim] += ShiftUp<4>(counts[j][dim]) + counts[j][dim];
                counts[j][dim] += prevCounts;
#elif MAX_LANE_WIDTH == 16
                counts[j][dim] = ShiftUp<1>(binCounts[j][dim]) + binCounts[j][dim];
                counts[j][dim] += ShiftUp<2>(counts[j][dim]) + counts[j][dim];
                counts[j][dim] += ShiftUp<4>(counts[j][dim]) + counts[j][dim];
                counts[j][dim] += prevCounts;
#endif
                prevCount = Shuffle<numBinsX - 1>(counts[j][dim]);

                LaneF32<numBinsX> extentX = binSumMaxX[dim] - binSumMinX[dim];
                LaneF32<numBinsX> extentY = binSumMaxZ[dim] - binSumMinY[dim];
                LaneF32<numBinsX> extentZ = binSumMaxY[dim] - binSumMinZ[dim];
                area[j][dim]              = FMA(extentX, extentY + extentZ, extentY * extentZ);
            }
        }
        for (u32 dim = 0; dim < 3; dim++)
        {
            binPrevMinX[dim] = pos_inf;
            binPrevMinY[dim] = pos_inf;
            binPrevMinZ[dim] = pos_inf;

            binPrevMaxX[dim] = neg_inf;
            binPrevMaxY[dim] = neg_inf;
            binPrevMaxZ[dim] = neg_inf;
        }

        LaneF32<numBinsX> lBestArea = pos_inf;
        LaneU32<numBinsX> lBestPos  = 0;
        LaneU32<numBinsX> lBestDIm  = 0;
        for (u32 j = numBinsY - 1; j >= 0; j--)
        {
            for (u32 dim = 0; dim < 3; dim++)
            {
                const LaneU32<numBinsX> d(dim);

                PrefixBackward<true>(binMinX[j][dim], binPrevMinX[dim], binSumMinX[j][dim]);
                PrefixBackward<true>(binMinY[j][dim], binPrevMinY[dim], binSumMinY[j][dim]);
                PrefixBackward<true>(binMinZ[j][dim], binPrevMinZ[dim], binSumMinZ[j][dim]);

                PrefixBackward<false>(binMaxX[j][dim], binPrevMaxX[dim], binSumMaxX[j][dim]);
                PrefixBackward<false>(binMaxY[j][dim], binPrevMaxY[dim], binSumMaxY[j][dim]);
                PrefixBackward<false>(binMaxZ[j][dim], binPrevMaxZ[dim], binSumMaxZ[j][dim]);
#if MAX_LANE_WIDTH == 4
                LaneF32<numBinsX> rCount = ShiftDown<1>(binCounts[j][dim]);
                rCount += ShiftDown<2>(rCount) + rCount;
                rCount += prevCount[dim];
#elif MAX_LANE_WIDTH == 8
                LaneF32<numBinsX> rCount = ShiftDown<1>(binCounts[j][dim]);
                rCount += ShiftDown<2>(rCount) + rCount;
                rCount += ShiftDown<4>(rCount) + rCount;
                rCount += prevCount[dim];
#elif MAX_LANE_WIDTH == 16
                LaneF32<numBinsX> rCount = ShiftDown<1>(binCounts[j][dim]);
                rCount += ShiftDown<2>(rCount) + rCount;
                rCount += ShiftDown<4>(rCount) + rCount;
                rCount += ShiftDown<8>(rCount) + rCount;
                rCount += prevCount[dim];
#endif
                prevCount[dim] = Shuffle<0>(rCount);

                LaneF32<numBinsX> extentX = binSumMaxX[dim] - binSumMinX[dim];
                LaneF32<numBinsX> extentY = binSumMaxZ[dim] - binSumMinY[dim];
                LaneF32<numBinsX> extentZ = binSumMaxY[dim] - binSumMinZ[dim];
                LaneF32<numBinsX> rArea   = FMA(extentX, extentY + extentZ, extentY * extentZ);

                LaneF32<numBinsX> sah = FMA(area[j][dim], counts[j][dim], rCount * rArea);

                lBestDim  = Select(sah < lBestArea, dim, lBestDim);
                lBestPos  = Select(sah < lBestArea, LaneU32<numBinsX>::Step(numBinsX * j), lBestPos);
                lBestArea = Select(sah < lBestArea, sah, lBestArea);
            }
        }

        f32 bestArea = pos_inf;
        u32 bestPos  = 0;
        u32 bestDim  = 0;
        for (u32 i = 0; i < numBinsX; i++)
        {
            if (lBestArea[i] < bestArea)
            {
                bestArea = lBestArea[i];
                bestPos  = lBestPos[i];
                bestDim  = lBestDim[i];
            }
        }
        return Split(bestArea, bestPos, bestDim);
    }
};
} // namespace rt

#endif
