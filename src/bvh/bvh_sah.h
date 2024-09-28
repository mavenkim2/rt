#ifndef BVH_SAH_H
#define BVH_SAH_H
#include "../math/simd_base.h"

// TODO:
// - spatial splits <- currently working on
// - explore octant order traversal,
// - partial rebraiding
// - stream (Fuetterling 2015, frusta)/packet traversal
// - ray sorting/binning(hyperion, or individually in each thread).
// - curves
//     - it seems that PBRT doesn't supported instanced curves, so the scene description files handle these weirdly.
//     look at converting these to instances?
// - support both BVH over all primitives and two level BVH.
//     - for BVH over all primitives, need polymorphism. will implement by indexing into an array of indices,
//     - can instances contain multiple types of primitives?
// - expand current BVH4 traversal code to BVH8

// crazier TODOs
// - stackless/stack compressed?
// - PLOC and agglomerative aggregate clustering
// - subdivision surfaces

// for the top level BVH, I think I can get rid of some of the meta fields maybe because the counts should be
// 1. probably can actually use 2 bits per node (node, invalid, leaf) for 2 bytes for 8-wide and 1 byte for 4-wide.
// not sure how much that would save or if it's even worth it, but something to consider.

namespace rt
{

const u32 PARALLEL_THRESHOLD = 4 * 1024;

// TODO: make sure to pad to max lane width, set min and maxes to pos_inf and neg_inf and count to zero for padded entries

template <i32 numBins>
struct HeuristicSAHBinnedAVX
{
    Lane8F32 bins[3][numBins];
    Lane4U32 binCounts[numBins];

    Lane8F32 scaleX;
    Lane8F32 scaleY;
    Lane8F32 scaleZ;

    Lane8F32 baseX;
    Lane8F32 baseY;
    Lane8F32 baseZ;

    Lane8F32 base;
    Lane8F32 scale;

    HeuristicSAHBinnedAVX() {}
    HeuristicSAHBinnedAVX(const AABB &base) { Init(Lane4F32(base.minP), Lane4F32(base.maxP)); }
    HeuristicSAHBinnedAVX(const Bounds &base) { Init(base.minP, base.maxP); }

    __forceinline void Init(const Lane4F32 inMin, const Lane4F32 inMax)
    {
        const Lane4F32 eps = 1e-34f;

        baseX               = Lane8F32(inMin[0]);
        baseY               = Lane8F32(inMin[1]);
        baseZ               = Lane8F32(inMin[2]);
        base                = Lane8F32(inMin, inMin);
        const Lane4F32 diag = Max(inMax - inMin, 0.f);

        Lane4F32 inScale = Select(diag > eps, Lane4F32((f32)numBins) / diag, Lane4F32(0.f));
        scaleX           = Lane8F32(inScale[0]);
        scaleY           = Lane8F32(inScale[1]);
        scaleZ           = Lane8F32(inScale[2]);
        scale            = Lane8F32(inScale, inScale);

        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 i = 0; i < numBins; i++)
            {
                bins[dim][i] = pos_inf;
            }
        }
        for (u32 i = 0; i < numBins; i++)
        {
            binCounts[i] = 0;
        }
    }

    __forceinline void BinTest(const PrimData *prims, u32 start, u32 count)
    {
        u32 i   = start;
        u32 end = start + count;

        u32 tempBinIndicesX[8];
        u32 tempBinIndicesY[8];
        u32 tempBinIndicesZ[8];

        Lane8F32 prev[8] = {
            prims[i].m256,
            prims[i + 1].m256,
            prims[i + 2].m256,
            prims[i + 3].m256,
            prims[i + 4].m256,
            prims[i + 5].m256,
            prims[i + 6].m256,
            prims[i + 7].m256,
        };
        Lane8F32 minX;
        Lane8F32 minY;
        Lane8F32 minZ;

        Lane8F32 maxX;
        Lane8F32 maxY;
        Lane8F32 maxZ;

        Transpose8x6(prev[0], prev[1], prev[2], prev[3],
                     prev[4], prev[5], prev[6], prev[7],
                     minX, minY, minZ, maxX, maxY, maxZ);

        const Lane8F32 centroidX0   = (minX - maxX) * 0.5f;
        const Lane8U32 binIndicesX0 = Clamp(Lane8U32(0), Lane8U32(numBins - 1), Flooru((centroidX0 - baseX) * scaleX));
        const Lane8F32 centroidY0   = (minY - maxY) * 0.5f;
        const Lane8U32 binIndicesY0 = Clamp(Lane8U32(0), Lane8U32(numBins - 1), Flooru((centroidY0 - baseY) * scaleY));
        const Lane8F32 centroidZ0   = (minZ - maxZ) * 0.5f;
        const Lane8U32 binIndicesZ0 = Clamp(Lane8U32(0), Lane8U32(numBins - 1), Flooru((centroidZ0 - baseZ) * scaleZ));

        i += 8;
        Lane8U32::Store(tempBinIndicesX, binIndicesX0);
        Lane8U32::Store(tempBinIndicesY, binIndicesY0);
        Lane8U32::Store(tempBinIndicesZ, binIndicesZ0);

        for (; i < end; i += 8)
        {
            const Lane8F32 prims8[8] = {
                prims[i].m256,
                prims[i + 1].m256,
                prims[i + 2].m256,
                prims[i + 3].m256,

                prims[i + 4].m256,
                prims[i + 5].m256,
                prims[i + 6].m256,
                prims[i + 7].m256};

            Transpose8x6(prims8[0], prims8[1], prims8[2], prims8[3],
                         prims8[4], prims8[5], prims8[6], prims8[7],
                         minX, minY, minZ, maxX, maxY, maxZ);

            const Lane8F32 centroidX = (minX - maxX) * 0.5f;

            u32 iX0 = tempBinIndicesX[0];
            u32 iY0 = tempBinIndicesY[0];
            u32 iZ0 = tempBinIndicesZ[0];
            binCounts[iX0][0]++;
            binCounts[iY0][1]++;
            binCounts[iZ0][2]++;
            bins[0][iX0] = Max(bins[0][iX0], prev[0]);
            bins[1][iY0] = Max(bins[1][iY0], prev[0]);
            bins[2][iZ0] = Max(bins[2][iZ0], prev[0]);
            prev[0]      = prims8[0];

            const Lane8U32 binIndicesX = Clamp(Lane8U32(0), Lane8U32(numBins - 1), Flooru((centroidX - baseX) * scaleX));

            u32 iX1 = tempBinIndicesX[1];
            u32 iY1 = tempBinIndicesY[1];
            u32 iZ1 = tempBinIndicesZ[1];
            binCounts[iX1][0]++;
            binCounts[iY1][1]++;
            binCounts[iZ1][2]++;
            bins[0][iX1] = Max(bins[0][iX1], prev[1]);
            bins[1][iY1] = Max(bins[1][iY1], prev[1]);
            bins[2][iZ1] = Max(bins[2][iZ1], prev[1]);
            prev[1]      = prims8[1];

            const Lane8F32 centroidY = (minY - maxY) * 0.5f;

            u32 iX2 = tempBinIndicesX[2];
            u32 iY2 = tempBinIndicesY[2];
            u32 iZ2 = tempBinIndicesZ[2];
            binCounts[iX2][0]++;
            binCounts[iY2][1]++;
            binCounts[iZ2][2]++;
            bins[0][iX2] = Max(bins[0][iX2], prev[2]);
            bins[1][iY2] = Max(bins[1][iY2], prev[2]);
            bins[2][iZ2] = Max(bins[2][iZ2], prev[2]);
            prev[2]      = prims8[2];

            const Lane8U32 binIndicesY = Clamp(Lane8U32(0), Lane8U32(numBins - 1), Flooru((centroidY - baseY) * scaleY));

            u32 iX3 = tempBinIndicesX[3];
            u32 iY3 = tempBinIndicesY[3];
            u32 iZ3 = tempBinIndicesZ[3];
            binCounts[iX3][0]++;
            binCounts[iY3][1]++;
            binCounts[iZ3][2]++;
            bins[0][iX3] = Max(bins[0][iX3], prev[3]);
            bins[1][iY3] = Max(bins[1][iY3], prev[3]);
            bins[2][iZ3] = Max(bins[2][iZ3], prev[3]);
            prev[3]      = prims8[3];

            const Lane8F32 centroidZ = (minZ - maxZ) * 0.5f;

            u32 iX4 = tempBinIndicesX[4];
            u32 iY4 = tempBinIndicesY[4];
            u32 iZ4 = tempBinIndicesZ[4];
            binCounts[iX4][0]++;
            binCounts[iY4][1]++;
            binCounts[iZ4][2]++;
            bins[0][iX4] = Max(bins[0][iX4], prev[4]);
            bins[1][iY4] = Max(bins[1][iY4], prev[4]);
            bins[2][iZ4] = Max(bins[2][iZ4], prev[4]);
            prev[4]      = prims8[4];

            const Lane8U32 binIndicesZ = Clamp(Lane8U32(0), Lane8U32(numBins - 1), Flooru((centroidZ - baseZ) * scaleZ));

            u32 iX5 = tempBinIndicesX[5];
            u32 iY5 = tempBinIndicesY[5];
            u32 iZ5 = tempBinIndicesZ[5];
            binCounts[iX5][0]++;
            binCounts[iY5][1]++;
            binCounts[iZ5][2]++;
            bins[0][iX5] = Max(bins[0][iX5], prev[5]);
            bins[1][iY5] = Max(bins[1][iY5], prev[5]);
            bins[2][iZ5] = Max(bins[2][iZ5], prev[5]);
            prev[5]      = prims8[5];

            u32 iX6 = tempBinIndicesX[6];
            u32 iY6 = tempBinIndicesY[6];
            u32 iZ6 = tempBinIndicesZ[6];
            binCounts[iX6][0]++;
            binCounts[iY6][1]++;
            binCounts[iZ6][2]++;
            bins[0][iX6] = Max(bins[0][iX6], prev[6]);
            bins[1][iY6] = Max(bins[1][iY6], prev[6]);
            bins[2][iZ6] = Max(bins[2][iZ6], prev[6]);
            prev[6]      = prims8[6];

            u32 iX7 = tempBinIndicesX[7];
            u32 iY7 = tempBinIndicesY[7];
            u32 iZ7 = tempBinIndicesZ[7];
            binCounts[iX7][0]++;
            binCounts[iY7][1]++;
            binCounts[iZ7][2]++;
            bins[0][iX7] = Max(bins[0][iX7], prev[7]);
            bins[1][iY7] = Max(bins[1][iY7], prev[7]);
            bins[2][iZ7] = Max(bins[2][iZ7], prev[7]);
            prev[7]      = prims8[7];

            Lane8U32::Store(tempBinIndicesX, binIndicesX);
            Lane8U32::Store(tempBinIndicesY, binIndicesY);
            Lane8U32::Store(tempBinIndicesZ, binIndicesZ);
        }
    }

    __forceinline void BinTest2(const PrimData *prims, u32 start, u32 count)
    {
        u32 i   = start;
        u32 end = start + count;

        u32 tempBinIndices[8];
        Lane8F32 prev0;
        Lane8F32 prev1;

        prev0 = prims[i].m256;
        prev1 = prims[i].m256;

        Lane8F32 binMin = Shuffle4<0, 2>(prev0, prev1);
        Lane8F32 binMax = Shuffle4<1, 3>(prev0, prev1);

        Lane8F32 centroid   = (binMax + binMin) * 0.5f;
        Lane8U32 binIndices = Clamp(Lane8U32(0), Lane8U32(numBins - 1), Flooru((centroid - base) * scale));

        Lane8U32::Store(tempBinIndices, binIndices);
        i += 2;

        for (; i < end - 1; i += 2)
        {
            const PrimData *prim0 = &prims[i];
            const PrimData *prim1 = &prims[i];

            binMin = Shuffle4<0, 2>(prim0->m256, prim1->m256);
            binMax = Shuffle4<1, 3>(prim0->m256, prim1->m256);

            centroid   = (binMax + binMin) * 0.5f;
            binIndices = Clamp(Lane8U32(0), Lane8U32(numBins - 1), Flooru((centroid - base) * scale));

            u32 binIndexX = tempBinIndices[0];
            u32 binIndexY = tempBinIndices[1];
            u32 binIndexZ = tempBinIndices[2];
            binCounts[binIndexX][0]++;
            binCounts[binIndexY][1]++;
            binCounts[binIndexZ][2]++;
            bins[0][binIndexX] = Max(bins[0][binIndexX], prev0);
            bins[1][binIndexY] = Max(bins[1][binIndexY], prev0);
            bins[2][binIndexZ] = Max(bins[2][binIndexZ], prev0);

            prev0 = prim0->m256;

            u32 binIndexX2 = tempBinIndices[4];
            u32 binIndexY2 = tempBinIndices[5];
            u32 binIndexZ2 = tempBinIndices[6];
            binCounts[binIndexX2][0]++;
            binCounts[binIndexY2][1]++;
            binCounts[binIndexZ2][2]++;
            bins[0][binIndexX2] = Max(bins[0][binIndexX2], prev1);
            bins[1][binIndexY2] = Max(bins[1][binIndexY2], prev1);
            bins[2][binIndexZ2] = Max(bins[2][binIndexZ2], prev1);

            prev1 = prim1->m256;

            Lane8U32::Store(tempBinIndices, binIndices);
        }
        u32 binIndexX = tempBinIndices[0];
        u32 binIndexY = tempBinIndices[1];
        u32 binIndexZ = tempBinIndices[2];
        binCounts[binIndexX][0]++;
        binCounts[binIndexY][1]++;
        binCounts[binIndexZ][2]++;
        bins[0][binIndexX] = Max(bins[0][binIndexX], prev0);
        bins[1][binIndexY] = Max(bins[1][binIndexY], prev0);
        bins[2][binIndexZ] = Max(bins[2][binIndexZ], prev0);

        u32 binIndexX2 = tempBinIndices[4];
        u32 binIndexY2 = tempBinIndices[5];
        u32 binIndexZ2 = tempBinIndices[6];
        binCounts[binIndexX2][0]++;
        binCounts[binIndexY2][1]++;
        binCounts[binIndexZ2][2]++;
        bins[0][binIndexX2] = Max(bins[0][binIndexX2], prev1);
        bins[1][binIndexY2] = Max(bins[1][binIndexY2], prev1);
        bins[2][binIndexZ2] = Max(bins[2][binIndexZ2], prev1);
    }
};

template <i32 numBins>
struct HeuristicSAHBinned
{
    StaticAssert(numBins >= 4, MoreThan4Bins);

    // static const u32 pow2NumBins = NextPowerOfTwo(numBins);
    // static const u32 numBinsX    = pow2NumBins < MAX_LANE_WIDTH ? pow2NumBins : MAX_LANE_WIDTH;
    // static const u32 binMask     = numBinsX - 1;
    // static const u32 binShift    = BsfConst(numBinsX);
    // static const u32 numBinsY    = (numBins + numBinsX - 1) / numBinsX; // MAX_LANE_WIDTH - 1) / MAX_LANE_WIDTH;

    Lane4F32 binMin[3][numBins];
    Lane4F32 binMax[3][numBins];
    Lane4U32 binCounts[numBins];

    Lane4F32 minP;
    Lane4F32 scale;

    HeuristicSAHBinned() {}
    HeuristicSAHBinned(const AABB &base) { Init(Lane4F32(base.minP), Lane4F32(base.maxP)); }
    HeuristicSAHBinned(const Bounds &base) { Init(base.minP, base.maxP); }

    __forceinline void Init(const Lane4F32 inMin, const Lane4F32 inMax)
    {
        const Lane4F32 eps  = 1e-34f;
        minP                = inMin;
        const Lane4F32 diag = Max(inMax - minP, 0.f);

        scale = Select(diag > eps, Lane4F32((f32)numBins) / diag, Lane4F32(0.f));

        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 i = 0; i < numBins; i++)
            {
                binMin[dim][i] = pos_inf;
                binMax[dim][i] = neg_inf;
            }
        }
        for (u32 i = 0; i < numBins; i++)
        {
            binCounts[i] = 0;
        }
    }

    __forceinline void Bin(const PrimData *prims, u32 start, u32 count)
    {
        u32 i   = start;
        u32 end = start + count;
        for (; i < end - 1; i += 2)
        {
            const PrimData *prim = &prims[i];
            // Load
            Lane4F32 prim0Min = prim->minP;
            Lane4F32 prim0Max = prim->maxP;

            Lane4F32 centroid   = (prim0Min + prim0Max) * 0.5f;
            Lane4U32 binIndices = Flooru((centroid - minP) * scale);
            binIndices          = Clamp(Lane4U32(0), Lane4U32(numBins - 1), binIndices);

            binMin[0][binIndices[0]] = Min(binMin[0][binIndices[0]], prim0Min);
            binMin[1][binIndices[1]] = Min(binMin[1][binIndices[1]], prim0Min);
            binMin[2][binIndices[2]] = Min(binMin[2][binIndices[2]], prim0Min);
            binMax[0][binIndices[0]] = Max(binMax[0][binIndices[0]], prim0Max);
            binMax[1][binIndices[1]] = Max(binMax[1][binIndices[1]], prim0Max);
            binMax[2][binIndices[2]] = Max(binMax[2][binIndices[2]], prim0Max);

            binCounts[binIndices[0]][0] += 1;
            binCounts[binIndices[1]][1] += 1;
            binCounts[binIndices[2]][2] += 1;

            const PrimData *prim1 = &prims[i + 1];
            Lane4F32 prim1Min     = prim1->minP;
            Lane4F32 prim1Max     = prim1->maxP;

            Lane4F32 centroid1   = (prim1Min + prim1Max) * 0.5f;
            Lane4U32 binIndices1 = Flooru((centroid1 - minP) * scale);
            binIndices1          = Clamp(Lane4U32(0), Lane4U32(numBins - 1), binIndices1);

            binMin[0][binIndices1[0]] = Min(binMin[0][binIndices1[0]], prim1Min);
            binMin[1][binIndices1[1]] = Min(binMin[1][binIndices1[1]], prim1Min);
            binMin[2][binIndices1[2]] = Min(binMin[2][binIndices1[2]], prim1Min);
            binMax[0][binIndices1[0]] = Max(binMax[0][binIndices1[0]], prim1Max);
            binMax[1][binIndices1[1]] = Max(binMax[1][binIndices1[1]], prim1Max);
            binMax[2][binIndices1[2]] = Max(binMax[2][binIndices1[2]], prim1Max);

            binCounts[binIndices1[0]][0] += 1;
            binCounts[binIndices1[1]][1] += 1;
            binCounts[binIndices1[2]][2] += 1;
        }
        if (i < end)
        {
            const PrimData *prim = &prims[i];
            Lane4F32 primMin     = prim->minP;
            Lane4F32 primMax     = prim->maxP;
            Lane4F32 centroid    = (primMin + primMax) * 0.5f;
            Lane4U32 binIndices  = Flooru((centroid - minP) * scale);
            binIndices           = Clamp(Lane4U32(0), Lane4U32(numBins - 1), binIndices);

            binMin[0][binIndices[0]] = Min(binMin[0][binIndices[0]], primMin);
            binMin[1][binIndices[1]] = Min(binMin[1][binIndices[1]], primMin);
            binMin[2][binIndices[2]] = Min(binMin[2][binIndices[2]], primMin);
            binMax[0][binIndices[0]] = Max(binMax[0][binIndices[0]], primMax);
            binMax[1][binIndices[1]] = Max(binMax[1][binIndices[1]], primMax);
            binMax[2][binIndices[2]] = Max(binMax[2][binIndices[2]], primMax);

            binCounts[binIndices[0]][0] += 1;
            binCounts[binIndices[1]][1] += 1;
            binCounts[binIndices[2]][2] += 1;
        }
    }

    __forceinline void Merge(const HeuristicSAHBinned<numBins> &other)
    {
        for (u32 i = 0; i < numBins; i++)
        {
            binCounts[i] += other.binCounts[i];
            for (u32 dim = 0; dim < 3; dim++)
            {
                binMin[dim][i] = Min(other.binMin[dim][i], binMin[dim][i]);
                binMax[dim][i] = Max(other.binMax[dim][i], binMax[dim][i]);
            }
        }
    }

    // NOTE: 1 << blockShift is the block size (e.g. intersecting groups of 4 triangles -> blockShift = 2)
    __forceinline Split Best(const u32 blockShift)
    {
        Lane4F32 minDimX = pos_inf;
        Lane4F32 minDimY = pos_inf;
        Lane4F32 minDimZ = pos_inf;

        Lane4F32 maxDimX = neg_inf;
        Lane4F32 maxDimY = neg_inf;
        Lane4F32 maxDimZ = neg_inf;

        Lane4U32 count = 0;
        Lane4U32 lCounts[numBins];
        Lane4F32 area[numBins] = {};

        const u32 blockAdd = (1 << blockShift) - 1;

        for (u32 i = 0; i < numBins - 1; i++)
        {
            count += binCounts[i];
            lCounts[i] = count;

            minDimX = Min(minDimX, binMin[0][i]);
            minDimY = Min(minDimY, binMin[1][i]);
            minDimZ = Min(minDimZ, binMin[2][i]);

            maxDimX = Max(maxDimX, binMax[0][i]);
            maxDimY = Max(maxDimY, binMax[1][i]);
            maxDimZ = Max(maxDimZ, binMax[2][i]);

            Lane4F32 minX, minY, minZ;
            Lane4F32 maxX, maxY, maxZ;
            Transpose3x3(minDimX, minDimY, minDimZ, minX, minY, minZ);
            Transpose3x3(maxDimX, maxDimY, maxDimZ, maxX, maxY, maxZ);

            Lane4F32 extentX = maxX - minX;
            Lane4F32 extentY = maxY - minY;
            Lane4F32 extentZ = maxZ - minZ;

            area[i] = FMA(extentX, extentY + extentZ, extentY * extentZ);
        }

        count = 0;

        minDimX = pos_inf;
        minDimY = pos_inf;
        minDimZ = pos_inf;

        maxDimX = neg_inf;
        maxDimY = neg_inf;
        maxDimZ = neg_inf;

        Lane4F32 lBestSAH = pos_inf;
        Lane4U32 lBestPos = 0;
        for (u32 i = numBins - 1; i >= 1; i--)
        {
            count += binCounts[i];

            minDimX = Min(minDimX, binMin[0][i]);
            minDimY = Min(minDimY, binMin[1][i]);
            minDimZ = Min(minDimZ, binMin[2][i]);

            maxDimX = Max(maxDimX, binMax[0][i]);
            maxDimY = Max(maxDimY, binMax[1][i]);
            maxDimZ = Max(maxDimZ, binMax[2][i]);

            Lane4F32 minX, minY, minZ;
            Lane4F32 maxX, maxY, maxZ;
            Transpose3x3(minDimX, minDimY, minDimZ, minX, minY, minZ);
            Transpose3x3(maxDimX, maxDimY, maxDimZ, maxX, maxY, maxZ);

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
            if (scale[dim] == 0.f) continue;

            if (lBestSAH[dim] < bestArea)
            {
                bestArea = lBestSAH[dim];
                bestPos  = lBestPos[dim];
                bestDim  = dim;
            }
        }
        f32 bestValue = (bestPos + 1) / (scale[bestDim]) + minP[bestDim];
        return Split(bestArea, bestPos, bestDim, bestValue);
    }
};

Split BinParallel(const Record &record, u32 blockSize = 1, HeuristicSAHBinned<32> *out = 0)
{
    Assert(IsPow2(blockSize));
    u32 blockShift = Bsf(blockSize);
    u32 total      = record.end - record.start;
    if (total < PARALLEL_THRESHOLD)
    {
        HeuristicSAHBinned<32> heuristic(record.centBounds);
        heuristic.Bin(record.data, record.start, total);
        if (out)
        {
            *out = heuristic;
        }
        return heuristic.Best(blockShift);
    }

    u32 taskSize                  = 1024;
    HeuristicSAHBinned<32> result = jobsystem::ParallelReduce<HeuristicSAHBinned<32>>(
        total, taskSize,
        [&](HeuristicSAHBinned<32> &bin, u32 start, u32 count) { bin.Bin(record.data, start, count); },
        [&](HeuristicSAHBinned<32> &a, const HeuristicSAHBinned<32> &b) { a.Merge(b); },
        record.centBounds);

    if (out)
    {
        *out = result;
    }
    return result.Best(blockShift);
}

__forceinline u32 ClipTriangle(const Vec3f &pA, const Vec3f &pB, const Vec3f &pC, const u32 dim,
                               const f32 clipPosA, const f32 clipPosB, Bounds &outBounds)
{
    u32 invalid = 0;

    Bounds out;

    Lane4F32 p0 = Lane4F32(pA);
    Lane4F32 p1 = Lane4F32(pB);
    Lane4F32 p2 = Lane4F32(pC);

    Lane4F32 edge0 = p1 - p0;

    f32 tA0 = (clipPosA - p0[dim]) / edge0[dim];
    f32 tB0 = (clipPosB - p0[dim]) / edge0[dim];

    invalid = invalid | ((tA0 > 1.f) && (tB0 > 1.f));
    invalid = invalid | (((tA0 < 0.f) && (tB0 < 0.f)) << 1);

    tA0 = Clamp(0.f, 1.f, tA0);
    tB0 = Clamp(0.f, 1.f, tB0);

    Lane4F32 clippedPointA0 = FMA(edge0, tA0, p0);
    Lane4F32 clippedPointB0 = FMA(edge0, tB0, p0);

    out.Extend(clippedPointA0);
    out.Extend(clippedPointB0);

    Lane4F32 edge1 = p2 - p1;

    f32 tA1 = (clipPosA - p1[dim]) / edge1[dim];
    f32 tB1 = (clipPosB - p1[dim]) / edge1[dim];

    invalid = invalid | ((tA1 > 1.f) && (tB1 > 1.f));
    invalid = invalid | (((tA1 < 0.f) && (tB1 < 0.f)) << 1);

    tA1 = Clamp(0.f, 1.f, tA1);
    tB1 = Clamp(0.f, 1.f, tB1);

    Lane4F32 clippedPointA1 = FMA(edge1, tA1, p1);
    Lane4F32 clippedPointB1 = FMA(edge1, tB1, p1);

    out.Extend(clippedPointA1);
    out.Extend(clippedPointB1);

    Lane4F32 edge2 = p0 - p2;

    f32 tA2 = (clipPosA - p2[dim]) / edge2[dim];
    f32 tB2 = (clipPosB - p2[dim]) / edge2[dim];

    invalid = invalid | ((tA2 > 1.f) && (tB2 > 1.f));
    invalid = invalid | (((tA2 < 0.f) && (tB2 < 0.f)) << 1);

    tA2 = Clamp(0.f, 1.f, tA2);
    tB2 = Clamp(0.f, 1.f, tB2);

    Lane4F32 clippedPointA2 = FMA(edge2, tA2, p2);
    Lane4F32 clippedPointB2 = FMA(edge2, tB2, p2);

    out.Extend(clippedPointA2);
    out.Extend(clippedPointB2);

    outBounds.minP = Select(Lane4F32((bool)invalid), Lane4F32(pos_inf), out.minP);
    outBounds.maxP = Select(Lane4F32((bool)invalid), Lane4F32(neg_inf), out.maxP);
    return 0;
}

__forceinline void Swap(const Lane8F32 &mask, Lane8F32 &a, Lane8F32 &b)
{
    Lane8F32 temp = a;
    a             = _mm256_blendv_ps(a, b, mask);
    b             = _mm256_blendv_ps(b, temp, mask);
}

__forceinline void ClipTriangleSimple(const TriangleMesh *mesh, const Bounds &bounds, const u32 faceIndex,
                                      const u32 dim, const f32 clipPos, Bounds &l, Bounds &r)
{
    Bounds left;
    Bounds right;
    /* clip triangle to left and right box by processing all edges */

    Lane4F32 v[] = {Lane4F32(mesh->p[mesh->indices[faceIndex * 3 + 0]]), Lane4F32(mesh->p[mesh->indices[faceIndex * 3 + 1]]),
                    Lane4F32(mesh->p[mesh->indices[faceIndex * 3 + 2]]), Lane4F32(mesh->p[mesh->indices[faceIndex * 3 + 0]])};

    for (size_t i = 0; i < 4; i++)
    {
        const Lane4F32 &v0 = v[i];
        const Lane4F32 &v1 = v[i + 1];
        const float v0d    = v0[dim];
        const float v1d    = v1[dim];

        if (v0d <= clipPos) left.Extend(v0);  // this point is on left side
        if (v0d >= clipPos) right.Extend(v0); // this point is on right side

        if ((v0d < clipPos && clipPos < v1d) || (v1d < clipPos && clipPos < v0d)) // the edge crosses the splitting location
        {
            Assert((v1d - v0d) != 0.0f);
            // f32 eps                = 1e-34f;
            f32 div                = Rcp(v1d - v0d);
            const float inv_length = v1d == v0d ? 0.f : div;
            const Lane4F32 c       = FMA((clipPos - v0d) * inv_length, v1 - v0, v0);
            left.Extend(c);
            right.Extend(c);
        }
    }

    // l       = left;
    // r       = right;
    l = Intersect(left, bounds);
    r = Intersect(right, bounds);
    // left_o = intersect(left, bounds);
    // r      = intersect(right, bounds);
}

static const Lane8F32 signFlipMask(-0.f, -0.f, -0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
static const u32 LUTAxis[] = {1, 2, 0};

struct ExtRange
{
    PrimData *data;
    u32 start;
    u32 count;  // number allocated
    u32 extEnd; // allocation end

    ExtRange(PrimData *data, u32 start, u32 count, u32 extEnd)
        : data(data), start(start), count(count), extEnd(extEnd) {}

    __forceinline u32 End() const { return start + count; }
    __forceinline u32 ExtSize() const { return extEnd - (start + count); }
    __forceinline u32 TotalSize() const { return extEnd - start; }
};

struct ExtRangeRef
{
    PrimRef *data;
    u32 start;
    u32 count;  // number allocated
    u32 extEnd; // allocation end

    ExtRangeRef(PrimRef *data, u32 start, u32 count, u32 extEnd)
        : data(data), start(start), count(count), extEnd(extEnd) {}

    __forceinline u32 End() const { return start + count; }
    __forceinline u32 ExtSize() const { return extEnd - (start + count); }
    __forceinline u32 TotalSize() const { return extEnd - start; }
};

template <i32 N>
struct QuantizedNode;

#define CREATE_NODE() __forceinline void operator()(const Record *records, const u32 numRecords, NodeType *result)
#define CREATE_LEAF() __forceinline void operator()(NodeType *leaf, PrimData *prims, u32 start, u32 end)
#define UPDATE_NODE() __forceinline void operator()()

template <i32 N>
struct QuantizedNode
{
    StaticAssert(N == 4 || N == 8, NMustBe4Or8);

    // TODO: the bottom 4 bits can be used for something (and maybe the top 7 bits too)
    QuantizedNode<N> *internalOffset;
    u8 lowerX[N];
    u8 lowerY[N];
    u8 lowerZ[N];

    // NOTE: upperX = 255 when node is invalid
    u8 upperX[N];
    u8 upperY[N];
    u8 upperZ[N];

    Vec3f minP;
    u32 leafOffset;
    u8 scale[3];

    // TODO: the last 5 bytes can be used for something
    u8 meta[N >> 2];
    u8 _pad[N == 8 ? 3 : 4];
};

template <i32 N>
struct CreateQuantizedNode
{
    using NodeType = QuantizedNode<N>;

    CREATE_NODE()
    {
        const f32 MIN_QUAN = 0.f;
        const f32 MAX_QUAN = 255.f;
        Lane4F32 boundsMinP(pos_inf);
        Lane4F32 boundsMaxP(neg_inf);

        for (u32 i = 0; i < numRecords; i++)
        {
            boundsMinP = Min(boundsMinP, records[i].geomBounds.minP);
            boundsMaxP = Max(boundsMaxP, records[i].geomBounds.maxP);
        }
        result->minP = ToVec3f(boundsMinP);

        Lane4F32 diff = boundsMaxP - boundsMinP;

        f32 expX = Ceil(Log2f(diff[0] / 255.f));
        f32 expY = Ceil(Log2f(diff[1] / 255.f));
        f32 expZ = Ceil(Log2f(diff[2] / 255.f));

        Lane4U32 shift = Flooru(Lane4F32(expX, expY, expZ, 0.f)) + 127;

        Lane4F32 pow = AsFloat(shift << 23);

        Vec3lf<N> powVec;
        // TODO: for N = 8, this needs to be shuffle across
        powVec.x = Shuffle<0>(pow);
        powVec.y = Shuffle<1>(pow);
        powVec.z = Shuffle<2>(pow);

        Assert(numRecords <= N);
        Vec3lf<N> min;
        Vec3lf<N> max;

        if constexpr (N == 4)
        {
            LaneF32<N> min02xy = UnpackLo(records[0].geomBounds.minP, records[2].geomBounds.minP);
            LaneF32<N> min13xy = UnpackLo(records[1].geomBounds.minP, records[3].geomBounds.minP);

            LaneF32<N> min02z_ = UnpackHi(records[0].geomBounds.minP, records[2].geomBounds.minP);
            LaneF32<N> min13z_ = UnpackHi(records[1].geomBounds.minP, records[3].geomBounds.minP);

            LaneF32<N> max02xy = UnpackLo(records[0].geomBounds.maxP, records[2].geomBounds.maxP);
            LaneF32<N> max13xy = UnpackLo(records[1].geomBounds.maxP, records[3].geomBounds.maxP);

            LaneF32<N> max02z_ = UnpackHi(records[0].geomBounds.maxP, records[2].geomBounds.maxP);
            LaneF32<N> max13z_ = UnpackHi(records[1].geomBounds.maxP, records[3].geomBounds.maxP);

            min.x = UnpackLo(min02xy, min13xy);
            min.y = UnpackHi(min02xy, min13xy);
            min.z = UnpackLo(min02z_, min13z_);

            max.x = UnpackLo(max02xy, max13xy);
            max.y = UnpackHi(max02xy, max13xy);
            max.z = UnpackLo(max02z_, max13z_);
        }
        else if constexpr (N == 8)
        {
            LaneF32<N> min04(records[0].geomBounds.minP, records[4].geomBounds.minP);
            LaneF32<N> min26(records[2].geomBounds.minP, records[6].geomBounds.minP);

            LaneF32<N> min15(records[1].geomBounds.minP, records[5].geomBounds.minP);
            LaneF32<N> min37(records[3].geomBounds.minP, records[7].geomBounds.minP);

            LaneF32<N> max04(records[0].geomBounds.maxP, records[4].geomBounds.maxP);
            LaneF32<N> max26(records[2].geomBounds.maxP, records[6].geomBounds.maxP);

            LaneF32<N> max15(records[1].geomBounds.maxP, records[5].geomBounds.maxP);
            LaneF32<N> max37(records[3].geomBounds.maxP, records[7].geomBounds.maxP);

            // x0 x2 y0 y2 x4 x6 y4 y6
            // x1 x3 y1 y3 x5 x7 y5 y7

            // z0 z2 _0 _2 z4 z6 _4 _6
            // z1 z3 _1 _3 z5 z7 _5 _7

            LaneF32<N> min0246xy = UnpackLo(min04, min26);
            LaneF32<N> min1357xy = UnpackLo(min15, min37);
            min.x                = UnpackLo(min0246xy, min1357xy);
            min.y                = UnpackHi(min0246xy, min1357xy);
            min.z                = UnpackLo(UnpackHi(min04, min26), UnpackHi(min15, min37));

            LaneF32<N> max0246xy = UnpackLo(max04, max26);
            LaneF32<N> max1357xy = UnpackLo(max15, max37);
            max.x                = UnpackLo(max0246xy, max1357xy);
            max.y                = UnpackHi(max0246xy, max1357xy);
            max.z                = UnpackLo(UnpackHi(max04, max26), UnpackHi(max15, max37));
        }

        Vec3lf<N> nodeMin;
        nodeMin.x = Shuffle<0>(boundsMinP);
        nodeMin.y = Shuffle<1>(boundsMinP);
        nodeMin.z = Shuffle<2>(boundsMinP);

        Vec3lf<N> qNodeMin = Floor((min - nodeMin) / powVec);
        Vec3lf<N> qNodeMax = Ceil((max - nodeMin) / powVec);

        Lane4F32 maskMinX = FMA(powVec.x, qNodeMin.x, nodeMin.x) > min.x;
        TruncateToU8(result->lowerX, Max(Select(maskMinX, qNodeMin.x - 1, qNodeMin.x), MIN_QUAN));
        Lane4F32 maskMinY = FMA(powVec.y, qNodeMin.y, nodeMin.y) > min.y;
        TruncateToU8(result->lowerY, Max(Select(maskMinY, qNodeMin.y - 1, qNodeMin.y), MIN_QUAN));
        Lane4F32 maskMinZ = FMA(powVec.z, qNodeMin.z, nodeMin.z) > min.z;
        TruncateToU8(result->lowerZ, Max(Select(maskMinZ, qNodeMin.z - 1, qNodeMin.z), MIN_QUAN));

        Lane4F32 maskMaxX = FMA(powVec.x, qNodeMax.x, nodeMin.x) < max.x;
        TruncateToU8(result->upperX, Min(Select(maskMaxX, qNodeMax.x + 1, qNodeMax.x), MAX_QUAN));
        Lane4F32 maskMaxY = FMA(powVec.y, qNodeMax.y, nodeMin.y) < max.y;
        TruncateToU8(result->upperY, Min(Select(maskMaxY, qNodeMax.y + 1, qNodeMax.y), MAX_QUAN));
        Lane4F32 maskMaxZ = FMA(powVec.z, qNodeMax.z, nodeMin.z) < max.z;
        TruncateToU8(result->upperZ, Min(Select(maskMaxZ, qNodeMax.z + 1, qNodeMax.z), MAX_QUAN));

        Assert(shift[0] <= 255 && shift[1] <= 255 && shift[2] <= 255);
        result->scale[0] = (u8)shift[0];
        result->scale[1] = (u8)shift[1];
        result->scale[2] = (u8)shift[2];
    }
};

template <i32 N>
struct UpdateQuantizedNode;

template <i32 N>
struct UpdateQuantizedNode
{
    using NodeType = QuantizedNode<N>;
    __forceinline void operator()(NodeType *parent, const Record *records, NodeType *children,
                                  const u32 *leafIndices, const u32 leafCount)
    {
        // NOTE: for leaves, top 3 bits represent binary count. bottom 5 bits represent offset from base offset.
        // 0 denotes a node, 1 denotes invalid.

        parent->internalOffset = children;
        for (u32 i = 0; i < N >> 2; i++)
        {
            parent->meta[i] = 0;
        }
        for (u32 i = 0; i < leafCount; i++)
        {
            parent->leafOffset   = records[leafIndices[0]].start;
            u32 leafIndex        = leafIndices[i];
            const Record *record = &records[leafIndex];
            u32 primCount        = record->end - record->start;
            u32 leafOffset       = record->start - parent->leafOffset;

            Assert(primCount >= 1 && primCount <= 3);
            Assert(leafOffset < 24);

            parent->meta[leafIndex >> 2] |= primCount << ((leafIndex & 3) << 1);
        }
    }
};

// NOTE: ptr MUST be aligned to 16 bytes, bottom 4 bits store the type, top 7 bits store the count

template <i32 N>
struct AABBNode
{
    struct Create
    {
        using NodeType = AABBNode;
        CREATE_NODE()
        {
        }
    };

    LaneF32<N> lowerX;
    LaneF32<N> lowerY;
    LaneF32<N> lowerZ;

    LaneF32<N> upperX;
    LaneF32<N> upperY;
    LaneF32<N> upperZ;
};

template <i32 N, typename NodeType>
struct BVHN
{
    BVHN() {}
    NodeType *root;
};

template <i32 N>
using QuantizedNode4 = QuantizedNode<4>;
template <i32 N>
using QuantizedNode8 = QuantizedNode<8>;

template <i32 N>
using BVHQuantized = BVHN<N, QuantizedNode<N>>;
typedef BVHQuantized<4> BVH4Quantized;
typedef BVHQuantized<8> BVH8Quantized;

template <i32 N, typename CreateNode, typename UpdateNode, typename Prim>
struct BuildFuncs
{
    using NodeType      = typename CreateNode::NodeType;
    using Primitive     = Prim;
    using BasePrimitive = Prim;

    CreateNode createNode;
    UpdateNode updateNode;
};

template <i32 N> //, typename CreateNode, typename UpdateNode, typename Prim>
struct BuildFuncs<N, CreateQuantizedNode<N>, UpdateQuantizedNode<N>, TriangleMesh>
{
    using NodeType      = typename CreateQuantizedNode<N>::NodeType;
    using Primitive     = TriangleMesh;
    using BasePrimitive = Triangle;

    CreateQuantizedNode<N> createNode;
    UpdateQuantizedNode<N> updateNode;
};

template <i32 N>
using BLAS_QuantizedNode_TriangleLeaf_Funcs =
    BuildFuncs<
        N,
        CreateQuantizedNode<N>,
        UpdateQuantizedNode<N>,
        TriangleMesh>;

// template <i32 N>
// using QuantizedInstanceFuncs =
//     BuildFunctions<
//         N,
//         typename CreateQuantizedNodeQuantizedNode<N>::Create,
//         typename CreateInstanceLeaf,
//         typename QuantizedNode<N>::Update>;

template <i32 N, typename BuildFunctions>
struct BVHBuilder
{
    using NodeType      = typename BuildFunctions::NodeType;
    using Primitive     = typename BuildFunctions::Primitive;
    using BasePrimitive = typename BuildFunctions::BasePrimitive;
    // BuildFunctions funcs;
    BuildFunctions f;

    Arena **arenas;
    Primitive *primitives;

    NodeType *BuildBVHRoot(BuildSettings settings, Record &record);
    __forceinline u32 BuildNode(BuildSettings settings, const Record &record, Record *childRecords, u32 &numChildren);
    void BuildBVH(BuildSettings settings, NodeType *parent, Record *records, u32 numChildren);

    BVHN<N, NodeType> BuildBVH(BuildSettings settings, Arena **inArenas, Primitive *inRawPrims, u32 count);
};

template <i32 N>
using BVHBuilderTriangleMesh = BVHBuilder<N, BLAS_QuantizedNode_TriangleLeaf_Funcs<N>>;

// template <i32 N>
// struct PolyLeaf
// {
//     LaneU32<N> indices;
// };

// template <>
// struct PolyLeaf<1>
// {
// };

// NOTE: for BVHs where the leaves are primitives containing any intersection type. not sure if this will actually exist
// CREATE_LEAF(CreatePolyLeaf)
// {
//     switch ((u32)prim.minP_geomID[3])
//     {
//         case PrimitiveType_Triangle:
//         {
//             u32 count              = end - start;
//             Triangle<N> *triangles = PushArray(arena, Triangle<N>, (end - start + N - 1) / N);
//             for (u32 i = start; i < end; i++)
//             {
//                 triangles[i].Fill(scene, prim[i], start, end);
//             }
//         }
//         break;
//         default:
//         {
//             Assert(!"Not implemented");
//         }
//     }
// }

template <i32 N, typename BuildFunctions>
typename BVHBuilder<N, BuildFunctions>::NodeType *BVHBuilder<N, BuildFunctions>::BuildBVHRoot(BuildSettings settings, Record &record)
{
    Record childRecords[N];
    u32 numChildren;
    u32 result = BuildNode(settings, record, childRecords, numChildren);

    NodeType *root = PushStruct(arenas[GetThreadIndex()], NodeType);
    if (result)
    {
        f.createNode(childRecords, numChildren, root);
        BuildBVH(settings, root, childRecords, numChildren);
    }
    // If the root is a leaf
    else
    {
        u32 leafIndex = 0;
        f.createNode(&record, 1, root);
        f.updateNode(root, &record, 0, &leafIndex, 1);
    }
    return root;
}

template <i32 N, typename BuildFunctions>
__forceinline u32 BVHBuilder<N, BuildFunctions>::BuildNode(BuildSettings settings, const Record &record,
                                                           Record *childRecords, u32 &numChildren)

{
    u32 total = record.end - record.start;
    Assert(total > 0);

    if (total == 1) return 0;

    HeuristicSAHBinned<32> heuristic;

    {
        Split split = BinParallel(record, 1, &heuristic);
        PartitionResult result;

        PartitionParallel(split, /*primitives, */ record, &result);

        // NOTE: multiply both by the area instead of dividing
        f32 area     = HalfArea(record.geomBounds);
        f32 leafSAH  = settings.intCost * area * total; //((total + (1 << settings.logBlockSize) - 1) >> settings.logBlockSize);
        f32 splitSAH = settings.travCost * area + settings.intCost * split.bestSAH;

        if (total <= settings.maxLeafSize && leafSAH <= splitSAH) return 0;

        childRecords[0] = Record(record.data, result.geomBoundsL, result.centBoundsL, record.start, result.mid);
        childRecords[1] = Record(record.data, result.geomBoundsR, result.centBoundsR, result.mid, record.end);
    }

    // N - 1 splits produces N children
    for (numChildren = 2; numChildren < N; numChildren++)
    {
        i32 bestChild = -1;
        f32 maxArea   = neg_inf;
        for (u32 recordIndex = 0; recordIndex < numChildren; recordIndex++)
        {
            Record &childRecord = childRecords[recordIndex];
            if (childRecord.Size() <= settings.maxLeafSize) continue;

            f32 childArea = HalfArea(childRecord.geomBounds);
            if (childArea > maxArea)
            {
                bestChild = recordIndex;
                maxArea   = childArea;
            }
        }
        if (bestChild == -1) break;

        Split split = BinParallel(childRecords[bestChild], 1, &heuristic);

        PartitionResult result;
        PartitionParallel(split, /*primitives, */ childRecords[bestChild], &result);

        Record &childRecord = childRecords[bestChild];
        PrimData *prim      = childRecord.data;
        u32 start           = childRecord.start;
        u32 end             = childRecord.end;

        Assert(start != end && start != result.mid && result.mid != end);

        childRecords[bestChild]   = Record(prim, result.geomBoundsL, result.centBoundsL, start, result.mid);
        childRecords[numChildren] = Record(prim, result.geomBoundsR, result.centBoundsR, result.mid, end);
    }

    // Test
    // for (u32 i = 0; i < numChildren; i++)
    // {
    //     Lane4F32 maskMin(False);
    //     Lane4F32 maskMax(False);
    //     Bounds test;
    //     Record &childRecord = childRecords[i];
    //     for (u32 j = childRecord.start; j < childRecord.end; j++)
    //     {
    //         AABB bounds = Primitive::Bounds(primitives, j);
    //
    //         Bounds b;
    //         b.minP = Lane4F32(bounds.minP);
    //         b.maxP = Lane4F32(bounds.maxP);
    //
    //         Assert(((Movemask(record.data[j].minP == b.minP) & 0x7) == 0x7) &&
    //                ((Movemask(record.data[j].maxP == b.maxP) & 0x7) == 0x7));
    //
    //         test.Extend(b);
    //         maskMin = maskMin | (b.minP == childRecord.geomBounds.minP);
    //         maskMax = maskMax | (b.maxP == childRecord.geomBounds.maxP);
    //         Assert(childRecord.geomBounds.Contains(b));
    //     }
    //     u32 minBits = Movemask(maskMin);
    //     u32 maxBits = Movemask(maskMax);
    //     Assert(((minBits & 0x7) == 0x7) && ((maxBits & 0x7) == 0x7));
    // }
    return 1;
}

template <i32 N, typename BuildFunctions>
void BVHBuilder<N, BuildFunctions>::BuildBVH(BuildSettings settings, NodeType *parent, Record *records, u32 inNumChildren)
{
    Record allChildRecords[N][N];
    u32 allNumChildren[N];

    u32 childNodeIndices[N];
    u32 childLeafIndices[N];
    u32 nodeCount = 0;
    u32 leafCount = 0;

    Assert(inNumChildren <= N);

    for (u32 childIndex = 0; childIndex < inNumChildren; childIndex++)
    {
        Record *childRecords = allChildRecords[childIndex];
        Record &record       = records[childIndex];
        u32 &numChildren     = allNumChildren[childIndex];
        u32 result           = BuildNode(settings, record, childRecords, numChildren);

        childNodeIndices[nodeCount] = childIndex;
        childLeafIndices[leafCount] = childIndex;
        nodeCount += result;
        leafCount += !result;
    }

    NodeType *children = PushArray(arenas[GetThreadIndex()], NodeType, nodeCount);

    // TODO: need to partition again so that the ranges pointed to by the leaves are consecutive in memory
    // TODO: consider whether we actually want to do this
    if (leafCount > 0)
    {
        u32 offset = pos_inf;
        u32 end    = neg_inf;

        TempArena temp = ScratchStart(0, 0);
        for (u32 i = 0; i < inNumChildren; i++)
        {
            offset = Min(offset, records[i].start);
            end    = Max(end, records[i].end);
        }

        u32 totalInternalNodeRange = 0;
        u32 totalLeafNodeRange     = 0;
        for (u32 i = 0; i < nodeCount; i++)
        {
            u32 internalNodeIndex = childNodeIndices[i];
            Record &record        = records[internalNodeIndex];
            totalInternalNodeRange += record.Size();
        }

        for (u32 i = 0; i < leafCount; i++)
        {
            u32 internalLeafIndex = childLeafIndices[i];
            Record &record        = records[internalLeafIndex];
            totalLeafNodeRange += record.Size();
        }

        Assert(totalLeafNodeRange + totalInternalNodeRange == end - offset);
        struct Range
        {
            u32 start;
            u32 end;
        };

        // TODO
        BasePrimitive *temporaryInternalNodeStorage = PushArray(temp.arena, BasePrimitive, totalInternalNodeRange);
        PrimData *tempInternalPrimData              = PushArray(temp.arena, PrimData, totalInternalNodeRange);
        Range *newInternalNodeRanges                = PushArray(temp.arena, Range, nodeCount);

        BasePrimitive *temporaryLeafNodeStorage = PushArray(temp.arena, BasePrimitive, totalLeafNodeRange);
        PrimData *tempLeafPrimData              = PushArray(temp.arena, PrimData, totalLeafNodeRange);
        Range *newLeafNodeRanges                = PushArray(temp.arena, Range, leafCount);

        u32 nodeOffset = 0;
        u32 leafOffset = 0;
        for (u32 i = 0; i < nodeCount; i++)
        {
            u32 internalNodeIndex = childNodeIndices[i];
            Record &record        = records[internalNodeIndex];

            newInternalNodeRanges[i].start = nodeOffset;
            newInternalNodeRanges[i].end   = nodeOffset + record.Size();
            for (u32 j = record.start; j < record.end; j++)
            {
                temporaryInternalNodeStorage[nodeOffset] = GetPrimitive<Primitive, BasePrimitive>(primitives, j);
                tempInternalPrimData[nodeOffset]         = record.data[j];
                nodeOffset++;
            }
        }
        for (u32 i = 0; i < leafCount; i++)
        {
            u32 leafNodeIndex = childLeafIndices[i];
            Record &record    = records[leafNodeIndex];

            newLeafNodeRanges[i].start = leafOffset;
            newLeafNodeRanges[i].end   = leafOffset + record.Size();
            for (u32 j = record.start; j < record.end; j++)
            {
                temporaryLeafNodeStorage[leafOffset] = GetPrimitive<Primitive, BasePrimitive>(primitives, j);
                tempLeafPrimData[leafOffset]         = record.data[j];
                leafOffset++;
            }
        }

        for (u32 i = 0; i < leafCount; i++)
        {
            u32 index           = childLeafIndices[i];
            Record &childRecord = records[index];
            Range *range        = &newLeafNodeRanges[i];
            childRecord.start   = offset;
            for (u32 j = range->start; j < range->end; j++)
            {
                StorePrimitive(primitives, temporaryLeafNodeStorage[j], offset);
                childRecord.data[offset] = tempLeafPrimData[j];
                offset++;
            }
            childRecord.end = offset;
        }
        for (u32 i = 0; i < nodeCount; i++)
        {
            u32 index           = childNodeIndices[i];
            Record &childRecord = records[index];
            Range *range        = &newInternalNodeRanges[i];
            u32 newStart        = offset;
            for (u32 j = range->start; j < range->end; j++)
            {
                StorePrimitive(primitives, temporaryInternalNodeStorage[j], offset);
                childRecord.data[offset] = tempInternalPrimData[j];
                offset++;
            }
            u32 newEnd = offset;
            for (u32 grandChildIndex = 0; grandChildIndex < allNumChildren[index]; grandChildIndex++)
            {
                Record &grandChildRecord = allChildRecords[index][grandChildIndex];
                i32 shift                = (i32)newStart - (i32)childRecord.start;
                grandChildRecord.start += shift;
                grandChildRecord.end += shift;
            }
            childRecord.start = newStart;
            childRecord.end   = newEnd;
        }

        Assert(offset == end);

        ScratchEnd(temp);
    }

    for (u32 i = 0; i < nodeCount; i++)
    {
        u32 childNodeIndex = childNodeIndices[i];
        f.createNode(allChildRecords[childNodeIndex], allNumChildren[childNodeIndex], &children[childNodeIndex]);
    }

    // Updates the parent
    f.updateNode(parent, records, children, childLeafIndices, leafCount);

    for (u32 i = 0; i < nodeCount; i++)
    {
        u32 childNodeIndex = childNodeIndices[i];
        BuildBVH(settings, &children[childNodeIndex], allChildRecords[childNodeIndex], allNumChildren[childNodeIndex]);
    }

    // return NodePtr::EncodeNode(node);
}

template <i32 N, typename BuildFunctions>
BVHN<N, typename BVHBuilder<N, BuildFunctions>::NodeType>
BVHBuilder<N, BuildFunctions>::BuildBVH(BuildSettings settings,
                                        Arena **inArenas,
                                        Primitive *inRawPrims,
                                        u32 count)
{
    arenas = inArenas;

    const u32 groupSize        = count / 16;
    jobsystem::Counter counter = {};

    PrimData *prims = PushArray(arenas[GetThreadIndex()], PrimData, count);

    jobsystem::KickJobs(&counter, count, groupSize, [&](jobsystem::JobArgs args) {
        u32 index         = args.jobId;
        AABB bounds       = Primitive::Bounds(inRawPrims, index);
        prims[index].minP = Lane4F32(bounds.minP);
        prims[index].maxP = Lane4F32(bounds.maxP);
        prims[index].SetGeomID(0);
        prims[index].SetPrimID(index);
    });
    jobsystem::WaitJobs(&counter);
    // for (u32 i = 0; i < count; i++)
    // {
    //     u32 index         = i;
    //     AABB bounds       = Primitive::Bounds(inRawPrims, index);
    //     prims[index].minP = Lane4F32(bounds.minP);
    //     prims[index].maxP = Lane4F32(bounds.maxP);
    //     prims[index].SetGeomID(0);
    //     prims[index].SetPrimID(index);
    // }

    primitives = inRawPrims;
    Record record;
    record = jobsystem::ParallelReduce<Record>(
        count, 1024,
        [&](Record &record, u32 start, u32 count) {
        for (u32 i = start; i < start + count; i++)
        {
            PrimData *prim    = &prims[i];
            Lane4F32 centroid = (prim->minP + prim->maxP) * 0.5f;
            record.geomBounds.Extend(prim->minP, prim->maxP);
            record.centBounds.Extend(centroid);
        } },
        [&](Record &a, Record &b) {
        a.geomBounds.Extend(b.geomBounds);
        a.centBounds.Extend(b.centBounds); });

    record.data  = prims;
    record.start = 0;
    record.end   = count;

    // BVH<N> *result = PushStruct(arenas[GetThreadIndex()], BVH<N>);
    BVHN<N, NodeType> result;
    result.root = BuildBVHRoot(settings, record);
    // if (settings.twoLevel)
    // {
    // }
    // else
    // {
    //     BuildBVH(settings, record);
    // }
    return result;
}

} // namespace rt

#endif
