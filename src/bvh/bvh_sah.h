#ifndef BVH_SAH_H
#define BVH_SAH_H
#include "../math/simd_base.h"

// TODO:
// - spatial splits <- currently working on
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

// far future TODOs (after moana is rendered)
// - explore octant order traversal,
// - stackless/stack compressed?
// - PLOC and agglomerative aggregate clustering
// - subdivision surfaces

// for the top level BVH, I think I can get rid of some of the meta fields maybe because the counts should be
// 1. probably can actually use 2 bits per node (node, invalid, leaf) for 2 bytes for 8-wide and 1 byte for 4-wide.
// not sure how much that would save or if it's even worth it, but something to consider.

namespace rt
{

// template <i32 numBins>
// struct HeuristicSAHBinnedAVX
// {
//     Lane8F32 bins[3][numBins];
//     Lane4U32 binCounts[numBins];
//
//     Lane8F32 scaleX;
//     Lane8F32 scaleY;
//     Lane8F32 scaleZ;
//
//     Lane8F32 baseX;
//     Lane8F32 baseY;
//     Lane8F32 baseZ;
//
//     Lane8F32 base;
//     Lane8F32 scale;
//
//     HeuristicSAHBinnedAVX() {}
//     HeuristicSAHBinnedAVX(const AABB &base) { Init(Lane4F32(base.minP), Lane4F32(base.maxP)); }
//     HeuristicSAHBinnedAVX(const Bounds &base) { Init(base.minP, base.maxP); }
//
//     __forceinline void Init(const Lane4F32 inMin, const Lane4F32 inMax)
//     {
//         const Lane4F32 eps = 1e-34f;
//
//         baseX               = Lane8F32(inMin[0]);
//         baseY               = Lane8F32(inMin[1]);
//         baseZ               = Lane8F32(inMin[2]);
//         base                = Lane8F32(inMin, inMin);
//         const Lane4F32 diag = Max(inMax - inMin, 0.f);
//
//         Lane4F32 inScale = Select(diag > eps, Lane4F32((f32)numBins) / diag, Lane4F32(0.f));
//         scaleX           = Lane8F32(inScale[0]);
//         scaleY           = Lane8F32(inScale[1]);
//         scaleZ           = Lane8F32(inScale[2]);
//         scale            = Lane8F32(inScale, inScale);
//
//         for (u32 dim = 0; dim < 3; dim++)
//         {
//             for (u32 i = 0; i < numBins; i++)
//             {
//                 bins[dim][i] = pos_inf;
//             }
//         }
//         for (u32 i = 0; i < numBins; i++)
//         {
//             binCounts[i] = 0;
//         }
//     }
//
//     __forceinline void BinTest(const PrimData *prims, u32 start, u32 count)
//     {
//         u32 i   = start;
//         u32 end = start + count;
//
//         u32 tempBinIndicesX[8];
//         u32 tempBinIndicesY[8];
//         u32 tempBinIndicesZ[8];
//
//         Lane8F32 prev[8] = {
//             prims[i].m256,
//             prims[i + 1].m256,
//             prims[i + 2].m256,
//             prims[i + 3].m256,
//             prims[i + 4].m256,
//             prims[i + 5].m256,
//             prims[i + 6].m256,
//             prims[i + 7].m256,
//         };
//         Lane8F32 minX;
//         Lane8F32 minY;
//         Lane8F32 minZ;
//
//         Lane8F32 maxX;
//         Lane8F32 maxY;
//         Lane8F32 maxZ;
//
//         Transpose8x6(prev[0], prev[1], prev[2], prev[3],
//                      prev[4], prev[5], prev[6], prev[7],
//                      minX, minY, minZ, maxX, maxY, maxZ);
//
//         const Lane8F32 centroidX0   = (minX - maxX) * 0.5f;
//         const Lane8U32 binIndicesX0 = Clamp(Lane8U32(0), Lane8U32(numBins - 1), Flooru((centroidX0 - baseX) * scaleX));
//         const Lane8F32 centroidY0   = (minY - maxY) * 0.5f;
//         const Lane8U32 binIndicesY0 = Clamp(Lane8U32(0), Lane8U32(numBins - 1), Flooru((centroidY0 - baseY) * scaleY));
//         const Lane8F32 centroidZ0   = (minZ - maxZ) * 0.5f;
//         const Lane8U32 binIndicesZ0 = Clamp(Lane8U32(0), Lane8U32(numBins - 1), Flooru((centroidZ0 - baseZ) * scaleZ));
//
//         i += 8;
//         Lane8U32::Store(tempBinIndicesX, binIndicesX0);
//         Lane8U32::Store(tempBinIndicesY, binIndicesY0);
//         Lane8U32::Store(tempBinIndicesZ, binIndicesZ0);
//
//         for (; i < end; i += 8)
//         {
//             const Lane8F32 prims8[8] = {
//                 prims[i].m256,
//                 prims[i + 1].m256,
//                 prims[i + 2].m256,
//                 prims[i + 3].m256,
//
//                 prims[i + 4].m256,
//                 prims[i + 5].m256,
//                 prims[i + 6].m256,
//                 prims[i + 7].m256};
//
//             Transpose8x6(prims8[0], prims8[1], prims8[2], prims8[3],
//                          prims8[4], prims8[5], prims8[6], prims8[7],
//                          minX, minY, minZ, maxX, maxY, maxZ);
//
//             const Lane8F32 centroidX = (minX - maxX) * 0.5f;
//
//             u32 iX0 = tempBinIndicesX[0];
//             u32 iY0 = tempBinIndicesY[0];
//             u32 iZ0 = tempBinIndicesZ[0];
//             binCounts[iX0][0]++;
//             binCounts[iY0][1]++;
//             binCounts[iZ0][2]++;
//             bins[0][iX0] = Max(bins[0][iX0], prev[0]);
//             bins[1][iY0] = Max(bins[1][iY0], prev[0]);
//             bins[2][iZ0] = Max(bins[2][iZ0], prev[0]);
//             prev[0]      = prims8[0];
//
//             const Lane8U32 binIndicesX = Clamp(Lane8U32(0), Lane8U32(numBins - 1), Flooru((centroidX - baseX) * scaleX));
//
//             u32 iX1 = tempBinIndicesX[1];
//             u32 iY1 = tempBinIndicesY[1];
//             u32 iZ1 = tempBinIndicesZ[1];
//             binCounts[iX1][0]++;
//             binCounts[iY1][1]++;
//             binCounts[iZ1][2]++;
//             bins[0][iX1] = Max(bins[0][iX1], prev[1]);
//             bins[1][iY1] = Max(bins[1][iY1], prev[1]);
//             bins[2][iZ1] = Max(bins[2][iZ1], prev[1]);
//             prev[1]      = prims8[1];
//
//             const Lane8F32 centroidY = (minY - maxY) * 0.5f;
//
//             u32 iX2 = tempBinIndicesX[2];
//             u32 iY2 = tempBinIndicesY[2];
//             u32 iZ2 = tempBinIndicesZ[2];
//             binCounts[iX2][0]++;
//             binCounts[iY2][1]++;
//             binCounts[iZ2][2]++;
//             bins[0][iX2] = Max(bins[0][iX2], prev[2]);
//             bins[1][iY2] = Max(bins[1][iY2], prev[2]);
//             bins[2][iZ2] = Max(bins[2][iZ2], prev[2]);
//             prev[2]      = prims8[2];
//
//             const Lane8U32 binIndicesY = Clamp(Lane8U32(0), Lane8U32(numBins - 1), Flooru((centroidY - baseY) * scaleY));
//
//             u32 iX3 = tempBinIndicesX[3];
//             u32 iY3 = tempBinIndicesY[3];
//             u32 iZ3 = tempBinIndicesZ[3];
//             binCounts[iX3][0]++;
//             binCounts[iY3][1]++;
//             binCounts[iZ3][2]++;
//             bins[0][iX3] = Max(bins[0][iX3], prev[3]);
//             bins[1][iY3] = Max(bins[1][iY3], prev[3]);
//             bins[2][iZ3] = Max(bins[2][iZ3], prev[3]);
//             prev[3]      = prims8[3];
//
//             const Lane8F32 centroidZ = (minZ - maxZ) * 0.5f;
//
//             u32 iX4 = tempBinIndicesX[4];
//             u32 iY4 = tempBinIndicesY[4];
//             u32 iZ4 = tempBinIndicesZ[4];
//             binCounts[iX4][0]++;
//             binCounts[iY4][1]++;
//             binCounts[iZ4][2]++;
//             bins[0][iX4] = Max(bins[0][iX4], prev[4]);
//             bins[1][iY4] = Max(bins[1][iY4], prev[4]);
//             bins[2][iZ4] = Max(bins[2][iZ4], prev[4]);
//             prev[4]      = prims8[4];
//
//             const Lane8U32 binIndicesZ = Clamp(Lane8U32(0), Lane8U32(numBins - 1), Flooru((centroidZ - baseZ) * scaleZ));
//
//             u32 iX5 = tempBinIndicesX[5];
//             u32 iY5 = tempBinIndicesY[5];
//             u32 iZ5 = tempBinIndicesZ[5];
//             binCounts[iX5][0]++;
//             binCounts[iY5][1]++;
//             binCounts[iZ5][2]++;
//             bins[0][iX5] = Max(bins[0][iX5], prev[5]);
//             bins[1][iY5] = Max(bins[1][iY5], prev[5]);
//             bins[2][iZ5] = Max(bins[2][iZ5], prev[5]);
//             prev[5]      = prims8[5];
//
//             u32 iX6 = tempBinIndicesX[6];
//             u32 iY6 = tempBinIndicesY[6];
//             u32 iZ6 = tempBinIndicesZ[6];
//             binCounts[iX6][0]++;
//             binCounts[iY6][1]++;
//             binCounts[iZ6][2]++;
//             bins[0][iX6] = Max(bins[0][iX6], prev[6]);
//             bins[1][iY6] = Max(bins[1][iY6], prev[6]);
//             bins[2][iZ6] = Max(bins[2][iZ6], prev[6]);
//             prev[6]      = prims8[6];
//
//             u32 iX7 = tempBinIndicesX[7];
//             u32 iY7 = tempBinIndicesY[7];
//             u32 iZ7 = tempBinIndicesZ[7];
//             binCounts[iX7][0]++;
//             binCounts[iY7][1]++;
//             binCounts[iZ7][2]++;
//             bins[0][iX7] = Max(bins[0][iX7], prev[7]);
//             bins[1][iY7] = Max(bins[1][iY7], prev[7]);
//             bins[2][iZ7] = Max(bins[2][iZ7], prev[7]);
//             prev[7]      = prims8[7];
//
//             Lane8U32::Store(tempBinIndicesX, binIndicesX);
//             Lane8U32::Store(tempBinIndicesY, binIndicesY);
//             Lane8U32::Store(tempBinIndicesZ, binIndicesZ);
//         }
//     }
//
//     __forceinline void BinTest2(const PrimData *prims, u32 start, u32 count)
//     {
//         u32 i   = start;
//         u32 end = start + count;
//
//         u32 tempBinIndices[8];
//         Lane8F32 prev0;
//         Lane8F32 prev1;
//
//         prev0 = prims[i].m256;
//         prev1 = prims[i].m256;
//
//         Lane8F32 binMin = Shuffle4<0, 2>(prev0, prev1);
//         Lane8F32 binMax = Shuffle4<1, 3>(prev0, prev1);
//
//         Lane8F32 centroid   = (binMax + binMin) * 0.5f;
//         Lane8U32 binIndices = Clamp(Lane8U32(0), Lane8U32(numBins - 1), Flooru((centroid - base) * scale));
//
//         Lane8U32::Store(tempBinIndices, binIndices);
//         i += 2;
//
//         for (; i < end - 1; i += 2)
//         {
//             const PrimData *prim0 = &prims[i];
//             const PrimData *prim1 = &prims[i];
//
//             binMin = Shuffle4<0, 2>(prim0->m256, prim1->m256);
//             binMax = Shuffle4<1, 3>(prim0->m256, prim1->m256);
//
//             centroid   = (binMax + binMin) * 0.5f;
//             binIndices = Clamp(Lane8U32(0), Lane8U32(numBins - 1), Flooru((centroid - base) * scale));
//
//             u32 binIndexX = tempBinIndices[0];
//             u32 binIndexY = tempBinIndices[1];
//             u32 binIndexZ = tempBinIndices[2];
//             binCounts[binIndexX][0]++;
//             binCounts[binIndexY][1]++;
//             binCounts[binIndexZ][2]++;
//             bins[0][binIndexX] = Max(bins[0][binIndexX], prev0);
//             bins[1][binIndexY] = Max(bins[1][binIndexY], prev0);
//             bins[2][binIndexZ] = Max(bins[2][binIndexZ], prev0);
//
//             prev0 = prim0->m256;
//
//             u32 binIndexX2 = tempBinIndices[4];
//             u32 binIndexY2 = tempBinIndices[5];
//             u32 binIndexZ2 = tempBinIndices[6];
//             binCounts[binIndexX2][0]++;
//             binCounts[binIndexY2][1]++;
//             binCounts[binIndexZ2][2]++;
//             bins[0][binIndexX2] = Max(bins[0][binIndexX2], prev1);
//             bins[1][binIndexY2] = Max(bins[1][binIndexY2], prev1);
//             bins[2][binIndexZ2] = Max(bins[2][binIndexZ2], prev1);
//
//             prev1 = prim1->m256;
//
//             Lane8U32::Store(tempBinIndices, binIndices);
//         }
//         u32 binIndexX = tempBinIndices[0];
//         u32 binIndexY = tempBinIndices[1];
//         u32 binIndexZ = tempBinIndices[2];
//         binCounts[binIndexX][0]++;
//         binCounts[binIndexY][1]++;
//         binCounts[binIndexZ][2]++;
//         bins[0][binIndexX] = Max(bins[0][binIndexX], prev0);
//         bins[1][binIndexY] = Max(bins[1][binIndexY], prev0);
//         bins[2][binIndexZ] = Max(bins[2][binIndexZ], prev0);
//
//         u32 binIndexX2 = tempBinIndices[4];
//         u32 binIndexY2 = tempBinIndices[5];
//         u32 binIndexZ2 = tempBinIndices[6];
//         binCounts[binIndexX2][0]++;
//         binCounts[binIndexY2][1]++;
//         binCounts[binIndexZ2][2]++;
//         bins[0][binIndexX2] = Max(bins[0][binIndexX2], prev1);
//         bins[1][binIndexY2] = Max(bins[1][binIndexY2], prev1);
//         bins[2][binIndexZ2] = Max(bins[2][binIndexZ2], prev1);
//     }
// };

template <i32 numBins>
struct HeuristicSAHBinned
{
    StaticAssert(numBins >= 4, MoreThan4Bins);

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

} // namespace rt

#endif
