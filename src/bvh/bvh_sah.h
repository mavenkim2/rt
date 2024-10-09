#ifndef BVH_SAH_H
#define BVH_SAH_H
#include "../math/simd_base.h"

// TODO:
// - partial rebraiding <-- currently working
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
    u32 total      = record.range.end - record.range.start;
    if (total < PARALLEL_THRESHOLD)
    {
        HeuristicSAHBinned<32> heuristic(record.centBounds);
        heuristic.Bin(record.data, record.range.start, total);
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

} // namespace rt

#endif
