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

struct PrimData
{
    Lane4F32 *minP;
    Lane4F32 *maxP;

    u32 total;
};

struct Split
{
    f32 bestSAH;
    u32 bestPos;
    u32 bestDim;
    f32 bestValue;

    Split(f32 sah, u32 pos, u32 dim, f32 val) : bestSAH(sah), bestPos(pos), bestDim(dim), bestValue(val) {}
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

    AABB base;
    Lane4F32 minP;
    Lane4F32 scale;

    HeuristicSAHBinned() {}
    HeuristicSAHBinned(const AABB &base)
    {
        const Lane4F32 eps  = 1e-34f;
        minP                = base.minP;
        const Lane4F32 diag = Max(base.maxP - base.minP, 0.f);

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

    __forceinline void Bin(const PrimData *prim, u32 start, u32 count)
    {
        u32 i   = start;
        u32 end = start + count;
        for (; i < end; i += 2)
        {
            // Load
            Lane4F32 prim0Min = prim->minP[i];
            Lane4F32 prim0Max = prim->maxP[i];

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

            Lane4F32 prim1Min = prim->minP[i + 1];
            Lane4F32 prim1Max = prim->maxP[i + 1];

            Lane4F32 centroid1   = (prim1Min + prim1Max) * 0.5f;
            Lane4U32 binIndices1 = Flooru((centroid1 - minP) * scale * (f32)numBins);
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
            Lane4F32 primMin    = prim->minP[i];
            Lane4F32 primMax    = prim->maxP[i];
            Lane4F32 centroid   = (primMin + primMax) * 0.5f;
            Lane4U32 binIndices = Flooru((centroid - minP) * scale * (f32)numBins);
            binIndices          = Clamp(Lane4U32(0), Lane4U32(numBins - 1), binIndices);

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

    __forceinline void Bin(PrimData *prim)
    {
        Bin(prim, 0, prim->total);
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

            // Lane4F32 a = UnpackLo(minDimX, minDimZ);
            // Lane4F32 b = UnpackLo(minDimY, minDimZ);
            //
            // Lane4F32 x = UnpackLo(a, minDimY);
            // Lane4F32 y = UnpackHi(a, b);
            // Lane4F32 z = Shuffle<0, 1, 2, 0>(UnpackHi(minDimX, minDimY), minDimZ);

            Lane4F32 extentDimX = maxDimX - minDimX;
            Lane4F32 extentDimY = maxDimY - minDimY;
            Lane4F32 extentDimZ = maxDimZ - minDimZ;

            area[i][0] = FMA(extentDimX[0], extentDimX[1] + extentDimX[2], extentDimX[1] * extentDimX[2]);
            area[i][1] = FMA(extentDimY[0], extentDimY[1] + extentDimY[2], extentDimY[1] * extentDimY[2]);
            area[i][2] = FMA(extentDimZ[0], extentDimZ[1] + extentDimZ[2], extentDimZ[1] * extentDimZ[2]);
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

            Lane4F32 extentDimX = maxDimX - minDimX;
            Lane4F32 extentDimY = maxDimY - minDimY;
            Lane4F32 extentDimZ = maxDimZ - minDimZ;

            const Lane4U32 lCount = (lCounts[i - 1] + blockAdd) >> blockShift;
            const Lane4U32 rCount = (count + blockAdd) >> blockShift;
            const Lane4F32 lArea  = area[i - 1];

            f32 areaDimX = FMA(extentDimX[0], extentDimX[1] + extentDimX[2], extentDimX[1] * extentDimX[2]);
            f32 areaDimY = FMA(extentDimY[0], extentDimY[1] + extentDimY[2], extentDimY[1] * extentDimY[2]);
            f32 areaDimZ = FMA(extentDimZ[0], extentDimZ[1] + extentDimZ[2], extentDimZ[1] * extentDimZ[2]);
            const Lane4F32 rArea(areaDimX, areaDimY, areaDimZ, 0.f);
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

Split BinParallel(const AABB &centroidBounds, PrimData *prim)
{
    u32 blockSize                 = 1024;
    HeuristicSAHBinned<32> result = jobsystem::ParallelReduce<HeuristicSAHBinned<32>>(
        prim->total, blockSize,
        [&](HeuristicSAHBinned<32> &bin, u32 start, u32 count) { bin.Bin(prim, start, count); },
        [&](HeuristicSAHBinned<32> &a, const HeuristicSAHBinned<32> &b) { a.Merge(b); },
        centroidBounds);
    return result.Best(0);
}

struct Bounds
{
    Lane4F32 minP;
    Lane4F32 maxP;

    Bounds() : minP(pos_inf), maxP(neg_inf) {}

    __forceinline void Extend(Lane4F32 inMin, Lane4F32 inMax)
    {
        minP = Min(minP, inMin);
        maxP = Max(maxP, inMax);
    }
};

u32 PartitionSerial(Split split, PrimData *prim)
{
    const u32 bestPos   = split.bestPos;
    const u32 bestDim   = split.bestDim;
    const f32 bestValue = split.bestValue;
    u32 l               = 0;
    u32 r               = prim->total - 1;

    Bounds lBounds;
    Bounds rBounds;

    for (;;)
    {
        b32 isLeft = true;
        do
        {
            Lane4F32 centroid = (prim->minP[l] + prim->maxP[l]) * 0.5f;
            isLeft            = centroid[bestDim] < bestValue;
        } while (isLeft && (lBounds.Extend(prim->minP[l], prim->maxP[l]), l++ <= r));
        do
        {
            Lane4F32 centroid = (prim->minP[r] + prim->maxP[r]) * 0.5f;
            isLeft            = centroid[bestDim] < bestValue;
        } while (!isLeft && (rBounds.Extend(prim->minP[r], prim->maxP[r]), l <= r--));
        if (r < l) break;

        Swap(prim->minP[l], prim->minP[r]);
        Swap(prim->maxP[l], prim->maxP[r]);

        lBounds.Extend(prim->minP[l], prim->maxP[l]);
        rBounds.Extend(prim->minP[r], prim->maxP[r]);
        l++;
        r--;
    }

    return r + 1;
}

// u32 PartitionSerialCrazy(Split split, PrimData *prim)
// {
//     u32 l = 0;
//     u32 r = soa->total - LaneXF32::N;
//
//     for (;;)
//     {
//         LaneXF32 mask(True);
//
//         do
//         {
//             LaneXF32 minX = LaneXF32::LoadU(soa->minX + l);
//             // LaneXF32 minY = LaneXF32::LoadU(soa->minY + l);
//             // LaneXF32 minZ = LaneXF32::LoadU(soa->minZ + l);
//
//             // LaneXF32 maxX = LaneXF32::LoadU(soa->maxX + l);
//             // LaneXF32 maxY = LaneXF32::LoadU(soa->maxY + l);
//             // LaneXF32 maxZ = LaneXF32::LoadU(soa->maxZ + l);
//
//             LaneXF32 centroid   = minX; //(minX + maxX) * 0.5f;
//             LaneXF32 isLeftMask = centroid < split.bestValue;
//
//             const u32 isLeftBitMask = Movemask(isLeftMask);
//             u32 numLeft             = PopCount(isLeftBitMask);
//             u32 storeMaskLeft       = (1 << numLeft) - 1;
//             u32 storeMaskRight      = (1 << (LaneXF32::N - numLeft)) - 1;
//
//             LaneXF32 storeMinXLeft  = Compact(isLeftBitMask, minX);
//             LaneXF32 storeMinXRight = Compact(~isLeftBitMask, minX);
//             LaneXF32::StoreU(storeMaskLeft, soa->minX + l, storeMinXLeft);
//             LaneXF32::StoreU(storeMaskRight, soa->minX + l + numLeft, storeMinXRight);
//
//             l += numLeft;
//         } while (l <= r && Any(mask));
//
//         mask = LaneXF32(True);
//         do
//         {
//             LaneXF32 minX = LaneXF32::LoadU(soa->minX + r);
//             // LaneXF32 maxX = LaneXF32::LoadU(soa->max + r);
//
//             LaneXF32 centroid   = minX; //(minX + maxX) * 0.5f;
//             LaneXF32 isLeftMask = centroid < split.bestValue;
//
//             const u32 isLeftBitMask = Movemask(isLeftMask);
//             u32 numLeft             = PopCount(isLeftBitMask);
//             u32 numRight            = LaneXF32::N - numLeft;
//             u32 storeMaskLeft       = (1 << numLeft) - 1;
//             u32 storeMaskRight      = (1 << numRight) - 1;
//
//             LaneXF32 storeMinXLeft  = Compact(isLeftBitMask, minX);
//             LaneXF32 storeMinXRight = Compact(~isLeftBitMask, minX);
//             LaneXF32::StoreU(storeMaskRight, soa->minX + r + numLeft, storeMinXRight);
//             LaneXF32::StoreU(storeMaskLeft, soa->minX + r, storeMinXLeft);
//             r -= numRight;
//
//         } while (l <= r && Any(mask));
//         if (l > r) break;
//
//         LaneXF32 left  = LaneXF32::LoadU(soa->minX + l);
//         LaneXF32 right = LaneXF32::LoadU(soa->minX + r);
//         LaneXF32::Store(soa->minX + l, right);
//         LaneXF32::Store(soa->minX + r, left);
//
//         l += LaneXF32::N;
//         r -= LaneXF32::N;
//     }
//     return r + 1;
// }

// u32 PartitionParallel(Split split, PrimData prim, AABB &leftBounds, AABB &rightBounds)
// {
// }

u32 PartitionParallel(Split split, PrimData prim)
{
    if (prim.total < 4 * 1024) return PartitionSerial(split, &prim);

    TempArena temp             = ScratchStart(0, 0);
    jobsystem::Counter counter = {};

    // TODO: hardcoded
    const u32 numJobs = 32; // 16; // jobsystem::numProcessors;

    const u32 blockSize = 16;
    StaticAssert(IsPow2(blockSize), Pow2BlockSize);

    const u32 blockShift        = Bsf(blockSize);
    const u32 numBlocksPerChunk = numJobs;
    const u32 chunkSize         = numBlocksPerChunk << blockShift;
    const u32 numChunks         = (prim.total + chunkSize - 1) / chunkSize;
    const u32 numBlocks         = numBlocksPerChunk * numChunks;
    const u32 bestDim           = split.bestDim;
    const f32 bestValue         = split.bestValue;

    auto isLeft = [&](u32 index) -> bool {
        f32 centroid = (prim.minP[index][bestDim] + prim.maxP[index][bestDim]) * 0.5f;
        return centroid < bestValue;
    };

    // Index of first element greater than the pivot
    u32 *vi   = PushArray(temp.arena, u32, numJobs);
    u32 *mid  = PushArray(temp.arena, u32, numJobs);
    u32 *ends = PushArray(temp.arena, u32, numJobs);

    // u32 *elementsInGroup0 = PushArray(temp.arena, u32, numChunks);

    // for (u32 i = 0; i < numChunks; i++)
    // {
    //     elementsInGroup0[i] = RandomInt(0, numBlocksPerChunk);
    // }

    Bounds *lBounds = PushArray(temp.arena, Bounds, numJobs);
    Bounds *rBounds = PushArray(temp.arena, Bounds, numJobs);
    for (u32 i = 0; i < numJobs; i++)
    {
        lBounds[i] = Bounds();
        rBounds[i] = Bounds();
    }

    jobsystem::KickJobs(&counter, numJobs, numJobs, [&](jobsystem::JobArgs args) {
        const u32 group = args.jobId;
        auto GetIndex   = [&](u32 index) {
            const u32 chunkIndex   = index >> blockShift;
            const u32 blockIndex   = group; //((elementsInGroup0[chunkIndex] + group) % numBlocksPerChunk);
            const u32 indexInBlock = index & (blockSize - 1);
            return chunkIndex * chunkSize + blockSize * blockIndex + indexInBlock;
        };

        u32 l             = 0;
        u32 r             = blockSize * numChunks - 1;
        u32 lastRIndex    = GetIndex(r);
        r                 = lastRIndex >= prim.total
                                ? (lastRIndex - prim.total) < (blockSize - 1)
                                      ? r - (lastRIndex - prim.total) - 1
                                      : r - (r & (blockSize - 1)) - 1
                                : r;
        u32 newLastRIndex = GetIndex(r);
        ends[args.jobId]  = r;

        for (;;)
        {
            u32 lIndex = GetIndex(l);
            for (; l <= r && isLeft(lIndex); lIndex = GetIndex(l))
            {
                // lBounds.Extend(prim.minP[lIndex], prim.maxP[lIndex]);
                l++;
            }
            u32 rIndex = GetIndex(r);
            for (; l < r && !isLeft(rIndex); rIndex = GetIndex(r))
            {
                // rBounds.Extend(prim.minP[rIndex], prim.maxP[rIndex]);
                r--;
            }
            if (l >= r) break;

            Swap(prim.minP[lIndex], prim.minP[rIndex]);
            Swap(prim.maxP[lIndex], prim.maxP[rIndex]);

            // lBounds.Extend(prim.minP[lIndex], prim.maxP[lIndex]);
            // rBounds.Extend(prim.minP[rIndex], prim.maxP[rIndex]);

            l++;
            r--;
        }
        vi[args.jobId]  = GetIndex(l);
        mid[args.jobId] = l;
    });

    jobsystem::WaitJobs(&counter);

    u32 globalMid = 0;
    for (u32 i = 0; i < numJobs; i++)
    {
        globalMid += mid[i];
    }

    struct Range
    {
        u32 start;
        u32 end;
        u32 group;
        __forceinline u32 Size() const { return end - start; }
    };

    auto GetIndex = [&](u32 index, u32 group) {
        const u32 chunkIndex   = index >> blockShift;
        const u32 blockIndex   = group; //((elementsInGroup0[chunkIndex] + group) % numBlocksPerChunk);
        const u32 indexInBlock = index & (blockSize - 1);
        return chunkIndex * chunkSize + blockSize * blockIndex + indexInBlock;
    };

    Range *leftMisplacedRanges  = PushArray(temp.arena, Range, numJobs);
    u32 lCount                  = 0;
    Range *rightMisplacedRanges = PushArray(temp.arena, Range, numJobs);
    u32 rCount                  = 0;

    u32 numMisplacedLeft  = 0;
    u32 numMisplacedRight = 0;

    for (u32 i = 0; i < numJobs; i++)
    {
        u32 globalIndex = GetIndex(mid[i], i);
        if (globalIndex < globalMid)
        {
            u32 diff = (((globalMid - globalIndex) / chunkSize) << blockShift) + ((globalMid - globalIndex) & (blockSize - 1));

            u32 check = GetIndex(mid[i] + diff - 1, i);

            int steps0 = 0;
            while (check > globalMid && !isLeft(check))
            {
                diff -= 1;
                steps0++;
                check = GetIndex(mid[i] + diff - 1, i);
            }
            int steps1 = 0;
            check      = GetIndex(mid[i] + diff + 1, i);
            while (check < globalMid && !isLeft(check))
            {
                steps1++;
                diff += 1;
                check = GetIndex(mid[i] + diff + 1, i);
            }
            if (steps1) diff++;

            rightMisplacedRanges[rCount] = {mid[i], mid[i] + diff, i};
            numMisplacedRight += rightMisplacedRanges[rCount].Size();
            rCount++;
        }
        else if (globalIndex > globalMid)
        {
            u32 diff = (((globalIndex - globalMid) / chunkSize) << blockShift) + ((globalIndex - globalMid) & (blockSize - 1));
            Assert(diff <= mid[i]);
            u32 check  = GetIndex(mid[i] - diff + 1, i);
            int steps0 = 0;
            while (check < globalMid && isLeft(check))
            {
                diff -= 1;
                steps0++;
                check = GetIndex(mid[i] - diff + 1, i);
                // u32 check2 = GetIndex(mid[i] - diff, i);
                // check      = GetIndex(mid[i] - diff - 1, i);
                // Assert(check2 >= globalMid);
                // Assert(check < globalMid);
                // int stop = 5;
            }
            if (steps0) diff--;

            check      = GetIndex(mid[i] - diff - 1, i);
            int steps1 = 0;
            while (check >= globalMid && isLeft(check))
            {
                steps1++;
                diff += 1;
                check = GetIndex(mid[i] - diff - 1, i);
            }
            leftMisplacedRanges[lCount] = {mid[i] - diff, mid[i], i};
            numMisplacedLeft += leftMisplacedRanges[lCount].Size();
            lCount++;
        }
    }

    Assert(numMisplacedLeft == numMisplacedRight);
    printf("Num misplaced: %u\n", numMisplacedLeft);

    u32 leftIndex  = 0;
    u32 rightIndex = 0;

    Range &lRange = leftMisplacedRanges[leftIndex];
    u32 lSize     = lRange.Size();
    u32 lIter     = 0;

    Range &rRange = rightMisplacedRanges[rightIndex];
    u32 rIter     = 0;
    u32 rSize     = rRange.Size();

    u32 testLCount = 0;
    u32 testRCount = 0;

    for (;;)
    {
        while (lSize != lIter && rSize != rIter)
        {
            u32 lIndex = GetIndex(lRange.start + lIter, lRange.group);
            u32 rIndex = GetIndex(rRange.start + rIter, rRange.group);

            Swap(prim.minP[lIndex], prim.minP[rIndex]);
            Swap(prim.maxP[lIndex], prim.maxP[rIndex]);

            lIter++;
            rIter++;
            testLCount++;
            testRCount++;
        }
        if (leftIndex == lCount - 1 && rightIndex == rCount - 1) break;
        if (rSize == rIter)
        {
            rightIndex++;
            rRange = rightMisplacedRanges[rightIndex];
            rIter  = 0;
            rSize  = rRange.Size();
        }
        if (lSize == lIter)
        {
            leftIndex++;
            lRange = leftMisplacedRanges[leftIndex];
            lIter  = 0;
            lSize  = lRange.Size();
        }
    }

#if 0
    u32 vMax = neg_inf;
    u32 vMin = pos_inf;
    Bounds lBound;
    Bounds rBound;
    for (u32 i = 0; i < numJobs; i++)
    {
        vMax = Max(vMax, vi[i]);
        vMin = Min(vMin, vi[i]);

        // lBound.Extend(lBounds[i]);
        // rBound.Extend(rBounds[i]);
    }

    ScratchEnd(temp);
    PrimData newPrim;
    newPrim.minP  = prim.minP + vMin;
    newPrim.maxP  = prim.maxP + vMin;
    newPrim.total = vMax - vMin;
    return vMin + PartitionParallel(split, newPrim);
#endif
    return globalMid;
}

// void BuildTree(PrimData prim)
// {
//
//     prim.Find();
// }

} // namespace rt

#endif
