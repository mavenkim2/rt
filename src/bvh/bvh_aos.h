#ifndef BVH_AOS_H
#define BVH_AOS_H
#include <atomic>
#include <utility>
namespace rt
{
//////////////////////////////
// AOS Partitions
//

template <i32 numBins>
u32 Partition(const ObjectBinner<numBins> *binner, PrimRef *data, u32 dim, u32 bestPos, i32 l, i32 r)
{
    for (;;)
    {
        while (l <= r)
        {
            PrimRef *lRef = &data[l];
            f32 max       = lRef->max[dim];
            f32 min       = lRef->min[dim];
            f32 centroid  = (max - min) * 0.5f;
            bool isRight  = binner->Bin(centroid, dim) >= bestPos;
            if (isRight) break;
            l++;
        }
        while (l <= r)
        {
            Assert(r >= 0);
            PrimRef *rRef = &data[r];
            f32 min       = rRef->min[dim];
            f32 max       = rRef->max[dim];
            f32 centroid  = (max - min) * 0.5f;

            bool isLeft = binner->Bin(centroid, dim) < bestPos;
            if (isLeft) break;
            r--;
        }
        if (l > r) break;

        Swap(data[l], data[r]);
        l++;
        r--;
    }
    return l;
}

template <typename GetIndex, i32 numBins>
u32 PartitionAndBounds(const ObjectBinner<numBins> *binner, PrimRef *data, u32 dim, u32 bestPos, i32 l, i32 r, GetIndex getIndex,
                       Lane8F32 &outGeomLeft, Lane8F32 &outGeomRight, Lane8F32 &outCentLeft, Lane8F32 &outCentRight, u32 chunkSize = 0)
{
    Lane8F32 masks[2] = {Lane8F32::Mask(false), Lane8F32::Mask(true)};

    Lane8F32 centLeft(neg_inf);
    Lane8F32 centRight(neg_inf);
    Lane8F32 geomLeft(neg_inf);
    Lane8F32 geomRight(neg_inf);

    i32 start = l;
    i32 end   = r;

    Lane8F32 lanes[8];
    for (;;)
    {
        Lane8F32 lVal;
        Lane8F32 rVal;
        u32 lIndex;
        u32 rIndex;
        while (l <= r)
        {
            lIndex            = getIndex(l);
            lVal              = Lane8F32::Load(&data[lIndex]);
            Lane4F32 min      = Lane4F32::Load(data[lIndex].min);
            Lane4F32 max      = Lane4F32::Load(data[lIndex].max);
            Lane4F32 centroid = (max - min) * 0.5f;

            u32 bin      = binner->Bin(centroid[dim], dim);
            bool isRight = bin >= bestPos;
            Lane8F32 c(-centroid, centroid);
            if (isRight)
            {
                centRight = Max(centRight, c);
                geomRight = Max(geomRight, lVal);
                break;
            }
            centLeft = Max(centLeft, c);
            geomLeft = Max(geomLeft, lVal);
            l++;
        }
        while (l <= r)
        {
            Assert(r >= 0);

            rIndex            = getIndex(r);
            rVal              = Lane8F32::Load(&data[rIndex]);
            Lane4F32 min      = Lane4F32::Load(data[rIndex].min);
            Lane4F32 max      = Lane4F32::Load(data[rIndex].max);
            Lane4F32 centroid = (max - min) * 0.5f;

            u32 bin     = binner->Bin(centroid[dim], dim);
            bool isLeft = bin < bestPos;
            Lane8F32 c(-centroid, centroid);
            if (isLeft)
            {
                centLeft = Max(centLeft, c);
                geomLeft = Max(geomLeft, lVal);
                break;
            }
            centRight = Max(centRight, c);
            geomRight = Max(geomRight, lVal);
            r--;
        }
        if (l > r) break;

        Lane8F32::Store(&data[lIndex], rVal);
        Lane8F32::Store(&data[rIndex], lVal);
    }

    outGeomLeft  = geomLeft;
    outGeomRight = geomRight;
    outCentLeft  = centLeft;
    outCentRight = centRight;
    return l;
}

template <i32 numBins>
u32 PartitionParallel(const ObjectBinner<numBins> *binner, PrimRef *data, Split split, u32 start, u32 count)
{
    static const u32 PARTITION_PARALLEL_THRESHOLD = 4 * 1024;
    if (count < PARTITION_PARALLEL_THRESHOLD)
    {
        Lane8F32 geomLeft;
        Lane8F32 geomRight;
        Lane8F32 centLeft;
        Lane8F32 centRight;
        return PartitionAndBounds(
            binner, data, split.bestDim, split.bestPos, start, start + count - 1, [&](u32 index) { return index; },
            geomLeft, geomRight, centLeft, centRight);
    }

    const u32 blockSize         = 512;
    const u32 blockMask         = blockSize - 1;
    const u32 blockShift        = Bsf(blockSize);
    const u32 numJobs           = Min(32u, (count + 511) / 512); // OS_NumProcessors();
    const u32 numBlocksPerChunk = numJobs;
    const u32 chunkSize         = blockSize * numBlocksPerChunk;
    Assert(IsPow2(chunkSize));

    const u32 numChunks = (count + chunkSize - 1) / chunkSize;
    u32 end             = start + count;

    TempArena temp = ScratchStart(0, 0);
    u32 *outMid    = PushArray(temp.arena, u32, numJobs);

    Lane8F32 *geomLefts  = PushArrayNoZero(temp.arena, Lane8F32, numJobs);
    Lane8F32 *geomRights = PushArrayNoZero(temp.arena, Lane8F32, numJobs);
    Lane8F32 *centLefts  = PushArrayNoZero(temp.arena, Lane8F32, numJobs);
    Lane8F32 *centRights = PushArrayNoZero(temp.arena, Lane8F32, numJobs);

    auto GetIndex = [&](u32 index, u32 group) {
        const u32 chunkIndex   = index >> blockShift;
        const u32 indexInBlock = index & blockMask;

        u32 outIndex = start + chunkIndex * chunkSize + (group << blockShift) + indexInBlock;
        return outIndex;
    };

    scheduler.ScheduleAndWait(numJobs, 1, [&](u32 jobID) {
        TIMED_FUNCTION(miscF);
        const u32 group = jobID;

        u32 l          = 0;
        u32 r          = (numChunks << blockShift) - 1;
        u32 lastRIndex = GetIndex(r, group);
        // r              = lastRIndex > rEndAligned ? r - 8 : r;

        r = lastRIndex >= end
                ? (lastRIndex - end) < (blockSize - 1)
                      ? r - (lastRIndex - end) - 1
                      : r - (r & (blockSize - 1)) - 1
                : r;

        // u32 lIndex = GetIndex(l, group);
        u32 rIndex = GetIndex(r, group);
        Assert(rIndex < end);

        auto GetIndexGroup = [&](u32 index) {
            return GetIndex(index, group);
        };

        outMid[jobID] = PartitionAndBounds(binner, data, split.bestDim, split.bestPos, l, r, GetIndexGroup,
                                           geomLefts[jobID], geomRights[jobID], centLefts[jobID], centRights[jobID], chunkSize);
    });

    // Partition the beginning and the end
    u32 minIndex  = pos_inf;
    u32 maxIndex  = neg_inf;
    u32 globalMid = start;
    for (u32 i = 0; i < numJobs; i++)
    {
        globalMid += outMid[i];
        minIndex = Min(GetIndex(outMid[i], i), minIndex);
        maxIndex = Max(GetIndex(outMid[i], i), maxIndex);
    }

    PerformanceCounter c = OS_StartCounter();
    // u32 out              = Partition(binner, data, split.bestDim, split.bestPos, minIndex, maxIndex - 1);
    u32 out = FixMisplacedRanges(data, numJobs, chunkSize, blockShift, blockSize, globalMid, outMid, GetIndex);
    printf("serial part: %f\n", OS_GetMilliseconds(c));

    Lane8F32 geomLeft;
    Lane8F32 geomRight;
    Lane8F32 centLeft;
    Lane8F32 centRight;

    ScratchEnd(temp);
    return out;
}

template <typename Binner>
void Partition(u32 lOffset, u32 rOffset, const Binner *binner, u32 start, u32 count, const PrimRef *data, const u32 *refs,
               u32 *outRefs, u32 dim, u32 bestPos, RecordAOSSplits &outLeft, RecordAOSSplits &outRight,
               u32 expectedLCount, u32 expectedRCount)
{
    u32 writeLocs[2]  = {lOffset, rOffset};
    Lane8F32 masks[2] = {Lane8F32::Mask(false), Lane8F32::Mask(true)};

    Lane8F32 lanes[8];

    Lane8F32 centLeft(neg_inf);
    Lane8F32 centRight(neg_inf);

    Lane8F32 geomLeft(neg_inf);
    Lane8F32 geomRight(neg_inf);

    u32 i            = start;
    u32 totalL       = 0;
    u32 totalR       = 0;
    u32 alignedCount = count & ~(LANE_WIDTH - 1);
    for (; i < start + alignedCount; i += LANE_WIDTH)
    {
        Transpose8x6(data[refs[i + 0]].m256, data[refs[i + 1]].m256, data[refs[i + 2]].m256, data[refs[i + 3]].m256,
                     data[refs[i + 4]].m256, data[refs[i + 5]].m256, data[refs[i + 6]].m256, data[refs[i + 7]].m256,
                     lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5]);
        Lane8F32 centroids[3] = {
            (lanes[3] - lanes[0]) * 0.5f,
            (lanes[4] - lanes[1]) * 0.5f,
            (lanes[5] - lanes[2]) * 0.5f,
        };
        Lane8U32 bin = binner->Bin(centroids[dim], dim);

        Lane8F32 mask = (AsFloat(bin) >= AsFloat(bestPos));
        u32 prevMask  = Movemask(mask);

        Transpose3x8(centroids[0], centroids[1], centroids[2],
                     lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7]);
        for (u32 chunkIndex = 0; chunkIndex < LANE_WIDTH; chunkIndex++)
        {
            u32 select = (prevMask >> chunkIndex) & 1;
            totalL += !select;
            totalR += select;
            centLeft  = MaskMax(masks[!select], centLeft, lanes[chunkIndex] ^ signFlipMask);
            centRight = MaskMax(masks[select], centRight, lanes[chunkIndex] ^ signFlipMask);

            outRefs[writeLocs[select]++] = refs[i + chunkIndex];

            geomLeft  = MaskMax(masks[!select], geomLeft, data[refs[i + chunkIndex]].m256);
            geomRight = MaskMax(masks[select], geomRight, data[refs[i + chunkIndex]].m256);
        }
    }
    for (; i < start + count; i++)
    {
        const PrimRef *primRef = &data[refs[i]];
        Lane4F32 min           = Extract4<0>(primRef->m256);
        Lane4F32 max           = Extract4<1>(primRef->m256);
        Lane4F32 centroid      = (max - min) * 0.5f;
        Lane8F32 c(-centroid, centroid);
        u32 bin                       = binner->Bin(centroid[dim], dim);
        bool isRight                  = bin >= bestPos;
        outRefs[writeLocs[isRight]++] = refs[i];
        totalR += isRight;
        totalL += !isRight;
        centLeft  = MaskMax(masks[!isRight], centLeft, c);
        centRight = MaskMax(masks[isRight], centRight, c);

        geomRight = MaskMax(masks[isRight], geomRight, primRef->m256);
        geomLeft  = MaskMax(masks[!isRight], geomLeft, primRef->m256);
    }
    Assert(expectedLCount == totalL);
    Assert(expectedRCount == totalR);
    outLeft.centBounds  = centLeft;
    outLeft.geomBounds  = geomLeft;
    outRight.centBounds = centRight;
    outRight.geomBounds = geomRight;
}

template <i32 numBins = 32>
struct HeuristicAOSObjectBinning
{
    Bounds8 bins[3][numBins];
    Lane4U32 counts[numBins];
    ObjectBinner<numBins> *binner;

    HeuristicAOSObjectBinning() {}
    HeuristicAOSObjectBinning(ObjectBinner<numBins> *binner) : binner(binner)
    {
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 i = 0; i < numBins; i++)
            {
                bins[dim][i] = Bounds8();
            }
        }
        for (u32 i = 0; i < numBins; i++)
        {
            counts[i] = 0;
        }
    }
    void Bin(const PrimRef *data, u32 start, u32 count)
    {
        u32 alignedCount = count - count % LANE_WIDTH;
        u32 i            = start;

        Lane8F32 prevLanes[8];
        alignas(32) u32 prevBinIndices[3][8];
        if (count >= LANE_WIDTH)
        {
            Lane8F32 temp[6];
            Transpose8x6(data[i + 0].m256, data[i + 1].m256, data[i + 2].m256, data[i + 3].m256,
                         data[i + 4].m256, data[i + 5].m256, data[i + 6].m256, data[i + 7].m256,
                         temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]);
            Lane8F32 centroids[3] = {
                (temp[3] - temp[0]) * 0.5f,
                (temp[4] - temp[1]) * 0.5f,
                (temp[5] - temp[2]) * 0.5f,
            };
            Lane8U32::Store(prevBinIndices[0], binner->Bin(centroids[0], 0));
            Lane8U32::Store(prevBinIndices[1], binner->Bin(centroids[1], 1));
            Lane8U32::Store(prevBinIndices[2], binner->Bin(centroids[2], 2));
            for (u32 c = 0; c < LANE_WIDTH; c++)
            {
                prevLanes[c] = data[i + c].m256;
            }
            i += LANE_WIDTH;
        }
        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            Lane8F32 temp[6];

            Transpose8x6(data[i + 0].m256, data[i + 1].m256, data[i + 2].m256, data[i + 3].m256,
                         data[i + 4].m256, data[i + 5].m256, data[i + 6].m256, data[i + 7].m256,
                         temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]);
            Lane8F32 centroids[] = {
                (temp[3] - temp[0]) * 0.5f,
                (temp[4] - temp[1]) * 0.5f,
                (temp[5] - temp[2]) * 0.5f,
            };

            Lane8U32 indicesX = binner->Bin(centroids[0], 0);
            Lane8U32 indicesY = binner->Bin(centroids[1], 1);
            Lane8U32 indicesZ = binner->Bin(centroids[2], 2);

            for (u32 dim = 0; dim < 3; dim++)
            {
                for (u32 b = 0; b < LANE_WIDTH; b++)
                {
                    u32 bin = prevBinIndices[dim][b];
                    bins[dim][bin].Extend(prevLanes[b]);
                    counts[bin][dim]++;
                }
            }

            Lane8U32::Store(prevBinIndices[0], indicesX);
            Lane8U32::Store(prevBinIndices[1], indicesY);
            Lane8U32::Store(prevBinIndices[2], indicesZ);
            for (u32 c = 0; c < LANE_WIDTH; c++)
            {
                prevLanes[c] = data[i + c].m256;
            }
        }
        if (count >= LANE_WIDTH)
        {
            for (u32 dim = 0; dim < 3; dim++)
            {
                for (u32 b = 0; b < LANE_WIDTH; b++)
                {
                    u32 bin = prevBinIndices[dim][b];
                    bins[dim][bin].Extend(prevLanes[b]);
                    counts[bin][dim]++;
                }
            }
        }
        for (; i < start + count; i++)
        {
            Lane4F32 low      = Extract4<0>(data[i].m256);
            Lane4F32 hi       = Extract4<1>(data[i].m256);
            Lane4F32 centroid = (hi - low) * 0.5f;
            u32 indices[]     = {
                binner->Bin(centroid[0], 0),
                binner->Bin(centroid[1], 1),
                binner->Bin(centroid[2], 2),
            };
            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 bin = indices[dim];
                bins[dim][bin].Extend(data[i].m256);
                counts[bin][dim]++;
            }
        }
    }

    void Bin(const PrimRef *data, const u32 *refs, u32 start, u32 count)
    {
        u32 alignedCount = count - count % LANE_WIDTH;
        u32 i            = start;

        Lane8F32 prevLanes[8];
        alignas(32) u32 prevBinIndices[3][8];
        if (count >= LANE_WIDTH)
        {
            Lane8F32 temp[6];
            Transpose8x6(data[refs[i + 0]].m256, data[refs[i + 1]].m256, data[refs[i + 2]].m256, data[refs[i + 3]].m256,
                         data[refs[i + 4]].m256, data[refs[i + 5]].m256, data[refs[i + 6]].m256, data[refs[i + 7]].m256,
                         temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]);
            Lane8F32 centroids[3] = {
                (temp[3] - temp[0]) * 0.5f,
                (temp[4] - temp[1]) * 0.5f,
                (temp[5] - temp[2]) * 0.5f,
            };
            Lane8U32::Store(prevBinIndices[0], binner->Bin(centroids[0], 0));
            Lane8U32::Store(prevBinIndices[1], binner->Bin(centroids[1], 1));
            Lane8U32::Store(prevBinIndices[2], binner->Bin(centroids[2], 2));
            for (u32 c = 0; c < LANE_WIDTH; c++)
            {
                prevLanes[c] = data[refs[i + c]].m256;
            }
            i += LANE_WIDTH;
        }
        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            Lane8F32 temp[6];

            Transpose8x6(data[refs[i + 0]].m256, data[refs[i + 1]].m256, data[refs[i + 2]].m256, data[refs[i + 3]].m256,
                         data[refs[i + 4]].m256, data[refs[i + 5]].m256, data[refs[i + 6]].m256, data[refs[i + 7]].m256,
                         temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]);
            Lane8F32 centroids[] = {
                (temp[3] - temp[0]) * 0.5f,
                (temp[4] - temp[1]) * 0.5f,
                (temp[5] - temp[2]) * 0.5f,
            };

            Lane8U32 indicesX = binner->Bin(centroids[0], 0);
            Lane8U32 indicesY = binner->Bin(centroids[1], 1);
            Lane8U32 indicesZ = binner->Bin(centroids[2], 2);

            for (u32 dim = 0; dim < 3; dim++)
            {
                for (u32 b = 0; b < LANE_WIDTH; b++)
                {
                    u32 bin = prevBinIndices[dim][b];
                    bins[dim][bin].Extend(prevLanes[b]);
                    counts[bin][dim]++;
                }
            }

            Lane8U32::Store(prevBinIndices[0], indicesX);
            Lane8U32::Store(prevBinIndices[1], indicesY);
            Lane8U32::Store(prevBinIndices[2], indicesZ);
            for (u32 c = 0; c < LANE_WIDTH; c++)
            {
                prevLanes[c] = data[refs[i + c]].m256;
            }
        }
        if (count >= LANE_WIDTH)
        {
            for (u32 dim = 0; dim < 3; dim++)
            {
                for (u32 b = 0; b < LANE_WIDTH; b++)
                {
                    u32 bin = prevBinIndices[dim][b];
                    bins[dim][bin].Extend(prevLanes[b]);
                    counts[bin][dim]++;
                }
            }
        }
        for (; i < start + count; i++)
        {
            Lane4F32 low      = Extract4<0>(data[refs[i]].m256);
            Lane4F32 hi       = Extract4<1>(data[refs[i]].m256);
            Lane4F32 centroid = (hi - low) * 0.5f;
            u32 indices[]     = {
                binner->Bin(centroid[0], 0),
                binner->Bin(centroid[1], 1),
                binner->Bin(centroid[2], 2),
            };
            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 bin = indices[dim];
                bins[dim][bin].Extend(data[refs[i]].m256);
                counts[bin][dim]++;
            }
        }
    }
    void Merge(const HeuristicAOSObjectBinning &other)
    {
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 i = 0; i < numBins; i++)
            {
                bins[dim][i].Extend(other.bins[dim][i]);
            }
        }
        for (u32 i = 0; i < numBins; i++)
        {
            counts[i] += other.counts[i];
        }
    }
};

template <i32 numBins = 16>
struct alignas(32) HeuristicAOSSplitBinning
{
    Bounds8 bins[3][numBins];
    Lane4U32 entryCounts[numBins];
    Lane4U32 exitCounts[numBins];

    SplitBinner<numBins> *binner;

    HeuristicAOSSplitBinning() {}
    HeuristicAOSSplitBinning(SplitBinner<numBins> *binner) : binner(binner)
    {
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 i = 0; i < numBins; i++)
            {
                bins[dim][i] = Bounds8();
            }
        }
        for (u32 i = 0; i < numBins; i++)
        {
            entryCounts[i] = 0;
            exitCounts[i]  = 0;
        }
    }

    void Bin(TriangleMesh *mesh, const PrimRef *data, u32 start, u32 count)
    {
        u32 binCounts[3][numBins]                                 = {};
        alignas(32) u32 binIndexStart[3][numBins][2 * LANE_WIDTH] = {};
        u32 faceIndices[3][numBins][2 * LANE_WIDTH]               = {};

        u32 i            = start;
        u32 alignedCount = count - count % LANE_WIDTH;

        Lane8F32 lanes[6];

        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            Transpose8x6(data[i + 0].m256, data[i + 1].m256, data[i + 2].m256, data[i + 3].m256,
                         data[i + 4].m256, data[i + 5].m256, data[i + 6].m256, data[i + 7].m256,
                         lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5]);

            Assert(All(-lanes[0] >= binner->base[0]));
            Assert(All(-lanes[1] >= binner->base[1]));
            Assert(All(-lanes[2] >= binner->base[2]));
            Assert(All(lanes[3] >= binner->base[0]));
            Assert(All(lanes[4] >= binner->base[1]));
            Assert(All(lanes[5] >= binner->base[2]));

            Lane8U32 indexMinArr[3] = {
                binner->BinMin(lanes[0], 0),
                binner->BinMin(lanes[1], 1),
                binner->BinMin(lanes[2], 2),
            };
            Lane8U32 indexMaxArr[3] = {
                binner->BinMax(lanes[3], 0),
                binner->BinMax(lanes[4], 1),
                binner->BinMax(lanes[5], 2),
            };

            Assert(All(AsFloat(indexMinArr[0]) <= AsFloat(indexMaxArr[0])));
            Assert(All(AsFloat(indexMinArr[1]) <= AsFloat(indexMaxArr[1])));
            Assert(All(AsFloat(indexMinArr[2]) <= AsFloat(indexMaxArr[2])));

            Lane8U32 binDiffX = indexMaxArr[0] - indexMinArr[0];
            Lane8U32 binDiffY = indexMaxArr[1] - indexMinArr[1];
            Lane8U32 binDiffZ = indexMaxArr[2] - indexMinArr[2];
            // x0 x1 x2 x3 y0 y1 y2 y3 x4 x5 x6 x7 y4 y5 y6 y7
            Lane8U32 out0 = PackU32(binDiffX, binDiffY);
            // x0 x1 x2 x3 |  y0 y1 y2 y3 |  z0 00 z1 00 |  z2 00 z3 00 |  x4 x5 x6 x7 |  y4 y5 y6 y7 |  z4 00 z5 00 | z6 00 z7 00

            alignas(32) u8 bytes[32];
            Lane8U32 out1 = PackU16(out0, binDiffZ);
            Lane8U32::Store(bytes, out1);

            u32 bitMask[3] = {};

            static const u32 order[3][LANE_WIDTH] = {
                {0, 1, 2, 3, 16, 17, 18, 19},
                {4, 5, 6, 7, 20, 21, 22, 23},
                {8, 10, 12, 14, 24, 26, 28, 30},
            };

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
                            bins[dim][indexMin].Extend(data[i + b].m256);
                        }
                        break;
                        default:
                        {
                            bitMask[dim] |= (1 << diff);
                            faceIndices[dim][diff][binCounts[dim][diff]]   = data[i + b].primID;
                            binIndexStart[dim][diff][binCounts[dim][diff]] = indexMin;
                            binCounts[dim][diff]++;
                        }
                    }
                }
            }
            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 numIters = PopCount(bitMask[dim]);
                for (u32 iter = 0; iter < numIters; iter++)
                {
                    u32 bin = Bsf(bitMask[dim]);
                    if (binCounts[dim][bin] >= LANE_WIDTH)
                    {
                        Assert(binCounts[dim][bin] <= ArrayLength(binIndexStart[0][0]));
                        binCounts[dim][bin] -= LANE_WIDTH;
                        u32 binCount = binCounts[dim][bin];

                        Bounds8 bounds[2][LANE_WIDTH];
                        Triangle8 tri     = Triangle8::Load(mesh, dim, faceIndices[dim][bin] + binCount);
                        Lane8U32 startBin = Lane8U32::LoadU(binIndexStart[dim][bin] + binCount);

                        for (u32 boundIndex = 0; boundIndex < LANE_WIDTH; boundIndex++)
                        {
                            bounds[0][boundIndex] = Bounds8(pos_inf);
                        }
                        alignas(32) u32 binIndices[LANE_WIDTH];

                        u32 current = 0;
                        for (u32 d = 0; d < bin; d++)
                        {
                            Lane8U32::Store(binIndices, startBin);
                            startBin += 1u;
                            Lane8F32 splitPos = binner->GetSplitValue(startBin, dim);

                            ClipTriangle(mesh, dim, tri, splitPos, bounds[current], bounds[!current]);

                            for (u32 b = 0; b < LANE_WIDTH; b++)
                            {
                                u32 binIndex = binIndices[b];
                                bins[dim][binIndex].Extend(bounds[current][b]);
                            }
                            current = !current;
                        }
                        for (u32 b = 0; b < LANE_WIDTH; b++)
                        {
                            u32 binIndex = binIndices[b] + 1;
                            bins[dim][binIndex].Extend(bounds[current][b]);
                        }
                        binCounts[dim][bin] = 0;
                    }
                    bitMask[dim] &= bitMask[dim] - 1;
                }
            }
        }
        // Finish the remaining primitives
        for (; i < start + count; i++)
        {
            const PrimRef *ref = &data[i];

            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 binIndexMin = binner->BinMin(ref->min[dim], dim);
                u32 binIndexMax = binner->BinMax(ref->max[dim], dim);

                Assert(binIndexMax >= binIndexMin);

                u32 diff = binIndexMax - binIndexMin;
                entryCounts[binIndexMin][dim] += 1;
                exitCounts[binIndexMax][dim] += 1;
                switch (diff)
                {
                    case 0:
                    {
                        bins[dim][binIndexMin].Extend(ref->m256);
                    }
                    break;
                    default:
                    {
                        faceIndices[dim][diff][binCounts[dim][diff]]   = ref->primID;
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
                Assert(remainingCount <= ArrayLength(binIndexStart[0][0]));

                const u32 numIters = ((remainingCount + 7) >> 3);
                for (u32 remaining = 0; remaining < numIters; remaining++)
                {
                    u32 numPrims = Min(remainingCount, 8u);
                    Bounds8 bounds[2][8];
                    for (u32 boundIndex = 0; boundIndex < LANE_WIDTH; boundIndex++)
                    {
                        bounds[0][boundIndex] = Bounds8(pos_inf);
                    }
                    Triangle8 tri     = Triangle8::Load(mesh, dim, faceIndices[dim][diff] + remaining * LANE_WIDTH); //, bounds[0]);
                    Lane8U32 startBin = Lane8U32::LoadU(binIndexStart[dim][diff] + remaining * LANE_WIDTH);

                    alignas(32) u32 binIndices[8];

                    u32 current = 0;
                    for (u32 d = 0; d < diff; d++)
                    {
                        Lane8U32::Store(binIndices, startBin);
                        startBin += 1u;
                        Lane8F32 splitPos = binner->GetSplitValue(startBin, dim);
                        ClipTriangle(mesh, dim, tri, splitPos, bounds[current], bounds[!current]);
                        for (u32 b = 0; b < numPrims; b++)
                        {
                            u32 binIndex = binIndices[b];
                            bins[dim][binIndex].Extend(bounds[current][b]);
                        }
                        current = !current;
                    }
                    for (u32 b = 0; b < numPrims; b++)
                    {
                        u32 binIndex = binIndices[b] + 1;
                        bins[dim][binIndex].Extend(bounds[current][b]);
                    }
                    remainingCount -= LANE_WIDTH;
                }
            }
        }
    }
    void Bin(TriangleMesh *mesh, const PrimRef *data, const u32 *refs, u32 start, u32 count)
    {
        u32 binCounts[3][numBins]                                 = {};
        alignas(32) u32 binIndexStart[3][numBins][2 * LANE_WIDTH] = {};
        u32 faceIndices[3][numBins][2 * LANE_WIDTH]               = {};

        u32 i            = start;
        u32 alignedCount = count - count % LANE_WIDTH;

        Lane8F32 lanes[6];

        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            Transpose8x6(data[refs[i + 0]].m256, data[refs[i + 1]].m256, data[refs[i + 2]].m256, data[refs[i + 3]].m256,
                         data[refs[i + 4]].m256, data[refs[i + 5]].m256, data[refs[i + 6]].m256, data[refs[i + 7]].m256,
                         lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5]);

            Assert(All(-lanes[0] >= binner->base[0]));
            Assert(All(-lanes[1] >= binner->base[1]));
            Assert(All(-lanes[2] >= binner->base[2]));
            Assert(All(lanes[3] >= binner->base[0]));
            Assert(All(lanes[4] >= binner->base[1]));
            Assert(All(lanes[5] >= binner->base[2]));

            Lane8U32 indexMinArr[3] = {
                binner->BinMin(lanes[0], 0),
                binner->BinMin(lanes[1], 1),
                binner->BinMin(lanes[2], 2),
            };
            Lane8U32 indexMaxArr[3] = {
                binner->BinMax(lanes[3], 0),
                binner->BinMax(lanes[4], 1),
                binner->BinMax(lanes[5], 2),
            };

            Assert(All(AsFloat(indexMinArr[0]) <= AsFloat(indexMaxArr[0])));
            Assert(All(AsFloat(indexMinArr[1]) <= AsFloat(indexMaxArr[1])));
            Assert(All(AsFloat(indexMinArr[2]) <= AsFloat(indexMaxArr[2])));

            Lane8U32 binDiffX = indexMaxArr[0] - indexMinArr[0];
            Lane8U32 binDiffY = indexMaxArr[1] - indexMinArr[1];
            Lane8U32 binDiffZ = indexMaxArr[2] - indexMinArr[2];
            // x0 x1 x2 x3 y0 y1 y2 y3 x4 x5 x6 x7 y4 y5 y6 y7
            Lane8U32 out0 = PackU32(binDiffX, binDiffY);
            // x0 x1 x2 x3 |  y0 y1 y2 y3 |  z0 00 z1 00 |  z2 00 z3 00 |  x4 x5 x6 x7 |  y4 y5 y6 y7 |  z4 00 z5 00 | z6 00 z7 00

            alignas(32) u8 bytes[32];
            Lane8U32 out1 = PackU16(out0, binDiffZ);
            Lane8U32::Store(bytes, out1);

            u32 bitMask[3] = {};

            static const u32 order[3][LANE_WIDTH] = {
                {0, 1, 2, 3, 16, 17, 18, 19},
                {4, 5, 6, 7, 20, 21, 22, 23},
                {8, 10, 12, 14, 24, 26, 28, 30},
            };

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
                            bins[dim][indexMin].Extend(data[refs[i + b]].m256);
                        }
                        break;
                        default:
                        {
                            bitMask[dim] |= (1 << diff);
                            faceIndices[dim][diff][binCounts[dim][diff]]   = data[refs[i + b]].primID;
                            binIndexStart[dim][diff][binCounts[dim][diff]] = indexMin;
                            binCounts[dim][diff]++;
                        }
                    }
                }
            }
            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 numIters = PopCount(bitMask[dim]);
                for (u32 iter = 0; iter < numIters; iter++)
                {
                    u32 bin = Bsf(bitMask[dim]);
                    if (binCounts[dim][bin] >= LANE_WIDTH)
                    {
                        Assert(binCounts[dim][bin] <= ArrayLength(binIndexStart[0][0]));
                        binCounts[dim][bin] -= LANE_WIDTH;
                        u32 binCount = binCounts[dim][bin];

                        Bounds8 bounds[2][LANE_WIDTH];
                        Triangle8 tri     = Triangle8::Load(mesh, dim, faceIndices[dim][bin] + binCount);
                        Lane8U32 startBin = Lane8U32::LoadU(binIndexStart[dim][bin] + binCount);

                        for (u32 boundIndex = 0; boundIndex < LANE_WIDTH; boundIndex++)
                        {
                            bounds[0][boundIndex] = Bounds8(pos_inf);
                        }
                        alignas(32) u32 binIndices[LANE_WIDTH];

                        u32 current = 0;
                        for (u32 d = 0; d < bin; d++)
                        {
                            Lane8U32::Store(binIndices, startBin);
                            startBin += 1u;
                            Lane8F32 splitPos = binner->GetSplitValue(startBin, dim);

                            ClipTriangle(mesh, dim, tri, splitPos, bounds[current], bounds[!current]);

                            for (u32 b = 0; b < LANE_WIDTH; b++)
                            {
                                u32 binIndex = binIndices[b];
                                bins[dim][binIndex].Extend(bounds[current][b]);
                            }
                            current = !current;
                        }
                        for (u32 b = 0; b < LANE_WIDTH; b++)
                        {
                            u32 binIndex = binIndices[b] + 1;
                            bins[dim][binIndex].Extend(bounds[current][b]);
                        }
                        binCounts[dim][bin] = 0;
                    }
                    bitMask[dim] &= bitMask[dim] - 1;
                }
            }
        }
        // Finish the remaining primitives
        for (; i < start + count; i++)
        {
            const PrimRef *ref = &data[refs[i]];

            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 binIndexMin = binner->BinMin(ref->min[dim], dim);
                u32 binIndexMax = binner->BinMax(ref->max[dim], dim);

                Assert(binIndexMax >= binIndexMin);

                u32 diff = binIndexMax - binIndexMin;
                entryCounts[binIndexMin][dim] += 1;
                exitCounts[binIndexMax][dim] += 1;
                switch (diff)
                {
                    case 0:
                    {
                        bins[dim][binIndexMin].Extend(ref->m256);
                    }
                    break;
                    default:
                    {
                        faceIndices[dim][diff][binCounts[dim][diff]]   = ref->primID;
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
                Assert(remainingCount <= ArrayLength(binIndexStart[0][0]));

                const u32 numIters = ((remainingCount + 7) >> 3);
                for (u32 remaining = 0; remaining < numIters; remaining++)
                {
                    u32 numPrims = Min(remainingCount, 8u);
                    Bounds8 bounds[2][8];
                    for (u32 boundIndex = 0; boundIndex < LANE_WIDTH; boundIndex++)
                    {
                        bounds[0][boundIndex] = Bounds8(pos_inf);
                    }
                    Triangle8 tri     = Triangle8::Load(mesh, dim, faceIndices[dim][diff] + remaining * LANE_WIDTH); //, bounds[0]);
                    Lane8U32 startBin = Lane8U32::LoadU(binIndexStart[dim][diff] + remaining * LANE_WIDTH);

                    alignas(32) u32 binIndices[8];

                    u32 current = 0;
                    for (u32 d = 0; d < diff; d++)
                    {
                        Lane8U32::Store(binIndices, startBin);
                        startBin += 1u;
                        Lane8F32 splitPos = binner->GetSplitValue(startBin, dim);
                        ClipTriangle(mesh, dim, tri, splitPos, bounds[current], bounds[!current]);
                        for (u32 b = 0; b < numPrims; b++)
                        {
                            u32 binIndex = binIndices[b];
                            bins[dim][binIndex].Extend(bounds[current][b]);
                        }
                        current = !current;
                    }
                    for (u32 b = 0; b < numPrims; b++)
                    {
                        u32 binIndex = binIndices[b] + 1;
                        bins[dim][binIndex].Extend(bounds[current][b]);
                    }
                    remainingCount -= LANE_WIDTH;
                }
            }
        }
    }
    // options:
    // 1. single pass
    //      a. with atomic and strided partitioning
    //      b. no atomic, "worse" partitioning
    // 2. double pass, calculate bounds while splitting
    // 3. double pass, calculate bounds while partitioning
    template <typename GetIndex>
    u32 Split(TriangleMesh *mesh, PrimRef *data, u32 dim, u32 bestPos, u32 l, u32 r,
              std::atomic<u32> &splitAtomic, GetIndex getIndex,
              RecordAOSSplits &outLeft, RecordAOSSplits &outRight)
    {
        u32 faceIDQueue[LANE_WIDTH * 2] = {};
        u32 refIDQueue[LANE_WIDTH * 2]  = {};
        Lane8F32 masks[2]               = {Lane8F32::Mask(false), Lane8F32::Mask(true)};

        Lane8F32 lanes[8];

        u32 splitCount = 0;

        Lane8F32 geomLeft(neg_inf);
        Lane8F32 geomRight(neg_inf);

        Lane8F32 centLeft(neg_inf);
        Lane8F32 centRight(neg_inf);

        u32 totalL = 0;
        u32 totalR = 0;

        Lane8F32 splitValue = binner->GetSplitValue(Lane8U32(bestPos), dim);

        for (;;)
        {
            bool lIsFullyRight = false;
            u32 lIndex;
            while (l <= r && !lIsFullyRight && splitCount < LANE_WIDTH)
            {
                lIndex            = getIndex(l);
                PrimRef &ref      = data[lIndex];
                Lane4F32 min      = Lane4F32::Load(ref.min);
                Lane4F32 max      = Lane4F32::Load(ref.max);
                Lane4F32 centroid = (max - min) * 0.5f;

                u32 minBin        = binner->BinMin(min[dim], dim);
                u32 maxBin        = binner->BinMax(max[dim], dim);
                lIsFullyRight     = minBin >= bestPos;
                bool lIsFullyLeft = maxBin < bestPos;
                bool isSplit      = !(lIsFullyRight || lIsFullyLeft); // minBin < bestPos && maxBin >= bestPos;
                Lane8F32 c(-centroid, centroid);

                centRight = MaskMax(masks[lIsFullyRight], centRight, c);
                geomRight = MaskMax(masks[lIsFullyRight], geomRight, ref.m256);
                centLeft  = MaskMax(masks[lIsFullyLeft], centLeft, c);
                geomLeft  = MaskMax(masks[lIsFullyLeft], geomLeft, ref.m256);

                faceIDQueue[splitCount] = ref.primID;
                refIDQueue[splitCount]  = lIndex;
                splitCount += isSplit;

                l += !lIsFullyRight;
            }

            u32 rPrimID        = 0;
            bool rIsSplit      = false;
            bool rIsFullyRight = true;
            u32 rIndex;
            while (l <= r && rIsFullyRight && splitCount < LANE_WIDTH)
            {
                rIndex            = getIndex(r);
                PrimRef &rRef     = data[rIndex];
                Lane4F32 min      = Lane4F32::Load(rRef.min);
                Lane4F32 max      = Lane4F32::Load(rRef.max);
                Lane4F32 centroid = (max - min) * 0.5f;

                u32 minBin       = binner->BinMin(min[dim], dim);
                u32 maxBin       = binner->BinMax(max[dim], dim);
                rIsFullyRight    = minBin >= bestPos;
                bool isFullyLeft = maxBin < bestPos;
                rIsSplit         = !(rIsFullyRight || isFullyLeft);
                Lane8F32 c(-centroid, centroid);
                rPrimID = rRef.primID;

                centRight = MaskMax(masks[rIsFullyRight], centRight, c);
                geomRight = MaskMax(masks[rIsFullyRight], geomRight, rRef.m256);
                centLeft  = MaskMax(masks[isFullyLeft], centLeft, c);
                geomLeft  = MaskMax(masks[isFullyLeft], geomLeft, rRef.m256);

                r -= rIsFullyRight;
            }

            if (l > r) break;
            faceIDQueue[splitCount] = rPrimID;
            refIDQueue[splitCount]  = lIndex;
            splitCount += rIsSplit;
            if (lIsFullyRight)
            {
                Swap(data[lIndex], data[rIndex]);
                l++;
                r--;
            }

            if (splitCount >= LANE_WIDTH)
            {
                splitCount -= LANE_WIDTH;
                u32 splitOffset = splitAtomic.fetch_add(LANE_WIDTH, std::memory_order_acq_rel);

                Lane8F32 gL[LANE_WIDTH];
                Lane8F32 gR[LANE_WIDTH];
                Lane8F32 cL[3];
                Lane8F32 cR[3];
                Lane8F32 outCentroidsL[8];
                Lane8F32 outCentroidsR[8];

                Triangle8 tri = Triangle8::Load(mesh, dim, faceIDQueue + splitCount);
                ClipTriangle(mesh, dim, faceIDQueue + splitCount, tri, splitValue, gL, gR, cL, cR);

                Transpose3x8(cL[0], cL[1], cL[2],
                             outCentroidsL[0], outCentroidsL[1], outCentroidsL[2], outCentroidsL[3],
                             outCentroidsL[4], outCentroidsL[5], outCentroidsL[6], outCentroidsL[7]);

                for (u32 queueIndex = 0; queueIndex < LANE_WIDTH; queueIndex++)
                {
                    const u32 refID    = refIDQueue[splitCount + queueIndex];
                    const u32 splitLoc = splitOffset++;
                    geomLeft           = Max(geomLeft, gL[queueIndex]);
                    geomRight          = Max(geomRight, gR[queueIndex]);
                    centLeft           = Max(centLeft, outCentroidsL[queueIndex]);
                    centRight          = Max(centRight, outCentroidsR[queueIndex]);

                    Lane8F32::Store(&data[refID].m256, gL[queueIndex]);
                    Lane8F32::Store(&data[splitLoc].m256, gR[queueIndex]);
                }
            }
        }

        // Flush the queue
        u32 remainingCount = splitCount;
        const u32 numIters = (remainingCount + 7) >> 3;
        u32 splitOffset    = splitAtomic.fetch_add(remainingCount, std::memory_order_acq_rel);
        for (u32 remaining = 0; remaining < numIters; remaining++)
        {
            u32 qStart   = remaining * LANE_WIDTH;
            u32 numPrims = Min(remainingCount, LANE_WIDTH);
            Lane8F32 gL[LANE_WIDTH];
            Lane8F32 gR[LANE_WIDTH];
            Lane8F32 cL[3];
            Lane8F32 cR[3];
            Lane8F32 outCentroidsL[8];
            Lane8F32 outCentroidsR[8];
            Triangle8 tri = Triangle8::Load(mesh, dim, faceIDQueue + qStart);
            ClipTriangle(mesh, dim, faceIDQueue + qStart, tri, splitValue, gL, gR, cL, cR);
            Transpose3x8(cL[0], cL[1], cL[2],
                         outCentroidsL[0], outCentroidsL[1], outCentroidsL[2], outCentroidsL[3],
                         outCentroidsL[4], outCentroidsL[5], outCentroidsL[6], outCentroidsL[7]);

            for (u32 queueIndex = 0; queueIndex < numPrims; queueIndex++)
            {
                const u32 refID    = refIDQueue[qStart + queueIndex];
                const u32 splitLoc = splitOffset++;
                geomLeft           = Max(geomLeft, gL[queueIndex]);
                geomRight          = Max(geomRight, gR[queueIndex]);
                centLeft           = Max(centLeft, outCentroidsL[queueIndex]);
                centRight          = Max(centRight, outCentroidsR[queueIndex]);

                Lane8F32::Store(&data[refID].m256, gL[queueIndex]);
                Lane8F32::Store(&data[splitLoc].m256, gR[queueIndex]);
            }

            remainingCount -= LANE_WIDTH;
        }
        // Assert(totalL == expectedL);
        // Assert(totalR == expectedR);
        // Assert(writeLocs[0] - outLStart == expectedL);
        // Assert(writeLocs[1] - outRStart == expectedR);
        outLeft.geomBounds  = geomLeft;
        outRight.geomBounds = geomRight;
        outLeft.centBounds  = centLeft;
        outRight.centBounds = centRight;
        return l;
    }

    // Splits and partitions at the same time
    void Split(TriangleMesh *mesh, PrimRef *data, const u32 *refs, u32 *outRefs, u32 outLStart, u32 outRStart, u32 splitOffset,
               u32 start, u32 count, struct Split split, RecordAOSSplits &outLeft, RecordAOSSplits &outRight,
               u32 expectedL, u32 expectedR)
    {
        u32 dim                         = split.bestDim;
        u32 alignedCount                = count - count % LANE_WIDTH;
        u32 faceIDQueue[LANE_WIDTH * 2] = {};
        u32 refIDQueue[LANE_WIDTH * 2]  = {};
        Lane8F32 masks[2]               = {Lane8F32::Mask(false), Lane8F32::Mask(true)};

        u32 i = start;
        Lane8F32 lanes[8];

        u32 splitCount = 0;

        Lane8F32 geomLeft(neg_inf);
        Lane8F32 geomRight(neg_inf);

        Bounds8F32 centLeft;
        Bounds8F32 centRight;

        u32 totalL = 0;
        u32 totalR = 0;

        u32 writeLocs[2]    = {outLStart, outRStart};
        Lane8F32 splitValue = binner->GetSplitValue(Lane8U32(split.bestPos), dim);
        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            Transpose8x8(data[refs[i + 0]].m256, data[refs[i + 1]].m256, data[refs[i + 2]].m256, data[refs[i + 3]].m256,
                         data[refs[i + 4]].m256, data[refs[i + 5]].m256, data[refs[i + 6]].m256, data[refs[i + 7]].m256,
                         lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7]);
            // See if the primitive needs to be split
            // Only add the primitive to the left and right bounds if it doesn't need to be split
            Lane8U32 faceIDs = AsUInt(lanes[7]);

            Lane8U32 binMin = binner->BinMin(lanes[dim], dim);
            Lane8U32 binMax = binner->BinMax(lanes[dim + 4], dim);
            Assert(All(AsFloat(binMax) >= AsFloat(binMin)));
            Lane8F32 isFullyLeftV  = AsFloat(binMax) < AsFloat(split.bestPos);
            Lane8F32 isFullyRightV = AsFloat(binMin) >= AsFloat(split.bestPos);

            Lane8F32 centroids[] = {
                (lanes[4] - lanes[0]) * 0.5f,
                (lanes[5] - lanes[1]) * 0.5f,
                (lanes[6] - lanes[2]) * 0.5f,
            };

            u32 lMaskBits = Movemask(isFullyLeftV);
            u32 rMaskBits = Movemask(isFullyRightV);
            // Extend next generation bounds
            centLeft.MaskExtend(isFullyLeftV, centroids[0], centroids[1], centroids[2]);
            centRight.MaskExtend(isFullyRightV, centroids[0], centroids[1], centroids[2]);
            u32 splitMask = (~(lMaskBits | rMaskBits)) & 0xff;

            for (u32 chunkIndex = 0; chunkIndex < LANE_WIDTH; chunkIndex++)
            {
                u32 refID        = refs[i + chunkIndex];
                u32 isFullyLeft  = (lMaskBits >> chunkIndex) & 1;
                u32 isFullyRight = (rMaskBits >> chunkIndex) & 1;

                geomLeft                           = MaskMax(masks[isFullyLeft], geomLeft, data[refID].m256);
                geomRight                          = MaskMax(masks[isFullyRight], geomRight, data[refID].m256);
                outRefs[writeLocs[isFullyRight]++] = refID;

                totalR += isFullyRight;
                totalL += !isFullyRight;
            }

            // Store primitives that need to be split
            Lane8U32 refIDs = Lane8U32::LoadU(refs + i);
            Lane8U32::StoreU(faceIDQueue + splitCount, MaskCompress(splitMask, faceIDs));
            Lane8U32::StoreU(refIDQueue + splitCount, MaskCompress(splitMask, refIDs));
            splitCount += PopCount(splitMask);

            if (splitCount >= LANE_WIDTH)
            {
                splitCount -= LANE_WIDTH;

                Lane8F32 gL[LANE_WIDTH];
                Lane8F32 gR[LANE_WIDTH];
                Lane8F32 cL[3];
                Lane8F32 cR[3];
                Triangle8 tri = Triangle8::Load(mesh, dim, faceIDQueue + splitCount);
                ClipTriangle(mesh, dim, faceIDQueue + splitCount, tri, splitValue, gL, gR, cL, cR);

                centLeft.Extend(cL[0], cL[1], cL[2]);
                centRight.Extend(cR[0], cR[1], cR[2]);

                // TODO: there could be a false sharing problem here. not a major one, but these writes invalidate
                // any previously cached PrimRefs, so all reads at the beginning of the outer loop are a cache miss
                for (u32 queueIndex = 0; queueIndex < LANE_WIDTH; queueIndex++)
                {
                    const u32 refID    = refIDQueue[splitCount + queueIndex];
                    const u32 splitLoc = splitOffset++;
                    geomLeft           = Max(geomLeft, gL[queueIndex]);
                    geomRight          = Max(geomRight, gR[queueIndex]);

                    Lane8F32::Store(&data[refID].m256, gL[queueIndex]);
                    Lane8F32::Store(&data[splitLoc].m256, gR[queueIndex]);
                    outRefs[writeLocs[1]++] = splitLoc;
                    totalR++;
                }
            }
        }
        for (; i < start + count; i++)
        {
            const PrimRef *primRef = &data[refs[i]];
            Lane4F32 min           = Extract4<0>(primRef->m256);
            Lane4F32 max           = Extract4<1>(primRef->m256);

            bool isFullyLeft  = binner->BinMax(primRef->max[dim], dim) < split.bestPos;
            bool isFullyRight = binner->BinMin(primRef->min[dim], dim) >= split.bestPos;
            Lane4F32 centroid = (max - min) * 0.5f;
            bool isSplit      = !(isFullyLeft || isFullyRight);

            Lane8F32 c(-centroid, centroid);
            faceIDQueue[splitCount] = primRef->primID;
            refIDQueue[splitCount]  = refs[i];
            splitCount += isSplit;
            geomLeft  = MaskMax(masks[isFullyLeft], geomLeft, primRef->m256);
            geomRight = MaskMax(masks[isFullyRight], geomRight, primRef->m256);

            centLeft.MaskExtend(masks[isFullyLeft], centroid[0], centroid[1], centroid[2]);
            centRight.MaskExtend(masks[isFullyRight], centroid[0], centroid[1], centroid[2]);

            totalL += !isFullyRight;
            totalR += isFullyRight;
            outRefs[writeLocs[isFullyRight]++] = refs[i];
        }
        // Flush the queue
        u32 remainingCount = splitCount;
        const u32 numIters = (remainingCount + 7) >> 3;
        for (u32 remaining = 0; remaining < numIters; remaining++)
        {
            u32 qStart   = remaining * LANE_WIDTH;
            u32 numPrims = Min(remainingCount, LANE_WIDTH);
            Lane8F32 gL[LANE_WIDTH];
            Lane8F32 gR[LANE_WIDTH];
            Lane8F32 cL[3];
            Lane8F32 cR[3];
            Triangle8 tri = Triangle8::Load(mesh, dim, faceIDQueue + qStart);
            ClipTriangle(mesh, dim, faceIDQueue + qStart, tri, splitValue, gL, gR, cL, cR);

            Lane8F32 mask = Lane8F32::Mask((1 << numPrims) - 1u);
            centLeft.MaskExtend(mask, cL[0], cL[1], cL[2]);
            centRight.MaskExtend(mask, cR[0], cR[1], cR[2]);
            for (u32 queueIndex = 0; queueIndex < numPrims; queueIndex++)
            {
                const u32 refID    = refIDQueue[qStart + queueIndex];
                const u32 splitLoc = splitOffset++;
                geomLeft           = Max(geomLeft, gL[queueIndex]);
                geomRight          = Max(geomRight, gR[queueIndex]);

                Lane8F32::Store(&data[refID].m256, gL[queueIndex]);
                Lane8F32::Store(&data[splitLoc].m256, gR[queueIndex]);
                outRefs[writeLocs[1]++] = splitLoc;
                totalR++;
            }

            remainingCount -= LANE_WIDTH;
        }
        Assert(totalL == expectedL);
        Assert(totalR == expectedR);
        Assert(writeLocs[0] - outLStart == expectedL);
        Assert(writeLocs[1] - outRStart == expectedR);
        outLeft.geomBounds  = geomLeft;
        outRight.geomBounds = geomRight;
        outLeft.centBounds  = Lane8F32(-ReduceMin(centLeft.minU), -ReduceMin(centLeft.minV), -ReduceMin(centLeft.minW), pos_inf,
                                       ReduceMax(centLeft.maxU), ReduceMax(centLeft.maxV), ReduceMax(centLeft.maxW), pos_inf);
        outRight.centBounds = Lane8F32(-ReduceMin(centRight.minU), -ReduceMin(centRight.minV), -ReduceMin(centRight.minW), pos_inf,
                                       ReduceMax(centRight.maxU), ReduceMax(centRight.maxV), ReduceMax(centRight.maxW), pos_inf);
    }

    void Merge(const HeuristicAOSSplitBinning<numBins> &other)
    {
        for (u32 i = 0; i < numBins; i++)
        {
            entryCounts[i] += other.entryCounts[i];
            exitCounts[i] += other.exitCounts[i];
            for (u32 dim = 0; dim < 3; dim++)
            {
                bins[dim][i].Extend(other.bins[dim][i]);
            }
        }
    }
};

// SBVH
static const f32 sbvhAlpha = 1e-5;
template <i32 numObjectBins = 32, i32 numSpatialBins = 16>
struct HeuristicSpatialSplits
{
    using Record = RecordAOSSplits;
    using HSplit = HeuristicAOSSplitBinning<numSpatialBins>;
    using OBin   = HeuristicAOSObjectBinning<numObjectBins>;

    TriangleMesh *mesh;
    f32 rootArea;
    PrimRef *primRefs;
    u32 *refs[2];

    std::atomic<u32> splitOffset;

    HeuristicSpatialSplits() {}
    HeuristicSpatialSplits(PrimRef *data, u32 *refs0, u32 *refs1, TriangleMesh *mesh, f32 rootArea, u32 splitOffset)
        : primRefs(data), refs{refs0, refs1}, mesh(mesh), rootArea(rootArea), splitOffset{splitOffset} {}

    Split Bin(const Record &record, u32 primary, u32 blockSize = 1)
    {
        // TIMED_FUNCTION(miscF);
        // Object splits
        TempArena temp    = ScratchStart(0, 0);
        u64 popPos        = ArenaPos(temp.arena);
        temp.arena->align = 32;

        // Stack allocate the heuristics since they store the centroid and geom bounds we'll need later
        ObjectBinner<numObjectBins> *objectBinner =
            PushStructConstruct(temp.arena, ObjectBinner<numObjectBins>)(record.centBounds);
        OBin *objectBinHeuristic = PushStructNoZero(temp.arena, OBin);
        ParallelForOutput objectOutput;
        if (record.count > PARALLEL_THRESHOLD)
        {
            const u32 groupSize = PARALLEL_THRESHOLD;
            objectOutput        = ParallelFor<OBin>(
                temp, record.start, record.count, groupSize,
                [&](OBin &binner, u32 start, u32 count) { binner.Bin(primRefs, refs[primary], start, count); },
                objectBinner);
            Reduce<OBin>(
                *objectBinHeuristic, objectOutput,
                [&](OBin &l, const OBin &r) { l.Merge(r); },
                objectBinner);
        }
        else
        {
            *objectBinHeuristic = OBin(objectBinner);
            objectBinHeuristic->Bin(primRefs, refs[primary], record.start, record.count);
            objectOutput.num       = 1;
            objectOutput.out       = objectBinHeuristic;
            objectOutput.groupSize = record.count;
        }
        struct Split objectSplit = BinBest(objectBinHeuristic->bins, objectBinHeuristic->counts, objectBinner);
        objectSplit.type         = Split::Object;

        Bounds8 geomBoundsL;
        for (u32 i = 0; i < objectSplit.bestPos; i++)
        {
            geomBoundsL.Extend(objectBinHeuristic->bins[objectSplit.bestDim][i]);
        }
        Bounds8 geomBoundsR;
        for (u32 i = objectSplit.bestPos; i < numObjectBins; i++)
        {
            geomBoundsR.Extend(objectBinHeuristic->bins[objectSplit.bestDim][i]);
        }

        f32 lambda = HalfArea(Intersect(geomBoundsL, geomBoundsR));
        if (lambda > sbvhAlpha * rootArea)
        {
            // Spatial splits
            SplitBinner<numSpatialBins> *splitBinner =
                PushStructConstruct(temp.arena, SplitBinner<numSpatialBins>)(record.geomBounds);

            HSplit *splitHeuristic = PushStructNoZero(temp.arena, HSplit);
            ParallelForOutput splitOutput;
            if (record.count > PARALLEL_THRESHOLD)
            {
                const u32 groupSize = PARALLEL_THRESHOLD;

                splitOutput = ParallelFor<HSplit>(
                    temp, record.start, record.count, groupSize,
                    [&](HSplit &binner, u32 start, u32 count) { binner.Bin(mesh, primRefs, refs[primary], start, count); },
                    splitBinner);
                Reduce<HSplit>(
                    *splitHeuristic, splitOutput,
                    [&](HSplit &l, const HSplit &r) { l.Merge(r); },
                    splitBinner);
            }
            else
            {
                *splitHeuristic = HSplit(splitBinner);
                splitHeuristic->Bin(mesh, primRefs, refs[primary], record.start, record.count);
                splitOutput.out       = splitHeuristic;
                splitOutput.num       = 1;
                splitOutput.groupSize = record.count;
            }
            struct Split spatialSplit = BinBest(splitHeuristic->bins,
                                                splitHeuristic->entryCounts, splitHeuristic->exitCounts, splitBinner);
            spatialSplit.type         = Split::Spatial;
            u32 lCount                = 0;
            for (u32 i = 0; i < spatialSplit.bestPos; i++)
            {
                lCount += splitHeuristic->entryCounts[i][spatialSplit.bestDim];
            }
            u32 rCount = 0;
            for (u32 i = spatialSplit.bestPos; i < numSpatialBins; i++)
            {
                rCount += splitHeuristic->exitCounts[i][spatialSplit.bestDim];
            }
            u32 totalNumSplits = lCount + rCount - record.count;
            if (spatialSplit.bestSAH < objectSplit.bestSAH && totalNumSplits <= record.ExtSize())
            {
                u32 threadSplitOffset      = splitOffset.fetch_add(totalNumSplits, std::memory_order_acq_rel);
                u32 threadSplitOffsetStart = threadSplitOffset;

                u32 *lOffsets     = PushArrayNoZero(temp.arena, u32, splitOutput.num);
                u32 *rOffsets     = PushArrayNoZero(temp.arena, u32, splitOutput.num);
                u32 *lCounts      = PushArrayNoZero(temp.arena, u32, splitOutput.num);
                u32 *rCounts      = PushArrayNoZero(temp.arena, u32, splitOutput.num);
                u32 *splitOffsets = PushArrayNoZero(temp.arena, u32, splitOutput.num);

                u32 lOffset = record.extStart;
                u32 rOffset = record.extEnd;
                u32 rTotal  = 0;
                u32 lTotal  = 0;
                for (u32 i = 0; i < splitOutput.num; i++)
                {
                    HSplit *h       = &((HSplit *)splitOutput.out)[i];
                    lOffsets[i]     = lOffset;
                    splitOffsets[i] = threadSplitOffset;
                    u32 entryCounts = 0;
                    u32 exitCounts  = 0;
                    for (u32 bin = 0; bin < spatialSplit.bestPos; bin++)
                    {
                        entryCounts += h->entryCounts[bin][spatialSplit.bestDim];
                    }
                    for (u32 bin = spatialSplit.bestPos; bin < numSpatialBins; bin++)
                    {
                        exitCounts += h->exitCounts[bin][spatialSplit.bestDim];
                    }

                    u32 groupSize = i == splitOutput.num - 1
                                        ? record.End() - (i * splitOutput.groupSize + record.start)
                                        : splitOutput.groupSize;

                    threadSplitOffset += (entryCounts + exitCounts - groupSize);

                    lCounts[i] = entryCounts;
                    rCounts[i] = exitCounts;

                    lOffset += entryCounts;
                    rOffset -= exitCounts;

                    lTotal += entryCounts;
                    rTotal += exitCounts;

                    rOffsets[i] = rOffset;
                }
                Assert(threadSplitOffset - threadSplitOffsetStart == totalNumSplits);
                spatialSplit.partitionPayload = PartitionPayload(lOffsets, rOffsets, lCounts, rCounts,
                                                                 splitOutput.num, splitOutput.groupSize);
                spatialSplit.ptr              = (void *)splitHeuristic;
                spatialSplit.splitOffsets     = splitOffsets;
                spatialSplit.allocPos         = popPos;
                spatialSplit.numLeft          = lTotal;
                spatialSplit.numRight         = rTotal;

                return spatialSplit;
            }
        }

        u32 *lOffsets = PushArrayNoZero(temp.arena, u32, objectOutput.num);
        u32 *rOffsets = PushArrayNoZero(temp.arena, u32, objectOutput.num);
        u32 *lCounts  = PushArrayNoZero(temp.arena, u32, objectOutput.num);
        u32 *rCounts  = PushArrayNoZero(temp.arena, u32, objectOutput.num);

        u32 lOffset = record.extStart;
        u32 rOffset = record.extEnd;
        u32 lTotal  = 0;
        u32 rTotal  = 0;
        for (u32 i = 0; i < objectOutput.num; i++)
        {
            OBin *h         = &((OBin *)objectOutput.out)[i];
            lOffsets[i]     = lOffset;
            u32 entryCounts = 0;
            u32 exitCounts  = 0;
            for (u32 bin = 0; bin < objectSplit.bestPos; bin++)
            {
                entryCounts += h->counts[bin][objectSplit.bestDim];
            }
            for (u32 bin = objectSplit.bestPos; bin < numObjectBins; bin++)
            {
                exitCounts += h->counts[bin][objectSplit.bestDim];
            }

            lCounts[i] = entryCounts;
            rCounts[i] = exitCounts;

            lOffset += entryCounts;
            rOffset -= exitCounts;

            lTotal += entryCounts;
            rTotal += exitCounts;

            rOffsets[i] = rOffset;
        }

        objectSplit.partitionPayload = PartitionPayload(lOffsets, rOffsets, lCounts, rCounts,
                                                        objectOutput.num, objectOutput.groupSize);
        objectSplit.ptr              = (void *)objectBinHeuristic;
        objectSplit.allocPos         = popPos;
        objectSplit.numLeft          = lTotal;
        objectSplit.numRight         = rTotal;
        return objectSplit;
    }
    void FlushState(struct Split split)
    {
        TempArena temp = ScratchStart(0, 0);
        ArenaPopTo(temp.arena, split.allocPos);
    }
    void Split(struct Split split, u32 current, const Record &record, Record &outLeft, Record &outRight)
    {
        // NOTE: Split must be called from the same thread as Bin
        TempArena temp = ScratchStart(0, 0);

        u32 *inRefs  = refs[current];
        u32 *outRefs = refs[!current];

        if (split.bestSAH == f32(pos_inf))
        {
            u32 lCount = record.count / 2;
            u32 rCount = record.count - lCount;
            u32 mid    = record.start + lCount;
            Bounds8 geomLeft;
            Bounds8 centLeft;
            Bounds8 geomRight;
            Bounds8 centRight;
            u32 lOffset = record.extStart;
            u32 rOffset = record.extEnd - rCount;
            for (u32 i = record.start; i < mid; i++) // record.start + record.count; i++)
            {
                PrimRef *ref      = &primRefs[inRefs[i]];
                Lane8F32 m256     = Lane8F32::Load(&ref->m256);
                Lane8F32 centroid = ((Shuffle4<1, 1>(m256) - Shuffle4<0, 0>(m256)) * 0.5f) ^ signFlipMask;
                geomLeft.Extend(m256);
                centLeft.Extend(centroid);
                outRefs[lOffset++] = inRefs[i];
            }
            for (u32 i = mid; i < record.End(); i++)
            {
                PrimRef *ref      = &primRefs[inRefs[i]];
                Lane8F32 m256     = Lane8F32::Load(&ref->m256);
                Lane8F32 centroid = ((Shuffle4<1, 1>(m256) - Shuffle4<0, 0>(m256)) * 0.5f) ^ signFlipMask;
                geomRight.Extend(m256);
                centRight.Extend(centroid);
                outRefs[rOffset++] = inRefs[i];
            }
            outLeft.geomBounds  = geomLeft.v;
            outLeft.centBounds  = centLeft.v;
            outRight.geomBounds = geomRight.v;
            outRight.centBounds = centRight.v;
        }
        else
        {
            PartitionPayload &payload     = split.partitionPayload;
            RecordAOSSplits *leftRecords  = PushArrayNoZero(temp.arena, RecordAOSSplits, payload.count);
            RecordAOSSplits *rightRecords = PushArrayNoZero(temp.arena, RecordAOSSplits, payload.count);
            switch (split.type)
            {
                case Split::Object:
                {
                    OBin *heuristic = (OBin *)(split.ptr);

                    if (payload.count > 1)
                    {
                        scheduler.ScheduleAndWait(payload.count, 1, [&](u32 jobID) {
                            u32 threadStart = record.start + payload.groupSize * jobID;
                            u32 count       = jobID == payload.count - 1 ? record.End() - threadStart : payload.groupSize;

                            Partition(payload.lOffsets[jobID], payload.rOffsets[jobID], heuristic->binner,
                                      threadStart, count, primRefs, inRefs, outRefs, split.bestDim, split.bestPos,
                                      leftRecords[jobID], rightRecords[jobID],
                                      payload.lCounts[jobID], payload.rCounts[jobID]);
                        });
                    }
                    else
                    {
                        Assert(record.count == payload.groupSize);
                        Assert(record.extStart == payload.lOffsets[0]);
                        Assert(record.extEnd - split.numRight == payload.rOffsets[0]);
                        Partition(record.extStart, record.extEnd - split.numRight, heuristic->binner,
                                  record.start, record.count, primRefs, inRefs, outRefs, split.bestDim, split.bestPos,
                                  leftRecords[0], rightRecords[0], split.numLeft, split.numRight);
                    }
                }
                break;
                case Split::Spatial:
                {
                    HSplit *heuristic = (HSplit *)(split.ptr);

                    if (payload.count > 1)
                    {
                        scheduler.ScheduleAndWait(payload.count, 1, [&](u32 jobID) {
                            u32 threadStart = record.start + payload.groupSize * jobID;
                            u32 count       = jobID == payload.count - 1 ? record.End() - threadStart : payload.groupSize;
                            heuristic->Split(mesh, primRefs, inRefs, outRefs, payload.lOffsets[jobID], payload.rOffsets[jobID],
                                             split.splitOffsets[jobID], threadStart, count, split, leftRecords[jobID], rightRecords[jobID],
                                             payload.lCounts[jobID], payload.rCounts[jobID]);
                        });
                    }
                    else
                    {
                        Assert(record.count == payload.groupSize);
                        Assert(record.extStart == payload.lOffsets[0]);
                        Assert(record.extEnd - split.numRight == payload.rOffsets[0]);

                        heuristic->Split(mesh, primRefs, inRefs, outRefs, payload.lOffsets[0], payload.rOffsets[0],
                                         split.splitOffsets[0], record.start, record.count, split, leftRecords[0], rightRecords[0],
                                         split.numLeft, split.numRight);
                    }
                }
                break;
            }
            Lane8F32 geomLeft(neg_inf);
            Lane8F32 geomRight(neg_inf);
            Lane8F32 centLeft(neg_inf);
            Lane8F32 centRight(neg_inf);
            for (u32 i = 0; i < payload.count; i++)
            {
                geomLeft  = Max(geomLeft, leftRecords[i].geomBounds);
                centLeft  = Max(centLeft, leftRecords[i].centBounds);
                geomRight = Max(geomRight, rightRecords[i].geomBounds);
                centRight = Max(centRight, rightRecords[i].centBounds);
            }
            outLeft.geomBounds  = geomLeft;
            outLeft.centBounds  = centLeft;
            outRight.geomBounds = geomRight;
            outRight.centBounds = centRight;
        }
        u32 numLeft  = split.numLeft;
        u32 numRight = split.numRight;

        f32 weight         = (f32)(numLeft) / (numLeft + numRight);
        u32 remainingSpace = (record.extEnd - record.extStart - numLeft - numRight);
        u32 extSizeLeft    = Min((u32)(remainingSpace * weight), remainingSpace);
        u32 extSizeRight   = remainingSpace - extSizeLeft;

        Assert(numLeft <= record.count);
        Assert(numRight <= record.count);

        outLeft.SetRange(record.extStart, record.extStart, numLeft, record.extStart + numLeft + extSizeLeft);
        outRight.SetRange(outLeft.extEnd, outLeft.extEnd + extSizeRight, numRight, record.extEnd);
        ArenaPopTo(temp.arena, split.allocPos);

        // error check
#if 0
        {
            for (u32 i = outLeft.start; i < outLeft.End(); i++)
            {
                const PrimRef *ref = &primRefs[outRefs[i]];
                Lane4F32 min       = Extract4<0>(ref->m256);
                Lane4F32 max       = Extract4<1>(ref->m256);
                Lane4F32 centroid  = (max - min) * 0.5f;
                // Assert(centroid[split.bestDim] < split.bestValue);
                Lane8F32 c(-centroid, centroid);
                Assert((Movemask(ref->m256 <= outLeft.geomBounds) & 0x77) == 0x77);
                Assert((Movemask(c <= outLeft.centBounds) & 0x77) == 0x77);
            }
            for (u32 i = outRight.start; i < outRight.End(); i++)
            {
                const PrimRef *ref = &primRefs[outRefs[i]];
                Lane4F32 min       = Extract4<0>(ref->m256);
                Lane4F32 max       = Extract4<1>(ref->m256);
                Lane4F32 centroid  = (max - min) * 0.5f;
                // Assert(centroid[split.bestDim] >= split.bestValue);
                Lane8F32 c(-centroid, centroid);
                Assert((Movemask(ref->m256 <= outRight.geomBounds) & 0x77) == 0x77);
                Assert((Movemask(c <= outRight.centBounds) & 0x77) == 0x77);
            }
        }
#endif
    }
};

template <typename Binner, i32 numBins>
Split BinBest(const Bounds8 bounds[3][numBins],
              const Lane4U32 *entryCounts,
              const Lane4U32 *exitCounts,
              const Binner *binner)
{

    Lane4F32 areas[numBins];
    Lane4U32 counts[numBins];
    Lane4U32 currentCount = 0;

    Bounds8 boundsX;
    Bounds8 boundsY;
    Bounds8 boundsZ;
    for (u32 i = 0; i < numBins - 1; i++)
    {
        currentCount += entryCounts[i];
        counts[i] = currentCount;

        boundsX.Extend(bounds[0][i]);
        boundsY.Extend(bounds[1][i]);
        boundsZ.Extend(bounds[2][i]);

        Lane4F32 minX, minY, minZ;
        Lane4F32 maxX, maxY, maxZ;
        Transpose3x3(Extract4<0>(boundsX.v), Extract4<0>(boundsY.v), Extract4<0>(boundsZ.v), minX, minY, minZ);
        Transpose3x3(Extract4<1>(boundsX.v), Extract4<1>(boundsY.v), Extract4<1>(boundsZ.v), maxX, maxY, maxZ);

        Lane4F32 extentX = maxX + minX;
        Lane4F32 extentY = maxY + minY;
        Lane4F32 extentZ = maxZ + minZ;

        areas[i] = FMA(extentX, extentY + extentZ, extentY * extentZ);
    }
    boundsX      = Bounds8();
    boundsY      = Bounds8();
    boundsZ      = Bounds8();
    currentCount = 0;
    Lane4F32 bestSAH(pos_inf);
    Lane4U32 bestPos(0);
    for (u32 i = numBins - 1; i >= 1; i--)
    {
        currentCount += exitCounts[i];

        boundsX.Extend(bounds[0][i]);
        boundsY.Extend(bounds[1][i]);
        boundsZ.Extend(bounds[2][i]);

        Lane4F32 minX, minY, minZ;
        Lane4F32 maxX, maxY, maxZ;
        Transpose3x3(Extract4<0>(boundsX.v), Extract4<0>(boundsY.v), Extract4<0>(boundsZ.v), minX, minY, minZ);
        Transpose3x3(Extract4<1>(boundsX.v), Extract4<1>(boundsY.v), Extract4<1>(boundsZ.v), maxX, maxY, maxZ);

        Lane4F32 extentX = maxX + minX;
        Lane4F32 extentY = maxY + minY;
        Lane4F32 extentZ = maxZ + minZ;

        Lane4F32 rArea = FMA(extentX, extentY + extentZ, extentY * extentZ);

        Lane4F32 sah = FMA(rArea, Lane4F32(currentCount), areas[i - 1] * Lane4F32(counts[i - 1]));

        bestPos = Select(sah < bestSAH, Lane4U32(i), bestPos);
        bestSAH = Select(sah < bestSAH, sah, bestSAH);
    }

    u32 bestDim       = 0;
    f32 bestSAHScalar = pos_inf;
    u32 bestPosScalar = 0;
    for (u32 dim = 0; dim < 3; dim++)
    {
        if (binner->scale[dim][0] == 0) continue;
        if (bestSAH[dim] < bestSAHScalar)
        {
            bestPosScalar = bestPos[dim];
            bestDim       = dim;
            bestSAHScalar = bestSAH[dim];
        }
    }
    f32 bestValue = binner->GetSplitValue(bestPosScalar, bestDim);
    return Split(bestSAHScalar, bestPosScalar, bestDim, bestValue);
}

template <typename Binner, i32 numBins>
Split BinBest(const Bounds8 bounds[3][numBins],
              const Lane4U32 *counts,
              const Binner *binner)
{
    return BinBest(bounds, counts, counts, binner);
}

} // namespace rt
#endif
