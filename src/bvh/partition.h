#ifndef BVH_PARTITION_H
#define BVH_PARTITION_H

namespace rt
{
void Swap(PrimDataSOA *data, u32 lIndex, u32 rIndex)
{
    ::rt::Swap(data->minX[lIndex], data->minX[rIndex]);
    ::rt::Swap(data->minY[lIndex], data->minY[rIndex]);
    ::rt::Swap(data->minZ[lIndex], data->minZ[rIndex]);
    ::rt::Swap(data->geomIDs[lIndex], data->geomIDs[rIndex]);
    ::rt::Swap(data->maxX[lIndex], data->maxX[rIndex]);
    ::rt::Swap(data->maxY[lIndex], data->maxY[rIndex]);
    ::rt::Swap(data->maxZ[lIndex], data->maxZ[rIndex]);
    ::rt::Swap(data->primIDs[lIndex], data->primIDs[rIndex]);
}
void Swap(PrimData *prims, u32 lIndex, u32 rIndex)
{
    Swap(prims[lIndex], prims[rIndex]);
}

struct PartitionResult
{
    Bounds geomBoundsL;
    Bounds geomBoundsR;

    Bounds centBoundsL;
    Bounds centBoundsR;

    u32 mid;

    PartitionResult() : geomBoundsL(Bounds()), geomBoundsR(Bounds()), centBoundsL(Bounds()), centBoundsR(Bounds()) {}
    PartitionResult(Bounds &gL, Bounds &gR, Bounds &cL, Bounds &cR, u32 mid)
        : geomBoundsL(gL), geomBoundsR(gR), centBoundsL(cL), centBoundsR(cR), mid(mid) {}

    void Extend(const PartitionResult &other)
    {
        geomBoundsL.Extend(other.geomBoundsL);
        geomBoundsR.Extend(other.geomBoundsR);
        centBoundsL.Extend(other.centBoundsL);
        centBoundsR.Extend(other.centBoundsR);
    }
};

struct Range
{
    u32 start;
    u32 end;
    u32 group;
    __forceinline u32 Size() const
    {
        if (end > start)
            return end - start;
        return 0;
    }
};

struct AdditionalMisplacedRanges
{
    Range leftRange;
    Range rightRange;
};

template <typename PrimitiveData, typename GetIndex>
u32 FixMisplacedRanges(PrimitiveData *data, u32 numJobs, u32 chunkSize, u32 blockShift,
                       u32 blockSize, u32 globalMid, u32 *mids, GetIndex getIndex, AdditionalMisplacedRanges *add = 0)
{
    TempArena temp              = ScratchStart(0, 0);
    u32 numMisplacedRanges      = add ? numJobs + 1 : numJobs;
    Range *leftMisplacedRanges  = PushArray(temp.arena, Range, numMisplacedRanges);
    u32 lCount                  = 0;
    Range *rightMisplacedRanges = PushArray(temp.arena, Range, numMisplacedRanges);
    u32 rCount                  = 0;

    u32 numMisplacedLeft  = 0;
    u32 numMisplacedRight = 0;

    if (add)
    {
        numMisplacedLeft += add->leftRange.Size();
        numMisplacedRight += add->rightRange.Size();
        leftMisplacedRanges[0]        = add->leftRange;
        leftMisplacedRanges[0].group  = 0xffffffff;
        rightMisplacedRanges[0]       = add->rightRange;
        rightMisplacedRanges[0].group = 0xffffffff;
        lCount++;
        rCount++;
    }

    for (u32 i = 0; i < numJobs; i++)
    {
        u32 globalIndex = getIndex(mids[i], i);
        if (globalIndex < globalMid)
        {
            u32 diff = (((globalMid - globalIndex) / chunkSize) << blockShift) + ((globalMid - globalIndex) & (blockSize - 1));

            u32 check = getIndex(mids[i] + diff - 1, i);

            while (check > globalMid)
            {
                diff -= 1;
                check = getIndex(mids[i] + diff - 1, i);
            }
            int steps1 = 0;
            check      = getIndex(mids[i] + diff + 1, i);
            while (check < globalMid)
            {
                steps1++;
                diff += 1;
                check = getIndex(mids[i] + diff + 1, i);
            }
            if (steps1) diff++;

            rightMisplacedRanges[rCount] = {mids[i], mids[i] + diff, i};
            numMisplacedRight += rightMisplacedRanges[rCount].Size();
            rCount++;
        }
        else if (globalIndex > globalMid)
        {
            u32 diff = (((globalIndex - globalMid) / chunkSize) << blockShift) + ((globalIndex - globalMid) & (blockSize - 1));
            Assert(diff <= mids[i]);
            u32 check  = getIndex(mids[i] - diff + 1, i);
            int steps0 = 0;
            while (check < globalMid)
            {
                diff -= 1;
                steps0++;
                check = getIndex(mids[i] - diff + 1, i);
            }
            if (steps0) diff--;

            check = getIndex(mids[i] - diff - 1, i);
            while (check >= globalMid)
            {
                diff += 1;
                check = getIndex(mids[i] - diff - 1, i);
            }
            leftMisplacedRanges[lCount] = {mids[i] - diff, mids[i], i};
            numMisplacedLeft += leftMisplacedRanges[lCount].Size();
            lCount++;
        }
    }

    Assert(numMisplacedLeft == numMisplacedRight);

    u32 leftIndex  = 0;
    u32 rightIndex = 0;

    Range &lRange = leftMisplacedRanges[leftIndex];
    u32 lSize     = lRange.Size();
    u32 lIter     = 0;

    Range &rRange = rightMisplacedRanges[rightIndex];
    u32 rIter     = 0;
    u32 rSize     = rRange.Size();

    for (;;)
    {
        while (lSize != lIter && rSize != rIter)
        {
            u32 lIndex = lRange.start + lIter;
            u32 rIndex = rRange.start + rIter;
            if (lRange.group != 0xffffffff)
            {
                lIndex = getIndex(lIndex, lRange.group);
            }
            if (rRange.group != 0xffffffff)
            {
                rIndex = getIndex(rIndex, rRange.group);
            }

            Swap(data, lIndex, rIndex);

            lIter++;
            rIter++;
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
    return globalMid;
}

//////////////////////////////
// Prim Data Partitions
//
void PartitionSerial(Split split, PrimData *prims, u32 start, u32 end, PartitionResult *result)
{
    const u32 bestPos   = split.bestPos;
    const u32 bestDim   = split.bestDim;
    const f32 bestValue = split.bestValue;
    u32 l               = start;
    u32 r               = end - 1;

    Bounds &gL = result->geomBoundsL;
    Bounds &gR = result->geomBoundsR;

    Bounds &cL = result->centBoundsL;
    Bounds &cR = result->centBoundsR;

    gL = Bounds();
    gR = Bounds();

    cL = Bounds();
    cR = Bounds();

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
            if (lCentroid[bestDim] >= bestValue) break;
            gL.Extend(lPrim->minP, lPrim->maxP);
            cL.Extend(lCentroid);
            l++;
        } while (l <= r);
        do
        {
            rPrim     = &prims[r];
            rCentroid = (rPrim->minP + rPrim->maxP) * 0.5f;
            if (rCentroid[bestDim] < bestValue) break;
            gR.Extend(rPrim->minP, rPrim->maxP);
            cR.Extend(rCentroid);
            r--;
        } while (l <= r);
        if (l > r) break;

        Swap(lPrim->minP, rPrim->minP);
        Swap(lPrim->maxP, rPrim->maxP);

        gL.Extend(lPrim->minP, lPrim->maxP);
        gR.Extend(rPrim->minP, rPrim->maxP);

        cL.Extend(rCentroid);
        cR.Extend(lCentroid);
        l++;
        r--;
    }

    result->mid = r + 1;
}

void PartitionParallel(Split split, PrimData *prims, u32 start, u32 end, PartitionResult *result)
{
    u32 total = end - start;
    if (total < PARALLEL_THRESHOLD)
    {
        PartitionSerial(split, prims, start, end, result);
        return;
    }

    TempArena temp = ScratchStart(0, 0);

    // TODO: hardcoded
    const u32 numJobs = 64;

    const u32 blockSize = 16;
    StaticAssert(IsPow2(blockSize), Pow2BlockSize);

    const u32 blockShift        = Bsf(blockSize);
    const u32 numBlocksPerChunk = numJobs;
    const u32 chunkSize         = numBlocksPerChunk << blockShift;
    const u32 numChunks         = (total + chunkSize - 1) / chunkSize;
    const u32 bestDim           = split.bestDim;
    const f32 bestValue         = split.bestValue;

    // Index of first element greater than the pivot
    u32 *vi   = PushArray(temp.arena, u32, numJobs);
    u32 *ends = PushArray(temp.arena, u32, numJobs);
    u32 *mids = PushArray(temp.arena, u32, numJobs);

    PartitionResult *results = PushArrayDefault<PartitionResult>(temp.arena, numJobs);

    Scheduler::Counter counter = {};
    scheduler.Schedule(
        &counter, numJobs, 1, [&](u32 jobID) {
            clock_t start   = clock();
            const u32 group = jobID;
            auto GetIndex   = [&](u32 index) {
                const u32 chunkIndex   = index >> blockShift;
                const u32 blockIndex   = group;
                const u32 indexInBlock = index & (blockSize - 1);

                return chunkIndex * chunkSize + blockSize * blockIndex + indexInBlock;
            };

            u32 l          = 0;
            u32 r          = blockSize * numChunks - 1;
            u32 lastRIndex = GetIndex(r);
            r              = lastRIndex >= total
                                 ? (lastRIndex - total) < (blockSize - 1)
                                       ? r - (lastRIndex - total) - 1
                                       : r - (r & (blockSize - 1)) - 1
                                 : r;
            ends[jobID]    = r;

            Bounds &gL = results[group].geomBoundsL;
            Bounds &gR = results[group].geomBoundsR;

            Bounds &cL = results[group].centBoundsL;
            Bounds &cR = results[group].centBoundsR;

            gL = Bounds();
            gR = Bounds();

            cL = Bounds();
            cR = Bounds();
            for (;;)
            {
                u32 lIndex;
                u32 rIndex;
                Lane4F32 lCentroid;
                Lane4F32 rCentroid;
                PrimData *lPrim;
                PrimData *rPrim;
                do
                {
                    lIndex    = GetIndex(l);
                    lPrim     = &prims[lIndex];
                    lCentroid = (lPrim->minP + lPrim->maxP) * 0.5f;
                    if (lCentroid[bestDim] >= bestValue) break;
                    gL.Extend(lPrim->minP, lPrim->maxP);
                    cL.Extend(lCentroid);
                    l++;
                } while (l <= r);

                do
                {
                    rIndex    = GetIndex(r);
                    rPrim     = &prims[rIndex];
                    rCentroid = (rPrim->minP + rPrim->maxP) * 0.5f;
                    if (rCentroid[bestDim] < bestValue) break;
                    gR.Extend(rPrim->minP, rPrim->maxP);
                    cR.Extend(rCentroid);
                    r--;

                } while (l <= r);
                if (l > r) break;

                Swap(lPrim->minP, rPrim->minP);
                Swap(lPrim->maxP, rPrim->maxP);

                gL.Extend(lPrim->minP, lPrim->maxP);
                gR.Extend(rPrim->minP, rPrim->maxP);
                cL.Extend(rCentroid);
                cR.Extend(lCentroid);

                l++;
                r--;
            }
            vi[jobID]   = GetIndex(l);
            mids[jobID] = l;
            clock_t end = clock();
            threadLocalStatistics[GetThreadIndex()].misc += u64(end - start);
        });

    scheduler.Wait(&counter);

    auto GetIndex = [&](u32 index, u32 group) {
        const u32 chunkIndex   = index >> blockShift;
        const u32 blockIndex   = group;
        const u32 indexInBlock = index & (blockSize - 1);
        return chunkIndex * chunkSize + blockSize * blockIndex + indexInBlock;
    };

    u32 globalMid = 0;
    for (u32 i = 0; i < numJobs; i++)
    {
        globalMid += mids[i];
    }

    result->mid = FixMisplacedRanges(prims, numJobs, chunkSize, blockShift, blockSize, globalMid, mids, GetIndex);

    for (u32 i = 0; i < numJobs; i++)
    {
        result->Extend(results[i]);
    }
}

template <typename Primitive>
void PartitionParallel(Split split, const Record &record, PartitionResult *result)
{
    PartitionParallel(split, record.data, record.start, record.end, result);
}

//////////////////////////////
// SOA Partitions
//
template <typename GetIndex>
u32 PartitionSerial(PrimDataSOA *data, u32 dim, f32 bestValue, u32 l, u32 r, Bounds &outLeft, Bounds &outRight, GetIndex getIndex)
{
    const u32 queueSize      = LANE_WIDTH * 2 - 1;
    const u32 rightQueueSize = queueSize + LANE_WIDTH;
    u32 leftQueue[queueSize];
    u32 leftCount = 0;

    u32 rightQueue[rightQueueSize];
    u32 rightCount = 0;

    f32 *minStream = data->arr[dim];
    f32 *maxStream = data->arr[dim + 4];
    u32 v          = LUTAxis[dim];
    u32 w          = LUTAxis[v];
    f32 *minVs     = data->arr[v];
    f32 *minWs     = data->arr[w];

    f32 *maxVs = data->arr[v + 4];
    f32 *maxWs = data->arr[w + 4];

    // bestValue *= 2.f;

    Bounds8F32 left(neg_inf);
    Bounds8F32 right(neg_inf);
    for (;;)
    {
        while (l <= r && leftCount < LANE_WIDTH)
        {
            u32 lIndex    = getIndex(l);
            Lane8F32 lMin = Lane8F32::LoadU(minStream + lIndex);
            Lane8F32 lMax = Lane8F32::LoadU(maxStream + lIndex);
            Lane8F32 minV = Lane8F32::LoadU(minVs + lIndex);
            Lane8F32 minW = Lane8F32::LoadU(minWs + lIndex);
            Lane8F32 maxV = Lane8F32::LoadU(maxVs + lIndex);
            Lane8F32 maxW = Lane8F32::LoadU(maxWs + lIndex);

            Lane8F32 lCentroid   = (lMax - lMin) * 0.5f;
            Lane8F32 notLeftMask = lCentroid >= bestValue;
            Lane8F32 leftMask    = lCentroid < bestValue;
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
            u32 rIndex    = getIndex(r);
            Lane8F32 min  = Lane8F32::LoadU(minStream + rIndex);
            Lane8F32 max  = Lane8F32::LoadU(maxStream + rIndex);
            Lane8F32 minV = Lane8F32::LoadU(minVs + rIndex);
            Lane8F32 minW = Lane8F32::LoadU(minWs + rIndex);
            Lane8F32 maxV = Lane8F32::LoadU(maxVs + rIndex);
            Lane8F32 maxW = Lane8F32::LoadU(maxWs + rIndex);

            Lane8F32 centroid     = (max - min) * 0.5f;
            Lane8F32 notRightMask = centroid < bestValue;
            Lane8F32 rightMask    = centroid >= bestValue;
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
            for (u32 i = 0; i < minCount; i++)
            {
                Swap(data, getIndex(leftQueue[i]), getIndex(rightQueue[queueSize - 1 - i]));
            }
            if (leftCount != minCount)
            {
                l = leftQueue[minCount];
                r += LANE_WIDTH;
            }
            else if (rightCount != minCount)
            {
                r = rightQueue[queueSize - 1 - minCount];
                l -= LANE_WIDTH;
            }
            for (;;)
            {
                u32 lIndex;
                while (l <= r)
                {
                    lIndex         = getIndex(l);
                    f32 min        = minStream[lIndex];
                    f32 max        = maxStream[lIndex];
                    f32 centroid   = (max - min) * 0.5f;
                    bool isNotLeft = centroid >= bestValue;
                    Lane8F32 leftMask(centroid < bestValue);
                    Lane8F32 rightMask(isNotLeft);
                    left.MaskExtendNegativeMin(leftMask, min, minVs[lIndex], minWs[lIndex], max, maxVs[lIndex], maxWs[lIndex]);
                    right.MaskExtendNegativeMin(rightMask, min, minVs[lIndex], minWs[lIndex], max, maxVs[lIndex], maxWs[lIndex]);
                    if (isNotLeft) break;
                    l++;
                }
                u32 rIndex;
                while (l <= r)
                {
                    rIndex          = getIndex(r);
                    f32 min         = minStream[rIndex];
                    f32 max         = maxStream[rIndex];
                    f32 centroid    = (max - min) * 0.5f;
                    bool isNotRight = centroid < bestValue;
                    Lane8F32 leftMask(isNotRight);
                    Lane8F32 rightMask(centroid >= bestValue);
                    left.MaskExtendNegativeMin(leftMask, min, minVs[rIndex], minWs[rIndex], max, maxVs[rIndex], maxWs[rIndex]);
                    right.MaskExtendNegativeMin(rightMask, min, minVs[rIndex], minWs[rIndex], max, maxVs[rIndex], maxWs[rIndex]);
                    if (isNotRight) break;
                    r--;
                }
                if (l > r) break;

                Swap(data, lIndex, rIndex);
                l++;
                r--;
            }
            break;
        }
        Assert(leftCount >= LANE_WIDTH);
        Assert(rightCount >= LANE_WIDTH);

        u32 minCount = Min(leftCount, rightCount);

        for (u32 i = 0; i < minCount; i++)
        {
            u32 leftIndex  = getIndex(leftQueue[i]);
            u32 rightIndex = getIndex(rightQueue[queueSize - 1 - i]);
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

    outLeft.minP[dim]  = -minScalars[0];
    outLeft.minP[v]    = -minScalars[1];
    outLeft.minP[w]    = -minScalars[2];
    outRight.minP[dim] = -minScalars[3];
    outRight.minP[v]   = -minScalars[4];
    outRight.minP[w]   = -minScalars[5];
    outLeft.maxP[dim]  = maxScalars[0];
    outLeft.maxP[v]    = maxScalars[1];
    outLeft.maxP[w]    = maxScalars[2];
    outRight.maxP[dim] = maxScalars[3];
    outRight.maxP[v]   = maxScalars[4];
    outRight.maxP[w]   = maxScalars[5];

    return l;
}

u32 PartitionParallel(Split split, ExtRange range, PrimDataSOA *data, Bounds &outLeft, Bounds &outRight)
{
    if (range.count < PARALLEL_THRESHOLD)
    {
        return PartitionSerial(data, split.bestDim, split.bestValue, range.start, range.End(), outLeft, outRight,
                               [&](u32 index) { return index; });
    }

    const u32 numPerCacheLine   = CACHE_LINE_SIZE / sizeof(f32);
    const u32 blockSize         = numPerCacheLine * 2;
    const u32 blockMask         = blockSize - 1;
    const u32 blockShift        = Bsf(blockSize);
    const u32 numJobs           = OS_NumProcessors();
    const u32 numBlocksPerChunk = numJobs;
    const u32 chunkSize         = blockSize * numBlocksPerChunk;

    // NOTE: these should be aligned on a cache line
    u32 align = numPerCacheLine;
    Assert(IsPow2(align));
    u32 lStartAligned   = AlignPow2(range.start, align);
    u32 rEndAligned     = range.End() & ~(align - 1);
    u32 alignedCount    = rEndAligned - lStartAligned;
    const u32 numChunks = (alignedCount + chunkSize - 1) / chunkSize;

    TempArena temp    = ScratchStart(0, 0);
    u32 *blockIndices = PushArray(temp.arena, u32, numChunks);
    u32 *outMid       = PushArray(temp.arena, u32, numJobs);
    for (u32 chunkIndex = 0; chunkIndex < numChunks; chunkIndex++)
    {
        blockIndices[chunkIndex] = RandomInt(0, numBlocksPerChunk);
    }

    auto GetIndex = [&](u32 index, u32 group) {
        const u32 chunkIndex   = index >> blockShift;
        const u32 blockIndex   = (blockIndices[chunkIndex] + group) & (numBlocksPerChunk - 1);
        const u32 indexInBlock = index & blockMask;

        u32 outIndex = lStartAligned + chunkIndex * chunkSize + (blockIndex << blockShift) + indexInBlock;
        return outIndex;
    };

    u64 tempAlign     = temp.arena->align;
    temp.arena->align = 16;
    Bounds *lefts     = PushArrayDefault<Bounds>(temp.arena, numJobs + 1);
    Bounds *rights    = PushArrayDefault<Bounds>(temp.arena, numJobs + 1);
    temp.arena->align = tempAlign;

    Scheduler::Counter counter = {};
    scheduler.Schedule(&counter, numJobs, 1, [&](u32 jobID) {
        const u32 group = jobID;

        u32 l          = 0;
        u32 r          = (numChunks << blockShift) - 1;
        u32 lastRIndex = GetIndex(r, group);
        Assert(!(lastRIndex >= rEndAligned && (lastRIndex - rEndAligned) == LANE_WIDTH));
        r = lastRIndex > rEndAligned
                ? (lastRIndex - rEndAligned) < (blockSize - 1)
                      ? r - (lastRIndex - rEndAligned) - LANE_WIDTH
                      : AlignPow2(r, blockSize) - blockSize - LANE_WIDTH
                : r - (LANE_WIDTH - 1);
        Assert(GetIndex(r, group) < rEndAligned);

        auto GetIndexInBlock = [&](u32 index) {
            return GetIndex(index, group);
        };
        outMid[jobID] = PartitionSerial(data, split.bestDim, split.bestValue, l, r,
                                        lefts[jobID], rights[jobID], GetIndexInBlock);
    });
    scheduler.Wait(&counter);

    // Partition the beginning and the end
    u32 leftOverMid      = range.start;
    u32 rightLeftOverMid = range.End();
    if (range.start != lStartAligned)
    {
        leftOverMid = PartitionSerial(data, split.bestDim, split.bestValue, range.start, lStartAligned,
                                      lefts[numJobs], rights[numJobs], [&](u32 index) { return index; });
    }
    if (range.End() != rEndAligned)
    {
        rightLeftOverMid = PartitionSerial(data, split.bestDim, split.bestValue, rEndAligned, range.End(),
                                           lefts[numJobs], rights[numJobs], [&](u32 index) { return index; });
    }

    u32 globalMid = 0;
    globalMid += leftOverMid - range.start + rightLeftOverMid - rEndAligned;
    for (u32 i = 0; i < numJobs; i++)
    {
        globalMid += outMid[i];
    }
    AdditionalMisplacedRanges ranges;
    ranges.rightRange.start = leftOverMid;
    ranges.rightRange.end   = lStartAligned;
    ranges.leftRange.start  = rEndAligned;
    ranges.leftRange.end    = rightLeftOverMid;

    u32 result = FixMisplacedRanges(data, numJobs, chunkSize, blockShift, blockSize, globalMid, outMid, GetIndex, &ranges);

    for (u32 i = 0; i < numJobs + 1; i++)
    {
        outLeft.Extend(lefts[i]);
        outRight.Extend(rights[i]);
    }
    return result;
}
} // namespace rt
#endif
