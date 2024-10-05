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

template <typename PrimitiveData, typename GetIndex, typename IsLeft>
u32 FixMisplacedRanges(PrimitiveData *data, u32 numJobs, u32 chunkSize, u32 blockShift,
                       u32 blockSize, u32 globalMid, u32 *mids, GetIndex getIndex, IsLeft isLeft, AdditionalMisplacedRanges *add = 0)
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
        if (add->leftRange.Size() > 0)
        {
            lCount++;
        }
        if (add->rightRange.Size() > 0)
        {
            rCount++;
        }
    }

    for (u32 i = 0; i < numJobs; i++)
    {
        u32 globalIndex = getIndex(mids[i], i);
        if (globalIndex < globalMid)
        {
            u32 diff = (((globalMid - globalIndex) / chunkSize) << blockShift) + ((globalMid - globalIndex) & (blockSize - 1));

            u32 check = getIndex(mids[i] + diff - 1, i);

            while (check >= globalMid)
            {
                Assert(!isLeft(check));
                diff -= 1;
                check = getIndex(mids[i] + diff - 1, i);
            }
            int steps1 = 0;
            check      = getIndex(mids[i] + diff + 1, i);
            while (check < globalMid)
            {
                Assert(isLeft(check));
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
                Assert(isLeft(check));
                diff -= 1;
                steps0++;
                check = getIndex(mids[i] - diff + 1, i);
            }
            if (steps0) diff--;

            check = getIndex(mids[i] - diff - 1, i);
            while (check >= globalMid)
            {
                Assert(!isLeft(check));
                diff += 1;
                check = getIndex(mids[i] - diff - 1, i);
            }
            leftMisplacedRanges[lCount] = {mids[i] - diff, mids[i], i};
            numMisplacedLeft += leftMisplacedRanges[lCount].Size();
            lCount++;
        }
    }

    Assert(numMisplacedLeft == numMisplacedRight);

    if (numMisplacedLeft == 0 && numMisplacedRight == 0)
    {
        ScratchEnd(temp);
        return globalMid;
    }

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
            Assert(rightIndex < rCount);
            Assert(leftIndex < lCount);
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
        if (leftIndex >= lCount - 1 && rightIndex >= rCount - 1) break;
        if (rSize == rIter && rightIndex < rCount)
        {
            rightIndex++;
            rRange = rightMisplacedRanges[rightIndex];
            rIter  = 0;
            rSize  = rRange.Size();
        }
        if (lSize == lIter && leftIndex < lCount)
        {
            leftIndex++;
            lRange = leftMisplacedRanges[leftIndex];
            lIter  = 0;
            lSize  = lRange.Size();
        }
    }
    ScratchEnd(temp);
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
            const u32 group = jobID;
            auto GetIndex   = [&](u32 index) {
                const u32 chunkIndex   = index >> blockShift;
                const u32 blockIndex   = group;
                const u32 indexInBlock = index & (blockSize - 1);

                return start + chunkIndex * chunkSize + blockSize * blockIndex + indexInBlock;
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
        });

    scheduler.Wait(&counter);

    auto GetIndex = [&](u32 index, u32 group) {
        const u32 chunkIndex   = index >> blockShift;
        const u32 blockIndex   = group;
        const u32 indexInBlock = index & (blockSize - 1);
        return chunkIndex * chunkSize + blockSize * blockIndex + indexInBlock;
    };

    u32 globalMid = start;
    for (u32 i = 0; i < numJobs; i++)
    {
        globalMid += mids[i];
    }

    auto IsLeft = [&](u32 index) -> bool {
        return true;
    };

    result->mid = FixMisplacedRanges(prims, numJobs, chunkSize, blockShift, blockSize, globalMid, mids, GetIndex, IsLeft);

    for (u32 i = 0; i < numJobs; i++)
    {
        result->Extend(results[i]);
    }
    ScratchEnd(temp);
}

template <typename Primitive>
void PartitionParallel(Split split, const Record &record, PartitionResult *result)
{
    PartitionParallel(split, record.data, record.start, record.end, result);
}

//////////////////////////////
// SOA Partitions
//
template <bool centroidPartition, typename GetIndex>
u32 PartitionSerial(PrimDataSOA *data, u32 dim, f32 bestValue, i32 l, i32 r, GetIndex getIndex)
{
    const u32 queueSize      = LANE_WIDTH * 2 - 1;
    const u32 rightQueueSize = queueSize + LANE_WIDTH;
    u32 leftQueue[queueSize];
    u32 leftCount = 0;

    u32 rightQueue[rightQueueSize];
    u32 rightCount = 0;

    f32 *minStream = data->arr[dim];
    f32 *maxStream = data->arr[dim + 4];

    i32 start = l;
    i32 end   = r;

    Lane8F32 negBestValue = -bestValue;
    r -= LANE_WIDTHi;
    for (;;)
    {
        while (l + LANE_WIDTHi <= r && leftCount == 0) //< LANE_WIDTH)
        {
            u32 lIndex   = getIndex(l);
            Lane8F32 min = Lane8F32::LoadU(minStream + lIndex);
            Lane8F32 rightMask;

            if constexpr (centroidPartition)
            {
                Lane8F32 max      = Lane8F32::LoadU(maxStream + lIndex);
                Lane8F32 centroid = (max - min) * 0.5f;
                rightMask         = centroid >= bestValue;
            }
            else
            {
                rightMask = min <= negBestValue;
            }

            u32 mask           = Movemask(rightMask);
            Lane8U32 leftRefId = Lane8U32::Step(l);

            Lane8U32::StoreU(leftQueue + leftCount, MaskCompress(mask, leftRefId));

            l += LANE_WIDTHi;
            leftCount += PopCount(mask);
        }
        while (l + LANE_WIDTHi <= r && rightCount == 0) //< LANE_WIDTH)
        {
            Assert(r >= 0);
            u32 rIndex   = getIndex(r);
            Lane8F32 min = Lane8F32::LoadU(minStream + rIndex);

            Lane8F32 leftMask;
            if constexpr (centroidPartition)
            {
                Lane8F32 max      = Lane8F32::LoadU(maxStream + rIndex);
                Lane8F32 centroid = (max - min) * 0.5f;
                leftMask          = centroid < bestValue;
            }
            else
            {
                leftMask = min > negBestValue;
            }
            const u32 mask = Movemask(leftMask);

            u32 notRightCount  = PopCount(mask);
            Lane8F32 storeMask = Lane8F32::Mask((1 << notRightCount) - 1u);

            Lane8U32 refID = Lane8U32::Step(r);
            // Store so that entries are sorted from smallest to largest
            rightCount += notRightCount;

            Lane8U32::StoreU(storeMask, rightQueue + queueSize - rightCount, MaskCompress(mask, refID));
            r -= LANE_WIDTHi;
        }
        if (l + LANE_WIDTHi > r)
        {
            u32 minCount = Min(leftCount, rightCount);
            for (u32 i = 0; i < minCount; i++)
            {
                Swap(data, getIndex(leftQueue[i]), getIndex(rightQueue[queueSize - 1 - i]));
            }
            if (leftCount != minCount)
            {
                l = leftQueue[minCount];
                r = Min(end - 1, r + LANE_WIDTHi - 1);
            }
            else if (rightCount != minCount)
            {
                r = rightQueue[queueSize - 1 - minCount];
                l -= LANE_WIDTHi;
            }
            else
            {
                r = Min(end - 1, r + LANE_WIDTHi - 1);
            }
            Assert(r >= 0);
            for (;;)
            {
                u32 lIndex;
                while (l <= r)
                {
                    lIndex  = getIndex(l);
                    f32 min = minStream[lIndex];
                    bool isRight;
                    if constexpr (centroidPartition)
                    {
                        f32 max      = maxStream[lIndex];
                        f32 centroid = ((Lane8F32(max) - Lane8F32(min)) * 0.5f)[0];
                        isRight      = (centroid >= bestValue);
                    }
                    else
                    {
                        isRight = All(Lane8F32(min) <= Lane8F32(-bestValue));
                    }
                    if (isRight) break;
                    l++;
                }
                u32 rIndex;
                while (l <= r)
                {
                    Assert(r >= 0);
                    rIndex  = getIndex(r);
                    f32 min = minStream[rIndex];

                    bool isLeft;
                    if constexpr (centroidPartition)
                    {
                        f32 max      = maxStream[rIndex];
                        f32 centroid = ((Lane8F32(max) - Lane8F32(min)) * 0.5f)[0];
                        isLeft       = (centroid < bestValue);
                    }
                    else
                    {
                        isLeft = All(Lane8F32(min) > Lane8F32(-bestValue));
                    }
                    if (isLeft) break;
                    r--;
                }
                if (l > r) break;

                Swap(data, lIndex, rIndex);
                l++;
                r--;
            }
            break;
        }
        // Assert(leftCount >= LANE_WIDTH);
        // Assert(rightCount >= LANE_WIDTH);

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

    // Assert(l < end);
    // Assert(r >= start);

    return l;
}

template <bool centroidPartition>
u32 PartitionSerialScalar(PrimDataSOA *data, u32 dim, f32 bestValue, i32 l, i32 r)
{
    f32 *minStream = data->arr[dim];
    f32 *maxStream = data->arr[dim + 4];
    r--;
    for (;;)
    {
        while (l <= r)
        {
            f32 min = minStream[l];
            bool isRight;
            if constexpr (centroidPartition)
            {
                f32 max      = maxStream[l];
                f32 centroid = ((Lane8F32(max) - Lane8F32(min)) * 0.5f)[0];
                isRight      = (centroid >= bestValue);
            }
            else
            {
                isRight = min <= -bestValue;
            }
            if (isRight) break;
            l++;
        }
        while (l <= r)
        {
            Assert(r >= 0);
            f32 min = minStream[r];

            bool isLeft;
            if constexpr (centroidPartition)
            {
                f32 max      = maxStream[r];
                f32 centroid = ((Lane8F32(max) - Lane8F32(min)) * 0.5f)[0];
                isLeft       = (centroid < bestValue);
            }
            else
            {
                isLeft = min > -bestValue;
            }
            if (isLeft) break;
            r--;
        }
        if (l > r) break;

        Swap(data, l, r);
        l++;
        r--;
    }
    return l;
}

u32 Partition(Split split, i32 l, i32 r, PrimRef *data)
{
    // u32 l   = 0;
    // u32 r   = range.count - 1;
    u32 dim = split.bestDim;
    for (;;)
    {
        while (l <= r)
        {
            PrimRef *lRef = &data[l];
            f32 max       = lRef->max[dim];
            f32 min       = lRef->min[dim];
            f32 centroid  = (max - min) * 0.5f;
            bool isRight  = (centroid >= split.bestValue);
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

            bool isLeft = (centroid < split.bestValue);
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

// ways of doing this:
// double buffer (better for multithread?)
// in place (better for single thread)
u32 Partition2(Split split, u32 l, u32 r, u32 outLStart, u32 outRStart, PrimRef *data, u32 *inRefs, u32 *outRefs)
{
    u32 dim          = split.bestDim;
    u32 writeLocs[2] = {outLStart, outRStart};

    const u32 fetchAmt = 64;
    u32 currentCount   = fetchAmt;
    for (u32 i = l; i < r; i++)
    {
        u32 ref                     = inRefs[i];
        PrimRef *primRef            = &data[ref];
        f32 min                     = primRef->min[dim];
        f32 max                     = primRef->max[dim];
        f32 centroid                = (max - min) * 0.5f;
        bool isRight                = centroid >= split.bestValue;
        outRefs[writeLocs[isRight]] = ref;
        writeLocs[isRight]++;
    }
    return 0;
}

u32 Partition3(Split split, u32 l, u32 r, u32 outLStart, u32 outRStart, PrimRef *data, u32 *inRefs, u32 *outRefs,
               Lane8F32 &outLeft, Lane8F32 &outRight)
{
    u32 dim           = split.bestDim;
    u32 writeLocs[2]  = {outLStart, outRStart};
    Lane8F32 masks[2] = {Lane8F32::Mask(false), Lane8F32::Mask(true)};

    Lane8F32 left(neg_inf);
    Lane8F32 right(neg_inf);

    Lane8F32 lanes[8];

    Lane8F32 centLeft(neg_inf);
    Lane8F32 centRight(neg_inf);
    // Bounds8F32 centLeft;
    // Bounds8F32 centRight;

    u32 v = LUTAxis[dim];
    u32 w = LUTAxis[v];

    u32 i = l;
    for (; i < r - (LANE_WIDTH - 1); i += LANE_WIDTH)
    {
        Transpose8x6(data[inRefs[i]].m256, data[inRefs[i + 1]].m256, data[inRefs[i + 2]].m256, data[inRefs[i + 3]].m256,
                     data[inRefs[i + 4]].m256, data[inRefs[i + 5]].m256, data[inRefs[i + 6]].m256, data[inRefs[i + 7]].m256,
                     lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5]);

        Lane8F32 centroid  = (lanes[dim + 3] - lanes[dim]) * 0.5f;
        Lane8F32 centroidV = (lanes[v + 3] - lanes[v]) * 0.5f;
        Lane8F32 centroidW = (lanes[w + 3] - lanes[w]) * 0.5f;

        Lane8F32 maskR = (centroid >= split.bestValue);
        Lane8F32 maskL = (centroid < split.bestValue);
        u32 prevMask   = Movemask(maskR);
        for (u32 b = 0; b < LANE_WIDTH; b++)
        {
            u32 select                   = (prevMask >> i) & 1;
            outRefs[writeLocs[select]++] = inRefs[i + b];
            left                         = MaskMax(masks[!select], left, data[inRefs[i + b]].m256);
            right                        = MaskMax(masks[select], right, data[inRefs[i + b]].m256);
        }

        Transpose3x8(centroid, centroidV, centroidW,
                     lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7]);
        for (u32 b = 0; b < LANE_WIDTH; b++)
        {
            centLeft  = Select(maskL, Max(centLeft, lanes[b] ^ signFlipMask), centLeft);
            centRight = Select(maskR, Max(centRight, lanes[b] ^ signFlipMask), centRight);
        }
        // centLeft.MaskExtendNegMin(maskL, centroid, centroidV, centroidW);
        // centRight.MaskExtendNegMin(maskR, centroid, centroidV, centroidW);
    }
    for (; i < r; i++)
    {
        u32 ref                       = inRefs[i];
        PrimRef *primRef              = &data[ref];
        f32 min                       = primRef->min[dim];
        f32 max                       = primRef->max[dim];
        f32 centroid                  = (max - min) * 0.5f;
        bool isRight                  = centroid >= split.bestValue;
        outRefs[writeLocs[isRight]++] = ref;
        if (isRight)
        {
            right = Max(right, primRef->m256);
        }
        else
        {
            left = Max(left, primRef->m256);
        }
    }
    outLeft  = left;
    outRight = right;
    return 0;
}

u32 PartitionParallel(PartitionPayload &payload, Split split, ExtRange range, PrimRef *data, u32 *inRefs, u32 *outRefs)
{
    if (range.count < 16 * 1024) // PARTITION_PARALLEL_THRESHOLD)
    {
        return Partition(split, range.start, range.End(), data);
    }

    TempArena temp = ScratchStart(0, 0);

    u32 groupSize = payload.groupSize;
    u32 end       = range.End();

    Lane8F32 *lBounds = PushArrayNoZero(temp.arena, Lane8F32, payload.count);
    Lane8F32 *rBounds = PushArrayNoZero(temp.arena, Lane8F32, payload.count);
    for (u32 i = 0; i < payload.count; i++)
    {
        lBounds[i] = neg_inf;
        rBounds[i] = neg_inf;
    }

    scheduler.ScheduleAndWait(payload.count, 1, [&](u32 jobID) {
        PerformanceCounter counter = OS_StartCounter();

        u32 start = range.start + groupSize * jobID;
        u32 end   = Min(start + groupSize, range.End());

        Partition3(split, start, end, payload.lOffsets[jobID], payload.rOffsets[jobID], data, inRefs, outRefs,
                   lBounds[jobID], rBounds[jobID]);
        threadLocalStatistics[GetThreadIndex()].miscF += OS_GetMilliseconds(counter);
    });

    return 0;
    // return out;
}

template <bool centroidPartition = false>
u32 PartitionParallel(Split split, ExtRange range, PrimDataSOA *data)
{
    static const u32 PARTITION_PARALLEL_THRESHOLD = 3 * 1024;
    if (range.count < 16 * 1024) // PARTITION_PARALLEL_THRESHOLD)
    {
        if (range.count < 1024)
        {
            return PartitionSerialScalar<centroidPartition>(data, split.bestDim, split.bestValue, range.start, range.End());
        }

        auto getIndex = [&](u32 index) { return index; };
        u32 out       = PartitionSerial<centroidPartition>(data, split.bestDim, split.bestValue,
                                                     range.start, range.End(), getIndex);
        return out;
    }

    const u32 numPerCacheLine   = CACHE_LINE_SIZE / sizeof(f32);
    const u32 blockSize         = numPerCacheLine * 2;
    const u32 blockMask         = blockSize - 1;
    const u32 blockShift        = Bsf(blockSize);
    const u32 numJobs           = OS_NumProcessors(); // * 2;
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

    scheduler.ScheduleAndWait(numJobs, 1, [&](u32 jobID) {
        const u32 group = jobID;

        u32 l          = 0;
        u32 r          = (numChunks << blockShift) - 1;
        u32 lastRIndex = GetIndex(r, group);
        Assert(!(lastRIndex >= rEndAligned && (lastRIndex - rEndAligned) == LANE_WIDTH));
        r = lastRIndex > rEndAligned
                ? (lastRIndex - rEndAligned) < (blockSize - 1)
                      ? r - (lastRIndex - rEndAligned)
                      : AlignPow2(r, blockSize) - blockSize
                : r + 1; // r - (LANE_WIDTH - 1);
        Assert(GetIndex(r - 1, group) < rEndAligned);

        auto GetIndexInBlock = [&](u32 index) {
            return GetIndex(index, group);
        };
        outMid[jobID] = PartitionSerial<centroidPartition>(data, split.bestDim, split.bestValue, l, r, GetIndexInBlock);
    });

    // Partition the beginning and the end
    u32 globalMid = range.start;

    u32 minIndex = pos_inf;
    u32 maxIndex = neg_inf;
    for (u32 i = 0; i < numJobs; i++)
    {
        globalMid += outMid[i];
        minIndex = Min(GetIndex(outMid[i], i), minIndex);
        maxIndex = Max(GetIndex(outMid[i], i), maxIndex);
    }
    f32 *minStream = data->arr[split.bestDim];
    f32 *maxStream = data->arr[split.bestDim + 4];

    u32 out = PartitionSerial<centroidPartition>(data, split.bestDim, split.bestValue,
                                                 minIndex, maxIndex, [&](u32 index) { return index; });

    // Parallelize the unaligned begin and end

    // u32 lCount = 0;
    i32 l = range.start;
    i32 r = range.End() - 1;
    Assert((maxStream[out] - minStream[out]) * 0.5f >= split.bestValue);
    for (;;)
    {
        while (l < (i32)lStartAligned && r >= (i32)rEndAligned && (maxStream[l] - minStream[l]) * 0.5f < split.bestValue)
        {
            // lCount++;
            l++;
        }
        while (l < (i32)lStartAligned && r >= (i32)rEndAligned && (maxStream[r] - minStream[r]) * 0.5f >= split.bestValue)
        {
            r--;
        }
        if (l >= (i32)lStartAligned || r < (i32)rEndAligned) break;
        Swap(data, l, r);
        // lCount++;
        l++;
        r--;
    }
    if (l < (i32)lStartAligned)
    {
        for (i32 i = l; i < (i32)lStartAligned; i++)
        {
            if ((maxStream[i] - minStream[i]) * 0.5f >= split.bestValue)
            {
                out--;
                Assert((maxStream[out] - minStream[out]) * 0.5f < split.bestValue);
                Swap(data, i, out);
            }
            // else
            // {
            //     lCount++;
            // }
        }
    }
    else if (r >= (i32)rEndAligned)
    {
        for (i32 i = r; i >= (i32)rEndAligned; i--)
        {
            if ((maxStream[i] - minStream[i]) * 0.5f < split.bestValue)
            {
                Assert((maxStream[out] - minStream[out]) * 0.5f >= split.bestValue);
                Swap(data, i, out);
                out++;
                // lCount++;
            }
        }
    }

    // error check
    // {
    //     u32 errors = 0;
    //     for (u32 i = range.start; i < range.start + range.count; i++)
    //     {
    //         f32 min      = minStream[i];
    //         f32 max      = maxStream[i];
    //         f32 centroid = (max - min) * 0.5f;
    //         if (i < out)
    //         {
    //             if (centroid >= split.bestValue) // || value > split.bestPos)
    //             {
    //                 errors++;
    //             }
    //         }
    //         else
    //         {
    //             if (centroid < split.bestValue) // || value <= split.bestPos)
    //             {
    //                 errors++;
    //             }
    //         }
    //     }
    //     threadLocalStatistics[GetThreadIndex()].misc += errors;
    // }

    // globalMid += lCount;
    // Assert(out == globalMid);

    ScratchEnd(temp);
    return out;
}

u32 PartitionParallelCentroids(Split split, ExtRange range, PrimDataSOA *data)
{
    return PartitionParallel<true>(split, range, data);
}

} // namespace rt
#endif
