#ifndef BVH_PARTITION_H
#define BVH_PARTITION_H

namespace rt
{
// void Swap(PrimDataSOA *data, u32 lIndex, u32 rIndex)
// {
//     ::rt::Swap(data->minX[lIndex], data->minX[rIndex]);
//     ::rt::Swap(data->minY[lIndex], data->minY[rIndex]);
//     ::rt::Swap(data->minZ[lIndex], data->minZ[rIndex]);
//     ::rt::Swap(data->geomIDs[lIndex], data->geomIDs[rIndex]);
//     ::rt::Swap(data->maxX[lIndex], data->maxX[rIndex]);
//     ::rt::Swap(data->maxY[lIndex], data->maxY[rIndex]);
//     ::rt::Swap(data->maxZ[lIndex], data->maxZ[rIndex]);
//     ::rt::Swap(data->primIDs[lIndex], data->primIDs[rIndex]);
// }
// void Swap(PrimData *prims, u32 lIndex, u32 rIndex)
// {
//     Swap(prims[lIndex], prims[rIndex]);
// }

template <typename PrimitiveData, typename GetIndex>
u32 FixMisplacedRanges(PrimitiveData *data, u32 numJobs, u32 chunkSize, u32 blockShift,
                       u32 blockSize, u32 globalMid, u32 *mids, GetIndex getIndex)
{
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
    TempArena temp              = ScratchStart(0, 0);
    u32 numMisplacedRanges      = numJobs;
    Range *leftMisplacedRanges  = PushArray(temp.arena, Range, numMisplacedRanges);
    u32 lCount                  = 0;
    Range *rightMisplacedRanges = PushArray(temp.arena, Range, numMisplacedRanges);
    u32 rCount                  = 0;

    u32 numMisplacedLeft  = 0;
    u32 numMisplacedRight = 0;

    for (u32 i = 0; i < numJobs; i++)
    {
        u32 globalIndex = getIndex(mids[i], i);
        if (globalIndex < globalMid)
        {
            u32 diff = (((globalMid - globalIndex) / chunkSize) << blockShift) + ((globalMid - globalIndex) & (blockSize - 1));

            u32 check = getIndex(mids[i] + diff - 1, i);

            while (check >= globalMid)
            {
                // Assert(!isLeft(check));
                diff -= 1;
                check = getIndex(mids[i] + diff - 1, i);
            }
            int steps1 = 0;
            check      = getIndex(mids[i] + diff + 1, i);
            while (check < globalMid)
            {
                // Assert(isLeft(check));
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
                // Assert(isLeft(check));
                diff -= 1;
                steps0++;
                check = getIndex(mids[i] - diff + 1, i);
            }
            if (steps0) diff--;

            check = getIndex(mids[i] - diff - 1, i);
            while (check >= globalMid)
            {
                // Assert(!isLeft(check));
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

    while (leftIndex != lCount && rightIndex != rCount)
    {
        while (lSize != lIter && rSize != rIter)
        {
            Assert(rightIndex < rCount);
            Assert(leftIndex < lCount);
            u32 lIndex = lRange.start + lIter;
            u32 rIndex = rRange.start + rIter;
            lIndex     = getIndex(lIndex, lRange.group);
            rIndex     = getIndex(rIndex, rRange.group);

            Swap(data[lIndex], data[rIndex]); //, lIndex, rIndex);

            lIter++;
            rIter++;
        }
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

#if 0
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

#endif
} // namespace rt
#endif
