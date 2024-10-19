#ifndef PARTIAL_REBRAID_H
#define PARTIAL_REBRAID_H
namespace rt
{

template <u32 N>
struct BuildRef
{
    f32 min[3];
    u32 objectID;
    f32 max[3];
    u32 numPrims;
    BVHNode<N> nodePtr;

    BVHNode<N> LeafID() const { return nodePtr; }
};

struct RebraidRecord
{
    f32 min[3];
    u32 start;
    f32 max[3];
    u32 count;
    u32 extEnd;

    u32 End() const
    {
        return start + count;
    }
    void SetRange(u32 inStart, u32 inCount, u32 inExtEnd)
    {
        start  = inStart;
        count  = inCount;
        extEnd = inExtEnd;
    }
};

// NOTE: row major affine transformation matrix
struct Transform
{
    f32 x[4];
    f32 y[4];
    f32 z[4];
};

// struct Instance2
// {
//     BVHNode bvhNode;
//     Transform transform;
// };

// void PartialRebraid(Scene *scene, Arena *arena, Instance2 *instances, u32 numInstances)
// {
//     BuildRef *b = PushArrayNoZero(arena, BuildRef, 4 * numInstances);
//
//     Lane4F32 min(neg_inf);
//     Lane4F32 max(pos_inf);
//     for (u32 i = 0; i < numInstances; i++)
//     {
//         Instance2 &instance = instances[i];
//         Assert((instance.bvhNode & 0xf) == 0);
//         b[i].nodePtr = bvhNode;
//     }
//
//     RebraidRecord record;
// }

static const f32 REBRAID_THRESHOLD = .1f;
template <u32 N>
void OpenBraid(RebraidRecord &record, BuildRef<N> *refs, u32 start, u32 count, std::atomic<u32> &refOffset)
{
    const u32 QUEUE_SIZE = 8;
    u32 choiceDim        = 0;
    f32 maxExtent        = neg_inf;
    for (u32 d = 0; d < 3; d++)
    {
        f32 extent = record.max[d] - record.min[d];
        if (extent > maxExtent)
        {
            maxExtent = extent;
            choiceDim = d;
        }
    }

    u32 refIDQueue[2 * QUEUE_SIZE] = {};
    u32 openCount                  = 0;
    for (u32 i = start; i < start + count; i++)
    {
        BuildRef &ref         = refs[i];
        refIDQueue[openCount] = i;
        bool isOpen           = (ref.max[choiceDim] - ref.min[choiceDim] > REBRAID_THRESHOLD * maxExtent);
        openCount += isOpen;

        // TODO: make sure that a compressed leaf node can't be opened
        if (openCount >= QUEUE_SIZE)
        {
            openCount -= QUEUE_SIZE;
            u32 numChildren[QUEUE_SIZE];
            u32 childAdd = 0;
            for (u32 refID = 0; refID < QUEUE_SIZE; refID++)
            {
                u32 num            = refs[refIDQueue[openCount + refID]].GetQuantizedNode()->GetNumChildren();
                numChildren[refID] = num;
                childAdd += num - 1;
            }
            u32 offset = refOffset.fetch_add(childAdd, std::memory_order_acq_rel);
            for (u32 testIndex = 0; testIndex < QUEUE_SIZE; testIndex++)
            {
                u32 refID            = refIDQueue[openCount + testIndex];
                QuantizedNode4 *node = refs[refID].GetQuantizedNode();
                f32 minX[4];
                f32 minY[4];
                f32 minZ[4];
                f32 maxX[4];
                f32 maxY[4];
                f32 maxZ[4];

                node->GetBounds(minX, minY, minZ, maxX, maxY, maxZ);

                // = children->Get ? ? ? ;
                refs[refID].min[0] = minX[0];
                refs[refID].min[1] = minY[0];
                refs[refID].min[2] = minZ[0];

                refs[refID].max[0] = maxX[0];
                refs[refID].max[1] = maxY[0];
                refs[refID].max[2] = maxZ[0];

                u32 numPrims         = Max(refs[refID].numPrims / numChildren[testIndex], 1);
                refs[refID].numPrims = numPrims;
                refs[refID].nodePtr  = node->Child(0);

                for (u32 b = 1; b < numChildren[testIndex]; b++)
                {
                    refs[offset].min[0]   = minX[b];
                    refs[offset].min[1]   = minY[b];
                    refs[offset].min[2]   = minZ[b];
                    refs[offset].objectID = refs[refID].objectID;
                    refs[offset].max[0]   = maxX[b];
                    refs[offset].max[1]   = maxY[b];
                    refs[offset].max[2]   = maxZ[b];
                    refs[offset].numPrims = numPrims;
                    refs[offset].nodePtr  = node->Child(b);

                    offset++;
                }
            }
        }
    }
}

template <u32 N, i32 numObjectBins = 32>
struct HeuristicPartialRebraid
{
    using Record = RebraidRecord;
    using OBin   = HeuristicAOSObjectBinning<numObjectBins, BuildRef<N>>;

    BuildRef<N> *buildRefs;

    HeuristicPartialRebraid() {}
    HeuristicPartialRebraid(BuildRef<N> *data) : primRefs(data) {}
    Split Bin(const Record &record)
    {
        u64 popPos = 0;
        if (record.count > PARALLEL_THRESHOLD)
        {
            std::atomic<u32> refOffset{record.End()};
            const u32 groupSize = PARALLEL_THRESHOLD;
            ParallelFor(
                record.start, record.count, groupSize,
                [&](u32 start, u32 count) { OpenBraid(record, buildRefs, start, count, refOffset); });
        }
        else
        {
            OpenBraid<N>(record, buildRefs, record.start, record.count, refOffset);
        }
        OBin *objectBinHeuristic;
        struct Split objectSplit = SAHObjectBinning(record, buildRefs, objectBinHeuristic, popPos);
        FinalizeObjectSplit(objectBinHeuristic, objectSplit, popPos);

        return objectSplit;
    }
    void FlushState(struct Split split)
    {
        TempArena temp = ScratchStart(0, 0);
        ArenaPopTo(temp.arena, split.allocPos);
    }
    void Split(struct Split split, const Record &record, Record &outLeft, Record &outRight)
    {
        TempArena temp = ScratchStart(0, 0);
        u32 mid;
        if (split.bestSAH == f32(pos_inf))
        {
            mid = SplitFallback(record, split, buildRefs, outLeft, outRight);
        }
        else
        {
            OBin *heuristic = (OBin *)(split.ptr);
            mid             = PartitionParallel(heuristic, buildRefs, split, record.start, record.count, outLeft, outRight);
        }
        MoveExtendedRanges(split, record, buildRefs, mid, outLeft, outRight);
        ArenaPopTo(temp.arena, split.allocPos);
    }
};

} // namespace rt
#endif
