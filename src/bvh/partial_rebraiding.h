#ifndef PARTIAL_REBRAID_H
#define PARTIAL_REBRAID_H
namespace rt
{

void GenerateBuildRefs(BRef *refs, Scene2 *scene, Scene2 *scenes, u32 start, u32 count, RecordAOSSplits &record)
{
    Bounds geom;
    Bounds cent;
    Instance *instances = scene->instances;
    for (u32 i = start; i < start + count; i++)
    {
        Instance &instance     = instances[i];
        AffineSpace &transform = scene->affineTransforms[instance.transformIndex];
        Assert(instance.geomID.GetType() == GT_InstanceType);
        u32 index       = instance.geomID.GetIndex() + 1;
        Scene2 *inScene = &scenes[index];
        BRef *ref       = &refs[i];

        Bounds bounds = Transform(transform, inScene->GetBounds());

        Lane4F32::StoreU(ref->min, -bounds.minP);
        Lane4F32::StoreU(ref->max, bounds.maxP);
        geom.Extend(bounds);
        cent.Extend(bounds.minP + bounds.maxP);

        ref->objectID = i;
        ref->nodePtr  = inScene->nodePtr;
        ref->numPrims = inScene->numPrims;
    }
    record.geomBounds = Lane8F32(-geom.minP, geom.maxP);
    record.centBounds = Lane8F32(-cent.minP, cent.maxP);
}

BRef *GenerateBuildRefs(Scene2 *scenes, u32 sceneNum, Arena *arena, RecordAOSSplits &record)
{
    Scene2 *scene    = &scenes[sceneNum];
    u32 numInstances = scene->numInstances;
    BRef *b          = PushArrayNoZero(arena, BRef, 4 * numInstances);

    if (numInstances > PARALLEL_THRESHOLD)
    {
        ParallelReduce<RecordAOSSplits>(
            &record, 0, numInstances, PARALLEL_THRESHOLD,
            [&](RecordAOSSplits &record, u32 jobID, u32 start, u32 count) {
                GenerateBuildRefs(b, scene, scenes, start, count, record);
            },
            [&](RecordAOSSplits &l, const RecordAOSSplits &r) {
                l.Merge(r);
            });
    }
    else
    {
        GenerateBuildRefs(b, scene, scenes, 0, numInstances, record);
    }
    return b;
}

static const f32 REBRAID_THRESHOLD = .1f;
template <bool parallel = true>
void OpenBraid(const RecordAOSSplits &record, BRef *refs, u32 start, u32 count,
               std::atomic<u32> &refOffset, u32 refO = 0)
{
    const u32 QUEUE_SIZE = 8;
    u32 choiceDim        = 0;
    f32 maxExtent        = neg_inf;
    for (u32 d = 0; d < 3; d++)
    {
        f32 extent = record.geomMax[d] - record.geomMin[d];
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
        BRef &ref             = refs[i];
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
                u32 num            = refs[refIDQueue[openCount + refID]].nodePtr.GetQuantizedNode()->GetNumChildren();
                numChildren[refID] = num;
                childAdd += num - 1;
            }
            u32 offset;
            if constexpr (parallel) offset = refOffset.fetch_add(childAdd, std::memory_order_acq_rel);
            else
            {
                refO += childAdd;
                offset = refO;
            }

            for (u32 testIndex = 0; testIndex < QUEUE_SIZE; testIndex++)
            {
                u32 refID   = refIDQueue[openCount + testIndex];
                QNode *node = refs[refID].nodePtr.GetQuantizedNode();
                f32 minX[DefaultN];
                f32 minY[DefaultN];
                f32 minZ[DefaultN];
                f32 maxX[DefaultN];
                f32 maxY[DefaultN];
                f32 maxZ[DefaultN];

                node->GetBounds(minX, minY, minZ, maxX, maxY, maxZ);

                // = children->Get ? ? ? ;
                refs[refID].min[0] = minX[0];
                refs[refID].min[1] = minY[0];
                refs[refID].min[2] = minZ[0];

                refs[refID].max[0] = maxX[0];
                refs[refID].max[1] = maxY[0];
                refs[refID].max[2] = maxZ[0];

                u32 numPrims         = Max(refs[refID].numPrims / numChildren[testIndex], 1u);
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

template <i32 numObjectBins = 32>
struct HeuristicPartialRebraid
{
    using Record  = RecordAOSSplits;
    using PrimRef = BRef;
    using OBin    = HeuristicAOSObjectBinning<numObjectBins, BRef>;

    BRef *buildRefs;

    HeuristicPartialRebraid() {}
    HeuristicPartialRebraid(BRef *data) : buildRefs(data)
    {
    }
    Split Bin(const Record &record)
    {
        u64 popPos = 0;
        // TODO: make the single threaded version not take an atomic
        std::atomic<u32> refOffset{record.End()};
        if (record.count > PARALLEL_THRESHOLD)
        {
            const u32 groupSize = PARALLEL_THRESHOLD;
            ParallelFor(
                record.start, record.count, groupSize,
                [&](u32 start, u32 count) { OpenBraid(record, buildRefs, start, count, refOffset); });
        }
        else
        {
            OpenBraid<false>(record, buildRefs, record.start, record.count, refOffset, record.End());
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
