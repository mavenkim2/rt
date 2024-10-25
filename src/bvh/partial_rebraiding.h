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

        ref->objectID = index;
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
    u32 extEnd       = 3 * numInstances;
    BRef *b          = PushArrayNoZero(arena, BRef, extEnd);

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
    record.SetRange(0, numInstances, extEnd);
    return b;
}

static const f32 REBRAID_THRESHOLD = .1f;

template <typename Heuristic, typename GetNode>
void OpenBraid(const RecordAOSSplits &record, BRef *refs, u32 start, u32 count, u32 offset, const u32 offsetMax,
               Heuristic &heuristic, GetNode &getNode)
{
    TIMED_FUNCTION(miscF);
    using NodeType       = typename GetNode::NodeType;
    const u32 QUEUE_SIZE = 8;

    for (u32 i = start; i < start + count; i++)
    {
        BRef &ref = refs[i];
        if (heuristic(ref))
        {
            NodeType *node = getNode(ref.nodePtr);
            alignas(4 * DefaultN) f32 minX[DefaultN];
            alignas(4 * DefaultN) f32 minY[DefaultN];
            alignas(4 * DefaultN) f32 minZ[DefaultN];
            alignas(4 * DefaultN) f32 maxX[DefaultN];
            alignas(4 * DefaultN) f32 maxY[DefaultN];
            alignas(4 * DefaultN) f32 maxZ[DefaultN];

            node->GetBounds(minX, minY, minZ, maxX, maxY, maxZ);

            // TODO: transform the bounds to world space
            // Instance &instance =
            //     Bounds(Lane4F32(minX[0], minY[0], minZ[0], 0), Lane4F32(maxX[0], maxY[0], maxZ[0], 0));

            ref.min[0] = -minX[0];
            ref.min[1] = -minY[0];
            ref.min[2] = -minZ[0];

            ref.max[0] = maxX[0];
            ref.max[1] = maxY[0];
            ref.max[2] = maxZ[0];

            u32 numC = node->GetNumChildren();
            Assert(numC > 0 && numC <= DefaultN);
            u32 numPrims = Max(ref.numPrims / numC, 1u);
            ref.numPrims = numPrims;

            BVHNodeType parent = ref.nodePtr;
            ref.nodePtr        = node->Child(0);
            Assert(ref.nodePtr.data != 0);

            for (u32 b = 1; b < numC; b++)
            {
                Assert(offset < offsetMax);
                refs[offset].min[0]   = -minX[b];
                refs[offset].min[1]   = -minY[b];
                refs[offset].min[2]   = -minZ[b];
                refs[offset].objectID = ref.objectID;
                refs[offset].max[0]   = maxX[b];
                refs[offset].max[1]   = maxY[b];
                refs[offset].max[2]   = maxZ[b];
                refs[offset].numPrims = numPrims;
                refs[offset].nodePtr  = node->Child(b);

                Assert(refs[offset].nodePtr.data != 0);

                offset++;
            }
        }
    }
    // wtf is going on ?

    if (offset != offsetMax)
    {
        // printf("offset %u\n", offset);
        // printf("offset max %u\n", offsetMax);
        // assert(0);
    }
}

struct GetQuantizedNode
{
    using NodeType = QNode;

    __forceinline NodeType *operator()(const BVHNodeType ptr) { return ptr.GetQuantizedNode(); }
};

template <typename GetNode, i32 numObjectBins = 32>
struct HeuristicPartialRebraid
{
    using Record  = RecordAOSSplits;
    using PrimRef = BRef;
    using OBin    = HeuristicAOSObjectBinning<numObjectBins, BRef>;

    BRef *buildRefs;
    GetNode getNode;

    HeuristicPartialRebraid() {}
    HeuristicPartialRebraid(BRef *data) : buildRefs(data) {}
    Split Bin(Record &record)
    {
        u32 choiceDim = 0;
        f32 maxExtent = neg_inf;
        for (u32 d = 0; d < 3; d++)
        {
            f32 extent = record.geomMax[d] - record.geomMin[d];
            if (extent > maxExtent)
            {
                maxExtent = extent;
                choiceDim = d;
            }
        }
        f32 threshold  = REBRAID_THRESHOLD * maxExtent;
        auto heuristic = [&](const BRef &ref) -> bool {
            return !ref.nodePtr.IsLeaf() && ref.max[choiceDim] - ref.min[choiceDim] > threshold;
        };

        u64 popPos = 0;
        // conditions to test:
        // 1. if all the nodes belong to the same geometry, break
        // 2. if there are 4 or less nodes, and there is no overlap between the bvhs, also break
        // 3. also obviously break if there is no space for node opening

        if (record.ExtSize() && record.count <= 4)
        {
            for (u32 i = 0; i < record.count; i++)
            {
                for (u32 j = i + 1; j < record.count; j++)
                {
                    Lane8F32 intersection = Min(buildRefs[record.start + i].Load(), buildRefs[record.start + j].Load());
                    if (None(-Extract4<0>(intersection) > Extract4<1>(intersection)))
                    {
                        record.extEnd = record.start + record.count;
                        break;
                    }
                }
            }
        }
        if (record.ExtSize())
        {
            u32 geomID = buildRefs[record.start].objectID;
            if (record.count > PARALLEL_THRESHOLD)
            {
                TempArena temp = ScratchStart(0, 0);
                struct Props
                {
                    u32 count     = 0;
                    bool sameType = true;
                };
                Props prop;
                ParallelForOutput output = ParallelFor<Props>(
                    temp, record.start, record.count, PARALLEL_THRESHOLD,
                    [&](Props &props, u32 jobID, u32 start, u32 count) {
                        bool commonGeomID = true;
                        u32 openCount     = 0;
                        for (u32 i = start; i < start + count; i++)
                        {
                            const BRef &ref = buildRefs[i];
                            u32 objectID    = ref.objectID;
                            commonGeomID &= objectID == geomID;
                            if (heuristic(ref)) openCount += getNode(ref.nodePtr)->GetNumChildren() - 1;
                        }
                        props.sameType &= commonGeomID;
                        props.count = openCount;
                    });
                Reduce(prop, output,
                       [&](Props &l, const Props &r) { l.sameType &= r.sameType; l.count += r.count; });
                if (prop.sameType || prop.count > record.ExtSize())
                {
                    record.extEnd = record.start + record.count;
                }
                else
                {
                    u32 offset   = record.End();
                    Props *props = (Props *)output.out;
                    for (u32 i = 0; i < output.num; i++)
                    {
                        u32 splits     = props[i].count;
                        props[i].count = offset;
                        Assert(offset < record.ExtEnd());
                        offset += splits;
                    }
                    ParallelFor(
                        record.start, record.count, PARALLEL_THRESHOLD,
                        [&](u32 jobID, u32 start, u32 count) {
                            Props &prop = props[jobID];
                            u32 max     = jobID == output.num - 1 ? offset : props[jobID + 1].count;
                            OpenBraid(record, buildRefs, start, count, prop.count, max, heuristic, getNode);
                        });
                }
                ScratchEnd(temp);
            }
            else
            {
                bool commonGeomID = true;
                u32 count         = 0;
                for (u32 i = record.start; i < record.start + record.count; i++)
                {
                    const BRef &ref = buildRefs[i];
                    u32 objectID    = ref.objectID;
                    commonGeomID &= objectID == geomID;
                    if (heuristic(ref)) count += getNode(ref.nodePtr)->GetNumChildren() - 1;
                }
                if (commonGeomID || count > record.ExtSize())
                {
                    record.extEnd = record.start + record.count;
                }
                else
                {
                    OpenBraid(record, buildRefs, record.start, record.count, record.End(), record.ExtEnd(), heuristic, getNode);
                }
            }
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
