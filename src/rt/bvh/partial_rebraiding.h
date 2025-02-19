#ifndef PARTIAL_REBRAID_H
#define PARTIAL_REBRAID_H
namespace rt
{

void GenerateBuildRefs(BRef *refs, ScenePrimitives *scene, u32 start, u32 count,
                       RecordAOSSplits &record)
{
    Bounds geom;
    Bounds cent;
    const Instance *instances = (const Instance *)scene->primitives;
    for (u32 i = start; i < start + count; i++)
    {
        const Instance &instance = instances[i];
        AffineSpace &transform   = scene->affineTransforms[instance.transformIndex];
        u32 index                = instance.id;
        Assert(scene->childScenes);
        ScenePrimitives *inScene = scene->childScenes[index];
        BRef *ref                = &refs[i];

        Bounds bounds = Transform(transform, inScene->GetBounds());
        Assert((Movemask(bounds.maxP >= bounds.minP) & 0x7) == 0x7);

        ref->StoreBounds(bounds);
        geom.Extend(bounds);
        cent.Extend(bounds.minP + bounds.maxP);

        ref->instanceID = i;
#ifdef USE_QUANTIZE_COMPRESS
        auto *node   = inScene->nodePtr.GetQuantizedCompressedNode();
        ref->nodePtr = uintptr_t(inScene->nodePtr.GetPtr());
        ref->type    = inScene->nodePtr.GetType();
#else
        ref->nodePtr = inScene->nodePtr;
#endif
        ref->numPrims = inScene->numFaces;

        ErrorExit(ref->nodePtr.data, "Invalid scene: %u\n", index);
    }
    record.geomBounds = Lane8F32(-geom.minP, geom.maxP);
    record.centBounds = Lane8F32(-cent.minP, cent.maxP);
}

// NOTE: either Scene or Scene2Inst
BRef *GenerateBuildRefs(ScenePrimitives *scene, Arena *arena, RecordAOSSplits &record)
{
    u32 numInstances = scene->numPrimitives;
    u32 extEnd       = 4 * numInstances;
    BRef *b          = PushArrayNoZero(arena, BRef, extEnd);

    if (numInstances > PARALLEL_THRESHOLD)
    {
        ParallelReduce<RecordAOSSplits>(
            &record, 0, numInstances, PARALLEL_THRESHOLD,
            [&](RecordAOSSplits &record, u32 jobID, u32 start, u32 count) {
                GenerateBuildRefs(b, scene, start, count, record);
            },
            [&](RecordAOSSplits &l, const RecordAOSSplits &r) { l.Merge(r); });
    }
    else
    {
        GenerateBuildRefs(b, scene, 0, numInstances, record);
    }
    record.SetRange(0, numInstances, extEnd);
    return b;
}

template <typename Node>
void SetNodePtr(BRef *ref, Node *node, int childIndex, const ScenePrimitives *scene,
                int instanceID)
{
#ifdef USE_QUANTIZE_COMPRESS
    int size     = scene->childScenes[instanceID]->bvhPrimSize;
    ref->nodePtr = node->Child(childIndex, size);
    ref->type    = node->GetType(childIndex);
    Assert(scene->childScenes);
#else
    ref->nodePtr = node->template Child(childIndex);
#endif
}

static const f32 REBRAID_THRESHOLD = .1f;

template <typename Heuristic, typename GetNode>
void OpenBraid(const ScenePrimitives *scene, RecordAOSSplits &record, BRef *refs, u32 start,
               u32 count, u32 offset, const u32 offsetMax, Heuristic &heuristic,
               GetNode &getNode)
{
    // TIMED_FUNCTION(miscF);
    using NodeType       = typename GetNode::NodeType;
    const u32 QUEUE_SIZE = 8;

    Bounds geomBounds;
    Bounds centBounds;
    const Instance *instances = (const Instance *)scene->primitives;
    for (u32 i = start; i < start + count; i++)
    {
        BRef &ref = refs[i];
        if (heuristic(ref))
        {
            NodeType *node = getNode(ref);

            LaneF32<DefaultN> min[3];
            LaneF32<DefaultN> max[3];

            node->GetBounds(min, max);

            Lane4F32 aosMin[DefaultN];
            Lane4F32 aosMax[DefaultN];

            if constexpr (DefaultN == 4)
            {
                Transpose3x4(min[0], min[1], min[2], aosMin[0], aosMin[1], aosMin[2],
                             aosMin[3]);
                Transpose3x4(max[0], max[1], max[2], aosMax[0], aosMax[1], aosMax[2],
                             aosMax[3]);
            }
            else
            {
                Transpose3x4(Extract4<0>(min[0]), Extract4<0>(min[1]), Extract4<0>(min[2]),
                             aosMin[0], aosMin[1], aosMin[2], aosMin[3]);
                Transpose3x4(Extract4<1>(min[0]), Extract4<1>(min[1]), Extract4<1>(min[2]),
                             aosMin[4], aosMin[5], aosMin[6], aosMin[7]);
                Transpose3x4(Extract4<0>(max[0]), Extract4<0>(max[1]), Extract4<0>(max[2]),
                             aosMax[0], aosMax[1], aosMax[2], aosMax[3]);
                Transpose3x4(Extract4<1>(max[0]), Extract4<1>(max[1]), Extract4<1>(max[2]),
                             aosMax[4], aosMax[5], aosMax[6], aosMax[7]);
            }

            const Instance &instance = instances[ref.instanceID];
            AffineSpace &transform   = scene->affineTransforms[instance.transformIndex];
            Bounds bounds0           = Transform(transform, Bounds(aosMin[0], aosMax[0]));

            geomBounds.Extend(bounds0);
            centBounds.Extend(bounds0.minP + bounds0.maxP);

            Assert((Movemask(bounds0.maxP >= bounds0.minP) & 0x7) == 0x7);
            u32 numC = node->GetNumChildren();
            Assert(numC > 0 && numC <= DefaultN);

            ref.SafeStoreBounds(bounds0);
            ref.numPrims = Max(ref.numPrims / numC, 1u);
            SetNodePtr(&ref, node, 0, scene, instance.id);
            Assert(ref.nodePtr.data != 0);

            for (u32 b = 1; b < numC; b++)
            {
                Assert(offset < offsetMax);
                Bounds bounds = Transform(transform, Bounds(aosMin[b], aosMax[b]));
                Assert((Movemask(bounds.maxP >= bounds.minP) & 0x7) == 0x7);
                geomBounds.Extend(bounds);
                centBounds.Extend(bounds.minP + bounds.maxP);
                refs[offset].StoreBounds(bounds);
                refs[offset].instanceID = ref.instanceID;
                refs[offset].numPrims   = ref.numPrims;
                SetNodePtr(&refs[offset], node, b, scene, instance.id);
                Assert(refs[offset].nodePtr.data != 0);

                offset++;
            }
        }
    }
    record.geomBounds = Lane8F32(-geomBounds.minP, geomBounds.maxP);
    record.centBounds = Lane8F32(-centBounds.minP, centBounds.maxP);
    Assert(offset == offsetMax);
}

struct GetQuantizedNode
{
    using NodeType = QNode;

    __forceinline NodeType *operator()(const BRef &ref)
    {
        Assert(ref.nodePtr.data);
        Assert(ref.nodePtr.IsQuantizedNode());
        return ref.nodePtr.GetQuantizedNode();
    }
};

#ifdef USE_QUANTIZE_COMPRESS
struct GetQuantizedCompressedNode
{
    using NodeType = QCNode;
    __forceinline QCNode *operator()(const BRef &ref)
    {
        Assert(ref.nodePtr.data);
        Assert(ref.type == BVHNodeN::tyQuantizedNode);
        // Assert(ptr.IsQuantizedNode());
        return (QCNode *)ref.nodePtr.data;
    }
};
#endif

// TODO: the number of nodes that this generates can vary very very wildly (like ~6 mil), find
// out why
template <typename GetNode, i32 numObjectBins = 32>
struct HeuristicPartialRebraid
{
    using Record  = RecordAOSSplits;
    using PrimRef = BRef;
    using OBin    = HeuristicAOSObjectBinning<numObjectBins, BRef>;

    ScenePrimitives *scene;
    BRef *buildRefs;
    GetNode getNode;
    u32 logBlockSize;

    HeuristicPartialRebraid() {}
    HeuristicPartialRebraid(ScenePrimitives *scene, BRef *data, u32 logBlockSize = 0)
        : scene(scene), buildRefs(data), logBlockSize(logBlockSize)
    {
    }
    Split Bin(Record &record)
    {
        const Instance *instances = (const Instance *)scene->primitives;
        u32 choiceDim             = 0;
        f32 maxExtent             = neg_inf;
        for (u32 d = 0; d < 3; d++)
        {
            f32 extent = record.geomMax[d] + record.geomMin[d];
            if (extent > maxExtent)
            {
                maxExtent = extent;
                choiceDim = d;
            }
        }
        f32 threshold = REBRAID_THRESHOLD * maxExtent;
#ifdef USE_QUANTIZE_COMPRESS
        auto heuristic = [&](const BRef &ref) -> bool {
            return ref.type < BVHNodeN::tyLeaf &&
                   ref.max[choiceDim] + ref.min[choiceDim] > threshold;
        };
#else
        auto heuristic = [&](const BRef &ref) -> bool {
            return !ref.nodePtr.IsLeaf() &&
                   ref.max[choiceDim] + ref.min[choiceDim] > threshold;
        };
#endif

        u64 popPos = 0;
        // conditions to test:
        // 1. if all the nodes belong to the same geometry, break
        // 2. if there are 4 or less nodes, and there is no overlap between the bvhs, also
        // break
        // 3. also obviously break if there is no space for node opening

        if (record.ExtSize() && record.count <= 4)
        {
            for (u32 i = 0; i < record.count; i++)
            {
                for (u32 j = i + 1; j < record.count; j++)
                {
                    Lane8F32 intersection = Min(buildRefs[record.start + i].Load(),
                                                buildRefs[record.start + j].Load());
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
            u32 geomID = instances[buildRefs[record.start].instanceID].id;
            if (record.count > PARALLEL_THRESHOLD)
            {
                TempArena temp = ScratchStart(0, 0);
                struct Props
                {
                    u32 count     = 0;
                    bool sameType = true;
                };
                Props prop;
                ParallelForOutput output =
                    ParallelFor<Props>(temp, record.start, record.count, PARALLEL_THRESHOLD,
                                       [&](Props &props, u32 jobID, u32 start, u32 count) {
                                           bool commonGeomID = true;
                                           u32 openCount     = 0;
                                           for (u32 i = start; i < start + count; i++)
                                           {
                                               const BRef &ref = buildRefs[i];
                                               u32 objectID    = instances[ref.instanceID].id;
                                               commonGeomID &= objectID == geomID;
                                               if (heuristic(ref))
                                               {
                                                   u32 numToAdd =
                                                       getNode(ref)->GetNumChildren() - 1;
                                                   openCount += numToAdd;
                                               }
                                           }
                                           props.sameType &= commonGeomID;
                                           props.count = openCount;
                                       });
                Reduce(prop, output, [&](Props &l, const Props &r) {
                    l.sameType &= r.sameType;
                    l.count += r.count;
                });
                if (prop.sameType)
                {
                    record.extEnd = record.start + record.count;
                }
                else if (prop.count && prop.count <= record.ExtSize())
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
                    RecordAOSSplits openRecord(neg_inf);
                    ParallelReduce(
                        &openRecord, record.start, record.count, PARALLEL_THRESHOLD,
                        [&](RecordAOSSplits &record, u32 jobID, u32 start, u32 count) {
                            Props &prop = props[jobID];
                            u32 max =
                                jobID == output.num - 1 ? offset : props[jobID + 1].count;
                            OpenBraid(scene, record, buildRefs, start, count, prop.count, max,
                                      heuristic, getNode);
                        },
                        [&](RecordAOSSplits &l, const RecordAOSSplits &r) { l.Merge(r); });
                    record.SafeMerge(openRecord);
                    record.count = offset - record.start;
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
                    u32 objectID    = instances[ref.instanceID].id;
                    commonGeomID &= objectID == geomID;
                    if (heuristic(ref)) count += getNode(ref)->GetNumChildren() - 1;
                }
                if (commonGeomID)
                {
                    record.extEnd = record.start + record.count;
                }
                else if (count && count <= record.ExtSize())
                {
                    RecordAOSSplits openRecord(neg_inf);
                    OpenBraid(scene, openRecord, buildRefs, record.start, record.count,
                              record.End(), record.End() + count, heuristic, getNode);
                    record.SafeMerge(openRecord);
                    record.count = record.End() + count - record.start;
                }
            }
        }
        OBin *objectBinHeuristic;
        struct Split objectSplit =
            SAHObjectBinning(record, buildRefs, objectBinHeuristic, popPos, logBlockSize);
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
            mid = PartitionParallel(heuristic, buildRefs, split, record.start, record.count,
                                    outLeft, outRight);
        }
        MoveExtendedRanges(split, record.End(), record, buildRefs, mid, outLeft, outRight);
        ArenaPopTo(temp.arena, split.allocPos);
    }
};

} // namespace rt
#endif
