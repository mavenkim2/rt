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

typedef BuildRef<4> BuildRef4;
typedef BuildRef<8> BuildRef8;

template <i32 N>
BuildRef<N> *GenerateBuildRefs(Scene2 *scene, Arena *arena, RecordAOSSplits &record)
{
    Instance *instances = scene->instances;
    u32 numInstances    = scene->numInstances;
    BuildRef<N> *b      = PushArrayNoZero(arena, BuildRef<N>, 4 * numInstances);

    Bounds geom;
    Bounds cent;
    for (u32 i = 0; i < numInstances; i++)
    {
        Instance &instance = instances[i];
        // TODO: make this work for groups of curves
        AffineSpace &transform = scene->affineTransforms[instance.transformIndex];
        QuadMeshGroup *group   = &scene->quadMeshGroups[instance.geomID.GetIndex()];
        BuildRef<N> *ref       = &b[i];

        Bounds bounds;
        u32 numPrims = 0;
        for (u32 meshIndex = 0; meshIndex < group->numMeshes; meshIndex++)
        {
            QuadMesh *mesh = &group->meshes[meshIndex];
            u32 numFaces   = mesh->numVertices / 4;
            numPrims += numFaces;
            for (u32 primIndex = 0; primIndex < numFaces; primIndex++)
            {
                Vec3f p0 = mesh->p[primIndex * 4 + 0];
                Vec3f p1 = mesh->p[primIndex * 4 + 1];
                Vec3f p2 = mesh->p[primIndex * 4 + 2];
                Vec3f p3 = mesh->p[primIndex * 4 + 3];

                Vec3f min = Min(p0, Min(p1, Min(p2, p3)));
                Vec3f max = Max(p0, Max(p1, Max(p2, p3)));
                Lane4F32 mins(min);
                Lane4F32 maxs(max);
                bounds.Extend(mins, maxs);
            }
        }

        Lane4F32::StoreU(ref->min, -bounds.minP);
        Lane4F32::StoreU(ref->max, bounds.maxP);
        geom.Extend(bounds);
        cent.Extend(bounds.minP + bounds.maxP);

        Assert((instance.bvhNode & 0xf) == 0);

        ref->objectID = i;
        ref->nodePtr  = group->nodePtr;
        ref->numPrims = numPrims;
    }
    record.geomBounds = Lane8F32(-geom.minP, geom.maxP);
    record.centBounds = Lane8F32(-cent.minP, cent.maxP);
    return b;
}

static const f32 REBRAID_THRESHOLD = .1f;
template <u32 N>
void OpenBraid(const RecordAOSSplits &record, BuildRef<N> *refs, u32 start, u32 count, std::atomic<u32> &refOffset)
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
        BuildRef<N> &ref      = refs[i];
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
            u32 offset = refOffset.fetch_add(childAdd, std::memory_order_acq_rel);
            for (u32 testIndex = 0; testIndex < QUEUE_SIZE; testIndex++)
            {
                u32 refID              = refIDQueue[openCount + testIndex];
                QuantizedNode<N> *node = refs[refID].nodePtr.GetQuantizedNode();
                f32 minX[N];
                f32 minY[N];
                f32 minZ[N];
                f32 maxX[N];
                f32 maxY[N];
                f32 maxZ[N];

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

template <u32 N, i32 numObjectBins = 32>
struct HeuristicPartialRebraid
{
    using Record        = RecordAOSSplits;
    using PrimitiveData = BuildRef<N>;
    using OBin          = HeuristicAOSObjectBinning<numObjectBins, BuildRef<N>>;

    BuildRef<N> *buildRefs;

    HeuristicPartialRebraid() {}
    HeuristicPartialRebraid(BuildRef<N> *data) : buildRefs(data) {}
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
