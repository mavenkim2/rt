#ifndef CREATE_CLUSTERS_H
#define CREATE_CLUSTERS_H
namespace rt
{

static const int clusterSize = 64;

template <i32 N, typename BuildFunctions>
BVHNode<N> BVHBuilder<N, BuildFunctions>::BuildBVH(const BuildSettings &settings,
                                                   Record &record, bool parallel)
{
    Assert(record.count > 0);

    Record childRecords[N];
    u32 numChildren = 0;
    if (record.count == 1)
    {
        return 0;
    }

    Split split = heuristic.Bin(record);

    const u32 blockAdd   = settings.blockAdd;
    const u32 blockShift = settings.logBlockSize;
    // NOTE: multiply both by the area instead of dividing
    f32 area     = HalfArea(record.geomBounds);
    f32 leafSAH  = settings.intCost * area * ((record.count + blockAdd) >> blockShift);
    f32 splitSAH = settings.travCost * area + settings.intCost * split.bestSAH;

    if (record.count < clusterSize && leafSAH <= splitSAH)
    {
        heuristic.FlushState(split);
        return 0;
    }
    heuristic.Split(split, record, childRecords[0], childRecords[1]);

    // N - 1 splits produces N children
    for (numChildren = 2; numChildren < N; numChildren++)
    {
        i32 bestChild = -1;
        f32 maxArea   = neg_inf;
        for (u32 recordIndex = 0; recordIndex < numChildren; recordIndex++)
        {
            Record &childRecord = childRecords[recordIndex];
            if (childRecord.count <= settings.maxLeafSize) continue;

            f32 childArea = HalfArea(childRecord.geomBounds);
            if (childArea > maxArea)
            {
                bestChild = recordIndex;
                maxArea   = childArea;
            }
        }
        if (bestChild == -1) break;

        split = heuristic.Bin(childRecords[bestChild]);

        Record out;
        heuristic.Split(split, childRecords[bestChild], out, childRecords[numChildren]);

        childRecords[bestChild] = out;
    }

    BVHNode<N> childNodes[N];

    if (parallel)
    {
        scheduler.ScheduleAndWait(numChildren, 1, [&](u32 jobID) {
            bool childParallel = childRecords[jobID].count >= BUILD_PARALLEL_THRESHOLD;
            childNodes[jobID]  = BuildBVH(settings, childRecords[jobID], childParallel);
        });
    }
    else
    {
        for (u32 i = 0; i < numChildren; i++)
        {
            childNodes[i] = BuildBVH(settings, childRecords[i], false);
        }
    }

    Arena *currentArena = arenas[GetThreadIndex()];

    for (u32 i = 0; i < numChildren; i++)
    {
        if (childNodes[i].data == 0)
        {
            u32 offset                = 0;
            const Record &childRecord = childRecords[i];
            u32 numPrims              = (childRecord.count + blockAdd) >> blockShift;
            Assert(numPrims <= settings.maxLeafSize);
            LeafType *primIDs =
                PushArrayNoZeroTagged(currentArena, LeafType, numPrims, MemoryType_BVH);
            Assert(currentArena->current->align == 16);
            u32 begin = childRecord.start;
            u32 end   = childRecord.start + childRecord.count;
            while (begin < end)
            {
                Assert(offset < numPrims);
                primIDs[offset++].Fill(scene, primRefs, begin, end);
            }
            Assert(begin == end);
            Assert(offset == numPrims);
            childNodes[i] = BVHNode<N>::EncodeLeaf(primIDs, numPrims);
        }
    }
    threadLocalStatistics[GetThreadIndex()].misc2 += 1;

    NodeType *node = PushStructNoZeroTagged(currentArena, NodeType, MemoryType_BVH);
    Assert(currentArena->current->align == 16);
    f.createNode(childRecords, numChildren, node);
    f.updateNode(node, childNodes, numChildren);

    auto result = BVHNode<N>::EncodeNode(node);
    return result;
}
} // namespace rt
#endif
