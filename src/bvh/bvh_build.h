#ifndef BVH_BUILD_H
#define BVH_BUILD_H
namespace rt
{
template <i32 N>
struct QuantizedNode;

#define CREATE_NODE() template <typename Record> \
__forceinline void operator()(const Record *records, const u32 numRecords, NodeType *result)

template <i32 N>
struct QuantizedNode
{
    StaticAssert(N == 4 || N == 8, NMustBe4Or8);

    // TODO: the bottom 4 bits can be used for something (and maybe the top 7 bits too)
    // QuantizedNode<N> *internalOffset;
    uintptr_t internalOffset;
    uintptr_t leafOffset;
    u8 lowerX[N];
    u8 lowerY[N];
    u8 lowerZ[N];

    // NOTE: upperX = 255 when node is invalid
    u8 upperX[N];
    u8 upperY[N];
    u8 upperZ[N];

    Vec3f minP;
    u8 scale[3];
    u8 meta;
};

template <i32 N>
struct CreateQuantizedNode
{
    using NodeType = QuantizedNode<N>;

    CREATE_NODE()
    {
        const f32 MIN_QUAN = 0.f;
        const f32 MAX_QUAN = 255.f;
        Lane4F32 boundsMinP(pos_inf);
        Lane4F32 boundsMaxP(neg_inf);

        for (u32 i = 0; i < numRecords; i++)
        {
            boundsMinP = Min(boundsMinP, records[i].geomBounds.minP);
            boundsMaxP = Max(boundsMaxP, records[i].geomBounds.maxP);
        }
        result->minP = ToVec3f(boundsMinP);

        Lane4F32 diff = boundsMaxP - boundsMinP;

        f32 expX = Ceil(Log2f(diff[0] / 255.f));
        f32 expY = Ceil(Log2f(diff[1] / 255.f));
        f32 expZ = Ceil(Log2f(diff[2] / 255.f));

        Lane4U32 shift = Flooru(Lane4F32(expX, expY, expZ, 0.f)) + 127;

        Lane4F32 pow = AsFloat(shift << 23);

        Vec3lf<N> powVec;
        // TODO: for N = 8, this needs to be shuffle across
        powVec.x = Shuffle<0>(pow);
        powVec.y = Shuffle<1>(pow);
        powVec.z = Shuffle<2>(pow);

        Assert(numRecords <= N);
        Vec3lf<N> min;
        Vec3lf<N> max;

        if constexpr (N == 4)
        {
            LaneF32<N> min02xy = UnpackLo(records[0].geomBounds.minP, records[2].geomBounds.minP);
            LaneF32<N> min13xy = UnpackLo(records[1].geomBounds.minP, records[3].geomBounds.minP);

            LaneF32<N> min02z_ = UnpackHi(records[0].geomBounds.minP, records[2].geomBounds.minP);
            LaneF32<N> min13z_ = UnpackHi(records[1].geomBounds.minP, records[3].geomBounds.minP);

            LaneF32<N> max02xy = UnpackLo(records[0].geomBounds.maxP, records[2].geomBounds.maxP);
            LaneF32<N> max13xy = UnpackLo(records[1].geomBounds.maxP, records[3].geomBounds.maxP);

            LaneF32<N> max02z_ = UnpackHi(records[0].geomBounds.maxP, records[2].geomBounds.maxP);
            LaneF32<N> max13z_ = UnpackHi(records[1].geomBounds.maxP, records[3].geomBounds.maxP);

            min.x = UnpackLo(min02xy, min13xy);
            min.y = UnpackHi(min02xy, min13xy);
            min.z = UnpackLo(min02z_, min13z_);

            max.x = UnpackLo(max02xy, max13xy);
            max.y = UnpackHi(max02xy, max13xy);
            max.z = UnpackLo(max02z_, max13z_);
        }
        else if constexpr (N == 8)
        {
            LaneF32<N> min04(records[0].geomBounds.minP, records[4].geomBounds.minP);
            LaneF32<N> min26(records[2].geomBounds.minP, records[6].geomBounds.minP);

            LaneF32<N> min15(records[1].geomBounds.minP, records[5].geomBounds.minP);
            LaneF32<N> min37(records[3].geomBounds.minP, records[7].geomBounds.minP);

            LaneF32<N> max04(records[0].geomBounds.maxP, records[4].geomBounds.maxP);
            LaneF32<N> max26(records[2].geomBounds.maxP, records[6].geomBounds.maxP);

            LaneF32<N> max15(records[1].geomBounds.maxP, records[5].geomBounds.maxP);
            LaneF32<N> max37(records[3].geomBounds.maxP, records[7].geomBounds.maxP);

            // x0 x2 y0 y2 x4 x6 y4 y6
            // x1 x3 y1 y3 x5 x7 y5 y7

            // z0 z2 _0 _2 z4 z6 _4 _6
            // z1 z3 _1 _3 z5 z7 _5 _7

            LaneF32<N> min0246xy = UnpackLo(min04, min26);
            LaneF32<N> min1357xy = UnpackLo(min15, min37);
            min.x                = UnpackLo(min0246xy, min1357xy);
            min.y                = UnpackHi(min0246xy, min1357xy);
            min.z                = UnpackLo(UnpackHi(min04, min26), UnpackHi(min15, min37));

            LaneF32<N> max0246xy = UnpackLo(max04, max26);
            LaneF32<N> max1357xy = UnpackLo(max15, max37);
            max.x                = UnpackLo(max0246xy, max1357xy);
            max.y                = UnpackHi(max0246xy, max1357xy);
            max.z                = UnpackLo(UnpackHi(max04, max26), UnpackHi(max15, max37));
        }

        Vec3lf<N> nodeMin;
        nodeMin.x = Shuffle<0>(boundsMinP);
        nodeMin.y = Shuffle<1>(boundsMinP);
        nodeMin.z = Shuffle<2>(boundsMinP);

        Vec3lf<N> qNodeMin = Floor((min - nodeMin) / powVec);
        Vec3lf<N> qNodeMax = Ceil((max - nodeMin) / powVec);

        Lane4F32 maskMinX = FMA(powVec.x, qNodeMin.x, nodeMin.x) > min.x;
        TruncateToU8(result->lowerX, Max(Select(maskMinX, qNodeMin.x - 1, qNodeMin.x), MIN_QUAN));
        Lane4F32 maskMinY = FMA(powVec.y, qNodeMin.y, nodeMin.y) > min.y;
        TruncateToU8(result->lowerY, Max(Select(maskMinY, qNodeMin.y - 1, qNodeMin.y), MIN_QUAN));
        Lane4F32 maskMinZ = FMA(powVec.z, qNodeMin.z, nodeMin.z) > min.z;
        TruncateToU8(result->lowerZ, Max(Select(maskMinZ, qNodeMin.z - 1, qNodeMin.z), MIN_QUAN));

        Lane4F32 maskMaxX = FMA(powVec.x, qNodeMax.x, nodeMin.x) < max.x;
        TruncateToU8(result->upperX, Min(Select(maskMaxX, qNodeMax.x + 1, qNodeMax.x), MAX_QUAN));
        Lane4F32 maskMaxY = FMA(powVec.y, qNodeMax.y, nodeMin.y) < max.y;
        TruncateToU8(result->upperY, Min(Select(maskMaxY, qNodeMax.y + 1, qNodeMax.y), MAX_QUAN));
        Lane4F32 maskMaxZ = FMA(powVec.z, qNodeMax.z, nodeMin.z) < max.z;
        TruncateToU8(result->upperZ, Min(Select(maskMaxZ, qNodeMax.z + 1, qNodeMax.z), MAX_QUAN));

        Assert(shift[0] <= 255 && shift[1] <= 255 && shift[2] <= 255);
        result->scale[0] = (u8)shift[0];
        result->scale[1] = (u8)shift[1];
        result->scale[2] = (u8)shift[2];
    }
};

template <i32 N>
struct UpdateQuantizedNode;

template <i32 N>
struct UpdateQuantizedNode;

template <>
struct UpdateQuantizedNode<4>
{
    using NodeType = QuantizedNode<4>;
    template <typename Record>
    __forceinline void operator()(Arena *arena, NodeType *parent, const Record *records, NodeType *children,
                                  const u32 *leafIndices, const u32 leafCount)
    {
        // NOTE: for leaves, top 3 bits represent binary count. bottom 5 bits represent offset from base offset.
        // 0 denotes a node, 1 denotes invalid.

        u32 primTotal = 0;
        for (u32 i = 0; i < leafCount; i++)
        {
            u32 leafIndex        = leafIndices[i];
            const Record *record = &records[leafIndex];
            primTotal += record->range.Size();
        }

        Assert(primTotal <= 12);
        u32 *primIndices       = PushArray(arena, u32, primTotal);
        parent->internalOffset = (uintptr_t)children;
        parent->leafOffset     = (uintptr_t)primIndices;
        parent->meta           = 0;
        u32 offset             = 0;
        for (u32 i = 0; i < leafCount; i++)
        {
            u32 leafIndex        = leafIndices[i];
            const Record *record = &records[leafIndex];
            u32 primCount        = record->range.Size();
            for (u32 j = record->range.start; j < record->range.End(); j++)
            {
                primIndices[offset++] = j;
            }

            Assert(primCount >= 1 && primCount <= 3);
            Assert(leafIndex >= 0 && leafIndex <= 3);

            parent->meta |= primCount << (leafIndex * 2);
        }
    }
};

template <>
struct UpdateQuantizedNode<8>
{
    using NodeType = QuantizedNode<8>;
    __forceinline void operator()(Arena *arena, NodeType *parent, const Record *records, NodeType *children,
                                  const u32 *leafIndices, const u32 leafCount)
    {
        // NOTE: for leaves, top 3 bits represent binary count. bottom 5 bits represent offset from base offset.
        // 0 denotes a node, 1 denotes invalid.
        Assert(((u64)parent->internalOffset & 0xf) == 0);
        Assert(((u64)parent->leafOffset & 0xf) == 0);

        uintptr_t internalOffset = (uintptr_t)children;

        u32 primTotal = 0;
        for (u32 i = 0; i < leafCount; i++)
        {
            u32 leafIndex        = leafIndices[i];
            const Record *record = &records[leafIndex];
            primTotal += record->range.end - record->range.start;
        }
        Assert(primTotal <= 24);
        u32 *primIndices     = PushArray(arena, u32, primTotal);
        uintptr_t leafOffset = (uintptr_t)primIndices;
        parent->meta         = 0;
        u32 offset           = 0;
        for (u32 i = 0; i < leafCount; i++)
        {
            u32 leafIndex        = leafIndices[i];
            const Record *record = &records[leafIndex];
            u32 primCount        = record->range.end - record->range.start;

            for (u32 j = record->range.start; j < record->range.end; j++)
            {
                primIndices[offset++] = j;
            }
            Assert(primCount >= 1 && primCount <= 3);

            if (leafIndex >= 6)
            {
                internalOffset |= (primCount << ((leafIndex - 6) * 2));
            }
            else if (leafIndex >= 4)
            {
                leafOffset |= (primCount << ((leafIndex - 4) * 2));
            }
            else
            {
                parent->meta |= primCount << (leafIndex * 2);
            }
        }
    }
};

// NOTE: ptr MUST be aligned to 16 bytes, bottom 4 bits store the type, top 7 bits store the count

template <i32 N>
struct AABBNode
{
    struct Create
    {
        using NodeType = AABBNode;
        CREATE_NODE()
        {
        }
    };

    LaneF32<N> lowerX;
    LaneF32<N> lowerY;
    LaneF32<N> lowerZ;

    LaneF32<N> upperX;
    LaneF32<N> upperY;
    LaneF32<N> upperZ;
};

template <i32 N, typename NodeType>
struct BVHN
{
    BVHN() {}
    NodeType *root;
};

template <i32 N>
using QuantizedNode4 = QuantizedNode<4>;
template <i32 N>
using QuantizedNode8 = QuantizedNode<8>;

template <i32 N>
using BVHQuantized = BVHN<N, QuantizedNode<N>>;
typedef BVHQuantized<4> BVH4Quantized;
typedef BVHQuantized<8> BVH8Quantized;

template <i32 N, typename Heur, typename CreateNode, typename UpdateNode, typename Prim>
struct BuildFuncs
{
    using NodeType  = typename CreateNode::NodeType;
    using Primitive = Prim;
    using Heuristic = Heur;

    CreateNode createNode;
    UpdateNode updateNode;
    Heuristic heuristic;
};

// template <i32 N, typename Heuristic> //, typename CreateNode, typename UpdateNode, typename Prim>
// struct BuildFuncs<N, Heuristic, CreateQuantizedNode<N>, UpdateQuantizedNode<N>, TriangleMesh>
// {
//     using NodeType  = typename CreateQuantizedNode<N>::NodeType;
//     using Primitive = TriangleMesh;
//
//     CreateQuantizedNode<N> createNode;
//     UpdateQuantizedNode<N> updateNode;
//     Heuristic heuristic;
// };

template <i32 N>
using BLAS_SOA_ObjectBin_QuantizedNode_TriangleLeaf_Funcs =
    BuildFuncs<
        N,
        HeuristicSOAObjectBinning<32>,
        CreateQuantizedNode<N>,
        UpdateQuantizedNode<N>,
        TriangleMesh>;

template <i32 N>
using BLAS_SOA_SBVH_QuantizedNode_TriangleLeaf_Funcs =
    BuildFuncs<
        N,
        HeuristicSpatialSplits<32, 16>,
        CreateQuantizedNode<N>,
        UpdateQuantizedNode<N>,
        TriangleMesh>;

template <i32 N, typename BuildFunctions>
struct BVHBuilder
{
    using NodeType      = typename BuildFunctions::NodeType;
    using Primitive     = typename BuildFunctions::Primitive;
    using Heuristic     = typename BuildFunctions::Heuristic;
    using Record        = typename Heuristic::Record;
    using PrimitiveData = typename Record::PrimitiveData;
    BuildFunctions f;

    Arena **arenas;
    Primitive *primitives;
    Heuristic heuristic;

    BVHBuilder() {}
    NodeType *BuildBVHRoot(BuildSettings settings, Record &record);
    __forceinline u32 BuildNode(BuildSettings settings, const Record &record, Record *childRecords, u32 &numChildren);
    void BuildBVH(BuildSettings settings, NodeType *parent, Record *records, u32 numChildren);

    BVHN<N, NodeType> BuildBVH(BuildSettings settings, Arena **inArenas, Primitive *inRawPrims, Record &record);
};

template <i32 N>
using SBVHBuilderTriangleMesh = BVHBuilder<N, BLAS_SOA_SBVH_QuantizedNode_TriangleLeaf_Funcs<N>>;
template <i32 N>
using BVHBuilderTriangleMesh = BVHBuilder<N, BLAS_SOA_ObjectBin_QuantizedNode_TriangleLeaf_Funcs<N>>;

template <i32 N, typename BuildFunctions>
typename BVHBuilder<N, BuildFunctions>::NodeType *BVHBuilder<N, BuildFunctions>::BuildBVHRoot(BuildSettings settings, Record &record)
{
    Record childRecords[N];
    u32 numChildren;
    u32 result = BuildNode(settings, record, childRecords, numChildren);

    Arena *arena   = arenas[GetThreadIndex()];
    NodeType *root = PushStruct(arena, NodeType);
    if (result)
    {
        f.createNode(childRecords, numChildren, root);
        BuildBVH(settings, root, childRecords, numChildren);
    }
    // If the root is a leaf
    else
    {
        u32 leafIndex = 0;
        f.createNode(&record, 1, root);
        f.updateNode(arena, root, &record, 0, &leafIndex, 1);
    }
    return root;
}

template <i32 N, typename BuildFunctions>
__forceinline u32 BVHBuilder<N, BuildFunctions>::BuildNode(BuildSettings settings, const Record &record,
                                                           Record *childRecords, u32 &numChildren)

{
    u32 total = record.range.Size();
    Assert(total > 0);

    if (total == 1) return 0;

    {
        Split split = heuristic.Bin(record, 1);
        heuristic.Split(split, record, childRecords[0], childRecords[1]);

        if (!(childRecords[0].range.count < record.range.count && childRecords[1].range.count < record.range.count))
        {
            split = heuristic.Bin(record, 1);
            heuristic.Split(split, record, childRecords[0], childRecords[1]);
            Assert(false);
        }

        // NOTE: multiply both by the area instead of dividing
        f32 area     = HalfArea(record.geomBounds);
        f32 leafSAH  = settings.intCost * area * total; //((total + (1 << settings.logBlockSize) - 1) >> settings.logBlockSize);
        f32 splitSAH = settings.travCost * area + settings.intCost * split.bestSAH;

        if (total <= settings.maxLeafSize && leafSAH <= splitSAH) return 0;
    }

    // N - 1 splits produces N children
    for (numChildren = 2; numChildren < N; numChildren++)
    {
        i32 bestChild = -1;
        f32 maxArea   = neg_inf;
        for (u32 recordIndex = 0; recordIndex < numChildren; recordIndex++)
        {
            Record &childRecord = childRecords[recordIndex];
            if (childRecord.range.Size() <= settings.maxLeafSize) continue;

            f32 childArea = HalfArea(childRecord.geomBounds);
            if (childArea > maxArea)
            {
                bestChild = recordIndex;
                maxArea   = childArea;
            }
        }
        if (bestChild == -1) break;

        Split split = heuristic.Bin(childRecords[bestChild]);

        Record out;
        heuristic.Split(split, childRecords[bestChild], out, childRecords[numChildren]);

        if (!(childRecords[bestChild].range.count < record.range.count))
        {
            split = heuristic.Bin(childRecords[bestChild], 1);
            heuristic.Split(split, childRecords[bestChild], out, childRecords[numChildren]);
            Assert(false);
        }
        // PartitionResult result;
        // PartitionParallel(split, childRecords[bestChild], &result);

        // Record &childRecord = childRecords[bestChild];
        // PrimData *prim      = childRecord.data;
        // u32 start           = childRecord.start;
        // u32 end             = childRecord.end;

        // Assert(start != end && start != result.mid && result.mid != end);

        childRecords[bestChild] = out; // Record(prim, result.geomBoundsL, result.centBoundsL, start, result.mid);
        // childRecords[numChildren] = Record(prim, result.geomBoundsR, result.centBoundsR, result.mid, end);
    }

    // Test
    // for (u32 i = 0; i < numChildren; i++)
    // {
    //     Lane4F32 maskMin(False);
    //     Lane4F32 maskMax(False);
    //     Bounds test;
    //     Record &childRecord = childRecords[i];
    //     for (u32 j = childRecord.start; j < childRecord.end; j++)
    //     {
    //         AABB bounds = Primitive::Bounds(primitives, j);
    //
    //         Bounds b;
    //         b.minP = Lane4F32(bounds.minP);
    //         b.maxP = Lane4F32(bounds.maxP);
    //
    //         Assert(((Movemask(record.data[j].minP == b.minP) & 0x7) == 0x7) &&
    //                ((Movemask(record.data[j].maxP == b.maxP) & 0x7) == 0x7));
    //
    //         test.Extend(b);
    //         maskMin = maskMin | (b.minP == childRecord.geomBounds.minP);
    //         maskMax = maskMax | (b.maxP == childRecord.geomBounds.maxP);
    //         Assert(childRecord.geomBounds.Contains(b));
    //     }
    //     u32 minBits = Movemask(maskMin);
    //     u32 maxBits = Movemask(maskMax);
    //     Assert(((minBits & 0x7) == 0x7) && ((maxBits & 0x7) == 0x7));
    // }
    return 1;
}

template <i32 N, typename BuildFunctions>
void BVHBuilder<N, BuildFunctions>::BuildBVH(BuildSettings settings, NodeType *parent, Record *records, u32 inNumChildren)
{
    Record allChildRecords[N][N];
    u32 allNumChildren[N];

    u32 childNodeIndices[N];
    u32 childLeafIndices[N];
    u32 nodeCount = 0;
    u32 leafCount = 0;

    Assert(inNumChildren <= N);

    // TODO: multithread this loop
    for (u32 childIndex = 0; childIndex < inNumChildren; childIndex++)
    {
        Record *childRecords = allChildRecords[childIndex];
        Record &record       = records[childIndex];
        u32 &numChildren     = allNumChildren[childIndex];
        u32 result           = BuildNode(settings, record, childRecords, numChildren);

        childNodeIndices[nodeCount] = childIndex;
        childLeafIndices[leafCount] = childIndex;
        nodeCount += result;
        leafCount += !result;
    }

    Arena *currentArena = arenas[GetThreadIndex()];

    NodeType *children = PushArray(currentArena, NodeType, nodeCount);

    for (u32 i = 0; i < nodeCount; i++)
    {
        u32 childNodeIndex = childNodeIndices[i];
        f.createNode(allChildRecords[childNodeIndex], allNumChildren[childNodeIndex], &children[childNodeIndex]);
    }

    // Updates the parent
    f.updateNode(currentArena, parent, records, children, childLeafIndices, leafCount);

    // TODO: multithread this loop
    for (u32 i = 0; i < nodeCount; i++)
    {
        u32 childNodeIndex = childNodeIndices[i];
        BuildBVH(settings, &children[childNodeIndex], allChildRecords[childNodeIndex], allNumChildren[childNodeIndex]);
    }
}

template <i32 N, typename BuildFunctions>
BVHN<N, typename BVHBuilder<N, BuildFunctions>::NodeType>
BVHBuilder<N, BuildFunctions>::BuildBVH(BuildSettings settings,
                                        Arena **inArenas,
                                        Primitive *inRawPrims,
                                        Record &record)
{
    arenas = inArenas;

    // const u32 groupSize = count / 16;

    // jobsystem::Counter counter = {};
    //
    // PrimData *prims = PushArray(arenas[GetThreadIndex()], PrimData, count);
    //
    // jobsystem::KickJobs(&counter, count, groupSize, [&](jobsystem::JobArgs args) {
    //     u32 index         = args.jobId;
    //     AABB bounds       = Primitive::Bounds(inRawPrims, index);
    //     prims[index].minP = Lane4F32(bounds.minP);
    //     prims[index].maxP = Lane4F32(bounds.maxP);
    //     prims[index].SetGeomID(0);
    //     prims[index].SetPrimID(index);
    // });
    // jobsystem::WaitJobs(&counter);
    // for (u32 i = 0; i < count; i++)
    // {
    //     u32 index         = i;
    //     AABB bounds       = Primitive::Bounds(inRawPrims, index);
    //     prims[index].minP = Lane4F32(bounds.minP);
    //     prims[index].maxP = Lane4F32(bounds.maxP);
    //     prims[index].SetGeomID(0);
    //     prims[index].SetPrimID(index);
    // }

    primitives = inRawPrims;
    // Record record;
    // record = jobsystem::ParallelReduce<Record>(
    //     count, 1024,
    //     [&](Record &record, u32 start, u32 count) {
    //     for (u32 i = start; i < start + count; i++)
    //     {
    //         PrimData *prim    = &prims[i];
    //         Lane4F32 centroid = (prim->minP + prim->maxP) * 0.5f;
    //         record.geomBounds.Extend(prim->minP, prim->maxP);
    //         record.centBounds.Extend(centroid);
    //     } },
    //     [&](Record &a, Record &b) {
    //     a.geomBounds.Extend(b.geomBounds);
    //     a.centBounds.Extend(b.centBounds); });

    // record.data  = record.data;
    // record.start = record.range.start;
    // record.end   = record.range.count;

    // BVH<N> *result = PushStruct(arenas[GetThreadIndex()], BVH<N>);
    BVHN<N, NodeType> result;
    result.root = BuildBVHRoot(settings, record);
    // if (settings.twoLevel)
    // {
    // }
    // else
    // {
    //     BuildBVH(settings, record);
    // }
    return result;
}

template <i32 N>
BVHQuantized<N> BuildQuantizedSBVH(BuildSettings settings,
                                   Arena **inArenas,
                                   TriangleMesh *mesh,
                                   RecordSOASplits &record)
{
    SBVHBuilderTriangleMesh<N> builder;
    HeuristicSpatialSplits heuristic(inArenas, mesh, HalfArea(record.geomBounds));
    builder.heuristic = heuristic;
    return builder.BuildBVH(settings, inArenas, mesh, record);
}

} // namespace rt
#endif
