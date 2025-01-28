#ifndef BVH_BUILD_H
#define BVH_BUILD_H
#include "bvh_types.h"
namespace rt
{

template <i32 N>
Lane4U32 ExponentialQuantize(const Lane4F32 &diff, Vec3lf<N> &scaleVec)
{
    const f32 divisor = 1 / 255.f;

    f32 expX = diff[0] == 0.f ? 0.f : Ceil(Log2f(diff[0] * divisor));
    f32 expY = diff[1] == 0.f ? 0.f : Ceil(Log2f(diff[1] * divisor));
    f32 expZ = diff[2] == 0.f ? 0.f : Ceil(Log2f(diff[2] * divisor));

    Lane4U32 shift = Flooru(Lane4F32(expX, expY, expZ, 0.f)) + 127;

    Lane4F32 pow = AsFloat(shift << 23);

    scaleVec.x = Shuffle<0>(LaneF32<N>(pow));
    scaleVec.y = Shuffle<1>(LaneF32<N>(pow));
    scaleVec.z = Shuffle<2>(LaneF32<N>(pow));
    return shift;
}

template <i32 N>
Vec3f ScalarQuantize(const Lane4F32 &diff, Vec3lf<N> &scaleVec)
{
    const f32 divisor = 1 / 255.f;

    f32 scaleX = diff[0] == 0.f ? 0.f : diff[0] * divisor;
    f32 scaleY = diff[1] == 0.f ? 0.f : diff[1] * divisor;
    f32 scaleZ = diff[2] == 0.f ? 0.f : diff[2] * divisor;

    scaleVec.x = Shuffle<0>(LaneF32<N>(scaleX));
    scaleVec.y = Shuffle<1>(LaneF32<N>(scaleY));
    scaleVec.z = Shuffle<2>(LaneF32<N>(scaleZ));

    return Vec3f(scaleX, scaleY, scaleZ);
}

template <i32 N>
struct CreateQuantizedNode
{
    template <typename Record, typename NodeType>
    __forceinline void operator()(const Record *records, const u32 numRecords,
                                  NodeType *result)
    {
        const f32 MIN_QUAN = 0.f;
        const f32 MAX_QUAN = 255.f;

        Lane8F32 bounds(neg_inf);
        for (u32 i = 0; i < numRecords; i++)
        {
            bounds = Max(bounds, Lane8F32::Load(&records[i].geomBounds));
        }
        result->meta        = u8((1u << numRecords) - 1);
        Lane4F32 boundsMinP = -Extract4<0>(bounds);
        Lane4F32 boundsMaxP = Extract4<1>(bounds);
        result->minP        = ToVec3f(boundsMinP);

        Lane4F32 diff = boundsMaxP - boundsMinP;
        Vec3lf<N> scaleVec;
#ifdef EXPONENTIAL_QUANTIZE
        Lane4U32 scale = ExponentialQuantize<N>(diff, scaleVec);
        Assert(scale[0] <= 255 && scale[1] <= 255 && scale[2] <= 255);
        result->scale[0] = (u8)scale[0];
        result->scale[1] = (u8)scale[1];
        result->scale[2] = (u8)scale[2];
#else
        Vec3f scale   = ScalarQuantize<N>(diff, scaleVec);
        result->scale = scale;
#endif

        Assert(numRecords <= N);
        Vec3lf<N> min;
        Vec3lf<N> max;

        if constexpr (N == 4)
        {
            Lane8F32 geomBounds[4] = {
                Lane8F32::Load(&records[0].geomBounds),
                Lane8F32::Load(&records[1].geomBounds),
                Lane8F32::Load(&records[2].geomBounds),
                Lane8F32::Load(&records[3].geomBounds),
            };
            Lane4F32 mins[4] = {
                Extract4<0>(geomBounds[0]),
                Extract4<0>(geomBounds[1]),
                Extract4<0>(geomBounds[2]),
                Extract4<0>(geomBounds[3]),
            };
            Lane4F32 maxs[4] = {
                Extract4<1>(geomBounds[0]),
                Extract4<1>(geomBounds[1]),
                Extract4<1>(geomBounds[2]),
                Extract4<1>(geomBounds[3]),
            };
            Transpose4x3(mins[0], mins[1], mins[2], mins[3], min.x, min.y, min.z);
            Transpose4x3(maxs[0], maxs[1], maxs[2], maxs[3], max.x, max.y, max.z);
        }
        else if constexpr (N == 8)
        {
            Lane8F32 geomBounds[8] = {
                Lane8F32::Load(&records[0].geomBounds), Lane8F32::Load(&records[1].geomBounds),
                Lane8F32::Load(&records[2].geomBounds), Lane8F32::Load(&records[3].geomBounds),
                Lane8F32::Load(&records[4].geomBounds), Lane8F32::Load(&records[5].geomBounds),
                Lane8F32::Load(&records[6].geomBounds), Lane8F32::Load(&records[7].geomBounds),
            };
            Transpose8x6(geomBounds[0], geomBounds[1], geomBounds[2], geomBounds[3],
                         geomBounds[4], geomBounds[5], geomBounds[6], geomBounds[7], min.x,
                         min.y, min.z, max.x, max.y, max.z);
        }
        min.x = FlipSign(min.x);
        min.y = FlipSign(min.y);
        min.z = FlipSign(min.z);

        Vec3lf<N> nodeMin;
        nodeMin.x = Shuffle<0>(LaneF32<N>(boundsMinP));
        nodeMin.y = Shuffle<1>(LaneF32<N>(boundsMinP));
        nodeMin.z = Shuffle<2>(LaneF32<N>(boundsMinP));

        Vec3lf<N> qNodeMin = Floor((min - nodeMin) / scaleVec);
        Vec3lf<N> qNodeMax = Ceil((max - nodeMin) / scaleVec);

        LaneF32<N> maskMinX = FMA(scaleVec.x, qNodeMin.x, nodeMin.x) > min.x;
        TruncateToU8(result->lowerX,
                     Max(Select(maskMinX, qNodeMin.x - 1, qNodeMin.x), MIN_QUAN));
        LaneF32<N> maskMinY = FMA(scaleVec.y, qNodeMin.y, nodeMin.y) > min.y;
        TruncateToU8(result->lowerY,
                     Max(Select(maskMinY, qNodeMin.y - 1, qNodeMin.y), MIN_QUAN));
        LaneF32<N> maskMinZ = FMA(scaleVec.z, qNodeMin.z, nodeMin.z) > min.z;
        TruncateToU8(result->lowerZ,
                     Max(Select(maskMinZ, qNodeMin.z - 1, qNodeMin.z), MIN_QUAN));

        LaneF32<N> maskMaxX = FMA(scaleVec.x, qNodeMax.x, nodeMin.x) < max.x;
        TruncateToU8(result->upperX,
                     Min(Select(maskMaxX, qNodeMax.x + 1, qNodeMax.x), MAX_QUAN));
        LaneF32<N> maskMaxY = FMA(scaleVec.y, qNodeMax.y, nodeMin.y) < max.y;
        TruncateToU8(result->upperY,
                     Min(Select(maskMaxY, qNodeMax.y + 1, qNodeMax.y), MAX_QUAN));
        LaneF32<N> maskMaxZ = FMA(scaleVec.z, qNodeMax.z, nodeMin.z) < max.z;
        TruncateToU8(result->upperZ,
                     Min(Select(maskMaxZ, qNodeMax.z + 1, qNodeMax.z), MAX_QUAN));
    }
};

template <i32 N>
struct UpdateQuantizedNode
{
    using NodeType = QuantizedNode<N>;
    __forceinline void operator()(NodeType *parent, BVHNode<N> *children, u32 numChildren)
    {
        for (u32 i = 0; i < numChildren; i++)
        {
            parent->children[i] = children[i];
        }
        for (u32 j = numChildren; j < N; j++)
        {
            parent->children[j] = BVHNode<N>::EncodeEmpty();
        }
    }
};

// NOTE: ptr MUST be aligned to 16 bytes, bottom 4 bits store the type, top 7 bits store the
// count

// template <i32 N>
// struct AABBNode
// {
//     struct Create
//     {
//         using NodeType = AABBNode;
//         CREATE_NODE()
//         {
//         }
//     };
//
//     LaneF32<N> lowerX;
//     LaneF32<N> lowerY;
//     LaneF32<N> lowerZ;
//
//     LaneF32<N> upperX;
//     LaneF32<N> upperY;
//     LaneF32<N> upperZ;
// };

template <i32 N, typename NodeType>
struct BVHN
{
    BVHN() {}
    NodeType *root;
};

template <i32 N>
using BVHQuantized = BVHN<N, QuantizedNode<N>>;
typedef BVHQuantized<4> BVH4Quantized;
typedef BVHQuantized<8> BVH8Quantized;

struct DefaultCompressedLeaf
{
};

template <i32 N, typename Heur, typename NT, typename CreateNode, typename UpdateNode,
          typename LT, typename CNT = DefaultCompressedLeaf>
struct BuildFuncs
{
    using NodeType           = NT;
    using CompressedNodeType = CNT;
    using Heuristic          = Heur;
    using LeafType           = LT;

    CreateNode createNode;
    UpdateNode updateNode;
    Heuristic heuristic;
};

template <i32 N>
using TLAS_PRB_QuantizedNode_Funcs =
    BuildFuncs<N, HeuristicPartialRebraid<GetQuantizedNode>, QuantizedNode<N>,
               CreateQuantizedNode<N>, UpdateQuantizedNode<N>, TLASLeaf,
               CompressedLeafNode<N>>;

template <i32 N, typename BuildFunctions>
struct BVHBuilder
{
    using NodeType           = typename BuildFunctions::NodeType;
    using CompressedNodeType = typename BuildFunctions::CompressedNodeType;
    using Heuristic          = typename BuildFunctions::Heuristic;
    using Record             = typename Heuristic::Record;
    using PrimRef            = typename Heuristic::PrimRef;
    using LeafType           = typename BuildFunctions::LeafType;
    static constexpr bool HasCompressedNode =
        !std::is_same_v<CompressedNodeType, DefaultCompressedLeaf>;

    BuildFunctions f;

    Arena **arenas;
    PrimRef *primRefs;
    const ScenePrimitives *scene;
    Heuristic heuristic;

    BVHBuilder() {}

    BVHNode<N> BuildBVH(BuildSettings settings, Arena **inArenas, Record &record);

    BVHNode<N> BuildBVHRoot(BuildSettings settings, Record &record);
    BVHNode<N> BuildBVH(const BuildSettings &settings, Record &record, bool parallel);
};

template <i32 N>
using PartialRebraidBuilder = BVHBuilder<N, TLAS_PRB_QuantizedNode_Funcs<N>>;

template <i32 N, typename BuildFunctions>
BVHNode<N> BVHBuilder<N, BuildFunctions>::BuildBVHRoot(BuildSettings settings, Record &record)
{
    settings.blockAdd = (1 << settings.logBlockSize) - 1;
    bool parallel     = record.count >= 8 * 1024;
    BVHNode<N> root   = BuildBVH(settings, record, parallel);
    if (root.data != 0) return root;

    Arena *currentArena = arenas[GetThreadIndex()];
    u32 offset          = 0;
    u32 count           = (record.count + settings.blockAdd) >> settings.logBlockSize;
    u8 *bytes           = PushArrayNoZeroTagged(currentArena, u8,
                                                sizeof(CompressedNodeType) + sizeof(LeafType) * count,
                                                MemoryType_BVH);
    Assert(currentArena->current->align == 16);
    CompressedNodeType *node = (CompressedNodeType *)bytes;
    LeafType *primIDs        = (LeafType *)(bytes + sizeof(CompressedNodeType));

    f.createNode(&record, 1, node);

    u32 begin = record.start;
    u32 end   = record.start + record.count;
    while (begin < end)
    {
        Assert(offset < count);
        primIDs[offset++].Fill(scene, primRefs, begin, end);
    }
    Assert(begin == end);
    node->offsets[0] = SafeTruncateU32ToU8(offset);

    return BVHNode<N>::EncodeCompressedNode(node);
}

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

    if (((record.count + blockAdd) >> blockShift) <= settings.maxLeafSize &&
        leafSAH <= splitSAH)
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
            bool childParallel = childRecords[jobID].count >= 8 * 1024;
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
    if constexpr (HasCompressedNode)
    {
        u32 leafCount = 0;
        u32 primTotal = 0;
        for (u32 i = 0; i < numChildren; i++)
        {
            if (childNodes[i].data == 0)
            {
                leafCount++;
                primTotal += (childRecords[i].count + blockAdd) >> blockShift;
            }
        }

        // Create a compressed leaf
        if (leafCount == numChildren)
        {
            threadLocalStatistics[GetThreadIndex()].misc += 1;
            u32 offset = 0;
            u8 *bytes  = PushArrayNoZeroTagged(
                currentArena, u8, sizeof(CompressedNodeType) + sizeof(LeafType) * primTotal,
                MemoryType_BVH);
            Assert(currentArena->current->align == 16);
            CompressedNodeType *node = (CompressedNodeType *)bytes;
            LeafType *primIDs        = (LeafType *)(bytes + sizeof(CompressedNodeType));

            f.createNode(childRecords, numChildren, node);

            for (u32 recordIndex = 0; recordIndex < numChildren; recordIndex++)
            {
                const Record &childRecord = childRecords[recordIndex];
                u32 begin                 = childRecord.start;
                u32 end                   = childRecord.start + childRecord.count;
                while (begin < end)
                {
                    Assert(offset < primTotal);
                    primIDs[offset++].Fill(scene, primRefs, begin, end);
                }
                Assert(begin == end);
                Assert(recordIndex < N);
                node->offsets[recordIndex] = SafeTruncateU32ToU8(offset);
            }
            Assert(offset == primTotal);
            return BVHNode<N>::EncodeCompressedNode(node);
        }
    }

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

template <i32 N, typename BuildFunctions>
BVHNode<N> BVHBuilder<N, BuildFunctions>::BuildBVH(BuildSettings settings, Arena **inArenas,
                                                   Record &record)
{

    arenas          = inArenas;
    BVHNode<N> head = BuildBVHRoot(settings, record);
    return head;
}

template <i32 N, GeometryType type, typename PrimRefType>
struct BVHHelper;

template <i32 N>
struct BVHHelper<N, GeometryType::TriangleMesh, PrimRefCompressed>
{
    using Polygon8 = Triangle8;
    using PrimType = TriangleCompressed<N>;
};

template <i32 N>
struct BVHHelper<N, GeometryType::TriangleMesh, PrimRef>
{
    using Polygon8 = Triangle8;
    using PrimType = Triangle<N>;
};

template <i32 N>
struct BVHHelper<N, GeometryType::QuadMesh, PrimRefCompressed>
{
    using Polygon8 = Quad8;
    using PrimType = QuadCompressed<N>;
};

template <i32 N>
struct BVHHelper<N, GeometryType::QuadMesh, PrimRef>
{
    using Polygon8 = Quad8;
    using PrimType = Quad<N>;
};

template <i32 N>
struct BVHHelper<N, GeometryType::CatmullClark, PrimRef>
{
    using PrimType = CatmullClarkPatch;
};

template <i32 N, i32 K, GeometryType type, typename PrimRef>
BVHNode<N> BuildQuantizedBVH(BuildSettings settings, Arena **inArenas,
                             const ScenePrimitives *scene, PrimRef *ref,
                             RecordAOSSplits &record)
{
    using BVHHelper = BVHHelper<K, type, PrimRef>;
    using Prim      = typename BVHHelper::PrimType;
    using BuildType = BuildFuncs<N, HeuristicObjectBinning<PrimRef>, QuantizedNode<N>,
                                 CreateQuantizedNode<N>, UpdateQuantizedNode<N>, Prim,
                                 CompressedLeafNode<N>>;

    using Builder = BVHBuilder<N, BuildType>;
    Builder builder;
    using Heuristic       = typename Builder::Heuristic;
    settings.logBlockSize = Bsf(K);
    new (&builder.heuristic) Heuristic(ref, scene, settings.logBlockSize);
    builder.primRefs = ref;
    builder.scene    = scene;
    return builder.BuildBVH(settings, inArenas, record);
}

template <typename PrimRef>
__forceinline BVHNodeN BuildQuantizedCatmullClarkBVH(BuildSettings settings, Arena **inArenas,
                                                     const ScenePrimitives *scene,
                                                     PrimRef *refs, RecordAOSSplits &record)
{
    return BuildQuantizedBVH<8, 1, GeometryType::CatmullClark>(settings, inArenas, scene, refs,
                                                               record);
}

template <i32 N, i32 K, GeometryType type, typename PrimRef>
BVHNode<N> BuildQuantizedSBVH(BuildSettings settings, Arena **inArenas,
                              const ScenePrimitives *scene, PrimRef *ref,
                              RecordAOSSplits &record)
{
    using BVHHelper = BVHHelper<K, type, PrimRef>;
    using Polygon8  = typename BVHHelper::Polygon8;
    using Prim      = typename BVHHelper::PrimType;
    using BuildType = BuildFuncs<N, HeuristicSpatialSplits<PrimRef, Polygon8>,
                                 QuantizedNode<N>, CreateQuantizedNode<N>,
                                 UpdateQuantizedNode<N>, Prim, CompressedLeafNode<N>>;
    using Builder   = BVHBuilder<N, BuildType>;
    Builder builder;
    using Heuristic       = typename Builder::Heuristic;
    settings.logBlockSize = Bsf(K);
    new (&builder.heuristic)
        Heuristic(ref, scene, HalfArea(record.geomBounds), settings.logBlockSize);
    builder.primRefs = ref;
    builder.scene    = scene;
    return builder.BuildBVH(settings, inArenas, record);
}

template <typename PrimRef>
__forceinline BVHNodeN BuildQuantizedTriSBVH(BuildSettings settings, Arena **inArenas,
                                             const ScenePrimitives *scene, PrimRef *refs,
                                             RecordAOSSplits &record)
{
    return BuildQuantizedSBVH<DefaultN, 8, GeometryType::TriangleMesh>(settings, inArenas,
                                                                       scene, refs, record);
}

template <typename PrimRef>
__forceinline BVHNodeN BuildQuantizedQuadSBVH(BuildSettings settings, Arena **inArenas,
                                              const ScenePrimitives *scene, PrimRef *refs,
                                              RecordAOSSplits &record)
{
    return BuildQuantizedSBVH<DefaultN, 8, GeometryType::QuadMesh>(settings, inArenas, scene,
                                                                   refs, record);
}

template <GeometryType type, typename PrimRef>
__forceinline BVHNodeN BuildQuantizedSBVH(BuildSettings settings, Arena **inArenas,
                                          const ScenePrimitives *scene, PrimRef *refs,
                                          RecordAOSSplits &record)
{
    return BuildQuantizedSBVH<DefaultN, 8, type>(settings, inArenas, scene, refs, record);
}

template <i32 N>
BVHNode<N> BuildTLASQuantized(BuildSettings settings, Arena **inArenas, ScenePrimitives *scene,
                              BuildRef<N> *refs, RecordAOSSplits &record)
{
    PartialRebraidBuilder<N> builder;
    using Heuristic = typename decltype(builder)::Heuristic;
    new (&builder.heuristic) Heuristic(scene, refs, settings.logBlockSize);
    settings.logBlockSize = 0;
    builder.primRefs      = refs;
    builder.scene         = scene;
    return builder.BuildBVH(settings, inArenas, record);
}

} // namespace rt
#endif
