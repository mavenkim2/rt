#ifndef BVH_BUILD_H
#define BVH_BUILD_H
#include "bvh_types.h"
namespace rt
{

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

        const f32 divisor = 1 / 255.f;

        f32 expX = diff[0] == 0.f ? 0.f : Ceil(Log2f(diff[0] * divisor));
        f32 expY = diff[1] == 0.f ? 0.f : Ceil(Log2f(diff[1] * divisor));
        f32 expZ = diff[2] == 0.f ? 0.f : Ceil(Log2f(diff[2] * divisor));

        Lane4U32 shift = Flooru(Lane4F32(expX, expY, expZ, 0.f)) + 127;

        Lane4F32 pow = AsFloat(shift << 23);

        Vec3lf<N> powVec;
        powVec.x = Shuffle<0>(LaneF32<N>(pow));
        powVec.y = Shuffle<1>(LaneF32<N>(pow));
        powVec.z = Shuffle<2>(LaneF32<N>(pow));

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

        Vec3lf<N> qNodeMin = Floor((min - nodeMin) / powVec);
        Vec3lf<N> qNodeMax = Ceil((max - nodeMin) / powVec);

        LaneF32<N> maskMinX = FMA(powVec.x, qNodeMin.x, nodeMin.x) > min.x;
        TruncateToU8(result->lowerX,
                     Max(Select(maskMinX, qNodeMin.x - 1, qNodeMin.x), MIN_QUAN));
        LaneF32<N> maskMinY = FMA(powVec.y, qNodeMin.y, nodeMin.y) > min.y;
        TruncateToU8(result->lowerY,
                     Max(Select(maskMinY, qNodeMin.y - 1, qNodeMin.y), MIN_QUAN));
        LaneF32<N> maskMinZ = FMA(powVec.z, qNodeMin.z, nodeMin.z) > min.z;
        TruncateToU8(result->lowerZ,
                     Max(Select(maskMinZ, qNodeMin.z - 1, qNodeMin.z), MIN_QUAN));

        LaneF32<N> maskMaxX = FMA(powVec.x, qNodeMax.x, nodeMin.x) < max.x;
        TruncateToU8(result->upperX,
                     Min(Select(maskMaxX, qNodeMax.x + 1, qNodeMax.x), MAX_QUAN));
        LaneF32<N> maskMaxY = FMA(powVec.y, qNodeMax.y, nodeMin.y) < max.y;
        TruncateToU8(result->upperY,
                     Min(Select(maskMaxY, qNodeMax.y + 1, qNodeMax.y), MAX_QUAN));
        LaneF32<N> maskMaxZ = FMA(powVec.z, qNodeMax.z, nodeMin.z) < max.z;
        TruncateToU8(result->upperZ,
                     Min(Select(maskMaxZ, qNodeMax.z + 1, qNodeMax.z), MAX_QUAN));

        Assert(shift[0] <= 255 && shift[1] <= 255 && shift[2] <= 255);
        result->scale[0] = (u8)shift[0];
        result->scale[1] = (u8)shift[1];
        result->scale[2] = (u8)shift[2];
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

#if 0
template <i32 N, i32 K>
using BLAS_SBVH_QuantizedNode_TriangleLeaf_Funcs =
    BuildFuncs<N, HeuristicSpatialSplits<Scene2Tri, PrimRefCompressed, Triangle8>,
               QuantizedNode<N>, CreateQuantizedNode<N>, UpdateQuantizedNode<N>,
               TriangleCompressed<K>, CompressedLeafNode<N>>;

template <i32 N, i32 K>
using BLAS_SBVH_QuantizedNode_QuadLeaf_Funcs =
    BuildFuncs<N, HeuristicSpatialSplits<Scene2Quad, PrimRefCompressed, Quad8>,
               QuantizedNode<N>, CreateQuantizedNode<N>, UpdateQuantizedNode<N>,
               QuadCompressed<K>, CompressedLeafNode<N>>;

template <i32 N, i32 K>
using BLAS_SBVH_QuantizedNode_TriangleLeaf_Scene_Funcs =
    BuildFuncs<N, HeuristicSpatialSplits<Scene2Tri, PrimRef, Triangle8>, QuantizedNode<N>,
               CreateQuantizedNode<N>, UpdateQuantizedNode<N>, Quad<K>, CompressedLeafNode<N>>;

template <i32 N, i32 K>
using BLAS_SBVH_QuantizedNode_QuadLeaf_Scene_Funcs =
    BuildFuncs<N, HeuristicSpatialSplits<Scene2Quad, PrimRef, Quad8>, QuantizedNode<N>,
               CreateQuantizedNode<N>, UpdateQuantizedNode<N>, Quad<K>, CompressedLeafNode<N>>;
#endif

template <i32 N, typename Scene>
using TLAS_PRB_QuantizedNode_Funcs =
    BuildFuncs<N, HeuristicPartialRebraid<Scene, GetQuantizedNode>, QuantizedNode<N>,
               CreateQuantizedNode<N>, UpdateQuantizedNode<N>, TLASLeaf<N>,
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

    BuildFunctions f;

    Arena **arenas;
    PrimRef *primRefs;
    Heuristic heuristic;

    BVHBuilder() {}

    BVHNode<N> BuildBVH(BuildSettings settings, Arena **inArenas, Record &record);

    BVHNode<N> BuildBVHRoot(BuildSettings settings, Record &record);
    BVHNode<N> BuildBVH(BuildSettings settings, Record &record, bool parallel);
};

#if 0
template <i32 N, i32 K>
using SBVHBuilderTriangleMesh =
    BVHBuilder<N, BLAS_SBVH_QuantizedNode_TriangleLeaf_Funcs<N, K>>;

template <i32 N, i32 K>
using SBVHBuilderSceneTris =
    BVHBuilder<N, BLAS_SBVH_QuantizedNode_TriangleLeaf_Scene_Funcs<N, K>>;

template <i32 N, i32 K>
using SBVHBuilderQuadMesh = BVHBuilder<N, BLAS_SBVH_QuantizedNode_QuadLeaf_Funcs<N, K>>;

template <i32 N, i32 K>
using SBVHBuilderSceneQuads =
    BVHBuilder<N, BLAS_SBVH_QuantizedNode_QuadLeaf_Scene_Funcs<N, K>>;
#endif

template <i32 N, typename Scene>
using PartialRebraidBuilder = BVHBuilder<N, TLAS_PRB_QuantizedNode_Funcs<N, Scene>>;

template <i32 N, typename BuildFunctions>
BVHNode<N> BVHBuilder<N, BuildFunctions>::BuildBVHRoot(BuildSettings settings, Record &record)
{
    bool parallel   = record.count >= 8 * 1024;
    BVHNode<N> root = BuildBVH(settings, record, parallel);
    if (root.data != 0) return root;

    Arena *currentArena = arenas[GetThreadIndex()];
    u32 offset          = 0;
    u8 *bytes           = PushArrayNoZeroTagged(
        currentArena, u8, sizeof(CompressedNodeType) + sizeof(LeafType) * record.count,
        MemoryType_BVH);
    CompressedNodeType *node = (CompressedNodeType *)bytes;
    LeafType *primIDs        = (LeafType *)(bytes + sizeof(CompressedNodeType));

    f.createNode(&record, 1, node);

    for (u32 primIndex = record.start; primIndex < record.start + record.count; primIndex++)
    {
        PrimRef *prim = &primRefs[primIndex];
        primIDs[offset++].Fill(prim);
    }
    return BVHNode<N>::EncodeCompressedNode(node);
}

template <i32 N, typename BuildFunctions>
BVHNode<N> BVHBuilder<N, BuildFunctions>::BuildBVH(BuildSettings settings, Record &record,
                                                   bool parallel)
{

    u32 total = record.count;
    Assert(total > 0);

    Record childRecords[N];
    u32 numChildren = 0;
    if (total == 1)
    {
        return 0;
    }

    Split split = heuristic.Bin(record);

    const u32 blockAdd   = (1 << settings.logBlockSize) - 1;
    const u32 blockShift = settings.logBlockSize;
    // NOTE: multiply both by the area instead of dividing
    f32 area     = HalfArea(record.geomBounds);
    f32 leafSAH  = settings.intCost * area * ((total + blockAdd) >> blockShift);
    f32 splitSAH = settings.travCost * area + settings.intCost * split.bestSAH;

    if (((total + blockAdd) >> blockShift) <= settings.maxLeafSize && leafSAH <= splitSAH)
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

    threadLocalStatistics[GetThreadIndex()].misc += 1;
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

    Arena *currentArena = arenas[GetThreadIndex()];
    // Create a compressed leaf
    if (leafCount == numChildren)
    {
        u32 offset = 0;
        u8 *bytes  = PushArrayNoZeroTagged(
            currentArena, u8, sizeof(CompressedNodeType) + sizeof(LeafType) * primTotal,
            MemoryType_BVH);
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
                primIDs[offset++].Fill(primRefs, begin, end);
            }
            Assert(begin == end);
            Assert(recordIndex < N);
            node->offsets[recordIndex] = SafeTruncateU32ToU8(offset);
        }
        Assert(offset == primTotal);
        return BVHNode<N>::EncodeCompressedNode(node);
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
            u32 begin = childRecord.start;
            u32 end   = childRecord.start + childRecord.count;
            while (begin < end)
            {
                Assert(offset < numPrims);
                primIDs[offset++].Fill(primRefs, begin, end);
            }
            Assert(begin == end);
            Assert(offset == numPrims);
            childNodes[i] = BVHNode<N>::EncodeLeaf(primIDs, numPrims);
        }
    }
    NodeType *node = PushStructNoZeroTagged(currentArena, NodeType, MemoryType_BVH);
    f.createNode(childRecords, numChildren, node);
    f.updateNode(node, childNodes, numChildren);

    return BVHNode<N>::EncodeNode(node);
}

template <i32 N, typename BuildFunctions>
BVHNode<N> BVHBuilder<N, BuildFunctions>::BuildBVH(BuildSettings settings, Arena **inArenas,
                                                   Record &record)
{

    arenas          = inArenas;
    BVHNode<N> head = BuildBVHRoot(settings, record);
    return head;
}

template <i32 N, typename Scene, typename PrimRefType>
struct BVHHelper;

template <i32 N>
struct BVHHelper<N, Scene2Tri, PrimRefCompressed>
{
    using Polygon8 = Triangle8;
    using PrimType = TriangleCompressed<N>;
};

template <i32 N>
struct BVHHelper<N, Scene2Tri, PrimRef>
{
    using Polygon8 = Triangle8;
    using PrimType = Triangle<N>;
};

template <i32 N>
struct BVHHelper<N, Scene2Quad, PrimRefCompressed>
{
    using Polygon8 = Quad8;
    using PrimType = QuadCompressed<N>;
};

template <i32 N>
struct BVHHelper<N, Scene2Quad, PrimRef>
{
    using Polygon8 = Quad8;
    using PrimType = Quad<N>;
};

template <i32 N, i32 K, typename SceneType, typename PrimRef>
BVHNode<N> BuildQuantizedSBVH(BuildSettings settings, Arena **inArenas, SceneType *scene,
                              PrimRef *ref, RecordAOSSplits &record)
{

    // template <i32 N, i32 K>
    // using BLAS_SBVH_QuantizedNode_TriangleLeaf_Funcs =
    using BVHHelper = BVHHelper<K, SceneType, PrimRef>;
    using Polygon8  = typename BVHHelper::Polygon8;
    using Prim      = typename BVHHelper::PrimType;
    using BuildType = BuildFuncs<N, HeuristicSpatialSplits<SceneType, PrimRef, Polygon8>,
                                 QuantizedNode<N>, CreateQuantizedNode<N>,
                                 UpdateQuantizedNode<N>, Prim, CompressedLeafNode<N>>;
    using Builder   = BVHBuilder<N, BuildType>;
    Builder builder;
    using Heuristic       = typename Builder::Heuristic;
    settings.logBlockSize = Bsf(K);
    new (&builder.heuristic)
        Heuristic(ref, scene, HalfArea(record.geomBounds), settings.logBlockSize);
    builder.primRefs = ref;
    return builder.BuildBVH(settings, inArenas, record);
}

template <typename SceneType, typename PrimRef>
__forceinline BVHNodeN BuildQuantizedSBVH(BuildSettings settings, Arena **inArenas,
                                          SceneType *scene, PrimRef *refs,
                                          RecordAOSSplits &record)
{
#if defined(USE_BVH4)
    return BuildQuantizedSBVH<4, 8>(settings, inArenas, scene, refs, record);
#elif defined(USE_BVH8)
    return BuildQuantizedSBVH<8, 8>(settings, inArenas, mesh, refs, record);
#endif
}

#if 0
template <i32 N>
BVHNode<N> BuildQuantizedTriSBVH(BuildSettings settings, Arena **inArenas, Scene2Tri *scene,
                                 PrimRefCompressed *ref, RecordAOSSplits &record)
{
    SBVHBuilderTriangleMesh<N, 8> builder;
    using Heuristic       = typename decltype(builder)::Heuristic;
    settings.logBlockSize = 3;
    new (&builder.heuristic)
        Heuristic(ref, scene, HalfArea(record.geomBounds), settings.logBlockSize);
    builder.primRefs = ref;
    return builder.BuildBVH(settings, inArenas, record);
}

template <i32 N>
BVHNode<N> BuildQuantizedTriSBVH(BuildSettings settings, Arena **inArenas, Scene2Tri *scene,
                                 PrimRef *ref, RecordAOSSplits &record)
{
    SBVHBuilderSceneTris<N, 8> builder;
    using Heuristic       = typename decltype(builder)::Heuristic;
    settings.logBlockSize = 3;
    new (&builder.heuristic)
        Heuristic(ref, scene, HalfArea(record.geomBounds), settings.logBlockSize);
    builder.primRefs = ref;
    return builder.BuildBVH(settings, inArenas, record);
}

template <i32 N>
BVHNode<N> BuildQuantizedQuadSBVH(BuildSettings settings, Arena **inArenas, Scene2Quad *scene,
                                  PrimRefCompressed *ref, RecordAOSSplits &record)
{
    SBVHBuilderQuadMesh<N, 8> builder;
    using Heuristic       = typename decltype(builder)::Heuristic;
    settings.logBlockSize = 3;
    new (&builder.heuristic)
        Heuristic(ref, scene, HalfArea(record.geomBounds), settings.logBlockSize);
    builder.primRefs = ref;
    return builder.BuildBVH(settings, inArenas, record);
}

template <i32 N>
BVHNode<N> BuildQuantizedQuadSBVH(BuildSettings settings, Arena **inArenas, Scene2Quad *scene,
                                  PrimRef *ref, RecordAOSSplits &record)
{
    SBVHBuilderSceneQuads<N, 8> builder;
    using Heuristic       = typename decltype(builder)::Heuristic;
    settings.logBlockSize = 3;
    new (&builder.heuristic) HeuristicSpatialSplits<Scene2>(
        ref, scene, HalfArea(record.geomBounds), settings.logBlockSize);
    builder.primRefs = ref;
    return builder.BuildBVH(settings, inArenas, record);
}
#endif

template <i32 N, typename Scene>
BVHNode<N> BuildTLASQuantized(BuildSettings settings, Arena **inArenas, Scene *scene,
                              BuildRef<N> *refs, RecordAOSSplits &record)
{
    PartialRebraidBuilder<N, Scene> builder;
    using Heuristic = typename decltype(builder)::Heuristic;
    new (&builder.heuristic) Heuristic(scene, refs, settings.logBlockSize);
    builder.primRefs = refs;
    return builder.BuildBVH(settings, inArenas, record);
}

template <typename Scene>
__forceinline BVHNodeN BuildTLASQuantized(BuildSettings settings, Arena **inArenas,
                                          Scene *scene, BRef *refs, RecordAOSSplits &record)
{
#if defined(USE_BVH4)
    return BuildTLASQuantized<4>(settings, inArenas, scene, refs, record);
#elif defined(USE_BVH8)
    return BuildTLASQuantized<Scene, 8>(settings, inArenas, scene, refs, record);
#endif
}

//////////////////////////////
// Helpers
//
#if 0
__forceinline BVHNodeN BuildQuantizedTriSBVH(BuildSettings settings, Arena **inArenas,
                                             TriangleMesh *mesh, PrimRefCompressed *refs,
                                             RecordAOSSplits &record)
{
#if defined(USE_BVH4)
    return BuildQuantizedTriSBVH<4>(settings, inArenas, mesh, refs, record);
#elif defined(USE_BVH8)
    return BuildQuantizedTriSBVH<8>(settings, inArenas, mesh, refs, record);
#endif
}

__forceinline BVHNodeN BuildQuantizedTriSBVH(BuildSettings settings, Arena **inArenas,
                                             Scene2Tri *scene, PrimRefCompressed *refs,
                                             RecordAOSSplits &record)
{
#if defined(USE_BVH4)
    return BuildQuantizedTriSBVH<4>(settings, inArenas, scene, refs, record);
#elif defined(USE_BVH8)
    return BuildQuantizedTriSBVH<8>(settings, inArenas, mesh, refs, record);
#endif
}

__forceinline BVHNodeN BuildQuantizedQuadSBVH(BuildSettings settings, Arena **inArenas,
                                              QuadMesh *mesh, PrimRefCompressed *refs,
                                              RecordAOSSplits &record)
{
#if defined(USE_BVH4)
    return BuildQuantizedQuadSBVH<4>(settings, inArenas, mesh, refs, record);
#elif defined(USE_BVH8)
    return BuildQuantizedQuadSBVH<8>(settings, inArenas, mesh, refs, record);
#endif
}

__forceinline BVHNodeN BuildQuantizedQuadSBVH(BuildSettings settings, Arena **inArenas,
                                              Scene2 *scene, PrimRef *refs,
                                              RecordAOSSplits &record)
{
#if defined(USE_BVH4)
    return BuildQuantizedQuadSBVH<4>(settings, inArenas, scene, refs, record);
#elif defined(USE_BVH8)
    return BuildQuantizedQuadSBVH<8>(settings, inArenas, scene, refs, record);
#endif
}

#endif

} // namespace rt
#endif
