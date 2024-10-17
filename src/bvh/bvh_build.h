#ifndef BVH_BUILD_H
#define BVH_BUILD_H
namespace rt
{
template <i32 N>
struct QuantizedNode;

template <i32 N>
struct CompressedLeafNode
{
    StaticAssert(N == 4 || N == 8, NMustBe4Or8);

    u8 scale[3];
    u8 meta;

    u8 lowerX[N];
    u8 lowerY[N];
    u8 lowerZ[N];

    u8 upperX[N];
    u8 upperY[N];
    u8 upperZ[N];

    Vec3f minP;
};

template <i32 N>
struct QuantizedNode
{
    StaticAssert(N == 4 || N == 8, NMustBe4Or8);

    uintptr_t children[N];
    u8 scale[3];
    u8 meta;

    u8 lowerX[N];
    u8 lowerY[N];
    u8 lowerZ[N];

    // NOTE: upperX = 255 when node is invalid
    u8 upperX[N];
    u8 upperY[N];
    u8 upperZ[N];

    Vec3f minP;

    // u32 GetNumChildren() const
    // {
    //     return ((internalOffset & 0xff) == 0) + ((leafOffset & 0xff) == 0) + ((meta & 0xff) == 0) + (((meta >> 4) & 0xff) == 0);
    // }

    void GetBounds(f32 *outMinX, f32 *outMinY, f32 *outMinZ, f32 *outMaxX, f32 *outMaxY, f32 *outMaxZ) const
    {
        LaneU32<N> lX = LaneU32<N>(*(u32 *)lowerX);
        LaneU32<N> lY = LaneU32<N>(*(u32 *)lowerY);
        LaneU32<N> lZ = LaneU32<N>(*(u32 *)lowerZ);

        LaneU32<N> uX = LaneU32<N>(*(u32 *)upperX);
        LaneU32<N> uY = LaneU32<N>(*(u32 *)upperY);
        LaneU32<N> uZ = LaneU32<N>(*(u32 *)upperZ);

        LaneU32<N> lExpandedMinX;
        LaneU32<N> lExpandedMinY;
        LaneU32<N> lExpandedMinZ;

        LaneU32<N> lExpandedMaxX;
        LaneU32<N> lExpandedMaxY;
        LaneU32<N> lExpandedMaxZ;
        if constexpr (N == 4)
        {
            lExpandedMinX = _mm_cvtepu16_epi32(_mm_cvtepu8_epi16(lX));
            lExpandedMinY = _mm_cvtepu16_epi32(_mm_cvtepu8_epi16(lY));
            lExpandedMinZ = _mm_cvtepu16_epi32(_mm_cvtepu8_epi16(lZ));

            lExpandedMaxX = _mm_cvtepu16_epi32(_mm_cvtepu8_epi16(uX));
            lExpandedMaxY = _mm_cvtepu16_epi32(_mm_cvtepu8_epi16(uY));
            lExpandedMaxZ = _mm_cvtepu16_epi32(_mm_cvtepu8_epi16(uZ));
        }
        else
        {
            lExpandedMinX = _mm256_cvtepu8_epi32(lX);
            lExpandedMinY = _mm256_cvtepu8_epi32(lY);
            lExpandedMinZ = _mm256_cvtepu8_epi32(lZ);
            lExpandedMaxX = _mm256_cvtepu8_epi32(uX);
            lExpandedMaxY = _mm256_cvtepu8_epi32(uY);
            lExpandedMaxZ = _mm256_cvtepu8_epi32(uZ);
        }
        LaneF32<N> minX(minP.x);
        LaneF32<N> minY(minP.y);
        LaneF32<N> minZ(minP.z);

        LaneF32<N> scaleX = AsFloat(LaneU32<N>(scale[0] << 23));
        LaneF32<N> scaleY = AsFloat(LaneU32<N>(scale[1] << 23));
        LaneF32<N> scaleZ = AsFloat(LaneU32<N>(scale[2] << 23));

        LaneF32<N>::Store(outMinX, minX + LaneF32<N>(lExpandedMinX) * scaleX);
        LaneF32<N>::Store(outMinY, minY + LaneF32<N>(lExpandedMinY) * scaleY);
        LaneF32<N>::Store(outMinZ, minZ + LaneF32<N>(lExpandedMinZ) * scaleZ);
        LaneF32<N>::Store(outMaxX, minX + LaneF32<N>(lExpandedMaxX) * scaleX);
        LaneF32<N>::Store(outMaxY, minY + LaneF32<N>(lExpandedMaxY) * scaleY);
        LaneF32<N>::Store(outMaxZ, minZ + LaneF32<N>(lExpandedMaxZ) * scaleZ);
    }
    QuantizedNode<N> GetBaseChildPtr() const
    {
        return (QuantizedNode<N> *)(internalOffset & ~(0xf));
    }
};

template <i32 N>
struct CreateQuantizedNode
{
    template <typename Record, typename NodeType>
    __forceinline void operator()(const Record *records, const u32 numRecords, NodeType *result)
    {
        const f32 MIN_QUAN = 0.f;
        const f32 MAX_QUAN = 255.f;

        Lane8F32 bounds(neg_inf);
        for (u32 i = 0; i < numRecords; i++)
        {
            bounds = Max(bounds, Lane8F32::Load(&records[i].geomBounds));
        }
        Lane4F32 boundsMinP = -Extract4<0>(bounds);
        Lane4F32 boundsMaxP = Extract4<1>(bounds);
        result->minP        = ToVec3f(boundsMinP);

        Lane4F32 diff = boundsMaxP - boundsMinP;

        f32 expX = diff[0] == 0.f ? 0.f : Ceil(Log2f(diff[0] / 255.f));
        f32 expY = diff[1] == 0.f ? 0.f : Ceil(Log2f(diff[1] / 255.f));
        f32 expZ = diff[2] == 0.f ? 0.f : Ceil(Log2f(diff[2] / 255.f));

        Lane4U32 shift = Flooru(Lane4F32(expX, expY, expZ, 0.f)) + 127;

        Lane4F32 pow = AsFloat(shift << 23);

        Vec3lf<N> powVec;
        // TODO: for N = 8, this needs to be shuffle across
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
                Lane8F32::Load(&records[2].geomBounds),
                Lane8F32::Load(&records[1].geomBounds),
                Lane8F32::Load(&records[3].geomBounds),
            };
            Lane4F32 mins[4] = {
                FlipSign(Extract4<0>(geomBounds[0])),
                FlipSign(Extract4<0>(geomBounds[1])),
                FlipSign(Extract4<0>(geomBounds[2])),
                FlipSign(Extract4<0>(geomBounds[3])),
            };
            Lane4F32 maxs[4] = {
                Extract4<1>(geomBounds[0]),
                Extract4<1>(geomBounds[1]),
                Extract4<1>(geomBounds[2]),
                Extract4<1>(geomBounds[3]),
            };
            LaneF32<N> min02xy = UnpackLo(mins[0], mins[2]);
            LaneF32<N> min13xy = UnpackLo(mins[1], mins[3]);

            LaneF32<N> min02z_ = UnpackHi(mins[0], mins[2]);
            LaneF32<N> min13z_ = UnpackHi(mins[1], mins[3]);

            LaneF32<N> max02xy = UnpackLo(maxs[0], maxs[2]);
            LaneF32<N> max13xy = UnpackLo(maxs[1], maxs[3]);

            LaneF32<N> max02z_ = UnpackHi(maxs[0], maxs[2]);
            LaneF32<N> max13z_ = UnpackHi(maxs[1], maxs[3]);

            min.x = UnpackLo(min02xy, min13xy);
            min.y = UnpackHi(min02xy, min13xy);
            min.z = UnpackLo(min02z_, min13z_);

            max.x = UnpackLo(max02xy, max13xy);
            max.y = UnpackHi(max02xy, max13xy);
            max.z = UnpackLo(max02z_, max13z_);
        }
        else if constexpr (N == 8)
        {
            Lane8F32 geomBounds[8] = {
                Lane8F32::Load(&records[0].geomBounds),
                Lane8F32::Load(&records[2].geomBounds),
                Lane8F32::Load(&records[1].geomBounds),
                Lane8F32::Load(&records[3].geomBounds),
                Lane8F32::Load(&records[4].geomBounds),
                Lane8F32::Load(&records[5].geomBounds),
                Lane8F32::Load(&records[6].geomBounds),
                Lane8F32::Load(&records[7].geomBounds),
            };
            Transpose8x6(geomBounds[0], geomBounds[1], geomBounds[2], geomBounds[3],
                         geomBounds[4], geomBounds[5], geomBounds[6], geomBounds[7],
                         min.x, min.y, min.z, max.x, max.y, max.z);
            min.x = FlipSign(min.x);
            min.y = FlipSign(min.y);
            min.z = FlipSign(min.z);
        }

        Vec3lf<N> nodeMin;
        nodeMin.x = Shuffle<0>(LaneF32<N>(boundsMinP));
        nodeMin.y = Shuffle<1>(LaneF32<N>(boundsMinP));
        nodeMin.z = Shuffle<2>(LaneF32<N>(boundsMinP));

        Vec3lf<N> qNodeMin = Floor((min - nodeMin) / powVec);
        Vec3lf<N> qNodeMax = Ceil((max - nodeMin) / powVec);

        LaneF32<N> maskMinX = FMA(powVec.x, qNodeMin.x, nodeMin.x) > min.x;
        TruncateToU8(result->lowerX, Max(Select(maskMinX, qNodeMin.x - 1, qNodeMin.x), MIN_QUAN));
        LaneF32<N> maskMinY = FMA(powVec.y, qNodeMin.y, nodeMin.y) > min.y;
        TruncateToU8(result->lowerY, Max(Select(maskMinY, qNodeMin.y - 1, qNodeMin.y), MIN_QUAN));
        LaneF32<N> maskMinZ = FMA(powVec.z, qNodeMin.z, nodeMin.z) > min.z;
        TruncateToU8(result->lowerZ, Max(Select(maskMinZ, qNodeMin.z - 1, qNodeMin.z), MIN_QUAN));

        LaneF32<N> maskMaxX = FMA(powVec.x, qNodeMax.x, nodeMin.x) < max.x;
        TruncateToU8(result->upperX, Min(Select(maskMaxX, qNodeMax.x + 1, qNodeMax.x), MAX_QUAN));
        LaneF32<N> maskMaxY = FMA(powVec.y, qNodeMax.y, nodeMin.y) < max.y;
        TruncateToU8(result->upperY, Min(Select(maskMaxY, qNodeMax.y + 1, qNodeMax.y), MAX_QUAN));
        LaneF32<N> maskMaxZ = FMA(powVec.z, qNodeMax.z, nodeMin.z) < max.z;
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
            primTotal += record->count;
        }

        Assert(primTotal <= 4 * 15);
        u32 *primIndices = PushArrayNoZero(arena, u32, primTotal);
        Assert(((uintptr_t)children & 0xf) == 0);
        Assert(((uintptr_t)primIndices & 0xf) == 0);
        parent->internalOffset = (uintptr_t)children;
        parent->leafOffset     = (uintptr_t)primIndices;
        parent->meta           = 0;
        u32 offset             = 0;
        for (u32 i = 0; i < leafCount; i++)
        {
            u32 leafIndex        = leafIndices[i];
            const Record *record = &records[leafIndex];
            for (u32 j = record->start; j < record->start + record->count; j++)
            {
                primIndices[offset++] = j;
            }

            Assert(record->count >= 1 && record->count <= 15);
            Assert(leafIndex >= 0 && leafIndex <= 3);

            switch (leafIndex)
            {
                case 0:
                {
                    parent->internalOffset |= record->count;
                }
                break;
                case 1:
                {
                    parent->leafOffset |= record->count;
                }
                break;
                default:
                {
                    parent->meta |= record->count << ((leafIndex - 2) << 2);
                }
            }
        }
    }
    __forceinline void operator()(NodeType *parent, uintptr_t *children, u32 numChildren)
    {
        for (u32 i = 0; i < numChildren; i++)
        {
            parent->children[i] = children[i];
        }
    }
};

// template <>
// struct UpdateQuantizedNode<8>
// {
//     using NodeType = QuantizedNode<8>;
//     template <typename Record>
//     __forceinline void operator()(Arena *arena, NodeType *parent, const Record *records, NodeType *children,
//                                   const u32 *leafIndices, const u32 leafCount)
//     {
//         // NOTE: for leaves, top 3 bits represent binary count. bottom 5 bits represent offset from base offset.
//         // 0 denotes a node, 1 denotes invalid.
//         Assert(((u64)parent->internalOffset & 0xf) == 0);
//         Assert(((u64)parent->leafOffset & 0xf) == 0);
//
//         uintptr_t internalOffset = (uintptr_t)children;
//
//         u32 primTotal = 0;
//         for (u32 i = 0; i < leafCount; i++)
//         {
//             u32 leafIndex        = leafIndices[i];
//             const Record *record = &records[leafIndex];
//             primTotal += record->range.count;
//         }
//         Assert(primTotal <= 24);
//         u32 *primIndices     = PushArrayNoZero(arena, u32, primTotal);
//         uintptr_t leafOffset = (uintptr_t)primIndices;
//         parent->meta         = 0;
//         u32 offset           = 0;
//         for (u32 i = 0; i < leafCount; i++)
//         {
//             u32 leafIndex        = leafIndices[i];
//             const Record *record = &records[leafIndex];
//             u32 primCount        = record->range.count;
//
//             for (u32 j = record->range.start; j < record->range.End(); j++)
//             {
//                 primIndices[offset++] = j;
//             }
//             Assert(primCount >= 1 && primCount <= 3);
//
//             if (leafIndex >= 6)
//             {
//                 internalOffset |= (primCount << ((leafIndex - 6) * 2));
//             }
//             else if (leafIndex >= 4)
//             {
//                 leafOffset |= (primCount << ((leafIndex - 4) * 2));
//             }
//             else
//             {
//                 parent->meta |= primCount << (leafIndex * 2);
//             }
//         }
//     }
// };

// NOTE: ptr MUST be aligned to 16 bytes, bottom 4 bits store the type, top 7 bits store the count

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
using QuantizedNode4 = QuantizedNode<4>;
template <i32 N>
using QuantizedNode8 = QuantizedNode<8>;

template <i32 N>
using BVHQuantized = BVHN<N, QuantizedNode<N>>;
typedef BVHQuantized<4> BVH4Quantized;
typedef BVHQuantized<8> BVH8Quantized;

template <i32 N, typename Heur, typename NT, typename CNT, typename CreateNode, typename UpdateNode, typename Prim>
struct BuildFuncs
{
    using NodeType           = NT;
    using CompressedNodeType = CNT;
    using Primitive          = Prim;
    using Heuristic          = Heur;

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
using BLAS_AOS_SBVH_QuantizedNode_TriangleLeaf_Funcs =
    BuildFuncs<
        N,
        HeuristicSpatialSplits<32, 16>,
        QuantizedNode<N>,
        CompressedLeafNode<N>,
        CreateQuantizedNode<N>,
        UpdateQuantizedNode<N>,
        TriangleMesh>;

template <i32 N, typename BuildFunctions>
struct BVHBuilder
{
    using NodeType           = typename BuildFunctions::NodeType;
    using CompressedNodeType = typename BuildFunctions::CompressedNodeType;
    using Primitive          = typename BuildFunctions::Primitive;
    using Heuristic          = typename BuildFunctions::Heuristic;
    using Record             = typename Heuristic::Record;
    using PrimitiveData      = typename Record::PrimitiveData;

    struct NextGenInfo
    {
        NodeType *grandChildren;
        u32 leafIndices[N];
        u32 numChildren;
        u32 leafCount;
    };

    BuildFunctions f;

    Arena **arenas;
    Primitive *primitives;
    PrimitiveData *data;
    Heuristic heuristic;

    BVHBuilder() {}
#if 0
    NodeType *BuildBVHRoot(BuildSettings settings, Record &record);
    __forceinline u32 BuildNode(BuildSettings settings, const Record &record, Record *childRecords, u32 &numChildren);

    void BuildBVH(Scheduler::Counter *counter, BuildSettings settings, NodeType *parent, const Record *records, u32 numChildren,
            bool parallel = true);
    NodeType *BuildBVHRoot2(BuildSettings settings, Record &record);

    void BuildBVH2(BuildSettings settings, const Record &record, NextGenInfo &info,
            Record *childRecords, bool parallel);
#endif

    BVHN<N, NodeType> BuildBVH(BuildSettings settings, Arena **inArenas, Primitive *inRawPrims, Record &record);
    uintptr_t BuildBVHRoot3(BuildSettings settings, Record &record);
    uintptr_t BuildBVH3(BuildSettings settings, const Record &record, bool parallel);
};

template <i32 N>
using SBVHBuilderTriangleMesh = BVHBuilder<N, BLAS_AOS_SBVH_QuantizedNode_TriangleLeaf_Funcs<N>>;

template <i32 N, typename BuildFunctions>
uintptr_t BVHBuilder<N, BuildFunctions>::BuildBVHRoot3(BuildSettings settings, Record &record)
{
    uintptr_t root = BuildBVH3(settings, record, true);
    return root;
}

template <i32 N, typename BuildFunctions>
uintptr_t BVHBuilder<N, BuildFunctions>::BuildBVH3(BuildSettings settings, const Record &record, bool parallel)
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

    // NOTE: multiply both by the area instead of dividing
    f32 area     = HalfArea(record.geomBounds);
    f32 leafSAH  = settings.intCost * area * total; //((total + (1 << settings.logBlockSize) - 1) >> settings.logBlockSize);
    f32 splitSAH = settings.travCost * area + settings.intCost * split.bestSAH;

    if (total <= settings.maxLeafSize && leafSAH <= splitSAH)
    {
        heuristic.FlushState(split);
        return 0;
    }
    heuristic.Split(split, record, childRecords[0], childRecords[1]);
    Assert(childRecords[0].count <= record.count && childRecords[1].count <= record.count);

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

        Assert(childRecords[0].count <= record.count && childRecords[1].count <= record.count);
        childRecords[bestChild] = out;
    }

    uintptr_t childNodes[N];

    if (parallel)
    {
        scheduler.ScheduleAndWait(numChildren, 1, [&](u32 jobID) {
            bool childParallel = childRecords[jobID].count >= 8 * 1024;
            childNodes[jobID]  = BuildBVH3(settings, childRecords[jobID], childParallel);
        });
    }
    else
    {
        for (u32 i = 0; i < numChildren; i++)
        {
            childNodes[i] = BuildBVH3(settings, childRecords[i], false);
        }
    }

    // threadLocalStatistics[GetThreadIndex()].misc += 1;
    u32 leafCount = 0;
    u32 primTotal = 0;
    for (u32 i = 0; i < numChildren; i++)
    {
        if (childNodes[i] == 0)
        {
            leafCount++;
            primTotal += childRecords[i].count;
        }
    }

    Arena *currentArena = arenas[GetThreadIndex()];
    u32 offset          = 0;
    // Create a compressed leaf
    if (leafCount == numChildren)
    {
        u8 *bytes                = PushArrayNoZeroTagged(currentArena, u8,
                                                         sizeof(CompressedNodeType) + sizeof(u32) * primTotal, MemoryType_BVH);
        CompressedNodeType *node = (CompressedNodeType *)bytes;
        u32 *primIndices         = (u32 *)(bytes + sizeof(CompressedNodeType));

        // TODO: not currently keeping track of which prims correspond with which leaf, either need to
        // use 4 8 bit offsets or using a triangle8 struct
        f.createNode(childRecords, numChildren, node);

        for (u32 recordIndex = 0; recordIndex < numChildren; recordIndex++)
        {
            const Record &childRecord = childRecords[recordIndex];
            for (u32 primIndex = childRecord.start; primIndex < childRecord.start + childRecord.count; primIndex++)
            {
                primIndices[offset++] = primIndex;
            }
        }
        return (uintptr_t)node;
    }

    u32 *primIndices = PushArrayNoZeroTagged(currentArena, u32, primTotal, MemoryType_BVH);
    for (u32 i = 0; i < numChildren; i++)
    {
        if (childNodes[i] == 0)
        {
            const Record &childRecord = childRecords[i];
            u32 numPrims              = childRecord.count;
            u32 *prims                = &primIndices[offset];
            for (u32 primIndex = childRecord.start; primIndex < childRecord.start + childRecord.count; primIndex++)
            {
                primIndices[offset++] = primIndex;
            }
            childNodes[i] = (uintptr_t)(prims);
        }
    }
    NodeType *node = PushStructNoZeroTagged(currentArena, NodeType, MemoryType_BVH);
    f.createNode(childRecords, numChildren, node);
    f.updateNode(node, childNodes, numChildren);
    return (uintptr_t)node;
}

#if 0
template <i32 N, typename BuildFunctions>
typename BVHBuilder<N, BuildFunctions>::NodeType *
BVHBuilder<N, BuildFunctions>::BuildBVHRoot2(BuildSettings settings, Record &record)
{
    Record childRecords[N];

    Arena *arena   = arenas[GetThreadIndex()];
    NodeType *root = PushStruct(arena, NodeType);
    NextGenInfo info;

    BuildBVH2(settings, record, info, childRecords, true);

    if (info.grandChildren)
    {
        f.createNode(childRecords, info.numChildren, root);
        f.updateNode(arena, root, childRecords, info.grandChildren, info.leafIndices, info.leafCount);
    }
    else
    {
        u32 leafIndex = 0;
        f.createNode(&record, 1, root);
        f.updateNode(arena, root, &record, 0, &leafIndex, 1);
    }
    return root;
}

template <i32 N, typename BuildFunctions>
void BVHBuilder<N, BuildFunctions>::BuildBVH2(BuildSettings settings, const Record &record, NextGenInfo &info,
                                              Record *childRecords, bool parallel)
{

    u32 total = record.count;
    Assert(total > 0);

    // Record childRecords[N];

    u32 numChildren = 0;
    if (total == 1)
    {
        info.grandChildren = 0;
        info.numChildren   = 0;
        return;
    }

    Split split = heuristic.Bin(record);

    // NOTE: multiply both by the area instead of dividing
    f32 area     = HalfArea(record.geomBounds);
    f32 leafSAH  = settings.intCost * area * total; //((total + (1 << settings.logBlockSize) - 1) >> settings.logBlockSize);
    f32 splitSAH = settings.travCost * area + settings.intCost * split.bestSAH;

    if (total <= settings.maxLeafSize && leafSAH <= splitSAH)
    {
        heuristic.FlushState(split);
        info.grandChildren = 0;
        info.numChildren   = 0;
        return;
    }
    heuristic.Split(split, record, childRecords[0], childRecords[1]);
    Assert(childRecords[0].count <= record.count && childRecords[1].count <= record.count);

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

        Assert(childRecords[0].count <= record.count && childRecords[1].count <= record.count);
        childRecords[bestChild] = out;
    }

    Record nextGenRecords[N][N];
    NextGenInfo nextGenInfo[N];

    if (parallel)
    {
        scheduler.ScheduleAndWait(numChildren, 1, [&](u32 jobID) {
            bool childParallel = childRecords[jobID].count >= 8 * 1024;
            NextGenInfo &info  = nextGenInfo[jobID];
            BuildBVH2(settings, childRecords[jobID], nextGenInfo[jobID], nextGenRecords[jobID], childParallel);
        });
    }
    else
    {
        for (u32 i = 0; i < numChildren; i++)
        {
            BuildBVH2(settings, childRecords[i], nextGenInfo[i], nextGenRecords[i], false);
        }
    }

    // info.leafCount = 0;
    u32 leafCount = 0;
    u32 nodeIndices[N];
    u32 nodeCount = 0;
    for (u32 i = 0; i < numChildren; i++)
    {
        u32 isLeaf = nextGenInfo[i].grandChildren == 0;
        if (isLeaf)
        {
            info.leafIndices[leafCount++] = i;
        }
        else
        {
            nodeIndices[nodeCount] = i;
            nodeCount++;
        }
    }

    Arena *currentArena = arenas[GetThreadIndex()];
    NodeType *children  = (NodeType *)(0xfffffffffffffff0);

    if (nodeCount)
    {
        children = PushArrayNoZeroTagged(currentArena, NodeType, nodeCount, MemoryType_Node);
    }

    // u8 *bytes = PushArrayNoZeroTagged(currentArena, u8, sizeof(NodeType) + sizeof(u32) * primTotal + (leafCount ? 4 : 0));
    for (u32 i = 0; i < nodeCount; i++)
    {
        u32 nodeIndex = nodeIndices[i];
        f.createNode(nextGenRecords[nodeIndex], nextGenInfo[nodeIndex].numChildren, &children[i]);
        f.updateNode(currentArena, &children[i], nextGenRecords[nodeIndex],
                     nextGenInfo[nodeIndex].grandChildren, nextGenInfo[nodeIndex].leafIndices, nextGenInfo[nodeIndex].leafCount);
    }
    info.grandChildren = children;
    info.numChildren   = numChildren;
    info.leafCount     = leafCount;

    // threadLocalStatistics[GetThreadIndex()].misc += leafCount != 0; // nodeCount == 0; // leafCount; // nodeCount;
}

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
        Scheduler::Counter counter = {};
        scheduler.Schedule(&counter, [&](u32 jobID) {
            BuildBVH(&counter, settings, root, childRecords, numChildren);
        });
        scheduler.Wait(&counter);
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

        // NOTE: multiply both by the area instead of dividing
        f32 area     = HalfArea(record.geomBounds);
        f32 leafSAH  = settings.intCost * area * total; //((total + (1 << settings.logBlockSize) - 1) >> settings.logBlockSize);
        f32 splitSAH = settings.travCost * area + settings.intCost * split.bestSAH;

        if (total <= settings.maxLeafSize)
        {
            heuristic.FlushState(split);
            return 0; // && leafSAH <= splitSAH) return 0;
        }

        heuristic.Split(split, record, childRecords[0], childRecords[1]);
        Assert(childRecords[0].range.count <= record.range.count && childRecords[1].range.count <= record.range.count);
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

        Assert(childRecords[0].range.count <= record.range.count && childRecords[1].range.count <= record.range.count);
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
void BVHBuilder<N, BuildFunctions>::BuildBVH(Scheduler::Counter *counter, BuildSettings settings,
                                             NodeType *parent, const Record *records, u32 inNumChildren,
                                             bool parallel)
{
    Arena *currentArena = arenas[GetThreadIndex()];
    Assert(currentArena->align == 32);
    Record allChildRecords[N][N];

    u32 allNumChildren[N];

    u32 childNodeIndices[N];
    u32 childLeafIndices[N];
    std::atomic<u32> nodeCount = 0;
    std::atomic<u32> leafCount = 0;

    Assert(inNumChildren <= N);

    // if (parallel)
    // {
    //     scheduler.ScheduleAndWait(inNumChildren, 1, [&](u32 jobID) {
    //         u32 childIndex       = jobID;
    //         Record *childRecords = allChildRecords[childIndex];
    //         const Record &record = records[childIndex];
    //         u32 &numChildren     = allNumChildren[childIndex];
    //         u32 result           = BuildNode(settings, record, childRecords, numChildren);
    //
    //         if (result)
    //         {
    //             u32 nodeIndex               = nodeCount.fetch_add(1);
    //             childNodeIndices[nodeIndex] = childIndex;
    //         }
    //         else
    //         {
    //             u32 leafIndex               = leafCount.fetch_add(1);
    //             childLeafIndices[leafIndex] = childIndex;
    //         }
    //     });
    // }
    // else
    // {
    for (u32 childIndex = 0; childIndex < inNumChildren; childIndex++)
    {
        Record *childRecords = allChildRecords[childIndex];
        const Record &record = records[childIndex];
        u32 &numChildren     = allNumChildren[childIndex];
        u32 result           = BuildNode(settings, record, childRecords, numChildren);

        if (result)
        {
            childNodeIndices[nodeCount++] = childIndex;
        }
        else
        {
            childLeafIndices[leafCount++] = childIndex;
        }
    }
    // }

    u32 finalNodeCount = nodeCount.load();
    threadLocalStatistics[GetThreadIndex()].misc += finalNodeCount;

    NodeType *children = PushArrayTagged(currentArena, NodeType, finalNodeCount, MemoryType_Node);

    for (u32 i = 0; i < finalNodeCount; i++)
    {
        u32 childNodeIndex = childNodeIndices[i];
        f.createNode(allChildRecords[childNodeIndex], allNumChildren[childNodeIndex], &children[i]);
    }

    // Updates the parent
    f.updateNode(currentArena, parent, records, children, childLeafIndices, leafCount.load());

    // Splits the thread pool
    // Range orderedRanges[N];
    // for (u32 i = 0; i < inNumChildren; i++)
    // {
    //     orderedRanges[i] = {records[i].range.start, records[i].range.count};
    // }
    // // TODO: 8-wide
    // for (u32 i = 0; i < inNumChildren; i++)
    // {
    //     const Record &record = records[i];
    //     bool split           = scheduler.SplitThreadPool(parentCount, record.range.count);
    //     parentCount -= record.range.count;
    //     if (!split)
    //     {
    //         dynamicPoolScheduler.Schedule(
    //     }
    // }

    if (parallel)
    {
        scheduler.Schedule(counter, finalNodeCount, 1, [=](u32 jobID) {
            u32 childNodeIndex = childNodeIndices[jobID];
            bool childParallel = records[childNodeIndex].range.count >= PARALLEL_THRESHOLD;
            BuildBVH(counter, settings, &children[jobID], allChildRecords[childNodeIndex],
                     allNumChildren[childNodeIndex], childParallel);
        });
    }
    else
    {
        for (u32 i = 0; i < finalNodeCount; i++)
        {
            u32 childNodeIndex = childNodeIndices[i];
            BuildBVH(counter, settings, &children[i], allChildRecords[childNodeIndex], allNumChildren[childNodeIndex], false);
        }
    }
}
#endif

template <i32 N, typename BuildFunctions>
BVHN<N, typename BVHBuilder<N, BuildFunctions>::NodeType>
BVHBuilder<N, BuildFunctions>::BuildBVH(BuildSettings settings,
                                        Arena **inArenas,
                                        Primitive *inRawPrims,
                                        Record &record)
{

    arenas     = inArenas;
    primitives = inRawPrims;
    BVHN<N, NodeType> result;
    // result.root = BuildBVHRoot(settings, record);
    // result.root = BuildBVHRoot2(settings, record);

    // TODO: change bvh to take a  NodePtr struct
    BuildBVHRoot3(settings, record);
    return result;
}

template <i32 N>
BVHQuantized<N> BuildQuantizedSBVH(BuildSettings settings,
                                   Arena **inArenas,
                                   TriangleMesh *mesh,
                                   PrimRef *ref,
                                   u32 *refIndices0,
                                   u32 *refIndices1,
                                   RecordAOSSplits &record)
{
    SBVHBuilderTriangleMesh<N> builder;
    // new (&builder.heuristic) HeuristicSpatialSplits(ref, refIndices0, refIndices1, mesh, HalfArea(record.geomBounds),
    //                                                 record.End());
    new (&builder.heuristic) HeuristicSpatialSplits(ref, mesh, HalfArea(record.geomBounds));
    return builder.BuildBVH(settings, inArenas, mesh, record);
}

} // namespace rt
#endif
