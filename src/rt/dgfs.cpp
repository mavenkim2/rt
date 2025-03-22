#include "base.h"
#include "math/basemath.h"
#include "scene.h"
#include "bvh/bvh_aos.h"
#include "bvh/bvh_types.h"
#include "shader_interop/dense_geometry_shaderinterop.h"
#include "shader_interop/ray_shaderinterop.h"
#include "dgfs.h"
#include "memory.h"
#include "scene.h"
#include "thread_context.h"
#include "platform.h"

namespace rt
{

typedef HeuristicObjectBinning<PrimRef> Heuristic;

ClusterBuilder::ClusterBuilder(Arena *arena, ScenePrimitives *scene, PrimRef *primRefs)
    : primRefs(primRefs)
{
    u32 numProcessors = OS_NumProcessors();
    arenas            = GetArenaArray(arena);
    threadClusters =
        StaticArray<ClusterList>(arenas[GetThreadIndex()], numProcessors, numProcessors);
    for (u32 i = 0; i < numProcessors; i++)
    {
        threadClusters[i].l = ChunkedLinkedList<RecordAOSSplits>(arenas[i]);
    }

    h = PushStructConstruct(arena, Heuristic)(primRefs, scene, 0);
}

BitVector::BitVector(Arena *arena, u32 maxNumBits) : maxNumBits(maxNumBits)
{
    bits = PushArray(arena, u32, (maxNumBits + 31) >> 5);
}
void BitVector::SetBit(u32 bit)
{
    Assert(bit < maxNumBits);
    bits[bit >> 5] |= 1ull << (bit & 31);
}
bool BitVector::GetBit(u32 bit)
{
    Assert(bit < maxNumBits);
    return bits[bit >> 5] & (1 << (bit & 31));
}

void DenseGeometryBuildData::Init()
{
    arena         = ArenaAlloc();
    byteBuffer    = ChunkedLinkedList<u8>(arena, 1024);
    headers       = ChunkedLinkedList<PackedDenseGeometryHeader>(arena);
    triangleOrder = ChunkedLinkedList<u32>(arena);

    // Debug
    types        = ChunkedLinkedList<TriangleStripType>(arena);
    firstUse     = ChunkedLinkedList<u32>(arena);
    reuse        = ChunkedLinkedList<u32>(arena);
    debugIndices = ChunkedLinkedList<u32>(arena);
}

void DenseGeometryBuildData::Merge(DenseGeometryBuildData &other)
{
    u32 offset = byteBuffer.Length();

    for (auto *node = other.headers.first; node != 0; node = node->next)
    {
        for (int i = 0; i < node->count; i++)
        {
            node->values[i].a += offset;
        }
    }

    headers.Merge(&other.headers);
    byteBuffer.Merge(&other.byteBuffer);
}

void WriteBits(u32 *data, u32 &position, u32 value, u32 numBits)
{
    Assert(numBits <= 32);
    uint dwordIndex = position >> 5;
    uint bitIndex   = position & 31;

    Assert(numBits == 32 || ((value & ((1u << numBits) - 1)) == value));

    data[dwordIndex] |= value << bitIndex;
    if (bitIndex + numBits > 32)
    {
        data[dwordIndex + 1] |= value >> (32 - bitIndex);
    }

    position += numBits;
}

void BitVector::WriteBits(u32 &position, u32 value, u32 numBits)
{
    Assert(position + numBits <= maxNumBits);
    rt::WriteBits(bits, position, value, numBits);
}

void ClusterBuilder::BuildClusters(RecordAOSSplits &record, bool parallel)
{
    auto *heuristic = (Heuristic *)h;
    const int N     = 4;
    Assert(record.count > 0);

    RecordAOSSplits childRecords[N];
    u32 numChildren = 0;

    Split split = heuristic->Bin(record);

    if (record.count <= MAX_CLUSTER_TRIANGLES)
    {
        u32 threadIndex                         = GetThreadIndex();
        threadClusters[threadIndex].l.AddBack() = record;
        heuristic->FlushState(split);
        return;
    }
    heuristic->Split(split, record, childRecords[0], childRecords[1]);

    // N - 1 splits produces N children
    for (numChildren = 2; numChildren < N; numChildren++)
    {
        i32 bestChild = -1;
        f32 maxArea   = neg_inf;
        for (u32 recordIndex = 0; recordIndex < numChildren; recordIndex++)
        {
            RecordAOSSplits &childRecord = childRecords[recordIndex];
            if (childRecord.count <= MAX_CLUSTER_TRIANGLES) continue;

            f32 childArea = HalfArea(childRecord.geomBounds);
            if (childArea > maxArea)
            {
                bestChild = recordIndex;
                maxArea   = childArea;
            }
        }
        if (bestChild == -1) break;

        split = heuristic->Bin(childRecords[bestChild]);

        RecordAOSSplits out;
        heuristic->Split(split, childRecords[bestChild], out, childRecords[numChildren]);

        childRecords[bestChild] = out;
    }

    if (parallel)
    {
        scheduler.ScheduleAndWait(numChildren, 1, [&](u32 jobID) {
            bool childParallel = childRecords[jobID].count >= BUILD_PARALLEL_THRESHOLD;
            BuildClusters(childRecords[jobID], childParallel);
        });
    }
    else
    {
        for (u32 i = 0; i < numChildren; i++)
        {
            BuildClusters(childRecords[i], false);
        }
    }
}

void ClusterBuilder::CreateDGFs(DenseGeometryBuildData *buildData, Mesh *meshes, int numMeshes,
                                Bounds &sceneBounds)
{
#if 0
    static const int b     = 24;
    static const f32 scale = (1 << (b - 1)) - 1.f;
    Vec3f sceneExtent      = ToVec3f(sceneBounds.maxP - sceneBounds.minP);

    i32 eX = (i32)Ceil(Log2f(sceneExtent[0] / scale));
    i32 eY = (i32)Ceil(Log2f(sceneExtent[1] / scale));
    i32 eZ = (i32)Ceil(Log2f(sceneExtent[2] / scale));

    f32 expX = AsFloat((eX + 127) << 23);
    f32 expY = AsFloat((eY + 127) << 23);
    f32 expZ = AsFloat((eZ + 127) << 23);

    Vec3f quantize = 1.f / Vec3f(expX, expY, expZ);
#endif

    i32 precision = 6;
    Assert(precision > CLUSTER_MIN_PRECISION);
    Vec3f quantize(AsFloat((127 + precision) << 23));

#if 0
    f32 offsetMinX = maxClusterEdgeLength[0] - 16.f;
    f32 offsetMinY = maxClusterEdgeLength[1] - 16.f;
    f32 offsetMinZ = maxClusterEdgeLength[2] - 16.f;

    static const int c = (1 << 23) - 1;
    f32 constraintMinX = Max(-sceneBounds.minP[0], 0.f) / c;
    f32 constraintMinY = Max(-sceneBounds.minP[1], 0.f) / c;
    f32 constraintMinZ = Max(-sceneBounds.minP[2], 0.f) / c;
    f32 constraintMaxX = Max(sceneBounds.maxP[0], 0.f) / (c + 1);
    f32 constraintMaxY = Max(sceneBounds.maxP[1], 0.f) / (c + 1);
    f32 constraintMaxZ = Max(sceneBounds.maxP[2], 0.f) / (c + 1);

    eX = Max(constraintMinX, Max(constraintMaxX, Max(offsetMinX, eX)));
    eY = Max(constraintMinY, Max(constraintMaxY, Max(offsetMinY, eY)));
    eZ = Max(constraintMinZ, Max(constraintMaxY, Max(offsetMinZ, eZ)));
#endif

    ScratchArena meshScratch;
    Vec3i **quantizedVertices = PushArrayNoZero(meshScratch.temp.arena, Vec3i *, numMeshes);

    u32 indexTotal = 0;

    for (int i = 0; i < numMeshes; i++)
    {
        Mesh &mesh = meshes[i];
        quantizedVertices[i] =
            PushArrayNoZero(meshScratch.temp.arena, Vec3i, mesh.numVertices);
        indexTotal += mesh.numIndices;

        // Quantize to global grid
        for (int j = 0; j < mesh.numVertices; j++)
        {
            quantizedVertices[i][j] = Vec3i(Round(mesh.p[j] * quantize));
        }
    }

    u32 total = 0;
    for (int threadIndex = 0; threadIndex < threadClusters.Length(); threadIndex++)
    {
        ClusterList &list = threadClusters[threadIndex];
        for (auto *node = list.l.first; node != 0; node = node->next)
        {
            total += node->count;
            for (int clusterIndex = 0; clusterIndex < node->count; clusterIndex++)
            {
                RecordAOSSplits &cluster = node->values[clusterIndex];
                CreateDGFs(buildData, meshScratch.temp.arena, meshes, quantizedVertices,
                           cluster, precision);
            }
        }
    }
    buildData->numBlocks = total;
}

u32 BitFieldPackU32(u32 val, u32 data, u32 &offset, u32 size)
{
    u32 o = offset & 31u;
    data  = data & ((1u << size) - 1u);
    val |= data << o;
    offset += size;
    return val;
}

u32 BitFieldPackI32(u32 bits, i32 data, u32 &offset, u32 size)
{
    u32 signBit = (data & 0x80000000) >> 31;
    return BitFieldPackU32(bits, data | (signBit << size), offset, size);
}

u32 BitFieldPackU32(Vec4u &bits, u32 data, u32 offset, u32 size)
{
    u32 d = data;
    Assert(offset < 128);
    u32 o     = offset & 31u;
    u32 index = o >> 5u;
    bits[index] |= data << o;
    if (o + size > 32u)
    {
        Assert(index + 1 < 4);
        bits[index + 1] |= (data >> (32u - o));
    }
    return offset + size;
}

u32 BitFieldPackI32(Vec4u &bits, i32 data, u32 offset, u32 size)
{
    u32 signBit = (data & 0x80000000) >> 31;
    return BitFieldPackU32(bits, data | (signBit << size), offset, size);
}

u32 BitAlignU32(u32 high, u32 low, u32 shift)
{
    shift &= 31u;

    u32 result = low >> shift;
    result |= shift > 0u ? (high << (32u - shift)) : 0u;
    return result;
}

void ClusterBuilder::CreateDGFs(DenseGeometryBuildData *buildDatas, Arena *arena, Mesh *meshes,
                                Vec3i **quantizedVertices, RecordAOSSplits &cluster,
                                int precision)
{
    static const u32 LUT[] = {1, 2, 0};
    struct HashedIndex
    {
        u32 geomID;
        u32 index;
        u32 clusterVertexIndex;
        u32 Hash() const { return MixBits(((u64)geomID << 32) | index); }
        bool operator==(const HashedIndex &other) const
        {
            return index == other.index && geomID == other.geomID;
        }
    };

    auto ComputeEdgeId = [](u32 indices[3], u32 edgeIndex) {
        u32 id0 = indices[edgeIndex];
        u32 id1 = indices[LUT[edgeIndex]];
        return id0 < id1 ? ((u64)id1 << 32) | id0 : ((u64)id0 << 32) | id1;
    };

    int start               = cluster.start;
    int clusterNumTriangles = cluster.count;
    Assert(clusterNumTriangles <= MAX_CLUSTER_TRIANGLES);

    TempArena clusterScratch = ScratchStart(&arena, 1);

    // Maps mesh indices to cluster indices
    HashMap<HashedIndex> vertexHashSet(clusterScratch.arena,
                                       NextPowerOfTwo(clusterNumTriangles * 3));

    // TODO: section 4.3?
    u32 *vertexIndices = PushArray(clusterScratch.arena, u32, clusterNumTriangles * 3);
    u32 *geomIDs       = PushArray(clusterScratch.arena, u32, clusterNumTriangles);
    u32 *clusterVertexIndexToMeshVertexIndex =
        PushArray(clusterScratch.arena, u32, clusterNumTriangles * 3);

    Vec3i *vertices = PushArrayNoZero(clusterScratch.arena, Vec3i, clusterNumTriangles * 3);

    u32 vertexCount = 0;
    u32 indexCount  = 0;

    auto GetIndices = [&vertexIndices](u32 ind[3], u32 triangleIndex) {
        ind[0] = vertexIndices[3 * triangleIndex + 0];
        ind[1] = vertexIndices[3 * triangleIndex + 1];
        ind[2] = vertexIndices[3 * triangleIndex + 2];
    };

    // Calculate the number of bits needed to represent the range of values in the
    // cluster
    Vec3i min(pos_inf);
    Vec3i max(neg_inf);

    for (int i = start; i < start + clusterNumTriangles; i++)
    {
        u32 geomID     = primRefs[i].geomID;
        Mesh &mesh     = meshes[geomID];
        u32 primID     = primRefs[i].primID;
        u32 indices[3] = {
            mesh.indices[3 * primID + 0],
            mesh.indices[3 * primID + 1],
            mesh.indices[3 * primID + 2],
        };

        geomIDs[i - start] = geomID;

        for (int indexIndex = 0; indexIndex < 3; indexIndex++)
        {
            min = Min(min, quantizedVertices[geomID][indices[indexIndex]]);
            max = Max(max, quantizedVertices[geomID][indices[indexIndex]]);

            auto *hashedIndex = vertexHashSet.Find({geomID, indices[indexIndex]});
            if (hashedIndex)
            {
                vertexIndices[indexCount++] = hashedIndex->clusterVertexIndex;
            }
            else
            {
                vertexHashSet.Add(clusterScratch.arena,
                                  {geomID, indices[indexIndex], vertexCount});
                clusterVertexIndexToMeshVertexIndex[vertexCount] = indices[indexIndex];
                vertexIndices[indexCount++]                      = vertexCount;

                vertices[vertexCount] = quantizedVertices[geomID][indices[indexIndex]];
                vertexCount++;
            }
        }
    }

    Assert(vertexCount <= MAX_CLUSTER_VERTICES);

    u32 numBitsX = min[0] == max[0] ? 0 : Log2Int(Max(max[0] - min[0], 1)) + 1;
    u32 numBitsY = min[1] == max[1] ? 0 : Log2Int(Max(max[1] - min[1], 1)) + 1;
    u32 numBitsZ = min[2] == max[2] ? 0 : Log2Int(Max(max[2] - min[2], 1)) + 1;

    Assert(numBitsX < 24 && numBitsY < 24 && numBitsZ < 24);
    Assert(numBitsX + numBitsY + numBitsZ < 64);
    Assert(Abs(min[0]) < (1 << 23) && Abs(min[1]) < (1 << 23) && Abs(min[2]) < (1 << 23));

    u32 numBits = vertexCount * (numBitsX + numBitsY + numBitsZ);

    Vec3u bitWidths(numBitsX, numBitsY, numBitsZ);

    // Build adjacency data for triangles
    u32 *counts  = PushArray(clusterScratch.arena, u32, clusterNumTriangles);
    u32 *offsets = PushArray(clusterScratch.arena, u32, clusterNumTriangles);

    struct EdgeKeyValue
    {
        u64 key;
        u32 geomID;
        u32 count;
        u32 value0;
        u32 value1;
        u32 Hash() const { return MixBits(key ^ geomID); }

        bool operator==(const EdgeKeyValue &other) const
        {
            return key == other.key && geomID == other.geomID;
        }
    };
    HashMap<EdgeKeyValue> edgeIDMap(clusterScratch.arena, NextPowerOfTwo(indexCount));

    // Find triangles with shared edge
    for (u32 triangleIndex = 0; triangleIndex < clusterNumTriangles; triangleIndex++)
    {
        u32 indices[3];
        GetIndices(indices, triangleIndex);
        u32 geomID = geomIDs[triangleIndex];

        for (int edgeIndex = 0; edgeIndex < 3; edgeIndex++)
        {
            u64 edgeID        = ComputeEdgeId(indices, edgeIndex);
            auto edgeKeyValue = edgeIDMap.Find({edgeID, geomID});
            if (edgeKeyValue == 0)
            {
                edgeIDMap.Add(clusterScratch.arena, {edgeID, geomID, 1, triangleIndex, ~0u});
            }
            else
            {
                // TODO: handle non manifolds
                Assert(edgeKeyValue->value1 == ~0u);
                edgeKeyValue->value1 = triangleIndex;
                edgeKeyValue->count  = 2;
            }
        }
    }

    // Find number of adjacent triangles for each triangle
    for (u32 triangleIndex = 0; triangleIndex < clusterNumTriangles; triangleIndex++)
    {
        u32 indices[3];
        GetIndices(indices, triangleIndex);
        u32 geomID = geomIDs[triangleIndex];
        u32 count  = 0;

        for (int edgeIndex = 0; edgeIndex < 3; edgeIndex++)
        {
            u64 edgeID        = ComputeEdgeId(indices, edgeIndex);
            auto edgeKeyValue = edgeIDMap.Find({edgeID, geomID});
            Assert(edgeKeyValue);
            count += edgeKeyValue->count - 1;
        }
        counts[triangleIndex] = count;
    }

    u32 offset = 0;
    for (u32 i = 0; i < clusterNumTriangles; i++)
    {
        offsets[i] = offset;
        offset += counts[i];
    }

    u32 *data = PushArray(clusterScratch.arena, u32, offset);

    for (u32 triangleIndex = 0; triangleIndex < clusterNumTriangles; triangleIndex++)
    {
        u32 indices[3];
        GetIndices(indices, triangleIndex);
        u32 geomID = geomIDs[triangleIndex];
        u32 count  = 0;

        for (int edgeIndex = 0; edgeIndex < 3; edgeIndex++)
        {
            u64 edgeID        = ComputeEdgeId(indices, edgeIndex);
            auto edgeKeyValue = edgeIDMap.Find({edgeID, geomID});
            Assert(edgeKeyValue);
            data[offsets[triangleIndex]] = triangleIndex == edgeKeyValue->value0
                                               ? edgeKeyValue->value1
                                               : edgeKeyValue->value0;
            offsets[triangleIndex] += edgeKeyValue->count - 1;
        }
    }

    for (u32 i = 0; i < clusterNumTriangles; i++)
    {
        offsets[i] -= counts[i];
    }

    // TODO:
    // 1. actually output the data in a format that the GPU can read and decode
    // 2. set up the custom primitive using vulkan
    // 3. handle multiple geom IDs in a single block
    // 4. handle vertices in multiple blocks (duplicate, or references)

    static const u32 removedBit = 0x80000000;
    auto FindMinValence         = [&]() {
        u32 minCount           = ~0u;
        u32 minValenceTriangle = 0;
        for (int i = 0; i < clusterNumTriangles; i++)
        {
            if (!(counts[i] & removedBit) && counts[i] < minCount)
            {
                minCount           = counts[i];
                minValenceTriangle = i;
            }
        }
        return minValenceTriangle;
    };

    u32 minValenceTriangle = FindMinValence();

    // NOTE: implicitly starts with "Restart"
    StaticArray<TriangleStripType> triangleStripTypes(clusterScratch.arena,
                                                      clusterNumTriangles);

    StaticArray<u32> newIndexOrder(clusterScratch.arena, clusterNumTriangles * 3);
    StaticArray<u32> debugIndexOrder(clusterScratch.arena, clusterNumTriangles * 3);
    StaticArray<u32> triangleOrder(clusterScratch.arena, clusterNumTriangles * 3);

    u32 prevTriangle = minValenceTriangle;

    auto RemoveFromNeighbor = [&](u32 tri, u32 neighbor) {
        u32 neighborCount = counts[neighbor] & ~removedBit;
        for (int j = 0; j < neighborCount; j++)
        {
            if (data[offsets[neighbor] + j] == tri)
            {
                Swap(data[offsets[neighbor] + j], data[offsets[neighbor] + neighborCount - 1]);
                counts[neighbor]--;
                return true;
            }
        }
        return false;
    };

    auto MarkUsed = [&](u32 triangle) { counts[minValenceTriangle] |= removedBit; };

    u32 firstIndices[3];
    GetIndices(firstIndices, minValenceTriangle);
    u32 firstGeomId = geomIDs[minValenceTriangle];
    newIndexOrder.Push(firstIndices[0]);
    newIndexOrder.Push(firstIndices[1]);
    newIndexOrder.Push(firstIndices[2]);

    debugIndexOrder.Push(firstIndices[0]);
    debugIndexOrder.Push(firstIndices[1]);
    debugIndexOrder.Push(firstIndices[2]);

    triangleOrder.Push(minValenceTriangle);

    MarkUsed(minValenceTriangle);

    // TODO: half edge structure instead of hash table?
    // Find the new ordering of triangles and construct the triangle strip
    for (u32 numAddedTriangles = 1; numAddedTriangles < clusterNumTriangles;
         numAddedTriangles++)
    {
        u32 newMinValenceTriangle = ~0u;

        TriangleStripType backTrackStripType = TriangleStripType::Restart;
        u32 count                            = counts[minValenceTriangle] & ~removedBit;
        // If triangle has no neighbors, attempt to backtrack
        if (!count)
        {
            MarkUsed(minValenceTriangle);
            // If can't backtrack, restart
            if (numAddedTriangles == 1 ||
                triangleStripTypes[numAddedTriangles - 2] == TriangleStripType::Restart ||
                triangleStripTypes[numAddedTriangles - 2] == TriangleStripType::Backtrack ||
                (counts[prevTriangle] & ~removedBit) == 0)
            {
                triangleStripTypes.Push(TriangleStripType::Restart);

                prevTriangle       = minValenceTriangle;
                minValenceTriangle = FindMinValence();
                u32 indices[3];
                GetIndices(indices, minValenceTriangle);
                u32 geomID = geomIDs[minValenceTriangle];

                newIndexOrder.Push(indices[0]);
                newIndexOrder.Push(indices[1]);
                newIndexOrder.Push(indices[2]);

                debugIndexOrder.Push(indices[0]);
                debugIndexOrder.Push(indices[1]);
                debugIndexOrder.Push(indices[2]);

                triangleOrder.Push(minValenceTriangle);
                continue;
            }
            // Backtrack
            TriangleStripType prevType = triangleStripTypes[numAddedTriangles - 2];
            minValenceTriangle         = prevTriangle;
            Assert(prevType == TriangleStripType::Edge1 ||
                   prevType == TriangleStripType::Edge2);
            backTrackStripType = prevType == TriangleStripType::Edge1
                                     ? TriangleStripType::Edge2
                                     : TriangleStripType::Edge1;

            triangleStripTypes.Push(TriangleStripType::Backtrack);

            // Get last neighbor from previous triangle
            Assert(counts[minValenceTriangle] & removedBit &&
                   (counts[minValenceTriangle] & ~removedBit) == 1);
            newMinValenceTriangle = data[offsets[minValenceTriangle]];
            counts[minValenceTriangle]--;
        }
        else
        {
            // Remove min valence triangle from neighbor's adjacency list
            u32 minCount = ~0u;
            u32 minI     = 0;
            for (int i = 0; i < count; i++)
            {
                u32 neighborTriangle = data[offsets[minValenceTriangle] + i];
                bool success = RemoveFromNeighbor(minValenceTriangle, neighborTriangle);
                Assert(success);
                // Find neighbor with minimum valence
                if (counts[neighborTriangle] < minCount)
                {
                    minCount              = counts[neighborTriangle];
                    newMinValenceTriangle = neighborTriangle;
                    minI                  = i;
                }
            }
            Swap(data[offsets[minValenceTriangle] + minI],
                 data[offsets[minValenceTriangle] + count - 1]);
            counts[minValenceTriangle]--;
            MarkUsed(minValenceTriangle);
            Assert(minCount != ~0u);
        }
        Assert(newMinValenceTriangle != ~0u);
        triangleOrder.Push(newMinValenceTriangle);

        // Find what edge is shared, and rotate
        u32 indices[3];
        GetIndices(indices, newMinValenceTriangle);
        u32 geomID = geomIDs[newMinValenceTriangle];
        for (int edgeIndex = 0; edgeIndex < 3; edgeIndex++)
        {
            u64 edgeID        = ComputeEdgeId(indices, edgeIndex);
            auto edgeKeyValue = edgeIDMap.Find({edgeID, geomID});
            Assert(edgeKeyValue);

            if (edgeKeyValue->value0 == minValenceTriangle ||
                edgeKeyValue->value1 == minValenceTriangle)
            {
                // Rotate
                // [0, 1, 2] -> [1, 2, 0]
                if (edgeIndex == 1)
                {
                    vertexIndices[3 * newMinValenceTriangle]     = indices[1];
                    vertexIndices[3 * newMinValenceTriangle + 1] = indices[2];
                    vertexIndices[3 * newMinValenceTriangle + 2] = indices[0];
                    newIndexOrder.Push(indices[0]);

                    debugIndexOrder.Push(indices[1]);
                    debugIndexOrder.Push(indices[2]);
                    debugIndexOrder.Push(indices[0]);
                }
                // [0, 1, 2] -> [2, 0, 1]
                else if (edgeIndex == 2)
                {
                    vertexIndices[3 * newMinValenceTriangle]     = indices[2];
                    vertexIndices[3 * newMinValenceTriangle + 1] = indices[0];
                    vertexIndices[3 * newMinValenceTriangle + 2] = indices[1];
                    newIndexOrder.Push(indices[1]);

                    debugIndexOrder.Push(indices[2]);
                    debugIndexOrder.Push(indices[0]);
                    debugIndexOrder.Push(indices[1]);
                }
                else
                {
                    newIndexOrder.Push(indices[2]);

                    debugIndexOrder.Push(indices[0]);
                    debugIndexOrder.Push(indices[1]);
                    debugIndexOrder.Push(indices[2]);
                }

                u32 oldIndices[3];
                GetIndices(oldIndices, minValenceTriangle);
                // Find whether we're attached to edge1 or 2 from previous triangle
                for (int oldEdgeIndex = 0; oldEdgeIndex < 3; oldEdgeIndex++)
                {
                    u64 oldEdgeId = ComputeEdgeId(oldIndices, oldEdgeIndex);

                    if (oldEdgeId == edgeID)
                    {
                        if (backTrackStripType != TriangleStripType::Restart)
                        {
                            Assert((backTrackStripType == TriangleStripType::Edge1 &&
                                    oldEdgeIndex == 1) ||
                                   (backTrackStripType == TriangleStripType::Edge2 &&
                                    oldEdgeIndex == 2));
                            break;
                        }

                        // If a restart occurs, a new triangle could connect with
                        // edge0. Rotate the old triangle so it's connected with
                        // edge1 instead
                        if (oldEdgeIndex == 0)
                        {
                            Assert(numAddedTriangles == 1 ||
                                   triangleStripTypes[numAddedTriangles - 2] ==
                                       TriangleStripType::Restart);
                            newIndexOrder[newIndexOrder.Length() - 4] = oldIndices[2];
                            newIndexOrder[newIndexOrder.Length() - 3] = oldIndices[0];
                            newIndexOrder[newIndexOrder.Length() - 2] = oldIndices[1];
                            vertexIndices[3 * minValenceTriangle]     = oldIndices[2];
                            vertexIndices[3 * minValenceTriangle + 1] = oldIndices[0];
                            vertexIndices[3 * minValenceTriangle + 2] = oldIndices[1];

                            debugIndexOrder[debugIndexOrder.Length() - 6] = oldIndices[2];
                            debugIndexOrder[debugIndexOrder.Length() - 5] = oldIndices[0];
                            debugIndexOrder[debugIndexOrder.Length() - 4] = oldIndices[1];

                            oldEdgeIndex = 1;
                        }
                        triangleStripTypes.Push(oldEdgeIndex == 1 ? TriangleStripType::Edge1
                                                                  : TriangleStripType::Edge2);
                        break;
                    }
                }
                break;
            }
        }
        prevTriangle       = minValenceTriangle;
        minValenceTriangle = newMinValenceTriangle;
    }

    // Append all data
    auto &buildData = buildDatas[0];

    u32 vertexBitStreamSize = (numBits + 7) >> 3;
    u32 baseAddress         = buildData.byteBuffer.Length();

    // Use the new index order to write compressed vertex buffer and reuse buffer
    u32 maxReuseBufferSize = clusterNumTriangles * 3;
    StaticArray<u32> mapOldIndexToDGFIndex(clusterScratch.arena, vertexCount, vertexCount);
    StaticArray<u32> reuseBuffer(clusterScratch.arena, maxReuseBufferSize);
    u32 maxReuseIndex = 0;

    BitVector firstUse(clusterScratch.arena, clusterNumTriangles * 3);
    u32 currentFirstUseBit = 0;
    BitVector usedVertices(clusterScratch.arena, vertexCount);
    u32 addedVertexCount = 0;

    u32 bitOffset = 0;
    auto *node    = buildData.byteBuffer.AddNode(vertexBitStreamSize);
    for (auto index : newIndexOrder)
    {
        if (usedVertices.GetBit(index))
        {
            currentFirstUseBit++;
            reuseBuffer.Push(mapOldIndexToDGFIndex[index]);
            maxReuseIndex = Max(mapOldIndexToDGFIndex[index], maxReuseIndex);
        }
        else
        {
            usedVertices.SetBit(index);
            firstUse.SetBit(currentFirstUseBit);
            currentFirstUseBit++;
            mapOldIndexToDGFIndex[index] = addedVertexCount;
            addedVertexCount++;

            for (int i = 0; i < 3; i++)
            {
                int p = vertices[index][i] - min[i];
                Assert(p >= 0);
                WriteBits((u32 *)node->values, bitOffset, p, bitWidths[i]);
            }
        }
    }

    for (u32 i = 0; i < debugIndexOrder.Length(); i++)
    {
        debugIndexOrder[i] = mapOldIndexToDGFIndex[debugIndexOrder[i]];
    }

    Assert(bitOffset == numBits);
    bitOffset = 0;

    // Write reuse buffer (byte aligned)
    u32 numIndexBits = Log2Int(maxReuseIndex) + 1;
    Assert(numIndexBits >= 1 && numIndexBits <= 8);
    u32 bitStreamSize = (numIndexBits * reuseBuffer.Length() + currentFirstUseBit +
                         triangleStripTypes.Length() * 2 + 7) >>
                        3;
    node = buildData.byteBuffer.AddNode(bitStreamSize);

    for (u32 reuseIndex : reuseBuffer)
    {
        WriteBits((u32 *)node->values, bitOffset, reuseIndex, numIndexBits);
    }

    // Write control bits
    u32 ctrlBitOffset = bitOffset;
    Assert(triangleStripTypes.Length() == clusterNumTriangles - 1u);
    for (auto &type : triangleStripTypes)
    {
        Assert((u32)type < 4);
        WriteBits((u32 *)node->values, bitOffset, (u32)type, 2);
    }

    // Write first use bits
    u32 firstBitWriteOffset  = 0;
    u32 firstUseBitWriteSize = Min(currentFirstUseBit - firstBitWriteOffset, 32u);
    while (firstBitWriteOffset != currentFirstUseBit)
    {
        WriteBits((u32 *)node->values, bitOffset, firstUse.bits[firstBitWriteOffset >> 5],
                  firstUseBitWriteSize);
        firstBitWriteOffset += firstUseBitWriteSize;
        firstUseBitWriteSize = Min(currentFirstUseBit - firstBitWriteOffset, 32u);
    }

    // Write header
    PackedDenseGeometryHeader packed = {};
    u32 headerOffset                 = 0;
    packed.a                         = baseAddress;
    headerOffset += 32;

    packed.b = BitFieldPackI32(packed.b, min[0], headerOffset, ANCHOR_WIDTH);
    packed.b = BitFieldPackU32(packed.b, clusterNumTriangles, headerOffset, 8);

    packed.c = BitFieldPackI32(packed.c, min[1], headerOffset, ANCHOR_WIDTH);
    packed.c = BitFieldPackU32(packed.c, numBitsX, headerOffset, 5);
    packed.c = BitFieldPackU32(packed.c, numIndexBits - 1, headerOffset, 3);

    packed.d = BitFieldPackI32(packed.d, min[2], headerOffset, ANCHOR_WIDTH);
    packed.d = BitFieldPackU32(packed.d, precision - CLUSTER_MIN_PRECISION, headerOffset, 8);

    packed.e = BitFieldPackU32(packed.e, vertexCount, headerOffset, 9);
    Assert(reuseBuffer.Length() < (1 << 8));
    packed.e = BitFieldPackU32(packed.e, reuseBuffer.Length(), headerOffset, 8);
    packed.e = BitFieldPackU32(packed.e, numBitsY, headerOffset, 5);
    packed.e = BitFieldPackU32(packed.e, numBitsZ, headerOffset, 5);

    buildData.headers.AddBack() = packed;

    // Reorder refs
    PrimRef *tempRefs = PushArrayNoZero(clusterScratch.arena, PrimRef, clusterNumTriangles);
    MemoryCopy(tempRefs, primRefs + start, clusterNumTriangles * sizeof(PrimRef));

    for (int i = 0; i < clusterNumTriangles; i++)
    {
        primRefs[start + i] = tempRefs[triangleOrder[i]];
    }

    // Debug
    auto *typesNode = buildData.types.AddNode(triangleStripTypes.Length());
    MemoryCopy(typesNode->values, triangleStripTypes.data,
               sizeof(TriangleStripType) * triangleStripTypes.Length());

    u32 numu32s        = (currentFirstUseBit + 31) >> 5;
    auto *firstUseNode = buildData.firstUse.AddNode(numu32s);
    MemoryCopy(firstUseNode->values, firstUse.bits, (currentFirstUseBit + 7) >> 3);

    auto *reuseNode = buildData.reuse.AddNode(reuseBuffer.Length());
    MemoryCopy(reuseNode->values, reuseBuffer.data, reuseBuffer.Length() * sizeof(u32));

    auto *debugIndexNode = buildData.debugIndices.AddNode(debugIndexOrder.Length());
    MemoryCopy(debugIndexNode->values, debugIndexOrder.data,
               debugIndexOrder.Length() * sizeof(u32));
    ScratchEnd(clusterScratch);
}

} // namespace rt
