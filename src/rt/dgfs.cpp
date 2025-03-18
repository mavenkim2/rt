#include "scene.h"
#include "bvh/bvh_aos.h"
#include "bvh/bvh_types.h"
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

void DenseGeometryBuildData::Init()
{
    arena      = ArenaAlloc();
    byteBuffer = ChunkedLinkedList<u8>(arena, 1024);
    offsets    = ChunkedLinkedList<u32>(arena);
}

void DenseGeometryBuildData::Merge(DenseGeometryBuildData &other)
{
    u32 offset = byteBuffer.Length();
    for (auto *node = other.offsets.first; node != 0; node = node->next)
    {
        for (int i = 0; i < node->count; i++)
        {
            node->values[i] += offset;
        }
    }
    byteBuffer.Merge(&other.byteBuffer);
    offsets.Merge(&other.offsets);
}

void AppendBitStream(std::vector<u8> &bitStream, u32 bitsToAdd, u32 &currentBitOffset,
                     u32 numBits)
{
    if (!numBits) return;
    Assert(currentBitOffset >= 0 && currentBitOffset < 8);

    u32 remaining = 0;
    if (currentBitOffset)
    {
        remaining = Min(8 - currentBitOffset, numBits);
        bitStream.back() |= ((bitsToAdd & ((1 << remaining) - 1)) << currentBitOffset);
    }

    u32 offset     = remaining + Min(8u, numBits - remaining);
    u32 prevOffset = remaining;

    while (prevOffset != numBits)
    {
        u32 mask = ((1 << offset) - 1) & ~((1 << prevOffset) - 1);
        bitStream.push_back(((bitsToAdd & mask) >> prevOffset) & 0xff);
        prevOffset = offset;
        offset += Min(8u, numBits - offset);
    }
    currentBitOffset = (currentBitOffset + numBits) & 7;
}

void ClusterBuilder::BuildClusters(RecordAOSSplits &record, bool parallel)

{
    auto *heuristic = (Heuristic *)h;
    const int N     = 4;
    Assert(record.count > 0);

    RecordAOSSplits childRecords[N];
    u32 numChildren = 0;

    Split split = heuristic->Bin(record);

    if (record.count <= clusterSize)
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
            if (childRecord.count <= clusterSize) continue;

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
            quantizedVertices[i][j] = Vec3i(mesh.p[j] * quantize + 0.5f);
        }
    }

    // TODO: this needs to be permanent

    for (int threadIndex = 0; threadIndex < threadClusters.Length(); threadIndex++)
    {
        ClusterList &list = threadClusters[threadIndex];
        for (auto *node = list.l.first; node != 0; node = node->next)
        {
            for (int clusterIndex = 0; clusterIndex < node->count; clusterIndex++)
            {
                RecordAOSSplits &cluster = node->values[clusterIndex];
                CreateDGFs(buildData, meshScratch.temp.arena, meshes, quantizedVertices,
                           cluster);
            }
        }
    }
}

void ClusterBuilder::CreateDGFs(DenseGeometryBuildData *buildDatas, Arena *arena, Mesh *meshes,
                                Vec3i **quantizedVertices, RecordAOSSplits &cluster)
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

    TempArena clusterScratch = ScratchStart(&arena, 1);

    // Maps mesh indices to cluster indices
    HashMap<HashedIndex> vertexHashSet(clusterScratch.arena,
                                       NextPowerOfTwo(clusterNumTriangles * 3));

    // TODO: section 4.3?
    u32 *vertexIndices = PushArray(clusterScratch.arena, u32, clusterNumTriangles * 3);
    u32 *geomIDs       = PushArray(clusterScratch.arena, u32, clusterNumTriangles);
    u32 *clusterVertexIndexToMeshVertexIndex =
        PushArray(clusterScratch.arena, u32, clusterNumTriangles * 3);

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
                vertexCount++;
            }
        }
    }

    u32 numBitsX = min[0] == max[0] ? 0 : Log2Int(Max(max[0] - min[0] - 1, 1)) + 1;
    u32 numBitsY = min[1] == max[1] ? 0 : Log2Int(Max(max[1] - min[1] - 1, 1)) + 1;
    u32 numBitsZ = min[2] == max[2] ? 0 : Log2Int(Max(max[2] - min[2] - 1, 1)) + 1;

    DenseGeometryHeader geometry;
    geometry.anchor       = min;
    geometry.bitWidths[0] = SafeTruncateU32ToU8(numBitsX);
    geometry.bitWidths[1] = SafeTruncateU32ToU8(numBitsY);
    geometry.bitWidths[2] = SafeTruncateU32ToU8(numBitsZ);

    int numBytes = (vertexCount * (numBitsX + numBitsY + numBitsZ) + 7) >> 3;

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

    // Find node with minimum valence

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

    enum class TriangleStripType
    {
        Restart,
        Edge1,
        Edge2,
        Backtrack,
    };

    // NOTE: implicitly starts with "Restart"
    StaticArray<TriangleStripType> triangleStripTypes(clusterScratch.arena,
                                                      clusterNumTriangles);

    std::vector<u8> vertexBitStream;
    vertexBitStream.reserve(numBytes);
    u32 bitOffset        = 0;
    u32 addedVertexCount = 0;

    std::vector<u32> mapOldIndexToDGFIndex(clusterNumTriangles * 3);

    std::vector<u8> reuseBuffer;
    u32 reuseBitOffset = 0;
    int numIndexBits   = Log2Int(NextPowerOfTwo(clusterNumTriangles));
    reuseBuffer.reserve(clusterNumTriangles * 3 * sizeof(u32));

    U128 firstUse          = {};
    u32 currentFirstUseBit = 0;
    u64 usedTriangles      = 0;
    u32 numReuse           = 0;

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

    auto CheckUsedTriangles = [&](u32 geomID, u32 index) {
        if (usedTriangles & (1 << index))
        {
            currentFirstUseBit++;
            Assert(index < (1 << numIndexBits));
            AppendBitStream(reuseBuffer, index, reuseBitOffset, numIndexBits);
            return true;
        }

        usedTriangles |= (1 << index);
        firstUse.SetBit(currentFirstUseBit);
        currentFirstUseBit++;
        mapOldIndexToDGFIndex[index] = addedVertexCount++;

        for (int i = 0; i < 3; i++)
        {
            AppendBitStream(
                vertexBitStream,
                quantizedVertices[geomID][clusterVertexIndexToMeshVertexIndex[index]][i] -
                    min[i],
                bitOffset, geometry.bitWidths[i]);
        }
        return false;
    };

    auto MarkUsed = [&](u32 triangle) { counts[minValenceTriangle] |= removedBit; };

    u32 firstIndices[3];
    GetIndices(firstIndices, minValenceTriangle);
    u32 firstGeomId = geomIDs[minValenceTriangle];
    CheckUsedTriangles(firstGeomId, firstIndices[0]);
    CheckUsedTriangles(firstGeomId, firstIndices[1]);
    CheckUsedTriangles(firstGeomId, firstIndices[2]);
    MarkUsed(minValenceTriangle);

    // TODO: half edge structure instead of hash table?
    for (u32 numAddedTriangles = 1; numAddedTriangles < clusterNumTriangles;
         numAddedTriangles++)
    {
        u32 newMinValenceTriangle = ~0u;

        TriangleStripType backTrackStripType = TriangleStripType::Restart;
        // If triangle has no neighbors, attempt to backtrack
        if (!counts[minValenceTriangle])
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

                CheckUsedTriangles(geomID, indices[0]);
                CheckUsedTriangles(geomID, indices[1]);
                CheckUsedTriangles(geomID, indices[2]);
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
            u32 count    = counts[minValenceTriangle] & ~removedBit;
            Assert(count);
            u32 minI = 0;
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

                    CheckUsedTriangles(geomID, indices[0]);
                }
                // [0, 1, 2] -> [2, 0, 1]
                else if (edgeIndex == 2)
                {
                    vertexIndices[3 * newMinValenceTriangle]     = indices[2];
                    vertexIndices[3 * newMinValenceTriangle + 1] = indices[0];
                    vertexIndices[3 * newMinValenceTriangle + 2] = indices[1];

                    CheckUsedTriangles(geomID, indices[1]);
                }
                else
                {
                    CheckUsedTriangles(geomID, indices[2]);
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
                            vertexIndices[3 * minValenceTriangle]     = oldIndices[2];
                            vertexIndices[3 * minValenceTriangle + 1] = oldIndices[0];
                            vertexIndices[3 * minValenceTriangle + 2] = oldIndices[1];
                            oldEdgeIndex                              = 1;
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
    auto &buildData         = buildDatas[0];
    u32 reuseBufferSize     = (u32)reuseBuffer.size();
    u32 vertexBitStreamSize = (u32)vertexBitStream.size();
    u32 firstUseSize        = (currentFirstUseBit + 7) >> 3;
    u32 controlBitsSize     = (triangleStripTypes.Length() * 2 + 7) >> 3;
    u32 size = sizeof(DenseGeometryHeader) + vertexBitStreamSize + reuseBufferSize +
               controlBitsSize + firstUseSize;
    Assert(size < (1 << 16));

    u32 currentOffset           = buildData.byteBuffer.Length();
    buildData.offsets.AddBack() = currentOffset;
    auto *node                  = buildData.byteBuffer.AddNode(size);
    offset                      = 0;
    MemoryCopy(node->values, &geometry, sizeof(DenseGeometryHeader));
    DenseGeometryHeader *header = (DenseGeometryHeader *)node->values;
    offset += sizeof(DenseGeometryHeader);
    MemoryCopy(node->values + offset, vertexBitStream.data(), vertexBitStreamSize);
    offset += vertexBitStreamSize;
    header->reuseStart = offset;
    MemoryCopy(node->values + offset, reuseBuffer.data(), reuseBufferSize);
    offset += reuseBufferSize;
    MemoryCopy(node->values, &firstUse, firstUseSize);
    offset += firstUseSize;

    u32 currentTypeBitOffset = 0;
    for (auto &type : triangleStripTypes)
    {
        node->values[offset] |= ((u32)type) << currentTypeBitOffset;
        currentTypeBitOffset += 2;
        offset += currentTypeBitOffset >> 3;
        currentTypeBitOffset = currentTypeBitOffset & 7;
    }

    ScratchEnd(clusterScratch);
}

} // namespace rt
