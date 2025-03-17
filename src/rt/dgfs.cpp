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
    threadClusters    = StaticArray<ClusterList>(arenas[GetThreadIndex()], numProcessors);
    for (u32 i = 0; i < numProcessors; i++)
    {
        threadClusters[i].l = ChunkedLinkedList<RecordAOSSplits>(arenas[i]);
    }

    h = PushStructConstruct(arena, Heuristic)(primRefs, scene, 0);
}

void AppendBitStream(std::vector<u8> &bitStream, u32 bitsToAdd, u32 &currentBitOffset,
                     u32 numBits)
{
    Assert(currentBitOffset >= 0 && currentBitOffset < 8);
    Assert(numBits);

    u32 remaining = 8 - currentBitOffset;
    bitStream.back() |= ((bitsToAdd & ((1 << remaining) - 1)) << currentBitOffset);

    u32 newBitOffset = 0;
    newBitOffset += Min(8u, numBits - remaining);
    u32 offset     = remaining + Min(8u, numBits - remaining);
    u32 prevOffset = remaining;

    while (prevOffset != numBits)
    {
        u32 mask = ((1 << offset) - 1) & ~((1 << prevOffset) - 1);
        bitStream.push_back(((bitsToAdd & mask) >> prevOffset) & 0xff);
        prevOffset = offset;
        offset += Min(8u, numBits - offset);
        newBitOffset += Min(8u, numBits - offset);
    }
    currentBitOffset = newBitOffset & 7;
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

void ClusterBuilder::CreateDGFs(Mesh *meshes, int numMeshes, Bounds &sceneBounds)
{
    static const int b     = 16;
    static const f32 scale = 1.f / ((1 << (b - 1)) - 1.f);
    Vec3f sceneExtent      = ToVec3f(sceneBounds.maxP - sceneBounds.minP);

    f32 eX = sceneExtent[0] * scale;
    f32 eY = sceneExtent[1] * scale;
    f32 eZ = sceneExtent[2] * scale;

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

    Vec3f quantize = 1.f / Vec3f(eX, eY, eZ);

    ScratchArena meshScratch;
    Vec3i **quantizedVertices = PushArrayNoZero(meshScratch.temp.arena, Vec3i *, numMeshes);

    for (int i = 0; i < numMeshes; i++)
    {
        Mesh &mesh = meshes[i];
        quantizedVertices[i] =
            PushArrayNoZero(meshScratch.temp.arena, Vec3i, mesh.numVertices);

        // Quantize to global grid
        for (int j = 0; j < mesh.numVertices; j++)
        {
            quantizedVertices[i][j] = mesh.p[j] * quantize;
        }

        // for (int j = 0; j < cluster.numVertices; j++)
        // {
        //     quantizedVertices[i][j] -= min;
        // }
    }

    for (int threadIndex = 0; threadIndex < threadClusters.Length(); threadIndex++)
    {
        ClusterList &list = threadClusters[threadIndex];
        for (auto *node = list.l.first; node != 0; node = node->next)
        {
            for (int clusterIndex = 0; clusterIndex < node->count; clusterIndex++)
            {
                RecordAOSSplits &cluster = node->values[clusterIndex];
                CreateDGFs(meshScratch.temp.arena, meshes, quantizedVertices, cluster);
            }
        }
    }
}

void ClusterBuilder::CreateDGFs(Arena *arena, Mesh *meshes, Vec3i **quantizedVertices,
                                RecordAOSSplits &cluster)
{
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

    auto ComputeEdgeId = [](u32 id0, u32 id1) {
        return id0 < id1 ? ((u64)id1 << 32) | id0 : ((u64)id0 << 32) | id1;
    };

    int start               = cluster.start;
    int clusterNumTriangles = cluster.count;

    TempArena clusterScratch = ScratchStart(&arena, 1);

    // Maps mesh indices to cluster indices
    HashMap<HashedIndex> vertexHashSet(clusterScratch.arena, clusterNumTriangles * 3);

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
    Vec3i max(pos_inf);

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

            auto *hashedIndex = vertexHashSet.Find({indices[indexIndex]});
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

    u32 numBitsX = Log2Int(Max(max[0] - min[0] - 1, 1)) + 1;
    u32 numBitsY = Log2Int(Max(max[1] - min[1] - 1, 1)) + 1;
    u32 numBitsZ = Log2Int(Max(max[1] - min[1] - 1, 1)) + 1;

    DenseGeometry geometry;
    geometry.anchor       = min;
    geometry.bitWidths[0] = SafeTruncateU32ToU8(numBitsX);
    geometry.bitWidths[1] = SafeTruncateU32ToU8(numBitsY);
    geometry.bitWidths[2] = SafeTruncateU32ToU8(numBitsZ);

    int numBytes = (vertexCount * (numBitsX + numBitsY + numBitsZ) + 7) >> 3;

    std::vector<u8> vertexBitStream;
    vertexBitStream.reserve(numBytes);
    u32 bitOffset        = 0;
    u32 addedVertexCount = 0;

    // Build adjacency data for triangles
    u32 *counts  = PushArray(clusterScratch.arena, u32, clusterNumTriangles);
    u32 *offsets = PushArray(clusterScratch.arena, u32, clusterNumTriangles);
    u32 *data    = PushArray(clusterScratch.arena, u32, clusterNumTriangles);

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
            u64 edgeID = ComputeEdgeId(indices[edgeIndex], indices[(edgeIndex + 1) & 3]);
            auto edgeKeyValue = edgeIDMap.Find({edgeID, geomID});
            if (edgeKeyValue == 0)
            {
                edgeIDMap.Add(clusterScratch.arena, {edgeID, geomID, 1, triangleIndex, ~0u});
            }
            else
            {
                // TODO: handle non manifolds
                if (edgeKeyValue->value1 == ~0u) Assert(0);
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
            u64 edgeID = ComputeEdgeId(indices[edgeIndex], indices[(edgeIndex + 1) & 3]);
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

    for (u32 triangleIndex = 0; triangleIndex < clusterNumTriangles; triangleIndex++)
    {
        u32 indices[3];
        GetIndices(indices, triangleIndex);
        u32 geomID = geomIDs[triangleIndex];
        u32 count  = 0;

        for (int edgeIndex = 0; edgeIndex < 3; edgeIndex++)
        {
            u64 edgeID = ComputeEdgeId(indices[edgeIndex], indices[(edgeIndex + 1) & 3]);
            auto edgeKeyValue = edgeIDMap.Find({edgeID, geomID});
            Assert(edgeKeyValue);
            data[offsets[triangleIndex]++] = triangleIndex == edgeKeyValue->value0
                                                 ? edgeKeyValue->value1
                                                 : edgeKeyValue->value0;
        }
    }

    for (u32 i = 0; i < clusterNumTriangles; i++)
    {
        offsets[i] -= counts[i];
    }

    // Find node with minimum valence
    auto FindMinValence = [&]() {
        u32 minCount           = pos_inf;
        u32 minValenceTriangle = 0;
        for (int i = 0; i < clusterNumTriangles; i++)
        {
            if (counts[i] < minCount)
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
        None,
        Restart,
        Edge1,
        Edge2,
        Backtrack,
    };

    // NOTE: implicitly starts with "Restart"
    std::vector<TriangleStripType> triangleStripTypes;
    triangleStripTypes.reserve(clusterNumTriangles);

    std::vector<u32> mapOldIndexToDGFIndex(clusterNumTriangles * 3);
    std::vector<u32> reuseBuffer;
    reuseBuffer.reserve(clusterNumTriangles * 3);

    u64 firstUse           = 0;
    u32 currentFirstUseBit = 0;
    u64 usedTriangles      = 0;
    u32 numReuse           = 0;

    u32 prevTriangle = minValenceTriangle;

    auto CheckUsedTriangles = [&](u32 geomID, u32 index) {
        if (usedTriangles & (1 << index))
        {
            currentFirstUseBit++;
            reuseBuffer.push_back(mapOldIndexToDGFIndex[index]);
            // Add to reuse buffer
            return true;
        }

        usedTriangles |= (1 << index);
        firstUse |= (1 << currentFirstUseBit);
        currentFirstUseBit++;
        mapOldIndexToDGFIndex[index] = addedVertexCount++;

        for (int i = 0; i < 3; i++)
        {
            AppendBitStream(vertexBitStream, quantizedVertices[geomID][index][i], bitOffset,
                            geometry.bitWidths[i]);
        }
        return false;
    };

    for (u32 numAddedTriangles = 1; numAddedTriangles < clusterNumTriangles;
         numAddedTriangles++)
    {
        u32 newMinValenceTriangle = ~0u;

        TriangleStripType backTrackStripType = TriangleStripType::None;
        // If triangle has no neighbors, attempt to backtrack
        if (!counts[minValenceTriangle])
        {
            // If can't backtrack, restart
            if (numAddedTriangles == 1 ||
                triangleStripTypes[numAddedTriangles - 1] == TriangleStripType::Restart ||
                triangleStripTypes[numAddedTriangles - 1] == TriangleStripType::Backtrack ||
                !counts[prevTriangle])
            {
                triangleStripTypes.push_back(TriangleStripType::Restart);

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
            TriangleStripType prevType = triangleStripTypes[numAddedTriangles - 1];
            minValenceTriangle         = prevTriangle;
            Assert(prevType == TriangleStripType::Edge1 ||
                   prevType == TriangleStripType::Edge2);
            backTrackStripType = prevType == TriangleStripType::Edge1
                                     ? TriangleStripType::Edge2
                                     : TriangleStripType::Edge1;

            triangleStripTypes.push_back(TriangleStripType::Backtrack);
        }

        Assert(counts[minValenceTriangle]);
        // Remove min valence triangle from neighbor's adjacency list
        u32 minCount = pos_inf;
        for (int i = 0; i < counts[minValenceTriangle]; i++)
        {
            u32 neighborTriangle = data[offsets[minValenceTriangle] + i];
            bool success         = false;
            u32 neighborCount    = counts[neighborTriangle];
            for (int j = 0; j < neighborCount; j++)
            {
                if (data[offsets[neighborTriangle] + j] == minValenceTriangle)
                {
                    Swap(data[offsets[neighborTriangle] + j],
                         data[offsets[neighborTriangle] + neighborCount - 1]);
                    counts[neighborTriangle]--;
                    success = true;
                    break;
                }
            }
            Assert(success);
            // Find neighbor with minimum valence
            if (counts[neighborTriangle] < minCount)
            {
                minCount              = counts[neighborTriangle];
                newMinValenceTriangle = neighborTriangle;
            }
        }
        Assert(newMinValenceTriangle != ~0u);

        // Find what edge is shared, and rotate
        u32 indices[3];
        GetIndices(indices, newMinValenceTriangle);
        u32 geomID = geomIDs[newMinValenceTriangle];
        for (int edgeIndex = 0; edgeIndex < 3; edgeIndex++)
        {
            u64 edgeID = ComputeEdgeId(indices[edgeIndex], indices[(edgeIndex + 1) & 3]);
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
                    u64 oldEdgeId = ComputeEdgeId(oldIndices[oldEdgeIndex],
                                                  oldIndices[(oldEdgeIndex + 1) & 3]);
                    if (oldEdgeId == edgeID)
                    {
                        if (backTrackStripType != TriangleStripType::None)
                        {
                            Assert((backTrackStripType == TriangleStripType::Edge1 &&
                                    oldEdgeId == 1) ||
                                   (backTrackStripType == TriangleStripType::Edge2 &&
                                    oldEdgeId == 2));
                            break;
                        }

                        // If a restart occurs, a new triangle could connect with
                        // edge0. Rotate the old triangle so it's connected with
                        // edge1 instead
                        if (oldEdgeIndex == 0)
                        {
                            vertexIndices[3 * minValenceTriangle]     = oldIndices[2];
                            vertexIndices[3 * minValenceTriangle + 1] = oldIndices[0];
                            vertexIndices[3 * minValenceTriangle + 2] = oldIndices[1];
                            oldEdgeIndex                              = 1;
                        }
                        triangleStripTypes.push_back(oldEdgeIndex == 1
                                                         ? TriangleStripType::Edge1
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
    ScratchEnd(clusterScratch);
}

} // namespace rt
