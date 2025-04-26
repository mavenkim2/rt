#include "base.h"
#include "bit_packing.h"
#include "containers.h"
#include "math/basemath.h"
#include "scene.h"
#include "bvh/bvh_aos.h"
#include "bvh/bvh_types.h"
#include "shader_interop/dense_geometry_shaderinterop.h"
#include "shader_interop/ray_shaderinterop.h"
#include "dgfs.h"
#include "graphics/ptex.h"
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
    headers    = ChunkedLinkedList<PackedDenseGeometryHeader>(arena);

    // Debug
#if 0
    firstUse     = ChunkedLinkedList<u32>(arena);
    reuse        = ChunkedLinkedList<u32>(arena);
#endif
    types                       = ChunkedLinkedList<TriangleStripType>(arena);
    debugIndices                = ChunkedLinkedList<u32>(arena);
    debugRestartCountPerDword   = ChunkedLinkedList<u32>(arena);
    debugRestartHighBitPerDword = ChunkedLinkedList<u32>(arena);
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

Mesh ConvertQuadToTriangleMesh(Arena *arena, Mesh mesh)
{
    Assert(mesh.numFaces * 4 == mesh.numIndices);
    u32 newNumIndices = mesh.numIndices / 4 * 6;
    u32 *newIndices   = PushArrayNoZero(arena, u32, mesh.numIndices / 4 * 6);

    u32 *faceIDs = PushArrayNoZero(arena, u32, mesh.numFaces * 2);
    for (int faceIndex = 0; faceIndex < mesh.numFaces; faceIndex++)
    {
        u32 id0 = mesh.indices[4 * faceIndex + 0];
        u32 id1 = mesh.indices[4 * faceIndex + 1];
        u32 id2 = mesh.indices[4 * faceIndex + 2];
        u32 id3 = mesh.indices[4 * faceIndex + 3];

        u32 *writeIndices = newIndices + 6 * faceIndex;

        writeIndices[0]            = id0;
        writeIndices[1]            = id1;
        writeIndices[2]            = id2;
        writeIndices[3]            = id0;
        writeIndices[4]            = id2;
        writeIndices[5]            = id3;
        faceIDs[2 * faceIndex + 0] = faceIndex;
        faceIDs[2 * faceIndex + 1] = faceIndex;
    }
    Mesh result       = mesh;
    result.indices    = newIndices;
    result.numIndices = newNumIndices;
    result.numFaces   = mesh.numFaces * 2;
    result.faceIDs    = faceIDs;
    return result;
}

void ClusterBuilder::CreateDGFs(ScenePrimitives *scene, DenseGeometryBuildData *buildData,
                                Mesh *meshes, int numMeshes, Bounds &sceneBounds)
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
    StaticArray<StaticArray<Vec3i>> quantizedVertices(meshScratch.temp.arena, numMeshes);
    StaticArray<StaticArray<u32>> octNormals(meshScratch.temp.arena, numMeshes);

    u32 indexTotal = 0;

    for (int i = 0; i < numMeshes; i++)
    {
        Mesh &mesh = meshes[i];
        quantizedVertices.Push(StaticArray<Vec3i>(meshScratch.temp.arena, mesh.numVertices));
        if (mesh.n)
        {
            octNormals.Push(StaticArray<u32>(meshScratch.temp.arena, mesh.numVertices));
        }
        indexTotal += mesh.numIndices;

        // Quantize to global grid
        for (int j = 0; j < mesh.numVertices; j++)
        {
            quantizedVertices[i].Push(Vec3i(Round(mesh.p[j] * quantize)));

            Vec3f n = Normalize(mesh.n[j]);
            octNormals[i].Push(EncodeOctahedral(n));
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
                CreateDGFs(scene, buildData, meshScratch.temp.arena, meshes, quantizedVertices,
                           octNormals, cluster, precision);
            }
        }
    }
    buildData->numBlocks = total;
}

void ClusterBuilder::CreateDGFs(ScenePrimitives *scene, DenseGeometryBuildData *buildDatas,
                                Arena *arena, Mesh *meshes,
                                const StaticArray<StaticArray<Vec3i>> &quantizedVertices,
                                const StaticArray<StaticArray<u32>> &octNormals,
                                RecordAOSSplits &cluster, int precision)
{
    Scene *rootScene       = GetScene();
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
    StaticArray<u32> vertexIndices(clusterScratch.arena, clusterNumTriangles * 3,
                                   clusterNumTriangles * 3);
    StaticArray<u32> primIDs(clusterScratch.arena, clusterNumTriangles, clusterNumTriangles);
    StaticArray<u32> geomIDs(clusterScratch.arena, clusterNumTriangles, clusterNumTriangles);
    StaticArray<u32> clusterVertexIndexToMeshVertexIndex(
        clusterScratch.arena, clusterNumTriangles * 3, clusterNumTriangles * 3);

    StaticArray<Vec3i> vertices(clusterScratch.arena, clusterNumTriangles * 3,
                                clusterNumTriangles * 3);
    StaticArray<u32> normals(clusterScratch.arena, clusterNumTriangles * 3,
                             clusterNumTriangles * 3);

    StaticArray<u32> faceIDs(clusterScratch.arena, clusterNumTriangles, clusterNumTriangles);

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

    Vec2u minOct(pos_inf);
    Vec2u maxOct(0);

    u32 geomIDStart = primRefs[start].geomID;

    u32 materialIDStart     = scene->primIndices[geomIDStart].materialID.GetIndex();
    bool constantMaterialID = true;

    u32 minFaceID   = pos_inf;
    u32 maxFaceID   = 0;
    bool hasFaceIDs = false;

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
        primIDs[i - start] = primID;

        u32 faceID         = mesh.faceIDs ? mesh.faceIDs[primID] : 0u;
        minFaceID          = Min(minFaceID, faceID);
        maxFaceID          = Max(maxFaceID, faceID);
        faceIDs[i - start] = faceID;

        u32 materialID = scene->primIndices[geomID].materialID.GetIndex();
        constantMaterialID &= materialID == materialIDStart;
        Material *material = rootScene->materials[materialID];

        hasFaceIDs |= (bool(mesh.faceIDs) && material->ptexReflectanceIndex != -1);

        for (int indexIndex = 0; indexIndex < 3; indexIndex++)
        {
            min = Min(min, quantizedVertices[geomID][indices[indexIndex]]);
            max = Max(max, quantizedVertices[geomID][indices[indexIndex]]);

            u32 normal = octNormals[geomID][indices[indexIndex]];
            Vec2u n    = Vec2u(normal >> 16, normal & ((1 << 16) - 1));
            minOct     = Min(minOct, n);
            maxOct     = Max(maxOct, n);

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
                normals[vertexCount]  = octNormals[geomID][indices[indexIndex]];
                vertexCount++;
            }
        }
    }

    Assert(vertexCount <= MAX_CLUSTER_VERTICES);

    u32 numBitsX = min[0] == max[0] ? 0 : Log2Int(Max(max[0] - min[0], 1)) + 1;
    u32 numBitsY = min[1] == max[1] ? 0 : Log2Int(Max(max[1] - min[1], 1)) + 1;
    u32 numBitsZ = min[2] == max[2] ? 0 : Log2Int(Max(max[2] - min[2], 1)) + 1;

    u32 numOctBitsX = minOct[0] == maxOct[0] ? 0 : Log2Int(Max(maxOct[0] - minOct[0], 1u)) + 1;
    u32 numOctBitsY = minOct[1] == maxOct[1] ? 0 : Log2Int(Max(maxOct[1] - minOct[1], 1u)) + 1;

    Assert(!hasFaceIDs || minFaceID != maxFaceID);
    u32 numFaceBits = hasFaceIDs ? Log2Int(Max(maxFaceID - minFaceID, 1u)) + 1 : 0;

    Assert(numOctBitsX <= 16 && numOctBitsY <= 16);
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
    // i need:
    // restart high bit per dword
    // edge high bit for BOTH edges accounting for backtracks
    // restart counts per dword

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
    StaticArray<u32> oldIndexOrder(clusterScratch.arena, clusterNumTriangles * 3);
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

    oldIndexOrder.Push(firstIndices[0]);
    oldIndexOrder.Push(firstIndices[1]);
    oldIndexOrder.Push(firstIndices[2]);

    triangleOrder.Push(minValenceTriangle);

    MarkUsed(minValenceTriangle);

    // TODO: half edge structure instead of hash table?
    // Find the new ordering of triangles and construct the triangle strip
    for (u32 numAddedTriangles = 1; numAddedTriangles < clusterNumTriangles;
         numAddedTriangles++)
    {
        u32 dwordIndex = numAddedTriangles >> 5;
        u32 bitIndex   = numAddedTriangles & 31u;

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

                oldIndexOrder.Push(indices[0]);
                oldIndexOrder.Push(indices[1]);
                oldIndexOrder.Push(indices[2]);

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

                    oldIndexOrder.Push(indices[0]);
                    oldIndexOrder.Push(indices[1]);
                    oldIndexOrder.Push(indices[2]);
                }
                // [0, 1, 2] -> [2, 0, 1]
                else if (edgeIndex == 2)
                {
                    vertexIndices[3 * newMinValenceTriangle]     = indices[2];
                    vertexIndices[3 * newMinValenceTriangle + 1] = indices[0];
                    vertexIndices[3 * newMinValenceTriangle + 2] = indices[1];
                    newIndexOrder.Push(indices[1]);

                    oldIndexOrder.Push(indices[0]);
                    oldIndexOrder.Push(indices[1]);
                    oldIndexOrder.Push(indices[2]);
                }
                else
                {
                    newIndexOrder.Push(indices[2]);

                    oldIndexOrder.Push(indices[0]);
                    oldIndexOrder.Push(indices[1]);
                    oldIndexOrder.Push(indices[2]);
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

    // 7 * 3 * 4 = 84 = 3 dwords
    FixedArray<u32, 4> restartHighBitPerDword = {0, 0, 0, 0};
    FixedArray<u32, 4> restartCountPerDword   = {1, 0, 0, 0};
    FixedArray<i32, 4> edge1HighBitPerDword   = {-1, -1, -1, -1};
    FixedArray<i32, 4> edge2HighBitPerDword   = {-1, -1, -1, -1};

    u32 prevEdge1HighBitPerDword = 0;
    u32 prevEdge2HighBitPerDword = 0;

    u32 bitVectorsSize = ((clusterNumTriangles >> 5u) + 1u) << 5u;
    BitVector backtrack(clusterScratch.arena, clusterNumTriangles);
    BitVector restart(clusterScratch.arena, clusterNumTriangles);
    BitVector edge(clusterScratch.arena, clusterNumTriangles);

    restart.SetBit(0);

    // Set bit masks
    for (u32 i = 1; i < clusterNumTriangles; i++)
    {
        u32 dwordIndex = i >> 5;
        u32 bitIndex   = i & 31u;

        TriangleStripType stripType = triangleStripTypes[i - 1];
        if (stripType == TriangleStripType::Backtrack)
        {
            Assert(i > 1);
            TriangleStripType prevType = triangleStripTypes[i - 2];
            Assert(prevType == TriangleStripType::Edge1 ||
                   prevType == TriangleStripType::Edge2);
            backtrack.SetBit(i);
            u32 prevDwordIndex = (i - 1) >> 5;
            if (prevType == TriangleStripType::Edge2)
            {
                Assert(edge2HighBitPerDword[prevDwordIndex] == (int)i - 1);
                edge1HighBitPerDword[prevDwordIndex] = i - 1;
                edge2HighBitPerDword[prevDwordIndex] = prevEdge2HighBitPerDword;
                edge.SetBit(i);
            }
            else if (prevType == TriangleStripType::Edge1)
            {
                Assert(edge1HighBitPerDword[prevDwordIndex] == (int)i - 1);
                edge2HighBitPerDword[prevDwordIndex] = i - 1;
                edge1HighBitPerDword[prevDwordIndex] = prevEdge1HighBitPerDword;
            }
        }
        else if (stripType == TriangleStripType::Edge1)
        {
            edge.SetBit(i);
            prevEdge1HighBitPerDword         = edge1HighBitPerDword[dwordIndex];
            edge1HighBitPerDword[dwordIndex] = i;
        }
        else if (stripType == TriangleStripType::Edge2)
        {
            prevEdge2HighBitPerDword         = edge2HighBitPerDword[dwordIndex];
            edge2HighBitPerDword[dwordIndex] = i;
        }
        else if (stripType == TriangleStripType::Restart)
        {
            restart.SetBit(i);
            restartHighBitPerDword[dwordIndex] = i;
            restartCountPerDword[dwordIndex]++;
        }
    }

    for (u32 i = 1; i < 4; i++)
    {
        restartCountPerDword[i] += restartCountPerDword[i - 1];
        restartHighBitPerDword[i] =
            Max(restartHighBitPerDword[i], restartHighBitPerDword[i - 1]);
        edge1HighBitPerDword[i] = Max(edge1HighBitPerDword[i], edge1HighBitPerDword[i - 1]);
        edge2HighBitPerDword[i] = Max(edge2HighBitPerDword[i], edge2HighBitPerDword[i - 1]);
    }

    // Append all data
    auto &buildData = buildDatas[0];

    u32 vertexBitStreamSize = (numBits + 7) >> 3;
    u32 normalBitStreamSize = (vertexCount * (numOctBitsX + numOctBitsY) + 7) >> 3;
    u32 faceBitStreamSize   = 4 + (((numFaceBits + 3) * clusterNumTriangles + 7) >> 3);
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

    u32 bitOffset       = 0;
    u32 normalBitOffset = 0;
    u32 faceBitOffset   = 0;

    u32 totalSize                                = 0;
    ChunkedLinkedList<u8>::ChunkNode *vertexNode = 0;
    if (vertexBitStreamSize)
    {
        totalSize += vertexBitStreamSize;
        vertexNode = buildData.byteBuffer.AddNode(vertexBitStreamSize);
    }
    ChunkedLinkedList<u8>::ChunkNode *normalNode = 0;
    if (normalBitStreamSize)
    {
        totalSize += normalBitStreamSize;
        normalNode = buildData.byteBuffer.AddNode(normalBitStreamSize);
    }
    ChunkedLinkedList<u8>::ChunkNode *faceIDNode = 0;
    if (hasFaceIDs)
    {
        totalSize += faceBitStreamSize;
        faceIDNode = buildData.byteBuffer.AddNode(faceBitStreamSize);
    }

    StaticArray<u32> newNormals(clusterScratch.arena, vertexCount);

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

            if (vertexNode)
            {
                for (int i = 0; i < 3; i++)
                {
                    int p = vertices[index][i] - min[i];
                    Assert(p >= 0);
                    WriteBits((u32 *)vertexNode->values, bitOffset, p, bitWidths[i]);
                }
            }
            u32 normal  = normals[index];
            u32 normalX = (normal >> 16) - minOct[0];
            u32 normalY = (normal & ((1 << 16) - 1)) - minOct[1];

            newNormals.Push(normal);
            if (normalNode)
            {
                WriteBits((u32 *)normalNode->values, normalBitOffset, normalX, numOctBitsX);
                WriteBits((u32 *)normalNode->values, normalBitOffset, normalY, numOctBitsY);
            }
        }
    }
    if (hasFaceIDs)
    {
        WriteBits((u32 *)faceIDNode->values, faceBitOffset, minFaceID, 32);
        for (u32 i = 0; i < triangleOrder.Length(); i++)
        {
            u32 index = triangleOrder[i];
            WriteBits((u32 *)faceIDNode->values, faceBitOffset, faceIDs[i] - minFaceID,
                      numFaceBits);

            // Write 2 bits to denote triangle rotation
            u32 indices[3];
            GetIndices(indices, index);
            u32 oldStartIndex = oldIndexOrder[3 * i];
            u32 rotate =
                oldStartIndex == indices[0] ? 0 : (oldStartIndex == indices[1] ? 2 : 1);

            WriteBits((u32 *)faceIDNode->values, faceBitOffset, rotate, 2);
            WriteBits((u32 *)faceIDNode->values, faceBitOffset, primIDs[index] & 1, 1);
        }
        Assert(faceBitOffset == 32 + (numFaceBits + 3) * clusterNumTriangles);
    }

    for (u32 i = 0; i < oldIndexOrder.Length(); i++)
    {
        oldIndexOrder[i] = mapOldIndexToDGFIndex[oldIndexOrder[i]];
    }

    Assert(bitOffset == numBits);
    Assert(normalBitOffset == vertexCount * (numOctBitsX + numOctBitsY));
    bitOffset = 0;

    // Write reuse buffer (byte aligned)
    u32 numIndexBits = Log2Int(Max(maxReuseIndex, 1u)) + 1;
    Assert(numIndexBits >= 1 && numIndexBits <= 8);
    u32 ctrlBitSize = ((clusterNumTriangles + 31) >> 5u) * 12u;
    u32 bitStreamSize =
        ((numIndexBits * reuseBuffer.Length() + currentFirstUseBit + 7) >> 3) + ctrlBitSize;
    u32 numBitStreamBits =
        numIndexBits * reuseBuffer.Length() + (ctrlBitSize << 3) + currentFirstUseBit;
    auto *node = buildData.byteBuffer.AddNode(bitStreamSize);
    totalSize += bitStreamSize;

    // Write control bits
    int numRemainingTriangles = clusterNumTriangles;
    u32 dwordIndex            = 0;
    while (numRemainingTriangles > 0)
    {
        WriteBits((u32 *)node->values, bitOffset, restart.bits[dwordIndex], 32);
        WriteBits((u32 *)node->values, bitOffset, edge.bits[dwordIndex], 32);
        WriteBits((u32 *)node->values, bitOffset, backtrack.bits[dwordIndex], 32);
        dwordIndex++;
        numRemainingTriangles -= 32;
    }

    for (u32 reuseIndex : reuseBuffer)
    {
        WriteBits((u32 *)node->values, bitOffset, reuseIndex, numIndexBits);
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

    Assert(bitOffset == numBitStreamBits);

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
    packed.e = BitFieldPackU32(packed.e, numOctBitsX, headerOffset, 5);

    packed.f = BitFieldPackU32(packed.f, restartHighBitPerDword[1], headerOffset, 6);
    packed.f = BitFieldPackU32(packed.f, restartHighBitPerDword[2], headerOffset, 7);
    packed.f = BitFieldPackI32(packed.f, edge1HighBitPerDword[0], headerOffset, 6);
    packed.f = BitFieldPackI32(packed.f, edge1HighBitPerDword[1], headerOffset, 7);
    packed.f = BitFieldPackI32(packed.f, edge2HighBitPerDword[0], headerOffset, 6);

    packed.g = BitFieldPackU32(packed.g, restartHighBitPerDword[0], headerOffset, 5);
    packed.g = BitFieldPackI32(packed.g, edge1HighBitPerDword[2], headerOffset, 8);
    packed.g = BitFieldPackI32(packed.g, edge2HighBitPerDword[1], headerOffset, 7);
    packed.g = BitFieldPackI32(packed.g, edge2HighBitPerDword[2], headerOffset, 8);
    // packed.g = BitFieldPackU32(packed.g, numFaceSizeBits - 1, headerOffset, 4);

    packed.h = BitFieldPackU32(packed.h, numOctBitsY, headerOffset, 5);
    packed.h = BitFieldPackU32(packed.h, restartCountPerDword[0], headerOffset, 6);
    packed.h = BitFieldPackU32(packed.h, restartCountPerDword[1], headerOffset, 7);
    packed.h = BitFieldPackU32(packed.h, restartCountPerDword[2], headerOffset, 8);
    packed.h = BitFieldPackU32(packed.h, numFaceBits, headerOffset, 6);
    // packed.h = BitFieldPackU32(packed.h, hasFaceIDs ? numPageOffsetBits : 0, headerOffset,
    // 6);

    headerOffset = 0;
    packed.i     = BitFieldPackU32(packed.i, minOct[0], headerOffset, 16);
    packed.i     = BitFieldPackU32(packed.i, minOct[1], headerOffset, 16);

    // Constant mode
    // TODO: compress this
    if (constantMaterialID)
    {
        u32 materialIndex = scene->primIndices[geomIDStart].materialID.GetIndex();
        materialIndex |= 0x80000000;
        packed.j = materialIndex;
    }
    else
    {
        u32 commonMSBs        = 32;
        u32 currentCommonMSBs = materialIDStart;

        u32 numUniqueMaterialIDs = 1;
        StaticArray<u32> uniqueMaterialIDs(clusterScratch.arena, clusterNumTriangles);
        StaticArray<u32> entryIndex(clusterScratch.arena, clusterNumTriangles,
                                    clusterNumTriangles);

        PrimitiveIndices *primIndices = scene->primIndices;
        uniqueMaterialIDs.Push(primIndices[geomIDs[triangleOrder[0]]].materialID.GetIndex());
        entryIndex[0] = 0;
        for (int i = 1; i < clusterNumTriangles; i++)
        {
            u32 triangleIndex = triangleOrder[i];
            u32 materialID    = primIndices[geomIDs[triangleIndex]].materialID.GetIndex();
            bool unique       = true;
            for (int j = 0; j < uniqueMaterialIDs.Length(); j++)
            {
                if (uniqueMaterialIDs[j] == materialID)
                {
                    entryIndex[i] = j;
                    unique        = false;
                    break;
                }
            }
            if (unique)
            {
                for (int bitIndex = commonMSBs; bitIndex >= 0; bitIndex--)
                {
                    if ((materialID >> (32 - bitIndex)) == currentCommonMSBs)
                    {
                        break;
                    }
                    currentCommonMSBs >>= 1;
                    commonMSBs--;
                }
                uniqueMaterialIDs.Push(materialID);
                entryIndex[i] = uniqueMaterialIDs.Length() - 1;
            }
        }

        Assert(uniqueMaterialIDs.Length() > 1);
        u32 entryBitLength      = Log2Int(uniqueMaterialIDs.Length() - 1) + 1;
        u32 materialTableOffset = totalSize;
        Assert(materialTableOffset < 0x80000000);

        // Common MSBs + LSB entries + per triangle entries
        u32 materialTableNumBits = commonMSBs +
                                   (32 - commonMSBs) * uniqueMaterialIDs.Length() +
                                   entryBitLength * clusterNumTriangles;
        u32 materialTableSize = (materialTableNumBits + 7) >> 3;
        auto *table           = buildData.byteBuffer.AddNode(materialTableSize);
        u32 tableBitOffset    = 0;
        WriteBits((u32 *)table->values, tableBitOffset, currentCommonMSBs, commonMSBs);
        for (u32 materialID : uniqueMaterialIDs)
        {
            u32 lsbMaterialID = materialID & ((1 << (32 - commonMSBs)) - 1u);
            WriteBits((u32 *)table->values, tableBitOffset, lsbMaterialID, (32 - commonMSBs));
        }
        for (u32 index : entryIndex)
        {
            WriteBits((u32 *)table->values, tableBitOffset, index, entryBitLength);
        }
        Assert(tableBitOffset == materialTableNumBits);

        Assert(materialTableOffset < (1 << 22));
        Assert(commonMSBs > 0);
        headerOffset = 0;
        packed.j     = BitFieldPackU32(packed.j, commonMSBs - 1, headerOffset, 5);
        packed.j     = BitFieldPackU32(packed.j, uniqueMaterialIDs.Length(), headerOffset, 5);
        packed.j     = BitFieldPackU32(packed.j, materialTableOffset, headerOffset, 22);
    }

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
#if 0

    u32 numu32s        = (currentFirstUseBit + 31) >> 5;
    auto *firstUseNode = buildData.firstUse.AddNode(numu32s);
    MemoryCopy(firstUseNode->values, firstUse.bits, (currentFirstUseBit + 7) >> 3);

    auto *reuseNode = buildData.reuse.AddNode(reuseBuffer.Length());
    MemoryCopy(reuseNode->values, reuseBuffer.data, reuseBuffer.Length() * sizeof(u32));
#endif

    u32 bitSize      = ((clusterNumTriangles + 31) >> 5) * sizeof(u32);
    auto *debug0Node = buildData.debugRestartCountPerDword.AddNode(vertexCount);
    MemoryCopy(debug0Node->values, newNormals.data, vertexCount * sizeof(u32));
    auto *debug1Node = buildData.debugRestartHighBitPerDword.AddNode(4);
    MemoryCopy(debug1Node->values, backtrack.bits, bitSize);

    auto *debugIndexNode = buildData.debugIndices.AddNode(oldIndexOrder.Length());
    MemoryCopy(debugIndexNode->values, oldIndexOrder.data,
               oldIndexOrder.Length() * sizeof(u32));
    ScratchEnd(clusterScratch);
}

struct PtexInfo
{
    string filename;
    TileMetadata *metaData;
    int numFaces;
};

} // namespace rt
