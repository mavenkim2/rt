#ifndef DGFS_H_
#define DGFS_H_
#include "math/simd_base.h"
#include "scene.h"
#include "bvh/bvh_types.h"

namespace rt
{

static const int maxClusterTriangles = 64;

struct DenseGeometry
{
    Vec3u anchor;
    u32 primitiveID;
    u8 *perVertexOffsets;
    u8 exponents[3];
};

//
// Vec3i QuantizeToWorldGrid(const Vec3f &pos, const Vec3f &modelExtent, const Vec3f &worldMin,
//                           int quantizationRate)
// {
//     f32 scale = (Pow(2, 24) - 1) / modelExtent;
//     return Vec3i((pos - worldMin + 0.5f) * scale);
// }

struct Cluster
{
    RecordAOSSplits record;
};

void DecodeDenseGeometry()
{
    // PopCount(something?) = numRestarts;
    // 2 * numRestarts;
    u32 ctrl;

    // if current ctrl is edge1, prev[2] & prev[1], 2r + index
    // prev[2] is always equal to 2r + k

    // shift it down so it hits the previous guy

    // 1 = edge1, 2 = edge2, 0 = ?
    // backtrack = 3

    // edgeMask = 0 if edge1, 1 if edge2
    int edge = edgeMask;

    // key insight is that 01 xor 11 = 10 and 10 xor 11 = 01, which swaps b/t edge1 and edge2
    // somehow get this mask, which is two bits set at positions 2i, 2i + 1,
    // if backtrack, otherwise 0
    int back = (mask & ~(backtrack << 2)) ^ (backtrack << 4);
    // even bits get edge2, odd bits get edge1

    u32 indices[3] = {0, 0, 3 + triIndex};
    indices[edge2] = 3 + triIndex - 1;

    u32 clearTopBits = (1 << triIndex) - 1;
    // find oldK, aka the last time the opposite of the current edge happened, -1
    u32 oldK        = Bsr(mask & clearTopBits);
    indices[!edge2] = 3 + (oldK - 1);

    // the problem is that with restart, you would essentially have to know
    // how many restarts occurred up to the point you care about
    // solution: just don't have restarts?
}

template <typename PrimRef>
void CreateDGFs(Cluster *clusters, int num, PrimRef *primRefs, Mesh &mesh, Bounds &sceneBounds,
                const Bounds &meshBounds, const Vec3f &maxClusterEdgeLength)
{
    static const int b     = 16;
    static const f32 scale = 1.f / ((1 << (b - 1)) - 1.f);
    Vec3f sceneExtent      = ToVec3f(sceneBounds.maxP - sceneBounds.minP);

    f32 eX = sceneExtent[0] * scale));
    f32 eY = sceneExtent[1] * scale));
    f32 eZ = sceneExtent[2] * scale));

    f32 offsetMinX = maxClusterEdgeLength[0] - 16.f;
    f32 offsetMinY = maxClusterEdgeLength[1] - 16.f;
    f32 offsetMinZ = maxClusterEdgeLength[2] - 16.f;

    static const int c = (1 << 23) - 1;
    f32 constraintMinX = Max(-meshBounds.minP[0], 0.f) / c;
    f32 constraintMinY = Max(-meshBounds.minP[1], 0.f) / c;
    f32 constraintMinZ = Max(-meshBounds.minP[2], 0.f) / c;
    f32 constraintMaxX = Max(meshBounds.maxP[0], 0.f) / (c + 1);
    f32 constraintMaxY = Max(meshBounds.maxP[1], 0.f) / (c + 1);
    f32 constraintMaxZ = Max(meshBounds.maxP[2], 0.f) / (c + 1);

    eX = Max(constraintMinX, Max(constraintMaxX, Max(offsetMinX, eX)));
    eY = Max(constraintMinY, Max(constraintMaxY, Max(offsetMinY, eY)));
    eZ = Max(constraintMinZ, Max(constraintMaxY, Max(offsetMinZ, eZ)));

    ScratchArena scratch;

    for (int clusterIndex = 0; clusterIndex < num; clusterIndex++)
    {
        Cluster &cluster = clusters[clusterIndex];

        u32 primID = primRefs[cluster.record.start].primID;
        u32 triangles[maxClusterTriangles];
        triangles[0] = primID;

        int start             = cluster.record.start;
        int clusterCount      = cluster.record.count;
        int clusterNumIndices = clusterCount * 3;

        u64 vertexBitVector = 0;

        // remap
        u32 vertexCount = 0;
        StaticArray<Vec3f> clusterVertices(scratch.temp.arena, maxClusterTriangles);
        u32 allIndices[clusterCount * 3];

        u32 remapPrimRefIDToClusterVertexIndex[][3];
        for (int i = start; i < start + clusterCount; i++)
        {
            u32 primID = primRefs[i].primID;
            // TODO: quad vs triangle?
            u32 indices[3] = {mesh.indices[3 * primID + 0], mesh.indices[3 * primID + 1],
                              mesh.indices[3 * primID + 2]};

            u32 newVertexCount = vertexCount;
            for (int j = 0; j < vertexCount; j++)
            {
                for (int indexIndex = 0; indexIndex < 3; indexIndex++)
                {
                    if (indices[indexIndex] == allIndices[j])
                    {
                        // the problem with this is that it requires duplicating vertices
                        remapPrimRefIDToClusterVertexIndex[3 * i + indexIndex] = j;
                    }
                }
            }
            allIndices[vertexCount++] = ;
        }

        // Build adjacency data for triangles
        u32 *counts  = PushArray(scratch.temp.arena, u32, clusterNumTriangles);
        u32 *offsets = PushArray(scratch.temp.arena, u32, clusterNumTriangles);
        u32 *data    = PushArray(scratch.temp.arena, u32, clusterNumTriangles);

        struct EdgeKeyValue
        {
            u64 key;
            u32 count;
            u32 value0;
            u32 value1;
            u32 Hash() const { return MixBits(key); }

            bool operator==(const EdgeKeyValue &other) const { return key == other.key; }
        };
        HashMap<EdgeKeyValue> edgeIDMap(scratch.temp.arena, NextPowerOfTwo(mesh.numIndices));

        // Find triangles with shared edge
        for (u32 triangleIndex = 0; triangleIndex < clusterNumTriangles; triangleIndex++)
        {
            u32 indices[3] = {vertexIndices[3 * clusterIndex],
                              vertexIndices[3 * clusterIndex + 1],
                              vertexIndices[3 * clusterIndex + 2]};

            for (int edgeIndex = 0; edgeIndex < 3; edgeIndex++)
            {
                u64 edgeID        = ComputeEdgeId(indices[0], indices[1]);
                auto edgeKeyValue = edgeIDMap.Find(edgeID);
                if (edgeKeyValue == 0)
                {
                    edgeIDMap.Add(scratch.temp.arena, {edgeID, 1, triangleIndex, ~0u});
                }
                else
                {
                    // TODO: handle non manifolds
                    if (edgeKeyValue.value1 == ~0u) Assert(0);
                    edgeKeyValue.value1 = triangleIndex;
                    edgeKeyValue.count  = 2;
                }
            }
        }

        // Find number of adjacent triangles for each triangle
        for (u32 triangleIndex = 0; triangleIndex < clusterNumTriangles; triangleIndex++)
        {
            u32 indices[3] = {vertexIndices[3 * triangleIndex],
                              vertexIndices[3 * triangleIndex + 1],
                              vertexIndices[3 * triangleIndex + 2]};
            u32 count      = 0;

            for (int edgeIndex = 0; edgeIndex < 3; edgeIndex++)
            {
                u64 edgeID        = ComputeEdgeId(indices[0], indices[1]);
                auto edgeKeyValue = edgeIDMap.Find(edgeID);
                Assert(edgeKeyValue);
                count += edgeKeyValue.count - 1;
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
            u32 indices[3] = {vertexIndices[3 * triangleIndex],
                              vertexIndices[3 * triangleIndex + 1],
                              vertexIndices[3 * triangleIndex + 2]};
            u32 count      = 0;

            for (int edgeIndex = 0; edgeIndex < 3; edgeIndex++)
            {
                u64 edgeID = ComputeEdgeId(indices[edgeIndex], indices[(edgeIndex + 1) & 3]);
                auto edgeKeyValue = edgeIDMap.Find(edgeID);
                Assert(edgeKeyValue);
                data[offsets[triangleIndex]++] = triangleIndex == edgeKeyValue.value0
                                                     ? edgeKeyValue.value1
                                                     : edgeKeyValue.value0;
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

        u64 firstUse           = 0;
        u32 currentFirstUseBit = 0;
        u64 usedTriangles      = 0;
        u32 numReuse           = 0;

        u32 prevTriangle = minValenceTriangle;

        auto GetIndices = [&vertexIndices](u32 ind[3], u32 triangleIndex) {
            ind[0] = vertexIndices[3 * triangleIndex + 0];
            ind[1] = vertexIndices[3 * triangleIndex + 1];
            ind[2] = vertexIndices[3 * triangleIndex + 2];
        };

        auto CheckUsedTriangles = [&](u32 index) {
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
            mapOldIndexToDGFIndex[index] = newVertices.size();
            newVertices.push_back(vertices[index]);
            return false;
        };

        for (u32 numAddedTriangles = 1; numAddedTriangles < clusterNumTriangles;
             numAddedTriangles++)
        {
            minCount                  = pos_inf;
            u32 newMinValenceTriangle = ~0u;

            TriangleStripType backTrackStripType = TriangleStripType::None;
            // If triangle has no neighbors, attempt to backtrack
            if (!counts[minValenceTriangle])
            {
                // If can't backtrack, restart
                if (numAddedTriangles == 1 ||
                    triangleStripTypes[numAddedTriangles - 1] == TriangleStripType::Restart ||
                    triangleStripTypes[numAddedTriangles - 1] ==
                        TriangleStripType::Backtrack ||
                    !counts[prevTriangle])
                {
                    triangleStripTypes.push_back(TriangleStripType::Restart);

                    prevTriangle       = minValenceTriangle;
                    minValenceTriangle = FindMinValence();
                    u32 indices[3];
                    GetIndices(indices, minValenceTriangle);

                    CheckUsedTriangles(indices[0]);
                    CheckUsedTriangles(indices[1]);
                    CheckUsedTriangles(indices[2]);
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
            for (int edgeIndex = 0; edgeIndex < 3; edgeIndex++)
            {
                u64 edgeID = ComputeEdgeId(indices[edgeIndex], indices[(edgeIndex + 1) & 3]);
                auto edgeKeyValue = edgeIDMap.Find(edgeID);
                Assert(edgeKeyValue);

                if (edgeKeyValue.value0 == minValenceTriangle ||
                    edgeKeyValue.value1 == minValenceTriangle)
                {
                    // Rotate
                    // [0, 1, 2] -> [1, 2, 0]
                    if (edgeIndex == 1)
                    {
                        vertexIndices[3 * newMinValenceTriangle]     = indices[1];
                        vertexIndices[3 * newMinValenceTriangle + 1] = indices[2];
                        vertexIndices[3 * newMinValenceTriangle + 2] = indices[0];

                        MemoryCopy(indices, vertexIndices + 3 * newMinValenceTriangle,
                                   sizeof(indices));
                        CheckUsedTriangles(indices[0]);
                    }
                    // [0, 1, 2] -> [2, 0, 1]
                    else if (edgeIndex == 2)
                    {
                        vertexIndices[3 * newMinValenceTriangle]     = indices[2];
                        vertexIndices[3 * newMinValenceTriangle + 1] = indices[0];
                        vertexIndices[3 * newMinValenceTriangle + 2] = indices[1];

                        MemoryCopy(indices, vertexIndices + 3 * newMinValenceTriangle,
                                   sizeof(indices));
                        CheckUsedTriangles(indices[1]);
                    }
                    else
                    {
                        CheckUsedTriangles(indices[2]);
                    }

                    u32 oldIndices[3];
                    GetIndices(oldIndices, minValenceTriangle);
                    // Find whether we're attached to edge1 or 2 from previous triangle
                    for (int oldEdgeIndex = 0; oldEdgeindex < 3; oldEdgeIndex++)
                    {
                        u64 oldEdgeId = ComputeEdgeId(oldIndices[oldEdgeIndex],
                                                      oldIndices[(oldEdgeIndex + 1) & 3]);
                        if (oldEdgeId == edgeId)
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
    }

    Vec3f worldMin;
    ScratchArena scratch;

    Vec3i min(pos_inf);
    Vec3i max(pos_inf);
    Vec3i *quantizedVertices = PushArrayNoZero(scratch.temp.arena, Vec3i, mesh.numVertices);

    // Quantize to global grid
    for (int i = 0; i < mesh.numVertices; i++)
    {
        quantizedVertices[i] = QuantizeToWorldGrid(mesh.p[i], worldMin);
        min                  = Min(min, quantizedVertices[i]);
        max                  = Max(max, quantizedVertices[i]);
    }

    // Calculate the number of bits needed to represent the range of values in the cluster
    int numBitsX = Log2Int(Max(max[0] - min[0] - 1, 1)) + 1;
    int numBitsY = Log2Int(Max(max[1] - min[1] - 1, 1)) + 1;
    int numBitsZ = Log2Int(Max(max[1] - min[1] - 1, 1)) + 1;

    int numBytes = (numClusterVertices * (numBitsX + numBitsY + numBitsZ) + 7) >> 3;

    int numClusterVertices;
    u8 *bitStream = PushArrayNoZero(arena, u8, numBytes);

    for (int i = 0; i < cluster.numVertices; i++)
    {
        quantizedVertices[i] -= min;
    }

    // Pack into blocks
    // need valences per vertex and neighbors per vertex

    // chose min valence, remove node, subtract valences of neighbors, choose
    // neighbor w/ min valence and repeat
    int numNeighbors = PushArrayNoZero(temp.arena, int, cluster.numVertices);
    for (int i = 0; i < cluster.numIndices; i++)
    {
    }
}
} // namespace rt
#endif
