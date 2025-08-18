#include "base.h"
#include "bit_packing.h"
#include "containers.h"
#include "math/basemath.h"
#include "bvh/bvh_aos.h"
#include "bvh/bvh_types.h"
#include "math/math.h"
#include "math/vec2.h"
#include "shader_interop/dense_geometry_shaderinterop.h"
#include "shader_interop/ray_shaderinterop.h"
#include "parallel.h"
#include "mesh.h"
#include "dgfs.h"
#include "memory.h"
#include "thread_context.h"
#include "platform.h"
#include <type_traits>

namespace rt
{

namespace ClusterBuilder
{

typedef HeuristicObjectBinning<PrimRef> Heuristic;
static void BuildClusters(StaticArray<RecordAOSSplits> &records, RecordAOSSplits &record,
                          Heuristic *heuristic, u32 maxTriangles)
{
    const int N = 4;
    Assert(record.count > 0);

    RecordAOSSplits childRecords[N];
    u32 numChildren = 0;

    Split split = heuristic->Bin(record);

    if (record.count <= maxTriangles)
    {
        records.Push(record);
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
            if (childRecord.count <= maxTriangles) continue;

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

    for (u32 i = 0; i < numChildren; i++)
    {
        BuildClusters(records, childRecords[i], heuristic, maxTriangles);
    }

    // if (parallel)
    // {
    //     scheduler.ScheduleAndWait(numChildren, 1, [&](u32 jobID) {
    //         bool childParallel = childRecords[jobID].count >= PARALLEL_THRESHOLD;
    //         BuildClusters(records, childRecords[jobID], heuristic, childParallel,
    //         maxTriangles);
    //     });
    // }
    // else
    // {
    // }
}

StaticArray<RecordAOSSplits> BuildClusters(Arena *arena, PrimRef *primRefs,
                                           RecordAOSSplits &record, u32 maxTriangles,
                                           u32 maxClusters)
{
    StaticArray<RecordAOSSplits> records(arena, maxClusters);
    Heuristic *h = PushStructConstruct(arena, Heuristic)(primRefs);

    BuildClusters(records, record, h, maxTriangles);
    return records;
}

} // namespace ClusterBuilder

DenseGeometryBuildData::DenseGeometryBuildData(Arena *arena) : arena(arena)
{
    geoByteBuffer     = ChunkedLinkedList<u8>(arena, 1024);
    shadingByteBuffer = ChunkedLinkedList<u8>(arena, 1024);
    headers           = ChunkedLinkedList<PackedDenseGeometryHeader>(arena);

    // Debug
#if 0
    firstUse     = ChunkedLinkedList<u32>(arena);
    reuse        = ChunkedLinkedList<u32>(arena);
#endif
    types                       = ChunkedLinkedList<TriangleStripType>(arena);
    debugFaceIDs                = ChunkedLinkedList<u32>(arena);
    debugIndices                = ChunkedLinkedList<u32>(arena);
    debugRestartCountPerDword   = ChunkedLinkedList<u32>(arena);
    debugRestartHighBitPerDword = ChunkedLinkedList<u32>(arena);
}

DGFTempResources::DGFTempResources()
    : hasAnyNormals(false), vertexCount(0), minP(pos_inf), maxP(neg_inf), minOct(Vec2u(65536)),
      maxOct(0), precision(6)
{
    Assert(precision > CLUSTER_MIN_PRECISION);
    quantizeP = AsFloat((127 + precision) << 23);
}

void DGFTempResources::UpdatePosBounds(const Vec3f &pos)
{
    Vec3i p(Round(pos * quantizeP));
    Assert(p.x != INT_MIN && p.y != INT_MIN && p.z != INT_MIN);

    minP = Min(minP, p);
    maxP = Max(maxP, p);
}

void DGFTempResources::UpdateNormalBounds(const Vec3f &normal)
{
    u32 octNormal = EncodeOctahedral(normal);
    Vec2u n       = Vec2u(octNormal >> 16, octNormal & ((1 << 16) - 1));
    minOct        = Min(minOct, n);
    maxOct        = Max(maxOct, n);
}

void DenseGeometryBuildData::WriteVertexData(const Mesh &mesh,
                                             const StaticArray<u32> &meshVertexIndices,
                                             DGFTempResources &tempResources)
{
    Vec3u bitWidths;
    bitWidths.x = tempResources.minP[0] == tempResources.maxP[0]
                      ? 0
                      : Log2Int(Max(tempResources.maxP[0] - tempResources.minP[0], 1)) + 1;
    bitWidths.y = tempResources.minP[1] == tempResources.maxP[1]
                      ? 0
                      : Log2Int(Max(tempResources.maxP[1] - tempResources.minP[1], 1)) + 1;
    bitWidths.z = tempResources.minP[2] == tempResources.maxP[2]
                      ? 0
                      : Log2Int(Max(tempResources.maxP[2] - tempResources.minP[2], 1)) + 1;

    u32 numBits = tempResources.vertexCount * (bitWidths.x + bitWidths.y + bitWidths.z);
    Assert(bitWidths.x < 32 && bitWidths.y < 32 && bitWidths.z < 32);

    u32 vertexBitStreamSize                      = (numBits + 7) >> 3;
    ChunkedLinkedList<u8>::ChunkNode *vertexNode = geoByteBuffer.AddNode(vertexBitStreamSize);

    u32 numOctBitsX =
        !tempResources.hasAnyNormals || tempResources.minOct[0] == tempResources.maxOct[0]
            ? 0
            : Log2Int(Max(tempResources.maxOct[0] - tempResources.minOct[0], 1u)) + 1;
    u32 numOctBitsY =
        !tempResources.hasAnyNormals || tempResources.minOct[1] == tempResources.maxOct[1]
            ? 0
            : Log2Int(Max(tempResources.maxOct[1] - tempResources.minOct[1], 1u)) + 1;

    Assert(numOctBitsX <= 16 && numOctBitsY <= 16);

    ErrorExit(
        Abs(tempResources.minP[0]) < (1 << 23) && Abs(tempResources.minP[1]) < (1 << 23) &&
            Abs(tempResources.minP[2]) < (1 << 23),
        "%i %i %i\n", tempResources.minP[0], tempResources.minP[1], tempResources.minP[2]);

    u32 vertexCount         = meshVertexIndices.Length();
    u32 normalBitStreamSize = (vertexCount * (numOctBitsX + numOctBitsY) + 7) >> 3;
    ChunkedLinkedList<u8>::ChunkNode *normalNode = 0;
    if (normalBitStreamSize)
    {
        normalNode = shadingByteBuffer.AddNode(normalBitStreamSize);
    }

    Vec3f quantizeP     = tempResources.quantizeP;
    u32 bitOffset       = 0;
    u32 normalBitOffset = 0;

    for (u32 index : meshVertexIndices)
    {
        Vec3i p = Vec3i(Round(mesh.p[index] * tempResources.quantizeP)) - tempResources.minP;
        for (int axis = 0; axis < 3; axis++)
        {
            WriteBits((u32 *)vertexNode->values, bitOffset, p[axis], bitWidths[axis]);
        }

        if (mesh.n)
        {
            Vec3f &n    = mesh.n[index];
            u32 normal  = EncodeOctahedral(n);
            u32 normalX = (normal >> 16) - tempResources.minOct[0];
            u32 normalY = (normal & ((1 << 16) - 1)) - tempResources.minOct[1];
            WriteBits((u32 *)normalNode->values, normalBitOffset, normalX, numOctBitsX);
            WriteBits((u32 *)normalNode->values, normalBitOffset, normalY, numOctBitsY);
        }
    }

    Assert(bitOffset == numBits);
    Assert(normalBitOffset == vertexCount * (numOctBitsX + numOctBitsY));

    PackedDenseGeometryHeader &packed = tempResources.packed;

    u32 headerOffset = 0;
    packed.b = BitFieldPackI32(packed.b, tempResources.minP[0], headerOffset, ANCHOR_WIDTH);

    headerOffset = 0;
    packed.c = BitFieldPackI32(packed.c, tempResources.minP[1], headerOffset, ANCHOR_WIDTH);
    packed.c = BitFieldPackU32(packed.c, bitWidths.x, headerOffset, 5);

    headerOffset = 0;
    packed.d = BitFieldPackI32(packed.d, tempResources.minP[2], headerOffset, ANCHOR_WIDTH);
    packed.d = BitFieldPackU32(packed.d, tempResources.precision - CLUSTER_MIN_PRECISION,
                               headerOffset, 8);

    headerOffset = 0;
    packed.e     = BitFieldPackU32(packed.e, vertexCount, headerOffset, 14);
    headerOffset += 8;
    packed.e = BitFieldPackU32(packed.e, bitWidths.y, headerOffset, 5);
    packed.e = BitFieldPackU32(packed.e, bitWidths.z, headerOffset, 5);

    headerOffset = 0;
    packed.h     = BitFieldPackU32(packed.h, numOctBitsX, headerOffset, 5);
    packed.h     = BitFieldPackU32(packed.h, numOctBitsY, headerOffset, 5);

    headerOffset = 0;
    packed.i     = BitFieldPackU32(packed.h, tempResources.minOct.x, headerOffset, 16);
    packed.i     = BitFieldPackU32(packed.h, tempResources.minOct.y, headerOffset, 16);
}

void DenseGeometryBuildData::WriteMaterialIDs(const StaticArray<u32> &materialIndices,
                                              DGFTempResources &tempResources)
{
    ScratchArena scratch;

    bool constantMaterialID = true;
    u32 materialStart       = materialIndices[0];

    for (u32 materialIndex : materialIndices)
    {
        if (materialIndex != materialStart)
        {
            constantMaterialID = false;
            break;
        }
    }

    PackedDenseGeometryHeader &packed = tempResources.packed;

    if (constantMaterialID)
    {
        u32 materialIndex = materialIndices[0];
        materialIndex |= 0x80000000;
        packed.j = materialIndex;
    }
    else
    {
        u32 commonMSBs        = 32;
        u32 currentCommonMSBs = materialIndices[0];

        u32 numUniqueMaterialIDs = 1;
        StaticArray<u32> uniqueMaterialIDs(scratch.temp.arena, materialIndices.Length());
        StaticArray<u32> entryIndex(scratch.temp.arena, materialIndices.Length(),
                                    materialIndices.Length());

        u32 materialIndex = materialIndices[0] & 0x7fffffff;
        uniqueMaterialIDs.Push(materialIndex);
        entryIndex[0] = 0;
        for (int primIndex = 1; primIndex < materialIndices.Length(); primIndex++)
        {
            u32 materialID = materialIndices[primIndex] & 0x7fffffff;
            bool unique    = true;
            for (int j = 0; j < uniqueMaterialIDs.Length(); j++)
            {
                if (uniqueMaterialIDs[j] == materialID)
                {
                    entryIndex[primIndex] = j;
                    unique                = false;
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
                entryIndex[primIndex] = uniqueMaterialIDs.Length() - 1;
            }
        }

        Assert(uniqueMaterialIDs.Length() > 1);
        u32 entryBitLength = Log2Int(uniqueMaterialIDs.Length() - 1) + 1;

        // Common MSBs + LSB entries + per triangle entries
        u32 materialTableNumBits = commonMSBs +
                                   (32 - commonMSBs) * uniqueMaterialIDs.Length() +
                                   entryBitLength * materialIndices.Length();
        u32 materialTableSize = (materialTableNumBits + 7) >> 3;
        auto *table           = shadingByteBuffer.AddNode(materialTableSize);
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

        Assert(commonMSBs > 0);
        u32 headerOffset = 0;

        packed.j = BitFieldPackU32(packed.j, commonMSBs - 1, headerOffset, 5);
        packed.j = BitFieldPackU32(packed.j, uniqueMaterialIDs.Length(), headerOffset, 5);
        // packed.j = BitFieldPackU32(packed.j, materialTableOffset, headerOffset, 22);
    }
}

void DenseGeometryBuildData::WriteVoxelData(StaticArray<CompressedVoxel> &voxels, Mesh &mesh,
                                            const StaticArray<u32> &materialIDs,
                                            StaticArray<u32> &voxelGeomIDs)
{
    ScratchArena scratch;
    DGFTempResources tempResources;

    tempResources.hasAnyNormals = (bool)mesh.n;

    StaticArray<u32> materialIndices(scratch.temp.arena, voxelGeomIDs.Length());

    u32 voxelTotal = 0;
    for (CompressedVoxel &voxel : voxels)
    {
        u64 bitMask      = voxel.bitMask;
        u32 vertexOffset = voxel.vertexOffset;

        u32 numVoxels = PopCount((u32)bitMask) + PopCount(bitMask >> 32u);

        voxelTotal += numVoxels;

        for (u32 voxelIndex = 0; voxelIndex < numVoxels; voxelIndex++)
        {
            u32 vertexIndex = vertexOffset + voxelIndex;
            tempResources.UpdatePosBounds(mesh.p[vertexIndex]);

            if (mesh.n)
            {
                Vec3f normal = mesh.n[vertexIndex];
                tempResources.UpdateNormalBounds(normal);
            }
        }
    }

    StaticArray<u32> voxelVertexIndices(scratch.temp.arena, voxelTotal);
    StaticArray<u32> voxelClusterVertexIndices(scratch.temp.arena, voxels.Length());

    voxelTotal = 0;
    for (CompressedVoxel &voxel : voxels)
    {
        u64 bitMask = voxel.bitMask;

        u32 numVoxels    = PopCount((u32)bitMask) + PopCount(bitMask >> 32u);
        u32 vertexOffset = voxel.vertexOffset;

        voxelClusterVertexIndices.Push(voxelTotal);
        for (u32 voxelIndex = 0; voxelIndex < numVoxels; voxelIndex++)
        {
            u32 vertexIndex   = vertexOffset + voxelIndex;
            u32 materialIndex = materialIDs[voxelGeomIDs[voxelTotal + voxelIndex]];
            materialIndices.Push(materialIndex);
            voxelVertexIndices.Push(vertexIndex);
        }
        voxelTotal += numVoxels;
    }

    tempResources.vertexCount = voxelTotal;

    u32 geoBaseAddress     = geoByteBuffer.Length();
    u32 shadingBaseAddress = shadingByteBuffer.Length();

    WriteVertexData(mesh, voxelVertexIndices, tempResources);

    u32 brickBitOffset      = 0;
    u32 brickOffset         = geoByteBuffer.Length();
    u32 numCompressedVoxels = voxels.Length();

    Assert(voxelClusterVertexIndices.Length() == numCompressedVoxels);
    u32 brickBitStreamSize = ((64 + 14) * numCompressedVoxels + 7u) >> 3u;
    auto *brickNode        = geoByteBuffer.AddNode(brickBitStreamSize);

    for (u32 brickIndex = 0; brickIndex < numCompressedVoxels; brickIndex++)
    {
        CompressedVoxel &brick = voxels[brickIndex];
        WriteBits((u32 *)brickNode->values, brickBitOffset, (u32)brick.bitMask, 32);
        WriteBits((u32 *)brickNode->values, brickBitOffset, brick.bitMask >> 32u, 32);
        WriteBits((u32 *)brickNode->values, brickBitOffset,
                  voxelClusterVertexIndices[brickIndex], 14); // MAX_CLUSTER_VERTICES_BIT);
    }

    Assert(materialIndices.Length() == voxelVertexIndices.Length());
    WriteMaterialIDs(materialIndices, tempResources);

    PackedDenseGeometryHeader &packed = tempResources.packed;
    packed.z                          = shadingBaseAddress;
    packed.a                          = geoBaseAddress;

    Assert(numCompressedVoxels < (1u << 14u));
    packed.g = numCompressedVoxels | (1u << 31u);

    headers.AddBack() = packed;
}

void DenseGeometryBuildData::WriteTriangleData(StaticArray<int> &triangleIndices,
                                               StaticArray<u32> &triangleGeomIDs, Mesh &mesh,
                                               StaticArray<u32> &materialIDs)
{
    ScratchArena scratch;
    DGFTempResources tempResources;

    StaticArray<u32> materialIndices(scratch.temp.arena, triangleIndices.Length());

    static const u32 LUT[] = {1, 2, 0};
    struct HashedIndex
    {
        // u32 geomID;
        u32 index;
        u32 clusterVertexIndex;
        // u32 Hash() const { return MixBits(((u64)geomID << 32) | index); }
        u32 Hash() const { return MixBits(index); }
        bool operator==(const HashedIndex &other) const { return index == other.index; }
    };

    auto ComputeEdgeId = [](u32 indices[3], u32 edgeIndex) {
        u32 id0 = indices[edgeIndex];
        u32 id1 = indices[LUT[edgeIndex]];
        return id0 < id1 ? ((u64)id1 << 32) | id0 : ((u64)id0 << 32) | id1;
    };

    u32 clusterNumTriangles = triangleIndices.Length();
    Assert(clusterNumTriangles <= MAX_CLUSTER_TRIANGLES);

    // Maps mesh indices to cluster indices
    HashMap<HashedIndex> vertexHashSet(scratch.temp.arena,
                                       NextPowerOfTwo(clusterNumTriangles * 3));

    StaticArray<u32> vertexIndices(scratch.temp.arena, clusterNumTriangles * 3,
                                   clusterNumTriangles * 3);
    StaticArray<u32> geomIDs(scratch.temp.arena, clusterNumTriangles);
    StaticArray<u32> clusterVertexIndexToMeshVertexIndex(scratch.temp.arena,
                                                         MAX_CLUSTER_VERTICES);

    u32 vertexCount = 0;
    u32 indexCount  = 0;

    auto GetIndices = [&vertexIndices](u32 ind[3], u32 triangleIndex) {
        ind[0] = vertexIndices[3 * triangleIndex + 0];
        ind[1] = vertexIndices[3 * triangleIndex + 1];
        ind[2] = vertexIndices[3 * triangleIndex + 2];
    };

    u32 minFaceID = pos_inf;
    u32 maxFaceID = 0;
    // bool hasFaceIDs    = false;
    tempResources.hasAnyNormals = (bool)mesh.n;

    Assert(triangleIndices.Length() == triangleGeomIDs.Length());
    for (u32 tri = 0; tri < triangleIndices.Length(); tri++)
    {
        int primID = triangleIndices[tri];
        u32 geomID = triangleGeomIDs[tri];

        geomIDs.Push(geomID);

#if 0
            u32 faceID         = mesh.faceIDs ? mesh.faceIDs[primID] : 0u;
            minFaceID          = Min(minFaceID, faceID);
            maxFaceID          = Max(maxFaceID, faceID);
            faceIDs[i - start] = faceID;
#endif

        // hasFaceIDs |= (bool(mesh.faceIDs) && bool(materialID >> 31u));

        for (int indexIndex = 0; indexIndex < 3; indexIndex++)
        {
            u32 vertexIndex = mesh.indices[3 * primID + indexIndex];
            tempResources.UpdatePosBounds(mesh.p[vertexIndex]);

            if (mesh.n)
            {
                Vec3f &normal = mesh.n[vertexIndex];
                tempResources.UpdateNormalBounds(normal);
            }

            auto *hashedIndex  = vertexHashSet.Find({vertexIndex});
            u32 newVertexIndex = ~0u;

            if (hashedIndex)
            {
                vertexIndices[indexCount++] = hashedIndex->clusterVertexIndex;
                newVertexIndex              = hashedIndex->clusterVertexIndex;
            }
            else
            {
                vertexHashSet.Add(scratch.temp.arena, {vertexIndex, vertexCount});
                vertexIndices[indexCount++] = vertexCount;
                clusterVertexIndexToMeshVertexIndex.Push(vertexIndex);

                vertexCount++;
            }
        }
    }

    // Build adjacency data for triangles
    u32 *counts  = PushArray(scratch.temp.arena, u32, clusterNumTriangles);
    u32 *offsets = PushArray(scratch.temp.arena, u32, clusterNumTriangles);

    struct EdgeKeyValue
    {
        u64 key;
        // u32 geomID;
        u32 count;
        u32 corner0;
        u32 corner1;

        static u32 Hash(u64 key) { return MixBits(key); }
        u32 Hash() const { return Hash(key); }

        bool operator==(const EdgeKeyValue &other) const { return key == other.key; }
    };
    HashIndex edgeIDMap(scratch.temp.arena, NextPowerOfTwo(indexCount),
                        NextPowerOfTwo(indexCount));
    StaticArray<EdgeKeyValue> edges(scratch.temp.arena, indexCount);

    // Find triangles with shared edge
    for (u32 triangleIndex = 0; triangleIndex < clusterNumTriangles; triangleIndex++)
    {
        u32 indices[3];
        GetIndices(indices, triangleIndex);
        // u32 geomID = geomIDs[triangleIndex];

        for (int edgeIndex = 0; edgeIndex < 3; edgeIndex++)
        {
            u64 edgeID = ComputeEdgeId(indices, edgeIndex);
            u32 corner = 3 * triangleIndex + edgeIndex;
            int hash   = EdgeKeyValue::Hash(edgeID);

            bool found = false;
            for (int hashIndex = edgeIDMap.FirstInHash(hash); hashIndex != -1;
                 hashIndex     = edgeIDMap.NextInHash(hashIndex))
            {
                EdgeKeyValue &key = edges[hashIndex];
                if (key.key == edgeID && key.corner1 == ~0u)
                {
                    found       = true;
                    key.corner1 = corner;
                    key.count++;
                    break;
                }
            }
            if (!found)
            {
                EdgeKeyValue newKey = {edgeID, 1, corner, ~0u};
                edges.Push(newKey);
                edgeIDMap.AddInHash(hash, edges.Length() - 1);
            }
        }
    }

    // Find number of adjacent triangles for each triangle
    for (u32 triangleIndex = 0; triangleIndex < clusterNumTriangles; triangleIndex++)
    {
        u32 indices[3];
        GetIndices(indices, triangleIndex);
        // u32 geomID = geomIDs[triangleIndex];
        u32 count = 0;

        for (int edgeIndex = 0; edgeIndex < 3; edgeIndex++)
        {
            u64 edgeID = ComputeEdgeId(indices, edgeIndex);
            u32 corner = 3 * triangleIndex + edgeIndex;
            int hash   = EdgeKeyValue::Hash(edgeID);
            for (int hashIndex = edgeIDMap.FirstInHash(hash); hashIndex != -1;
                 hashIndex     = edgeIDMap.NextInHash(hashIndex))
            {
                EdgeKeyValue &key = edges[hashIndex];
                if (key.key == edgeID && (corner == key.corner0 || corner == key.corner1))
                {
                    count += key.count - 1;
                    break;
                }
            }
        }
        counts[triangleIndex] = count;
    }

    u32 offset = 0;
    for (u32 i = 0; i < clusterNumTriangles; i++)
    {
        offsets[i] = offset;
        offset += counts[i];
    }

    u32 *data = PushArray(scratch.temp.arena, u32, offset);

    for (u32 triangleIndex = 0; triangleIndex < clusterNumTriangles; triangleIndex++)
    {
        u32 indices[3];
        GetIndices(indices, triangleIndex);
        // u32 geomID = geomIDs[triangleIndex];
        u32 count = 0;

        for (int edgeIndex = 0; edgeIndex < 3; edgeIndex++)
        {
            u64 edgeID = ComputeEdgeId(indices, edgeIndex);
            u32 corner = 3 * triangleIndex + edgeIndex;
            int hash   = EdgeKeyValue::Hash(edgeID);
            for (int hashIndex = edgeIDMap.FirstInHash(hash); hashIndex != -1;
                 hashIndex     = edgeIDMap.NextInHash(hashIndex))
            {
                EdgeKeyValue &key = edges[hashIndex];
                if (key.key == edgeID && (corner == key.corner0 || corner == key.corner1))
                {
                    data[offsets[triangleIndex]] =
                        corner == key.corner0 ? key.corner1 : key.corner0;
                    offsets[triangleIndex] += key.count - 1;
                    break;
                }
            }
        }
    }

    for (u32 i = 0; i < clusterNumTriangles; i++)
    {
        offsets[i] -= counts[i];
    }

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
    StaticArray<TriangleStripType> triangleStripTypes(scratch.temp.arena, clusterNumTriangles);
    StaticArray<u32> newIndexOrder(scratch.temp.arena, clusterNumTriangles * 3);
    StaticArray<u32> triangleOrder(scratch.temp.arena, clusterNumTriangles * 3);

    u32 prevTriangle = minValenceTriangle;

    auto RemoveFromNeighbor = [&](u32 tri, u32 neighbor) {
        u32 neighborCount = counts[neighbor] & ~removedBit;
        for (int j = 0; j < neighborCount; j++)
        {
            if (data[offsets[neighbor] + j] / 3 == tri)
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
    newIndexOrder.Push(firstIndices[0]);
    newIndexOrder.Push(firstIndices[1]);
    newIndexOrder.Push(firstIndices[2]);

    triangleOrder.Push(minValenceTriangle);

    MarkUsed(minValenceTriangle);

    // Find the new ordering of triangles and construct the triangle strip
    for (u32 numAddedTriangles = 1; numAddedTriangles < clusterNumTriangles;
         numAddedTriangles++)
    {
        u32 dwordIndex = numAddedTriangles >> 5;
        u32 bitIndex   = numAddedTriangles & 31u;

        u32 newMinValenceCorner = ~0u;

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

                newIndexOrder.Push(indices[0]);
                newIndexOrder.Push(indices[1]);
                newIndexOrder.Push(indices[2]);

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
            newMinValenceCorner = data[offsets[minValenceTriangle]];
            counts[minValenceTriangle]--;
        }
        else
        {
            // Remove min valence triangle from neighbor's adjacency list
            u32 minCount = ~0u;
            u32 minI     = 0;
            for (int i = 0; i < count; i++)
            {
                u32 neighborCorner   = data[offsets[minValenceTriangle] + i];
                u32 neighborTriangle = data[offsets[minValenceTriangle] + i] / 3;
                bool success = RemoveFromNeighbor(minValenceTriangle, neighborTriangle);
                Assert(success);
                // Find neighbor with minimum valence
                if (counts[neighborTriangle] < minCount)
                {
                    minCount            = counts[neighborTriangle];
                    newMinValenceCorner = neighborCorner;
                    minI                = i;
                }
            }
            Swap(data[offsets[minValenceTriangle] + minI],
                 data[offsets[minValenceTriangle] + count - 1]);
            counts[minValenceTriangle]--;
            MarkUsed(minValenceTriangle);
            Assert(minCount != ~0u);
        }

        Assert(newMinValenceCorner != ~0u);
        u32 newMinValenceTriangle = newMinValenceCorner / 3;
        triangleOrder.Push(newMinValenceTriangle);

        // Find what edge is shared, and rotate
        u32 indices[3];
        GetIndices(indices, newMinValenceTriangle);

        u32 edgeIndex = newMinValenceCorner % 3;
        u64 edgeID    = ComputeEdgeId(indices, edgeIndex);
        // Rotate
        // [0, 1, 2] -> [1, 2, 0]
        if (edgeIndex == 1)
        {
            vertexIndices[3 * newMinValenceTriangle]     = indices[1];
            vertexIndices[3 * newMinValenceTriangle + 1] = indices[2];
            vertexIndices[3 * newMinValenceTriangle + 2] = indices[0];
            newIndexOrder.Push(indices[0]);
        }
        // [0, 1, 2] -> [2, 0, 1]
        else if (edgeIndex == 2)
        {
            vertexIndices[3 * newMinValenceTriangle]     = indices[2];
            vertexIndices[3 * newMinValenceTriangle + 1] = indices[0];
            vertexIndices[3 * newMinValenceTriangle + 2] = indices[1];
            newIndexOrder.Push(indices[1]);
        }
        else
        {
            newIndexOrder.Push(indices[2]);
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
                    Assert(
                        (backTrackStripType == TriangleStripType::Edge1 &&
                         oldEdgeIndex == 1) ||
                        (backTrackStripType == TriangleStripType::Edge2 && oldEdgeIndex == 2));
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
        prevTriangle       = minValenceTriangle;
        minValenceTriangle = newMinValenceTriangle;
    }

    for (u32 tri : triangleOrder)
    {
        u32 materialIndex = materialIDs[geomIDs[tri]];
        materialIndices.Push(materialIndex);
    }

#if 0
    ChunkedLinkedList<u8>::ChunkNode *faceIDNode = 0;
    if (hasFaceIDs)
    {
        totalSize += faceBitStreamSize;
        faceIDNode = buildData.shadingByteBuffer.AddNode(faceBitStreamSize);
    }

    if (hasFaceIDs)
    {
        WriteBits((u32 *)faceIDNode->values, faceBitOffset, minFaceID, 32);
        for (u32 i = 0; i < triangleOrder.Length(); i++)
        {
            u32 index = triangleOrder[i];
            WriteBits((u32 *)faceIDNode->values, faceBitOffset, faceIDs[index] - minFaceID,
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
#endif

    u32 geoBaseAddress     = geoByteBuffer.Length();
    u32 shadingBaseAddress = shadingByteBuffer.Length();

    StaticArray<u32> reuseBuffer(scratch.temp.arena, clusterNumTriangles * 3);
    BitVector firstUse(scratch.temp.arena, clusterNumTriangles * 3);

    u32 faceBitOffset = 0;

    BitVector usedVertices(scratch.temp.arena, MAX_CLUSTER_VERTICES);
    StaticArray<u32> mapOldIndexToDGFIndex(scratch.temp.arena, vertexCount, vertexCount);
    u32 addedVertexCount   = 0;
    u32 currentFirstUseBit = 0;
    u32 maxReuseIndex      = 0;

    StaticArray<u32> meshVertexIndices(scratch.temp.arena, MAX_CLUSTER_VERTICES);
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

            meshVertexIndices.Push(clusterVertexIndexToMeshVertexIndex[index]);
        }
    }

    Assert(addedVertexCount == vertexCount);
    tempResources.vertexCount = addedVertexCount;
    WriteVertexData(mesh, meshVertexIndices, tempResources);

    FixedArray<u32, 4> restartHighBitPerDword = {0, 0, 0, 0};
    FixedArray<u32, 4> restartCountPerDword   = {1, 0, 0, 0};
    FixedArray<i32, 4> edge1HighBitPerDword   = {-1, -1, -1, -1};
    FixedArray<i32, 4> edge2HighBitPerDword   = {-1, -1, -1, -1};

    u32 prevEdge1HighBitPerDword = 0;
    u32 prevEdge2HighBitPerDword = 0;

    BitVector backtrack(scratch.temp.arena, clusterNumTriangles);
    BitVector restart(scratch.temp.arena, clusterNumTriangles);
    BitVector edge(scratch.temp.arena, clusterNumTriangles);

    restart.SetBit(0);

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

    // Write reuse buffer (byte aligned)
    u32 numIndexBits = Log2Int(Max(maxReuseIndex, 1u)) + 1;
    Assert(numIndexBits >= 1 && numIndexBits <= 8);
    u32 ctrlBitSize = ((clusterNumTriangles + 31) >> 5u) * 12u;
    u32 bitStreamSize =
        ((numIndexBits * reuseBuffer.Length() + currentFirstUseBit + 7) >> 3) + ctrlBitSize;
    u32 numBitStreamBits =
        numIndexBits * reuseBuffer.Length() + (ctrlBitSize << 3) + currentFirstUseBit;
    auto *node = geoByteBuffer.AddNode(bitStreamSize);

    // Write control bits
    int numRemainingTriangles = clusterNumTriangles;
    u32 dwordIndex            = 0;
    u32 indexBitOffset        = 0;
    while (numRemainingTriangles > 0)
    {
        WriteBits((u32 *)node->values, indexBitOffset, restart.bits[dwordIndex], 32);
        WriteBits((u32 *)node->values, indexBitOffset, edge.bits[dwordIndex], 32);
        WriteBits((u32 *)node->values, indexBitOffset, backtrack.bits[dwordIndex], 32);
        dwordIndex++;
        numRemainingTriangles -= 32;
    }
    for (u32 reuseIndex : reuseBuffer)
    {
        WriteBits((u32 *)node->values, indexBitOffset, reuseIndex, numIndexBits);
    }
    // Write first use bits
    u32 firstBitWriteOffset  = 0;
    u32 firstUseBitWriteSize = Min(currentFirstUseBit - firstBitWriteOffset, 32u);
    while (firstBitWriteOffset != currentFirstUseBit)
    {
        WriteBits((u32 *)node->values, indexBitOffset, firstUse.bits[firstBitWriteOffset >> 5],
                  firstUseBitWriteSize);
        firstBitWriteOffset += firstUseBitWriteSize;
        firstUseBitWriteSize = Min(currentFirstUseBit - firstBitWriteOffset, 32u);
    }

    Assert(indexBitOffset == numBitStreamBits);

    WriteMaterialIDs(materialIndices, tempResources);

    PackedDenseGeometryHeader &packed = tempResources.packed;

    u32 headerOffset = 0;
    packed.z         = shadingBaseAddress;
    packed.a         = geoBaseAddress;

    headerOffset = ANCHOR_WIDTH;
    packed.b     = BitFieldPackU32(packed.b, clusterNumTriangles, headerOffset, 8);

    headerOffset = ANCHOR_WIDTH + 5;
    packed.c     = BitFieldPackU32(packed.c, numIndexBits - 1, headerOffset, 3);

    headerOffset = 0;
    Assert(reuseBuffer.Length() < (1 << 8));
    packed.e = BitFieldPackU32(packed.e, reuseBuffer.Length(), headerOffset, 8);

    headerOffset = 0;

    packed.f = BitFieldPackU32(packed.f, restartHighBitPerDword[1], headerOffset, 6);
    packed.f = BitFieldPackU32(packed.f, restartHighBitPerDword[2], headerOffset, 7);
    packed.f = BitFieldPackI32(packed.f, edge1HighBitPerDword[0], headerOffset, 6);
    packed.f = BitFieldPackI32(packed.f, edge1HighBitPerDword[1], headerOffset, 7);
    packed.f = BitFieldPackI32(packed.f, edge2HighBitPerDword[0], headerOffset, 6);

    packed.g = BitFieldPackU32(packed.g, restartHighBitPerDword[0], headerOffset, 5);
    packed.g = BitFieldPackI32(packed.g, edge1HighBitPerDword[2], headerOffset, 8);
    packed.g = BitFieldPackI32(packed.g, edge2HighBitPerDword[1], headerOffset, 7);
    packed.g = BitFieldPackI32(packed.g, edge2HighBitPerDword[2], headerOffset, 8);

    headerOffset = 5;
    packed.h     = BitFieldPackU32(packed.h, restartCountPerDword[0], headerOffset, 6);
    packed.h     = BitFieldPackU32(packed.h, restartCountPerDword[1], headerOffset, 7);
    packed.h     = BitFieldPackU32(packed.h, restartCountPerDword[2], headerOffset, 8);
    // packed.h     = BitFieldPackU32(packed.h, numFaceBits, headerOffset, 6);

    headers.AddBack() = packed;
}

} // namespace rt
