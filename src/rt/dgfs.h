#ifndef DGFS_H_
#define DGFS_H_

#include "containers.h"
#include "math/simd_base.h"
#include "shader_interop/dense_geometry_shaderinterop.h"

namespace rt
{

struct Mesh;
struct PrimRef;
struct RecordAOSSplits;
struct ScenePrimitives;

static const int maxClusterTriangles = 128;
static const int clusterSize         = 128;

struct U128
{
    u64 vals[2];
    void operator|=(U128 other)
    {
        vals[0] |= other.vals[0];
        vals[1] |= other.vals[1];
    }
    void SetBit(u32 currentFirstUseBit)
    {
        vals[currentFirstUseBit >> 6] |= 1ull << (currentFirstUseBit & 63);
    }
};

struct BitVector
{
    u32 *bits;
    u32 maxNumBits;
    BitVector(Arena *arena, u32 maxNumBits);
    void SetBit(u32 bit);
    bool GetBit(u32 bit);
};

struct alignas(CACHE_LINE_SIZE) ClusterList
{
    ChunkedLinkedList<RecordAOSSplits> l;
};

struct alignas(CACHE_LINE_SIZE) ClusterExtents
{
    Vec3f extent;
};

struct alignas(CACHE_LINE_SIZE) DenseGeometryBuildData
{
    Arena *arena;
    ChunkedLinkedList<u8> byteBuffer;
    ChunkedLinkedList<PackedDenseGeometryHeader> headers;
    u32 numBlocks;

    void Init();
    void Merge(DenseGeometryBuildData &other);
};

struct ClusterBuilder
{
    Arena **arenas;
    StaticArray<ClusterList> threadClusters;
    PrimRef *primRefs;
    void *h;

    ClusterBuilder(Arena *arena, ScenePrimitives *scene, PrimRef *primRefs);
    void BuildClusters(RecordAOSSplits &record, bool parallel);
    void CreateDGFs(DenseGeometryBuildData *buildData, Mesh *meshes, int numMeshes,
                    Bounds &sceneBounds);
    void CreateDGFs(DenseGeometryBuildData *data, Arena *arena, Mesh *meshes,
                    Vec3i **quantizedVertices, RecordAOSSplits &cluster);
};

#if 0
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
#endif

} // namespace rt
#endif
