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
struct TileMetadata;

enum class TriangleStripType
{
    Restart,
    Edge1,
    Edge2,
    Backtrack,
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

    // Debug
#if 0
    ChunkedLinkedList<u32> firstUse;
    ChunkedLinkedList<u32> reuse;
#endif
    ChunkedLinkedList<u32> debugFaceIDs;
    ChunkedLinkedList<TriangleStripType> types;
    ChunkedLinkedList<u32> debugIndices;

    ChunkedLinkedList<u32> debugRestartCountPerDword;
    ChunkedLinkedList<u32> debugRestartHighBitPerDword;

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
    void CreateDGFs(ScenePrimitives *scene, DenseGeometryBuildData *buildData, Mesh *meshes,
                    int numMeshes, Bounds &sceneBounds);
    void CreateDGFs(ScenePrimitives *scene, DenseGeometryBuildData *buildDatas, Arena *arena,
                    Mesh *meshes, const StaticArray<StaticArray<Vec3i>> &quantizedVertices,
                    const StaticArray<StaticArray<u32>> &normals, RecordAOSSplits &cluster,
                    int precision);
};

void WriteBits(u32 *data, u32 &position, u32 value, u32 numBits);
Mesh ConvertQuadToTriangleMesh(Arena *arena, Mesh mesh);

} // namespace rt
#endif
