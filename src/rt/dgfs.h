#ifndef DGFS_H_
#define DGFS_H_

#include "containers.h"
#include "math/simd_base.h"
#include "math/vec3.h"
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
    ChunkedLinkedList<u8> geoByteBuffer;
    ChunkedLinkedList<u8> shadingByteBuffer;
    ChunkedLinkedList<PackedDenseGeometryHeader> headers;

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

    DenseGeometryBuildData();
};

struct DGFTempResources
{
    StaticArray<TriangleStripType> triangleStripTypes;

    StaticArray<u32> newIndexOrder;
    StaticArray<u32> voxelVertexIndices;

    Array<u32> materialIndices;
};

struct ClusterIndices
{
    StaticArray<Vec2u> triangleIndices;
    StaticArray<u32> brickIndices;
};

struct CompressedVoxel
{
    u64 bitMask;
    u32 vertexOffset;
};

struct ClusterBuilder
{
    Arena **arenas;
    StaticArray<ClusterList> threadClusters;
    PrimRef *primRefs;
    void *h;

    Vec3f quantizeP;
    int precision;
    bool hasAnyNormals;
    u32 vertexCount;

    Vec3i minP;
    Vec3i maxP;

    Vec2u minOct;
    Vec2u maxOct;

    ClusterBuilder() {}
    ClusterBuilder(Arena *arena, PrimRef *primRefs);
    void BuildClusters(RecordAOSSplits &record, bool parallel,
                       u32 maxTriangles = MAX_CLUSTER_TRIANGLES);

    ClusterIndices GetClusterIndices(Arena *arena, RecordAOSSplits &record);
    void Triangles(Arena *arena, Mesh &mesh, StaticArray<Vec2u> &triangleIndices,
                   const StaticArray<u32> &materialIndices, DGFTempResources &tempResources);
    void Voxels(Arena *arena, Mesh &mesh, StaticArray<CompressedVoxel> &voxels,
                StaticArray<u32> &brickIndices, const StaticArray<u32> &materialIndices,
                StaticArray<u32> &voxelGeomIDs, DGFTempResources &tempResources);
    void WriteData(Mesh &mesh, DGFTempResources &tempResources,
                   StaticArray<CompressedVoxel> &compressedVoxels,
                   DenseGeometryBuildData &buildData);
    void CreateDGFs(const StaticArray<u32> &materialIDs, DenseGeometryBuildData &buildData,
                    Mesh &mesh, Bounds &sceneBounds, StaticArray<CompressedVoxel> *voxels = 0,
                    StaticArray<u32> *voxelGeomIDs = 0);
};

void WriteBits(u32 *data, u32 &position, u32 value, u32 numBits);

} // namespace rt
#endif
