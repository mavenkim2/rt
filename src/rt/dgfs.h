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
struct SGGXCompact;
struct TileMetadata;

enum class TriangleStripType
{
    Restart,
    Edge1,
    Edge2,
    Backtrack,
};

struct CompressedVoxel
{
    u64 bitMask;
    u32 vertexOffset;
};

// struct ClusterBuilder
// {
//     Arena **arenas;
//     StaticArray<RecordAOSSplits> records;
//
//     PrimRef *primRefs;
//     void *h;
//
//     ClusterIndices GetClusterIndices(Arena *arena, RecordAOSSplits &record);
// };

struct DGFTempResources
{
    // Serialization
    Vec3f quantizeP;
    int precision;
    bool hasAnyNormals;

    Vec3i minP;
    Vec3i maxP;

    Vec2u minOct;
    Vec2u maxOct;

    PackedDenseGeometryHeader packed = {};

    DGFTempResources();
    void UpdatePosBounds(const Vec3f &p);
    void UpdateNormalBounds(const Vec3f &normal);
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

    DenseGeometryBuildData(Arena *arena);

    void WriteVertexData(const Mesh &mesh, const StaticArray<u32> &meshVertexIndices,
                         const StaticArray<u32> &meshNormalIndices,
                         DGFTempResources &tempResources);
    void WriteMaterialIDs(const StaticArray<u32> &materialIndices,
                          DGFTempResources &tempResources);
    void WriteVoxelData(StaticArray<CompressedVoxel> &voxels, Mesh &mesh,
                        const StaticArray<u32> &materialIDs, StaticArray<u32> &voxelGeomIDs,
                        f32 *coverages, SGGXCompact *sggx);
    void WriteTriangleData(StaticArray<int> &triangleIndices, StaticArray<u32> &geomIDs,
                           Mesh &mesh, StaticArray<u32> &materialIDs);
};

namespace ClusterBuilder
{

StaticArray<RecordAOSSplits> BuildClusters(Arena *arena, PrimRef *primRefs,
                                           RecordAOSSplits &record, u32 maxTriangles,
                                           u32 maxClusters);

}; // namespace ClusterBuilder

} // namespace rt
#endif
