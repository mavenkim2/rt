#ifndef AS_SHADER_INTEROP_H_
#define AS_SHADER_INTEROP_H_

#ifdef __cplusplus
namespace rt
{
#endif

// #define USE_PROCEDURAL_CLUSTER_INTERSECTION

#define RAY_TRACING_ADDRESS_STRIDE               8
#define FILL_CLUSTER_BOTTOM_LEVEL_INFO_GROUPSIZE 32

#define MAX_CLUSTERS_PER_PAGE_BITS 8
#define MAX_CLUSTERS_PER_PAGE      (1u << MAX_CLUSTERS_PER_PAGE_BITS)

static const uint32_t kFillInstanceDescsThreads = 32;

struct ClusterPageData
{
    uint32_t clusterStart;
    uint32_t clusterCount;
    uint32_t blasIndex;
};

struct DecodeClusterData
{
    uint32_t indexBufferOffset;
    uint32_t vertexBufferOffset;
    uint32_t blasIndex;
};

struct BLASData
{
    uint clusterStartIndex;
    uint clusterCount;
};

struct ClusterData
{
    uint32_t indexBufferOffset;
    uint32_t vertexBufferOffset;
};

struct AccelerationStructureInstance
{
    float transform[3][4];
    uint32_t instanceID : 24;
    uint32_t instanceMask : 8;
    uint32_t instanceContributionToHitGroupIndex : 24;
    uint32_t flags : 8;
    uint64_t blasDeviceAddress;
};

struct BUILD_CLUSTERS_BOTTOM_LEVEL_INFO
{
    uint32_t clusterReferencesCount;
    uint32_t clusterReferencesStride;
    uint64_t clusterReferences;
};

struct IndirectArgs
{
    uint32_t clasCount;
};

struct BUILD_CLUSTERS_TRIANGLE_INFO
{
    uint32_t clusterId;
    uint32_t clusterFlags;
    uint32_t triangleCount : 9;
    uint32_t vertexCount : 9;
    uint32_t positionTruncateBitCount : 6;
    uint32_t indexFormat : 4;
    uint32_t opacityMicromapIndexFormat : 4;
    uint32_t baseGeometryIndexAndFlags;
    uint32_t indexBufferStride : 16;
    uint32_t vertexBufferStride : 16;
    uint32_t geometryIndexAndFlagsBufferStride : 16;
    uint32_t opacityMicromapIndexBufferStride : 16;
    uint64_t indexBuffer;
    uint64_t vertexBuffer;
    uint64_t geometryIndexAndFlagsBuffer;
    uint64_t opacityMicromapArray;
    uint64_t opacityMicromapIndexBuffer;
};

struct GeometryIndexAndFlags
{
    uint32_t geometryIndex : 24;
    uint32_t reserved : 5;
    uint32_t geometryFlags : 3;
};

struct FillClusterTriangleInfoPushConstant
{
    uint numClusters;
    uint indexBufferBaseAddressLowBits;
    uint indexBufferBaseAddressHighBits;
    uint vertexBufferBaseAddressLowBits;
    uint vertexBufferBaseAddressHighBits;
};

struct FillClusterBottomLevelInfoPushConstant
{
    uint blasCount;
    uint arrayBaseAddressLowBits;
    uint arrayBaseAddressHighBits;
};

struct NumPushConstant
{
    uint num;
};

#define GLOBALS_VERTEX_BUFFER_OFFSET_INDEX 0
#define GLOBALS_INDEX_BUFFER_OFFSET_INDEX  1
#define GLOBALS_CLAS_COUNT_INDEX           2
#define GLOBALS_BLAS_COUNT_INDEX           3
#define GLOBALS_BLAS_CLAS_COUNT_INDEX      4

#ifdef __cplusplus
}
#endif

#endif
