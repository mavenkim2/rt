#ifndef AS_SHADER_INTEROP_H_
#define AS_SHADER_INTEROP_H_

#include "hlsl_cpp_compat.h"

#ifdef __cplusplus
namespace rt
{
#endif

#define RAY_TRACING_ADDRESS_STRIDE               8
#define FILL_CLUSTER_BOTTOM_LEVEL_INFO_GROUPSIZE 32

#define MAX_CLUSTERS_PER_PAGE_BITS 8
#define MAX_CLUSTERS_PER_PAGE      (1u << MAX_CLUSTERS_PER_PAGE_BITS)

#define CLUSTER_PAGE_SIZE_BITS 17
#define CLUSTER_PAGE_SIZE      (1u << CLUSTER_PAGE_SIZE_BITS)

#define CLUSTER_FILE_MAGIC 0x6A697975

#define NUM_CLUSTER_HEADER_FLOAT4S 4

#define MAX_CANDIDATE_NODES    (1u << 21u)
#define MAX_CANDIDATE_CLUSTERS (1u << 24)
#define MAX_VISIBLE_CLUSTERS   (1u << 22)

static const uint32_t kFillInstanceDescsThreads = 32;

struct ClusterPageHeader
{
    uint numClusters;
};

struct ClusterFileHeader
{
    uint magic;
    uint numPages;
    uint numNodes;
};

struct DecodeClusterData
{
    uint32_t pageIndex;
    uint32_t clusterIndex;
    uint32_t indexBufferOffset;
    uint32_t vertexBufferOffset;
};

struct BLASData
{
    uint instanceIndex;
    uint clusterStartIndex;
    uint clusterCount;
};

struct AccelerationStructureInstance
{
#ifdef __cplusplus
    float transform[3][4];
#else
    row_major float3x4 transform;
#endif
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
    uint indexBufferBaseAddressLowBits;
    uint indexBufferBaseAddressHighBits;
    uint vertexBufferBaseAddressLowBits;
    uint vertexBufferBaseAddressHighBits;
};

struct FillClusterBottomLevelInfoPushConstant
{
    uint arrayBaseAddressLowBits;
    uint arrayBaseAddressHighBits;
};

struct NumPushConstant
{
    uint num;
};

#define CHILDREN_PER_HIERARCHY_NODE      4
#define CHILDREN_PER_HIERARCHY_NODE_BITS 2

struct PackedHierarchyNode
{
    float4 lodBounds[CHILDREN_PER_HIERARCHY_NODE];
    float maxParentError[CHILDREN_PER_HIERARCHY_NODE];
    uint childOffset[CHILDREN_PER_HIERARCHY_NODE];
    uint flags;
};

struct GPUInstance
{
#ifdef __cplusplus
    float renderFromObject[3][4];
#else
    row_major float3x4 renderFromObject;
#endif
    uint globalRootNodeOffset;
};

struct TestDenseGeometry
{
    uint baseAddress;
    uint geoBaseAddress;
    uint shadBaseAddress;

    float4 lodBounds;
    float lodError;

    int3 anchor;
    uint2 octBase;

    uint3 posBitWidths;
    uint2 octBitWidths;
    uint indexBitWidth;
    uint numFaceIDBits;

    uint numTriangles;
    uint numVertices;

    uint normalOffset;
    uint faceIDOffset;
    uint indexOffset;
    uint ctrlOffset;
    uint firstBitOffset;
    int posPrecision;
    uint materialInfo;

    uint3 numPrevRestartsBeforeDwords;
    uint3 prevHighRestartBeforeDwords;
    int3 prevHighEdge1BeforeDwords;
    int3 prevHighEdge2BeforeDwords;

    bool debug;
};

#define GLOBALS_VERTEX_BUFFER_OFFSET_INDEX 0
#define GLOBALS_INDEX_BUFFER_OFFSET_INDEX  1
#define GLOBALS_CLAS_COUNT_INDEX           2
#define GLOBALS_DECODE_INDIRECT_Y          3
#define GLOBALS_DECODE_INDIRECT_Z          4

#define GLOBALS_BLAS_COUNT_INDEX      5
#define GLOBALS_BLAS_CLAS_COUNT_INDEX 6

#define GLOBALS_CLAS_INDIRECT_X 7
#define GLOBALS_CLAS_INDIRECT_Y 8
#define GLOBALS_CLAS_INDIRECT_Z 9

#define GLOBALS_BLAS_INDIRECT_X 10
#define GLOBALS_BLAS_INDIRECT_Y 11
#define GLOBALS_BLAS_INDIRECT_Z 12

#define GLOBALS_SIZE 13

#ifdef __cplusplus
}
#endif

#endif
