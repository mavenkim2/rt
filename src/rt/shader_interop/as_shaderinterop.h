#ifndef AS_SHADER_INTEROP_H_
#define AS_SHADER_INTEROP_H_

#include "hlsl_cpp_compat.h"

#ifdef __cplusplus
namespace rt
{
#endif

#define TRACE_BRICKS
#define RAY_TRACING_ADDRESS_STRIDE               8
#define FILL_CLUSTER_BOTTOM_LEVEL_INFO_GROUPSIZE 32

#define MAX_CLUSTERS_PER_PAGE_BITS 8u
#define MAX_CLUSTERS_PER_PAGE      (1u << MAX_CLUSTERS_PER_PAGE_BITS)
#define MAX_PARTS_PER_GROUP_BITS   3u
#define MAX_PARTS_PER_GROUP        (1u << MAX_PARTS_PER_GROUP_BITS)

#define MAX_CLUSTERS_PER_GROUP_BITS 5u
#define MAX_CLUSTERS_PER_GROUP      (1u << MAX_CLUSTERS_PER_GROUP_BITS)

#define CLUSTER_PAGE_SIZE_BITS 17
#define CLUSTER_PAGE_SIZE      (1u << CLUSTER_PAGE_SIZE_BITS)

#define CLUSTER_FILE_MAGIC 0x6A697975

#define NUM_CLUSTER_HEADER_FLOAT4S 6

#define MAX_CANDIDATE_NODES    (1u << 22u)
#define MAX_CANDIDATE_CLUSTERS (1u << 24)
#define MAX_VISIBLE_CLUSTERS   (1u << 22)

#define MAX_CLUSTERS_PER_BLAS 8192

#define CLUSTER_STREAMING_LEAF_FLAG 0x1

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
    uint numVoxelClusters;
    bool hasTruncatedEllipsoid;
};

struct DecodeClusterData
{
    uint32_t pageIndex;
    uint32_t clusterIndex;
    // NOTE: reused as a "brick offset" for voxel clusters
    uint32_t indexBufferOffset;
    uint32_t vertexBufferOffset;
};

struct VoxelPageDecodeData
{
    uint32_t offset;
    uint clusterStartIndex;
    uint clusterEndIndex;
    int pageIndex;
};

struct BLASData
{
    uint instanceID;
    uint clusterStartIndex;
    uint clusterCount;

    uint addressIndex;
};

struct BLASVoxelInfo
{
    uint64_t address;
    uint clusterID;
    uint instanceIndex;
};

struct GPUTruncatedEllipsoid
{
#ifdef __cplusplus
    float transform[3][4];
#else
    row_major float3x4 transform;
#endif
    float4 sphere;
    float3 boundsMin;
    float3 boundsMax;
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

struct INSTANTIATE_CLUSTER_TEMPLATE_INFO
{
    uint32_t clusterIdOffset;
    uint32_t geometryIndexOffset : 24;
    uint32_t reserved : 8;
    uint64_t clusterTemplateAddress;
    struct
    {
        uint64_t startAddress;
        uint64_t strideInBytes;
    } vertexBuffer;
};

struct BUILD_RANGE_INFO
{
    uint32_t primitiveCount;
    uint32_t primitiveOffset;
    uint32_t firstVertex;
    uint32_t transformOffset;
};

#define PTLAS_TYPE_WRITE_INSTANCE              0
#define PTLAS_TYPE_UPDATE_INSTANCE             1
#define PTLAS_TYPE_WRITE_PARTITION_TRANSLATION 2

struct PTLAS_INDIRECT_COMMAND
{
    uint32_t opType;
    uint32_t argCount;
    struct
    {
        uint64_t startAddress;
        uint64_t strideInBytes;
    };
};

#define PTLAS_WRITE_INSTANCE_INFO_STRIDE  104
#define PTLAS_UPDATE_INSTANCE_INFO_STRIDE 16

struct PTLAS_WRITE_INSTANCE_INFO
{
#ifdef __cplusplus
    float transform[3][4];
#else
    row_major float3x4 transform;
#endif
    float explicitAABB[6];
    uint32_t instanceID;
    uint32_t instanceMask;
    uint32_t instanceContributionToHitGroupIndex;
    uint32_t instanceFlags;
    uint32_t instanceIndex;
    uint32_t partitionIndex;
    uint64_t accelerationStructure;
};

struct PTLAS_UPDATE_INSTANCE_INFO
{
    uint32_t instanceIndex;
    uint32_t instanceContributionToHitGroupIndex;
    uint64_t accelerationStructure;
};

#ifdef __cplusplus
static_assert(sizeof(PTLAS_WRITE_INSTANCE_INFO) == PTLAS_WRITE_INSTANCE_INFO_STRIDE);
static_assert(sizeof(PTLAS_UPDATE_INSTANCE_INFO) == PTLAS_UPDATE_INSTANCE_INFO_STRIDE);
#endif

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
    uint clusterOffset;
};

struct DefragPushConstant
{
    uint evictedPageStart;
    uint numEvictedPages;
};

struct AddressPushConstant
{
    uint addressLowBits;
    uint addressHighBits;
};

struct PtlasPushConstant
{
    uint64_t writeAddress;
    uint64_t updateAddress;
};

struct InstanceCullingPushConstant
{
    uint64_t oneBlasAddress;
    uint num;
};

struct InstanceStreamingPushConstant
{
    uint numEvictedPartitions;
    uint numNewInstances;
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
    float3 center[CHILDREN_PER_HIERARCHY_NODE];
    float3 extents[CHILDREN_PER_HIERARCHY_NODE];
    float maxParentError[CHILDREN_PER_HIERARCHY_NODE];
    uint childRef[CHILDREN_PER_HIERARCHY_NODE];
    uint leafInfo[CHILDREN_PER_HIERARCHY_NODE];
};

struct GPUTransform
{
    uint32_t scaleX_scaleY;
    uint32_t scaleZ_rotX;
    uint32_t rotY_rotZ;
    uint32_t rotW_translateX;
    uint32_t translateY_translateZ;

#ifdef __cplusplus
    GPUTransform(uint16_t scaleX, uint16_t scaleY, uint16_t scaleZ, uint16_t rotX,
                 uint16_t rotY, uint16_t rotZ, uint16_t rotW, uint16_t translateX,
                 uint16_t translateY, uint16_t translateZ)
    {
        scaleX_scaleY         = scaleX | ((u32)scaleY << 16u);
        scaleZ_rotX           = scaleZ | ((u32)rotX << 16u);
        rotY_rotZ             = rotY | ((u32)rotZ << 16u);
        rotW_translateX       = rotW | ((u32)translateX << 16u);
        translateY_translateZ = translateY | ((u32)translateZ << 16u);
    }
#endif
};

#ifndef __cplusplus
float3x4 ConvertGPUMatrix(GPUTransform transform, float3 anchor, float3 scale)
{
    float scaleX = f16tof32(transform.scaleX_scaleY);
    float scaleY = f16tof32(transform.scaleX_scaleY >> 16u);
    float scaleZ = f16tof32(transform.scaleZ_rotX);

    float4 q =
        float4(f16tof32(transform.scaleZ_rotX >> 16u), f16tof32(transform.rotY_rotZ),
               f16tof32(transform.rotY_rotZ >> 16u), f16tof32(transform.rotW_translateX));

    float translateX = (transform.rotW_translateX >> 16u) * scale[0] + anchor[0];
    float translateY = (transform.translateY_translateZ & 0xffff) * scale[1] + anchor[1];
    float translateZ = (transform.translateY_translateZ >> 16u) * scale[2] + anchor[2];

    float4 quat = normalize(q);
    float xx    = quat.x * quat.x;
    float yy    = quat.y * quat.y;
    float zz    = quat.z * quat.z;
    float xy    = quat.x * quat.y;
    float xz    = quat.x * quat.z;
    float yz    = quat.y * quat.z;
    float wx    = quat.w * quat.x;
    float wy    = quat.w * quat.y;
    float wz    = quat.w * quat.z;

    float3x4 result =
        float3x4(scaleX * (1 - 2 * (yy + zz)), scaleY * 2 * (xy - wz), scaleZ * 2 * (xz + wy),
                 translateX, scaleX * 2 * (xy + wz), scaleY * (1 - 2 * (xx + zz)),
                 scaleZ * 2 * (yz - wx), translateY, scaleX * 2 * (xz - wy),
                 scaleY * 2 * (yz + wx), scaleZ * (1 - 2 * (xx + yy)), translateZ);
    return result;
}
#endif

#define GPU_INSTANCE_FLAG_CULL         (1u << 0u)
#define GPU_INSTANCE_FLAG_MERGED       (1u << 1u)
#define GPU_INSTANCE_FLAG_WAS_RENDERED (1u << 2u)
#define GPU_INSTANCE_FLAG_FREED        (1u << 3u)
#define GPU_INSTANCE_FLAG_INDIV        (1u << 4u)

struct GPUInstance
{
    uint transformIndex;
    uint globalRootNodeOffset;
    uint resourceID;
    uint partitionIndex;
    uint voxelAddressOffset;
    uint clusterLookupTableOffset;
    uint flags;
};

#define PARTITION_FLAG_INSTANCES_RENDERED (1u << 0)
#define PARTITION_FLAG_PROXY_RENDERED     (1u << 1)

struct PartitionInfo
{
    float4 lodBounds;
    float lodError;
    float3 base;
    float3 scale;
    uint32_t transformOffset;
    uint32_t transformCount;
    uint32_t flags;
};

struct GPUClusterFixup
{
    uint offset;
};

struct Brick
{
    uint64_t bitMask;
    uint vertexOffset;
};

#define RESOURCE_FLAG_ONE_CLUSTER (1u << 0u)
struct Resource
{
    uint finestAddressIndex;
    uint clusterLookupTableOffset;
    uint voxelAddressOffset;
    uint flags;
};

struct VoxelAddressTableEntry
{
    uint64_t address;
    uint32_t tableOffset;
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

#define GLOBALS_NEW_PAGE_DATA_BYTES 13
#define GLOBALS_OLD_PAGE_DATA_BYTES 14

#define GLOBALS_DEFRAG_CLAS_COUNT 15

#define GLOBALS_BLAS_FINAL_COUNT_INDEX 16
#define GLOBALS_BLAS_BYTES             17

#define GLOBALS_PTLAS_OP_TYPE_COUNT_INDEX 18

#define GLOBALS_VISIBLE_CLUSTER_COUNT_INDEX 19

#define GLOBALS_DEBUG 20

#define GLOBALS_UNUSED_CHECK_INDEX 21

#define GLOBALS_PTLAS_UPDATE_COUNT_INDEX 22
#define GLOBALS_PTLAS_WRITE_COUNT_INDEX  23

#define GLOBALS_NEW_INSTANCE_COUNT           24
#define GLOBALS_FREED_PARTITION_COUNT        25
#define GLOBALS_VISIBLE_PARTITION_COUNT      26
#define GLOBALS_VISIBLE_PARTITION_INDIRECT_X 27
#define GLOBALS_VISIBLE_PARTITION_INDIRECT_Y 28
#define GLOBALS_VISIBLE_PARTITION_INDIRECT_Z 29

#define GLOBALS_SIZE 30

#ifdef __cplusplus
}
#endif

#endif
