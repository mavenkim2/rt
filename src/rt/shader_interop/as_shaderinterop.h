#ifndef AS_SHADER_INTEROP_H_
#define AS_SHADER_INTEROP_H_

#ifdef __cplusplus
namespace rt
{
#endif

static const uint32_t kFillInstanceDescsThreads = 32;

struct AccelerationStructureInstance
{
#ifdef __cplusplus
    float transform[12];
#else
    float4 transform[3];
#endif
    uint32_t instanceID : 24;
    uint32_t instanceMask : 8;
    uint32_t instanceContributionToHitGroupIndex : 24;
    uint32_t flags : 8;
    uint64_t blasDeviceAddress;
};

struct ClusterBottomLevelInfo
{
    uint32_t clusterReferencesCount;
    uint32_t clusterReferencesStride;
    uint64_t clusterReferences;
};

struct IndirectArgs
{
    uint32_t clasCount;
};

struct BuildClasDesc
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

struct DecodePushConstant
{
    uint numHeaders;
    uint indexBufferBaseAddressLowBits;
    uint indexBufferBaseAddressHighBits;
    uint vertexBufferBaseAddressLowBits;
    uint vertexBufferBaseAddressHighBits;
};

#define GLOBALS_VERTEX_BUFFER_OFFSET_INDEX 0
#define GLOBALS_INDEX_BUFFER_OFFSET_INDEX  1
#define GLOBALS_CLAS_COUNT_INDEX           2
#define GLOBALS_BLAS_COUNT_INDEX           3

#ifdef __cplusplus
}
#endif

#endif
