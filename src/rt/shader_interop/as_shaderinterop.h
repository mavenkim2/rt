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

#ifdef __cplusplus
}
#endif

#endif
