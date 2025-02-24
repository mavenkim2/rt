#ifndef THREAD_STATISTICS_H
#define THREAD_STATISTICS_H
namespace rt
{
// NOTE: all member have to be u64
struct alignas(CACHE_LINE_SIZE) ThreadStatistics
{
    // u64 rayPrimitiveTests;
    // u64 rayAABBTests;
    u64 bvhIntersectionTime;
    u64 primitiveIntersectionTime;
    u64 integrationTime;
    u64 samplingTime;

    u64 misc;
    u64 misc2;
    u64 misc3;
    u64 misc4;
    f64 miscF;
};

struct ThreadMemoryStatistics
{
    u64 totalFileMemory;
    u64 totalShapeMemory;
    u64 totalMaterialMemory;
    u64 totalTextureMemory;
    u64 totalLightMemory;
    u64 totalInstanceMemory;
    u64 totalStringMemory;
    u64 totalBVHMemory;
    u64 totalOtherMemory;
};

static ThreadStatistics *threadLocalStatistics;
static ThreadMemoryStatistics *threadMemoryStatistics;
} // namespace rt
#endif
