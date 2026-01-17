#ifndef GPU_PATH_GUIDING_H_
#define GPU_PATH_GUIDING_H_

#include "../base.h"
#include "path_guiding_util.h"
#include "device.h"

namespace rt
{

enum PathGuidingKernels : int
{
    PATH_GUIDING_KERNEL_INITIALIZE_SAMPLES,
    PATH_GUIDING_KERNEL_UPDATE_MIXTURE,
    PATH_GUIDING_KERNEL_PARTIAL_UPDATE_MIXTURE,
    PATH_GUIDING_KERNEL_UPDATE_SPLIT_STATISTICS,
    PATH_GUIDING_KERNEL_SPLIT_COMPONENTS,
    PATH_GUIDING_KERNEL_MERGE_COMPONENTS,
    PATH_GUIDING_KERNEL_UPDATE_COMPONENT_DISTANCES,
    PATH_GUIDING_KERNEL_REPROJECT_SAMPLES,
    PATH_GUIDING_KERNEL_GET_SAMPLE_BOUNDS,
    PATH_GUIDING_KERNEL_CALCULATE_SPLIT_LOCATIONS,
    PATH_GUIDING_KERNEL_BEGIN_LEVEL,
    PATH_GUIDING_KERNEL_CALCULATE_CHILD_INDICES,
    PATH_GUIDING_KERNEL_CREATE_WORK_ITEMS,
    PATH_GUIDING_KERNEL_CALCULATE_NODE_STATISTICS,
    PATH_GUIDING_KERNEL_BUILD_KDTREE,
    PATH_GUIDING_KERNEL_GET_CHILD_NODE_OFFSET,
    PATH_GUIDING_KERNEL_FIND_LEAF_NODES,
    PATH_GUIDING_KERNEL_WRITE_SAMPLES_SOA,
    PATH_GUIDING_KERNEL_PRINT_STATS,

    PATH_GUIDING_KERNEL_MAX,
};

static const string pathGuidingKernelNames[] = {
    "InitializeSamples",
    "UpdateMixture",
    "PartialUpdateMixture",
    "UpdateSplitStatistics",
    "SplitComponents",
    "MergeComponents",
    "UpdateComponentDistances",
    "ReprojectSamples",
    "GetSampleBounds",
    "CalculateSplitLocations",
    "BeginLevel",
    "CalculateChildIndices",
    "CreateWorkItems",
    "CalculateNodeStatistics",
    "BuildKDTree",
    "GetChildNodeOffset",
    "FindLeafNodes",
    "WriteSamplesToSOA",
    "PrintStatistics",
};

struct PathGuiding
{
    // TODO: hardcoded
    const uint32_t numSamples  = 1u << 22u;
    const uint32_t maxNumNodes = 1u << 16u;

    GPUArena *gpuArena;
    Device *device;
    KernelHandle handles[PATH_GUIDING_KERNEL_MAX];
    bool initialized;

    // CPU + GPU visible memory
    uint32_t *numSamplesBuffer;

    // GPU visible memory
    KDTreeBuildState *treeBuildState;
    KDTreeNode *nodes;
    SampleStatistics *sampleStatistics;
    Bounds3f *rootBounds;

    PathGuiding(Device *device);
    void Update();
};

} // namespace rt

#endif
