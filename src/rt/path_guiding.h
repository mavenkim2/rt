#ifndef PATH_GUIDING_H_
#define PATH_GUIDING_H_

#include "graphics/vulkan.h"
#include "graphics/render_graph.h"

namespace rt
{

struct PathGuider
{
    GPUBuffer vmmsBuffer;
    ResourceHandle vmmsBufferHandle;

    GPUBuffer sampleVMMIndicesBuffer;
    ResourceHandle sampleVMMIndicesBufferHandle;

    GPUBuffer sampleDirectionsBuffer;
    ResourceHandle sampleDirectionsBufferHandle;

    ResourceHandle vmmOffsetsHandle;
    ResourceHandle vmmCountsHandle;
    ResourceHandle statisticsHandle;

    PushConstant push;
    DescriptorSetLayout initializeVMMLayout = {};
    VkPipeline initializeVMMPipeline;

    DescriptorSetLayout prefixSumLayout = {};
    VkPipeline prefixSumPipeline;

    DescriptorSetLayout wemLayout = {};
    VkPipeline wemPipeline;

    PathGuider(Arena *arena);
    void PathGuiding();
};

} // namespace rt

#endif
