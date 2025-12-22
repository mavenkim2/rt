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

    PushConstant push;
    DescriptorSetLayout initializeVMMLayout = {};
    VkPipeline initializeVMMPipeline;

    PathGuider(Arena *arena);
    void PathGuiding();
};

} // namespace rt

#endif
