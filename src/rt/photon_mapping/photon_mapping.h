#include "../graphics/vulkan.h"
#include "../graphics/render_graph.h"

namespace rt
{
struct Arena;

struct PhotonMapper
{
    static const u32 maxPhotons = (1u << 20u);
    PhotonMapper(Arena *arena);

    ResourceHandle indirectBuffer;

    ResourceHandle numPhotonsBuffer;
    ResourceHandle globalHistogramBuffer;
    ResourceHandle partitionHistogram;
    ResourceHandle kdTreeTags;

    ResourceHandle sortTempTags;
    ResourceHandle tempIndices0;
    ResourceHandle tempIndices1;

    ResourceHandle photonBoundsBuffer;
    ResourceHandle photonIntermediateBoundsBuffer;
    GPUBuffer photonPositionsBuffer;
    ResourceHandle photonPositionsBufferHandle;

    GPUBuffer kdTreeDimensionsBuffer;
    ResourceHandle kdTreeDimensionsBufferHandle;

    // Sort
    PushConstant sortPush;
    DescriptorSetLayout upsweepLayout = {};
    VkPipeline upsweepPipeline;

    DescriptorSetLayout spineLayout = {};
    VkPipeline spinePipeline;

    DescriptorSetLayout downsweepLayout = {};
    VkPipeline downsweepPipeline;

    // KD Tree
    PushConstant updateTagsPush;
    DescriptorSetLayout updateTagsLayout = {};
    VkPipeline updateTagsPipeline;

    DescriptorSetLayout calculateBoundsLayout = {};
    VkPipeline calculateBoundsPipeline;

    DescriptorSetLayout finalizeCalculateBoundsLayout = {};
    VkPipeline finalizeCalculateBoundsPipeline;

    DescriptorSetLayout createSortKeysLayout = {};
    VkPipeline createSortKeysPipeline;

    DescriptorSetLayout prepareIndirectLayout = {};
    VkPipeline prepareIndirectPipeline;

    DescriptorSetLayout initializeIndicesLayout = {};
    VkPipeline initializeIndicesPipeline;

    // temp
    DescriptorSetLayout generateRandomPointsLayout = {};
    VkPipeline generateRandomPointsPipeline;

    void Sort();
    void BuildKDTree();
};

} // namespace rt
