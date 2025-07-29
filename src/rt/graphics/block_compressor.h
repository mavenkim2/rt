#ifndef BLOCK_COMPRESSOR_H_
#define BLOCK_COMPRESSOR_H_

#include <functional>
#include "vulkan.h"

namespace rt
{

using BlockCopyFunction = std::function<void(u8 *, int)>;

struct SubmissionInfo
{
    u32 numSubmissionX;
    u32 numSubmissionY;
    u32 width;
    u32 height;
};

struct BlockCompressor
{
    GPUImage gpuSrcImages[2];
    GPUImage uavImages[2];
    GPUBuffer submissionBuffers[2];
    void *mappedPtrs[2];
    GPUBuffer readbackBuffers[2];
    DescriptorSet descriptorSets[2];
    Semaphore semaphores[2];
    CommandBuffer *cmds[2];
    u32 numSubmissions  = 0;
    u32 submissionIndex = 0;

    u32 gpuSubmissionWidth;
    u32 submissionSize;
    u32 gpuOutputWidth;
    u32 outputSize;

    VkFormat inputFormat;
    VkFormat outputFormat;

    DescriptorSetLayout bcLayout;
    int inputBinding;
    int outputBinding;
    VkPipeline pipeline;
    Shader shader;

    BlockCompressor(u32 gpuSubmissionWidth, VkFormat inFormat, VkFormat outFormat,
                    u32 numMips = 0);
    void CopyBlockCompressedResultsToDisk(const BlockCopyFunction &func);
    void SubmitBlockCompressedCommands(u8 *in);
};

} // namespace rt

#endif
