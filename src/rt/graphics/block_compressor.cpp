#include "block_compressor.h"
#include <functional>

namespace rt
{

BlockCompressor::BlockCompressor(u32 gpuSubmissionWidth, VkFormat inFormat, VkFormat outFormat,
                                 u32 numMips)
    : gpuSubmissionWidth(gpuSubmissionWidth), inputFormat(inFormat), outputFormat(outFormat)
{

    submissionSize = Sqr(gpuSubmissionWidth) * GetFormatSize(inFormat);

    u32 blockShift = GetBlockShift(outFormat);
    gpuOutputWidth = gpuSubmissionWidth >> blockShift;

    outputSize   = 0;
    u32 mipWidth = gpuOutputWidth;
    for (int i = 0; i < numMips - blockShift; i++)
    {
        outputSize += Sqr(mipWidth) * GetFormatSize(outFormat);
        mipWidth >>= 1;
    }

    // Allocate GPU resources
    ImageDesc blockCompressedImageDesc(
        ImageType::Type2D, gpuSubmissionWidth, gpuSubmissionWidth, 1, numMips, 1, baseFormat,
        MemoryUsage::GPU_ONLY, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_TILING_OPTIMAL);
    gpuSrcImages[0] = device->CreateImage(blockCompressedImageDesc);
    gpuSrcImages[1] = device->CreateImage(blockCompressedImageDesc);

    ImageDesc uavDesc(ImageType::Type2D, gpuOutputWidth, gpuOutputWidth, 1,
                      numMips - blockShift, 1, VK_FORMAT_R32G32_UINT, MemoryUsage::GPU_ONLY,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                      VK_IMAGE_TILING_OPTIMAL);

    uavImages[0] = device->CreateImage(uavDesc);
    uavImages[1] = device->CreateImage(uavDesc);

    submissionBuffers[0] = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                submissionSize, MemoryUsage::CPU_TO_GPU);
    submissionBuffers[1] = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                submissionSize, MemoryUsage::CPU_TO_GPU);

    Assert(submissionBuffers[0].mappedPtr && submissionBuffers[1].mappedPtr);
    mappedPtrs[0] = submissionBuffers[0].mappedPtr;
    mappedPtrs[1] = submissionBuffers[1].mappedPtr;

    readbackBuffers[0] = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, outputSize,
                                              MemoryUsage::GPU_TO_CPU);
    readbackBuffers[1] = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, outputSize,
                                              MemoryUsage::GPU_TO_CPU);

    descriptorSets[0] = bcLayout.CreateNewDescriptorSet();
    descriptorSets[1] = bcLayout.CreateNewDescriptorSet();

    semaphores[0] = device->CreateSemaphore();
    semaphores[1] = device->CreateSemaphore();

    cmds[0] = device->BeginCommandBuffer(QueueType_Compute);
    cmds[0]->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    cmds[1] = device->BeginCommandBuffer(QueueType_Compute);
    cmds[1]->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    ScratchArena scratch;
    string shaderName = "../src/shaders/block_compress.spv";
    string data       = OS_ReadFile(scratch.temp.arena, shaderName);
    shader            = device->CreateShader(ShaderStage::Compute, "block_compress", data);

    bcLayout = {};
    inputBinding =
        bcLayout.AddBinding(0, DescriptorType::SampledImage, VK_SHADER_STAGE_COMPUTE_BIT);
    outputBinding =
        bcLayout.AddBinding(1, DescriptorType::StorageImage, VK_SHADER_STAGE_COMPUTE_BIT);
    bcLayout.AddImmutableSamplers();
    pipeline = device->CreateComputePipeline(&shader, &bcLayout);
}

void BlockCompressor::SubmitBlockCompressedCommands(u8 *in)
{
    // Submission currently packed image to GPU
    GPUImage *srcImage        = &gpuSrcImages[submissionIndex];
    GPUImage *uavImage        = &uavImages[submissionIndex];
    GPUBuffer *transferBuffer = &submissionBuffers[submissionIndex];
    GPUBuffer *readbackBuffer = &readbackBuffers[submissionIndex];
    DescriptorSet *set        = &descriptorSets[submissionIndex];
    CommandBuffer *cmd        = cmds[submissionIndex];

    BufferImageCopy copy = {};
    copy.layerCount      = 1;
    copy.extent          = Vec3u(gpuSubmissionWidth, gpuSubmissionWidth, 1);

    // Debug copy
    {
        MemoryCopy((u8 *)mappedPtrs[submissionIndex], in, submissionSize);
    }

    // Copy from buffer to src image
    {
        cmd->Barrier(srcImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        cmd->FlushBarriers();
        cmd->CopyImage(transferBuffer, srcImage, &copy, 1);
    }

    // Block compress
    {
        cmd->Barrier(srcImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        cmd->Barrier(uavImage, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_WRITE_BIT);
        cmd->FlushBarriers();

        set->Bind(inputBinding, srcImage);
        set->Bind(outputBinding, uavImage);
        cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, set, bcLayout.pipelineLayout);
        cmd->Dispatch((gpuSubmissionWidth + 7) >> 3, (gpuSubmissionWidth + 7) >> 3, 1);
    }

    // Copy from uav to buffer
    {
        ScratchArena scratch;
        u32 numMips             = uavImages[0].desc.numMips;
        BufferImageCopy *copies = PushArray(scratch.temp.arena, BufferImageCopy, numMips);
        u32 bufferOffset        = 0;
        u32 outputMipWidth      = gpuOutputWidth;
        for (int i = 0; i < numMips; i++)
        {
            BufferImageCopy &copy = copies[i];
            copy.bufferOffset     = bufferOffset;
            copy.mipLevel         = i;
            copy.layerCount       = 1;
            copy.extent           = Vec3u(outputMipWidth, outputMipWidth, 1);

            outputMipWidth >>= 1
        }
        cmd->Barrier(uavImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        cmd->FlushBarriers();
        cmd->CopyImageToBuffer(readbackBuffer, uavImage, copies, numMips);
    }

    // Submit command buffer
    semaphores[submissionIndex].signalValue++;
    cmd->SignalOutsideFrame(semaphores[submissionIndex]);
    device->SubmitCommandBuffer(cmds[submissionIndex], false, true);
    cmds[submissionIndex] = device->BeginCommandBuffer(QueueType_Compute);
    cmds[submissionIndex]->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    numSubmissions++;
    submissionIndex = numSubmissions & 1;
}

void BlockCompressor::CopyBlockCompressedResultsToDisk(BlockCopyFunction &func)
{
    if (numSubmissions > 0)
    {
        int lastSubmissionIndex   = (submissionIndex - 1) & 1;
        GPUBuffer *readbackBuffer = &readbackBuffers[lastSubmissionIndex];
        device->Wait(semaphores[lastSubmissionIndex]);

        descriptorSets[lastSubmissionIndex].Reset();

        func(readbackBuffers[lastSubmissionIndex.mappedPtr], lastSubmissionIndex);
    }
}

} // namespace rt
