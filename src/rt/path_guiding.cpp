#include "path_guiding.h"
#include "sampling.h"
#include "shader_interop/path_guiding_shaderinterop.h"
#include "thread_context.h"

namespace rt
{

PathGuider::PathGuider(Arena *arena)
{
    string shaderName = "../src/shaders/initialize_vmms.spv";
    string shaderData = OS_ReadFile(arena, shaderName);
    Shader shader = device->CreateShader(ShaderStage::Compute, "initialize vmms", shaderData);

    push.size   = sizeof(u32);
    push.stage  = ShaderStage::Compute;
    push.offset = 0;
    initializeVMMLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                   VK_SHADER_STAGE_COMPUTE_BIT);
    initializeVMMPipeline =
        device->CreateComputePipeline(&shader, &initializeVMMLayout, &push, "initialize vmms");

    shaderName = "../src/shaders/prefix_sum.spv";
    shaderData = OS_ReadFile(arena, shaderName);
    shader     = device->CreateShader(ShaderStage::Compute, "prefix sum", shaderData);

    for (int i = 0; i <= 1; i++)
    {
        prefixSumLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                   VK_SHADER_STAGE_COMPUTE_BIT);
    }
    prefixSumPipeline =
        device->CreateComputePipeline(&shader, &prefixSumLayout, &push, "prefix sum pipeline");

    shaderName = "../src/shaders/wem.spv";
    shaderData = OS_ReadFile(arena, shaderName);
    shader     = device->CreateShader(ShaderStage::Compute, "wem", shaderData);

    for (int i = 0; i <= 5; i++)
    {
        wemLayout.AddBinding(i, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);
    }
    wemPipeline = device->CreateComputePipeline(&shader, &wemLayout, &push, "wem pipeline");

    RenderGraph *rg = GetRenderGraph();
    u32 numVMMs     = 32;
    u32 numSamples  = 64;
    vmmsBuffer =
        device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(VMM) * numVMMs);
    vmmsBufferHandle = rg->RegisterExternalResource("vmms buffer", &vmmsBuffer);

    // path guiding samples
    sampleVMMIndicesBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                  sizeof(u32) * numSamples);
    sampleVMMIndicesBufferHandle =
        rg->RegisterExternalResource("sample vmm indices buffer", &sampleVMMIndicesBuffer);

    sampleDirectionsBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                  sizeof(Vec3f) * numSamples);
    sampleDirectionsBufferHandle =
        rg->RegisterExternalResource("sample directions buffer", &sampleDirectionsBuffer);

    // end
    vmmOffsetsHandle = rg->CreateBufferResource(
        "vmm offsets", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(u32) * numVMMs);
    vmmCountsHandle = rg->CreateBufferResource(
        "vmm counts", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(u32) * numVMMs);
    statisticsHandle =
        rg->CreateBufferResource("sample statistics", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 sizeof(Statistics) * numSamples);
}

void PathGuider::PathGuiding()
{
    RenderGraph *rg = GetRenderGraph();
    u32 numVmms     = 32;
    u32 numSamples  = 64;

    rg->StartComputePass(
          initializeVMMPipeline, initializeVMMLayout, 1,
          [&](CommandBuffer *cmd) {
              cmd->Dispatch(1, 1, 1);
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
              cmd->FlushBarriers();
          },
          &push, &numVmms)
        .AddHandle(vmmsBufferHandle, ResourceUsageType::Write);

    ScratchArena scratch;
    StaticArray<u32> sampleVMMIndices(scratch.temp.arena, 64);
    StaticArray<Vec3f> sampleDirections(scratch.temp.arena, 64);

    RNG rng;
    for (u32 i = 0; i < 64; i++)
    {
        u32 vmmIndex = i % 32;
        Vec3f dir    = SampleUniformSphere(Vec2f(rng.Uniform<f32>(), rng.Uniform<f32>()));
        sampleVMMIndices.Push(vmmIndex);
        sampleDirections.Push(dir);
    }

    u32 *indices      = rg->Allocate(sampleVMMIndices.data, 64);
    Vec3f *directions = rg->Allocate(sampleDirections.data, 64);
    rg->StartPass(0, [&, indices, numSamples](CommandBuffer *cmd) {
        cmd->SubmitBuffer(&sampleVMMIndicesBuffer, indices, sizeof(u32) * numSamples);
        cmd->SubmitBuffer(&sampleDirectionsBuffer, indices, sizeof(Vec3f) * numSamples);
        cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
    });

    rg->StartComputePass(
          prefixSumPipeline, prefixSumLayout, 2,
          [&](CommandBuffer *cmd) {
              cmd->Dispatch(1, 1, 1);
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
              cmd->FlushBarriers();
          },
          &push, &numSamples)
        .AddHandle(sampleVMMIndicesBufferHandle, ResourceUsageType::Read)
        .AddHandle(vmmOffsetsHandle, ResourceUsageType::Write);

    rg->StartComputePass(
          wemPipeline, wemLayout, 6,
          [&, numSamples](CommandBuffer *cmd) {
              cmd->Dispatch(
                  (numSamples + PATH_GUIDING_GROUP_SIZE - 1) / PATH_GUIDING_GROUP_SIZE, 1, 1);
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
              cmd->FlushBarriers();

              {
                  RenderGraph *rg     = GetRenderGraph();
                  GPUBuffer *buffer   = rg->GetBuffer(statisticsHandle);
                  GPUBuffer readback0 = device->CreateBuffer(
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, buffer->size, MemoryUsage::GPU_TO_CPU);
                  cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                               VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                               VK_ACCESS_2_TRANSFER_READ_BIT);
                  cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                               VK_ACCESS_2_TRANSFER_READ_BIT);
                  // cmd->Barrier(
                  //     img, VK_IMAGE_LAYOUT_GENERAL,
                  //     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  //     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  // VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  //     VK_ACCESS_2_SHADER_WRITE_BIT,
                  // VK_ACCESS_2_TRANSFER_READ_BIT,
                  //     QueueType_Ignored, QueueType_Ignored, level, 1);

                  //
                  // cmd->Barrier(VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                  //     //              VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  //     // VK_ACCESS_2_SHADER_WRITE_BIT,
                  //     //              VK_ACCESS_2_TRANSFER_READ_BIT);
                  cmd->FlushBarriers();

                  // BufferImageCopy copy = {};
                  // copy.mipLevel        = level;
                  // copy.extent          = Vec3u(width, height, 1);

                  cmd->CopyBuffer(&readback0, buffer);
                  // cmd->CopyBuffer(&readback2, buffer1);
                  // cmd->CopyImageToBuffer(&readback0, img, &copy, 1);
                  Semaphore testSemaphore   = device->CreateSemaphore();
                  testSemaphore.signalValue = 1;
                  cmd->SignalOutsideFrame(testSemaphore);
                  device->SubmitCommandBuffer(cmd);
                  device->Wait(testSemaphore);

                  Statistics *data = (Statistics *)readback0.mappedPtr;
                  int stop         = 5;
              }
          })
        .AddHandle(vmmsBufferHandle, ResourceUsageType::Read)
        .AddHandle(sampleDirectionsBufferHandle, ResourceUsageType::Read)
        .AddHandle(sampleVMMIndicesBufferHandle, ResourceUsageType::Read)
        .AddHandle(vmmOffsetsHandle, ResourceUsageType::Read)
        .AddHandle(vmmCountsHandle, ResourceUsageType::Write)
        .AddHandle(statisticsHandle, ResourceUsageType::Write);

    // rg->StartComputePass(prefixSumPipeline, prefixSumLayout, 2,
    //                      [&](CommandBuffer *cmd) {
    //                          cmd->Dispatch(1, 1, 1);
    //                          cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    //                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    //                                       VK_ACCESS_2_SHADER_WRITE_BIT,
    //                                       VK_ACCESS_2_SHADER_READ_BIT);
    //                          cmd->FlushBarriers();
    //                      })
    //     .AddHandle(sampleVMMIndicesBufferHandle, ResourceUsageType::Read)
    //     .AddHandle(vmmOffsetsHandle, ResourceUsageType::Write);
}

} // namespace rt
