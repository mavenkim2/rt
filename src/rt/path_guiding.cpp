#include "path_guiding.h"
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

    RenderGraph *rg = GetRenderGraph();
    vmmsBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(VMM) * 32);
    vmmsBufferHandle = rg->RegisterExternalResource("vmms buffer", &vmmsBuffer);

    // string shaderName = "../src/shaders/wem.spv";
    // string shaderData = OS_ReadFile(arena, shaderName);
    // Shader shader =
    //     device->CreateShader(ShaderStage::Compute, "weighted expectation step", shaderData);
}

void PathGuider::PathGuiding()
{
    RenderGraph *rg = GetRenderGraph();
    u32 numVmms     = 32;
    rg->StartComputePass(
          initializeVMMPipeline, initializeVMMLayout, 1,
          [&](CommandBuffer *cmd) {
              cmd->Dispatch(1, 1, 1);
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
              cmd->FlushBarriers();

              {
                  // RenderGraph *rg   = GetRenderGraph();
                  GPUBuffer readback0 =
                      device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, vmmsBuffer.size,
                                           MemoryUsage::GPU_TO_CPU);
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

                  cmd->CopyBuffer(&readback0, &vmmsBuffer);
                  // cmd->CopyBuffer(&readback2, buffer1);
                  // cmd->CopyImageToBuffer(&readback0, img, &copy, 1);
                  Semaphore testSemaphore   = device->CreateSemaphore();
                  testSemaphore.signalValue = 1;
                  cmd->SignalOutsideFrame(testSemaphore);
                  device->SubmitCommandBuffer(cmd);
                  device->Wait(testSemaphore);

                  VMM *data = (VMM *)readback0.mappedPtr;
                  int stop  = 5;
              }
          },
          &push, &numVmms)
        .AddHandle(vmmsBufferHandle, ResourceUsageType::Write);

    ScratchArena scratch;
    // StaticArray<PathGuidingSample> samples(scratch.temp.arena, 32);
    // for (u32 i = 0; i < 64; i++)
    // {
    //     PathGuidingSample sample;
    //     sample.vmmIndex = i % 32;
    //     samples.Push(sample);
    // }
}

} // namespace rt
