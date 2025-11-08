#include "photon_mapping.h"
#include "../shader_interop/sort_shaderinterop.h"
#include "../shader_interop/kd_tree_shaderinterop.h"
#include "../thread_context.h"
namespace rt
{

PhotonMapper::PhotonMapper(Arena *arena)
{
    sortPush.size   = sizeof(u32);
    sortPush.stage  = ShaderStage::Compute;
    sortPush.offset = 0;

    // upsweep
    string shaderName = "../src/shaders/upsweep.spv";
    string shaderData = OS_ReadFile(arena, shaderName);
    Shader shader =
        device->CreateShader(ShaderStage::Compute, "radix sort upsweep", shaderData);

    for (int i = 0; i <= 3; i++)
    {
        upsweepLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                 VK_SHADER_STAGE_COMPUTE_BIT);
    }

    upsweepPipeline = device->CreateComputePipeline(&shader, &upsweepLayout, &sortPush,
                                                    "radix sort upsweep pipeline");

    // spine
    shaderName = "../src/shaders/spine.spv";
    shaderData = OS_ReadFile(arena, shaderName);
    shader     = device->CreateShader(ShaderStage::Compute, "radix sort spine", shaderData);

    for (int i = 0; i <= 2; i++)
    {
        spineLayout.AddBinding(i, DescriptorType::StorageBuffer, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    spinePipeline = device->CreateComputePipeline(&shader, &spineLayout, &sortPush,
                                                  "radix sort spine pipeline");

    // downsweep
    shaderName = "../src/shaders/downsweep.spv";
    shaderData = OS_ReadFile(arena, shaderName);
    shader = device->CreateShader(ShaderStage::Compute, "radix sort downsweep", shaderData);

    for (int i = 0; i <= 6; i++)
    {
        downsweepLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                   VK_SHADER_STAGE_COMPUTE_BIT);
    }

    downsweepPipeline = device->CreateComputePipeline(&shader, &downsweepLayout, &sortPush,
                                                      "radix sort downsweep pipeline");

    // update tags
    shaderName = "../src/shaders/update_tags.spv";
    shaderData = OS_ReadFile(arena, shaderName);
    shader =
        device->CreateShader(ShaderStage::Compute, "kd tree build update tags", shaderData);

    for (int i = 0; i <= 5; i++)
    {
        updateTagsLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                    VK_SHADER_STAGE_COMPUTE_BIT);
    }

    updateTagsPipeline = device->CreateComputePipeline(&shader, &updateTagsLayout, &sortPush,
                                                       "kd tree build update tags pipeline");

    // calculate bounds
    shaderName = "../src/shaders/calculate_bounds.spv";
    shaderData = OS_ReadFile(arena, shaderName);
    shader =
        device->CreateShader(ShaderStage::Compute, "kd tree calculate bounds", shaderData);

    for (int i = 0; i <= 2; i++)
    {
        calculateBoundsLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    }

    calculateBoundsPipeline = device->CreateComputePipeline(&shader, &calculateBoundsLayout, 0,
                                                            "kd tree calculate bounds");

    // finalize bounds
    shaderName = "../src/shaders/finalize_calculate_bounds.spv";
    shaderData = OS_ReadFile(arena, shaderName);
    shader = device->CreateShader(ShaderStage::Compute, "kd tree finalize calculate bounds",
                                  shaderData);

    for (int i = 0; i <= 2; i++)
    {
        finalizeCalculateBoundsLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                                 VK_SHADER_STAGE_COMPUTE_BIT);
    }

    finalizeCalculateBoundsPipeline = device->CreateComputePipeline(
        &shader, &finalizeCalculateBoundsLayout, 0, "kd tree finalize calculate bounds");

    // prepare indirect
    shaderName = "../src/shaders/kd_tree_prepare_indirect.spv";
    shaderData = OS_ReadFile(arena, shaderName);
    shader =
        device->CreateShader(ShaderStage::Compute, "kd tree prepare indirect", shaderData);

    for (int i = 0; i <= 1; i++)
    {
        prepareIndirectLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    }

    prepareIndirectPipeline = device->CreateComputePipeline(
        &shader, &prepareIndirectLayout, &sortPush, "kd tree prepare indirect");

    // create sort keys
    shaderName = "../src/shaders/create_sort_keys.spv";
    shaderData = OS_ReadFile(arena, shaderName);
    shader =
        device->CreateShader(ShaderStage::Compute, "kd tree create sort keys", shaderData);

    for (int i = 0; i <= 3; i++)
    {
        createSortKeysLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    }

    createSortKeysPipeline = device->CreateComputePipeline(&shader, &createSortKeysLayout, 0,
                                                           "kd tree create sort keys");

    // initialize indices
    shaderName = "../src/shaders/initialize_indices.spv";
    shaderData = OS_ReadFile(arena, shaderName);
    shader =
        device->CreateShader(ShaderStage::Compute, "kd tree initialize indices", shaderData);

    for (int i = 0; i <= 0; i++)
    {
        initializeIndicesLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                           VK_SHADER_STAGE_COMPUTE_BIT);
    }

    initializeIndicesPipeline = device->CreateComputePipeline(
        &shader, &initializeIndicesLayout, 0, "kd tree initialize indices pipeline");

    // generate random points
    shaderName = "../src/shaders/generate_random_points.spv";
    shaderData = OS_ReadFile(arena, shaderName);
    shader = device->CreateShader(ShaderStage::Compute, "generate random points", shaderData);

    for (int i = 0; i <= 1; i++)
    {
        generateRandomPointsLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                              VK_SHADER_STAGE_COMPUTE_BIT);
    }
    generateRandomPointsPipeline = device->CreateComputePipeline(
        &shader, &generateRandomPointsLayout, 0, "generate random points pipeline");

    // buffers
    RenderGraph *rg = GetRenderGraph();
    kdTreeDimensionsBuffer =
        device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, maxPhotons * sizeof(u32));
    kdTreeDimensionsBufferHandle =
        rg->RegisterExternalResource("kd tree dimensions", &kdTreeDimensionsBuffer);

    kdTreeTags   = rg->CreateBufferResource("kd tree build tags 0",
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                            sizeof(u64) * maxPhotons);
    sortTempTags = rg->CreateBufferResource(
        "kd tree build tags 1", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(u64) * maxPhotons);
    tempIndices0 = rg->CreateBufferResource(
        "kd tree indices 0", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(u32) * maxPhotons);
    tempIndices1 = rg->CreateBufferResource(
        "kd tree indices 1", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(u32) * maxPhotons);

    //
    // VkDeviceSize InoutSize(uint32_t elementCount) { return Align(elementCount *
    // sizeof(uint32_t), 16); }

    globalHistogramBuffer = rg->CreateBufferResource("sort global histogram buffer",
                                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                     sizeof(u32) * (8 * RADIX + 4));

    u32 histogramSize =
        (4 + (maxPhotons + PARTITION_SIZE - 1) / PARTITION_SIZE * RADIX) * sizeof(u32);
    partitionHistogram = rg->CreateBufferResource(
        "sort partition histogram buffer", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, histogramSize);

    photonPositionsBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                 maxPhotons * sizeof(Vec3f));
    photonPositionsBufferHandle =
        rg->RegisterExternalResource("photon positions", &photonPositionsBuffer);

    photonBoundsBuffer             = rg->CreateBufferResource("photon mapping photon bounds",
                                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                              sizeof(Vec3f) * 2);
    photonIntermediateBoundsBuffer = rg->CreateBufferResource(
        "photon mapping intermediate bounds",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(Vec3f) * 2 * (maxPhotons >> (1 + KD_TREE_REDUCTION_BITS)));
    numPhotonsBuffer = rg->CreateBufferResource(
        "photon mapping num photons", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(u32));

    indirectBuffer = rg->CreateBufferResource("photon mapping indirect buffer",
                                              VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                              sizeof(u32) * KD_TREE_INDIRECT_SIZE);
}

void PhotonMapper::Sort()
{
    RenderGraph *rg = GetRenderGraph();
    ScratchArena scratch;

    for (int i = 0; i < 4; i++)
    {
        ResourceHandle keys0 = (i & 1) ? sortTempTags : kdTreeTags;
        ResourceHandle keys1 = (i & 1) ? kdTreeTags : sortTempTags;

        ResourceHandle values0 = (i & 1) ? tempIndices1 : tempIndices0;
        ResourceHandle values1 = (i & 1) ? tempIndices0 : tempIndices1;
        // upsweep
        string str = PushStr8F(scratch.temp.arena, "upsweep %u", i);
        // rg->StartIndirectComputePass(str,
        u32 numGroups = (maxPhotons + PARTITION_SIZE - 1) / PARTITION_SIZE;
        rg->StartComputePass(
              upsweepPipeline, upsweepLayout, 4,
              // indirectBuffer, sizeof(u32) * SORT_KEYS_INDIRECT_X,
              [&, numGroups](CommandBuffer *cmd) {
                  cmd->Dispatch(numGroups, 1, 1);
                  cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                  cmd->FlushBarriers();
              },
              &sortPush, &i)
            .AddHandle(numPhotonsBuffer, ResourceUsageType::Read)
            .AddHandle(globalHistogramBuffer, ResourceUsageType::Write)
            .AddHandle(partitionHistogram, ResourceUsageType::Write)
            .AddHandle(keys0, ResourceUsageType::Read);

        // spine
        str = PushStr8F(scratch.temp.arena, "spine %u", i);
        string spineName;
        spineName.str  = rg->Allocate(str.str, str.size);
        spineName.size = str.size;
        rg->StartComputePass(spinePipeline, spineLayout, 3,
                             [spineName](CommandBuffer *cmd) {
                                 device->BeginEvent(cmd, spineName);
                                 cmd->Dispatch(RADIX, 1, 1);
                                 cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                              VK_ACCESS_2_SHADER_WRITE_BIT,
                                              VK_ACCESS_2_SHADER_READ_BIT);
                                 cmd->FlushBarriers();
                                 device->EndEvent(cmd);
                             })
            .AddHandle(numPhotonsBuffer, ResourceUsageType::Read)
            .AddHandle(globalHistogramBuffer, ResourceUsageType::RW)
            .AddHandle(partitionHistogram, ResourceUsageType::RW);

        // downsweep
        str = PushStr8F(scratch.temp.arena, "downsweep %u", i);
        // rg->StartIndirectComputePass(str,
        rg->StartComputePass(downsweepPipeline, downsweepLayout, 7,
                             // indirectBuffer, sizeof(u32) * SORT_KEYS_INDIRECT_X,
                             [numGroups](CommandBuffer *cmd) {
                                 cmd->Dispatch(numGroups, 1, 1);
                                 cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                              VK_ACCESS_2_SHADER_WRITE_BIT,
                                              VK_ACCESS_2_SHADER_READ_BIT);
                                 cmd->FlushBarriers();
                             })
            .AddHandle(numPhotonsBuffer, ResourceUsageType::Read)
            .AddHandle(globalHistogramBuffer, ResourceUsageType::Read)
            .AddHandle(partitionHistogram, ResourceUsageType::Read)
            .AddHandle(keys0, ResourceUsageType::Read)
            .AddHandle(keys1, ResourceUsageType::Write)
            .AddHandle(values0, ResourceUsageType::Read)
            .AddHandle(values1, ResourceUsageType::Write);
    }
}

void PhotonMapper::BuildKDTree()
{
    RenderGraph *rg = GetRenderGraph();

    rg->StartComputePass(generateRandomPointsPipeline, generateRandomPointsLayout, 2,
                         [&](CommandBuffer *cmd) {
                             cmd->Dispatch((1u << 15u), 1, 1);
                             cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          VK_ACCESS_2_SHADER_WRITE_BIT,
                                          VK_ACCESS_2_SHADER_READ_BIT);
                             cmd->FlushBarriers();
                         })
        .AddHandle(photonPositionsBufferHandle, ResourceUsageType::Write)
        .AddHandle(photonBoundsBuffer, ResourceUsageType::Write);

    // Get the KD tree bounds
    // rg->StartIndirectComputePass( "Calculate Photon Bounds",
    rg->StartComputePass(calculateBoundsPipeline, calculateBoundsLayout, 3,
                         // indirectBuffer,  sizeof(u32) * KD_TREE_INDIRECT_X,
                         [&](CommandBuffer *cmd) {
                             cmd->Dispatch((1u << 10u), 1, 1);
                             cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          VK_ACCESS_2_SHADER_WRITE_BIT,
                                          VK_ACCESS_2_SHADER_READ_BIT);
                             cmd->FlushBarriers();
                         })
        .AddHandle(photonIntermediateBoundsBuffer, ResourceUsageType::Write)
        .AddHandle(photonPositionsBufferHandle, ResourceUsageType::Read)
        .AddHandle(numPhotonsBuffer, ResourceUsageType::Read);

    rg->StartComputePass(finalizeCalculateBoundsPipeline, finalizeCalculateBoundsLayout, 3,
                         [&](CommandBuffer *cmd) {
                             cmd->Dispatch(1, 1, 1);
                             cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          VK_ACCESS_2_SHADER_WRITE_BIT,
                                          VK_ACCESS_2_SHADER_READ_BIT);
                             cmd->FlushBarriers();
                         })
        .AddHandle(photonBoundsBuffer, ResourceUsageType::Write)
        .AddHandle(photonIntermediateBoundsBuffer, ResourceUsageType::Read)
        .AddHandle(numPhotonsBuffer, ResourceUsageType::Read);

    // Build left balanced KD tree
    // TODO need to readback N

    u32 N = maxPhotons;
    u32 L = Log2Int(N) + 1;

    u32 offset = 0;
    ScratchArena scratch;

    rg->StartPass(2,
                  [&](CommandBuffer *cmd) {
                      RenderGraph *rg   = GetRenderGraph();
                      GPUBuffer *buffer = rg->GetBuffer(kdTreeTags);
                      cmd->ClearBuffer(buffer);
                      buffer = rg->GetBuffer(kdTreeDimensionsBufferHandle);
                      cmd->ClearBuffer(buffer);
                      cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT);
                      cmd->FlushBarriers();
                  })
        .AddHandle(kdTreeTags, ResourceUsageType::Write)
        .AddHandle(kdTreeDimensionsBufferHandle, ResourceUsageType::Write);

    rg->StartComputePass(initializeIndicesPipeline, initializeIndicesLayout, 1,
                         [&](CommandBuffer *cmd) {
                             cmd->Dispatch(maxPhotons / 32u, 1, 1);
                             cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          VK_ACCESS_2_SHADER_WRITE_BIT,
                                          VK_ACCESS_2_SHADER_READ_BIT);
                             cmd->FlushBarriers();
                         })
        .AddHandle(tempIndices0, ResourceUsageType::Write);

    // for (u32 i = 0; i < L - 1; i++)
    u32 i = 0;
    {
        // Prepare Indirect
        // rg->StartComputePass(prepareIndirectPipeline, prepareIndirectLayout, 2,
        //                      [](CommandBuffer *cmd) {
        //                          cmd->Dispatch(1, 1, 1);
        //                          cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        //                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
        //                                           VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
        //                                       VK_ACCESS_2_SHADER_WRITE_BIT,
        //                                       VK_ACCESS_2_SHADER_READ_BIT |
        //                                           VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
        //                          cmd->FlushBarriers();
        //                      })
        //     .AddHandle(numPhotonsBuffer, ResourceUsageType::RW)
        //     .AddHandle(indirectBuffer, ResourceUsageType::RW);

        // Create sort keys
        string str = PushStr8F(scratch.temp.arena, "Photon Mapping: Create Sort Keys %u", i);
        // rg->StartIndirectComputePass(str,
        rg->StartComputePass(createSortKeysPipeline, createSortKeysLayout, 4,
                             // indirectBuffer, sizeof(u32) * KD_TREE_INDIRECT_X,
                             [&](CommandBuffer *cmd) {
                                 // TODO: hardcoded
                                 cmd->Dispatch((1u << 15u), 1, 1);
                                 cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                              VK_ACCESS_2_SHADER_WRITE_BIT,
                                              VK_ACCESS_2_SHADER_READ_BIT);
                                 cmd->FlushBarriers();
                             })
            .AddHandle(photonPositionsBufferHandle, ResourceUsageType::Read)
            .AddHandle(kdTreeDimensionsBufferHandle, ResourceUsageType::Read)
            .AddHandle(kdTreeTags, ResourceUsageType::RW, -1, offset)
            .AddHandle(tempIndices0, ResourceUsageType::Read, -1, offset);

        Sort();

        // Update Tags
        str = PushStr8F(scratch.temp.arena, "Photon Mapping: Update Tags %u", i);
        // rg->StartIndirectComputePass(str,
        rg->StartComputePass(
              updateTagsPipeline, updateTagsLayout, 5,
              // indirectBuffer, sizeof(u32) * KD_TREE_INDIRECT_X,
              [](CommandBuffer *cmd) {
                  cmd->Dispatch(maxPhotons / 32, 1, 1);
                  cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                  cmd->FlushBarriers();
              },
              &sortPush, &i)
            .AddHandle(kdTreeTags, ResourceUsageType::RW)
            .AddHandle(tempIndices0, ResourceUsageType::Read)
            .AddHandle(kdTreeDimensionsBufferHandle, ResourceUsageType::RW)
            .AddHandle(photonPositionsBufferHandle, ResourceUsageType::Read)
            .AddHandle(photonBoundsBuffer, ResourceUsageType::Read);

        rg->StartPass(0, [&](CommandBuffer *cmd) {
            // if (debug) // numBlas - 1)
            {
                RenderGraph *rg     = GetRenderGraph();
                GPUBuffer *buffer   = rg->GetBuffer(kdTreeTags);
                GPUBuffer readback0 = device->CreateBuffer(
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT, buffer->size, MemoryUsage::GPU_TO_CPU);

                cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                             VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                             VK_ACCESS_2_TRANSFER_READ_BIT);
                cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                             VK_ACCESS_2_TRANSFER_READ_BIT);
                // cmd->Barrier(&imageOut, VK_IMAGE_LAYOUT_GENERAL,
                //              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                // VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                //              VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                // VK_ACCESS_2_SHADER_WRITE_BIT,
                //              VK_ACCESS_2_TRANSFER_READ_BIT);

                //
                //
                //
                // cmd->Barrier(VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                //     // VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                //     // VK_ACCESS_2_SHADER_WRITE_BIT,
                //     //              VK_ACCESS_2_TRANSFER_READ_BIT);
                // cmd->FlushBarriers();

                // BufferImageCopy copy = {};
                // copy.extent =
                //     Vec3u(filterWeightsImage.desc.width,
                // filterWeightsImage.desc.height,
                // 1);

                cmd->CopyBuffer(&readback0, buffer);
                // cmd->CopyBuffer(&readback2, buffer1);
                // cmd->CopyImageToBuffer(&readback0,
                // &filterWeightsImage,
                // &copy, 1);
                Semaphore testSemaphore   = device->CreateSemaphore();
                testSemaphore.signalValue = 1;
                cmd->SignalOutsideFrame(testSemaphore);
                device->SubmitCommandBuffer(cmd);
                device->Wait(testSemaphore);

                u64 *data = (u64 *)readback0.mappedPtr;

                int stop = 5;
            }
        });

        offset += (1u << i);
    }
}

} // namespace rt
