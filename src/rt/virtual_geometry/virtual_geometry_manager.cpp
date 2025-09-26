#include "../bit_packing.h"
#include "../base.h"
#include "../debug.h"
#include "../radix_sort.h"
#include "../shader_interop/as_shaderinterop.h"
#include "../shader_interop/bit_twiddling_shaderinterop.h"
#include "../shader_interop/dense_geometry_shaderinterop.h"
#include "../shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../thread_context.h"
#include "virtual_geometry_manager.h"
#include "../graphics/vulkan.h"
#include "../math/sphere.h"
#include "../parallel.h"
#include "../scene.h"
#include "../bvh/bvh_types.h"
#include "../bvh/bvh_aos.h"

namespace rt
{

static_assert(sizeof(PTLAS_WRITE_INSTANCE_INFO) ==
              sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV));
static_assert(sizeof(PTLAS_UPDATE_INSTANCE_INFO) ==
              sizeof(VkPartitionedAccelerationStructureUpdateInstanceDataNV));

VirtualGeometryManager::VirtualGeometryManager(CommandBuffer *cmd, Arena *arena,
                                               u32 targetWidth, u32 targetHeight)
    : physicalPages(arena, maxPages + 2), virtualTable(arena, maxVirtualPages),
      meshInfos(arena, maxInstances), numInstances(0), numAllocatedPartitions(0),
      virtualInstanceOffset(0), voxelAddressOffset(0), clusterLookupTableOffset(0),
      currentClusterTotal(0), currentTriangleClusterTotal(0), totalNumVirtualPages(0),
      totalNumNodes(0), lruHead(-1), lruTail(-1), resourceSharingInfoOffset(0)
{
    for (u32 i = 0; i < maxVirtualPages; i++)
    {
        virtualTable.Push(VirtualPage{0, PageFlag::NonResident, -1});
    }

    instanceIDFreeRanges = StaticArray<Range>(arena, 32);
    instanceIDFreeRanges.Push(Range{0, maxInstances});

    string decodeDgfClustersName   = "../src/shaders/decode_dgf_clusters.spv";
    string decodeDgfClustersData   = OS_ReadFile(arena, decodeDgfClustersName);
    Shader decodeDgfClustersShader = device->CreateShader(
        ShaderStage::Compute, "decode dgf clusters", decodeDgfClustersData);

    string decodeVoxelClustersName   = "../src/shaders/decode_voxel_aabbs.spv";
    string decodeVoxelClustersData   = OS_ReadFile(arena, decodeVoxelClustersName);
    Shader decodeVoxelClustersShader = device->CreateShader(
        ShaderStage::Compute, "decode voxel clusters", decodeVoxelClustersData);

    string fillClusterTriangleInfoName   = "../src/shaders/fill_cluster_triangle_info.spv";
    string fillClusterTriangleInfoData   = OS_ReadFile(arena, fillClusterTriangleInfoName);
    Shader fillClusterTriangleInfoShader = device->CreateShader(
        ShaderStage::Compute, "fill cluster triangle info", fillClusterTriangleInfoData);

    string clasDefragName = "../src/shaders/clas_defrag.spv";
    string clasDefragData = OS_ReadFile(arena, clasDefragName);
    Shader clasDefragShader =
        device->CreateShader(ShaderStage::Compute, "clas defrag", clasDefragData);

    string computeClasAddressesName   = "../src/shaders/compute_clas_addresses.spv";
    string computeClasAddressesData   = OS_ReadFile(arena, computeClasAddressesName);
    Shader computeClasAddressesShader = device->CreateShader(
        ShaderStage::Compute, "compute clas addresses", computeClasAddressesData);

    string hierarchyTraversalName   = "../src/shaders/hierarchy_traversal.spv";
    string hierarchyTraversalData   = OS_ReadFile(arena, hierarchyTraversalName);
    Shader hierarchyTraversalShader = device->CreateShader(
        ShaderStage::Compute, "hierarchy traversal", hierarchyTraversalData);

    string writeClasDefragAddressesName   = "../src/shaders/write_clas_defrag_addresses.spv";
    string writeClasDefragAddressesData   = OS_ReadFile(arena, writeClasDefragAddressesName);
    Shader writeClasDefragAddressesShader = device->CreateShader(
        ShaderStage::Compute, "write clas defrag addresses", writeClasDefragAddressesData);

    string fillBlasAddressArrayName   = "../src/shaders/fill_blas_address_array.spv";
    string fillBlasAddressArrayData   = OS_ReadFile(arena, fillBlasAddressArrayName);
    Shader fillBlasAddressArrayShader = device->CreateShader(
        ShaderStage::Compute, "fill blas address array", fillBlasAddressArrayData);

    string fillClusterBLASName       = "../src/shaders/fill_cluster_bottom_level_info.spv";
    string fillClusterBLASInfoData   = OS_ReadFile(arena, fillClusterBLASName);
    Shader fillClusterBLASInfoShader = device->CreateShader(
        ShaderStage::Compute, "fill cluster bottom level info", fillClusterBLASInfoData);

    string getBlasAddressOffsetName   = "../src/shaders/get_blas_address_offset.spv";
    string getBlasAddressOffsetData   = OS_ReadFile(arena, getBlasAddressOffsetName);
    Shader getBlasAddressOffsetShader = device->CreateShader(
        ShaderStage::Compute, "get blas address offset", getBlasAddressOffsetData);

    string computeBlasAddressesName   = "../src/shaders/compute_blas_addresses.spv";
    string computeBlasAddressesData   = OS_ReadFile(arena, computeBlasAddressesName);
    Shader computeBlasAddressesShader = device->CreateShader(
        ShaderStage::Compute, "compute blas addresses", computeBlasAddressesData);

    string ptlasWriteInstancesName   = "../src/shaders/ptlas_write_instances.spv";
    string ptlasWriteInstancesData   = OS_ReadFile(arena, ptlasWriteInstancesName);
    Shader ptlasWriteInstancesShader = device->CreateShader(
        ShaderStage::Compute, "ptlas write instances", ptlasWriteInstancesData);

    string clusterFixupName = "../src/shaders/cluster_fixup.spv";
    string clusterFixupData = OS_ReadFile(arena, clusterFixupName);
    Shader clusterFixupShader =
        device->CreateShader(ShaderStage::Compute, "cluster fixup", clusterFixupData);

    string ptlasUpdateUnusedInstancesName = "../src/shaders/ptlas_update_unused_instances.spv";
    string ptlasUpdateUnusedInstancesData = OS_ReadFile(arena, ptlasUpdateUnusedInstancesName);
    Shader ptlasUpdateUnusedInstancesShader = device->CreateShader(
        ShaderStage::Compute, "ptlas update unused instances", ptlasUpdateUnusedInstancesData);

    string ptlasWriteCommandInfosName   = "../src/shaders/ptlas_write_command_infos.spv";
    string ptlasWriteCommandInfosData   = OS_ReadFile(arena, ptlasWriteCommandInfosName);
    Shader ptlasWriteCommandInfosShader = device->CreateShader(
        ShaderStage::Compute, "ptlas write command infos", ptlasWriteCommandInfosData);

    string initializeInstanceFreeListName = "../src/shaders/initialize_instance_freelist.spv";
    string initializeInstanceFreeListData = OS_ReadFile(arena, initializeInstanceFreeListName);
    Shader initializeInstanceFreeListShader = device->CreateShader(
        ShaderStage::Compute, "initialize instance free list", initializeInstanceFreeListData);

    string fillFinestClusterBlasName =
        "../src/shaders/fill_finest_cluster_bottom_level_info.spv";
    string fillFinestClusterBlasData   = OS_ReadFile(arena, fillFinestClusterBlasName);
    Shader fillFinestClusterBlasShader = device->CreateShader(
        ShaderStage::Compute, "fill finest cluster blas", fillFinestClusterBlasData);

    string ptlasUpdatePartitionsName   = "../src/shaders/ptlas_update_partitions.spv";
    string ptlasUpdatePartitionsData   = OS_ReadFile(arena, ptlasUpdatePartitionsName);
    Shader ptlasUpdatePartitionsShader = device->CreateShader(
        ShaderStage::Compute, "ptlas update partitions", ptlasUpdatePartitionsData);

    string instanceStreamingName   = "../src/shaders/instance_streaming.spv";
    string instanceStreamingData   = OS_ReadFile(arena, instanceStreamingName);
    Shader instanceStreamingShader = device->CreateShader(
        ShaderStage::Compute, "instance streaming", instanceStreamingData);

    string decodeMergedInstancesName   = "../src/shaders/decode_merged_instances.spv";
    string decodeMergedInstancesData   = OS_ReadFile(arena, decodeMergedInstancesName);
    Shader decodeMergedInstancesShader = device->CreateShader(
        ShaderStage::Compute, "decode merged instances", decodeMergedInstancesData);

    string mergedInstancesTestName = "../src/shaders/merged_instances_test.spv";
    string mergedInstancesTestData = OS_ReadFile(arena, mergedInstancesTestName);
    Shader mergedInstancesShader   = device->CreateShader(
        ShaderStage::Compute, "merged instances test", mergedInstancesTestData);

    string freeInstancesName = "../src/shaders/free_instances.spv";
    string freeInstancesData = OS_ReadFile(arena, freeInstancesName);
    Shader freeInstancesShader =
        device->CreateShader(ShaderStage::Compute, "free instances", freeInstancesData);

    string allocateInstancesName   = "../src/shaders/allocate_instances.spv";
    string allocateInstancesData   = OS_ReadFile(arena, allocateInstancesName);
    Shader allocateInstancesShader = device->CreateShader(
        ShaderStage::Compute, "allocate instances", allocateInstancesData);

    string instanceCullingShaderName = "../src/shaders/instance_culling.spv";
    string instanceCullingShaderData = OS_ReadFile(arena, instanceCullingShaderName);
    Shader instanceCullingShader     = device->CreateShader(
        ShaderStage::Compute, "instance culling", instanceCullingShaderData);

    string assignInstancesShaderName = "../src/shaders/assign_instances.spv";
    string assignInstancesShaderData = OS_ReadFile(arena, assignInstancesShaderName);
    Shader assignInstancesShader     = device->CreateShader(
        ShaderStage::Compute, "assign instances", assignInstancesShaderData);

    string generateMipsShaderName = "../src/shaders/generate_mips_naive.spv";
    string generateMipsShaderData = OS_ReadFile(arena, generateMipsShaderName);
    Shader generateMipsShader =
        device->CreateShader(ShaderStage::Compute, "generate mips", generateMipsShaderData);

    string reprojectDepthShaderName = "../src/shaders/reproject_depth.spv";
    string reprojectDepthShaderData = OS_ReadFile(arena, reprojectDepthShaderName);
    Shader reprojectDepthShader = device->CreateShader(ShaderStage::Compute, "reproject depth",
                                                       reprojectDepthShaderData);
    // string generateMipsShaderName = "../src/shaders/generate_mips_naive.spv";

    // initialize instance free list
    for (int i = 0; i <= 1; i++)
    {
        initializeFreeListLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                            VK_SHADER_STAGE_COMPUTE_BIT);
    }
    initializeFreeListPipeline = device->CreateComputePipeline(
        &initializeInstanceFreeListShader, &initializeFreeListLayout, 0,
        "initialize instance free list");

    // update partitions
    for (int i = 0; i <= 0; i++)
    {
        ptlasUpdatePartitionsLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                               VK_SHADER_STAGE_COMPUTE_BIT);
    }
    ptlasUpdatePartitionsPipeline = device->CreateComputePipeline(
        &ptlasUpdatePartitionsShader, &ptlasUpdatePartitionsLayout, 0,
        "ptlas update partitions");

    // fill cluster triangle info
    fillClusterTriangleInfoPush.stage  = ShaderStage::Compute;
    fillClusterTriangleInfoPush.size   = sizeof(FillClusterTriangleInfoPushConstant);
    fillClusterTriangleInfoPush.offset = 0;

    for (int i = 0; i <= 4; i++)
    {
        fillClusterTriangleInfoLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                                 VK_SHADER_STAGE_COMPUTE_BIT);
    }
    fillClusterTriangleInfoLayout.AddBinding((u32)RTBindings::ClusterPageData,
                                             DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    fillClusterTriangleInfoPipeline = device->CreateComputePipeline(
        &fillClusterTriangleInfoShader, &fillClusterTriangleInfoLayout,
        &fillClusterTriangleInfoPush, "fill cluster triangle info");

    for (int i = 0; i <= 3; i++)
    {
        decodeDgfClustersLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                           VK_SHADER_STAGE_COMPUTE_BIT);
    }
    decodeDgfClustersLayout.AddBinding((u32)RTBindings::ClusterPageData,
                                       DescriptorType::StorageBuffer,
                                       VK_SHADER_STAGE_COMPUTE_BIT);
    decodeDgfClustersPipeline = device->CreateComputePipeline(
        &decodeDgfClustersShader, &decodeDgfClustersLayout, 0, "decode dgf clusters");

    decodeVoxelClustersPush.stage  = ShaderStage::Compute;
    decodeVoxelClustersPush.size   = sizeof(NumPushConstant);
    decodeVoxelClustersPush.offset = 0;
    for (int i = 0; i <= 2; i++)
    {
        decodeVoxelClustersLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    }
    decodeVoxelClustersLayout.AddBinding((u32)RTBindings::ClusterPageData,
                                         DescriptorType::StorageBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    decodeVoxelClustersPipeline =
        device->CreateComputePipeline(&decodeVoxelClustersShader, &decodeVoxelClustersLayout,
                                      &decodeVoxelClustersPush, "decode voxel clusters");

    clasDefragPush.stage  = ShaderStage::Compute;
    clasDefragPush.size   = sizeof(DefragPushConstant);
    clasDefragPush.offset = 0;
    for (u32 i = 0; i <= 7; i++)
    {
        clasDefragLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                    VK_SHADER_STAGE_COMPUTE_BIT);
    }
    clasDefragPipeline = device->CreateComputePipeline(
        &clasDefragShader, &clasDefragLayout, &clasDefragPush, "clas defrag pipeline");

    computeClasAddressesPush.stage  = ShaderStage::Compute;
    computeClasAddressesPush.size   = sizeof(AddressPushConstant);
    computeClasAddressesPush.offset = 0;
    for (int i = 0; i <= 5; i++)
    {
        computeClasAddressesLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                              VK_SHADER_STAGE_COMPUTE_BIT);
    }
    computeClasAddressesLayout.AddBinding((u32)RTBindings::ClusterPageData,
                                          DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    computeClasAddressesPipeline = device->CreateComputePipeline(
        &computeClasAddressesShader, &computeClasAddressesLayout, &computeClasAddressesPush);
    hierarchyTraversalLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(1, DescriptorType::UniformBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    for (int i = 2; i <= 12; i++)
    {
        hierarchyTraversalLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                            VK_SHADER_STAGE_COMPUTE_BIT);
    }

    hierarchyTraversalPipeline = device->CreateComputePipeline(
        &hierarchyTraversalShader, &hierarchyTraversalLayout, 0, "hierarchy traversal");

    for (int i = 0; i <= 4; i++)
    {
        writeClasDefragLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    }
    writeClasDefragPipeline =
        device->CreateComputePipeline(&writeClasDefragAddressesShader, &writeClasDefragLayout);

    // fill cluster bottom level info
    fillClusterBottomLevelInfoPush.offset = 0;
    fillClusterBottomLevelInfoPush.size   = sizeof(AddressPushConstant);
    fillClusterBottomLevelInfoPush.stage  = ShaderStage::Compute;

    for (int i = 0; i <= 3; i++)
    {
        fillClusterBLASInfoLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    }
    fillClusterBLASInfoPipeline = device->CreateComputePipeline(
        &fillClusterBLASInfoShader, &fillClusterBLASInfoLayout,
        &fillClusterBottomLevelInfoPush, "fill cluster bottom level info");

    // fill finest
    fillFinestClusterBottomLevelInfoPush.offset = 0;
    fillFinestClusterBottomLevelInfoPush.size   = sizeof(AddressPushConstant);
    fillFinestClusterBottomLevelInfoPush.stage  = ShaderStage::Compute;

    for (int i = 0; i <= 4; i++)
    {
        fillFinestClusterBLASInfoLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                                   VK_SHADER_STAGE_COMPUTE_BIT);
    }
    fillFinestClusterBLASInfoPipeline = device->CreateComputePipeline(
        &fillFinestClusterBlasShader, &fillFinestClusterBLASInfoLayout,
        &fillFinestClusterBottomLevelInfoPush, "fill finest blas");

    // fill blas address array
    for (int i = 0; i <= 5; i++)
    {
        fillBlasAddressArrayLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                              VK_SHADER_STAGE_COMPUTE_BIT);
    }
    fillBlasAddressArrayLayout.AddBinding(16, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);

    fillBlasAddressArrayPipeline =
        device->CreateComputePipeline(&fillBlasAddressArrayShader, &fillBlasAddressArrayLayout,
                                      0, "fill blas address array");

    // get blas address offset
    for (int i = 0; i <= 1; i++)
    {
        getBlasAddressOffsetLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                              VK_SHADER_STAGE_COMPUTE_BIT);
    }
    getBlasAddressOffsetPipeline =
        device->CreateComputePipeline(&getBlasAddressOffsetShader, &getBlasAddressOffsetLayout,
                                      0, "get blas address offset");

    // compute blas addresses
    computeBlasAddressesPush.size   = sizeof(AddressPushConstant);
    computeBlasAddressesPush.offset = 0;
    computeBlasAddressesPush.stage  = ShaderStage::Compute;
    for (int i = 0; i <= 4; i++)
    {
        computeBlasAddressesLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                              VK_SHADER_STAGE_COMPUTE_BIT);
    }
    computeBlasAddressesPipeline =
        device->CreateComputePipeline(&computeBlasAddressesShader, &computeBlasAddressesLayout,
                                      &computeBlasAddressesPush, "compute blas addresses");

    // ptlas write instances
    for (int i = 0; i <= 8; i++)
    {
        ptlasWriteInstancesLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    }
    ptlasWriteInstancesPipeline = device->CreateComputePipeline(
        &ptlasWriteInstancesShader, &ptlasWriteInstancesLayout, 0, "ptlas write instances");

    // ptlas update unused instances
    for (int i = 0; i <= 2; i++)
    {
        ptlasUpdateUnusedInstancesLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                                    VK_SHADER_STAGE_COMPUTE_BIT);
    }
    ptlasUpdateUnusedInstancesPipeline = device->CreateComputePipeline(
        &ptlasUpdateUnusedInstancesShader, &ptlasUpdateUnusedInstancesLayout, 0,
        "ptlas write instances");

    // ptlas write command infos
    ptlasWriteCommandInfosPush.size   = sizeof(PtlasPushConstant);
    ptlasWriteCommandInfosPush.offset = 0;
    ptlasWriteCommandInfosPush.stage  = ShaderStage::Compute;

    for (int i = 0; i <= 2; i++)
    {
        ptlasWriteCommandInfosLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                                VK_SHADER_STAGE_COMPUTE_BIT);
    }
    ptlasWriteCommandInfosPipeline = device->CreateComputePipeline(
        &ptlasWriteCommandInfosShader, &ptlasWriteCommandInfosLayout,
        &ptlasWriteCommandInfosPush, "ptlas write command infos");

    instanceStreamingPush.size   = sizeof(InstanceStreamingPushConstant);
    instanceStreamingPush.offset = 0;
    instanceStreamingPush.stage  = ShaderStage::Compute;
    for (int i = 0; i <= 3; i++)
    {
        instanceStreamingLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                           VK_SHADER_STAGE_COMPUTE_BIT);
    }
    instanceStreamingPipeline =
        device->CreateComputePipeline(&instanceStreamingShader, &instanceStreamingLayout,
                                      &instanceStreamingPush, "instance streaming pipeline");

    for (int i = 0; i <= 6; i++)
    {
        decodeMergedInstancesLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                               VK_SHADER_STAGE_COMPUTE_BIT);
    }
    decodeMergedInstancesPipeline = device->CreateComputePipeline(
        &decodeMergedInstancesShader, &decodeMergedInstancesLayout, 0,
        "decode merged instances pipeline");

    // merged instances test
    mergedInstancesTestPush.size   = sizeof(MergedInstancesPushConstant);
    mergedInstancesTestPush.offset = 0;
    mergedInstancesTestPush.stage  = ShaderStage::Compute;
    for (int i = 0; i <= 4; i++)
    {
        mergedInstancesTestLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    }
    mergedInstancesTestLayout.AddBinding(5, DescriptorType::UniformBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    mergedInstancesTestLayout.AddBinding(6, DescriptorType::SampledImage,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    mergedInstancesTestLayout.AddImmutableSamplers();
    mergedInstancesTestPipeline =
        device->CreateComputePipeline(&mergedInstancesShader, &mergedInstancesTestLayout,
                                      &mergedInstancesTestPush, "merged instances pipeline");

    // free instances
    for (int i = 0; i <= 3; i++)
    {
        freeInstancesLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                       VK_SHADER_STAGE_COMPUTE_BIT);
    }
    freeInstancesPipeline = device->CreateComputePipeline(
        &freeInstancesShader, &freeInstancesLayout, 0, "free instances pipeline");

    // allocate instances
    for (int i = 0; i <= 7; i++)
    {
        allocateInstancesLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                           VK_SHADER_STAGE_COMPUTE_BIT);
    }
    allocateInstancesPipeline = device->CreateComputePipeline(
        &allocateInstancesShader, &allocateInstancesLayout, 0, "allocate instance pipeline");

    // instance culling
    instanceCullingLayout.AddBinding(0, DescriptorType::UniformBuffer,
                                     VK_SHADER_STAGE_COMPUTE_BIT);
    for (int i = 1; i <= 10; i++)
    {
        instanceCullingLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    }

    instanceCullingPipeline = device->CreateComputePipeline(
        &instanceCullingShader, &instanceCullingLayout, 0, "instance culling");

    // cluster fixup
    clusterFixupPush.size   = sizeof(NumPushConstant);
    clusterFixupPush.offset = 0;
    clusterFixupPush.stage  = ShaderStage::Compute;
    clusterFixupLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                  VK_SHADER_STAGE_COMPUTE_BIT);
    clusterFixupLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                  VK_SHADER_STAGE_COMPUTE_BIT);
    clusterFixupPipeline = device->CreateComputePipeline(
        &clusterFixupShader, &clusterFixupLayout, &clusterFixupPush, "cluster fixups");

    // assign instances
    for (int i = 0; i <= 6; i++)
    {
        assignInstancesLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    }
    assignInstancesPipeline = device->CreateComputePipeline(
        &assignInstancesShader, &assignInstancesLayout, 0, "assign instances");

    // generate mips
    generateMipsLayout.AddBinding(0, DescriptorType::SampledImage,
                                  VK_SHADER_STAGE_COMPUTE_BIT);
    generateMipsLayout.AddBinding(1, DescriptorType::StorageImage,
                                  VK_SHADER_STAGE_COMPUTE_BIT);
    generateMipsLayout.AddImmutableSamplers();
    generateMipsPipeline = device->CreateComputePipeline(
        &generateMipsShader, &generateMipsLayout, 0, "generate mips pipeline");

    // reproject depth
    reprojectDepthLayout.AddBinding(0, DescriptorType::SampledImage,
                                    VK_SHADER_STAGE_COMPUTE_BIT);
    reprojectDepthLayout.AddBinding(1, DescriptorType::StorageImage,
                                    VK_SHADER_STAGE_COMPUTE_BIT);
    reprojectDepthLayout.AddBinding(2, DescriptorType::UniformBuffer,
                                    VK_SHADER_STAGE_COMPUTE_BIT);
    reprojectDepthLayout.AddImmutableSamplers();
    reprojectDepthPipeline = device->CreateComputePipeline(
        &reprojectDepthShader, &reprojectDepthLayout, 0, "reproject depth pipeline");

    RenderGraph *rg = GetRenderGraph();
    // Buffers
    maxWriteClusters        = MAX_CLUSTERS_PER_PAGE * maxPages;
    streamingRequestsBuffer = rg->CreateBufferResource(
        "streaming requests buffer",
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        sizeof(StreamingRequest) * maxStreamingRequestsPerFrame);
    readbackBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(StreamingRequest) * maxStreamingRequestsPerFrame, MemoryUsage::GPU_TO_CPU);
    readbackBufferHandle =
        rg->RegisterExternalResource("virtual geometry readback", &readbackBuffer);

    pageUploadBuffer      = {};
    fixupBuffer           = {};
    voxelTransferBuffer   = {};
    voxelPageDecodeBuffer = {};
    // clusterLookupTableBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    //                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    //                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    //                                                 sizeof(u32) * 1000);

    evictedPagesBuffer = rg->CreateBufferResource("evicted pages buffer",
                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                  sizeof(u32) * maxPageInstallsPerFrame);

    clusterPageDataBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                 maxPages * CLUSTER_PAGE_SIZE);
    clusterPageDataBufferHandle =
        rg->RegisterExternalResource("cluster page data buffer", &clusterPageDataBuffer);
    hierarchyNodeBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                               maxNodes * sizeof(PackedHierarchyNode));
    hierarchyNodeBufferHandle =
        rg->RegisterExternalResource("hierarchy node buffer", &hierarchyNodeBuffer);
    clusterFixupBuffer = rg->CreateBufferResource(
        "cluster fixup buffer",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        maxClusterFixupsPerFrame * sizeof(GPUClusterFixup));

    clusterAccelAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * (maxPages * MAX_CLUSTERS_PER_PAGE));

    clusterAccelAddressesHandle =
        rg->RegisterExternalResource("cluster accel addresses", &clusterAccelAddresses);
    clusterAccelSizes = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * (maxPages * MAX_CLUSTERS_PER_PAGE));
    clusterAccelSizesHandle =
        rg->RegisterExternalResource("cluster accel sizes buffer", &clusterAccelSizes);

    indexBuffer = rg->CreateBufferResource(
        "index buffer",
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        maxNumTriangles * 3);

    vertexBuffer = rg->CreateBufferResource(
        "vertex buffer",
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        maxNumVertices * sizeof(Vec3f));
    clasGlobalsBuffer = rg->CreateBufferResource(
        "clas globals buffer",
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        sizeof(u32) * GLOBALS_SIZE);

    u32 expectedSize = maxPages * MAX_CLUSTERS_PER_PAGE * 2000;

    u32 clasScratchSize, clasAccelerationStructureSize;
    device->GetCLASBuildSizes(CLASOpMode::ExplicitDestinations,
                              maxPages * MAX_CLUSTERS_PER_PAGE,
                              maxPages * MAX_CLUSTERS_PER_PAGE * MAX_CLUSTER_TRIANGLES,
                              maxPages * MAX_CLUSTERS_PER_PAGE * MAX_CLUSTER_TRIANGLE_VERTICES,
                              MAX_CLUSTER_TRIANGLES, MAX_CLUSTER_TRIANGLE_VERTICES,
                              clasScratchSize, clasAccelerationStructureSize);

    u32 temp;
    device->GetCLASBuildSizes(CLASOpMode::ExplicitDestinations, maxNumClusters,
                              maxNumTriangles, maxNumVertices, MAX_CLUSTER_TRIANGLES,
                              MAX_CLUSTER_TRIANGLE_VERTICES, clasScratchSize, temp);

    Assert(clasAccelerationStructureSize <= expectedSize);
    clasImplicitData =
        device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                             expectedSize);
    clasImplicitDataHandle =
        rg->RegisterExternalResource("clas implicit data", &clasImplicitData);
    clasScratchBuffer = rg->CreateBufferResource(
        "clas scratch buffer",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        clasScratchSize);

    // VkAccelerationStructureGeometryKHR geometry;
    // geometry.geometryType   = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    // geometry.geometry.aabbs = {
    //     VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR};
    // VkAccelerationStructureBuildRangeInfoKHR buildRange = {};
    // buildRange.primitiveCount                           = 400896;
    // u32 maxPrimitiveCounts                              = 400896;
    // u32 testScratch, testSize;
    // device->GetBuildSizes(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, &geometry, 1,
    //                       &buildRange, &maxPrimitiveCounts, testScratch, testSize);

    u32 moveScratchSize, moveStructureSize;
    device->GetMoveBuildSizes(CLASOpMode::ExplicitDestinations, maxWriteClusters,
                              clasImplicitData.size, false, moveScratchSize,
                              moveStructureSize);

    moveScratchBuffer = rg->CreateBufferResource(
        "move scratch buffer",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        moveScratchSize);

    blasDataBuffer                    = rg->CreateBufferResource("blas data buffer",
                                                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                                 sizeof(BLASData) * maxInstances);
    buildClusterBottomLevelInfoBuffer = rg->CreateBufferResource(
        "build cluster bottom level info buffer",
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(BUILD_CLUSTERS_BOTTOM_LEVEL_INFO) * maxInstances);
    blasClasAddressBuffer = rg->CreateBufferResource(
        "blas clas address buffer",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * maxTotalClusterCount);

    blasAccelAddresses = rg->CreateBufferResource(
        "blas accel addresses buffer",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * maxInstances);
    blasAccelSizes = rg->CreateBufferResource(
        "blas accel sizes buffer",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * maxInstances);

    // voxelAABBBuffer   = {};
    // voxelAddressTable = device->CreateBuffer(
    //     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
    //         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    //     sizeof(VoxelAddressTableEntry) * MAX_CLUSTERS_PER_PAGE * maxPages);

    instanceUploadBuffer = {};
    resourceBitVector    = rg->CreateBufferResource("resource bit vector buffer",
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1024);

    ptlasIndirectCommandBuffer = rg->CreateBufferResource(
        "ptlas indirect buffer",
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(PTLAS_INDIRECT_COMMAND) * 3);

    ptlasWriteInfosBuffer = rg->CreateBufferResource(
        "ptlas write infos buffer",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(PTLAS_WRITE_INSTANCE_INFO) * maxInstances);
    ptlasUpdateInfosBuffer = rg->CreateBufferResource(
        "ptlas update infos buffer",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(PTLAS_UPDATE_INSTANCE_INFO) * maxInstances);

    tempInstanceBuffer      = {};
    evictedPartitionsBuffer = {};

    instanceFreeListBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                  sizeof(u32) * (maxInstances + 1));

    instanceFreeListBufferHandle =
        rg->RegisterExternalResource("instance free list buffer", &instanceFreeListBuffer);
    // debugBuffer =
    //     device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(Vec2f) * (1u <<
    //     21u));

    instancesBuffer       = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                 sizeof(GPUInstance) * maxInstances);
    instancesBufferHandle = rg->RegisterExternalResource("instances buffer", &instancesBuffer);

    cmd->StartBindingCompute(initializeFreeListPipeline, &initializeFreeListLayout)
        .Bind(&instanceFreeListBuffer)
        .Bind(&instancesBuffer)
        .End();
    cmd->Dispatch(1, 1, 1);

    decodeClusterDataBuffer = rg->CreateBufferResource(
        "decode cluster data buffer",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(DecodeClusterData) * maxNumClusters);
    buildClusterTriangleInfoBuffer = rg->CreateBufferResource(
        "build cluster triangle info buffer",
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(BUILD_CLUSTERS_TRIANGLE_INFO) * maxNumClusters);
    clasPageInfoBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                              sizeof(CLASPageInfo) * maxPages);
    clasPageInfoBufferHandle =
        rg->RegisterExternalResource("clas page info buffer", &clasPageInfoBuffer);

    moveDescriptors = rg->CreateBufferResource(
        "move descriptors",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(u64) * maxPages * MAX_CLUSTERS_PER_PAGE);

    moveDstAddresses = rg->CreateBufferResource(
        "move dst addresses",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(u64) * maxPages * MAX_CLUSTERS_PER_PAGE);

    moveDstSizes = rg->CreateBufferResource(
        "move dst sizes",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * maxPages * MAX_CLUSTERS_PER_PAGE);

    visibleClustersBuffer = rg->CreateBufferResource(
        "visible clusters buffer",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        MAX_VISIBLE_CLUSTERS * sizeof(VisibleCluster));

    queueBuffer = rg->CreateBufferResource(
        "queue buffer", VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(Queue));

    candidateNodeBuffer = rg->CreateBufferResource(
        "candidate node buffer",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(CandidateNode) * (MAX_CANDIDATE_NODES));

    candidateClusterBuffer = rg->CreateBufferResource(
        "candidate cluster buffer",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(Vec2u) * (MAX_CANDIDATE_CLUSTERS));

    AABB aabb;
    aabb.minX = -1;
    aabb.minY = -1;
    aabb.minZ = -1;
    aabb.maxX = 1;
    aabb.maxY = 1;
    aabb.maxZ = 1;

    u32 numLevels    = GetNumLevels(targetWidth, targetHeight);
    depthPyramidDesc = ImageDesc(ImageType::Type2D, targetWidth, targetHeight, 1, numLevels, 1,
                                 VK_FORMAT_R32_SFLOAT, MemoryUsage::GPU_ONLY,
                                 VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                 VK_IMAGE_TILING_OPTIMAL);
    depthPyramid     = rg->CreateImageResource("depth pyramid", depthPyramidDesc);

    ScratchArena scratch;
    StaticArray<Subresource> subresources(scratch.temp.arena, numLevels);
    for (u32 i = 0; i < numLevels; i++)
    {
        Subresource subresource;
        subresource.baseMip = i;
        subresource.numMips = 1;
        subresources.Push(subresource);
    }
    rg->CreateImageSubresources(depthPyramid, subresources);
}

void VirtualGeometryManager::UnlinkLRU(int pageIndex)
{
    Page &page = physicalPages[pageIndex];

    int prevPage = page.prevPage;
    int nextPage = page.nextPage;

    if (prevPage == -1 && nextPage == -1)
    {
        lruHead = lruTail = -1;
        return;
    }

    if (prevPage != -1)
    {
        physicalPages[prevPage].nextPage = nextPage;
    }
    else
    {
        Assert(lruHead == pageIndex);
        lruHead = nextPage;
    }
    page.prevPage = -1;

    if (nextPage != -1)
    {
        physicalPages[nextPage].prevPage = prevPage;
    }
    else
    {
        Assert(lruTail == pageIndex);
        lruTail = prevPage;
    }
    page.nextPage = -1;
}

void VirtualGeometryManager::LinkLRU(int index)
{
    physicalPages[index].nextPage = lruHead;
    physicalPages[index].prevPage = -1;

    if (lruHead != -1) physicalPages[lruHead].prevPage = index;
    else lruTail = index;

    lruHead = index;
}

void VirtualGeometryManager::EditRegistration(u32 instanceID, u32 pageIndex, bool add)
{
    MeshInfo &meshInfo = meshInfos[instanceID];
    Graph<u32> &graph  = meshInfo.pageToParentPageGraph;

    int physicalPageIndex = virtualTable[meshInfo.virtualPageOffset + pageIndex].pageIndex;
    Assert(add || (!add && physicalPageIndex != -1));

    Page &physicalPage = physicalPages[physicalPageIndex];
    Assert(add || (!add && physicalPage.numDependents == 0));

    int increment = add ? 1 : -1;

    for (int parentPageIndex = graph.offsets[pageIndex];
         parentPageIndex < graph.offsets[pageIndex + 1]; parentPageIndex++)
    {
        u32 parentIndex             = graph.data[parentPageIndex];
        u32 virtualIndex            = meshInfo.virtualPageOffset + parentIndex;
        int parentPhysicalPageIndex = virtualTable[virtualIndex].pageIndex;
        Assert(parentPhysicalPageIndex != -1);
        physicalPages[parentPhysicalPageIndex].numDependents += increment;
    }

    if (!add)
    {
        UnlinkLRU(physicalPageIndex);
    }
    else
    {
        LinkLRU(physicalPageIndex);
    }
}

u32 VirtualGeometryManager::AddNewMesh(Arena *arena, CommandBuffer *cmd, string filename)
{
    string clusterPageData = OS_ReadFile(arena, filename);

    Tokenizer tokenizer;
    tokenizer.input  = clusterPageData;
    tokenizer.cursor = clusterPageData.str;

    ClusterFileHeader clusterFileHeader;
    GetPointerValue(&tokenizer, &clusterFileHeader);
    u32 numPages         = clusterFileHeader.numPages;
    u32 numNodes         = clusterFileHeader.numNodes;
    u32 numVoxelClusters = clusterFileHeader.numVoxelClusters;
    bool hasEllipsoid    = clusterFileHeader.hasTruncatedEllipsoid;

    u8 *pageData = tokenizer.cursor;

    Advance(&tokenizer, clusterFileHeader.numPages * CLUSTER_PAGE_SIZE);
    PackedHierarchyNode *nodes = (PackedHierarchyNode *)tokenizer.cursor;

    Assert(clusterFileHeader.magic == CLUSTER_FILE_MAGIC);

    Bounds bounds;
    u32 count = 0;
    Vec4f spheres[4];
    for (u32 i = 0; i < CHILDREN_PER_HIERARCHY_NODE; i++)
    {
        if (nodes[0].childRef[i] == ~0u) continue;
        Vec3f minP = nodes[0].center[i] - nodes[0].extents[i];
        Vec3f maxP = nodes[0].center[i] + nodes[0].extents[i];

        Bounds b(minP, maxP);

        bounds.Extend(b);
        spheres[i] = nodes[0].lodBounds[i];
    }

    Vec3f boundsMin = ToVec3f(bounds.minP);
    Vec3f boundsMax = ToVec3f(bounds.maxP);

    Vec4f lodBounds = ConstructSphereFromSpheres(spheres, count);

    Advance(&tokenizer, clusterFileHeader.numNodes * sizeof(PackedHierarchyNode));

    // Graphs
    Graph<u32> pageToParentPage;
    pageToParentPage.offsets = (u32 *)tokenizer.cursor;
    Advance(&tokenizer, sizeof(u32) * (clusterFileHeader.numPages + 1));
    pageToParentPage.data = (u32 *)tokenizer.cursor;
    Advance(&tokenizer, sizeof(u32) * pageToParentPage.offsets[clusterFileHeader.numPages]);

    Graph<ClusterFixup> pageToParentCluster;
    pageToParentCluster.offsets = (u32 *)tokenizer.cursor;
    Advance(&tokenizer, sizeof(u32) * (clusterFileHeader.numPages + 1));
    pageToParentCluster.data = (ClusterFixup *)tokenizer.cursor;
    Advance(&tokenizer,
            sizeof(ClusterFixup) * (pageToParentCluster.offsets[clusterFileHeader.numPages]));

    u32 numVoxelClusterGroupFixups;
    GetPointerValue(&tokenizer, &numVoxelClusterGroupFixups);
    StaticArray<VoxelClusterGroupFixup> fixups((VoxelClusterGroupFixup *)tokenizer.cursor,
                                               numVoxelClusterGroupFixups);
    Advance(&tokenizer, sizeof(VoxelClusterGroupFixup) * numVoxelClusterGroupFixups);

    u32 *pageOffsets  = PushArray(arena, u32, clusterFileHeader.numPages + 1);
    u32 *pageOffsets1 = &pageOffsets[1];

    for (u32 nodeIndex = 0; nodeIndex < numNodes; nodeIndex++)
    {
        PackedHierarchyNode &node = nodes[nodeIndex];
        for (u32 childIndex = 0; childIndex < CHILDREN_PER_HIERARCHY_NODE; childIndex++)
        {
            if (node.leafInfo[childIndex] == ~0u) continue;

            // Map page to nodes
            // u32 pageIndex = node.childRef[childIndex];
            u32 bitOffset = MAX_CLUSTERS_PER_PAGE_BITS + MAX_CLUSTERS_PER_GROUP_BITS;
            u32 numPages  = BitFieldExtractU32(node.leafInfo[childIndex],
                                               MAX_PARTS_PER_GROUP_BITS, bitOffset);
            bitOffset += MAX_PARTS_PER_GROUP_BITS;
            u32 pageStartIndex =
                BitFieldExtractU32(node.leafInfo[childIndex], 32u - bitOffset, bitOffset);

            for (u32 pageIndex = pageStartIndex; pageIndex < pageStartIndex + numPages;
                 pageIndex++)
            {
                pageOffsets1[pageIndex]++;
            }
        }
    }

    u32 totalCount = 0;
    for (u32 pageIndex = 0; pageIndex < numPages; pageIndex++)
    {
        u32 count               = pageOffsets1[pageIndex];
        pageOffsets1[pageIndex] = totalCount;
        totalCount += count;
    }

    HierarchyFixup *pageToNodeData = PushArrayNoZero(arena, HierarchyFixup, totalCount);

    ScratchArena scratch(&arena, 1);
    StaticArray<ResourceSharingInfo> resourceSharingInfos(scratch.temp.arena, 32);
    for (u32 i = 0; i < 32; i++)
    {
        ResourceSharingInfo info;
        info.smallestBounds.w = FLT_MAX;
        info.smallestError    = FLT_MAX;
        info.maxError         = 0.f;
        resourceSharingInfos.Push(info);
    }
    u32 maxDepth = 0;

    for (u32 nodeIndex = 0; nodeIndex < numNodes; nodeIndex++)
    {
        PackedHierarchyNode &node = nodes[nodeIndex];
        for (u32 childIndex = 0; childIndex < CHILDREN_PER_HIERARCHY_NODE; childIndex++)
        {
            if (node.leafInfo[childIndex] == ~0u) continue;

            // Map page to nodes

            u32 bitOffset             = 0;
            u32 clusterPageStartIndex = BitFieldExtractU32(
                node.leafInfo[childIndex], MAX_CLUSTERS_PER_PAGE_BITS, bitOffset);
            bitOffset += MAX_CLUSTERS_PER_PAGE_BITS;
            u32 clusterCount = BitFieldExtractU32(node.leafInfo[childIndex],
                                                  MAX_CLUSTERS_PER_GROUP_BITS, bitOffset) +
                               1;
            bitOffset += MAX_CLUSTERS_PER_GROUP_BITS;
            u32 numPages = BitFieldExtractU32(node.leafInfo[childIndex],
                                              MAX_PARTS_PER_GROUP_BITS, bitOffset);
            bitOffset += MAX_PARTS_PER_GROUP_BITS;
            u32 pageStartIndex =
                BitFieldExtractU32(node.leafInfo[childIndex], 32u - bitOffset, bitOffset);
            u32 pageDelta = node.childRef[childIndex] - pageStartIndex;
            HierarchyFixup fixup(nodeIndex, childIndex, pageStartIndex, pageDelta, numPages);

            for (u32 pageIndex = pageStartIndex; pageIndex < pageStartIndex + numPages;
                 pageIndex++)
            {
                u32 index             = pageOffsets1[pageIndex]++;
                pageToNodeData[index] = fixup;
            }

            u32 pageIndex   = node.childRef[childIndex];
            u32 numClusters = *(u32 *)(pageData + CLUSTER_PAGE_SIZE * pageIndex);
            u32 depth       = *(u32 *)(pageData + (CLUSTER_PAGE_SIZE * pageIndex) + 4 +
                                 numClusters * 16 * 4 + clusterPageStartIndex * 16 + 4);
            maxDepth        = Max(maxDepth, depth);

            resourceSharingInfos[depth].maxError =
                Max(resourceSharingInfos[depth].maxError, node.maxParentError[childIndex]);
            resourceSharingInfos[depth].smallestError = Min(
                resourceSharingInfos[depth].smallestError, node.maxParentError[childIndex]);
            if (node.lodBounds[childIndex].w < resourceSharingInfos[depth].smallestBounds.w)
            {
                resourceSharingInfos[depth].smallestBounds = node.lodBounds[childIndex];
            }
        }
    }

    Graph<HierarchyFixup> graph;
    graph.offsets = pageOffsets;
    graph.data    = pageToNodeData;

    for (u32 nodeIndex = 0; nodeIndex < numNodes; nodeIndex++)
    {
        for (u32 i = 0; i < CHILDREN_PER_HIERARCHY_NODE; i++)
        {
            if (nodes[nodeIndex].leafInfo[i] != ~0u)
            {
                nodes[nodeIndex].childRef[i] = ~0u;
            }
        }
    }

    cmd->SubmitBuffer(&hierarchyNodeBuffer, nodes, sizeof(PackedHierarchyNode) * numNodes,
                      sizeof(PackedHierarchyNode) * totalNumNodes);

    MeshInfo meshInfo                 = {};
    meshInfo.pageToHierarchyNodeGraph = graph;
    meshInfo.hierarchyNodeOffset      = totalNumNodes;
    meshInfo.virtualPageOffset        = totalNumVirtualPages;
    meshInfo.pageData                 = pageData;
    meshInfo.pageToParentClusters     = pageToParentCluster;
    meshInfo.pageToParentPageGraph    = pageToParentPage;
    meshInfo.totalNumVoxelClusters    = numVoxelClusters;
    meshInfo.boundsMin                = boundsMin;
    meshInfo.boundsMax                = boundsMax;
    meshInfo.voxelClusterGroupFixups  = fixups;
    meshInfo.voxelBLASBitmask         = 0;
    meshInfo.voxelAddressOffset       = voxelAddressOffset;
    meshInfo.clusterLookupTableOffset = clusterLookupTableOffset;
    meshInfo.lodBounds                = lodBounds;
    meshInfo.numLodLevels             = maxDepth + 1;
    meshInfo.resourceSharingInfos =
        StaticArray<ResourceSharingInfo>(arena, maxDepth + 1, maxDepth + 1);
    MemoryCopy(meshInfo.resourceSharingInfos.data, resourceSharingInfos.data,
               sizeof(ResourceSharingInfo) * (maxDepth + 1));
    meshInfo.resourceSharingInfoOffset = resourceSharingInfoOffset;

    if (hasEllipsoid)
    {
        GetPointerValue(&tokenizer, &meshInfo.ellipsoid);
    }
    else
    {
        meshInfo.ellipsoid.sphere.w = 0.f;
    }

    voxelAddressOffset += meshInfo.voxelClusterGroupFixups.Length();
    clusterLookupTableOffset += numVoxelClusters;

    meshInfo.nodes    = nodes;
    meshInfo.numNodes = numNodes;

    totalNumNodes += numNodes;
    totalNumVirtualPages += numPages;
    resourceSharingInfoOffset += meshInfo.numLodLevels;

    meshInfos.Push(meshInfo);

    return meshInfos.Length() - 1;
}

void VirtualGeometryManager::FinalizeResources(CommandBuffer *cmd)
{
    ScratchArena scratch;
    StaticArray<Resource> resources(scratch.temp.arena, meshInfos.Length());
    StaticArray<AABB> resourceAABBs(scratch.temp.arena, meshInfos.Length());
    StaticArray<GPUTruncatedEllipsoid> truncatedEllipsoids(scratch.temp.arena,
                                                           meshInfos.Length());
    Array<ResourceSharingInfo> resourceSharingInfos(scratch.temp.arena,
                                                    16 * meshInfos.Length());

    RenderGraph *rg = GetRenderGraph();
    u32 clasBlasScratchSize, blasSize;
    // maxWriteClusters = 200000;

    // TODO IMPORTANT: calculate this for real
    device->GetClusterBLASBuildSizes(CLASOpMode::ExplicitDestinations, 50000, 40000,
                                     meshInfos.Length(), clasBlasScratchSize, blasSize);

    clasBlasImplicitBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        blasSize);
    clasBlasImplicitHandle =
        rg->RegisterExternalResource("clas blas implicit buffer", &clasBlasImplicitBuffer);
    blasScratchBuffer = rg->CreateBufferResource(
        "blas scratch buffer",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        clasBlasScratchSize);

    for (MeshInfo &meshInfo : meshInfos)
    {
        Resource resource     = {};
        resource.flags        = meshInfo.numNodes == 1 && (*(u32 *)meshInfo.pageData == 1)
                                    ? RESOURCE_FLAG_ONE_CLUSTER
                                    : 0;
        resource.lodBounds    = meshInfo.lodBounds;
        resource.numLodLevels = meshInfo.numLodLevels;
        resource.resourceSharingInfoOffset = meshInfo.resourceSharingInfoOffset;
        resource.globalRootNodeOffset      = meshInfo.hierarchyNodeOffset;
        resources.Push(resource);

        for (ResourceSharingInfo &info : meshInfo.resourceSharingInfos)
        {
            resourceSharingInfos.Push(info);
        }

        AABB aabb;
        aabb.minX = meshInfo.boundsMin[0];
        aabb.minY = meshInfo.boundsMin[1];
        aabb.minZ = meshInfo.boundsMin[2];
        aabb.maxX = meshInfo.boundsMax[0];
        aabb.maxY = meshInfo.boundsMax[1];
        aabb.maxZ = meshInfo.boundsMax[2];

        resourceAABBs.Push(aabb);

        GPUTruncatedEllipsoid ellipsoid;
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                ellipsoid.transform[r][c] = meshInfo.ellipsoid.transform[c][r];
            }
        }
        ellipsoid.sphere    = meshInfo.ellipsoid.sphere;
        ellipsoid.boundsMin = meshInfo.ellipsoid.boundsMin;
        ellipsoid.boundsMax = meshInfo.ellipsoid.boundsMax;
        truncatedEllipsoids.Push(ellipsoid);
    }

    resourceBuffer       = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                sizeof(Resource) * meshInfos.Length());
    resourceBufferHandle = rg->RegisterExternalResource("resource buffer", &resourceBuffer);
    cmd->SubmitBuffer(&resourceBuffer, resources.data, sizeof(Resource) * meshInfos.Length());

    resourceAABBBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                              sizeof(AABB) * resourceAABBs.Length());
    resourceAABBBufferHandle =
        rg->RegisterExternalResource("resource aabb buffer", &resourceAABBBuffer);
    cmd->SubmitBuffer(&resourceAABBBuffer, resourceAABBs.data,
                      sizeof(AABB) * resourceAABBs.Length());

    resourceTruncatedEllipsoidsBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        sizeof(TruncatedEllipsoid) * truncatedEllipsoids.Length());
    resourceTruncatedEllipsoidsBufferHandle =
        rg->RegisterExternalResource("ellisoid buffer", &resourceTruncatedEllipsoidsBuffer);
    cmd->SubmitBuffer(&resourceTruncatedEllipsoidsBuffer, truncatedEllipsoids.data,
                      sizeof(TruncatedEllipsoid) * truncatedEllipsoids.Length());

    resourceSharingInfosBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(ResourceSharingInfo) * resourceSharingInfos.Length());
    resourceSharingInfosBufferHandle =
        rg->RegisterExternalResource("resource sharing buffer", &resourceSharingInfosBuffer);
    cmd->SubmitBuffer(&resourceSharingInfosBuffer, resourceSharingInfos.data,
                      sizeof(ResourceSharingInfo) * resourceSharingInfos.Length());

    maxMinLodLevelBuffer = rg->CreateBufferResource("max min lod level buffer",
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                    sizeof(Vec2u) * meshInfos.Length());
}

void VirtualGeometryManager::RecursePageDependencies(StaticArray<VirtualPageHandle> &pages,
                                                     u32 instanceID, u32 pageIndex,
                                                     u32 priority)
{
    MeshInfo &meshInfo = meshInfos[instanceID];

    VirtualPage &virtualPage = virtualTable[meshInfo.virtualPageOffset + pageIndex];
    if (virtualPage.priority == 0)
    {
        pages.Push(VirtualPageHandle{instanceID, pageIndex});
    }

    if (priority <= virtualPage.priority) return;
    virtualPage.priority = priority;

    Graph<u32> &graph = meshInfo.pageToParentPageGraph;
    u32 numParents    = graph.offsets[pageIndex + 1] - graph.offsets[pageIndex];

    for (int parentPageIndex = 0; parentPageIndex < numParents; parentPageIndex++)
    {
        u32 parentPage = graph.data[graph.offsets[pageIndex] + parentPageIndex];
        VirtualPage &parentVirtualPage = virtualTable[meshInfo.virtualPageOffset + parentPage];
        if (priority + 1 > parentVirtualPage.priority)
        {
            RecursePageDependencies(pages, instanceID, parentPage, priority + 1);
        }
    }
}

bool VirtualGeometryManager::VerifyPageDependencies(u32 virtualOffset, u32 startPage,
                                                    u32 numPages)
{
    for (u32 pageIndex = startPage; pageIndex < startPage + numPages; pageIndex++)
    {
        VirtualPage &virtualPage = virtualTable[virtualOffset + pageIndex];
        if (virtualPage.pageIndex == -1)
        {
            return false;
        }
    }
    return true;
}

bool VirtualGeometryManager::CheckDuplicatedFixup(u32 virtualOffset, u32 pageIndex,
                                                  u32 startPage, u32 numPages)
{
    for (u32 otherPageIndex = startPage; otherPageIndex < pageIndex; otherPageIndex++)
    {
        VirtualPage &virtualPage = virtualTable[virtualOffset + otherPageIndex];
        Assert(virtualPage.pageIndex != -1);
        if (virtualPage.pageFlag == PageFlag::ResidentThisFrame)
        {
            return true;
        }
    }
    return false;
}

static void GetBrickMax(u64 bitMask, Vec3u &maxP)
{
    maxP.z = 4u - (LeadingZeroCount64(bitMask) >> 4u);

    u32 bits = (u32)bitMask | u32(bitMask >> 32u);
    bits |= bits << 16u;
    maxP.y = 4u - (LeadingZeroCount(bits) >> 2u);

    bits |= bits << 8u;
    bits |= bits << 4u;
    maxP.x = 4u - LeadingZeroCount(bits);

    Assert(maxP.x <= 4 && maxP.y <= 4 && maxP.z <= 4);
}

static Brick DecodeBrick(u8 *pageData, u32 brickIndex, u32 baseAddress, u32 brickOffset)
{
    u32 bitsOffset = (64 + 14) * brickIndex;
    u32 byteOffset = baseAddress + brickOffset + (bitsOffset >> 3u);
    u32 bitOffset  = bitsOffset & 7u;

    Vec3u data = *(Vec3u *)(pageData + byteOffset);

    Vec3u packed;
    packed.x = BitAlignU32(data.y, data.x, bitOffset);
    packed.y = BitAlignU32(data.z, data.y, bitOffset);
    packed.z = data.z >> bitOffset;

    Brick brick;
    brick.bitMask = packed.x;
    brick.bitMask |= ((u64)packed.y << 32u);
    brick.vertexOffset = BitFieldExtractU32(packed.z, 14, 0);

    return brick;
}

static Vec3f DecodePosition(u8 *pageData, u32 vertexIndex, Vec3u posBitWidths, Vec3i anchor,
                            u32 baseAddress, u32 geoBaseAddress)
{
    const uint bitsPerVertex = posBitWidths[0] + posBitWidths[1] + posBitWidths[2];
    const uint bitsOffset    = vertexIndex * bitsPerVertex;

    u32 byteOffset = baseAddress + geoBaseAddress + (bitsOffset >> 3u);
    u32 bitOffset  = bitsOffset & 7u;
    Vec3u data     = *(Vec3u *)(pageData + byteOffset);

    Vec2u packed =
        Vec2u(BitAlignU32(data.y, data.x, bitOffset), BitAlignU32(data.z, data.y, bitOffset));

    Vec3i pos = Vec3i(0, 0, 0);
    pos.x     = BitFieldExtractU32(packed.x, posBitWidths.x, 0);
    packed.x  = BitAlignU32(packed.y, packed.x, posBitWidths.x);

    packed.y >>= posBitWidths.x;
    pos.y = BitFieldExtractU32(packed.x, posBitWidths.y, 0);

    packed.x = BitAlignU32(packed.y, packed.x, posBitWidths.y);
    pos.z    = BitFieldExtractU32(packed.x, posBitWidths.z, 0);

    const float scale = AsFloat((127u - 6u) << 23u);
    return Vec3f(pos + anchor) * scale;
}

bool VirtualGeometryManager::ProcessInstanceRequests(CommandBuffer *cmd)
{
#if 0
    ScratchArena scratch;
    u32 *proxyCounts    = (u32 *)partitionReadbackBuffer.mappedPtr;
    const u32 threshold = 128;

    // u32 total = 0;
    // for (u32 i = 0; i < maxPartitions; i++)
    // {
    //     total += proxyCounts[i];
    // }
    // Print("num hits: %u\n", total);

    Array<u32> newPartitions(scratch.temp.arena, maxPartitions);
    Array<u32> emptyPartitions(scratch.temp.arena, 128);
    u32 numInstancesToUpload    = 0;
    u32 numInstancesToStreamOut = 0;

    for (u32 partitionIndex = 0; partitionIndex < maxPartitions; partitionIndex++)
    {
        if (newPartitions.Length() == maxPartitions) break;
        if (proxyCounts[partitionIndex] && !partitionStreamedIn.GetBit(partitionIndex))
        {
            newPartitions.Push(partitionIndex);
            numInstancesToUpload += partitionInstanceGraph.offsets[partitionIndex + 1] -
                                    partitionInstanceGraph.offsets[partitionIndex];
        }
        else if (proxyCounts[partitionIndex] == 0 &&
                 partitionStreamedIn.GetBit(partitionIndex))
        {
            emptyPartitions.Push(partitionIndex);
            numInstancesToStreamOut += partitionInstanceGraph.offsets[partitionIndex + 1] -
                                       partitionInstanceGraph.offsets[partitionIndex];
        }
    }

    if (numInstancesToUpload == 0) return false;

    if (instanceUploadBuffer.size)
    {
        device->DestroyBuffer(&instanceUploadBuffer);
    }

    instanceUploadBuffer = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                sizeof(GPUInstance) * numInstancesToUpload +
                                                    sizeof(u32) * emptyPartitions.Length(),
                                                MemoryUsage::CPU_TO_GPU);

    u32 offset       = 0;
    u32 evictedIndex = 0;

    for (u32 partition : newPartitions)
    {
        if (numAllocatedPartitions >= maxPartitions &&
            evictedIndex >= emptyPartitions.Length())
            break;

        u32 count = partitionInstanceGraph.offsets[partition + 1] -
                    partitionInstanceGraph.offsets[partition];

        u32 allocatedPartitionIndex = ~0u;
        Assert(allocatedPartitionIndices[partition] == ~0u);
        if (numAllocatedPartitions < maxPartitions)
        {
            allocatedPartitionIndex = numAllocatedPartitions++;
        }
        else
        {
            allocatedPartitionIndex =
                allocatedPartitionIndices[emptyPartitions[evictedIndex++]];
        }
        Assert(allocatedPartitionIndex != ~0u);
        allocatedPartitionIndices[partition] = allocatedPartitionIndex;
        for (u32 i = 0; i < count; i++)
        {
            partitionInstanceGraph.data[partitionInstanceGraph.offsets[partition] + i]
                .partitionIndex = allocatedPartitionIndex;
        }
        MemoryCopy((u8 *)instanceUploadBuffer.mappedPtr + offset,
                   partitionInstanceGraph.data + partitionInstanceGraph.offsets[partition],
                   sizeof(GPUInstance) * count);
        offset += sizeof(GPUInstance) * count;
        partitionStreamedIn.SetBit(partition);
    }
    offset = 0;

    if (numInstances < maxInstances)
    {
        u32 numToCopy = Min(maxInstances - numInstances, numInstancesToUpload);
        BufferToBufferCopy copy;
        copy.srcOffset = 0;
        copy.dstOffset = sizeof(GPUInstance) * numInstances;
        copy.size      = sizeof(GPUInstance) * numToCopy;
        numInstances += numToCopy;
        numInstancesToUpload -= numToCopy;
        offset = copy.size;

        cmd->CopyBuffer(&instancesBuffer, &instanceUploadBuffer, &copy, 1);
    }

    if (numInstancesToUpload)
    {
        // while (emptyPartitions.Length() > newPartitions.Length() &&
        //        numInstancesToUpload < numInstancesToStreamOut)
        // {
        //     u32 partition = emptyPartitions.Back();
        //     u32 count     = partitionInstanceGraph.offsets[partition + 1] -
        //                 partitionInstanceGraph.offsets[partition];
        //     if (numInstancesToStreamOut - count >= numInstancesToUpload)
        //     {
        //         emptyPartitions.Pop();
        //         numInstancesToStreamOut -= count;
        //     }
        //     else
        //     {
        //         break;
        //     }
        // }

        while (numInstancesToUpload > numInstancesToStreamOut ||
               newPartitions.Length() > emptyPartitions.Length())
        {
            u32 partition = newPartitions.Pop();
            u32 count     = partitionInstanceGraph.offsets[partition + 1] -
                        partitionInstanceGraph.offsets[partition];
            numInstancesToUpload -= count;

            partitionStreamedIn.UnsetBit(partition);
            allocatedPartitionIndices[partition] = ~0u;
        }
        Assert(numInstancesToStreamOut >= numInstancesToUpload);
        Assert(emptyPartitions.Length() >= newPartitions.Length());

        for (u32 emptyPartition : emptyPartitions)
        {
            partitionStreamedIn.UnsetBit(emptyPartition);
            allocatedPartitionIndices[emptyPartition] = ~0u;
        }

        if (tempInstanceBuffer.size)
        {
            device->DestroyBuffer(&tempInstanceBuffer);
        }
        tempInstanceBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                  sizeof(GPUInstance) * numInstancesToUpload);
        if (evictedPartitionsBuffer.size)
        {
            device->DestroyBuffer(&evictedPartitionsBuffer);
        }
        evictedPartitionsBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                       sizeof(u32) * emptyPartitions.Length());

        {
            BufferToBufferCopy copy;
            copy.srcOffset = offset;
            copy.dstOffset = 0;
            copy.size      = sizeof(GPUInstance) * numInstancesToUpload;
            cmd->CopyBuffer(&tempInstanceBuffer, &instanceUploadBuffer, &copy, 1);
            offset += copy.size;
        }
        {
            MemoryCopy((u8 *)instanceUploadBuffer.mappedPtr + offset, emptyPartitions.data,
                       sizeof(u32) * emptyPartitions.Length());
            BufferToBufferCopy copy;
            copy.srcOffset = offset;
            copy.dstOffset = 0;
            copy.size      = sizeof(u32) * emptyPartitions.Length();
            cmd->CopyBuffer(&evictedPartitionsBuffer, &instanceUploadBuffer, &copy, 1);
        }

        cmd->Barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();

        InstanceStreamingPushConstant pc;
        pc.numNewInstances      = numInstancesToUpload;
        pc.numEvictedPartitions = emptyPartitions.Length();

        cmd->StartBindingCompute(instanceStreamingPipeline, &instanceStreamingLayout)
            .Bind(&instancesBuffer)
            .Bind(&tempInstanceBuffer)
            .Bind(&evictedPartitionsBuffer)
            .Bind(&clasGlobalsBuffer)
            .PushConstants(&instanceStreamingPush, &pc)
            .End();
        cmd->Dispatch((numInstances + 31) / 32, 1, 1);
    }

#endif
    return true;
}

void VirtualGeometryManager::UpdateHZB()
{
    RenderGraph *rg = GetRenderGraph();
    u32 numLevels   = depthPyramidDesc.numMips;
    u32 width       = depthPyramidDesc.width;
    u32 height      = depthPyramidDesc.height;
    for (u32 level = 0; level < numLevels - 1; level++)
    {
        rg->StartPass(
              2,
              [width, height, level, depthPyramid = this->depthPyramid,
               generateMipsPipeline = this->generateMipsPipeline,
               &generateMipsLayout  = this->generateMipsLayout,
               &depthPyramidDesc    = this->depthPyramidDesc](CommandBuffer *cmd) {
                  RenderGraph *rg = GetRenderGraph();
                  GPUImage *img   = rg->GetImage(depthPyramid);

                  cmd->Barrier(img, VK_IMAGE_LAYOUT_GENERAL,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                               QueueType_Ignored, QueueType_Ignored, level, 1);
                  // cmd->Barrier(img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                  //              VK_PIPELINE_STAGE_2_NONE,
                  //              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_NONE,
                  //              VK_ACCESS_2_SHADER_READ_BIT, QueueType_Ignored,
                  //              QueueType_Ignored, level + 1, 1);
                  cmd->FlushBarriers();

                  ResourceBinding binding =
                      cmd->StartBindingCompute(generateMipsPipeline, &generateMipsLayout);

                  binding.Bind(img, level);
                  binding.Bind(img, level + 1);
                  binding.End();

                  u32 groupCountX = (width + 7) / 8;
                  u32 groupCountY = (height + 7) / 8;
                  groupCountX     = groupCountX == 0 ? 1 : groupCountX;
                  groupCountY     = groupCountY == 0 ? 1 : groupCountY;

                  cmd->Dispatch(groupCountX, groupCountY, 1);
              })
            .AddHandle(depthPyramid, ResourceUsageType::Read, level)
            .AddHandle(depthPyramid, ResourceUsageType::Write, level + 1);
        width  = Max(1u, width >> 1);
        height = Max(1u, height >> 1);
    }
    rg->StartPass(1, [depthPyramid = this->depthPyramid, numLevels](CommandBuffer *cmd) {
          RenderGraph *rg = GetRenderGraph();
          GPUImage *img   = rg->GetImage(depthPyramid);
          cmd->Barrier(img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT, QueueType_Ignored, QueueType_Ignored,
                       numLevels - 1, 1);
          cmd->FlushBarriers();
      }).AddHandle(depthPyramid, ResourceUsageType::Write, numLevels - 1);
}

void VirtualGeometryManager::ReprojectDepth(u32 targetWidth, u32 targetHeight,
                                            ResourceHandle depth, ResourceHandle scene)
{
    RenderGraph *rg = GetRenderGraph();
    u32 groupCountX = (targetWidth + 7) / 8;
    u32 groupCountY = (targetHeight + 7) / 8;
    rg->StartComputePass(reprojectDepthPipeline, reprojectDepthLayout, 3,
                         [targetWidth, targetHeight, groupCountX, groupCountY,
                          depthPyramid = this->depthPyramid](CommandBuffer *cmd) {
                             GPUImage *image = GetRenderGraph()->GetImage(depthPyramid);

                             cmd->Barrier(image, VK_IMAGE_LAYOUT_UNDEFINED,
                                          VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE,
                                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          VK_ACCESS_2_NONE, VK_ACCESS_2_SHADER_WRITE_BIT,
                                          QueueType_Ignored, QueueType_Ignored, 0, 1);
                             cmd->FlushBarriers();

                             device->BeginEvent(cmd, "Reproject Depth");
                             cmd->Dispatch(groupCountX, groupCountY, 1);
                             device->EndEvent(cmd);
                         })
        .AddHandle(depth, ResourceUsageType::Read)
        .AddHandle(depthPyramid, ResourceUsageType::Write, 0)
        .AddHandle(scene, ResourceUsageType::Read);
}

void VirtualGeometryManager::ProcessRequests(CommandBuffer *cmd, bool test)
{
    RenderGraph *rg            = GetRenderGraph();
    StreamingRequest *requests = (StreamingRequest *)readbackBuffer.mappedPtr;
    u32 numRequests            = requests[0].pageIndex_numPages;
    numRequests                = Min(numRequests, maxStreamingRequestsPerFrame - 1u);
    Print("num requests: %u\n", numRequests);
    requests++;

    if (numRequests == 0) return;

    // u32 writeIndex
    // Radix sort by priority
    // Divide into pages that need to be loaded and pages that are already resident
    // fix up the hierarchy nodes

    // TODO: async copy
    // u32 batchIndex  = (requestBatchWriteIndex + (maxQueueBatches - 1)) %
    // (maxQueueBatches); u32 numRequests = streamingRequestBatches[batchIndex];

    ScratchArena scratch;
    u32 requestStartIndex = 0; // batchIndex * maxStreamingRequestsPerFrame;

    union Float
    {
        f32 f;
        u32 u;
    };

    StaticArray<VirtualPageHandle> pages(scratch.temp.arena, maxVirtualPages);
    u32 minPage = 200;
    for (u32 requestIndex = 0; requestIndex < numRequests; requestIndex++)
    {
        StreamingRequest &request = requests[requestIndex];
        u32 pageCount =
            BitFieldExtractU32(request.pageIndex_numPages, MAX_PARTS_PER_GROUP_BITS, 0);
        u32 pageStartIndex = request.pageIndex_numPages >> MAX_PARTS_PER_GROUP_BITS;

        minPage = Min(minPage, pageStartIndex);

        for (u32 pageIndex = pageStartIndex; pageIndex < pageStartIndex + pageCount;
             pageIndex++)
        {
            Float streamingPriority;
            streamingPriority.f = request.priority;

            RecursePageDependencies(pages, request.instanceID, pageIndex,
                                    streamingPriority.u + 1);
        }
    }

    struct PageHandle
    {
        u32 sortKey;
        u32 index;
    };

    StaticArray<VirtualPageHandle> unloadedRequests(scratch.temp.arena, pages.Length());
    StaticArray<PageHandle> pageHandles(scratch.temp.arena, pages.Length());

    for (VirtualPageHandle page : pages)
    {
        u32 virtualPageOffset   = meshInfos[page.instanceID].virtualPageOffset;
        int virtualPageIndex    = virtualPageOffset + page.pageIndex;
        VirtualPage virtualPage = virtualTable[virtualPageIndex];
        Assert(virtualPage.priority != 0);

        int physicalPageIndex = virtualPage.pageIndex;

        if (physicalPageIndex == -1)
        {
            unloadedRequests.Push(page);

            PageHandle handle;
            handle.sortKey = virtualPage.priority;
            handle.index   = unloadedRequests.Length() - 1;
            pageHandles.Push(handle);
        }
        else
        {
            UnlinkLRU(physicalPageIndex);
            LinkLRU(physicalPageIndex);
        }
    }

    SortHandles<PageHandle, false>(pageHandles.data, pageHandles.Length());

    // Get pages
    u32 pagesToUpdate = physicalPages.Length();
    StaticArray<int> pageIndices(scratch.temp.arena, unloadedRequests.Length());
    while (physicalPages.Length() < maxPages &&
           pageIndices.Length() < unloadedRequests.Length() &&
           pageIndices.Length() < maxPageInstallsPerFrame)
    {
        Page page         = {};
        page.virtualIndex = -1;
        physicalPages.Push(page);

        pageIndices.Push(physicalPages.Length() - 1);
    }

    u32 evictedPageStart = pageIndices.Length();

    // Upload pages to GPU and apply hierarchy fixups
    StaticArray<BufferToBufferCopy> pageInstallCopies(scratch.temp.arena,
                                                      maxPageInstallsPerFrame);
    StaticArray<BufferToBufferCopy> nodeInstallCopies(scratch.temp.arena, 2 * maxNodes);
    StaticArray<u32> hierarchyFixupData(scratch.temp.arena, 2 * maxNodes);
    u32 offset = 0;

    StaticArray<VirtualPageHandle> uninstallRequests(scratch.temp.arena,
                                                     unloadedRequests.Length());
    StaticArray<VirtualPageHandle> installRequests(scratch.temp.arena,
                                                   unloadedRequests.Length());
    StaticArray<u32> updatedResourceIndices(scratch.temp.arena, meshInfos.Length());

    u32 clustersToAdd  = 0;
    u32 pagesToInstall = Min(unloadedRequests.Length(), maxPageInstallsPerFrame);
    int lruIndex       = lruTail;

    if (pageUploadBuffer.size)
    {
        device->DestroyBuffer(&pageUploadBuffer);
        pageUploadBuffer.size = 0;
    }
    if (pagesToInstall)
    {
        pageUploadBuffer =
            device->CreateBuffer(VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
                                 CLUSTER_PAGE_SIZE * pagesToInstall, MemoryUsage::CPU_TO_GPU);
    }

    for (int requestIndex = 0; requestIndex < pagesToInstall; requestIndex++)
    {
        PageHandle handle         = pageHandles[requestIndex];
        VirtualPageHandle request = unloadedRequests[handle.index];

        if (requestIndex >= evictedPageStart)
        {
            while (lruIndex != -1)
            {
                int index = lruIndex;
                lruIndex  = physicalPages[lruIndex].prevPage;
                if (physicalPages[index].numDependents == 0 &&
                    virtualTable[physicalPages[index].virtualIndex].priority == 0)
                {
                    pageIndices.Push(index);
                    break;
                }
            }
            if (lruIndex == -1) break;
        }

        updatedResourceIndices.PushUnique(request.instanceID);

        u32 gpuPageIndex = pageIndices[requestIndex];
        u32 virtualPageIndex =
            meshInfos[request.instanceID].virtualPageOffset + request.pageIndex;
        virtualTable[virtualPageIndex].pageIndex = gpuPageIndex;
        virtualTable[virtualPageIndex].pageFlag  = PageFlag::ResidentThisFrame;

        Page &page = physicalPages[gpuPageIndex];
        if (page.virtualIndex != -1)
        {
            currentClusterTotal -= page.numClusters;
            currentTriangleClusterTotal -= page.numTriangleClusters;
            EditRegistration(page.handle.instanceID, page.handle.pageIndex, false);
            virtualTable[page.virtualIndex].pageIndex = -1;
            virtualTable[page.virtualIndex].pageFlag  = PageFlag::NonResident;

            uninstallRequests.Push(page.handle);
        }
        page.handle       = request;
        page.virtualIndex = virtualPageIndex;

        EditRegistration(page.handle.instanceID, page.handle.pageIndex, true);
        installRequests.Push(page.handle);

        BufferToBufferCopy copy;
        copy.srcOffset = offset;
        copy.dstOffset = CLUSTER_PAGE_SIZE * gpuPageIndex;
        copy.size      = CLUSTER_PAGE_SIZE;
        pageInstallCopies.Push(copy);

        u8 *buffer = meshInfos[request.instanceID].pageData;
        u8 *src    = buffer + CLUSTER_PAGE_SIZE * request.pageIndex;

        u32 numClustersInPage = *(u32 *)src;

        clustersToAdd += numClustersInPage;
        physicalPages[gpuPageIndex].numClusters = numClustersInPage;

        MemoryCopy((u8 *)pageUploadBuffer.mappedPtr + offset, src, CLUSTER_PAGE_SIZE);
        offset += CLUSTER_PAGE_SIZE;
    }

    for (VirtualPageHandle page : pages)
    {
        virtualTable[meshInfos[page.instanceID].virtualPageOffset + page.pageIndex].priority =
            0;
    }

    Array<GPUClusterFixup> gpuClusterFixup(scratch.temp.arena, uninstallRequests.Length());

    auto PrepareHierarchyFixup = [&](u32 instanceID, u32 pageIndex, bool uninstall) {
        MeshInfo &meshInfo       = meshInfos[instanceID];
        u32 hierarchyNodeOffset  = meshInfo.hierarchyNodeOffset;
        u32 *offsets             = meshInfo.pageToHierarchyNodeGraph.offsets;
        HierarchyFixup *nodeData = meshInfo.pageToHierarchyNodeGraph.data;

        for (u32 graphIndex = offsets[pageIndex]; graphIndex < offsets[pageIndex + 1];
             graphIndex++)
        {
            HierarchyFixup &fixup = nodeData[graphIndex];

            if (!uninstall &&
                !VerifyPageDependencies(meshInfo.virtualPageOffset, fixup.GetPageStartIndex(),
                                        fixup.GetNumPages()))
                continue;

            if (!uninstall &&
                CheckDuplicatedFixup(meshInfo.virtualPageOffset, pageIndex,
                                     fixup.GetPageStartIndex(), fixup.GetNumPages()))
                continue;

            u32 hierarchyNodeIndex = fixup.GetNodeIndex();
            u32 childIndex         = fixup.GetChildIndex();

            BufferToBufferCopy hierarchyCopy;
            hierarchyCopy.srcOffset = nodeInstallCopies.Length() * sizeof(u32);
            hierarchyCopy.dstOffset =
                sizeof(PackedHierarchyNode) * (hierarchyNodeOffset + hierarchyNodeIndex) +
                OffsetOf(PackedHierarchyNode, childRef) + sizeof(u32) * childIndex;
            hierarchyCopy.size = sizeof(u32);

            nodeInstallCopies.Push(hierarchyCopy);

            Assert(uninstall ||
                   virtualTable[meshInfo.virtualPageOffset + fixup.GetPageIndex()].pageIndex !=
                       -1);

            u32 gpuPageIndex =
                uninstall ? ~0u
                          : virtualTable[meshInfo.virtualPageOffset + fixup.GetPageIndex()]
                                .pageIndex;

            hierarchyFixupData.Push(gpuPageIndex);
        }
    };

    auto PrepareClusterFixup = [&](u32 instanceID, u32 pageIndex, bool uninstall) {
        // When adding page, must check if this completes the group
        // When removing page, must check if this uncompletes the group
        MeshInfo &meshInfo         = meshInfos[instanceID];
        Graph<ClusterFixup> &graph = meshInfo.pageToParentClusters;

        for (u32 parentClusterIndex = graph.offsets[pageIndex];
             parentClusterIndex < graph.offsets[pageIndex + 1]; parentClusterIndex++)
        {
            ClusterFixup parentCluster = graph.data[parentClusterIndex];

            if (!uninstall && !VerifyPageDependencies(meshInfo.virtualPageOffset,
                                                      parentCluster.GetPageStartIndex(),
                                                      parentCluster.GetNumPages()))
                continue;

            if (!uninstall && CheckDuplicatedFixup(meshInfo.virtualPageOffset, pageIndex,
                                                   parentCluster.GetPageStartIndex(),
                                                   parentCluster.GetNumPages()))
                continue;

            u32 parentPageIndex = parentCluster.GetPageIndex();
            VirtualPage &virtualPage =
                virtualTable[meshInfo.virtualPageOffset + parentPageIndex];

            if (virtualPage.pageIndex == -1)
            {
                Assert(uninstall);
                continue;
            }
            u32 numClusters = physicalPages[virtualPage.pageIndex].numClusters;

            GPUClusterFixup fixup;
            fixup.offset = CLUSTER_PAGE_SIZE * virtualPage.pageIndex + sizeof(u32) +
                           sizeof(Vec4u) * (parentCluster.GetClusterIndex() + numClusters * 4);
            Assert((fixup.offset & 1) == 0);
            fixup.offset |= uninstall ? 0x0 : 0x1;
            gpuClusterFixup.Push(fixup);
        }
    };

    // Voxel acceleration structures
    u32 triangleClustersToAdd = 0;
    u32 numAABBs              = 0;

    // Uninstalls
    for (VirtualPageHandle uninstallRequest : uninstallRequests)
    {
        // Only fixup parent clusters if they are still going to be resident
        PrepareClusterFixup(uninstallRequest.instanceID, uninstallRequest.pageIndex, true);
        PrepareHierarchyFixup(uninstallRequest.instanceID, uninstallRequest.pageIndex, true);
    }

    // Installs
    for (VirtualPageHandle installRequest : installRequests)
    {
        PrepareClusterFixup(installRequest.instanceID, installRequest.pageIndex, false);
        PrepareHierarchyFixup(installRequest.instanceID, installRequest.pageIndex, false);
    }

    // Voxels
    Array<AccelBuildInfo> buildInfos(scratch.temp.arena, pagesToInstall);
    Array<VoxelPageDecodeData> voxelPageDecodeData(scratch.temp.arena, pagesToInstall);
    Array<Vec2u> voxelAddressTableOffsets(scratch.temp.arena, pagesToInstall);
    Array<Vec2u> clusterLookupTableCopies(scratch.temp.arena,
                                          pageIndices.Length() * MAX_CLUSTERS_PER_PAGE);

#if 0
    for (u32 resourceIndex : updatedResourceIndices)
    {
        MeshInfo &meshInfo = meshInfos[resourceIndex];
        for (VoxelClusterGroupFixup &fixup : meshInfo.voxelClusterGroupFixups)
        {
            bool wasBuilt = (meshInfo.voxelBLASBitmask & (1u << fixup.depth)) != 0u;
            bool pageDependenciesFound = VerifyPageDependencies(
                meshInfo.virtualPageOffset, fixup.pageStartIndex, fixup.numPages);

            u32 depth = fixup.depth;
            if (!wasBuilt && pageDependenciesFound)
            {
                meshInfo.voxelBLASBitmask |= 1u << fixup.depth;

                AccelBuildInfo buildInfo  = {};
                buildInfo.primitiveOffset = sizeof(AABB) * numAABBs;

                voxelAddressTableOffsets.Push(
                    {meshInfo.voxelAddressOffset + depth,
                     meshInfo.clusterLookupTableOffset + fixup.clusterOffset});
                u32 totalClusterCount = 0;
                for (u32 groupPageIndex = fixup.pageStartIndex;
                     groupPageIndex < fixup.pageStartIndex + fixup.numPages; groupPageIndex++)
                {
                    int physicalPage =
                        virtualTable[meshInfo.virtualPageOffset + groupPageIndex].pageIndex;
                    Assert(physicalPage != -1);
                    int numClusters = physicalPages[physicalPage].numClusters;

                    int clusterStartIndex =
                        groupPageIndex == fixup.pageStartIndex ? fixup.clusterStartIndex : 0;
                    int clusterEndIndex =
                        groupPageIndex == fixup.pageStartIndex + fixup.numPages - 1
                            ? fixup.clusterEndIndex
                            : numClusters;
                    u32 clusterCount = clusterEndIndex - clusterStartIndex;

                    VoxelPageDecodeData decodeData;
                    decodeData.pageIndex         = physicalPage;
                    decodeData.clusterStartIndex = clusterStartIndex;
                    decodeData.clusterEndIndex   = clusterEndIndex;
                    decodeData.offset            = numAABBs;
                    voxelPageDecodeData.Push(decodeData);

                    for (int clusterIndex = clusterStartIndex; clusterIndex < clusterEndIndex;
                         clusterIndex++)
                    {
                        Vec2u data;
                        data.x = (physicalPage << MAX_CLUSTERS_PER_PAGE_BITS) | clusterIndex;
                        data.y = sizeof(u32) * (meshInfo.clusterLookupTableOffset +
                                                fixup.clusterOffset + totalClusterCount++);
                        clusterLookupTableCopies.Push(data);
                    }
                    numAABBs += MAX_CLUSTER_TRIANGLES * clusterCount;
                }

                buildInfo.primitiveCount = MAX_CLUSTER_TRIANGLES * totalClusterCount;
                buildInfos.Push(buildInfo);
                Assert(numAABBs ==
                       buildInfo.primitiveOffset / sizeof(AABB) + buildInfo.primitiveCount);
            }
            else if (wasBuilt && !pageDependenciesFound)
            {
                Assert(0);
            }
        }
    }
#endif

    for (VirtualPageHandle installRequest : installRequests)
    {
        PageFlag &pageFlag =
            virtualTable[meshInfos[installRequest.instanceID].virtualPageOffset +
                         installRequest.pageIndex]
                .pageFlag;
        Assert(pageFlag == PageFlag::ResidentThisFrame);
        pageFlag = PageFlag::Resident;
    }

    // TODO: direct storage
    if (pageIndices.Length())
    {
        u32 pageInstallSize     = pageIndices.Length() * sizeof(u32);
        u32 gpuClusterFixupSize = sizeof(GPUClusterFixup) * gpuClusterFixup.Length();
        u32 hierarchyFixupSize  = sizeof(u32) * hierarchyFixupData.Length();
        u32 totalSize           = hierarchyFixupSize + gpuClusterFixupSize + pageInstallSize;

        if (fixupBuffer.size)
        {
            device->DestroyBuffer(&fixupBuffer);
        }
        fixupBuffer = device->CreateBuffer(VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT, totalSize,
                                           MemoryUsage::CPU_TO_GPU);
        Assert(totalSize);

        u32 offset = hierarchyFixupSize;
        MemoryCopy(fixupBuffer.mappedPtr, hierarchyFixupData.data, hierarchyFixupSize);

        u32 numPageInstallCopies = pageInstallCopies.Length();
        BufferToBufferCopy *pageCopies =
            rg->Allocate(pageInstallCopies.data, numPageInstallCopies);
        u32 numNodeInstallCopies = nodeInstallCopies.Length();
        BufferToBufferCopy *nodeCopies =
            rg->Allocate(nodeInstallCopies.data, numNodeInstallCopies);
        rg->StartPass(2,
                      [&clusterPageDataBuffer = this->clusterPageDataBuffer,
                       &hierarchyNodeBuffer   = this->hierarchyNodeBuffer,
                       &pageUploadBuffer      = this->pageUploadBuffer,
                       &fixupBuffer           = this->fixupBuffer, pageCopies, nodeCopies,
                       numPageInstallCopies, numNodeInstallCopies](CommandBuffer *cmd) {
                          cmd->CopyBuffer(&clusterPageDataBuffer, &pageUploadBuffer,
                                          pageCopies, numPageInstallCopies);
                          cmd->CopyBuffer(&hierarchyNodeBuffer, &fixupBuffer, nodeCopies,
                                          numNodeInstallCopies);
                      })
            .AddHandle(clusterPageDataBufferHandle, ResourceUsageType::Write)
            .AddHandle(hierarchyNodeBufferHandle, ResourceUsageType::Write);

        MemoryCopy((u8 *)fixupBuffer.mappedPtr + offset, pageIndices.data, pageInstallSize);

        rg->StartPass(1, [rg, &fixupBuffer = this->fixupBuffer,
                          &evictedPagesBufferHandle = this->evictedPagesBuffer, offset,
                          pageInstallSize](CommandBuffer *cmd) {
              u32 dstOffset;
              GPUBuffer *evictedPagesBuffer =
                  rg->GetBuffer(evictedPagesBufferHandle, dstOffset);

              BufferToBufferCopy pageInstallCopy;
              pageInstallCopy.srcOffset = offset;
              pageInstallCopy.dstOffset = dstOffset;
              pageInstallCopy.size      = pageInstallSize;

              cmd->CopyBuffer(evictedPagesBuffer, &fixupBuffer, &pageInstallCopy, 1);
          }).AddHandle(evictedPagesBuffer, ResourceUsageType::Write);

        offset += pageInstallSize;

        // TODO: this still triggers
        Assert(gpuClusterFixup.Length() <= maxClusterFixupsPerFrame);

        // Cluster fixups
        if (gpuClusterFixup.Length())
        {
            MemoryCopy((u8 *)fixupBuffer.mappedPtr + offset, gpuClusterFixup.data,
                       gpuClusterFixupSize);
            rg->StartPass(1, [&clusterFixupBufferHandle = this->clusterFixupBuffer,
                              &fixupBuffer              = this->fixupBuffer, offset,
                              gpuClusterFixupSize](CommandBuffer *cmd) {
                  u32 dstOffset;
                  GPUBuffer *clusterFixupBuffer =
                      GetRenderGraph()->GetBuffer(clusterFixupBufferHandle, dstOffset);

                  BufferToBufferCopy clusterFixupsCopy;
                  clusterFixupsCopy.srcOffset = offset;
                  clusterFixupsCopy.dstOffset = dstOffset;
                  clusterFixupsCopy.size      = gpuClusterFixupSize;
                  cmd->CopyBuffer(clusterFixupBuffer, &fixupBuffer, &clusterFixupsCopy, 1);
              }).AddHandle(clusterFixupBuffer, ResourceUsageType::Write);
            offset += gpuClusterFixupSize;
        }

        rg->StartPass(0, [](CommandBuffer *cmd) {
            cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
            cmd->FlushBarriers();
        });

        NumPushConstant pc;
        pc.num        = gpuClusterFixup.Length();
        u32 numFixups = gpuClusterFixup.Length();
        rg->StartComputePass(
              clusterFixupPipeline, clusterFixupLayout, 2,
              [numFixups](CommandBuffer *cmd) {
                  device->BeginEvent(cmd, "Cluster Fixups");
                  cmd->Dispatch((numFixups + 31) >> 5, 1, 1);
                  device->EndEvent(cmd);
              },
              &clusterFixupPush, &pc)
            .AddHandle(clusterFixupBuffer, ResourceUsageType::Read)
            .AddHandle(clusterPageDataBufferHandle, ResourceUsageType::RW);
    }

    for (int requestIndex = 0; requestIndex < pagesToInstall; requestIndex++)
    {
        PageHandle handle         = pageHandles[requestIndex];
        VirtualPageHandle request = unloadedRequests[handle.index];

        u32 page           = pageIndices[requestIndex];
        MeshInfo &meshInfo = meshInfos[request.instanceID];
        u8 *src            = meshInfo.pageData + CLUSTER_PAGE_SIZE * request.pageIndex;

        u32 numClusters            = *(u32 *)src;
        u32 clusterHeaderSOAStride = numClusters * 16;
        u32 numTriangleClusters    = 0;

        for (u32 clusterIndex = 0; clusterIndex < numClusters; clusterIndex++)
        {
            u32 clusterHeaderSOAStride = numClusters * 16;
            u32 baseOffset = request.pageIndex * CLUSTER_PAGE_SIZE + 4 + clusterIndex * 16;
            Vec4u packed[NUM_CLUSTER_HEADER_FLOAT4S];
            for (int i = 0; i < NUM_CLUSTER_HEADER_FLOAT4S; i++)
            {
                packed[i] =
                    *(Vec4u *)(meshInfo.pageData + baseOffset + i * clusterHeaderSOAStride);
            }

            if ((packed[2].w >> 31u) == 0)
            {
                numTriangleClusters++;
            }
        }

        physicalPages[page].numTriangleClusters = numTriangleClusters;

        triangleClustersToAdd += numTriangleClusters;
    }

    u32 newClasOffset = currentTriangleClusterTotal;
    currentTriangleClusterTotal += triangleClustersToAdd;

#if 0
    int buildIndex = TIMED_CPU_RANGE_BEGIN();

    StaticArray<AccelBuildInfo> staticBuildInfos(buildInfos.data, buildInfos.Length());
    if (numAABBs)
    {
        StaticArray<AccelerationStructureSizes> sizes =
            device->GetBLASBuildSizes(scratch.temp.arena, staticBuildInfos);

        StaticArray<AccelerationStructureCreate> creates(scratch.temp.arena,
                                                         buildInfos.Length());
        u32 totalScratch = 0;
        u32 totalAccel   = 0;
        for (AccelerationStructureSizes sizeInfo : sizes)
        {
            totalAccel = AlignPow2(totalAccel, 256);

            AccelerationStructureCreate create = {};
            create.accelSize                   = sizeInfo.accelSize;
            create.bufferOffset                = totalAccel;
            create.type                        = AccelerationStructureType::Bottom;

            creates.Push(create);

            totalAccel += sizeInfo.accelSize;
            totalScratch += sizeInfo.scratchSize;
        }

        GPUBuffer scratchBuffer = device->CreateBuffer(
            VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
            totalScratch);
        GPUBuffer accelBuffer =
            device->CreateBuffer(VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                     VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                 totalAccel);

        for (AccelerationStructureCreate &create : creates)
        {
            create.buffer = &accelBuffer;
        }

        if (voxelAABBBuffer.size)
        {
            device->DestroyBuffer(&voxelAABBBuffer);
        }
        voxelAABBBuffer = device->CreateBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            sizeof(AABB) * numAABBs);

        u32 voxelDecodeSize = sizeof(VoxelPageDecodeData) * voxelPageDecodeData.Length();
        if (voxelPageDecodeBuffer.size)
        {
            device->DestroyBuffer(&voxelPageDecodeBuffer);
        }
        voxelPageDecodeBuffer = device->CreateBuffer(VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT |
                                                         VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                                     voxelDecodeSize);

        u32 voxelPageFixupsSize   = sizeof(VoxelPageDecodeData) * voxelPageDecodeData.Length();
        u32 voxelAddressTableSize = sizeof(u64) * voxelAddressTableOffsets.Length();
        u32 clusterLookupTableCopiesSize = sizeof(u32) * clusterLookupTableCopies.Length();
        u32 transferSize                 = sizeof(u64) * creates.Length() + voxelDecodeSize +
                           voxelPageFixupsSize + voxelAddressTableSize +
                           clusterLookupTableCopiesSize;

        // TODO: have unified system for GPU transfers
        if (voxelTransferBuffer.size)
        {
            device->DestroyBuffer(&voxelTransferBuffer);
        }
        voxelTransferBuffer = device->CreateBuffer(VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
                                                   transferSize, MemoryUsage::CPU_TO_GPU);

        uint64_t scratchDataDeviceAddress = device->GetDeviceAddress(&scratchBuffer);
        uint64_t aabbDeviceAddress        = device->GetDeviceAddress(&voxelAABBBuffer);
        totalScratch                      = 0;
        totalAccel                        = 0;

        device->CreateAccelerationStructures(creates);

        for (u32 sizeIndex = 0; sizeIndex < sizes.Length(); sizeIndex++)
        {
            AccelerationStructureSizes &sizeInfo = sizes[sizeIndex];
            AccelerationStructureCreate &create  = creates[sizeIndex];

            AccelBuildInfo &buildInfo          = buildInfos[sizeIndex];
            buildInfo.scratchDataDeviceAddress = scratchDataDeviceAddress + totalScratch;
            buildInfo.dataDeviceAddress        = aabbDeviceAddress;
            buildInfo.as                       = create.as;

            totalScratch += sizeInfo.scratchSize;
        }

        StaticArray<BufferToBufferCopy> copies(scratch.temp.arena, creates.Length());
        StaticArray<BufferToBufferCopy> clusterLookupTableBufferCopy(
            scratch.temp.arena, clusterLookupTableCopies.Length());

        u32 offset = 0;
        Assert(voxelAddressTableOffsets.Length() == creates.Length());
        for (u32 i = 0; i < voxelAddressTableOffsets.Length(); i++)
        {
            AccelerationStructureCreate &create = creates[i];
            Assert(sizeof(create.asDeviceAddress) == sizeof(u64));

            VoxelAddressTableEntry entry;
            entry.address     = create.asDeviceAddress;
            entry.tableOffset = voxelAddressTableOffsets[i].y;

            BufferToBufferCopy copy;
            copy.srcOffset = offset;
            copy.dstOffset = sizeof(VoxelAddressTableEntry) * voxelAddressTableOffsets[i].x;
            copy.size      = sizeof(VoxelAddressTableEntry);
            copies.Push(copy);

            MemoryCopy((u8 *)voxelTransferBuffer.mappedPtr + copy.srcOffset, &entry,
                       copy.size);

            offset += copy.size;
        }

        BufferToBufferCopy decodeCopy;
        decodeCopy.srcOffset = offset;
        decodeCopy.dstOffset = 0;
        decodeCopy.size      = sizeof(VoxelPageDecodeData) * voxelPageDecodeData.Length();
        MemoryCopy((u8 *)voxelTransferBuffer.mappedPtr + decodeCopy.srcOffset,
                   voxelPageDecodeData.data, decodeCopy.size);

        offset += decodeCopy.size;

        for (Vec2u &data : clusterLookupTableCopies)
        {
            BufferToBufferCopy copy;
            copy.srcOffset = offset;
            copy.dstOffset = data.y;
            copy.size      = sizeof(u32);
            MemoryCopy((u8 *)voxelTransferBuffer.mappedPtr + copy.srcOffset, &data.x,
                       copy.size);
            offset += copy.size;

            clusterLookupTableBufferCopy.Push(copy);
        }

        cmd->CopyBuffer(&voxelAddressTable, &voxelTransferBuffer, copies.data,
                        copies.Length());
        cmd->CopyBuffer(&voxelPageDecodeBuffer, &voxelTransferBuffer, &decodeCopy, 1);
        cmd->CopyBuffer(&clusterLookupTableBuffer, &voxelTransferBuffer,
                        clusterLookupTableBufferCopy.data,
                        clusterLookupTableBufferCopy.Length());
        cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
    }
    TIMED_RANGE_END(buildIndex);
#endif

    u32 numEvictedPages = pageIndices.Length() - evictedPageStart;

    u64 address = device->GetDeviceAddress(clasImplicitData.buffer);

    if (pageIndices.Length() == 0) return;

    // Prepare move descriptors
    {
        DefragPushConstant pc;
        pc.evictedPageStart = evictedPageStart;
        pc.numEvictedPages  = numEvictedPages;

        rg->StartComputePass(
              clasDefragPipeline, clasDefragLayout, 8,
              [pagesToUpdate,
               clasGlobalsBuffer = this->clasGlobalsBuffer](CommandBuffer *cmd) {
                  device->BeginEvent(cmd, "Prepare Defrag Clas");

                  cmd->Dispatch(pagesToUpdate, 1, 1);

                  cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                               VK_ACCESS_2_SHADER_WRITE_BIT,
                               VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                   VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
                  cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                  cmd->FlushBarriers();
                  device->EndEvent(cmd);
              },
              &clasDefragPush, &pc)
            .AddHandle(evictedPagesBuffer, ResourceUsageType::Read)
            .AddHandle(clasPageInfoBufferHandle, ResourceUsageType::RW)
            .AddHandle(clusterAccelAddressesHandle, ResourceUsageType::Read)
            .AddHandle(clusterAccelSizesHandle, ResourceUsageType::Read)
            .AddHandle(clasGlobalsBuffer, ResourceUsageType::Write)
            .AddHandle(moveDescriptors, ResourceUsageType::Write)
            .AddHandle(moveDstAddresses, ResourceUsageType::Write)
            .AddHandle(moveDstSizes, ResourceUsageType::Write);
    }

    if (numEvictedPages)
    {
        Assert(0);
        Print("Evicting %u pages\n", numEvictedPages);

        u32 maxMovedBytes = clasImplicitData.size;
        rg->StartPass(
              6,
              [moveDescriptors   = this->moveDescriptors,
               moveScratchBuffer = this->moveScratchBuffer,
               moveDstAddresses = this->moveDstAddresses, moveDstSizes = this->moveDstSizes,
               maxPages = this->maxPages, clasGlobalsBuffer = this->clasGlobalsBuffer,
               maxWriteClusters = this->maxWriteClusters, maxMovedBytes](CommandBuffer *cmd) {
                  RenderGraph *rg = GetRenderGraph();
                  u32 moveInfoOffset, moveInfoSize, scratchBufferOffset, srcInfosCountOffset;
                  GPUBuffer *moveInfo =
                      rg->GetBuffer(moveDescriptors, moveInfoOffset, moveInfoSize);
                  u64 moveInfoAddress = device->GetDeviceAddress(moveInfo) + moveInfoOffset;

                  GPUBuffer *scratchBuffer =
                      rg->GetBuffer(moveScratchBuffer, scratchBufferOffset);
                  u64 scratchBufferAddress =
                      device->GetDeviceAddress(scratchBuffer) + scratchBufferOffset;

                  u32 moveAddressesOffset, moveSizesOffset;
                  GPUBuffer *moveAddresses =
                      rg->GetBuffer(moveDstAddresses, moveAddressesOffset);
                  u64 dstAddressesAddress =
                      device->GetDeviceAddress(moveAddresses) + moveAddressesOffset;
                  u32 dstAddressesSize = sizeof(u64) * (maxPages * MAX_CLUSTERS_PER_PAGE);

                  GPUBuffer *moveSizes = rg->GetBuffer(moveDstSizes, moveSizesOffset);
                  u64 dstSizesAddress  = device->GetDeviceAddress(moveSizes) + moveSizesOffset;
                  u32 dstSizesSize     = sizeof(u32) * (maxPages * MAX_CLUSTERS_PER_PAGE);

                  GPUBuffer *srcInfosCount =
                      rg->GetBuffer(clasGlobalsBuffer, srcInfosCountOffset);
                  u64 srcInfosCountAddress = device->GetDeviceAddress(srcInfosCount) +
                                             srcInfosCountOffset +
                                             sizeof(u32) * GLOBALS_DEFRAG_CLAS_COUNT;

                  device->BeginEvent(cmd, "Defrag Clas");
                  cmd->MoveCLAS(CLASOpMode::ExplicitDestinations, 0, scratchBufferAddress,
                                moveInfoAddress, moveInfoSize, dstAddressesAddress,
                                dstAddressesSize, dstSizesAddress, dstSizesSize,
                                srcInfosCountAddress, maxWriteClusters, maxMovedBytes, false);
                  device->EndEvent(cmd);
              })
            .AddHandle(moveScratchBuffer, ResourceUsageType::RW)
            .AddHandle(moveDstAddresses, ResourceUsageType::Read)
            .AddHandle(moveDstSizes, ResourceUsageType::Read)
            .AddHandle(moveDescriptors, ResourceUsageType::Read)
            .AddHandle(clasGlobalsBuffer, ResourceUsageType::Read)
            .AddHandle(clasImplicitDataHandle, ResourceUsageType::RW);

        rg->StartComputePass(writeClasDefragPipeline, writeClasDefragLayout, 5,
                             [pagesToUpdate](CommandBuffer *cmd) {
                                 device->BeginEvent(cmd, "Write CLAS Defrag Addresses");

                                 cmd->Dispatch(pagesToUpdate, 1, 1);
                                 cmd->Barrier(
                                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                     VK_ACCESS_2_SHADER_READ_BIT);
                                 cmd->FlushBarriers();
                                 device->EndEvent(cmd);
                             })
            .AddHandle(clasPageInfoBufferHandle, ResourceUsageType::Read)
            .AddHandle(moveDstAddresses, ResourceUsageType::Read)
            .AddHandle(moveDstSizes, ResourceUsageType::Read)
            .AddHandle(clusterAccelAddressesHandle, ResourceUsageType::Write)
            .AddHandle(clusterAccelSizesHandle, ResourceUsageType::Write);
    }

    u32 numNewPages = pageIndices.Length();
    // Write cluster build descriptors
    {

        rg->StartComputePass(
              fillClusterTriangleInfoPipeline, fillClusterTriangleInfoLayout, 6,
              [indexBuffer = this->indexBuffer, vertexBuffer = this->vertexBuffer,
               &fillClusterTriangleInfoPush   = this->fillClusterTriangleInfoPush,
               &fillClusterTriangleInfoLayout = this->fillClusterTriangleInfoLayout,
               newClasOffset, numNewPages,
               clasGlobalsBuffer = this->clasGlobalsBuffer](CommandBuffer *cmd) {
                  RenderGraph *rg = GetRenderGraph();
                  u32 indexBufferOffset, vertexBufferOffset, indexSize, vertexSize;
                  GPUBuffer *indexGPUBuffer =
                      rg->GetBuffer(indexBuffer, indexBufferOffset, indexSize);
                  GPUBuffer *vertexGPUBuffer =
                      rg->GetBuffer(vertexBuffer, vertexBufferOffset, vertexSize);
                  u64 indexBufferAddress =
                      device->GetDeviceAddress(indexGPUBuffer) + indexBufferOffset;
                  u64 vertexBufferAddress =
                      device->GetDeviceAddress(vertexGPUBuffer) + vertexBufferOffset;

                  FillClusterTriangleInfoPushConstant fillPc;
                  fillPc.indexBufferBaseAddressLowBits = indexBufferAddress & 0xffffffff;
                  fillPc.indexBufferBaseAddressHighBits =
                      (indexBufferAddress >> 32u) & 0xffffffff;
                  fillPc.vertexBufferBaseAddressLowBits = vertexBufferAddress & 0xffffffff;
                  fillPc.vertexBufferBaseAddressHighBits =
                      (vertexBufferAddress >> 32u) & 0xffffffff;
                  fillPc.clusterOffset = newClasOffset;
                  cmd->PushConstants(&fillClusterTriangleInfoPush, &fillPc,
                                     fillClusterTriangleInfoLayout.pipelineLayout);
                  cmd->Dispatch(numNewPages, 1, 1);
                  cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
                  cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                               VK_ACCESS_2_SHADER_WRITE_BIT,
                               VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
                  // cmd->FlushBarriers();
                  cmd->FlushBarriers();
              })
            .AddHandle(evictedPagesBuffer, ResourceUsageType::Read)
            .AddHandle(buildClusterTriangleInfoBuffer, ResourceUsageType::Write)
            .AddHandle(decodeClusterDataBuffer, ResourceUsageType::Write)
            .AddHandle(clasGlobalsBuffer, ResourceUsageType::Write)
            .AddHandle(clasPageInfoBufferHandle, ResourceUsageType::Write)
            .AddHandle(clusterPageDataBufferHandle, ResourceUsageType::Read);
    }

    // Decode the clusters
    {
        rg->StartIndirectComputePass(
              "Decode Installed Pages", decodeDgfClustersPipeline, decodeDgfClustersLayout, 5,
              clasGlobalsBuffer, sizeof(u32) * GLOBALS_CLAS_COUNT_INDEX,
              [clasGlobalsBuffer = this->clasGlobalsBuffer,
               info              = this->buildClusterTriangleInfoBuffer](CommandBuffer *cmd) {
                  device->BeginEvent(cmd, "Decode Installed Pages");
                  cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                               VK_ACCESS_2_SHADER_WRITE_BIT,
                               VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                   VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
                  cmd->FlushBarriers();
                  device->EndEvent(cmd);
                  static u32 count = 0;
              })
            .AddHandle(indexBuffer, ResourceUsageType::Write)
            .AddHandle(vertexBuffer, ResourceUsageType::Write)
            .AddHandle(decodeClusterDataBuffer, ResourceUsageType::Read)
            .AddHandle(clasGlobalsBuffer, ResourceUsageType::Read)
            .AddHandle(clusterPageDataBufferHandle, ResourceUsageType::Read);

        Assert(voxelPageDecodeData.Length() == 0);
        // if (voxelPageDecodeData.Length())
        // {
        //     NumPushConstant num;
        //     num.num = voxelPageDecodeData.Length();
        //     cmd->StartBindingCompute(decodeVoxelClustersPipeline,
        //     &decodeVoxelClustersLayout)
        //         .Bind(&clasGlobalsBuffer)
        //         .Bind(&voxelAABBBuffer)
        //         .Bind(&voxelPageDecodeBuffer)
        //         .Bind(&clusterPageDataBuffer)
        //         .PushConstants(&decodeVoxelClustersPush, &num)
        //         .End();
        //     cmd->Dispatch(num.num, 1, 1);
        // }
    }

    // if (staticBuildInfos.Length())
    // {
    //     Assert(0);
    //     cmd->BuildCustomBLAS(staticBuildInfos);
    // }

    {
        AddressPushConstant pc;
        pc.addressLowBits  = address & 0xffffffff;
        pc.addressHighBits = address >> 32u;

        rg->StartPass(
              4,
              [maxNumTriangles = this->maxNumTriangles, maxNumVertices = this->maxNumVertices,
               maxNumClusters                 = this->maxNumClusters, newClasOffset, rg,
               buildClusterTriangleInfoBuffer = this->buildClusterTriangleInfoBuffer,
               clasScratchBuffer              = this->clasScratchBuffer,
               &clusterAccelSizes             = this->clusterAccelSizes,
               clasGlobalsBuffer              = this->clasGlobalsBuffer,
               maxPages                       = this->maxPages](CommandBuffer *cmd) {
                  device->BeginEvent(cmd, "Compute CLAS Sizes");
                  u32 triangleInfoOffset, triangleInfoSize, clasScratchBufferOffset,
                      srcInfosCountOffset;
                  GPUBuffer *buildClusterTriangleInfo = rg->GetBuffer(
                      buildClusterTriangleInfoBuffer, triangleInfoOffset, triangleInfoSize);
                  u64 buildClusterTriangleInfoAddress =
                      device->GetDeviceAddress(buildClusterTriangleInfo) + triangleInfoOffset;

                  GPUBuffer *scratchBuffer =
                      rg->GetBuffer(clasScratchBuffer, clasScratchBufferOffset);
                  u64 scratchBufferAddress =
                      device->GetDeviceAddress(scratchBuffer) + clasScratchBufferOffset;

                  u64 dstSizesAddress = device->GetDeviceAddress(&clusterAccelSizes) +
                                        sizeof(u32) * newClasOffset;
                  u32 dstSizesSize =
                      sizeof(u32) * (maxPages * MAX_CLUSTERS_PER_PAGE - newClasOffset);

                  GPUBuffer *srcInfosCount =
                      rg->GetBuffer(clasGlobalsBuffer, srcInfosCountOffset);
                  u64 srcInfosCountAddress = device->GetDeviceAddress(srcInfosCount) +
                                             srcInfosCountOffset +
                                             sizeof(u32) * GLOBALS_CLAS_COUNT_INDEX;

                  cmd->ComputeCLASSizes(buildClusterTriangleInfoAddress, triangleInfoSize,
                                        scratchBufferAddress, dstSizesAddress, dstSizesSize,
                                        srcInfosCountAddress, maxNumTriangles, maxNumVertices,
                                        maxNumClusters);

                  cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                               VK_ACCESS_2_SHADER_READ_BIT);
                  cmd->FlushBarriers();
                  device->EndEvent(cmd);
              })
            .AddHandle(buildClusterTriangleInfoBuffer, ResourceUsageType::Read)
            .AddHandle(clasScratchBuffer, ResourceUsageType::RW)
            .AddHandle(clusterAccelSizesHandle, ResourceUsageType::Read)
            .AddHandle(clasGlobalsBuffer, ResourceUsageType::Read);

        rg->StartComputePass(
              computeClasAddressesPipeline, computeClasAddressesLayout, 7,
              [&clasAccelAddresses = this->clusterAccelAddresses,
               &decode             = this->clusterAccelSizes, numNewPages,
               clasGlobalsBuffer   = this->clasGlobalsBuffer](CommandBuffer *cmd) {
                  device->BeginEvent(cmd, "Compute New Clas Addresses");
                  cmd->Dispatch(numNewPages, 1, 1);
                  cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                               VK_ACCESS_2_SHADER_WRITE_BIT,
                               VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
                  cmd->FlushBarriers();
                  device->EndEvent(cmd);
              },
              &computeClasAddressesPush, &pc)
            .AddHandle(evictedPagesBuffer, ResourceUsageType::Read)
            .AddHandle(clusterAccelAddressesHandle, ResourceUsageType::Write)
            .AddHandle(clusterAccelSizesHandle, ResourceUsageType::Read)
            .AddHandle(clasGlobalsBuffer, ResourceUsageType::Write)
            .AddHandle(clasPageInfoBufferHandle, ResourceUsageType::Write)
            .AddHandle(decodeClusterDataBuffer, ResourceUsageType::Read)
            .AddHandle(clusterPageDataBufferHandle, ResourceUsageType::Read);
    }

    // Build the CLAS
    rg->StartPass(
          8,
          [&clasImplicitData              = this->clasImplicitData,
           &clusterAccelAddresses         = this->clusterAccelAddresses,
           &clusterAccelSizes             = this->clusterAccelSizes,
           clasScratchBuffer              = this->clasScratchBuffer,
           buildClusterTriangleInfoBuffer = this->buildClusterTriangleInfoBuffer,
           maxPages = this->maxPages, maxNumTriangles = this->maxNumTriangles,
           maxNumVertices = this->maxNumVertices, maxNumClusters = this->maxNumClusters,
           clasGlobalsBuffer = this->clasGlobalsBuffer, newClasOffset,
           blasDataBuffer    = this->blasDataBuffer](CommandBuffer *cmd) {
              RenderGraph *rg = GetRenderGraph();
              u32 triangleInfoOffset, triangleInfoSize, clasScratchBufferOffset,
                  srcInfosCountOffset;
              GPUBuffer *buildClusterTriangleInfo = rg->GetBuffer(
                  buildClusterTriangleInfoBuffer, triangleInfoOffset, triangleInfoSize);
              u64 buildClusterTriangleInfoAddress =
                  device->GetDeviceAddress(buildClusterTriangleInfo) + triangleInfoOffset;

              GPUBuffer *scratchBuffer =
                  rg->GetBuffer(clasScratchBuffer, clasScratchBufferOffset);
              u64 scratchBufferAddress =
                  device->GetDeviceAddress(scratchBuffer) + clasScratchBufferOffset;

              u64 dstSizesAddress =
                  device->GetDeviceAddress(&clusterAccelSizes) + sizeof(u32) * newClasOffset;
              u32 dstSizesSize =
                  sizeof(u32) * (maxPages * MAX_CLUSTERS_PER_PAGE - newClasOffset);
              u64 dstAddressesAddress = device->GetDeviceAddress(&clusterAccelAddresses) +
                                        sizeof(u32) * newClasOffset;
              u32 dstAddressesSize =
                  sizeof(u64) * (maxPages * MAX_CLUSTERS_PER_PAGE - newClasOffset);

              GPUBuffer *srcInfosCount = rg->GetBuffer(clasGlobalsBuffer, srcInfosCountOffset);
              u64 srcInfosCountAddress = device->GetDeviceAddress(srcInfosCount) +
                                         srcInfosCountOffset +
                                         sizeof(u32) * GLOBALS_CLAS_COUNT_INDEX;

              u64 dstImplicitData = device->GetDeviceAddress(&clasImplicitData);

              device->BeginEvent(cmd, "Build New CLAS");
              cmd->BuildCLAS(CLASOpMode::ExplicitDestinations, dstImplicitData,
                             scratchBufferAddress, buildClusterTriangleInfoAddress,
                             triangleInfoSize, dstAddressesAddress, dstAddressesSize,
                             dstSizesAddress, dstSizesSize, srcInfosCountAddress,
                             maxNumTriangles, maxNumVertices, maxNumClusters);
              cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
              cmd->FlushBarriers();
              device->EndEvent(cmd);
          })
        .AddHandle(clasImplicitDataHandle, ResourceUsageType::Write)
        .AddHandle(clasScratchBuffer, ResourceUsageType::RW)
        .AddHandle(buildClusterTriangleInfoBuffer, ResourceUsageType::Read)
        .AddHandle(clusterAccelAddressesHandle, ResourceUsageType::Read)
        .AddHandle(clusterAccelSizesHandle, ResourceUsageType::Read)
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::Read)
        .AddHandle(vertexBuffer, ResourceUsageType::Read)
        .AddHandle(indexBuffer, ResourceUsageType::Read);
}

void VirtualGeometryManager::PrepareInstances(CommandBuffer *cmd, ResourceHandle sceneBuffer,
                                              bool ptlas)
{
    RenderGraph *rg = GetRenderGraph();
    MergedInstancesPushConstant pc;
    pc.num        = maxPartitions;
    pc.firstFrame = device->frameCount == 0;

    rg->StartComputePass(
          mergedInstancesTestPipeline, mergedInstancesTestLayout, 7,
          [maxPartitions = this->maxPartitions, &infos = this->partitionInfosBuffer,
           sceneBuffer](CommandBuffer *cmd) {
              device->BeginEvent(cmd, "Merged Instances Test");
              cmd->Dispatch((maxPartitions + 31) / 32, 1, 1);
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
              cmd->FlushBarriers();
              device->EndEvent(cmd);
          },
          &mergedInstancesTestPush, &pc)
        .AddHandle(partitionInfosBufferHandle, ResourceUsageType::RW)
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::Write)
        .AddHandle(visiblePartitionsBuffer, ResourceUsageType::Write)
        .AddHandle(evictedPartitionsBuffer, ResourceUsageType::Write)
        .AddHandle(instanceFreeListBufferHandle, ResourceUsageType::RW)
        .AddHandle(sceneBuffer, ResourceUsageType::Read)
        .AddHandle(depthPyramid, ResourceUsageType::Read);

    rg->StartComputePass(freeInstancesPipeline, freeInstancesLayout, 4,
                         [maxInstances = this->maxInstances](CommandBuffer *cmd) {
                             device->BeginEvent(cmd, "Free instances");
                             cmd->Dispatch(maxInstances / 64, 1, 1);
                             cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          VK_ACCESS_2_SHADER_WRITE_BIT,
                                          VK_ACCESS_2_SHADER_READ_BIT |
                                              VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
                             cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          VK_ACCESS_2_SHADER_WRITE_BIT,
                                          VK_ACCESS_2_SHADER_WRITE_BIT);
                             cmd->FlushBarriers();
                             device->EndEvent(cmd);
                         })
        .AddHandle(instancesBufferHandle, ResourceUsageType::RW)
        .AddHandle(evictedPartitionsBuffer, ResourceUsageType::Read)
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::Read)
        .AddHandle(instanceFreeListBufferHandle, ResourceUsageType::RW);

    rg->StartComputePass(
          allocateInstancesPipeline, allocateInstancesLayout, 8,
          [maxPartitions = this->maxPartitions](CommandBuffer *cmd) {
              device->BeginEvent(cmd, "Allocate instances");
              cmd->Dispatch((maxPartitions + 31) / 32, 1, 1);
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
              cmd->FlushBarriers();
              device->EndEvent(cmd);
          })
        .AddHandle(visiblePartitionsBuffer, ResourceUsageType::Read)
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::Read)
        .AddHandle(instanceTransformsBufferHandle, ResourceUsageType::Read)
        .AddHandle(partitionInfosBufferHandle, ResourceUsageType::Read)
        .AddHandle(instanceFreeListBufferHandle, ResourceUsageType::RW)
        .AddHandle(instancesBufferHandle, ResourceUsageType::Write)
        .AddHandle(resourceBufferHandle, ResourceUsageType::Read)
        .AddHandle(instanceResourceIDsBufferHandle, ResourceUsageType::Read);

    rg->StartComputePass(
          instanceCullingPipeline, instanceCullingLayout, 11,
          [&instances = this->instancesBuffer, maxInstances = this->maxInstances,
           blasDataBuffer = this->blasDataBuffer](CommandBuffer *cmd) {
              device->BeginEvent(cmd, "Instance Culling");
              cmd->Dispatch(maxInstances / 64, 1, 1);
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                               VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
              cmd->FlushBarriers();
              device->EndEvent(cmd);
          })
        .AddHandle(sceneBuffer, ResourceUsageType::Read)
        .AddHandle(instancesBufferHandle, ResourceUsageType::RW)
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::Write)
        .AddHandle(resourceAABBBufferHandle, ResourceUsageType::Read)
        .AddHandle(instanceTransformsBufferHandle, ResourceUsageType::Read)
        .AddHandle(partitionInfosBufferHandle, ResourceUsageType::Read)
        .AddHandle(streamingRequestsBuffer, ResourceUsageType::Write)
        .AddHandle(resourceBufferHandle, ResourceUsageType::Read)
        .AddHandle(resourceSharingInfosBufferHandle, ResourceUsageType::Read)
        .AddHandle(maxMinLodLevelBuffer, ResourceUsageType::Write)
        .AddHandle(blasDataBuffer, ResourceUsageType::Write);

    rg->StartComputePass(
          assignInstancesPipeline, assignInstancesLayout, 7,
          [maxInstances = this->maxInstances,
           blasData     = this->blasDataBuffer](CommandBuffer *cmd) {
              device->BeginEvent(cmd, "Assign Instances");
              cmd->Dispatch(maxInstances / 64, 1, 1);
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
              cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
              cmd->FlushBarriers();
              device->EndEvent(cmd);
          })
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::Write)
        .AddHandle(blasDataBuffer, ResourceUsageType::Write)
        .AddHandle(instancesBufferHandle, ResourceUsageType::Read)
        .AddHandle(maxMinLodLevelBuffer, ResourceUsageType::Read)
        .AddHandle(candidateNodeBuffer, ResourceUsageType::Write)
        .AddHandle(queueBuffer, ResourceUsageType::Write)
        .AddHandle(resourceBitVector, ResourceUsageType::Write);
}

void VirtualGeometryManager::HierarchyTraversal(CommandBuffer *cmd,
                                                ResourceHandle gpuSceneBuffer)

{
    RenderGraph *rg = GetRenderGraph();
    rg->StartComputePass(hierarchyTraversalPipeline, hierarchyTraversalLayout, 13,
                         [candidateNodes = this->candidateNodeBuffer](CommandBuffer *cmd) {
                             device->BeginEvent(cmd, "Hierarchy Traversal");

                             cmd->Dispatch(1440, 1, 1);
                             cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                              VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                          VK_ACCESS_2_SHADER_WRITE_BIT,
                                          VK_ACCESS_2_SHADER_READ_BIT |
                                              VK_ACCESS_2_TRANSFER_READ_BIT);
                             cmd->FlushBarriers();

                             device->EndEvent(cmd);
                         })
        .AddHandle(queueBuffer, ResourceUsageType::RW)
        .AddHandle(gpuSceneBuffer, ResourceUsageType::Read)
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::RW)
        .AddHandle(candidateNodeBuffer, ResourceUsageType::RW)
        .AddHandle(candidateClusterBuffer, ResourceUsageType::RW)
        .AddHandle(instancesBufferHandle, ResourceUsageType::Read)
        .AddHandle(hierarchyNodeBufferHandle, ResourceUsageType::Read)
        .AddHandle(visibleClustersBuffer, ResourceUsageType::Write)
        .AddHandle(clusterPageDataBufferHandle, ResourceUsageType::Read)
        .AddHandle(blasDataBuffer, ResourceUsageType::RW)
        .AddHandle(streamingRequestsBuffer, ResourceUsageType::Write)
        .AddHandle(instanceTransformsBufferHandle, ResourceUsageType::Read)
        .AddHandle(partitionInfosBufferHandle, ResourceUsageType::Read);
}

void VirtualGeometryManager::BuildClusterBLAS(CommandBuffer *cmd)
{
    RenderGraph *rg = GetRenderGraph();

    rg->StartIndirectComputePass(
          "Get BLAS Address Offset", getBlasAddressOffsetPipeline, getBlasAddressOffsetLayout,
          2, clasGlobalsBuffer, sizeof(u32) * GLOBALS_BLAS_INDIRECT_X,
          [](CommandBuffer *cmd) {
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
              cmd->FlushBarriers();
          })
        .AddHandle(blasDataBuffer, ResourceUsageType::RW)
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::RW);

    rg->StartIndirectComputePass(
          "Fill BLAS Address Array", fillBlasAddressArrayPipeline, fillBlasAddressArrayLayout,
          6, clasGlobalsBuffer, sizeof(u32) * GLOBALS_CLAS_INDIRECT_X,
          [visibleClustersBuffer = this->visibleClustersBuffer,
           clasGlobalsBuffer     = this->clasGlobalsBuffer](CommandBuffer *cmd) {
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
              cmd->FlushBarriers();
          })
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::Read)
        .AddHandle(visibleClustersBuffer, ResourceUsageType::Read)
        .AddHandle(blasDataBuffer, ResourceUsageType::RW)
        .AddHandle(clusterAccelAddressesHandle, ResourceUsageType::Read)
        .AddHandle(blasClasAddressBuffer, ResourceUsageType::Write)
        .AddHandle(clasPageInfoBufferHandle, ResourceUsageType::Read);

    rg->StartComputePass(
          fillFinestClusterBLASInfoPipeline, fillFinestClusterBLASInfoLayout, 5,
          [fillFinestClusterBLASInfoLayout = this->fillFinestClusterBLASInfoLayout,
           &fillFinestClusterBLASPush      = this->fillFinestClusterBottomLevelInfoPush,
           blasClasAddressBuffer           = this->blasClasAddressBuffer,
           clasGlobalsHandle               = this->clasGlobalsBuffer,
           &clusterAccelAddresses          = this->clusterAccelAddresses](CommandBuffer *cmd) {
              u32 offset, indirectOffset;
              GPUBuffer *buffer = GetRenderGraph()->GetBuffer(blasClasAddressBuffer, offset);
              GPUBuffer *clasGlobalsBuffer =
                  GetRenderGraph()->GetBuffer(clasGlobalsHandle, indirectOffset);
              indirectOffset += sizeof(u32) * GLOBALS_BLAS_INDIRECT_X;

              u64 blasClasAddressBufferAddress = device->GetDeviceAddress(buffer) + offset;

              AddressPushConstant pc;
              pc.addressLowBits  = blasClasAddressBufferAddress & (~0u);
              pc.addressHighBits = blasClasAddressBufferAddress >> 32u;

              device->BeginEvent(cmd, "Fill Finest Cluster BLAS");

              cmd->PushConstants(&fillFinestClusterBLASPush, &pc,
                                 fillFinestClusterBLASInfoLayout.pipelineLayout);
              cmd->DispatchIndirect(clasGlobalsBuffer, indirectOffset);

              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
              cmd->FlushBarriers();
              device->EndEvent(cmd);
          })
        .AddHandle(blasDataBuffer, ResourceUsageType::RW)
        .AddHandle(buildClusterBottomLevelInfoBuffer, ResourceUsageType::Write)
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::RW)
        .AddHandle(instancesBufferHandle, ResourceUsageType::Read)
        .AddHandle(resourceBufferHandle, ResourceUsageType::Write);

    rg->StartIndirectComputePass(
          "Fill Cluster BLAS Info", fillClusterBLASInfoPipeline, fillClusterBLASInfoLayout, 4,
          clasGlobalsBuffer, sizeof(u32) * GLOBALS_BLAS_INDIRECT_X,
          [](CommandBuffer *cmd) {
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                               VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
              cmd->FlushBarriers();
          })
        .AddHandle(blasDataBuffer, ResourceUsageType::RW)
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::Read)
        .AddHandle(instancesBufferHandle, ResourceUsageType::Read)
        .AddHandle(resourceBufferHandle, ResourceUsageType::Read);

    u32 numResources = meshInfos.Length();
    rg->StartPass(
          4,
          [maxTotalClusterCount = this->maxTotalClusterCount,
           maxClusterCountPerAccelerationStructure =
               this->maxClusterCountPerAccelerationStructure,
           numResources,
           buildClusterBottomLevelInfoHandle = this->buildClusterBottomLevelInfoBuffer,
           blasScratchBuffer = this->blasScratchBuffer, blasAccelSizes = this->blasAccelSizes,
           clasGlobalsBuffer = this->clasGlobalsBuffer](CommandBuffer *cmd) {
              RenderGraph *rg = GetRenderGraph();
              u32 infoOffset, infoSize, scratchOffset, srcInfosCountOffset;
              GPUBuffer *buildClusterBottomLevelInfoBuffer =
                  rg->GetBuffer(buildClusterBottomLevelInfoHandle, infoOffset, infoSize);
              u64 srcInfosArray =
                  device->GetDeviceAddress(buildClusterBottomLevelInfoBuffer) + infoOffset;

              GPUBuffer *scratchBuffer = rg->GetBuffer(blasScratchBuffer, scratchOffset);
              u64 scratchBufferAddress =
                  device->GetDeviceAddress(scratchBuffer) + scratchOffset;

              u32 blasAccelSizeOffset;
              GPUBuffer *blasAccelSizesBuffer =
                  rg->GetBuffer(blasAccelSizes, blasAccelSizeOffset);
              u64 dstSizesAddress =
                  device->GetDeviceAddress(blasAccelSizesBuffer) + blasAccelSizeOffset;
              u32 dstSizesSize = sizeof(u32) * numResources;

              GPUBuffer *srcInfosCount = rg->GetBuffer(clasGlobalsBuffer, srcInfosCountOffset);
              u64 srcInfosCountAddress = device->GetDeviceAddress(srcInfosCount) +
                                         srcInfosCountOffset +
                                         sizeof(u32) * GLOBALS_BLAS_FINAL_COUNT_INDEX;

              device->BeginEvent(cmd, "Compute BLAS Sizes");
              cmd->ComputeBLASSizes(srcInfosArray, infoSize, scratchBufferAddress,
                                    dstSizesAddress, dstSizesSize, srcInfosCountAddress, 50000,
                                    40000, numResources);

              cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                           VK_ACCESS_2_SHADER_READ_BIT);
              cmd->FlushBarriers();
              device->EndEvent(cmd);
          })
        .AddHandle(buildClusterBottomLevelInfoBuffer, ResourceUsageType::Read)
        .AddHandle(blasScratchBuffer, ResourceUsageType::RW)
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::Read)
        .AddHandle(blasAccelSizes, ResourceUsageType::Write);

    u64 blasBufferAddress = device->GetDeviceAddress(clasBlasImplicitBuffer.buffer);

    AddressPushConstant pc;
    pc.addressLowBits  = blasBufferAddress & (~0u);
    pc.addressHighBits = blasBufferAddress >> 32u;

    rg->StartIndirectComputePass(
          computeBlasAddressesPipeline, computeBlasAddressesLayout, 4, clasGlobalsBuffer,
          sizeof(u32) * GLOBALS_BLAS_INDIRECT_X,
          [clasGlobalsBuffer = this->clasGlobalsBuffer,
           blasAccel         = this->blasClasAddressBuffer](CommandBuffer *cmd) {
              device->BeginEvent(cmd, "Compute BLAS Addresses");
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
              cmd->FlushBarriers();
              device->EndEvent(cmd);
          },
          &computeBlasAddressesPush, &pc)
        .AddHandle(blasAccelSizes, ResourceUsageType::Read)
        .AddHandle(blasAccelAddresses, ResourceUsageType::Write)
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::RW)
        .AddHandle(blasDataBuffer, ResourceUsageType::Read);

    rg->StartPass(
          7,
          [&blasImplicit      = this->clasBlasImplicitBuffer,
           blasAccelAddresses = this->blasAccelAddresses,
           blasAccelSizes = this->blasAccelSizes, blasScratchBuffer = this->blasScratchBuffer,
           info = this->buildClusterBottomLevelInfoBuffer, maxInstances = this->maxInstances,
           maxClusterCountPerAccelerationStructure =
               this->maxClusterCountPerAccelerationStructure,
           maxTotalClusterCount = this->maxTotalClusterCount,
           clasGlobalsBuffer    = this->clasGlobalsBuffer, numResources](CommandBuffer *cmd) {
              device->BeginEvent(cmd, "Build BLAS");

              RenderGraph *rg = GetRenderGraph();
              u32 infoOffset, infoSize, scratchOffset, srcInfosCountOffset;
              GPUBuffer *infoBuffer = rg->GetBuffer(info, infoOffset, infoSize);
              u64 infoAddress       = device->GetDeviceAddress(infoBuffer) + infoOffset;

              GPUBuffer *scratchBuffer = rg->GetBuffer(blasScratchBuffer, scratchOffset);
              u64 scratchBufferAddress =
                  device->GetDeviceAddress(scratchBuffer) + scratchOffset;

              u32 addressOffset, sizesOffset;
              u64 dstSizesAddress =
                  device->GetDeviceAddress(rg->GetBuffer(blasAccelSizes, sizesOffset)) +
                  sizesOffset;
              u32 dstSizesSize = sizeof(u32) * maxInstances;
              u64 dstAddressesAddress =
                  device->GetDeviceAddress(rg->GetBuffer(blasAccelAddresses, addressOffset)) +
                  addressOffset;
              u32 dstAddressesSize = sizeof(u64) * maxInstances;

              GPUBuffer *srcInfosCount = rg->GetBuffer(clasGlobalsBuffer, srcInfosCountOffset);
              u64 srcInfosCountAddress = device->GetDeviceAddress(srcInfosCount) +
                                         srcInfosCountOffset +
                                         sizeof(u32) * GLOBALS_BLAS_FINAL_COUNT_INDEX;

              u64 dstImplicitData = device->GetDeviceAddress(&blasImplicit);

              cmd->BuildClusterBLAS(CLASOpMode::ExplicitDestinations, dstImplicitData,
                                    scratchBufferAddress, infoAddress, infoSize,
                                    dstAddressesAddress, dstAddressesSize, dstSizesAddress,
                                    dstSizesSize, srcInfosCountAddress, 40000, 50000,
                                    numResources);
              cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                           VK_ACCESS_2_SHADER_READ_BIT);
              cmd->FlushBarriers();
              device->EndEvent(cmd);
          })
        .AddHandle(clasBlasImplicitHandle, ResourceUsageType::Write)
        .AddHandle(blasAccelAddresses, ResourceUsageType::Read)
        .AddHandle(blasAccelSizes, ResourceUsageType::Read)
        .AddHandle(blasScratchBuffer, ResourceUsageType::RW)
        .AddHandle(buildClusterBottomLevelInfoBuffer, ResourceUsageType::Read)
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::Read)
        .AddHandle(blasClasAddressBuffer, ResourceUsageType::Read);
}

void VirtualGeometryManager::AllocateInstances(StaticArray<GPUInstance> &inInstances)
{
#if 0
    for (GPUInstance &instance : gpuInstances)
    {
        MeshInfo &meshInfo          = meshInfos[instance.resourceID];
        instance.voxelAddressOffset = meshInfo.voxelAddressOffset;
        // u32 num            = meshInfo.totalNumVoxelClusters + 1;
        // u32 offset         = virtualInstanceOffset;
        // virtualInstanceOffset += num;
        //
        // instance.virtualInstanceIDOffset = offset;
        // for (Range &range : instanceIDFreeRanges)
        // {
        //     if (range.end - range.begin >= numInstances)
        //     {
        //         u32 start = range.begin;
        //         if (range.end - range.begin > numInstances)
        //         {
        //             range.begin = range.begin + numInstances;
        //         }
        //         else
        //         {
        //             Swap(instanceIDFreeRanges[instanceIDFreeRanges.Length() - 1], range);
        //             instanceIDFreeRanges.size_ -= 1;
        //         }
        //
        //         // instance.instanceIDStart = start;
        //         found = true;
        //         break;
        //     }
        // }
        // Assert(found);
    }
    numInstances += gpuInstances.Length();
#endif
}

void VirtualGeometryManager::BuildPTLAS(CommandBuffer *cmd)
{
    // cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    //              VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    // cmd->Barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    //              VK_ACCESS_2_SHADER_WRITE_BIT,
    //              VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
    // cmd->FlushBarriers();
    RenderGraph *rg = GetRenderGraph();

    rg->StartIndirectComputePass(
          "Update PTLAS Instances", ptlasWriteInstancesPipeline, ptlasWriteInstancesLayout, 9,
          clasGlobalsBuffer, sizeof(u32) * GLOBALS_BLAS_INDIRECT_X,
          [info = this->ptlasWriteInfosBuffer, blasAccel = this->blasAccelAddresses,
           clasGlobalsBuffer = this->clasGlobalsBuffer](CommandBuffer *cmd) {
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
              cmd->FlushBarriers();

              // if (device->frameCount > 100)
              // if (level == 1)
              // {
              //     RenderGraph *rg   = GetRenderGraph();
              //     GPUBuffer *buffer = rg->GetBuffer(clasGlobalsBuffer);
              //     // GPUBuffer *buffer   = &infos;
              //     GPUBuffer readback0 = device->CreateBuffer(
              //         VK_BUFFER_USAGE_TRANSFER_DST_BIT, buffer->size, MemoryUsage::GPU_TO_CPU);
              //     cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
              //                  VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              //                  VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
              //                  VK_ACCESS_2_TRANSFER_READ_BIT);
              //     cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              //                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
              //                  VK_ACCESS_2_TRANSFER_READ_BIT);
              //     // cmd->Barrier(
              //     //     img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
              //     //     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              //     //     VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
              //     //     QueueType_Ignored, QueueType_Ignored, level, 1);
              //
              //     // cmd->Barrier(VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
              //     //     //              VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              //     //     // VK_ACCESS_2_SHADER_WRITE_BIT,
              //     //     //              VK_ACCESS_2_TRANSFER_READ_BIT);
              //     // cmd->FlushBarriers();
              //
              //     // BufferImageCopy copy = {};
              //     // copy.mipLevel        = level;
              //     // copy.extent          = Vec3u(width, height, 1);
              //
              //     cmd->CopyBuffer(&readback0, buffer);
              //     // cmd->CopyImageToBuffer(&readback0, img, &copy, 1);
              //     Semaphore testSemaphore   = device->CreateSemaphore();
              //     testSemaphore.signalValue = 1;
              //     cmd->SignalOutsideFrame(testSemaphore);
              //     device->SubmitCommandBuffer(cmd);
              //     device->Wait(testSemaphore);
              //
              //     u32 *data = (u32 *)readback0.mappedPtr;
              //
              //     int stop = 5;
              // }
          })
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::RW)
        .AddHandle(ptlasWriteInfosBuffer, ResourceUsageType::Write)
        .AddHandle(ptlasUpdateInfosBuffer, ResourceUsageType::Write)
        .AddHandle(blasAccelAddresses, ResourceUsageType::Read)
        .AddHandle(blasDataBuffer, ResourceUsageType::Read)
        .AddHandle(instancesBufferHandle, ResourceUsageType::RW)
        .AddHandle(resourceAABBBufferHandle, ResourceUsageType::Read)
        .AddHandle(instanceTransformsBufferHandle, ResourceUsageType::Read)
        .AddHandle(partitionInfosBufferHandle, ResourceUsageType::Read);

    // Reset free list counts to 0 if they're negative
    // {
    //     cmd->StartBindingCompute(ptlasUpdatePartitionsPipeline,
    //     &ptlasUpdatePartitionsLayout)
    //         .Bind(&instanceIDFreeListBuffer)
    //         .End();
    //     cmd->Dispatch(1, 1, 1);
    //
    //     cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    //                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    //                  VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_WRITE_BIT |
    //                  VK_ACCESS_2_SHADER_READ_BIT);
    //     cmd->FlushBarriers();
    // }

    rg->StartComputePass(
          ptlasUpdateUnusedInstancesPipeline, ptlasUpdateUnusedInstancesLayout, 3,
          [maxInstances = this->maxInstances](CommandBuffer *cmd) {
              device->BeginEvent(cmd, "PTLAS Update Unused Instances");
              cmd->Dispatch(maxInstances / 64, 1, 1);
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
              cmd->FlushBarriers();
              device->EndEvent(cmd);
          })
        .AddHandle(instancesBufferHandle, ResourceUsageType::Read)
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::Write)
        .AddHandle(ptlasWriteInfosBuffer, ResourceUsageType::Write);

    // Update command infos
    rg->StartComputePass(
          ptlasWriteCommandInfosPipeline, ptlasWriteCommandInfosLayout, 2,
          [ptlasWriteInfosBuffer  = this->ptlasWriteInfosBuffer,
           ptlasUpdateInfosBuffer = this->ptlasUpdateInfosBuffer,
           &push                  = this->ptlasWriteCommandInfosPush,
           &layout                = this->ptlasWriteCommandInfosLayout,
           info                   = this->ptlasIndirectCommandBuffer](CommandBuffer *cmd) {
              RenderGraph *rg = GetRenderGraph();
              u32 writeOffset, updateOffset;
              PtlasPushConstant pc;
              pc.writeAddress =
                  device->GetDeviceAddress(rg->GetBuffer(ptlasWriteInfosBuffer, writeOffset)) +
                  writeOffset;
              pc.updateAddress = device->GetDeviceAddress(
                                     rg->GetBuffer(ptlasUpdateInfosBuffer, updateOffset)) +
                                 updateOffset;

              device->BeginEvent(cmd, "Write PTLAS Command Infos");
              cmd->PushConstants(&push, &pc, layout.pipelineLayout);
              cmd->Dispatch(1, 1, 1);
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                               VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
              cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                           VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
              cmd->FlushBarriers();
              device->EndEvent(cmd);
          })
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::RW)
        .AddHandle(ptlasIndirectCommandBuffer, ResourceUsageType::Write);

    rg->StartPass(
          6,
          [&tlasAccelBuffer  = this->tlasAccelBuffer,
           tlasScratchHandle = this->tlasScratchHandle,
           ptlasIndirect     = this->ptlasIndirectCommandBuffer,
           clasGlobalsBuffer = this->clasGlobalsBuffer, maxPartitions = this->maxPartitions,
           ptlasWriteInfosBuffer = this->ptlasWriteInfosBuffer,
           blasAccel             = this->blasAccelAddresses](CommandBuffer *cmd) {
              RenderGraph *rg = GetRenderGraph();

              u64 accel = device->GetDeviceAddress(&tlasAccelBuffer);
              u32 scratchOffset, indirectOffset;
              u64 scratch =
                  device->GetDeviceAddress(rg->GetBuffer(tlasScratchHandle, scratchOffset)) +
                  scratchOffset;
              u64 indirect =
                  device->GetDeviceAddress(rg->GetBuffer(ptlasIndirect, indirectOffset)) +
                  indirectOffset;

              u32 srcInfosOffset;
              GPUBuffer *srcInfosBuffer = rg->GetBuffer(clasGlobalsBuffer, srcInfosOffset);
              u64 srcInfosCount = device->GetDeviceAddress(srcInfosBuffer) + srcInfosOffset +
                                  sizeof(u32) * GLOBALS_PTLAS_OP_TYPE_COUNT_INDEX;

              device->BeginEvent(cmd, "Build PTLAS");
              cmd->BuildPTLAS(accel, scratch, indirect, srcInfosCount, (1u << 22), 1024,
                              maxPartitions, 0);
              device->EndEvent(cmd);
          })
        .AddHandle(tlasAccelHandle, ResourceUsageType::Write)
        .AddHandle(tlasScratchHandle, ResourceUsageType::RW)
        .AddHandle(ptlasIndirectCommandBuffer, ResourceUsageType::Read)
        .AddHandle(ptlasWriteInfosBuffer, ResourceUsageType::Read)
        .AddHandle(ptlasUpdateInfosBuffer, ResourceUsageType::Read)
        .AddHandle(clasGlobalsBuffer, ResourceUsageType::Read);
}

struct GetHierarchyNode
{
    using NodeType = TempHierarchyNode;
    __forceinline TempHierarchyNode *operator()(const BRef &ref)
    {
        return (TempHierarchyNode *)ref.nodePtr.GetPtr();
    }
};

void VirtualGeometryManager::BuildHierarchy(PrimRef *refs, RecordAOSSplits &record,
                                            std::atomic<u32> &numPartitions,
                                            StaticArray<u32> &partitionIndices,
                                            StaticArray<RecordAOSSplits> &records,
                                            bool parallel)
{
    typedef HeuristicObjectBinning<PrimRef> Heuristic;

    HeuristicObjectBinning<PrimRef> heuristic(refs, 0);

    Assert(record.count > 0);

    RecordAOSSplits childRecords[2];
    u32 numChildren = 2;

    Split split = heuristic.Bin(record);

    if (record.count <= 1024)
    {
        u32 threadIndex = GetThreadIndex();
        heuristic.FlushState(split);

        u32 partitionIndex = numPartitions.fetch_add(1, std::memory_order_relaxed);

        Assert(partitionIndex < records.Length());
        records[partitionIndex] = record;

        for (int i = 0; i < record.count; i++)
        {
            PrimRef &ref                 = refs[record.start + i];
            partitionIndices[ref.primID] = partitionIndex;
        }

        return;
    }

    heuristic.Split(split, record, childRecords[0], childRecords[1]);

    if (parallel)
    {
        scheduler.ScheduleAndWait(numChildren, 1, [&](u32 jobID) {
            bool childParallel = childRecords[jobID].count >= BUILD_PARALLEL_THRESHOLD;
            BuildHierarchy(refs, childRecords[jobID], numPartitions, partitionIndices, records,
                           childParallel);
        });
    }
    else
    {
        for (int i = 0; i < numChildren; i++)
        {
            BuildHierarchy(refs, childRecords[i], numPartitions, partitionIndices, records,
                           false);
        }
    }
}

inline AffineSpace QuatToMatrix(Vec4f a)
{
    Vec4f normalized = Normalize(a);
    f32 xx           = normalized.x * normalized.x;
    f32 yy           = normalized.y * normalized.y;
    f32 zz           = normalized.z * normalized.z;
    f32 xy           = normalized.x * normalized.y;
    f32 xz           = normalized.x * normalized.z;
    f32 yz           = normalized.y * normalized.z;
    f32 wx           = normalized.w * normalized.x;
    f32 wy           = normalized.w * normalized.y;
    f32 wz           = normalized.w * normalized.z;

    AffineSpace result(Vec3f(1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)),
                       Vec3f(2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)),
                       Vec3f(2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy)));
    return result;
}

static Vec4f MatrixToQuat(AffineSpace &m)
{
    Vec4f result;
    if (m[0][0] + m[1][1] + m[2][2] > 0.f)
    {
        f64 t    = m[0][0] + m[1][1] + m[2][2] + 1.;
        f64 s    = Sqrt(t) * .5;
        result.w = (f32)(s * t);
        result.z = (f32)((m[0][1] - m[1][0]) * s);
        result.y = (f32)((m[2][0] - m[0][2]) * s);
        result.x = (f32)((m[1][2] - m[2][1]) * s);
    }
    else if (m[0][0] > m[1][1] && m[0][0] > m[2][2])
    {
        f64 t    = m[0][0] - m[1][1] - m[2][2] + 1.;
        f64 s    = Sqrt(t) * 0.5;
        result.x = (f32)(s * t);
        result.y = (f32)((m[0][1] + m[1][0]) * s);
        result.z = (f32)((m[2][0] + m[0][2]) * s);
        result.w = (f32)((m[1][2] - m[2][1]) * s);
    }
    else if (m[1][1] > m[2][2])
    {
        f64 t    = -m[0][0] + m[1][1] - m[2][2] + 1.;
        f64 s    = Sqrt(t) * 0.5;
        result.y = (f32)(s * t);
        result.x = (f32)((m[0][1] + m[1][0]) * s);
        result.w = (f32)((m[2][0] - m[0][2]) * s);
        result.z = (f32)((m[1][2] + m[2][1]) * s);
    }
    else
    {
        f64 t    = -m[0][0] - m[1][1] + m[2][2] + 1.;
        f64 s    = Sqrt(t) * .5;
        result.z = (f32)(s * t);
        result.w = (f32)((m[0][1] - m[1][0]) * s);
        result.x = (f32)((m[2][0] + m[0][2]) * s);
        result.y = (f32)((m[1][2] + m[2][1]) * s);
    }

    result = Normalize(result);
    return result;
}

void VirtualGeometryManager::Test(Arena *arena, CommandBuffer *cmd,
                                  StaticArray<Instance> &inputInstances,
                                  StaticArray<AffineSpace> &transforms)
{
    ScratchArena scratch(&arena, 1);
    scratch.temp.arena->align = 16;

    partitionStreamedIn = BitVector(arena, inputInstances.Length());

    // u32 numInstances;
    // Instance *instances = PushArrayNoZero(scratch.temp.arena, Instance, numInstances);
    // AffineSpace *transforms;

    TempHierarchyNode *nodes =
        PushArrayNoZero(scratch.temp.arena, TempHierarchyNode, totalNumNodes);

    u32 nodeIndex = 0;
    for (auto &meshInfo : meshInfos)
    {
        for (int i = 0; i < meshInfo.numNodes; i++)
        {
            nodes[nodeIndex].node  = &meshInfo.nodes[i];
            nodes[nodeIndex].nodes = nodes;
            nodeIndex++;
        }
    }
    StaticArray<PrimRef> refs(scratch.temp.arena, inputInstances.Length());

    Bounds geom;
    Bounds cent;
    RecordAOSSplits record;

    for (int i = 0; i < inputInstances.Length(); i++)
    {
        Instance &instance = inputInstances[i];

        MeshInfo &meshInfo = meshInfos[instance.id];
        if (meshInfo.ellipsoid.sphere.w == 0.f) continue;

        u32 resourceIndex = instance.id;

        Bounds bounds(meshInfos[resourceIndex].boundsMin, meshInfos[resourceIndex].boundsMax);
        AffineSpace &transform = transforms[instance.transformIndex];

        bounds = Transform(transform, bounds);

        PrimRef ref;
        ref.minX   = -bounds.minP[0];
        ref.minY   = -bounds.minP[1];
        ref.minZ   = -bounds.minP[2];
        ref.maxX   = bounds.maxP[0];
        ref.maxY   = bounds.maxP[1];
        ref.maxZ   = bounds.maxP[2];
        ref.primID = i;

        refs.Push(ref);

        geom.Extend(bounds);
        cent.Extend(bounds.minP + bounds.maxP);
    }
    record.geomBounds = Lane8F32(-geom.minP, geom.maxP);
    record.centBounds = Lane8F32(-cent.minP, cent.maxP);
    record.SetRange(0, refs.Length());

    u32 numNodes                   = 0;
    std::atomic<u32> numPartitions = 0;
    StaticArray<u32> partitionIndices(scratch.temp.arena, inputInstances.Length(),
                                      inputInstances.Length());
    StaticArray<RecordAOSSplits> records(scratch.temp.arena, inputInstances.Length(),
                                         inputInstances.Length());

    bool parallel = inputInstances.Length() >= BUILD_PARALLEL_THRESHOLD;

    BuildHierarchy(refs.data, record, numPartitions, partitionIndices, records, parallel);

    // Repeat with non ellipsoid instances
    refs.Clear();
    geom   = Bounds();
    cent   = Bounds();
    record = RecordAOSSplits();
    for (int i = 0; i < inputInstances.Length(); i++)
    {
        Instance &instance = inputInstances[i];

        MeshInfo &meshInfo = meshInfos[instance.id];
        if (meshInfo.ellipsoid.sphere.w != 0.f) continue;

        u32 resourceIndex = instance.id;

        Bounds bounds(meshInfos[resourceIndex].boundsMin, meshInfos[resourceIndex].boundsMax);
        AffineSpace &transform = transforms[instance.transformIndex];

        bounds = Transform(transform, bounds);

        PrimRef ref;
        ref.minX   = -bounds.minP[0];
        ref.minY   = -bounds.minP[1];
        ref.minZ   = -bounds.minP[2];
        ref.maxX   = bounds.maxP[0];
        ref.maxY   = bounds.maxP[1];
        ref.maxZ   = bounds.maxP[2];
        ref.primID = i;

        refs.Push(ref);

        geom.Extend(bounds);
        cent.Extend(bounds.minP + bounds.maxP);
    }
    record.geomBounds = Lane8F32(-geom.minP, geom.maxP);
    record.centBounds = Lane8F32(-cent.minP, cent.maxP);
    record.SetRange(0, refs.Length());

    BuildHierarchy(refs.data, record, numPartitions, partitionIndices, records, parallel);

    partitionInstanceGraph.InitializeStatic(
        arena, inputInstances.Length(), numPartitions.load(),
        [&](u32 instanceIndex, u32 *offsets, Instance *data = 0) {
            u32 partition = partitionIndices[instanceIndex];
            u32 dataIndex = offsets[partition]++;
            if (data)
            {
                data[dataIndex] = inputInstances[instanceIndex];
            }
            return 1;
        });

    u32 finalNumPartitions = numPartitions.load();
    maxPartitions          = finalNumPartitions;

    allocatedPartitionIndices =
        StaticArray<u32>(arena, finalNumPartitions, finalNumPartitions);
    MemorySet(allocatedPartitionIndices.data, 0xff, sizeof(u32) * finalNumPartitions);

    StaticArray<AccelBuildInfo> accelBuildInfos(scratch.temp.arena, finalNumPartitions);
    StaticArray<Vec2u> mergedPartitionIndices(scratch.temp.arena, finalNumPartitions);

    StaticArray<GPUTransform> instanceTransforms(scratch.temp.arena, inputInstances.Length());
    StaticArray<u32> partitionResourceIDs(scratch.temp.arena, inputInstances.Length());
    StaticArray<PartitionInfo> partitionInfos(scratch.temp.arena, finalNumPartitions);
    u32 numAABBs = 0;

    union Float
    {
        u32 u;
        f32 f;
    };
    // Store the calculating bounding proxies
    for (u32 partitionIndex = 0; partitionIndex < finalNumPartitions; partitionIndex++)
    {
        f32 minTranslate[3];
        f32 maxTranslate[3];

        for (u32 i = 0; i < 3; i++)
        {
            minTranslate[i] = pos_inf;
            maxTranslate[i] = neg_inf;
        }

        float maxError = 0.f;
        u32 count      = partitionInstanceGraph.offsets[partitionIndex + 1] -
                    partitionInstanceGraph.offsets[partitionIndex];
        Vec4f *spheres = PushArrayNoZero(scratch.temp.arena, Vec4f, count);
        for (u32 transformIndex = partitionInstanceGraph.offsets[partitionIndex];
             transformIndex < partitionInstanceGraph.offsets[partitionIndex + 1];
             transformIndex++)
        {
            Instance &instance = partitionInstanceGraph.data[transformIndex];
            auto &transform    = transforms[instance.transformIndex];
            for (u32 i = 0; i < 3; i++)
            {
                minTranslate[i] = Min(minTranslate[i], transform[3][i]);
                maxTranslate[i] = Max(maxTranslate[i], transform[3][i]);
            }

            MeshInfo &meshInfo             = meshInfos[instance.id];
            AffineSpace worldFromEllipsoid = transform * Inverse(meshInfo.ellipsoid.transform);

            Bounds bounds(meshInfo.ellipsoid.sphere.xyz - meshInfo.ellipsoid.sphere.w,
                          meshInfo.ellipsoid.sphere.xyz + meshInfo.ellipsoid.sphere.w);

            Vec3f extent = ToVec3f(bounds.maxP - bounds.minP);
            float scaleX = LengthSquared(worldFromEllipsoid[0]);
            float scaleY = LengthSquared(worldFromEllipsoid[1]);
            float scaleZ = LengthSquared(worldFromEllipsoid[2]);

            float minScale = Sqrt(Min(scaleX, Min(scaleY, scaleZ)));
            float maxScale = Sqrt(Max(scaleX, Max(scaleY, scaleZ)));

            float error         = Max(extent.x, Max(extent.y, extent.z));
            float instanceError = error * maxScale;

            maxError = Max(maxError, instanceError);

            Vec3f sphereCenter = TransformP(worldFromEllipsoid, meshInfo.ellipsoid.sphere.xyz);
            f32 sphereRadius   = meshInfo.ellipsoid.sphere.w * maxScale;

            Vec4f sphere(sphereCenter, sphereRadius);
            spheres[transformIndex - partitionInstanceGraph.offsets[partitionIndex]] = sphere;
        }

        Vec4f lodBounds = ConstructSphereFromSpheres(spheres, count);

        PartitionInfo info   = {};
        info.base            = Vec3f(minTranslate[0], minTranslate[1], minTranslate[2]);
        info.scale           = Vec3f((maxTranslate[0] - minTranslate[0]) / 65535.f,
                                     (maxTranslate[1] - minTranslate[1]) / 65535.f,
                                     (maxTranslate[2] - minTranslate[2]) / 65535.f);
        info.transformOffset = partitionInstanceGraph.offsets[partitionIndex];
        info.transformCount =
            partitionInstanceGraph.offsets[partitionIndex + 1] - info.transformOffset;

        info.lodBounds = lodBounds;
        info.lodError  = maxError;

        for (u32 transformIndex = partitionInstanceGraph.offsets[partitionIndex];
             transformIndex < partitionInstanceGraph.offsets[partitionIndex + 1];
             transformIndex++)
        {
            auto &transform =
                transforms[partitionInstanceGraph.data[transformIndex].transformIndex];

            f32 scaleX    = Length(transform[0]);
            f32 scaleY    = Length(transform[1]);
            f32 scaleZ    = Length(transform[2]);
            f32 scales[3] = {scaleX, scaleY, scaleZ};

            AffineSpace rotation = transform;

            u16 translate16[3];
            for (u32 c = 0; c < 3; c++)
            {
                rotation[c] /= scales[c];

                translate16[c] = u16((transform[3][c] - minTranslate[c]) /
                                         (maxTranslate[c] - minTranslate[c]) * 65535.f +
                                     0.5f);
            }
            Vec4f quat = MatrixToQuat(rotation);

            __m128i i = _mm_cvtps_ph(_mm_setr_ps(scales[0], scales[1], scales[2], 0.f), 0);
            alignas(16) u16 scale16[8];
            _mm_store_si128((__m128i *)scale16, i);

            i = _mm_cvtps_ph(_mm_setr_ps(quat[0], quat[1], quat[2], quat[3]), 0);
            alignas(16) u16 rot16[8];
            _mm_store_si128((__m128i *)rot16, i);

            GPUTransform gpuTransform(scale16[0], scale16[1], scale16[2], rot16[0], rot16[1],
                                      rot16[2], rot16[3], translate16[0], translate16[1],
                                      translate16[2]);
            instanceTransforms.Push(gpuTransform);
            partitionResourceIDs.Push(partitionInstanceGraph.data[transformIndex].id);
        }

        u32 numEllipsoids = 0;
        for (u32 instanceIndex = partitionInstanceGraph.offsets[partitionIndex];
             instanceIndex < partitionInstanceGraph.offsets[partitionIndex + 1];
             instanceIndex++)
        {
            Instance &instance = partitionInstanceGraph.data[instanceIndex];
            MeshInfo &meshInfo = meshInfos[instance.id];
            if (meshInfo.ellipsoid.sphere.w != 0.f)
            {
                numEllipsoids++;
            }
        }

        if (numEllipsoids)
        {
            AccelBuildInfo accelBuildInfo  = {};
            accelBuildInfo.primitiveOffset = sizeof(AABB) * numAABBs;
            accelBuildInfo.primitiveCount  = numEllipsoids;
            numAABBs += accelBuildInfo.primitiveCount;

            accelBuildInfos.Push(accelBuildInfo);
            mergedPartitionIndices.Push(Vec2u(partitionIndex));
            info.flags = PARTITION_FLAG_HAS_PROXIES;
        }
        partitionInfos.Push(info);
    }

    RenderGraph *rg = GetRenderGraph();
    if (accelBuildInfos.Length())
    {
        StaticArray<AccelerationStructureSizes> sizes =
            device->GetBLASBuildSizes(scratch.temp.arena, accelBuildInfos);

        StaticArray<AccelerationStructureCreate> creates(scratch.temp.arena,
                                                         accelBuildInfos.Length());

        u32 totalScratch = 0;
        u32 totalAccel   = 0;
        for (AccelerationStructureSizes sizeInfo : sizes)
        {
            totalAccel = AlignPow2(totalAccel, 256);

            AccelerationStructureCreate create = {};
            create.accelSize                   = sizeInfo.accelSize;
            create.bufferOffset                = totalAccel;
            create.type                        = AccelerationStructureType::Bottom;

            creates.Push(create);

            totalAccel += sizeInfo.accelSize;
            totalScratch += sizeInfo.scratchSize;
        }

        blasProxyScratchBuffer = device->CreateBuffer(
            VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
            totalScratch);
        GPUBuffer accelBuffer =
            device->CreateBuffer(VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                     VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                 totalAccel);

        mergedInstancesAABBBuffer = device->CreateBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            sizeof(AABB) * numAABBs);

        uint64_t scratchDataDeviceAddress = device->GetDeviceAddress(&blasProxyScratchBuffer);
        uint64_t aabbDeviceAddress = device->GetDeviceAddress(&mergedInstancesAABBBuffer);
        totalScratch               = 0;

        for (AccelerationStructureCreate &create : creates)
        {
            create.buffer = &accelBuffer;
        }
        device->CreateAccelerationStructures(creates);

        for (u32 sizeIndex = 0; sizeIndex < sizes.Length(); sizeIndex++)
        {
            AccelerationStructureSizes &sizeInfo = sizes[sizeIndex];
            AccelerationStructureCreate &create  = creates[sizeIndex];

            AccelBuildInfo &buildInfo          = accelBuildInfos[sizeIndex];
            buildInfo.scratchDataDeviceAddress = scratchDataDeviceAddress + totalScratch;
            buildInfo.dataDeviceAddress        = aabbDeviceAddress;
            buildInfo.as                       = create.as;

            totalScratch += sizeInfo.scratchSize;

            partitionInfos[mergedPartitionIndices[sizeIndex].x].mergedProxyDeviceAddress =
                create.asDeviceAddress;
            mergedPartitionIndices[sizeIndex].y = buildInfo.primitiveOffset / sizeof(AABB);
        }
    }

    partitionInfosBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        sizeof(PartitionInfo) * (partitionInfos.Length()));
    partitionInfosBufferHandle =
        rg->RegisterExternalResource("partition infos buffer", &partitionInfosBuffer);
    visiblePartitionsBuffer = rg->CreateBufferResource("visible partitions buffer",
                                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                       sizeof(u32) * finalNumPartitions);
    evictedPartitionsBuffer = rg->CreateBufferResource("evicted partitions buffer",
                                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                       sizeof(Vec2u) * finalNumPartitions);

    u32 tlasScratchSize, tlasAccelSize;
    // device->GetPTLASBuildSizes(maxInstances, maxInstances / maxPartitions,
    // maxPartitions, 0,
    //                            tlasScratchSize, tlasAccelSize);
    device->GetPTLASBuildSizes(2 * maxInstances, 1024, maxPartitions, 0, tlasScratchSize,
                               tlasAccelSize);

    // u32 test, test2;
    // device->GetPTLASBuildSizes(1u << 22u, 1024, 4096, 0, test, test2);

    tlasAccelBuffer =
        device->CreateBuffer(VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             tlasAccelSize);
    tlasAccelHandle   = rg->RegisterExternalResource("tlas accel buffer", &tlasAccelBuffer);
    tlasScratchHandle = rg->CreateBufferResource(
        "tlas scratch buffer",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        tlasScratchSize);

    // partitionCountsBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    //                                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    //                                              sizeof(u32) * finalNumPartitions);
    // partitionReadbackBuffer =
    //     device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    //                          sizeof(u32) * finalNumPartitions, MemoryUsage::GPU_TO_CPU);

    u32 transformSize        = sizeof(instanceTransforms[0]) * instanceTransforms.Length();
    instanceTransformsBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, transformSize);
    instanceTransformsBufferHandle =
        rg->RegisterExternalResource("instance transforms buffer", &instanceTransformsBuffer);

    instanceResourceIDsBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        sizeof(u32) * partitionResourceIDs.Length());
    instanceResourceIDsBufferHandle = rg->RegisterExternalResource(
        "instance resource ids buffer", &instanceResourceIDsBuffer);

    partitionsAndOffset = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        sizeof(Vec2u) * mergedPartitionIndices.Length());

    cmd->SubmitBuffer(&instanceTransformsBuffer, instanceTransforms.data, transformSize);
    cmd->SubmitBuffer(&instanceResourceIDsBuffer, partitionResourceIDs.data,
                      instanceResourceIDsBuffer.size);

    cmd->SubmitBuffer(&partitionInfosBuffer, partitionInfos.data,
                      sizeof(PartitionInfo) * (finalNumPartitions));

    cmd->SubmitBuffer(&partitionsAndOffset, mergedPartitionIndices.data,
                      sizeof(Vec2u) * (mergedPartitionIndices.Length()));

    cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    cmd->FlushBarriers();

    if (accelBuildInfos.Length())
    {
        cmd->StartBindingCompute(decodeMergedInstancesPipeline, &decodeMergedInstancesLayout)
            .Bind(&instanceTransformsBuffer)
            .Bind(&resourceAABBBuffer)
            .Bind(&mergedInstancesAABBBuffer)
            .Bind(&partitionInfosBuffer)
            .Bind(&resourceTruncatedEllipsoidsBuffer)
            .Bind(&partitionsAndOffset)
            .Bind(&instanceResourceIDsBuffer)
            .End();

        cmd->Dispatch(mergedPartitionIndices.Length(), 1, 1);

        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        cmd->FlushBarriers();

        cmd->BuildCustomBLAS(accelBuildInfos);
    }
}

} // namespace rt
