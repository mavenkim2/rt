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
      currentClusterTotal(0), currentGeoMemoryTotal(0), totalNumVirtualPages(0),
      totalNumNodes(0), lruHead(-1), lruTail(-1), resourceSharingInfoOffset(0)
{
    for (u32 i = 0; i < maxVirtualPages; i++)
    {
        virtualTable.Push(VirtualPage{0, PageFlag::NonResident, -1});
    }

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

    string writeClasDefragAddressesName   = "../src/shaders/write_clas_defrag_addresses.spv";
    string writeClasDefragAddressesData   = OS_ReadFile(arena, writeClasDefragAddressesName);
    Shader writeClasDefragAddressesShader = device->CreateShader(
        ShaderStage::Compute, "write clas defrag addresses", writeClasDefragAddressesData);

    string fillBlasAddressArrayName   = "../src/shaders/fill_blas_address_array.spv";
    string fillBlasAddressArrayData   = OS_ReadFile(arena, fillBlasAddressArrayName);
    Shader fillBlasAddressArrayShader = device->CreateShader(
        ShaderStage::Compute, "fill blas address array", fillBlasAddressArrayData);

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

    for (int i = 0; i <= 2; i++)
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

    decodeDgfPush.size   = sizeof(DecodePushConstant);
    decodeDgfPush.offset = 0;
    decodeDgfPush.stage  = ShaderStage::Compute;
    for (int i = 0; i <= 3; i++)
    {
        decodeDgfClustersLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                           VK_SHADER_STAGE_COMPUTE_BIT);
    }
    decodeDgfClustersLayout.AddBinding((u32)RTBindings::ClusterPageData,
                                       DescriptorType::StorageBuffer,
                                       VK_SHADER_STAGE_COMPUTE_BIT);
    decodeDgfClustersPipeline =
        device->CreateComputePipeline(&decodeDgfClustersShader, &decodeDgfClustersLayout,
                                      &decodeDgfPush, "decode dgf clusters");

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
    computeClasAddressesPush.size   = sizeof(ComputeCLASAddressesPushConstant);
    computeClasAddressesPush.offset = 0;
    for (int i = 0; i <= 4; i++)
    {
        computeClasAddressesLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                              VK_SHADER_STAGE_COMPUTE_BIT);
    }
    computeClasAddressesLayout.AddBinding((u32)RTBindings::ClusterPageData,
                                          DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    computeClasAddressesPipeline = device->CreateComputePipeline(
        &computeClasAddressesShader, &computeClasAddressesLayout, &computeClasAddressesPush);

    for (int i = 0; i <= 4; i++)
    {
        writeClasDefragLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    }
    writeClasDefragPipeline =
        device->CreateComputePipeline(&writeClasDefragAddressesShader, &writeClasDefragLayout);

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

    // compute blas addresses
    computeBlasAddressesPush.size   = sizeof(ComputeBLASAddressesPushConstant);
    computeBlasAddressesPush.offset = 0;
    computeBlasAddressesPush.stage  = ShaderStage::Compute;
    for (int i = 0; i <= 3; i++)
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
    for (int i = 1; i <= 7; i++)
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
    for (int i = 0; i <= 3; i++)
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
    maxWriteClusters = MAX_CLUSTERS_PER_PAGE * maxPages;

    pageUploadBuffer = {};
    blasUploadBuffer = {};

    evictedPagesBuffer = rg->CreateBufferResource("evicted pages buffer",
                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                  sizeof(u32) * maxPageInstallsPerFrame);

    clusterPageDataBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, megabytes(512));
    clusterPageDataBufferHandle =
        rg->RegisterExternalResource("cluster page data buffer", &clusterPageDataBuffer);

    clusterAccelAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * (maxPages * MAX_CLUSTERS_PER_PAGE));

    clusterAccelAddressesHandle =
        rg->RegisterExternalResource("cluster accel addresses", &clusterAccelAddresses);
    clusterAccelSizes = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * (maxPages * MAX_CLUSTERS_PER_PAGE));
    clusterAccelSizesHandle =
        rg->RegisterExternalResource("cluster accel sizes buffer", &clusterAccelSizes);

    indexBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        118000000);

    vertexBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(Vec3f) * 2 * 22000000);
    // maxNumVertices * sizeof(Vec3f));

    clasGlobalsBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        sizeof(u32) * GLOBALS_SIZE);

    totalAccelSizesBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                 sizeof(u32) * 2);
    clasGlobalsBufferHandle =
        rg->RegisterExternalResource("clas globals buffer", &clasGlobalsBuffer);

    u32 expectedSize = maxPages * MAX_CLUSTERS_PER_PAGE * 2000;

    clasScratchBuffer = {};

    clasImplicitData =
        device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                             expectedSize);
    clasImplicitDataHandle =
        rg->RegisterExternalResource("clas implicit data", &clasImplicitData);

    blasScratchBuffer = {};

    clasBlasImplicitBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        megabytes(512));

    clasBlasImplicitHandle =
        rg->RegisterExternalResource("clas blas implicit buffer", &clasBlasImplicitBuffer);

    blasDataBuffer                    = rg->CreateBufferResource("blas data buffer",
                                                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                                 sizeof(BLASData) * maxInstances);
    buildClusterBottomLevelInfoBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(BUILD_CLUSTERS_BOTTOM_LEVEL_INFO) * maxInstances);

    blasAccelAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * maxInstances);
    blasAccelAddressesHandle =
        rg->RegisterExternalResource("blas accel addresses", &blasAccelAddresses);
    blasAccelSizes = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * maxInstances);

    resourceBitVector = rg->CreateBufferResource("resource bit vector buffer",
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

    decodeClusterDataBuffer        = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                          sizeof(DecodeClusterData) * 320000);
    buildClusterTriangleInfoBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(BUILD_CLUSTERS_TRIANGLE_INFO) * 320000);

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

u32 VirtualGeometryManager::AddNewMesh2(Arena *arena, CommandBuffer *cmd, string filename)
{
    string clusterPageData = OS_ReadFile(arena, filename);

    Tokenizer tokenizer;
    tokenizer.input  = clusterPageData;
    tokenizer.cursor = clusterPageData.str;

    ClusterFileHeader2 fileHeader;
    GetPointerValue(&tokenizer, &fileHeader);

    u32 numClusters = fileHeader.numClusters;

    u32 size = tokenizer.input.size - sizeof(ClusterFileHeader2);

    if (pageUploadBuffer.size)
    {
        device->DestroyBuffer(&pageUploadBuffer);
        pageUploadBuffer.size = 0;
    }
    pageUploadBuffer = device->CreateBuffer(VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT, size,
                                            MemoryUsage::CPU_TO_GPU);
    MemoryCopy((u8 *)pageUploadBuffer.mappedPtr, tokenizer.cursor, size);

    BufferToBufferCopy copy = {};
    copy.srcOffset          = 0;
    copy.dstOffset          = currentGeoMemoryTotal;
    copy.size               = size;
    cmd->CopyBuffer(&clusterPageDataBuffer, &pageUploadBuffer, &copy, 1);
    cmd->Barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    cmd->FlushBarriers();

    u32 newClasOffset = currentClusterTotal;
    u32 baseAddress   = currentGeoMemoryTotal;
    currentGeoMemoryTotal += size;
    currentGeoMemoryTotal = AlignPow2(currentGeoMemoryTotal, 4);

    u64 indexBufferAddress  = device->GetDeviceAddress(&indexBuffer);
    u64 vertexBufferAddress = device->GetDeviceAddress(&vertexBuffer);

    FillClusterTriangleInfoPushConstant fillPc;
    fillPc.indexBufferBaseAddressLowBits   = indexBufferAddress & 0xffffffff;
    fillPc.indexBufferBaseAddressHighBits  = (indexBufferAddress >> 32u) & 0xffffffff;
    fillPc.vertexBufferBaseAddressLowBits  = vertexBufferAddress & 0xffffffff;
    fillPc.vertexBufferBaseAddressHighBits = (vertexBufferAddress >> 32u) & 0xffffffff;
    fillPc.clusterOffset                   = newClasOffset;
    fillPc.baseAddress                     = baseAddress;
    fillPc.numClusters                     = numClusters;

    cmd->StartBindingCompute(fillClusterTriangleInfoPipeline, &fillClusterTriangleInfoLayout)
        .Bind(&buildClusterTriangleInfoBuffer)
        .Bind(&decodeClusterDataBuffer)
        .Bind(&clasGlobalsBuffer)
        .Bind(&clusterPageDataBuffer)
        .PushConstants(&fillClusterTriangleInfoPush, &fillPc)
        .End();

    u32 numGroups   = (numClusters + 31) / 32;
    u32 groupCountX = numGroups & 65535;
    u32 groupCountY = 1;
    cmd->Dispatch(groupCountX, groupCountY, 1);
    cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                 VK_ACCESS_2_SHADER_READ_BIT);
    cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_WRITE_BIT,
                 VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT);
    cmd->FlushBarriers();

    // static u32 count = 0;
    // if (count > 0)
    // {
    //     // RenderGraph *rg   = GetRenderGraph();
    //     GPUBuffer readback0 =
    //         device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    //                              buildClusterTriangleInfoBuffer.size,
    //                              MemoryUsage::GPU_TO_CPU);
    //     cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    //                  VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    //                  VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    //                  VK_ACCESS_2_TRANSFER_READ_BIT);
    //     cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    //     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    //                  VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    //     // cmd->Barrier(
    //     //     img, VK_IMAGE_LAYOUT_GENERAL,
    //     //     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    //     //     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    //     // VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    //     //     VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
    //     //     QueueType_Ignored, QueueType_Ignored, level, 1);
    //
    //     // cmd->Barrier(VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
    //     //     //              VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    //     //     // VK_ACCESS_2_SHADER_WRITE_BIT,
    //     //     //              VK_ACCESS_2_TRANSFER_READ_BIT);
    //     cmd->FlushBarriers();
    //
    //     // BufferImageCopy copy = {};
    //     // copy.mipLevel        = level;
    //     // copy.extent          = Vec3u(width, height, 1);
    //
    //     cmd->CopyBuffer(&readback0, &buildClusterTriangleInfoBuffer);
    //     // cmd->CopyBuffer(&readback2, buffer1);
    //     // cmd->CopyImageToBuffer(&readback0, img, &copy, 1);
    //     Semaphore testSemaphore   = device->CreateSemaphore();
    //     testSemaphore.signalValue = 1;
    //     cmd->SignalOutsideFrame(testSemaphore);
    //     device->SubmitCommandBuffer(cmd);
    //     device->Wait(testSemaphore);
    //
    //     BUILD_CLUSTERS_TRIANGLE_INFO *data =
    //         (BUILD_CLUSTERS_TRIANGLE_INFO *)readback0.mappedPtr;
    //
    //     int stop = 5;
    // }
    // count++;

    // Decode the clusters
    cmd->ClearBuffer(&indexBuffer);
    cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
    cmd->FlushBarriers();

    DecodePushConstant decodePush;
    decodePush.baseAddress = baseAddress;
    cmd->StartBindingCompute(decodeDgfClustersPipeline, &decodeDgfClustersLayout)
        .Bind(&indexBuffer)
        .Bind(&vertexBuffer)
        .Bind(&decodeClusterDataBuffer)
        .Bind(&clasGlobalsBuffer)
        .Bind(&clusterPageDataBuffer)
        .PushConstants(&decodeDgfPush, &decodePush)
        .End();

    cmd->Dispatch(groupCountX, groupCountY, 1);
    cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                 VK_ACCESS_2_SHADER_WRITE_BIT,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                     VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    cmd->FlushBarriers();

    u32 clasScratchSize, clasAccelerationStructureSize;
    device->GetCLASBuildSizes(CLASOpMode::ExplicitDestinations, numClusters, 118000000 / 3,
                              2 * 22000000, MAX_CLUSTER_TRIANGLES,
                              MAX_CLUSTER_TRIANGLE_VERTICES, clasScratchSize,
                              clasAccelerationStructureSize);

    if (clasScratchSize > clasScratchBuffer.size)
    {
        device->DestroyBuffer(&clasScratchBuffer);
        clasScratchBuffer.size = 0;
    }
    if (clasScratchBuffer.size == 0)
    {
        clasScratchBuffer = device->CreateBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            clasScratchSize);
    }

    u64 address = device->GetDeviceAddress(clasImplicitData.buffer);

    ComputeCLASAddressesPushConstant pc;
    pc.addressLowBits  = address & 0xffffffff;
    pc.addressHighBits = address >> 32u;
    pc.clasOffset      = newClasOffset;

    cmd->ComputeCLASSizes(&buildClusterTriangleInfoBuffer, &clasScratchBuffer,
                          &clusterAccelSizes, &clasGlobalsBuffer,
                          sizeof(u32) * GLOBALS_CLAS_COUNT_INDEX, newClasOffset, 118000000 / 3,
                          2 * 22000000, numClusters);

    cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                 VK_ACCESS_2_SHADER_READ_BIT);
    cmd->FlushBarriers();

    cmd->StartBindingCompute(computeClasAddressesPipeline, &computeClasAddressesLayout)
        .Bind(&clusterAccelAddresses)
        .Bind(&clusterAccelSizes)
        .Bind(&clasGlobalsBuffer)
        .Bind(&decodeClusterDataBuffer)
        .Bind(&totalAccelSizesBuffer)
        .Bind(&clusterPageDataBuffer)
        .PushConstants(&computeClasAddressesPush, &pc)
        .End();

    cmd->Dispatch(groupCountX, groupCountY, 1);
    cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                 VK_ACCESS_2_SHADER_WRITE_BIT,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                     VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    cmd->FlushBarriers();

    cmd->BuildCLAS(CLASOpMode::ExplicitDestinations, &clasImplicitData, &clasScratchBuffer,
                   &buildClusterTriangleInfoBuffer, &clusterAccelAddresses, &clusterAccelSizes,
                   &clasGlobalsBuffer, sizeof(u32) * GLOBALS_CLAS_COUNT_INDEX, numClusters,
                   118000000 / 3, 2 * 22000000, newClasOffset);
    cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                 VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    cmd->FlushBarriers();

    u64 deviceAddress =
        device->GetDeviceAddress(&clusterAccelAddresses) + sizeof(u64) * newClasOffset;

    BUILD_CLUSTERS_BOTTOM_LEVEL_INFO bottom;
    bottom.clusterReferences       = deviceAddress;
    bottom.clusterReferencesStride = 8;
    bottom.clusterReferencesCount  = numClusters;

    if (blasUploadBuffer.size == 0)
    {
        blasUploadBuffer = device->CreateBuffer(VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
                                                sizeof(bottom), MemoryUsage::CPU_TO_GPU);
    }
    MemoryCopy(blasUploadBuffer.mappedPtr, &bottom, sizeof(BUILD_CLUSTERS_BOTTOM_LEVEL_INFO));

    copy.srcOffset = 0;
    copy.dstOffset = 0;
    copy.size      = sizeof(BUILD_CLUSTERS_BOTTOM_LEVEL_INFO);
    cmd->CopyBuffer(&buildClusterBottomLevelInfoBuffer, &blasUploadBuffer, &copy, 1);
    cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    cmd->FlushBarriers();

    u32 clasBlasScratchSize, blasSize;
    device->GetClusterBLASBuildSizes(CLASOpMode::ExplicitDestinations, numClusters,
                                     numClusters, 1, clasBlasScratchSize, blasSize);

    if (blasScratchBuffer.size)
    {
        device->DestroyBuffer(&blasScratchBuffer);
    }
    blasScratchBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        clasBlasScratchSize);

    cmd->ComputeBLASSizes(&buildClusterBottomLevelInfoBuffer, &blasScratchBuffer,
                          &blasAccelSizes, &clasGlobalsBuffer,
                          sizeof(u32) * GLOBALS_BLAS_FINAL_COUNT_INDEX, numClusters,
                          numClusters, 1, meshInfos.Length());

    cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                 VK_ACCESS_2_SHADER_READ_BIT);
    cmd->FlushBarriers();

    u64 blasBufferAddress = device->GetDeviceAddress(clasBlasImplicitBuffer.buffer);

    ComputeBLASAddressesPushConstant blasPc;
    blasPc.addressLowBits  = blasBufferAddress & (~0u);
    blasPc.addressHighBits = blasBufferAddress >> 32u;
    blasPc.blasOffset      = meshInfos.Length();

    cmd->StartBindingCompute(computeBlasAddressesPipeline, &computeBlasAddressesLayout)
        .Bind(&blasAccelSizes)
        .Bind(&blasAccelAddresses)
        .Bind(&clasGlobalsBuffer)
        .Bind(&totalAccelSizesBuffer)
        .PushConstants(&computeBlasAddressesPush, &blasPc)
        .End();
    cmd->Dispatch(1, 1, 1);
    cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                 VK_ACCESS_2_SHADER_WRITE_BIT,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    cmd->FlushBarriers();

    cmd->BuildClusterBLAS(CLASOpMode::ExplicitDestinations, &clasBlasImplicitBuffer,
                          &blasScratchBuffer, &buildClusterBottomLevelInfoBuffer,
                          &blasAccelAddresses, &blasAccelSizes, &clasGlobalsBuffer,
                          sizeof(u32) * GLOBALS_BLAS_FINAL_COUNT_INDEX, numClusters,
                          numClusters, 1, meshInfos.Length());
    cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                 VK_ACCESS_2_SHADER_READ_BIT);
    cmd->FlushBarriers();

    MeshInfo meshInfo = {};
    // meshInfo.pageData  = pageData;
    meshInfo.boundsMin = fileHeader.boundsMin;
    meshInfo.boundsMax = fileHeader.boundsMax;

    meshInfo.ellipsoid.sphere.w = 0.f;
    meshInfo.dataOffset         = baseAddress;

    currentClusterTotal += numClusters;

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

    RenderGraph *rg = GetRenderGraph();
    u32 clasBlasScratchSize, blasSize;
    // maxWriteClusters = 200000;

    for (MeshInfo &meshInfo : meshInfos)
    {
        Resource resource    = {};
        resource.baseAddress = meshInfo.dataOffset;
        resources.Push(resource);

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
           sceneBuffer, visiblePartitions              = this->visiblePartitionsBuffer,
           freePartitions = this->evictedPartitionsBuffer](CommandBuffer *cmd) {
              device->BeginEvent(cmd, "Merged Instances Test");
              cmd->Dispatch((maxPartitions + 31) / 32, 1, 1);
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
              cmd->FlushBarriers();
              device->EndEvent(cmd);
          },
          &mergedInstancesTestPush, &pc)
        .AddHandle(partitionInfosBufferHandle, ResourceUsageType::RW)
        .AddHandle(clasGlobalsBufferHandle, ResourceUsageType::Write)
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
        .AddHandle(clasGlobalsBufferHandle, ResourceUsageType::Read)
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
        .AddHandle(clasGlobalsBufferHandle, ResourceUsageType::Read)
        .AddHandle(instanceTransformsBufferHandle, ResourceUsageType::Read)
        .AddHandle(partitionInfosBufferHandle, ResourceUsageType::Read)
        .AddHandle(instanceFreeListBufferHandle, ResourceUsageType::RW)
        .AddHandle(instancesBufferHandle, ResourceUsageType::Write)
        .AddHandle(resourceBufferHandle, ResourceUsageType::Read)
        .AddHandle(instanceResourceIDsBufferHandle, ResourceUsageType::Read);

    rg->StartComputePass(
          instanceCullingPipeline, instanceCullingLayout, 8,
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
        .AddHandle(clasGlobalsBufferHandle, ResourceUsageType::Write)
        .AddHandle(resourceAABBBufferHandle, ResourceUsageType::Read)
        .AddHandle(instanceTransformsBufferHandle, ResourceUsageType::Read)
        .AddHandle(partitionInfosBufferHandle, ResourceUsageType::Read)
        .AddHandle(resourceBufferHandle, ResourceUsageType::Read)
        .AddHandle(blasDataBuffer, ResourceUsageType::Write);

    rg->StartComputePass(
          assignInstancesPipeline, assignInstancesLayout, 4,
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
        .AddHandle(clasGlobalsBufferHandle, ResourceUsageType::Write)
        .AddHandle(blasDataBuffer, ResourceUsageType::Write)
        .AddHandle(instancesBufferHandle, ResourceUsageType::Read)
        .AddHandle(resourceBitVector, ResourceUsageType::Write);
}

void VirtualGeometryManager::BuildPTLAS(CommandBuffer *cmd, GPUBuffer *debug)
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
          clasGlobalsBufferHandle, sizeof(u32) * GLOBALS_BLAS_INDIRECT_X,
          [&clasGlobalsBuffer = this->clasGlobalsBuffer, debug](CommandBuffer *cmd) {
              cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
              cmd->FlushBarriers();

              cmd->CopyBuffer(debug, &clasGlobalsBuffer);
          })
        .AddHandle(clasGlobalsBufferHandle, ResourceUsageType::RW)
        .AddHandle(ptlasWriteInfosBuffer, ResourceUsageType::Write)
        .AddHandle(ptlasUpdateInfosBuffer, ResourceUsageType::Write)
        .AddHandle(blasAccelAddressesHandle, ResourceUsageType::Read)
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
        .AddHandle(clasGlobalsBufferHandle, ResourceUsageType::Write)
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
        .AddHandle(clasGlobalsBufferHandle, ResourceUsageType::RW)
        .AddHandle(ptlasIndirectCommandBuffer, ResourceUsageType::Write);

    rg->StartPass(
          6,
          [&tlasAccelBuffer   = this->tlasAccelBuffer,
           tlasScratchHandle  = this->tlasScratchHandle,
           ptlasIndirect      = this->ptlasIndirectCommandBuffer,
           &clasGlobalsBuffer = this->clasGlobalsBuffer, maxPartitions = this->maxPartitions,
           ptlasWriteInfosBuffer = this->ptlasWriteInfosBuffer](CommandBuffer *cmd) {
              RenderGraph *rg = GetRenderGraph();

              u64 accel = device->GetDeviceAddress(&tlasAccelBuffer);
              u32 scratchOffset, indirectOffset;
              u64 scratch =
                  device->GetDeviceAddress(rg->GetBuffer(tlasScratchHandle, scratchOffset)) +
                  scratchOffset;
              u64 indirect =
                  device->GetDeviceAddress(rg->GetBuffer(ptlasIndirect, indirectOffset)) +
                  indirectOffset;

              GPUBuffer *srcInfosBuffer = &clasGlobalsBuffer;
              u64 srcInfosCount         = device->GetDeviceAddress(srcInfosBuffer) +
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
        .AddHandle(clasGlobalsBufferHandle, ResourceUsageType::Read);
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

bool VirtualGeometryManager::Test(Arena *arena, CommandBuffer *cmd,
                                  StaticArray<Instance> &inputInstances,
                                  StaticArray<AffineSpace> &transforms)
{
    ScratchArena scratch(&arena, 1);
    scratch.temp.arena->align = 16;

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

    std::atomic<u32> numPartitions = 0;
    StaticArray<u32> partitionIndices(scratch.temp.arena, inputInstances.Length(),
                                      inputInstances.Length());
    StaticArray<RecordAOSSplits> records(scratch.temp.arena, inputInstances.Length(),
                                         inputInstances.Length());

    bool parallel = inputInstances.Length() >= BUILD_PARALLEL_THRESHOLD;

    if (refs.Length())
    {
        BuildHierarchy(refs.data, record, numPartitions, partitionIndices, records, parallel);
    }

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

    if (refs.Length())
    {
        BuildHierarchy(refs.data, record, numPartitions, partitionIndices, records, parallel);
    }

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

    if (mergedPartitionIndices.Length())
    {
        partitionsAndOffset = device->CreateBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sizeof(Vec2u) * mergedPartitionIndices.Length());
        cmd->SubmitBuffer(&partitionsAndOffset, mergedPartitionIndices.data,
                          sizeof(Vec2u) * (mergedPartitionIndices.Length()));
    }

    cmd->SubmitBuffer(&instanceTransformsBuffer, instanceTransforms.data, transformSize);
    cmd->SubmitBuffer(&instanceResourceIDsBuffer, partitionResourceIDs.data,
                      instanceResourceIDsBuffer.size);

    cmd->SubmitBuffer(&partitionInfosBuffer, partitionInfos.data,
                      sizeof(PartitionInfo) * (finalNumPartitions));

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
        return true;
    }
    return false;
}

} // namespace rt
