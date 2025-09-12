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

VirtualGeometryManager::VirtualGeometryManager(CommandBuffer *cmd, Arena *arena)
    : physicalPages(arena, maxPages + 2), virtualTable(arena, maxVirtualPages),
      meshInfos(arena, maxInstances), numInstances(0), numAllocatedPartitions(0),
      virtualInstanceOffset(0), voxelAddressOffset(0), clusterLookupTableOffset(0),
      currentClusterTotal(0), currentTriangleClusterTotal(0), totalNumVirtualPages(0),
      totalNumNodes(0), lruHead(-1), lruTail(-1)
{
    for (u32 i = 0; i < maxVirtualPages; i++)
    {
        virtualTable.Push(VirtualPage{0, PageFlag::NonResident, -1});
    }

    u64 totalNumBytes = 0;

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

    // initialize instance free list
    for (int i = 0; i <= 0; i++)
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
    for (int i = 2; i <= 11; i++)
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

    for (int i = 0; i <= 5; i++)
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

    for (int i = 0; i <= 6; i++)
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
    for (int i = 0; i <= 10; i++)
    {
        ptlasWriteInstancesLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    }
    ptlasWriteInstancesLayout.AddBinding(16, DescriptorType::StorageBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    ptlasWriteInstancesPipeline = device->CreateComputePipeline(
        &ptlasWriteInstancesShader, &ptlasWriteInstancesLayout, 0, "ptlas write instances");

    // ptlas update unused instances
    for (int i = 0; i <= 4; i++)
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

    for (int i = 0; i <= 4; i++)
    {
        decodeMergedInstancesLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                               VK_SHADER_STAGE_COMPUTE_BIT);
    }
    decodeMergedInstancesPipeline = device->CreateComputePipeline(
        &decodeMergedInstancesShader, &decodeMergedInstancesLayout, 0,
        "decode merged instances pipeline");

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

    // Buffers
    maxWriteClusters        = MAX_CLUSTERS_PER_PAGE * maxPages;
    streamingRequestsBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        sizeof(StreamingRequest) * maxStreamingRequestsPerFrame);
    totalNumBytes += streamingRequestsBuffer.size;
    readbackBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(StreamingRequest) * maxStreamingRequestsPerFrame, MemoryUsage::GPU_TO_CPU);
    totalNumBytes += readbackBuffer.size;
    pageUploadBuffer         = {};
    fixupBuffer              = {};
    voxelTransferBuffer      = {};
    voxelPageDecodeBuffer    = {};
    clusterLookupTableBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                    sizeof(u32) * 1000);

    // uploadBuffer     = device->CreateBuffer(
    //     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    //     maxPageInstallsPerFrame * (sizeof(u32) + CLUSTER_PAGE_SIZE) +
    //         2 * maxNodes * sizeof(u32) + maxClusterFixupsPerFrame * sizeof(GPUClusterFixup)
    //         + MAX_CLUSTERS_PER_PAGE * maxPageInstallsPerFrame * sizeof(u64) + sizeof(u32) *
    //         MAX_CLUSTERS_PER_PAGE * maxPageInstallsPerFrame,
    //     MemoryUsage::CPU_TO_GPU);
    // totalNumBytes += uploadBuffer.size;

    evictedPagesBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                              sizeof(u32) * maxPageInstallsPerFrame);
    totalNumBytes += evictedPagesBuffer.size;
    clusterPageDataBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                 maxPages * CLUSTER_PAGE_SIZE);
    totalNumBytes += clusterPageDataBuffer.size;
    hierarchyNodeBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                               maxNodes * sizeof(PackedHierarchyNode));
    totalNumBytes += hierarchyNodeBuffer.size;
    clusterFixupBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        maxClusterFixupsPerFrame * sizeof(GPUClusterFixup));
    totalNumBytes += clusterFixupBuffer.size;

    clusterAccelAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * (maxPages * MAX_CLUSTERS_PER_PAGE));
    totalNumBytes += clusterAccelAddresses.size;
    clusterAccelSizes = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * (maxPages * MAX_CLUSTERS_PER_PAGE));
    totalNumBytes += clusterAccelSizes.size;

    indexBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        maxNumTriangles * 3);
    totalNumBytes += indexBuffer.size;

    vertexBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        maxNumVertices * sizeof(Vec3f));
    totalNumBytes += vertexBuffer.size;
    clasGlobalsBuffer = device->CreateBuffer(
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
                              MAX_CLUSTER_TRIANGLES, MAX_CLUSTER_VERTICES, clasScratchSize,
                              clasAccelerationStructureSize);

    Assert(clasAccelerationStructureSize <= expectedSize);
    clasImplicitData =
        device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                             expectedSize);
    totalNumBytes += clasImplicitData.size;
    clasScratchBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, clasScratchSize);
    totalNumBytes += clasScratchBuffer.size;

    u32 moveScratchSize, moveStructureSize;
    device->GetMoveBuildSizes(CLASOpMode::ExplicitDestinations, maxWriteClusters,
                              clasImplicitData.size, false, moveScratchSize,
                              moveStructureSize);

    moveScratchBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, moveScratchSize);

    totalNumBytes += moveScratchBuffer.size;
    u32 clasBlasScratchSize, clasBlasAccelSize;
    // maxWriteClusters = 200000;
    device->GetClusterBLASBuildSizes(CLASOpMode::ExplicitDestinations, maxTotalClusterCount,
                                     maxClusterCountPerAccelerationStructure, maxInstances,
                                     clasBlasScratchSize, clasBlasAccelSize);

    u32 blasSize = megabytes(512); // clasBlasAccelSize * maxInstances;

    clasBlasScratchBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        clasBlasScratchSize);
    totalNumBytes += clasBlasScratchBuffer.size;

    clasBlasImplicitBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        blasSize);
    totalNumBytes += clasBlasImplicitBuffer.size;

    blasDataBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                          sizeof(BLASData) * maxInstances);
    totalNumBytes += blasDataBuffer.size;
    buildClusterBottomLevelInfoBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(BUILD_CLUSTERS_BOTTOM_LEVEL_INFO) * maxInstances);
    totalNumBytes += buildClusterBottomLevelInfoBuffer.size;
    blasClasAddressBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * maxTotalClusterCount);
    totalNumBytes += blasClasAddressBuffer.size;

    blasAccelAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * maxInstances);
    totalNumBytes += blasAccelAddresses.size;
    blasAccelSizes = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * maxInstances);
    totalNumBytes += blasAccelSizes.size;

    voxelAABBBuffer   = {};
    voxelAddressTable = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        sizeof(VoxelAddressTableEntry) * MAX_CLUSTERS_PER_PAGE * maxPages);
    totalNumBytes += voxelAddressTable.size;

    instanceUploadBuffer = {};
    resourceBitVector    = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1024);
    resourceBuffer       = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1024);

    ptlasIndirectCommandBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(PTLAS_INDIRECT_COMMAND) * 3);
    totalNumBytes += ptlasIndirectCommandBuffer.size;

    ptlasWriteInfosBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(PTLAS_WRITE_INSTANCE_INFO) * maxInstances);
    totalNumBytes += ptlasWriteInfosBuffer.size;
    ptlasUpdateInfosBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(PTLAS_UPDATE_INSTANCE_INFO) * maxInstances);
    totalNumBytes += ptlasUpdateInfosBuffer.size;
    ptlasInstanceBitVectorBuffer =
        device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ((1u << 24u) + 7u) >> 3u);
    ptlasInstanceFrameBitVectorBuffer0 =
        device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ((1u << 24u) + 7u) >> 3u);
    ptlasInstanceFrameBitVectorBuffer1 =
        device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ((1u << 24u) + 7u) >> 3u);
    instanceBitmasksBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 65536);
    tempInstanceBuffer      = {};
    evictedPartitionsBuffer = {};

    totalNumBytes += ptlasInstanceBitVectorBuffer.size;
    totalNumBytes += ptlasInstanceFrameBitVectorBuffer0.size;
    totalNumBytes += ptlasInstanceFrameBitVectorBuffer1.size;
    totalNumBytes += instanceBitmasksBuffer.size;

    // virtualInstanceTableBuffer =
    //     device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(u32) * 1u << 24u);
    // instanceIDFreeListBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    //                                                 sizeof(u32) * ((1u << 24u) + 1u));
    debugBuffer =
        device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(Vec2f) * (1u << 21u));

    // totalNumBytes += virtualInstanceTableBuffer.size;
    // totalNumBytes += instanceIDFreeListBuffer.size;
    // cmd->StartBindingCompute(initializeFreeListPipeline, &initializeFreeListLayout)
    //     .Bind(&instanceIDFreeListBuffer)
    //     .End();
    // cmd->Dispatch(1, 1, 1);

    decodeClusterDataBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                   sizeof(DecodeClusterData) * maxNumClusters);
    totalNumBytes += decodeClusterDataBuffer.size;
    buildClusterTriangleInfoBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(BUILD_CLUSTERS_TRIANGLE_INFO) * maxNumClusters);
    totalNumBytes += buildClusterTriangleInfoBuffer.size;
    clasPageInfoBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                              sizeof(CLASPageInfo) * maxPages);
    totalNumBytes += clasPageInfoBuffer.size;

    moveDescriptors = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(u64) * maxPages * MAX_CLUSTERS_PER_PAGE);
    totalNumBytes += moveDescriptors.size;

    moveDstAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(u64) * maxPages * MAX_CLUSTERS_PER_PAGE);
    totalNumBytes += moveDstAddresses.size;

    moveDstSizes = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * maxPages * MAX_CLUSTERS_PER_PAGE);
    totalNumBytes += moveDstSizes.size;

    AABB aabb;
    aabb.minX = -1;
    aabb.minY = -1;
    aabb.minZ = -1;
    aabb.maxX = 1;
    aabb.maxY = 1;
    aabb.maxZ = 1;

    TransferBuffer tempAabbBuffer = cmd->SubmitBuffer(
        &aabb,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        sizeof(AABB));

    GPUAccelerationStructurePayload payload = cmd->BuildCustomBLAS(&tempAabbBuffer.buffer, 1);
    oneBlasBuildAddress                     = payload.as.address;

    Print("%llu total bytes\n", totalNumBytes);
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
    for (u32 i = 0; i < CHILDREN_PER_HIERARCHY_NODE; i++)
    {
        if (nodes[0].childRef[i] == ~0u) continue;
        Vec3f minP = nodes[0].center[i] - nodes[0].extents[i];
        Vec3f maxP = nodes[0].center[i] + nodes[0].extents[i];

        Bounds b(minP, maxP);

        bounds.Extend(b);
    }

    Vec3f boundsMin = ToVec3f(bounds.minP);
    Vec3f boundsMax = ToVec3f(bounds.maxP);

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

    for (u32 nodeIndex = 0; nodeIndex < numNodes; nodeIndex++)
    {
        PackedHierarchyNode &node = nodes[nodeIndex];
        for (u32 childIndex = 0; childIndex < CHILDREN_PER_HIERARCHY_NODE; childIndex++)
        {
            if (node.leafInfo[childIndex] == ~0u) continue;

            // Map page to nodes

            u32 bitOffset = MAX_CLUSTERS_PER_PAGE_BITS + MAX_CLUSTERS_PER_GROUP_BITS;
            u32 numPages  = BitFieldExtractU32(node.leafInfo[childIndex],
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
        }
    }

    Graph<HierarchyFixup> graph;
    graph.offsets = pageOffsets;
    graph.data    = pageToNodeData;

#if 0
    // Get bounds of instance ref
    for (u32 rebraidIndex = 0; rebraidIndex < numRebraid; rebraidIndex++)
    {
        u32 rebraid               = rebraidIndices[rebraidIndex];
        PackedHierarchyNode &node = nodes[rebraid];
        InstanceRef ref;
        ref.bounds[0] = pos_inf;
        ref.bounds[1] = pos_inf;
        ref.bounds[2] = pos_inf;

        ref.bounds[3] = neg_inf;
        ref.bounds[4] = neg_inf;
        ref.bounds[5] = neg_inf;
        for (int childIndex = 0; childIndex < CHILDREN_PER_HIERARCHY_NODE; childIndex++)
        {
            if (node.childRef[childIndex] == ~0u) continue;

            for (int axis = 0; axis < 3; axis++)
            {
                float maxVal = node.center[childIndex][axis] + node.extents[childIndex][axis];
                float minVal = node.center[childIndex][axis] - node.extents[childIndex][axis];

                ref.bounds[axis]     = Min(ref.bounds[axis], minVal);
                ref.bounds[3 + axis] = Max(ref.bounds[3 + axis], maxVal);
            }
        }
        ref.instanceID = meshInfos.Length();
        ref.nodeOffset = rebraid;
        instanceRefs.Push(ref);
    }
#endif

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

    if (hasEllipsoid)
    {
        GetPointerValue(&tokenizer, &meshInfo.ellipsoid);
    }

    voxelAddressOffset += meshInfo.voxelClusterGroupFixups.Length();
    clusterLookupTableOffset += numVoxelClusters;

    meshInfo.nodes    = nodes;
    meshInfo.numNodes = numNodes;

    totalNumNodes += numNodes;
    totalNumVirtualPages += numPages;

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
    for (MeshInfo &meshInfo : meshInfos)
    {
        Resource resource = {};
        // resource.maxClusters = meshInfo.numFinestClusters;
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
        ellipsoid.sphere = meshInfo.ellipsoid.sphere;
        truncatedEllipsoids.Push(ellipsoid);
    }
    cmd->SubmitBuffer(&resourceBuffer, resources.data, sizeof(Resource) * meshInfos.Length());

    resourceAABBBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                              sizeof(AABB) * resourceAABBs.Length());
    cmd->SubmitBuffer(&resourceAABBBuffer, resourceAABBs.data,
                      sizeof(AABB) * resourceAABBs.Length());

    resourceTruncatedEllipsoidsBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        sizeof(TruncatedEllipsoid) * truncatedEllipsoids.Length());
    cmd->SubmitBuffer(&resourceTruncatedEllipsoidsBuffer, truncatedEllipsoids.data,
                      sizeof(TruncatedEllipsoid) * truncatedEllipsoids.Length());
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

    return true;
}

void VirtualGeometryManager::ProcessRequests(CommandBuffer *cmd, bool test)
{
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
        Assert(totalSize);
        fixupBuffer = device->CreateBuffer(VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT, totalSize,
                                           MemoryUsage::CPU_TO_GPU);

        u32 offset = hierarchyFixupSize;
        MemoryCopy(fixupBuffer.mappedPtr, hierarchyFixupData.data, hierarchyFixupSize);

        cmd->CopyBuffer(&clusterPageDataBuffer, &pageUploadBuffer, pageInstallCopies.data,
                        pageInstallCopies.Length());
        cmd->CopyBuffer(&hierarchyNodeBuffer, &fixupBuffer, nodeInstallCopies.data,
                        nodeInstallCopies.Length());

        BufferToBufferCopy pageInstallCopy;
        pageInstallCopy.srcOffset = offset;
        pageInstallCopy.dstOffset = 0;
        pageInstallCopy.size      = pageInstallSize;
        MemoryCopy((u8 *)fixupBuffer.mappedPtr + pageInstallCopy.srcOffset, pageIndices.data,
                   pageInstallSize);

        offset += pageInstallCopy.size;

        cmd->CopyBuffer(&evictedPagesBuffer, &fixupBuffer, &pageInstallCopy, 1);

        // TODO: this still triggers
        Assert(gpuClusterFixup.Length() <= maxClusterFixupsPerFrame);

        // Cluster fixups
        if (gpuClusterFixup.Length())
        {
            BufferToBufferCopy clusterFixupsCopy;
            clusterFixupsCopy.srcOffset = offset;
            clusterFixupsCopy.dstOffset = 0;
            clusterFixupsCopy.size      = gpuClusterFixupSize;
            MemoryCopy((u8 *)fixupBuffer.mappedPtr + clusterFixupsCopy.srcOffset,
                       gpuClusterFixup.data, clusterFixupsCopy.size);
            cmd->CopyBuffer(&clusterFixupBuffer, &fixupBuffer, &clusterFixupsCopy, 1);
            cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
            offset += clusterFixupsCopy.size;
        }

        cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();

        device->BeginEvent(cmd, "Cluster Fixups");
        NumPushConstant pc;
        pc.num = gpuClusterFixup.Length();
        cmd->StartBindingCompute(clusterFixupPipeline, &clusterFixupLayout)
            .Bind(&clusterFixupBuffer)
            .Bind(&clusterPageDataBuffer)
            .PushConstants(&clusterFixupPush, &pc)
            .End();
        cmd->Dispatch((gpuClusterFixup.Length() + 31) >> 5, 1, 1);
        device->EndEvent(cmd);
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

    u32 numEvictedPages = pageIndices.Length() - evictedPageStart;

    u64 address = device->GetDeviceAddress(clasImplicitData.buffer);

    if (pageIndices.Length() == 0) return;

    // Prepare move descriptors
    {
        DefragPushConstant pc;
        pc.evictedPageStart = evictedPageStart;
        pc.numEvictedPages  = numEvictedPages;

        device->BeginEvent(cmd, "Prepare Defrag Clas");
        cmd->StartBindingCompute(clasDefragPipeline, &clasDefragLayout)
            .Bind(&evictedPagesBuffer)
            .Bind(&clasPageInfoBuffer)
            .Bind(&clusterAccelAddresses)
            .Bind(&clusterAccelSizes)
            .Bind(&clasGlobalsBuffer)
            .Bind(&moveDescriptors)
            .Bind(&moveDstAddresses)
            .Bind(&moveDstSizes)
            .PushConstants(&clasDefragPush, &pc)
            .End();

        cmd->Dispatch(pagesToUpdate, 1, 1);

        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                         VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }

    if (numEvictedPages)
    {
        Print("Evicting %u pages\n", numEvictedPages);

        device->BeginEvent(cmd, "Defrag Clas");
        cmd->MoveCLAS(CLASOpMode::ExplicitDestinations, NULL, &moveScratchBuffer,
                      &moveDstAddresses, &moveDstSizes, &moveDescriptors, &clasGlobalsBuffer,
                      sizeof(u32) * GLOBALS_DEFRAG_CLAS_COUNT, maxWriteClusters,
                      clasImplicitData.size, false);
        device->EndEvent(cmd);

        device->BeginEvent(cmd, "Write CLAS Defrag Addresses");

        cmd->StartBindingCompute(writeClasDefragPipeline, &writeClasDefragLayout)
            .Bind(&clasPageInfoBuffer)
            .Bind(&moveDstAddresses)
            .Bind(&moveDstSizes)
            .Bind(&clusterAccelAddresses)
            .Bind(&clusterAccelSizes)
            .End();

        cmd->Dispatch(pagesToUpdate, 1, 1);
        cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                     VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }

    // Write cluster build descriptors
    {
        device->BeginEvent(cmd, "Fill Cluster Triangle Info");
        u64 indexBufferAddress  = device->GetDeviceAddress(indexBuffer.buffer);
        u64 vertexBufferAddress = device->GetDeviceAddress(vertexBuffer.buffer);
        FillClusterTriangleInfoPushConstant fillPc;
        fillPc.indexBufferBaseAddressLowBits   = indexBufferAddress & 0xffffffff;
        fillPc.indexBufferBaseAddressHighBits  = (indexBufferAddress >> 32u) & 0xffffffff;
        fillPc.vertexBufferBaseAddressLowBits  = vertexBufferAddress & 0xffffffff;
        fillPc.vertexBufferBaseAddressHighBits = (vertexBufferAddress >> 32u) & 0xffffffff;
        fillPc.clusterOffset                   = newClasOffset;

        cmd->StartBindingCompute(fillClusterTriangleInfoPipeline,
                                 &fillClusterTriangleInfoLayout)
            .Bind(&evictedPagesBuffer)
            .Bind(&buildClusterTriangleInfoBuffer)
            .Bind(&decodeClusterDataBuffer)
            .Bind(&clasGlobalsBuffer)
            .Bind(&clasPageInfoBuffer)
            .Bind(&clusterPageDataBuffer)
            .PushConstants(&fillClusterTriangleInfoPush, &fillPc)
            .End();

        cmd->Dispatch(pageIndices.Length(), 1, 1);
        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_SHADER_READ_BIT);
        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }

    // Decode the clusters
    {
        device->BeginEvent(cmd, "Decode Installed Pages");
        cmd->StartBindingCompute(decodeDgfClustersPipeline, &decodeDgfClustersLayout)
            .Bind(&indexBuffer)
            .Bind(&vertexBuffer)
            .Bind(&decodeClusterDataBuffer)
            .Bind(&clasGlobalsBuffer)
            .Bind(&clusterPageDataBuffer)
            .End();

        cmd->DispatchIndirect(&clasGlobalsBuffer, sizeof(u32) * GLOBALS_CLAS_COUNT_INDEX);

        NumPushConstant num;
        num.num = voxelPageDecodeData.Length();
        cmd->StartBindingCompute(decodeVoxelClustersPipeline, &decodeVoxelClustersLayout)
            .Bind(&clasGlobalsBuffer)
            .Bind(&voxelAABBBuffer)
            .Bind(&voxelPageDecodeBuffer)
            .Bind(&clusterPageDataBuffer)
            .PushConstants(&decodeVoxelClustersPush, &num)
            .End();
        cmd->Dispatch(num.num, 1, 1);

        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                         VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }

    if (staticBuildInfos.Length())
    {
        cmd->BuildCustomBLAS(staticBuildInfos);
    }

    // Compute the CLAS addresses
    {
        device->BeginEvent(cmd, "Compute New CLAS Addresses");
        AddressPushConstant pc;
        pc.addressLowBits  = address & 0xffffffff;
        pc.addressHighBits = address >> 32u;

        cmd->ComputeCLASSizes(&buildClusterTriangleInfoBuffer, &clasScratchBuffer,
                              &clusterAccelSizes, &clasGlobalsBuffer,
                              sizeof(u32) * GLOBALS_CLAS_COUNT_INDEX, newClasOffset,
                              maxNumTriangles, maxNumVertices, maxNumClusters);

        cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                     VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();

        cmd->StartBindingCompute(computeClasAddressesPipeline, &computeClasAddressesLayout)
            .Bind(&evictedPagesBuffer)
            .Bind(&clusterAccelAddresses)
            .Bind(&clusterAccelSizes)
            .Bind(&clasGlobalsBuffer)
            .Bind(&clasPageInfoBuffer)
            .Bind(&decodeClusterDataBuffer)
            .Bind(&clusterPageDataBuffer)
            .PushConstants(&computeClasAddressesPush, &pc)
            .End();

        cmd->Dispatch(pageIndices.Length(), 1, 1);

        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }

    // Build the CLAS
    {
        device->BeginEvent(cmd, "Build New CLAS");
        cmd->BuildCLAS(CLASOpMode::ExplicitDestinations, &clasImplicitData, &clasScratchBuffer,
                       &buildClusterTriangleInfoBuffer, &clusterAccelAddresses,
                       &clusterAccelSizes, &clasGlobalsBuffer,
                       sizeof(u32) * GLOBALS_CLAS_COUNT_INDEX, maxNumClusters, maxNumTriangles,
                       maxNumVertices, newClasOffset);
        device->EndEvent(cmd);
    }
}

void VirtualGeometryManager::HierarchyTraversal(CommandBuffer *cmd, GPUBuffer *queueBuffer,
                                                GPUBuffer *gpuSceneBuffer,
                                                GPUBuffer *workItemQueueBuffer,
                                                GPUBuffer *gpuInstancesBuffer,
                                                GPUBuffer *visibleClustersBuffer)

{
    device->BeginEvent(cmd, "Hierarchy Traversal");

    cmd->StartBindingCompute(hierarchyTraversalPipeline, &hierarchyTraversalLayout)
        .Bind(queueBuffer)
        .Bind(gpuSceneBuffer)
        .Bind(&clasGlobalsBuffer)
        .Bind(workItemQueueBuffer, 0, MAX_CANDIDATE_NODES * sizeof(Vec4u))
        .Bind(workItemQueueBuffer, MAX_CANDIDATE_NODES * sizeof(Vec4u),
              MAX_CANDIDATE_CLUSTERS * sizeof(Vec4u))
        .Bind(gpuInstancesBuffer)
        .Bind(&hierarchyNodeBuffer)
        .Bind(visibleClustersBuffer)
        .Bind(&clusterPageDataBuffer)
        .Bind(&blasDataBuffer)
        .Bind(&streamingRequestsBuffer)
        .Bind(&instanceBitmasksBuffer)
        .End();

    cmd->Dispatch(1440, 1, 1);
    cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_SHADER_WRITE_BIT,
                 VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
    cmd->FlushBarriers();

    device->EndEvent(cmd);
}

void VirtualGeometryManager::BuildClusterBLAS(CommandBuffer *cmd,
                                              GPUBuffer *visibleClustersBuffer,
                                              GPUBuffer *gpuInstancesBuffer)
{

    {
        device->BeginEvent(cmd, "Get BLAS Address Offset");
        // Calculate where clas addresses should be written for each blas

        cmd->StartBindingCompute(getBlasAddressOffsetPipeline, &getBlasAddressOffsetLayout)
            .Bind(&blasDataBuffer)
            .Bind(&clasGlobalsBuffer)
            .End();

        cmd->DispatchIndirect(&clasGlobalsBuffer, sizeof(u32) * GLOBALS_BLAS_INDIRECT_X);
        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }

    {
        // Write the clas addresses to a new buffer
        device->BeginEvent(cmd, "Fill BLAS Address Array");
        cmd->StartBindingCompute(fillBlasAddressArrayPipeline, &fillBlasAddressArrayLayout)
            .Bind(&clasGlobalsBuffer)
            .Bind(visibleClustersBuffer)
            .Bind(&blasDataBuffer)
            .Bind(&clusterAccelAddresses)
            .Bind(&blasClasAddressBuffer)
            .Bind(&clasPageInfoBuffer)
            .End();

        cmd->DispatchIndirect(&clasGlobalsBuffer, sizeof(u32) * GLOBALS_CLAS_INDIRECT_X);
        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }

    u64 blasClasAddressBufferAddress = device->GetDeviceAddress(blasClasAddressBuffer.buffer);
    AddressPushConstant pc;
    pc.addressLowBits  = blasClasAddressBufferAddress & (~0u);
    pc.addressHighBits = blasClasAddressBufferAddress >> 32u;

    // Fill finest blas first
    {
        device->BeginEvent(cmd, "Fill Finest Cluster BLAS");

        cmd->StartBindingCompute(fillFinestClusterBLASInfoPipeline,
                                 &fillFinestClusterBLASInfoLayout)
            .Bind(&blasDataBuffer)
            .Bind(&buildClusterBottomLevelInfoBuffer)
            .Bind(&clasGlobalsBuffer)
            .Bind(gpuInstancesBuffer)
            .Bind(&resourceBuffer)
            .Bind(&resourceBitVector)
            .Bind(&instanceBitmasksBuffer)
            .PushConstants(&fillFinestClusterBottomLevelInfoPush, &pc)
            .End();

        cmd->DispatchIndirect(&clasGlobalsBuffer, sizeof(u32) * GLOBALS_BLAS_INDIRECT_X);

        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }

    {
        // Fill out the BUILD_CLUSTERS_BOTTOM_LEVEL_INFO descriptors
        device->BeginEvent(cmd, "Fill Cluster BLAS Info");

        cmd->StartBindingCompute(fillClusterBLASInfoPipeline, &fillClusterBLASInfoLayout)
            .Bind(&blasDataBuffer)
            .Bind(&buildClusterBottomLevelInfoBuffer)
            .Bind(&clasGlobalsBuffer)
            .Bind(gpuInstancesBuffer)
            .Bind(&resourceBuffer)
            .Bind(&instanceBitmasksBuffer)
            .PushConstants(&fillClusterBottomLevelInfoPush, &pc)
            .End();

        cmd->DispatchIndirect(&clasGlobalsBuffer, sizeof(u32) * GLOBALS_BLAS_INDIRECT_X);

        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                         VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }

    {
        u64 blasBufferAddress = device->GetDeviceAddress(clasBlasImplicitBuffer.buffer);

        device->BeginEvent(cmd, "Compute BLAS Addresses");
        AddressPushConstant pc;
        pc.addressLowBits  = blasBufferAddress & (~0u);
        pc.addressHighBits = blasBufferAddress >> 32u;

        // Compute BLAS sizes
        cmd->ComputeBLASSizes(
            &buildClusterBottomLevelInfoBuffer, &clasBlasScratchBuffer, &blasAccelSizes,
            &clasGlobalsBuffer, sizeof(u32) * GLOBALS_BLAS_FINAL_COUNT_INDEX,
            maxTotalClusterCount, maxClusterCountPerAccelerationStructure, maxInstances);
        cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                     VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();

        // Compute BLAS addresses
        cmd->StartBindingCompute(computeBlasAddressesPipeline, &computeBlasAddressesLayout)
            .Bind(&blasAccelSizes)
            .Bind(&blasAccelAddresses)
            .Bind(&clasGlobalsBuffer)
            .Bind(&blasDataBuffer)
            .PushConstants(&computeBlasAddressesPush, &pc)
            .End();

        cmd->DispatchIndirect(&clasGlobalsBuffer, sizeof(u32) * GLOBALS_BLAS_INDIRECT_X);
        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }

    {
        // Build the BLASes
        device->BeginEvent(cmd, "Build BLAS");

        cmd->BuildClusterBLAS(
            CLASOpMode::ExplicitDestinations, &clasBlasImplicitBuffer, &clasBlasScratchBuffer,
            &buildClusterBottomLevelInfoBuffer, &blasAccelAddresses, &blasAccelSizes,
            &clasGlobalsBuffer, sizeof(u32) * GLOBALS_BLAS_FINAL_COUNT_INDEX,
            maxClusterCountPerAccelerationStructure, maxTotalClusterCount, maxInstances);
        cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                     VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }
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

void VirtualGeometryManager::BuildPTLAS(CommandBuffer *cmd, GPUBuffer *gpuInstancesBuffer,
                                        GPUBuffer *debugReadback)
{

    uint buffer = device->frameCount & 1;
    GPUBuffer *lastFrameBitVector =
        buffer ? &ptlasInstanceFrameBitVectorBuffer0 : &ptlasInstanceFrameBitVectorBuffer1;
    GPUBuffer *thisFrameBitVector =
        buffer ? &ptlasInstanceFrameBitVectorBuffer1 : &ptlasInstanceFrameBitVectorBuffer0;
    //
    // cmd->ClearBuffer(thisFrameBitVector);
    // cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    //              VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
    // cmd->FlushBarriers();

    // Update/write PTLAS instances
    {
        device->BeginEvent(cmd, "Update PTLAS Instances");
        u64 writeAddress  = device->GetDeviceAddress(ptlasWriteInfosBuffer.buffer);
        u64 updateAddress = device->GetDeviceAddress(ptlasUpdateInfosBuffer.buffer);

        PtlasPushConstant pc;
        pc.writeAddress  = writeAddress;
        pc.updateAddress = updateAddress;

        cmd->StartBindingCompute(ptlasWriteInstancesPipeline, &ptlasWriteInstancesLayout)
            .Bind(&ptlasInstanceBitVectorBuffer)
            .Bind(thisFrameBitVector)
            .Bind(&clasGlobalsBuffer)
            .Bind(&ptlasWriteInfosBuffer)
            .Bind(&ptlasUpdateInfosBuffer)
            .Bind(&blasAccelAddresses)
            .Bind(&blasDataBuffer)
            .Bind(gpuInstancesBuffer)
            .Bind(&resourceAABBBuffer)
            .Bind(&voxelAddressTable)
            .Bind(&instanceBitmasksBuffer)
            // .Bind(&debugBuffer)
            .End();

        cmd->DispatchIndirect(&clasGlobalsBuffer, sizeof(u32) * GLOBALS_BLAS_INDIRECT_X);

        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }

    {
        // cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        // VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        //              VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        // cmd->FlushBarriers();
        // cmd->CopyBuffer(debugReadback, &clasGlobalsBuffer);
        // cmd->CopyBuffer(debugRdbck2, &instanceIDFreeListBuffer);
    }

    // Reset free list counts to 0 if they're negative
    // {
    //     cmd->StartBindingCompute(ptlasUpdatePartitionsPipeline,
    //     &ptlasUpdatePartitionsLayout)
    //         .Bind(&instanceIDFreeListBuffer)
    //         .End();
    //     cmd->Dispatch(1, 1, 1);
    //
    //     cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    //                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
    //                  VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
    //     cmd->FlushBarriers();
    // }

    // Update unused instances
    {
        device->BeginEvent(cmd, "PTLAS Update Unused Instances");
        cmd->StartBindingCompute(ptlasUpdateUnusedInstancesPipeline,
                                 &ptlasUpdateUnusedInstancesLayout)
            .Bind(lastFrameBitVector)
            .Bind(thisFrameBitVector)
            .Bind(&clasGlobalsBuffer)
            .Bind(&ptlasUpdateInfosBuffer)
            .Bind(&ptlasWriteInfosBuffer)
            .End();
        cmd->Dispatch(1440, 1, 1);
        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }

    // Update command infos
    {
        PtlasPushConstant pc;
        pc.writeAddress  = device->GetDeviceAddress(&ptlasWriteInfosBuffer);
        pc.updateAddress = device->GetDeviceAddress(&ptlasUpdateInfosBuffer);

        device->BeginEvent(cmd, "Write PTLAS Command Infos");
        cmd->StartBindingCompute(ptlasWriteCommandInfosPipeline, &ptlasWriteCommandInfosLayout)
            .Bind(&clasGlobalsBuffer)
            .Bind(&ptlasIndirectCommandBuffer)
            .PushConstants(&ptlasWriteCommandInfosPush, &pc)
            .End();
        cmd->Dispatch(1, 1, 1);
        device->EndEvent(cmd);
    }

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

    cmd->BuildPTLAS(&tlasAccelBuffer, &tlasScratchBuffer, &ptlasIndirectCommandBuffer,
                    &clasGlobalsBuffer, sizeof(u32) * GLOBALS_PTLAS_OP_TYPE_COUNT_INDEX,
                    (1u << 21), 1024, maxPartitions, 0);
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

void VirtualGeometryManager::Test(Arena *arena, CommandBuffer *cmd,
                                  StaticArray<GPUInstance> &inputInstances)
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
    PrimRef *refs = PushArrayNoZero(scratch.temp.arena, PrimRef, inputInstances.Length());

    Bounds geom;
    Bounds cent;
    RecordAOSSplits record;
    for (int i = 0; i < inputInstances.Length(); i++)
    {
        PrimRef &ref          = refs[i];
        GPUInstance &instance = inputInstances[i];

        u32 resourceIndex = instance.resourceID;

        Bounds bounds(meshInfos[resourceIndex].boundsMin, meshInfos[resourceIndex].boundsMax);
        AffineSpace transform(instance.worldFromObject[0][0], instance.worldFromObject[0][1],
                              instance.worldFromObject[0][2], instance.worldFromObject[0][3],
                              instance.worldFromObject[1][0], instance.worldFromObject[1][1],
                              instance.worldFromObject[1][2], instance.worldFromObject[1][3],
                              instance.worldFromObject[2][0], instance.worldFromObject[2][1],
                              instance.worldFromObject[2][2], instance.worldFromObject[2][3]);

        bounds = Transform(transform, bounds);

        ref.minX   = -bounds.minP[0];
        ref.minY   = -bounds.minP[1];
        ref.minZ   = -bounds.minP[2];
        ref.maxX   = bounds.maxP[0];
        ref.maxY   = bounds.maxP[1];
        ref.maxZ   = bounds.maxP[2];
        ref.primID = i;

        geom.Extend(bounds);
        cent.Extend(bounds.minP + bounds.maxP);
    }
    record.geomBounds = Lane8F32(-geom.minP, geom.maxP);
    record.centBounds = Lane8F32(-cent.minP, cent.maxP);
    record.SetRange(0, inputInstances.Length());

    u32 numNodes                   = 0;
    std::atomic<u32> numPartitions = 0;
    StaticArray<u32> partitionIndices(scratch.temp.arena, inputInstances.Length(),
                                      inputInstances.Length());
    StaticArray<RecordAOSSplits> records(scratch.temp.arena, inputInstances.Length(),
                                         inputInstances.Length());

    bool parallel = inputInstances.Length() >= BUILD_PARALLEL_THRESHOLD;

    BuildHierarchy(refs, record, numPartitions, partitionIndices, records, parallel);

    partitionInstanceGraph.InitializeStatic(
        arena, inputInstances.Length(), numPartitions.load(),
        [&](u32 instanceIndex, u32 *offsets, GPUInstance *data = 0) {
            u32 partition = partitionIndices[instanceIndex];
            u32 dataIndex = offsets[partition]++;
            if (data)
            {
                data[dataIndex] = inputInstances[instanceIndex];
                // TODO temp
                inputInstances[instanceIndex].partitionIndex = partition;
                data[dataIndex].groupIndex                   = partition;
            }
            return 1;
        });

    u32 finalNumPartitions = numPartitions.load();
    maxPartitions          = finalNumPartitions;

    allocatedPartitionIndices =
        StaticArray<u32>(arena, finalNumPartitions, finalNumPartitions);
    MemorySet(allocatedPartitionIndices.data, 0xff, sizeof(u32) * finalNumPartitions);
    proxyInstances = StaticArray<GPUInstance>(arena, finalNumPartitions);

    StaticArray<AccelBuildInfo> accelBuildInfos(scratch.temp.arena, finalNumPartitions);

    StaticArray<float> instanceTransforms(scratch.temp.arena, 12 * inputInstances.Length());
    u32 numTransforms = 0;
    u32 numAABBs      = 0;
    // Store the calculating bounding proxies
    for (u32 partitionIndex = 0; partitionIndex < finalNumPartitions; partitionIndex++)
    {
        RecordAOSSplits &record = records[partitionIndex];
        Bounds bounds(record.geomBounds);
        AABB aabb;
        aabb.minX = bounds.minP[0];
        aabb.minY = bounds.minP[1];
        aabb.minZ = bounds.minP[2];
        aabb.maxX = bounds.maxP[0];
        aabb.maxY = bounds.maxP[1];
        aabb.maxZ = bounds.maxP[2];

        GPUInstance gpuInstance = {};

        Vec3<double> minP(ToVec3f(bounds.minP));
        Vec3<double> maxP(ToVec3f(bounds.maxP));
        Vec3<double> center  = (maxP + minP) * 0.5;
        Vec3<double> extents = (maxP - minP) * 0.5;

        AffineSpace transform =
            AffineSpace::Translate(Vec3f(center)) * AffineSpace::Scale(Vec3f(extents));
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                gpuInstance.worldFromObject[r][c] = transform[c][r];
            }
        }
        gpuInstance.resourceID = ~0u;
        gpuInstance.flags      = GPU_INSTANCE_FLAG_MERGED;

        // TODO:
        gpuInstance.partitionIndex = partitionIndex % maxPartitions;
        gpuInstance.groupIndex     = partitionIndex;
        proxyInstances.Push(gpuInstance);

        for (u32 transformIndex = partitionInstanceGraph.offsets[partitionIndex];
             transformIndex < partitionInstanceGraph.offsets[partitionIndex + 1];
             transformIndex++)
        {
            MemoryCopy(instanceTransforms.data + 12 * numTransforms,
                       partitionInstanceGraph.data[transformIndex].worldFromObject,
                       sizeof(f32) * 12);
            numTransforms++;
        }

        u32 count = 0;
        for (u32 instanceIndex = partitionInstanceGraph.offsets[partitionIndex];
             instanceIndex < partitionInstanceGraph.offsets[partitionIndex + 1];
             instanceIndex++)
        {
            GPUInstance &instance = partitionInstanceGraph.data[instanceIndex];
            MeshInfo &meshInfo    = meshInfos[instance.resourceID];
            if (meshInfo.ellipsoid.sphere.w != 0.f)
            {
                count++;
            }
        }

        AccelBuildInfo accelBuildInfo  = {};
        accelBuildInfo.primitiveOffset = sizeof(AABB) * numAABBs;
        accelBuildInfo.primitiveCount  = count;
        numAABBs += accelBuildInfo.primitiveCount;

        accelBuildInfos.Push(accelBuildInfo);
    }

    // TODO: separate this out
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

    GPUBuffer scratchBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
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

    uint64_t scratchDataDeviceAddress = device->GetDeviceAddress(&scratchBuffer);
    uint64_t aabbDeviceAddress        = device->GetDeviceAddress(&mergedInstancesAABBBuffer);
    totalScratch                      = 0;

    for (AccelerationStructureCreate &create : creates)
    {
        create.buffer = &accelBuffer;
    }
    totalAccel = 0;
    device->CreateAccelerationStructures(creates);

    StaticArray<u64> accelDeviceAddresses(scratch.temp.arena, finalNumPartitions);

    for (u32 sizeIndex = 0; sizeIndex < sizes.Length(); sizeIndex++)
    {
        AccelerationStructureSizes &sizeInfo = sizes[sizeIndex];
        AccelerationStructureCreate &create  = creates[sizeIndex];

        accelDeviceAddresses.Push(create.asDeviceAddress);
        AccelBuildInfo &buildInfo          = accelBuildInfos[sizeIndex];
        buildInfo.scratchDataDeviceAddress = scratchDataDeviceAddress + totalScratch;
        buildInfo.dataDeviceAddress        = aabbDeviceAddress;
        buildInfo.as                       = create.as;

        totalScratch += sizeInfo.scratchSize;
    }

    mergedPartitionDeviceAddresses = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                          sizeof(u64) * finalNumPartitions);

    instanceGroupTransformOffsets = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        sizeof(u32) * (finalNumPartitions + 1u));

    u32 tlasScratchSize, tlasAccelSize;
    device->GetPTLASBuildSizes(maxInstances, 1024, maxPartitions, 0, tlasScratchSize,
                               tlasAccelSize);

    tlasScratchBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        tlasScratchSize);
    tlasAccelBuffer =
        device->CreateBuffer(VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             tlasAccelSize);
    // totalNumBytes += tlasAccelBuffer.size;
    // totalNumBytes += tlasScratchBuffer.size;

    instancesBuffer                = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                          sizeof(GPUInstance) * maxInstances);
    partitionCountsBuffer          = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                          sizeof(u32) * finalNumPartitions);
    partitionErrorThresholdsBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                          sizeof(f32) * finalNumPartitions);
    partitionReadbackBuffer =
        device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             sizeof(u32) * finalNumPartitions, MemoryUsage::GPU_TO_CPU);
    instanceTransformsBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                    sizeof(f32) * 12 * numTransforms);
    cmd->SubmitBuffer(&instanceTransformsBuffer, instanceTransforms.data,
                      sizeof(f32) * 12 * numTransforms);

    cmd->SubmitBuffer(&instancesBuffer, proxyInstances.data,
                      sizeof(GPUInstance) * proxyInstances.Length());
    numInstances = proxyInstances.Length();

    // cmd->SubmitBuffer(&instancesBuffer, inputInstances.data,
    //                   sizeof(GPUInstance) * inputInstances.Length());
    // numInstances = inputInstances.Length();
    cmd->SubmitBuffer(&instanceGroupTransformOffsets, partitionInstanceGraph.offsets,
                      sizeof(u32) * (finalNumPartitions + 1));
    cmd->SubmitBuffer(&mergedPartitionDeviceAddresses, accelDeviceAddresses.data,
                      sizeof(u64) * finalNumPartitions);

    cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    cmd->FlushBarriers();

    cmd->StartBindingCompute(decodeMergedInstancesPipeline, &decodeMergedInstancesLayout)
        .Bind(&instanceTransformsBuffer)
        .Bind(&resourceAABBBuffer)
        .Bind(&mergedInstancesAABBBuffer)
        .Bind(&partitionErrorThresholdsBuffer)
        .Bind(&instanceGroupTransformOffsets)
        .End();

    cmd->Dispatch(finalNumPartitions, 1, 1);

    cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                 VK_ACCESS_2_SHADER_WRITE_BIT,
                 VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    cmd->FlushBarriers();

    cmd->BuildCustomBLAS(accelBuildInfos);
}

} // namespace rt
