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

namespace rt
{

static_assert(sizeof(PTLAS_WRITE_INSTANCE_INFO) ==
              sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV));
static_assert(sizeof(PTLAS_UPDATE_INSTANCE_INFO) ==
              sizeof(VkPartitionedAccelerationStructureUpdateInstanceDataNV));

VirtualGeometryManager::VirtualGeometryManager(Arena *arena)
    : physicalPages(arena, maxPages + 2), virtualTable(arena, maxVirtualPages),
      meshInfos(arena, 1), instanceRefs(arena, maxInstances), currentClusterTotal(0),
      totalNumVirtualPages(0), totalNumNodes(0), lruHead(-1), lruTail(-1)
{
    for (u32 i = 0; i < maxVirtualPages; i++)
    {
        virtualTable.Push(-1);
    }

    string decodeDgfClustersName   = "../src/shaders/decode_dgf_clusters.spv";
    string decodeDgfClustersData   = OS_ReadFile(arena, decodeDgfClustersName);
    Shader decodeDgfClustersShader = device->CreateShader(
        ShaderStage::Compute, "decode dgf clusters", decodeDgfClustersData);

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

    // fill cluster triangle info
    fillClusterTriangleInfoPush.stage  = ShaderStage::Compute;
    fillClusterTriangleInfoPush.size   = sizeof(FillClusterTriangleInfoPushConstant);
    fillClusterTriangleInfoPush.offset = 0;

    fillClusterTriangleInfoLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    fillClusterTriangleInfoLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    fillClusterTriangleInfoLayout.AddBinding(2, DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    fillClusterTriangleInfoLayout.AddBinding(3, DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    fillClusterTriangleInfoLayout.AddBinding(4, DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    fillClusterTriangleInfoLayout.AddBinding((u32)RTBindings::ClusterPageData,
                                             DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    fillClusterTriangleInfoPipeline = device->CreateComputePipeline(
        &fillClusterTriangleInfoShader, &fillClusterTriangleInfoLayout,
        &fillClusterTriangleInfoPush, "fill cluster triangle info");

    decodeDgfClustersLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                       VK_SHADER_STAGE_COMPUTE_BIT);
    decodeDgfClustersLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                       VK_SHADER_STAGE_COMPUTE_BIT);
    decodeDgfClustersLayout.AddBinding(2, DescriptorType::StorageBuffer,
                                       VK_SHADER_STAGE_COMPUTE_BIT);
    decodeDgfClustersLayout.AddBinding(3, DescriptorType::StorageBuffer,
                                       VK_SHADER_STAGE_COMPUTE_BIT);
    decodeDgfClustersLayout.AddBinding((u32)RTBindings::ClusterPageData,
                                       DescriptorType::StorageBuffer,
                                       VK_SHADER_STAGE_COMPUTE_BIT);
    decodeDgfClustersPipeline = device->CreateComputePipeline(
        &decodeDgfClustersShader, &decodeDgfClustersLayout, 0, "decode dgf clusters");

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
    computeClasAddressesLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    computeClasAddressesLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    computeClasAddressesLayout.AddBinding(2, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    computeClasAddressesLayout.AddBinding(3, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    computeClasAddressesLayout.AddBinding(4, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    computeClasAddressesLayout.AddBinding((u32)RTBindings::ClusterPageData,
                                          DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    computeClasAddressesPipeline = device->CreateComputePipeline(
        &computeClasAddressesShader, &computeClasAddressesLayout, &computeClasAddressesPush);

    hierarchyTraversalLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(1, DescriptorType::UniformBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(2, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(3, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(4, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(5, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(6, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(7, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(9, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding((u32)RTBindings::ClusterPageData,
                                        DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(10, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);
    hierarchyTraversalLayout.AddBinding(11, DescriptorType::StorageBuffer,
                                        VK_SHADER_STAGE_COMPUTE_BIT);

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

    fillClusterBLASInfoLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    fillClusterBLASInfoLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    fillClusterBLASInfoLayout.AddBinding(2, DescriptorType::StorageBuffer,
                                         VK_SHADER_STAGE_COMPUTE_BIT);
    fillClusterBLASInfoPipeline = device->CreateComputePipeline(
        &fillClusterBLASInfoShader, &fillClusterBLASInfoLayout,
        &fillClusterBottomLevelInfoPush, "fill cluster bottom level info");

    // fill blas address array
    fillBlasAddressArrayLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    fillBlasAddressArrayLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    fillBlasAddressArrayLayout.AddBinding(2, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    fillBlasAddressArrayLayout.AddBinding(3, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    fillBlasAddressArrayLayout.AddBinding(4, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    fillBlasAddressArrayLayout.AddBinding(5, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);

    fillBlasAddressArrayPipeline =
        device->CreateComputePipeline(&fillBlasAddressArrayShader, &fillBlasAddressArrayLayout,
                                      0, "fill blas address array");

    // get blas address offset
    getBlasAddressOffsetLayout.AddBinding(0, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    getBlasAddressOffsetLayout.AddBinding(1, DescriptorType::StorageBuffer,
                                          VK_SHADER_STAGE_COMPUTE_BIT);
    getBlasAddressOffsetPipeline =
        device->CreateComputePipeline(&getBlasAddressOffsetShader, &getBlasAddressOffsetLayout,
                                      0, "get blas address offset");

    // compute blas addresses
    computeBlasAddressesPush.size   = sizeof(AddressPushConstant);
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

    ptlasWriteInstancesPush.size   = sizeof(PtlasPushConstant);
    ptlasWriteInstancesPush.offset = 0;
    ptlasWriteInstancesPush.stage  = ShaderStage::Compute;
    for (int i = 0; i <= 8; i++)
    {
        ptlasWriteInstancesLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
    }
    ptlasWriteInstancesPipeline =
        device->CreateComputePipeline(&ptlasWriteInstancesShader, &ptlasWriteInstancesLayout,
                                      &ptlasWriteInstancesPush, "compute blas addresses");

    // Buffers
    maxWriteClusters        = MAX_CLUSTERS_PER_PAGE * maxPages;
    instanceRefBuffer       = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                   sizeof(InstanceRef) * maxInstances);
    streamingRequestsBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        sizeof(StreamingRequest) * maxStreamingRequestsPerFrame);
    readbackBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(StreamingRequest) * maxStreamingRequestsPerFrame, MemoryUsage::GPU_TO_CPU);
    uploadBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        maxPageInstallsPerFrame * (sizeof(u32) + CLUSTER_PAGE_SIZE) + maxNodes * sizeof(u32),
        MemoryUsage::CPU_TO_GPU);

    evictedPagesBuffer    = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                 sizeof(u32) * maxPageInstallsPerFrame);
    clusterPageDataBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                 maxPages * CLUSTER_PAGE_SIZE);
    hierarchyNodeBuffer   = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                 maxNodes * sizeof(PackedHierarchyNode));

    clusterAccelAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * (maxPages * MAX_CLUSTERS_PER_PAGE));
    clusterAccelSizes = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * (maxPages * MAX_CLUSTERS_PER_PAGE));

    indexBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        maxNumTriangles * 3);

    vertexBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        maxNumVertices * sizeof(Vec3f));
    clasGlobalsBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        sizeof(u32) * GLOBALS_SIZE);

    // TODO
    u32 numBlas      = 1;
    u32 expectedSize = maxPages * MAX_CLUSTERS_PER_PAGE * 2000;

    u32 clasScratchSize, clasAccelerationStructureSize;
    device->GetCLASBuildSizes(CLASOpMode::ExplicitDestinations,
                              maxPages * MAX_CLUSTERS_PER_PAGE,
                              maxPages * MAX_CLUSTERS_PER_PAGE * MAX_CLUSTER_TRIANGLES,
                              maxPages * MAX_CLUSTERS_PER_PAGE * MAX_CLUSTER_VERTICES,
                              clasScratchSize, clasAccelerationStructureSize);

    Assert(clasAccelerationStructureSize <= expectedSize);
    clasImplicitData = device->CreateBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, expectedSize);
    clasScratchBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, clasScratchSize);

    u32 moveScratchSize, moveStructureSize;
    device->GetMoveBuildSizes(CLASOpMode::ExplicitDestinations, maxWriteClusters,
                              clasImplicitData.size, false, moveScratchSize,
                              moveStructureSize);

    moveScratchBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, moveScratchSize);

    u32 clasBlasScratchSize, clasBlasAccelSize;
    device->GetClusterBLASBuildSizes(CLASOpMode::ExplicitDestinations, maxWriteClusters,
                                     maxWriteClusters, 1, clasBlasScratchSize,
                                     clasBlasAccelSize);
    // u32 maxWriteClusters = 200000;
    // device->GetClusterBLASBuildSizes(CLASOpMode::ExplicitDestinations, maxTotalClusterCount,
    //                                  maxClusterCountPerAccelerationStructure, maxInstances,
    //                                  clasBlasScratchSize, clasBlasAccelSize);

    u32 blasSize = megabytes(512); // clasBlasAccelSize * maxInstances;

    clasBlasScratchBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        clasBlasScratchSize);

    clasBlasImplicitBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        blasSize);

    blasDataBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                          sizeof(BLASData) * maxInstances);
    buildClusterBottomLevelInfoBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(BUILD_CLUSTERS_BOTTOM_LEVEL_INFO) * maxInstances);
    blasClasAddressBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * maxWriteClusters);

    blasAccelAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * maxInstances);
    blasAccelSizes = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * maxInstances);

    u32 tlasScratchSize, tlasAccelSize;
    device->GetPTLASBuildSizes(maxInstances, maxInstances, 1, 0, tlasScratchSize,
                               tlasAccelSize);

    tlasScratchBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, tlasScratchSize);
    tlasAccelBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, tlasAccelSize);
    ptlasIndirectCommandBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(PTLAS_INDIRECT_COMMAND));

    ptlasWriteInfosBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(PTLAS_WRITE_INSTANCE_INFO) * maxInstances);
    ptlasUpdateInfosBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(PTLAS_UPDATE_INSTANCE_INFO) * maxInstances);
    ptlasInstanceBitVectorBuffer =
        device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, maxInstances >> 3u);

    ptlas = device->CreatePTLAS(&tlasAccelBuffer);

    decodeClusterDataBuffer        = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                          sizeof(DecodeClusterData) * maxNumClusters);
    buildClusterTriangleInfoBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(BUILD_CLUSTERS_TRIANGLE_INFO) * maxNumClusters);
    clasPageInfoBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                              sizeof(CLASPageInfo) * maxPages);

    moveDescriptors = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(u64) * maxPages * MAX_CLUSTERS_PER_PAGE);

    moveDstAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(u64) * maxPages * MAX_CLUSTERS_PER_PAGE);

    moveDstSizes = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * maxPages * MAX_CLUSTERS_PER_PAGE);
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

u32 VirtualGeometryManager::AddNewMesh(Arena *arena, CommandBuffer *cmd, u8 *pageData,
                                       PackedHierarchyNode *nodes, u32 *rebraidIndices,
                                       u32 numNodes, u32 numPages, u32 numRebraid)
{
    u32 *pageOffsets  = PushArray(arena, u32, numPages + 1);
    u32 *pageOffsets1 = &pageOffsets[1];

    for (u32 nodeIndex = 0; nodeIndex < numNodes; nodeIndex++)
    {
        PackedHierarchyNode &node = nodes[nodeIndex];
        for (u32 childIndex = 0; childIndex < CHILDREN_PER_HIERARCHY_NODE; childIndex++)
        {
            if (node.leafInfo[childIndex] == ~0u) continue;

            // Map page to nodes
            u32 pageIndex = node.childRef[childIndex];
            pageOffsets1[pageIndex]++;
        }
    }

    u32 totalCount = 0;
    for (u32 pageIndex = 0; pageIndex < numPages; pageIndex++)
    {
        u32 count               = pageOffsets1[pageIndex];
        pageOffsets1[pageIndex] = totalCount;
        totalCount += count;
    }

    u32 *pageToNodeData = PushArrayNoZero(arena, u32, totalCount);

    for (u32 nodeIndex = 0; nodeIndex < numNodes; nodeIndex++)
    {
        PackedHierarchyNode &node = nodes[nodeIndex];
        for (u32 childIndex = 0; childIndex < CHILDREN_PER_HIERARCHY_NODE; childIndex++)
        {
            if (node.leafInfo[childIndex] == ~0u) continue;

            // Map page to nodes
            u32 pageIndex = node.childRef[childIndex];
            u32 data      = (nodeIndex << CHILDREN_PER_HIERARCHY_NODE_BITS) | childIndex;

            u32 index             = pageOffsets1[pageIndex]++;
            pageToNodeData[index] = data;
        }
    }

    PageToHierarchyNodeGraph graph;
    graph.offsets = pageOffsets;
    graph.data    = pageToNodeData;

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
        ref.instanceID = 0;
        ref.nodeOffset = rebraid;
        instanceRefs.Push(ref);
    }

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

    MeshInfo meshInfo;
    meshInfo.graph               = graph;
    meshInfo.rebraid             = rebraidIndices;
    meshInfo.hierarchyNodeOffset = totalNumNodes;
    meshInfo.virtualPageOffset   = totalNumVirtualPages;
    meshInfo.pageData            = pageData;
    meshInfo.numRebraid          = numRebraid;

    totalNumNodes += numNodes;
    totalNumVirtualPages += numPages;

    meshInfos.Push(meshInfo);

    return meshInfos.Length() - 1;
}

void VirtualGeometryManager::ProcessRequests(CommandBuffer *cmd)
{
    // TIMED_CPU();
    StreamingRequest *requests = (StreamingRequest *)readbackBuffer.mappedPtr;
    u32 numRequests            = requests[0].pageIndex_numPages;
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
    struct Handle
    {
        u64 sortKey;
        int index;
        u32 instanceID;
    };

    union Float
    {
        f32 f;
        int i;
    };

    u32 totalNumPages = 0;
    for (u32 requestIndex = 0; requestIndex < numRequests; requestIndex++)
    {
        u32 pageIndex_numPages = requests[requestIndex].pageIndex_numPages;
        u32 pageCount = BitFieldExtractU32(pageIndex_numPages, MAX_PARTS_PER_GROUP_BITS, 0);
        u32 pageStartIndex = pageIndex_numPages >> MAX_PARTS_PER_GROUP_BITS;

        totalNumPages += pageCount;
    }
    Handle *handles = PushArrayNoZero(scratch.temp.arena, Handle, totalNumPages);

    // Prune duplicate page requests
    u32 numHandles = 0;
    for (u32 requestIndex = 0; requestIndex < numRequests; requestIndex++)
    {
        StreamingRequest &request = requests[requestIndex];
        u32 pageCount =
            BitFieldExtractU32(request.pageIndex_numPages, MAX_PARTS_PER_GROUP_BITS, 0);
        u32 pageStartIndex = request.pageIndex_numPages >> MAX_PARTS_PER_GROUP_BITS;

        for (u32 pageIndex = pageStartIndex; pageIndex < pageStartIndex + pageCount;
             pageIndex++)
        {
            u32 handleIndex                 = numHandles++;
            handles[handleIndex].sortKey    = ((u64)pageIndex << 32u) | request.instanceID;
            handles[handleIndex].index      = requestIndex;
            handles[handleIndex].instanceID = request.instanceID;
        }
    }

    SortHandles(handles, numHandles);

    u64 prevKey             = handles[0].sortKey;
    u32 numCompactedHandles = 0;
    f32 maxPriority         = neg_inf;

    // Compact handles
    for (u32 handleIndex = 0; handleIndex < numHandles; handleIndex++)
    {
        Handle handle                   = handles[handleIndex];
        const StreamingRequest &request = requests[handle.index];

        if (prevKey != handle.sortKey)
        {
            maxPriority = neg_inf;
            prevKey     = handle.sortKey;
            numCompactedHandles++;
        }

        if (request.priority > maxPriority)
        {
            Float f;
            f.f = request.priority;

            maxPriority                             = request.priority;
            handles[numCompactedHandles].sortKey    = f.i;
            handles[numCompactedHandles].index      = handle.sortKey >> 32u;
            handles[numCompactedHandles].instanceID = handle.instanceID;
        }
    }
    numCompactedHandles++;

    SortHandles(handles, numCompactedHandles);

    struct PageRequest
    {
        u32 instanceID;
        int virtualPageIndex;
        u32 pageIndex;
    };
    StaticArray<PageRequest> unloadedRequests(scratch.temp.arena, numCompactedHandles);
    int firstUsedLRUPage = -1;

    // Sorted by ascending priority
    for (u32 requestIndex = 0; requestIndex < numCompactedHandles; requestIndex++)
    {
        Handle &handle = handles[requestIndex];

        u32 pageIndex = handle.index;

        u32 virtualPageOffset = meshInfos[handle.instanceID].virtualPageOffset;

        int virtualPageIndex  = virtualPageOffset + pageIndex;
        int physicalPageIndex = virtualTable[virtualPageIndex];

        if (physicalPageIndex == -1)
        {
            PageRequest pageRequest;
            pageRequest.instanceID       = handle.instanceID;
            pageRequest.virtualPageIndex = virtualPageIndex;
            pageRequest.pageIndex        = pageIndex;
            unloadedRequests.Push(pageRequest);
        }
        else
        {
            firstUsedLRUPage = firstUsedLRUPage == -1 ? physicalPageIndex : firstUsedLRUPage;
            UnlinkLRU(physicalPageIndex);
            LinkLRU(physicalPageIndex);
        }
    }

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

        LinkLRU(physicalPages.Length() - 1);
        pageIndices.Push(physicalPages.Length() - 1);
    }

    u32 evictedPageStart = pageIndices.Length();

    if (pageIndices.Length() < maxPageInstallsPerFrame &&
        pageIndices.Length() != unloadedRequests.Length())
    {
        Assert(lruTail != -1);
        int lruIndex = lruTail;

        while (pageIndices.Length() < maxPageInstallsPerFrame &&
               pageIndices.Length() < unloadedRequests.Length() &&
               lruIndex != firstUsedLRUPage && lruIndex != lruHead)
        {
            pageIndices.Push(lruIndex);
            currentClusterTotal -= physicalPages[lruIndex].numClusters;
            lruIndex = physicalPages[lruIndex].prevPage;
        }
    }

    u32 newClasOffset = currentClusterTotal;

    // Upload pages to GPU and apply hierarchy fixups
    StaticArray<BufferToBufferCopy> pageInstallCopies(scratch.temp.arena,
                                                      maxPageInstallsPerFrame);
    StaticArray<BufferToBufferCopy> nodeInstallCopies(scratch.temp.arena, maxNodes);
    u32 offset = 0;

    for (int requestIndex = 0; requestIndex < pageIndices.Length(); requestIndex++)
    {
        PageRequest &request = unloadedRequests[unloadedRequests.Length() - 1 - requestIndex];
        u32 gpuPageIndex     = pageIndices[requestIndex];
        virtualTable[request.virtualPageIndex] = gpuPageIndex;

        Page &page = physicalPages[gpuPageIndex];
        if (page.virtualIndex != -1)
        {
            virtualTable[page.virtualIndex] = -1;
        }
        page.virtualIndex = request.virtualPageIndex;

        BufferToBufferCopy copy;
        copy.srcOffset = offset;
        copy.dstOffset = CLUSTER_PAGE_SIZE * gpuPageIndex;
        copy.size      = CLUSTER_PAGE_SIZE;
        pageInstallCopies.Push(copy);

        u8 *buffer = meshInfos[request.instanceID].pageData;
        u8 *src    = buffer + CLUSTER_PAGE_SIZE * request.pageIndex;

        u32 numClustersInPage = *(u32 *)src;

        currentClusterTotal += numClustersInPage;
        physicalPages[gpuPageIndex].numClusters = numClustersInPage;

        MemoryCopy((u8 *)uploadBuffer.mappedPtr + offset, src, CLUSTER_PAGE_SIZE);
        offset += CLUSTER_PAGE_SIZE;

        // Fix up hierarchy nodes
        MeshInfo &meshInfo      = meshInfos[request.instanceID];
        u32 hierarchyNodeOffset = meshInfo.hierarchyNodeOffset;
        u32 *offsets            = meshInfo.graph.offsets;
        u32 *nodeData           = meshInfo.graph.data;

        for (u32 graphIndex = offsets[request.pageIndex];
             graphIndex < offsets[request.pageIndex + 1]; graphIndex++)
        {
            u32 hierarchyNodeIndex = nodeData[graphIndex] >> CHILDREN_PER_HIERARCHY_NODE_BITS;
            u32 childIndex         = nodeData[graphIndex] & (CHILDREN_PER_HIERARCHY_NODE - 1u);

            BufferToBufferCopy hierarchyCopy;
            hierarchyCopy.srcOffset =
                hierarchyUploadBufferOffset + nodeInstallCopies.Length() * sizeof(u32);
            hierarchyCopy.dstOffset =
                sizeof(PackedHierarchyNode) * (hierarchyNodeOffset + hierarchyNodeIndex) +
                OffsetOf(PackedHierarchyNode, childRef) + sizeof(u32) * childIndex;
            hierarchyCopy.size = sizeof(u32);

            nodeInstallCopies.Push(hierarchyCopy);

            *(u32 *)((u8 *)uploadBuffer.mappedPtr + hierarchyCopy.srcOffset) = gpuPageIndex;
        }
    }

    // TODO: direct storage
    if (pageIndices.Length())
    {
        cmd->CopyBuffer(&clusterPageDataBuffer, &uploadBuffer, pageInstallCopies.data,
                        pageInstallCopies.Length());
        cmd->CopyBuffer(&hierarchyNodeBuffer, &uploadBuffer, nodeInstallCopies.data,
                        nodeInstallCopies.Length());
        u32 pageInstallSize = pageIndices.Length() * sizeof(u32);
        BufferToBufferCopy pageInstallCopy;
        pageInstallCopy.srcOffset = evictedPagesOffset;
        pageInstallCopy.dstOffset = 0;
        pageInstallCopy.size      = pageInstallSize;
        MemoryCopy((u8 *)uploadBuffer.mappedPtr + evictedPagesOffset, pageIndices.data,
                   pageInstallSize);

        cmd->CopyBuffer(&evictedPagesBuffer, &uploadBuffer, &pageInstallCopy, 1);

        cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
    }

    u32 numEvictedPages = pageIndices.Length() - evictedPageStart;

    u64 address = device->GetDeviceAddress(clasImplicitData.buffer);
    // Prepare move descriptors
    if (pagesToUpdate && pageIndices.Length())
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
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_SHADER_READ_BIT);
        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
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

    if (pageIndices.Length() == 0) return;

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
        cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                     VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
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
        .Bind(&blasDataBuffer)
        .Bind(&clusterPageDataBuffer)
        .Bind(&streamingRequestsBuffer)
        .Bind(&instanceRefBuffer)
        .End();

    cmd->Dispatch(1440, 1, 1);
    cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                 VK_ACCESS_2_SHADER_READ_BIT);
    cmd->FlushBarriers();

    device->EndEvent(cmd);
}

void VirtualGeometryManager::BuildClusterBLAS(CommandBuffer *cmd,
                                              GPUBuffer *visibleClustersBuffer)
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
                     VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }

    // if (device->frameCount > 100)
    // {
    //     GPUBuffer readback = device->CreateBuffer(
    //         VK_BUFFER_USAGE_TRANSFER_DST_BIT, blasDataBuffer.size, MemoryUsage::GPU_TO_CPU);
    //     Semaphore testSemaphore   = device->CreateSemaphore();
    //     testSemaphore.signalValue = 1;
    //     cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    //     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    //                  VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    //     cmd->FlushBarriers();
    //     cmd->CopyBuffer(&readback, &blasDataBuffer);
    //     cmd->SignalOutsideFrame(testSemaphore);
    //
    //     device->SubmitCommandBuffer(cmd);
    //     device->Wait(testSemaphore);
    //
    //     BLASData *data = (BLASData *)readback.mappedPtr;
    //     // BUILD_CLUSTERS_BOTTOM_LEVEL_INFO *data =
    //     //     (BUILD_CLUSTERS_BOTTOM_LEVEL_INFO *)readback.mappedPtr;
    //     // AccelerationStructureInstance *data =
    //     //     (AccelerationStructureInstance *)readback.mappedPtr;
    //     // StreamingRequest *data = (StreamingRequest *)readback.mappedPtr;
    //     // PTLAS_WRITE_INSTANCE_INFO *data =
    //     //     (PTLAS_WRITE_INSTANCE_INFO *)readback.mappedPtr;
    //     // Vec4u *data = (Vec4u *)readback.mappedPtr;
    //
    //     int stop = 5;
    // }

    {
        u64 blasClasAddressBufferAddress =
            device->GetDeviceAddress(blasClasAddressBuffer.buffer);
        AddressPushConstant pc;
        pc.addressLowBits  = blasClasAddressBufferAddress & (~0u);
        pc.addressHighBits = blasClasAddressBufferAddress >> 32u;

        // Fill out the BUILD_CLUSTERS_BOTTOM_LEVEL_INFO descriptors
        device->BeginEvent(cmd, "Fill Cluster BLAS Info");

        cmd->StartBindingCompute(fillClusterBLASInfoPipeline, &fillClusterBLASInfoLayout)
            .Bind(&blasDataBuffer)
            .Bind(&buildClusterBottomLevelInfoBuffer)
            .Bind(&clasGlobalsBuffer)
            .PushConstants(&fillClusterBottomLevelInfoPush, &pc)
            .End();

        cmd->DispatchIndirect(&clasGlobalsBuffer, sizeof(u32) * GLOBALS_BLAS_INDIRECT_X);

        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
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
        // cmd->ComputeBLASSizes(
        //     &buildClusterBottomLevelInfoBuffer, &clasBlasScratchBuffer, &blasAccelSizes,
        //     &clasGlobalsBuffer, sizeof(u32) * GLOBALS_BLAS_FINAL_COUNT_INDEX,
        //     maxTotalClusterCount, maxClusterCountPerAccelerationStructure, maxInstances);
        cmd->ComputeBLASSizes(&buildClusterBottomLevelInfoBuffer, &clasBlasScratchBuffer,
                              &blasAccelSizes, &clasGlobalsBuffer,
                              sizeof(u32) * GLOBALS_BLAS_FINAL_COUNT_INDEX, maxWriteClusters,
                              maxWriteClusters, 1);
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
        cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                     VK_ACCESS_2_TRANSFER_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }
}

void VirtualGeometryManager::BuildPTLAS(CommandBuffer *cmd, GPUBuffer *gpuInstances)
{
    // Update/write PTLAS instances
    {
        u64 writeAddress  = device->GetDeviceAddress(ptlasWriteInfosBuffer.buffer);
        u64 updateAddress = device->GetDeviceAddress(ptlasUpdateInfosBuffer.buffer);

        PtlasPushConstant pc;
        pc.writeAddress  = writeAddress;
        pc.updateAddress = updateAddress;

        cmd->StartBindingCompute(ptlasWriteInstancesPipeline, &ptlasWriteInstancesLayout)
            .Bind(&blasAccelAddresses)
            .Bind(&clasGlobalsBuffer)
            .Bind(&blasDataBuffer)
            .Bind(gpuInstances)
            .Bind(&ptlasWriteInfosBuffer)
            .Bind(&ptlasUpdateInfosBuffer)
            .Bind(&ptlasIndirectCommandBuffer)
            .Bind(&instanceRefBuffer)
            .Bind(&ptlasInstanceBitVectorBuffer)
            .PushConstants(&ptlasWriteInstancesPush, &pc)
            .End();

        cmd->DispatchIndirect(&clasGlobalsBuffer, sizeof(u32) * GLOBALS_BLAS_INDIRECT_X);

        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        cmd->FlushBarriers();
    }

    cmd->BuildPTLAS(&tlasAccelBuffer, &tlasScratchBuffer, &ptlasIndirectCommandBuffer,
                    &clasGlobalsBuffer, sizeof(u32) * GLOBALS_PTLAS_OP_TYPE_COUNT_INDEX,
                    maxInstances, maxInstances, 1, 0);
}

} // namespace rt
