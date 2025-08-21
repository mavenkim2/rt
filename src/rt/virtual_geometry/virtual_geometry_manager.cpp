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
#include "../bvh/bvh_aos.h"
#include "../bvh/partial_rebraiding.h"

namespace rt
{

static_assert(sizeof(PTLAS_WRITE_INSTANCE_INFO) ==
              sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV));
static_assert(sizeof(PTLAS_UPDATE_INSTANCE_INFO) ==
              sizeof(VkPartitionedAccelerationStructureUpdateInstanceDataNV));

VirtualGeometryManager::VirtualGeometryManager(CommandBuffer *cmd, Arena *arena)
    : physicalPages(arena, maxPages + 2), virtualTable(arena, maxVirtualPages),
      meshInfos(arena, maxInstances), currentClusterTotal(0), currentTriangleClusterTotal(0),
      totalNumVirtualPages(0), totalNumNodes(0), lruHead(-1), lruTail(-1)
{
    for (u32 i = 0; i < maxVirtualPages; i++)
    {
        virtualTable.Push(VirtualPage{0, PageFlag::NonResident, -1});
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

    string clusterFixupName = "../src/shaders/cluster_fixup.spv";
    string clusterFixupData = OS_ReadFile(arena, clusterFixupName);
    Shader clusterFixupShader =
        device->CreateShader(ShaderStage::Compute, "cluster fixup", clusterFixupData);

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
    for (int i = 0; i <= 7; i++)
    {
        fillBlasAddressArrayLayout.AddBinding(i, DescriptorType::StorageBuffer,
                                              VK_SHADER_STAGE_COMPUTE_BIT);
    }

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
    readbackBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(StreamingRequest) * maxStreamingRequestsPerFrame, MemoryUsage::GPU_TO_CPU);
    uploadBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        maxPageInstallsPerFrame * (sizeof(u32) + CLUSTER_PAGE_SIZE) +
            2 * maxNodes * sizeof(u32) + maxClusterFixupsPerFrame * sizeof(GPUClusterFixup) +
            MAX_CLUSTERS_PER_PAGE * maxPageInstallsPerFrame * sizeof(u64),
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
    clusterFixupBuffer    = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        maxClusterFixupsPerFrame * sizeof(GPUClusterFixup));

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

    u32 expectedSize = maxPages * MAX_CLUSTERS_PER_PAGE * 2000;

    u32 clasScratchSize, clasAccelerationStructureSize;
    device->GetCLASBuildSizes(CLASOpMode::ExplicitDestinations,
                              maxPages * MAX_CLUSTERS_PER_PAGE,
                              maxPages * MAX_CLUSTERS_PER_PAGE * MAX_CLUSTER_TRIANGLES,
                              maxPages * MAX_CLUSTERS_PER_PAGE * MAX_CLUSTER_TRIANGLE_VERTICES,
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
        sizeof(u64) * maxTotalClusterCount);

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

    voxelAABBBuffer      = {};
    voxelAddressTable    = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                sizeof(u64) * MAX_CLUSTERS_PER_PAGE * maxPages);
    voxelBlasInfosBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                sizeof(BLASVoxelInfo) * maxTotalClusterCount);

    u32 tlasScratchSize, tlasAccelSize;
    device->GetPTLASBuildSizes(maxInstances, maxInstancesPerPartition, maxPartitions, 0,
                               tlasScratchSize, tlasAccelSize);

    tlasScratchBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, tlasScratchSize);
    tlasAccelBuffer =
        device->CreateBuffer(VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             tlasAccelSize);
    ptlasIndirectCommandBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(PTLAS_INDIRECT_COMMAND) * 3);

    ptlasWriteInfosBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(PTLAS_WRITE_INSTANCE_INFO) * maxInstances);
    ptlasUpdateInfosBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(PTLAS_UPDATE_INSTANCE_INFO) * maxInstances);
    ptlasInstanceBitVectorBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                        maxInstances >> 3u);

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
    u32 numPages = clusterFileHeader.numPages;
    u32 numNodes = clusterFileHeader.numNodes;

    u8 *pageData = tokenizer.cursor;

    Advance(&tokenizer, clusterFileHeader.numPages * CLUSTER_PAGE_SIZE);
    PackedHierarchyNode *nodes = (PackedHierarchyNode *)tokenizer.cursor;

    Assert(clusterFileHeader.magic == CLUSTER_FILE_MAGIC);

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

    MeshInfo meshInfo;
    meshInfo.pageToHierarchyNodeGraph = graph;
    meshInfo.hierarchyNodeOffset      = totalNumNodes;
    meshInfo.virtualPageOffset        = totalNumVirtualPages;
    meshInfo.pageData                 = pageData;
    meshInfo.pageToParentClusters     = pageToParentCluster;
    meshInfo.pageToParentPageGraph    = pageToParentPage;

    meshInfo.nodes    = nodes;
    meshInfo.numNodes = numNodes;

    totalNumNodes += numNodes;
    totalNumVirtualPages += numPages;

    meshInfos.Push(meshInfo);

    return meshInfos.Length() - 1;
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
    for (u32 otherPageIndex = pageIndex + 1; otherPageIndex < startPage + numPages;
         otherPageIndex++)
    {
        VirtualPage &virtualPage = virtualTable[virtualOffset + pageIndex];
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

    union Float
    {
        f32 f;
        u32 u;
    };

    StaticArray<VirtualPageHandle> pages(scratch.temp.arena, maxVirtualPages);
    for (u32 requestIndex = 0; requestIndex < numRequests; requestIndex++)
    {
        StreamingRequest &request = requests[requestIndex];
        u32 pageCount =
            BitFieldExtractU32(request.pageIndex_numPages, MAX_PARTS_PER_GROUP_BITS, 0);
        u32 pageStartIndex = request.pageIndex_numPages >> MAX_PARTS_PER_GROUP_BITS;

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
    u32 offset = 0;

    StaticArray<VirtualPageHandle> uninstallRequests(scratch.temp.arena,
                                                     unloadedRequests.Length());
    StaticArray<VirtualPageHandle> installRequests(scratch.temp.arena,
                                                   unloadedRequests.Length());

    u32 clustersToAdd  = 0;
    u32 pagesToInstall = Min(unloadedRequests.Length(), maxPageInstallsPerFrame);
    int lruIndex       = lruTail;
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

        MemoryCopy((u8 *)uploadBuffer.mappedPtr + offset, src, CLUSTER_PAGE_SIZE);
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
            hierarchyCopy.srcOffset =
                hierarchyUploadBufferOffset + nodeInstallCopies.Length() * sizeof(u32);
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

            *(u32 *)((u8 *)uploadBuffer.mappedPtr + hierarchyCopy.srcOffset) = gpuPageIndex;
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

        // TODO: this still triggers
        Assert(gpuClusterFixup.Length() <= maxClusterFixupsPerFrame);

        // Cluster fixups
        if (gpuClusterFixup.Length())
        {
            BufferToBufferCopy clusterFixupsCopy;
            clusterFixupsCopy.srcOffset = clusterFixupOffset;
            clusterFixupsCopy.dstOffset = 0;
            clusterFixupsCopy.size      = sizeof(GPUClusterFixup) * gpuClusterFixup.Length();
            MemoryCopy((u8 *)uploadBuffer.mappedPtr + clusterFixupsCopy.srcOffset,
                       gpuClusterFixup.data, clusterFixupsCopy.size);
            cmd->CopyBuffer(&clusterFixupBuffer, &uploadBuffer, &clusterFixupsCopy, 1);
            cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
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

    // Voxel acceleration structures
    StaticArray<AABB> aabbs(scratch.temp.arena,
                            pagesToInstall * MAX_CLUSTERS_PER_PAGE * MAX_CLUSTER_TRIANGLES);
    StaticArray<BLASBuildInfo> buildInfos(scratch.temp.arena,
                                          pagesToInstall * MAX_CLUSTERS_PER_PAGE);
    struct VoxelBLASInfo
    {
        u32 pageIndex;
        u32 clusterIndex;
    };
    StaticArray<VoxelBLASInfo> voxelBlasInfos(scratch.temp.arena,
                                              pagesToInstall * MAX_CLUSTERS_PER_PAGE);

    u32 totalNumBricks        = 0;
    u32 triangleClustersToAdd = 0;

    for (int requestIndex = 0; requestIndex < pagesToInstall; requestIndex++)
    {
        PageHandle handle         = pageHandles[requestIndex];
        VirtualPageHandle request = unloadedRequests[handle.index];
        u32 page                  = pageIndices[requestIndex];

        MeshInfo &meshInfo = meshInfos[request.instanceID];

        u8 *src                    = meshInfo.pageData + CLUSTER_PAGE_SIZE * request.pageIndex;
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
                continue;
            }

            Vec3i anchor;
            Vec3u posBitWidths;

            u32 geoBaseAddress = packed[1].y;
            u32 numVertices    = BitFieldExtractU32(packed[2].y, 14, 0);
            anchor[0]          = BitFieldExtractI32((int)packed[1].z, ANCHOR_WIDTH, 0);
            anchor[1]          = BitFieldExtractI32((int)packed[1].w, ANCHOR_WIDTH, 0);
            posBitWidths[0]    = BitFieldExtractU32(packed[1].w, 5, ANCHOR_WIDTH);
            anchor[2]          = BitFieldExtractI32((int)packed[2].x, ANCHOR_WIDTH, 0);
            posBitWidths[1]    = BitFieldExtractU32(packed[2].y, 5, 22);
            posBitWidths[2]    = BitFieldExtractU32(packed[2].y, 5, 27);
            u32 numBricks      = packed[2].w & 0x7fffffff;
            f32 lodError       = AsFloat(packed[3].w);
            u32 baseAddress    = request.pageIndex * CLUSTER_PAGE_SIZE;

            if (numBricks)
            {
                u32 vertexBitWidth = posBitWidths.x + posBitWidths.y + posBitWidths.z;
                u32 brickOffset = geoBaseAddress + ((numBricks * vertexBitWidth + 7u) >> 3u);
                BLASBuildInfo buildInfo   = {};
                buildInfo.primitiveOffset = sizeof(AABB) * totalNumBricks;
                buildInfo.primitiveCount  = numBricks;
                totalNumBricks += numBricks;

                buildInfos.Push(buildInfo);

                VoxelBLASInfo voxelBlasInfo;
                voxelBlasInfo.pageIndex    = page;
                voxelBlasInfo.clusterIndex = clusterIndex;
                voxelBlasInfos.Push(voxelBlasInfo);

                for (u32 brickIndex = 0; brickIndex < numBricks; brickIndex++)
                {
                    Brick brick =
                        DecodeBrick(meshInfo.pageData, brickIndex, baseAddress, brickOffset);
                    Assert(brick.vertexOffset < numVertices);
                    Vec3f position =
                        DecodePosition(meshInfo.pageData, brickIndex, posBitWidths, anchor,
                                       baseAddress, geoBaseAddress);

                    Vec3u maxP;
                    GetBrickMax(brick.bitMask, maxP);

                    Vec3f aabbMin = position;
                    Vec3f aabbMax = position + Vec3f(maxP) * lodError;

                    AABB aabb;
                    aabb.minX = aabbMin.x;
                    aabb.minY = aabbMin.y;
                    aabb.minZ = aabbMin.z;
                    aabb.maxX = aabbMax.x;
                    aabb.maxY = aabbMax.y;
                    aabb.maxZ = aabbMax.z;

                    aabbs.Push(aabb);
                }
            }
        }
        physicalPages[page].numTriangleClusters = numTriangleClusters;
        triangleClustersToAdd += numTriangleClusters;
    }

    u32 newClasOffset = currentTriangleClusterTotal;
    currentTriangleClusterTotal += triangleClustersToAdd;

    if (aabbs.Length())
    {
        device->BeginEvent(cmd, "Build Voxel AABBs");
        StaticArray<AccelerationStructureSizes> sizes =
            device->GetBuildSizes(scratch.temp.arena, buildInfos);

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

        if (voxelAABBBuffer.size < sizeof(AABB) * aabbs.Length())
        {
            if (voxelAABBBuffer.size)
            {
                device->DestroyBuffer(&voxelAABBBuffer);
            }

            voxelAABBBuffer = device->CreateBuffer(
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                sizeof(AABB) * aabbs.Length());
        }

        uint64_t scratchDataDeviceAddress = device->GetDeviceAddress(&scratchBuffer);
        uint64_t aabbDeviceAddress        = device->GetDeviceAddress(&voxelAABBBuffer);
        totalScratch                      = 0;
        totalAccel                        = 0;

        device->CreateAccelerationStructures(creates);

        for (u32 sizeIndex = 0; sizeIndex < sizes.Length(); sizeIndex++)
        {
            AccelerationStructureSizes &sizeInfo = sizes[sizeIndex];
            AccelerationStructureCreate &create  = creates[sizeIndex];

            BLASBuildInfo &buildInfo           = buildInfos[sizeIndex];
            buildInfo.scratchDataDeviceAddress = scratchDataDeviceAddress + totalScratch;
            buildInfo.dataDeviceAddress        = aabbDeviceAddress;
            buildInfo.as                       = create.as;

            totalScratch += sizeInfo.scratchSize;
        }

        cmd->SubmitBuffer(&voxelAABBBuffer, aabbs.data, sizeof(AABB) * aabbs.Length());
        cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        cmd->FlushBarriers();

        cmd->BuildCustomBLAS(buildInfos);

        cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                     VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);

        StaticArray<BufferToBufferCopy> copies(scratch.temp.arena, creates.Length());

        for (u32 createIndex = 0; createIndex < creates.Length(); createIndex++)
        {
            AccelerationStructureCreate &create = creates[createIndex];
            Assert(sizeof(create.asDeviceAddress) == sizeof(u64));
            VoxelBLASInfo &info = voxelBlasInfos[createIndex];

            BufferToBufferCopy copy;
            copy.srcOffset = voxelBlasOffset + createIndex * sizeof(u64);
            copy.dstOffset =
                sizeof(u64) * (info.pageIndex * MAX_CLUSTERS_PER_PAGE + info.clusterIndex);
            copy.size = sizeof(u64);

            MemoryCopy((u8 *)uploadBuffer.mappedPtr + voxelBlasOffset +
                           createIndex * sizeof(u64),
                       &create.asDeviceAddress, sizeof(u64));

            copies.Push(copy);
        }
        cmd->CopyBuffer(&voxelAddressTable, &uploadBuffer, copies.data, copies.Length());
        cmd->Barrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
    }

    u32 numEvictedPages = pageIndices.Length() - evictedPageStart;

    u64 address = device->GetDeviceAddress(clasImplicitData.buffer);

    // Prepare move descriptors
    if (pagesToUpdate && pageIndices.Length())
    {
        Print("Prepare defrag\n");
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
        // cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        //              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        //              VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }

    // if (device->frameCount > 10)
    // {
    //     GPUBuffer readback = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    //                                               vertexBuffer.size,
    //                                               MemoryUsage::GPU_TO_CPU);
    //
    //     // cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    //     //              VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    //     //              VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    //     //              VK_ACCESS_2_TRANSFER_READ_BIT);
    //     cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    //     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    //                  VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    //     cmd->FlushBarriers();
    //     cmd->CopyBuffer(&readback, &vertexBuffer);
    //     Semaphore testSemaphore   = device->CreateSemaphore();
    //     testSemaphore.signalValue = 1;
    //     cmd->SignalOutsideFrame(testSemaphore);
    //     device->SubmitCommandBuffer(cmd);
    //     device->Wait(testSemaphore);
    //
    //     Vec3f *data = (Vec3f *)readback.mappedPtr;
    //
    //     int stop = 5;
    // }

    u32 templateOffset = maxPages * MAX_CLUSTERS_PER_PAGE;
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
        .Bind(&blasDataBuffer)
        .Bind(&clusterPageDataBuffer)
        .Bind(&streamingRequestsBuffer)
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
            .Bind(&voxelAddressTable)
            .Bind(&voxelBlasInfosBuffer)
            .End();

        cmd->DispatchIndirect(&clasGlobalsBuffer, sizeof(u32) * GLOBALS_CLAS_INDIRECT_X);
        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }

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

void VirtualGeometryManager::BuildPTLAS(CommandBuffer *cmd, GPUBuffer *gpuInstances)
{
    // Update/write PTLAS instances
    {
        device->BeginEvent(cmd, "Update PTLAS Instances");
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
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                         VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
        cmd->FlushBarriers();
        device->EndEvent(cmd);
    }

    cmd->BuildPTLAS(&tlasAccelBuffer, &tlasScratchBuffer, &ptlasIndirectCommandBuffer,
                    &clasGlobalsBuffer, sizeof(u32) * GLOBALS_PTLAS_OP_TYPE_COUNT_INDEX,
                    maxInstances, maxInstancesPerPartition, maxPartitions, 0);
}

struct GetHierarchyNode
{
    using NodeType = TempHierarchyNode;
    __forceinline TempHierarchyNode *operator()(const BRef &ref)
    {
        return (TempHierarchyNode *)ref.nodePtr.GetPtr();
    }
};

#if 0
void VirtualGeometryManager::BuildHierarchy(ScenePrimitives *scene, BuildRef4 *bRefs,
                                            RecordAOSSplits &record,
                                            std::atomic<u32> &numPartitions,
                                            std::atomic<u32> &instanceRefCount, bool parallel)
{
    typedef HeuristicPartialRebraid<GetHierarchyNode> Heuristic;
    const u32 instancesPerPartition = 128;

    Heuristic heuristic(scene, bRefs, 0);

    Assert(record.start < maxInstances);
    Assert(record.count > 0);

    RecordAOSSplits childRecords[CHILDREN_PER_HIERARCHY_NODE];
    u32 numChildren = 0;

    Split split = heuristic.Bin(record);

    f32 area     = HalfArea(record.geomBounds);
    f32 intCost  = 1.f;
    f32 travCost = 1.f;
    f32 leafSAH  = intCost * area * record.count;
    f32 splitSAH = travCost * area + intCost * split.bestSAH;

    if (record.count <= instancesPerPartition) //&& leafSAH <= splitSAH)
    {
        u32 threadIndex = GetThreadIndex();
        heuristic.FlushState(split);

        u32 partitionIndex = numPartitions.fetch_add(1, std::memory_order_relaxed);
        u32 instanceRefStartIndex =
            instanceRefCount.fetch_add(record.Count(), std::memory_order_relaxed);
        for (int i = record.start; i < record.End(); i++)
        {
            InstanceRef ref;
            BRef &bRef     = bRefs[i];
            ref.instanceID = bRef.instanceID;
            for (int j = 0; j < 3; j++)
            {
                ref.bounds[j]     = -bRef.min[j];
                ref.bounds[3 + j] = bRef.max[j];
            }
            ref.nodeOffset = ((TempHierarchyNode *)bRef.nodePtr.GetPtr())->node -
                             meshInfos[bRef.instanceID].nodes;
            ref.partitionIndex                                        = partitionIndex;
            newInstanceRefs[instanceRefStartIndex + i - record.start] = ref;
        }

        return;
    }
    heuristic.Split(split, record, childRecords[0], childRecords[1]);

    // N - 1 splits produces N children
    for (numChildren = 2; numChildren < CHILDREN_PER_HIERARCHY_NODE; numChildren++)
    {
        i32 bestChild = -1;
        f32 maxArea   = neg_inf;
        for (u32 recordIndex = 0; recordIndex < numChildren; recordIndex++)
        {
            RecordAOSSplits &childRecord = childRecords[recordIndex];
            if (childRecord.count <= instancesPerPartition) continue;

            f32 childArea = HalfArea(childRecord.geomBounds);
            if (childArea > maxArea)
            {
                bestChild = recordIndex;
                maxArea   = childArea;
            }
        }
        if (bestChild == -1) break;

        split = heuristic.Bin(childRecords[bestChild]);

        RecordAOSSplits out;
        heuristic.Split(split, childRecords[bestChild], out, childRecords[numChildren]);

        childRecords[bestChild] = out;
    }

    // InstanceNode *nodes = PushArrayNoZero(arena, InstanceNode, numChildren);
    if (parallel)
    {
        scheduler.ScheduleAndWait(numChildren, 1, [&](u32 jobID) {
            bool childParallel = childRecords[jobID].count >= BUILD_PARALLEL_THRESHOLD;
            BuildHierarchy(scene, bRefs, childRecords[jobID], numPartitions, instanceRefCount,
                           childParallel);
        });
    }
    else
    {
        for (int i = 0; i < numChildren; i++)
        {
            BuildHierarchy(scene, bRefs, childRecords[i], numPartitions, instanceRefCount,
                           false);
        }
    }
}

void VirtualGeometryManager::Test(ScenePrimitives *scene)
{
    ScratchArena scratch;
    scratch.temp.arena->align = 16;

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
    BRef *bRefs = PushArrayNoZero(scratch.temp.arena, BRef, maxInstances);

    newInstanceRefs = StaticArray<InstanceRef>(scratch.temp.arena, maxInstances, maxInstances);

    Bounds geom;
    Bounds cent;
    RecordAOSSplits record;
    Instance *instances = (Instance *)scene->primitives;
    for (int i = 0; i < scene->numPrimitives; i++)
    {
        BRef &bRef         = bRefs[i];
        Instance &instance = instances[i];

        u32 sceneIndex = scene->childScenes[instance.id]->sceneIndex;

        Bounds bounds;
        TempHierarchyNode *child = &nodes[meshInfos[sceneIndex].hierarchyNodeOffset];

        for (int childIndex = 0; childIndex < CHILDREN_PER_HIERARCHY_NODE; childIndex++)
        {
            if (child->node->childRef[childIndex] == ~0u) continue;
            for (int axis = 0; axis < 3; axis++)
            {
                bounds.minP[axis] = Min(child->node->center[childIndex][axis] -
                                            child->node->extents[childIndex][axis],
                                        bounds.minP[axis]);
                bounds.maxP[axis] = Max(child->node->center[childIndex][axis] +
                                            child->node->extents[childIndex][axis],
                                        bounds.maxP[axis]);
            }
        }

        AffineSpace &transform = scene->affineTransforms[instance.transformIndex];
        bounds                 = Transform(transform, bounds);

        bRef.StoreBounds(bounds);
        geom.Extend(bounds);
        cent.Extend(bounds.minP + bounds.maxP);
        bRef.instanceID = sceneIndex;

        bRef.nodePtr = uintptr_t(child);
        if (child->GetNumChildren() == 0)
        {
            bRef.nodePtr.data |= BVHNode4::tyLeaf;
        }
        bRef.numPrims = 0;
    }
    record.geomBounds = Lane8F32(-geom.minP, geom.maxP);
    record.centBounds = Lane8F32(-cent.minP, cent.maxP);
    record.start      = 0;
    record.count      = scene->numPrimitives;
    record.extEnd     = maxInstances;

    u32 numNodes                     = 0;
    std::atomic<u32> numPartitions   = 0;
    std::atomic<u32> numInstanceRefs = 0;
    bool parallel                    = scene->numPrimitives >= BUILD_PARALLEL_THRESHOLD;

    BuildHierarchy(scene, bRefs, record, numPartitions, numInstanceRefs, parallel);
    Assert(numPartitions <= maxPartitions);

    newInstanceRefs.size() = numInstanceRefs;

    int stop = 5;
    // Bounds bounds;
    // basically:
    // uniform voxel grid over the scene
    // store what istances are inside each voxel
    // secondary rays dda through the voxel grid
    const u32 voxelX = 4096;
    const u32 voxelY = 4096;
    const u32 voxelZ = 8;
}
#endif

} // namespace rt
