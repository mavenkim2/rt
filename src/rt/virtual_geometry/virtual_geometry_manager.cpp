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

VirtualGeometryManager::VirtualGeometryManager(Arena *arena)
    : physicalPages(arena, maxPages), virtualTable(arena, maxVirtualPages),
      meshInfos(arena, 1), currentClusterTotal(0), totalNumVirtualPages(0), totalNumNodes(0)
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
    clasDefragPush.size   = sizeof(NumPushConstant);
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

    // Buffers
    maxWriteClusters = MAX_CLUSTERS_PER_PAGE * maxPages;

    streamingRequestsBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        sizeof(StreamingRequest) * maxStreamingRequestsPerFrame);
    readbackBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                 maxNodes * sizeof(PackedHierarchyNode));

    clusterAccelAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * (maxWriteClusters + maxPageInstallsPerFrame * MAX_CLUSTERS_PER_PAGE));
    clusterAccelSizes = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * (maxWriteClusters + maxPageInstallsPerFrame * MAX_CLUSTERS_PER_PAGE));

    indexBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        megabytes(256));

    vertexBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        megabytes(320));
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
    device->GetMoveBuildSizes(CLASOpMode::ExplicitDestinations, maxNumClusters,
                              clasAccelerationStructureSize, true, moveScratchSize,
                              moveStructureSize);

    moveScratchBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, moveScratchSize);

    decodeClusterDataBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sizeof(DecodeClusterData) * maxWriteClusters);
    buildClusterTriangleInfoBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(BUILD_CLUSTERS_TRIANGLE_INFO) * maxWriteClusters);
    clasPageInfoBuffer = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                              sizeof(CLASPageInfo) * maxPages);

    moveDescriptors = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * maxPages * MAX_CLUSTERS_PER_PAGE);

    moveDstAddresses = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u64) * maxPages * MAX_CLUSTERS_PER_PAGE);

    moveDstSizes = device->CreateBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        sizeof(u32) * maxPages * MAX_CLUSTERS_PER_PAGE);
}

void VirtualGeometryManager::UnlinkLRU(int pageIndex)
{
    Page &page = physicalPages[pageIndex];
    Assert(page.prevPage != -1 && page.nextPage != -1);

    int prevPage                     = page.prevPage;
    int nextPage                     = page.nextPage;
    physicalPages[prevPage].nextPage = nextPage;
    physicalPages[nextPage].prevPage = prevPage;
}

void VirtualGeometryManager::LinkLRU(int index)
{
    int nextPage                  = physicalPages[lruHead].nextPage;
    physicalPages[index].nextPage = nextPage;
    physicalPages[index].prevPage = lruHead;

    physicalPages[nextPage].prevPage = index;
    physicalPages[lruHead].nextPage  = index;
}

u32 VirtualGeometryManager::AddNewMesh(Arena *arena, CommandBuffer *cmd, u8 *pageData,
                                       PackedHierarchyNode *nodes, u32 numNodes, u32 numPages)
{
    u32 *pageOffsets  = PushArray(arena, u32, numPages + 1);
    u32 *pageOffsets1 = &pageOffsets[1];

    const u32 pageShift =
        MAX_CLUSTERS_PER_PAGE_BITS + MAX_CLUSTERS_PER_GROUP_BITS + MAX_PARTS_PER_GROUP_BITS;

    for (u32 nodeIndex = 0; nodeIndex < numNodes; nodeIndex++)
    {
        PackedHierarchyNode &node = nodes[nodeIndex];
        for (u32 childIndex = 0; childIndex < CHILDREN_PER_HIERARCHY_NODE; childIndex++)
        {
            if (node.leafInfo[childIndex] == ~0u) continue;

            // Map page to nodes
            u32 pageIndex = node.leafInfo[childIndex] >> pageShift;
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
            u32 pageIndex = node.leafInfo[childIndex] >> pageShift;
            u32 data      = (nodeIndex << CHILDREN_PER_HIERARCHY_NODE_BITS) | childIndex;

            u32 index             = pageOffsets1[pageIndex]++;
            pageToNodeData[index] = data;
        }
    }

    PageToHierarchyNodeGraph graph;
    graph.offsets = pageOffsets;
    graph.data    = pageToNodeData;

    cmd->SubmitBuffer(&hierarchyNodeBuffer, nodes, sizeof(PackedHierarchyNode) * numNodes,
                      sizeof(PackedHierarchyNode) * totalNumNodes);

    MeshInfo meshInfo;
    meshInfo.graph               = graph;
    meshInfo.hierarchyNodeOffset = totalNumNodes;
    meshInfo.virtualPageOffset   = totalNumVirtualPages;

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
    // u32 batchIndex  = (requestBatchWriteIndex + (maxQueueBatches - 1)) % (maxQueueBatches);
    // u32 numRequests = streamingRequestBatches[batchIndex];

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

    // TODO: sort by instance ID too
    u64 prevKey             = handles[0].sortKey;
    u32 numCompactedHandles = 0;
    f32 maxPriority         = 0.f;

    // Compact handles
    for (u32 handleIndex = 0; handleIndex < numHandles; handleIndex++)
    {
        Handle handle                   = handles[handleIndex];
        const StreamingRequest &request = requests[handle.index];

        if (prevKey != handle.sortKey)
        {
            maxPriority = 0.f;
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
            firstUsedLRUPage =
                requestIndex == 0 && pageIndex == 0 ? physicalPageIndex : firstUsedLRUPage;
            UnlinkLRU(physicalPageIndex);
            LinkLRU(physicalPageIndex);
        }
    }

    // Get pages
    StaticArray<int> pageIndices(scratch.temp.arena, unloadedRequests.Length());
    while (physicalPages.Length() < maxPages && pageIndices.Length() < maxPageInstallsPerFrame)
    {
        Page page         = {};
        page.virtualIndex = -1;
        physicalPages.Push(page);

        LinkLRU(physicalPages.Length() - 1);
        pageIndices.Push(physicalPages.Length() - 1);
    }

    if (pageIndices.Length() < maxPageInstallsPerFrame &&
        pageIndices.Length() != unloadedRequests.Length())
    {
        int lruIndex = physicalPages[lruTail].prevPage;

        while (pageIndices.Length() < maxPageInstallsPerFrame &&
               pageIndices.Length() < unloadedRequests.Length() &&
               lruIndex != firstUsedLRUPage && lruIndex != lruHead)
        {
            pageIndices.Push(lruIndex);
            lruIndex = physicalPages[lruIndex].prevPage;
            currentClusterTotal -= physicalPages[lruIndex].numClusters;
        }
    }

    u32 newClasOffset = currentClusterTotal;

    // Upload pages to GPU and apply hierarchy fixups
    BufferToBufferCopy *copies =
        PushArrayNoZero(scratch.temp.arena, BufferToBufferCopy, 2 * pageIndices.Length());
    u32 hierarchyFixupOffset = pageIndices.Length();
    u32 hierarchyFixups      = 0;
    u32 offset               = 0;

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

        BufferToBufferCopy &copy = copies[requestIndex];
        copy.srcOffset           = offset;
        copy.dstOffset           = CLUSTER_PAGE_SIZE * gpuPageIndex;
        copy.size                = CLUSTER_PAGE_SIZE;

        u8 *buffer = meshInfos[request.instanceID].pageData;
        u8 *src = buffer + sizeof(ClusterFileHeader) + CLUSTER_PAGE_SIZE * request.pageIndex;

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
            u32 hierarchyFixupIndex = hierarchyFixups++;

            u32 hierarchyNodeIndex = nodeData[graphIndex] >> CHILDREN_PER_HIERARCHY_NODE_BITS;
            u32 childIndex = nodeData[graphIndex] & ((1u << CHILDREN_PER_HIERARCHY_NODE) - 1u);

            BufferToBufferCopy &hierarchyCopy =
                copies[hierarchyFixupOffset + hierarchyFixupIndex];
            hierarchyCopy.srcOffset =
                hierarchyUploadBufferOffset + hierarchyFixupIndex * sizeof(u32);
            hierarchyCopy.dstOffset =
                sizeof(PackedHierarchyNode) * (hierarchyNodeOffset + hierarchyNodeIndex) +
                sizeof(PackedHierarchyNode) * hierarchyNodeIndex +
                OffsetOf(PackedHierarchyNode, childRef) + sizeof(u32) * childIndex;
            hierarchyCopy.size = sizeof(u32);

            *(u32 *)((u8 *)uploadBuffer.mappedPtr + hierarchyCopy.srcOffset) = gpuPageIndex;
        }
    }

    cmd->CopyBuffer(&clusterPageDataBuffer, &uploadBuffer, copies, pageIndices.Length());
    cmd->CopyBuffer(&hierarchyNodeBuffer, &uploadBuffer, copies + hierarchyFixupOffset,
                    hierarchyFixups);

    u32 evictedPagesSize = pageIndices.Length() * sizeof(u32);
    BufferToBufferCopy evictedPageCopy;
    evictedPageCopy.srcOffset = evictedPagesOffset;
    evictedPageCopy.dstOffset = 0;
    evictedPageCopy.size      = evictedPagesSize;
    MemoryCopy((u8 *)uploadBuffer.mappedPtr + evictedPagesOffset, pageIndices.data,
               evictedPagesSize);

    cmd->CopyBuffer(&evictedPagesBuffer, &uploadBuffer, &evictedPageCopy, 1);

    // Prepare move descriptors
    {
        // number of resident pages
        cmd->StartBindingCompute(clasDefragPipeline, clasDefragLayout)
            .Bind(&evictedPagesBuffer)
            .Bind(&clasPageInfoBuffer)
            .Bind(&clusterAccelAddresses)
            .Bind(&clusterAccelSizes)
            .Bind(&clasGlobalsBuffer)
            .Bind(&moveDescriptors)
            .Bind(&moveDstAddresses)
            .Bind(&moveDstSizes);

        cmd->Dispatch(maxPages, 1, 1);

        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        cmd->FlushBarriers();
    }

    // Evict old CLAS
    {
        cmd->MoveCLAS(CLASOpMode::ExplicitDestinations, NULL, &moveScratchBuffer,
                      &moveDstAddresses, &moveDstSizes, &moveDescriptors, &clasGlobalsBuffer,
                      GLOBALS_CLAS_COUNT_INDEX, maxWriteClusters, clusterPageDataBuffer.size);

        // TODO split barriers?
        cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                     VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();
    }

    // Write cluster build descriptors
    {
        u64 indexBufferAddress  = device->GetDeviceAddress(indexBuffer.buffer);
        u64 vertexBufferAddress = device->GetDeviceAddress(vertexBuffer.buffer);
        FillClusterTriangleInfoPushConstant fillPc;
        fillPc.indexBufferBaseAddressLowBits   = indexBufferAddress & 0xffffffff;
        fillPc.indexBufferBaseAddressHighBits  = (indexBufferAddress >> 32u) & 0xffffffff;
        fillPc.vertexBufferBaseAddressLowBits  = vertexBufferAddress & 0xffffffff;
        fillPc.vertexBufferBaseAddressHighBits = (vertexBufferAddress >> 32u) & 0xffffffff;

        cmd->StartBindingCompute(fillClusterTriangleInfoPipeline,
                                 fillClusterTriangleInfoLayout)
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
    }

    // Decode the clusters
    {
        cmd->StartBindingCompute(decodeDgfClustersPipeline, decodeDgfClustersLayout)
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
    }

    // Compute the CLAS addresses
    {
        cmd->ComputeCLASSizes(&buildClusterTriangleInfoBuffer, &clasScratchBuffer,
                              &clusterAccelSizes, &clasGlobalsBuffer,
                              sizeof(u32) * GLOBALS_CLAS_COUNT_INDEX, newClasOffset,
                              maxNumTriangles, maxNumVertices, maxNumClusters);

        cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                     VK_ACCESS_2_SHADER_READ_BIT);
        cmd->FlushBarriers();

        cmd->StartBindingCompute(computeClasAddressesPipeline, computeClasAddressesLayout)
            .Bind(&evictedPagesBuffer)
            .Bind(&clusterAccelAddresses)
            .Bind(&clusterAccelSizes)
            .Bind(&clasGlobalsBuffer)
            .Bind(&clasPageInfoBuffer)
            .End();

        cmd->Dispatch(pageIndices.Length(), 1, 1);

        cmd->Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        cmd->FlushBarriers();
    }

    // Build the CLAS
    {
        cmd->BuildCLAS(CLASOpMode::ExplicitDestinations, &clasImplicitData, &clasScratchBuffer,
                       &buildClusterTriangleInfoBuffer, &clusterAccelAddresses,
                       &clusterAccelSizes, &clasGlobalsBuffer,
                       sizeof(u32) * GLOBALS_CLAS_COUNT_INDEX, maxNumClusters, maxNumTriangles,
                       maxNumVertices, newClasOffset);
        cmd->Barrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                     VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        cmd->FlushBarriers();
    }
}

} // namespace rt
