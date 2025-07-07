#include "../base.h"
#include "../debug.h"
#include "../graphics/vulkan.h"
#include "../shader_interop/hierarchy_traversal_shaderinterop.h"

namespace rt
{

struct VirtualGeometryStreamer
{
    const u32 streamingPoolSize            = megabytes(512);
    const u32 maxStreamingRequestsPerFrame = (1u << 18u);
    const u32 maxPageInstallsPerFrame      = 128;
    const u32 maxQueueBatches              = 4;
    const u32 hierarchyUploadBufferOffset  = maxPageInstallsPerFrame * CLUSTER_PAGE_SIZE;

    struct StreamingRequestBatch
    {
        u32 numRequests;
    };

    struct Page
    {
        int nextPage;
        int prevPage;
    };

    struct HierarchyNodeRef
    {
        static const u32 None = (1u << 29) - 1u;
        u32 nextNode : 29;
        u32 childIndex : 3;
    };

    GPUBuffer hierarchyNodeBuffer;
    GPUBuffer clusterPageDataBuffer;

    GPUBuffer uploadBuffer;
    GPUBuffer *readbackBuffer;

    u32 requestBatchWriteIndex;
    RingBuffer<StreamingRequestBatch> streamingRequestBatches;
    StaticArray<StreamingRequest, maxQueueBatches * maxStreamingRequestsPerFrame>
        streamingRequests;

    HashIndex pageToHierarchyHash;

    int lruHead;
    int lruTail;

    // Range in virtual page space for each instanced geometry
    StaticArray<Vec2u> instanceRanges;

    StaticArray<HierarchyNodeRef> pageToHierarchyNode;
    StaticArray<u32> virtualPageToHierarchyNodeRefHead;

    StaticArray<int> virtualTable;
    StaticArray<Page> physicalPages;

    VirtualGeometryStreamer();
    void ProcessRequests(CommandBuffer *cmd, u8 *buffer);
    void UnlinkLRU(int pageIndex);
    void LinkLRU(int index);
};

void VirtualGeometryStreamer::UnlinkLRU(int pageIndex)
{
    Page &page = physicalPages[pageIndex];
    Assert(page.prevPage != -1 && page.nextPage != -1);

    int prevPage                     = page.prevPage;
    int nextPage                     = page.nextPage;
    physicalPages[prevPage].nextPage = nextPage;
    physicalPages[nextPage].prevPage = prevPage;
}

void VirtualGeometryStreamer::LinkLRU(int index)
{
    int nextPage                  = physicalPages[lruHead].nextPage;
    physicalPages[index].nextPage = nextPage;
    physicalPages[index].prevPage = lruHead;

    physicalPages[nextPage].prevPage = index;
    physicalPages[lruHead].nextPage  = index;
}

void VirtualGeometryStreamer::ProcessRequests(CommandBuffer *cmd, u8 *buffer)
{
    TIMED_CPU();

    StreamingRequest *requests = (StreamingRequest *)readbackBuffer->mappedPtr;
    u32 numRequests            = requests[0].pageIndex_numClusters_clusterStartIndex;
    requests++;

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
        int sortKey;
        int index;
    };

    union Float
    {
        f32 f;
        int i;
    };

    Handle *handles = PushArrayNoZero(scratch.temp.arena, Handle, numRequests);
    for (u32 requestIndex = 0; requestIndex < numRequests; requestIndex++)
    {
        Float f;
        f.f = Max(0.f, request[requestIndex].priority);

        handles[requestIndex].sortKey = f.i;
        handles[requestIndex].index   = request[requestIndex]
    }

    SortHandles(handles, numRequests);

    struct PageRequest
    {
        u32 instanceID;
        u32 pageIndex;
    };
    StaticArray<PageRequest> unloadedRequests(scratch.temp.arena, numRequests);
    int firstUsedLRUPage = -1;

    // Sorted by ascending priority
    for (u32 requestIndex = 0; requestIndex < numRequests; requestIndex++)
    {
        Handle &handle            = handles[requestIndex];
        StreamingRequest &request = streamingRequests[handle.index];

        u32 pageStartIndex = request.pageIndex_numPages >> 2;
        u32 pageCount      = request.pageIndex_numPages & 0x3;

        Vec2u instanceRange = instanceRanges[request.instanceID];
        Assert(pageStartIndex + pageCount < instanceRange.y);

        for (int pageIndex = 0; pageIndex < pageCount; pageIndex++)
        {
            int physicalPageIndex = virtualTable[instanceRange.x + pageStartIndex + pageIndex];

            // Need to load
            if (physicalPageIndex == -1)
            {
                PageRequest pageRequest;
                pageRequest.instanceID = request.instanceID;
                pageRequest.pageIndex  = pageStartIndex + pageIndex;
                unloadedRequests.Push(pageRequest);
            }
            else
            {
                firstUsedLRUPage =
                    requestIndex == 0 && pageIndex == 0 ? physicalPageIndex : firstUsedLRUPage;
                UnlinkLRU(physicalPageIndex);
                LinkLRU(physicalPageIndex);
            }
            // Move to back of LRU
            Page &page = physicalPages[physicalPageIndex];
        }
        Vec2u physicalPageInfo = virtualTable[instanceID];
    }

    // Free pages
    u32 numPages = 0;
    int lruIndex = pages[lruTail].prevPage;
    StaticArray<int> pageIndices(scratch.temp.arena, unloadedRequests.Length());

    while (numPages < unloadedRequests.Length() && lruIndex != firstUsedLRUPage &&
           lruIndex != lruHead)
    {
        pageIndices.Push(lruIndex);
        lruIndex = pages[lruIndex].prevPage;
    }

    // Upload pages to GPU and apply hierarchy fixups
    BufferToBufferCopy *copies =
        PushArrayNoZero(scratch.temp.arena, BufferToBufferCopy, 2 * pageIndices.Length());
    u32 hierarchyFixupOffset = pageIndices.Length();
    u32 hierarchyFixups      = 0;
    u32 offset               = 0;

    for (int requestIndex = 0; requestIndex < pageIndices.Length(); requestIndex++)
    {
        PageRequest &request = unloadedRequests[unloadedRequests.Length() - 1 - requestIndex];

        u32 gpuPageIndex = pageIndices[requestIndex];
        Page &page       = physicalPages[gpuPageIndex];

        BufferToBufferCopy &copy = copies[requestIndex];
        copy.srcOffset           = offset;
        copy.dstOffset           = CLUSTER_PAGE_SIZE * gpuPageIndex;
        copy.size                = CLUSTER_PAGE_SIZE;

        u8 *src = buffer + sizeof(ClusterFileHeader) + CLUSTER_PAGE_SIZE * request.pageIndex;
        MemoryCopy(uploadBuffer.mappedPtr + offset, src, CLUSTER_PAGE_SIZE);
        offset += CLUSTER_PAGE_SIZE;

        // Fix up hierarchy nodes
        // TODO: support multiple instance IDs
        u32 hierarchyNodeIndex = virtualPageToHierarchyNodeRefHead[request.pageIndex];
        while (hierarchyNodeIndex != HierarchyNodeRef::None)
        {
            u32 hierarchyFixupIndex = hierarchyFixups++;
            BufferToBufferCopy &hierarchyCopy =
                copies[hierarchyFixupOffset + hierarchyFixupIndex];
            hierarchyCopy.srcOffset =
                hierarchyUploadBufferOffset + hierarchyFixupIndex * sizeof(u32);
            hierarchyCopy.dstOffset = sizeof(PackedHierarchyNode) * hierarchyNodeIndex +
                                      OffsetOf(PackedHierarchyNode, childOffset) +
                                      sizeof(u32) * childIndex;
            hierarchyCopy.size = sizeof(u32);

            *(u32 *)(uploadBuffer.mappedPtr + hierarchyCopy.srcOffset) = gpuPageIndex;
        }
    }

    cmd->CopyBuffer(&clusterPageDataBuffer, &uploadBuffer, copies, pageIndices.Length());
    cmd->CopyBuffer(&hierarchyNodeBuffer, &uploadBuffer, copies + hierarchyFixupOffset,
                    hierarchyFixups);

    // CLAS build requests
    for (PageRequest &request : unloadedPages)
    {
    }

    // Write cluster build descriptors
    {
        cmd->BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, fillClusterTriangleInfoPipeline);
        DescriptorSet ds = fillClusterTriangleInfoLayout.CreateDescriptorSet();
        ds.Bind(&buildClusterTriangleInfoBuffer)
            .Bind(&decodeClusterDataBuffer)
            .Bind(&clasGlobalsBuffer)
            .Bind(&clasPageInfoBuffer)
            .Bind(&clusterPageDataBuffer);

        cmd->BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, &ds,
                                fillClusterTriangleInfoLayout.pipelineLayout);

        cmd->PushConstants(&fillClusterTriangleInfoPush, &fillPc,
                           fillClusterTriangleInfoLayout.pipelineLayout);

        cmd->Dispatch(header.numPages, 1, 1);
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
    }
    // Build the CLAS
    {
    }
}

} // namespace rt
