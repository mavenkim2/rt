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

    struct StreamingRequestBatch
    {
        u32 numRequests;
    };

    struct Page
    {
        int nextPage;
        int prevPage;
    };

    u32 requestBatchWriteIndex;
    RingBuffer<StreamingRequestBatch> streamingRequestBatches;
    StaticArray<StreamingRequest, maxQueueBatches * maxStreamingRequestsPerFrame>
        streamingRequests;

    int lruHead;
    int lruTail;

    // Range in virtual page space for each instanced geometry
    StaticArray<Vec2u> instanceRanges;

    StaticArray<int> virtualTable;
    StaticArray<Page> physicalPages;
};

void VirtualGeometryStreamer::ProcessRequests(GPUBuffer *readbackBuffer)
{
    StreamingRequest *requests = (StreamingRequest *)readbackBuffer->mappedPtr;
    u32 numRequests            = requests[0].pageIndex_numClusters_clusterStartIndex;
    requests++;

    // u32 writeIndex
    // Radix sort by priority
    // Divide into pages that need to be loaded and pages that are already resident
    // fix up the hierarchy nodes

    // TODO: actual async copy
    u32 batchIndex  = (requestBatchWriteIndex + (maxQueueBatches - 1)) % (maxQueueBatches);
    u32 numRequests = streamingRequestBatches[batchIndex];

    u32 requestStartIndex = batchIndex * maxStreamingRequestsPerFrame;

    ScratchArena scratch;

    struct PageRequest
    {
        u32 pageIndex;
    };
    StaticArray<PageRequest> unloadedPages;

    for (u32 requestIndex = 0; requestIndex < numRequests; requestIndex++)
    {
        StreamingRequest &request = streamingRequests[requestStartIndex + requestIndex];

        u32 pageStartIndex = request.pageIndex_numClusters_clusterStartIndex >> ? ;
        u32 pageCount      = request.pageCo;

        Vec2u instanceRange = instanceRanges[request.instanceID];
        Assert(pageStartIndex + pageCount < instanceRange.y);

        for (int pageIndex = 0; pageIndex < pageCount; pageIndex++)
        {
            int physicalPageIndex = virtualTable[instanceRange.x + pageStartIndex + pageIndex];

            // Need to load
            if (physicalPageIndex == -1)
            {
                unloadedPages.Push(request);
            }
            else
            {
                UnlinkLRU(physicalPageIndex);
                LinkLRU(pageIndex);
            }
            // Move to back of LRU
            Page &page = physicalPages[physicalPageIndex];
        }
        Vec2u physicalPageInfo = virtualTable[instanceID];
    }

    // Upload pages to GPU
    GPUBuffer uploadBuffer;
    u32 offset = 0;
    for (PageRequest &request : unloadedPages)
    {
        u8 *buffer;
        u8 *src = buffer + sizeof(ClusterFileHeader) + CLUSTER_PAGE_SIZE * request.pageIndex;
        MemoryCopy(uploadBuffer.mappedPtr + offset, src, CLUSTER_PAGE_SIZE);
        offset += CLUSTER_PAGE_SIZE;
    }

    // Fix hierarchy nodes

    // CLAS build requests
} // namespace rt
