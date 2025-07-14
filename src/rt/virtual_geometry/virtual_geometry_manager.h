#ifndef VIRTUAL_GEOMETRY_MANAGER_H_
#define VIRTUAL_GEOMETRY_MANAGER_H_

#include "../base.h"
#include "../graphics/vulkan.h"

namespace rt
{

struct VirtualGeometryManager
{
    const u32 streamingPoolSize = megabytes(512);
    const u32 maxPages          = 1024; // streamingPoolSize >> CLUSTER_PAGE_SIZE_BITS;
    const u32 maxVirtualPages   = maxPages << 8u;

    const u32 maxNodes                     = 1u << 17u;
    const u32 maxStreamingRequestsPerFrame = (1u << 18u);
    const u32 maxPageInstallsPerFrame      = 128;
    const u32 maxQueueBatches              = 4;

    const u32 hierarchyUploadBufferOffset = maxPageInstallsPerFrame * CLUSTER_PAGE_SIZE;
    const u32 evictedPagesOffset = hierarchyUploadBufferOffset + sizeof(u32) * maxNodes;

    const u32 maxNumTriangles =
        maxPageInstallsPerFrame * MAX_CLUSTERS_PER_PAGE * MAX_CLUSTER_TRIANGLES;
    const u32 maxNumVertices =
        maxPageInstallsPerFrame * MAX_CLUSTERS_PER_PAGE * MAX_CLUSTER_VERTICES;
    const u32 maxNumClusters = maxPageInstallsPerFrame * MAX_CLUSTERS_PER_PAGE;

    struct StreamingRequestBatch
    {
        u32 numRequests;
    };

    struct Page
    {
        u32 numClusters;
        int virtualIndex;
        int nextPage;
        int prevPage;
    };

    struct PageToHierarchyNodeGraph
    {
        u32 *offsets;
        u32 *data;
    };

    struct MeshInfo
    {
        PageToHierarchyNodeGraph graph;

        // TODO: replace with filename
        u8 *pageData;
        u32 hierarchyNodeOffset;
        u32 virtualPageOffset;
    };

    u32 currentClusterTotal;
    u32 totalNumVirtualPages;
    u32 totalNumNodes;
    u32 maxWriteClusters;

    // Render resources
    PushConstant fillClusterTriangleInfoPush;
    DescriptorSetLayout fillClusterTriangleInfoLayout = {};
    VkPipeline fillClusterTriangleInfoPipeline;

    DescriptorSetLayout decodeDgfClustersLayout = {};
    VkPipeline decodeDgfClustersPipeline;

    PushConstant clasDefragPush;
    DescriptorSetLayout clasDefragLayout = {};
    VkPipeline clasDefragPipeline;

    PushConstant computeClasAddressesPush;
    DescriptorSetLayout computeClasAddressesLayout = {};
    VkPipeline computeClasAddressesPipeline;

    DescriptorSetLayout hierarchyTraversalLayout = {};
    VkPipeline hierarchyTraversalPipeline;

    DescriptorSetLayout writeClasDefragLayout = {};
    VkPipeline writeClasDefragPipeline;

    GPUBuffer evictedPagesBuffer;
    GPUBuffer hierarchyNodeBuffer;
    GPUBuffer clusterPageDataBuffer;

    GPUBuffer uploadBuffer;
    GPUBuffer streamingRequestsBuffer;
    GPUBuffer readbackBuffer;

    GPUBuffer clusterAccelAddresses;
    GPUBuffer clusterAccelSizes;

    // For defrag
    GPUBuffer tempClusterAccelAddresses;
    GPUBuffer tempClusterAccelSizes;

    GPUBuffer indexBuffer;
    GPUBuffer vertexBuffer;
    GPUBuffer clasGlobalsBuffer;

    GPUBuffer decodeClusterDataBuffer;
    GPUBuffer buildClusterTriangleInfoBuffer;
    GPUBuffer clasPageInfoBuffer;

    GPUBuffer clasImplicitData;
    GPUBuffer clasScratchBuffer;

    GPUBuffer moveScratchBuffer;

    GPUBuffer moveDescriptors;
    GPUBuffer moveDstAddresses;
    GPUBuffer moveDstSizes;

    // u32 requestBatchWriteIndex;
    // RingBuffer<StreamingRequestBatch> streamingRequestBatches;
    // StaticArray<StreamingRequest, maxQueueBatches * maxStreamingRequestsPerFrame>
    //     streamingRequests;

    int lruHead;
    int lruTail;

    // Range in virtual page space for each instanced geometry
    StaticArray<MeshInfo> meshInfos;

    StaticArray<int> virtualTable;
    StaticArray<Page> physicalPages;

    VirtualGeometryManager(Arena *arena);
    void ProcessRequests(CommandBuffer *cmd);
    u32 AddNewMesh(Arena *arena, CommandBuffer *cmd, u8 *pageData, PackedHierarchyNode *nodes,
                   u32 numNodes, u32 numPages);
    void HierarchyTraversal(CommandBuffer *cmd, GPUBuffer *queueBuffer,
                            GPUBuffer *gpuSceneBuffer, GPUBuffer *workItemQueueBuffer,
                            GPUBuffer *gpuInstancesBuffer, GPUBuffer *visibleClustersBuffer,
                            GPUBuffer *blasDataBuffer);
    void UnlinkLRU(int pageIndex);
    void LinkLRU(int index);
};

} // namespace rt

#endif
