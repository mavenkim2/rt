#ifndef VIRTUAL_GEOMETRY_MANAGER_H_
#define VIRTUAL_GEOMETRY_MANAGER_H_

#include "../base.h"
#include "../graphics/vulkan.h"

namespace rt
{

template <typename T>
struct Graph
{
    u32 *offsets;
    T *data;
};

struct VirtualPageHandle
{
    u32 instanceID;
    u32 pageIndex;
};

struct ClusterFixup
{
    u32 pageIndex_clusterIndex;
    u32 pageStartIndex_numPages;

    ClusterFixup(u32 pageIndex, u32 clusterIndex, u32 pageStartIndex, u32 numPages)
    {
        pageIndex_clusterIndex  = (pageIndex << MAX_CLUSTERS_PER_PAGE_BITS) | clusterIndex;
        pageStartIndex_numPages = (pageStartIndex << MAX_PARTS_PER_GROUP_BITS) | numPages;
    }
    u32 GetPageIndex() { return pageIndex_clusterIndex >> MAX_CLUSTERS_PER_PAGE_BITS; }
    u32 GetClusterIndex() { return pageIndex_clusterIndex & (MAX_CLUSTERS_PER_PAGE - 1); }
    u32 GetPageStartIndex() { return pageStartIndex_numPages >> MAX_PARTS_PER_GROUP_BITS; }
    u32 GetNumPages() { return pageStartIndex_numPages & (MAX_PARTS_PER_GROUP - 1); }
};

struct HierarchyFixup
{
    u32 nodeIndex_childIndex;
    u32 pageStartIndex_pageIndex_numPages;
    HierarchyFixup(u32 nodeIndex, u32 childIndex, u32 pageStartIndex, u32 pageDelta,
                   u32 numPages)
    {
        nodeIndex_childIndex = (nodeIndex << CHILDREN_PER_HIERARCHY_NODE_BITS) | (childIndex);
        pageStartIndex_pageIndex_numPages =
            (pageStartIndex << (2 * MAX_PARTS_PER_GROUP_BITS)) |
            (pageDelta << MAX_PARTS_PER_GROUP_BITS) | numPages;
    }
    u32 GetNodeIndex() { return nodeIndex_childIndex >> CHILDREN_PER_HIERARCHY_NODE_BITS; }
    u32 GetChildIndex() { return nodeIndex_childIndex & (CHILDREN_PER_HIERARCHY_NODE - 1); }
    u32 GetPageStartIndex()
    {
        return pageStartIndex_pageIndex_numPages >> (2 * MAX_PARTS_PER_GROUP_BITS);
    }
    u32 GetNumPages() { return pageStartIndex_pageIndex_numPages & (MAX_PARTS_PER_GROUP - 1); }
    u32 GetPageIndex()
    {
        return GetPageStartIndex() +
               ((pageStartIndex_pageIndex_numPages >> MAX_PARTS_PER_GROUP_BITS) &
                (MAX_PARTS_PER_GROUP - 1));
    }
};

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
    const u32 clusterFixupOffset = evictedPagesOffset + sizeof(u32) * maxPageInstallsPerFrame;

    const u32 maxNumTriangles =
        maxPageInstallsPerFrame * MAX_CLUSTERS_PER_PAGE * MAX_CLUSTER_TRIANGLES;
    const u32 maxNumVertices =
        maxPageInstallsPerFrame * MAX_CLUSTERS_PER_PAGE * MAX_CLUSTER_VERTICES;
    const u32 maxNumClusters = maxPageInstallsPerFrame * MAX_CLUSTERS_PER_PAGE;

    const u32 maxInstances                            = 1024;
    const u32 maxTotalClusterCount                    = MAX_CLUSTERS_PER_BLAS * maxInstances;
    const u32 maxClusterCountPerAccelerationStructure = MAX_CLUSTERS_PER_BLAS;

    struct StreamingRequestBatch
    {
        u32 numRequests;
    };

    struct VirtualPage
    {
        u32 priority;
        int pageIndex;
    };

    struct Page
    {
        u32 numClusters;

        VirtualPageHandle handle;
        u32 numDependents;

        int virtualIndex;
        int nextPage;
        int prevPage;
    };

    struct MeshInfo
    {
        Graph<HierarchyFixup> pageToHierarchyNodeGraph;

        Graph<u32> pageToParentPageGraph;
        Graph<ClusterFixup> pageToParentClusters;

        // TODO: replace with filename
        u8 *pageData;
        u32 *rebraid;
        u32 hierarchyNodeOffset;
        u32 virtualPageOffset;
        u32 numRebraid;
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

    PushConstant fillClusterBottomLevelInfoPush;
    DescriptorSetLayout fillClusterBLASInfoLayout = {};
    VkPipeline fillClusterBLASInfoPipeline;

    DescriptorSetLayout fillBlasAddressArrayLayout = {};
    VkPipeline fillBlasAddressArrayPipeline;

    DescriptorSetLayout getBlasAddressOffsetLayout = {};
    VkPipeline getBlasAddressOffsetPipeline;

    PushConstant computeBlasAddressesPush;
    DescriptorSetLayout computeBlasAddressesLayout = {};
    VkPipeline computeBlasAddressesPipeline;

    PushConstant ptlasWriteInstancesPush;
    DescriptorSetLayout ptlasWriteInstancesLayout = {};
    VkPipeline ptlasWriteInstancesPipeline;

    PushConstant clusterFixupPush;
    DescriptorSetLayout clusterFixupLayout = {};
    VkPipeline clusterFixupPipeline;

    GPUBuffer evictedPagesBuffer;
    GPUBuffer hierarchyNodeBuffer;
    GPUBuffer clusterPageDataBuffer;

    GPUBuffer clusterFixupBuffer;

    GPUBuffer instanceRefBuffer;

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

    GPUBuffer clasBlasScratchBuffer;

    GPUBuffer clasBlasImplicitBuffer;

    GPUBuffer blasDataBuffer;
    GPUBuffer buildClusterBottomLevelInfoBuffer;
    GPUBuffer blasClasAddressBuffer;
    GPUBuffer blasAccelAddresses;
    GPUBuffer blasAccelSizes;

    GPUBuffer tlasScratchBuffer;
    GPUBuffer tlasAccelBuffer;

    GPUBuffer ptlasIndirectCommandBuffer;
    GPUBuffer ptlasWriteInfosBuffer;
    GPUBuffer ptlasUpdateInfosBuffer;
    GPUBuffer ptlasInstanceBitVectorBuffer;
    VkAccelerationStructureKHR ptlas;

    // u32 requestBatchWriteIndex;
    // RingBuffer<StreamingRequestBatch> streamingRequestBatches;
    // StaticArray<StreamingRequest, maxQueueBatches * maxStreamingRequestsPerFrame>
    //     streamingRequests;

    int lruHead;
    int lruTail;

    // Range in virtual page space for each instanced geometry
    StaticArray<MeshInfo> meshInfos;
    StaticArray<InstanceRef> instanceRefs;

    StaticArray<VirtualPage> virtualTable;
    StaticArray<Page> physicalPages;

    VirtualGeometryManager(Arena *arena);
    void EditRegistration(u32 instanceID, u32 pageIndex, bool add);
    void RecursePageDependencies(StaticArray<VirtualPageHandle> &pages, u32 instanceID,
                                 u32 pageIndex, u32 priority);
    bool VerifyPageDependencies(u32 virtualOffset, u32 startPage, u32 numPages);
    void ProcessRequests(CommandBuffer *cmd);
    u32 AddNewMesh(Arena *arena, CommandBuffer *cmd, string filename);
    void HierarchyTraversal(CommandBuffer *cmd, GPUBuffer *queueBuffer,
                            GPUBuffer *gpuSceneBuffer, GPUBuffer *workItemQueueBuffer,
                            GPUBuffer *gpuInstancesBuffer, GPUBuffer *visibleClustersBuffer);
    void BuildClusterBLAS(CommandBuffer *cmd, GPUBuffer *visibleClustersBuffer);
    void BuildPTLAS(CommandBuffer *cmd, GPUBuffer *gpuInstances);
    void UnlinkLRU(int pageIndex);
    void LinkLRU(int index);
};

} // namespace rt

#endif
