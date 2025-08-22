#ifndef VIRTUAL_GEOMETRY_MANAGER_H_
#define VIRTUAL_GEOMETRY_MANAGER_H_

#include "../base.h"
#include "../bvh/bvh_types.h"
#include "../graphics/vulkan.h"

namespace rt
{

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

struct RecordAOSSplits;
template <i32 N>
struct BuildRef;

struct alignas(16) TempHierarchyNode
{
    PackedHierarchyNode *node;
    TempHierarchyNode *nodes;

    u32 GetNumChildren()
    {
        u32 count = 0;
        for (u32 i = 0; i < CHILDREN_PER_HIERARCHY_NODE; i++)
        {
            count += node->childRef[i] != ~0u;
        }
        return count;
    };

    void GetBounds(Lane4F32 &outMinX, Lane4F32 &outMinY, Lane4F32 &outMinZ, Lane4F32 &outMaxX,
                   Lane4F32 &outMaxY, Lane4F32 &outMaxZ)
    {
        for (int i = 0; i < 4; i++)
        {
            if (node->childRef[i] != ~0u)
            {
                outMinX[i] = node->center[i].x - node->extents[i].x;
                outMaxX[i] = node->center[i].x + node->extents[i].x;

                outMinY[i] = node->center[i].y - node->extents[i].y;
                outMaxY[i] = node->center[i].y + node->extents[i].y;

                outMinZ[i] = node->center[i].z - node->extents[i].z;
                outMaxZ[i] = node->center[i].z + node->extents[i].z;
            }
        }
    }
    BVHNode4 Child(u32 childIndex)
    {
        BVHNode4 ptr;
        Assert(node->childRef[childIndex] != ~0u);
        TempHierarchyNode *child = &nodes[node->childRef[childIndex]];
        ptr.data                 = (uintptr_t)child;
        BVHNode4::CheckAlignment(child);
        if (child->GetNumChildren() == 0)
        {
            ptr.data |= BVHNode4::tyLeaf;
        }
        return ptr;
    }
};

struct VirtualGeometryManager
{
    const u32 streamingPoolSize = megabytes(512);
    const u32 maxPages          = 1024; // streamingPoolSize >> CLUSTER_PAGE_SIZE_BITS;
    const u32 maxVirtualPages   = maxPages << 8u;

    const u32 instanceIDStart = maxPages * MAX_CLUSTERS_PER_PAGE;

    const u32 maxNodes                     = 1u << 17u;
    const u32 maxStreamingRequestsPerFrame = (1u << 18u);
    const u32 maxPageInstallsPerFrame      = 128;
    const u32 maxQueueBatches              = 4;

    const u32 maxNumTriangles =
        maxPageInstallsPerFrame * MAX_CLUSTERS_PER_PAGE * MAX_CLUSTER_TRIANGLES;
    const u32 maxNumVertices =
        maxPageInstallsPerFrame * MAX_CLUSTERS_PER_PAGE * MAX_CLUSTER_TRIANGLE_VERTICES;
    const u32 maxNumClusters = maxPageInstallsPerFrame * MAX_CLUSTERS_PER_PAGE;

    const u32 maxInstances             = 1u << 21;
    const u32 maxInstancesPerPartition = 128;
    const u32 maxPartitions            = 1u << 14;    // 2048;
    const u32 maxTotalClusterCount = 8 * 1024 * 1024; // MAX_CLUSTERS_PER_BLAS * maxInstances;
    const u32 maxClusterCountPerAccelerationStructure = MAX_CLUSTERS_PER_BLAS;

    const u32 maxClusterFixupsPerFrame = 2 * maxPageInstallsPerFrame * MAX_CLUSTERS_PER_PAGE;

    const u32 hierarchyUploadBufferOffset = maxPageInstallsPerFrame * CLUSTER_PAGE_SIZE;
    const u32 evictedPagesOffset = hierarchyUploadBufferOffset + sizeof(u32) * maxNodes;
    const u32 clusterFixupOffset = evictedPagesOffset + sizeof(u32) * maxPageInstallsPerFrame;
    const u32 voxelBlasOffset =
        clusterFixupOffset + maxClusterFixupsPerFrame * sizeof(GPUClusterFixup);
    const u32 voxelRangeOffset =
        voxelBlasOffset + MAX_CLUSTERS_PER_PAGE * sizeof(u64) * maxPageInstallsPerFrame;

    enum class PageFlag
    {
        NonResident,
        ResidentThisFrame,
        Resident,
    };

    struct StreamingRequestBatch
    {
        u32 numRequests;
    };

    struct VirtualPage
    {
        u32 priority;
        PageFlag pageFlag;
        int pageIndex;
    };

    enum class VoxelStatus
    {
        None,
        Built,
        Compacted,
    };

    struct Range
    {
        u32 begin;
        u32 end;
    };

    struct Page
    {
        u32 numTriangleClusters;
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
        u32 hierarchyNodeOffset;
        u32 virtualPageOffset;

        u32 totalNumVoxelClusters;

        PackedHierarchyNode *nodes;
        u32 numNodes;
    };

    u32 currentClusterTotal;
    u32 currentTriangleClusterTotal;
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

    DescriptorSetLayout ptlasWriteInstancesLayout = {};
    VkPipeline ptlasWriteInstancesPipeline;

    DescriptorSetLayout ptlasUpdateUnusedInstancesLayout = {};
    VkPipeline ptlasUpdateUnusedInstancesPipeline;

    PushConstant ptlasWriteCommandInfosPush;
    DescriptorSetLayout ptlasWriteCommandInfosLayout = {};
    VkPipeline ptlasWriteCommandInfosPipeline;

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
    GPUBuffer ptlasInstanceFrameBitVectorBuffer0;
    GPUBuffer ptlasInstanceFrameBitVectorBuffer1;

    GPUBuffer voxelAABBBuffer;
    GPUBuffer voxelBlasBuffer;
    GPUBuffer voxelAddressTable;
    GPUBuffer voxelBlasInfosBuffer;
    GPUBuffer voxelCompactedBlasBuffer;

    // u32 requestBatchWriteIndex;
    // RingBuffer<StreamingRequestBatch> streamingRequestBatches;
    // StaticArray<StreamingRequest, maxQueueBatches * maxStreamingRequestsPerFrame>
    //     streamingRequests;

    int lruHead;
    int lruTail;

    // Range in virtual page space for each instanced geometry
    StaticArray<MeshInfo> meshInfos;
    // StaticArray<InstanceRef> instanceRefs;

    StaticArray<VirtualPage> virtualTable;
    StaticArray<Page> physicalPages;
    StaticArray<Range> instanceIDFreeRanges;
    StaticArray<Range> pageClusterIDRanges;

    VirtualGeometryManager(CommandBuffer *cmd, Arena *arena);
    void EditRegistration(u32 instanceID, u32 pageIndex, bool add);
    void RecursePageDependencies(StaticArray<VirtualPageHandle> &pages, u32 instanceID,
                                 u32 pageIndex, u32 priority);
    bool VerifyPageDependencies(u32 virtualOffset, u32 startPage, u32 numPages);
    bool CheckDuplicatedFixup(u32 virtualOffset, u32 pageIndex, u32 startPage, u32 numPages);
    void ProcessRequests(CommandBuffer *cmd);
    u32 AddNewMesh(Arena *arena, CommandBuffer *cmd, string filename);
    void HierarchyTraversal(CommandBuffer *cmd, GPUBuffer *queueBuffer,
                            GPUBuffer *gpuSceneBuffer, GPUBuffer *workItemQueueBuffer,
                            GPUBuffer *gpuInstancesBuffer, GPUBuffer *visibleClustersBuffer);
    void BuildClusterBLAS(CommandBuffer *cmd, GPUBuffer *visibleClustersBuffer);
    void AllocateInstances(StaticArray<GPUInstance> &gpuInstances);
    void BuildPTLAS(CommandBuffer *cmd, GPUBuffer *gpuInstances, GPUBuffer *blasSceneBounds);
    void UnlinkLRU(int pageIndex);
    void LinkLRU(int index);

    // void Test(ScenePrimitives *scene);
    // void BuildHierarchy(ScenePrimitives *scene, BuildRef4 *bRefs, RecordAOSSplits &record,
    //                     std::atomic<u32> &numPartitions, std::atomic<u32> &instanceRefCount,
    //                     bool parallel);
};

} // namespace rt

#endif
