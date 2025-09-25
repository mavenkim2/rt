#ifndef VIRTUAL_GEOMETRY_MANAGER_H_
#define VIRTUAL_GEOMETRY_MANAGER_H_

#include "../base.h"
#include "../bvh/bvh_types.h"
#include "../graphics/vulkan.h"
#include "../graphics/render_graph.h"

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

struct TruncatedEllipsoid
{
    AffineSpace transform;
    Vec4f sphere;
    Vec3f boundsMin;
    Vec3f boundsMax;
};

struct VoxelClusterGroupFixup
{
    u32 clusterStartIndex;
    u32 clusterEndIndex;
    u32 pageStartIndex;
    u32 numPages;
    u32 clusterOffset;
    u32 depth;
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
    const u32 maxTotalClusterCount = 8 * 1024 * 1024; // MAX_CLUSTERS_PER_BLAS * maxInstances;
    const u32 maxClusterCountPerAccelerationStructure = MAX_CLUSTERS_PER_BLAS;

    const u32 maxClusterFixupsPerFrame = 2 * maxPageInstallsPerFrame * MAX_CLUSTERS_PER_PAGE;

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

        StaticArray<VoxelClusterGroupFixup> voxelClusterGroupFixups;
        StaticArray<ResourceSharingInfo> resourceSharingInfos;

        TruncatedEllipsoid ellipsoid;

        u32 voxelBLASBitmask;
        u32 voxelAddressOffset;
        u32 clusterLookupTableOffset;

        Vec3f boundsMin;
        Vec3f boundsMax;

        // TODO: replace with filename
        u8 *pageData;
        u32 hierarchyNodeOffset;
        u32 virtualPageOffset;

        u32 totalNumVoxelClusters;
        u32 numFinestClusters;

        PackedHierarchyNode *nodes;
        u32 numNodes;

        Vec4f lodBounds;
        u32 numLodLevels;
        u32 resourceSharingInfoOffset;
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

    PushConstant decodeVoxelClustersPush;
    DescriptorSetLayout decodeVoxelClustersLayout = {};
    VkPipeline decodeVoxelClustersPipeline;

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

    DescriptorSetLayout initializeFreeListLayout = {};
    VkPipeline initializeFreeListPipeline;

    DescriptorSetLayout ptlasUpdatePartitionsLayout = {};
    VkPipeline ptlasUpdatePartitionsPipeline;

    PushConstant fillFinestClusterBottomLevelInfoPush;
    DescriptorSetLayout fillFinestClusterBLASInfoLayout = {};
    VkPipeline fillFinestClusterBLASInfoPipeline;

    PushConstant instanceStreamingPush;
    DescriptorSetLayout instanceStreamingLayout = {};
    VkPipeline instanceStreamingPipeline;

    DescriptorSetLayout decodeMergedInstancesLayout = {};
    VkPipeline decodeMergedInstancesPipeline;

    PushConstant mergedInstancesTestPush;
    DescriptorSetLayout mergedInstancesTestLayout = {};
    VkPipeline mergedInstancesTestPipeline;

    DescriptorSetLayout freeInstancesLayout = {};
    VkPipeline freeInstancesPipeline;

    DescriptorSetLayout allocateInstancesLayout = {};
    VkPipeline allocateInstancesPipeline;

    PushConstant instanceCullingPush;
    DescriptorSetLayout instanceCullingLayout = {};
    VkPipeline instanceCullingPipeline;

    DescriptorSetLayout assignInstancesLayout = {};
    VkPipeline assignInstancesPipeline        = {};

    ResourceHandle evictedPagesBuffer;
    GPUBuffer hierarchyNodeBuffer;
    ResourceHandle hierarchyNodeBufferHandle;
    GPUBuffer clusterPageDataBuffer;
    ResourceHandle clusterPageDataBufferHandle;

    ResourceHandle clusterFixupBuffer;
    ResourceHandle voxelPageDecodeBuffer;

    GPUBuffer pageUploadBuffer;
    GPUBuffer fixupBuffer;
    GPUBuffer voxelTransferBuffer;

    // ResourceHandle uploadBuffer;
    ResourceHandle streamingRequestsBuffer;
    GPUBuffer readbackBuffer;
    ResourceHandle readbackBufferHandle;

    ResourceHandle visibleClustersBuffer;
    ResourceHandle queueBuffer;

    ResourceHandle candidateNodeBuffer;
    ResourceHandle candidateClusterBuffer;

    GPUBuffer clusterAccelAddresses;
    ResourceHandle clusterAccelAddressesHandle;
    GPUBuffer clusterAccelSizes;
    ResourceHandle clusterAccelSizesHandle;

    ResourceHandle indexBuffer;
    ResourceHandle vertexBuffer;
    ResourceHandle clasGlobalsBuffer;

    ResourceHandle decodeClusterDataBuffer;
    ResourceHandle buildClusterTriangleInfoBuffer;
    GPUBuffer clasPageInfoBuffer;
    ResourceHandle clasPageInfoBufferHandle;

    ResourceHandle clasScratchBuffer;
    GPUBuffer clasImplicitData;
    ResourceHandle clasImplicitDataHandle;

    ResourceHandle moveScratchBuffer;
    ResourceHandle moveDescriptors;
    ResourceHandle moveDstAddresses;
    ResourceHandle moveDstSizes;

    GPUBuffer clasBlasImplicitBuffer;
    ResourceHandle clasBlasImplicitHandle;

    ResourceHandle blasScratchBuffer;
    ResourceHandle blasDataBuffer;
    ResourceHandle buildClusterBottomLevelInfoBuffer;
    ResourceHandle blasClasAddressBuffer;
    ResourceHandle blasAccelAddresses;
    ResourceHandle blasAccelSizes;

    ResourceHandle ptlasIndirectCommandBuffer;
    ResourceHandle ptlasWriteInfosBuffer;
    ResourceHandle ptlasUpdateInfosBuffer;

    // ResourceHandle virtualInstanceTableBuffer;
    // ResourceHandle instanceIDFreeListBuffer;
    ResourceHandle debugBuffer;

    GPUBuffer resourceBuffer;
    ResourceHandle resourceBufferHandle;
    ResourceHandle resourceBitVector;

    // ResourceHandle ptlasInstanceBitVectorBuffer;
    // ResourceHandle ptlasInstanceFrameBitVectorBuffer0;
    // ResourceHandle ptlasInstanceFrameBitVectorBuffer1;
    GPUBuffer resourceSharingInfosBuffer;
    ResourceHandle resourceSharingInfosBufferHandle;
    ResourceHandle maxMinLodLevelBuffer;

    ResourceHandle voxelAABBBuffer;
    ResourceHandle voxelBlasBuffer;
    ResourceHandle voxelAddressTable;
    ResourceHandle voxelCompactedBlasBuffer;
    // ResourceHandle clusterLookupTableBuffer;

    GPUBuffer tlasAccelBuffer;
    ResourceHandle tlasAccelHandle;
    ResourceHandle tlasScratchHandle;

    GPUBuffer blasProxyScratchBuffer;

    GPUBuffer resourceAABBBuffer;
    ResourceHandle resourceAABBBufferHandle;
    GPUBuffer resourceTruncatedEllipsoidsBuffer;
    ResourceHandle resourceTruncatedEllipsoidsBufferHandle;
    GPUBuffer partitionInfosBuffer;
    ResourceHandle partitionInfosBufferHandle;
    // ResourceHandle partitionReadbackBuffer;
    GPUBuffer instanceTransformsBuffer;
    ResourceHandle instanceTransformsBufferHandle;
    GPUBuffer instanceResourceIDsBuffer;
    ResourceHandle instanceResourceIDsBufferHandle;
    // ResourceHandle partitionCountsBuffer;

    GPUBuffer mergedPartitionDeviceAddresses;
    ResourceHandle mergedPartitionDeviceAddressesHandle;
    GPUBuffer instancesBuffer;
    ResourceHandle instancesBufferHandle;
    BitVector partitionStreamedIn;
    Graph<Instance> partitionInstanceGraph;
    StaticArray<u32> allocatedPartitionIndices;

    GPUBuffer mergedInstancesAABBBuffer;

    u32 numStreamedInstances;
    ResourceHandle instanceUploadBuffer;
    ResourceHandle tempInstanceBuffer;

    GPUBuffer instanceFreeListBuffer;
    ResourceHandle instanceFreeListBufferHandle;
    ResourceHandle visiblePartitionsBuffer;
    ResourceHandle evictedPartitionsBuffer;

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
    u32 virtualInstanceOffset;
    u32 voxelAddressOffset;
    u32 clusterLookupTableOffset;
    u32 resourceSharingInfoOffset;

    u32 maxPartitions;
    u32 numAllocatedPartitions;
    u32 numInstances;

    VirtualGeometryManager(CommandBuffer *cmd, Arena *arena);
    void EditRegistration(u32 instanceID, u32 pageIndex, bool add);
    void RecursePageDependencies(StaticArray<VirtualPageHandle> &pages, u32 instanceID,
                                 u32 pageIndex, u32 priority);
    bool VerifyPageDependencies(u32 virtualOffset, u32 startPage, u32 numPages);
    bool CheckDuplicatedFixup(u32 virtualOffset, u32 pageIndex, u32 startPage, u32 numPages);
    bool ProcessInstanceRequests(CommandBuffer *cmd);
    void ProcessRequests(CommandBuffer *cmd, bool test);
    u32 AddNewMesh(Arena *arena, CommandBuffer *cmd, string filename);
    void FinalizeResources(CommandBuffer *cmd);
    void PrepareInstances(CommandBuffer *cmd, ResourceHandle sceneBuffer, bool ptlas);
    void HierarchyTraversal(CommandBuffer *cmd, ResourceHandle gpuSceneBuffer);
    void BuildClusterBLAS(CommandBuffer *cmd);
    void AllocateInstances(StaticArray<GPUInstance> &gpuInstances);
    void BuildPTLAS(CommandBuffer *cmd);
    void UnlinkLRU(int pageIndex);
    void LinkLRU(int index);

    void Test(Arena *arena, CommandBuffer *cmd, StaticArray<Instance> &inputInstances,
              StaticArray<AffineSpace> &transforms);
    void BuildHierarchy(PrimRef *refs, RecordAOSSplits &record,
                        std::atomic<u32> &numPartitions, StaticArray<u32> &partitionIndices,
                        StaticArray<RecordAOSSplits> &records, bool parallel);
};

} // namespace rt

#endif
