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
    const u32 maxInstances             = 1u << 22;
    const u32 maxInstancesPerPartition = 1024u;

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
        u32 numClusters;

        VirtualPageHandle handle;

        u32 numDependents;

        int virtualIndex;
        int nextPage;
        int prevPage;
    };

    struct MeshInfo
    {
        TruncatedEllipsoid ellipsoid;

        Vec3f boundsMin;
        Vec3f boundsMax;

        Vec4f lodBounds;
        u32 clusterOffset;
        u32 dataOffset;
        u32 numClusters;
    };

    u32 currentClusterTotal;
    u32 currentGeoMemoryTotal;
    u32 totalNumVirtualPages;
    u32 totalNumNodes;

    // Render resources
    PushConstant fillClusterTriangleInfoPush;
    DescriptorSetLayout fillClusterTriangleInfoLayout = {};
    VkPipeline fillClusterTriangleInfoPipeline;

    PushConstant decodeDgfPush;
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

    DescriptorSetLayout writeClasDefragLayout = {};
    VkPipeline writeClasDefragPipeline;

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

    DescriptorSetLayout decodeMergedInstancesLayout = {};
    VkPipeline decodeMergedInstancesPipeline;

    PushConstant mergedInstancesTestPush;
    DescriptorSetLayout mergedInstancesTestLayout = {};
    VkPipeline mergedInstancesTestPipeline;

    PushConstant freeInstancesPush;
    DescriptorSetLayout freeInstancesLayout = {};
    VkPipeline freeInstancesPipeline;

    DescriptorSetLayout allocateInstancesLayout = {};
    VkPipeline allocateInstancesPipeline;

    DescriptorSetLayout compactFreeListLayout = {};
    VkPipeline compactFreeListPipeline;

    DescriptorSetLayout generateMipsLayout = {};
    VkPipeline generateMipsPipeline        = {};

    DescriptorSetLayout reprojectDepthLayout = {};
    VkPipeline reprojectDepthPipeline        = {};

    GPUBuffer clusterPageDataBuffer;
    ResourceHandle clusterPageDataBufferHandle;

    GPUBuffer pageUploadBuffer;
    GPUBuffer blasUploadBuffer;

    GPUBuffer clusterAccelAddresses;
    ResourceHandle clusterAccelAddressesHandle;
    GPUBuffer clusterAccelSizes;
    ResourceHandle clusterAccelSizesHandle;

    GPUBuffer indexBuffer;
    GPUBuffer vertexBuffer;
    GPUBuffer clasGlobalsBuffer;

    GPUBuffer totalAccelSizesBuffer;

    ResourceHandle clasGlobalsBufferHandle;

    GPUBuffer decodeClusterDataBuffer;
    GPUBuffer buildClusterTriangleInfoBuffer;

    GPUBuffer clasScratchBuffer;
    GPUBuffer clasImplicitData;
    ResourceHandle clasImplicitDataHandle;

    GPUBuffer clasBlasImplicitBuffer;
    ResourceHandle clasBlasImplicitHandle;

    GPUBuffer blasScratchBuffer;
    ResourceHandle allocatedInstancesBuffer;
    GPUBuffer buildClusterBottomLevelInfoBuffer;

    GPUBuffer blasAccelAddresses;
    ResourceHandle blasAccelAddressesHandle;
    GPUBuffer blasAccelSizes;

    ResourceHandle ptlasIndirectCommandBuffer;
    ResourceHandle ptlasWriteInfosBuffer;
    ResourceHandle ptlasUpdateInfosBuffer;

    // ResourceHandle virtualInstanceTableBuffer;
    // ResourceHandle instanceIDFreeListBuffer;
    ResourceHandle debugBuffer;

    GPUBuffer resourceBuffer;
    ResourceHandle resourceBufferHandle;

    ResourceHandle instanceBitVectorHandle;

    // ResourceHandle ptlasInstanceBitVectorBuffer;
    // ResourceHandle ptlasInstanceFrameBitVectorBuffer0;
    // ResourceHandle ptlasInstanceFrameBitVectorBuffer1;
    // GPUBuffer resourceSharingInfosBuffer;
    // ResourceHandle resourceSharingInfosBufferHandle;
    // ResourceHandle maxMinLodLevelBuffer;

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
    GPUBuffer partitionInfosUploadBuffer;
    ResourceHandle partitionInfosBufferHandle;
    // ResourceHandle partitionReadbackBuffer;
    GPUBuffer instanceTransformsBuffer;
    GPUBuffer instanceTransformsUploadBuffer;
    GPUBuffer resourceIDsUploadBuffer;
    ResourceHandle instanceTransformsBufferHandle;
    GPUBuffer instanceResourceIDsBuffer;
    ResourceHandle instanceResourceIDsBufferHandle;
    // ResourceHandle partitionCountsBuffer;
    GPUBuffer partitionsAndOffset;

    GPUBuffer instancesBuffer;
    ResourceHandle instancesBufferHandle;
    Graph<Instance> partitionInstanceGraph;
    StaticArray<u32> allocatedPartitionIndices;

    GPUBuffer mergedInstancesAABBBuffer;

    u32 numStreamedInstances;

    GPUBuffer instanceFreeListBuffer;
    GPUBuffer instanceFreeListBuffer2;
    ResourceHandle instanceFreeListBufferHandle;
    ResourceHandle instanceFreeListBuffer2Handle;
    ResourceHandle freedInstancesBuffer;
    ResourceHandle visiblePartitionsBuffer;
    ResourceHandle evictedPartitionsBuffer;

    ResourceHandle depthPyramid;
    ImageDesc depthPyramidDesc;

    // u32 requestBatchWriteIndex;
    // RingBuffer<StreamingRequestBatch> streamingRequestBatches;
    // StaticArray<StreamingRequest, maxQueueBatches * maxStreamingRequestsPerFrame>
    //     streamingRequests;

    int lruHead;
    int lruTail;

    // Range in virtual page space for each instanced geometry
    StaticArray<MeshInfo> meshInfos;
    // StaticArray<InstanceRef> instanceRefs;

    u32 virtualInstanceOffset;
    u32 voxelAddressOffset;
    u32 clusterLookupTableOffset;
    u32 resourceSharingInfoOffset;

    u32 maxPartitions;

    VirtualGeometryManager(Arena *arena, u32 targetWidth, u32 targetHeight, u32 numBlas);
    void UpdateHZB();
    void ReprojectDepth(u32 targetWidth, u32 targetHeight, ResourceHandle depth,
                        ResourceHandle scene);
    bool ProcessInstanceRequests(CommandBuffer *cmd);
    void ProcessRequests(CommandBuffer *cmd, bool test);
    u32 AddNewMesh(Arena *arena, CommandBuffer *cmd, string filename, bool debug);
    void FinalizeResources(CommandBuffer *cmd);
    void PrepareInstances(CommandBuffer *cmd, ResourceHandle sceneBuffer, bool ptlas,
                          GPUBuffer *debug);
    void BuildPTLAS(CommandBuffer *cmd, GPUBuffer *debug);
    void UnlinkLRU(int pageIndex);
    void LinkLRU(int index);

    bool AddInstances(Arena *arena, CommandBuffer *cmd, StaticArray<Instance> &inputInstances,
                      StaticArray<AffineSpace> &transforms, string filename);
    void BuildHierarchy(PrimRef *refs, RecordAOSSplits &record,
                        std::atomic<u32> &numPartitions, StaticArray<u32> &partitionIndices,
                        StaticArray<RecordAOSSplits> &records, bool parallel);
};

} // namespace rt

#endif
