#include "cull.hlsli"
#include "../dense_geometry.hlsli"
#include "../wave_intrinsics.hlsli"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "lod_error_test.hlsli"

#define MAX_NODES_PER_BATCH 16
#define PACKED_NODE_SIZE 12

typedef uint4 WorkItem;

// 8 bit bounding box planes
struct InstanceHierarchyNode 
{
    float3 min[CHILDREN_PER_HIERARCHY_NODE];
    float3 max[CHILDREN_PER_HIERARCHY_NODE];
    uint childOffset;
    uint numChildren;
};

struct PackedInstanceHierarchyNode 
{
    float3 minBounds;
    uint minX;
    uint minY;
    uint minZ;
    uint maxX;
    uint maxY;
    uint maxZ;

    uint childOffset;
    uint scale_numChildren;
};

#define NUM_CANDIDATE_NODES_INDEX 0
#define NODE_SIZE 3

globallycoherent RWStructuredBuffer<Queue> queue : register(u0);

ConstantBuffer<GPUScene> gpuScene : register(b1);
RWStructuredBuffer<uint> globals : register(u2);

globallycoherent RWStructuredBuffer<WorkItem> nodeQueue : register(u3);
globallycoherent RWStructuredBuffer<WorkItem> leafQueue : register(u4);

StructuredBuffer<GPUInstance> gpuInstances : register(t5);
StructuredBuffer<PackedHierarchyNode> hierarchyNodes : register(t6);
RWStructuredBuffer<VisibleCluster> selectedClusters : register(u7);
RWStructuredBuffer<BLASData> blasDatas : register(u9);

RWStructuredBuffer<StreamingRequest> requests : register(u10);
StructuredBuffer<GPUTransform> instanceTransforms : register(t11);
StructuredBuffer<PartitionInfo> partitionInfos : register(t12);

struct ClusterCull 
{
    void ProcessNode(WorkItem workItem, uint childIndex)
    {
        CandidateNode candidateNode;
        candidateNode.instanceID = workItem.x;
        candidateNode.nodeOffset = workItem.y;
        candidateNode.blasIndex = workItem.z;
        candidateNode.flags = workItem.w;

        GPUInstance instance = gpuInstances[candidateNode.instanceID];
        uint nodeOffset = instance.globalRootNodeOffset + candidateNode.nodeOffset;
        PackedHierarchyNode node = hierarchyNodes[nodeOffset];

        bool isValid = node.childRef[childIndex] != ~0u;
        bool isLeaf = node.leafInfo[childIndex] != ~0u;
        bool isVisible = false;
        bool highestDetail = candidateNode.flags & CANDIDATE_NODE_FLAG_HIGHEST_DETAIL;
        bool streamingOnly = candidateNode.flags & CANDIDATE_NODE_FLAG_STREAMING_ONLY;

        float2 edgeScales = float2(0, 0);
        float minScale = 0.f;
        float test = 0.f;
        float priority = 0.f;

        if (isValid || isLeaf)
        {
            float4 lodBounds = node.lodBounds[childIndex];
            float maxParentError = node.maxParentError[childIndex];
            
            PartitionInfo info = partitionInfos[instance.partitionIndex];

            float3x4 renderFromObject = ConvertGPUMatrix(instanceTransforms[instance.transformIndex], info.base, info.scale);
            Translate(renderFromObject, -gpuScene.cameraP);

            float scaleX = length2(float3(renderFromObject[0].x, renderFromObject[1].x, renderFromObject[2].x));
            float scaleY = length2(float3(renderFromObject[0].y, renderFromObject[1].y, renderFromObject[2].y));
            float scaleZ = length2(float3(renderFromObject[0].z, renderFromObject[1].z, renderFromObject[2].z));

            float3 scale = float3(scaleX, scaleY, scaleZ);
            scale = sqrt(scale);
            minScale = min(scale.x, min(scale.y, scale.z));
            float maxScale = max(scale.x, max(scale.y, scale.z));

            test = minScale;

            bool cull = instance.flags & GPU_INSTANCE_FLAG_CULL;
            edgeScales = TestNode(renderFromObject, gpuScene.cameraFromRender, lodBounds, maxScale, test, cull);

            float threshold = maxParentError * minScale * gpuScene.lodScale;

            isVisible = highestDetail || edgeScales.x <= threshold;
            isValid &= isVisible;

            priority = threshold == 0.f ? 0.f : threshold / edgeScales.x;
        }

#if 0
        if (isVisible)
        {
            float3 minP = node.center[childIndex] - node.extents[childIndex];
            float3 maxP = node.center[childIndex] + node.extents[childIndex];

            bool cull = FrustumCull(gpuScene.clipFromRender, instance.worldFromObject, 
                minP, maxP, gpuScene.p22, gpuScene.p23);
            isVisible &= !cull;
        }
#endif

        bool isNodeValid = isValid && isVisible && !isLeaf;
        uint nodeWriteOffset;
        WaveInterlockedAddScalarTest(queue[0].nodeWriteOffset, isNodeValid, 1, nodeWriteOffset);

        uint nodesToAdd = WaveActiveCountBits(isNodeValid);
        if (WaveIsFirstLane())
        {
            InterlockedAdd(queue[0].numNodes, nodesToAdd);
        }

        if (isNodeValid)
        {
            WorkItem childCandidateNode;
            childCandidateNode.x = candidateNode.instanceID;
            childCandidateNode.y = node.childRef[childIndex];
            childCandidateNode.z = candidateNode.blasIndex;
            childCandidateNode.w = candidateNode.flags;

            nodeQueue[nodeWriteOffset] = childCandidateNode;
        }

        DeviceMemoryBarrier();

        if (isLeaf && (isValid || isVisible))
        {
            uint leafInfo = node.leafInfo[childIndex];

            uint pageClusterIndex = BitFieldExtractU32(leafInfo, MAX_CLUSTERS_PER_PAGE_BITS, 0);
            uint numClusters = BitFieldExtractU32(leafInfo, MAX_CLUSTERS_PER_GROUP_BITS, MAX_CLUSTERS_PER_PAGE_BITS) + 1;
            uint numPages = BitFieldExtractU32(leafInfo, MAX_PARTS_PER_GROUP_BITS, MAX_CLUSTERS_PER_PAGE_BITS + MAX_CLUSTERS_PER_GROUP_BITS);
            uint localPageIndex = leafInfo >> (MAX_CLUSTERS_PER_PAGE_BITS + MAX_CLUSTERS_PER_GROUP_BITS + MAX_PARTS_PER_GROUP_BITS);

            if (isValid && !streamingOnly)
            {
                uint leafWriteOffset;
                InterlockedAdd(queue[0].leafWriteOffset, numClusters, leafWriteOffset);

                uint gpuPageIndex = node.childRef[childIndex];

                const int maxClusters = 1 << 24;
                int clampedNumClusters = min((int)numClusters, maxClusters - (int)leafWriteOffset);

                for (int i = 0; i < clampedNumClusters; i++)
                {
                    WorkItem candidateCluster;
                    candidateCluster.x = (gpuPageIndex << MAX_CLUSTERS_PER_PAGE_BITS) | (pageClusterIndex + i);
                    candidateCluster.y = candidateNode.instanceID;
                    candidateCluster.z = candidateNode.blasIndex;
                    candidateCluster.w = candidateNode.flags;

                    leafQueue[leafWriteOffset + i] = candidateCluster;
                }
            }

            if (isVisible && !highestDetail)
            {
                StreamingRequest request;
                request.priority = priority;
                request.instanceID = instance.resourceID;
                request.pageIndex_numPages = (localPageIndex << MAX_PARTS_PER_GROUP_BITS) | numPages;

                uint requestIndex;
                InterlockedAdd(requests[0].pageIndex_numPages, 1, requestIndex);
                const uint maxNumRequests = 1u << 18u;
                if (requestIndex < maxNumRequests - 1u)
                {
                    requests[requestIndex + 1] = request;
                }
            }
        }

        DeviceMemoryBarrier();
    }

    void ProcessLeaf(WorkItem workItem, uint leafReadOffset)
    {
        // Make sure the candidate cluster fits
        uint pageIndex = workItem.x >> MAX_CLUSTERS_PER_PAGE_BITS;
        uint clusterIndex = workItem.x & (MAX_CLUSTERS_PER_PAGE - 1);
        uint instanceID = workItem.y;
        uint blasIndex = workItem.z;
        uint flags = workItem.w;
        bool highestDetail = flags & CANDIDATE_NODE_FLAG_HIGHEST_DETAIL;

        GPUInstance instance = gpuInstances[instanceID];

        uint baseAddress = GetClusterPageBaseAddress(pageIndex);
        uint numClusters = GetNumClustersInPage(baseAddress);
        DenseGeometry header = GetDenseGeometryHeader(baseAddress, numClusters, clusterIndex);

        bool isVoxel = header.numBricks;
        float4 lodBounds = header.lodBounds;
        float lodError = header.lodError;

        PartitionInfo info = partitionInfos[instance.partitionIndex];
        float3x4 renderFromObject = ConvertGPUMatrix(instanceTransforms[instance.transformIndex], info.base, info.scale);

        Translate(renderFromObject, -gpuScene.cameraP);
        float scaleX = length2(float3(renderFromObject[0].x, renderFromObject[1].x, renderFromObject[2].x));
        float scaleY = length2(float3(renderFromObject[0].y, renderFromObject[1].y, renderFromObject[2].y));
        float scaleZ = length2(float3(renderFromObject[0].z, renderFromObject[1].z, renderFromObject[2].z));

        float3 scale = float3(scaleX, scaleY, scaleZ);
        scale = sqrt(scale);
        float minScale = min(scale.x, min(scale.y, scale.z));
        float maxScale = max(scale.x, max(scale.y, scale.z));

        float test;

        bool cull = instance.flags & GPU_INSTANCE_FLAG_CULL;
        float2 edgeScales = TestNode(renderFromObject, gpuScene.cameraFromRender, lodBounds, maxScale, test, cull);

        uint depth = header.depth;
        bool isValid = (!highestDetail && ((edgeScales.x > gpuScene.lodScale * lodError * minScale) || (header.flags & CLUSTER_STREAMING_LEAF_FLAG)))
                        || (highestDetail && (header.flags & CLUSTER_STREAMING_LEAF_FLAG));

        uint clusterOffset;
        WaveInterlockedAddScalarTest(globals[GLOBALS_VISIBLE_CLUSTER_COUNT_INDEX], isValid && !isVoxel, 1, clusterOffset);

        if (isValid && !isVoxel)
        {
            bool isVoxel = (bool)header.numBricks;
            VisibleCluster cluster;
            cluster.pageIndex_clusterIndex = (pageIndex << MAX_CLUSTERS_PER_PAGE_BITS) | clusterIndex;
            cluster.blasIndex = blasIndex;
            InterlockedAdd(blasDatas[blasIndex].clusterCount, 1);

            selectedClusters[clusterOffset] = cluster;
#if 0
            // Instances
            uint instanceOffset;
            WaveInterlockedAddScalarTest(globals[GLOBALS_VISIBLE_INSTANCE_COUNT_INDEX], isValid, 1, instanceOffset);
            visibleInstances[instanceOffset] = instanceIndex;
#endif
        }
    }
};

template <typename Traversal>
void TraverseHierarchy(uint dtID)
{
    uint waveIndex = WaveGetLaneIndex();
    uint waveLaneCount = WaveGetLaneCount();
    uint nodesPerBatch = waveLaneCount;

    uint nodeReadOffset = 0;
    uint childIndex = 0;
    uint leafReadOffset = ~0u;

    bool processed = true;
    bool noMoreNodes = false;

    WorkItem workItem;
    workItem.x = ~0u;
    workItem.y = ~0u;
    workItem.z = ~0u;
    Traversal traversal;

    uint depth = 0;

    for (;;)
    {
        if (!noMoreNodes)
        {
            uint numNodesToRead = WaveActiveCountBits(processed) >> CHILDREN_PER_HIERARCHY_NODE_BITS;
            uint newNodeReadOffset = 0;

            if (WaveIsFirstLane() && numNodesToRead)
            {
                InterlockedAdd(queue[0].nodeReadOffset, numNodesToRead, newNodeReadOffset);
            }
            newNodeReadOffset = WaveReadLaneFirst(newNodeReadOffset);
            uint sourceIndex = WavePrefixCountBits(processed);
            if (numNodesToRead)
            {
                nodeReadOffset = processed ? newNodeReadOffset + (sourceIndex >> CHILDREN_PER_HIERARCHY_NODE_BITS) : nodeReadOffset;
                childIndex = processed ? sourceIndex & (CHILDREN_PER_HIERARCHY_NODE - 1) : childIndex;
            }

            if ((waveIndex & (CHILDREN_PER_HIERARCHY_NODE - 1)) == 0)
            {
                workItem = nodeQueue[nodeReadOffset];
            }
            workItem = WaveReadLaneAt(workItem, waveIndex & ~(CHILDREN_PER_HIERARCHY_NODE - 1));

            processed = false;
            // TODO: ?
            bool isValid = (workItem.x != ~0u && workItem.y != ~0u && workItem.z != ~0u) && nodeReadOffset < queue[0].nodeWriteOffset;
            
            if (isValid)
            {
                traversal.ProcessNode(workItem, childIndex);
                processed = true;
                //nodeQueue[nodeReadOffset] = ~0u;
            }

            int numNodesCompleted = (int)(WaveActiveCountBits(isValid) >> CHILDREN_PER_HIERARCHY_NODE_BITS);
            if (WaveIsFirstLane())
            {
                InterlockedAdd(queue[0].numNodes, -numNodesCompleted);
            }
        }

        depth++;

        // Process leaves
        if (WaveActiveAllTrue(!processed))
        {
            if (leafReadOffset == ~0u)
            {
                InterlockedAdd(queue[0].leafReadOffset, 1, leafReadOffset);
            }
            uint leafWriteOffset = queue[0].leafWriteOffset;
            
            if (noMoreNodes && WaveActiveAllTrue(leafReadOffset >= leafWriteOffset))
            {
                break;
            }

            uint batchSize = WaveActiveCountBits(leafReadOffset < leafWriteOffset);

            if ((!noMoreNodes && batchSize == WaveGetLaneCount()) || (noMoreNodes && leafReadOffset < leafWriteOffset))
            {
                workItem = leafQueue[leafReadOffset];
                if (workItem.x != ~0u && workItem.y != ~0u && workItem.z != ~0u)
                {
                    traversal.ProcessLeaf(workItem, leafReadOffset);
                    leafReadOffset = ~0u;
                }
            }

            uint numNodes;
            if (WaveIsFirstLane())
            {
                numNodes = queue[0].numNodes;
            }
            numNodes = WaveReadLaneFirst(numNodes);

            if (!noMoreNodes && numNodes == 0)
            {
                noMoreNodes = true;
            }
        }
    }
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    TraverseHierarchy<ClusterCull>(dispatchThreadID.x);
}

