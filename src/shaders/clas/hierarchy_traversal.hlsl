#include "../common.hlsli"
#include "../dense_geometry.hlsli"
#include "../wave_intrinsics.hlsli"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"

struct Queue 
{
    uint nodeReadOffset;
    uint nodeWriteOffset;
    uint numNodes;

    uint leafReadOffset;
    uint leafWriteOffset;
};

// TODO: don't hardcode this 
static const float zNear = 1e-2f;
static const float zFar = 1000.f;

// See if child should be visited
float2 TestNode(float3x4 objectToRender, float3x4 cameraFromRender, float4 lodBounds)
{
    // also candidate clusters/candidate nodes? 

    // Find length to cluster center
    float3 scale = float3(length2(objectToRender[0].xyz), length2(objectToRender[1].xyz), 
                          length2(objectToRender[2].xyz)); 
    scale = sqrt(scale);
    float maxScale = max(scale.x, max(scale.y, scale.z));

    float3 center = mul(objectToRender, float4(lodBounds.xyz, 1.f));
    float radius = lodBounds.w * maxScale;
    float distSqr = length2(center);

    // Find angle between vector to cluster center and view vector
    float3 cameraForward = cameraFromRender[2].xyz;
    float z = dot(cameraForward, center);
    float x = distSqr - z * z;

    // Find angle between vector to cluster center and vector to tangent point on sphere
    float distTangentSqr = distSqr - radius * radius;

    float distTangent = sqrt(max(0.f, distTangentSqr));

    // Find cosine of the above angles subtracted/added
    float invDistSqr = rcp(distSqr);
    float cosSub = (z * distTangent + x * radius) * invDistSqr;
    float cosAdd = (z * distTangent - x * radius) * invDistSqr;

    // Clipping
    float depth = z - zNear;
    if (distSqr < 0.f || cosSub * distTangent < zNear)
    {
        float cosSubX = max(0.f, x - sqrt(radius * radius - depth * depth));
        cosSub = zNear / sqrt(cosSubX * cosSubX + zNear * zNear);
    }
    if (distSqr < 0.f || cosAdd * distTangent < zNear)
    {
        float cosAddX = x + sqrt(radius * radius - depth * depth);
        cosAdd = zNear / sqrt(cosAddX * cosAddX + zNear * zNear);
    }

    float minZ = max(z - radius, zNear);
    float maxZ = z + radius; //max(z + radius, zFar);

    return z + radius > zNear ? float2(minZ * cosAdd, maxZ * cosSub) : float2(0, 0);
}

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

groupshared uint groupNodeReadOffset;

globallycoherent RWStructuredBuffer<Queue> queue : register(u0);

ConstantBuffer<GPUScene> gpuScene : register(b1);
RWStructuredBuffer<uint> globals : register(u2);

RWStructuredBuffer<WorkItem> nodeQueue : register(u3);
RWStructuredBuffer<WorkItem> leafQueue : register(u4);

StructuredBuffer<GPUInstance> gpuInstances : register(t5);
StructuredBuffer<PackedHierarchyNode> hierarchyNodes : register(t6);
RWStructuredBuffer<VisibleCluster> selectedClusters : register(u7);
RWStructuredBuffer<BLASData> blasDatas : register(u8);

struct ClusterCull 
{
    void ProcessNode(WorkItem workItem, uint childIndex)
    {
        CandidateNode candidateNode;
        candidateNode.instanceID = workItem.x;
        candidateNode.nodeOffset = workItem.y;
        candidateNode.blasIndex = workItem.z;

        GPUInstance instance = gpuInstances[candidateNode.instanceID];
        uint nodeOffset = instance.globalRootNodeOffset + candidateNode.nodeOffset;
        PackedHierarchyNode node = hierarchyNodes[nodeOffset];

        bool isValid = node.childOffset[childIndex] != ~0u;
        // TODO 
        bool isLeaf = false;

        if (isValid)
        {
            float4 lodBounds = node.lodBounds[childIndex];
            float maxParentError = node.maxParentError[childIndex];
            float2 edgeScales = TestNode(instance.objectToRender, gpuScene.cameraFromRender, lodBounds);

            isValid &= edgeScales.x <= maxParentError;
        }

        uint nodeWriteOffset;
        WaveInterlockedAddScalarTest(queue[0].nodeWriteOffset, isValid && !isLeaf, 1, nodeWriteOffset);

        //DeviceMemoryBarrier();

        if (isValid && !isLeaf)
        {
            WorkItem childCandidateNode;
            childCandidateNode.x = candidateNode.instanceID;
            childCandidateNode.y = node.childOffset[childIndex];
            childCandidateNode.z = candidateNode.blasIndex;
            childCandidateNode.w = 0;

            nodeQueue[nodeWriteOffset] = childCandidateNode;
        }

        DeviceMemoryBarrier();

        if (isValid && isLeaf)
        {
            uint leafWriteOffset;
            uint leafInfo = node.childOffset[childIndex];
            uint numClusters = leafInfo & ((1u << 5u) - 1u);
            WaveInterlockedAdd(queue[0].leafWriteOffset, numClusters, leafWriteOffset);
            // DeviceMemoryBarrier();

            uint pageIndex = leafInfo >> 10u;
            uint pageClusterIndex = (leafInfo >> 5u) & ((1u << 5u) - 1u);

            const int maxClusters = 1 << 24;
            int clampedNumClusters = min((int)numClusters, maxClusters - (int)leafWriteOffset);

            for (int i = 0; i < clampedNumClusters; i++)
            {
                WorkItem candidateCluster;
                candidateCluster.x = (pageIndex << MAX_CLUSTERS_PER_PAGE_BITS) | (pageClusterIndex + i);
                candidateCluster.y = candidateNode.instanceID;
                candidateCluster.z = candidateNode.blasIndex;
                candidateCluster.w = 0x80000000;

                leafQueue[leafWriteOffset + i] = candidateCluster;
            }

            DeviceMemoryBarrier();
        }
    }

    void ProcessLeaf(WorkItem workItem)
    {
        // Make sure the candidate cluster fits
        uint pageIndex = workItem.x >> MAX_CLUSTERS_PER_PAGE_BITS;
        uint clusterIndex = workItem.x & (MAX_CLUSTERS_PER_PAGE - 1);
        uint instanceID = workItem.y;
        uint blasIndex = workItem.z;

        GPUInstance instance = gpuInstances[instanceID];

        uint baseAddress = GetClusterPageBaseAddress(pageIndex);
        uint numClusters = GetNumClustersInPage(baseAddress);
        DenseGeometry header = GetDenseGeometryHeader(baseAddress, numClusters, clusterIndex);

        float4 lodBounds = header.lodBounds;
        float lodError = header.lodError;
        float2 edgeScales = TestNode(instance.objectToRender, gpuScene.cameraFromRender, lodBounds);

        bool isValid = edgeScales.x > lodError;

        uint clusterOffset;
        WaveInterlockedAddScalarTest(globals[GLOBALS_CLAS_COUNT_INDEX], isValid, 1, clusterOffset);

        if (isValid)
        {
            VisibleCluster cluster;
            cluster.pageIndex = pageIndex;
            cluster.clusterIndex = clusterIndex;
            cluster.instanceID = instanceID;
            cluster.blasIndex = blasIndex;

            InterlockedAdd(blasDatas[blasIndex].clusterCount, 1);

            selectedClusters[clusterOffset] = cluster;
        }
    }
};

template <typename Traversal>
void TraverseHierarchy()
{
    uint waveIndex = WaveGetLaneIndex();
    uint waveLaneCount = WaveGetLaneCount();
    uint nodesPerBatch = waveLaneCount;

    uint nodeReadOffset = 0;
    uint childIndex = 0;
    uint leafReadOffset = ~0u;

    bool processed = false;
    bool noMoreNodes = false;

    WorkItem workItem;
    Traversal traversal;

    for (;;)
    {
        if (!noMoreNodes)
        {
            uint numNodesToRead = 0;
            if (WaveIsFirstLane())
            {
                numNodesToRead = WaveActiveCountBits(!processed) >> CHILDREN_PER_HIERARCHY_NODE_BITS;
                if (numNodesToRead > 0)
                {
                    InterlockedAdd(queue[0].nodeReadOffset, numNodesToRead, nodeReadOffset);
                }
            }
            numNodesToRead = WaveReadLaneFirst(numNodesToRead);
            if (numNodesToRead)
            {
                WorkItem newWorkItem;
                if ((numNodesToRead & (CHILDREN_PER_HIERARCHY_NODE - 1)) != 0)
                {
                    printf("bad\n");
                }
                if (waveIndex < numNodesToRead)
                {
                    nodeReadOffset = WaveReadLaneFirst(nodeReadOffset) + waveIndex;
                    newWorkItem = nodeQueue[nodeReadOffset];
                }
                uint sourceIndex = WavePrefixCountBits(!processed);
                if (!processed)
                {
                    workItem = WaveReadLaneAt(newWorkItem, sourceIndex >> CHILDREN_PER_HIERARCHY_NODE_BITS);
                    childIndex = sourceIndex & (CHILDREN_PER_HIERARCHY_NODE - 1);
                }
            }

            processed = false;
            bool isValid = (workItem.x != ~0u && workItem.y != ~0u && workItem.z != ~0u);
            
            if (isValid)
            {
                // if node
                traversal.ProcessNode(workItem, childIndex);
                processed = true;
            }
        }
            
        // Process leaves
        if (WaveActiveAllTrue(!processed))
        {
            if (leafReadOffset == ~0u)
            {
                WaveInterlockedAdd(queue[0].leafReadOffset, 1, leafReadOffset);
            }
            uint leafWriteOffset = queue[0].leafWriteOffset;
            
            if (noMoreNodes && WaveActiveAllTrue(leafReadOffset >= leafWriteOffset))
            {
                break;
            }

            uint batchSize = WaveActiveCountBits(leafReadOffset < leafWriteOffset);

            if ((!noMoreNodes && batchSize == 64) || (noMoreNodes && leafReadOffset < leafWriteOffset))
            {
                workItem = leafQueue[leafReadOffset];
                traversal.ProcessLeaf(workItem);
                leafReadOffset = ~0u;
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
}

