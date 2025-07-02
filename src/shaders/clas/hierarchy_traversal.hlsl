#include "../dense_geometry.hlsli"
#include "../wave_intrinsics.hlsli"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"

// TODO: don't hardcode this 
static const float zNear = 1e-2f;
static const float zFar = 1000.f;

// See if child should be visited
float2 TestNode(float3x4 renderFromObject, float3x4 cameraFromRender, float4 lodBounds, float maxScale, out float test)
{
    // Find length to cluster center
    float3 center = mul(renderFromObject, float4(lodBounds.xyz, 1.f));
    float radius = lodBounds.w * maxScale;
    float distSqr = length2(center);

    // Find angle between vector to cluster center and view vector
    float3 cameraForward = -cameraFromRender[2].xyz;
    float zf = abs(dot(cameraForward, center));
    float zr = abs(dot(cameraFromRender[0].xyz, center));
    float zu = abs(dot(cameraFromRender[1].xyz, center));

    //float z = dot(cameraForward, center);
    float z = max(zf, max(zr, zu));

    float x = distSqr - z * z;
    x = sqrt(max(0.f, x));

    // Find angle between vector to cluster center and vector to tangent point on sphere
    float distTangentSqr = distSqr - radius * radius;

    float distTangent = sqrt(max(0.f, distTangentSqr));

    // Find cosine of the above angles subtracted/added
    float invDistSqr = rcp(distSqr);
    float cosSub = (z * distTangent + x * radius) * invDistSqr;
    float cosAdd = (z * distTangent - x * radius) * invDistSqr;

    test = cosAdd;

    // Clipping
    float depth = z - zNear;
    if (distSqr < 0.f || cosSub * distTangent < zNear)
    {
        float cosSubX = max(0.f, x - sqrt(radius * radius - depth * depth));
        cosSub = zNear * rsqrt(cosSubX * cosSubX + zNear * zNear);
    }
    if (distSqr < 0.f || cosAdd * distTangent < zNear)
    {
        float cosAddX = x + sqrt(radius * radius - depth * depth);
        cosAdd = zNear * rsqrt(cosAddX * cosAddX + zNear * zNear);
    }

    float minZ = max(z - radius, zNear);
    float maxZ = max(z + radius, zNear);

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

globallycoherent RWStructuredBuffer<Queue> queue : register(u0);

ConstantBuffer<GPUScene> gpuScene : register(b1);
RWStructuredBuffer<uint> globals : register(u2);

globallycoherent RWStructuredBuffer<WorkItem> nodeQueue : register(u3);
globallycoherent RWStructuredBuffer<WorkItem> leafQueue : register(u4);

StructuredBuffer<GPUInstance> gpuInstances : register(t5);
StructuredBuffer<PackedHierarchyNode> hierarchyNodes : register(t6);
RWStructuredBuffer<VisibleCluster> selectedClusters : register(u7);
RWStructuredBuffer<BLASData> blasDatas : register(u9);

// Debug 
RWStructuredBuffer<uint4> debugLeaves : register(u10);

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
        bool isLeaf = bool(node.childOffset[childIndex] >> 31u);

        float2 edgeScales = float2(0, 0);
        float minScale = 0.f;
        float test = 0.f;
        if (isValid)
        {
            float4 lodBounds = node.lodBounds[childIndex];
            float maxParentError = node.maxParentError[childIndex];

            float3x4 renderFromObject = instance.renderFromObject;

            float3 scale = float3(length2(renderFromObject[0].xyz), length2(renderFromObject[1].xyz), 
                              length2(renderFromObject[2].xyz)); 
            scale = sqrt(scale);
            minScale = min(scale.x, min(scale.y, scale.z));
            float maxScale = max(scale.x, max(scale.y, scale.z));

            edgeScales = TestNode(renderFromObject, gpuScene.cameraFromRender, lodBounds, maxScale, test);

            isValid &= edgeScales.x <= maxParentError * minScale;
        }

        uint nodeWriteOffset;
        WaveInterlockedAddScalarTest(queue[0].nodeWriteOffset, isValid && !isLeaf, 1, nodeWriteOffset);

        uint nodesToAdd = WaveActiveCountBits(isValid && !isLeaf);
        if (WaveIsFirstLane())
        {
            InterlockedAdd(queue[0].numNodes, nodesToAdd);
        }

        //DeviceMemoryBarrier();

        if (isValid && !isLeaf)
        {
            WorkItem childCandidateNode;
            childCandidateNode.x = candidateNode.instanceID;
            childCandidateNode.y = node.childOffset[childIndex];
            childCandidateNode.z = candidateNode.blasIndex;
            childCandidateNode.w = asuint(minScale);

            nodeQueue[nodeWriteOffset] = childCandidateNode;
        }

        DeviceMemoryBarrier();

        if (isValid && isLeaf)
        {
            uint leafWriteOffset;
            uint leafInfo = node.childOffset[childIndex] & 0x7fffffff;
            uint numClusters = leafInfo & ((1u << 5u) - 1u);
            InterlockedAdd(queue[0].leafWriteOffset, numClusters, leafWriteOffset);
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
                candidateCluster.w = asuint(edgeScales.x);//0x80000000;

                leafQueue[leafWriteOffset + i] = candidateCluster;
            }

        }
#if 0
        else if (!isValid && isLeaf)
        {
            uint leafWriteOffset;
            uint leafInfo = node.childOffset[childIndex] & 0x7fffffff;
            uint numClusters = leafInfo & ((1u << 5u) - 1u);
            InterlockedAdd(queue[0].debugLeafWriteOffset, numClusters, leafWriteOffset);
            // DeviceMemoryBarrier();

            uint pageIndex = leafInfo >> 10u;
            uint pageClusterIndex = (leafInfo >> 5u) & ((1u << 5u) - 1u);

            const int maxClusters = 1 << 24;
            int clampedNumClusters = min((int)numClusters, maxClusters - (int)leafWriteOffset);

            for (int i = 0; i < clampedNumClusters; i++)
            {
                WorkItem candidateCluster;
                candidateCluster.x = ~0u;
                candidateCluster.y = asuint(test);
                candidateCluster.z = node.maxParentError[childIndex];
                candidateCluster.w = asuint(edgeScales.x);//0x80000000;

                debugLeaves[leafWriteOffset + i] = candidateCluster;
            }
        }
#endif

        DeviceMemoryBarrier();
    }

    void ProcessLeaf(WorkItem workItem, uint leafReadOffset)
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

        float3x4 renderFromObject = instance.renderFromObject;
        float3 scale = float3(length2(renderFromObject[0].xyz), length2(renderFromObject[1].xyz), 
                              length2(renderFromObject[2].xyz)); 
        scale = sqrt(scale);
        float minScale = min(scale.x, min(scale.y, scale.z));
        float maxScale = max(scale.x, max(scale.y, scale.z));

        float test;
        float2 edgeScales = TestNode(renderFromObject, gpuScene.cameraFromRender, lodBounds, maxScale, test);

        bool isValid = edgeScales.x > lodError * minScale;

        uint clusterOffset;
        WaveInterlockedAddScalarTest(globals[GLOBALS_CLAS_COUNT_INDEX], isValid, 1, clusterOffset);

        if (isValid)
        {
            VisibleCluster cluster;
#if 0
            cluster.pageIndex = asuint(edgeScales.x);
            cluster.clusterIndex = asuint(lodError);
            cluster.instanceID = asuint(gpuScene.cameraFromRender[0].z);
            cluster.blasIndex = asuint(gpuScene.cameraFromRender[0].w);
#endif
#if 1
            cluster.pageIndex = pageIndex;
            cluster.clusterIndex = clusterIndex;
            cluster.instanceID = instanceID;
            cluster.blasIndex = blasIndex;
#endif

            InterlockedAdd(blasDatas[blasIndex].clusterCount, 1);

            selectedClusters[clusterOffset] = cluster;
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
            bool isValid = (workItem.x != ~0u && workItem.y != ~0u && workItem.z != ~0u);
            
            if (isValid)
            {
                traversal.ProcessNode(workItem, childIndex);
                processed = true;
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

