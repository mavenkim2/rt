struct Queue 
{
    uint nodeReadOffset;
    uint nodeWriteOffset;

    uint leafReadOffset;
    uint leafWriteOffset;
};

// See if child should be visited
float2 TestNode(float3x4 objectToRender, float3x3 renderToCamera, float4 lodBounds)
{
    // also candidate clusters/candidate nodes? 

    // Find length to cluster center
    float4 lodBounds;
    float3x4 objectToRender;

    float3 center = mul(objectToRender, float4(lodBounds.xyz, 1.f));
    float radius = ?;
    float distSqr = length2(center);

    // Find angle between vector to cluster center and view vector
    float z = dot(camera.forward, center);
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
        float cosAddX = x + sqrt(radius * radius - depth * depth));
        cosAdd = zNear / sqrt(cosAddX * cosAddX + zNear * zNear);
    }

    float minZ = max(z - radius, zNear);
    float maxZ = max(z + radius, zFar);

    return z + radius > zNear ? float2(minZ * cosAdd, maxZ * cosSub) : float2(0, 0);
}

#define MAX_NODES_PER_BATCH 16
#define PACKED_NODE_SIZE 12

groupshared uint groupNodeReadOffset;

globallycoherent RWStructuredBuffer<Queue> queue : register(u0);
globallycoherent RWByteAddressBuffer nodesBuffer : register(u1);

void WriteNode()
{
    nodesBuffer.Store(PACKED_NODE_SIZE
}

struct Instance 
{
    float3x4 objectToRender;
    uint globalRootNodeOffset;
};

struct InstanceData 
{
    uint instanceID;
    uint nodeOffset;
};

struct CandidateNode 
{
    uint instanceID;
    uint nodeOffset;
    uint blasIndex;
};

enum WorkItemType 
{
    Node = 0,
    Leaf = 1,
};

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

InstanceHierarchyNode UnpackInstanceHierarchyNode(PackedInstanceHierarchyNode packed)
{
    InstanceHierarchyNode node;

    return node;
}

#define NUM_CANDIDATE_NODES_INDEX 0
#define NODE_SIZE 3

RWStructuredBuffer<uint> globals : register(u0);
StructuredBuffer<WorkItem> workItemQueue : register(t1);
StructuredBuffer<InstanceData> instanceDatas : register(t2);
StructuredBuffer<HierarchyNode> hierarchyNodes : register(t3);

RWStructuredBuffer<uint> nodes : register(u4);

#if 0
struct InstanceHierarchyCull
{
    void ProcessNode() 
    {
        // intersect AABB against node
        uint childIndex = ? & (CHILDREN_PER_HIERARCHY_NODE - 1);
        
        // Write node batches
        InstanceHierarchyNode node;

        // Child is leaf
        if (node.childOffset == 0)
        {
        }

        InterlockedAdd(queue.leafWriteOffset, );
    }

    void ProcessLeaf(uint4 data) 
    {
        InstanceHierarchyNode node;
        // intersect the node
        
        bool bIntersect = false;
        if (childIndex < node.numChildren)
        {
            bIntersect = ?;
        }

        uint candidateNodeIndex;
        WaveInterlockedAddScalarTest(globals[NUM_CANDIDATE_NODES_INDEX], bIntersect, 1, candidateNodeIndex);
        if (bIntersect)
        {
            uint instanceDataIndex = node.childOffset + childIndex;
            InstanceData instanceData = instanceDatas[instanceDataIndex];

            uint3 candidateNode;
            nodes[NODE_SIZE * candidateNodeIndex + 0] = instanceData.instanceID;
            nodes[NODE_SiZE * candidateNodeIndex + 1] = instanceData.nodeOffset;
            nodes[NODE_SiZE * candidateNodeIndex + 2] = instanceData.blasIndex;

            //CandidateNode candidateNode;
            //candidateNode.instanceID = instanceData.instanceID;
            //candidateNode.nodeOffset = instanceData.nodeOffset;
            //candidateNode.blasIndex  = blasIndex;
        }
    }
};
#endif

struct ClusterCull 
{
    void ProcessNode(WorkItem workItem, uint nodeReadOffset)
    {
        //uint nodeIndex = ? >> 2;
        uint childIndex = nodeReadOffset & (CHILDREN_PER_HIERARCHY_NODE - 1);

        CandidateNode candidateNode;
        candidateNode.instanceID = workItem.x;
        candidateNode.nodeOffset = workItem.y;
        candidateNode.blasIndex = workItem.z;

        Instance instance = instances[candidateNode.instanceID];
        uint nodeOffset = instance.globalRootNodeOffset + candidateNode.nodeOffset;
        HierarchyNode node = hierarchyNodes[nodeOffset];

        bool isValid = childIndex < numChildren;
        bool isLeaf  false;

        if (isValid)
        {
            float4 lodBounds = node.lodBounds[childIndex];
            float maxParentError = node.maxParentError[childIndex];
            float2 edgeScales = TestNode(instance.objectToRender, , lodBounds);

            isValid &= edgeScales <= maxParentError;
        }

        uint nodeWriteOffset;
        WaveInterlockedAddScalarTest(queue.nodeWriteOffset, isValid && !isLeaf, 1, nodeWriteOffset);

        //DeviceMemoryBarrier();

        if (isValid && !isLeaf)
        {
            WorkItem childCandidateNode;
            childCandidateNode.x = candidateNode.instanceID;
            childCandidateNode.y = node.childOffset + childIndex;
            childCandidateNode.z = candidateNode.blasIndex;
            childCandidateNode.w = 0;

            workItemQueue[nodeWriteOffset] = childCandidateNode;
        }

        DeviceMemoryBarrier();

        if (isValid && isLeaf)
        {
            uint leafWriteOffset;
            uint numClusters = node.numChildren;
            WaveInterlockedAdd(queue.leafWriteOffset, numClusters, leafWriteOffset);
            // DeviceMemoryBarrier();

            uint pageIndex = node.childOffset >> MAX_CLUSTERS_PER_PAGE_BITS;
            uint pageClusterIndex = node.childOffset & (MAX_CLUSTERS_PER_PAGE - 1);

            const int maxClusters = 1 << 24;
            int clampedNumClusters = min((int)numClusters, maxClusters - (int)leafWriteOffset);

            for (int i = 0; i < clampedNumClusters; i++)
            {
                WorkItem candidateCluster;
                candidateCluster.x = (pageIndex << MAX_CLUSTERS_PER_PAGE_BITS) | (pageClusterIndex + i);
                candidateCluster.y = candidateNode.instanceID;
                candidateCluster.z = candidateNode.blasIndex;
                candidateCluster.w = 0x80000000;
                workItemQueue[leafWriteOffset + i] = candidateCluster;
            }

            DeviceMemoryBarrier();
        }

        //DeviceMemoryBarrier();
    }

    void ProcessLeaf(WorkItem workItem)
    {
        // Make sure the candidate cluster fits
        uint clusterOffset;
        WaveInterlockedAdd(globals[GLOBALS_CLAS_COUNT_INDEX]
        uint pageIndex = workItem.x >> MAX_CLUSTERS_PER_PAGE_BITS;
        uint clusterIndex = workItem.x & (MAX_CLUSTERS_PER_PAGE - 1);
        uint instanceID = workItem.y;
        uint blasIndex = workItem.z;
    }
};

template <typename Traversal>
void TraverseHierarchy()
{
    uint waveIndex = WaveGetLaneIndex();
    uint waveLaneCount = WaveGetLaneCount();
    uint nodesPerBatch = waveLaneCount;

    uint nodeReadOffset = 0;

    bool processed = false;
    bool noMoreNodes = false;

    WorkItem workItem;
    Traversal traversal;

    for (;;)
    {
        if (!noMoreNodes)
        {
            if (WaveIsFirstLane())
            {
                uint numNodesToRead = WaveActiveCountBits(!processed);
                if (numNodesToRead > 0)
                {
                    InterlockedAdd(queue.nodeReadOffset, numNodesToRead, nodeReadOffset);
                }
            }
            if (!processed)
            {
                nodeReadOffset = WaveReadLaneFirst(nodeReadOffset) + WavePrefixCountBits(!processed);
                uint workItemIndex = nodeReadOffset >> 2;
                if ((waveIndex & (CHILDREN_PER_HIERARCHY_NODE - 1)) == 0)
                {
                    workItem = workItemQueue[workItemIndex];
                }
                uint readLane = waveIndex & ~(CHILDREN_PER_HIERARCHY_NODE - 1);
                workItem = WaveReadLaneAt(workItem, readLane);
            }
            processed = false;
            
            bool isValid = (workItem.x != ~0u && workItem.y != ~0u && workItem.z != ~0u);
            
            if (isValid)
            {
                // if node
                traversal.ProcessNode(workItem);
                processed = true;
            }
        }
            
        // Process leaves
        if (WaveActiveAllTrue(!processed))
        {
            uint leafReadOffset;
            WaveInterlockedAdd(queue.leafReadOffset, 1, leafReadOffset);
            uint leafWriteOffset = queue.leafWriteOffset;

            if (leafReadOffset < leafWriteOffset)
            {
                workItem = workItemQueue[leafReadOffset];
                traversal.ProcessLeaf(workItem);
            }
        }
    }
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
}

