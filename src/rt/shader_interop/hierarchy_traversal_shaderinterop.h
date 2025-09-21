#ifndef HIERARCHY_TRAVERSAL_SHADERINTEROP_H_
#define HIERARCHY_TRAVERSAL_SHADERINTEROP_H_
#ifdef __cplusplus
namespace rt
{
#endif

struct Queue
{
    uint nodeReadOffset;
    uint nodeWriteOffset;
    int numNodes;

    uint leafReadOffset;
    uint leafWriteOffset;

    uint debugLeafWriteOffset;
};

#define CANDIDATE_NODE_FLAG_STREAMING_ONLY (1u << 0u)
#define CANDIDATE_NODE_FLAG_HIGHEST_DETAIL (1u << 1u)

struct CandidateNode
{
    uint instanceID;
    uint nodeOffset;
    uint blasIndex;
    uint flags;
};

struct VisibleCluster
{
    uint pageIndex_clusterIndex;
    uint blasIndex;
};

struct CLASPageInfo
{
    uint addressStartIndex;
    uint accelByteOffset;
    uint clasCount;
    uint clasSize;
    uint numTriangleClusters;

    uint tempClusterOffset;
};

struct StreamingRequest
{
    float priority;
    uint instanceID;
    uint pageIndex_numPages;
};

#ifdef __cplusplus
}
#endif
#endif
