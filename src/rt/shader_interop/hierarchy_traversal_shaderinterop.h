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

struct CandidateNode
{
    uint instanceID;
    uint nodeOffset;
    uint blasIndex;
    uint pad;
};

struct VisibleCluster
{
    uint isVoxel_pageIndex_clusterIndex;
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
