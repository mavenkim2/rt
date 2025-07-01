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
    uint pageIndex;
    uint clusterIndex;
    uint instanceID;
    uint blasIndex;
};

struct CLASPageInfo
{
    uint addressStartIndex;
};

#ifdef __cplusplus
}
#endif
#endif
