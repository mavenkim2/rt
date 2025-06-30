#ifndef HIERARCHY_TRAVERSAL_SHADERINTEROP_H_
#define HIERARCHY_TRAVERSAL_SHADERINTEROP_H_
#ifdef __cplusplus
namespace rt
{
#endif

struct CandidateNode
{
    uint instanceID;
    uint nodeOffset;
    uint blasIndex;
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
