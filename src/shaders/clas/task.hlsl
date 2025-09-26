#include "cull.hlsli"
#include "../dense_geometry.hlsli"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<VisibleCluster> visibleClusters : register(t0);
StructuredBuffer<GPUTransform> instanceTransforms : register(t1);
ConstantBuffer<GPUScene> gpuScene : register(b2);
StructuredBuffer<GPUInstance> instances : register(t3);
StructuredBuffer<PartitionInfo> partitionInfos : register(t4);

#define THREAD_GROUP_SIZE 32
struct Payload 
{
    uint clusterIndices[THREAD_GROUP_SIZE];
};

groupshared uint numVisibleClusters;
groupshared Payload payload;

[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
#if 0
    if (groupIndex == 0)
    {
        numVisibleClusters = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    
    uint visibleClusterIndex = dtID.x;
    VisibleCluster cluster = visibleClusters[visibleClusterIndex];
    uint instanceIndex;
    GPUInstance instance = instances[instanceIndex];
    PartitionInfo info = partitionInfos[instance.partitionIndex];
    float3x4 worldFromObject = ConvertGPUMatrix(instanceTransforms[instance.transformIndex], info.base, info.scale);

    uint clusterID = cluster.pageIndex_clusterIndex;
    uint pageIndex = GetPageIndexFromClusterID(clusterID); 
    uint clusterIndex = GetClusterIndexFromClusterID(clusterID);

    uint basePageAddress = GetClusterPageBaseAddress(pageIndex);
    uint numClusters = GetNumClustersInPage(basePageAddress);
    DenseGeometry dg = GetDenseGeometryHeader(basePageAddress, numClusters, clusterIndex);

    // TODO occlusion culling
    bool cull = FrustumCull(gpuScene.clipFromRender, worldFromObject, dg.boundsMin, dg.boundsMax, gpuScene.p22, gpuScene.p23);
    if (!cull && dg.numTriangles)
    {
        uint payloadIndex;
        InterlockedAdd(numVisibleClusters, 1, payloadIndex);
        payload.clusterIndices[payloadIndex] = clusterID;
    }

    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(numClusters, 1, 1, payload);
#endif
}
