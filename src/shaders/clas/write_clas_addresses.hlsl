#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<CLASPageInfo> clasPageInfos : register(t0);
StructuredBuffer<DecodeClusterData> decodeClusterDatas : register(t1);

StructuredBuffer<uint64_t> srcBuildAddresses : register(t2);
StructuredBuffer<uint32_t> srcBuildSizes : register(t3);

StructuredBuffer<uint64_t> srcTemplateAddresses : register(t4);
StructuredBuffer<uint32_t> srcTemplateSizes : register(t5);

RWStructuredBuffer<uint64_t> dstAddresses : register(u6);
RWStructuredBuffer<uint32_t> dstSizes : register(u7);

#define THREADS_PER_GROUP 32
[numthreads(THREADS_PER_GROUP, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint clusterDataIndex = dtID.x;
    if (clusterDataIndex.x >+ GLOBALS_CLAS_COUNT_INDEX) return;

    DecodeClusterData data = decodeClusterDatas[clusterDataIndex];

    CLASPageInfo clasPageInfo = clasPageInfos[data.pageIndex];
    uint clusterIndex = data.clusterIndex;

    uint srcIndex = clasPageInfo.tempClusterOffset + clusterIndex;
    uint64_t address = 0;
    uint size = 0;
    if (data.addressIndex >> 31u)
    {
        uint index = data.addressIndex & 0x7fffffffu;
        address = srcTemplateAddresses[index];
        size = srcTemplateSizes[index];
    }
    else 
    {
        uint index = data.addressIndex;
        address = srcBuildAddresses[index];
        size = srcBuildSizes[index];
    }

    uint dstIndex = clasPageInfo.addressStartIndex + data.clusterIndex;
    dstAddresses[dstIndex] = address;
    dstSizes[dstIndex] = size;
}
