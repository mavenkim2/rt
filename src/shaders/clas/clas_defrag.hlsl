#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<uint> evictedPages : register(t0);
RWStructuredBuffer<CLASPageInfo> clasPageInfos : register(u1);

StructuredBuffer<uint64_t> clasAddresses : register(t2);
StructuredBuffer<uint> clasSizes : register(t3);

RWStructuredBuffer<uint> globals : register(u4);

RWStructuredBuffer<uint64_t> moveDescriptors : register(u5);
RWStructuredBuffer<uint64_t> moveDstAddresses : register(u6);
RWStructuredBuffer<uint> moveDstSizes : register(u7);

[[vk::push_constant]] DefragPushConstant pc;

groupshared uint pageShiftLeftClas;
groupshared uint pageShiftLeftBytes;
groupshared uint pageEvicted;

groupshared uint descriptorStartIndex;

#define THREADS_PER_GROUP 32

[numthreads(THREADS_PER_GROUP, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    if (groupIndex == 0)
    {
        pageShiftLeftClas = 0;
        pageShiftLeftBytes = 0;
        pageEvicted = false;
    }
    GroupMemoryBarrierWithGroupSync();

    uint pageIndex = groupID.x;
    CLASPageInfo clasPageInfo = clasPageInfos[pageIndex];
    uint addressStartIndex = clasPageInfo.addressStartIndex;
    uint accelByteOffset = clasPageInfo.accelByteOffset;

    uint shiftLeftClas = 0;
    uint shiftLeftBytes = 0;
    uint evicted = 0;

    // Add up the CLAS accel size/count of all the pages to the left
    for (uint index = pc.evictedPageStart + groupIndex; index < pc.evictedPageStart + pc.numEvictedPages; index += THREADS_PER_GROUP)
    {
        uint evictedPage = evictedPages[index];
        CLASPageInfo evictedClasPageInfo = clasPageInfos[evictedPage];
        evicted |= evictedPage == pageIndex;

        if (evictedClasPageInfo.accelByteOffset < accelByteOffset)
        {
            shiftLeftBytes += evictedClasPageInfo.clasSize;
        }
        if (evictedClasPageInfo.addressStartIndex < addressStartIndex)
        {
            shiftLeftClas += evictedClasPageInfo.clasCount;
        }
    }

    InterlockedAdd(pageShiftLeftBytes, shiftLeftBytes);
    InterlockedAdd(pageShiftLeftClas, shiftLeftClas);
    InterlockedOr(pageEvicted, evicted);

    GroupMemoryBarrierWithGroupSync();

    if (groupIndex == 0 && !pageEvicted)
    {
        InterlockedAdd(globals[GLOBALS_OLD_PAGE_DATA_BYTES], clasPageInfo.clasSize);
    }

    if (pageEvicted || (pageShiftLeftClas == 0 && pageShiftLeftBytes == 0))
    {
        if (groupIndex == 0)
        {
            clasPageInfos[pageIndex].tempClusterOffset = ~0u;
        }
        return;
    }

    if (pc.numEvictedPages == 0) return;

    if (groupIndex == 0)
    {
        InterlockedAdd(globals[GLOBALS_DEFRAG_CLAS_COUNT], clasPageInfo.clasCount, descriptorStartIndex);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint clusterIndex = groupIndex; clusterIndex < clasPageInfo.clasCount; clusterIndex += THREADS_PER_GROUP)
    {
        uint addressIndex = addressStartIndex + clusterIndex;
        uint64_t srcAddress = clasAddresses[addressIndex];
        uint64_t dstAddress = srcAddress - pageShiftLeftBytes;
        uint srcSize = clasSizes[addressIndex];

        uint descriptorIndex = descriptorStartIndex + clusterIndex;

        moveDescriptors[descriptorIndex] = srcAddress;
        moveDstAddresses[descriptorIndex] = dstAddress;
        moveDstSizes[descriptorIndex] = srcSize;
    }

    if (groupIndex == 0)
    {
        clasPageInfos[pageIndex].addressStartIndex -= pageShiftLeftClas;
        clasPageInfos[pageIndex].accelByteOffset -= pageShiftLeftBytes;
        clasPageInfos[pageIndex].tempClusterOffset = descriptorStartIndex;
    }
}
