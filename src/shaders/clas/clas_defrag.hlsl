#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<uint> evictedPages : register(t0);
RWStructuredBuffer<CLASPageInfo> clasPageInfos : register(u1);

RWStructuredBuffer<uint64_t> clasAddresses : register(u2);
RWStructuredBuffer<uint> clasSizes : register(u3);

RWStructuredBuffer<uint> globals : register(u4);

RWStructuredBuffer<uint64_t> moveDescriptors : register(u5);
RWStructuredBuffer<uint64_t> moveDstAddresses : register(u6);
RWStructuredBuffer<uint64_t> moveDstSizes : register(u7);

[[vk::push_constant]] NumPushConstant pc;

groupshared uint pageShiftLeftClas;
groupshared uint pageShiftLeftBytes;
groupshared uint pageEvicted;

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

    InterlockedAdd(globals[GLOBALS_OLD_PAGE_DATA_BYTES], clasPageInfo.clasSize);

    uint shiftLeftClas = 0;
    uint shiftLeftBytes = 0;
    uint evicted = 0;

    // Add up the CLAS accel size/count of all the pages to the left
    for (uint pageIndex = groupIndex; pageIndex < pc.num; pageIndex += THREADS_PER_GROUP)
    {
        uint evictedPage = evictedPages[pageIndex];
        CLASPageInfo clasPageInfo = clasPageInfos[evictedPage];

        if (clasPageInfo.addressStartIndex >= addressStartIndex) continue;

        evicted |= evictedPage == pageIndex;

        shiftLeftBytes += clasPageInfo.clasSize;
        shiftLeftClas += clasPageInfo.clasCount;
    }

    InterlockedAdd(pageShiftLeftBytes, shiftLeftBytes);
    InterlockedAdd(pageShiftLeftClas, shiftLeftClas);
    InterlockedOr(pageEvicted, evicted);

    GroupMemoryBarrierWithGroupSync();

    if (pageEvicted || (pageShiftLeftClas == 0 && pageShiftLeftBytes == 0)) return;

    for (uint clusterIndex = groupIndex; clusterIndex < clasPageInfo.clasCount; clusterIndex += THREADS_PER_GROUP)
    {
        uint64_t srcAddress = clasAddresses[addressStartIndex + clusterIndex];
        uint64_t dstAddress = srcAddress - shiftLeftBytes;
        clasAddresses[addressStartIndex - shiftLeftClas] = srcAddress;

        uint srcSize = clasSizes[addressStartIndex + clusterIndex];
        clasSizes[addressStartIndex - shiftLeftClas] = srcSize;

        uint descriptorIndex;
        InterlockedAdd(globals[GLOBALS_DEFRAG_CLAS_COUNT], 1, descriptorIndex);

        moveDescriptors[descriptorIndex] = srcAddress;
        moveDstAddresses[descriptorIndex] = dstAddress;
        moveDstSizes[descriptorIndex] = srcSize;
    }

    if (groupIndex == 0)
    {
        clasPageInfos[pageIndex].addressStartIndex -= shiftLeftClas;
    }
}
