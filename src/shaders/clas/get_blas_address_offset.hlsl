#include "../../rt/shader_interop/as_shaderinterop.h"

RWStructuredBuffer<BLASData> blasDatas : register(u0);
RWStructuredBuffer<uint> globals : register(u1);

[[vk::push_constant]] FillClusterBottomLevelInfoPushConstant pc;

[numthreads(32, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint blasIndex = DTid.x;
    if (blasIndex >= pc.blasCount) return;

    uint clasStartIndex;
    InterlockedAdd(globals[GLOBALS_CLAS_COUNT_INDEX], blasDatas[blasIndex].clusterCount, clasStartIndex);

    blasDatas[blasIndex].clusterStartIndex = clasStartIndex;
    blasDatas[blasIndex].clusterCount = 0;
}
