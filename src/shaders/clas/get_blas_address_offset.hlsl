#include "../../rt/shader_interop/as_shaderinterop.h"

RWStructuredBuffer<BLASData> blasDatas : register(u0);
RWStructuredBuffer<uint> globals : register(u1);

[[vk::push_constant]] NumPushConstant pc;

[numthreads(32, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint blasIndex = DTid.x;
    if (blasIndex >= pc.num) return;

    uint clasStartIndex;
    InterlockedAdd(globals[GLOBALS_BLAS_CLAS_COUNT_INDEX], blasDatas[blasIndex].clusterCount, clasStartIndex);

    blasDatas[blasIndex].clusterStartIndex = clasStartIndex;
    blasDatas[blasIndex].clusterCount = 0;
}
