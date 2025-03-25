#include "../rt/shader_interop/as_shaderinterop.h"

struct BLASData
{
    uint clasAddressStartIndex;
    uint clasCount;
};

// Input
StructuredBuffer<BLASData> blasData : register(t0);

// Output
RWStructuredBuffer<ClusterBottomLevelInfo> clusterBottomLevelInfo : register(u1);

[numthreads()]
void main(uint3 DTid: SV_DispatchThreadID)
{
    uint blasIndex = DTid.x;
    if (blasIndex >= blasCount) return;

    clusterBottomLevelInfo[blasIndex].clusterReferencesCount = blasData[blasIndex].clasCount;
    clusterBottomLevelInfo[blasIndex].clusterReferences = ?;
}
