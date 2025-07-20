#include "../../rt/shader_interop/as_shaderinterop.h"
StructuredBuffer<GPUClusterFixup> clusterFixups : register(t0);
RWByteAddressBuffer clusterPageData : register(u1);

[[vk::push_constant]] NumPushConstant pc;

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint fixupIndex = dtID.x;
    if (fixupIndex >= pc.num) return;
    GPUClusterFixup fixup = clusterFixups[fixupIndex];
    uint offset = fixup.offset & ~0x1;
    if (fixup.offset & 1)
    {
        clusterPageData.InterlockedAnd(offset, ~CLUSTER_STREAMING_LEAF_FLAG);
    }
    else 
    {
        clusterPageData.InterlockedOr(offset, CLUSTER_STREAMING_LEAF_FLAG);
    }
}
