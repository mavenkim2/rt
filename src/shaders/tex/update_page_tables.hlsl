#include "../common.hlsli"
#include "../../rt/shader_interop/virtual_textures_shaderinterop.h"

StructuredBuffer<PageTableUpdateRequest> requests : register(t0);

[[vk::push_constant]] PageTableUpdatePushConstant pc;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x >= pc.numRequests) return;

    PageTableUpdateRequest request = requests[dispatchThreadID.x];
    uint descriptorIndex = pc.bindlessPageTableStartIndex + request.mipLevel;
    uint2 virtualPage = request.virtualPage;
    RWTexture2D<uint> pageTableMip = bindlessRWTextureUint[NonUniformResourceIndex(descriptorIndex)];
    pageTableMip[virtualPage] = request.packed;
}
