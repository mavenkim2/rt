#include "../../rt/shader_interop/virtual_textures_shaderinterop.h"

StructuredBuffer<PageTableUpdateRequest> requests : register(t0);
RWTexture1D<uint> pageTable : register(u1);

[[vk::push_constant]] PageTableUpdatePushConstant pc;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x >= pc.numRequests) return;

    PageTableUpdateRequest request = requests[dispatchThreadID.x];
    pageTable[request.faceIndex] = request.physicalPageStart;
}
