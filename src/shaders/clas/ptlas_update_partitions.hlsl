#include "../../rt/shader_interop/as_shaderinterop.h"

RWStructuredBuffer<int> instanceIDFreeList : register(u0);
RWStructuredBuffer<uint> globals : register(u1);

[numthreads(1, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    if (dtID.x != 0) return;
    instanceIDFreeList[0] = max(instanceIDFreeList[0], 0);
    globals[GLOBALS_DEBUG2] = instanceIDFreeList[0];
}
