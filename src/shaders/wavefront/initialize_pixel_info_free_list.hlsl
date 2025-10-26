#include "../../rt/shader_interop/wavefront_shaderinterop.h"
RWStructuredBuffer<int> freeList : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    int maxNum = 2 * WAVEFRONT_QUEUE_SIZE;
    if (dtID.x == 0)
    {
        freeList[0] = maxNum;
    }
    freeList[dtID.x + 1] = maxNum - dtID.x - 1;
}
