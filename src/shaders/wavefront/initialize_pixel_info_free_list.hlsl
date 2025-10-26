#include "../../rt/shader_interop/wavefront_shaderinterop.h"
RWStructuredBuffer<int> freeList : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    int maxNum = WAVEFRONT_QUEUE_SIZE + 1;
    freeList[0] = maxNum;
    for (int i = 0; i < maxNum; i++)
    {
        freeList[i + 1] = maxNum - i - 1;
    }
}
