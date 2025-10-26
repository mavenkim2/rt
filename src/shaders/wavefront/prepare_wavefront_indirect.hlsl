#include "../../rt/shader_interop/wavefront_shaderinterop.h"
#include "wavefront_helper.hlsli"

RWStructuredBuffer<WavefrontQueue> queues : register(u0);
RWStructuredBuffer<uint> indirectBuffer : register(u1);

[[vk::push_constant]] WavefrontPushConstant pc;

[numthreads(1, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    if (dtID.x != 0) return;

    int finishedQueueIndex = pc.finishedQueueIndex;
    if (finishedQueueIndex != -1)
    {
        queues[finishedQueueIndex].readOffset = queues[finishedQueueIndex].writeOffset;
    }
    int dispatchQueueIndex = pc.dispatchQueueIndex;
    if (dispatchQueueIndex != -1)
    {
        uint numToDispatch = queues[dispatchQueueIndex].writeOffset 
                             - queues[dispatchQueueIndex].readOffset;

        bool isRayTrace = dispatchQueueIndex == WAVEFRONT_RAY_QUEUE_INDEX;
        uint numToDispatchX = isRayTrace ? numToDispatch : (numToDispatch + 31) / 32;

        indirectBuffer[3 * dispatchQueueIndex] = numToDispatchX;
        indirectBuffer[3 * dispatchQueueIndex + 1] = 1;
        indirectBuffer[3 * dispatchQueueIndex + 2] = 1;
    }
}
