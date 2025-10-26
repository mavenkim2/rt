#include "../../rt/shader_interop/wavefront_shaderinterop.h"
#include "../../rt/shader_interop/radix_sort_shaderinterop.h"
#include "wavefront_helper.hlsli"

RWStructuredBuffer<WavefrontQueue> queues : register(u0);
RWStructuredBuffer<uint> indirectBuffer : register(u1);
RWStructuredBuffer<uint> numElementsToSort : register(u2);
RWStructuredBuffer<uint> tileInfo : register(u3);

[[vk::push_constant]] WavefrontPushConstant pc;

[numthreads(1, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    if (dtID.x != 0) return;

    bool flush = pc.flush;

    int finishedQueueIndex = pc.finishedQueueIndex;
    if (finishedQueueIndex != -1)
    {
        if (indirectBuffer[3 * finishedQueueIndex] != 0)
        {
            queues[finishedQueueIndex].readOffset = flush ? queues[finishedQueueIndex].writeOffset : 
                queues[finishedQueueIndex].readOffset + WAVEFRONT_WORKING_SET_SIZE;
        }
    }

    int dispatchQueueIndex = pc.dispatchQueueIndex;

    uint numToDispatch = queues[dispatchQueueIndex].writeOffset 
                         - queues[dispatchQueueIndex].readOffset;

    if (dispatchQueueIndex == WAVEFRONT_GENERATE_CAMERA_RAYS_INDEX)
    {
        uint numRaysEnqueued = queues[WAVEFRONT_RAY_QUEUE_INDEX].writeOffset - queues[WAVEFRONT_RAY_QUEUE_INDEX].readOffset;
        if (numRaysEnqueued >= WAVEFRONT_WORKING_SET_SIZE && !flush)
        {
            indirectBuffer[3 * dispatchQueueIndex] = 0;
            indirectBuffer[3 * dispatchQueueIndex + 1] = 0;
            indirectBuffer[3 * dispatchQueueIndex + 2] = 0;
        }
        else 
        {
            const uint tileWidth   = 8;
            const uint tileNumRays = 64;
            const uint maxRays = WAVEFRONT_WORKING_SET_SIZE;

            // TODO IMPORTANT
            const uint targetWidth = 2560;
            const uint targetHeight = 1440;

            const uint maxTiles    = (maxRays + tileNumRays - 1) / tileNumRays;
            const uint numTilesX   = (targetWidth + tileWidth - 1) / tileWidth;
            const uint numTilesY   = (targetHeight + tileWidth - 1) / tileWidth;
            const uint totalTiles  = numTilesX * numTilesY;

            uint tileIndex = tileInfo[1];
            uint numTilesInDispatch = min(totalTiles - tileIndex, maxTiles);

            if (tileIndex > totalTiles)
            {
                indirectBuffer[3 * dispatchQueueIndex] = 0;
                indirectBuffer[3 * dispatchQueueIndex + 1] = 0;
                indirectBuffer[3 * dispatchQueueIndex + 2] = 0;
            }
            else if (numRaysEnqueued + numTilesInDispatch <= WAVEFRONT_QUEUE_SIZE)
            {
                tileInfo[0] = tileIndex;
                tileInfo[1] += maxTiles;

                indirectBuffer[3 * dispatchQueueIndex] = numTilesInDispatch;
                indirectBuffer[3 * dispatchQueueIndex + 1] = 1;
                indirectBuffer[3 * dispatchQueueIndex + 2] = 1;
            }
            else 
            {
                indirectBuffer[3 * dispatchQueueIndex] = 0;
                indirectBuffer[3 * dispatchQueueIndex + 1] = 0;
                indirectBuffer[3 * dispatchQueueIndex + 2] = 0;
            }
        }
    }
    else if (numToDispatch < WAVEFRONT_WORKING_SET_SIZE && !flush)
    {
        indirectBuffer[3 * dispatchQueueIndex] = 0;
        indirectBuffer[3 * dispatchQueueIndex + 1] = 0;
        indirectBuffer[3 * dispatchQueueIndex + 2] = 0;

        if (dispatchQueueIndex == WAVEFRONT_RAY_QUEUE_INDEX)
        {
            indirectBuffer[3 * WAVEFRONT_RAY_SORT_INDEX] = 0;
            indirectBuffer[3 * WAVEFRONT_RAY_SORT_INDEX + 1] = 0;
            indirectBuffer[3 * WAVEFRONT_RAY_SORT_INDEX + 2] = 0;
        }

    }
    else 
    {
        uint num = flush ? numToDispatch : min(numToDispatch, WAVEFRONT_WORKING_SET_SIZE);
        if (dispatchQueueIndex == WAVEFRONT_RAY_QUEUE_INDEX)
        {
            indirectBuffer[3 * dispatchQueueIndex] = num;
            indirectBuffer[3 * dispatchQueueIndex + 1] = 1;
            indirectBuffer[3 * dispatchQueueIndex + 2] = 1;

            indirectBuffer[3 * WAVEFRONT_RAY_SORT_INDEX] = (num + SORT_WORKGROUP_SIZE - 1) / SORT_WORKGROUP_SIZE;
            indirectBuffer[3 * WAVEFRONT_RAY_SORT_INDEX + 1] = 1;
            indirectBuffer[3 * WAVEFRONT_RAY_SORT_INDEX + 2] = 1;

            numElementsToSort[0] = num;
        }
        else 
        {
            indirectBuffer[3 * dispatchQueueIndex] = (num + 31) / 32;
            indirectBuffer[3 * dispatchQueueIndex + 1] = 1;
            indirectBuffer[3 * dispatchQueueIndex + 2] = 1;
        }
    }
}
