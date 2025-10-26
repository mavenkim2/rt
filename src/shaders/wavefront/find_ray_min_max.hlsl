#include "../../rt/shader_interop/radix_sort_shaderinterop.h"
#include "../../rt/shader_interop/wavefront_shaderinterop.h"
#include "wavefront_helper.hlsli"

StructuredBuffer<WavefrontQueue> queues : register(t0);
ConstantBuffer<WavefrontDescriptors> descriptors : register(b1);

#define WORKGROUP_SIZE 256
#define SUBGROUP_SIZE 32

groupshared float3 minPs[WORKGROUP_SIZE / SUBGROUP_SIZE];
groupshared float3 maxPs[WORKGROUP_SIZE / SUBGROUP_SIZE];

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex, uint3 groupID : SV_GroupID)
{
    uint start = queues[WAVEFRONT_RAY_QUEUE_INDEX].readOffset;
    uint end = queues[WAVEFRONT_RAY_QUEUE_INDEX].writeOffset;
    uint queueIndex = start + dtID.x;

    if (queueIndex >= end) return;
    queueIndex %= WAVEFRONT_QUEUE_SIZE;

    if (groupIndex == 0)
    {
        for (int i = 0; i < WORKGROUP_SIZE / SUBGROUP_SIZE; i++)
        {
            minPs[i] = FLT_MAX;
            maxPs[i] = -FLT_MAX;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    float3 pos = GetFloat3(descriptors.rayQueuePosIndex, queueIndex);

    float3 minP = WaveActiveMin(pos);
    float3 maxP = WaveActiveMax(pos);

    if (WaveIsFirstLane())
    {
        uint waveInGroupIndex = groupIndex / WaveGetLaneCount();
        minPs[waveInGroupIndex] = minP;
        maxPs[waveInGroupIndex] = maxP;
    }

    GroupMemoryBarrierWithGroupSync();

    uint maxWavesInGroup = WORKGROUP_SIZE / WaveGetLaneCount();

    uint waveIndex = WaveGetLaneIndex();
    if (waveIndex < maxWavesInGroup)
    {
        float3 minPos = minPs[waveIndex];
        minP = WaveActiveMin(minPos);

        float3 maxPos = maxPs[waveIndex];
        maxP = WaveActiveMax(maxPos);

        if (groupIndex == 0)
        {
            StoreFloat3(minP, descriptors.rayQueueMinPosIndex, groupID.x);
            StoreFloat3(maxP, descriptors.rayQueueMaxPosIndex, groupID.x);
        }
    }
}
