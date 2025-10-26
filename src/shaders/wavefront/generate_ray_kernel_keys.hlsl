#include "../common.hlsli"
#include "wavefront_helper.hlsli"
#include "../../rt/shader_interop/radix_sort_shaderinterop.h"
#include "../../rt/shader_interop/wavefront_shaderinterop.h"

StructuredBuffer<WavefrontQueue> queues : register(t0);
ConstantBuffer<WavefrontDescriptors> descriptors : register(b1);
RWStructuredBuffer<SortKey> sortKeys : register(u2);
StructuredBuffer<uint> indirectBuffer : register(t3);

#define WORKGROUP_SIZE 256
#define SUBGROUP_SIZE 32

groupshared float3 minPs[WORKGROUP_SIZE / SUBGROUP_SIZE];
groupshared float3 maxPs[WORKGROUP_SIZE / SUBGROUP_SIZE];

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    uint start = queues[WAVEFRONT_RAY_QUEUE_INDEX].readOffset;
    uint end = queues[WAVEFRONT_RAY_QUEUE_INDEX].writeOffset;
    uint queueIndex = start + dtID.x;

    if (queueIndex >= end) return;
    queueIndex %= WAVEFRONT_QUEUE_SIZE;

    // Calculate the scale
    float3 maxPos = -FLT_MAX;
    float3 minPos = FLT_MAX;

    uint numGroups = indirectBuffer[3 * WAVEFRONT_RAY_QUEUE_INDEX + 0];
    uint waveLaneCount = WaveGetLaneCount();

    for (uint i = groupIndex; i < numGroups; i += WORKGROUP_SIZE)
    {
        float3 minP = GetFloat3(descriptors.rayQueueMinPosIndex, i);
        float3 maxP = GetFloat3(descriptors.rayQueueMaxPosIndex, i);

        float3 minP_ = WaveActiveMin(minP);
        float3 maxP_ = WaveActiveMax(maxP);

        minPos = min(minP_, minPos);
        maxPos = max(maxP_, maxPos);
    }

    if (WaveIsFirstLane())
    {
        uint waveInGroupIndex = groupIndex / WaveGetLaneCount();
        minPs[waveInGroupIndex] = minPos;
        maxPs[waveInGroupIndex] = maxPos;
    }

    GroupMemoryBarrierWithGroupSync();

    uint maxWavesInGroup = WORKGROUP_SIZE / WaveGetLaneCount();
    uint waveIndex = WaveGetLaneIndex() < maxWavesInGroup;
    if (waveIndex < maxWavesInGroup)
    {
        minPos = WaveActiveMin(minPs[waveIndex]);
        maxPos = WaveActiveMax(maxPs[waveIndex]);
    }
    minPos = WaveReadLaneFirst(minPos);
    maxPos = WaveReadLaneFirst(maxPos);

    // 9 bits per origin dimension
    float3 scale = 511.f / (maxPos - minPos);

    float3 pos = GetFloat3(descriptors.rayQueuePosIndex, queueIndex);
    float3 dir = GetFloat3(descriptors.rayQueueDirIndex, queueIndex);

    uint3 quantizedP = uint3((pos - minPos) * scale + 0.5f);

    uint posKey = (MortonCode3(quantizedP.x)) | (MortonCode3(quantizedP.y) << 1u) | (MortonCode3(quantizedP.z) << 2u);
    uint dirKey = ((dir.z >= 0.f) << 2u) | ((dir.y >= 0.f) << 1u) | ((dir.x >= 0.f));
    uint key = (posKey << 3u) | dirKey;

    SortKey sortKey;
    sortKey.key = key;
    sortKey.index = queueIndex;

    sortKeys[dtID.x] = sortKey;
}
