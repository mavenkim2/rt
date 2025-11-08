#include "../common.hlsli"
#include "../../rt/shader_interop/kd_tree_shaderinterop.h"

RWStructuredBuffer<float3> intermediateBounds : register(u0);
StructuredBuffer<float3> points : register(t1);
StructuredBuffer<uint> numBuffer : register(t2);

// TODO: platform specific, minimum 4 on some archs
groupshared float3 mins[KD_TREE_REDUCTION_SIZE / 32];
groupshared float3 maxs[KD_TREE_REDUCTION_SIZE / 32];

[numthreads(KD_TREE_REDUCTION_SIZE, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    uint num = (1u << 20u);//numBuffer[0];
    if (dtID.x >= num) return;

    uint blockIndex = groupID.x;
    uint blockDim = KD_TREE_REDUCTION_SIZE;

    uint laneIndex = WaveGetLaneIndex();
    uint laneCount = WaveGetLaneCount();
    uint waveIndex = groupIndex / laneCount;
    uint waveCount = KD_TREE_REDUCTION_SIZE / laneCount;

    uint dataIndex = blockIndex * (blockDim * 2) + groupIndex;
    uint numBlocks = 1024u;
    uint gridSize = (blockDim * 2) * numBlocks;

    float3 laneMinP = FLT_MAX;
    float3 laneMaxP = -FLT_MAX;

    while (dataIndex < num)
    {
        float3 pt0 = points[dataIndex];
        float3 pt1 = dataIndex + blockDim < num ? points[dataIndex + blockDim] : pt0;

        laneMinP = min(laneMinP, min(pt0, pt1));
        laneMaxP = max(laneMaxP, max(pt0, pt1));

        dataIndex += gridSize;
    }

    float3 waveMinP = WaveActiveMin(laneMinP);
    float3 waveMaxP = WaveActiveMax(laneMaxP);

    if (laneIndex == 0)
    {
        mins[groupIndex / laneCount] = waveMinP;
        maxs[groupIndex / laneCount] = waveMaxP;
    }
    GroupMemoryBarrierWithGroupSync();

    if (groupIndex < KD_TREE_REDUCTION_SIZE / laneCount)
    {
        laneMinP = mins[groupIndex];
        laneMaxP = maxs[groupIndex];
        waveMinP = WaveActiveMin(laneMinP);
        waveMaxP = WaveActiveMax(laneMaxP);

        if (groupIndex == 0)
        {
            intermediateBounds[2 * blockIndex] = waveMinP;
            intermediateBounds[2 * blockIndex + 1] = waveMaxP;
        }
    }
}
