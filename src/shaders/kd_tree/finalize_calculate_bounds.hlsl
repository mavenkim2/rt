#include "../common.hlsli"
#include "../../rt/shader_interop/kd_tree_shaderinterop.h"

RWStructuredBuffer<float3> bounds : register(u0);
StructuredBuffer<float3> intermediateBounds : register(t1);
StructuredBuffer<uint> numBuffer : register(t2);

// TODO: platform specific, minimum 4 on some archs
groupshared float3 mins[KD_TREE_REDUCTION_SIZE / 32];
groupshared float3 maxs[KD_TREE_REDUCTION_SIZE / 32];

[numthreads(KD_TREE_REDUCTION_SIZE, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    // TODO hardcoded
    uint num = 1024;
    if (dtID.x >= num) return;

    uint blockIndex = groupID.x;
    uint blockDim = KD_TREE_REDUCTION_SIZE;

    uint laneIndex = WaveGetLaneIndex();
    uint laneCount = WaveGetLaneCount();
    uint waveIndex = groupIndex / laneCount;
    uint waveCount = KD_TREE_REDUCTION_SIZE / laneCount;

    uint dataIndex = blockIndex * (blockDim * 2) + groupIndex;

    float3 boundsMin0 = intermediateBounds[2 * dataIndex];
    float3 boundsMax0 = intermediateBounds[2 * dataIndex + 1];

    float3 boundsMin1 = dataIndex + blockDim < num ? intermediateBounds[2 * (dataIndex + blockDim)] : boundsMin0;
    float3 boundsMax1 = dataIndex + blockDim < num ? intermediateBounds[2 * (dataIndex + blockDim) + 1] : boundsMax0;

    float3 laneMinP = min(boundsMin0, boundsMin1);
    float3 laneMaxP = max(boundsMax0, boundsMax1);

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
            bounds[0] = waveMinP;
            bounds[1] = waveMaxP;
        }
    }
}
