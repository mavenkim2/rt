#include "path_guiding.hlsli"

StructuredBuffer<ParallaxVMM> vmms : register(t0);
StructuredBuffer<PathGuidingSample> samples : register(t1);

StructuredBuffer<uint> vmmOffsets : register(t2);
RWStructuredBuffer<uint> vmmCounts : register(u3);
RWStructuredBuffer<Statistics> vmmStatistics : register(u4);

[numthreads(PATH_GUIDING_GROUP_SIZE, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 dtID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    // Weighted Expectation: Calculate soft assignment weight
    uint totalNumSamples = 0;
    if (dtID.x >= totalNumSamples) return;

    PathGuidingSample sample = samples[dtID.x];
    ParallaxVMM vmm = vmms[sample.vmmIndex];

    float assignments[MAX_COMPONENTS];
    uint numAssignments = 0;
    float V = 0.f;

    for (uint i = 0; i < vmm.numComponents; i++)
    {
        float cosTheta = dot(sample.dir, vmm.directions[i]);
        float v = vmm.normalizations[i] * exp(vmm.kappas[i] * min(cosTheta - 1.f, 0.f));
        assignments[i] = vmm.weights[i] * v;
        V += assignments[i];
    }

    // TODO: what do I do here?
    if (V <= 1e-16f)
    {
        return;
    }

    float invV = 1.f / V;
    for (uint i = 0; i < numAssignments; i++)
    {
        assignments[i] *= invV;
    }

    uint4 mask = WaveMatch(sample.vmmIndex);
    int4 highLanes = (int4)(firstbithigh(mask) | uint4(0, 0x20, 0x40, 0x60));
    uint highLane = (uint)max(max(max(highLanes.x, highLanes.y), highLanes.z), highLanes.w);
    bool leader = WaveGetLaneIndex() == highLane;

    float weightedLogLikelihood = sample.weight * log(V);
    float result = WaveMultiPrefixSum(weightedLogLikelihood, mask);

    float3 directions[MAX_COMPONENTS];

    // TODO: make sure this works
    for (uint i = 0; i < vmm.numComponents; i++)
    {
        directions[i] = WaveMultiPrefixSum(assignments[i] * sample.weight * sample.dir, mask);
        assignments[i] = WaveMultiPrefixSum(assignments[i] * sample.weight, mask);
    }

    if (leader)
    {
        uint index;
        InterlockedAdd(vmmCounts[sample.vmmIndex], index, 1);
        index = vmmOffsets[sample.vmmIndex] + index;

        vmmStatistics[index].weightedLogLikelihood = weightedLogLikelihood;
        for (uint i = 0; i < vmm.numComponents; i++)
        {
            vmmStatistics[index].sumWeightedDirections[i] = directions[i];
            vmmStatistics[index].sumWeights[i] = assignments[i];
        }
    }
}
