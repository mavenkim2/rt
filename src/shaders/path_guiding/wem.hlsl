#include "path_guiding.hlsli"

StructuredBuffer<VMM> vmms : register(t0);
StructuredBuffer<float3> sampleDirections : register(t1);
StructuredBuffer<uint> sampleVMMIndices : register(t2);

StructuredBuffer<uint> vmmOffsets : register(t3);
StructuredBuffer<uint> vmmCounts : register(t4);
RWStructuredBuffer<Statistics> vmmStatistics : register(u5);

struct Num 
{
    uint num;
};

[[vk::push_constant]] Num num;

groupshared Statistics statistics_;

[numthreads(PATH_GUIDING_GROUP_SIZE, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 dtID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    uint totalNumSamples = num.num;
    uint vmmIndex = groupID.x;
    uint sampleCount = vmmCounts[vmmIndex];
    uint offset = vmmOffsets[vmmIndex];
    VMM vmm = vmms[vmmIndex];

    if (groupIndex == 0)
    {
        for (uint i = 0; i < vmm.numComponents; i++)
        {
            statistics_.sumWeights[i] = 0.f;
        }
        statistics_.weightedLogLikelihood = 0.f;
    }

    uint laneIndex = WaveGetLaneIndex();
    Statistics statistics;

    for (uint sampleIndex = groupIndex; sampleIndex < sampleCount; sampleIndex += PATH_GUIDING_GROUP_SIZE)
    {
        float3 sampleDirection = sampleDirections[sampleIndex];
        float V = 0.f;

        for (uint componentIndex = 0; componentIndex < vmm.numComponents; componentIndex++)
        {
            float cosTheta = dot(sampleDirection, vmm.directions[componentIndex]);
            float norm = CalculateVMFNormalization(vmm.kappas[componentIndex]);
            float v = norm * exp(vmm.kappas[componentIndex] * min(cosTheta - 1.f, 0.f));
            statistics.sumWeights[componentIndex] = vmm.weights[componentIndex] * v;

            V += statistics.sumWeights[componentIndex];
        }

        // TODO: what do I do here?
        if (V <= 1e-16f) continue;

        float invV = 1.f / V;
        for (uint i = 0; i < vmm.numComponents; i++)
        {
            statistics.sumWeights[i] *= invV;
        }
        statistics.weightedLogLikelihood = log(V); // * sampleWeight

        for (uint componentIndex = 0; componentIndex < vmm.numComponents; componentIndex++)
        {
            float softAssignmentWeight = statistics.sumWeights[componentIndex];
            float sumWeights = WaveActiveSum(softAssignmentWeight);
            float weightedLogLikelihood = WaveActiveSum(statistics.weightedLogLikelihood);

            if (groupIndex == 0)
            {
                statistics_.sumWeights[componentIndex] += sumWeights;
                statistics_.weightedLogLikelihood += weightedLogLikelihood;
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Normalize
    if (groupIndex < MAX_COMPONENTS)
    {
        float componentWeight = statistics_.sumWeights[groupIndex];
        float normWeight = WaveActiveSum(componentWeight);
        normWeight = normWeight > FLT_EPSILON ? float(sampleCount) / normWeight : 0.f;
        statistics_.sumWeights[groupIndex] *= normWeight;
    }
 
    GroupMemoryBarrierWithGroupSync();

    if (groupIndex == 0)
    {
        vmmStatistics[vmmIndex] = statistics_;
    }

#if 0
    float weightedLogLikelihood = sample.weight * log(V);

    float3 directions[MAX_COMPONENTS];

    for (uint i = 0; i < vmm.numComponents; i++)
    {
        directions[i] = assignments[i] * sample.weight * sampleDirection;
        assignments[i] = assignments[i] * sample.weight;
    }
    uint index;
    InterlockedAdd(vmmCounts[vmmIndex], 1, index);
    index = vmmOffsets[vmmIndex] + index;

    vmmStatistics[index] = statistics;
#endif
}
