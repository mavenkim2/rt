#include "path_guiding.hlsli"

StructuredBuffer<VMM> vmms : register(t0);
StructuredBuffer<float3> sampleDirections : register(t1);
StructuredBuffer<uint> sampleVMMIndices : register(t2);

StructuredBuffer<uint> vmmOffsets : register(t3);
RWStructuredBuffer<uint> vmmCounts : register(u4);
RWStructuredBuffer<Statistics> vmmStatistics : register(u5);

struct Num 
{
    uint num;
};

[[vk::push_constant]] Num num;

[numthreads(PATH_GUIDING_GROUP_SIZE, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 dtID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    uint totalNumSamples = num.num;
    if (dtID.x >= totalNumSamples) return;

    float3 sampleDirection = sampleDirections[dtID.x];
    uint vmmIndex = sampleVMMIndices[dtID.x];
    VMM vmm = vmms[vmmIndex];

    Statistics statistics;
    float V = 0.f;

    for (uint i = 0; i < vmm.numComponents; i++)
    {
        float cosTheta = dot(sampleDirection, vmm.directions[i]);
        float norm = CalculateVMFNormalization(vmm.kappas[i]);
        float v = norm * exp(vmm.kappas[i] * min(cosTheta - 1.f, 0.f));
        statistics.sumWeights[i] = vmm.weights[i] * v;
        V += statistics.sumWeights[i];
    }

    // TODO: what do I do here?
    if (V <= 1e-16f)
    {
        return;
    }

    float invV = 1.f / V;
    for (uint i = 0; i < vmm.numComponents; i++)
    {
        statistics.sumWeights[i] *= invV;
    }
    statistics.weightedLogLikelihood = log(V);

#if 0
    float weightedLogLikelihood = sample.weight * log(V);

    float3 directions[MAX_COMPONENTS];

    for (uint i = 0; i < vmm.numComponents; i++)
    {
        directions[i] = assignments[i] * sample.weight * sampleDirection;
        assignments[i] = assignments[i] * sample.weight;
    }
#endif

    uint index;
    InterlockedAdd(vmmCounts[vmmIndex], 1, index);
    index = vmmOffsets[vmmIndex] + index;

    vmmStatistics[index] = statistics;
}
