#include "path_guiding.hlsli"

StructuredBuffer<VMM> vmms : register(t0);
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
    VMM vmm = vmms[sample.vmmIndex];

    float assignments[MAX_COMPONENTS];
    uint numAssignments = 0;
    float V = 0.f;

    for (uint i = 0; i < vmm.numComponents; i++)
    {
        float cosTheta = dot(sample.dir, vmm.directions[i]);
        float norm = CalculateVMFNormalization(vmm.kappas[i]);
        float v = norm * exp(vmm.kappas[i] * min(cosTheta - 1.f, 0.f));
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

    float weightedLogLikelihood = sample.weight * log(V);

    float3 directions[MAX_COMPONENTS];

    for (uint i = 0; i < vmm.numComponents; i++)
    {
        directions[i] = assignments[i] * sample.weight * sample.dir;
        assignments[i] = assignments[i] * sample.weight;
    }

    uint index;
    InterlockedAdd(vmmCounts[sample.vmmIndex], 1, index);
    index = vmmOffsets[sample.vmmIndex] + index;
}
