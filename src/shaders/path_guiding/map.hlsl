#include "path_guiding.hlsli"

RWStructuredBuffer<VMM> vmms : register(u0);
StructuredBuffer<Statistics> vmmStatistics : register(t1);
StructuredBuffer<uint> vmmCounts : register(t2);

//StructuredBuffer<uint> vmmCounts : register(t2);

struct Num 
{
    uint num;
};

[[vk::push_constant]] Num num;

[numthreads(PATH_GUIDING_GROUP_SIZE, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 dtID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    //uint totalNumVMMs = num.num;
    //if (dtID.x >= totalNumVMMs) return;

    const float weightPrior = 0.01f;

    uint vmmIndex = groupID.x;
    VMM vmm = vmms[vmmIndex];
    Statistics statistics = vmmStatistics[groupID.x];
    uint numSamples = vmmCounts[vmmIndex];

    // TODO: should it be one thread per VMM instead?
    // Update weights
    if (groupIndex < vmm.numComponents)
    {
        // TODO IMPORTANT: handle previous stats
        float weight = statistics.sumWeights[groupIndex];
        weight = (weightPrior + weight) / (weightPrior * vmm.numComponents + numSamples);

        vmms[vmmIndex].weights[groupIndex] = weight;
    }
    else if (groupIndex < MAX_COMPONENTS)
    {
        vmms[vmmIndex].weights[groupIndex] = 0.f;
    }

    // Update kappas and directions
    // TODO IMPORTANT: handle previous stats
    uint totalNumSamples = numSamples;

    const float currentEstimationWeight = numSamples / totalNumSamples;
    const float previousEstimationWeight = 1.f - currentEstimationWeight;

    const float meanCosinePrior = 0.f;
    const float meanCosinePriorStrength = 0.2f;
    const float maxKappa = 32000.f;
    const float maxMeanCosine = KappaToMeanCosine(maxKappa);

    if (groupIndex < vmm.numComponents)
    {
        float3 currentMeanDirection = statistics.sumWeights[groupIndex] > 0.f 
                                    ? statistics.sumWeightedDirections[groupIndex] / statistics.sumWeights[groupIndex] : 0.f;
        float3 previousMeanDirection = 0.f;

        float3 meanDirection = currentMeanDirection * currentEstimationWeight + previousMeanDirection * previousEstimationWeight;
        float meanCosine = length(meanDirection);

        if (meanCosine > 0.f)
        {
            vmms[vmmIndex].directions[groupIndex] = meanDirection / meanCosine;
        }
        float partialNumSamples = totalNumSamples * vmms[vmmIndex].weights[groupIndex];
        meanCosine = (meanCosinePrior * meanCosinePriorStrength + meanCosine * partialNumSamples) / (meanCosinePriorStrength + partialNumSamples);
        meanCosine = min(meanCosine, maxMeanCosine);
        vmms[vmmIndex].kappas[groupIndex] = MeanCosineToKappa(meanCosine);
    }
    else if (groupIndex < MAX_COMPONENTS)
    {
        vmms[vmmIndex].directions[groupIndex] = 0.f;
        vmms[vmmIndex].kappas[groupIndex] = 0.f;
    }
}
