#include "path_guiding.hlsli"

RWStructuredBuffer<ParallaxVMM> vmms : register(u0);
StructuredBuffer<Statistics> vmmStatistics : register(t1);

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint totalNumVMMs = 0;
    if (dtID.x >= totalNumVMMs) return;

    ParallaxVMM vmm = vmms[dtID.x];
    Statistics statistics = vmmStatistics[dtID.x];

    // maximum a posteriori step
    // update weights
    // TODO: config for this
    const float weightPrior = 0.1f;
    // TODO IMPORTANT
    uint numSamples = 0;

    for (uint i = 0; i < vmm.numComponents; i++)
    {
        float weight = statistics.sumWeights[i];
        weight = (weightPrior + weight) / (weightPrior * vmm.numComponents + numSamples);
        vmm.weights[i] = weight;
    }
}
