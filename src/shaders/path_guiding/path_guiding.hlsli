#ifndef PATH_GUIDING_HLSLI_
#define PATH_GUIDING_HLSLI_

#include "../common.hlsli"
#include "../../rt/shader_interop/path_guiding_shaderinterop.h"

struct VMM
{
    float kappas[MAX_COMPONENTS];
    float3 directions[MAX_COMPONENTS];
    float weights[MAX_COMPONENTS];

    uint numComponents;
};

struct Statistics 
{
    float weightedLogLikelihood;
    float3 sumWeightedDirections[MAX_COMPONENTS];
    float sumWeights[MAX_COMPONENTS];
};

float CalculateVMFNormalization(float kappa)
{
    float eMinus2Kappa = exp(-2.f * kappa);
    float norm = kappa / (2.f * PI * (1.f - eMinus2Kappa));
    norm = kappa > 0.f ? norm : 1.f / (4 * PI);
    return norm;
}

#endif
