#ifndef PATH_GUIDING_HLSLI_
#define PATH_GUIDING_HLSLI_

#include "../common.hlsli"
#include "../../rt/shader_interop/path_guiding_shaderinterop.h"

float CalculateVMFNormalization(float kappa)
{
    float eMinus2Kappa = exp(-2.f * kappa);
    float norm = kappa / (2.f * PI * (1.f - eMinus2Kappa));
    norm = kappa > 0.f ? norm : 1.f / (4 * PI);
    return norm;
}

float KappaToMeanCosine(float kappa)
{
    float meanCosine = 1.f / tanh(kappa) - 1.f / kappa;
    return kappa > 0.f ? meanCosine : 0.f;
}

float MeanCosineToKappa(float meanCosine)
{
    const float meanCosine2 = meanCosine * meanCosine;
    return (meanCosine * 3.f - meanCosine * meanCosine2) / (1.f - meanCosine2);
}

#endif
