#ifndef PATH_GUIDING_HLSLI_
#define PATH_GUIDING_HLSLI_

#include "../../rt/shader_interop/path_guiding_shaderinterop.h"

struct PathGuidingSample
{
    float3 pos;
    float3 dir;
    float3 radiance;
    float pdf;
    float weight;

    uint vmmIndex;
};

// TODO: jagged array?
struct ParallaxVMM
{
    float kappas[MAX_COMPONENTS];
    float normalizations[MAX_COMPONENTS];
    float directions[MAX_COMPONENTS];
    float weights[MAX_COMPONENTS];

    uint numComponents;
};

struct Statistics 
{
    float weightedLogLikelihood;
    float3 sumWeightedDirections[MAX_COMPONENTS];
    float sumWeights[MAX_COMPONENTS];
};

#endif
