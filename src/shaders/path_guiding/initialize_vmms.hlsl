#include "path_guiding.hlsli"

RWStructuredBuffer<VMM> vmms : register(u0);

struct Num 
{
    uint num;
};

[[vk::push_constant]] Num pc;

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    if (dtID.x >= pc.num) return;
    const float gr = 1.618033988749895f;
    const uint numComponents = MAX_COMPONENTS / 2;

    VMM vmm;
    vmm.numComponents = numComponents;

    const float weight = 1.f / float(numComponents);
    const float kappa = 5.f;
    uint l = numComponents - 1;
    for (uint n = 0; n < MAX_COMPONENTS; n++)
    {
        float3 uniformDirection;
        if (n < l + 1)
        {
            float phi = 2.0f * PI * ((float)n / gr);
            float z = 1.0f - ((2.0f * n + 1.0f) / float(l + 1));
            float sinTheta = sqrt(1.f - max(z * z, 1.f));

            // cos(theta) = z
            // sin(theta) = sin(arccos(z)) = sqrt(1 - z^2)
            float3 mu = float3(sinTheta * cos(phi), sinTheta * sin(phi), z);
            uniformDirection = mu;
        }
        else
        {
            uniformDirection = float3(0, 0, 1);
        }

        vmm.directions[n] = uniformDirection;
        if (n < numComponents)
        {
            vmm.kappas[n] = kappa;
            vmm.weights[n] = weight;
        }
        else
        {
            vmm.kappas[n] = 0.0f;
            vmm.weights[n] = 0.0f;
            //vmm.normalizations[i][j] = ONE_OVER_FOUR_PI;
            //vmm.eMinus2Kappa[i][j] = 1.0f;
            //vmm._meanCosines[i][j] = 0.0f;
        }
    }

    vmms[dtID.x] = vmm;
}
