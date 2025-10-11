#include "../rt/shader_interop/as_shaderinterop.h"

Texture2D<float4> frameOutput : register(t0);
RWTexture2D<float4> accumulate : register(u1);

Texture2D<float> weights : register(t2);
RWTexture2D<float> totalWeights : register(u3);

[[vk::push_constant]] NumPushConstant pc;

[numthreads(8, 8, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint width, height;
    accumulate.GetDimensions(width, height);
    if (any(dtID.xy >= uint2(width, height))) return;

    float4 currentValue = pc.num == 1 ? 0.f : accumulate[dtID.xy];
    float4 newValue = frameOutput[dtID.xy];

    float totalWeight = pc.num == 1 ? 0.f : totalWeights[dtID.xy];
    float weight = weights[dtID.xy];
    accumulate[dtID.xy] = currentValue + (newValue - currentValue) / float(pc.num);
    //accumulate[dtID.xy] = currentValue + weight * (newValue - currentValue) / (totalWeight + weight);
    //totalWeights[dtID.xy] = totalWeight + weight;
}
