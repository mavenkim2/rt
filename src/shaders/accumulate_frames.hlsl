#include "../rt/shader_interop/as_shaderinterop.h"

Texture2D<float4> frameOutput : register(t0);
//RWStructuredBuffer<half3> accumulate : register(u1);
RWStructuredBuffer<float3> accumulate : register(u1);

Texture2D<float4> albedo : register(t2);
RWStructuredBuffer<float3> accumulatedAlbedo : register(u3);

[[vk::push_constant]] NumPushConstant pc;

[numthreads(8, 8, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint width, height;
    frameOutput.GetDimensions(width, height);
    if (any(dtID.xy >= uint2(width, height))) return;

    uint index = width * dtID.y + dtID.x;
    float4 currentValue = pc.num == 1 ? 0.f : float4(float3(accumulate[index]), 1.f);
    float4 newValue = frameOutput[dtID.xy];
    accumulate[index] = (currentValue + (newValue - currentValue) / float(pc.num)).xyz;

    float4 currentAlbedo = pc.num == 1 ? 0.f : float4(float3(accumulatedAlbedo[index]), 1.f);
    float4 newAlbedo = albedo[dtID.xy];

    accumulatedAlbedo[index] = (currentAlbedo + (newAlbedo - currentAlbedo) / float(pc.num)).xyz;
}
