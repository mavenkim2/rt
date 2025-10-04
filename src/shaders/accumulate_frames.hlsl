#include "../rt/shader_interop/as_shaderinterop.h"

Texture2D<float4> frameOutput : register(t0);
RWTexture2D<half4> accumulate : register(u1);

[[vk::push_constant]] NumPushConstant pc;

[numthreads(8, 8, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint width, height;
    accumulate.GetDimensions(width, height);
    if (any(dtID.xy >= uint2(width, height))) return;

    float4 currentValue = pc.num == 1 ? 0.f : accumulate[dtID.xy];
    float4 newValue = frameOutput[dtID.xy];
    accumulate[dtID.xy] = currentValue + (newValue - currentValue) / float(pc.num);
}
