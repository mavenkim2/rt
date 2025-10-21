#include "../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<float3> denoisedOutput : register(t0);
RWTexture2D<float4> finalImage : register(u1);

[[vk::push_constant]] NumPushConstant pc;

[numthreads(8, 8, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint width, height;
    finalImage.GetDimensions(width, height);
    if (any(dtID.xy >= uint2(width, height))) return;

    uint index = width * dtID.y + dtID.x;
    float3 toneMapped = denoisedOutput[index];
    toneMapped = toneMapped / (toneMapped + 1);
    float4 r = float4(toneMapped, 1.f);
    //float4 r = denoisedOutput[index];
    finalImage[dtID.xy] = r;
}

