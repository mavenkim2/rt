#include "../rt/shader_interop/gpu_scene_shaderinterop.h"

Texture2D<float> depthBuffer : register(t0);
RWTexture2D<float2> motionVectors : register(u1);
ConstantBuffer<GPUScene> thisScene : register(b2);
ConstantBuffer<GPUScene> lastScene : register(b3);

[numthreads(8, 8, 1)]
void main(uint3 dtID: SV_DispatchThreadID)
{
    uint2 pixel = dtID.xy;
    uint width, height;
    depthBuffer.GetDimensions(width, height);
    if (any(pixel >= uint2(width, height))) return;

    float depth = depthBuffer[pixel]; 
    float2 uv = (float2(pixel) + 0.5f) / float2(width, height);

    float4 clipPos = float4(uv.x * 2.f - 1.f, 1.f - uv.y * 2.f, depth, 1.f);
    float4 prevClipPos = mul(thisScene.prevClipFromClip, clipPos);
    prevClipPos.xy /= prevClipPos.w;

    float2 velocity = prevClipPos.xy - clipPos.xy;
    velocity *= float2(0.5f, -0.5f);
    velocity = float2(abs(velocity.x) < 1e-7f ? 0.f : velocity.x, abs(velocity.y) < 1e-7f ? 0.f : velocity.y);

    motionVectors[pixel] = velocity;
}
