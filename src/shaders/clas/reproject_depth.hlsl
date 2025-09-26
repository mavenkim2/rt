#include "../common.hlsli"
#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"

Texture2D<float> lastDepthBuffer : register(t0);
RWTexture2D<float> newDepthBuffer : register(u1);

ConstantBuffer<GPUScene> thisScene : register(b2);

[numthreads(8, 8, 1)]
void main(uint3 dtID: SV_DispatchThreadID)
{
    uint2 pixel = dtID.xy;
    uint width, height;
    newDepthBuffer.GetDimensions(width, height);
    if (any(pixel >= uint2(width, height))) return;

    float2 uv = (float2(pixel) + 0.5f) / float2(width, height);

    float4 clipPos = float4(uv.x * 2.f - 1.f, 1.f - uv.y * 2.f, 1.f, 1.f);
    float4 prevClipPos = mul(thisScene.prevClipFromClip, clipPos);

    prevClipPos.xyz /= prevClipPos.w;

    // Sample depth
    float2 prevUv = prevClipPos.xy * float2(0.5f, -0.5f) + 0.5f;
    bool outside = any(prevUv < 0.f) || any(prevUv > 1.f);
    prevUv = clamp(prevUv, 0.f, 1.f);

    float depth = lastDepthBuffer.SampleLevel(samplerNearestClamp, prevUv, 0).r;

    // Reproject depth to current frame
    float4 lastClipPos = float4(prevClipPos.x, prevClipPos.y, depth, 1.f);
    float4 newClipPos = mul(thisScene.clipFromPrevClip, lastClipPos);
    float newDepth = outside ? 0.f : newClipPos.z / newClipPos.w;

    newDepthBuffer[pixel] = newDepth;
}
