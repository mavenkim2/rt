#include "../rt/shader_interop/gpu_scene_shaderinterop.h"

Texture2D<float> depthBuffer : register(t0);
RWTexture2D<half2> motionVectors : register(u1);
ConstantBuffer<GPUScene> thisScene : register(b2);
ConstantBuffer<GPUScene> lastScene : register(b2);

[numthreads(8, 8, 1)]
void main(uint3 dtID: SV_DispatchThreadID)
{
    uint2 pixel = dtID.xy;
    uint width, height;
    depthBuffer.GetDimensions(width, height);
    if (any(pixel >= uint2(width, height))) return;

    float depth = depthBuffer[pixel];

    float4 viewPos = mul(thisScene.cameraFromRaster, float4(pixel, 0.f, 1.f));
    viewPos /= viewPos.w;
    float3 worldPos = mul(thisScene.renderFromCamera, viewPos);

    float4 lastNdcPos = mul(lastScene.clipFromRender, float4(worldPos, 1.f));
    lastNdcPos /= lastNdcPos.w;

    float2 uvPrev = float2((lastNdcPos.x + 1.f) * 0.5f, (1.f - lastNdcPos.y) * 0.5f);
    float2 uv = pixel / float2(width, height);

    float2 velocity = (uvPrev - float2(lastScene.jitterX, lastScene.jitterY)) - (uv - float2(thisScene.jitterX, thisScene.jitterY));
    motionVectors[pixel] = velocity;
}
