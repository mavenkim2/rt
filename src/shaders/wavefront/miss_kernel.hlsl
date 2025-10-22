#include "wavefront_helper.hlsli"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"
#include "../../rt/shader_interop/wavefront_shaderinterop.h"
#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "../../rt/shader_interop/ray_shaderinterop.h"
#include "../lights/envmap.hlsli"
#include "../bit_twiddling.hlsli"

RWStructuredBuffer<WavefrontQueue> queues : register(u0);
ConstantBuffer<WavefrontDescriptors> descriptors : register(b1);
ConstantBuffer<GPUScene> scene : register(b2);
StructuredBuffer<PixelInfo> pixelInfos : register(t3);
RWTexture2D<float4> albedo : register(u4);
RWStructuredBuffer<float3> normals : register(u5);
RWTexture2D<float4> image : register(u6);

[[vk::push_constant]] RayPushConstant push;

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint start = queues[WAVEFRONT_MISS_QUEUE_INDEX].readOffset;
    uint end = queues[WAVEFRONT_MISS_QUEUE_INDEX].writeOffset;
    uint queueIndex = start + dtID.x;
    if (queueIndex >= end) return;
    queueIndex %= WAVEFRONT_QUEUE_SIZE;

    uint pixelIndex = GetUint(descriptors.missQueuePixelIndex, queueIndex);
    float3 dir = GetFloat3(descriptors.missQueueDirIndex, queueIndex);

    PixelInfo pixelInfo = pixelInfos[pixelIndex];
    float3 radiance = pixelInfo.radiance;
    float3 throughput = pixelInfo.throughput;
    uint2 pixelLoc = uint2(BitFieldExtractU32(pixelInfo.pixelLocation_specularBounce, 15, 0),
                           BitFieldExtractU32(pixelInfo.pixelLocation_specularBounce, 15, 15));
    bool specularBounce = bool(BitFieldExtractU32(pixelInfo.pixelLocation_specularBounce, 1, 30));

    float3 imageLe = EnvMapLe(scene.lightFromRender, push.envMap, dir);
    radiance += throughput * imageLe;

    uint depth = push.depth;

    if (depth == 0 || (depth == 1 && specularBounce))
    {
        uint index = scene.width * pixelLoc.y + pixelLoc.x;
        albedo[pixelLoc] = 0;
        normals[index] = 0;
    }
    image[pixelLoc] = float4(radiance, 1);
}
