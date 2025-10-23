#include "wavefront_helper.hlsli"
#include "../sampling.hlsli"
#include "../rt.hlsli"
#include "../../rt/shader_interop/ray_shaderinterop.h"
#include "../../rt/shader_interop/wavefront_shaderinterop.h"
#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"

RWStructuredBuffer<WavefrontQueue> queues : register(u0);
ConstantBuffer<WavefrontDescriptors> descriptors : register(b1);
RWStructuredBuffer<PixelInfo> pixelInfos : register(u2);
ConstantBuffer<GPUScene> scene : register(b3);
StructuredBuffer<float> filterCDF : register(t4);
StructuredBuffer<float> filterValues : register(t5);

[[vk::push_constant]] GenerateRayPushConstant push;

[numthreads(64, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    uint imageWidth = push.imageWidth;
    uint imageHeight = push.imageHeight;
    uint startTileIndex = push.startTileIndex;
    const uint tileWidth = 8;
    const uint numTilesX = (imageWidth + tileWidth - 1) / tileWidth;
    const uint numTilesY = (imageHeight + tileWidth - 1) / tileWidth;

    uint tileIndex = startTileIndex + groupID.x;
    uint tileX = tileIndex % numTilesX;
    uint tileY = tileIndex / numTilesX;
    uint pixelInTileX = groupIndex % tileWidth;
    uint pixelInTileY = groupIndex / tileWidth;

    uint2 swizzledThreadID = uint2(tileX * tileWidth + pixelInTileX, tileY * tileWidth + pixelInTileY);

    if (any(swizzledThreadID.xy >= uint2(imageWidth, imageHeight))) return;
    
    RNG rng = RNG::Init(RNG::PCG3d(swizzledThreadID.xyx).zy, push.frameNum);

    // Generate Ray
    float2 sample = rng.Uniform2D();
    const float2 filterRadius = float2(0.5, 0.5);

    // First, sample marginal
    int offsetY, offsetX;
    float pdf = 1;
    float2 sampleP = 0;

    float marginalIntegral = push.filterIntegral;
    float radius = 1.5f;
    int piecewiseConstant2DWidth = 32 * radius;
    float2 minD = -radius;
    float2 maxD = radius;

    for (int i = piecewiseConstant2DWidth - 1; i >= 0; i--)
    {
        if (filterCDF[i] <= sample.y)
        {
            offsetY = i;

            float cdfRange = filterCDF[i + 1] - filterCDF[i];
            float du = (sample.y - filterCDF[i]) * (cdfRange > 0.f ? 1.f / cdfRange : 0.f);
            float t = saturate((i + du) / (float)piecewiseConstant2DWidth);
            sampleP.y = lerp(minD.y, maxD.y, t);

            break;
        }
    }
    
    for (int i = piecewiseConstant2DWidth - 1; i >= 0; i--)
    {
        uint cdfIndex = (piecewiseConstant2DWidth + 1) * (1 + offsetY) + i;
        if (filterCDF[cdfIndex] <= sample.x)
        {
            offsetX = i;
            pdf = abs(filterValues[piecewiseConstant2DWidth * offsetY + offsetX]) / marginalIntegral;

            float cdfRange = filterCDF[cdfIndex + 1] - filterCDF[cdfIndex];
            float du = (sample.x - filterCDF[cdfIndex]) * (cdfRange > 0.f ? 1.f / cdfRange : 0.f);
            float t = saturate((i + du) / (float)piecewiseConstant2DWidth);
            sampleP.x = lerp(minD.x, maxD.x, t);

            break;
        }
    }

    float2 filterSample = sampleP;

    filterSample += float2(0.5, 0.5) + float2(swizzledThreadID);
    float2 pLens = rng.Uniform2D();

    float3 pos;
    float3 dir;
    float3 dpdx, dpdy, dddx, dddy;
    GenerateRay(scene, filterSample, pLens, pos, dir, dpdx, dpdy, dddx, dddy);

    uint writeIndex;
    InterlockedAdd(queues[WAVEFRONT_RAY_QUEUE_INDEX].writeOffset, 1, writeIndex);
    writeIndex %= WAVEFRONT_QUEUE_SIZE;

    uint pixelIndex = dtID.x;

    StoreFloat3(pos, descriptors.rayQueuePosIndex, writeIndex);
    StoreFloat3(dir, descriptors.rayQueueDirIndex, writeIndex);
    StoreUint(pixelIndex, descriptors.rayQueuePixelIndex, writeIndex);

    PixelInfo info;
    info.radiance = 0;
    info.throughput = 1;
    info.pixelLocation_specularBounce = (swizzledThreadID.y << 15u) | swizzledThreadID.x;
    info.rayConeWidth = 0.f;
    info.rayConeSpread = atan(2.f * tan(scene.fov / 2.f) / scene.height);
    info.rngState = rng.State;

    pixelInfos[pixelIndex] = info;
}
