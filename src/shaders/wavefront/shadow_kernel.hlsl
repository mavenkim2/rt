#include "wavefront_helper.hlsli"
#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "../../rt/shader_interop/wavefront_shaderinterop.h"
#include "../../rt/shader_interop/ray_shaderinterop.h"
#include "../lights/envmap.hlsli"
#include "../lights_temp.hlsli"

RaytracingAccelerationStructure accel : register(t0);
ConstantBuffer<WavefrontDescriptors> descriptors : register(b1);
RWStructuredBuffer<PixelInfo> pixelInfos : register(u2);
ConstantBuffer<GPUScene> scene : register(b3);

[[vk::push_constant]] RayPushConstant push;

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    float3 origin;
    float3 dir;
    float tMax;
    uint lightSampleIndex_deltaLight;
    float3 bsdfVal;
    float lightPdf;
    uint pixelIndex;

    uint lightSampleIndex = lightSampleIndex_deltaLight & 0x7fffffffu;
    bool deltaLight = lightSampleIndex_deltaLight >> 31u;

    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> occludedQuery;
    RayDesc occludedDesc;
    occludedDesc.Origin = origin;
    occludedDesc.Direction = dir;
    occludedDesc.TMin = 0.f;
    occludedDesc.TMax = tMax;
    occludedQuery.TraceRayInline(accel, RAY_FLAG_NONE, 0xff, occludedDesc);
    
    occludedQuery.Proceed();
    
    if (occludedQuery.CommittedStatus() == COMMITTED_NOTHING)
    {
        float3 L = deltaLight ? 
                    EnvMapLe(scene.lightFromRender, push.envMap, dir)
                    : areaLightColors[lightSampleIndex].xyz;
        
        float3 r = pixelInfos[pixelIndex].throughput * bsdfVal * L / lightPdf;
        pixelInfos[pixelIndex].radiance += r;
    }
}
