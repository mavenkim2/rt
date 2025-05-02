#include "common.hlsli"
#include "rt.hlsli"
#include "sampling.hlsli"
#include "payload.hlsli"
#include "dense_geometry.hlsli"
#include "../rt/shader_interop/ray_shaderinterop.h"
#include "../rt/shader_interop/hit_shaderinterop.h"
#include "../rt/shader_interop/debug_shaderinterop.h"
#include "../rt/shader_interop/dense_geometry_shaderinterop.h"
#include "ray_triangle_intersection.hlsli"
//#include "../rt/nvapi.h"

[[vk::push_constant]] RayPushConstant push;

RaytracingAccelerationStructure accel : register(t0);

RWTexture2D<half4> image : register(u1);

ConstantBuffer<GPUScene> scene : register(b2);
StructuredBuffer<RTBindingData> rtBindingData : register(t3);
StructuredBuffer<GPUMaterial> materials : register(t4);
ConstantBuffer<ShaderDebugInfo> debugInfo: register(b7);

[shader("raygeneration")]
void main() 
{
    uint3 id = DispatchRaysIndex();
    RNG rng = RNG::Init(RNG::PCG3d(id.xyx).zy, push.frameNum);

    // Generate Ray
    float2 sample = rng.Uniform2D();
    const float2 filterRadius = float2(0.5, 0.5);
    float2 filterSample = float2(lerp(-filterRadius.x, filterRadius.x, sample[0]), 
                                 lerp(-filterRadius.y, filterRadius.y, sample[1]));
    filterSample += float2(0.5, 0.5) + float2(id.xy);
    float2 pLens = rng.Uniform2D();

    const int maxDepth = 2;
    int depth = 0;

    float3 pos;
    float3 dir;
    float3 dpdx, dpdy, dddx, dddy;
    GenerateRay(scene, filterSample, pLens, pos, dir, dpdx, dpdy, dddx, dddy);

    RayPayload payload;
    payload.rng = rng;
    payload.radiance = 0;
    payload.throughput = 1;
    payload.missed = false;

    while (true)
    {
        RayDesc desc;
        desc.Origin = pos;
        desc.Direction = dir;
        desc.TMin = 0;
        desc.TMax = FLT_MAX;

        TraceRay(accel, RAY_FLAG_NONE, 0xff, 0, 1, 0, desc, payload);

        bool terminate = payload.missed || depth++ >= maxDepth;
        uint hint = terminate ? 1 : 0;

        //NvReorderThread(hit, hint, 1);
        //NvInvokeHitObject(accel, hit, payload);

        pos = payload.pos;
        dir = payload.dir;

        //NvReorderThread(terminate, 1);
        if (terminate) break;
    }

    image[id.xy] = float4(payload.radiance, 1);
}
