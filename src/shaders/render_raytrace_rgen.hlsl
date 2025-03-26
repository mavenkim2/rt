#include "bxdf.hlsli"
#include "common.hlsli"
#include "rt.hlsli"
#include "../rt/shader_interop/ray_shaderinterop.h"
#include "../rt/shader_interop/hit_shaderinterop.h"
#include "../rt/shader_interop/debug_shaderinterop.h"

[[vk::push_constant]] RayPushConstant push;

RaytracingAccelerationStructure accel : register(t0);
RWTexture2D<float4> image : register(u1);
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

        TraceRay(accel, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, desc, payload);
        if (payload.missed || depth++ >= maxDepth) break;

        float3 p = payload.intersectPosition;
        float3 gn = payload.geometricNormal;
        float3 n = payload.shadingNormal;
        float eta = payload.eta;

        //float3 d = WorldRayDirection();
        float3 wo = float3(1, 0, 0);//-normalize(d);

        float u = rng.Uniform();
        float R = FrDielectric(dot(wo, n), eta);

        float T = 1 - R;
        float pr = R, pt = T;

        if (u < pr / (pr + pt))
        {
            dir = Reflect(wo, n);
        }
        else
        {
            float etap;
            bool valid = Refract(wo, n, eta, etap, dir);
            if (!valid)
            {
                payload.radiance = float3(0, 0, 0);
                break;
            }
        
            payload.throughput /= etap * etap;
        }

        float3 origin = TransformP(ObjectToWorld3x4(), p);
        dir = TransformV(ObjectToWorld3x4(), normalize(dir));
        pos = OffsetRayOrigin(origin, gn);

    }

    image[id.xy] = float4(payload.radiance, 1);
}
