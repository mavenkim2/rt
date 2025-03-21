#include "common.hlsli"
#include "rt.hlsli"

RaytracingAccelerationStructure accel : register(t0);
RWTexture2D<float4> image : register(u1);
ConstantBuffer<GPUScene> scene : register(b2);

[shader("raygeneration")]
void main() 
{
    uint3 id = DispatchRaysIndex();
    // Seed based on frame number and pixel position
    uint seed = 0;//Hash(id);

    // Generate Ray
    float2 sample = Get2Random(seed);
    const float2 filterRadius = float2(0.5, 0.5);
    float2 filterSample = float2(lerp(-filterRadius.x, filterRadius.x, sample[0]), 
                                 lerp(-filterRadius.y, filterRadius.y, sample[1]));
    filterSample += float2(0.5, 0.5) + float2(id.xy);
    float2 pLens = Get2Random(seed);

    RayDesc desc;
    desc.TMin = 0;
    desc.TMax = FLT_MAX;
    RayPayload payload;
    payload.throughput = 1;
    GenerateRay(scene, filterSample, pLens, desc.Origin, desc.Direction,
                payload.pxOffset, payload.pyOffset, payload.dxOffset, payload.dyOffset);

    TraceRay(accel, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, desc, payload);

    image[id.xy] = float4(payload.radiance, 1);
}
