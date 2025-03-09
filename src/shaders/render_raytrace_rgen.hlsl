#include "common.hlsli"

ConstantBuffer<GPUScene> scene;
RaytracingAccelerationStructure accel : register(t0);

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
    scene.GenerateRay(filterSample, pLens, desc, payload);

    TraceRay(accel, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 1, 0, desc, payload);
}
