#include "bit_twiddling.hlsli"
#include "common.hlsli"
#include "bxdf.hlsli"
#include "payload.hlsli"
#include "rt.hlsli"
#include "dgf_intersect.hlsli"
#include "sampling.hlsli"
#include "../rt/shader_interop/as_shaderinterop.h"
#include "../rt/shader_interop/hit_shaderinterop.h"
#include "../rt/shader_interop/ray_shaderinterop.h"
#include "../rt/shader_interop/debug_shaderinterop.h"

RaytracingAccelerationStructure accel : register(t0);
RWTexture2D<float4> image : register(u1);
ConstantBuffer<GPUScene> scene : register(b2);
StructuredBuffer<RTBindingData> rtBindingData : register(t3);
StructuredBuffer<GPUMaterial> materials : register(t4);
ConstantBuffer<ShaderDebugInfo> debugInfo: register(b7);

[shader("closesthit")]
void main(inout RayPayload payload : SV_RayPayload, BuiltInTriangleIntersectionAttributes attr : SV_IntersectionAttributes) 
{
    payload.objectToWorld = ObjectToWorld3x4();
    payload.objectRayDir = ObjectRayDirection();
    payload.bary = attr.barycentrics;
    payload.rayT = RayTCurrent();
    payload.hitKind = HitKind();
}
