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
    uint bindingDataIndex = InstanceID() + GeometryIndex();
    uint primitiveIndex = PrimitiveIndex();

    uint2 blockTriangleIndices = DecodeBlockAndTriangleIndex(primitiveIndex, HitKind());
    uint blockIndex = blockTriangleIndices[0];
    uint triangleIndex = blockTriangleIndices[1];

    DenseGeometry dg = GetDenseGeometryHeader(blockIndex);
    uint3 vids = dg.DecodeTriangle(triangleIndex);

    float3 p0 = dg.DecodePosition(vids[0]);
    float3 p1 = dg.DecodePosition(vids[1]);
    float3 p2 = dg.DecodePosition(vids[2]);

    float3 n0 = dg.DecodeNormal(vids[0]);
    float3 n1 = dg.DecodeNormal(vids[1]);
    float3 n2 = dg.DecodeNormal(vids[2]);

    float3 gn = normalize(cross(p0 - p2, p1 - p2));

    float2 bary = attr.barycentrics;
    float3 n = normalize(n0 + (n1 - n0) * bary.x + (n2 - n0) * bary.y);

    // Get material
    RTBindingData bindingData = rtBindingData[0];
    GPUMaterial material = materials[bindingData.materialIndex];
    float eta = material.eta;

    float3 origin = p0 + (p1 - p0) * bary.x + (p2 - p0) * bary.y;
    origin = TransformP(ObjectToWorld3x4(), origin);
    float3 wo = -normalize(ObjectRayDirection());

    RNG rng = payload.rng;
    float2 u = rng.Uniform2D();

    float3 dir    = SampleCosineHemisphere(u);
    dir.z         = wo.z < 0 ? -wi.z : wi.z;
    LaneNF32 pdf = CosineHemispherePDF(abs(wi.z));

    payload.rng = rng;
    payload.throughput = R * InvPi * rcp(pdf);

    payload.rng = rng;
    payload.dir = TransformV(ObjectToWorld3x4(), normalize(dir));
    payload.pos = OffsetRayOrigin(origin, gn);
    payload.throughput = throughput;
}
