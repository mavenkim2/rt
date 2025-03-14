#include "common.hlsli"
#include "bxdf.hlsli"
#include "../rt/shader_interop/hit_shaderinterop.h"
#include "../rt/shader_interop/ray_shaderinterop.h"

RaytracingAccelerationStructure accel : register(t0);
StructuredBuffer<RTBindingData> rtBindingData : register(t3);
StructuredBuffer<GPUMaterial> materials : register(t4);

[shader("closesthit")]
void main(inout RayPayload payload, BuiltInTriangleIntersectionAttributes attr) 
{
    uint bindingDataIndex = InstanceID() + GeometryIndex();
    uint vertexBufferIndex = 3 * bindingDataIndex;
    uint indexBufferIndex = 3 * bindingDataIndex + 1;
    uint normalBufferIndex = 3 * bindingDataIndex + 2;
    uint primID = PrimitiveIndex();

    uint index0 = bindlessUints[indexBufferIndex][3 * primID + 0];
    uint index1 = bindlessUints[indexBufferIndex][3 * primID + 1];
    uint index2 = bindlessUints[indexBufferIndex][3 * primID + 2];

    uint normal0 = bindlessUints[normalBufferIndex][index0];
    uint normal1 = bindlessUints[normalBufferIndex][index1];
    uint normal2 = bindlessUints[normalBufferIndex][index2];

#if 1
    float3 n0 = DecodeOctahedral(normal0);
    float3 n1 = DecodeOctahedral(normal1);
    float3 n2 = DecodeOctahedral(normal2);

    float3 n = normalize(n0 + (n1 - n0) * attr.barycentrics[0] + (n2 - n0) * attr.barycentrics[1]);

    // Get material
    RTBindingData bindingData = rtBindingData[bindingDataIndex];
    GPUMaterial material = materials[bindingData.materialIndex];
    float eta = material.eta;

    float3 wo = -normalize(WorldRayDirection());

    float u = 0.5;//GetRandom();
    float R = FrDielectric(dot(wo, n), eta);

    float T = 1 - R;
    float pr = R, pt = T;

    payload.radiance = float3(R, R, R);
    
    RayDesc newRay;
    newRay.Origin = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    newRay.Direction = normalize(Reflect(wo, n));

    printf("%f %f %f\n", newRay.Direction.x, newRay.Direction.y, newRay.Direction.z);
   //if (u < pr / (pr + pt))
   //{
   //}
   //else
   //{
   //    float etap;
   //    bool valid = Refract(wo, n, eta, etap, newRay.Direction);
   //    if (!valid)
   //    {
   //        payload.radiance = float3(0, 0, 0);
   //        return;
   //    }
   //
   //    payload.throughput /= eta * eta;
   //}

    newRay.TMin = 0.01;
    newRay.TMax = FLT_MAX;
    //TraceRay(accel, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, newRay, payload);
#endif
}
