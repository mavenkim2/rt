#include "bit_twiddling.hlsli"
#include "common.hlsli"
#include "bxdf.hlsli"
#include "payload.hlsli"
#include "rt.hlsli"
#include "dgf_intersect.hlsli"
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

#ifndef USE_PROCEDURAL_CLUSTER_INTERSECTION
#define NV_SHADER_EXTN_SLOT u999
#include "../third_party/nvapi/nvHLSLExtns.h"
StructuredBuffer<ClusterData> clusterData : register(t8);
StructuredBuffer<float3> vertices : register(t9);
ByteAddressBuffer indices : register(t10);
#endif

[shader("closesthit")]
void main(inout RayPayload payload : SV_RayPayload, BuiltInTriangleIntersectionAttributes attr : SV_IntersectionAttributes) 
{
#ifdef USE_PROCEDURAL_CLUSTER_INTERSECTION 
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
#else
    //uint clusterID = NvRtGetClusterID();
    float3x3 positions = NvRtTriangleObjectPositions();
    float3 p0 = positions[0];
    float3 p1 = positions[1];
    float3 p2 = positions[2];

    uint bindingDataIndex = InstanceID() + GeometryIndex();

#if 0
    uint primitiveIndex = PrimitiveIndex();
    uint clusterIndex = NvRtGetClusterID();
    ClusterData cData = clusterData[clusterIndex];
    uint ibOffset = cData.indexBufferOffset;

    uint2 offsets = GetAlignedAddressAndBitOffset(ibOffset + 3 * primitiveIndex, 0);
    uint2 indexData = indices.Load2(offsets[0]);
    uint packedIndices = BitAlignU32(indexData.y, indexData.x, offsets[1]);

    uint3 indices;
    indices[0] = BitFieldExtractU32(packedIndices, 8, 0);
    indices[1] = BitFieldExtractU32(packedIndices, 8, 8);
    indices[2] = BitFieldExtractU32(packedIndices, 8, 16);

    indices += cData.vertexBufferOffset;

    float3 p0 = vertices[indices[0]];
    float3 p1 = vertices[indices[1]];
    float3 p2 = vertices[indices[2]];

   //uint normal0 = bindlessUints[normalBufferIndex][index0];
   //uint normal1 = bindlessUints[normalBufferIndex][index1];
   //uint normal2 = bindlessUints[normalBufferIndex][index2];
#endif

    float3 gn = normalize(cross(p0 - p2, p1 - p2));
    float3 n0 = gn;
    float3 n1 = gn;
    float3 n2 = gn;
#endif
    float2 bary = attr.barycentrics;
    float3 n = normalize(n0 + (n1 - n0) * bary.x + (n2 - n0) * bary.y);

    // Get material
    RTBindingData bindingData = rtBindingData[0];
    GPUMaterial material = materials[bindingData.materialIndex];
    float eta = material.eta;

    float3 wo = -normalize(ObjectRayDirection());

    RNG rng = payload.rng;
    float u = rng.Uniform();
    float R = FrDielectric(dot(wo, n), eta);

    float T = 1 - R;
    float pr = R, pt = T;

    float3 origin = p0 + (p1 - p0) * bary.x + (p2 - p0) * bary.y;

    float3 throughput = payload.throughput;

    float3 dir;
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
            throughput = 0;
        }
        else 
        {
            throughput /= etap * etap;
        }
    }

    origin = TransformP(ObjectToWorld3x4(), origin);

    payload.rng = rng;
    payload.dir = TransformV(ObjectToWorld3x4(), normalize(dir));
    payload.pos = OffsetRayOrigin(origin, gn);
    payload.throughput = throughput;
}
