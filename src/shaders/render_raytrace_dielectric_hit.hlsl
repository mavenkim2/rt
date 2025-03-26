#include "bit_twiddling.hlsli"
#include "common.hlsli"
#include "bxdf.hlsli"
#include "dense_geometry.hlsli"
#include "../rt/shader_interop/as_shaderinterop.h"
#include "../rt/shader_interop/dense_geometry_shaderinterop.h"
#include "../rt/shader_interop/hit_shaderinterop.h"
#include "../rt/shader_interop/ray_shaderinterop.h"
#include "nvapi.hlsli"

RaytracingAccelerationStructure accel : register(t0);
StructuredBuffer<RTBindingData> rtBindingData : register(t3);
StructuredBuffer<GPUMaterial> materials : register(t4);
#if 0
StructuredBuffer<ClusterData> clusterData : register(t7);
StructuredBuffer<float3> vertices : register(t8);
ByteAddressBuffer indices : register(t9);
#endif

[shader("closesthit")]
void main(inout RayPayload payload : SV_RayPayload, BuiltInTriangleIntersectionAttributes attr : SV_IntersectionAttributes) 
{
    uint bindingDataIndex = InstanceID() + GeometryIndex();
    uint primitiveIndex = PrimitiveIndex();

    uint blockIndex = primitiveIndex >> MAX_CLUSTER_TRIANGLES_BIT;
    uint triangleIndex = primitiveIndex & (MAX_CLUSTER_TRIANGLES - 1);

    DenseGeometry dg = GetDenseGeometryHeader(blockIndex);
    uint3 vids = dg.DecodeTriangle(triangleIndex);

    float3 p0 = dg.DecodePosition(vids[0]);
    float3 p1 = dg.DecodePosition(vids[1]);
    float3 p2 = dg.DecodePosition(vids[2]);

    float3 n0 = dg.DecodeNormal(vids[0]);
    float3 n1 = dg.DecodeNormal(vids[1]);
    float3 n2 = dg.DecodeNormal(vids[2]);

    float3 gn = normalize(cross(p0 - p2, p1 - p2));

#if 0
    uint clusterIndex = NvRtGetClusterID();
    ClusterData cData = clusterData[clusterIndex];
    uint ibOffset = cData.indexBufferOffset;

    uint2 offsets = GetAlignedAddressAndBitOffset(ibOffset + 3 * primID, 0);
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

    uint normal0 = bindlessUints[normalBufferIndex][index0];
    uint normal1 = bindlessUints[normalBufferIndex][index1];
    uint normal2 = bindlessUints[normalBufferIndex][index2];

    float3 n0 = DecodeOctahedral(normal0);
    float3 n1 = DecodeOctahedral(normal1);
    float3 n2 = DecodeOctahedral(normal2);

#endif
    float2 bary = attr.barycentrics;
    float3 n = normalize(n0 + (n1 - n0) * bary.x + (n2 - n0) * bary.y);
    float3 p = p0 + (p1 - p0) * bary.x + (p2 - p0) * bary.y;

    // Get material
    RTBindingData bindingData = rtBindingData[0];
    GPUMaterial material = materials[bindingData.materialIndex];
    float eta = material.eta;

    payload.eta = eta;
    payload.intersectPosition = p;
    payload.geometricNormal = gn;
    payload.shadingNormal = n;
}
