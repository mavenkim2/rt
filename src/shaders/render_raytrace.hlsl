#include "common.hlsli"
#include "bxdf.hlsli"
#include "rt.hlsli"
#include "ray_triangle_intersection.hlsli"
#include "sampling.hlsli"
#include "dense_geometry.hlsli"
#include "../rt/shader_interop/ray_shaderinterop.h"
#include "../rt/shader_interop/hit_shaderinterop.h"
#include "../rt/shader_interop/debug_shaderinterop.h"

RaytracingAccelerationStructure accel : register(t0);
RWTexture2D<float4> image : register(u1);
ConstantBuffer<GPUScene> scene : register(b2);
StructuredBuffer<RTBindingData> rtBindingData : register(t3);
StructuredBuffer<GPUMaterial> materials : register(t4);
ConstantBuffer<ShaderDebugInfo> debugInfo: register(b7);

StructuredBuffer<AABB> aabbs : register(t8);

[[vk::push_constant]] RayPushConstant push;

[numthreads(PATH_TRACE_NUM_THREADS_X, PATH_TRACE_NUM_THREADS_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    // TODO: thread swizzling to get greater coherence, and SOA?
    RNG rng = RNG::Init(RNG::PCG3d(DTid.xyx).zy, push.frameNum);

    // Generate Ray
    float2 sample = rng.Uniform2D();
    const float2 filterRadius = float2(0.5, 0.5);
    float2 filterSample = float2(lerp(-filterRadius.x, filterRadius.x, sample[0]), 
                                 lerp(-filterRadius.y, filterRadius.y, sample[1]));
    filterSample += float2(0.5, 0.5) + float2(DTid.xy);
    float2 pLens = rng.Uniform2D();

    float3 throughput = float3(1, 1, 1);
    float3 radiance = float3(0, 0, 0);

    const int maxDepth = 2;
    int depth = 0;

    float3 pos;
    float3 dir;
    float3 dpdx, dpdy, dddx, dddy;
    GenerateRay(scene, filterSample, pLens, pos, dir, dpdx, dpdy, dddx, dddy);

    while (true)
    {
        RayDesc desc;
        desc.Origin = pos;
        desc.Direction = dir;
        desc.TMin = 0;
        desc.TMax = FLT_MAX;
        
#ifndef USE_PROCEDURAL_CLUSTER_INTERSECTION
        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> query;
        query.TraceRayInline(accel, RAY_FLAG_NONE, 0xff, desc);
        query.Proceed();
#else
        RayQuery<RAY_FLAG_SKIP_TRIANGLES | RAY_FLAG_FORCE_OPAQUE> query;
        query.TraceRayInline(accel, RAY_FLAG_NONE, 0xff, desc);
        float2 bary;

        // TODO: explore design space (e.g. have primitive be a cluster, instead of triangle 
        // in cluster)
        while (query.Proceed()) 
        {
            if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
            {
                uint primitiveIndex = query.CandidatePrimitiveIndex();

                uint blockIndex = primitiveIndex >> MAX_CLUSTER_TRIANGLES_BIT;
                uint triangleIndex = primitiveIndex & (MAX_CLUSTER_TRIANGLES - 1);

                DenseGeometry dg = GetDenseGeometryHeader(blockIndex);
                uint3 vids = dg.DecodeTriangle(triangleIndex, all(DTid.xy == debugInfo.mousePos));

                float3 p0 = dg.DecodePosition(vids[0]);
                float3 p1 = dg.DecodePosition(vids[1]);
                float3 p2 = dg.DecodePosition(vids[2]);

                float tHit;
                float2 tempBary;

                bool result = 
                RayTriangleIntersectionMollerTrumbore(query.CandidateObjectRayOrigin(), 
                                                      query.CandidateObjectRayDirection(), 
                                                      p0, p1, p2,
                                                      tHit, tempBary);

                result &= (query.RayTMin() < tHit && tHit <= query.CommittedRayT());
 
                if (!result) continue;

                bary = tempBary;
                query.CommitProceduralPrimitiveHit(tHit);
            }
        }
#endif
        // TODO: emitter intersection
        if (depth++ >= maxDepth) break;

#ifndef USE_PROCEDURAL_CLUSTER_INTERSECTION
        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            uint bindingDataIndex = query.CommittedInstanceID() + query.CommittedGeometryIndex();
            uint vertexBufferIndex = 3 * bindingDataIndex;
            uint indexBufferIndex = 3 * bindingDataIndex + 1;
            uint normalBufferIndex = 3 * bindingDataIndex + 2;
            uint primID = query.CommittedPrimitiveIndex();

            uint index0 = bindlessUints[NonUniformResourceIndex(indexBufferIndex)][NonUniformResourceIndex(3 * primID + 0)];
            uint index1 = bindlessUints[NonUniformResourceIndex(indexBufferIndex)][NonUniformResourceIndex(3 * primID + 1)];
            uint index2 = bindlessUints[NonUniformResourceIndex(indexBufferIndex)][NonUniformResourceIndex(3 * primID + 2)];

            uint normal0 = bindlessUints[NonUniformResourceIndex(normalBufferIndex)][NonUniformResourceIndex(index0)];
            uint normal1 = bindlessUints[NonUniformResourceIndex(normalBufferIndex)][NonUniformResourceIndex(index1)];
            uint normal2 = bindlessUints[NonUniformResourceIndex(normalBufferIndex)][NonUniformResourceIndex(index2)];

            float3 p0 = bindlessFloat3s[NonUniformResourceIndex(vertexBufferIndex)][NonUniformResourceIndex(index0)];
            float3 p1 = bindlessFloat3s[NonUniformResourceIndex(vertexBufferIndex)][NonUniformResourceIndex(index1)];
            float3 p2 = bindlessFloat3s[NonUniformResourceIndex(vertexBufferIndex)][NonUniformResourceIndex(index2)];

            float3 n0 = DecodeOctahedral(normal0);
            float3 n1 = DecodeOctahedral(normal1);
            float3 n2 = DecodeOctahedral(normal2);

            float2 bary = query.CommittedTriangleBarycentrics();

            float3 gn = normalize(cross(p[0] - p[2], p[1] - p[2]));
#else
        if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
        {
            uint bindingDataIndex = query.CommittedInstanceID() + query.CommittedGeometryIndex();
            uint primitiveIndex = query.CommittedPrimitiveIndex();

            uint blockIndex = primitiveIndex >> MAX_CLUSTER_TRIANGLES_BIT;
            uint triangleIndex = primitiveIndex & (MAX_CLUSTER_TRIANGLES - 1);

            DenseGeometry dg = GetDenseGeometryHeader(blockIndex);
            uint3 vids = dg.DecodeTriangle(triangleIndex);

            float3 p0 = dg.DecodePosition(vids[0]);
            float3 p1 = dg.DecodePosition(vids[1]);
            float3 p2 = dg.DecodePosition(vids[2]);

            float3 gn = normalize(cross(p0 - p2, p1 - p2));

            float3 n0 = dg.DecodeNormal(vids[0]);
            float3 n1 = dg.DecodeNormal(vids[1]);
            float3 n2 = dg.DecodeNormal(vids[2]);

#endif
            float3 n = normalize(n0 + (n1 - n0) * bary[0] + (n2 - n0) * bary[1]);

            // Get material
            RTBindingData bindingData = rtBindingData[NonUniformResourceIndex(bindingDataIndex)];
            GPUMaterial material = materials[NonUniformResourceIndex(bindingData.materialIndex)];
            float eta = material.eta;

            float3 wo = -normalize(query.CommittedObjectRayDirection());

            float u = rng.Uniform();
            float R = FrDielectric(dot(wo, n), eta);

            float T = 1 - R;
            float pr = R, pt = T;

            float3 origin = p0 + (p1 - p0) * bary.x + (p2 - p0) * bary.y;

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
                    radiance = float3(0, 0, 0);
                    break;
                }
            
                throughput /= etap * etap;
            }

            origin = TransformP(query.CommittedObjectToWorld3x4(), origin);
            dir = TransformV(query.CommittedObjectToWorld3x4(), normalize(dir));
            pos = OffsetRayOrigin(origin, gn);
        }
        else 
        {
            float3 d = normalize(mul(scene.lightFromRender, 
                                 float4(query.WorldRayDirection(), 0)).xyz);

            // Equal area sphere to square
            float x = abs(d.x), y = abs(d.y), z = abs(d.z);

            // Compute the radius r
            float r = sqrt(1 - z); // r = sqrt(1-|z|)

            // Compute the argument to atan (detect a=0 to avoid div-by-zero)
            float a = max(x, y), b = min(x, y);
            b = a == 0 ? 0 : b / a;

            // Polynomial approximation of atan(x)*2/pi, x=b
            // Coefficients for 6th degree minimax approximation of atan(x)*2/pi,
            // x=[0,1].
            const float t1 = 0.406758566246788489601959989e-5;
            const float t2 = 0.636226545274016134946890922156;
            const float t3 = 0.61572017898280213493197203466e-2;
            const float t4 = -0.247333733281268944196501420480;
            const float t5 = 0.881770664775316294736387951347e-1;
            const float t6 = 0.419038818029165735901852432784e-1;
            const float t7 = -0.251390972343483509333252996350e-1;
            float phi      = mad(b, mad(b, mad(b, mad(b, mad(b, mad(b, t7, t6), t5), t4), 
                                    t3), t2), t1);

            // Extend phi if the input is in the range 45-90 degrees (u<v)
            if (x < y) phi = 1 - phi;

            // Find (u,v) based on (r,phi)
            float v = phi * r;
            float u = r - v;

            if (d.z < 0)
            {
                // southern hemisphere -> mirror u,v
                swap(u, v);
                u = 1 - u;
                v = 1 - v;
            }

            // Move (u,v) to the correct quadrant based on the signs of (x,y)
            u = copysign(u, d.x);
            v = copysign(v, d.y);
            float2 uv = float2(0.5f * (u + 1), 0.5f * (v + 1));

            uv = uv[0] < 0 ? float2(-uv[0], 1 - uv[1]) : uv;
            uv = uv[0] >= 1 ? float2(2 - uv[0], 1 - uv[1]) : uv;

            uv = uv[1] < 0 ? float2(1 - uv[0], -uv[1]) : uv;
            uv = uv[1] >= 1 ? float2(1 - uv[0], 2 - uv[1]) : uv;

            float3 imageLe = bindlessTextures[push.envMap].SampleLevel(samplerLinearClamp, uv, 0).rgb;
            radiance += throughput * imageLe;
            break;
        }
    }
    image[DTid.xy] = float4(radiance, 1);
}
