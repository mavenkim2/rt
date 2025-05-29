#define ShaderInvocationReorderNV 5383
[[vk::ext_capability(ShaderInvocationReorderNV)]]
[[vk::ext_extension("SPV_NV_shader_invocation_reorder")]]

#include "common.hlsli"
#include "bsdf/bxdf.hlsli"
#include "bsdf/bsdf.hlsli"
#include "rt.hlsli"
#include "ray_triangle_intersection.hlsli"
#include "sampling.hlsli"
#include "payload.hlsli"
#include "dense_geometry.hlsli"
#include "dgf_intersect.hlsli"
#include "tex/virtual_textures.hlsli"
#include "tex/ray_cones.hlsli"
#include "tex/ptex.hlsli"
#include "../rt/shader_interop/as_shaderinterop.h"
#include "../rt/shader_interop/ray_shaderinterop.h"
#include "../rt/shader_interop/hit_shaderinterop.h"
#include "../rt/shader_interop/debug_shaderinterop.h"
#include "ser.hlsli"
#include "wave_intrinsics.hlsli"

#define SER

RaytracingAccelerationStructure accel : register(t0);
RWTexture2D<half4> image : register(u1);
ConstantBuffer<GPUScene> scene : register(b2);
StructuredBuffer<RTBindingData> rtBindingData : register(t3);
StructuredBuffer<GPUMaterial> materials : register(t4);
ConstantBuffer<ShaderDebugInfo> debugInfo: register(b7);

#ifndef USE_PROCEDURAL_CLUSTER_INTERSECTION
StructuredBuffer<ClusterData> clusterData : register(t8);
StructuredBuffer<float3> vertices : register(t9);
ByteAddressBuffer indices : register(t10);
#endif

RWStructuredBuffer<uint> feedbackBuffer : register(u12);

[[vk::push_constant]] RayPushConstant push;

[shader("raygeneration")]
void main()
{
    uint3 id = DispatchRaysIndex();
    uint2 swizzledThreadID = id.xy;

    uint imageWidth, imageHeight;
    image.GetDimensions(imageWidth, imageHeight);
    if (any(swizzledThreadID.xy >= uint2(imageWidth, imageHeight))) return;
    
    RNG rng = RNG::Init(RNG::PCG3d(swizzledThreadID.xyx).zy, push.frameNum);

    // Generate Ray
    float2 sample = rng.Uniform2D();
    const float2 filterRadius = float2(0.5, 0.5);
    float2 filterSample = float2(lerp(-filterRadius.x, filterRadius.x, sample[0]), 
                                 lerp(-filterRadius.y, filterRadius.y, sample[1]));
    filterSample += float2(0.5, 0.5) + float2(swizzledThreadID);
    float2 pLens = rng.Uniform2D();

    float3 throughput = float3(1, 1, 1);
    float3 radiance = float3(0, 0, 0);

    const int maxDepth = 2;
    int depth = 0;

    float3 pos;
    float3 dir;
    float3 dpdx, dpdy, dddx, dddy;
    GenerateRay(scene, filterSample, pLens, pos, dir, dpdx, dpdy, dddx, dddy);

    RayCone rayCone;
    rayCone.width = 0.f;
    rayCone.spreadAngle = atan(2.f * tan(scene.fov / 2.f) / scene.height);

    bool printDebug = all(swizzledThreadID == debugInfo.mousePos);

    uint2 feedbackRequest = ~0u;

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
#ifdef SER
        HitObjectNV hitObject;
        CreateHitObjectNV();
        TraceRayHitObjectNV(hitObject, accel, RAY_FLAG_SKIP_TRIANGLES | RAY_FLAG_FORCE_OPAQUE, 
                            0xff, 0, 1, 0, pos, 0, dir, FLT_MAX, payload);
        ReorderThreadWithHitNV(hitObject, 0, 0);
#else
        RayQuery<RAY_FLAG_SKIP_TRIANGLES | RAY_FLAG_FORCE_OPAQUE> query;
        query.TraceRayInline(accel, RAY_FLAG_NONE, 0xff, desc);
        float2 bary;
        uint hitKind;

        uint numIters = 0;

        while (query.Proceed()) 
        {
            numIters++;
            if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
            {
                uint primitiveIndex = query.CandidatePrimitiveIndex();
                uint instanceID = query.CandidateInstanceID();
                float3 o = query.CandidateObjectRayOrigin();
                float3 d = query.CandidateObjectRayDirection();

                float tHit = 0;
                uint kind = 0;
                float2 tempBary = 0;

                bool result = IntersectCluster(instanceID, primitiveIndex, o, d, query.RayTMin(), 
                                               query.CommittedRayT(), tHit, kind, tempBary, numIters > 1000);

                if (!result) continue;

                bary = tempBary;
                hitKind = kind;
                query.CommitProceduralPrimitiveHit(tHit);
            }
        }
#endif
#endif

#ifdef SER
        if (IsMissNV(hitObject))
#else
        bool isMiss = query.CommittedStatus() != COMMITTED_PROCEDURAL_PRIMITIVE_HIT;
        if (isMiss)
#endif
        {
            float3 d = normalize(mul(scene.lightFromRender, float4(dir, 0)).xyz);

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

        // TODO: emitter intersection
        if (depth++ >= maxDepth)
        {
            break;
        }

#ifndef USE_PROCEDURAL_CLUSTER_INTERSECTION
        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            float3x3 positions = NvRtCommittedTriangleObjectPositions(query);
            float3 p0 = positions[0];
            float3 p1 = positions[1];
            float3 p2 = positions[2];

            uint blockIndex = NvRtGetCommittedClusterID(query);
            uint triangleIndex = query.CommittedPrimitiveIndex();
            float2 bary = query.CommittedTriangleBarycentrics();
#else
#ifdef SER
        if (IsHitNV(hitObject))
        {
            InvokeHitObjectNV(hitObject, payload);

            uint primitiveIndex = GetPrimitiveIndexNV(hitObject);
            uint instanceID = GetInstanceIDNV(hitObject);
            float3x4 objectToWorld = transpose(GetObjectToWorldNV(hitObject));
            float3 objectRayDir = GetObjectRayDirectionNV(hitObject);
            float2 bary = payload.bary;
            float rayT = payload.rayT;
            uint hitKind = 0;//GetHitKindNV(hitObject);
#else
        if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
        {
            uint primitiveIndex = query.CommittedPrimitiveIndex();
            uint instanceID = query.CommittedInstanceID();
            float3x4 objectToWorld = query.CommittedObjectToWorld3x4();
            float rayT = query.CommittedRayT();
            float3 objectRayDir = query.CommittedObjectRayDirection();
#endif
            uint2 blockTriangleIndices = DecodeBlockAndTriangleIndex(primitiveIndex, hitKind);
            uint blockIndex = blockTriangleIndices[0];
            uint triangleIndex = blockTriangleIndices[1];
#endif
            DenseGeometry dg = GetDenseGeometryHeader(instanceID, blockIndex, printDebug);

            uint materialID = dg.DecodeMaterialID(triangleIndex);
            uint2 pageInformation = dg.DecodeFaceIDAndRotateInfo(triangleIndex);
            uint3 vids = dg.DecodeTriangle(triangleIndex);

            uint faceID = pageInformation.x;
            // Rotate to original order
            uint rotate = BitFieldExtractU32(pageInformation.y, 2, 0);
            uint isSecondFace = BitFieldExtractU32(pageInformation.y, 1, 2);

            uint3 oldVids = vids;
            vids = (rotate == 1 ? vids.zxy : (rotate == 2 ? vids.yzx : vids));
            float2 oldBary = bary;
            bary.x = (rotate == 1 ? 1 - oldBary.x - oldBary.y : (rotate == 2 ? oldBary.y : oldBary.x));
            bary.y = (rotate == 1 ? oldBary.x : (rotate == 2 ? 1 - oldBary.x - oldBary.y : oldBary.y));

#ifdef USE_PROCEDURAL_CLUSTER_INTERSECTION
            float3 p0 = dg.DecodePosition(vids[0]);
            float3 p1 = dg.DecodePosition(vids[1]);
            float3 p2 = dg.DecodePosition(vids[2]);
#endif

            if (0)
            {
                printf("%u %u %u type: %u\noldvids: %u %u %u\nvids: %u %u %u\n", instanceID, blockIndex, triangleIndex, rotate,
                        oldVids[0], oldVids[1], oldVids[2], vids[0], vids[1], vids[2]);
            }


            float3 gn = normalize(cross(p0 - p2, p1 - p2));

            float3 n0 = dg.DecodeNormal(vids[0]);
            float3 n1 = dg.DecodeNormal(vids[1]);
            float3 n2 = dg.DecodeNormal(vids[2]);

            
            // Calculate triangle differentials
            // p0 + dpdu * u + dpdv * v = p1

            float2 uv0 = float2(0, 0);
            float2 uv1 = isSecondFace ? float2(1, 1) : float2(1, 0);
            float2 uv2 = isSecondFace ? float2(0, 1) : float2(1, 1);

            float2 duv10 = uv1 - uv0;
            float2 duv20 = uv2 - uv0;
            float det = mad(duv10.x, duv20.y, -duv10.y * duv20.x);

            float3 dpdu, dpdv, dndu, dndv = 0;
            float3 dp10 = p1 - p0;
            float3 dp20 = p2 - p0;
            float3 dn10 = n1 - n0;
            float3 dn20 = n2 - n0;

            if (abs(det) < 1e-9f)
            {
                float2x3 tb = BuildOrthonormalBasis(gn);
                dpdu = tb[0];
                dpdv = tb[1];
            }
            else 
            {
                float invDet = rcp(det);

                dpdu = mad(duv20.y, dp10, -duv10.y * dp20) * invDet;
                dpdv = mad(-duv20.x, dp10, duv10.x * dp20) * invDet;
                
                dndu = mad(duv20.y, dn10, -duv10.y * dn20) * invDet;
                dndv = mad(-duv20.x, dn10, duv10.x * dn20) * invDet;
            }

            float3 n = normalize(n0 + dn10 * bary[0] + dn20 * bary[1]);

            float3 ss = dpdu;
            float3 ts = cross(n, ss);
            if (dot(ts, ts) > 0)
            {
                ss = cross(ts, n);
            }
            else
            {
                float2x3 tb = BuildOrthonormalBasis(n);
                ss = tb[0];
                ts = tb[1];
            }

            ss = normalize(ss);
            ts = cross(n, ss);

            // Get material
            GPUMaterial material = materials[NonUniformResourceIndex(materialID)];

            float3 origin = p0 + dp10 * bary.x + dp20 * bary.y;
            float2 uv = uv0 + duv10 * bary.x + duv20 * bary.y;

            float3 wo = normalize(float3(dot(ss, -objectRayDir), dot(ts, -objectRayDir), dot(n, -objectRayDir)));

            float2 sample = rng.Uniform2D();

            float surfaceSpreadAngle = depth == 1 ? rayCone.CalculatePrimaryHitUnifiedSurfaceSpreadAngle(dir, n, p0, p1, p2, n0, n1, n2) 
                                                  : rayCone.CalculateSecondaryHitSurfaceSpreadAngle(dir, n, p0, p1, p2, n0, n1, n2);

            float filterU = rng.Uniform();

            uint2 virtualPage = ~0u;
            switch (material.type) 
            {
                case GPUMaterialType::Dielectric:
                {
                    dir = SampleDielectric(wo, material.eta, sample, throughput, printDebug);
                }
                break;
                case GPUMaterialType::Diffuse: 
                {
                    // Get base face data
                    Ptex::FaceData faceData = Ptex::GetFaceData(material, faceID);
                    int2 dim = int2(1u << faceData.log2Dim.x, 1u << faceData.log2Dim.y);

                    rayCone.Propagate(surfaceSpreadAngle, rayT);
                    float lambda = rayCone.ComputeTextureLOD(p0, p1, p2, uv0, uv1, uv2, dir, n, dim, printDebug);
                    uint mipLevel = (uint)lambda;

                    float4 reflectance = SampleStochasticCatmullRomBorderless(faceData, material, faceID, uv, mipLevel, filterU, printDebug);
                    dir = SampleDiffuse(reflectance.xyz, wo, sample, throughput, printDebug);

                    if (depth == 1)
                    {
                        float2 newUv = faceData.rotate ? float2(1 - uv.y, uv.x) : uv;
                        uint2 virtualPage = VirtualTexture::CalculateVirtualPage(faceData.faceOffset, newUv, dim, mipLevel);
                        const uint feedbackMipLevel = VirtualTexture::ClampMipLevel(dim, mipLevel);
                        feedbackRequest = PackFeedbackEntry(virtualPage.x, virtualPage.y, material.textureIndex, feedbackMipLevel);
                    }
                }
                break;
            }


            if (dir.z == 0) break;
            origin = TransformP(objectToWorld, origin);
            dir = ss * dir.x + ts * dir.y + n * dir.z;
            dir = normalize(TransformV(objectToWorld, dir));
            pos = OffsetRayOrigin(origin, gn, printDebug);
        }

#if 1
        // Warp based russian roulette
        // https://www.nvidia.com/en-us/on-demand/session/gdc25-gdc1002/
        const float continuationProb = min(1.f, Luminance(throughput));
        const uint activeLaneCount = WaveActiveCountBits(true);
        const uint totalLaneCount = WaveGetLaneCount();
        const float activeLaneRatio = (float)activeLaneCount / (float)totalLaneCount;
        const float activeLaneRatioThreshold = .3f;
        const float groupContinuationProb = continuationProb * saturate(activeLaneRatio / activeLaneRatioThreshold);

        bool russianRoulette = rng.Uniform() >= groupContinuationProb;
#endif
        uint hint = (dir.x >= 0) | ((dir.y >= 0) << 1) | ((dir.z >= 0) << 2);
        //NvReorderThread(russianRoulette, 1);

        if (russianRoulette) break;
        throughput /= groupContinuationProb;
    }
    image[swizzledThreadID] = float4(radiance, 1);

    // Write virtual texture feedback back to main memory
    uint4 mask = WaveMatch(feedbackRequest.x);
    int4 highLanes = (int4)(firstbithigh(mask) | uint4(0, 0x20, 0x40, 0x60));
    uint highLane = (uint)max(max(highLanes.x, highLanes.y), max(highLanes.z, highLanes.w));
    bool isHighLane = all(feedbackRequest != ~0u) && WaveGetLaneIndex() == highLane;

    uint index;
    WaveInterlockedAddScalarTest(feedbackBuffer[0], isHighLane, 2, index);
    if (isHighLane)
    {
        feedbackBuffer[index + 1] = feedbackRequest.x;
        feedbackBuffer[index + 2] = feedbackRequest.y;
    }
}
