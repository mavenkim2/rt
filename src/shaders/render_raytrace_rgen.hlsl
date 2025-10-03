#define ShaderInvocationReorderNV 5383
#define RayTracingClusterAccelerationStructureNV 5437

[[vk::ext_capability(ShaderInvocationReorderNV)]]
[[vk::ext_extension("SPV_NV_shader_invocation_reorder")]]
[[vk::ext_capability(RayTracingClusterAccelerationStructureNV)]]
[[vk::ext_extension("SPV_NV_cluster_acceleration_structure")]]

#include "common.hlsli"
#include "hit.hlsli"
#include "bsdf/bxdf.hlsli"
#include "bsdf/bsdf.hlsli"
#include "bsdf/disney_bsdf.hlsli"
#include "rt.hlsli"
#include "ray_triangle_intersection.hlsli"
#include "sampling.hlsli"
#include "payload.hlsli"
#include "dense_geometry.hlsli"
#include "tex/virtual_textures.hlsli"
#include "tex/ray_cones.hlsli"
#include "tex/ptex.hlsli"
#include "../rt/shader_interop/as_shaderinterop.h"
#include "../rt/shader_interop/ray_shaderinterop.h"
#include "../rt/shader_interop/hit_shaderinterop.h"
#include "../rt/shader_interop/debug_shaderinterop.h"
#include "nvidia/ser.hlsli"
#include "nvidia/clas.hlsli"
#include "wave_intrinsics.hlsli"
#include "voxel.hlsli"

RaytracingAccelerationStructure accel : register(t0);
RWTexture2D<half4> image : register(u1);
ConstantBuffer<GPUScene> scene : register(b2);
StructuredBuffer<GPUMaterial> materials : register(t4);
ConstantBuffer<ShaderDebugInfo> debugInfo: register(b7);

//RWStructuredBuffer<uint> feedbackBuffer : register(u12);
//StructuredBuffer<uint> clusterLookupTable : register(t13);
StructuredBuffer<GPUInstance> gpuInstances : register(t14);
//RWStructuredBuffer<uint> proxyCounts : register(u15);
StructuredBuffer<GPUTruncatedEllipsoid> truncatedEllipsoids : register(t16);
StructuredBuffer<GPUTransform> instanceTransforms : register(t17);
StructuredBuffer<PartitionInfo> partitionInfos : register(t18);
StructuredBuffer<uint> partitionResourceIDs : register(t19);
StructuredBuffer<Resource> resources : register(t20);

//RWStructuredBuffer<float2> debugBuffer : register(u19);
//RWStructuredBuffer<uint> globals : register(u20);

RWTexture2D<float> depthBuffer : register(u21);
RWTexture2D<float4> normalRougnessBuffer : register(u22);
RWTexture2D<float4> diffuseAlbedo : register(u23);
RWTexture2D<float4> specularAlbedo : register(u24);
RWTexture2D<float> specularHitDistance : register(u25);

[[vk::push_constant]] RayPushConstant push;

void IntersectAABB(float3 boundsMin, float3 boundsMax, float3 o, float3 invDir, out float tEntry, out float tLeave)
{
    float3 tIntersectMin = (boundsMin - o) * invDir;
    float3 tIntersectMax = (boundsMax - o) * invDir;

    float3 tMin = min(tIntersectMin, tIntersectMax);
    float3 tMax = max(tIntersectMin, tIntersectMax);

    tEntry = max(tMin.x, max(tMin.y, tMin.z));
    tLeave = min(tMax.x, min(tMax.y, tMax.z));
}

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
    //float2 sample = rng.Uniform2D();
    const float2 filterRadius = float2(0.5, 0.5);
    float2 filterSample = float2(0, 0);
    //float2(lerp(-filterRadius.x, filterRadius.x, sample[0]), lerp(-filterRadius.y, filterRadius.y, sample[1]));
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
        
        RayQuery<RAY_FLAG_NONE | RAY_FLAG_FORCE_OPAQUE> query;
        query.TraceRayInline(accel, RAY_FLAG_NONE, 0xff, desc);

        uint hitKind = 0;

        float hitT = FLT_MAX;

        while (query.Proceed())
        {
            if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
            {
                uint instanceID = query.CandidateInstanceID();
                GPUInstance instance = gpuInstances[instanceID];
                uint resourceID = instance.resourceID;

#if 0
                InterlockedAdd(proxyCounts[gpuInstances[instanceID].groupIndex], 1);
                if (resourceID == ~0u)
                {
                    continue;
                }
#endif

                if (instance.flags & GPU_INSTANCE_FLAG_MERGED)
                {
                    uint primIndex = query.CandidatePrimitiveIndex();
                    PartitionInfo info = partitionInfos[instance.partitionIndex];
                    resourceID = partitionResourceIDs[info.transformOffset + primIndex];

                    GPUTruncatedEllipsoid ellipsoid = truncatedEllipsoids[resourceID];
                    float3x4 worldFromObject = ConvertGPUMatrix(instanceTransforms[info.transformOffset + primIndex], info.base, info.scale);
                    float3x4 objectFromWorld = Inverse(worldFromObject);
                    float3 rayPos = mul(ellipsoid.transform, float4(mul(objectFromWorld, float4(query.WorldRayOrigin(), 1.f)), 1.f));
                    float3 rayDir = mul(ellipsoid.transform, float4(mul(objectFromWorld, float4(query.WorldRayDirection(), 0.f)), 0.f));
                    rayPos -= ellipsoid.sphere.xyz;

                    // Ray sphere test
                    float a = dot(rayDir, rayDir);
                    float b = 2.f * dot(rayDir, rayPos);
                    float c = dot(rayPos, rayPos) - ellipsoid.sphere.w * ellipsoid.sphere.w;
                    float l = length(rayPos - b / (2.f * a) * rayDir);
                    float discrim = 4 * a * (ellipsoid.sphere.w + l) * (ellipsoid.sphere.w - l);

#if 0
                    if (depth == 1)
                    {
                        uint debugIndex;
                        InterlockedAdd(globals[GLOBALS_DEBUG], 1, debugIndex);
                        if (debugIndex < 1u << 21u)
                        {
                            debugBuffer[debugIndex] = float2(objectFromWorld[0][3], objectFromWorld[1][3]);
                        }
                    }
#endif

                    if (discrim < 0.f)
                    {
                        continue;
                    }
                    discrim = sqrt(discrim);
                    float q = -.5f * (b < 0.f ? b - discrim : b + discrim);
                    float t0 = q / a;
                    float t1 = c / q;

                    if (t0 > t1)
                    {
                        float temp = t0;
                        t0 = t1;
                        t1 = temp;
                    }
                    if (t0 > hitT || t1 <= 0.f)
                    {
                        continue;
                    }

                    float tHit = t0;
                    if (tHit <= 0.f)
                    {
                        tHit = t1;
                        if (t1 > hitT) continue;
                    }
                    hitT = tHit;
                    query.CommitProceduralPrimitiveHit(tHit);

#if 1
                    float tEntry, tLeave;
                    IntersectAABB(ellipsoid.boundsMin - ellipsoid.sphere.xyz, ellipsoid.boundsMax - ellipsoid.sphere.xyz, rayPos, rcp(rayDir), tEntry, tLeave);
                    tHit = max(tHit, tEntry);
                    tLeave = min(t1, tLeave);
                    if (tHit <= tLeave)
                    {
                        hitT = tHit;
                        query.CommitProceduralPrimitiveHit(tHit);
                    }
#endif
                    continue;
                }
            }
        }

        if (query.CommittedStatus() == COMMITTED_NOTHING)
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

            if (depth == 0)
            {
                depthBuffer[swizzledThreadID] = 0.f;
                specularAlbedo[swizzledThreadID] = float4(0.5, 0.5, 0.5, 1.f);
                normalRougnessBuffer[swizzledThreadID] = 0.f;
                diffuseAlbedo[swizzledThreadID] = 0.f;
                specularHitDistance[swizzledThreadID] = 0.f;
            }
            break;
        }

        // TODO: emitter intersection
        if (depth++ >= maxDepth)
        {
            break;
        }

        uint materialID = 0;
        HitInfo hitInfo = (HitInfo)0;
        float3 objectRayDir = query.CommittedObjectRayDirection();
        uint instanceID = query.CommittedInstanceID();
        float3x4 objectToWorld = query.CommittedObjectToWorld3x4();
        float rayT = query.CommittedRayT();

        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            uint clusterID = GetClusterIDNV(query, RayQueryCommittedIntersectionKHR);
            uint pageIndex = GetPageIndexFromClusterID(clusterID); 
            uint clusterIndex = GetClusterIndexFromClusterID(clusterID);

            uint triangleIndex = query.CommittedPrimitiveIndex();
            float2 bary = query.CommittedTriangleBarycentrics();
            uint baseAddress = resources[instanceID].baseAddress;
            DenseGeometry dg = GetDenseGeometryHeader2(clusterID, baseAddress);

            uint materialID = dg.DecodeMaterialID(triangleIndex) + 1;
            uint3 vids = dg.DecodeTriangle(triangleIndex);

#if 0
            uint2 pageInformation = dg.DecodeFaceIDAndRotateInfo(triangleIndex);
            uint faceID = pageInformation.x;
            // Rotate to original order
            uint rotate = BitFieldExtractU32(pageInformation.y, 2, 0);
            uint isSecondFace = BitFieldExtractU32(pageInformation.y, 1, 2);

            uint3 oldVids = vids;
            vids = (rotate == 1 ? vids.zxy : (rotate == 2 ? vids.yzx : vids));
            float2 oldBary = bary;
            bary.x = (rotate == 1 ? 1 - oldBary.x - oldBary.y : (rotate == 2 ? oldBary.y : oldBary.x));
            bary.y = (rotate == 1 ? oldBary.x : (rotate == 2 ? 1 - oldBary.x - oldBary.y : oldBary.y));
#endif

            float3 p0 = dg.DecodePosition(vids[0]);
            float3 p1 = dg.DecodePosition(vids[1]);
            float3 p2 = dg.DecodePosition(vids[2]);

            float3 gn = normalize(cross(p0 - p2, p1 - p2));
            // TODO
            //if (dot(gn, query.WorldRayDirection()) > 0.f)
            //{
                //gn = -gn;
            //}

            float3 n0, n1, n2;
            if (dg.HasNormals())
            {
                n0 = dg.DecodeNormal(vids[0]);
                n1 = dg.DecodeNormal(vids[1]);
                n2 = dg.DecodeNormal(vids[2]);
            }
            //else 
            {
                n0 = gn;
                n1 = gn;
                n2 = gn;
            }

            // Calculate triangle differentials
            // p0 + dpdu * u + dpdv * v = p1

            float2 uv0 = float2(0, 0);
            float2 uv1 = float2(0, 1);//isSecondFace ? float2(1, 1) : float2(1, 0);
            float2 uv2 = float2(1, 1);//isSecondFace ? float2(0, 1) : float2(1, 1);

            hitInfo = CalculateTriangleHitInfo(p0, p1, p2, n0, n1, n2, gn, uv0, uv1, uv2, bary);

        }
        else if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
        {
            uint instanceID = query.CommittedInstanceID();

            hitInfo.hitP = query.CommittedObjectRayOrigin() + objectRayDir * rayT;

            if (gpuInstances[instanceID].flags & GPU_INSTANCE_FLAG_MERGED)
            {
                uint primIndex = query.CommittedPrimitiveIndex();
                GPUInstance instance = gpuInstances[instanceID];
                PartitionInfo info = partitionInfos[instance.partitionIndex];

                uint resourceID = partitionResourceIDs[info.transformOffset + primIndex]; 

                GPUTruncatedEllipsoid ellipsoid = truncatedEllipsoids[resourceID];
                float3x4 worldFromObject = ConvertGPUMatrix(instanceTransforms[info.transformOffset + primIndex], info.base, info.scale);
                float3x4 objectFromWorld_ = Inverse(worldFromObject);
                float4x4 objectFromWorld = float4x4(
                    objectFromWorld_[0][0], objectFromWorld_[0][1], objectFromWorld_[0][2], objectFromWorld_[0][3], 
                    objectFromWorld_[1][0], objectFromWorld_[1][1], objectFromWorld_[1][2], objectFromWorld_[1][3], 
                    objectFromWorld_[2][0], objectFromWorld_[2][1], objectFromWorld_[2][2], objectFromWorld_[2][3], 
                    0, 0, 0, 1.f);
                float3x4 inverseTransform = mul(ellipsoid.transform, objectFromWorld);
                float3 sphereNormal = normalize(mul(inverseTransform, float4(hitInfo.hitP, 1.f)) - ellipsoid.sphere.xyz);
                float3x3 normalTransform = transpose(float3x3(inverseTransform[0].xyz, inverseTransform[1].xyz, inverseTransform[2].xyz));
                float3 normal = normalize(mul(normalTransform, sphereNormal));

                hitInfo.n = normal;
                hitInfo.gn = hitInfo.n;
                float2x3 tb = BuildOrthonormalBasis(hitInfo.n);
                hitInfo.ss = tb[0];
                hitInfo.ts = tb[1];
            }
        }

        // Get material
        GPUMaterial material = materials[NonUniformResourceIndex(materialID)];
        // Ray cone
        float4 reflectance;
#if 0
        if (material.textureIndex != -1)
        {
        }
        Ptex::FaceData faceData = Ptex::GetFaceData(material, faceID);
        int2 dim = int2(1u << faceData.log2Dim.x, 1u << faceData.log2Dim.y);

        float4 reflectance = SampleStochasticCatmullRomBorderless(faceData, material, faceID, uv, mipLevel, filterU, printDebug);
        float surfaceSpreadAngle = depth == 1 ? rayCone.CalculatePrimaryHitUnifiedSurfaceSpreadAngle(dir, hitInfo.n, p0, p1, p2, n0, n1, n2) 
                                                  : rayCone.CalculateSecondaryHitSurfaceSpreadAngle(dir, hitInfo.n, p0, p1, p2, n0, n1, n2);
        rayCone.Propagate(surfaceSpreadAngle, rayT);
        float lambda = rayCone.ComputeTextureLOD(p0, p1, p2, uv0, uv1, uv2, dir, hitInfo.n, dim, printDebug);
        uint mipLevel = (uint)lambda;
#endif

        float3 wo = normalize(float3(dot(hitInfo.ss, -objectRayDir), dot(hitInfo.ts, -objectRayDir),
                              dot(hitInfo.n, -objectRayDir)));

        float2 sample = rng.Uniform2D();

        float filterU = rng.Uniform();

        uint2 virtualPage = ~0u;
        switch (material.type) 
        {
            case GPUMaterialType::Disney: 
            {
                dir = SampleDisney(material, sample, 
            }
            break;
            case GPUMaterialType::Dielectric:
            {
                dir = SampleDielectric(wo, material.ior, sample, throughput, printDebug);
            }
            break;
            default:
            //case GPUMaterialType::Diffuse: 
            {
                // Get base face data

                // Debug
#if 0
                uint hash = Hash(instanceID);
                float4 reflectance;
                reflectance.x = max(.2f, ((hash >> 0) & 0xff) / 255.f);
                reflectance.y = max(.2f, ((hash >> 8) & 0xff) / 255.f);
                reflectance.z = max(.2f, ((hash >> 16) & 0xff) / 255.f);
                reflectance.w = 1.f;
#endif

                float4 reflectance = float4(0.f, 1.f, 0.f, 1.f);

                //dir = SampleDiffuse(reflectance.xyz, wo, sample, throughput, printDebug);
                dir = SampleDisneyThin(sample, throughput, wo);
#if 0
                if (depth == 1)
                {
                    float2 newUv = faceData.rotate ? float2(1 - uv.y, uv.x) : uv;
                    uint2 virtualPage = VirtualTexture::CalculateVirtualPage(faceData.faceOffset, newUv, dim, mipLevel);
                    const uint feedbackMipLevel = VirtualTexture::ClampMipLevel(dim, mipLevel);
                    feedbackRequest = PackFeedbackEntry(virtualPage.x, virtualPage.y, material.textureIndex, feedbackMipLevel);
                }
#endif
            }
            break;
        }


        if (dir.z == 0)
        {
            break;
        }

        float3 origin = TransformP(objectToWorld, hitInfo.hitP);

        if (depth == 1)
        {
            float3 r0 = float3(objectToWorld[0].xyz);
            float3 r1 = float3(objectToWorld[1].xyz);
            float3 r2 = float3(objectToWorld[2].xyz);
            float3x3 adjugate = float3x3(cross(r1, r2), cross(r2, r0), cross(r0, r1));

            float4 clipPos = mul(scene.clipFromRender, float4(origin, 1.f));
            float viewDepth = clipPos.z / clipPos.w;
            depthBuffer[swizzledThreadID] = viewDepth;
            normalRougnessBuffer[swizzledThreadID] = float4(normalize(mul(adjugate, hitInfo.n)), 1.f);

            // TODO
            float3 baseColor = float3(.554, .689, .374);
            baseColor = pow(baseColor, 2.2f);
            diffuseAlbedo[swizzledThreadID] = float4(baseColor, 1.f);
            specularAlbedo[swizzledThreadID] = 0.f;
            specularHitDistance[swizzledThreadID] = 0.f;
        }

        dir = hitInfo.ss * dir.x + hitInfo.ts * dir.y + hitInfo.n * dir.z;
        dir = normalize(TransformV(objectToWorld, dir));

        pos = OffsetRayOrigin(origin, hitInfo.gn, printDebug);

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

#if 0
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
#endif
}
