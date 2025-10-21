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
#include "lights_temp.hlsli"
#include "lights/envmap.hlsli"

RaytracingAccelerationStructure accel : register(t0);
RWTexture2D<float4> image : register(u1);
ConstantBuffer<GPUScene> scene : register(b2);
StructuredBuffer<GPUMaterial> materials : register(t4);
ConstantBuffer<ShaderDebugInfo> debugInfo: register(b7);

RWStructuredBuffer<uint> feedbackBuffer : register(u12);
RWStructuredBuffer<float3> debugBuffer : register(u13);

StructuredBuffer<GPUInstance> gpuInstances : register(t14);
StructuredBuffer<GPUTruncatedEllipsoid> truncatedEllipsoids : register(t16);
StructuredBuffer<GPUTransform> instanceTransforms : register(t17);
StructuredBuffer<PartitionInfo> partitionInfos : register(t18);
StructuredBuffer<uint> partitionResourceIDs : register(t19);
StructuredBuffer<Resource> resources : register(t20);

RWTexture2D<float> depthBuffer : register(u21);
RWTexture2D<float4> albedo : register(u22);
RWStructuredBuffer<float3> normals : register(u23);

StructuredBuffer<float> filterCDF : register(t26);
StructuredBuffer<float> filterValues : register(t27);

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
    float2 sample = rng.Uniform2D();
    const float2 filterRadius = float2(0.5, 0.5);

#if 1
    // First, sample marginal
    int offsetY, offsetX;
    //float pdfY, pdfX;
    float pdf = 1;
    float2 sampleP = 0;

    float marginalIntegral = push.filterIntegral;
    float radius = 1.5f;
    int piecewiseConstant2DWidth = 32 * radius;
    float2 minD = -radius;
    float2 maxD = radius;

    for (int i = piecewiseConstant2DWidth - 1; i >= 0; i--)
    {
        if (filterCDF[i] <= sample.y)
        {
            offsetY = i;
            //pdfY = filterValues[i] / mitchellIntegral;

            float cdfRange = filterCDF[i + 1] - filterCDF[i];
            float du = (sample.y - filterCDF[i]) * (cdfRange > 0.f ? 1.f / cdfRange : 0.f);
            float t = saturate((i + du) / (float)piecewiseConstant2DWidth);
            sampleP.y = lerp(minD.y, maxD.y, t);

            break;
        }
    }
    
    for (int i = piecewiseConstant2DWidth - 1; i >= 0; i--)
    {
        uint cdfIndex = (piecewiseConstant2DWidth + 1) * (1 + offsetY) + i;
        if (filterCDF[cdfIndex] <= sample.x)
        {
            offsetX = i;
            pdf = abs(filterValues[piecewiseConstant2DWidth * offsetY + offsetX]) / marginalIntegral;
            //pdfX = filterValues[i] / mitchellIntegral;

            float cdfRange = filterCDF[cdfIndex + 1] - filterCDF[cdfIndex];
            float du = (sample.x - filterCDF[cdfIndex]) * (cdfRange > 0.f ? 1.f / cdfRange : 0.f);
            float t = saturate((i + du) / (float)piecewiseConstant2DWidth);
            sampleP.x = lerp(minD.x, maxD.x, t);

            break;
        }
    }

    float2 filterSample = sampleP;

#else
    float2 filterSample = float2(lerp(-filterRadius.x, filterRadius.x, sample[0]), lerp(-filterRadius.y, filterRadius.y, sample[1]));
#endif

    filterSample += float2(0.5, 0.5) + float2(swizzledThreadID);
    float2 pLens = rng.Uniform2D();

    const uint maxDepth = 3;
    uint depth = 0;

    half3 throughput = 1;
    half3 radiance = 0;

    float3 pos;
    float3 dir;
    float3 dpdx, dpdy, dddx, dddy;
    GenerateRay(scene, filterSample, pLens, pos, dir, dpdx, dpdy, dddx, dddy);

    RayCone rayCone;
    rayCone.width = 0.f;
    rayCone.spreadAngle = atan(2.f * tan(scene.fov / 2.f) / scene.height);

    bool printDebug = all(swizzledThreadID == debugInfo.mousePos);

    bool specularBounce = false;
    float bsdfPdf = 0.f;

    while (true)
    {
        RayDesc desc;
        desc.Origin = pos;
        desc.Direction = dir;
        desc.TMin = 0;
        desc.TMax = FLT_MAX;
        
        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> query;
        query.TraceRayInline(accel, RAY_FLAG_NONE, 0xff, desc);

        query.Proceed();

        if (query.CommittedStatus() == COMMITTED_NOTHING)
        {
            float3 imageLe = EnvMapLe(scene.lightFromRender, push.envMap, dir);
            if (1)//specularBounce || depth == 0)
            {
                radiance += throughput * imageLe;
            }
            else 
            {
                float lightPdf = .1f / (4 * PI);
                float weight = Sqr(lightPdf) / (Sqr(lightPdf) + Sqr(bsdfPdf));
                radiance += throughput * weight * imageLe;
            }

            if (depth == 0 || (depth == 1 && specularBounce))
            {
                depthBuffer[swizzledThreadID] = 0.f;
                //normalRougnessBuffer[swizzledThreadID] = 0.f;
                //diffuseAlbedo[swizzledThreadID] = 0.f;
                uint index = scene.width * swizzledThreadID.y + swizzledThreadID.x;
                albedo[swizzledThreadID] = 0;
                normals[index] = 0;
            }
            break;
        }

        // TODO: emitter intersection
        if (depth++ >= maxDepth)
        {
            break;
        }

#if 0
        [[vk::ext_storage_class(HitObjectAttributeNV)]] BuiltInTriangleIntersectionAttributes attributes;
        attributes.barycentrics = query.CommittedTriangleBarycentrics();

        HitObjectNV hitObject;
        CreateHitObjectNV();
        SERMakeHit(hitObject, accel, query.CommittedInstanceID(), query.CommittedGeometryIndex(),
                   query.CommittedPrimitiveIndex(), 0, 0u, 0u, 
                   desc.Origin, desc.TMin, desc.Direction, desc.TMax, attributes);
        ReorderThreadWithHitNV(hitObject, depth, 2);
#endif

        uint materialID = 0;
        HitInfo hitInfo = (HitInfo)0;
        float3 objectRayDir = query.CommittedObjectRayDirection();
        uint instanceID = query.CommittedInstanceID();
        float rayT = query.CommittedRayT();

        //if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            uint clusterID = GetClusterIDNV(query, RayQueryCommittedIntersectionKHR);
            uint pageIndex = GetPageIndexFromClusterID(clusterID); 
            uint clusterIndex = GetClusterIndexFromClusterID(clusterID);

            uint triangleIndex = query.CommittedPrimitiveIndex();
            float2 bary = query.CommittedTriangleBarycentrics();
            uint baseAddress = resources[instanceID].baseAddress;
            DenseGeometry dg = GetDenseGeometryHeader2(clusterID, baseAddress);

            materialID = dg.DecodeMaterialID(triangleIndex);
            uint3 vids = dg.DecodeTriangle(triangleIndex);

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

            float3 p0 = dg.DecodePosition(vids[0]);
            float3 p1 = dg.DecodePosition(vids[1]);
            float3 p2 = dg.DecodePosition(vids[2]);

            float3 gn = normalize(cross(p0 - p2, p1 - p2));

            float3 n0, n1, n2;
            if (dg.HasNormals())
            {
                n0 = dg.DecodeNormal(vids[0]);
                n1 = dg.DecodeNormal(vids[1]);
                n2 = dg.DecodeNormal(vids[2]);
            }
            else 
            {
                n0 = gn;
                n1 = gn;
                n2 = gn;
            }

            // Calculate triangle differentials
            // p0 + dpdu * u + dpdv * v = p1

            float2 uv0 = float2(0, 0);
            float2 uv1 = isSecondFace ? float2(1, 1) : float2(1, 0);
            float2 uv2 = isSecondFace ? float2(0, 1) : float2(1, 1);

            hitInfo = CalculateTriangleHitInfo(p0, p1, p2, n0, n1, n2, gn, uv0, uv1, uv2, bary);
            hitInfo.faceID = faceID;
        }
#if 0
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
#endif

        // Get material
        GPUMaterial material = materials[materialID];
        // Ray cone
        float4 reflectance = 1;
        float filterU = rng.Uniform();
        if (material.textureIndex != -1)
        {
            Ptex::FaceData faceData = Ptex::GetFaceData(material, hitInfo.faceID);
            int2 dim = int2(1u << faceData.log2Dim.x, 1u << faceData.log2Dim.y);

            float surfaceSpreadAngle = depth == 1 ? rayCone.CalculatePrimaryHitUnifiedSurfaceSpreadAngle(dir, hitInfo.n, hitInfo.p0, hitInfo.p1, hitInfo.p2, hitInfo.n0, hitInfo.n1, hitInfo.n2) 
                : rayCone.CalculateSecondaryHitSurfaceSpreadAngle(dir, hitInfo.n, hitInfo.p0, hitInfo.p1, hitInfo.p2, hitInfo.n0, hitInfo.n1, hitInfo.n2);
            rayCone.Propagate(surfaceSpreadAngle, rayT);
            float lambda = rayCone.ComputeTextureLOD(hitInfo.p0, hitInfo.p1, hitInfo.p2, hitInfo.uv0, hitInfo.uv1, hitInfo.uv2, dir, hitInfo.n, dim);
            uint mipLevel = (uint)lambda;
            mipLevel = min(mipLevel, max(faceData.log2Dim.x, faceData.log2Dim.y));
            uint tileIndex = 0;
            reflectance = SampleStochasticCatmullRomBorderless(faceData, material, hitInfo.faceID, hitInfo.uv, mipLevel, filterU, tileIndex);

            if (depth == 1)
            {
                uint2 feedbackRequest = uint2(material.textureIndex | (tileIndex << 16u), hitInfo.faceID | (mipLevel << 28u));
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
        }
        else 
        {
            reflectance = material.baseColor;
        }

        float3x4 objectToWorld = query.CommittedObjectToWorld3x4();
        float3 origin = TransformP(objectToWorld, hitInfo.hitP);

        float3 r0 = float3(objectToWorld[0].xyz);
        float3 r1 = float3(objectToWorld[1].xyz);
        float3 r2 = float3(objectToWorld[2].xyz);
        float3x3 adjugate = float3x3(cross(r1, r2), cross(r2, r0), cross(r0, r1));
        hitInfo.gn = normalize(mul(adjugate, hitInfo.gn));
        hitInfo.n = normalize(mul(adjugate, hitInfo.n));
        hitInfo.ss = normalize(mul(objectToWorld, float4(hitInfo.ss, 0.f)));
        //hitInfo.ts = normalize(mul(objectToWorld, float4(hitInfo.ts, 0.f)));

        float3 frameY = normalize(cross(hitInfo.n, hitInfo.ss));

        float3 wo = normalize(float3(dot(hitInfo.ss, -query.WorldRayDirection()), 
                                     dot(frameY, -query.WorldRayDirection()),
                                     dot(hitInfo.n, -query.WorldRayDirection())));

        // NEE
        if (!(material.roughness == 0.f && material.specTrans == 1.f))
        {
            float lightSample = rng.Uniform();
            float weightTotal = 0.f;
            uint lightSampleIndex = 0;
            float chosenImportance = 0.f;

            // TODO hardcoded
            for (int i = 0; i < 22; i++)
            {
                float4 color = areaLightColors[i];

                float importance = .3 * color.x + .6 * color.y + .1 * color.z;
                weightTotal += importance;
                float prob = importance / weightTotal;

                if (lightSample < prob)
                {
                    lightSample /= prob;
                    lightSampleIndex = i;
                    chosenImportance = importance;
                }
                else 
                {
                    lightSample = (lightSample - prob) / (1 - prob);
                }
            }
            float lightPdf = chosenImportance / weightTotal;
            float3 lightSampleDirection;

            float2 lightDirSample = rng.Uniform2D();
            float tMax = FLT_MAX;
            bool deltaLight = false;

            if (lightSample < .1)
            {
                lightSampleDirection = SampleUniformSphere(lightDirSample);
                lightPdf = .1f / (4 * PI);
                deltaLight = true;
            }
            else 
            {
                float2 areaLightDim = areaLightDims[lightSampleIndex];

                float3 p[4] = 
                {
                    float3(areaLightDim.x, areaLightDim.y, 0.f) / 2,
                    float3(-areaLightDim.x, areaLightDim.y, 0.f) / 2,
                    float3(-areaLightDim.x, -areaLightDim.y, 0.f) / 2,
                    float3(areaLightDim.x, -areaLightDim.y, 0.f) / 2,
                };

                float3x4 areaLightTransform = areaLightTransforms[lightSampleIndex];
                Translate(areaLightTransform, -scene.cameraBase);

                p[0] = mul(areaLightTransform, float4(p[0], 1));
                p[1] = mul(areaLightTransform, float4(p[1], 1));
                p[2] = mul(areaLightTransform, float4(p[2], 1));
                p[3] = mul(areaLightTransform, float4(p[3], 1));

                float3 v00 = normalize(p[0] - origin);
                float3 v10 = normalize(p[1] - origin);
                float3 v01 = normalize(p[3] - origin);
                float3 v11 = normalize(p[2] - origin);

                float3 p01        = p[1] - p[0];
                float3 p02        = p[2] - p[0];
                float3 p03        = p[3] - p[0];
                float3 lightSamplePos;
                float area0        = 0.5f * length(cross(p01, p02));
                float area1        = 0.5f * length(cross(p02, p03));

                float div  = 1.f / (area0 + area1);
                float prob = area0 * div;
                // Then sample the triangle by area
                if (lightDirSample[0] < prob)
                {
                    lightDirSample[0]       = lightDirSample[0] / prob;
                    float3 bary = SampleUniformTriangle(lightDirSample);
                    lightSamplePos = bary[0] * p[0] + bary[1] * p[1] + bary[2] * p[2];
                }
                else
                {
                    lightDirSample[0]       = (lightDirSample[0] - prob) / (1 - prob);
                    float3 bary = SampleUniformTriangle(lightDirSample);
                    lightSamplePos = bary[0] * p[0] + bary[1] * p[2] + bary[2] * p[3];
                }
                lightSampleDirection = normalize(lightSamplePos - origin);
                float samplePointPdf = div * length2(origin - lightSamplePos) / abs(dot(hitInfo.n, lightSampleDirection));
                lightPdf *= .9f * samplePointPdf;
                tMax = length(origin - lightSamplePos) * .99f;
            }

            float lightBsdfPdf;
            float3 wi = normalize(float3(dot(hitInfo.ss, lightSampleDirection), 
                                         dot(frameY, lightSampleDirection),
                                         dot(hitInfo.n, lightSampleDirection)));
            float3 bsdfVal = EvaluateDisney(material, reflectance.xyz, wo, wi, lightBsdfPdf)
                             * abs(dot(hitInfo.n, lightSampleDirection));
            lightBsdfPdf = 0;

            if (all(bsdfVal > 0.f))
            {
                RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> occludedQuery;
                RayDesc occludedDesc;
                occludedDesc.Origin = OffsetRayOrigin(origin, hitInfo.gn, printDebug);
                occludedDesc.Direction = lightSampleDirection;
                occludedDesc.TMin = 0.f;
                occludedDesc.TMax = tMax;
                occludedQuery.TraceRayInline(accel, RAY_FLAG_NONE, 0xff, occludedDesc);

                occludedQuery.Proceed();

                if (occludedQuery.CommittedStatus() == COMMITTED_NOTHING)
                {
                    //float weight = Sqr(lightPdf) / (Sqr(lightBsdfPdf) + Sqr(lightPdf));
                    float3 L = deltaLight ? 
                                EnvMapLe(scene.lightFromRender, push.envMap, lightSampleDirection)
                                : areaLightColors[lightSampleIndex].xyz;
                    
                    //float3 r = throughput * bsdfVal * weight * L / lightPdf;
                    float3 r = throughput * bsdfVal * L / lightPdf;
                    radiance += r;
                }
            }
        }

        float2 sample = rng.Uniform2D();
        float sample2 = rng.Uniform();

        float3 sample3 = float3(sample, sample2);

        uint2 virtualPage = ~0u;
        dir = SampleDisney(material, sample3, reflectance.xyz, throughput, wo, bsdfPdf);

        if (dir.z == 0)
        {
            break;
        }

        bool bounceWasSpecular = material.roughness == 0.f && material.specTrans == 1.f;
        uint index = scene.width * swizzledThreadID.y + swizzledThreadID.x;
        if (depth == 1 && !bounceWasSpecular)
        {
            normals[index] = hitInfo.n;
            albedo[swizzledThreadID] = bounceWasSpecular ? 1 : reflectance;
        }
        else if (depth == 2 && specularBounce)
        {
            normals[index] = hitInfo.n;
            albedo[swizzledThreadID] = 1;
        }

        specularBounce = bounceWasSpecular;
        dir = hitInfo.ss * dir.x + frameY * dir.y + hitInfo.n * dir.z;

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
}
