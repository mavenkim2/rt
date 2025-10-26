#include "wavefront_helper.hlsli"
#include "../dense_geometry.hlsli"
#include "../../rt/shader_interop/ray_shaderinterop.h"
#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "../../rt/shader_interop/wavefront_shaderinterop.h"
#include "../../rt/shader_interop/hit_shaderinterop.h"
#include "../hit.hlsli"
#include "../tex/ptex.hlsli"
#include "../lights_temp.hlsli"
#include "../lights/envmap.hlsli"
#include "../tex/ray_cones.hlsli"
#include "../wave_intrinsics.hlsli"
#include "../bsdf/disney_bsdf.hlsli"
#include "../rt.hlsli"

RaytracingAccelerationStructure accel : register(t0);
ConstantBuffer<WavefrontDescriptors> descriptors : register(b1);
RWStructuredBuffer<WavefrontQueue> queues : register(u2);
ConstantBuffer<GPUScene> scene : register(b3);
StructuredBuffer<Resource> resources : register(t4);
StructuredBuffer<GPUTransform> instanceTransforms : register(t5);
StructuredBuffer<GPUMaterial> materials : register(t6);
StructuredBuffer<PartitionInfo> partitionInfos : register(t7);
StructuredBuffer<GPUInstance> gpuInstances : register(t8);

RWTexture2D<float4> albedo : register(u9);
RWStructuredBuffer<float3> normals : register(u10);
RWTexture2D<float4> image : register(u11);

RWStructuredBuffer<PixelInfo> pixelInfos : register(u12);
RWStructuredBuffer<uint> feedbackBuffer : register(u13);

[[vk::push_constant]] RayPushConstant push;

[numthreads(32, 1, 1)]
void main(uint3 dtID: SV_DispatchThreadID)
{
    uint start = queues[WAVEFRONT_SHADE_QUEUE_INDEX].readOffset;
    uint end = queues[WAVEFRONT_SHADE_QUEUE_INDEX].writeOffset;

    uint queueIndex = start + dtID.x;
    if (queueIndex >= end) return;

    // TODO: need to set readOffset to writeOffset, probably in an indirect pass
    queueIndex %= WAVEFRONT_QUEUE_SIZE;

    // so we need...
    // depth, radiance, ray cone, direction, rayT, 
    // do i make direction part of pixel info? or just toss it in the queue as well

    // Shading queue data
    uint clusterID_triangleIndex = GetUint(descriptors.hitShadingQueueClusterIDIndex, queueIndex);
    uint instanceID = GetUint(descriptors.hitShadingQueueInstanceIDIndex, queueIndex);
    float2 bary = GetFloat2(descriptors.hitShadingQueueBaryIndex, queueIndex);
    uint pixelIndex = GetUint(descriptors.hitShadingQueueInstanceIDIndex, queueIndex);
    float rayT = GetFloat(descriptors.hitShadingQueueRayTIndex, queueIndex);
    float3 dir = GetFloat3(descriptors.hitShadingQueueDirIndex, queueIndex);

    // Per pixel data
    PixelInfo pixelInfo = pixelInfos[pixelIndex];
    uint packed = pixelInfo.pixelLocation_specularBounce;
    float3 throughput = pixelInfo.throughput;
    float3 radiance = pixelInfo.radiance;

    RayCone rayCone;
    rayCone.width = pixelInfo.rayConeWidth;
    rayCone.spreadAngle = pixelInfo.rayConeSpread;

    // TODO: rays may not share depth
    uint depth = push.depth;

    uint2 pixelLocation = uint2(BitFieldExtractU32(packed, 15, 0),
                                BitFieldExtractU32(packed, 15, 15));
    bool specularBounce = bool(BitFieldExtractU32(packed, 1, 30));

    RNG rng;
    rng.State = GetUint(descriptors.hitShadingQueueRNGIndex, queueIndex);

    uint triangleIndex = BitFieldExtractU32(clusterID_triangleIndex, MAX_CLUSTER_TRIANGLES_BIT, 0);
    uint clusterID = BitFieldExtractU32(clusterID_triangleIndex, 32u - MAX_CLUSTER_TRIANGLES_BIT, MAX_CLUSTER_TRIANGLES_BIT);

    uint pageIndex = GetPageIndexFromClusterID(clusterID); 
    uint clusterIndex = GetClusterIndexFromClusterID(clusterID);
    
    uint baseAddress = resources[instanceID].baseAddress;
    DenseGeometry dg = GetDenseGeometryHeader2(clusterID, baseAddress);
    
    uint materialID = dg.DecodeMaterialID(triangleIndex);
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
    
    HitInfo hitInfo = CalculateTriangleHitInfo(p0, p1, p2, n0, n1, n2, gn, uv0, uv1, uv2, bary);
    hitInfo.faceID = faceID;

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

    GPUInstance instance = gpuInstances[instanceID];
    PartitionInfo info = partitionInfos[instance.partitionIndex];
    float3x4 objectToWorld = ConvertGPUMatrix(instanceTransforms[instance.transformIndex], info.base, info.scale);
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

    float3 wo = normalize(float3(dot(hitInfo.ss, -dir), dot(frameY, -dir), dot(hitInfo.n, -dir)));

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
            occludedDesc.Origin = OffsetRayOrigin(origin, hitInfo.gn);
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
                
                float3 r = throughput * bsdfVal * L / lightPdf;
                radiance += r;
            }
        }
    }

    float2 sample = rng.Uniform2D();
    float sample2 = rng.Uniform();

    float3 sample3 = float3(sample, sample2);

    uint2 virtualPage = ~0u;
    float bsdfPdf;
    dir = SampleDisney(material, sample3, reflectance.xyz, throughput, wo, bsdfPdf);

    if (dir.z == 0) return;

    bool bounceWasSpecular = material.roughness == 0.f && material.specTrans == 1.f;
    uint index = scene.width * pixelLocation.y + pixelLocation.x;
    if (depth == 1 && !bounceWasSpecular)
    {
        normals[index] = hitInfo.n;
        albedo[pixelLocation] = bounceWasSpecular ? 1 : reflectance;
    }
    else if (depth == 2 && specularBounce)
    {
        normals[index] = hitInfo.n;
        albedo[pixelLocation] = 1;
    }

    specularBounce = bounceWasSpecular;

    float3 newDir = hitInfo.ss * dir.x + frameY * dir.y + hitInfo.n * dir.z;

    float3 pos = OffsetRayOrigin(origin, hitInfo.gn);
    float continuationProb = min(1.f, Luminance(throughput));

    bool russianRoulette = rng.Uniform() >= continuationProb;

    if (russianRoulette) 
    {
        image[pixelLocation] = float4(radiance, 1);
    }
    else 
    {
        throughput /= continuationProb;

        pixelInfos[pixelIndex].pixelLocation_specularBounce |= (specularBounce) << 30u;
        pixelInfos[pixelIndex].radiance = radiance;
        pixelInfos[pixelIndex].throughput = throughput;
        pixelInfos[pixelIndex].rayConeWidth = rayCone.width;
        pixelInfos[pixelIndex].rayConeSpread = rayCone.spreadAngle;

        uint writeOffset;
        InterlockedAdd(queues[WAVEFRONT_RAY_QUEUE_INDEX].writeOffset, 1, writeOffset);
        writeOffset %= WAVEFRONT_QUEUE_SIZE;

        StoreFloat3(pos, descriptors.rayQueueRWPosIndex, writeOffset);
        StoreFloat3(newDir, descriptors.rayQueueRWDirIndex, writeOffset);
        StoreUint(pixelIndex, descriptors.rayQueueRWPixelIndex, writeOffset);
    }
}

