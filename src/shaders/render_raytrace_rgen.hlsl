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
StructuredBuffer<uint> clusterLookupTable : register(t13);
StructuredBuffer<GPUInstance> gpuInstances : register(t14);
RWStructuredBuffer<uint> proxyCounts : register(u15);
StructuredBuffer<GPUTruncatedEllipsoid> truncatedEllipsoids : register(t16);
StructuredBuffer<GPUTransform> instanceTransforms : register(t17);
StructuredBuffer<PartitionInfo> partitionInfos : register(t18);

RWStructuredBuffer<float2> debugBuffer : register(u19);
RWStructuredBuffer<uint> globals : register(u20);

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
                    GPUTruncatedEllipsoid ellipsoid = truncatedEllipsoids[resourceID];
                    uint primIndex = query.CandidatePrimitiveIndex();
                    PartitionInfo info = partitionInfos[instance.partitionIndex];
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
                    if (depth == 0)
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

                    // TODO: clip against bounding box?
#if 0
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
#ifndef TRACE_BRICKS
#error
                uint levelTableOffset = query.CandidateInstanceID();
                uint primIndex = query.CandidatePrimitiveIndex();

                uint brickIndex = (primIndex >> 6) & 127;
                uint voxelIndex = primIndex & 63;
                uint tableOffset = primIndex >> 13;

                uint clusterID = clusterLookupTable[levelTableOffset + tableOffset];

                uint pageIndex = GetPageIndexFromClusterID(clusterID); 
                uint clusterIndex = GetClusterIndexFromClusterID(clusterID);

                uint basePageAddress = GetClusterPageBaseAddress(pageIndex);
                uint numClusters = GetNumClustersInPage(basePageAddress);
                DenseGeometry dg = GetDenseGeometryHeader(basePageAddress, numClusters, clusterIndex);

                float3 position = dg.DecodePosition(brickIndex);
                float voxelSize = dg.lodError;
                float rcpVoxelSize = 1.f / voxelSize;

                uint x         = voxelIndex & 3u;
                uint y         = (voxelIndex >> 2u) & 3u;
                uint z         = voxelIndex >> 4u;

                position += float3(x, y, z) * voxelSize;
                //float3 boundsMin = position + float3(x, y, z) * voxelSize; 
                //float3 boundsMax = boundsMin + voxelSize;

                //float3 objectRayOrigin = query.CandidateObjectRayOrigin();
                //float3 objectRayDir = query.CandidateObjectRayDirection();
                //float3 invDir = rcp(objectRayDir);
                //float tEntry, tLeave;
                //float3 tMin;

                //IntersectAABB(boundsMin, boundsMax, objectRayOrigin, invDir, tEntry, tLeave, tMin);

                float3 objectRayOrigin = (query.CandidateObjectRayOrigin() - position) * rcpVoxelSize;
                float3 objectRayDir = query.CandidateObjectRayDirection() * rcpVoxelSize;
                float3 invDir =
                    float3(abs(objectRayDir.x) < 1e-8f ? 0.f : 1.f / objectRayDir.x, 
                           abs(objectRayDir.y) < 1e-8f ? 0.f : 1.f / objectRayDir.y, 
                           abs(objectRayDir.z) < 1e-8f ? 0.f : 1.f / objectRayDir.z);

                float tEntry, tLeave;
                float3 tMin;
                IntersectAABB(float3(0, 0, 0), float3(1, 1, 1), objectRayOrigin, invDir, tEntry, tLeave, tMin);

                tLeave = min(tLeave, hitT);
                tEntry = max(tEntry, 0);

                if (tEntry <= tLeave)
                {
#if 0
                    Brick brick = dg.DecodeBrick(brickIndex);
                    uint vertexOffset = GetVoxelVertexOffset(brick.vertexOffset, brick.bitMask, voxelIndex);
                    float alpha = dg.DecodeCoverage(vertexOffset);

                    if (rng.Uniform() < alpha)
                    {
                        hitT = tEntry;

                        query.CommitProceduralPrimitiveHit(tEntry);
                        break;
                    }
#else
                    hitT = tEntry;
                    query.CommitProceduralPrimitiveHit(tEntry);
#endif
                }
#else
                uint tableBaseOffset = gpuInstances[instanceID].clusterLookupTableOffset;
                //uint resourceID = query.CandidateInstanceID();
                uint primIndex = query.CandidatePrimitiveIndex();

                uint brickIndex = primIndex & 127;
                uint tableOffset = primIndex >> 7;

                uint clusterID = clusterLookupTable[tableBaseOffset + tableOffset];

                uint pageIndex = GetPageIndexFromClusterID(clusterID); 
                uint clusterIndex = GetClusterIndexFromClusterID(clusterID);

                uint basePageAddress = GetClusterPageBaseAddress(pageIndex);
                uint numClusters = GetNumClustersInPage(basePageAddress);
                DenseGeometry dg = GetDenseGeometryHeader(basePageAddress, numClusters, clusterIndex);

                Brick brick = dg.DecodeBrick(brickIndex);

                float3 position = dg.DecodePosition(brickIndex);
                float voxelSize = dg.lodError;
                float rcpVoxelSize = rcp(voxelSize);

                uint3 voxelMax;
                GetBrickMax(brick.bitMask, voxelMax);
                //float3 boundsMin = position * rcpVoxelSize; 
                //float3 boundsMax = boundsMin + float3(voxelMax);

                float3 objectRayOrigin = (query.CandidateObjectRayOrigin() - position) * rcpVoxelSize;
                float3 objectRayDir = query.CandidateObjectRayDirection() * rcpVoxelSize;
                float3 invDir =
                    float3(abs(objectRayDir.x) < 1e-8f ? 0.f : 1.f / objectRayDir.x, 
                           abs(objectRayDir.y) < 1e-8f ? 0.f : 1.f / objectRayDir.y, 
                           abs(objectRayDir.z) < 1e-8f ? 0.f : 1.f / objectRayDir.z);

                float tEntry, tLeave;
                IntersectAABB(float3(0, 0, 0), float3(voxelMax), objectRayOrigin, invDir, tEntry, tLeave);

                tLeave = min(tLeave, hitT);
                tEntry = max(tEntry, 0);

                if (tEntry <= tLeave)
                {
                    float tHit = tEntry;
                    float maxT = tLeave;

                    int3 step = int3(objectRayDir.x >= 0 ? 1 : -1, objectRayDir.y >= 0 ? 1 : -1, 
                                     objectRayDir.z >= 0 ? 1 : -1);
                    int3 add = int3(objectRayDir.x >= 0 ? 1 : 0, objectRayDir.y >= 0 ? 1 : 0,
                                    objectRayDir.z >= 0 ? 1 : 0);

                    float3 stepT = abs(invDir);

                    float3 intersectP = objectRayOrigin + objectRayDir * tHit;

                    int3 voxel = floor(intersectP);

                    float3 nextTime = tHit + ((voxel + add) - intersectP) * invDir;

                    // DDA
                    for (;;)
                    {
                        if (tHit >= maxT || any(and(objectRayDir < 0, voxel < 0)) || any(and(objectRayDir >= 0, voxel >= voxelMax))) break;

                        if (all(voxel >= 0) && all(voxel < voxelMax))
                        {
                            uint bit = voxel.x + voxel.y * 4 + voxel.z * 16;
                            if (brick.bitMask & (1u << bit))
                            {
#if 0
                                uint vertexOffset = GetVoxelVertexOffset(brick.vertexOffset, brick.bitMask, bit);
                                float alpha = dg.DecodeCoverage(vertexOffset);

                                if (rng.Uniform() < alpha)
                                {
                                    hitKind = bit;
                                    hitT = tHit;

                                    query.CommitProceduralPrimitiveHit(tHit);
                                    break;
                                }
#else
                                hitKind = bit;
                                hitT = tHit;

                                query.CommitProceduralPrimitiveHit(tHit);
                                break;
#endif
                            }
                        }

                        float nextT = min(nextTime.x, min(nextTime.y, nextTime.z));
                        uint axis = nextT == nextTime.x ? 0 : (nextT == nextTime.y ? 1 : 2);

                        voxel[axis] += step[axis];
                        nextTime[axis] += stepT[axis];
                        tHit = nextT;
                    }
                }
#endif
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
#if 0
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
#endif
            uint basePageAddress = GetClusterPageBaseAddress(pageIndex);
            uint numClusters = GetNumClustersInPage(basePageAddress);
            DenseGeometry dg = GetDenseGeometryHeader(basePageAddress, numClusters, clusterIndex);

            uint materialID = dg.DecodeMaterialID(triangleIndex);
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

            float3 n0, n1, n2;
            if (dg.HasNormals())
            {
                //n0 = dg.DecodeNormal(vids[0]);
                //n1 = dg.DecodeNormal(vids[1]);
                //n2 = dg.DecodeNormal(vids[2]);
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
            float2 uv1 = float2(0, 1);//isSecondFace ? float2(1, 1) : float2(1, 0);
            float2 uv2 = float2(1, 1);//isSecondFace ? float2(0, 1) : float2(1, 1);

            hitInfo = CalculateTriangleHitInfo(p0, p1, p2, n0, n1, n2, uv0, uv1, uv2, bary);

            // Ray cone
#if 0
            Ptex::FaceData faceData = Ptex::GetFaceData(material, faceID);
            int2 dim = int2(1u << faceData.log2Dim.x, 1u << faceData.log2Dim.y);

            float4 reflectance = SampleStochasticCatmullRomBorderless(faceData, material, faceID, uv, mipLevel, filterU, printDebug);
            float surfaceSpreadAngle = depth == 1 ? rayCone.CalculatePrimaryHitUnifiedSurfaceSpreadAngle(dir, hitInfo.n, p0, p1, p2, n0, n1, n2) 
                                                  : rayCone.CalculateSecondaryHitSurfaceSpreadAngle(dir, hitInfo.n, p0, p1, p2, n0, n1, n2);
            rayCone.Propagate(surfaceSpreadAngle, rayT);
            float lambda = rayCone.ComputeTextureLOD(p0, p1, p2, uv0, uv1, uv2, dir, hitInfo.n, dim, printDebug);
            uint mipLevel = (uint)lambda;
#endif
        }
        else if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
        {
            uint instanceID = query.CommittedInstanceID();

            hitInfo.hitP = query.CommittedObjectRayOrigin() + objectRayDir * rayT;

            if (gpuInstances[instanceID].flags & GPU_INSTANCE_FLAG_MERGED)
            {
                uint primIndex = query.CommittedPrimitiveIndex();
                GPUInstance instance = gpuInstances[instanceID];
                uint resourceID = instance.resourceID;
                GPUTruncatedEllipsoid ellipsoid = truncatedEllipsoids[resourceID];
                PartitionInfo info = partitionInfos[instance.partitionIndex];
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
            else 
            {
#ifndef TRACE_BRICKS
#error
                uint primIndex = query.CommittedPrimitiveIndex();
                uint brickIndex = primIndex >> 6u;
                hitKind = primIndex & 63u;
#else
                uint primIndex = query.CandidatePrimitiveIndex();
                uint brickIndex = primIndex & 127;
                uint tableOffset = primIndex >> 7;
#endif
                uint tableBaseOffset = gpuInstances[instanceID].clusterLookupTableOffset;
                uint clusterID = clusterLookupTable[tableBaseOffset + tableOffset];

                uint pageIndex = GetPageIndexFromClusterID(clusterID); 
                uint clusterIndex = GetClusterIndexFromClusterID(clusterID);

                uint basePageAddress = GetClusterPageBaseAddress(pageIndex);
                uint numClusters = GetNumClustersInPage(basePageAddress);
                DenseGeometry dg = GetDenseGeometryHeader(basePageAddress, numClusters, clusterIndex);

                materialID = dg.DecodeMaterialID(brickIndex);
                Brick brick = dg.DecodeBrick(brickIndex);

                uint vertexOffset = GetVoxelVertexOffset(brick.vertexOffset, brick.bitMask, hitKind);
                float3 normal = dg.DecodeNormal(vertexOffset);

#if 1
                float2x3 wBasis = BuildOrthonormalBasis(-objectRayDir);
                float3 wk = wBasis[0];
                float3 wj = wBasis[1];
                SGGX sggx = dg.DecodeSGGX(vertexOffset);

                float3 wm = sggx.SampleSGGX(-objectRayDir, wk, wj, rng.Uniform2D());

                if (any(isnan(wm)) || any(isinf(wm)))
                {
                    hitInfo.n = normal;
                    hitInfo.gn = hitInfo.n;
                }
                else 
                {
                    hitInfo.n = wm;
                    hitInfo.gn = normal;
                }
                float2x3 tb = BuildOrthonormalBasis(hitInfo.n);
                hitInfo.ss = tb[0];
                hitInfo.ts = tb[1];

#else
                hitInfo.n = dg.DecodeNormal(vertexOffset);
                hitInfo.gn = hitInfo.n;
                float2x3 tb = BuildOrthonormalBasis(hitInfo.n);
                hitInfo.ss = tb[0];
                hitInfo.ts = tb[1];
#endif
            }
        }

        // Get material
        GPUMaterial material = materials[NonUniformResourceIndex(materialID)];

        float3 wo = normalize(float3(dot(hitInfo.ss, -objectRayDir), dot(hitInfo.ts, -objectRayDir),
                              dot(hitInfo.n, -objectRayDir)));

        float2 sample = rng.Uniform2D();

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

                dir = SampleDiffuse(reflectance.xyz, wo, sample, throughput, printDebug);
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
