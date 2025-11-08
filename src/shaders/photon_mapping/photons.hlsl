#define RayTracingClusterAccelerationStructureNV 5437

[[vk::ext_capability(RayTracingClusterAccelerationStructureNV)]]
[[vk::ext_extension("SPV_NV_cluster_acceleration_structure")]]

#include "../common.hlsli"
#include "../dense_geometry.hlsli"
#include "../sampling.hlsli"
#include "../tex/virtual_textures.hlsli"
#include "../tex/ray_cones.hlsli"
#include "../tex/ptex.hlsli"
#include "../nvidia/clas.hlsli"
#include "../hit.hlsli"

struct Photon
{
    float3 hitP;
    float3 throughput;
};

RWStructuredBuffer<Photon> photons : register(u0);
RWStructuredBuffer<uint> photonCount : register(u1);
RaytracingAccelerationStructure accel : register(t2);
StructuredBuffer<GPUMaterial> materials : register(t3);
StructuredBuffer<GPUInstance> gpuInstances : register(t4);
StructuredBuffer<Resource> resources : register(t5);

[shader("raygeneration")]
void main()
{
    uint depth = 0;
    const uint maxDepth = 3;

    static float4 causticLightColor = pow(2, 13.6) * float4(1.0, 0.7681251, 0.56915444, 1.0);
    float3 throughput = causticLightColor;

    float3x4 renderFromLight;
    float3 sceneCenter;
    float sceneRadius;

    RNG rng;

    bool specularBounce = false;
    uint diffuseBounces = 0;

    float3 w = mul(renderFromLight, float4(0, 0, 1, 0));
    float2x3 basis = BuildOrthonormalBasis(w);
    float2 diskPt = SampleUniformDiskConcentric(rng.Uniform2D());

    float3 pDisk = sceneCenter + sceneRadius * (basis[0] * diskPt.x + basis[1] * diskPt.y);
    float3 pos = pDisk + sceneRadius * w;
    float3 dir = -w;

    throughput /= (PI * Sqr(sceneRadius));

    for (int depth = 0; depth < maxDepth; depth++)
    {
        RayDesc desc;
        desc.Origin = pos;
        desc.Direction = dir;
        desc.TMin = 0;
        desc.TMax = FLT_MAX;

        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> query;
        query.TraceRayInline(accel, RAY_FLAG_NONE, 0xff, desc);

        query.Proceed();

        if (query.CommittedStatus() == COMMITTED_NOTHING) return;

        uint clusterID = GetClusterIDNV(query, RayQueryCommittedIntersectionKHR);
        uint pageIndex = GetPageIndexFromClusterID(clusterID); 
        uint clusterIndex = GetClusterIndexFromClusterID(clusterID);

        uint triangleIndex = query.CommittedPrimitiveIndex();
        float2 bary = query.CommittedTriangleBarycentrics();
        uint instanceID = query.CommittedInstanceID();
        uint baseAddress = resources[instanceID].baseAddress;
        DenseGeometry dg = GetDenseGeometryHeader2(clusterID, baseAddress);

        uint materialID = dg.DecodeMaterialID(triangleIndex);
        uint3 vids = dg.DecodeTriangle(triangleIndex);

        uint2 pageInformation = dg.DecodeFaceIDAndRotateInfo(triangleIndex);
        uint faceID = pageInformation.x;
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

        // Ray cone
        GPUMaterial material = materials[materialID];
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

        float3 frameY = normalize(cross(hitInfo.n, hitInfo.ss));

        float3 wo = normalize(float3(dot(hitInfo.ss, -query.WorldRayDirection()), 
                                     dot(frameY, -query.WorldRayDirection()),
                                     dot(hitInfo.n, -query.WorldRayDirection())));

        // Sample scattering
        float2 sample = rng.Uniform2D();
        float sample2 = rng.Uniform();

        float3 sample3 = float3(sample, sample2);

        uint2 virtualPage = ~0u;
        dir = SampleDisney(material, sample3, reflectance.xyz, throughput, wo, bsdfPdf);

        if (dir.z == 0)
        {
            break;
        }

        if (specularBounce)
        {
            uint index;
            InterlockedAdd(photonCount[0], 1, index);

            Photon photon;
            photon.throughput = throughput;
            photon.hitP = origin;

            photons[index] = photon;
        }

        bool bounceWasSpecular = material.roughness == 0.f && material.specTrans == 1.f;
        specularBounce = bounceWasSpecular;
        diffuseBounces += !bounceWasSpecular;
        diffuseBounces = specularBounce ? 0 : diffuseBounces;
        if (diffuseBounces >= 2) return;

        dir = hitInfo.ss * dir.x + frameY * dir.y + hitInfo.n * dir.z;
        pos = OffsetRayOrigin(origin, hitInfo.gn);
    }
}
