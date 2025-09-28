#include "../common.hlsli"
#include "cull.hlsli"
#include "../bit_twiddling.hlsli"
#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"
#include "lod_error_test.hlsli"

// TODO: hierarchical instance culling
ConstantBuffer<GPUScene> gpuScene : register(b0);
RWStructuredBuffer<GPUInstance> gpuInstances : register(u1);
RWStructuredBuffer<uint> globals : register(u2);
StructuredBuffer<AABB> aabbs : register(t3);

StructuredBuffer<GPUTransform> instanceTransforms : register(t4);
StructuredBuffer<PartitionInfo> partitionInfos : register(t5);
StructuredBuffer<Resource> resources : register(t6);

#if 0
StructuredBuffer<ResourceSharingInfo> resourceSharingInfos : register(t8);
RWStructuredBuffer<uint2> maxMinLodLevel : register(u9);
#endif
RWStructuredBuffer<BLASData> blasDatas : register(u8);

[numthreads(64, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint instanceIndex = dtID.x;
    if (instanceIndex >= 1u << 21u) return;

    GPUInstance instance = gpuInstances[instanceIndex];

    if (instance.flags & GPU_INSTANCE_FLAG_FREED) return;

    // Proxy
    if (instance.flags & GPU_INSTANCE_FLAG_MERGED)
    {
        BLASData blasData = (BLASData)0;
        blasData.instanceID = instanceIndex;
        uint blasIndex;
        InterlockedAdd(globals[GLOBALS_BLAS_COUNT_INDEX], 1, blasIndex);
        blasDatas[blasIndex] = blasData;
        return;
    }

#if 0
    {
        if (pc.oneBlasAddress != 0)
        {
            AABB aabb;
            uint64_t address = mergedPartitionDeviceAddresses[instance.partitionIndex];
            float3x4 worldFromObject = float3x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
            WritePTLASDescriptors(worldFromObject, address, instanceIndex, instanceIndex,
                    aabb, false, 0x0u);
            return;
        }
    }
    {
        if (pc.oneBlasAddress != 0)
        {
            uint proxyIndex = instance.partitionIndex;
            AABB aabb;
            aabb.minX = -1;
            aabb.minY = -1;
            aabb.minZ = -1;
            aabb.maxX = 1;
            aabb.maxY = 1;
            aabb.maxZ = 1;
            WritePTLASDescriptors(instance, pc.oneBlasAddress, instanceIndex, instanceIndex,
                                  //aabb, true, 0x10u);
                                  aabb, false, 0x0u, 0);
        }
    }
#endif

#if 0
    Resource resource = resources[instance.resourceID];

    PartitionInfo info = partitionInfos[instance.partitionIndex];
    float3x4 renderFromObject = ConvertGPUMatrix(instanceTransforms[instance.transformIndex], info.base, info.scale); 
    AABB aabb = aabbs[instance.resourceID];

    float4 temp;
    float tempfloat, tempfloat2;
    bool cull = FrustumCull(gpuScene.clipFromRender, renderFromObject, 
            float3(aabb.minX, aabb.minY, aabb.minZ),
            float3(aabb.maxX, aabb.maxY, aabb.maxZ), gpuScene.p22, gpuScene.p23, temp, tempfloat, tempfloat2);
    Translate(renderFromObject, -gpuScene.cameraP);

    // BLAS Sharing
    // https://github.com/nvpro-samples/vk_lod_clusters/blob/main/docs/blas_sharing.md
    float scaleX = length2(float3(renderFromObject[0].x, renderFromObject[1].x, renderFromObject[2].x));
    float scaleY = length2(float3(renderFromObject[0].y, renderFromObject[1].y, renderFromObject[2].y));
    float scaleZ = length2(float3(renderFromObject[0].z, renderFromObject[1].z, renderFromObject[2].z));

    float3 scale = float3(scaleX, scaleY, scaleZ);
    scale = sqrt(scale);
    float minScale = min(scale.x, min(scale.y, scale.z));
    float maxScale = max(scale.x, max(scale.y, scale.z));

    uint minLevel = 0;
    uint maxLevel = resource.numLodLevels - 1;
    bool testMin = true;
    for (uint lodLevel = 0; lodLevel < resource.numLodLevels; lodLevel++)
    {
        uint infoIndex = resource.resourceSharingInfoOffset + lodLevel;
        ResourceSharingInfo info = resourceSharingInfos[infoIndex];
        float test;

        if (testMin)
        {
            float2 edgeScales = TestNode(renderFromObject, gpuScene.cameraFromRender, resource.lodBounds, maxScale, test, cull);
            if (edgeScales.x <= info.maxError * gpuScene.lodScale * minScale)
            {
                testMin = false;
                minLevel = lodLevel;
            }
        }
        if (!testMin)
        {
            float4 lodBounds = resource.lodBounds;
            float3 cameraForward = -gpuScene.cameraFromRender[2].xyz;
            lodBounds.xyz = mul(renderFromObject, float4(lodBounds.xyz, 1.f));
            lodBounds.w = lodBounds.w * maxScale;
            lodBounds.xyz += normalize(cameraForward) * (lodBounds.w - maxScale * info.smallestBounds.w);
            lodBounds.w = info.smallestBounds.w * maxScale;

            float3x4 identity = float3x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);

            float2 edgeScales = TestNode(identity, gpuScene.cameraFromRender, lodBounds, 1.f, test, cull);
            if (edgeScales.x <= info.smallestError * gpuScene.lodScale * minScale)
            {
                maxLevel = lodLevel;
                break;
            }
        }
    }

    // TODO: don't hardcode
    if (maxLevel <= 3)
    {
        uint packed = (minLevel << 27u) | instanceIndex;
        InterlockedMax(maxMinLodLevel[instance.resourceID].x, packed);
    }
    InterlockedMin(maxMinLodLevel[instance.resourceID].y, maxLevel);
    gpuInstances[instanceIndex].minLodLevel = minLevel;
    gpuInstances[instanceIndex].maxLodLevel = maxLevel;

    if (cull)
    {
        gpuInstances[instanceIndex].flags |= GPU_INSTANCE_FLAG_CULL;
    }
    else 
    {
        gpuInstances[instanceIndex].flags &= ~GPU_INSTANCE_FLAG_CULL;
    }
#endif
    //gpuInstances[instanceIndex].flags &= ~(GPU_INSTANCE_FLAG_MERGED_INSTANCE | GPU_INSTANCE_FLAG_SHARED_INSTANCE);
}
