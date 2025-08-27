#include "../common.hlsli"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../bit_twiddling.hlsli"
#include "../dense_geometry.hlsli"

StructuredBuffer<uint64_t> blasAddresses : register(t0);
RWStructuredBuffer<uint> globals : register(u1);
StructuredBuffer<BLASData> blasDatas : register(t2);
RWStructuredBuffer<GPUInstance> gpuInstances : register(u3);

RWStructuredBuffer<PTLAS_WRITE_INSTANCE_INFO> ptlasInstanceWriteInfos : register(u4);
RWStructuredBuffer<PTLAS_UPDATE_INSTANCE_INFO> ptlasInstanceUpdateInfos : register(u5);

RWStructuredBuffer<uint> renderedBitVector : register(u6);
RWStructuredBuffer<uint> thisFrameBitVector : register(u7);
StructuredBuffer<AABB> aabbs : register(t9);
StructuredBuffer<uint64_t> tlasAddresses : register(t10);
StructuredBuffer<uint64_t> blasVoxelAddressTable : register(t11);

void WritePTLASDescriptors(GPUInstance instance, uint64_t address, uint instanceIndex, uint instanceID, AABB aabb, bool update)
{
    uint partition = instance.partitionIndex;
    uint2 offsets = uint2(instanceID >> 5u, instanceID & 31u);
    bool wasRendered = renderedBitVector[offsets.x] & (1u << offsets.y);

    if (1)//!wasRendered)
    {
        uint descriptorIndex;
        InterlockedAdd(globals[GLOBALS_PTLAS_WRITE_COUNT_INDEX], 1, descriptorIndex);

        PTLAS_WRITE_INSTANCE_INFO instanceInfo = (PTLAS_WRITE_INSTANCE_INFO)0;
        if (update)
        {
            float3 minP = float3(FLT_MAX, FLT_MAX, FLT_MAX);
            float3 maxP = float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
            for (int z = 0; z < 2; z++)
            {
                for (int y = 0; y < 2; y++)
                {
                    for (int x = 0; x < 2; x ++)
                    {
                        float3 p = float3(x ? aabb.maxX : aabb.minX, y ? aabb.maxY : aabb.minY, z ? aabb.maxZ : aabb.minZ);
                        float3 pos = mul(instance.worldFromObject, float4(p, 1.f));
                        minP = min(minP, pos);
                        maxP = max(maxP, pos);
                    }
                }
            }
            for (int i = 0; i < 3; i++)
            {
                instanceInfo.explicitAABB[i] = minP[i];
                instanceInfo.explicitAABB[3 + i] = maxP[i];
            }
            //instanceInfo.instanceFlags |= 0x10u;
        }

        instanceInfo.transform = instance.worldFromObject;
        instanceInfo.instanceID = instanceID;
        instanceInfo.instanceMask = 0xff;
        instanceInfo.instanceContributionToHitGroupIndex = 0;
        instanceInfo.instanceIndex = instanceIndex;
        instanceInfo.partitionIndex = partition;
        instanceInfo.accelerationStructure = address;

        ptlasInstanceWriteInfos[descriptorIndex] = instanceInfo;
        InterlockedOr(renderedBitVector[offsets.x], (1u << offsets.y));
    }
    else if (update)
    {
        uint descriptorIndex;
        InterlockedAdd(globals[GLOBALS_PTLAS_UPDATE_COUNT_INDEX], 1, descriptorIndex);

        PTLAS_UPDATE_INSTANCE_INFO instanceInfo;
        instanceInfo.instanceIndex = instanceIndex;
        instanceInfo.instanceContributionToHitGroupIndex = 0;
        instanceInfo.accelerationStructure = address;

        ptlasInstanceUpdateInfos[descriptorIndex] = instanceInfo;
    }

    InterlockedOr(thisFrameBitVector[offsets.x], 1u << offsets.y);
}

[numthreads(32, 1, 1)] 
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint blasIndex = dtID.x;
    if (blasIndex >= globals[GLOBALS_BLAS_COUNT_INDEX]) return;

    BLASData blasData = blasDatas[blasIndex];
    GPUInstance instance = gpuInstances[blasData.instanceID];

    uint64_t address = 0;
    uint instanceID = 0;
    uint instanceIndex = blasData.instanceID;
    // TLAS built over cluster BLAS and voxel BLAS
    if (blasData.tlasIndex != ~0u)
    {
        address = tlasAddresses[blasData.tlasIndex];
        instanceID = 2;
        return;
    }
    // Singular voxel BLAS
    else if (blasData.voxelClusterCount == 1)
    {
        uint clusterID = blasData.addressIndex;
        address = blasVoxelAddressTable[clusterID];
        instanceID = clusterID;
        return;
    }
    // Singular cluster BLAS
    else
    {
        instanceID = 1;
        address = blasAddresses[blasData.addressIndex];
    }

    AABB aabb = aabbs[instance.resourceID];
    WritePTLASDescriptors(instance, address, blasData.instanceID, blasIndex, /*instanceID, */aabb, true);
#if 0
    if (blasData.voxelClusterCount) 
    {
        for (int index = 0; index < blasData.voxelClusterCount; index++)
        {
            BLASVoxelInfo info = blasVoxelInfos[blasData.voxelClusterStartIndex + index];
            uint pageIndex = info.clusterID >> MAX_CLUSTERS_PER_PAGE_BITS;
            uint clusterIndex = info.clusterID & (MAX_CLUSTERS_PER_PAGE - 1);

            uint basePageAddress = GetClusterPageBaseAddress(pageIndex);
            uint numClusters = GetNumClustersInPage(basePageAddress);
            DenseGeometry header = GetDenseGeometryHeader(basePageAddress, numClusters, clusterIndex);

            AABB aabb;
            aabb.minX = header.boundsMin.x;
            aabb.minY = header.boundsMin.y;
            aabb.minZ = header.boundsMin.z;
            aabb.maxX = header.boundsMax.x;
            aabb.maxY = header.boundsMax.y;
            aabb.maxZ = header.boundsMax.z;

        //AABB aabb = aabbs[instance.resourceID];

            CLASPageInfo pageInfo = clasPageInfos[pageIndex];

            uint virtualInstanceID = instance.virtualInstanceIDOffset + 1 + info.instanceIndex;

            WritePTLASDescriptors(instance, info.address, virtualInstanceID, info.clusterID, aabb, false);
        }
    }

    if (blasData.clusterCount) 
    {
        uint64_t address = blasAddresses[blasData.addressIndex];

        AABB aabb = aabbs[instance.resourceID];
        WritePTLASDescriptors(instance, address, instance.virtualInstanceIDOffset, 0, aabb, true);
    }
#endif
}
