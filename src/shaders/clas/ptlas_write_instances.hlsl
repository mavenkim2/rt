#include "../common.hlsli"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../bit_twiddling.hlsli"
#include "ptlas_write_instances.hlsli"

StructuredBuffer<uint64_t> blasAddresses : register(t5);
StructuredBuffer<BLASData> blasDatas : register(t6);
RWStructuredBuffer<GPUInstance> gpuInstances : register(u7);
StructuredBuffer<AABB> aabbs : register(t8);

[numthreads(32, 1, 1)] 
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint blasIndex = dtID.x;
    if (blasIndex >= globals[GLOBALS_BLAS_COUNT_INDEX]) return;

    BLASData blasData = blasDatas[blasIndex];
    if (blasData.clusterCount == 0) return;

    GPUInstance instance = gpuInstances[blasData.instanceID];

    uint64_t address = blasAddresses[blasData.addressIndex];

    AABB aabb = aabbs[instance.resourceID];
    WritePTLASDescriptors(instance, address, instance.virtualInstanceIDOffset, 0, aabb, true);
}
