#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<uint> blasSizes : register(t0);
RWStructuredBuffer<uint64_t> blasAddresses : register(u1);
RWStructuredBuffer<uint> globals : register(u2);
RWStructuredBuffer<AccelerationStructureInstance> instanceDescriptors : register(u3);  
RWStructuredBuffer<uint2> offsetsAndCounts : register(u4);  
StructuredBuffer<BLASData> blasDatas : register(t5);

[[vk::push_constant]] AddressPushConstant pc;

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint blasIndex = dtID.x;
    if (blasIndex >= globals[GLOBALS_BLAS_COUNT_INDEX]) return;

    BLASData blasData = blasDatas[blasIndex];
    if (blasData.clusterCount == 0) return;

    uint addressIndex = blasData.addressIndex;
    uint blasSize = blasSizes[addressIndex];
    uint blasByteOffset;
    InterlockedAdd(globals[GLOBALS_BLAS_BYTES], blasSize, blasByteOffset);

    uint64_t blasBaseAddress = ((uint64_t)pc.addressHighBits << 32u) | (uint64_t)pc.addressLowBits;
    uint64_t blasAddress = blasBaseAddress + blasByteOffset;

    blasAddresses[addressIndex] = blasAddress;

    if (blasData.tlasIndex != ~0u)
    {
        float3x4 transform = float3x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
        AccelerationStructureInstance instanceDescriptor;
        instanceDescriptor.transform = transform;
        instanceDescriptor.instanceID = 0;
        instanceDescriptor.instanceMask = 0xff;
        instanceDescriptor.instanceContributionToHitGroupIndex = 0;
        instanceDescriptor.flags = 0;
        instanceDescriptor.blasDeviceAddress = blasAddress;

        uint descriptorIndex;
        uint tlasIndex = blasDatas[blasIndex].tlasIndex + 1;
        InterlockedAdd(offsetsAndCounts[tlasIndex].y, 1, descriptorIndex);
        descriptorIndex += offsetsAndCounts[tlasIndex].x;
        instanceDescriptors[descriptorIndex] = instanceDescriptor;
    }
}
