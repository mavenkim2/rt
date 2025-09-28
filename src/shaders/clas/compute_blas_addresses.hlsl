#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<uint> blasSizes : register(t0);
RWStructuredBuffer<uint64_t> blasAddresses : register(u1);
RWStructuredBuffer<uint> globals : register(u2);
RWStructuredBuffer<uint> totalBlasSizes : register(u3);

[[vk::push_constant]] ComputeBLASAddressesPushConstant pc;

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint blasIndex = dtID.x;
    if (blasIndex >= globals[GLOBALS_BLAS_FINAL_COUNT_INDEX]) return;

    uint blasSize = blasSizes[pc.blasOffset + blasIndex];
    uint blasByteOffset;
    InterlockedAdd(totalBlasSizes[1], blasSize, blasByteOffset);

    uint64_t blasBaseAddress = ((uint64_t)pc.addressHighBits << 32u) | (uint64_t)pc.addressLowBits;
    uint64_t blasAddress = blasBaseAddress + blasByteOffset;

    blasAddresses[pc.blasOffset + blasIndex] = blasAddress;
}
