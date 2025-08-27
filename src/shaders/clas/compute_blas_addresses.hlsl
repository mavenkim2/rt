#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<uint> blasSizes : register(t0);
RWStructuredBuffer<uint64_t> blasAddresses : register(u1);
RWStructuredBuffer<uint> globals : register(u2);

[[vk::push_constant]] AddressPushConstant pc;

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint blasIndex = dtID.x;
    if (blasIndex >= globals[GLOBALS_BLAS_FINAL_COUNT_INDEX]) return;

    uint blasSize = blasSizes[blasIndex];
    uint blasByteOffset;
    InterlockedAdd(globals[GLOBALS_BLAS_BYTES], blasSize, blasByteOffset);

    uint64_t blasBaseAddress = ((uint64_t)pc.addressHighBits << 32u) | (uint64_t)pc.addressLowBits;

    blasAddresses[blasIndex] = blasBaseAddress + blasByteOffset;
}
