#include "path_guiding.hlsli"

StructuredBuffer<ParallaxVMM> vmms : register(t0);
RWStructuredBuffer<uint> pathGuidingGlobals : register(u1);
RWStructuredBuffer<uint> vmmOffsets : register(u2);

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    uint numVMMs = 0;
    if (dtID.x >= numVMMs) return;
    uint numSamples = 0;

    ParallaxVMM vmm = vmms[dtID.x];
    uint count = 1 + (vmm.numComponents * numSamples + PATH_GUIDING_GROUP_SIZE - 1) / PATH_GUIDING_GROUP_SIZE;

    uint index;
    InterlockedAdd(pathGuidingGlobals[0], index, count);
    vmmOffsets[dtID.x] = index;
}
