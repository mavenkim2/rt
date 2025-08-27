#include "../../rt/shader_interop/as_shaderinterop.h"

StructuredBuffer<uint> lastFrameBitVector : register(t0);
StructuredBuffer<uint> thisFrameBitVector : register(t1);
RWStructuredBuffer<uint> globals : register(u2);
RWStructuredBuffer<PTLAS_UPDATE_INSTANCE_INFO> ptlasInstanceUpdateInfos : register(u3);

[numthreads(32, 1, 1)]
void main(uint dtID : SV_DispatchThreadID)
{
    uint frameBits = thisFrameBitVector[dtID.x];
    uint lastFrameBits = lastFrameBitVector[dtID.x];

    uint unusedMask = lastFrameBits & ~frameBits;
    while (unusedMask)
    {
        uint instanceIndex = 32 * dtID.x + firstbitlow(unusedMask);
        uint descriptorIndex;
        InterlockedAdd(globals[GLOBALS_PTLAS_UPDATE_COUNT_INDEX], 1, descriptorIndex);

        PTLAS_UPDATE_INSTANCE_INFO instanceInfo;
        instanceInfo.instanceIndex = instanceIndex;
        instanceInfo.instanceContributionToHitGroupIndex = 0;
        instanceInfo.accelerationStructure = 0;

        ptlasInstanceUpdateInfos[descriptorIndex] = instanceInfo;

        unusedMask &= unusedMask - 1;
    }
}
