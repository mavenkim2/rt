#include "../../rt/shader_interop/as_shaderinterop.h"
RWStructuredBuffer<uint> clasGlobalsBuffer : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    if (dtID.x == 0)
    {
        clasGlobalsBuffer[GLOBALS_CLAS_INDIRECT_X] = (clasGlobalsBuffer[GLOBALS_CLAS_COUNT_INDEX] + 31u) >> 5u;
        clasGlobalsBuffer[GLOBALS_CLAS_INDIRECT_Y] = 1;
        clasGlobalsBuffer[GLOBALS_CLAS_INDIRECT_Z] = 1;
    }
}
