#include "../../rt/shader_interop/as_shaderinterop.h"
RWStructuredBuffer<uint> clasGlobalsBuffer : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    if (dtID.x == 0)
    {
        clasGlobalsBuffer[GLOBALS_CLAS_INDIRECT_X] = (clasGlobalsBuffer[GLOBALS_VISIBLE_CLUSTER_COUNT_INDEX] + 31u) >> 5u;
        clasGlobalsBuffer[GLOBALS_CLAS_INDIRECT_Y] = 1;
        clasGlobalsBuffer[GLOBALS_CLAS_INDIRECT_Z] = 1;

        clasGlobalsBuffer[GLOBALS_BLAS_INDIRECT_X] = (clasGlobalsBuffer[GLOBALS_BLAS_COUNT_INDEX] + 128u) >> 7u;;
        clasGlobalsBuffer[GLOBALS_BLAS_INDIRECT_Y] = 1;
        clasGlobalsBuffer[GLOBALS_BLAS_INDIRECT_Z] = 1;

        clasGlobalsBuffer[GLOBALS_FREED_INSTANCE_INDIRECT_X] = (clasGlobalsBuffer[GLOBALS_FREED_INSTANCE_COUNT_INDEX] + 31u) >> 5u;
        clasGlobalsBuffer[GLOBALS_FREED_INSTANCE_INDIRECT_Y] = 1;
        clasGlobalsBuffer[GLOBALS_FREED_INSTANCE_INDIRECT_Z] = 1;

        uint numGroups = (clasGlobalsBuffer[GLOBALS_ALLOCATED_INSTANCE_COUNT_INDEX] + 31u) >> 5u;
        uint numGroupsY = uint(ceil(numGroups / float(65535)));
        uint numGroupsX = uint(ceil(numGroups / float(numGroupsY)));
        clasGlobalsBuffer[GLOBALS_ALLOCATE_INSTANCE_INDIRECT_X] = numGroupsX;
        clasGlobalsBuffer[GLOBALS_ALLOCATE_INSTANCE_INDIRECT_Y] = numGroupsY;
        clasGlobalsBuffer[GLOBALS_ALLOCATE_INSTANCE_INDIRECT_Z] = 1;
    }
}
