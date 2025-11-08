#include "../../rt/shader_interop/kd_tree_shaderinterop.h"
#include "../../rt/shader_interop/sort_shaderinterop.h"

RWStructuredBuffer<int> elementCounts : register(u0);
RWStructuredBuffer<uint> indirectBuffer : register(u1);

struct Push 
{
    uint num;
};

[[vk::push_constant]] Push pc;

[numthreads(1, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    if (dtID.x != 0) return;

    uint level = pc.num;

    if (level > 0)
    {
        elementCounts[0] -= (1l << (level - 1));
    }

    uint value = elementCounts[0];

    uint createSortX = (value + KD_TREE_WORKGROUP_SIZE - 1) / KD_TREE_WORKGROUP_SIZE;
    uint sortX = (value + SORT_WORKGROUP_SIZE - 1) / SORT_WORKGROUP_SIZE;

    indirectBuffer[KD_TREE_INDIRECT_X] = createSortX;
    indirectBuffer[KD_TREE_INDIRECT_Y] = 1;
    indirectBuffer[KD_TREE_INDIRECT_Z] = 1;
    indirectBuffer[SORT_KEYS_INDIRECT_X] = sortX;
    indirectBuffer[SORT_KEYS_INDIRECT_Y] = 1;
    indirectBuffer[SORT_KEYS_INDIRECT_Z] = 1;
}
