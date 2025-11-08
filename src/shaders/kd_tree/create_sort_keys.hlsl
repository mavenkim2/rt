#include "../../rt/shader_interop/kd_tree_shaderinterop.h"

StructuredBuffer<float3> points : register(t0);
StructuredBuffer<uint> dims : register(t1);
RWStructuredBuffer<uint64_t> tags : register(u2);
StructuredBuffer<uint> indices : register(t3);

[numthreads(KD_TREE_WORKGROUP_SIZE, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    // TODO
    uint index = indices[dtID.x];
    float3 pt = points[index];

    uint dim = dims[index];
    float pointCoord = pt[dim];
    uint pointKey = pointCoord > 0.f ? (0x80000000u | asuint(pointCoord)) 
                                     : ~asuint(pointCoord);

    uint tag = uint(tags[dtID.x]);
    uint64_t newTag = (uint64_t(tag) << 32u) | pointKey;

    tags[dtID.x] = newTag;
}
