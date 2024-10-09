#ifndef PARTIAL_REBRAID_H
#define PARTIAL_REBRAID_H
namespace rt
{

struct BuildRef
{
    f32 min[3];
    u32 objectID;
    f32 max[3];
    u32 numPrims;
    uintptr_t nodePtr;
};

struct RecordRebraid
{
    f32 min[3];
    u32 start;
    f32 max[3];
    u32 count;
    u32 extEnd;

    u32 End() const
    {
        return start + count;
    }
};

// NOTE: row major affine transformation matrix
struct Transform
{
    f32 x[4];
    f32 y[4];
    f32 z[4];
};

struct Instance2
{
    uintptr_t bvhNode;
    Transform transform;
};

void PartialRebraid(Scene *scene, Arena *arena, Instance2 *instances, u32 numInstances)
{
    BuildRef *b = PushArrayNoZero(arena, BuildRef, 4 * numInstances);

    Lane4F32 min(neg_inf);
    Lane4F32 max(pos_inf);
    for (u32 i = 0; i < numInstances; i++)
    {
        Instance2 &instance = instances[i];
        Assert((instance.bvhNode & 0xf) == 0);
        b[i].nodePtr = bvhNode;
    }

    RecordRebraid record;
}

static const f32 REBRAID_THRESHOLD = .1f;
void OpenBraid(RecordRebraid &record, BuildRef *refs, u32 start, u32 count, u32 newRefOffset)
{
    u32 choiceDim = 0;
    f32 maxExtent = neg_inf;
    for (u32 d = 0; d < 3; d++)
    {
        f32 extent = record.max[d] - record.min[d];
        if (extent > maxExtent)
        {
            maxExtent = extent;
            choiceDim = d;
        }
    }
    for (u32 i = start; i < start + count; i++)
    {
        BuildRef &ref = refs[i];
        if (ref.max[choiceDim] - ref.min[choiceDim] > REBRAID_THRESHOLD * maxExtent)
        {
            QuantizedNode4 *node     = (QuantizedNode4 *)ref.nodePtr;
            QuantizedNode4 *children = (QuantizedNode4 *)node->internalOffset;
            u32 numChildren          = node->GetNumChildren();

            f32 minX[4];
            f32 minY[4];
            f32 minZ[4];
            f32 maxX[4];
            f32 maxY[4];
            f32 maxZ[4];

            if (numChildren)
            {
                ref.nodePtr         = (uintptr_t)&children[0];
                BuildRef *newRefs[] = {
                    &refs[newRefOffset + 1],
                    &refs[newRefOffset + 2],
                    &refs[newRefOffset + 3],
                };
                newRefOffset += 3;
                // TODO: get the bounds, transform them to world space
                for (u32 childIndex = 1; childIndex < numChildren; childIndex++)
                {
                    BuildRef &newRef = refs[newRefOffset++];
                }
            }
        }
    }
}

} // namespace rt
#endif
