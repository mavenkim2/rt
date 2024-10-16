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
void OpenBraid(RecordRebraid &record, BuildRef *refs, u32 start, u32 count, std::atomic<u32> &refOffset)
{
    const u32 QUEUE_SIZE = 8;
    u32 choiceDim        = 0;
    f32 maxExtent        = neg_inf;
    for (u32 d = 0; d < 3; d++)
    {
        f32 extent = record.max[d] - record.min[d];
        if (extent > maxExtent)
        {
            maxExtent = extent;
            choiceDim = d;
        }
    }

    u32 refIDQueue[2 * QUEUE_SIZE] = {};
    u32 openCount                  = 0;
    for (u32 i = start; i < start + count; i++)
    {
        BuildRef &ref         = refs[i];
        refIDQueue[openCount] = i;
        bool isOpen           = (ref.max[choiceDim] - ref.min[choiceDim] > REBRAID_THRESHOLD * maxExtent);
        openCount += isOpen;

        if (openCount >= QUEUE_SIZE)
        {
            openCount -= QUEUE_SIZE;
            BuildRef &refs[QUEUE_SIZE] = {
                refs[refIDQueue[openCount + 0]],
                refs[refIDQueue[openCount + 1]],
                refs[refIDQueue[openCount + 2]],
                refs[refIDQueue[openCount + 3]],
                refs[refIDQueue[openCount + 4]],
                refs[refIDQueue[openCount + 5]],
                refs[refIDQueue[openCount + 6]],
                refs[refIDQueue[openCount + 7]],
            };
            QuantizedNode4 *nodes[QUEUE_SIZE] = {
                (QuantizedNode4 *)refs[0].nodePtr,
                (QuantizedNode4 *)refs[1].nodePtr,
                (QuantizedNode4 *)refs[2].nodePtr,
                (QuantizedNode4 *)refs[3].nodePtr,
                (QuantizedNode4 *)refs[4].nodePtr,
                (QuantizedNode4 *)refs[5].nodePtr,
                (QuantizedNode4 *)refs[6].nodePtr,
                (QuantizedNode4 *)refs[7].nodePtr,
            };
            u32 numChildren[QUEUE_SIZE];

            u32 totalNumChildren = 0;
            u32 childAdd         = 0;
            for (u32 i = 0; i < QUEUE_SIZE; i++)
            {
                u32 childCount = node->GetNumChildren();
                numChildren[i] = childCount;
                totalNumChildren += childCount;
                childAdd += childCount != 0 ? childCount - 1 : 0;
            }
            u32 offset = refOffset.fetch_add(childAdd, std::memory_order_acq_rel);
            for (u32 i = 0; i < QUEUE_SIZE; i++)
            {
                u32 refID                = refIDQueue[openCount + i];
                QuantizedNode *node      = nodes[i];
                QuantizedNode4 *children = (QuantizedNode4 *)node->GetBaseChildPtr();
                f32 minX[4];
                f32 minY[4];
                f32 minZ[4];
                f32 maxX[4];
                f32 maxY[4];
                f32 maxZ[4];

                node->GetBounds(minX, minY, minZ, maxX, maxY, maxZ);

                // = children->Get ? ? ? ;
                refs[refID].min[0] = minX[0];
                refs[refID].min[1] = minY[0];
                refs[refID].min[2] = minZ[0];

                refs[refID].max[0] = maxX[0];
                refs[refID].max[1] = maxY[0];
                refs[refID].max[2] = maxZ[0];

                u32 numPrims         = Max(refs[refID].numPrims / numChildren[i], 1);
                refs[refID].numPrims = numPrims;

                for (u32 b = 1; b < numChildren[i]; b++)
                {
                    refs[offset].min[0]   = minX[b];
                    refs[offset].min[1]   = minY[b];
                    refs[offset].min[2]   = minZ[b];
                    refs[offset].max[0]   = maxX[b];
                    refs[offset].max[1]   = maxY[b];
                    refs[offset].max[2]   = maxZ[b];
                    refs[offset].objectID = refs[refID].objectID;
                    refs[offset].numPrims = numPrims;
                    refs[offset].nodePtr  = (uintptr_t)(children + b);

                    offset++;
                }
            }
        }
    }
}

} // namespace rt
#endif
