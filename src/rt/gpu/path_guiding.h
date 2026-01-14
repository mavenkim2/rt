#ifndef GPU_PATH_GUIDING_H_
#define GPU_PATH_GUIDING_H_

#include "../base.h"
#include "device.h"

namespace rt
{

struct KDTreeNode
{
    float splitPos;
    u32 childIndex_dim;

    // TODO: remove?
    int parentIndex;

    u32 offset;
    u32 count;
    u32 vmmIndex;

    RT_DEVICE u32 GetChildIndex() const { return (childIndex_dim << 2) >> 2; }
    RT_DEVICE void SetChildIndex(u32 childIndex)
    {
        Assert(childIndex < (1u << 30u));
        childIndex_dim |= childIndex;
    }
    RT_DEVICE u32 GetSplitDim() const { return childIndex_dim >> 30; }
    RT_DEVICE void SetSplitDim(u32 dim)
    {
        Assert(dim < 3);
        childIndex_dim |= dim << 30;
    }
    RT_DEVICE bool HasChild() const { return childIndex_dim != ~0u; }
    RT_DEVICE bool IsChild() const { return (childIndex_dim >> 30u) == 3u; }
};

enum PathGuidingKernels : int
{
    PATH_GUIDING_KERNEL_INITIALIZE_SAMPLES,
    PATH_GUIDING_KERNEL_MAX,
};

static const string pathGuidingKernelNames[] = {
    "InitializeSamples",
};

struct PathGuiding
{
    PathGuiding();
    Device *device;
    KernelHandle handles[PATH_GUIDING_KERNEL_MAX];
};

} // namespace rt

#endif
