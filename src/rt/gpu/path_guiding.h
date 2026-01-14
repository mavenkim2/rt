#ifndef GPU_PATH_GUIDING_H_
#define GPU_PATH_GUIDING_H_

#include "../base.h"
#include "path_guiding_util.h"
#include "device.h"

namespace rt
{

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
    PathGuiding(Device *device);
    Device *device;
    KernelHandle handles[PATH_GUIDING_KERNEL_MAX];
};

} // namespace rt

#endif
