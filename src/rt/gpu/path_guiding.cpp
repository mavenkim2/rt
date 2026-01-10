#include "path_guiding.h"

namespace rt
{

PathGuiding::PathGuiding()
{
    // TODO: hardcoded
    string path         = "../src/rt/gpu/kernel.cubin";
    ModuleHandle handle = device->RegisterModule(path);

    string kernels[1];
    kernels[0] = "help";
    device->RegisterKernels(kernels, ArrayLength(kernels), handle);

    // cuModuleLoadDataEx();
}

} // namespace rt
