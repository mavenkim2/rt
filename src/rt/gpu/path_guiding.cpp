#include "path_guiding.h"

namespace rt
{

PathGuiding::PathGuiding()
{
    // TODO: hardcoded
    string path         = "../src/rt/gpu/kernel.cubin";
    ModuleHandle handle = device->RegisterModule(path);

    for (int kernel = 0; kernel < PATH_GUIDING_KERNEL_MAX; kernel++)
    {
        handles[kernel] =
            device->RegisterKernels((const char *)pathGuidingKernelNames[handle].str, handle);
    }

    u32 numVMMs         = 32;
    u32 numBlocks       = numVMMs;
    const u32 blockSize = 256;

    device->ExecuteKernel(handles[PATH_GUIDING_KERNEL_INITIALIZE_SAMPLES], numBlocks,
                          blockSize, numVMMs);
}

} // namespace rt
