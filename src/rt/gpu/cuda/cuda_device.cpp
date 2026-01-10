#include "../../platform.h"
#include "cuda_device.h"

namespace rt
{

CUDADevice::CUDADevice() { arena = ArenaAlloc(); }

ModuleHandle CUDADevice::RegisterModule(string module)
{
    CUmodule cuModule;
    string data     = OS_ReadFile(arena, module);
    CUresult result = cuModuleLoadData(&cuModule, data.str);
    Assert(result != CUDA_SUCCESS);
    cudaModules.push_back(cuModule);

    ModuleHandle handle = (u32)(cudaModules.size() - 1);
    return handle;
}

void CUDADevice::RegisterKernels(string *kernels, u32 count, ModuleHandle moduleHandle)
{
    for (u32 i = 0; i < count; i++)
    {
        string kernel   = kernels[i];
        CUmodule module = cudaModules[moduleHandle];
        CUfunction func;
        CUresult result = cuModuleGetFunction(&func, module, (char *)kernel.str);
        Assert(result == CUDA_SUCCESS);
    }
}

} // namespace rt
