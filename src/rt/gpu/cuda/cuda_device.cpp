#include "../../platform.h"
#include "cuda_device.h"

namespace rt
{

CUDADevice::CUDADevice()
{
    CUDA_ASSERT(cuInit(0));
    CUDA_ASSERT(cuDeviceGet(&cudaDevice, 0));

    CUDA_ASSERT(cuDevicePrimaryCtxRetain(&cudaContext, cudaDevice));

    arena = ArenaAlloc();
}

ModuleHandle CUDADevice::RegisterModule(string module)
{
    CUDA_ASSERT(cuCtxPushCurrent(cudaContext));

    CUmodule cuModule;
    string data = OS_ReadFile(arena, module);
    CUDA_ASSERT(cuModuleLoadData(&cuModule, data.str));

    u32 numFunctions = 0;
    CUDA_ASSERT(cuModuleGetFunctionCount(&numFunctions, cuModule));

    CUfunction *functions = PushArray(arena, CUfunction, numFunctions);
    CUDA_ASSERT(cuModuleEnumerateFunctions(functions, numFunctions, cuModule));

    for (u32 i = 0; i < numFunctions; i++)
    {
        const char *name;
        CUDA_ASSERT(cuFuncGetName(&name, functions[i]));
        int stop = 5;
    }

    cudaModules.push_back(cuModule);
    ModuleHandle handle = (u32)(cudaModules.size() - 1);

    CUDA_ASSERT(cuCtxPopCurrent(0));

    return handle;
}

void CUDADevice::RegisterKernels(string *kernels, u32 count, ModuleHandle moduleHandle)
{
    // for (u32 i = 0; i < count; i++)
    // {
    //     string kernel   = kernels[i];
    //     CUmodule module = cudaModules[moduleHandle];
    //     CUfunction func;
    //     CUresult result = cuModuleGetFunction(&func, module, (char *)kernel.str);
    //     if (result != CUDA_SUCCESS)
    //     {
    //         const char *errorName   = 0;
    //         const char *errorString = 0;
    //         cuGetErrorName(result, errorString);
    //         cuGetErrorString(result, errorString);
    //         Error(0, "CUDA Error: %S (%S)", errorName, errorString);
    //     }
    //     Assert(result == CUDA_SUCCESS);
    // }
}

} // namespace rt
