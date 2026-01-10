#ifndef CUDA_DEVICE_H_
#define CUDA_DEVICE_H_

#include "../../string.h"
#include "../device.h"
#include <cuda.h>

namespace rt
{

#define CUDA_ASSERT(statement)                                                                \
    {                                                                                         \
        CUresult result = statement;                                                          \
        if (result != CUDA_SUCCESS)                                                           \
        {                                                                                     \
            const char *name;                                                                 \
            cuGetErrorString(result, &name);                                                  \
            printf("CUDA Error: %s in %s (%s:%d)", name, #statement, __FILE__, __LINE__);     \
            Trap();                                                                           \
        }                                                                                     \
    }

struct CUDADevice : Device
{
    CUdevice cudaDevice;
    CUcontext cudaContext;
    Arena *arena;
    std::vector<Module> modules;
    std::vector<CUmodule> cudaModules;
    std::vector<CUfunction *> cudaKernels;

    // std::vector<Kernel> kernels;
    // std::vector<CUfunction> cudaKernels;

    CUDADevice();
    ModuleHandle RegisterModule(string module) override;
    void RegisterKernels(string *kernels, u32 count, ModuleHandle module) override;
};
} // namespace rt

#endif
