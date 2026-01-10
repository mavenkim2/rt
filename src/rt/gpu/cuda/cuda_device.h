#ifndef CUDA_DEVICE_H_
#define CUDA_DEVICE_H_

#include "../../string.h"
#include "../device.h"
#include <cuda.h>

namespace rt
{
struct CUDADevice : Device
{
    Arena *arena;
    std::vector<Module> modules;
    std::vector<CUmodule> cudaModules;

    std::vector<Kernel> kernels;
    std::vector<CUfunction> cudaKernels;

    CUDADevice();
    ModuleHandle RegisterModule(string module) override;
    void RegisterKernels(string *kernels, u32 count, ModuleHandle module) override;
};
} // namespace rt

#endif
