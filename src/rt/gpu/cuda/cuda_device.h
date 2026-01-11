#ifndef CUDA_DEVICE_H_
#define CUDA_DEVICE_H_

#include "../../string.h"
#include "../../thread_context.h"
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
    std::vector<CUfunction> cudaKernels;

    CUDADevice();
    ModuleHandle RegisterModule(string module) override;
    KernelHandle RegisterKernels(const char *kernel, ModuleHandle module) override;

    void ExecuteKernelInternal(KernelHandle handle, uint32_t numBlocks, uint32_t blockSize,
                               void **params, u32 paramCount) override
    {
        CUDA_ASSERT(cuCtxPushCurrent(cudaContext));
        ScratchArena scratch;
        CUfunction kernel = cudaKernels[handle];

        // TODO: alternative streams. also, how do I ensure order is right?
        cuLaunchKernel(kernel, numBlocks, 1, 1, blockSize, 1, 1, 0, 0, params, 0);
        CUDA_ASSERT(cuCtxPopCurrent(0));
    }
};
} // namespace rt

#endif
