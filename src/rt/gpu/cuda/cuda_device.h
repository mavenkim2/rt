#ifndef CUDA_DEVICE_H_
#define CUDA_DEVICE_H_

#include "../../string.h"
#include "../../thread_context.h"
#include "../device.h"

#include <cuda.h>
#include <cuda_runtime.h>

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

struct CUDAArena
{
    void *ptr;
    uintptr_t base;
    u32 totalSize;
    u32 offset;

    void Init(u32 size)
    {
        cudaMalloc(&ptr, size);
        totalSize = size;
        base      = uintptr_t(ptr);
        offset    = 0;
    }

    void *Alloc(u32 size, uintptr_t alignment)
    {
        uintptr_t current = uintptr_t((char *)ptr + offset);
        uintptr_t aligned = (current + alignment - 1) & ~(alignment - 1);
        offset            = (aligned - base) + size;

        Assert(offset <= totalSize);
        return (void *)aligned;
    }

    template <typename T>
    T *Alloc(u32 count, u32 alignment = 0)
    {
        alignment = alignment == 0 ? sizeof(T) : alignment;
        return (T *)Alloc(sizeof(T) * count, alignment);
    }

    void Clear() { offset = 0; }
    // void Release() { cudaFree; }
};

struct CUDADevice : Device
{
    CUdevice cudaDevice;
    CUcontext cudaContext;
    Arena *arena;

    CUDAArena cudaArena;
    std::vector<Module> modules;
    std::vector<CUmodule> cudaModules;
    std::vector<CUfunction> cudaKernels;

    CUDADevice();
    ModuleHandle RegisterModule(string module) override;
    KernelHandle RegisterKernels(const char *kernel, ModuleHandle module) override;

    void ExecuteKernelInternal(KernelHandle handle, uint32_t numBlocks, uint32_t blockSize,
                               void **params, u32 paramCount) override;

    void *Alloc(u32 size, uintptr_t alignment) override;
};
} // namespace rt

#endif
