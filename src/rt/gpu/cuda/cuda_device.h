#ifndef CUDA_DEVICE_H_
#define CUDA_DEVICE_H_

#include "../../string.h"
#include "../../thread_context.h"
#include "../device.h"

#include <cuda.h>

namespace rt
{

struct CUDADevice;

struct CUDAContextScope
{
    CUDAContextScope(CUDADevice *device);
    ~CUDAContextScope();
};

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

struct CUDAArena : GPUArena
{
    CUDADevice *device;
    uintptr_t ptr;

    size_t granularity;
    size_t reservedSize;
    size_t committedSize;

    uintptr_t offset;

    CUDAArena() = default;
    CUDAArena(CUDADevice *device, size_t maxSize);

    void *Alloc(size_t size, uintptr_t alignment) override;
    void Clear() override;
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

    GPUArena *CreateArena(size_t maxSize) override;
    void *Alloc(u32 size, uintptr_t alignment) override;
    void MemZero(void *ptr, uint64_t size) override;
    void MemSet(void *ptr, char ch, uint64_t size) override;

protected:
    void ExecuteKernelInternal(KernelHandle handle, uint32_t numBlocks, uint32_t blockSize,
                               void **params, u32 paramCount) override;
};
} // namespace rt

#endif
