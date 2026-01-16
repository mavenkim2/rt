#ifndef DEVICE_H_
#define DEVICE_H_

#include "../string.h"
#include "platform.h"

namespace rt
{

typedef u32 ModuleHandle;
typedef u32 KernelHandle;

struct GPUArena
{
    virtual void *Alloc(size_t size, uintptr_t alignment) = 0;
    virtual void Clear()                                  = 0;

    template <typename T>
    T *Alloc(u32 count, u32 alignment = 0)
    {
        alignment = alignment == 0 ? sizeof(T) : alignment;
        return (T *)Alloc(sizeof(T) * count, alignment);
    }
};

struct Device
{
    virtual ModuleHandle RegisterModule(string module)                            = 0;
    virtual KernelHandle RegisterKernels(const char *kernel, ModuleHandle module) = 0;
    template <typename... Args>
    void ExecuteKernel(KernelHandle handle, uint32_t numBlocks, uint32_t blockSize,
                       Args... args)
    {
        void *paramArray[] = {(void *)&args...};
        ExecuteKernelInternal(handle, numBlocks, blockSize, paramArray, sizeof...(args));
    }

    virtual void *Alloc(u32 size, uintptr_t alignment) = 0;
    template <typename T>
    T *Alloc(u32 count, u32 alignment = 0)
    {
        alignment = alignment == 0 ? sizeof(T) : alignment;
        return (T *)Alloc(sizeof(T) * count, alignment);
    }
    virtual void MemZero(void *ptr, uint64_t size) = 0;
    virtual GPUArena *CreateArena(size_t maxSize)  = 0;

protected:
    virtual void ExecuteKernelInternal(KernelHandle handle, uint32_t numBlocks,
                                       uint32_t blockSize, void **params, u32 paramCount) = 0;
};

struct Module
{
    string name;
    string data;
};

struct Kernel
{
    string name;
    ModuleHandle moduleHandle;
};

} // namespace rt

#endif
