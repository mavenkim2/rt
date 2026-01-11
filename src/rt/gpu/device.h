#ifndef DEVICE_H_
#define DEVICE_H_

#include "../string.h"

namespace rt
{

typedef u32 ModuleHandle;
typedef u32 KernelHandle;

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
