#ifndef DEVICE_H_
#define DEVICE_H_

#include "../string.h"

namespace rt
{

typedef u32 ModuleHandle;

struct Device
{
    virtual ModuleHandle RegisterModule(string module)                            = 0;
    virtual void RegisterKernels(string *kernels, u32 count, ModuleHandle module) = 0;
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
