#include "../../platform.h"
#include "cuda_device.h"

namespace rt
{

CUDAContextScope::CUDAContextScope(CUDADevice *device)
{
    CUDA_ASSERT(cuCtxPushCurrent(device->cudaContext));
}

CUDAContextScope::~CUDAContextScope() { CUDA_ASSERT(cuCtxPopCurrent(0)); }

CUDAArena::CUDAArena(CUDADevice *device, size_t maxSize)
    : device(device), committedSize(0), offset(0)
{
    CUDAContextScope scope(device);
    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id         = device->cudaDevice;

    CUDA_ASSERT(
        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    reservedSize = ((maxSize + granularity - 1) / granularity) * granularity;
    CUDA_ASSERT(cuMemAddressReserve(&ptr, reservedSize, 0, 0, 0));
}

void *CUDAArena::Alloc(size_t size, uintptr_t alignment)
{
    uintptr_t current   = ptr + offset;
    uintptr_t aligned   = (current + alignment - 1) & ~(alignment - 1);
    uintptr_t newOffset = (aligned - ptr) + size;

    if (newOffset > reservedSize)
    {
        ErrorExit(0, "CUDA arena out of memory. Capacity of %u exceeded.\n", reservedSize);
        return 0;
    }
    else if (newOffset > committedSize)
    {
        CUDAContextScope scope(device);
        size_t commitSize = size > granularity
                                ? granularity * ((size + granularity - 1) / granularity)
                                : granularity;

        ErrorExit(commitSize + committedSize <= reservedSize,
                  "CUDA arena out of memory. Capacity of %u exceeded.\n", reservedSize);

        CUmemAllocationProp prop = {};
        prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id         = device->cudaDevice;

        CUmemGenericAllocationHandle handle;
        CUDA_ASSERT(cuMemCreate(&handle, commitSize, &prop, 0));
        CUDA_ASSERT(cuMemMap(CUdeviceptr(ptr + committedSize), commitSize, 0, handle, 0));
        CUDA_ASSERT(cuMemRelease(handle));

        // TODO: save commit offsets for unmapping when popping

        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id     = device->cudaDevice;
        accessDesc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        CUDA_ASSERT(
            cuMemSetAccess(CUdeviceptr(ptr + committedSize), commitSize, &accessDesc, 1));

        offset = newOffset;
        committedSize += commitSize;

        return (void *)aligned;
    }
    else
    {
        Assert(offset < reservedSize && offset < committedSize);
        offset = newOffset;
        return (void *)aligned;
    }
}

void CUDAArena::Clear() { offset = 0; }

// TODO: hardcoded
CUDADevice::CUDADevice()
{
    CUDA_ASSERT(cuInit(0));
    CUDA_ASSERT(cuDeviceGet(&cudaDevice, 0));

    CUDA_ASSERT(cuDevicePrimaryCtxRetain(&cudaContext, cudaDevice));

    arena     = ArenaAlloc();
    cudaArena = CUDAArena(this, megabytes(512));
}

ModuleHandle CUDADevice::RegisterModule(string module)
{
    CUDAContextScope scope(this);

    CUmodule cuModule;
    string data = OS_ReadFile(arena, module);
    CUDA_ASSERT(cuModuleLoadData(&cuModule, data.str));

    u32 numFunctions = 0;
    CUDA_ASSERT(cuModuleGetFunctionCount(&numFunctions, cuModule));

#if 0
    CUfunction *functions = PushArray(arena, CUfunction, numFunctions);
    CUDA_ASSERT(cuModuleEnumerateFunctions(functions, numFunctions, cuModule));

    for (u32 i = 0; i < numFunctions; i++)
    {
        const char *name;
        CUDA_ASSERT(cuFuncGetName(&name, functions[i]));
        int stop = 5;
    }
#endif

    cudaModules.push_back(cuModule);
    ModuleHandle handle = (u32)(cudaModules.size() - 1);

    return handle;
}

KernelHandle CUDADevice::RegisterKernels(const char *kernel, ModuleHandle moduleHandle)
{
    CUDAContextScope scope(this);
    CUmodule module = cudaModules[moduleHandle];
    CUfunction func;
    CUDA_ASSERT(cuModuleGetFunction(&func, module, kernel));

    cudaKernels.push_back(func);
    KernelHandle handle = KernelHandle(cudaKernels.size() - 1);
    return handle;
}

void CUDADevice::ExecuteKernelInternal(KernelHandle handle, uint32_t numBlocks,
                                       uint32_t blockSize, void **params, u32 paramCount)
{
    CUDAContextScope scope(this);
    ScratchArena scratch;
    CUfunction kernel = cudaKernels[handle];

    // TODO: alternative streams. also, how do I ensure order is right?
    CUDA_ASSERT(cuLaunchKernel(kernel, numBlocks, 1, 1, blockSize, 1, 1, 0, 0, params, 0));
}

GPUArena *CUDADevice::CreateArena(size_t maxSize)
{
    CUDAArena *a = PushStructConstruct(arena, CUDAArena)(this, maxSize);
    return a;
}

void *CUDADevice::Alloc(u32 size, uintptr_t alignment)
{
    return cudaArena.Alloc(size, alignment);
}

void CUDADevice::MemZero(void *ptr, uint64_t size)
{
    CUDAContextScope scope(this);
    CUDA_ASSERT(cuMemsetD8((CUdeviceptr)ptr, 0, size));
}

void CUDADevice::MemSet(void *ptr, char ch, uint64_t size)
{
    CUDAContextScope scope(this);
    CUDA_ASSERT(cuMemsetD8((CUdeviceptr)ptr, ch, size));
}

// void CUDADevice::CopyFromDevice(void *)
// {
//     CUDA_ASSERT(cuMemcpyDtoHAsync(mem.host_pointer, (CUdeviceptr)mem.device_pointer,
//                                   mem.memory_size(), cuda_stream_));
// }

} // namespace rt
