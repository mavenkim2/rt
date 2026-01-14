#pragma once

#ifdef __CUDACC__
#define RT_DEVICE     __host__ __device__
#define RT_GPU_DEVICE __device__
#include <cuda_runtime.h>
#else
#define RT_DEVICE
#define RT_GPU_DEVICE
#endif

namespace rt
{
#ifdef WITH_CUDA
#endif
} // namespace rt
