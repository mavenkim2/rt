#pragma once

#ifdef __CUDACC__
#define RT_DEVICE __host__ __device__
#include <cuda_runtime.h>
#else
#define RT_DEVICE
#endif

namespace rt
{
#ifdef WITH_CUDA
#endif
} // namespace rt
