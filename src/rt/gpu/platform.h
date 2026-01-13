#pragma once

#ifdef WITH_CUDA
#define GPU_DEVICE __host__ __device__
#include <cuda_runtime.h>
#endif

namespace rt
{
#ifdef WITH_CUDA
#endif
} // namespace rt
