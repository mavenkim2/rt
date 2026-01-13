#ifdef __CUDACC__
#pragma warning test
__device__ void atomicAdd(float3 *a, float3 b)
{
    atomicAdd(&a->x, b.x);
    atomicAdd(&a->y, b.y);
    atomicAdd(&a->z, b.z);
}

__device__ int3 atomicAdd(int3 *a, int3 b)
{
    int3 result;
    result.x = atomicAdd(&a->x, b.x);
    result.y = atomicAdd(&a->y, b.y);
    result.z = atomicAdd(&a->z, b.z);
    return result;
}

__device__ longlong3 atomicAdd(longlong3 *a, longlong3 b)
{
    longlong3 result;
    result.x = (long long)atomicAdd((unsigned long long *)&a->x, (unsigned long long)b.x);
    result.y = (long long)atomicAdd((unsigned long long *)&a->y, (unsigned long long)b.y);
    result.z = (long long)atomicAdd((unsigned long long *)&a->z, (unsigned long long)b.z);
    return result;
}

__device__ void atomicAdd(uint3 *a, uint3 b)
{
    atomicAdd(&a->x, b.x);
    atomicAdd(&a->y, b.y);
    atomicAdd(&a->z, b.z);
}

__device__ void atomicMin(uint3 *a, uint3 b)
{
    atomicMin(&a->x, b.x);
    atomicMin(&a->y, b.y);
    atomicMin(&a->z, b.z);
}

__device__ void atomicMax(uint3 *a, uint3 b)
{
    atomicMax(&a->x, b.x);
    atomicMax(&a->y, b.y);
    atomicMax(&a->z, b.z);
}
#endif
