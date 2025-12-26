#include <stdio.h>
#include "../base.h"
#include "helper_math.h"
#include "../thread_context.h"

#define PI             3.14159265358979323846f
#define MAX_COMPONENTS 32
#define WARP_SIZE      32

namespace rt
{

struct RNG
{
    __device__ static uint PCG(uint x)
    {
        uint state = x * 747796405u + 2891336453u;
        uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    // Ref: M. Jarzynski and M. Olano, "Hash Functions for GPU Rendering," Journal of Computer
    // Graphics Techniques, 2020.
    static uint3 PCG3d(uint3 v)
    {
        v = v * 1664525u + 1013904223u;
        v.x += v.y * v.z;
        v.y += v.z * v.x;
        v.z += v.x * v.y;
        v ^= v >> 16u;
        v.x += v.y * v.z;
        v.y += v.z * v.x;
        v.z += v.x * v.y;
        return v;
    }

    // Ref: M. Jarzynski and M. Olano, "Hash Functions for GPU Rendering," Journal of Computer
    // Graphics Techniques, 2020.
    static uint4 PCG4d(uint4 v)
    {
        v = v * 1664525u + 1013904223u;
        v.x += v.y * v.w;
        v.y += v.z * v.x;
        v.z += v.x * v.y;
        v.w += v.y * v.z;
        v ^= v >> 16u;
        v.x += v.y * v.w;
        v.y += v.z * v.x;
        v.z += v.x * v.y;
        v.w += v.y * v.z;
        return v;
    }

    static RNG Init(uint2 pixel, uint frame)
    {
        RNG rng;
#if 0
        rng.State = RNG::PCG(pixel.x + RNG::PCG(pixel.y + RNG::PCG(frame)));
#else
        rng.State = RNG::PCG3d(make_uint3(pixel, frame)).x;
#endif

        return rng;
    }

    static RNG Init(uint2 pixel, uint frame, uint idx)
    {
        RNG rng;
        rng.State = RNG::PCG4d(make_uint4(pixel, frame, idx)).x;

        return rng;
    }

    __device__ static RNG Init(uint idx, uint frame)
    {
        RNG rng;
        rng.State = rng.PCG(idx + PCG(frame));

        return rng;
    }

    static RNG Init(uint seed)
    {
        RNG rng;
        rng.State = seed;

        return rng;
    }

    __device__ uint UniformUint()
    {
        this->State = this->State * 747796405u + 2891336453u;
        uint word = ((this->State >> ((this->State >> 28u) + 4u)) ^ this->State) * 277803737u;

        return (word >> 22u) ^ word;
    }

    __device__ float Uniform()
    {
#if 0
    	return asfloat(0x3f800000 | (UniformUint() >> 9)) - 1.0f;
#else
        // For 32-bit floats, any integer in [0, 2^24] can be represented exactly and
        // there may be rounding errors for anything larger, e.g. 2^24 + 1 is rounded
        // down to 2^24.
        // Given random integers, we can right shift by 8 bits to get integers in
        // [0, 2^24 - 1]. After division by 2^-24, we have uniform numbers in [0, 1).
        // Ref: https://prng.di.unimi.it/
        return float(UniformUint() >> 8) * 0x1p-24f;
#endif
    }

    // Returns samples in [0, bound)
    __device__ uint UniformUintBounded(uint bound)
    {
        uint32_t threshold = (~bound + 1u) % bound;

        for (;;)
        {
            uint32_t r = UniformUint();

            if (r >= threshold) return r % bound;
        }
    }

    // Returns samples in [0, bound). Biased but faster than #UniformUintBounded():
    // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
    __device__ uint UniformUintBounded_Faster(uint bound)
    {
        return (uint)(Uniform() * float(bound));
    }

    __device__ float2 Uniform2D()
    {
        float u0 = Uniform();
        float u1 = Uniform();

        return make_float2(u0, u1);
    }

    __device__ float3 Uniform3D()
    {
        float u0 = Uniform();
        float u1 = Uniform();
        float u2 = Uniform();

        return make_float3(u0, u1, u2);
    }

    __device__ float4 Uniform4D()
    {
        float u0 = Uniform();
        float u1 = Uniform();
        float u2 = Uniform();
        float u3 = Uniform();

        return make_float4(u0, u1, u2, u3);
    }

    uint State;
};

__device__ float3 SampleUniformSphere(float2 u)
{
    float z   = 1 - 2 * u.x;
    float r   = sqrt(1 - z * z);
    float phi = 2 * PI * u.y;
    return make_float3(r * cos(phi), r * sin(phi), z);
}

template <typename T>
__device__ T WarpReduceSum(T val)
{
#pragma unroll
    for (int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__device__ T WarpReadLaneAt(T val, uint32_t thread)
{
    val = __shfl_sync(0xffffffff, val, thread);
    return val;
}

template <typename T>
__device__ T WarpReadLaneFirst(T val)
{
    val = __shfl_sync(0xffffffff, val, 0);
    return val;
}

struct VMM
{
    float kappas[MAX_COMPONENTS];
    float3 directions[MAX_COMPONENTS];
    float weights[MAX_COMPONENTS];

    uint32_t numComponents;
};

struct Statistics
{
    float weightedLogLikelihood;
    float sumWeights[MAX_COMPONENTS];
    float3 sumWeightedDirections[MAX_COMPONENTS];
};

__device__ float CalculateVMFNormalization(float kappa)
{
    float eMinus2Kappa = exp(-2.f * kappa);
    float norm         = kappa / (2.f * PI * (1.f - eMinus2Kappa));
    norm               = kappa > 0.f ? norm : 1.f / (4 * PI);
    return norm;
}

float KappaToMeanCosine(float kappa)
{
    float meanCosine = 1.f / tanh(kappa) - 1.f / kappa;
    return kappa > 0.f ? meanCosine : 0.f;
}

float MeanCosineToKappa(float meanCosine)
{
    const float meanCosine2 = meanCosine * meanCosine;
    return (meanCosine * 3.f - meanCosine * meanCosine2) / (1.f - meanCosine2);
}

__global__ void helloWorld() { printf("hello world\n"); }

__global__ void InitializeVMMS(VMM *vmms, uint32_t numVMMs)
{
    uint32_t vmmIndex = threadIdx.x;
    if (vmmIndex >= numVMMs) return;
    const float gr               = 1.618033988749895f;
    const uint32_t numComponents = MAX_COMPONENTS / 2;

    VMM vmm;
    vmm.numComponents = numComponents;

    const float weight = 1.f / float(numComponents);
    const float kappa  = 5.f;
    uint32_t l         = numComponents - 1;
    for (uint32_t n = 0; n < MAX_COMPONENTS; n++)
    {
        float3 uniformDirection;
        if (n < l + 1)
        {
            float phi      = 2.0f * ((float)n / gr);
            float z        = 1.0f - ((2.0f * n + 1.0f) / float(l + 1));
            float sinTheta = sqrt(1.f - min(z * z, 1.f));

            // cos(theta) = z
            // sin(theta) = sin(arccos(z)) = sqrt(1 - z^2)
            float3 mu        = make_float3(sinTheta * cos(phi), sinTheta * sin(phi), z);
            uniformDirection = mu;
        }
        else
        {
            uniformDirection = make_float3(0, 0, 1);
        }

        vmm.directions[n] = uniformDirection;
        if (n < numComponents)
        {
            vmm.kappas[n]  = kappa;
            vmm.weights[n] = weight;
        }
        else
        {
            vmm.kappas[n]  = 0.0f;
            vmm.weights[n] = 0.0f;
            // vmm.normalizations[i][j] = ONE_OVER_FOUR_PI;
            // vmm.eMinus2Kappa[i][j] = 1.0f;
            // vmm._meanCosines[i][j] = 0.0f;
        }
    }

    vmms[vmmIndex] = vmm;
    printf("%f %f %f\n", vmm.directions[0].x, vmm.directions[0].y, vmm.directions[0].z);
}

__global__ void InitializeSamples(float3 *__restrict__ sampleDirections, uint32_t numSamples)
{
    uint32_t sampleIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (sampleIndex < numSamples)
    {
        RNG rng                       = RNG::Init(sampleIndex, numSamples);
        float2 u                      = rng.Uniform2D();
        float3 dir                    = SampleUniformSphere(u);
        sampleDirections[sampleIndex] = dir;
        printf("sampleindex: %u, dir: %f %f %f\n", sampleIndex, u.x, u.y, dir.z);
    }
}

__global__ void WeightedExpectation(const VMM *__restrict__ vmms,
                                    const float3 *__restrict__ sampleDirections,
                                    const uint32_t *__restrict__ vmmOffsets,
                                    const uint32_t *__restrict__ vmmCounts,
                                    uint32_t totalNumSamples)
{
    __shared__ Statistics statistics_;

    uint32_t threadIndex = threadIdx.x;
    uint32_t vmmIndex    = blockIdx.x;
    uint32_t sampleCount = vmmCounts[vmmIndex];
    uint32_t offset      = vmmOffsets[vmmIndex];

    __shared__ VMM vmm;

    if (threadIndex == 0)
    {
        vmm = vmms[vmmIndex];
        for (uint32_t i = 0; i < vmm.numComponents; i++)
        {
            statistics_.sumWeights[i] = 0.f;
        }
        statistics_.weightedLogLikelihood = 0.f;
    }

    __syncthreads();

    Statistics statistics;
    const uint32_t numIters = (sampleCount + blockDim.x - 1) / blockDim.x;

    for (uint32_t sampleIndex = threadIndex; sampleIndex < sampleCount;
         sampleIndex += blockDim.x)
    {
        float3 sampleDirection = sampleDirections[sampleIndex + offset];
        float V                = 0.f;

        for (uint32_t componentIndex = 0; componentIndex < vmm.numComponents; componentIndex++)
        {
            float cosTheta = dot(sampleDirection, vmm.directions[componentIndex]);
            float norm     = CalculateVMFNormalization(vmm.kappas[componentIndex]);
            float v = norm * __expf(vmm.kappas[componentIndex] * min(cosTheta - 1.f, 0.f));
            statistics.sumWeights[componentIndex] = vmm.weights[componentIndex] * v;

            V += statistics.sumWeights[componentIndex];
        }
        // TODO: what do I do here?
        if (V <= 1e-16f) continue;

        float invV = 1.f / V;
        for (uint32_t i = 0; i < vmm.numComponents; i++)
        {
            statistics.sumWeights[i] *= invV;
        }
        statistics.weightedLogLikelihood = log(V); // * sampleWeight
    }

    // __shared__
    for (uint32_t componentIndex = 0; componentIndex < vmm.numComponents; componentIndex++)
    {
        float softAssignmentWeight  = statistics.sumWeights[componentIndex];
        float sumWeights            = WarpReduceSum(softAssignmentWeight);
        float weightedLogLikelihood = WarpReduceSum(statistics.weightedLogLikelihood);

        if ((threadIndex & (WARP_SIZE - 1)) == 0)
        {
            atomicAdd(&statistics_.sumWeights[componentIndex], sumWeights);
            // statistics_.sumWeights[componentIndex] += sumWeights;
            // statistics_.weightedLogLikelihood += weightedLogLikelihood;
        }
    }

    __syncthreads();

    // Normalize
    if (threadIndex < WARP_SIZE)
    {
        float componentWeight = statistics_.sumWeights[threadIndex];
        float normWeight      = WarpReduceSum(componentWeight);
        normWeight            = WarpReadLaneFirst(normWeight);
        normWeight = normWeight > FLT_EPSILON ? float(sampleCount) / normWeight : 0.f;
        statistics_.sumWeights[threadIndex] *= normWeight;
    }

    __syncthreads();

    if (threadIndex == 0)
    {
        // vmmStatistics[vmmIndex] = statistics_;
    }
}

// __global void WeightedMAP()
// {
//     // uint totalNumVMMs = num.num;
//     // if (dtID.x >= totalNumVMMs) return;
//
//     const float weightPrior = 0.01f;
//
//     uint vmmIndex         = groupID.x;
//     VMM vmm               = vmms[vmmIndex];
//     Statistics statistics = vmmStatistics[groupID.x];
//     uint numSamples       = vmmCounts[vmmIndex];
//
//     // TODO: should it be one thread per VMM instead?
//     // Update weights
//     if (groupIndex < vmm.numComponents)
//     {
//         // TODO IMPORTANT: handle previous stats
//         float weight = statistics.sumWeights[groupIndex];
//         weight       = (weightPrior + weight) / (weightPrior * vmm.numComponents +
//         numSamples);
//
//         vmms[vmmIndex].weights[groupIndex] = weight;
//     }
//     else if (groupIndex < MAX_COMPONENTS)
//     {
//         vmms[vmmIndex].weights[groupIndex] = 0.f;
//     }
//
//     // Update kappas and directions
//     uint totalNumSamples = numSamples;
//
//     const float currentEstimationWeight  = numSamples / totalNumSamples;
//     const float previousEstimationWeight = 1.f - currentEstimationWeight;
//
//     const float meanCosinePrior         = 0.f;
//     const float meanCosinePriorStrength = 0.2f;
//     const float maxKappa                = 32000.f;
//     const float maxMeanCosine           = KappaToMeanCosine(maxKappa);
//
//     if (groupIndex < vmm.numComponents)
//     {
//         float3 currentMeanDirection  = statistics.sumWeights[groupIndex] > 0.f
//                                            ? statistics.sumWeightedDirections[groupIndex] /
//                                                 statistics.sumWeights[groupIndex]
//                                            : 0.f;
//         float3 previousMeanDirection = 0.f;
//
//         float3 meanDirection = currentMeanDirection * currentEstimationWeight +
//                                previousMeanDirection * previousEstimationWeight;
//         float meanCosine = length(meanDirection);
//
//         if (meanCosine > 0.f)
//         {
//             vmms[vmmIndex].directions[groupIndex] = meanDirection / meanCosine;
//         }
//         float partialNumSamples = totalNumSamples * vmms[vmmIndex].weights[groupIndex];
//         meanCosine =
//             (meanCosinePrior * meanCosinePriorStrength + meanCosine * partialNumSamples) /
//             (meanCosinePriorStrength + partialNumSamples);
//         meanCosine                        = min(meanCosine, maxMeanCosine);
//         vmms[vmmIndex].kappas[groupIndex] = MeanCosineToKappa(meanCosine);
//     }
//     else if (groupIndex < MAX_COMPONENTS)
//     {
//         vmms[vmmIndex].directions[groupIndex] = 0.f;
//         vmms[vmmIndex].kappas[groupIndex]     = 0.f;
//     }
// }

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

void test()
{
    uint32_t numVMMs    = 32;
    uint32_t numSamples = 64;

    CUDAArena allocator;
    allocator.Init(megabytes(1));

    printf("hello world\n");
    VMM *vmms                = allocator.Alloc<VMM>(numVMMs, 4);
    float3 *sampleDirections = allocator.Alloc<float3>(numSamples, 4);
    uint32_t *vmmOffsets     = allocator.Alloc<uint32_t>(numVMMs, 4);
    uint32_t *vmmCounts      = allocator.Alloc<uint32_t>(numSamples, 4);

    InitializeVMMS<<<1, numVMMs>>>(vmms, numVMMs);
    InitializeSamples<<<1, numSamples>>>(sampleDirections, numSamples);

    // WeightedExpectation<<<1, 1>>>(vmms, const int *__restrict sampleDirections,
    //                               const int *__restrict vmmOffsets,
    //                               const int *__restrict vmmCounts, int totalNumSamples);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    // CPU execution is asynchronous, so we must wait for the GPU to finish
    // before the program exits (otherwise the printfs might get cut off).
    cudaDeviceSynchronize();
}

} // namespace rt

int main()
{
    rt::test();
    return 0;
}
