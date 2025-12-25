#include <stdio.h>
#include "helper_math.h"

#define PI 3.14159265358979323846f
static const float FLT_EPSILON = 1.192092896e-07f;

#define MAX_COMPONENTS 32
#define WARP_SIZE      32

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

float CalculateVMFNormalization(float kappa)
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
            float v        = norm * exp(vmm.kappas[componentIndex] * min(cosTheta - 1.f, 0.f));
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
        normWeight > FLT_EPSILON ? float(sampleCount) / normWeight : 0.f;
        statistics_.sumWeights[threadIndex] *= normWeight;
    }

    __syncthreads();

    if (threadIndex == 0)
    {
        vmmStatistics[vmmIndex] = statistics_;
    }
}

int main()
{
    printf("Hello World from CPU!\n");

    uint32_t numVMMs    = 32;
    uint32_t numSamples = 64;
    VMM *vmms;

    uint32_t totalSize = sizeof(VMM) * numVMMs;
    totalSize += sizeof(u32) * (2 * numVMMs);
    totalSize += sizeof(float3) * (2 * numVMMs);

    cudaMalloc(&vmms, sizeof(VMM) * numVMMs);
    InitializeVMMS<<<1, numVMMs>>>(vmms, numVMMs);

    WeightedExpectation<<<1, 1>>>(vmms, const int *__restrict sampleDirections,
                                  const int *__restrict vmmOffsets,
                                  const int *__restrict vmmCounts, int totalNumSamples);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    // CPU execution is asynchronous, so we must wait for the GPU to finish
    // before the program exits (otherwise the printfs might get cut off).
    cudaDeviceSynchronize();

    return 0;
}
