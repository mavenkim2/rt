#include <assert.h>
#include <stdio.h>
#include "../base.h"
#include "helper_math.h"
#include "../thread_context.h"

#define PI             3.14159265358979323846f
#define MAX_COMPONENTS 32
#define WARP_SIZE      32

static_assert(MAX_COMPONENTS <= WARP_SIZE, "too many max components");

__device__ void atomicAdd(float3 *a, float3 b)
{
    atomicAdd(&a->x, b.x);
    atomicAdd(&a->y, b.y);
    atomicAdd(&a->z, b.z);
}

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

template <>
__device__ float3 WarpReduceSum(float3 val)
{
    val.x = WarpReduceSum(val.x);
    val.y = WarpReduceSum(val.y);
    val.z = WarpReduceSum(val.z);
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

template <typename T>
__device__ T WarpReduceSumBroadcast(T val)
{
    val = WarpReduceSum(val);
    return WarpReadLaneFirst(val);
}

inline __device__ void BuildOrthonormalBasis(const float3 &n, float3 &t0, float3 &t1)
{
    const float s = n.z >= 0.f ? 1.f : -1.f;
    const float a = -1.0 / (s + n.z);
    const float b = n.x * n.y * a;

    t0 = make_float3(1.f + n.x * n.x * a * s, s * b, -s * n.x);
    t1 = make_float3(b, s + n.y * n.y * a, -n.y);
}

inline __device__ float3 Map2DTo3D(float2 vec2D)
{
    float3 vec3D = make_float3(0.f);
    float norm2  = vec2D.x * vec2D.x + vec2D.y * vec2D.y;
    float length = norm2 > 0.f ? sqrt(norm2) : 0.f;
    float sinc   = length > FLT_EPSILON ? sin(length) / length : 0.f;

    vec3D.x = length > 0.0f ? vec2D.x * sinc : vec3D.x;
    vec3D.y = length > 0.0f ? vec2D.y * sinc : vec3D.y;
    vec3D.z = cos(length);

    return vec3D;
}

struct VMM
{
    float kappas[MAX_COMPONENTS];
    float directionX[MAX_COMPONENTS];
    float directionY[MAX_COMPONENTS];
    float directionZ[MAX_COMPONENTS];
    float weights[MAX_COMPONENTS];

    uint32_t numComponents;

    inline __device__ float3 ReadDirection(uint32_t componentIndex) const
    {
        float3 dir;
        dir.x = directionX[componentIndex];
        dir.y = directionY[componentIndex];
        dir.z = directionZ[componentIndex];
        return dir;
    }

    inline __device__ void WriteDirection(uint32_t componentIndex, float3 dir)
    {
        directionX[componentIndex] = dir.x;
        directionY[componentIndex] = dir.y;
        directionZ[componentIndex] = dir.z;
    }

    __device__ void UpdateComponent(uint32_t componentIndex, float kappa, float3 dir,
                                    float weight)
    {
        kappas[componentIndex] = kappa;
        WriteDirection(componentIndex, dir);
        weights[componentIndex] = weight;
    }
};

inline __device__ float VMFProduct(float kappa0, float normalization0, float3 meanDirection0,
                                   float kappa1, float normalization1, float3 meanDirection1,
                                   float &productKappa, float &productNormalization,
                                   float3 &productMeanDirection)
{
    productMeanDirection = kappa0 * meanDirection0 + kappa1 * meanDirection1;
    productKappa         = sqrt(dot(productMeanDirection, productMeanDirection));

    productNormalization      = 1.f / (4.f * PI);
    float productEMinus2Kappa = 1.f;
    if (productKappa > 1e-3f)
    {
        productEMinus2Kappa  = __expf(-2.0f * productKappa);
        productNormalization = productKappa / (2.f * PI * (1.0f - productEMinus2Kappa));
        productMeanDirection /= productKappa;
    }
    else
    {
        productKappa         = 0.0f;
        productMeanDirection = meanDirection0;
    }

    float scale     = (normalization0 * normalization1) / productNormalization;
    float cosTheta0 = dot(meanDirection0, productMeanDirection);
    float cosTheta1 = dot(meanDirection1, productMeanDirection);

    scale *= __expf(kappa0 * (cosTheta0 - 1.0f) + kappa1 * (cosTheta1 - 1.0f));

    return scale;
}

struct Statistics
{
    float weightedLogLikelihood;
    float sumWeights[MAX_COMPONENTS];
    float3 sumWeightedDirections[MAX_COMPONENTS];

    uint32_t numSamples;
};

__device__ float CalculateVMFNormalization(float kappa)
{
    float eMinus2Kappa = exp(-2.f * kappa);
    float norm         = kappa / (2.f * PI * (1.f - eMinus2Kappa));
    norm               = kappa > 0.f ? norm : 1.f / (4 * PI);
    return norm;
}

__device__ float KappaToMeanCosine(float kappa)
{
    float meanCosine = 1.f / tanh(kappa) - 1.f / kappa;
    return kappa > 0.f ? meanCosine : 0.f;
}

__device__ float MeanCosineToKappa(float meanCosine)
{
    const float meanCosine2 = meanCosine * meanCosine;
    return (meanCosine * 3.f - meanCosine * meanCosine2) / (1.f - meanCosine2);
}

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

        vmm.WriteDirection(n, uniformDirection);
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
    }
}

__global__ void AssignSamples(uint32_t *vmmOffsets, uint32_t *vmmCounts, uint32_t numVMMs,
                              uint32_t numSamples)
{
    assert(numSamples >= numVMMs);
    uint32_t samplesPerVMM = numSamples / numVMMs;
    uint32_t count         = 0;
    for (uint i = 0; i < numVMMs; i++)
    {
        vmmOffsets[i] = count;
        vmmCounts[i]  = samplesPerVMM;
        count += samplesPerVMM;
    }
}

inline __device__ float SoftAssignment(const VMM &vmm, float3 sampleDirection,
                                       float *softAssignment)
{
    float V = 0.f;

    for (uint32_t componentIndex = 0; componentIndex < vmm.numComponents; componentIndex++)
    {
        float cosTheta = dot(sampleDirection, vmm.ReadDirection(componentIndex));
        // TODO: precompute in shared memory
        float norm = CalculateVMFNormalization(vmm.kappas[componentIndex]);
        float v    = norm * __expf(vmm.kappas[componentIndex] * min(cosTheta - 1.f, 0.f));
        softAssignment[componentIndex] = vmm.weights[componentIndex] * v;

        V += softAssignment[componentIndex];
    }

    if (V <= 1e-16f) V = 0.f;

    return V;
}

template <bool masked = false>
inline __device__ void UpdateMixtureParameters(VMM &sharedVMM, Statistics &sharedStatistics,
                                               Statistics &previousStatistics,
                                               const float3 *sampleDirections,
                                               float &sumSampleWeights, uint32_t sampleOffset,
                                               uint32_t sampleCount, uint32_t mask = 0)
{
    assert(!masked || (masked && mask != 0));

    const uint32_t maxNumIterations     = 100;
    const float weightPrior             = 0.01f;
    const float meanCosinePrior         = 0.f;
    const float meanCosinePriorStrength = 0.2f;
    const float maxKappa                = 32000.f;
    const float maxMeanCosine           = KappaToMeanCosine(maxKappa);
    const float convergenceThreshold    = 0.005f;
    const uint32_t numSampleBatches     = (sampleCount + blockDim.x - 1) / blockDim.x;

    float previousLogLikelihood = 0.f;

    const uint32_t threadIndex = threadIdx.x;
    for (uint32_t iteration = 0; iteration < maxNumIterations; iteration++)
    {
        if (threadIndex < sharedVMM.numComponents)
        {
            sharedStatistics.sumWeights[threadIndex]            = 0.f;
            sharedStatistics.sumWeightedDirections[threadIndex] = make_float3(0.f);
        }
        if (threadIndex == 0) sharedStatistics.weightedLogLikelihood = 0.f;

        __syncthreads();

        // Weighted Expectation
        for (uint32_t batch = 0; batch < numSampleBatches; batch++)
        {
            Statistics statistics;
            uint32_t sampleIndex   = threadIndex + blockDim.x * batch;
            bool hasData           = sampleIndex < sampleCount;
            float V                = 0.f;
            float3 sampleDirection = make_float3(0.f);

            if (hasData)
            {
                sampleDirection = sampleDirections[sampleIndex + sampleOffset];
                V = SoftAssignment(sharedVMM, sampleDirection, statistics.sumWeights);
            }

            hasData            = V > 0.f;
            float invV         = hasData ? 1.f / V : 0.f;
            float sampleWeight = 1.f;

            statistics.weightedLogLikelihood = hasData ? sampleWeight * log(V) : 0.f;

            // TODO IMPORTANT: test accumulating in registers and then reduce at the end,
            // instead of reducing in the loop
            for (uint32_t i = 0; i < sharedVMM.numComponents; i++)
            {
                statistics.sumWeights[i] = hasData ? statistics.sumWeights[i] * invV : 0.f;

                float softAssignmentWeight = sampleWeight * statistics.sumWeights[i];
                float3 sumWeightedDirection =
                    sampleDirection * sampleWeight * statistics.sumWeights[i];

                float sumWeights             = WarpReduceSum(softAssignmentWeight);
                float3 sumWeightedDirections = WarpReduceSum(sumWeightedDirection);

                if ((threadIndex & (WARP_SIZE - 1)) == 0)
                {
                    atomicAdd(&sharedStatistics.sumWeights[i], sumWeights);
                    atomicAdd(&sharedStatistics.sumWeightedDirections[i],
                              sumWeightedDirections);
                }
            }

            float weightedLogLikelihood = WarpReduceSum(statistics.weightedLogLikelihood);
            if ((threadIndex & (WARP_SIZE - 1)) == 0)
            {
                // printf("log like: %f\n", weightedLogLikelihood);
                atomicAdd(&sharedStatistics.weightedLogLikelihood, weightedLogLikelihood);
            }
        }

        __syncthreads();

        // Weighted MAP Update
        if (threadIndex < MAX_COMPONENTS)
        {
            // NOTE: if MAX_COMPONENTS > 32, this no longer works
            // Normalize
            float componentWeight = sharedStatistics.sumWeights[threadIndex];
            float normWeight      = WarpReduceSumBroadcast(componentWeight);

            if (threadIndex == 0) sumSampleWeights = normWeight;

            normWeight = normWeight > FLT_EPSILON ? float(sampleCount) / normWeight : 0.f;
            sharedStatistics.sumWeights[threadIndex] *= normWeight;

            // Update weights
            const uint32_t totalNumSamples = sampleCount + previousStatistics.numSamples;
            float weight                   = sharedStatistics.sumWeights[threadIndex] +
                           previousStatistics.sumWeights[threadIndex];
            weight = threadIndex >= sharedVMM.numComponents
                         ? 0.f
                         : (weightPrior + weight) /
                               (weightPrior * sharedVMM.numComponents + sampleCount);

            const bool threadIsEnabled = mask & (1u << threadIndex);

            float sumMaskedWeights   = threadIsEnabled ? weight : 0.f;
            float sumUnmaskedWeights = threadIsEnabled ? 0.f : weight;

            if (!masked || threadIsEnabled)
            {
                sharedVMM.weights[threadIndex] = weight;
            }

            if constexpr (masked)
            {
                // Makes sure weights always sum to one
                float invSumWeights = 1.f / WarpReduceSumBroadcast(sumMaskedWeights);
                invSumWeights *= 1.f - WarpReduceSumBroadcast(sumUnmaskedWeights);
                sharedVMM.weights[threadIndex] *= invSumWeights;
            }

            // Update kappas and directions
            if (!masked || threadIsEnabled)
            {
                const float currentEstimationWeight  = float(sampleCount) / totalNumSamples;
                const float previousEstimationWeight = 1.f - currentEstimationWeight;

                float3 currentMeanDirection =
                    sharedStatistics.sumWeights[threadIndex] > 0.f
                        ? sharedStatistics.sumWeightedDirections[threadIndex] /
                              sharedStatistics.sumWeights[threadIndex]
                        : make_float3(0.f);
                float3 previousMeanDirection =
                    previousStatistics.sumWeights[threadIndex] > 0.f
                        ? previousStatistics.sumWeightedDirections[threadIndex] /
                              previousStatistics.sumWeights[threadIndex]
                        : make_float3(0.f);

                float3 meanDirection = currentMeanDirection * currentEstimationWeight +
                                       previousMeanDirection * previousEstimationWeight;

                float meanCosine = length(meanDirection);

                // TODO: make sure uninitialized components have correct default state?
                if (meanCosine > 0.f)
                {
                    sharedVMM.WriteDirection(threadIndex, meanDirection / meanCosine);
                }

                float partialNumSamples = totalNumSamples * sharedVMM.weights[threadIndex];

                meanCosine = (meanCosinePrior * meanCosinePriorStrength +
                              meanCosine * partialNumSamples) /
                             (meanCosinePriorStrength + partialNumSamples);
                meanCosine                    = min(meanCosine, maxMeanCosine);
                float kappa                   = threadIndex < sharedVMM.numComponents
                                                    ? MeanCosineToKappa(meanCosine)
                                                    : 0.f;
                sharedVMM.kappas[threadIndex] = kappa;
            }
        }

        __syncthreads();

        // Convergence check
        if (iteration > 0)
        {
            float logLikelihood = sharedStatistics.weightedLogLikelihood;
            float relLogLikelihoodDifference =
                fabs(logLikelihood - previousLogLikelihood) / fabs(previousLogLikelihood);

            if (relLogLikelihoodDifference < convergenceThreshold) break;
            previousLogLikelihood = logLikelihood;
        }
    }
}

// TODO IMPORTANT:
// - handle unassigned components in MAP update
// - make sure to update statistics and rewrite to memory
// - make sure to get chi square statistics fromm previous update
// - make sure to load sample weights and pdfs for chi square splitting
// - proper criteria for splitting/merging
// - make sure to test performance of atomics vs block reduce, different block sizes
__global__ void UpdateMixture(const VMM *__restrict__ vmms,
                              Statistics *__restrict__ previousStatisticsArray,
                              const float3 *__restrict__ sampleDirections,
                              const uint32_t *__restrict__ vmmOffsets,
                              const uint32_t *__restrict__ vmmCounts)
{
    __shared__ Statistics sharedStatistics;
    __shared__ Statistics previousStatistics;
    __shared__ VMM sharedVMM;
    __shared__ float sumSampleWeights;

    uint32_t threadIndex = threadIdx.x;
    uint32_t vmmIndex    = blockIdx.x;
    uint32_t sampleCount = vmmCounts[vmmIndex];
    uint32_t offset      = vmmOffsets[vmmIndex];

    if (threadIndex == 0)
    {
        sharedVMM = vmms[vmmIndex];
        for (uint32_t i = 0; i < sharedVMM.numComponents; i++)
        {
            sharedStatistics.sumWeights[i]              = 0.f;
            sharedStatistics.sumWeightedDirections[i]   = make_float3(0.f);
            previousStatistics.sumWeights[i]            = 0.f;
            previousStatistics.sumWeightedDirections[i] = make_float3(0.f);
        }
        sharedStatistics.weightedLogLikelihood   = 0.f;
        previousStatistics.weightedLogLikelihood = 0.f;
        previousStatistics.numSamples            = 0;
    }

    __syncthreads();

    const float maxKappa           = 32000.f;
    const float maxMeanCosine      = KappaToMeanCosine(maxKappa);
    const float splittingThreshold = 0.5f;

    const uint32_t numSampleBatches = (sampleCount + blockDim.x - 1) / blockDim.x;

    UpdateMixtureParameters(sharedVMM, sharedStatistics, previousStatistics, sampleDirections,
                            sumSampleWeights, offset, sampleCount);

    // Splitting
    __shared__ float3 sharedCovarianceTotals[MAX_COMPONENTS];
    __shared__ float sharedChiSquareTotals[MAX_COMPONENTS];
    __shared__ float sharedSumWeights[MAX_COMPONENTS];

    {
        const uint32_t totalNumSamples = previousStatistics.numSamples + sampleCount;
        const float mcEstimate         = sumSampleWeights / float(totalNumSamples);
        uint32_t numUsedSamples        = 0;

        // TODO IMPORTANT: these totals need to include totals from previous
        // update passes

        if (threadIndex < MAX_COMPONENTS)
        {
            sharedCovarianceTotals[threadIndex] = make_float3(0.f);
            sharedChiSquareTotals[threadIndex]  = 0.f;
            sharedSumWeights[threadIndex]       = 0.f;
        }

        __syncthreads();

        // Update split statistics
        for (uint32_t batch = 0; batch < numSampleBatches; batch++)
        {
            Statistics statistics;
            uint32_t sampleIndex   = threadIndex + blockDim.x * batch;
            bool hasData           = sampleIndex < sampleCount;
            float V                = 0.f;
            float3 sampleDirection = make_float3(0.f);

            if (hasData)
            {
                sampleDirection = sampleDirections[sampleIndex + offset];
                V = SoftAssignment(sharedVMM, sampleDirection, statistics.sumWeights);
            }

            // TODO: load from memory
            float sampleWeight = 1.f;
            float samplePDF    = 1.f;
            float sampleLi     = 1.f;

            hasData    = V > 0.f;
            float invV = V > 0.f ? 1.f / V : 0.f;
            numUsedSamples += hasData;

            // See equations 21 and 22 in Robust Fitting of Parallax-Aware Mixtures for
            // Path Guiding. Note that this evaluates p^2/q - 2p + q, while the paper only
            // evaluates p^2/q
            for (uint32_t componentIndex = 0; componentIndex < sharedVMM.numComponents;
                 componentIndex++)
            {
                float vmfPdf          = hasData ? statistics.sumWeights[componentIndex] : 0.f;
                float partialValuePDF = (vmfPdf * sampleLi) / (mcEstimate * V);

                float chiSquareEstimate =
                    (vmfPdf * sampleLi * sampleLi) / (V * V * mcEstimate * mcEstimate);
                chiSquareEstimate -= 2.f * partialValuePDF;
                chiSquareEstimate += vmfPdf;
                chiSquareEstimate /= samplePDF;
                chiSquareEstimate = hasData ? chiSquareEstimate : 0.f;

                // Calculate covariances
                float3 t0, t1;
                float3 vmmComponentDir = sharedVMM.ReadDirection(componentIndex);
                BuildOrthonormalBasis(vmmComponentDir, t0, t1);
                float3 localDirection =
                    make_float3(dot(t0, sampleDirection), dot(t1, sampleDirection),
                                dot(vmmComponentDir, sampleDirection));

                float softAssignmentWeight = vmfPdf * invV;
                float assignedWeight       = softAssignmentWeight * sampleWeight;

                float covXX = assignedWeight * localDirection.x * localDirection.x;
                float covYY = assignedWeight * localDirection.y * localDirection.y;
                float covXY = assignedWeight * localDirection.x * localDirection.y;
                float3 cov  = make_float3(covXX, covYY, covXY);

                // TODO IMPORTANT: save as above
                chiSquareEstimate = WarpReduceSum(chiSquareEstimate);
                cov               = WarpReduceSum(cov);
                assignedWeight    = WarpReduceSum(assignedWeight);

                if ((threadIndex & (WARP_SIZE - 1)) == 0)
                {
                    atomicAdd(&sharedChiSquareTotals[componentIndex], chiSquareEstimate);
                    atomicAdd(&sharedCovarianceTotals[componentIndex], cov);
                    atomicAdd(&sharedSumWeights[componentIndex], assignedWeight);
                }
            }
        }

        __shared__ uint32_t sharedSplitMask;
        // Split components
        if (threadIndex < MAX_COMPONENTS)
        {
            numUsedSamples = WarpReduceSum(numUsedSamples);
            numUsedSamples = WarpReadLaneFirst(numUsedSamples);

            float chiSquareEstimate               = sharedChiSquareTotals[threadIndex];
            uint32_t rank                         = 0;
            const uint32_t numAvailableComponents = MAX_COMPONENTS - sharedVMM.numComponents;
            for (uint32_t i = 0; i < MAX_COMPONENTS; i++)
            {
                float chiSquare    = sharedChiSquareTotals[i];
                uint32_t increment = chiSquare > chiSquareEstimate ||
                                     (chiSquare == chiSquareEstimate && i < threadIndex);
                rank += increment;
            }

            bool split = sharedChiSquareTotals[threadIndex] > splittingThreshold &&
                         rank < numAvailableComponents;

            uint32_t splitMask     = __ballot_sync(0xffffffff, split);
            uint32_t numComponents = sharedVMM.numComponents;
            uint32_t newComponentIndex =
                numComponents + __popc(splitMask & ((1u << threadIndex) - 1));

            uint componentIndex = threadIndex;

            if (split)
            {
                assert(newComponentIndex < MAX_COMPONENTS);

                float sumWeights = sharedSumWeights[componentIndex];
                float3 cov       = sharedCovarianceTotals[componentIndex] / sumWeights;

                float negB = cov.x + cov.y;
                float discriminant =
                    sqrt((cov.x - cov.y) * (cov.x - cov.y) - 4 * cov.z * cov.z);
                float eigenValue0   = 0.5f * (negB + discriminant);
                float2 eigenVector0 = make_float2(-cov.z, cov.x - eigenValue0);

                float norm0 = dot(eigenVector0, eigenVector0);
                norm0       = norm0 > FLT_EPSILON ? rsqrt(norm0) : 1.f;
                eigenVector0 *= norm0;

                if (discriminant > 1e-8f)
                {
                    float2 temp     = eigenValue0 * eigenVector0 * 0.5f;
                    float2 meanDir0 = temp;
                    float2 meanDir1 = -temp;

                    float3 meanDirection = sharedVMM.ReadDirection(componentIndex);

                    float3 basis0, basis1;
                    BuildOrthonormalBasis(meanDirection, basis0, basis1);

                    float3 meanDir3D0 = Map2DTo3D(meanDir0);
                    float3 meanDir3D1 = Map2DTo3D(meanDir1);

                    float3 meanDirection0 = basis0 * meanDir3D0.x + basis1 * meanDir3D0.y +
                                            meanDirection * meanDir3D0.z;
                    float3 meanDirection1 = basis0 * meanDir3D1.x + basis1 * meanDir3D1.y +
                                            meanDirection * meanDir3D1.z;

                    float meanCosine  = KappaToMeanCosine(sharedVMM.kappas[componentIndex]);
                    float meanCosine0 = meanCosine / abs(dot(meanDirection0, meanDirection));
                    meanCosine0       = min(meanCosine0, maxMeanCosine);

                    float kappa  = MeanCosineToKappa(meanCosine0);
                    float weight = sharedVMM.weights[componentIndex] * 0.5f;

                    sharedVMM.UpdateComponent(componentIndex, kappa, meanDirection0, weight);
                    sharedVMM.UpdateComponent(newComponentIndex, kappa, meanDirection1,
                                              weight);

                    // TODO: update and split statistics
                }
            }

            if (threadIndex == 0)
            {
                sharedVMM.numComponents += __popc(splitMask);
                sharedSplitMask = splitMask;
            }
        }

        __syncthreads();

        if (threadIndex < MAX_COMPONENTS)
        {
            // printf("thread: %u, %f %f %f %f %f\n", threadIndex,
            // sharedVMM.kappas[threadIndex],
            //        sharedVMM.directions[threadIndex].x, sharedVMM.directions[threadIndex].y,
            //        sharedVMM.directions[threadIndex].z, sharedVMM.weights[threadIndex]);
        }

        // Update split components
        UpdateMixtureParameters<true>(sharedVMM, sharedStatistics, previousStatistics,
                                      sampleDirections, sumSampleWeights, offset, sampleCount,
                                      sharedSplitMask);
    }

    // Reuse shared memory
    float *mergeKappas      = sharedSumWeights;
    float *mergeProducts    = sharedChiSquareTotals;
    float3 *mergeDirections = sharedCovarianceTotals;

    // Merging
    {
        const uint32_t numComponents   = sharedVMM.numComponents;
        const uint32_t numPairs        = (numComponents * numComponents - numComponents) / 2;
        const uint32_t numMergeBatches = (numPairs + blockDim.x - 1) / blockDim.x;

        if (threadIndex < sharedVMM.numComponents)
        {
            float kappa          = sharedVMM.kappas[threadIndex];
            float normalization  = CalculateVMFNormalization(kappa);
            float3 meanDirection = sharedVMM.ReadDirection(threadIndex);

            float productKappa, productNormalization;
            float3 productMeanDirection;
            float scale = VMFProduct(kappa, normalization, meanDirection, kappa, normalization,
                                     meanDirection, productKappa, productNormalization,
                                     productMeanDirection);

            mergeKappas[threadIndex]     = productKappa;
            mergeProducts[threadIndex]   = scale;
            mergeDirections[threadIndex] = productMeanDirection;
        }

        __syncthreads();

        for (uint32_t batch = 0; batch < numMergeBatches; batch++)
        {
            // 0  1  2  3  4  5  6  7  8
            // 1  9 10 11 12 13 14 15 16
            // 2 10 17 18 19 20 21 22 23
            // 3 11 18 24 25 26 27 28 29

            // TODO: instead of recomputing the chi square estimate for
            // every merge iteration, cache in shared memory and update only the pairs
            // for the components that have changed

            // index % components < index / components ? : index - index / components

            // Calculate the merge cost for the requisite pair
            const uint32_t componentIndex0 = 0;
            const uint32_t componentIndex1 = 0;

            float kappa0          = sharedVMM.kappas[componentIndex0];
            float normalization0  = CalculateVMFNormalization(kappa0);
            float kappa1          = sharedVMM.kappas[componentIndex1];
            float normalization1  = CalculateVMFNormalization(kappa1);
            float weight0         = sharedVMM.weights[componentIndex0];
            float weight1         = sharedVMM.weights[componentIndex1];
            float3 meanDirection0 = sharedVMM.ReadDirection(componentIndex0);
            float3 meanDirection1 = sharedVMM.ReadDirection(componentIndex1);

            float weight00 = weight0 * weight0;

            float productKappa, productNormalization;
            float3 productMeanDirection;

            float scale00 = mergeProducts[componentIndex0];
            float scale11 = mergeProducts[componentIndex1];
            float scale01 = VMFProduct(kappa0, normalization0, meanDirection0, kappa1,
                                       normalization1, meanDirection1, productKappa,
                                       productNormalization, productMeanDirection);

            // is this even worth it? like this is a non trivial amount of effort.
            // like it's not hard but it's not free. plus there's a potential for errors
            // over just copying the code dumbly.
        }
    }
}

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
    uint32_t numVMMs    = 4;
    uint32_t numSamples = numVMMs * WARP_SIZE;

    CUDAArena allocator;
    allocator.Init(megabytes(1));

    printf("hello world\n");
    VMM *vmms                 = allocator.Alloc<VMM>(numVMMs, 4);
    float3 *sampleDirections  = allocator.Alloc<float3>(numSamples, 4);
    uint32_t *vmmOffsets      = allocator.Alloc<uint32_t>(numVMMs, 4);
    uint32_t *vmmCounts       = allocator.Alloc<uint32_t>(numSamples, 4);
    Statistics *vmmStatistics = allocator.Alloc<Statistics>(numVMMs, 4);

    // Initialization
    InitializeVMMS<<<1, numVMMs>>>(vmms, numVMMs);
    InitializeSamples<<<1, numSamples>>>(sampleDirections, numSamples);
    AssignSamples<<<1, 1>>>(vmmOffsets, vmmCounts, numVMMs, numSamples);

    UpdateMixture<<<numVMMs, WARP_SIZE>>>(vmms, vmmStatistics, sampleDirections, vmmOffsets,
                                          vmmCounts);

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
