// Based on code from OpenPGL (https://github.com/OpenPathGuidingLibrary/openpgl)
// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Modifications
// Copyright 2026 Maven Kim
// SPDX-License-Identifier: Apache-2.0
//
// This file contains algorithms and core logic derived from OpenPGL,
// restructured and adapted for CUDA execution.

#include <assert.h>
#include <cfloat>
#include <stdio.h>
#include "../base.h"
#include "helper_math.h"
#include "../thread_context.h"
#include <type_traits>
#include "path_guiding_util.h"

#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCuda.h"

#define PI             3.14159265358979323846f
#define MAX_COMPONENTS 32
#define WARP_SIZE      32
#define WARP_SHIFT     5u

static_assert(MAX_COMPONENTS <= WARP_SIZE, "too many max components");

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

inline __device__ void GetSplitDimensionAndPosition(float3 sampleMean, float3 sampleVariance,
                                                    float3 sampleBoundExtend,
                                                    uint32_t &splitDim, float &splitPos)
{
    auto maxDimension = [](const float3 &v) -> uint32_t {
        return v.x > v.y ? (v.x > v.z ? 0 : 2) : (v.y > v.z ? 1 : 2);
    };

    if (sampleVariance.x == sampleVariance.y && sampleVariance.y == sampleVariance.z)
    {
        if (sampleBoundExtend.x == sampleBoundExtend.y &&
            sampleBoundExtend.y == sampleBoundExtend.z)
        {
            splitDim = (splitDim + 1) % 3;
            splitPos = Get(sampleMean, splitDim);
        }
        else
        {
            splitDim = maxDimension(sampleBoundExtend);
            splitPos = Get(sampleMean, splitDim);
        }
    }
    else
    {
        splitDim = maxDimension(sampleVariance);
        splitPos = Get(sampleMean, splitDim);
    }
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

inline __device__ float2 Map3DTo2D(float3 vec3D)
{
    float2 vec2D = make_float2(0.f);

    float cosTheta = fmaxf(-1.0f, fminf(1.0f, vec3D.z));
    float alpha    = acos(cosTheta);
    float sinAlpha = sqrt(1 - fminf(cosTheta * cosTheta, 1.f));
    float inv_sinc = sinAlpha != 0.f ? alpha / sinAlpha : 0.f;

    vec2D.x = alpha > 0.f ? vec3D.x * inv_sinc : vec2D.x;
    vec2D.y = alpha > 0.f ? vec3D.y * inv_sinc : vec2D.y;
    return vec2D;
}

inline __device__ float3 FromLocal(float3 t0, float3 t1, float3 t2, float3 x)
{
    return make_float3(dot(t0, x), dot(t1, x), dot(t2, x));
}

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

__device__ float VMFIntegratedDivision(const float3 &meanDirection0, const float &kappa0,
                                       const float &normalization0,
                                       const float3 &meanDirection1, const float &kappa1,
                                       const float &normalization1)
{
    float eMinus2Kappa1         = __expf(-2.f * kappa1);
    float3 productMeanDirection = kappa0 * meanDirection0 + kappa1 * meanDirection1;

    float productKappa = sqrt(dot(productMeanDirection, productMeanDirection));

    float productNormalization = 1.0f / (4.0f * PI);
    float productEMinus2Kappa  = 1.0f;
    if (productKappa > 1e-3f)
    {
        productEMinus2Kappa  = __expf(-2.0f * productKappa);
        productNormalization = productKappa / (2.0f * PI * (1.0f - productEMinus2Kappa));
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

    // TODO IMPORTANT: shouldn't the 1 - eMinus2Kappa1 be squared???
    scale *= (4.0f * PI * PI * (1.0f - eMinus2Kappa1)) / (kappa1 * kappa1);
    scale *= __expf((kappa0 * (cosTheta0 - 1.0f) + kappa1 * (cosTheta1 + 1.0f)));

    return scale;
}

inline __device__ void AddVMMStatistics(VMMStatistics &left, const VMMStatistics &right)
{
    const uint32_t threadIndex = threadIdx.x;
    if (threadIndex < MAX_COMPONENTS)
    {
        left.sumWeightedDirections[threadIndex] += right.sumWeightedDirections[threadIndex];
        left.sumWeights[threadIndex] += right.sumWeights[threadIndex];
    }
    if (threadIndex == 0)
    {
        left.sumSampleWeights += right.sumSampleWeights;
        left.numSamples += right.numSamples;
    }
}

__device__ float CalculateVMFNormalization(float kappa)
{
    float eMinus2Kappa = __expf(-2.f * kappa);
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

#define GPU_KERNEL extern "C" __global__ void

GPU_KERNEL InitializeSamples(SampleStatistics *statistics, KDTreeBuildState *buildState,
                             KDTreeNode *nodes, SampleData *samples,
                             SOAFloat3 sampleDirections, SOAFloat3 samplePositions,
                             uint32_t *__restrict__ sampleIndices, uint32_t numSamples,
                             float3 sceneMin, float3 sceneMax)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        nodes[0].count  = numSamples;
        nodes[0].offset = 0;
        // nodes[0].childIndex_dim = 1;
        nodes[0].parentIndex = -1;

        buildState->totalNumNodes     = 1;
        buildState->numNodes          = 1;
        buildState->nextLevelNumNodes = 0;

        for (uint32_t i = 0; i < 3; i++)
        {
            statistics[i] = SampleStatistics();
        }
    }

    uint32_t sampleIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (sampleIndex < numSamples)
    {
        RNG rng      = RNG::Init(sampleIndex, numSamples);
        float2 u     = rng.Uniform2D();
        float3 dir   = SampleUniformSphere(u);
        float weight = rng.Uniform();
        weight       = weight < 1e-3f ? 1e-3f : weight;
        float pdf    = rng.Uniform();
        pdf          = pdf < 1e-3f ? 1e-3f : pdf;

        float3 p = make_float3(rng.Uniform2D(), rng.Uniform());
        p        = p * (sceneMax - sceneMin) + sceneMin;

        SampleData data;
        data.pos    = p;
        data.dir    = dir;
        data.weight = weight;
        data.pdf    = pdf;

        samples[sampleIndex]          = data;
        sampleDirections[sampleIndex] = dir;
        sampleIndices[sampleIndex]    = sampleIndex;

        samplePositions[sampleIndex] = p;
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
inline __device__ void
UpdateMixtureParameters(VMM &sharedVMM, VMMStatistics &sharedStatistics,
                        const VMMStatistics &previousStatistics, SOAFloat3 sampleDirections,
                        float *sampleWeights, float &sumSampleWeights,
                        float &sharedWeightedLogLikelihood, uint32_t sampleOffset,
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
        if (threadIndex < MAX_COMPONENTS)
        {
            sharedStatistics.sumWeights[threadIndex]            = 0.f;
            sharedStatistics.sumWeightedDirections[threadIndex] = make_float3(0.f);
        }
        if (threadIndex == 0) sharedWeightedLogLikelihood = 0.f;

        __syncthreads();

        // Weighted Expectation
        for (uint32_t batch = 0; batch < numSampleBatches; batch++)
        {
            VMMStatistics statistics;
            uint32_t sampleIndex   = threadIndex + blockDim.x * batch;
            bool hasData           = sampleIndex < sampleCount;
            float V                = 0.f;
            float3 sampleDirection = make_float3(0.f);
            float sampleWeight     = 1.f;

            if (hasData)
            {
                sampleDirection = sampleDirections[sampleIndex + sampleOffset];
                sampleWeight    = sampleWeights[sampleIndex + sampleOffset];
                V = SoftAssignment(sharedVMM, sampleDirection, statistics.sumWeights);
            }

            hasData                     = V > 0.f;
            float invV                  = hasData ? 1.f / V : 0.f;
            float weightedLogLikelihood = hasData ? sampleWeight * __logf(V) : 0.f;

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

            weightedLogLikelihood = WarpReduceSum(weightedLogLikelihood);

            if ((threadIndex & (WARP_SIZE - 1)) == 0)
            {
                atomicAdd(&sharedWeightedLogLikelihood, weightedLogLikelihood);
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
            uint32_t totalNumSamples = sampleCount;
            if constexpr (!masked)
            {
                totalNumSamples += previousStatistics.numSamples;
            }

            float weight = sharedStatistics.sumWeights[threadIndex];

            if constexpr (!masked)
            {
                weight += previousStatistics.sumWeights[threadIndex];
            }

            weight = threadIndex >= sharedVMM.numComponents
                         ? 0.f
                         : (weightPrior + weight) /
                               (weightPrior * sharedVMM.numComponents + totalNumSamples);

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
                assert(invSumWeights > 0.f);
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

                float3 meanDirection = currentMeanDirection * currentEstimationWeight;

                if constexpr (!masked)
                {
                    meanDirection += previousMeanDirection * previousEstimationWeight;
                }

                float meanCosine = length(meanDirection);

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
            float logLikelihood = sharedWeightedLogLikelihood;
            float relLogLikelihoodDifference =
                fabs(logLikelihood - previousLogLikelihood) / fabs(previousLogLikelihood);

            if (relLogLikelihoodDifference < convergenceThreshold) break;
            previousLogLikelihood = logLikelihood;
        }

        __syncthreads();
    }
}

// TODO IMPORTANT:
// - handle unassigned components in MAP update
// - make sure to update statistics and rewrite to memory
// - make sure to get chi square statistics fromm previous update
// - proper criteria for splitting/merging
template <bool partial = false>
__device__ void UpdateMixtureHelper(
    VMM *__restrict__ vmms, VMMStatistics *__restrict__ currentStatistics,
    const SOASampleData samples, const uint32_t *__restrict__ leafNodeIndices,
    const uint32_t *__restrict__ numLeafNodes, const KDTreeNode *__restrict__ nodes,
    const uint32_t *__restrict__ numWorkItems = 0, const WorkItem *workItems = 0)
{
    assert(!partial || (partial && numWorkItems && workItems));

    __shared__ VMMStatistics sharedStatistics;
    __shared__ VMM sharedVMM;
    __shared__ float sumSampleWeights;
    __shared__ float weightedLogLikelihood;

    const uint32_t threadIndex = threadIdx.x;

    uint32_t limit, nodeIndex, sharedSplitMask;
    if constexpr (!partial)
    {
        limit = *numLeafNodes;
    }
    else
    {
        assert(numWorkItems);
        assert(workItems);
        limit = *numWorkItems;
    }

    if (blockIdx.x >= limit) return;

    if constexpr (!partial)
    {
        nodeIndex = leafNodeIndices[blockIdx.x];
    }
    else
    {
        WorkItem workItem = workItems[blockIdx.x];
        nodeIndex         = workItem.nodeIndex;
        sharedSplitMask   = workItem.offset;
    }

    const KDTreeNode &node = nodes[nodeIndex];
    uint32_t sampleCount   = node.count;
    uint32_t offset        = node.offset;
    uint32_t vmmIndex      = node.vmmIndex;

    VMMStatistics &previousStatistics = currentStatistics[vmmIndex];
    if (threadIndex == 0)
    {
        sharedVMM = vmms[vmmIndex];
    }

    __syncthreads();

    const uint32_t numSampleBatches = (sampleCount + blockDim.x - 1) / blockDim.x;

    if constexpr (!partial)
    {
        UpdateMixtureParameters(sharedVMM, sharedStatistics, previousStatistics, samples.dir,
                                samples.weights, sumSampleWeights, weightedLogLikelihood,
                                offset, sampleCount);
        __syncthreads();

        AddVMMStatistics(previousStatistics, sharedStatistics);

        if (threadIndex == 0)
        {
            vmms[vmmIndex]                      = sharedVMM;
            previousStatistics.sumSampleWeights = sumSampleWeights;
            previousStatistics.numSamples       = sampleCount;
            previousStatistics.numSamplesAfterLastSplit += sampleCount;
            previousStatistics.numSamplesAfterLastMerge += sampleCount;
        }
    }
    else
    {
        UpdateMixtureParameters<true>(
            sharedVMM, sharedStatistics, previousStatistics, samples.dir, samples.weights,
            sumSampleWeights, weightedLogLikelihood, offset, sampleCount, sharedSplitMask);
        __syncthreads();

        VMM &vmm = vmms[vmmIndex];
        if (threadIndex < MAX_COMPONENTS)
        {
            const bool threadIsEnabled = sharedSplitMask & (1u << threadIndex);
            if (threadIsEnabled)
            {
                vmm.kappas[threadIndex] = sharedVMM.kappas[threadIndex];
                vmm.WriteDirection(threadIndex, sharedVMM.ReadDirection(threadIndex));
                vmm.weights[threadIndex] = sharedVMM.weights[threadIndex];

                previousStatistics.sumWeights[threadIndex] =
                    sharedStatistics.sumWeights[threadIndex];
                previousStatistics.sumWeightedDirections[threadIndex] =
                    sharedStatistics.sumWeightedDirections[threadIndex];
            }
        }
    }
}

GPU_KERNEL
UpdateMixture(VMM *__restrict__ vmms, VMMStatistics *__restrict__ currentStatistics,
              const SOASampleData samples, const uint32_t *__restrict__ leafNodeIndices,
              const uint32_t *__restrict__ numLeafNodes, const KDTreeNode *__restrict__ nodes)
{
    UpdateMixtureHelper<false>(vmms, currentStatistics, samples, leafNodeIndices, numLeafNodes,
                               nodes);
}

GPU_KERNEL
PartialUpdateMixture(VMM *__restrict__ vmms, VMMStatistics *__restrict__ currentStatistics,
                     const SOASampleData samples, const uint32_t *__restrict__ leafNodeIndices,
                     const uint32_t *__restrict__ numLeafNodes,
                     const KDTreeNode *__restrict__ nodes,
                     const uint32_t *__restrict__ numWorkItems, const WorkItem *workItems)
{
    UpdateMixtureHelper<true>(vmms, currentStatistics, samples, leafNodeIndices, numLeafNodes,
                              nodes, numWorkItems, workItems);
}

GPU_KERNEL UpdateSplitStatistics(VMM *__restrict__ vmms,
                                 VMMStatistics *__restrict__ currentStatistics,
                                 SplitStatistics *__restrict__ splitStatistics,
                                 const SOASampleData samples,
                                 const uint32_t *__restrict__ leafNodeIndices,
                                 const uint32_t *__restrict__ numLeafNodes,
                                 const KDTreeNode *__restrict__ nodes)
{
    __shared__ float sharedChiSquareTotals[MAX_COMPONENTS];

    const uint32_t threadIndex = threadIdx.x;
    const uint32_t wid         = threadIndex >> WARP_SHIFT;

    if (blockIdx.x >= *numLeafNodes) return;

    const KDTreeNode &node = nodes[leafNodeIndices[blockIdx.x]];
    uint32_t sampleCount   = node.count;
    uint32_t offset        = node.offset;

    // const uint32_t totalNumSamples = previousStatistics.numSamples + sampleCount;
    const uint32_t totalNumSamples  = sampleCount;
    float sumSampleWeights          = currentStatistics[node.vmmIndex].sumSampleWeights;
    const float mcEstimate          = sumSampleWeights / float(totalNumSamples);
    uint32_t numUsedSamples         = 0;
    const uint32_t numSampleBatches = (sampleCount + blockDim.x - 1) / blockDim.x;

    const float maxKappa        = 32000.f;
    const float maxMeanCosine   = KappaToMeanCosine(maxKappa);
    VMM &sharedVMM              = vmms[node.vmmIndex];
    SplitStatistics &splitStats = splitStatistics[node.vmmIndex];

    if (threadIndex < MAX_COMPONENTS)
    {
        sharedChiSquareTotals[threadIndex] = 0.f;
    }

    __syncthreads();

    // Update split statistics
    for (uint32_t batch = 0; batch < numSampleBatches; batch++)
    {
        VMMStatistics statistics;
        uint32_t sampleIndex   = threadIndex + blockDim.x * batch;
        bool hasData           = sampleIndex < sampleCount;
        float V                = 0.f;
        float3 sampleDirection = make_float3(0.f);
        float sampleWeight     = 1.f;
        float samplePDF        = 1.f;

        if (hasData)
        {
            sampleDirection = samples.dir[sampleIndex + offset];
            sampleWeight    = samples.weights[sampleIndex + offset];
            samplePDF       = samples.pdfs[sampleIndex + offset];

            V = SoftAssignment(sharedVMM, sampleDirection, statistics.sumWeights);
        }

        float sampleLi = sampleWeight * samplePDF;
        hasData        = V > 0.f;
        float invV     = V > 0.f ? 1.f / V : 0.f;
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

            chiSquareEstimate = WarpReduceSum(chiSquareEstimate);
            cov               = WarpReduceSum(cov);
            assignedWeight    = WarpReduceSum(assignedWeight);

            assert(isfinite(cov.x) && !isnan(cov.x));
            assert(isfinite(cov.y) && !isnan(cov.y));
            assert(isfinite(cov.z) && !isnan(cov.z));
            assert(isfinite(chiSquareEstimate) && !isnan(chiSquareEstimate));

            if ((threadIndex & (WARP_SIZE - 1)) == 0)
            {
                // atomicAdd(&splitStats.chiSquareTotals[componentIndex], chiSquareEstimate);
                atomicAdd(&sharedChiSquareTotals[componentIndex], chiSquareEstimate);
                atomicAdd(&splitStats.covarianceTotals[componentIndex], cov);
                atomicAdd(&splitStats.sumWeights[componentIndex], assignedWeight);
            }
        }
    }

    __syncthreads();

    if (threadIndex < MAX_COMPONENTS)
    {
        // TODO: floating point precision issues?
        float chi                = sharedChiSquareTotals[threadIndex];
        uint32_t totalNumSamples = sampleCount + splitStats.numSamples[threadIndex];
        splitStats.chiSquareTotals[threadIndex] +=
            (chi - splitStats.chiSquareTotals[threadIndex] * sampleCount) / totalNumSamples;
        // if (chi != 0.f)
        // {
        //     printf("chi: %f %f\n", chi, splitStats.chiSquareTotals[threadIndex]);
        // }
        splitStats.numSamples[threadIndex] = totalNumSamples;
    }
}

GPU_KERNEL
SplitComponents(VMM *__restrict__ vmms, VMMStatistics *__restrict__ currentStatistics,
                SplitStatistics *__restrict__ splitStatistics, const SOASampleData samples,
                const uint32_t *__restrict__ leafNodeIndices,
                const uint32_t *__restrict__ numLeafNodes,
                const KDTreeNode *__restrict__ nodes, uint32_t *__restrict__ numWorkItems,
                WorkItem *__restrict__ workItems)
{
    const float splittingThreshold = 0.5f;
    const float maxKappa           = 32000.f;
    const float maxMeanCosine      = KappaToMeanCosine(maxKappa);
    static_assert(MAX_COMPONENTS <= WARP_SIZE,
                  "MAX_COMPONENTS must not be greater than the warp size");

    if (blockIdx.x >= *numLeafNodes) return;

    uint32_t nodeIndex     = leafNodeIndices[blockIdx.x];
    const KDTreeNode &node = nodes[nodeIndex];
    uint32_t vmmIndex      = node.vmmIndex;
    uint32_t sampleCount   = node.count;
    uint32_t offset        = node.offset;

    const uint32_t threadIndex = threadIdx.x;

    VMM &vmm                     = vmms[vmmIndex];
    SplitStatistics &splitStats  = splitStatistics[vmmIndex];
    VMMStatistics &vmmStatistics = currentStatistics[vmmIndex];

    if (vmmStatistics.numSamplesAfterLastSplit < MIN_SPLIT_SAMPLES) return;

    __shared__ uint32_t sharedSplitMask;
    // Split components
    if (threadIndex < MAX_COMPONENTS)
    {
        float chiSquareEstimate               = splitStats.chiSquareTotals[threadIndex];
        uint32_t rank                         = 0;
        const uint32_t numAvailableComponents = MAX_COMPONENTS - vmm.numComponents;
        for (uint32_t i = 0; i < MAX_COMPONENTS; i++)
        {
            float chiSquare    = splitStats.chiSquareTotals[i];
            uint32_t increment = chiSquare > chiSquareEstimate ||
                                 (chiSquare == chiSquareEstimate && i < threadIndex);
            rank += increment;
        }

        bool split = splitStats.chiSquareTotals[threadIndex] > splittingThreshold &&
                     rank < numAvailableComponents;

        uint32_t splitMask     = __ballot_sync(0xffffffff, split);
        uint32_t numComponents = vmm.numComponents;
        uint32_t newComponentIndex =
            numComponents + __popc(splitMask & ((1u << threadIndex) - 1));

        uint componentIndex = threadIndex;

        if (split)
        {
            assert(newComponentIndex < MAX_COMPONENTS);

            float sumWeights = splitStats.sumWeights[componentIndex];
            float3 cov       = splitStats.covarianceTotals[componentIndex] / sumWeights;

            float negB          = cov.x + cov.y;
            float discriminant  = sqrt((cov.x - cov.y) * (cov.x - cov.y) - 4 * cov.z * cov.z);
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

                float3 meanDirection = vmm.ReadDirection(componentIndex);

                float3 basis0, basis1;
                BuildOrthonormalBasis(meanDirection, basis0, basis1);

                float3 meanDir3D0 = Map2DTo3D(meanDir0);
                float3 meanDir3D1 = Map2DTo3D(meanDir1);

                float3 meanDirection0 = basis0 * meanDir3D0.x + basis1 * meanDir3D0.y +
                                        meanDirection * meanDir3D0.z;
                float3 meanDirection1 = basis0 * meanDir3D1.x + basis1 * meanDir3D1.y +
                                        meanDirection * meanDir3D1.z;

                float meanCosine  = KappaToMeanCosine(vmm.kappas[componentIndex]);
                float meanCosine0 = meanCosine / abs(dot(meanDirection0, meanDirection));
                meanCosine0       = min(meanCosine0, maxMeanCosine);

                float kappa    = MeanCosineToKappa(meanCosine0);
                float weight   = vmm.weights[componentIndex] * 0.5f;
                float distance = vmm.distances[componentIndex];

                vmm.UpdateComponent_(componentIndex, kappa, meanDirection0, weight);
                vmm.UpdateComponent(newComponentIndex, kappa, meanDirection1, weight,
                                    distance);

                splitStats.SetComponent(componentIndex, make_float3(0.f), 0.f, 0.f, 0);
                splitStats.SetComponent(newComponentIndex, make_float3(0.f), 0.f, 0.f, 0);

                float sumStatsWeights = vmmStatistics.sumWeights[componentIndex] / 2.f;
                vmmStatistics.SetComponent(componentIndex, sumStatsWeights,
                                           meanDirection0 * meanCosine0 * sumStatsWeights);
                vmmStatistics.SetComponent(newComponentIndex, sumStatsWeights,
                                           meanDirection1 * meanCosine0 * sumStatsWeights);
            }
        }

        if (threadIndex == 0)
        {
            uint32_t newNumComponents = __popc(splitMask);
            assert(vmm.numComponents + newNumComponents <= MAX_COMPONENTS);

            uint32_t maskExtend = ((1u << newNumComponents) - 1u) << vmm.numComponents;

            vmm.numComponents += newNumComponents;
            vmmStatistics.numSamplesAfterLastSplit = 0;
            sharedSplitMask                        = splitMask | maskExtend;
        }
    }

    __syncthreads();

    // Update split components
    if (threadIndex == 0 && sharedSplitMask)
    {
        printf("vmm: %u\n", vmmIndex);
        WorkItem workItem;
        workItem.nodeIndex = nodeIndex;
        workItem.offset    = sharedSplitMask;
        workItem.count     = 0;

        uint32_t workItemIndex   = atomicAdd(numWorkItems, 1);
        workItems[workItemIndex] = workItem;
    }
}

GPU_KERNEL MergeComponents(VMM *__restrict__ vmms,
                           VMMStatistics *__restrict__ currentStatistics,
                           SplitStatistics *__restrict__ splitStatistics,
                           const SOASampleData samples,
                           const uint32_t *__restrict__ leafNodeIndices,
                           const uint32_t *__restrict__ numLeafNodes,
                           const KDTreeNode *__restrict__ nodes)
{
    const float mergeThreshold = 0.025f;
    __shared__ float mergeKappas[MAX_COMPONENTS];
    __shared__ float mergeProducts[MAX_COMPONENTS];
    __shared__ float3 mergeDirections[MAX_COMPONENTS];
    __shared__ VMM sharedVMM;

    __shared__ uint2 componentsToMerge;
    __shared__ float minMergeCost;

    if (blockIdx.x >= *numLeafNodes) return;

    uint32_t threadIndex               = threadIdx.x;
    uint32_t nodeIndex                 = leafNodeIndices[blockIdx.x];
    const KDTreeNode &node             = nodes[nodeIndex];
    uint32_t vmmIndex                  = node.vmmIndex;
    uint32_t sampleCount               = node.count;
    uint32_t offset                    = node.offset;
    const VMMStatistics &vmmStatistics = currentStatistics[vmmIndex];

    if (threadIndex == 0)
    {
        sharedVMM    = vmms[vmmIndex];
        minMergeCost = FLT_MAX;
    }
    __syncthreads();

    if (vmmStatistics.numSamplesAfterLastMerge < MIN_MERGE_SAMPLES) return;

    // Merging
    for (;;)
    {
        const uint32_t numComponents = sharedVMM.numComponents;

        const uint32_t numPairs        = (numComponents * numComponents - numComponents) / 2;
        const uint32_t numMergeBatches = (numPairs + blockDim.x - 1) / blockDim.x;

        if (threadIndex < sharedVMM.numComponents)
        {
            float kappa = sharedVMM.kappas[threadIndex];
            // if (blockIdx.x == 0)
            // {
            //     printf("weight: %f\n", sharedVMM.weights[threadIndex]);
            //     printf("%u, thread: %u, %f\n", numComponents, threadIndex, kappa);
            //     float3 dir = sharedVMM.ReadDirection(threadIndex);
            //     printf("%f %f %f\n", dir.x, dir.y, dir.z);
            // }
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

        // TODO: I'm pretty sure they don't combine components that were just split?
        for (uint32_t batch = 0; batch < numMergeBatches; batch++)
        {
            // X 0 1 2  3  4   5  6  7
            // X X 8 9  10 11 12 13 14
            // X X X 15 16 17 18 19 20

            // Calculate the merge cost for the requisite pair
            uint32_t indexInBatch = batch * blockDim.x + threadIndex;
            float a               = 1 + (numComponents - 1) * 2;
            uint32_t row          = 1 + uint32_t(a - sqrt(a * a - 8.f * indexInBatch)) / 2;

            uint32_t newIndexInBatch = indexInBatch + row * (row + 1) / 2;

            const uint32_t componentIndex0 = newIndexInBatch / numComponents;
            const uint32_t componentIndex1 = newIndexInBatch % numComponents;

            float chiSquareIJ = FLT_MAX;

            if (indexInBatch < numPairs)
            {
                float kappa0          = sharedVMM.kappas[componentIndex0];
                float normalization0  = CalculateVMFNormalization(kappa0);
                float kappa1          = sharedVMM.kappas[componentIndex1];
                float normalization1  = CalculateVMFNormalization(kappa1);
                float weight0         = sharedVMM.weights[componentIndex0];
                float weight1         = sharedVMM.weights[componentIndex1];
                float3 meanDirection0 = sharedVMM.ReadDirection(componentIndex0);
                float3 meanDirection1 = sharedVMM.ReadDirection(componentIndex1);

                float kappa00         = mergeKappas[componentIndex0];
                float normalization00 = CalculateVMFNormalization(kappa00);
                float scale00         = mergeProducts[componentIndex0];
                float3 dir00          = mergeDirections[componentIndex0];

                float kappa11         = mergeKappas[componentIndex1];
                float normalization11 = CalculateVMFNormalization(kappa11);
                float scale11         = mergeProducts[componentIndex1];
                float3 dir11          = mergeDirections[componentIndex1];

                float weight         = weight0 + weight1;
                float3 meanDirection = weight0 * KappaToMeanCosine(kappa0) * meanDirection0 +
                                       weight1 * KappaToMeanCosine(kappa1) * meanDirection1;
                meanDirection /= weight;
                float meanCosine = dot(meanDirection, meanDirection);
                float kappa      = 0.f;

                if (meanCosine > 0.f)
                {
                    meanCosine = sqrt(meanCosine);
                    kappa      = MeanCosineToKappa(meanCosine);
                    meanDirection /= meanCosine;
                }

                // what's left?
                //     fitting vs updating distinction for both statistics and the tree
                //     handling distances for parallax
                //     runtime importance sampling, etc

                else
                {
                    meanDirection = meanDirection0;
                }

                float normalization = CalculateVMFNormalization(kappa);

                float kappa01, normalization01;
                float3 dir01;
                float scale01 =
                    VMFProduct(kappa0, normalization0, meanDirection0, kappa1, normalization1,
                               meanDirection1, kappa01, normalization01, dir01);

                chiSquareIJ = 0.f;
                chiSquareIJ +=
                    (weight0 * weight0 / weight) *
                    (scale00 * VMFIntegratedDivision(dir00, kappa00, normalization00,
                                                     -meanDirection, kappa, normalization));
                chiSquareIJ +=
                    (weight1 * weight1 / weight) *
                    (scale11 * VMFIntegratedDivision(dir11, kappa11, normalization11,
                                                     -meanDirection, kappa, normalization));
                chiSquareIJ +=
                    2.0f * (weight0 * weight1 / weight) *
                    (scale01 * VMFIntegratedDivision(dir01, kappa01, normalization01,
                                                     -meanDirection, kappa, normalization));
                chiSquareIJ -= weight;
            }

            float minChi = WarpReduceMin(chiSquareIJ);
            minChi       = WarpReadLaneFirst(minChi);
            if (minChi == chiSquareIJ && minChi < minMergeCost)
            {
                minMergeCost        = minChi;
                componentsToMerge.x = componentIndex0;
                componentsToMerge.y = componentIndex1;
            }
            __syncthreads();
        }

        // if (threadIndex == 0)
        // {
        //     printf("%f\n", minMergeCost);
        // }
        if (minMergeCost < mergeThreshold)
        {
            printf("good: %f %u %u\n", minMergeCost, componentsToMerge.x, componentsToMerge.y);

            // Merge components
            if (threadIndex == 0)
            {
                float weightI, weightJ, weightK;
                float3 meanDirectionI, meanDirectionJ, meanDirectionK;
                const uint32_t index0 = componentsToMerge.x;
                const uint32_t index1 = componentsToMerge.y;

                // Update VMM
                {

                    const float weight0 = sharedVMM.weights[index0];
                    const float weight1 = sharedVMM.weights[index1];
                    weightI             = weight0;
                    weightJ             = weight1;

                    const float meanCosine0 = KappaToMeanCosine(sharedVMM.kappas[index0]);
                    const float meanCosine1 = KappaToMeanCosine(sharedVMM.kappas[index1]);

                    float kappa  = 0.0f;
                    float weight = weight0 + weight1;
                    weightK      = weight;

                    meanDirectionI = sharedVMM.ReadDirection(index0);
                    meanDirectionJ = sharedVMM.ReadDirection(index1);

                    float3 meanDirection = weight0 * meanCosine0 * meanDirectionI +
                                           weight1 * meanCosine1 * meanDirectionJ;

                    meanDirection /= weight;

                    float meanCosine = dot(meanDirection, meanDirection);

                    if (meanCosine > 0.0f)
                    {
                        meanCosine = sqrt(meanCosine);

                        kappa = MeanCosineToKappa(meanCosine);
                        kappa = kappa < 1e-3f ? 0.f : kappa;

                        meanDirection /= meanCosine;
                    }
                    else
                    {
                        meanDirection = sharedVMM.ReadDirection(index0);
                    }

                    sharedVMM.weights[index0] = weight;
                    sharedVMM.kappas[index0]  = kappa;

                    meanDirectionK = meanDirection;
                    sharedVMM.WriteDirection(index0, meanDirection);

                    const float distance0 = sharedVMM.distances[index0];
                    const float distance1 = sharedVMM.distances[index1];

                    float newDistance = weight0 * distance0 + weight1 * distance1;
                    newDistance /= (weight0 + weight1);

                    uint32_t last = sharedVMM.numComponents - 1;
                    sharedVMM.UpdateComponent(index1, sharedVMM.kappas[last],
                                              sharedVMM.ReadDirection(last),
                                              sharedVMM.weights[last], newDistance);
                    sharedVMM.UpdateComponent(last, 0.f, make_float3(0.f), 0.f, 0.f);
                    sharedVMM.numComponents -= 1;
                }

                // Update split statistics
                const uint32_t last = sharedVMM.numComponents;
                {
                    SplitStatistics &splitStats = splitStatistics[vmmIndex];

                    float3 frame0, frame1;
                    BuildOrthonormalBasis(meanDirectionK, frame0, frame1);

                    float2 meanDirection2DItoK =
                        Map3DTo2D(FromLocal(frame0, frame1, meanDirectionK, meanDirectionI));
                    float2 meanDirection2DJtoK =
                        Map3DTo2D(FromLocal(frame0, frame1, meanDirectionK, meanDirectionJ));

                    const float inv_weightK = (weightK > 0.f) ? 1.f / weightK : 1.f;

                    const float sumWeightsI = splitStats.sumWeights[index0];
                    const float sumWeightsJ = splitStats.sumWeights[index1];
                    const float sumWeightsK = sumWeightsI + sumWeightsJ;

                    const float3 covarianceI =
                        (sumWeightsI > 0.f) ? splitStats.covarianceTotals[index0] / sumWeightsI
                                            : make_float3(0.f);
                    const float3 covarianceJ =
                        (sumWeightsJ > 0.f) ? splitStats.covarianceTotals[index1] / sumWeightsJ
                                            : make_float3(0.f);

                    const float2 meanDirectionK2D = make_float2(0.f);
                    float3 meanII = make_float3(meanDirection2DItoK.x * meanDirection2DItoK.x,
                                                meanDirection2DItoK.y * meanDirection2DItoK.y,
                                                meanDirection2DItoK.x * meanDirection2DItoK.y);
                    float3 meanJJ = make_float3(meanDirection2DJtoK.x * meanDirection2DJtoK.x,
                                                meanDirection2DJtoK.y * meanDirection2DJtoK.y,
                                                meanDirection2DJtoK.x * meanDirection2DJtoK.y);
                    float3 covarianceK = weightI * covarianceI + weightI * meanII +
                                         weightJ * covarianceJ + weightJ * meanJJ;
                    covarianceK *= inv_weightK;
                    const float3 sampleCovarianceK = covarianceK * sumWeightsK;

                    // TODO: is this ever used?
                    // const float sumAssignedSamplesK =
                    // sumAssignedSamples[tmpI.quot][tmpI.rem] +
                    //                                   sumAssignedSamples[tmpJ.quot][tmpJ.rem];

                    const float numSamplesK =
                        inv_weightK * (weightI * splitStats.numSamples[index0] +
                                       weightJ * splitStats.numSamples[index1]);
                    const float chiSquareMCEstimatesK = splitStats.chiSquareTotals[index0] +
                                                        splitStats.chiSquareTotals[index1];

                    splitStats.SetComponent(index0, sampleCovarianceK, chiSquareMCEstimatesK,
                                            sumWeightsK, numSamplesK);
                    // sumAssignedSamples[tmpI.quot][tmpI.rem]   = sumAssignedSamplesK;

                    splitStats.SetComponent(index1, splitStats.covarianceTotals[last],
                                            splitStats.chiSquareTotals[last],
                                            splitStats.sumWeights[last],
                                            splitStats.numSamples[last]);

                    splitStats.SetComponent(last, make_float3(0.f), 0.f, 0.f, 0);
                }

                // Update vmm statistics
                {
                    VMMStatistics &vmmStatistics = currentStatistics[vmmIndex];
                    vmmStatistics.sumWeightedDirections[index0] +=
                        vmmStatistics.sumWeightedDirections[index1];
                    vmmStatistics.sumWeights[index0] += vmmStatistics.sumWeights[index1];
                    // sumOfDistanceWeightes[tmpIdx0.quot][tmpIdx0.rem] +=
                    //     sumOfDistanceWeightes[tmpIdx1.quot][tmpIdx1.rem];

                    vmmStatistics.sumWeightedDirections[index1] =
                        vmmStatistics.sumWeightedDirections[last];
                    vmmStatistics.sumWeights[index1] = vmmStatistics.sumWeights[last];
                    // sumOfDistanceWeightes[tmpIdx1.quot][tmpIdx1.rem] =
                    //     sumOfDistanceWeightes[tmpIdx2.quot][tmpIdx2.rem];

                    // reseting the statistics of the last component
                    vmmStatistics.SetComponent(last, 0.f, make_float3(0.f));
                    // sumOfDistanceWeightes[tmpIdx2.quot][tmpIdx2.rem]     = 0.0f;
                }
            }
        }
        else break;
    }

    if (threadIndex == 0)
    {
        VMMStatistics &vmmStatistics           = currentStatistics[vmmIndex];
        vmmStatistics.numSamplesAfterLastMerge = 0;
    }
}

GPU_KERNEL UpdateComponentDistances(VMM *vmms, KDTreeNode *nodes, VMMStatistics *vmmStatistics,
                                    SOASampleData samples,
                                    const uint32_t *__restrict__ leafNodeIndices,
                                    const uint32_t *__restrict__ numLeafNodes)
{
    __shared__ float batchDistances[MAX_COMPONENTS];
    __shared__ float batchSumWeights[MAX_COMPONENTS];

    if (blockIdx.x >= *numLeafNodes) return;

    if (threadIdx.x < MAX_COMPONENTS)
    {
        batchDistances[threadIdx.x]  = 0.f;
        batchSumWeights[threadIdx.x] = 0.f;
    }

    __syncthreads();

    uint32_t nodeIndex     = leafNodeIndices[blockIdx.x];
    const KDTreeNode &node = nodes[nodeIndex];
    uint32_t sampleCount   = node.count;
    uint32_t sampleOffset  = node.offset;
    uint32_t vmmIndex      = node.vmmIndex;

    const VMM &vmm = vmms[vmmIndex];

    const uint32_t numSampleBatches = (sampleCount + blockDim.x - 1) / blockDim.x;

    for (uint32_t batch = 0; batch < numSampleBatches; batch++)
    {
        VMMStatistics statistics;
        uint32_t sampleIndex   = threadIdx.x + blockDim.x * batch;
        bool hasData           = sampleIndex < sampleCount;
        float V                = 0.f;
        float3 sampleDirection = make_float3(0.f);
        float sampleWeight     = 1.f;
        float sampleDistance   = 0.f;

        if (hasData)
        {
            sampleDirection = samples.dir[sampleIndex + sampleOffset];
            sampleWeight    = samples.weights[sampleIndex + sampleOffset];
            sampleDistance  = 1.f / samples.distances[sampleIndex + sampleOffset];
            V               = SoftAssignment(vmm, sampleDirection, statistics.sumWeights);
        }

        hasData = V > 0.f;

        for (uint32_t componentIndex = 0; componentIndex < vmm.numComponents; componentIndex++)
        {
            float weight = vmm.weights[componentIndex] > FLT_EPSILON
                               ? sampleWeight * statistics.sumWeights[componentIndex] *
                                     statistics.sumWeights[componentIndex] /
                                     (vmm.weights[componentIndex] * V)
                               : 0.f;

            atomicAdd(&batchDistances[componentIndex], weight * sampleDistance);
            atomicAdd(&batchSumWeights[componentIndex], weight);
        }
    }

    if (threadIdx.x < MAX_COMPONENTS)
    {
        VMMStatistics &vmmStats = vmmStatistics[vmmIndex];
        VMM &vmm                = vmms[vmmIndex];

        // NOTE: this assumes that the distances are 0 initialized
        float sumInverseDistances = batchDistances[threadIdx.x];
        sumInverseDistances +=
            vmm.distances[threadIdx.x] > 0.0f
                ? vmmStatistics[vmmIndex].sumOfDistanceWeights[threadIdx.x] /
                      vmm.distances[threadIdx.x]
                : 0.f;

        vmmStats.sumOfDistanceWeights[threadIdx.x] += batchSumWeights[threadIdx.x];
        vmm.distances[threadIdx.x] =
            vmmStats.sumOfDistanceWeights[threadIdx.x] / sumInverseDistances;

        // vmmStatistics[vmmIndex].sumOfDistanceWeights[threadIdx.x] =
        //     batchSumWeights[threadIdx.x];
        // vmm.distances[threadIdx.x] =
        //     batchDistances[threadIdx.x] > FLT_EPSILON
        //         ? batchSumWeights[threadIdx.x] / batchDistances[threadIdx.x]
        //         : 0.f;
    }
}

// Reprojects samples to VMM
GPU_KERNEL ReprojectSamples(SampleStatistics *statistics, SOASampleData samples,
                            const uint32_t *__restrict__ leafNodeIndices,
                            const uint32_t *__restrict__ numLeafNodes,
                            const KDTreeNode *__restrict__ nodes, const Bounds3f *bounds)

{
    if (blockIdx.x >= *numLeafNodes) return;

    uint32_t nodeIndex     = leafNodeIndices[blockIdx.x];
    const KDTreeNode &node = nodes[nodeIndex];
    uint32_t sampleCount   = node.count;
    uint32_t sampleOffset  = node.offset;
    uint32_t vmmIndex      = node.vmmIndex;

    SampleStatistics &sampleStatistics = statistics[nodeIndex];

    Bounds3f scaledBounds = nodeIndex == 0 ? *bounds : statistics[node.parentIndex].bounds;
    float3 center         = scaledBounds.GetCenter();
    scaledBounds.SetBoundsMin(center + (scaledBounds.GetBoundsMin() - center) *
                                           INTEGER_SAMPLE_STATS_BOUND_SCALE);
    scaledBounds.SetBoundsMax(center + (scaledBounds.GetBoundsMax() - center) *
                                           INTEGER_SAMPLE_STATS_BOUND_SCALE);

    center            = scaledBounds.GetCenter();
    float3 halfExtent = scaledBounds.GetHalfExtent();
    float3 sampleMean, sampleVariance;
    sampleStatistics.ConvertToFloat(center, halfExtent, sampleMean, sampleVariance);

    float norm        = dot(sampleVariance, sampleVariance);
    norm              = max(FLT_EPSILON, norm);
    float minDistance = sqrt(norm);
    minDistance       = 3.f * 3.f * sqrt(minDistance);

    const uint32_t numSampleBatches = (sampleCount + blockDim.x - 1) / blockDim.x;
    for (uint32_t batch = 0; batch < numSampleBatches; batch++)
    {
        uint32_t sampleIndex = threadIdx.x + blockDim.x * batch;

        float sampleDistance = samples.distances[sampleIndex + sampleOffset];

        if (isinf(sampleDistance))
        {
            samples.pos[sampleIndex + sampleOffset] = sampleMean;
            continue;
        }
        else if (!(sampleDistance > 0.0f))
        {
            continue;
        }

        const float distance        = fmaxf(minDistance, sampleDistance);
        const float3 samplePosition = samples.pos[sampleIndex + sampleOffset];
        float3 sampleDirection      = samples.dir[sampleIndex + sampleOffset];

        const float3 originPosition = samplePosition + sampleDirection * distance;
        float3 newDirection         = originPosition - sampleMean;
        const float newDistance     = length(newDirection);

        // TODO: are these ever used again?
        // sample.position.x = pivotPoint[0];
        // sample.position.y = pivotPoint[1];
        // sample.position.z = pivotPoint[2];

        newDirection =
            newDistance > FLT_EPSILON ? newDirection / newDistance : sampleDirection;
        samples.distances[sampleIndex + sampleOffset] =
            newDistance > FLT_EPSILON ? newDistance : distance;
        samples.dir[sampleIndex + sampleOffset] = newDirection;
    }
}

#define PARTITION_THREADS_IN_BLOCK 256
#define PARTITION_WARPS_IN_BLOCK   (PARTITION_THREADS_IN_BLOCK >> WARP_SHIFT)

struct PartitionSharedState
{
    uint32_t numLefts[PARTITION_WARPS_IN_BLOCK];
    uint32_t numRights[PARTITION_WARPS_IN_BLOCK];
    uint32_t leftOffset;
    uint32_t rightOffset;

    // NOTE: Must be executed by first warp
    __device__ void Clear()
    {
        if (threadIdx.x < PARTITION_WARPS_IN_BLOCK)
        {
            numLefts[threadIdx.x]  = 0;
            numRights[threadIdx.x] = 0;
        }
    }
};

__device__ void PartitionSamples(uint32_t *sampleIndices, uint32_t *newSampleIndices,
                                 float splitPos, uint32_t splitDim, uint32_t numSamples,
                                 uint32_t *sampleLeftOffset, uint32_t *sampleRightOffset,
                                 SOAFloat3 samplePositions, PartitionSharedState &state,
                                 bool active)
{
    uint32_t *numLefts    = state.numLefts;
    uint32_t *numRights   = state.numRights;
    uint32_t &leftOffset  = state.leftOffset;
    uint32_t &rightOffset = state.rightOffset;

    const uint32_t wid = threadIdx.x >> WARP_SHIFT;

    uint32_t sampleIndexIndex = threadIdx.x;

    uint32_t dim = splitDim;

    uint32_t sampleIndex = active ? sampleIndices[sampleIndexIndex] : 0u;

    float *array =
        dim == 0 ? samplePositions.x : (dim == 1 ? samplePositions.y : samplePositions.z);
    float samplePosDim = active ? array[sampleIndex] : 0.f;
    bool isLeft        = active && samplePosDim < splitPos;
    bool isRight       = active && samplePosDim >= splitPos;

    uint32_t mask      = __activemask();
    uint32_t leftMask  = __ballot_sync(mask, isLeft);
    uint32_t rightMask = __ballot_sync(mask, isRight);

    if ((threadIdx.x & (WARP_SIZE - 1)) == 0)
    {
        numLefts[wid]  = __popc(leftMask);
        numRights[wid] = __popc(rightMask);
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        uint32_t totalLeft  = 0;
        uint32_t totalRight = 0;
        for (uint32_t i = 0; i < PARTITION_WARPS_IN_BLOCK; i++)
        {
            uint32_t numLeft = numLefts[i];
            numLefts[i]      = totalLeft;
            totalLeft += numLeft;

            uint32_t numRight = numRights[i];
            numRights[i]      = totalRight;
            totalRight += numRight;
        }
        leftOffset  = atomicAdd(sampleLeftOffset, totalLeft);
        rightOffset = atomicAdd(sampleRightOffset, totalRight);
    }

    __syncthreads();

    uint32_t threadInWarp = threadIdx.x & 31u;
    uint32_t lOffset =
        leftOffset + numLefts[wid] + __popc(leftMask & ((1u << threadInWarp) - 1u));
    uint32_t rOffset =
        rightOffset + numRights[wid] + __popc(rightMask & ((1u << threadInWarp) - 1u));

    uint32_t offset = (isLeft ? lOffset : numSamples - 1 - rOffset);

    if (active)
    {
        newSampleIndices[offset] = sampleIndex;
    }
}

// you first find the split position. then you partition. then you recursively call
// the kernel on the children. in gpu land you push the work to the queue.

template <uint32_t blockSize, typename MergeType, typename AddFunc>
__device__ void BlockReduction(MergeType *totals, MergeType &out, uint32_t numElements,
                               uint32_t blockIndex, uint32_t stride, AddFunc &&addFunc)
{
    static constexpr uint32_t numWarps = blockSize >> WARP_SHIFT;
    static_assert(numWarps <= WARP_SIZE, "Too many warps");
    // const uint32_t numElementsPerThread = (numElements + blockSize - 1) / blockSize;

    uint32_t wId = threadIdx.x >> WARP_SHIFT;

    MergeType total;
    total.Clear();

    uint32_t i = threadIdx.x + blockIndex * blockSize;
    while (i < numElements)
    {
        addFunc(total, i);
        i += stride;
    }

    total.WarpReduce();

    if ((threadIdx.x & 31) == 0)
    {
        totals[wId] = total;
    }

    __syncthreads();

    if (wId == 0)
    {
        MergeType finalTotal = threadIdx.x < numWarps ? totals[threadIdx.x] : MergeType();
        finalTotal.WarpReduce();
        if (threadIdx.x == 0)
        {
            out.AtomicMerge(finalTotal);
        }
    }

    // MergeType finalTotal = threadIdx.x < numWarps ? totals[threadIdx.x] : MergeType();
    // finalTotal.WarpReduce();
    // if (threadIdx.x == 0)
    // {
    //     out.AtomicMerge(finalTotal);
    // }
    // return finalTotal;
}

template <uint32_t blockSize, typename MergeType, typename AddFunc>
__device__ void BlockReductionGridStride(MergeType *totals, MergeType &out,
                                         uint32_t numElements, AddFunc &&addFunc)
{
    BlockReduction<blockSize>(totals, out, numElements, blockIdx.x, blockSize * gridDim.x,
                              addFunc);
}

GPU_KERNEL GetSampleBounds(Bounds3f *sampleBounds, SOAFloat3 samplePositions,
                           uint32_t numSamples)
{
    const uint32_t blockSize = 256;
    assert(blockDim.x == blockSize);

    static constexpr uint32_t numWarps = blockSize >> WARP_SHIFT;
    __shared__ Bounds3f totals[numWarps];

    BlockReductionGridStride<blockSize>(totals, *sampleBounds, numSamples,
                                        [&](Bounds3f &bounds, uint32_t index) {
                                            float3 samplePos = samplePositions[index];
                                            bounds.Extend(samplePos);
                                        });
}

GPU_KERNEL CalculateSplitLocations(KDTreeBuildState *buildState, KDTreeNode *nodes,
                                   SampleStatistics *statistics,
                                   uint32_t *__restrict__ nodeIndices)
{
    uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadIndex >= buildState->numNodes) return;

    int nodeIndex    = nodeIndices[threadIndex];
    KDTreeNode *node = &nodes[nodeIndex];

    if (node->HasChild())
    {
        Bounds3f scaledBounds =
            nodeIndex == 0 ? statistics[0].bounds : statistics[node->parentIndex].bounds;
        float3 center = scaledBounds.GetCenter();
        scaledBounds.SetBoundsMin(center + (scaledBounds.GetBoundsMin() - center) *
                                               INTEGER_SAMPLE_STATS_BOUND_SCALE);
        scaledBounds.SetBoundsMax(center + (scaledBounds.GetBoundsMax() - center) *
                                               INTEGER_SAMPLE_STATS_BOUND_SCALE);

        center            = scaledBounds.GetCenter();
        float3 halfExtent = scaledBounds.GetHalfExtent();

        float3 sampleMean, sampleVariance;
        statistics[nodeIndex].ConvertToFloat(center, halfExtent, sampleMean, sampleVariance);

        uint32_t splitDim;
        float splitPos;
        GetSplitDimensionAndPosition(sampleMean, sampleVariance, 2.f * halfExtent, splitDim,
                                     splitPos);

        node->SetSplitDim(splitDim);
        node->splitPos = splitPos;

        assert(!isnan(sampleMean.x) && !isnan(sampleMean.y) && !isnan(sampleMean.z));
        assert(!isnan(sampleVariance.x) && !isnan(sampleVariance.y) &&
               !isnan(sampleVariance.z));

        assert(isfinite(sampleMean.x) && isfinite(sampleMean.y) && isfinite(sampleMean.z));
        assert(isfinite(sampleVariance.x) && isfinite(sampleVariance.y) &&
               isfinite(sampleVariance.z));
    }
}

GPU_KERNEL BeginLevel(KDTreeBuildState *buildState)
{
    buildState->numReduceWorkItems    = 0;
    buildState->numPartitionWorkItems = 0;
    buildState->numNodes              = buildState->nextLevelNumNodes;
    buildState->nextLevelNumNodes     = 0;
}

GPU_KERNEL CalculateChildIndices(KDTreeNode *__restrict__ nodes,
                                 KDTreeBuildState *__restrict__ buildState,
                                 const uint32_t *__restrict__ nodeIndices,
                                 uint32_t *__restrict__ nextNodeIndices)
{
    for (uint32_t nodeIndexIndex = 0; nodeIndexIndex < buildState->numNodes; nodeIndexIndex++)
    {
        uint32_t nodeIndex = nodeIndices[nodeIndexIndex];
        KDTreeNode &node   = nodes[nodeIndex];

        bool split    = node.count > MAX_SAMPLES_PER_LEAF;
        bool hasChild = node.HasChild();

        if (split || hasChild)
        {
            uint32_t childIndex = ~0u;
            if (!hasChild)
            {
                childIndex = buildState->totalNumNodes;
                buildState->totalNumNodes += 2;
                node.SetChildIndex(childIndex);
            }
            else
            {
                childIndex = node.GetChildIndex();
            }

            assert(childIndex != ~0u);

            nextNodeIndices[buildState->nextLevelNumNodes]     = childIndex;
            nextNodeIndices[buildState->nextLevelNumNodes + 1] = childIndex + 1;
            buildState->nextLevelNumNodes += 2;
            nodes[childIndex].count     = 0;
            nodes[childIndex + 1].count = 0;
        }
        else
        {
            node.childIndex_dim = ~0u;
        }
    }
}

GPU_KERNEL CreateWorkItems(KDTreeBuildState *buildState, WorkItem *reduceWorkItems,
                           WorkItem *partitionWorkItems, KDTreeNode *nodes,
                           SampleStatistics *stats, uint32_t *nodeIndices)
{
    assert(blockDim.x == 256);
    const uint32_t blockSize = 256;

    if (blockIdx.x >= buildState->numNodes) return;

    uint32_t nodeIndex = nodeIndices[blockIdx.x];

    __shared__ KDTreeNode node;
    __shared__ uint32_t reduceStart;
    __shared__ uint32_t partitionStart;

    if (threadIdx.x == 0)
    {
        node = nodes[nodeIndex];
    }

    __syncthreads();

    uint32_t numWorkItems = (node.count + blockSize - 1) / blockSize;
    bool split            = node.HasChild();
    bool reduce           = node.IsChild();

    if (threadIdx.x == 0)
    {
        if (reduce)
        {
            reduceStart      = atomicAdd(&buildState->numReduceWorkItems, numWorkItems);
            stats[nodeIndex] = SampleStatistics();
        }
        if (split)
        {
            partitionStart = atomicAdd(&buildState->numPartitionWorkItems, numWorkItems);
        }
    }

    __syncthreads();

    if (split || reduce)
    {
        for (uint32_t i = threadIdx.x; i < numWorkItems; i += blockSize)
        {
            WorkItem workItem;
            workItem.nodeIndex = nodeIndex;
            workItem.offset    = i * blockSize;
            workItem.count     = min(node.count - workItem.offset, blockSize);

            if (reduce)
            {
                reduceWorkItems[reduceStart + i] = workItem;
            }

            if (split)
            {
                partitionWorkItems[partitionStart + i] = workItem;
            }
        }
    }
}

GPU_KERNEL CalculateNodeStatistics(KDTreeBuildState *buildState, WorkItem *workItems,
                                   KDTreeNode *nodes, SampleStatistics *statistics,
                                   SOAFloat3 samplePositions, uint32_t *sampleIndices)
{
    const uint32_t blockSize = 256;
    assert(blockSize == blockDim.x);
    static constexpr uint32_t numWarps = blockSize >> WARP_SHIFT;

    __shared__ WorkItem workItem;
    __shared__ KDTreeNode node;
    __shared__ SampleStatistics totals[numWarps];

    for (uint32_t workItemIndex = blockIdx.x; workItemIndex < buildState->numReduceWorkItems;
         workItemIndex += gridDim.x)
    {
        if (threadIdx.x == 0)
        {
            workItem = workItems[workItemIndex];
            node     = nodes[workItem.nodeIndex];
        }

        __syncthreads();

        const uint32_t *blockSampleIndices = sampleIndices + node.offset + workItem.offset;
        Bounds3f scaledBounds              = workItem.nodeIndex == 0 ? statistics[0].bounds
                                                                     : statistics[node.parentIndex].bounds;

        SampleStatistics &blockStatistics = statistics[workItem.nodeIndex];

        float3 center = scaledBounds.GetCenter();
        scaledBounds.SetBoundsMin(center + (scaledBounds.GetBoundsMin() - center) *
                                               INTEGER_SAMPLE_STATS_BOUND_SCALE);
        scaledBounds.SetBoundsMax(center + (scaledBounds.GetBoundsMax() - center) *
                                               INTEGER_SAMPLE_STATS_BOUND_SCALE);

        center               = scaledBounds.GetCenter();
        float3 invHalfExtent = scaledBounds.GetHalfExtent();

        invHalfExtent.x = invHalfExtent.x > 0.f ? 1.f / invHalfExtent.x : 0.f;
        invHalfExtent.y = invHalfExtent.y > 0.f ? 1.f / invHalfExtent.y : 0.f;
        invHalfExtent.z = invHalfExtent.z > 0.f ? 1.f / invHalfExtent.z : 0.f;

        BlockReduction<blockSize>(totals, blockStatistics, workItem.count, 0, blockSize,
                                  [&](SampleStatistics &total, uint32_t index) {
                                      float3 samplePos =
                                          samplePositions[blockSampleIndices[index]];

                                      total.AddSample(samplePos, center, invHalfExtent);
                                  });

        __syncthreads();
    }
}

GPU_KERNEL BuildKDTree(KDTreeBuildState *buildState, WorkItem *workItems, KDTreeNode *nodes,
                       uint32_t *sampleIndices, uint32_t *newSampleIndices,
                       SOAFloat3 samplePositions)
{
    __shared__ KDTreeNode node;
    __shared__ WorkItem workItem;

    __shared__ PartitionSharedState partitionState;

    for (uint32_t workItemIndex = blockIdx.x;
         workItemIndex < buildState->numPartitionWorkItems; workItemIndex += gridDim.x)
    {
        if (threadIdx.x == 0)
        {
            partitionState.Clear();
            workItem = workItems[workItemIndex];
            node     = nodes[workItem.nodeIndex];
        }
        __syncthreads();

        uint32_t splitDim = node.GetSplitDim();
        float splitPos    = node.splitPos;

        uint32_t *blockSampleIndices = sampleIndices + node.offset + workItem.offset;
        uint32_t *blockOutputIndices = newSampleIndices + node.offset;

        uint32_t childIndex = node.GetChildIndex();

        // if (threadIdx.x < workItem.count)
        bool active = threadIdx.x < workItem.count;
        {
            PartitionSamples(blockSampleIndices, blockOutputIndices, splitPos, splitDim,
                             node.count, &nodes[childIndex].count,
                             &nodes[childIndex + 1].count, samplePositions, partitionState,
                             active);
        }

        __syncthreads();
    }
}

GPU_KERNEL GetChildNodeOffset(KDTreeBuildState *buildState, KDTreeNode *nodes, uint32_t level,
                              uint32_t *nodeIndices)
{
    uint32_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIndex >= buildState->numNodes) return;

    int parentIndex  = nodeIndices[threadIndex];
    KDTreeNode *node = &nodes[parentIndex];

    if (node->HasChild())
    {
        uint32_t childIndex = node->GetChildIndex();
        uint32_t offset     = node->offset;

        KDTreeNode *left  = &nodes[childIndex];
        KDTreeNode *right = &nodes[childIndex + 1];

        if (node->count != left->count + right->count)
        {
            printf("level: %u, %u %u %u\n", level, node->count, left->count, right->count);
            assert(node->count == left->count + right->count);
        }
        left->offset  = offset;
        right->offset = offset + left->count;

        left->parentIndex  = parentIndex;
        right->parentIndex = parentIndex;
    }
}

GPU_KERNEL FindLeafNodes(KDTreeBuildState *buildState, KDTreeNode *nodes,
                         uint32_t *nodeIndices, uint32_t *numLeafNodes, VMMMapState *states,
                         VMM *vmms)
{
    const uint32_t blockSize = 256;
    assert(blockDim.x == blockSize);
    // uint32_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    // if (threadIndex >= levelInfo->start) return;
    // KDTreeNode &node = nodes[threadIndex];

    for (uint32_t threadIndex = 0; threadIndex < buildState->totalNumNodes; threadIndex++)
    {
        KDTreeNode &node = nodes[threadIndex];
        if (node.IsChild())
        {
            uint32_t index     = (*numLeafNodes)++; // atomicAdd(numLeafNodes, 1);
            node.vmmIndex      = index;
            nodeIndices[index] = threadIndex;

            states[index].numWorkItems          = (node.count + blockSize - 1) / blockSize;
            states[index].numSamples            = node.count;
            states[index].previousLogLikelihood = 0.f;
            states[index].iteration             = 0;

            vmms[index].Initialize();
        }
    }
    printf("%u\n", *numLeafNodes);
}

GPU_KERNEL
WriteSamplesToSOA(const uint32_t *__restrict__ numLeafNodes,
                  uint32_t *__restrict__ leafIndices, uint32_t *__restrict__ sampleIndices,
                  uint32_t *__restrict__ newSampleIndices, KDTreeNode *__restrict__ nodes,
                  SampleData *__restrict__ sampleData, SOASampleData soaSampleData)
{
    uint32_t num = *numLeafNodes;
    if (blockIdx.x >= num) return;
    KDTreeNode &node = nodes[leafIndices[blockIdx.x]];

    if (!node.IsChild()) return;

    uint32_t depth    = 0;
    uint32_t *indices = (depth & 1) ? newSampleIndices : sampleIndices;

    for (uint32_t index = threadIdx.x; index < node.count; index += blockDim.x)
    {
        uint32_t sampleIndex               = indices[node.offset + index];
        soaSampleData[node.offset + index] = sampleData[sampleIndex];
    }
}

GPU_KERNEL PrintStatistics(KDTreeBuildState *buildState, SampleStatistics *stats,
                           KDTreeNode *nodes)
{
    // uint32_t t = 6;
    //
    // for (uint32_t i = 0; i < MAX_COMPONENTS; i++)
    // {
    //     // printf("%f ", splitStats[t].chiSquareTotals[i]);
    //     printf("%f ", vmm[t].weights[i]);
    //     printf("%f ", vmmStats[t].sumWeights[i]);
    // }
    //
    // printf("\n");

    printf("total num nodes: %u\n", buildState->totalNumNodes);
    for (uint32_t i = 0; i < buildState->totalNumNodes; i++)
    {
        // KDTreeNode &node = nodes[i];
        // printf("node: %u %u %f\n", i, node.GetSplitDim(), node.splitPos);

        SampleStatistics &s = stats[i];
        float3 bmin         = s.bounds.GetBoundsMin();
        longlong3 mean      = s.mean;
        // printf("node: %u, %u, %lld %lld %lld\n", i, s.numSamples, mean.x, mean.y,
        // mean.z);
        printf("node: %u, %u, %u, %f %f %f\n", i, nodes[i].count, s.numSamples, bmin.x, bmin.y,
               bmin.z);
    }

    // for (uint32_t i = 191; i < 192; i++)
    // {
    //     KDTreeNode *node                   = &nodes[i];
    //     SampleStatistics *sampleStatistics = &stats[i];
    //     float3 boundsMin                   = sampleStatistics->bounds.GetBoundsMin();
    //     float3 boundsMax                   = sampleStatistics->bounds.GetBoundsMax();
    //     longlong3 mean                     = sampleStatistics->mean;
    //     longlong3 variance                 = sampleStatistics->variance;
    //
    //     uint3 boundsMinUint = sampleStatistics->bounds.GetBoundsMinUint();
    //     uint3 boundsMaxUint = sampleStatistics->bounds.GetBoundsMaxUint();
    //
    //     printf("par: %u %u %f\n", node->parentIndex,
    //     nodes[node->parentIndex].GetSplitDim(),
    //            nodes[node->parentIndex].splitPos);
    //     printf("node: %u %u %f\n", i, node->GetSplitDim(), node->splitPos);
    //     printf("why? %u %u %u %u %u %u\n", boundsMinUint.x, boundsMinUint.y,
    //     boundsMinUint.z,
    //            boundsMaxUint.x, boundsMaxUint.y, boundsMaxUint.z);
    //
    //     printf("off count: %u %u,  bounds: %f %f %f %f %f %f\n mean: %lld %lld %lld "
    //            "var: %lld %lld %lld, %u\n\n",
    //            node->offset, node->count, boundsMin.x, boundsMin.y, boundsMin.z,
    //            boundsMax.x, boundsMax.y, boundsMax.z, mean.x, mean.y, mean.z,
    //            variance.x, variance.y, variance.z, sampleStatistics->numSamples);
    // }
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

        assert(offset <= totalSize);
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
    cudaFree(0);

    const uint32_t blockSize = 256;
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    int numBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, UpdateMixture, blockSize,
                                                  0);

    int maxGridSize = props.multiProcessorCount * numBlocksPerSM;
    printf("blocks, grid: %i %i\n", numBlocksPerSM, maxGridSize);

    uint32_t numSamples  = 1u << 22u; // numVMMs * WARP_SIZE;
    uint32_t maxNumNodes = 1u << 16u;

    CUDAArena allocator;
    allocator.Init(megabytes(512));

    printf("hello world\n");
    Bounds3f *sampleBounds = allocator.Alloc<Bounds3f>(1, 4u);

    KDTreeNode *nodes                  = allocator.Alloc<KDTreeNode>(maxNumNodes, 4u);
    SampleStatistics *sampleStatistics = allocator.Alloc<SampleStatistics>(maxNumNodes, 8u);
    VMM *vmms                          = allocator.Alloc<VMM>(maxNumNodes, 4u);
    VMMStatistics *vmmStatistics       = allocator.Alloc<VMMStatistics>(maxNumNodes, 4);
    SplitStatistics *splitStatistics   = allocator.Alloc<SplitStatistics>(maxNumNodes, 4);

    cudaMemset(nodes, 0, sizeof(KDTreeNode) * maxNumNodes);
    cudaMemset(vmmStatistics, 0, sizeof(VMMStatistics) * maxNumNodes);
    cudaMemset(splitStatistics, 0, sizeof(SplitStatistics) * maxNumNodes);

    SampleData *samples = allocator.Alloc<SampleData>(numSamples, 4u);

    uint32_t *leafNodeIndices = allocator.Alloc<uint32_t>(maxNumNodes + 1, 4u);
    uint32_t *numLeafNodes    = leafNodeIndices;
    leafNodeIndices += 1;
    float *samplePosX = allocator.Alloc<float>(numSamples, 4u);
    float *samplePosY = allocator.Alloc<float>(numSamples, 4u);
    float *samplePosZ = allocator.Alloc<float>(numSamples, 4u);

    float *sampleDirX      = allocator.Alloc<float>(numSamples, 128u);
    float *sampleDirY      = allocator.Alloc<float>(numSamples, 4u);
    float *sampleDirZ      = allocator.Alloc<float>(numSamples, 4u);
    float *sampleWeights   = allocator.Alloc<float>(numSamples, 4u);
    float *samplePdfs      = allocator.Alloc<float>(numSamples, 4u);
    float *sampleDistances = allocator.Alloc<float>(numSamples, 4u);

    SOAFloat3 samplePositions  = {samplePosX, samplePosY, samplePosZ};
    SOAFloat3 sampleDirections = {sampleDirX, sampleDirY, sampleDirZ};

    uint32_t *sampleIndices    = allocator.Alloc<uint32_t>(numSamples, 4u);
    uint32_t *newSampleIndices = allocator.Alloc<uint32_t>(numSamples, 4u);

    WorkItem *reductionWorkItems = allocator.Alloc<WorkItem>(2 * numSamples / blockSize, 4u);
    WorkItem *partitionWorkItems = allocator.Alloc<WorkItem>(2 * numSamples / blockSize, 4u);
    WorkItem *vmmWorkItems       = allocator.Alloc<WorkItem>(numSamples, 4u);

    cudaMemset(vmmWorkItems, 0, sizeof(WorkItem) * numSamples);

    VMMMapState *vmmMapStates = allocator.Alloc<VMMMapState>(maxNumNodes, 4u);

    float3 sceneMin = make_float3(0.f);
    float3 sceneMax = make_float3(100.f);

    printf("amount alloc: %u\n", allocator.offset);

    // int minGridSize, maxBlockSize;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlockSize, ReduceSamples<256>,
    // 256, 0); printf("%i\n", maxBlockSize);

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#if 0
    nvtxRangePush("start");
    InitializeSamples<<<(numSamples + 255) / 256, 256>>>(
        sampleStatistics, sampleBounds, levelInfo, nodes, samples, sampleDirections,
        samplePositions, sampleIndices, numSamples, sceneMin, sceneMax);

    // testing reduce
    uint32_t numBlocks = (numSamples + 255) / 256;
    GetSampleBounds<blockSize><<<numBlocks, blockSize>>>(sampleBounds, samplePosX, samplePosY,
                                                         samplePosZ, numSamples);

    cudaEventRecord(start);

    // what do i need to do
    // - update kd tree logic
    // - link vmm code, get vmm code working

    const uint32_t nodeBlocks = (maxNumNodes + WARP_SIZE - 1) / WARP_SIZE;
    for (uint32_t level = 0; level < MAX_TREE_DEPTH; level++)
    {
        uint32_t *s0 = (level & 1) ? newSampleIndices : sampleIndices;
        uint32_t *s1 = (level & 1) ? sampleIndices : newSampleIndices;

        if (level > 0)
        {
            GetChildNodeOffset<<<nodeBlocks, WARP_SIZE>>>(levelInfo, nodes, level);
        }
        BeginLevel<<<1, 1>>>(levelInfo, reduceQueue, partitionQueue);

        CalculateChildIndices<<<1, 1>>>(levelInfo, nodes);
        CreateWorkItems<blockSize>
            <<<numBlocks, blockSize>>>(reduceQueue, reductionWorkItems, partitionQueue,
                                       partitionWorkItems, nodes, levelInfo, sampleStatistics);

        CalculateNodeStatistics<blockSize>
            <<<numBlocks, blockSize>>>(reduceQueue, reductionWorkItems, nodes,
                                       sampleStatistics, samplePositions, s0, sampleBounds);

        CalculateSplitLocations<<<nodeBlocks, WARP_SIZE>>>(levelInfo, nodes, sampleStatistics,
                                                           sampleBounds);

        BuildKDTree<<<numBlocks, blockSize>>>(partitionQueue, partitionWorkItems, level,
                                              levelInfo, nodes, s0, s1, samplePositions);
    }

    // TODO: reuse temporary buffers from kd tree build
    cudaEventRecord(stop);

    // VMM update
    SOASampleData soaSampleData;
    soaSampleData.pos       = samplePositions;
    soaSampleData.dir       = sampleDirections;
    soaSampleData.weights   = sampleWeights;
    soaSampleData.pdfs      = samplePdfs;
    soaSampleData.distances = sampleDistances;

    cudaEvent_t vmmStart, vmmStop;
    cudaEventCreate(&vmmStart);
    cudaEventCreate(&vmmStop);
    cudaEventRecord(vmmStart);

    FindLeafNodes<blockSize>
        <<<1, 1>>>(levelInfo, nodes, leafNodeIndices, numLeafNodes, vmmMapStates, vmms);
    WriteSamplesToSOA<<<nodeBlocks, WARP_SIZE>>>(numLeafNodes, leafNodeIndices, sampleIndices,
                                                 newSampleIndices, nodes, samples,
                                                 soaSampleData);

    UpdateMixture<<<256, 512>>>(vmms, vmmStatistics, soaSampleData, leafNodeIndices,
                                numLeafNodes, nodes);

    UpdateSplitStatistics<<<numBlocks, blockSize>>>(vmms, vmmStatistics, splitStatistics,
                                                    soaSampleData, leafNodeIndices,
                                                    numLeafNodes, nodes);
    SplitComponents<<<(maxNumNodes + WARP_SIZE - 1) / WARP_SIZE, WARP_SIZE>>>(
        vmms, vmmStatistics, splitStatistics, soaSampleData, leafNodeIndices, numLeafNodes,
        nodes, vmmQueue, vmmWorkItems);

    PartialUpdateMixture<<<256, blockSize>>>(vmms, vmmStatistics, soaSampleData,
                                             leafNodeIndices, numLeafNodes, nodes, vmmQueue,
                                             vmmWorkItems);

    MergeComponents<<<256, blockSize>>>(vmms, vmmStatistics, splitStatistics, soaSampleData,
                                        leafNodeIndices, numLeafNodes, nodes);

    nvtxRangePop();
    cudaEventRecord(vmmStop);
    PrintStatistics<<<1, 1>>>(sampleStatistics, nodes, levelInfo, numLeafNodes, vmms,
                              vmmMapStates, splitStatistics, vmmStatistics);
#endif

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    // CPU execution is asynchronous, so we must wait for the GPU to finish
    // before the program exits (otherwise the printfs might get cut off).
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("KD Tree GPU Time: %f ms\n", milliseconds);

    // cudaEventElapsedTime(&milliseconds, vmmStart, vmmStop);
    // printf("VMM GPU Time: %f ms\n", milliseconds);
}

} // namespace rt

int main()
{
    rt::test();
    return 0;
}
