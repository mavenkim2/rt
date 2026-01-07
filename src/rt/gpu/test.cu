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
#include <cuda_runtime.h>
#include <type_traits>

#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCuda.h"

#define PI             3.14159265358979323846f
#define MAX_COMPONENTS 32
#define WARP_SIZE      32
#define WARP_SHIFT     5u

static_assert(MAX_COMPONENTS <= WARP_SIZE, "too many max components");

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

template <>
__device__ int3 WarpReduceSum(int3 val)
{
    val.x = WarpReduceSum(val.x);
    val.y = WarpReduceSum(val.y);
    val.z = WarpReduceSum(val.z);
    return val;
}

template <>
__device__ longlong3 WarpReduceSum(longlong3 val)
{
    val.x = WarpReduceSum(val.x);
    val.y = WarpReduceSum(val.y);
    val.z = WarpReduceSum(val.z);
    return val;
}

template <>
__device__ uint3 WarpReduceSum(uint3 val)
{
    val.x = WarpReduceSum(val.x);
    val.y = WarpReduceSum(val.y);
    val.z = WarpReduceSum(val.z);
    return val;
}

template <typename T>
__device__ T WarpReduceMin(T val)
{
#pragma unroll
    for (int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1)
    {
        T temp = __shfl_down_sync(0xffffffff, val, offset);
        val    = min(val, temp);
    }
    return val;
}

template <>
__device__ float3 WarpReduceMin(float3 val)
{
    val.x = WarpReduceMin(val.x);
    val.y = WarpReduceMin(val.y);
    val.z = WarpReduceMin(val.z);
    return val;
}

template <>
__device__ uint3 WarpReduceMin(uint3 val)
{
    val.x = WarpReduceMin(val.x);
    val.y = WarpReduceMin(val.y);
    val.z = WarpReduceMin(val.z);
    return val;
}

template <typename T>
__device__ T WarpReduceMax(T val)
{
#pragma unroll
    for (int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1)
    {
        T temp = __shfl_down_sync(0xffffffff, val, offset);
        val    = max(val, temp);
    }
    return val;
}

template <>
__device__ float3 WarpReduceMax(float3 val)
{
    val.x = WarpReduceMax(val.x);
    val.y = WarpReduceMax(val.y);
    val.z = WarpReduceMax(val.z);
    return val;
}

template <>
__device__ uint3 WarpReduceMax(uint3 val)
{
    val.x = WarpReduceMax(val.x);
    val.y = WarpReduceMax(val.y);
    val.z = WarpReduceMax(val.z);
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

inline __device__ uint32_t FloatToOrderedUint(float f)
{
    uint32_t u    = __float_as_uint(f);
    uint32_t mask = (u & 0x80000000) ? ~0u : 0x80000000;
    return u ^ mask;
}

inline __device__ float OrderedUintToFloat(uint32_t u)
{
    uint32_t mask = (u & 0x80000000) ? 0x80000000 : ~0u;
    return __uint_as_float(u ^ mask);
}

inline __device__ uint3 FloatToOrderedUint(float3 f)
{
    uint3 u;
    u.x = FloatToOrderedUint(f.x);
    u.y = FloatToOrderedUint(f.y);
    u.z = FloatToOrderedUint(f.z);
    return u;
}

inline __device__ float3 OrderedUintToFloat(uint3 u)
{
    float3 f;
    f.x = OrderedUintToFloat(u.x);
    f.y = OrderedUintToFloat(u.y);
    f.z = OrderedUintToFloat(u.z);
    return f;
}

template <typename T>
struct Bounds
{
    T boundsMin;
    T boundsMax;

    __device__ void Extend(T value)
    {
        boundsMin = min(boundsMin, value);
        boundsMax = max(boundsMax, value);
    }

    __device__ void WarpReduce()
    {
        boundsMin = WarpReduceMin(boundsMin);
        boundsMax = WarpReduceMax(boundsMax);
    }

    __device__ void AtomicMerge(const Bounds &other) {}
};

struct SOAFloat3Ref
{
    float *x, *y, *z;
    const uint32_t index;

    __device__ SOAFloat3Ref &operator=(const float3 &val)
    {
        x[index] = val.x;
        y[index] = val.y;
        z[index] = val.z;
        return *this;
    }

    __device__ operator float3() const { return make_float3(x[index], y[index], z[index]); }

    __device__ SOAFloat3Ref &operator=(const SOAFloat3Ref &other)
    {
        float3 val = (float3)other;
        return (*this = val);
    }
};

struct SOAFloat3
{
    float *x;
    float *y;
    float *z;

    __device__ SOAFloat3Ref operator[](uint32_t index) { return {x, y, z, index}; }

    __device__ float3 operator[](uint32_t index) const
    {
        return make_float3(x[index], y[index], z[index]);
    }

    __device__ SOAFloat3 operator+(uint32_t offset)
    {
        SOAFloat3 other;
        other.x = x + offset;
        other.y = y + offset;
        other.z = z + offset;
        return other;
    }
};

struct SampleData
{
    float3 pos;
    float3 dir;
    float weight;
    float pdf;
};

struct SOASampleDataRef
{
    SOAFloat3Ref pos;
    SOAFloat3Ref dir;
    float *weights;
    float *pdfs;
    const uint32_t index;

    __device__ SOASampleDataRef &operator=(const SampleData &val)
    {
        pos            = val.pos;
        dir            = val.dir;
        weights[index] = val.weight;
        pdfs[index]    = val.pdf;
        return *this;
    }

    // __device__ operator float3() const { return make_float3(x[index], y[index], z[index]); }
    //
    // __device__ SOASampleDataRef &operator=(const SOAFloat3Ref &other)
    // {
    //     float3 val = (float3)other;
    //     return (*this = val);
    // }
};

struct SOASampleData
{
    SOAFloat3 pos;
    SOAFloat3 dir;

    float *weights;
    float *pdfs;

    __device__ SOASampleDataRef operator[](uint32_t index)
    {
        return {pos[index], dir[index], weights, pdfs, index};
    }

    __device__ SampleData operator[](uint32_t index) const
    {
        return {pos[index], dir[index], weights[index], pdfs[index]};
    }
};

template <>
struct Bounds<float3>
{
private:
    uint3 boundsMin;
    uint3 boundsMax;

public:
    __device__ Bounds()
    {
        const uint pos_inf = 0xff800000;
        const uint neg_inf = ~0xff800000;

        boundsMin = make_uint3(pos_inf);
        boundsMax = make_uint3(neg_inf);
    }

    __device__ void Clear()
    {
        const uint pos_inf = 0xff800000;
        const uint neg_inf = ~0xff800000;

        boundsMin = make_uint3(pos_inf);
        boundsMax = make_uint3(neg_inf);
    }

    __device__ void Extend(float3 value)
    {
        uint3 val = FloatToOrderedUint(value);
        boundsMin = min(boundsMin, val);
        boundsMax = max(boundsMax, val);
    }

    __device__ void WarpReduce()
    {
        boundsMin = WarpReduceMin(boundsMin);
        boundsMax = WarpReduceMax(boundsMax);
    }

    __device__ void AtomicMerge(const Bounds &other)
    {
        atomicMin(&boundsMin, other.boundsMin);
        atomicMax(&boundsMax, other.boundsMax);
    }

    __device__ void SetBoundsMin(float3 f) { boundsMin = FloatToOrderedUint(f); }
    __device__ void SetBoundsMax(float3 f) { boundsMax = FloatToOrderedUint(f); }

    __device__ float3 GetBoundsMin() const { return OrderedUintToFloat(boundsMin); }
    __device__ float3 GetBoundsMax() const { return OrderedUintToFloat(boundsMax); }

    __device__ uint3 GetBoundsMinUint() const { return boundsMin; }
    __device__ uint3 GetBoundsMaxUint() const { return boundsMax; }

    __device__ float3 GetCenter() const
    {
        float3 bMin = GetBoundsMin();
        float3 bMax = GetBoundsMax();

        return (bMin + bMax) * 0.5f;
    }

    __device__ float3 GetHalfExtent() const
    {
        float3 bMin = GetBoundsMin();
        float3 bMax = GetBoundsMax();

        return (bMax - bMin) * 0.5f;
    }
};

typedef Bounds<float3> Bounds3f;

struct LevelInfo
{
    uint32_t start;
    uint32_t count;
    uint32_t childCount;
};

struct KDTreeNode
{
    float splitPos;
    uint32_t childIndex_dim;

    // TODO: remove?
    int parentIndex;

    uint32_t offset;
    uint32_t count;
    uint32_t vmmIndex;

    __device__ uint32_t GetChildIndex() const { return (childIndex_dim << 2) >> 2; }
    __device__ void SetChildIndex(uint32_t childIndex)
    {
        assert(childIndex < (1u << 30u));
        childIndex_dim |= childIndex;
    }
    __device__ uint32_t GetSplitDim() const { return childIndex_dim >> 30; }
    __device__ void SetSplitDim(uint32_t dim)
    {
        assert(dim < 3);
        childIndex_dim |= dim << 30;
    }
    __device__ bool HasChild() const { return childIndex_dim != ~0u; }
    __device__ bool IsChild() const { return (childIndex_dim >> 30u) == 3u; }
};

#define INTEGER_BINS                     float(1 << 18)
#define INTEGER_SAMPLE_STATS_BOUND_SCALE (1.0f + 2.f / INTEGER_BINS)
#define MAX_TREE_DEPTH                   32
#define MAX_SAMPLES_PER_LEAF             32000

struct SampleStatistics
{
    Bounds3f bounds;
    // int3 mean;
    // int3 variance;
    longlong3 mean;
    longlong3 variance;

    uint32_t numSamples;

    __device__ SampleStatistics()
        : numSamples(0), bounds(), mean(make_longlong3(0, 0, 0)),
          variance(make_longlong3(0, 0, 0))
    {
    }

    __device__ void Clear()
    {
        numSamples = 0;
        bounds.Clear();
        mean     = make_longlong3(0, 0, 0);
        variance = make_longlong3(0, 0, 0);
    }

    __device__ void AddSample(float3 pos, float3 center, float3 invHalfExtent)
    {
        numSamples++;
        float3 tmpSample = (pos - center) * invHalfExtent;

        float3 tmpVariance = ((tmpSample * tmpSample) * INTEGER_BINS);

        mean += make_longlong3(tmpSample * INTEGER_BINS);
        variance += make_longlong3(tmpVariance);

        bounds.Extend(pos);
    }

    __device__ void WarpReduce()
    {
        bounds.WarpReduce();
        mean       = WarpReduceSum(mean);
        variance   = WarpReduceSum(variance);
        numSamples = WarpReduceSum(numSamples);
    }

    __device__ inline void AtomicMerge(const SampleStatistics &other)
    {
        bounds.AtomicMerge(other.bounds);
        atomicAdd(&mean, other.mean);
        atomicAdd(&variance, other.variance);
        atomicAdd(&numSamples, other.numSamples);
    }

    __device__ void ConvertToFloat(float3 center, float3 halfExtent, float3 &outMean,
                                   float3 &outVariance)
    {
        longlong3 intMean = mean;
        longlong3 intVar  = variance;

        float invNumSamples = 1.f / numSamples;
        float3 sampleMean   = make_float3(intMean) * invNumSamples;
        sampleMean /= INTEGER_BINS;

        float3 sampleMeanBin = sampleMean;
        sampleMean           = sampleMean * halfExtent + center;

        float3 lowerCollectedSampleBound = bounds.GetBoundsMin();
        float3 upperCollectedSampleBound = bounds.GetBoundsMax();

        float3 collectedSampleBoundExtend =
            (upperCollectedSampleBound - lowerCollectedSampleBound);
        float3 halfCollectedSampleBoundExtend =
            (upperCollectedSampleBound - lowerCollectedSampleBound) * 0.5f;
        float3 halfBinSize = make_float3(0.5f / INTEGER_BINS) * halfExtent;

        sampleMean.x = sampleMean.x <= lowerCollectedSampleBound.x
                           ? lowerCollectedSampleBound.x +
                                 min(halfCollectedSampleBoundExtend.x, halfBinSize.x)
                           : sampleMean.x;
        sampleMean.y = sampleMean.y <= lowerCollectedSampleBound.y
                           ? lowerCollectedSampleBound.y +
                                 min(halfCollectedSampleBoundExtend.y, halfBinSize.y)
                           : sampleMean.y;
        sampleMean.z = sampleMean.z <= lowerCollectedSampleBound.z
                           ? lowerCollectedSampleBound.z +
                                 min(halfCollectedSampleBoundExtend.z, halfBinSize.z)
                           : sampleMean.z;
        sampleMean.x = sampleMean.x >= upperCollectedSampleBound.x
                           ? upperCollectedSampleBound.x -
                                 min(halfCollectedSampleBoundExtend.x, halfBinSize.x)
                           : sampleMean.x;
        sampleMean.y = sampleMean.y >= upperCollectedSampleBound.y
                           ? upperCollectedSampleBound.y -
                                 min(halfCollectedSampleBoundExtend.y, halfBinSize.y)
                           : sampleMean.y;
        sampleMean.z = sampleMean.z >= upperCollectedSampleBound.z
                           ? upperCollectedSampleBound.z -
                                 min(halfCollectedSampleBoundExtend.z, halfBinSize.z)
                           : sampleMean.z;

        float3 sampleVariance =
            (make_float3(intVar.x, intVar.y, intVar.z) / (INTEGER_BINS)) * invNumSamples;
        sampleVariance -= sampleMeanBin * sampleMeanBin;
        sampleVariance = make_float3(max(0.f, sampleVariance.x), max(0.f, sampleVariance.y),
                                     max(0.f, sampleVariance.z));

        sampleVariance = sampleVariance * (halfExtent * halfExtent);
        sampleVariance.x =
            min(collectedSampleBoundExtend.x * collectedSampleBoundExtend.x, sampleVariance.x);
        sampleVariance.y =
            min(collectedSampleBoundExtend.y * collectedSampleBoundExtend.y, sampleVariance.y);
        sampleVariance.z =
            min(collectedSampleBoundExtend.z * collectedSampleBoundExtend.z, sampleVariance.z);

        sampleVariance.x =
            upperCollectedSampleBound.x - lowerCollectedSampleBound.x < 2.f / INTEGER_BINS
                ? 0.f
                : sampleVariance.x;
        sampleVariance.y =
            upperCollectedSampleBound.y - lowerCollectedSampleBound.y < 2.f / INTEGER_BINS
                ? 0.f
                : sampleVariance.y;
        sampleVariance.z =
            upperCollectedSampleBound.z - lowerCollectedSampleBound.z < 2.f / INTEGER_BINS
                ? 0.f
                : sampleVariance.z;

        outVariance = sampleVariance;
        outMean     = sampleMean;
    }
};

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

struct VMM
{
    float kappas[MAX_COMPONENTS];
    float directionX[MAX_COMPONENTS];
    float directionY[MAX_COMPONENTS];
    float directionZ[MAX_COMPONENTS];
    float weights[MAX_COMPONENTS];

    uint32_t numComponents;

    inline __device__ void Initialize()
    {
        const float gr = 1.618033988749895f;
        numComponents  = MAX_COMPONENTS / 2;

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

            WriteDirection(n, uniformDirection);
            if (n < numComponents)
            {
                kappas[n]  = kappa;
                weights[n] = weight;
            }
            else
            {
                kappas[n]  = 0.0f;
                weights[n] = 0.0f;
                // vmm.normalizations[i][j] = ONE_OVER_FOUR_PI;
                // vmm.eMinus2Kappa[i][j] = 1.0f;
                // vmm._meanCosines[i][j] = 0.0f;
            }
        }
    }

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

struct VMMStatistics
{
    float weightedLogLikelihood;
    float sumSampleWeights;
    float sumWeights[MAX_COMPONENTS];
    float3 sumWeightedDirections[MAX_COMPONENTS];

    uint32_t numSamples;

    __device__ void SetComponent(uint32_t componentIndex, float sumWeight,
                                 float3 sumWeightedDirection)
    {
        assert(componentIndex < MAX_COMPONENTS);
        sumWeights[componentIndex]            = sumWeight;
        sumWeightedDirections[componentIndex] = sumWeightedDirection;
    }
};

struct SplitStatistics
{
    float3 covarianceTotals[MAX_COMPONENTS];
    float chiSquareTotals[MAX_COMPONENTS];
    float sumWeights[MAX_COMPONENTS];
    uint32_t numSamples[MAX_COMPONENTS];

    __device__ void SetComponent(uint32_t componentIndex, float3 covarianceTotal,
                                 float chiSquareTotal, float sumWeight, uint32_t num)
    {
        covarianceTotals[componentIndex] = covarianceTotal;
        chiSquareTotals[componentIndex]  = chiSquareTotal;
        sumWeights[componentIndex]       = sumWeight;
        numSamples[componentIndex]       = num;
    }
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

struct Queue
{
    uint32_t readOffset;
    uint32_t writeOffset;
};

struct WorkItem
{
    uint32_t nodeIndex;
    uint32_t offset;
    uint32_t count;
};

struct VMMMapState
{
    int numWorkItems;
    uint32_t numSamples;
    uint32_t sampleOffset;
    float previousLogLikelihood;
    uint32_t iteration;
};

__global__ void InitializeSamples(SampleStatistics *statistics, Bounds3f *sampleBounds,
                                  LevelInfo *levelInfo, KDTreeNode *nodes, SampleData *samples,
                                  SOAFloat3 sampleDirections, SOAFloat3 samplePositions,
                                  uint32_t *__restrict__ sampleIndices, uint32_t numSamples,
                                  float3 sceneMin, float3 sceneMax)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *sampleBounds = Bounds3f();

        nodes[0].count  = numSamples;
        nodes[0].offset = 0;
        // nodes[0].childIndex_dim = 1;
        nodes[0].parentIndex = -1;

        levelInfo[0].start      = 0;
        levelInfo[0].count      = 0;
        levelInfo[0].childCount = 1;

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
inline __device__ void
UpdateMixtureParameters(VMM &sharedVMM, VMMStatistics &sharedStatistics,
                        VMMStatistics &previousStatistics, SOAFloat3 sampleDirections,
                        float *sampleWeights, float &sumSampleWeights, uint32_t sampleOffset,
                        uint32_t sampleCount, uint32_t mask = 0, bool doPrint = false)
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
        // if (threadIndex == 0 && doPrint)
        // {
        //     printf("wtf");
        // }
        if (threadIndex < MAX_COMPONENTS)
        {
            sharedStatistics.sumWeights[threadIndex]            = 0.f;
            sharedStatistics.sumWeightedDirections[threadIndex] = make_float3(0.f);
        }
        if (threadIndex == 0) sharedStatistics.weightedLogLikelihood = 0.f;

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

            hasData    = V > 0.f;
            float invV = hasData ? 1.f / V : 0.f;

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
            if (iteration == 0 && doPrint)
            {
                printf("weight: %u %f %f\n", threadIndex, sharedVMM.kappas[threadIndex],
                       sharedStatistics.sumWeights[threadIndex]);
            }
            sharedStatistics.sumWeights[threadIndex] *= normWeight;

            // Update weights
            const uint32_t totalNumSamples = sampleCount; // + previousStatistics.numSamples;
            float weight =
                sharedStatistics
                    .sumWeights[threadIndex]; // + previousStatistics.sumWeights[threadIndex];
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
                const float currentEstimationWeight = float(sampleCount) / totalNumSamples;
                // const float previousEstimationWeight = 1.f - currentEstimationWeight;

                float3 currentMeanDirection =
                    sharedStatistics.sumWeights[threadIndex] > 0.f
                        ? sharedStatistics.sumWeightedDirections[threadIndex] /
                              sharedStatistics.sumWeights[threadIndex]
                        : make_float3(0.f);
                // float3 previousMeanDirection =
                //     previousStatistics.sumWeights[threadIndex] > 0.f
                //         ? previousStatistics.sumWeightedDirections[threadIndex] /
                //               previousStatistics.sumWeights[threadIndex]
                //         : make_float3(0.f);

                float3 meanDirection = currentMeanDirection *
                                       currentEstimationWeight; // + previousMeanDirection *
                                                                // previousEstimationWeight;

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

            // if (iteration >= 4 && threadIndex == 0)
            // {
            //     printf("previousLogLikelihood: %f, %u\n", previousLogLikelihood, iteration);
            // }
        }
    }
}

// TODO IMPORTANT:
// - handle unassigned components in MAP update
// - make sure to update statistics and rewrite to memory
// - make sure to get chi square statistics fromm previous update
// - proper criteria for splitting/merging

__global__ void
UpdateMixture(VMM *__restrict__ vmms, VMMStatistics *__restrict__ currentStatistics,
              const SOASampleData samples, const uint32_t *__restrict__ leafNodeIndices,
              const uint32_t *__restrict__ numLeafNodes, const KDTreeNode *__restrict__ nodes)
{
    __shared__ VMMStatistics sharedStatistics;
    __shared__ VMMStatistics previousStatistics;
    __shared__ VMM sharedVMM;
    __shared__ float sumSampleWeights;

    const uint32_t threadIndex = threadIdx.x;

    if (blockIdx.x >= *numLeafNodes) return;

    uint32_t vmmIndex      = leafNodeIndices[blockIdx.x];
    const KDTreeNode &node = nodes[vmmIndex];
    uint32_t sampleCount   = node.count;
    uint32_t offset        = node.offset;

    if (threadIndex == 0)
    {
        // TODO: need to make sure the VMM maps to the right node
        sharedVMM = vmms[node.vmmIndex];
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

    const uint32_t numSampleBatches = (sampleCount + blockDim.x - 1) / blockDim.x;

    UpdateMixtureParameters(sharedVMM, sharedStatistics, previousStatistics, samples.dir,
                            samples.weights, sumSampleWeights, offset, sampleCount, 0,
                            node.vmmIndex == 0);

    __syncthreads();

    if (threadIndex == 0)
    {
        vmms[node.vmmIndex]                               = sharedVMM;
        currentStatistics[node.vmmIndex].sumSampleWeights = sumSampleWeights;
        currentStatistics[node.vmmIndex].numSamples       = sampleCount;
    }

#if 0

    // Reuse shared memory
    float *mergeKappas      = sharedSumWeights;
    float *mergeProducts    = sharedChiSquareTotals;
    float3 *mergeDirections = sharedCovarianceTotals;

    // Merging
    {
        const uint32_t numComponents = sharedVMM.numComponents;

        const uint32_t numPairs        = (numComponents * numComponents - numComponents) / 2;
        const uint32_t numMergeBatches = (numPairs + blockDim.x - 1) / blockDim.x;

        if (threadIndex < sharedVMM.numComponents)
        {
            float kappa = sharedVMM.kappas[threadIndex];
            if (blockIdx.x == 0)
            {
                printf("weight: %f\n", sharedVMM.weights[threadIndex]);
                printf("%u, thread: %u, %f\n", numComponents, threadIndex, kappa);
                float3 dir = sharedVMM.ReadDirection(threadIndex);
                printf("%f %f %f\n", dir.x, dir.y, dir.z);
            }
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

            // TODO: instead of recomputing the chi square estimate for
            // every merge iteration, cache in shared memory and update only the pairs
            // for the components that have changed

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

                if (blockIdx.x == 0)
                {
                    // printf("%u, thread: %u, %u %u %f\n", numComponents, threadIndex,
                    //        componentIndex0, componentIndex1, chiSquareIJ);
                }
            }
        }
    }
#endif
}

__global__ void UpdateSplitStatistics(VMM *__restrict__ vmms,
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

__global__ void SplitComponents(VMM *__restrict__ vmms,
                                VMMStatistics *__restrict__ currentStatistics,
                                SplitStatistics *__restrict__ splitStatistics,
                                const SOASampleData samples,
                                const uint32_t *__restrict__ leafNodeIndices,
                                const uint32_t *__restrict__ numLeafNodes,
                                const KDTreeNode *__restrict__ nodes)
{
    const float splittingThreshold = 0.5f;
    const float maxKappa           = 32000.f;
    const float maxMeanCosine      = KappaToMeanCosine(maxKappa);
    static_assert(MAX_COMPONENTS <= WARP_SIZE);

    if (blockIdx.x >= *numLeafNodes) return;

    uint32_t nodeIndex     = leafNodeIndices[blockIdx.x];
    const KDTreeNode &node = nodes[nodeIndex];
    uint32_t sampleCount   = node.count;
    uint32_t offset        = node.offset;

    const uint32_t threadIndex = threadIdx.x;

    VMM &vmm                     = vmms[node.vmmIndex];
    SplitStatistics &splitStats  = splitStatistics[node.vmmIndex];
    VMMStatistics &vmmStatistics = currentStatistics[node.vmmIndex];

    __shared__ uint32_t sharedSplitMask;
    // Split components
    if (threadIndex < MAX_COMPONENTS)
    {
        // numUsedSamples = WarpReduceSum(numUsedSamples);
        // numUsedSamples = WarpReadLaneFirst(numUsedSamples);

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

                float kappa  = MeanCosineToKappa(meanCosine0);
                float weight = vmm.weights[componentIndex] * 0.5f;

                vmm.UpdateComponent(componentIndex, kappa, meanDirection0, weight);
                vmm.UpdateComponent(newComponentIndex, kappa, meanDirection1, weight);

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
            sharedSplitMask = splitMask | maskExtend;
        }
    }

    __syncthreads();

    // Update split components
    // if (sharedSplitMask)
    // {
    //     UpdateMixtureParameters<true>(sharedVMM, sharedStatistics, previousStatistics,
    //                                   samples.dir, samples.weights, sumSampleWeights,
    //                                   offset, sampleCount, sharedSplitMask);
    // }
}

template <uint32_t blockSize>
__global__ void CreateVMMWorkItems(const uint32_t *leafNodeIndices,
                                   const uint32_t *numLeafNodes, KDTreeNode *nodes,
                                   WorkItem *workItems, Queue *queue)
{
    if (blockIdx.x < *numLeafNodes)
    {
        KDTreeNode &node      = nodes[leafNodeIndices[blockIdx.x]];
        uint32_t numWorkItems = (node.count + blockSize - 1) / blockSize;

        __shared__ uint32_t workItemOffset;

        if (threadIdx.x == 0)
        {
            workItemOffset = atomicAdd(&queue->writeOffset, numWorkItems);
        }

        __syncthreads();

        for (uint32_t i = threadIdx.x; i < numWorkItems; i += blockDim.x)
        {
            WorkItem workItem;
            workItem.nodeIndex            = node.vmmIndex;
            workItem.offset               = node.offset + blockSize * i;
            workItem.count                = min(blockSize, node.count - blockSize * i);
            workItems[workItemOffset + i] = workItem;

            // if (blockIdx.x == 0)
            // {
            //     printf("%u %u %u %u %u %u\n", i, node.vmmIndex, workItem.offset,
            //            workItem.count, node.count, i * blockSize);
            // }
        }
    }
}

#if 0
template <uint32_t blockSize>
__global__ void UpdateMixturePersistent(
    Queue *queue, WorkItem *workItems, VMM *__restrict__ vmms,
    // VMMStatistics *__restrict__ previousStatisticsArray,
    VMMStatistics *__restrict__ currentStatistics, const SOASampleData samples,
    const uint32_t *__restrict__ leafNodeIndices, uint32_t *__restrict__ numLeafNodes,
    const KDTreeNode *__restrict__ nodes, VMMMapState *__restrict__ vmmMapStates,
    uint32_t queueMax)
{
    __shared__ VMMStatistics previousStatistics;
    __shared__ VMM sharedVMM;
    __shared__ float sumSampleWeights;
    __shared__ WorkItem workItem;
    __shared__ uint32_t workItemIndex;
    __shared__ uint32_t workItemOffset;
    __shared__ bool doUpdate;
    __shared__ bool processed;

    const uint32_t threadIndex = threadIdx.x;

    // uint32_t vmmIndex      = leafNodeIndices[blockIdx.x];
    // const KDTreeNode &node = nodes[vmmIndex];
    // uint32_t sampleCount   = node.count;
    // uint32_t offset        = node.offset;

    // if (threadIndex == 0)
    // {
    //     // TODO: need to make sure the VMM maps to the right node
    //     sharedVMM = vmms[node.vmmIndex];
    //     for (uint32_t i = 0; i < sharedVMM.numComponents; i++)
    //     {
    //         previousStatistics.sumWeights[i]            = 0.f;
    //         previousStatistics.sumWeightedDirections[i] = make_float3(0.f);
    //     }
    //     previousStatistics.weightedLogLikelihood = 0.f;
    //     previousStatistics.numSamples            = 0;
    //
    //     // TODO: not everytime
    //     sharedVMM.Initialize();
    // }

    // __syncthreads();

    const float splittingThreshold = 0.5f;

    constexpr bool masked = true;
    uint32_t mask         = ~0u;

    const uint32_t maxNumIterations     = 100;
    const float weightPrior             = 0.01f;
    const float meanCosinePrior         = 0.f;
    const float meanCosinePriorStrength = 0.2f;
    const float maxKappa                = 32000.f;
    const float maxMeanCosine           = KappaToMeanCosine(maxKappa);
    const float convergenceThreshold    = 0.005f;

    float previousLogLikelihood = 0.f;

    if (threadIndex == 0)
    {
        processed = true;
    }
    __syncthreads();

    for (;;)
    {
        if (threadIndex == 0)
        {
            if (processed)
            {
                workItemIndex = atomicAdd(&queue->readOffset, 1);
                processed     = false;
            }
            // volatile???
            volatile WorkItem *w = &workItems[workItemIndex];
            workItem             = *(WorkItem *)w;
            volatile VMM *v      = &vmms[workItem.nodeIndex];
            sharedVMM            = *(VMM *)v;
            // if (workItemIndex < queue->writeOffset)
            // {
            // }
        }

        __syncthreads();

        if (workItem.count == 0)
        {
            __threadfence();
            if (*(volatile uint32_t *)numLeafNodes == 0)
            {
                break;
            }
            continue;
        }

        volatile VMMStatistics *vVmmStatistics = &currentStatistics[workItem.nodeIndex];
        VMMStatistics &vmmStatistics           = *(VMMStatistics *)vVmmStatistics;

        // Weighted Expectation
        VMMStatistics statistics;
        uint32_t sampleIndex = threadIdx.x + workItem.offset;
        bool hasData         = threadIdx.x < workItem.count;

        float V                = 0.f;
        float3 sampleDirection = make_float3(0.f);
        float sampleWeight     = 1.f;

        if (hasData)
        {
            sampleDirection = samples.dir[sampleIndex];
            sampleWeight    = samples.weights[sampleIndex];
            V = SoftAssignment(sharedVMM, sampleDirection, statistics.sumWeights);
        }

        hasData    = V > 0.f;
        float invV = hasData ? 1.f / V : 0.f;

        statistics.weightedLogLikelihood = hasData ? sampleWeight * log(V) : 0.f;

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
                atomicAdd(&vmmStatistics.sumWeights[i], sumWeights);
                atomicAdd(&vmmStatistics.sumWeightedDirections[i], sumWeightedDirections);
            }
        }

        float weightedLogLikelihood = WarpReduceSum(statistics.weightedLogLikelihood);
        if ((threadIndex & (WARP_SIZE - 1)) == 0)
        {
            // printf("log like: %f\n", weightedLogLikelihood);
            atomicAdd(&vmmStatistics.weightedLogLikelihood, weightedLogLikelihood);
        }

        if (threadIdx.x == 0)
        {
            __threadfence();
            uint32_t index = atomicSub(&vmmMapStates[workItem.nodeIndex].numWorkItems, 1);
            assert(index >= 1);
            doUpdate = index == 1;
        }

        __syncthreads();

        // Weighted MAP Update
        if (doUpdate && threadIndex < MAX_COMPONENTS)
        {
            // Normalize
            volatile VMM *vVMM                     = &vmms[workItem.nodeIndex];
            volatile VMMStatistics *vVmmStatistics = &currentStatistics[workItem.nodeIndex];

            VMM &vmm                     = *(VMM *)vVMM;
            VMMStatistics &vmmStatistics = *(VMMStatistics *)vVmmStatistics;

            float componentWeight = vmmStatistics.sumWeights[threadIndex];
            float normWeight      = WarpReduceSumBroadcast(componentWeight);

            if (threadIndex == 0) sumSampleWeights = normWeight;

            // TODO:
            uint32_t sampleCount = vmmMapStates[workItem.nodeIndex].numSamples;
            normWeight = normWeight > FLT_EPSILON ? float(sampleCount) / normWeight : 0.f;
            vmmStatistics.sumWeights[threadIndex] *= normWeight;

            // Update weights
            const uint32_t totalNumSamples = sampleCount; // + previousStatistics.numSamples;
            float weight =
                vmmStatistics
                    .sumWeights[threadIndex]; // + previousStatistics.sumWeights[threadIndex];
            weight =
                threadIndex >= vmm.numComponents
                    ? 0.f
                    : (weightPrior + weight) / (weightPrior * vmm.numComponents + sampleCount);

            const bool threadIsEnabled = mask & (1u << threadIndex);

            float sumMaskedWeights   = threadIsEnabled ? weight : 0.f;
            float sumUnmaskedWeights = threadIsEnabled ? 0.f : weight;

            if (!masked || threadIsEnabled)
            {
                vmm.weights[threadIndex] = weight;
            }

            if constexpr (masked)
            {
                // Makes sure weights always sum to one
                float invSumWeights = 1.f / WarpReduceSumBroadcast(sumMaskedWeights);
                invSumWeights *= 1.f - WarpReduceSumBroadcast(sumUnmaskedWeights);
                vmm.weights[threadIndex] *= invSumWeights;
            }

            // Update kappas and directions
            if (!masked || threadIsEnabled)
            {
                const float currentEstimationWeight  = float(sampleCount) / totalNumSamples;
                const float previousEstimationWeight = 1.f - currentEstimationWeight;

                float3 currentMeanDirection =
                    vmmStatistics.sumWeights[threadIndex] > 0.f
                        ? vmmStatistics.sumWeightedDirections[threadIndex] /
                              vmmStatistics.sumWeights[threadIndex]
                        : make_float3(0.f);
                // float3 previousMeanDirection =
                //     previousStatistics.sumWeights[threadIndex] > 0.f
                //         ? previousStatistics.sumWeightedDirections[threadIndex] /
                //               previousStatistics.sumWeights[threadIndex]
                //         : make_float3(0.f);

                float3 meanDirection = currentMeanDirection * currentEstimationWeight;
                // + previousMeanDirection * previousEstimationWeight;

                float meanCosine = length(meanDirection);

                // TODO: make sure uninitialized components have correct default state?
                if (meanCosine > 0.f)
                {
                    vmm.WriteDirection(threadIndex, meanDirection / meanCosine);
                }

                float partialNumSamples = totalNumSamples * vmm.weights[threadIndex];

                meanCosine = (meanCosinePrior * meanCosinePriorStrength +
                              meanCosine * partialNumSamples) /
                             (meanCosinePriorStrength + partialNumSamples);
                meanCosine = min(meanCosine, maxMeanCosine);
                float kappa =
                    threadIndex < vmm.numComponents ? MeanCosineToKappa(meanCosine) : 0.f;
                vmm.kappas[threadIndex] = kappa;
            }

            if (threadIndex == 0)
            {
                volatile VMMMapState &state = vmmMapStates[workItem.nodeIndex];
                uint32_t iteration          = state.iteration;
                if (iteration > 0)
                {
                    float previousLogLikelihood =
                        vmmMapStates[workItem.nodeIndex].previousLogLikelihood;
                    float logLikelihood = vmmStatistics.weightedLogLikelihood;
                    float relLogLikelihoodDifference =
                        fabs(logLikelihood - previousLogLikelihood) /
                        fabs(previousLogLikelihood);

                    if (iteration + 1 >= maxNumIterations ||
                        relLogLikelihoodDifference < convergenceThreshold)
                    {
                        atomicSub(numLeafNodes, 1);
                        doUpdate = false;
                    }
                    else
                    {
                        state.previousLogLikelihood = logLikelihood;
                        state.iteration++;
                    }
                }
                else
                {
                    float logLikelihood         = vmmStatistics.weightedLogLikelihood;
                    state.previousLogLikelihood = logLikelihood;
                    state.iteration++;
                }
            }
            __threadfence();
        }

        if (threadIndex == 0)
        {
            processed = true;
        }

        __syncthreads();

        if (doUpdate)
        {
            // TODO: volatile?
            volatile VMMMapState &state = vmmMapStates[workItem.nodeIndex];
            VMMStatistics &statistics   = currentStatistics[workItem.nodeIndex];
            uint32_t numWorkItems       = (state.numSamples + blockSize - 1) / blockSize;
            if (threadIdx.x == 0)
            {
                state.numWorkItems = numWorkItems;
                workItemOffset     = atomicAdd(&queue->writeOffset, numWorkItems);

                if (state.iteration >= 4)
                {
                    printf("%f %u %u %u %u\n", state.previousLogLikelihood, workItemOffset,
                           queue->readOffset, numWorkItems, state.iteration);
                }
                if (state.previousLogLikelihood < -30000.f)
                {
                    printf("help me: %f\n", state.previousLogLikelihood);
                }

                if (!(workItemOffset + numWorkItems <= queueMax))
                {
                    assert(0);
                }
            }

            if (threadIndex < MAX_COMPONENTS)
            {
                statistics.sumWeights[threadIndex]            = 0.f;
                statistics.sumWeightedDirections[threadIndex] = make_float3(0.f);
            }

            if (threadIndex == 0)
            {
                statistics.numSamples            = 0;
                statistics.weightedLogLikelihood = 0.f;
            }

            __syncthreads();
            __threadfence();

            for (uint32_t i = threadIdx.x; i < numWorkItems; i += blockDim.x)
            {
                WorkItem newWorkItem;
                newWorkItem.nodeIndex         = workItem.nodeIndex;
                newWorkItem.offset            = state.sampleOffset + blockSize * i;
                newWorkItem.count             = 0;
                workItems[workItemOffset + i] = newWorkItem;
            }

            __threadfence();

            for (uint32_t i = threadIdx.x; i < numWorkItems; i += blockDim.x)
            {
                workItems[workItemOffset + i].count =
                    min(blockSize, state.numSamples - blockSize * i);
            }

            __threadfence();
        }
    }
}
#endif

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
    static_assert(numWarps <= WARP_SIZE);
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

template <uint32_t blockSize>
__global__ void GetSampleBounds(Bounds3f *sampleBounds, float *sampleX, float *sampleY,
                                float *sampleZ, uint32_t numSamples)
{
    static constexpr uint32_t numWarps = blockSize >> WARP_SHIFT;
    __shared__ Bounds3f totals[numWarps];

    BlockReductionGridStride<blockSize>(
        totals, *sampleBounds, numSamples, [&](Bounds3f &bounds, uint32_t index) {
            float3 samplePos = make_float3(sampleX[index], sampleY[index], sampleZ[index]);
            bounds.Extend(samplePos);
        });
}

__global__ void CalculateSplitLocations(LevelInfo *info, KDTreeNode *nodes,
                                        SampleStatistics *statistics, Bounds3f *bounds = 0)
{
    uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadIndex >= info->count) return;

    int nodeIndex    = info->start + threadIndex;
    KDTreeNode *node = &nodes[nodeIndex];

    assert(nodeIndex != 0 || (nodeIndex == 0 && bounds));
    if (node->HasChild())
    {
        Bounds3f scaledBounds =
            nodeIndex == 0 ? *bounds : statistics[node->parentIndex].bounds;
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

__global__ void BeginLevel(LevelInfo *levelInfo, Queue *queue, Queue *queue2)
{
    // printf("level %u %u %u\n", levelInfo->start, levelInfo->count, levelInfo->childCount);
    // printf("queue: %u\n", queue->writeOffset);

    levelInfo->start += levelInfo->count;
    levelInfo->count      = levelInfo->childCount;
    levelInfo->childCount = 0;
    queue->readOffset     = 0;
    queue->writeOffset    = 0;

    queue2->readOffset  = 0;
    queue2->writeOffset = 0;
}

__global__ void Pass(uint32_t *sampleIndices, uint32_t *newSampleIndices, uint32_t level)
{
    if (blockIdx.x == 0)
    {
        printf("%u %u\n", sampleIndices[threadIdx.x], newSampleIndices[threadIdx.x]);
    }
}

__global__ void CalculateChildIndices(LevelInfo *info, KDTreeNode *nodes)
{

    for (uint32_t nodeIndex = info->start; nodeIndex < info->start + info->count; nodeIndex++)
    {
        KDTreeNode &node = nodes[nodeIndex];

        bool split = node.count > MAX_SAMPLES_PER_LEAF;

        {
            // stats[nodeIndex] = SampleStatistics();
            if (split)
            {
                assert(info->start == 1 || node.parentIndex != 0);
                node.SetChildIndex(info->start + info->count + info->childCount);
                // atomicAdd(&info->childCount, 2));
                info->childCount += 2;
            }
            else
            {
                node.childIndex_dim = ~0u;
            }
        }
    }
}

template <uint32_t blockSize>
__global__ void CreateWorkItems(Queue *reduceQueue, WorkItem *reduceWorkItems,
                                Queue *partitionQueue, WorkItem *partitionWorkItems,
                                KDTreeNode *nodes, LevelInfo *info, SampleStatistics *stats)
{
    if (blockIdx.x >= info->count) return;

    // TODO: this is bad and slow for now
    uint32_t nodeIndex = info->start + blockIdx.x;

    __shared__ KDTreeNode node;
    __shared__ uint32_t reduceStart;
    __shared__ uint32_t partitionStart;

    if (threadIdx.x == 0)
    {
        node = nodes[nodeIndex];
    }

    __syncthreads();

    uint32_t numWorkItems = (node.count + blockSize - 1) / blockSize;
    bool split            = node.count > MAX_SAMPLES_PER_LEAF;

    if (threadIdx.x == 0)
    {
        reduceStart      = atomicAdd(&reduceQueue->writeOffset, numWorkItems);
        stats[nodeIndex] = SampleStatistics();
        // KDTreeNode &node = nodes[nodeIndex];

        if (split)
        {
            partitionStart = atomicAdd(&partitionQueue->writeOffset, numWorkItems);
            // node.SetChildIndex(info->start + info->count + atomicAdd(&info->childCount, 2));
        }
        else
        {
            // node.childIndex_dim = ~0u;
        }
    }

    __syncthreads();

    for (uint32_t i = threadIdx.x; i < numWorkItems; i += blockSize)
    {
        WorkItem workItem;
        workItem.nodeIndex = nodeIndex;
        workItem.offset    = i * blockSize;
        workItem.count     = min(node.count - workItem.offset, blockSize);

        reduceWorkItems[reduceStart + i] = workItem;

        if (split)
        {
            partitionWorkItems[partitionStart + i] = workItem;
        }
    }
}

template <uint32_t blockSize>
__global__ void CalculateNodeStatistics(Queue *queue, WorkItem *workItems, KDTreeNode *nodes,
                                        SampleStatistics *statistics,
                                        SOAFloat3 samplePositions, uint32_t *sampleIndices,
                                        Bounds3f *bounds = 0)
{
    static constexpr uint32_t numWarps = blockSize >> WARP_SHIFT;

    __shared__ WorkItem workItem;
    __shared__ KDTreeNode node;
    __shared__ SampleStatistics totals[numWarps];

    for (uint32_t workItemIndex = blockIdx.x; workItemIndex < queue->writeOffset;
         workItemIndex += gridDim.x)
    {
        if (threadIdx.x == 0)
        {
            workItem = workItems[workItemIndex];
            node     = nodes[workItem.nodeIndex];
        }

        __syncthreads();

        assert((workItem.nodeIndex == 0 && bounds) || workItem.nodeIndex != 0);

        const uint32_t *blockSampleIndices = sampleIndices + node.offset + workItem.offset;
        Bounds3f scaledBounds              = statistics[node.parentIndex].bounds;

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

__global__ void BuildKDTree(Queue *queue, WorkItem *workItems, uint32_t level,
                            LevelInfo *levelInfo, KDTreeNode *nodes, uint32_t *sampleIndices,
                            uint32_t *newSampleIndices, SOAFloat3 samplePositions)
{
    __shared__ KDTreeNode node;
    __shared__ WorkItem workItem;

    __shared__ PartitionSharedState partitionState;

    for (uint32_t workItemIndex = blockIdx.x; workItemIndex < queue->writeOffset;
         workItemIndex += gridDim.x)
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

__global__ void GetChildNodeOffset(LevelInfo *levelInfo, KDTreeNode *nodes, uint32_t level)
{
    uint32_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIndex >= levelInfo->count) return;

    int parentIndex  = levelInfo->start + threadIndex;
    KDTreeNode *node = &nodes[levelInfo->start + threadIndex];

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

template <uint32_t blockSize>
__global__ void FindLeafNodes(LevelInfo *levelInfo, KDTreeNode *nodes, uint32_t *nodeIndices,
                              uint32_t *numLeafNodes, VMMMapState *states, VMM *vmms)
{
    // uint32_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    // if (threadIndex >= levelInfo->start) return;
    // KDTreeNode &node = nodes[threadIndex];

    for (uint32_t threadIndex = 0; threadIndex < levelInfo->start; threadIndex++)
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

__global__ void
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

__global__ void PrintStatistics(SampleStatistics *stats, KDTreeNode *nodes, LevelInfo *info,
                                uint32_t *test, VMM *vmm, VMMMapState *states,
                                SplitStatistics *splitStats)
{
    uint32_t t = 0;

    for (uint32_t i = 0; i < MAX_COMPONENTS; i++)
    {
        printf("%f ", splitStats[0].chiSquareTotals[i]);
    }

    printf("\n");

    // printf("num: %u\n", *test);
    // for (uint32_t i = 0; i < info->start + info->count; i++)
    // {
    //     // KDTreeNode &node = nodes[i];
    //     // printf("node: %u %u %f\n", i, node.GetSplitDim(), node.splitPos);
    //
    //     SampleStatistics &s = stats[i];
    //     float3 bmin         = s.bounds.GetBoundsMin();
    //     longlong3 mean      = s.mean;
    //     // printf("node: %u, %u, %lld %lld %lld\n", i, s.numSamples, mean.x, mean.y,
    //     // mean.z);
    //     printf("node: %u, %u, %u, %f %f %f\n", i, nodes[i].count, s.numSamples, bmin.x,
    //     bmin.y,
    //            bmin.z);
    // }
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
    //     printf("par: %u %u %f\n", node->parentIndex, nodes[node->parentIndex].GetSplitDim(),
    //            nodes[node->parentIndex].splitPos);
    //     printf("node: %u %u %f\n", i, node->GetSplitDim(), node->splitPos);
    //     printf("why? %u %u %u %u %u %u\n", boundsMinUint.x, boundsMinUint.y,
    //     boundsMinUint.z,
    //            boundsMaxUint.x, boundsMaxUint.y, boundsMaxUint.z);
    //
    //     printf("off count: %u %u,  bounds: %f %f %f %f %f %f\n mean: %lld %lld %lld "
    //            "var: %lld %lld %lld, %u\n\n",
    //            node->offset, node->count, boundsMin.x, boundsMin.y, boundsMin.z,
    //            boundsMax.x, boundsMax.y, boundsMax.z, mean.x, mean.y, mean.z, variance.x,
    //            variance.y, variance.z, sampleStatistics->numSamples);
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

    // int numBlocksPerSM;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &numBlocksPerSM, UpdateMixturePersistent<blockSize>, blockSize, 0);
    //
    // int maxGridSize = props.multiProcessorCount * numBlocksPerSM;
    // printf("blocks, grid: %i %i\n", numBlocksPerSM, maxGridSize);

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

    float *sampleDirX    = allocator.Alloc<float>(numSamples, 32u);
    float *sampleDirY    = allocator.Alloc<float>(numSamples, 4u);
    float *sampleDirZ    = allocator.Alloc<float>(numSamples, 4u);
    float *sampleWeights = allocator.Alloc<float>(numSamples, 4u);
    float *samplePdfs    = allocator.Alloc<float>(numSamples, 4u);

    SOAFloat3 samplePositions  = {samplePosX, samplePosY, samplePosZ};
    SOAFloat3 sampleDirections = {sampleDirX, sampleDirY, sampleDirZ};

    uint32_t *sampleIndices    = allocator.Alloc<uint32_t>(numSamples, 4u);
    uint32_t *newSampleIndices = allocator.Alloc<uint32_t>(numSamples, 4u);

    LevelInfo *levelInfo = allocator.Alloc<LevelInfo>(1, 4u);

    Queue *reduceQueue           = allocator.Alloc<Queue>(1, 4u);
    Queue *partitionQueue        = allocator.Alloc<Queue>(1, 4u);
    Queue *vmmQueue              = allocator.Alloc<Queue>(1, 4u);
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
        // Pass<<<1, blockSize>>>(s0, s1, level);
    }

    // TODO: reuse temporary buffers from kd tree build
    cudaEventRecord(stop);

    // VMM update
    SOASampleData soaSampleData;
    soaSampleData.pos     = samplePositions;
    soaSampleData.dir     = sampleDirections;
    soaSampleData.weights = sampleWeights;
    soaSampleData.pdfs    = samplePdfs;

    cudaEvent_t vmmStart, vmmStop;
    cudaEventCreate(&vmmStart);
    cudaEventCreate(&vmmStop);
    cudaEventRecord(vmmStart);

    FindLeafNodes<blockSize>
        <<<1, 1>>>(levelInfo, nodes, leafNodeIndices, numLeafNodes, vmmMapStates, vmms);
    WriteSamplesToSOA<<<nodeBlocks, WARP_SIZE>>>(numLeafNodes, leafNodeIndices, sampleIndices,
                                                 newSampleIndices, nodes, samples,
                                                 soaSampleData);

    CreateVMMWorkItems<blockSize><<<(maxNumNodes + blockSize - 1) / blockSize, blockSize>>>(
        leafNodeIndices, numLeafNodes, nodes, vmmWorkItems, vmmQueue);

    // UpdateMixturePersistent<blockSize><<<maxGridSize * 2, blockSize>>>(
    //     vmmQueue, vmmWorkItems, vmms, vmmStatistics, soaSampleData, leafNodeIndices,
    //     numLeafNodes, nodes, vmmMapStates, numSamples);
    UpdateMixture<<<(maxNumNodes + blockSize - 1) / blockSize, blockSize>>>(
        vmms, vmmStatistics, soaSampleData, leafNodeIndices, numLeafNodes, nodes);
    UpdateSplitStatistics<<<numBlocks, blockSize>>>(vmms, vmmStatistics, splitStatistics,
                                                    soaSampleData, leafNodeIndices,
                                                    numLeafNodes, nodes);
    SplitComponents<<<(maxNumNodes + WARP_SIZE - 1) / WARP_SIZE, WARP_SIZE>>>(
        vmms, vmmStatistics, splitStatistics, soaSampleData, leafNodeIndices, numLeafNodes,
        nodes);

    nvtxRangePop();
    cudaEventRecord(vmmStop);
    PrintStatistics<<<1, 1>>>(sampleStatistics, nodes, levelInfo, numLeafNodes, vmms,
                              vmmMapStates, splitStatistics);

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

    cudaEventElapsedTime(&milliseconds, vmmStart, vmmStop);
    printf("VMM GPU Time: %f ms\n", milliseconds);
}

} // namespace rt

int main()
{
    rt::test();
    return 0;
}
