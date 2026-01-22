#pragma once

#include <assert.h>
#include "platform.h"
#include "helper_math.h"
#include "../util/atomic.h"

#define PI             3.14159265358979323846f
#define MAX_COMPONENTS 32
#define WARP_SIZE      32
#define WARP_SHIFT     5u

namespace rt
{

// TODO IMPORTANT: for other gpu platforms, change #ifdef __CUDACC__

#ifdef __CUDACC__
template <typename T>
RT_GPU_DEVICE T WarpReduceSum(T val)
{
#pragma unroll
    for (int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <>
RT_GPU_DEVICE float3 WarpReduceSum(float3 val)
{
    val.x = WarpReduceSum(val.x);
    val.y = WarpReduceSum(val.y);
    val.z = WarpReduceSum(val.z);
    return val;
}

template <>
RT_GPU_DEVICE int3 WarpReduceSum(int3 val)
{
    val.x = WarpReduceSum(val.x);
    val.y = WarpReduceSum(val.y);
    val.z = WarpReduceSum(val.z);
    return val;
}

template <>
RT_GPU_DEVICE longlong3 WarpReduceSum(longlong3 val)
{
    val.x = WarpReduceSum(val.x);
    val.y = WarpReduceSum(val.y);
    val.z = WarpReduceSum(val.z);
    return val;
}

template <>
RT_GPU_DEVICE uint3 WarpReduceSum(uint3 val)
{
    val.x = WarpReduceSum(val.x);
    val.y = WarpReduceSum(val.y);
    val.z = WarpReduceSum(val.z);
    return val;
}

template <typename T>
RT_GPU_DEVICE T WarpReduceMin(T val)
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
RT_GPU_DEVICE float3 WarpReduceMin(float3 val)
{
    val.x = WarpReduceMin(val.x);
    val.y = WarpReduceMin(val.y);
    val.z = WarpReduceMin(val.z);
    return val;
}

template <>
RT_GPU_DEVICE uint3 WarpReduceMin(uint3 val)
{
    val.x = WarpReduceMin(val.x);
    val.y = WarpReduceMin(val.y);
    val.z = WarpReduceMin(val.z);
    return val;
}

template <typename T>
RT_GPU_DEVICE T WarpReduceMax(T val)
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
RT_GPU_DEVICE float3 WarpReduceMax(float3 val)
{
    val.x = WarpReduceMax(val.x);
    val.y = WarpReduceMax(val.y);
    val.z = WarpReduceMax(val.z);
    return val;
}

template <>
RT_GPU_DEVICE uint3 WarpReduceMax(uint3 val)
{
    val.x = WarpReduceMax(val.x);
    val.y = WarpReduceMax(val.y);
    val.z = WarpReduceMax(val.z);
    return val;
}

template <typename T>
RT_GPU_DEVICE T WarpReadLaneAt(T val, uint32_t thread)
{
    val = __shfl_sync(0xffffffff, val, thread);
    return val;
}

template <>
RT_GPU_DEVICE float3 WarpReadLaneAt(float3 val, uint32_t thread)
{
    val.x = __shfl_sync(0xffffffff, val.x, thread);
    val.y = __shfl_sync(0xffffffff, val.y, thread);
    val.z = __shfl_sync(0xffffffff, val.z, thread);
    return val;
}

template <typename T>
RT_GPU_DEVICE T WarpReadLaneFirst(T val)
{
    val = __shfl_sync(0xffffffff, val, 0);
    return val;
}

template <typename T>
RT_GPU_DEVICE T WarpReduceSumBroadcast(T val)
{
    val = WarpReduceSum(val);
    return WarpReadLaneFirst(val);
}

inline RT_GPU_DEVICE uint32_t FloatToUint(float f) { return __float_as_uint(f); }
inline RT_GPU_DEVICE float UintToFloat(uint u) { return __uint_as_float(u); }

inline RT_DEVICE uint32_t FloatToOrderedUint(float f)
{
    uint32_t u    = FloatToUint(f);
    uint32_t mask = (u & 0x80000000) ? ~0u : 0x80000000;
    return u ^ mask;
}

inline RT_DEVICE float OrderedUintToFloat(uint32_t u)
{
    uint32_t mask = (u & 0x80000000) ? 0x80000000 : ~0u;
    return UintToFloat(u ^ mask);
}

inline RT_DEVICE uint3 FloatToOrderedUint(float3 f)
{
    uint3 u;
    u.x = FloatToOrderedUint(f.x);
    u.y = FloatToOrderedUint(f.y);
    u.z = FloatToOrderedUint(f.z);
    return u;
}

inline RT_DEVICE float3 OrderedUintToFloat(uint3 u)
{
    float3 f;
    f.x = OrderedUintToFloat(u.x);
    f.y = OrderedUintToFloat(u.y);
    f.z = OrderedUintToFloat(u.z);
    return f;
}
#endif

template <typename T>
struct GPUBounds
{
    T boundsMin;
    T boundsMax;

#ifdef __CUDACC__
    RT_DEVICE void Extend(T value)
    {
        boundsMin = min(boundsMin, value);
        boundsMax = max(boundsMax, value);
    }

    RT_DEVICE void WarpReduce()
    {
        boundsMin = WarpReduceMin(boundsMin);
        boundsMax = WarpReduceMax(boundsMax);
    }
#endif
};

struct SOAFloat3Ref
{
    float *x, *y, *z;
    const uint32_t index;

#ifdef __CUDACC__
    RT_DEVICE SOAFloat3Ref &operator=(const float3 &val)
    {
        x[index] = val.x;
        y[index] = val.y;
        z[index] = val.z;
        return *this;
    }

    RT_DEVICE operator float3() const { return make_float3(x[index], y[index], z[index]); }

    RT_DEVICE SOAFloat3Ref &operator=(const SOAFloat3Ref &other)
    {
        float3 val = (float3)other;
        return (*this = val);
    }
#endif
};

struct SOAFloat3
{
    float *x;
    float *y;
    float *z;

#ifdef __CUDACC__
    RT_DEVICE SOAFloat3Ref operator[](uint32_t index) { return {x, y, z, index}; }

    RT_DEVICE float3 operator[](uint32_t index) const
    {
        return make_float3(x[index], y[index], z[index]);
    }

    RT_DEVICE SOAFloat3 operator+(uint32_t offset)
    {
        SOAFloat3 other;
        other.x = x + offset;
        other.y = y + offset;
        other.z = z + offset;
        return other;
    }
#endif
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

#ifdef __CUDACC__
    RT_DEVICE SOASampleDataRef &operator=(const SampleData &val)
    {
        pos            = val.pos;
        dir            = val.dir;
        weights[index] = val.weight;
        pdfs[index]    = val.pdf;
        return *this;
    }
#endif
};

struct SOASampleData
{
    SOAFloat3 pos;
    SOAFloat3 dir;

    float *weights;
    float *pdfs;
    float *distances;

#ifdef __CUDACC__
    RT_DEVICE SOASampleDataRef operator[](uint32_t index)
    {
        return {pos[index], dir[index], weights, pdfs, index};
    }

    RT_DEVICE SampleData operator[](uint32_t index) const
    {
        return {pos[index], dir[index], weights[index], pdfs[index]};
    }
#endif
};

template <>
struct GPUBounds<float3>
{
private:
    uint3 boundsMin;
    uint3 boundsMax;

#ifdef __CUDACC__
public:
    RT_DEVICE GPUBounds()
    {
        const uint pos_inf = 0xff800000;
        const uint neg_inf = ~0xff800000;

        boundsMin = make_uint3(pos_inf);
        boundsMax = make_uint3(neg_inf);
    }

    RT_DEVICE void Clear()
    {
        const uint pos_inf = 0xff800000;
        const uint neg_inf = ~0xff800000;

        boundsMin = make_uint3(pos_inf);
        boundsMax = make_uint3(neg_inf);
    }

    RT_DEVICE void Extend(float3 value)
    {
        uint3 val = FloatToOrderedUint(value);
        boundsMin = min(boundsMin, val);
        boundsMax = max(boundsMax, val);
    }

    RT_DEVICE void WarpReduce()
    {
        boundsMin = WarpReduceMin(boundsMin);
        boundsMax = WarpReduceMax(boundsMax);
    }

    RT_DEVICE void AtomicMerge(const GPUBounds &other)
    {
        atomicMin(&boundsMin, other.boundsMin);
        atomicMax(&boundsMax, other.boundsMax);
    }

    RT_DEVICE void SetBoundsMin(float3 f) { boundsMin = FloatToOrderedUint(f); }
    RT_DEVICE void SetBoundsMax(float3 f) { boundsMax = FloatToOrderedUint(f); }

    RT_DEVICE float3 GetBoundsMin() const { return OrderedUintToFloat(boundsMin); }
    RT_DEVICE float3 GetBoundsMax() const { return OrderedUintToFloat(boundsMax); }

    RT_DEVICE uint3 GetBoundsMinUint() const { return boundsMin; }
    RT_DEVICE uint3 GetBoundsMaxUint() const { return boundsMax; }

    RT_DEVICE float3 GetCenter() const
    {
        float3 bMin = GetBoundsMin();
        float3 bMax = GetBoundsMax();

        return (bMin + bMax) * 0.5f;
    }

    RT_DEVICE float3 GetHalfExtent() const
    {
        float3 bMin = GetBoundsMin();
        float3 bMax = GetBoundsMax();

        return (bMax - bMin) * 0.5f;
    }
#endif
};

typedef GPUBounds<float3> Bounds3f;

struct KDTreeBuildState
{
    // persists across updates
    uint32_t totalNumNodes;

    // temporary values
    uint32_t numNodes;
    uint32_t nextLevelNumNodes;
    uint32_t numReduceWorkItems;
    uint32_t numPartitionWorkItems;
};

struct KDTreeNode
{
    float splitPos;

    // NOTE: This variable can be in one of three states.
    // 1. Bits all set: child node
    // 2. Child index bits set, dim = 3: internal node, split location not determined yet
    // 3. Child index bits set, 0 < dim < 3: internal node
    uint32_t childIndex_dim;

    // TODO: remove?
    int parentIndex;

    uint32_t offset;
    uint32_t count;
    uint32_t vmmIndex;

    RT_DEVICE uint32_t GetChildIndex() const { return (childIndex_dim << 2) >> 2; }
    RT_DEVICE void SetChildIndex(uint32_t childIndex)
    {
        assert(childIndex < (1u << 30u));
        uint32_t dim   = GetSplitDim();
        childIndex_dim = childIndex | (dim << 30);
    }
    RT_DEVICE uint32_t GetSplitDim() const { return childIndex_dim >> 30; }
    RT_DEVICE void SetSplitDim(uint32_t dim)
    {
        assert(dim < 3);
        uint32_t childIndex = GetChildIndex();
        childIndex_dim      = childIndex | (dim << 30);
    }
    RT_DEVICE bool HasChild() const { return childIndex_dim != ~0u; }
    RT_DEVICE bool IsChild() const { return (childIndex_dim >> 30u) == 3u; }
};

#define INTEGER_BINS                     float(1 << 18)
#define INTEGER_SAMPLE_STATS_BOUND_SCALE (1.0f + 2.f / INTEGER_BINS)
#define MAX_TREE_DEPTH                   24
#define MAX_SAMPLES_PER_LEAF             32000
#define MIN_SPLIT_SAMPLES                (MAX_SAMPLES_PER_LEAF / 8u)
#define MIN_MERGE_SAMPLES                (MAX_SAMPLES_PER_LEAF / 4u)

struct SampleStatistics
{
    Bounds3f bounds;
    longlong3 mean;
    longlong3 variance;

    uint32_t numSamples;

#ifdef __CUDACC__
    RT_DEVICE SampleStatistics()
        : numSamples(0), bounds(), mean(make_longlong3(0, 0, 0)),
          variance(make_longlong3(0, 0, 0))
    {
    }

    RT_DEVICE void Clear()
    {
        numSamples = 0;
        bounds.Clear();
        mean     = make_longlong3(0, 0, 0);
        variance = make_longlong3(0, 0, 0);
    }

    RT_DEVICE void AddSample(float3 pos, float3 center, float3 invHalfExtent)
    {
        numSamples++;
        float3 tmpSample = (pos - center) * invHalfExtent;

        float3 tmpVariance = ((tmpSample * tmpSample) * INTEGER_BINS);

        mean += make_longlong3(tmpSample * INTEGER_BINS);
        variance += make_longlong3(tmpVariance);

        bounds.Extend(pos);
    }

    RT_DEVICE void WarpReduce()
    {
        bounds.WarpReduce();
        mean       = WarpReduceSum(mean);
        variance   = WarpReduceSum(variance);
        numSamples = WarpReduceSum(numSamples);
    }

    RT_DEVICE inline void AtomicMerge(const SampleStatistics &other)
    {
        bounds.AtomicMerge(other.bounds);
        atomicAdd(&mean, other.mean);
        atomicAdd(&variance, other.variance);
        atomicAdd(&numSamples, other.numSamples);
    }

    RT_DEVICE void ConvertToFloat(float3 center, float3 halfExtent, float3 &outMean,
                                  float3 &outVariance) const
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
#endif
};

struct VMM
{
    float kappas[MAX_COMPONENTS];
    float directionX[MAX_COMPONENTS];
    float directionY[MAX_COMPONENTS];
    float directionZ[MAX_COMPONENTS];
    float weights[MAX_COMPONENTS];
    float distances[MAX_COMPONENTS];

    uint32_t numComponents;

#ifdef __CUDACC__

    inline RT_GPU_DEVICE void Initialize()
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
            distances[n] = 0.f;
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

    inline RT_GPU_DEVICE float3 ReadDirection(uint32_t componentIndex) const
    {
        float3 dir;
        dir.x = directionX[componentIndex];
        dir.y = directionY[componentIndex];
        dir.z = directionZ[componentIndex];
        return dir;
    }

    inline RT_GPU_DEVICE void WriteDirection(uint32_t componentIndex, float3 dir)
    {
        directionX[componentIndex] = dir.x;
        directionY[componentIndex] = dir.y;
        directionZ[componentIndex] = dir.z;
    }

    RT_GPU_DEVICE void UpdateComponent_(uint32_t componentIndex, float kappa, float3 dir,
                                        float weight)
    {
        kappas[componentIndex] = kappa;
        WriteDirection(componentIndex, dir);
        weights[componentIndex] = weight;
    }

    RT_GPU_DEVICE void UpdateComponent(uint32_t componentIndex, float kappa, float3 dir,
                                       float weight, float distance)
    {
        kappas[componentIndex] = kappa;
        WriteDirection(componentIndex, dir);
        weights[componentIndex]   = weight;
        distances[componentIndex] = distance;
    }
#endif
};

struct VMMStatistics
{
    float sumSampleWeights;
    float sumWeights[MAX_COMPONENTS];
    float sumOfDistanceWeights[MAX_COMPONENTS];
    float3 sumWeightedDirections[MAX_COMPONENTS];

    // TODO: decay?
    uint32_t numSamples;

    uint32_t numSamplesAfterLastSplit;
    uint32_t numSamplesAfterLastMerge;

#ifdef __CUDACC__
    RT_GPU_DEVICE void SetComponent(uint32_t componentIndex, float sumWeight,
                                    float3 sumWeightedDirection)
    {
        assert(componentIndex < MAX_COMPONENTS);
        sumWeights[componentIndex]            = sumWeight;
        sumWeightedDirections[componentIndex] = sumWeightedDirection;
    }
#endif
};

struct SplitStatistics
{
    float3 covarianceTotals[MAX_COMPONENTS];
    float chiSquareTotals[MAX_COMPONENTS];
    float sumWeights[MAX_COMPONENTS];
    uint32_t numSamples[MAX_COMPONENTS];

#ifdef __CUDACC__
    RT_GPU_DEVICE void SetComponent(uint32_t componentIndex, float3 covarianceTotal,
                                    float chiSquareTotal, float sumWeight, uint32_t num)
    {
        covarianceTotals[componentIndex] = covarianceTotal;
        chiSquareTotals[componentIndex]  = chiSquareTotal;
        sumWeights[componentIndex]       = sumWeight;
        numSamples[componentIndex]       = num;
    }
#endif
};

struct WorkItem
{
    uint32_t nodeIndex;
    uint32_t offset;
    uint32_t count;
};

struct VMMUpdateWorkItem
{
    uint32_t vmmIndex_isNew;
    uint32_t sharedSplitMask;
    uint32_t offset;
    uint32_t count;

    RT_DEVICE bool IsNew() const { return vmmIndex_isNew >> 31u; }
    RT_DEVICE uint32_t GetVMMIndex() const { return (vmmIndex_isNew << 1u) >> 1u; }

    RT_DEVICE void SetIsNew(bool isNew)
    {
        uint32_t vmmIndex = GetVMMIndex();
        vmmIndex_isNew    = vmmIndex | (isNew << 31u);
    }

    RT_DEVICE void SetVMMIndex(uint32_t vmmIndex)
    {
        assert(vmmIndex < (1u << 31u));
        bool isNew     = IsNew();
        vmmIndex_isNew = vmmIndex | (isNew << 31u);
    }
};

struct VMMUpdateState
{
    uint32_t numVMMs;
    uint32_t totalNumVMMs;
    uint32_t maxNumVMMs;
};

} // namespace rt
