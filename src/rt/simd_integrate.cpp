
#include "bxdf.h"
namespace rt
{

struct SortKey
{
    u32 value;
    u32 index;
};

struct RayState
{
    Ray2 ray;
    SampledSpectrum L;
    SampledSpectrum beta;
    SampledSpectrum etaScale;
    PathFlags pathFlags;
    u32 depth;
    Sampler sampler;
};

typedef u32 RayStateHandle;

static const u32 simdQueueLength = 512;
struct RayQueue
{
    // lightpdf, le
    RayStateHandle queue[simdQueueLength];
    u32 count;
    void Flush(u32 add)
    {
        if (count + add <= simdQueueLength) return;

        u32 alignedCount = count & (IntN - 1);
        u32 start        = count - alignedCount;
        for (u32 i = start; i < start + alignedCount; i++)
        {
        }
    }
    void Push() {}
};

// going to do per material instead of per material type, since it's easier
struct DielectricBxDFQueue
{
    f32 eta;
    TrowbridgeReitzDistribution mfDistrib;
};

// 1. intersect ray, add to shading queue
// 2. shading queue can either be per material instance or per material type
// - either: write directly in soa format, or write in aos and transpose data members
//      - i'm going to transpose. i'm also going to support jank mega kernels by sampling
//      lights early, calculating type ranges, and simd along those ranges
//      - then why can't i just use ranges for the materials as well?

// ways of doing this:
// - after intersection write to queue, then:
// 1. sort using a "mega key", compute on ranges
// 2. wavefront queues per type, pass keys between queues
//      a. or per instance

// - queues local or global?
// - do I want to construct megakernels? from what I'm envisioning, either we have a lot of
// queues (representing one combination of kernels), or a lot of masking/wasted execution

struct ShadingQueue
{
    using BxDF = typename Material::BxDF;
    SurfaceInteraction queue[simdQueueLength];
    u32 count = 0;
    ShadingQueuePtex() {}
    void Flush() // SortKey *keys, SurfaceInteraction *values, u32 num)
    {
        TempArena temp = ScratchStart(0, 0);

        u32 alignedCount = count & (IntN - 1);
        count -= alignedCount;
        u32 start = count;

        // IMPORTANT:
        // for ptex, radix sort using:
        // light type, mesh id, face id
        // for no ptex, radix sort using:
        // light type, uv
        // Create radix sort keys, get the light type

        SortKey *keys0 = PushArrayNoZero(temp.arena, SortKey, alignedCount);
        SortKey *keys1 = PushArrayNoZero(temp.arena, SortKey, alignedCount);
        SortKey keys0[queueFlushSize];
        SortKey keys1[queueFlushSize];
        f32 *pmfs            = PushArrayNoZero(temp.arena, f32, alignedCount);
        LightHandle *handles = PushArrayNoZero(temp.arena, LightHandle, alignedCount);
        for (u32 i = 0; i < alignedCount; i++)
        {
            SurfaceInteraction &intr = queue[start + i];
            // TODO: get this somehow
            Sampler sampler;
            LightHandle handle = UniformLightSample(sampler.Get1D(), &pmfs[i]);
            handles[i]         = handle;

            keys0[i].value = GenerateKey(intr, handle.GetType());
            keys0[i].index = start + i;
        }

        // Radix sort
        for (u32 iter = 3; iter >= 0; iter--)
        {
            u32 shift        = iter * 8;
            u32 buckets[255] = {};
            // Calculate # in each radix
            for (u32 i = 0; i < queueFlushSize; i++)
            {
                SortKey *key = &keys0[i];
                buckets[(key->value >> shift) & 0xff]++;
            }
            // Prefix sum
            u32 total = 0;
            for (u32 i = 0; i < 255; i++)
            {
                u32 count  = buckets[i];
                buckets[i] = total;
                total += count;
            }

            // TODO: calculate ranges here
            // Place in correct position
            for (u32 i = 0; i < queueFlushSize; i++)
            {
                SortKey &sortKey      = &keys0[i];
                u32 key               = (sortKey.value >> shift) & 0xff;
                keys1[buckets[key]++] = sortKey;
            }
            Swap(keys0, keys1);
        }

        // Convert to AOSOA
        u32 limit = queueFlushSize % (IntN);
        // SurfaceInteractionsN aosoaIntrs[queueFlushSize / IntN];
        for (u32 i = 0; i < alignedCount;)
        {
            const u32 prefetchDistance = IntN * 2;
            alignas(32) SurfaceInteraction intrs[IntN];
            if (i + prefetchDistance < alignedCount)
            {
                for (u32 j = 0; j < IntN; j++)
                {
                    _mm_prefetch((char *)&queue[keys[i + prefetchDistance + j].index],
                                 _MM_HINT_T0);
                    intrs[j] = queue[keys0[i + j].index];
                }
            }
            SurfaceInteractionsN aosoaIntrs; //&out = aosoaIntrs[aosoaIndex];
                                             // Transpose p, n, uv
            Transpose(intrs, aosoaIntrs);
            // Transpose8x8(Lane8F32::Load(&intrs[0]), Lane8F32::Load(&intrs[1]),
            // Lane8F32::Load(&intrs[2]), Lane8F32::Load(&intrs[3]),
            //              Lane8F32::Load(&intrs[4]), Lane8F32::Load(&intrs[5]),
            //              Lane8F32::Load(&intrs[6]), Lane8F32::Load(&intrs[7]), out.p.x,
            //              out.p.y, out.p.z, out.n.x, out.n.y, out.n.z, out.uv.x, out.uv.y);
            // Transpose the rest

            MaskF32 continuationMask = LaneNF32::Mask<true>();
            BSDFBase<BxDF> bsdf      = Material::Evaluate(aosoaIntrs);

            template <i32 width>
            struct LightSamples
            {
                SampledSpectrum Le;
                Vec3IF32 samplePoint;
                LaneNF32 pdf;
            };

            Sampler samplers[];

            //////////////////////////////
            // Next event estimation
            //
            if (Any(!IsSpecular(bsdf.Flags())))
            {
                RayStateHandle rayStateHandles[IntN];

                // Sample lights
                alignas(32) LightHandle itrHandles[IntN];
                alignas(32) f32 pdfs[IntN];
                for (u32 j = 0; j < IntN; j++)
                {
                    u32 index     = keys[i + j].index - start;
                    itrHandles[j] = handles[index];
                    pdfs[j]       = pmfs[index];
                }
                LaneIU32 handles  = LaneIU32::Load(itrHandles);
                u32 type          = itrHandles[0].GetType();
                LaneIU32 laneType = LaneIU32(itrHandles[0].GetType());
                MaskF32 mask      = (handles & laneType) == laneType;
                u32 maskBits      = Movemask(mask);
                u32 add           = PopCount(maskBits);
                LaneNF32 lightPdf = LaneNF32::Load(pdfs);

                // TODO: get samplers
                LightSamples sample = SampleLi(type, handles, add, aosoaIntrs, lambda?, samplers);//scene, lightHandle, intr, lambda, u);
                mask &= sample.pdf == 0.f;
                lightPdf *= sample.pdf;
                // f32 scatterPdf;
                // SampledSpectrum f_hat;
                Vec3IF32 wi = Normalize(sample.samplePoint - aosoaIntrs.p);

                LaneNF32 scatterPdf;
                SampledSpectrum f =
                    BxDF::EvaluateSample(-ray.d, wi, scatterPdf, TransportMode::Radiance) *
                    AbsDot(aosoaIntrs.shading.n, wi);
                // TODO: need to == with 0.f for every wavelength, and then combine together
                mask &= f.GetMask();

                maskBits = Movemask(mask);
                // Shoot occlusion rays
                for (u32 j = 0; j < IntN; j++)
                {
                    if (maskBits & (1 << j))
                    {
                        // Shoot occlusion ray
                        // maskBits &= Occluded();
                    }
                }

                // Power heuristic
                LaneNF32 w_l = lightPdf / (Sqr(lightPdf) + Sqr(scatterPdf));
                // TODO: to prevent atomics, need to basically have thread permanently take a
                // tile
                L += Select(LaneNF32::Mask(maskbits), f * beta * w_l * sample.Le, 0.f);
                i += add;
            }
            else
            {
                i += IntN;
            }

            // TODO: things I need to simd:
            // - sampler
            // - ray
            // - path throughput weight
            // - path flags
            // - path eta scale for russian roulette

            // TODO: should I copy data (e.g. path throughput weights) so that stages can
            // happen in whatever order? otherwise, i need to ensure that next event
            // estimation, etc. happens before the bsdf sample is generated, otherwise the path
            // throughput weight needed for nee is lost
            Sampler *samplers[IntN];
            Ray2 *rays[IntN];
            LaneF32<IntN> u;
            Vec2lf<IntN> uv;

            BSDFSample<IntN> sample =
                bsdf.GenerateSample(-ray.d, u, uv, TransportMode ::Radiance, BSDFFlags::RT);
            MaskF32 mask;
            mask = Select(sample.pdf == 0.f, 1, 0);
            beta *= sample.f * AbsDot(intr.shading.n, sample.wi) / sample.pdf;

            // store back path throughput
            // store back radiance
            // store back depth
            // store back eta scale

            // TODO: path flags, set specular bounce to true
            // pathFlags &= bsdf.IsSpecular();

            // Spawn a new ray, push to the ray queue
            // Russian roulette
        }

        ScratchEnd(temp);
    }
}
};
} // namespace rt
