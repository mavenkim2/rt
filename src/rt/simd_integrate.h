#ifndef SIMD_INTEGRATE_H
#define SIMD_INTEGRATE_H
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
    Vec2u pixel;
    SampledSpectrum L;
    SampledSpectrum beta;
    SampledSpectrum etaScale;
    SampledWavelengths lambda;
    // PathFlags pathFlags;
    f32 bsdfPdf;
    bool specularBounce;
    u32 depth;
    Sampler sampler;
    SurfaceInteraction si;
};

typedef ChunkedLinkedList<RayState, 8192> RayStateList;
typedef RayStateList::ChunkNode RayStateNode;

struct RayStateHandle
{
    RayStateNode *node;
    u32 index;

    const RayState *GetRayState() const { return &node->values[index]; }
    RayState *GetRayState() { return &node->values[index]; }
};

typedef ChunkedLinkedList<RayStateHandle, 8192> RayStateFreeList;

struct ShadingHandle
{
    u64 shadingKey;
    RayStateHandle rayStateHandle;
};

static const u32 QUEUE_LENGTH    = 1024;
static const u32 FLUSH_THRESHOLD = 512;

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

template <typename T>
struct ThreadLocalQueue
{
    typedef void (*Handler)(void *values, u32 count);
    Handler handler;

    T values[QUEUE_LENGTH];
    u32 count;

    void Flush()
    {
        if (count > FLUSH_THRESHOLD)
        {
            count -= FLUSH_THRESHOLD;
            handler(values + count, count);
        }
    }
    void Finalize() { handler(values, count); }
    void Push(const T &item)
    {
        Assert(count < QUEUE_LENGTH);
        values[count++] = item;
    }
};

typedef ThreadLocalQueue<ShadingHandle> ShadingQueue;
typedef ThreadLocalQueue<RayStateHandle> RayQueue;

struct ShadingThreadState
{
    // Queues:
    ShadingQueue shadingQueues[(u32)MaterialTypes::Max];
    // TODO: distinction b/t primary and secondary rays?
    RayQueue rayQueue;

    RayStateList rayStates;
    RayStateFreeList rayFreeList;
};

struct ShadingGlobals
{
    u32 maxDepth;
};

thread_local ShadingThreadState *shadingThreadState_;
static ShadingGlobals *shadingGlobals_;

ShadingThreadState *GetShadingThreadState() { return shadingThreadState_; }
ShadingGlobals *GetShadingGlobals() { return shadingGlobals_; }
} // namespace rt
#endif
