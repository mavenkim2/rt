#ifndef SIMD_INTEGRATE_H
#define SIMD_INTEGRATE_H

#include "base.h"
#include "containers.h"
#include "integrate.h"
#include "memory.h"
#include "parallel.h"

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
    SampledWavelengths lambda;
    // PathFlags pathFlags;
    f32 etaScale;
    f32 bsdfPdf;
    u32 depth;
    bool specularBounce;
    struct ZSobolSampler sampler;
    SurfaceInteraction si;
};

typedef ChunkedLinkedList<RayState, 4096> RayStateList;
typedef RayStateList::ChunkNode RayStateNode;

struct RayStateHandle
{
    RayState *state = 0;

    bool IsValid() const { return state != 0; }
    const RayState *GetRayState() const { return state; }
    RayState *GetRayState() { return state; }
};

typedef ChunkedLinkedList<RayStateHandle, 4096> RayStateFreeList;

struct ShadingHandle
{
    u32 sortKey;
    RayStateHandle rayStateHandle;
};

static const u32 QUEUE_LENGTH = 1024;

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

#define QUEUE_HANDLER(name)                                                                   \
    void name(TempArena inScratch, struct ShadingThreadState *state, void *values, u32 count)

template <typename T>
struct ThreadLocalQueue
{
    typedef QUEUE_HANDLER((*Handler));
    Handler handler;

    T values[QUEUE_LENGTH];
    u32 count;

    // bool Push(ShadingThreadState *state, const T &item);

    void Push(TempArena temp, ShadingThreadState *state, T *entries, int numEntries);
    bool Flush(ShadingThreadState *state);
};

// template <typename T>

template <typename T>
struct SharedShadeQueue
{
    T values[QUEUE_LENGTH];
    typedef void (*Handler)(TempArena temp, struct ShadingThreadState *state, T *values,
                            u32 count, Material *material);

    Handler handler;

    u32 count;

    Material *material;
    Mutex mutex;

    void Push(TempArena temp, ShadingThreadState *state, T *entries, int numEntries);
    bool Flush(ShadingThreadState *state);
};

// typedef ThreadLocalQueue<ShadingHandle> ShadingQueue;
typedef SharedShadeQueue<ShadingHandle> ShadingQueue;
typedef ThreadLocalQueue<RayStateHandle> RayQueue;

struct alignas(CACHE_LINE_SIZE) ShadingThreadState
{
    // Queues:
    // ShadingQueue shadingQueues[(u32)MaterialTypes::Max];
    // TODO: distinction b/t primary and secondary rays?
    RayQueue rayQueue;

    RayStateList rayStates;
    RayStateFreeList rayFreeList;

    PtexTexture *texture;

    void *buffer;

    Arena *scratchArenas[2];
    u64 pos[2];
    u32 last;
    u32 current;
};

struct ShadingGlobals
{
    ShadingQueue *shadingQueues;
    u32 numShadingQueues;
    Vec3f *rgbValues;
    Camera *camera;
    u32 width;
    u32 height;
    u32 maxDepth;
};

extern ShadingThreadState *shadingThreadState_;
extern ShadingGlobals *shadingGlobals_;

template <typename MaterialType>
void ShadingQueueHandler(TempArena inScratch, struct ShadingThreadState *state,
                         ShadingHandle *values, u32 count, Material *m);

inline ShadingThreadState *GetShadingThreadState()
{
    return &shadingThreadState_[GetThreadIndex()];
}
inline ShadingThreadState *GetShadingThreadState(u32 index)
{
    return &shadingThreadState_[index];
}
inline ShadingGlobals *GetShadingGlobals() { return shadingGlobals_; }
inline ShadingQueue *GetShadingQueue(u32 index)
{
    return &shadingGlobals_->shadingQueues[index];
}
QUEUE_HANDLER(RayIntersectionHandler);
void RenderSIMD(Arena **arenas, RenderParams2 &params);

} // namespace rt
#endif
