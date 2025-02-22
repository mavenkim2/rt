#include "simd_integrate.h"
namespace rt
{

template <typename T>
void ThreadLocalQueue<T>::Push(TempArena scratch, ShadingThreadState *state, T *entries,
                               int numEntries)
{
    TempArena temp = ScratchStart(&scratch.arena, 1);
    u32 total      = numEntries + count;
    if (total >= ArrayLength(values))
    {
        T *newEntries = (T *)PushArrayNoZero(temp.arena, u8, sizeof(T) * total);
        MemoryCopy(newEntries, values, sizeof(T) * count);
        MemoryCopy(newEntries + count, entries, sizeof(T) * numEntries);

        ScratchEnd(scratch);
        count = 0;
        handler(temp, state, newEntries, total);
    }
    else
    {
        MemoryCopy(values + count, entries, sizeof(T) * numEntries);
        ScratchEnd(scratch);
        count += numEntries;
    }
}

template <typename T>
bool ThreadLocalQueue<T>::Flush(ShadingThreadState *state)
{
    if (count == 0) return true;

    TempArena temp = ScratchStart(0, 0);
    u32 total      = count;
    T *newEntries  = (T *)PushArrayNoZero(temp.arena, u8, sizeof(T) * total);
    MemoryCopy(newEntries, values, sizeof(T) * total);
    count = 0;
    handler(temp, state, values, total);
    return false;
}

template <typename T>
void SharedShadeQueue<T>::Push(TempArena scratch, ShadingThreadState *state, T *entries,
                               int numEntries)
{
    TempArena temp = ScratchStart(&scratch.arena, 1);
    BeginMutex(&mutex);
    u32 total  = numEntries + count;
    u32 offset = count;
    if (total >= ArrayLength(values))
    {
        T *newEntries = (T *)PushArrayNoZero(temp.arena, u8, sizeof(T) * total);
        MemoryCopy(newEntries, values, sizeof(T) * count);
        count = 0;
        EndMutex(&mutex);

        MemoryCopy(newEntries + offset, entries, sizeof(T) * numEntries);

        ScratchEnd(scratch);
        Assert(material);
        handler(temp, state, newEntries, total, material);
    }
    else
    {
        MemoryCopy(values + count, entries, sizeof(T) * numEntries);
        count += numEntries;
        EndMutex(&mutex);

        ScratchEnd(scratch);
    }
}

template <typename T>
bool SharedShadeQueue<T>::Flush(ShadingThreadState *state)
{
    TempArena temp = ScratchStart(0, 0);
    bool result    = TryMutex(&mutex);
    if (!result) BeginMutex(&mutex);
    u32 total = count;
    if (count == 0)
    {
        EndMutex(&mutex);
        return result;
    }

    T *newEntries = (T *)PushArrayNoZero(temp.arena, u8, sizeof(T) * count);
    MemoryCopy(newEntries, values, sizeof(T) * count);
    count = 0;
    EndMutex(&mutex);

    handler(temp, state, newEntries, total, material);

    return false;
}

void FreeRayState(ShadingThreadState *state, RayStateHandle handle)
{
    state->rayFreeList.AddBack() = handle;
}

RayStateHandle AllocRayState(ShadingThreadState *state)
{
    RayStateHandle handle;
    state->rayFreeList.Pop(&handle);
    if (!handle.IsValid())
    {
        Assert(state->rayFreeList.totalCount == 0);
        for (u32 i = 0; i < 4096; i++)
        {
            RayState *rayState           = &state->rayStates.AddBack();
            state->rayFreeList.AddBack() = RayStateHandle{rayState};
        }
        state->rayFreeList.Pop(&handle);
        Assert(handle.IsValid());
    }
    MemoryZero(handle.GetRayState(), sizeof(RayState));
    return handle;
}

void WriteRadiance(const Vec2u &pixel, const SampledSpectrum &L,
                   const SampledWavelengths &lambda)
{
    Vec3f rgb               = ConvertRadianceToRGB(L, lambda);
    ShadingGlobals *globals = GetShadingGlobals();
    globals->rgbValues[pixel.x + globals->width * pixel.y] += rgb;
}

void TerminateRay(ShadingThreadState *shadeState, RayStateHandle handle)
{
    RayState *state = handle.GetRayState();
    WriteRadiance(state->pixel, state->L, state->lambda);
    FreeRayState(shadeState, handle);
}

// that's the problem. basically it's ray1->material1->ray2->material2,
// so  if ray1 fills materials2, then material1 can also fill material2...
void RenderSIMD(Arena *arena, RenderParams2 &params)
{
    u32 width              = params.width;
    u32 height             = params.height;
    u32 spp                = params.spp;
    Vec2f &filterRadius    = params.filterRadius;
    Mat4 &cameraFromRaster = params.cameraFromRaster;
    Mat4 &renderFromCamera = params.renderFromCamera;
    u32 maxDepth           = params.maxDepth;
    f32 lensRadius         = params.lensRadius;
    f32 focalLength        = params.focalLength;
    f32 maxComponentValue  = 10.f;

    // parallel for over tiles
    u32 pixelWidth  = width;
    u32 pixelHeight = height;
    if (params.pixelMin != params.pixelMax)
    {
        params.pixelMax = Min(params.pixelMax, Vec2u(width, height));
        params.pixelMin = Min(params.pixelMin, Vec2u(width, height));
        Assert(params.pixelMax[0] > params.pixelMin[0]);
        Assert(params.pixelMax[1] > params.pixelMin[1]);
        pixelWidth  = params.pixelMax[0] - params.pixelMin[0];
        pixelHeight = params.pixelMax[1] - params.pixelMin[1];
    }
    u32 tileWidth  = 16;
    u32 tileHeight = 16;
    u32 tileCountX = (pixelWidth + tileWidth - 1) / tileWidth;
    u32 tileCountY = (pixelHeight + tileHeight - 1) / tileHeight;
    u32 taskCount  = tileCountX * tileCountY;

    // TODO: instead of adding all tasks at once, add them to the thread queue
    // once the # queued is under a certain threshold (to save space)
    Image image;
    image.width         = width;
    image.height        = height;
    image.bytesPerPixel = sizeof(u32);
    image.contents      = PushArrayNoZero(arena, u8, GetImageSize(&image));

    std::atomic<u32> numTiles = 0;

    // Camera differentials
    Vec3f org      = TransformP(cameraFromRaster, Vec3f(0.f, 0.f, 0.f));
    Vec3f dxCamera = TransformP(cameraFromRaster, Vec3f(1.f, 0.f, 0.f)) - org;
    Vec3f dyCamera = TransformP(cameraFromRaster, Vec3f(0.f, 1.f, 0.f)) - org;

    Camera camera(cameraFromRaster, renderFromCamera, dxCamera, dyCamera, focalLength,
                  lensRadius, spp);

    GenerateMinimumDifferentials(camera, params, width, height, taskCount, tileCountX,
                                 tileWidth, tileHeight, pixelWidth, pixelHeight);

    ShadingGlobals *globals = GetShadingGlobals();
    globals->rgbValues      = PushArray(arena, Vec3f, width * height);
    globals->camera         = &camera;
    globals->width          = width;
    globals->height         = height;
    globals->maxDepth       = maxDepth;

    ParallelFor2D(
        Vec2i(params.pixelMin),
        Vec2i(params.pixelMin) + Vec2i((int)pixelWidth, (int)pixelHeight),
        Vec2i(tileWidth, tileHeight), [&](int jobID, Vec2i start, Vec2i end) {
            TempArena temp                         = ScratchStart(0, 0);
            ShadingThreadState *shadingThreadState = GetShadingThreadState();
            RayStateHandle *handles =
                PushArrayNoZero(temp.arena, RayStateHandle, QUEUE_LENGTH);
            u32 handleCount = 0;

            ZSobolSampler sampler(spp, Vec2i(width, height));
            for (u32 y = start.y; y < end.y; y++)
            {
                for (u32 x = start.x; x < end.x; x++)
                {
                    u32 *out = GetPixelPointer(&image, x, y);
                    Vec2u pPixel(x, y);
                    Vec3f rgb(0.f);
                    for (u32 i = 0; i < spp; i++)
                    {
                        sampler.StartPixelSample(Vec2i(x, y), i);
                        SampledWavelengths lambda = SampleVisible(sampler.Get1D());
                        Vec2f u                   = sampler.GetPixel2D();
                        // TODO: motion blur
                        sampler.Get1D();
                        // box filter
                        Vec2f filterSample =
                            Vec2f(Lerp(u[0], -filterRadius.x, filterRadius.x),
                                  Lerp(u[1], -filterRadius.y, filterRadius.y));
                        // converts from continuous to discrete coordinates
                        filterSample += Vec2f(0.5f, 0.5f) + Vec2f(pPixel);
                        Vec2f pLens = sampler.Get2D();

                        Ray2 ray = camera.GenerateRayDifferentials(filterSample, pLens);

                        f32 cameraWeight = 1.f;

                        RayStateHandle handle = AllocRayState(shadingThreadState);
                        RayState *rayState    = handle.GetRayState();
                        Assert(rayState->depth == 0);
                        rayState->ray      = ray;
                        rayState->beta     = SampledSpectrum(1.f);
                        rayState->etaScale = 1.f;
                        rayState->pixel    = pPixel;
                        rayState->lambda   = lambda;

                        MemoryCopy(&rayState->sampler, &sampler, sizeof(ZSobolSampler));

                        handles[handleCount++] = handle;
                        if (handleCount == QUEUE_LENGTH)
                        {
                            shadingThreadState->rayQueue.Push(temp, shadingThreadState,
                                                              handles, handleCount);
                            handleCount = 0;

                            temp = ScratchStart(0, 0);
                            handles =
                                PushArrayNoZero(temp.arena, RayStateHandle, QUEUE_LENGTH);
                        }
                    }
                }
            }
            shadingThreadState->rayQueue.Push(temp, shadingThreadState, handles, handleCount);

            ScratchEnd(temp);
            u32 n = numTiles.fetch_add(1);
            fprintf(stderr, "\rRaycasting %d%%...    ", u32(100.f * n / taskCount));
            fflush(stdout);
        });

    u32 numProcessors = OS_NumProcessors();
    // Flush all queues
    scheduler.ScheduleAndWait(numProcessors, 1, [&](u32 jobID) {
        ShadingThreadState *state = GetShadingThreadState(jobID);
        ShadingGlobals *globals   = GetShadingGlobals();
        bool done                 = true;
        u32 numShadingQueues      = globals->numShadingQueues;
        u32 start                 = u32((f32)jobID / numProcessors);

        do
        {
            done = true;
            done &= state->rayQueue.Flush(state);

            for (u32 i = 0; i < numShadingQueues; i++)
            {
                done &= globals->shadingQueues[(start + i) % numShadingQueues].Flush(state);
            }

        } while (!done);
    });

    for (u32 i = 0; i < numProcessors; i++)
    {
        ShadingThreadState *state = GetShadingThreadState(i);
        Assert(state->rayQueue.count == 0);
    }
    for (u32 i = 0; i < globals->numShadingQueues; i++)
    {
        Assert(globals->shadingQueues[i].count == 0);
    }

    ParallelFor2D(Vec2i(params.pixelMin),
                  Vec2i(params.pixelMin) + Vec2i((int)pixelWidth, (int)pixelHeight),
                  Vec2i(tileWidth, tileHeight), [&](int jobID, Vec2i start, Vec2i end) {
                      for (u32 y = start.y; y < end.y; y++)
                      {
                          for (u32 x = start.x; x < end.x; x++)
                          {
                              u32 *out = GetPixelPointer(&image, x, y);

                              Vec3f rgb = globals->rgbValues[x + image.width * y];

                              rgb /= f32(spp);
                              rgb = Mul(RGBColorSpace::sRGB->XYZToRGB, rgb);
                              if (rgb.x != rgb.x) rgb.x = 0.f;
                              if (rgb.y != rgb.y) rgb.y = 0.f;
                              if (rgb.z != rgb.z) rgb.z = 0.f;

                              // f32 r = 255.f * rgb.x;
                              // f32 g = 255.f * rgb.y;
                              // f32 b = 255.f * rgb.z;
                              f32 r = 255.f * ExactLinearToSRGB(rgb.x);
                              f32 g = 255.f * ExactLinearToSRGB(rgb.y);
                              f32 b = 255.f * ExactLinearToSRGB(rgb.z);
                              f32 a = 255.f;

                              Assert(r <= 255.f && g <= 255.f && b <= 255.f);

                              u32 color =
                                  (RoundFloatToU32(a) << 24) | (RoundFloatToU32(r) << 16) |
                                  (RoundFloatToU32(g) << 8) | (RoundFloatToU32(b) << 0);
                              *out = color;
                          }
                      }
                  });
    WriteImage(&image, "image.bmp");
    printf("done\n");
}

template <typename Handle>
void SortHandles(Handle *shadingHandles, u32 count)
{
    TempArena temp    = ScratchStart(0, 0);
    size_t handleSize = sizeof(shadingHandles[0].sortKey);
    Assert(handleSize == 4 || handleSize == 8);

    Handle *keys0 = (Handle *)shadingHandles;
    Handle *keys1 = PushArrayNoZero(temp.arena, Handle, count);

    // Radix sort
    for (int iter = (int)handleSize - 1; iter >= 0; iter--)
    {
        u32 shift = iter * 8;
        Assert(shift < 64);
        u32 buckets[256] = {};
        // Calculate # in each radix
        for (u32 i = 0; i < count; i++)
        {
            const Handle &key = keys0[i];
            u32 bucket        = (key.sortKey >> shift) & 0xff;
            Assert(bucket < 256);
            buckets[bucket]++;
        }
        // Prefix sum
        u32 total = 0;
        for (u32 i = 0; i < 256; i++)
        {
            u32 bucketCount = buckets[i];
            buckets[i]      = total;
            total += bucketCount;
        }

        // Place in correct position
        for (u32 i = 0; i < count; i++)
        {
            const Handle &key = keys0[i];
            u32 bucket        = (key.sortKey >> shift) & 0xff;
            u32 index         = buckets[bucket]++;
            Assert(index < count);
            keys1[index] = key;
        }
        Swap(keys0, keys1);
    }
    ScratchEnd(temp);
}

QUEUE_HANDLER(RayIntersectionHandler)
{
    RayStateHandle *handles = (RayStateHandle *)values;
    ShadingGlobals *globals = GetShadingGlobals();
    Scene *scene            = GetScene();

    struct SortEntry
    {
        u32 sortKey;
        u32 faceID;
        RayStateHandle handle;
    };

    TempArena temp = ScratchStart(&inScratch.arena, 1);

    SortEntry *nextHandles = PushArray(temp.arena, SortEntry, count);
    u32 nextHandleCount    = 0;

    for (u32 index = 0; index < count; index++)
    {
        RayStateHandle handle = handles[index];
        RayState *rayState    = handle.GetRayState();
        SurfaceInteraction si;
        bool intersect = Intersect(scene, rayState->ray, si);

        if (!intersect)
        {
            if (rayState->specularBounce || rayState->depth == 0)
            {
                for (auto &light : scene->infiniteLights)
                {
                    SampledSpectrum Le = light->Le(rayState->ray.d, rayState->lambda);
                    rayState->L += rayState->beta * Le;
                }
            }
            else
            {
                for (auto &light : scene->infiniteLights)
                {
                    SampledSpectrum Le = light->Le(rayState->ray.d, rayState->lambda);

                    f32 pdf      = LightPDF(scene);
                    f32 lightPdf = pdf * (f32)light->PDF_Li(rayState->ray.d, true);

                    f32 w_l = PowerHeuristic(1, rayState->bsdfPdf, 1, lightPdf);
                    // NOTE: beta already contains the cosine, bsdf, and pdf terms
                    rayState->L += rayState->beta * w_l * Le;
                }
            }

            TerminateRay(state, handle);
        }
        else
        {
            rayState->si                  = si;
            MaterialHandle materialHandle = (MaterialHandle)si.materialIDs;
            u32 materialIndex             = materialHandle.GetIndex();

            nextHandles[nextHandleCount].sortKey = materialIndex;
            nextHandles[nextHandleCount].faceID  = si.faceIndices;
            nextHandles[nextHandleCount].handle  = handle;
            nextHandleCount++;
        }
    }

    ScratchEnd(inScratch);

    // Sort by material index
    SortHandles(nextHandles, nextHandleCount);
    SortEntry *handleStop  = nextHandles + nextHandleCount;
    SortEntry *handleStart = nextHandles;
    while (handleStart != handleStop)
    {
        TempArena scratch = ScratchStart(&inScratch.arena, 1);
        // Contiguous ranges of the same material get pushed together
        u32 materialIndex    = handleStart->sortKey;
        SortEntry *handleEnd = handleStart;
        while (handleEnd->sortKey == materialIndex && handleEnd != handleStop)
        {
            handleEnd++;
        }

        unsigned range                = unsigned(handleEnd - handleStart);
        ShadingHandle *shadingHandles = PushArrayNoZero(scratch.arena, ShadingHandle, range);

        for (unsigned index = 0; index < range; index++)
        {
            shadingHandles[index].sortKey        = handleStart[index].faceID;
            shadingHandles[index].rayStateHandle = handleStart[index].handle;
        }
        u32 index = MaterialHandle(materialIndex) ? materialIndex : 0;
        Assert(index < globals->numShadingQueues);
        ShadingQueue *queue = GetShadingQueue(index);
        queue->Push(scratch, state, shadingHandles, range);

        handleStart = handleEnd;
    }
    ScratchEnd(temp);
}

template <typename MaterialType>
void ShadingQueueHandler(TempArena inScratch, struct ShadingThreadState *state,
                         ShadingHandle *values, u32 count, Material *m)
{
    ShadingGlobals *globals = GetShadingGlobals();

    ShadingHandle *handles = values;
    SortHandles(handles, count);

    Scene *scene = GetScene();

    MaterialType *material = (MaterialType *)m;

    material->Start(state);

    TempArena temp = ScratchStart(&inScratch.arena, 1);

    RayStateHandle *rayStateHandles = PushArrayNoZero(temp.arena, RayStateHandle, count);
    u32 rayStateHandleCount         = 0;

    for (u32 i = 0; i < count; i++)
    {
        ShadingHandle handle = handles[i];
        RayState *rayState   = handle.rayStateHandle.GetRayState();

        // Ray state stuff
        Ray2 &ray                  = rayState->ray;
        SurfaceInteraction &si     = rayState->si;
        ZSobolSampler &sampler     = rayState->sampler;
        SampledSpectrum &beta      = rayState->beta;
        f32 &etaScale              = rayState->etaScale;
        SampledWavelengths &lambda = rayState->lambda;
        SampledSpectrum &L         = rayState->L;
        f32 &bsdfPdf               = rayState->bsdfPdf;
        bool &specularBounce       = rayState->specularBounce;

        if constexpr (std::is_same_v<MaterialType, NullMaterial>)
        {
            specularBounce = true;
            ray.o          = OffsetRayOrigin(si.p, si.pError, si.n, ray.d);
            ray.tFar       = pos_inf;
            if (ray.pxOffset != Vec3f(pos_inf))
            {
                ray.pxOffset = ray.pxOffset + si.tHit * ray.dxOffset;
                ray.pyOffset = ray.pyOffset + si.tHit * ray.dyOffset;
            }
            // TODO: handle empty materials differently from medium interfaces
            if (rayState->depth++ >= globals->maxDepth)
            {
                TerminateRay(state, handle.rayStateHandle);
                continue;
            }
        }
        else
        {

            Vec3f dpdx, dpdy;
            f32 dudx, dvdx, dudy, dvdy;
            CalculateFilterWidths(rayState->ray, *globals->camera, si.p, si.n, si.dpdu,
                                  si.dpdv, dpdx, dpdy, dudx, dvdx, dudy, dvdy);

            ScratchArena scratch;

            {
                // BxDF evaluation
                ErrorExit(si.faceIndices == handle.sortKey, "face: %u, sort: %u\n",
                          si.faceIndices, handle.sortKey);
                BxDF bxdf = material->Evaluate(scratch.temp.arena, si, lambda,
                                               Vec4f(dudx, dvdx, dudy, dvdy));
                BSDF bsdf(bxdf, si.shading.dpdu, si.shading.n);

                if (rayState->depth++ >= globals->maxDepth)
                {
                    TerminateRay(state, handle.rayStateHandle);
                    continue;
                }

                // Next Event Estimation
                // Choose light source for direct lighting calculation
                if (!IsSpecular(bsdf.Flags()))
                {
                    // DebugBreak();
                    f32 lightU = sampler.Get1D();
                    f32 pmf;
                    Light *light = UniformLightSample(scene, lightU, &pmf);
                    Vec2f sample = sampler.Get2D();
                    if (light)
                    {
                        // Sample point on the light source
                        LightSample ls = light->SampleLi(si, sample, lambda, true);
                        if (ls.pdf)
                        {
                            // Evaluate BSDF for light sample, check visibility with shadow ray
                            f32 p_b;
                            SampledSpectrum f = bsdf.EvaluateSample(-ray.d, ls.wi, p_b) *
                                                AbsDot(si.shading.n, ls.wi);
                            if (f && !Occluded(scene, ray, si, ls))
                            {
                                // Calculate contribution
                                f32 lightPdf = pmf * ls.pdf;

                                if (IsDeltaLight(ls.lightType))
                                {
                                    L += beta * f * ls.L / lightPdf;
                                }
                                else
                                {
                                    f32 w_l = PowerHeuristic(1, lightPdf, 1, p_b);
                                    L += beta * f * w_l * ls.L / lightPdf;
                                }
                            }
                        }
                    }
                }

                // sample bsdf, calculate pdf
                f32 u             = sampler.Get1D();
                BSDFSample sample = bsdf.GenerateSample(-ray.d, u, sampler.Get2D());
                if (sample.pdf == 0.f)
                {
                    TerminateRay(state, handle.rayStateHandle);
                    continue;
                }
                // beta *= sample.f / sample.pdf;
                beta *= sample.f * AbsDot(si.shading.n, sample.wi) / sample.pdf;
                bsdfPdf        = sample.pdf;
                specularBounce = sample.IsSpecular();
                if (sample.IsTransmissive()) etaScale *= Sqr(sample.eta);

                // Offset ray and compute ray diferentials if necessary
                ray.o = OffsetRayOrigin(si.p, si.pError, si.n, sample.wi);

                ray.d    = sample.wi;
                ray.tFar = pos_inf;

                // Compute ray differentials for specular reflection or transmission
                // Compute common factors for specular ray differentials
                UpdateRayDifferentials(ray, sample.wi, si.p, si.shading.n, si.shading.dndu,
                                       si.shading.dndv, dpdx, dpdy, dudx, dvdx, dudy, dvdy,
                                       sample.eta, sample.flags);
            }
            // Russian roulette
            {
                SampledSpectrum rrBeta = beta * etaScale;
                f32 q                  = rrBeta.MaxComponentValue();
                if (rayState->depth > 1 && q < 1.f)
                {
                    if (sampler.Get1D() < Max(0.f, 1 - q))
                    {
                        TerminateRay(state, handle.rayStateHandle);
                        continue;
                    }

                    beta /= q;
                    // TODO: infinity check for beta
                }
            }
        }
        rayStateHandles[rayStateHandleCount++] = handle.rayStateHandle;
    }
    material->Stop();

    ScratchEnd(inScratch);
    state->rayQueue.Push(temp, state, rayStateHandles, rayStateHandleCount);
}

#if 0
struct ShadingQueue
{
    using BxDF = typename Material::BxDF;
    SurfaceInteraction queue[simdQueueLength];
    u32 count = 0;
    ShadingQueue() {}
    void Flush() // SortKey *keys, SurfaceInteraction *values, u32 num)
    {
        TempArena temp = ScratchStart(0, 0);

        u32 alignedCount = count & (IntN - 1);
        count -= alignedCount;
        u32 start = count;

        // f32 *pmfs = PushArrayNoZero(temp.arena, f32, alignedCount);
        // LightHandle *handles = PushArrayNoZero(temp.arena, LightHandle, alignedCount);
        // for (u32 i = 0; i < alignedCount; i++)
        // {
        //     SurfaceInteraction &intr = queue[start + i];
        //     // TODO: get this somehow
        //     Sampler sampler;
        //     LightHandle handle = UniformLightSample(sampler.Get1D(), &pmfs[i]);
        //     handles[i]         = handle;
        //
        //     keys0[i].value = GenerateKey(intr, handle.GetType());
        //     keys0[i].index = start + i;
        // }
        // IMPORTANT:
        // for ptex, radix sort using:
        // light type, mesh id, face id
        // for no ptex, radix sort using:
        // light type, uv
        // Create radix sort keys, get the light type

        // Convert to AOSOA
        // u32 limit = queueFlushSize % (IntN);
        // SurfaceInteractionsN aosoaIntrs[queueFlushSize / IntN];
        // for (u32 i = 0; i < alignedCount;)
        // {
        //     const u32 prefetchDistance = IntN * 2;
        //     alignas(32) SurfaceInteraction intrs[IntN];
        //     if (i + prefetchDistance < alignedCount)
        //     {
        //         for (u32 j = 0; j < IntN; j++)
        //         {
        //             _mm_prefetch((char *)&queue[keys[i + prefetchDistance + j].index],
        //                          _MM_HINT_T0);
        //             intrs[j] = queue[keys0[i + j].index];
        //         }
        //     }
        //     SurfaceInteractionsN aosoaIntrs; //&out = aosoaIntrs[aosoaIndex];
        //                                      // Transpose p, n, uv
        //     Transpose(intrs, aosoaIntrs);
        //     // Transpose8x8(Lane8F32::Load(&intrs[0]), Lane8F32::Load(&intrs[1]),
        //     // Lane8F32::Load(&intrs[2]), Lane8F32::Load(&intrs[3]),
        //     //              Lane8F32::Load(&intrs[4]), Lane8F32::Load(&intrs[5]),
        //     //              Lane8F32::Load(&intrs[6]), Lane8F32::Load(&intrs[7]), out.p.x,
        //     //              out.p.y, out.p.z, out.n.x, out.n.y, out.n.z, out.uv.x,
        //     out.uv.y);
        //     // Transpose the rest
        //
        //     MaskF32 continuationMask = LaneNF32::Mask<true>();
        //     BSDFBase<BxDF> bsdf      = Material::Evaluate(aosoaIntrs);
        //
        //     template <i32 width>
        //     struct LightSamples
        //     {
        //         SampledSpectrum Le;
        //         Vec3IF32 samplePoint;
        //         LaneNF32 pdf;
        //     };
        //
        //     Sampler samplers[];
        //
        //     //////////////////////////////
        //     // Next event estimation
        //     //
        //     if (Any(!IsSpecular(bsdf.Flags())))
        //     {
        //         RayStateHandle rayStateHandles[IntN];
        //
        //         // Sample lights
        //         alignas(32) LightHandle itrHandles[IntN];
        //         alignas(32) f32 pdfs[IntN];
        //         for (u32 j = 0; j < IntN; j++)
        //         {
        //             u32 index     = keys[i + j].index - start;
        //             itrHandles[j] = handles[index];
        //             pdfs[j]       = pmfs[index];
        //         }
        //         LaneIU32 handles  = LaneIU32::Load(itrHandles);
        //         u32 type          = itrHandles[0].GetType();
        //         LaneIU32 laneType = LaneIU32(itrHandles[0].GetType());
        //         MaskF32 mask      = (handles & laneType) == laneType;
        //         u32 maskBits      = Movemask(mask);
        //         u32 add           = PopCount(maskBits);
        //         LaneNF32 lightPdf = LaneNF32::Load(pdfs);
        //
        //         // TODO: get samplers
        //         LightSamples sample = SampleLi(type, handles, add, aosoaIntrs, lambda?,
        //         samplers);//scene, lightHandle, intr, lambda, u); mask &= sample.pdf == 0.f;
        //         lightPdf *= sample.pdf;
        //         // f32 scatterPdf;
        //         // SampledSpectrum f_hat;
        //         Vec3IF32 wi = Normalize(sample.samplePoint - aosoaIntrs.p);
        //
        //         LaneNF32 scatterPdf;
        //         SampledSpectrum f =
        //             BxDF::EvaluateSample(-ray.d, wi, scatterPdf, TransportMode::Radiance) *
        //             AbsDot(aosoaIntrs.shading.n, wi);
        //         // TODO: need to == with 0.f for every wavelength, and then combine together
        //         mask &= f.GetMask();
        //
        //         maskBits = Movemask(mask);
        //         // Shoot occlusion rays
        //         for (u32 j = 0; j < IntN; j++)
        //         {
        //             if (maskBits & (1 << j))
        //             {
        //                 // Shoot occlusion ray
        //                 // maskBits &= Occluded();
        //             }
        //         }
        //
        //         // Power heuristic
        //         LaneNF32 w_l = lightPdf / (Sqr(lightPdf) + Sqr(scatterPdf));
        //         // TODO: to prevent atomics, need to basically have thread permanently take
        //         a
        //         // tile
        //         L += Select(LaneNF32::Mask(maskbits), f * beta * w_l * sample.Le, 0.f);
        //         i += add;
        //     }
        //     else
        //     {
        //         i += IntN;
        //     }
        //
        //     // TODO: things I need to simd:
        //     // - sampler
        //     // - ray
        //     // - path throughput weight
        //     // - path flags
        //     // - path eta scale for russian roulette
        //
        //     // TODO: should I copy data (e.g. path throughput weights) so that stages can
        //     // happen in whatever order? otherwise, i need to ensure that next event
        //     // estimation, etc. happens before the bsdf sample is generated, otherwise the
        //     path
        //     // throughput weight needed for nee is lost
        //     Sampler *samplers[IntN];
        //     Ray2 *rays[IntN];
        //     LaneF32<IntN> u;
        //     Vec2lf<IntN> uv;
        //
        //     BSDFSample<IntN> sample =
        //         bsdf.GenerateSample(-ray.d, u, uv, TransportMode ::Radiance, BSDFFlags::RT);
        //     MaskF32 mask;
        //     mask = Select(sample.pdf == 0.f, 1, 0);
        //     beta *= sample.f * AbsDot(intr.shading.n, sample.wi) / sample.pdf;
        //
        //     // store back path throughput
        //     // store back radiance
        //     // store back depth
        //     // store back eta scale
        //
        //     // TODO: path flags, set specular bounce to true
        //     // pathFlags &= bsdf.IsSpecular();
        //
        //     // Spawn a new ray, push to the ray queue
        //     // Russian roulette
        // }

        ScratchEnd(temp);
    }
}
};
#endif
} // namespace rt
