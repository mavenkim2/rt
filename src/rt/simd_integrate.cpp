#include "simd_integrate.h"
namespace rt
{

void FreeRayState(RayStateHandle handle)
{
    ShadingThreadState *state    = GetShadingThreadState();
    state->rayFreeList.AddBack() = handle;
}

RayStateHandle AllocRayState()
{
    ShadingThreadState *state = GetShadingThreadState();
    RayStateHandle handle     = state->rayFreeList.Pop();
    MemoryZero(handle.GetRayState(), sizeof(RayState));
    return handle;
}

void WriteRadiance(const Vec2u &pixel, const SampledSpectrum &L,
                   const SampledWavelengths &lambda)
{
    Vec3f rgb               = ConvertRadianceToRGB(L, lambda);
    ShadingGlobals *globals = GetShadingGlobals();
    globals->radiances[pixel.x + (height - pixel.y - 1) * width] += rgb;
}

void TerminateRay(RayStateHandle handle)
{
    WriteRadiance();
    FreeRayState(handle);
}

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
    u32 tileWidth  = 64;
    u32 tileHeight = 64;
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
    Vec3f dxCamera = TransformV(cameraFromRaster, Vec3f(1.f, 0.f, 0.f));
    Vec3f dyCamera = TransformV(cameraFromRaster, Vec3f(0.f, 1.f, 0.f));

    Camera camera(cameraFromRaster, renderFromCamera, dxCamera, dyCamera, focalLength,
                  lensRadius, spp);

    GenerateMinimumDifferentials(camera, params, width, height, taskCount, tileCountX,
                                 tileWidth, tileHeight, pixelWidth, pixelHeight);

    scheduler.ScheduleAndWait(taskCount, 1, [&](u32 jobID) {
        ShadingThreadState *shadingThreadState = GetShadingThreadState();
        u32 tileX                              = jobID % tileCountX;
        u32 tileY                              = jobID / tileCountX;
        Vec2u minPixelBounds(params.pixelMin[0] + tileWidth * tileX,
                             params.pixelMin[1] + tileHeight * tileY);
        Vec2u maxPixelBounds(
            Min(params.pixelMin[0] + tileWidth * (tileX + 1), params.pixelMin[0] + pixelWidth),
            Min(params.pixelMin[1] + tileHeight * (tileY + 1),
                params.pixelMin[1] + pixelHeight));

        Assert(maxPixelBounds.x >= minPixelBounds.x && minPixelBounds.x >= 0 &&
               maxPixelBounds.x <= width);
        Assert(maxPixelBounds.y >= minPixelBounds.y && minPixelBounds.y >= 0 &&
               maxPixelBounds.y <= height);

        ZSobolSampler sampler(spp, Vec2i(width, height));
        for (u32 y = minPixelBounds.y; y < maxPixelBounds.y; y++)
        {
            for (u32 x = minPixelBounds.x; x < maxPixelBounds.x; x++)
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
                    Vec2f filterSample = Vec2f(Lerp(u[0], -filterRadius.x, filterRadius.x),
                                               Lerp(u[1], -filterRadius.y, filterRadius.y));
                    // converts from continuous to discrete coordinates
                    filterSample += Vec2f(0.5f, 0.5f) + Vec2f(pPixel);
                    Vec2f pLens = sampler.Get2D();

                    Ray2 ray = camera.GenerateRayDifferentials(filterSample, pLens);

                    f32 cameraWeight = 1.f;

                    RayStateHandle handle = AllocRayState();
                    RayState *rayState    = handle.GetRayState();
                    rayState->ray         = ray;
                    rayState->pixel       = pPixel;

                    // Clone sampler
                    rayState->sampler = sampler.Clone();
                    shadingThreadState->rayQueue.Push(handle);
                    shadingThreadState->rayQueue.Flush();

                    SampledSpectrum L =
                        cameraWeight * Li(ray, camera, sampler, maxDepth, lambda);
                    // convert radiance to rgb, add and divide
                    L               = SafeDiv(L, lambda.PDF());
                    f32 r           = (Spectra::X().Sample(lambda) * L).Average();
                    f32 g           = (Spectra::Y().Sample(lambda) * L).Average();
                    f32 b           = (Spectra::Z().Sample(lambda) * L).Average();
                    f32 m           = Max(r, Max(g, b));
                    Vec3f sampleRgb = Vec3f(r, g, b);
                    if (m > maxComponentValue)
                    {
                        sampleRgb *= maxComponentValue / m;
                    }
                    rgb += sampleRgb;
                }
                // TODO: filter importance sampling
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

                u32 color = (RoundFloatToU32(a) << 24) | (RoundFloatToU32(r) << 16) |
                            (RoundFloatToU32(g) << 8) | (RoundFloatToU32(b) << 0);
                *out = color;
            }
        }
        u32 n = numTiles.fetch_add(1);
        fprintf(stderr, "\rRaycasting %d%%...    ", u32(100.f * n / taskCount));
        fflush(stdout);
    });

    // TODO: write rgb to image
    // scheduler.ScheduleAndWait()
    // {
    // }
    WriteImage(&image, "image.bmp");
    printf("done\n");
}

void RayIntersectionHandler(RayStateHandle *handles, u32 count)
{
    ShadingThreadState *state = GetShadingThreadState();
    Scene *scene              = GetScene();

    for (u32 i = 0; i < count; i++)
    {
        RayStateHandle handle = handles[i];
        RayState *rayState    = handle.GetRayState();
        SurfaceInteraction si;
        bool intersect = Intersect(scene, rayState->ray, si);

        if (!intersect)
        {
            if (rayState->specularBounce || rayState->depth == 0)
            {
                ForEachTypeSubset(
                    scene->lights,
                    [&](auto *array, u32 count) {
                        using Light = std::remove_reference_t<decltype(*array)>;
                        for (u32 i = 0; i < count; i++)
                        {
                            Light &light       = array[i];
                            SampledSpectrum Le = Light::Le(&light, ray.d, lambda);
                            rayState->L += rayState->beta * Le;
                        }
                    },
                    InfiniteLightTypes());
            }
            else
            {
                ForEachTypeSubset(
                    scene->lights,
                    [&](auto *array, u32 count) {
                        using Light = std::remove_reference_t<decltype(*array)>;
                        for (u32 i = 0; i < count; i++)
                        {
                            Light &light       = array[i];
                            SampledSpectrum Le = Light::Le(&light, ray.d, rayState->lambda);

                            f32 pdf      = LightPDF(scene);
                            f32 lightPdf = pdf * (f32)Light::PDF_Li(&light, ray.d, true);

                            f32 w_l = PowerHeuristic(1, rayState->bsdfPdf, 1, lightPdf);
                            // NOTE: beta already contains the cosine, bsdf, and pdf terms
                            rayState->L += rayState->beta * w_l * Le;
                        }
                    },
                    InfiniteLightTypes());
            }

            TerminateRay(handle);
        }
        else if (rayState->depth++ >= maxDepth)
        {
            return;
        }
        else
        {
            rayState->si                  = si;
            MaterialHandle materialHandle = (MaterialHandle)si.materialIDs;
            u32 materialType              = (u32)materialHandle.GetType();
            u32 materiaIndex              = materialHandle.GetIndex();
            u64 shadingKey                = ((u64)materiaIndex << 32) | si.faceIndices;
            state->shadingQueues[materialType].Push(ShadingHandle{shadingKey, handle});
        }
    }
    for (u32 i = 0; i < (u32)MaterialTypes::Max; i++)
    {
        state->shadingQueues[i].Flush();
    }
}

template <typename MaterialType>
void ShadingQueueHandler(void *values, u32 count)
{
    ShadingThreadState *state = GetShadingThreadState();
    ShadingHandle *keys0      = (ShadingHandle *)values;
    ShadingHandle *keys1      = PushArrayNoZero(temp.arena, SortKey, alignedCount);

    // Radix sort
    for (u32 iter = 3; iter >= 0; iter--)
    {
        u32 shift        = iter * 8;
        u32 buckets[255] = {};
        // Calculate # in each radix
        for (u32 i = 0; i < queueFlushSize; i++)
        {
            ShadingHandle *key = &keys0[i];
            buckets[(key->shadingKey >> shift) & 0xff]++;
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
            ShadingHandle &sortKey = &keys0[i];
            u32 key                = (sortKey.shadingKey >> shift) & 0xff;
            keys1[buckets[key]++]  = sortKey;
        }
        Swap(keys0, keys1);
    }

    Scene *scene = GetScene();
    for (u32 i = 0; i < count; i++)
    {
        ShadingHandle handle   = keys0[i];
        MaterialType *material = (MaterialType *)scene->materials[handle.shadingKey >> 32];
        RayState *rayState     = state->rayStates[handle];

        // Ray state stuff
        Ray2 &ray                 = rayState->ray;
        SurfaceInteraction &si    = rayState->si;
        Sampler &sampler          = rayState->sampler;
        SampledSpectrum &beta     = rayState->beta;
        SampledSpectrum &etaScale = rayState->etaScale;
        bool &specularBounce      = rayState->specularBounce;

        Vec3f dpdx, dpdy;
        f32 dudx, dvdx, dudy, dvdy;
        CalculateFilterWidths(rayState->ray, rayState->si.p, rayState->si.n, dpdx, dpdy, dudx,
                              dvdx, dudy, dvdy);

        ScratchArena scratch;

        // Offset ray and compute ray diferentials if necessary
        {
            ray.o = OffsetRayOrigin(si.p, si.pError, si.n, sample.wi);

            ray.d    = sample.wi;
            ray.tFar = pos_inf;

            // Compute ray differentials for specular reflection or transmission
            // Compute common factors for specular ray differentials
            UpdateRayDifferentials(ray, sample.wi, si.shading.n, si.shading.dndu,
                                   si.shading.dndv, dudx, dvdx, dudy, dvdy, sample.eta);
        }
        {
            // BxDF evaluation
            BxDF bxdf = material->Evaluate(scratch.temp.arena, si, lambda,
                                           Vec4f(dudx, dvdx, dudy, dvdy));
            BSDF bsdf(bxdf, si.shading.dpdu, si.shading.n);

            // Next Event Estimation
            // Choose light source for direct lighting calculation
            if (!IsSpecular(bsdf.Flags()))
            {
                f32 lightU = sampler.Get1D();
                f32 pmf;
                LightHandle lightHandle = UniformLightSample(scene, lightU, &pmf);
                Vec2f sample            = sampler.Get2D();
                if (bool(lightHandle))
                {
                    // Sample point on the light source
                    LightSample ls = SampleLi(scene, lightHandle, si, lambda, sample, true);
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
                TerminateRay(handle.rayStateHandle);
                continue;
            }
            // beta *= sample.f / sample.pdf;
            beta *= sample.f * AbsDot(si.shading.n, sample.wi) / sample.pdf;
            bsdfPdf        = sample.pdf;
            specularBounce = sample.IsSpecular();
            if (sample.IsTransmissive()) etaScale *= Sqr(sample.eta);
        }
        // Russian roulette
        {
            SampledSpectrum rrBeta = beta * etaScale;
            f32 q                  = rrBeta.MaxComponentValue();
            if (depth > 1 && q < 1.f)
            {
                if (sampler.Get1D() < Max(0.f, 1 - q))
                {
                    TerminateRay(handle.rayStateHandle);
                    continue;
                }

                beta /= q;
                // TODO: infinity check for beta
            }
        }

        // Push to secondary ray queue
        state->rayQueue.Push(handle.rayStateHandle);
    }
    state->rayQueue.Flush();
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
