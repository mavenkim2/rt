#include "integrate.h"
#include "lights.h"
#include "bsdf.h"
#include "scene.h"
#include <type_traits>

namespace rt
{
// TODO
// - loading volumes
// - volumetric
//      - ratio tracking, residual ratio tracking, delta tracking <-- done but untested
//      - virtual density segments?
// - bvh intersection and triangle intersection
// - shading, ptex, materials, textures
//      - ray differentials

// after that's done:
// - equiangular sampling
// - simd queues for everything (radiance evaluation, shading, ray streams?)
// - bdpt, metropolis, vcm/upbp, mcm?
// - subdivision surfaces

// harder stuff
// - covariance tracing
// - path guiding

//////////////////////////////
// Textures and materials
//
void InitializePtex()
{
    u32 maxFiles  = 100;
    size_t maxMem = gigabytes(4);
    cache         = Ptex::PtexCache::create(maxFiles, maxMem, true, 0, &errorHandler);
}

template <typename RflShader>
DiffuseBxDF DiffuseMaterial<RflShader>::GetBxDF(SurfaceInteractionsN &intr, DiffuseMaterial **materials, Vec4lfn &filterWidths,
                                                SampledWavelengthsN &lambda)
{
    // TODO: vectorized texture evaluation?
    // TODO: sampled spectrum vectorized
    RflShader *shaders[IntN];
    for (u32 i = 0; i < IntN; i++)
    {
        shaders[i] = &materials[i]->rflShader;
    }
    SampledSpectrumN sampledSpectra = RflShader::Evaluate(intr, filterWidths, shaders, lambda);
    return DiffuseBxDF(sampledSpectra);
}

template <typename RflShader>
DiffuseBxDF DiffuseMaterial<RflShader>::GetBxDF(SurfaceInteraction &intr, Vec4lfn &filterWidths, SampledWavelengthsN &lambda)
{
    return DiffuseMaterial::GetBxDF(intr, &this);
}

template <typename RflShader, typename TrmShader>
DiffuseTransmissionBxDF DiffuseTransmissionMaterial<RflShader, TrmShader>::GetBxDF(SurfaceInteractionsN &intr,
                                                                                   DiffuseTransmissionMaterial **materials, Vec4lfn &filterWidths,
                                                                                   SampledWavelengthsN &lambda)
{
    RflShader *rflShaders[IntN];
    TrmShader *trmShaders[IntN];
    for (u32 i = 0; i < IntN; i++)
    {
        rflShaders[i] = &materials[i]->rflShader;
        trmShaders[i] = &materials[i]->trmShaders;
    }
    SampledSpectrumN r = RflShader::Evaluate(intr, filterWidths, rflShaders, lambda);
    SampledSpectrumN t = TrmShader::Evaluate(intr, filterWidths, trmShaders, lambda);
    return DiffuseTransmissionBxDF(r, t);
}

template <typename RflShader, typename TrmShader>
DiffuseTransmissionBxDF DiffuseTransmissionMaterial<RflShader, TrmShader>::GetBxDF(SurfaceInteractionsN &intr, Vec4lfn &filterWidths,
                                                                                   SampledWavelengthsN &lambda)
{
    return DiffuseTransmissionMaterial::GetBxDF(intr, &this);
}

template <typename RghShader, typename IORShader>
DielectricBxDF DielectricMaterial<RghShader, IORShader>::GetBxDF(SurfaceInteractionsN &intr, DielectricMaterial **materials, Vec4lfn &filterWidths,
                                                                 SampledWavelengthsN &lambda)
{
    RghShader *rghShaders[IntN];
    LaneNF32 eta;
    for (u32 i = 0; i < IntN; i++)
    {
        rghShaders[i] = &materials[i]->rghShader;
        Set(eta, i)   = materials[i]->ior(Get(lambda[0], i));
    }
    // NOTE: for dispersion (i.e. wavelength dependent IORs), we terminate every wavelength except the first
    if constexpr (!std::is_same_v<Spectrum, ConstantSpectrum>)
    {
        lambda.TerminateSecondary();
    }

    // TODO: anisotropic roughness
    LaneNF32 roughness = RghShader::Evaluate(intr, rghShaders, filterWidths, lambda);
    roughness          = TrowbridgeReitzDistribution::RoughnessToAlpha(roughness);
    TrowbridgeReitzDistribution distrib(roughness, roughness);
    return DielectricBxDF(eta, distrib);
}

template <typename RghShader, typename IORShader>
DielectricBxDF DielectricMaterial<RghShader, IORShader>::GetBxDF(SurfaceInteractionsN &intr, Vec4lfn &filterWidths, SampledWavelengthsN &lambda)
{
    return DielectricMaterial::GetBxDF(intr, &this);
}

typedef u32 PathFlags;
enum
{
    PathFlags_SpecularBounce,
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

// TODO: one for each type of material
// TODO: if no intersection, then shove to the end? beginning? of the queue
template <typename Material>
struct ShadingQueuePtex
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
        for (u32 i = 0; i < alignedCount;) //, aosoaIndex = 0; i < queueFlushSize; i += IntN, aosoaIndex++)
        {
            const u32 prefetchDistance = IntN * 2;
            alignas(32) SurfaceInteraction intrs[IntN];
            if (i + prefetchDistance < alignedCount)
            {
                for (u32 j = 0; j < IntN; j++)
                {
                    _mm_prefetch((char *)&queue[keys[i + prefetchDistance + j].index], _MM_HINT_T0);
                    intrs[j] = queue[keys0[i + j].index];
                }
            }
            SurfaceInteractionsN aosoaIntrs; //&out = aosoaIntrs[aosoaIndex];
                                             // Transpose p, n, uv
            Transpose(intrs, aosoaIntrs);
            // Transpose8x8(Lane8F32::Load(&intrs[0]), Lane8F32::Load(&intrs[1]), Lane8F32::Load(&intrs[2]), Lane8F32::Load(&intrs[3]),
            //              Lane8F32::Load(&intrs[4]), Lane8F32::Load(&intrs[5]), Lane8F32::Load(&intrs[6]), Lane8F32::Load(&intrs[7]),
            //              out.p.x, out.p.y, out.p.z, out.n.x, out.n.y, out.n.z, out.uv.x, out.uv.y);
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
                SampledSpectrum f = BxDF::EvaluateSample(-ray.d, wi, scatterPdf, TransportMode::Radiance) * AbsDot(aosoaIntrs.shading.n, wi);
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
                // TODO: to prevent atomics, need to basically have thread permanently take a tile
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

            // TODO: should I copy data (e.g. path throughput weights) so that stages can happen in whatever order? otherwise,
            // i need to ensure that next event estimation, etc. happens before the bsdf sample is generated, otherwise the
            // path throughput weight needed for nee is lost
            Sampler *samplers[IntN];
            Ray2 *rays[IntN];
            LaneF32<IntN> u;
            Vec2lf<IntN> uv;

            BSDFSample<IntN> sample = bsdf.GenerateSample(-ray.d, u, uv, TransportMode ::Radiance, BSDFFlags::RT);
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
};

template <>
bool SurfaceInteraction::ComputeShading(BSDF &bsdf)
{
    // TODO:
    // auto material = scene->materials.Get(0, materialIDs.value);
    // auto Dispatch = [&](auto v) {
    //     using BSDFType = std::decay<decltype(v)>::type;
    //     bsdf           = &v;
    //     BSDFType *b =
    // };

    return {};
}

template <typename Sampler>
SampledSpectrum Li(Ray2 &ray, Sampler &sampler, u32 maxDepth, SampledWavelengths &lambda);

void Render(Arena *arena, RenderParams2 &params) // Vec2i imageDim, Vec2f filterRadius, u32 spp, Mat4 &cameraFromRaster, )
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
    u32 tileWidth  = 64;
    u32 tileHeight = 64;
    u32 tileCountX = (width + tileWidth - 1) / tileWidth;
    u32 tileCountY = (height + tileHeight - 1) / tileHeight;
    u32 taskCount  = tileCountX * tileCountY;

    Image image;
    image.width         = width;
    image.height        = height;
    image.bytesPerPixel = sizeof(u32);
    image.contents      = PushArrayNoZero(arena, u8, GetImageSize(&image));

    scheduler.ScheduleAndWait(taskCount, 1, [&](u32 jobID) {
        u32 tileX = jobID % tileCountX;
        u32 tileY = jobID / tileCountX;
        Vec2u minPixelBounds(tileWidth * tileX, tileHeight * tileY);
        Vec2u maxPixelBounds(Min(tileWidth * (tileX + 1), width), Min((tileY + 1) * tileHeight, height));

        Assert(maxPixelBounds.x >= minPixelBounds.x && minPixelBounds.x >= 0 && maxPixelBounds.x <= width);
        Assert(maxPixelBounds.y >= minPixelBounds.y && minPixelBounds.y >= 0 && maxPixelBounds.y <= height);

        ZSobolSampler sampler(spp, Vec2i(width, height));
        for (u32 y = minPixelBounds.y; y < maxPixelBounds.y; y++)
        {
            u32 *out = GetPixelPointer(&image, minPixelBounds.x, y);
            for (u32 x = minPixelBounds.x; x < maxPixelBounds.x; x++)
            {
                Vec2u pPixel(x, y);
                Vec3f rgb(0.f);
                for (u32 i = 0; i < spp; i++)
                {
                    sampler.StartPixelSample(Vec2i(x, y), i);
                    SampledWavelengths lambda = SampleVisible(sampler.Get1D());
                    // box filter
                    Vec2f u            = sampler.Get2D();
                    Vec2f filterSample = Vec2f(Lerp(u[0], -filterRadius.x, filterRadius.x), Lerp(u[1], -filterRadius.y, filterRadius.y));
                    // converts from continuous to discrete coordinates
                    filterSample += Vec2f(0.5f, 0.5f) + Vec2f(pPixel);
                    Vec2f pLens = sampler.Get2D();

                    Vec3f pCamera = TransformP(cameraFromRaster, Vec3f(filterSample, 0.f));
                    Ray2 ray(Vec3f(0.f, 0.f, 0.f), Normalize(pCamera), pos_inf);
                    if (lensRadius > 0.f)
                    {
                        pLens = lensRadius * SampleUniformDiskConcentric(pLens);

                        // point on plane of focus
                        f32 t        = focalLength / -ray.d.z;
                        Vec3f pFocus = ray(t);
                        ray.o        = Vec3f(pLens.x, pLens.y, 0.f);
                        // ensure ray intersects focal point
                        ray.d = Normalize(pFocus - ray.o);
                    }
                    ray               = Transform(renderFromCamera, ray);
                    f32 cameraWeight  = 1.f;
                    SampledSpectrum L = cameraWeight * Li(ray, sampler, maxDepth, lambda);
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

                // f32 m = Max(rgb.x, Max(rgb.y, rgb.z));
                // if (m > 1.f)
                // {
                //     rgb *= 1.f / m;
                // }
                f32 r = 255.f * rgb.x;
                f32 g = 255.f * rgb.y;
                f32 b = 255.f * rgb.z;
                f32 a = 255.f;

                Assert(r <= 255.f && g <= 255.f && b <= 255.f);

                u32 color = (RoundFloatToU32(a) << 24) | (RoundFloatToU32(r) << 16) | (RoundFloatToU32(g) << 8) | (RoundFloatToU32(b) << 0);
                *out++    = color;
            }
        }
    });
    WriteImage(&image, "image.bmp");
    printf("done\n");
}

f32 PowerHeuristic(u32 numA, f32 pdfA, u32 numB, f32 pdfB)
{
    f32 a = Sqr(numA * pdfA);
    f32 b = Sqr(numB * pdfB);
    return a / (a + b);
}

void EvaluateMaterial(Arena *arena, SurfaceInteraction &si, BSDF *bsdf, SampledWavelengths &lambda)
{
    using MaterialTypes = Scene2::MaterialTypes;
    Dispatch(
        [&](auto t) {
            using MaterialType = std::decay_t<decltype(t)>;
            MaterialType::Evaluate(arena, si, lambda, bsdf);
        },
        Scene2::MaterialTypes(), MaterialHandle::GetType(si.materialIDs));
}

template <typename Sampler>
SampledSpectrum Li(Ray2 &ray, Sampler &sampler, u32 maxDepth, SampledWavelengths &lambda)
{
    Scene2 *scene = GetScene();
    u32 depth     = 0;
    SampledSpectrum L(0.f);
    SampledSpectrum beta(1.f);

    bool specularBounce = false;
    f32 bsdfPdf         = 0.f;
    f32 etaScale        = 1.f;

    SurfaceInteraction prevSi;

    for (;;)
    {
        SurfaceInteraction si;
        // TODO: not hardcoded
        bool intersect = BVHTriangleIntersectorCmp4::Intersect(ray, scene->nodePtr, si);

        // If no intersection, sample "infinite" lights (e.g environment maps, sun, etc.)
        if (!intersect)
        {
            // Eschew MIS when last bounce is specular or depth is zero (because either light wasn't previously sampled,
            // or it wasn't sampled with MIS)
            if (specularBounce || depth == 0)
            {
                ForEachTypeSubset(
                    scene->lights,
                    [&](auto *array, u32 count) {
                        using Light = std::remove_reference_t<decltype(*array)>;
                        for (u32 i = 0; i < count; i++)
                        {
                            Light &light       = array[i];
                            SampledSpectrum Le = Light::Le(&light, ray.d, lambda);
                            L += beta * Le;
                        }
                    },
                    Scene2::InfiniteLightTypes());
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
                            SampledSpectrum Le = Light::Le(&light, ray.d, lambda);

                            f32 pdf      = LightPDF(scene);
                            f32 lightPdf = pdf * (f32)Light::PDF_Li(&light, ray.d, true);

                            f32 w_l = PowerHeuristic(1, bsdfPdf, 1, lightPdf);
                            // NOTE: beta already contains the cosine, bsdf, and pdf terms
                            L += beta * w_l * Le;
                        }
                    },
                    Scene2::InfiniteLightTypes());
            }

            break;
        }
        // If intersected with a light
        if (si.lightIndices)
        {
            Assert(0);
            DiffuseAreaLight *light = &scene->GetAreaLights()[LightHandle(si.lightIndices).GetIndex()];
            if (specularBounce || depth == 0)
            {
                SampledSpectrum Le = DiffuseAreaLight::Le(light, si.n, -ray.d, lambda);
                L += beta * Le;
            }
            else
            {
                SampledSpectrum Le = DiffuseAreaLight::Le(light, si.n, -ray.d, lambda);
                // probability of sampling the light * probability of sampling point on light
                f32 pmf      = LightPDF(scene);
                f32 lightPdf = pmf * DiffuseAreaLight::PDF_Li(scene, si.lightIndices, prevSi.p, si, true);
                f32 w_l      = PowerHeuristic(1, bsdfPdf, 1, lightPdf);
                // NOTE: beta already contains the cosine, bsdf, and pdf terms
                L += beta * w_l * Le;
            }
        }

        if (depth >= maxDepth)
        {
            break;
        }

        TempArena temp = ScratchStart(0, 0);
        // BSDF bsdf; // = si.GetBSDF();
        BSDF bsdf;
        EvaluateMaterial(temp.arena, si, &bsdf, lambda);

        // Next Event Estimation
        // Choose light source for direct lighting calculation
        f32 lightU = sampler.Get1D();
        f32 pmf;
        LightHandle handle = UniformLightSample(scene, lightU, &pmf);
        if (bool(handle))
        {
            Vec2f sample = sampler.Get2D();
            // Sample point on the light source
            LightSample ls = SampleLi(scene, handle, si, lambda, sample);
            if (ls.pdf)
            {
                // Evaluate BSDF for light sample, check visibility with shadow ray
                f32 p_b;
                SampledSpectrum f = bsdf.EvaluateSample(-ray.d, ls.wi, p_b) * AbsDot(si.shading.n, ls.wi);
                if (f && !BVHTriangleIntersector4::Occluded(ray, scene->nodePtr))
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

        // sample bsdf, calculate pdf
        BSDFSample sample = bsdf.GenerateSample(-ray.d, sampler.Get1D(), sampler.Get2D());
        beta *= sample.f * AbsDot(si.shading.n, sample.wi) / sample.pdf;
        bsdfPdf        = sample.pdf;
        specularBounce = sample.IsSpecular();
        if (sample.IsTransmissive()) etaScale *= Sqr(sample.eta);

        // Spawn new ray
        prevSi   = si;
        ray.o    = si.p;
        ray.d    = sample.wi;
        ray.tFar = pos_inf;

        // Russian Roulette
        SampledSpectrum rrBeta = beta * sample.eta;
        f32 q                  = rrBeta.MaxComponentValue();
        if (depth > 1 && q < 1.f)
        {
            if (sampler.Get1D() < Max(0.f, 1 - q)) break;

            beta /= q;
            // TODO: infinity check for beta
        }
        ScratchEnd(temp);
    }
    return L;
}

//////////////////////////////
// Volumes
//

#if 0
void VolumeAggregate::Build(Arena *arena)
{
    Scene2 *scene = GetScene();
    const f32 T   = -1.f / std::log(0.5f);
    // Loop over the bounds of the volume
    Bounds bounds;
    ForEachType(scene->volumes, [&](auto *array, u32 count) {
        for (u32 i = 0; i < count; i++)
        {
            bounds.Extend(array[i].bounds);
        }
    });
    volumeBounds = bounds;

    f32 maxExtent = neg_inf;
    Lane4F32 diag = bounds.Diagonal();
    for (u32 i = 0; i < 3; i++)
    {
        if (diag[i] > maxExtent)
        {
            maxExtent = diag[i];
        }
    }

    root                = PushStruct(arena, OctreeNode);
    root->extinctionMin = pos_inf;
    struct StackEntry
    {
        OctreeNode *node;
        Bounds b;
    };
    ForEachType(scene->volumes, [&](auto *array, u32 count) {
        for (u32 i = 0; i < count; i++)
        {
            auto *volume = &array[i];
            // Volume *volume = &scene->volumes[i];
            StackEntry stack[64];
            i32 stackPtr      = 0;
            stack[stackPtr++] = StackEntry{root, bounds};

            while (stackPtr > 0)
            {
                StackEntry entry = stack[--stackPtr];
                Bounds &b        = entry.b;
                OctreeNode *node = entry.node;
                // Get minorant and majorant
                f32 extinctionMin, extinctionMax;
                volume->QueryExtinction(bounds, extinctionMin, extinctionMax);
                if (!extinctionMax) continue;

                node->volumeHandles[node->numVolumes++] = VolumeHandle(i);
                node->extinctionMax                     = Max(node->extinctionMax, extinctionMax);
                node->extinctionMin                     = Min(node->extinctionMin, extinctionMin);
                // max(R) - min(R) * diag(R) > T
                bool divide = (extinctionMax - extinctionMin) * Length(ToVec3f(b.Diagonal())) > T;
                if (divide)
                {
                    if (!node->children)
                    {
                        node->children = PushArray(arena, OctreeNode, 8);
                        for (u32 childIndex = 0; childIndex < 8; childIndex++)
                        {
                            node->children[i].extinctionMin = node->extinctionMin;
                            node->children[i].extinctionMax = node->extinctionMax;
                        }
                    }
                    Lane4F32 centroid = b.Centroid();
                    Lane4F32 mins[2]  = {b.minP, centroid};
                    Lane4F32 maxs[2]  = {centroid, b.maxP};
                    for (u32 childIndex = 0; childIndex < 8; childIndex++)
                    {
                        Lane4F32 minP(mins[childIndex & 1][0], mins[(childIndex & 3) >> 1][1],
                                      mins[childIndex >> 2][2], 0.f);

                        Lane4F32 maxP(maxs[childIndex & 1][0], maxs[(childIndex & 3) >> 1][1],
                                      maxs[childIndex >> 2][2], 0.f);
                        Bounds newBounds(minP, maxP);
                        stack[stackPtr++] = {&node->children[childIndex], newBounds};
                    }
                }
            }
        }
    });
}

bool VolumeAggregate::Iterator::Next(RaySegment &segment)
{
    while (stackPtr)
    {
        StackEntry &entry = entries[--stackPtr];
        OctreeNode *node  = entry.node;
        Assert(node);

        // If leaf
        if (!node->children)
        {
            segment = RaySegment(entry.tMin, entry.tMax, node->extinctionMin, node->extinctionMax, cExtinct, node->volumeHandles);
            return true;
        }

        Bounds &b         = entry.b;
        Lane4F32 centroid = b.Centroid();

        // Calculate bounds, intersect ray
        Lane8F32 minX = Blend<0xaa>(Lane8F32(b.minP[0]), Lane8F32(centroid[0]));
        Lane8F32 minY = Blend<0xcc>(Lane8F32(b.minP[1]), Lane8F32(centroid[1]));
        Lane8F32 minZ = Blend<0xf0>(Lane8F32(b.minP[2]), Lane8F32(centroid[2]));

        Lane8F32 maxX = Blend<0xaa>(Lane8F32(centroid[0]), Lane8F32(b.maxP[0]));
        Lane8F32 maxY = Blend<0xcc>(Lane8F32(centroid[1]), Lane8F32(b.maxP[1]));
        Lane8F32 maxZ = Blend<0xf0>(Lane8F32(centroid[2]), Lane8F32(b.maxP[2]));

        Lane8F32 tMinX = (minX - ray->o[0]) * invRayDx;
        Lane8F32 tMaxX = (maxX - ray->o[0]) * invRayDx;

        Lane8F32 tMinY = (minY - ray->o[1]) * invRayDy;
        Lane8F32 tMaxY = (maxY - ray->o[1]) * invRayDy;

        Lane8F32 tMinZ = (minZ - ray->o[2]) * invRayDz;
        Lane8F32 tMaxZ = (maxZ - ray->o[2]) * invRayDz;

        const Lane8F32 tEntryX = Min(tMaxX[0], tMinX[0]);
        const Lane8F32 tLeaveX = Max(tMinX[0], tMaxX[0]);

        const Lane8F32 tEntryY = Min(tMaxY[1], tMinY[1]);
        const Lane8F32 tLeaveY = Max(tMinY[1], tMaxY[1]);

        const Lane8F32 tEntryZ = Min(tMaxZ[2], tMinZ[2]);
        const Lane8F32 tLeaveZ = Max(tMinZ[2], tMaxZ[2]);

        Lane8F32 tEntry        = Max(tEntryX, Max(tEntryY, Max(tEntryZ, tMinEpsilon)));
        Lane8F32 tLeave        = Min(tLeaveX, Min(tLeaveY, Min(tLeaveZ, tMax)));
        Lane8F32 intersectMask = tEntry <= tLeave;
        u32 maskBits           = Movemask(intersectMask);

        Lane8F32 t_hgfedcba = Select(intersectMask, tEntry, pos_inf);

        // Find the indices of each node (distance sorted)
        Lane8F32 t_aaaaaaaa = Shuffle<0>(t_hgfedcba);
        Lane8F32 t_edbcbbca = ShuffleReverse<4, 3, 1, 2, 1, 1, 2, 0>(t_hgfedcba);
        Lane8F32 t_gfcfeddb = ShuffleReverse<6, 5, 2, 5, 4, 3, 3, 1>(t_hgfedcba);
        Lane8F32 t_hhhgfgeh = ShuffleReverse<7, 7, 7, 6, 5, 6, 4, 7>(t_hgfedcba);

        const u32 mask0 = Movemask(t_aaaaaaaa < t_gfcfeddb);
        const u32 mask1 = Movemask(t_edbcbbca < t_gfcfeddb);
        const u32 mask2 = Movemask(t_edbcbbca < t_hhhgfgeh);
        const u32 mask3 = Movemask(t_gfcfeddb < t_hhhgfgeh);

        const u32 mask = mask0 | (mask1 << 8) | (mask2 << 16) | (mask3 << 24);

        u32 indices[] = {
            PopCount(~mask & 0x000100ed),
            PopCount((mask ^ 0x002c2c00) & 0x002c2d00),
            PopCount((mask ^ 0x20121200) & 0x20123220),
            PopCount((mask ^ 0x06404000) & 0x06404602),
            PopCount((mask ^ 0x08808000) & 0x0a828808),
            PopCount((mask ^ 0x50000000) & 0x58085010),
            PopCount((mask ^ 0x80000000) & 0x94148080),
            PopCount(mask & 0xe0e10000),
        };

        // Add to stack
        Lane4F32 mins[]               = {b.minP, centroid};
        Lane4F32 maxs[]               = {centroid, b.maxP};
        const u32 numIntersectedNodes = PopCount(maskBits);
        for (u32 i = 0; i < 8; i++)
        {
            Lane4F32 minP(mins[i & 1][0], mins[(i & 3) >> 1][1],
                          mins[i >> 2][2], 0.f);
            Lane4F32 maxP(maxs[i & 1][0], maxs[(i & 3) >> 1][1],
                          maxs[i >> 2][2], 0.f);
            Bounds newBounds(minP, maxP);
            entries[stackPtr + (numIntersectedNodes - 1 - indices[i]) & 7] =
                StackEntry(&node->children[i], newBounds, tEntry[i], tLeave[i]);
        }
        stackPtr += numIntersectedNodes;
    }
    return false;
}

// One sample MIS estimator
__forceinline f32 MISWeight(SampledSpectrum spec, u32 channel = 0)
{
    return f32(NSampledWavelengths) / spec.Sum();
}

// weight = p(u, lambda0) / (1/m * (sum(spec0) + sum(spec1)))
__forceinline f32 MISWeight(SampledSpectrum spec0, SampledSpectrum spec1, u32 channel = 0)
{
    return f32(NSampledWavelengths) * spec0[channel] / (spec0 + spec1).Sum();
}

template <bool residualRatioTracking, typename F>
SampledSpectrum SampleTMaj(Scene2 *scene, Ray2 &ray, f32 tHit, f32 xi, Sampler sampler, const SampledWavelengths &lambda, const F &callback)
{
    tHit *= Length(ray.d);
    ray.d                      = Normalize(ray.d);
    VolumeAggregate &aggregate = scene->aggregate;
    // TODO: get this from the medium somehow
    SampledSpectrum cExtinct;

    VolumeAggregate::Iterator itr = aggregate.CreateIterator(&ray, cExtinct, tHit);
    RaySegment segment;

    bool rngInitialized = false;
    RNG rng;
    // NOTE: contains majorant transmittance (starting from the previous vertex)
    SampledSpectrum tMaj(1.f);
    while (itr.Next(segment))
    {
        bool terminated = false;

        f32 cMaj                     = segment.cMaj[0];
        SampledSpectrum cSpectrumMaj = segment.cMaj;
        if constexpr (residualRatioTracking)
        {
            cSpectrumMaj -= segment.cMin;
            cMaj -= segment.cMin[0];
        }

        f32 tMax = segment.tMax;
        f32 tMin = segment.tMin;
        if (cMaj == 0)
        {
            tMaj *= FastExp(-(Min(FLT_MAX, tMax) - tMin) * cSpectrumMaj);
            continue;
        }

        if (!rngInitialized)
        {
            rng            = RNG(Hash(sampler.Get1D()), Hash(sampler.Get1D()));
            rngInitialized = true;
        }

        for (;;)
        {
            f32 t = t - (std::log(1 - xi) / cMaj);
            xi    = rng.Uniform<f32>();
            if (t > tMax)
            {
                f32 dT = Min(FLT_MAX, tMax) - tMin;
                tMaj *= FastExp(-dT * cSpectrumMaj);
                // if constexpr (residualRatioTracking) tRay *= FastExp(-dT * segment.cMin[0]);
                break;
            }
            else
            {
                tMaj *= FastExp(-(t - tMin) * cSpectrumMaj);
                Vec3f p = ray(t);

                NanoVDBVolume &volume = scene->volumes.Get<NanoVDBVolume>()[segment.handles[0].index];
                SampledSpectrum cAbsorb, cScatter, Le;
                volume.Extinction(p, lambda, cAbsorb, cScatter, Le);
                const PhaseFunction &phase = volume.PhaseFunction();
                // TODO: build cdf over extinction coefficients for multiple volumes, select a random volume
                // how does this work? what value of the majorant do I use? do I do maxDensity * (sum of extinction for
                // all volumes), or do I do maxDensity * extinction of selected volume, does absorption
                if (!callback(rng, p, cSpectrumMaj, tMaj, cAbsorb, cScatter, Le, phase))
                {
                    terminated = true;
                    break;
                }
                tMaj = SampledSpectrum(1.f);
                tMin = t;
            }
        }
        if (terminated) return SampledSpectrum(1.f);
    }
    return tMaj;
}

bool IsValidVolume(u32 volumeIndex)
{
    return volumeIndex != invalidVolume;
}
// Manually intersect every quad in every mesh
bool Intersect(Ray2 &r, SurfaceInteraction &intr)
{
    f32 tHit      = pos_inf;
    f32 tMin      = tMinEpsilon;
    bool result   = false;
    u32 typeIndex = 0;
    u32 index     = 0;
    ForEachType(GetScene()->primitives, [&](auto *array, u32 count) {
        using Primitive = std::remove_reference_t<decltype(*array)>;
        Ray2 ray        = r;
        for (u32 i = 0; i < count; i++)
        {
            bool hit = array[i].Intersect(ray, intr, tHit);
            result |= hit;
            typeIndex = hit ? IndexOf<Primitive, Scene2::ShapeTypes>::count : typeIndex;
            index     = hit ? i : index;
        }
    });
    // If ray direction is opposite normal, we are entering the medium, otherwise we are exiting
    // else
    // {
    //     intr.volumeIndices = r.volumeIndex;
    // }
    return tHit != f32(pos_inf);
}

SampledSpectrum VolumetricIntegrator(Ray2 &ray, Sampler sampler,
                                     SampledWavelengths &lambda, u32 maxDepth)
{
    // TODO:
    // 3. multiple volumes
    // 4. virtual density segments, and other sampling methods
    //      a. equiangular sampling
    SampledSpectrum beta(1.f), L(1.f), p_l(1.f), p_u(1.f);
    SurfaceInteraction prevIntr;
    bool specularBounce = false;
    u32 depth           = 0;
    f32 etaScale        = 1.f;

    Scene2 *scene = GetScene();

    for (;;)
    {
        SurfaceInteraction intr;
        // TODO: tMin epsilon (for now)
        bool intersect = Intersect(ray, intr);

        // Volume intersection
        {
            bool scattered  = false;
            bool terminated = false;

            // RNG rng(Hash(sampler.Get1D()), Hash(sampler.Get1D()));

            SampledSpectrum tMaj = SampleTMaj<false>(
                scene, ray, f32(intr.tHit), sampler.Get1D(), sampler, lambda,
                [&](RNG &rng, Vec3f p, const SampledSpectrum &cMaj, const SampledSpectrum &tMaj,
                    const SampledSpectrum &cAbsorb, const SampledSpectrum &cScatter,
                    const SampledSpectrum &Le, const PhaseFunction &phase) {
                    if (!beta)
                    {
                        terminated = true;
                        return false;
                    }
                    // TODO: select base on throughput instead of just using the first wavelength?
                    f32 pAbsorb  = cAbsorb[0] / cMaj[0];
                    f32 pScatter = cScatter[0] / cMaj[0];
                    f32 pNull    = Max(0.f, 1 - pAbsorb - pScatter);

                    f32 xi = rng.Uniform<f32>();

                    if (depth < maxDepth && Le)
                    {
                        // probability of emission (1) * probability of sampling that point
                        f32 pdf               = cMaj[0] * tMaj[0];
                        SampledSpectrum betap = beta * tMaj / pdf;
                        SampledSpectrum p_e   = p_u * cMaj * tMaj / pdf;
                        L += betap * Le * cAbsorb * MISWeight(p_e);
                    }
                    // Emit
                    if (xi < pAbsorb)
                    {
                        terminated = true;
                        return false;
                    }
                    // Scatter
                    else if (xi < pAbsorb + pScatter)
                    {
                        if (depth++ >= maxDepth)
                        {
                            terminated = true;
                            return false;
                        }
                        // probability of being scattered * probability of sampling that point
                        f32 pdf = cScatter[0] * tMaj[0];
                        beta *= tMaj * cScatter / pdf;
                        p_u *= tMaj * cScatter / pdf;

                        // Next event estimation (for once scattered direct illumination)
                        Vec3f wi;
                        NEESample neeSample = VolumetricSampleEmitter(intr, ray, sampler, beta, p_u, lambda, wi);
                        f32 scatterPdf;
                        SampledSpectrum f = phase.EvaluateSample(-ray.d, wi, &scatterPdf);
                        neeSample.p_u *= scatterPdf;
                        L += neeSample.L_beta_tray * f *
                             MISWeight(neeSample.p_l, neeSample.delta ? SampledSpectrum(0.f) : neeSample.p_u);

                        // Generate new scatter direction for indirect illumination
                        PhaseFunctionSample phaseSample = phase.GenerateSample(-ray.d, sampler.Get2D());
                        if (phaseSample.pdf == 0)
                        {
                            terminated = true;
                            return false;
                        }
                        else
                        {
                            beta *= phaseSample.p / phaseSample.pdf;
                            p_l            = p_u / phaseSample.pdf;
                            ray.o          = p;
                            ray.d          = phaseSample.wi;
                            specularBounce = false;
                            return false;
                        }
                    }
                    // Null Scatter
                    else
                    {
                        SampledSpectrum cNull = Max(SampledSpectrum(0.f), cMaj - cAbsorb - cScatter);
                        f32 pdf               = cNull[0] * tMaj[0];
                        beta *= tMaj * cNull / pdf;
                        beta = Select(pdf, beta, SampledSpectrum(0.f));
                        p_u *= tMaj * cNull / pdf;
                        p_l *= tMaj * cMaj / pdf;
                        return beta && p_u;
                    }
                });
            if (terminated || !beta || !p_u) return L;
            if (scattered) continue;
            beta *= tMaj / tMaj[0];
            p_u *= tMaj / tMaj[0];
            p_l *= tMaj / tMaj[0];
        }

        // If ray doesn't intersect with anything, sum contribution from infinite lights and return
        if (!intersect)
        {
            // Eschew MIS when last bounce is specular or depth is zero (because either light wasn't previously sampled,
            // or it wasn't sampled with MIS)
            bool noMisFlag = specularBounce || depth == 0;
            if (specularBounce || depth == 0)
            {
                ForEachTypeSubset(
                    scene->lights, [&](auto *array, u32 count) {
                        using Light = std::remove_reference_t<decltype(*array)>;
                        for (u32 i = 0; i < count; i++)
                        {
                            Light &light       = array[i];
                            SampledSpectrum Le = Light::Le(&light, ray.d, lambda);
                            L += beta * Le * MISWeight(p_u);
                        }
                    },
                    Scene2::InfiniteLightTypes());
            }
            else
            {
                ForEachTypeSubset(
                    scene->lights, [&](auto *array, u32 count) {
                        using Light = std::remove_reference_t<decltype(*array)>;
                        for (u32 i = 0; i < count; i++)
                        {
                            Light &light       = array[i];
                            SampledSpectrum Le = Light::Le(&light, ray.d, lambda);

                            f32 pdf      = LightPDF(scene);
                            f32 lightPdf = pdf * (f32)Light::PDF_Li(&light, ray.d, true);

                            p_l *= lightPdf;
                            L += beta * Le * MISWeight(p_u, p_l);
                        }
                    },
                    Scene2::InfiniteLightTypes());
            }
            break;
            // sample infinite area lights, environment map, and return
        }

        //////////////////////////////
        // Emitter Intersection
        //
        if ((u32)intr.lightIndices)
        {
            DiffuseAreaLight *light = &scene->GetAreaLights()[u32(intr.lightIndices)];
            SampledSpectrum Le      = DiffuseAreaLight::Le(light, intr.n, -ray.d, lambda);
            if (depth == 0 || specularBounce)
            {
                L += beta * Le * MISWeight(p_u);
            }
            else
            {
                f32 pdf = LightPDF(scene);
                pdf *= (f32)DiffuseAreaLight::PDF_Li(scene, intr.lightIndices, prevIntr.p, intr, true);
                p_l *= pdf;
                L += beta * Le * MISWeight(p_u, p_l);
            }
        }

        BSDF bsdf;
        if (!intr.ComputeShading(bsdf))
        {
            // denotes boundary between medium, no event
            ray.o = intr.p;
            continue;
            // skip intersection, expand the differentials
        }
        if (depth++ >= maxDepth) return L;

        //////////////////////////////
        // Emitter Sampling
        //
        if (!IsSpecular(bsdf.Flags()))
        {
            Vec3f wi;
            NEESample neeSample = VolumetricSampleEmitter(intr, ray, sampler, beta, p_u, lambda, wi);
            f32 scatterPdf;
            SampledSpectrum f = bsdf.EvaluateSample(-ray.d, wi, scatterPdf);
            neeSample.p_u *= scatterPdf;
            L += neeSample.L_beta_tray * f * AbsDot(Vec3f(intr.shading.n), wi) *
                 MISWeight(neeSample.p_l, neeSample.delta ? SampledSpectrum(0.f) : neeSample.p_u);
        }

        //////////////////////////////
        // BSDF Sampling
        //
        BSDFSample sample = bsdf.GenerateSample(-ray.d, sampler.Get1D(), sampler.Get2D());
        if (!sample.pdf) return L;
        beta *= sample.f * AbsDot(Vec3f(intr.shading.n), sample.wi) / sample.pdf;
        p_l            = p_u / sample.pdf;
        specularBounce = IsSpecular(bsdf.Flags());
        if (sample.IsTransmissive())
        {
            etaScale *= Sqr(sample.eta);
        }
        ray.o = intr.p;
        ray.d = sample.wi;

        //////////////////////////////
        // Russian Roulette
        //
        SampledSpectrum rrBeta = etaScale * beta / (p_u + p_l).Average();
        f32 q                  = rrBeta.MaxComponentValue();
        f32 uRR                = sampler.Get1D();
        if (depth > 1 && q < 1.f)
        {
            if (uRR >= q) break;
            beta /= q;
        }
    }
    return L;
}

NEESample VolumetricSampleEmitter(const SurfaceInteraction &intr, Ray2 &ray, Sampler sampler,
                                  SampledSpectrum beta, const SampledSpectrum &p, const SampledWavelengths &lambda, Vec3f &wi)
{
    Scene2 *scene = GetScene();
    f32 lightPdf;
    LightHandle lightHandle = UniformLightSample(scene, sampler.Get1D(), &lightPdf);
    Vec2f u                 = sampler.Get2D();
    if (!lightHandle) return {};
    LightSample sample = SampleLi(scene, lightHandle, intr, lambda, u);
    if (sample.pdf == 0.f) return {};
    lightPdf *= f32(sample.pdf);
    // f32 scatterPdf;
    // SampledSpectrum f_hat;
    wi = Normalize(sample.samplePoint - intr.p);
    // if (bsdf)
    // {
    //     // f_hat = bsdf->f(wo, wi) * AbsDot(intr.shading.n, wi);
    //     f_hat = bsdf->EvaluateSample(wo, wi, &scatterPdf) * AbsDot(intr.shading.n, wi);
    //     // TODO: switch to EvaluateSample, GenerateSample interface (instead of having a separate PDF function)
    //     // for both bsdfs and phase functions
    //     // bsdf->EvaluateSample(wo, wi) * AbsDot(intr.shading.n);
    // }
    // else
    // {
    //     // Sample the phase function
    //     f_hat = bsdf->EvaluateSample(wo, wi, &scatterPdf);
    // }

    // Residual ratio tracking
    SampledSpectrum tRay(1.f), p_u(1.f), p_l(1.f);

    // RNG rng(Hash(sampler.Get1D()), Hash(sampler.Get1D()));
    SampledSpectrum tMaj = SampleTMaj<false>(
        scene, ray, f32(intr.tHit), sampler.Get1D(), sampler, lambda,
        [&](RNG &rng, Vec3f p, const SampledSpectrum &cMaj, const SampledSpectrum &tMaj,
            const SampledSpectrum &cAbsorb, const SampledSpectrum &cScatter,
            const SampledSpectrum &Le, const PhaseFunction &phase) {
            SampledSpectrum cNull = Max(SampledSpectrum(0.f), cMaj - cAbsorb - cScatter);

            // Ratio tracking code
            f32 pdf = tMaj[0] * cMaj[0];
            tRay *= tMaj * cNull / pdf;
            p_u *= tMaj * cNull / pdf;
            p_l *= tMaj * cMaj / pdf;

            // Residual ratio tracking
            // f32 rMaj = cMaj - cMnt;
            // f32 pdf  = tResidual[0] * rMaj[0];

            // Probability of sampling point along ray * probability of
            // (cMaj - cMnt) * tMnt * (1 - ((cAbsorb + cScatter - cMnt) / (cMaj - cMnt)))
            // rMaj * tMnt * (1 - (cAbsorb + cScatter - cMnt) / rMaj)
            // tMnt * (rMaj - (cAbsorb + cScatter - cMnt))
            // tMnt * (cMaj - (cAbsorb + cScatter))
            // tRay *= tResidual * cNull / pdf;
            // p_u *= tResidual * cNull / pdf;
            // p_l *= tResidual * rMaj / pdf;
            // p_l *= ;

            // Russian roulette
            if ((tRay / (p_l + p_u).Average()).MaxComponentValue() < 0.05f)
            {
                f32 q = 0.25f;
                if (rng.Uniform<f32>() < q)
                {
                    tRay = SampledSpectrum(0.f);
                }
                else
                {
                    tRay /= 1.f - q;
                }
            }
            if (!tRay) return false;
            return true;
        });

    // p_u *= p * scatterPdf;
    p_l *= p * lightPdf;
    beta *= tMaj / tMaj[0];
    p_u *= tMaj / tMaj[0];
    p_l *= tMaj / tMaj[0];

    return NEESample{beta * tRay * sample.L, p_l, p_u, IsDeltaLight(sample.lightType)};
    // if (IsDeltaLight(IsSpecular(bsdf)))
    // {
    //     return beta * t_ray * sample.L * MISWeight(p_l);
    // }
    // else
    // {
    //     return beta * t_ray * sample.L * MISWeight(p_l, p_u);
    // }
}
#endif

#if 0
void VirtualDensitySegments(const RayDifferential &ray)
{
    // TODO: things I don't understand
    // 1. how are candidate points generated? tracking without termination to the end of the segment??
    // 2. how do you choose between scattering, absoprtion, and null scattering, or do you even choose?

    // do you delta track along the segment until:
    // 1. you get absorbed, goodbye
    // 2. you scatter, find the candidate location for direct illumination + scattering location using the below
    // 3. null scatter, meaning you continue

    RNG rng;
    const u32 N = 8;

    Vec3f lightDir;

    // Pick light

    // Generate virtual density segments

    // Equiangular sampling
    f32 thetaB, thetaA, D;
    f32 tMax;

    auto EquiSampInverse = [&](f32 u) -> f32 {
        return D * Tan((1 - u) * thetaB + u * thetaA);
    };

    // Generate equal importance segments
    f32 f;
    f32 tSegment[N + 1];
    for (u32 i = 0, f = 0.f; i < N; i++, f++)
    {
                f32 u       = f / N;
                tSegment[i] = EquiSampInverse(u);
    }
    tSegment[N] = EquiSampInverse(1.f);

    f32 virtualMajorants[N];
    const f32 c = 1.f;
    for (u32 i = 0; i < N; i++)
    {
                virtualMajorant[i] = c / (tSegment[i + 1] - tSegment[i]);
    }

    u32 currentVirtualIndex = 0;
    bool done               = false;
    f32 tMin                = ray.t;
    while (!done)
    {
                // generate ray segments here
                RaySegment segment;

                f32 majorant       = Max(segment.majorant, virtualMajorants[currentVirtualIndex]);
                f32 subSegmentTMax = Min(segment.tMax, tSegment[currentVirtualIndex + 1]);
                for (;;)
                {
                    // Generate sample along current majorant segment by sampling the exponential function
                    f32 u = rng.Uniform<f32>();
                    f32 t = tMin - std::log(1 - u) / majorant;
                    // Take the max of the majorant
                    if (t < subSegmentTMax)
                    {
                    }
                    else
                    {
                        // if t is past ray segment, fetch a new one
                        if (t >= segment.tMax)
                        {
                            tMin = segment.tMax;
                            // do stuff here
                            break;
                        }
                        // if it's past only the subsegment
                        else
                        {
                            currentVirtualIndex++;
                            Assert(currentVirtualIndex < N);
                            majorant       = Max(segment.majorant, virtualMajorants[currentVirtualIndex]);
                            subSegmentTMax = Min(segment.tMax, tSegment[currentVirtualIndex + 1]);
                            continue;
                        }
                    }
                }
    }

    // I'm leaning towards that you only do this when you scatter

    // Calculate weights from candidate sample locations
    // Compute discrete CDF from weights and draw sample

    // Sample scattering direction
    Vec3f wi[N];
    f32 pmf[N];
    // Generate N directions from sampling the phase function, calculate weights by compute phase function
    for (u32 i = 0; i < N; i++)
    {
                wi[i]  = SampleHenyeyGreenstein(-ray.d, segment.g, Vec2f(rng.Uniform<f32>(), rng.Uniform<f32>()));
                pmf[i] = HenyeyGreenstein(lightDir, wi[i], segment.g);
    }

    // Get sample from discrete CDF
    f32 total = 0.f;
    f32 limit = rng.Uniform<f32>() * pmf[i + 1];
    u32 index = 0;
    for (u32 i = 0; i < N; i++)
    {
                if (total + pmf[i] >= limit)
                {
                    index = i;
                    break;
                }
                total += pmf[i];
    }

    // NOTE: beta is not updated because HenyeyGreenstein is perfectly importance sampled
}
#endif

} // namespace rt
