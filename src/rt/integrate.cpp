#include "integrate.h"
#include "bxdf.h"
#include "lights.h"
#include "bsdf.h"
#include "math/basemath.h"
#include "scene.h"
#include "spectrum.h"
#include <cstring>
#include <type_traits>

namespace rt
{
// TODO
// BUGS:
// - shading normals seem wrong for some elements, check subdivision code
// - when the build process receives garbage data, it produces a bvh that
// causes the intersection to infinite loop?
// - rare deadlock in bvh construction code
//
// - area lights w/ light tree traversal, adaptive splitting, and accounting for BSDF
// - https://fpsunflower.github.io/ckulla/data/many-lights-hpg2018.pdf
// - remove duplicate materials (and maybe geometry), see if this leads to coherent texture
// reads
// - curves
// - simd queues for everything (radiance evaluation, shading, ray streams?)
// - optimize bvh memory consumption?

// - bdpt, metropolis, vcm, upbp, mcm?
// - volumetric rendering
//      - ratio tracking, residual ratio tracking, delta tracking <-- done but untested
//      - virtual density segments?
// - disney bsdf
// - multiple scattering
// - adaptive sampling + path splitting
// - equiangular sampling
// - manifold next event estimation
// - wave optics :D
// - cache points
//
// - memory mapped files for treelets??? (i.e. single level massive bvh)
// - covariance tracing
// - path guiding
// - non exponential free flight
// - photon planes & volumes

// - what is zero variance theory? half space light transport? dwivedi sampling?
// - crazy idea: render this on the gpu (investigate how nanite handles the cpu-gpu
// intercommunication to retrieve the right LODs)

//////////////////////////////
// Textures and materials
//
void InitializePtex()
{
    u32 maxFiles  = 100;
    size_t maxMem = gigabytes(4);
    // cache         = Ptex::PtexCache::create(maxFiles, maxMem, true, 0, &errorHandler);
    cache = Ptex::PtexCache::create(maxFiles, maxMem, true, &ptexInputHandler, &errorHandler);
    // cache = Ptex::PtexCache::create(maxFiles, maxMem, true, &handler2, &errorHandler);
}

typedef u32 PathFlags;
enum
{
    PathFlags_SpecularBounce,
};

void DefocusBlur(const Vec3f &dIn, const Vec2f &pLens, const f32 focalLength, Vec3f &o,
                 Vec3f &d)
{
    f32 t        = focalLength / -dIn.z;
    Vec3f pFocus = dIn * t;
    o            = Vec3f(pLens.x, pLens.y, 0.f);
    d            = Normalize(pFocus - o);
}

Vec3f IntersectRayPlane(const Vec3f &planeN, const Vec3f &planeP, const Vec3f &rayP,
                        const Vec3f &rayD)
{
    f32 d   = Dot(planeN, planeP);
    f32 t   = (d - Dot(planeN, rayP)) / Dot(planeN, rayD);
    Vec3f p = rayP + t * rayD;
    return p;
}

template <typename Sampler>
SampledSpectrum Li(Ray2 &ray, Camera &camera, Sampler &sampler, u32 maxDepth,
                   SampledWavelengths &lambda);

void GenerateMinimumDifferentials(Camera &camera, RenderParams2 &params, u32 width, u32 height,
                                  u32 taskCount, u32 tileCountX, u32 tileWidth, u32 tileHeight,
                                  u32 pixelWidth, u32 pixelHeight)
{
    ParallelReduce(
        &camera.diff, 0, taskCount, 1,
        [&](CameraDifferentials &camDiffs, u32 jobID, u32 start, u32 count) {
            Vec3f &minPosX = camDiffs.minPosX;
            Vec3f &minPosY = camDiffs.minPosY;
            Vec3f &minDirX = camDiffs.minDirX;
            Vec3f &minDirY = camDiffs.minDirY;

            f32 minPosXLength = pos_inf;
            f32 minPosYLength = pos_inf;
            f32 minDirXLength = pos_inf;
            f32 minDirYLength = pos_inf;

            auto FindMin = [&](const Vec3f &check, const Vec3f &origin, f32 &length,
                               Vec3f &minV) {
                f32 diff = Length(check - origin);
                if (diff < length)
                {
                    length = diff;
                    minV   = check - origin;
                }
            };

            for (u32 i = start; i < start + count; i++)
            {
                u32 tileX = i % tileCountX;
                u32 tileY = i / tileCountX;
                Vec2u minPixelBounds(params.pixelMin[0] + tileWidth * tileX,
                                     params.pixelMin[1] + tileHeight * tileY);
                Vec2u maxPixelBounds(Min(params.pixelMin[0] + tileWidth * (tileX + 1),
                                         params.pixelMin[0] + pixelWidth),
                                     Min(params.pixelMin[1] + tileHeight * (tileY + 1),
                                         params.pixelMin[1] + pixelHeight));

                Assert(maxPixelBounds.x >= minPixelBounds.x && minPixelBounds.x >= 0 &&
                       maxPixelBounds.x <= width);
                Assert(maxPixelBounds.y >= minPixelBounds.y && minPixelBounds.y >= 0 &&
                       maxPixelBounds.y <= height);

                for (u32 y = minPixelBounds.y; y < maxPixelBounds.y; y++)
                {
                    for (u32 x = minPixelBounds.x; x < maxPixelBounds.x; x++)
                    {
                        Vec2f pFilm((f32)x, (f32)y);
                        Vec2f pLens(.5f);
                        Ray2 ray = camera.GenerateRayDifferentials(pFilm, pLens);

                        FindMin(ray.pxOffset, ray.o, minPosXLength, minPosX);
                        FindMin(ray.pyOffset, ray.o, minPosYLength, minPosY);
                        FindMin(ray.dxOffset, ray.d, minDirXLength, minDirX);
                        FindMin(ray.dyOffset, ray.d, minDirYLength, minDirY);
                    }
                }
            }
        },
        [&](CameraDifferentials &l, const CameraDifferentials &r) { l.Merge(r); });
}

Vec3f ConvertRadianceToRGB(const SampledSpectrum &Lin, const SampledWavelengths &lambda,
                           u32 maxComponentValue = 10)
{
    SampledSpectrum L = SafeDiv(Lin, lambda.PDF());
    f32 r             = (Spectra::X().Sample(lambda) * L).Average();
    f32 g             = (Spectra::Y().Sample(lambda) * L).Average();
    f32 b             = (Spectra::Z().Sample(lambda) * L).Average();
    f32 m             = Max(r, Max(g, b));
    Vec3f sampleRgb   = Vec3f(r, g, b);
    if (m > maxComponentValue)
    {
        sampleRgb *= maxComponentValue / m;
    }
    return sampleRgb;
}

void Render(Arena *arena, RenderParams2 &params)
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
    Vec3f org      = TransformP(cameraFromRaster, Vec3f(0.f, 0.f, 0.f));
    Vec3f dxCamera = TransformP(cameraFromRaster, Vec3f(1.f, 0.f, 0.f)) - org;
    Vec3f dyCamera = TransformP(cameraFromRaster, Vec3f(0.f, 1.f, 0.f)) - org;

    Camera camera(cameraFromRaster, renderFromCamera, dxCamera, dyCamera, focalLength,
                  lensRadius, spp);

    GenerateMinimumDifferentials(camera, params, width, height, taskCount, tileCountX,
                                 tileWidth, tileHeight, pixelWidth, pixelHeight);

    scheduler.ScheduleAndWait(taskCount, 1, [&](u32 jobID) {
        u32 tileX = jobID % tileCountX;
        u32 tileY = jobID / tileCountX;
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

                GetDebug()->pixel = pPixel;
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

                    // Vec3f temp = TransformP(cameraFromRaster, Vec3f(0.25f, 0.25f, 0.f));
                    // ray.radius = 0.f;
                    // ray.spread = Max(temp[0], temp[1]);
                    // printf("%f spread\n", ray.spread);
                    f32 cameraWeight = 1.f;

                    SampledSpectrum L =
                        cameraWeight * Li(ray, camera, sampler, maxDepth, lambda);

                    rgb += ConvertRadianceToRGB(L, lambda);
                    // convert radiance to rgb, add and divide
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
                f32 m = Max(r, Max(g, b));

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
    WriteImage(&image, "image.bmp");
    printf("done\n");
}

f32 PowerHeuristic(u32 numA, f32 pdfA, u32 numB, f32 pdfB)
{
    f32 a = Sqr(numA * pdfA);
    f32 b = Sqr(numB * pdfB);
    return a / (a + b);
}

Vec3f OffsetRayOrigin(const Vec3f &p, const Vec3f &err, const Vec3f &n, const Vec3f &wi)
{
    f32 d        = Dot(err, Abs(n));
    d            = Select(Dot(wi, n) < 0, -d, d);
    Vec3f offset = n * d;
    Vec3f outP   = p + offset;

    for (int i = 0; i < 3; ++i)
    {
        if (offset[i] > 0) outP[i] = NextFloatUp(outP[i]);
        else if (offset[i] < 0) outP[i] = NextFloatDown(outP[i]);
    }
    return outP;
}

bool Occluded(Scene *scene, Ray2 &ray)
{
    Assert(scene->scene.occludedFunc);
    return scene->scene.occludedFunc(&scene->scene, scene->scene.nodePtr, ray);
}
bool Occluded(Scene *scene, Ray2 &r, SurfaceInteraction &si, LightSample &ls)
{
    const f32 shadowRayEpsilon = 1 - .0001f;

    Vec3f from = OffsetRayOrigin(si.p, si.pError, si.n, ls.wi);
    // Vec3f to                   = OffsetRayOrigin(ls.p, ls.rayOrigin);
    Ray2 ray(from, ls.samplePoint - from, shadowRayEpsilon);
    // return BVH4QuadCLIntersectorCmp8::Occluded(ray, nodePtr);
    return Occluded(scene, ray);
}

bool Intersect(Scene *scene, Ray2 &ray, SurfaceInteraction &si)
{
    Assert(scene->scene.intersectFunc);
    return scene->scene.intersectFunc(&scene->scene, scene->scene.nodePtr, ray, si);
}

void CalculateFilterWidths(const Ray2 &ray, const Camera &camera, const Vec3f &p,
                           const Vec3f &n, const Vec3f &dpdu, const Vec3f &dpdv, Vec3f &dpdx,
                           Vec3f &dpdy, f32 &dudx, f32 &dvdx, f32 &dudy, f32 &dvdy)
{
    if (ray.pxOffset != Vec3f(pos_inf))
    {
        Vec3f px = IntersectRayPlane(n, p, ray.pxOffset, ray.dxOffset);
        Vec3f py = IntersectRayPlane(n, p, ray.pyOffset, ray.dyOffset);

        dpdx = px - p;
        dpdy = py - p;
    }
    else
    {
        // Estimate ray differentials from camera
        Vec3f px =
            IntersectRayPlane(n, p, ray.o + camera.diff.minPosX, ray.d + camera.diff.minDirX);
        Vec3f py =
            IntersectRayPlane(n, p, ray.o + camera.diff.minPosY, ray.d + camera.diff.minDirY);

        dpdx = camera.sppScale * (px - p);
        dpdy = camera.sppScale * (py - p);
    }

    // Solve overdetermined linear system
    // dpdx = dpdu * dudx
    // dpdx = dpdv * dvdx
    // Solve using linear least squares
    // (At * A)^-1 * At * b = x
    f32 ata00  = Dot(dpdu, dpdu);
    f32 ata01  = Dot(dpdu, dpdv);
    f32 ata11  = Dot(dpdv, dpdv);
    f32 det    = FMS(ata00, ata11, Sqr(ata01));
    f32 invDet = det == 0.f ? 0.f : 1.f / det;

    f32 atb0x = Dot(dpdu, dpdx);
    f32 atb1x = Dot(dpdv, dpdx);
    f32 atb0y = Dot(dpdu, dpdy);
    f32 atb1y = Dot(dpdv, dpdy);

    // dudx = invDet * (ata11 * atb0x - atb1x * ata01);
    dudx = Clamp(invDet * FMS(ata11, atb0x, ata01 * atb1x), -1e8f, 1e8f);
    dvdx = Clamp(invDet * FMS(ata00, atb1x, ata01 * atb0x), -1e8f, 1e8f);

    // dvdx =invDet * (-ata01 * atb0x + ata00 * atb1x)
    dudy = Clamp(invDet * FMS(ata11, atb0y, ata01 * atb1y), -1e8f, 1e8f);
    dvdy = Clamp(invDet * FMS(ata00, atb1y, ata01 * atb0y), -1e8f, 1e8f);

    // printf("dpdx: %f %f %f, dpdy: %f %f %f, dudx: %f, dvdx: %f, dudy: %f, dvdy: %f\n",
    // dpdx.x,
    //        dpdx.y, dpdx.z, dpdy.x, dpdy.y, dpdy.z, dudx, dvdx, dudy, dvdy);
}

__forceinline void UpdateRayDifferentials(Ray2 &ray, const Vec3f &wi, const Vec3f &p, Vec3f n,
                                          const Vec3f &dndu, const Vec3f &dndv,
                                          const Vec3f &dpdx, const Vec3f &dpdy, const f32 dudx,
                                          const f32 dvdx, const f32 dudy, const f32 dvdy,
                                          f32 eta, u32 flags)
{
    if (ray.pxOffset != Vec3f(pos_inf))
    {
        Vec3f dndx = dndu * dudx + dndv * dvdx;
        Vec3f dndy = dndu * dudy + dndv * dvdy;

        Vec3f wo = -ray.d;

        Vec3f dwodx = -ray.dxOffset - wo;
        Vec3f dwody = -ray.dyOffset - wo;

        if (flags == (u32)BxDFFlags::SpecularReflection)
        {
            // Initialize origins of specular differential rays
            ray.pxOffset = p + dpdx;
            ray.pyOffset = p + dpdy;

            // Compute differential reflected directions
            f32 dwoDotn_dx = Dot(dwodx, n) + Dot(wo, dndx);
            f32 dwoDotn_dy = Dot(dwody, n) + Dot(wo, dndy);
            ray.dxOffset   = wi - dwodx + 2 * Vec3f(Dot(wo, n) * dndx + dwoDotn_dx * n);
            ray.dyOffset   = wi - dwody + 2 * Vec3f(Dot(wo, n) * dndy + dwoDotn_dy * n);
        }
        else if (flags == (u32)BxDFFlags::SpecularTransmission)
        {
            // Initialize origins of specular differential rays
            ray.pxOffset = p + dpdx;
            ray.pyOffset = p + dpdy;

            // Compute differential transmitted directions
            // Find oriented surface normal for transmission
            if (Dot(wo, n) < 0)
            {
                n    = -n;
                dndx = -dndx;
                dndy = -dndy;
            }

            // Compute partial derivatives of $\mu$
            f32 dwoDotn_dx = Dot(dwodx, n) + Dot(wo, dndx);
            f32 dwoDotn_dy = Dot(dwody, n) + Dot(wo, dndy);
            f32 mu         = Dot(wo, n) / eta - AbsDot(wi, n);
            f32 dmudx      = dwoDotn_dx * (1 / eta + 1 / Sqr(eta) * Dot(wo, n) / Dot(wi, n));
            f32 dmudy      = dwoDotn_dy * (1 / eta + 1 / Sqr(eta) * Dot(wo, n) / Dot(wi, n));

            ray.dxOffset = wi - eta * dwodx + mu * dndx + dmudx * n;
            ray.dyOffset = wi - eta * dwody + mu * dndy + dmudy * n;
        }
        else
        {
            ray.pxOffset = Vec3f(pos_inf);
        }
        // Squash potentially troublesome differentials
        if (LengthSquared(ray.pxOffset) > 1e16f || LengthSquared(ray.pyOffset) > 1e16f ||
            LengthSquared(ray.dxOffset) > 1e16f || LengthSquared(ray.dyOffset) > 1e16f)
            ray.pxOffset = Vec3f(pos_inf);
    }
}

template <typename Sampler>
SampledSpectrum Li(Ray2 &ray, Camera &camera, Sampler &sampler, u32 maxDepth,
                   SampledWavelengths &lambda)
{
    Scene *scene = GetScene();
    u32 depth    = 0;
    SampledSpectrum L(0.f);
    SampledSpectrum beta(1.f);

    bool specularBounce = false;
    f32 bsdfPdf         = 0.f;
    f32 etaScale        = 1.f;

    SurfaceInteraction prevSi;

    for (;;)
    {
        SurfaceInteraction si;
        // TODO IMPORTANT: need to robustly prevent self intersections. right now at grazing
        // angles the reflected ray is re-intersecting with the ocean, which causes the ray to
        // effectively transmit when it should reflect

        // bool intersect = BVH4QuadCLIntersectorCmp8::Intersect(ray, scene->nodePtr, si);
        bool intersect = Intersect(scene, ray, si);
        // BVH4TriangleCLIntersectorCmp1::Intersect(ray, scene->nodePtr, si);

        // If no intersection, sample "infinite" lights (e.g environment maps, sun, etc.)
        if (!intersect)
        {
            // Eschew MIS when last bounce is specular or depth is zero (because either light
            // wasn't previously sampled, or it wasn't sampled with MIS)
            if (specularBounce || depth == 0)
            {
                for (auto &light : scene->infiniteLights)
                {
                    SampledSpectrum Le = light->Le(ray.d, lambda);
                    L += beta * Le;
                }
            }
            else
            {
                for (auto &light : scene->infiniteLights)
                {
                    SampledSpectrum Le = light->Le(ray.d, lambda);

                    f32 pdf      = LightPDF(scene);
                    f32 lightPdf = pdf * (f32)light->PDF_Li(ray.d, true);

                    f32 w_l = PowerHeuristic(1, bsdfPdf, 1, lightPdf);
                    // NOTE: beta already contains the cosine, bsdf, and pdf terms
                    L += beta * w_l * Le;
                }
            }

            break;
        }

        // If intersected with a light
        if (si.lightIndices)
        {
            Assert(0);
            // DiffuseAreaLight *light =
            //     &scene->GetAreaLights()[LightHandle(si.lightIndices).GetIndex()];
            // if (specularBounce || depth == 0)
            // {
            //     SampledSpectrum Le = DiffuseAreaLight::Le(light, si.n, -ray.d, lambda);
            //     L += beta * Le;
            // }
            // else
            // {
            //     SampledSpectrum Le = DiffuseAreaLight::Le(light, si.n, -ray.d, lambda);
            //     // probability of sampling the light * probability of sampling point on
            //     light f32 pmf = LightPDF(scene); f32 lightPdf =
            //         pmf * DiffuseAreaLight::PDF_Li(scene, si.lightIndices, prevSi.p, si,
            //         true);
            //     f32 w_l = PowerHeuristic(1, bsdfPdf, 1, lightPdf);
            //     // NOTE: beta already contains the cosine, bsdf, and pdf terms
            //     L += beta * w_l * Le;
            // }
        }

        if (depth++ >= maxDepth)
        {
            break;
        }

        ScratchArena scratch;

        f32 dudx, dudy, dvdx, dvdy = 0.f;
        Vec3f dpdx, dpdy;
        CalculateFilterWidths(ray, camera, si.p, si.n, si.dpdu, si.dpdv, dpdx, dpdy, dudx,
                              dvdx, dudy, dvdy);
        // f32 newRadius = ray.radius + ray.spread * si.tHit;

        // Calculate differential of surface parameterization w.r.t image plane (i.e.
        // dudx, dudy, dvdx, dvdy)
        if (MaterialHandle(si.materialIDs).GetType() == MaterialTypes::Interface)
        {
            specularBounce = true;
            ray.o          = OffsetRayOrigin(si.p, si.pError, si.n, ray.d);
            ray.tFar       = pos_inf;
            if (ray.pxOffset != Vec3f(pos_inf))
            {
                ray.pxOffset = ray.pxOffset + si.tHit * ray.dxOffset;
                ray.pyOffset = ray.pyOffset + si.tHit * ray.dyOffset;
            }
            continue;
        }

        Material *material = scene->materials[MaterialHandle(si.materialIDs).GetIndex()];
        BxDF bxdf =
            material->Evaluate(scratch.temp.arena, si, lambda, Vec4f(dudx, dvdx, dudy, dvdy));
        BSDF bsdf(bxdf, si.shading.dpdu, si.shading.n);

        // Next Event Estimation
        // Choose light source for direct lighting calculation
        if (!IsSpecular(bsdf.Flags()))
        {
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
                    SampledSpectrum f =
                        bsdf.EvaluateSample(-ray.d, ls.wi, p_b) * AbsDot(si.shading.n, ls.wi);
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
        if (sample.pdf == 0.f) break;
        // beta *= sample.f / sample.pdf;
        beta *= sample.f * AbsDot(si.shading.n, sample.wi) / sample.pdf;
        bsdfPdf        = sample.pdf;
        specularBounce = sample.IsSpecular();
        if (sample.IsTransmissive()) etaScale *= Sqr(sample.eta);

        // Spawn new ray
        prevSi = si;
        // ray.o  = si.p;

        // Update spread
        // {
        //     BxDFFlags flags = (BxDFFlags)sample.flags;
        //     f32 newSpread;
        //
        //     f32 pdfSpread = .125f / Sqrt(sample.pdf);
        //     if (EnumHasAnyFlags(flags, BxDFFlags::Reflection))
        //     {
        //         Assert(sample.pdf > 0.f);
        //         f32 spread = 2.f * si.curvature * ray.radius;
        //         newSpread  = Max(pdfSpread, spread);
        //     }
        //     else if (EnumHasAnyFlags(flags, BxDFFlags::Transmission))
        //     {
        //         f32 spread =
        //             sample.eta * ray.spread - (1.f - sample.eta) * si.curvature *
        //             ray.radius;
        //         newSpread = Max(pdfSpread, ray.spread);
        //     }
        //
        //     ray.radius = newRadius;
        //     ray.spread = Clamp(newSpread, -1.f, 1.f);
        // }

        // Offset ray along geometric normal + compute specular ray differentials
        {
            ray.o = OffsetRayOrigin(si.p, si.pError, si.n, sample.wi);

            ray.d    = sample.wi;
            ray.tFar = pos_inf;

            // Compute ray differentials for specular reflection or transmission
            // Compute common factors for specular ray differentials
            UpdateRayDifferentials(ray, sample.wi, si.p, si.shading.n, si.shading.dndu,
                                   si.shading.dndv, dpdx, dpdy, dudx, dvdx, dudy, dvdy,
                                   sample.eta, sample.flags);

            // if there is a non-specular interaction, revert to using
        }

        // Russian Roulette
        SampledSpectrum rrBeta = beta * etaScale;
        f32 q                  = rrBeta.MaxComponentValue();
        if (depth > 1 && q < 1.f)
        {
            if (sampler.Get1D() < Max(0.f, 1 - q)) break;

            beta /= q;
            // TODO: infinity check for beta
        }
    }
    return L;
}

#if 0
struct KDTreeNode
{
    union
    {
        struct
        {
            int axis;
            f32 split;
        };
        struct
        {
            int start;
            int count;
        };
    };

    KDTreeNode *left;
    KDTreeNode *right;

    bool IsLeaf() const { return left == 0; }
};

// double buffer
void BuildKDTreeHelper(Arena *arena, Vec3f *cachePoints, Vec3f *cachePointsBuffer, int start,
                       int num)
{
    Bounds geomBounds;
    // Calculate the maximum extent to split by
    // TODO: parallel for this when num is sufficiently large
    for (int i = start; i < start + num; i++)
    {
        geomBounds.Extend(Lane4F32(cachePoints[i]));
    }

    // Calculate the maximum extent
    int axis      = -1;
    f32 maxExtent = neg_inf;
    for (int i = 0; i < 3; i++)
    {
        f32 extent = geomBounds.maxP[i] - geomBounds.minP[i];
        if (extent > maxExtent)
        {
            maxExtent = extent;
            axis      = i;
        }
    }

    // TODO: calculate a splitting plane that evenly subdivides the points? vs
    // just splitting in the middle
    f32 splitPlane = (geomBounds.maxP[axis] + geomBounds.minP[axis]) / 2.f;

    KDTreeNode *node = PushStruct(arena, KDTreeNode);
    node->split      = splitPlane;

    // TODO: parallel for
    int numLeft  = 0;
    int numRight = 0;
    for (int i = start; i < start + num; i++)
    {
        if (cachePoints[i][axis] >= splitPlane) numRight++;
        else numLeft++;
    }
    int leftOffset  = start;
    int rightOffset = start + numLeft;
    for (int i = start; i < start + num; i++)
    {
        int choice = cachePoints[i][axis] >= splitPlane;
        cachePointsBuffer[choice ? rightOffset++ : leftOffset++] = cachePoints[i];
    }
}

void BuildKDTree(Arena **arenas, const Vec3f *cachePoints) {}

struct KNN
{
    int index;
    f32 distanceSqr;
};

template <i32 kMax>
void FindKNearestNeighbors(KDTreeNode *node, const Vec3f &candidatePoint, int index, KNN *kNN,
                           int &k, const int *cachePointIndexList, const Vec3f *cachePoints,
                           const f32 &maxMinDistance)
{
    // Remove 20 kNN
    const f32 searchRadius = 1e-5;

    int stackPtr     = 1;
    KDTreeNode *node = stack[--stackPtr];
    if (node->IsLeaf())
    {
        for (int i = node->start; i < node->start + node->count; i++)
        {
            int index          = cachePointIndexList[i];
            const Vec3f &point = cachePoints[index];
            f32 distance       = LengthSquared(point - candidatePoint);
            if (distance < maxMinDistance)
            {
                int replaceIndex = 0;
                // Maintain sorted list of 20 closest neighbors within search radius
                for (; replaceIndex < k; replaceIndex++)
                {
                    if (distance < kNN[replaceIndex].distanceSquared)
                    {
                        // Shift to the right
                        memmove(kNN + replaceIndex + 1, kNN + replaceIndex,
                                sizeof(KNN) * (k - replaceIndex - 1));
                        break;
                    }
                }
                kNN[replaceIndex] = {index, disance};
                maxMinDistance    = kNN[k].distance;
                k += k < kMax;
            }
        }
    }
    else
    {
        int choice = p[node->axis] >= node->split;
        FindKNearestNeighbors(choice ? node->right : node->left);
        // Backtrack through the tree
        if (Abs(p[node->axis] - node->split) < maxMinDistance)
        {
            FindKNearestNeighbors(choice ? node->left : node->right);
        }
    }
}

struct LightDistribution
{
    struct LightEstimate
    {
        Light *light;
        f32 pdf;
    };

    std::vector<Light *> nearbyLights;
    std::vector<LightEstimate> farLights[6];
    f32 radius;

    void MergeFarDistributions(LightDistribution &ld, f32 weight = 1.f)
    {
        for (int i = 0; i < 6; i++)
        {
            farLights[i].reserve(farLights[i].size() + ld.farLights[i].size());
            for (auto lightEst : ld.farLights[i])
            {
                lightEst.pdf *= weight;
                farLights[i].push_back(lightEst);
            }
        }
    }
};

// TODO: atomically mark for deletion all except the lowest index
template <typename F>
void MarkForDeletion(KNN *knn, int numNeighbors, std::atomic<int> *flags, F func)
{
    int minIndex = (int)pos_inf;
    for (int i = 0; i < numNeighbors; i++)
    {
        if (func(knn[i])) minIndex = Min(knn[i].index);
    }
    for (int i = 0; i < numNeighbors; i++)
    {
        if (func(knn) && knn[i].index != minIndex)
        {
            flags[knn[i].index].store(1, std::memory_order_relaxed);
        }
    }
}

// idea for later: do this at runtime
// 1. when you hit an object for the first time, that's when you distribute points
// within the bounding box of that object
// 2. weighted reservoir sampling with a russian roulette termination based on the ratio
// between an additional light's probability and the total running weight. does this
// even make any sense?
// 3.

struct CachePoints
{
    StaticArray<KDTreeNode> kdTreeNodes;
};

void ConstructCachePoints(Arena **arenas, const Camera &camera,
                          LightDistribution *distributions, int seed = 0)
{
    // Step 1: Generate 100,000 candidate points randomly distributed within the
    // individual bounding boxes

    const int totalNumCandidatePoints = 100000;

    // TODO: get list of bounds of all objects
    Bounds *bounds = 0;

    const f64 totalVolume = 0.f;

    int numCandidatePoints = totalNumCandidatePoints * bounds->Volume() / totalVolume;

    // TODO: make sure the object we're looking at is one
    // where nee is actually computed (i.e. no area lights, no null material objects)
    RNG rng(seed);
    for (int i = 0; i < numCandidatePoints; i++)
    {
        // Generate random points within the bounding box
        f32 x = rng.Uniform<f32>();
        f32 y = rng.Uniform<f32>();
        f32 z = rng.Uniform<f32>();

        Vec3f p;
        p.x = Lerp(x, bounds.minP[0], bounds.maxP[0]);
        p.y = Lerp(y, bounds.minP[1], bounds.maxP[1]);
        p.z = Lerp(z, bounds.minP[2], bounds.maxP[2]);

        // TODO: trace pilot paths?
    }

    // Step 2: Trace pilot paths


    // Step 2: Build KD Tree over points
    Arena *kdTreeArena = ArenaAlloc();
    StaticArray<KDTreeNode> kdTreeNodes(kdTreeArena, totalNumCandidiatePoints);
    kdTreeNodes.Resize(totalNumCandidatePoints);

    BuildKDTree();

    // How do I estimate the contribution of a light to a point? what even is
    // fluence?
    // fluence is only used for curves/volume points where surface normal isn't defined.
    // otherwise, use irradiance at the point

    // TODO: handle radiance varying over the surface of the
    // like sample a point on the light source, get the radiance and cosines and

    Scene *scene = GetScene();

    // TODO: for the similarity metric, how do you calaculate the probability for
    // "near" lights? for far lights, it's obvious that you use the irradiance estimates.
    // but
    const f32 D = 4.f;

    for (int i = 0; i < numCachePoints; i++)
    {
        LightDistribution &distribution = distributions[i];
        const Vec3f &point              = cachePoints[i];

        const int k = 20;
        KNN kNN[k];
        FindKNearestNeighbors<k>(root, point, ?, kNN, etc.);

        // Calculate radius as average of the distance to the k nearest neighbors
        f32 radius = 0.f;
        for (int i = 0; i < k; i++)
        {
            radius += kNN[i].distance;
        }
        radius /= f32(k);

        distribution.radius = radius;

        // Calculate importance of all lights in scene

        static const int farLightBinMax = 256;
        FarLight farLights[6][farLightBinMax];
        int farLightCount[6] = {};
        f32 importance[6]    = {};
        f32 minContribution[6];
        for (int cardinalIndex = 0; cardinalIndex < 6; cardinalIndex++)
        {
            minContribution[cardinalIndex] = pos_inf;
        }

        for (int lightIndex = 0; lightIndex < scene->numLights; lightIndex++)
        {
            Light *light = scene->lights[lightIndex];
            bool isNear  = LengthSquared(light->Centroid() - point) < radius * D;
            if (isNear)
            {
                // girard's theorem for the solid angle?
                distribution.nearbyLights.push_back(light);
            }
            else
            {
                Vec3f cardinalNormals[] = {
                    Vec3f(1, 0, 0),  Vec3f(-1, 0, 0), Vec3f(0, 1, 0),
                    Vec3f(0, -1, 0), Vec3f(0, 0, 1),  Vec3f(0, 0, -1),
                };
                // TODO: we're sampling visible wavelengths here since we're
                // using spectral, maybe there's a better way?
                // Compute irradiance estimate for six cardinal bins

                SurfaceInteraction intr;
                intr.p = point;
                // Store list accounting for 97% of the energy reaching the point
                for (int cardinalIndex = 0; cardinalIndex < 6; cardinalIndex++)
                {
                    intr.n             = cardinalNormals[cardinalIndex];
                    LightSample sample = light->SampleLi();
                    f32 estimate       = sample.L.MaxValue() *
                                   AbsDot(sample.samplePoint, cardinalNormals[cardinalIndex]) /
                                   sample.pdf;

                    // Find "far lights" with greatest contribution to current bin
                    auto &farLights = distribution.farLights[cardinalIndex];
                    if (farLightCount[cardinalIndex] < farLightBinMax)
                    {
                        farLights[cardinalIndex][farLightCount[cardinalIndex]++] = {light,
                                                                                    estimate};
                        minContribution[cardinalIndex] =
                            Min(minContribution[cardinalIndex], estimate);
                    }
                    else if (estimate > minContribution[cardinalIndex])
                    {
                        f32 newMin = pos_inf;
                        for (int cullLightIndex = 0; cullLightIndex < farLightBinMax;
                             cullLightIndex++)
                        {
                            newMin = Min(newMin, farLights[cardinalIndex][cullLightIndex]);
                            if (farLights[cardinalIndex][cullLightIndex].estimate ==
                                minContribution[cardinalIndex])
                            {
                                farLights[cardinalIndex][cullLightIndex] = {light, estimate};
                            }
                        }
                    }
                }
            }
        }
    }

    // Calculate similarity metric between point and its neighbors
    for (int i = 0; i < numCachePoints; i++)
    {
        const int k = 20;
        KNN kNN[k];
        int numNeighbors = FindKNearestNeighbors<k>(root, point, ?, kNN, etc.);
        // TODO: should I also consider the nearby lights???
        LightDistribution &distribution = distributions[i];

        int sizeA = distribution.nearbyLightCount;
        for (int cardinalIndex = 0; cardinalIndex < 6; cardinalIndex++)
        {
            sizeA += farLightCounts[cardinalIndex];
        }

        f32 M = 0.f;

        // TODO: maybe should consider the total probability of a light in all bins
        for (int neighborIndex = 0; neighborIndex < k; neighborIndex++)
        {
            const KNN &neighbor                     = kNN[neighborIndex];
            LightDistribution &neighborDistribution = distributions[neighbor.index];
            int sizeB                               = neighborDistribution.nearbyLightCount;

            f32 S = 0.f;
            for (int cardinalIndex = 0; cardinalIndex < 6; cardinalIndex++)
            {
                sizeB += neighborDistribution.farLightCounts[cardinalIndex];

                for (int farLightIndex = 0;
                     farLightIndex < distribution.farLightCounts[cardinalIndex];
                     farLightIndex++)
                {
                    const FarLight &light =
                        distribution.farLights[cardinalIndex][farLightIndex];
                    for (int neighborLightIndex = 0;
                         neighborLightIndex <
                         neighborDistribution.farLightCounts[cardinalIndex];
                         neighborLightIndex++)
                    {
                        const FarLight &neighborLight =
                            neighborDistribution.farLights[cardinalIndex][neighborLightIndex];
                        if (light.light == neighborLight.light)
                        {
                            S += 1 - Abs(light.pdf - neighborLight.pdf) /
                                         (light.pdf + neighborLight.pdf);
                        }
                    }
                }
            }
            M += 2 * S / (sizeA + sizeB);
        }
        M /= f32(k);
        // TODO: minimum projected screen space size of 3 pixels
        distribution.radius = Max(distribution.radius * M, .25f * distribution.radius);

        MarkForDeletion(kNN, numNeighbors, flags, [&](const KNN &neighbor) {
            return neighbor.distance < distribution.radius;
        });
    }

    int rootIndex = BuildKDTree();

    // Last step: blur light distributions across cache points
    ParallelFor([&](int jobID, int start, int count) {
        for (int i = start; i < start + count; i++)
        {
            static const int k = 16;

            LightDistribution &distribution = distributions[i];
            KNN knn[k];
            int numNeighbors = FindKNearestNeighbors<k>(
                nodes, rootIndex, const int &candidatePoint, int index, KNN *kNN, int &k,
                const int *cachePointIndexList, const int *cachePoints,
                const f32 &maxMinDistance);

            f32 maxDistance = neg_inf;
            f32 minDistance = pos_inf;
            for (int kIndex = 0; kIndex < k; kIndex++)
            {
                maxDistance = Max(maxDistance, knn[kIndex].distanceSquared);
                minDistance = Min(minDistance, knn[kIndex].distanceSquared);
            }
            maxDistance = Sqrt(maxDistance);

            f32 weights[17];
            weights[0] = 1.f;

            f32 total = 1.f;
            for (int kIndex = 0; kIndex < k; kIndex++)
            {
                if (knn[kIndex].distanceSquared == minDistance)
                {
                    weights[kIndex + 1] = 1.f;
                    total += 1.f;
                }
                else if (knn[kIndex].distanceSquared == maxDistance)
                {
                    weights[kIndex + 1] = 1.f / 16.f;
                    total += 1.f / 16.f;
                }
                else
                {
                    f32 weight =
                        Lerp(1.f, 1.f / 16.f, knn[kIndex].distanceSquared / maxDistance);
                    weights[kIndex + 1] = weight;
                    total += weight;
                }
            }
            total                 = 1.f / total;
            int farLightTotals[6] = {};
            int nearCount         = 0;

            for (int kIndex = 0; kIndex < ArrayLength(weights); kIndex++)
            {
                weights[kIndex] *= total;
                const LightDistribution &neighborDistribution =
                    distributions[knn[kIndex].index];
                nearCount += neighborDistribution.nearbyLightCount;
                for (int farIndex = 0; farIndex < 6; farIndex++)
                {
                    farLightTotals[farIndex] += neighborDistribution.farLightCounts[farIndex];
                }
            }

            // Merge far distributions
            LightDistribution &newDistribution = newDistributions[i];
            newDistribution.nearbyLights       = distribution.nearbyLights;
            newDistribution.radius             = distribution.radius;
            newDistribution.nearbyLightCount   = nearCount;

            newDistribution.SetFarLightData(arena, farLightTotals);

            newDistribution.MergeFarDistributions(distribution, weights[0]);
            for (int kIndex = 0; kIndex < numNeighbors; kIndex++)
            {
                newDistribution.MergeFarDistributions(distributions[knn[kIndex].index],
                                                      weights[kIndex + 1]);
            }
        }
    });
    distributions.clear();
}
#endif

#if 0
template <typename Sampler>
void ManifoldNextEventEstimation(Sampler &sampler, u32 currentDepth, u32 maxDepth,
                                 SampledWavelengths &lambda)
{
    struct ManifoldSurfaceInteraction
    {
        SurfaceInteraction si;
    };
    const u32 MAX_MANIFOLD_OCCLUDERS = 2;
    ManifoldSurfaceInteraction msiData[MAX_MANIFOLD_OCCLUDERS];
    u32 msiDataCount = 0;

    auto helper =
        [&]() {
            f32 theta = Acos(w.z);
            f32 phi   = Atan2(w.y, w.x);
        }

    Vec3f rayOrigin;
    if (!IsSpecular(bsdf.Flags()))
    {
        // Sample light source
        f32 lightU = sampler.Get1D();
        f32 pmf;
        LightHandle handle = UniformLightSample(scene, lightU, &pmf);
        Vec2f sample       = sampler.Get2D();
        if (!bool(handle)) return;
        // Sample point on the light source
        LightSample ls = SampleLi(scene, handle, si, lambda, sample, true);
        if (!ls.pdf) return;

        Assert(maxDepth >= currentDepth);
        u32 numBounces = maxDepth - currentDepth;
        if (numBounces == 0)
        {
            // normal NEE
        }
        else
        {
        }

        // Generate seed path
        for (;;)
        {
            ManifoldSurfaceInteraction &msi = msiData[msiDataCount];
            bool intersect                  = Intersect(scene, ray, msi.si);
            if (!intersect)
            {
                break;
            }
            else
            {
                ScratchArena scratch;
                BSDF bsdf;
                EvaluateMaterial(scratch.temp.arena, si, &bsdf, lambda);
                if (!EnumHasAnyFlags(bsdf.Flags(), BxDFFlags::SpecularTransmission)) return;

                Vec3f wo = -ray.d;
                f32 pdf;
                BSDFSample sample = bsdf.GenerateSample(
                    -ray.d, sampler.Get1D(), sampler.Get2D(), TransportMode::Radiance,
                    BxDFFlags::SpecularTransmission);
                if (!sample.f || !sample.pdf) return;

                ray.d = sample.wi;
                msiDataCount++;
            }
        }

        // Handle paths
        for (u32 index = 0; index < msiDataCount; index++)
        {
            Vec3f p0 = index == 0 ? rayOrigin : msiData[index - 1].si.p;
            Vec3f p2 = index + 1 == msiDataCount ? ls.samplePoint : msiData[index + 1].si.p;
            ManifoldSurfaceInteraction &currentHit = msiData[index];
            // TODO: change differential geometry?
            Vec3f wo     = p0 - currentHit.si.p;
            Vec3f wi     = p2 - currentHit.si.p;
            bool outside = Dot(wo, currentHit.si.shading.n) > 0.f;
            if (!outside) Swap(wo, wi);

            // Manifold walk
            RefractPartialDerivatives(wo);
                // Constraint solving until under a threshold between these computed partial derivatives and 
        }

        // Evaluate BSDF for light sample, check visibility with shadow ray
        f32 p_b;
        SampledSpectrum f =
            bsdf.EvaluateSample(-ray.d, ls.wi, p_b) * AbsDot(si.shading.n, ls.wi);
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
#endif

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
                    InfiniteLightTypes());
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
                    InfiniteLightTypes());
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
