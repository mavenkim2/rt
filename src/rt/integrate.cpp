#include "integrate.h"
#include "lights.h"
#include "bsdf.h"
#include "scene.h"
#include <type_traits>
#include <Ptexture.h>

namespace rt
{
// TODO
// - loading volumes
// - creating objects from the parsed scene packets

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

// dreams
// - covariance tracing

static Ptex::PtexCache *cache;
struct : public PtexErrorHandler
{
    void reportError(const char *error) override { Error(0, "%s", error); }
} errorHandler;

enum class ColorEncoding
{
    Linear,
    SRGB,
};

template <i32 numChannels>
struct PtexTexture
{
    string filename;
    ColorEncoding encoding;
    f32 scale;
    PtexTexture(string filename, ColorEncoding encoding = ColorEncoding::SRGB, f32 scale = 1.f)
        : filename(filename), encoding(encoding), scale(scale) {}

    auto Evaluate(const Vec2f &uv, const Vec4f &filterWidth, u32 faceIndex)
    {
        Assert(cache);
        Ptex::String error;
        Ptex::PtexTexture *texture = cache->get((char *)filename.str, error);
        Assert(texture);
        Ptex::PtexFilter::Options opts(Ptex::PtexFilter::FilterType::f_bspline);
        Ptex::PtexFilter *filter = Ptex::PtexFilter::getFilter(texture, opts);
        i32 nc                   = texture->numChannels();
        Assert(nc == numChannels);

        // TODO: ray differentials
        // f32 filterWidth = 0.75f;
        // Vec2f uv(0.5f, 0.5f);

        f32 out[numChannels];
        filter->eval(out, 0, nc, faceIndex, uv[0], uv[1], filterWidth[0], filterWidth[1], filterWidth[2], filterWidth[3]);
        texture->release();
        filter->release();

        // Convert to srgb
        if constexpr (numChannels == 1) return out[0];

        if (encoding == ColorEncoding::SRGB)
        {
            for (i32 i = 0; i < nc; i++)
            {
                out[i] = ExactLinearToSRGB(out[i]);
            }
        }
        for (i32 i = 0; i < nc; i++)
        {
            out[i] *= scale;
        }

        Assert(numChannels == 3);
        return Vec3f(out[0], out[1], out[2]);
    }
};

static Scene2 *scene;

// template <typename Texture>
struct NormalMap
{
    template <i32 width>
    void Evaluate(SurfaceInteraction<width> &intrs)
    {
        Vec3f ns(2 * normalMap.BilerpChannel(uv, wrap), -1);
        ns = Normalize(ns);

        f32 dpduLength    = Length(dpdu);
        f32 dpdvLength    = Length(dpdv);
        dpdu              = dpdu / length;
        AffineSpace frame = AffineSpace(dpdu, Cross(ns, intrs.shading.dpdu), intrs.shading.ns);
        // Transform to world space
        ns   = TransformV(frame, ns);
        dpdu = Normalize(dpdu - Dot(dpdu, ns) * ns) * dpduLength;
        dpdv = Normalize(Cross(ns, dpdu)) * dpdvLength;
    }
};

template <typename TextureType, i32 numChannels>
struct ImageTextureShader;

template <typename TextureType>
struct ImageTextureShader<1>
{
    TextureType texture;
    ImageTextureShader() {}
    template <typename T, i32 width>
    static LaneF32<width> Evaluate(SurfaceInteraction<width> &intrs, Vec4lf<width> &filterWidths,
                                   LaneF32<width> &dfdu, LaneF32<width> &dfdv, const ImageTextureShader<TextureType, 1> **textures)
    {
        alignas(4 * width) f32 results[width];
        // Finite differencing
        LaneF32<width> du = .5f * Abs(filterWidths[0], filterWidths[2]);
        du                = Select(du == 0.f, 0.0005f, du);
        LaneF32<width> dv = .5f * Abs(filterWidths[1], filterWidths[3]);
        dv                = Select(dv == 0.f, 0.0005f, dv);

        for (u32 i = 0; i < width; i++)
        {
            Vec2f uv(intrs.uv[0][i], intrs.uv[1][i]);
            Vec4f filterWidth(filterWidths[0][i], filterWidths[1][i], filterWidths[2][i], filterWidths[3][i]);

            results[i]      = textures[i]->texture.Evaluate(uv, filterWidth, intrs.faceIndex[i]);
            results.dfdu[i] = textures[i]->texture.Evaluate(uv + Vec2f(du[i], 0.f), filterWidth, intrs.faceIndex[i]);
            results.dfdv[i] = textures[i]->texture.Evaluate(uv + Vec2f(0.f, dv[i]), filterWidth, intrs.faceIndex[i]);
        }
        return LaneF32<width>::Load(results);
    }
};

// template <typename TextureType>
// struct ImageTextureShader<3>
// {
//     ImageTextureShader() {}
//     template <typename T, i32 width>
//     static Vec3lf<width> Evaluate(SurfaceInteraction<width> &intrs, Vec4lf<width> &filterWidths,
//                                   const ImageTexture **textures)
//     {
//         Vec3lf<width> result;
//
//         for (u32 i = 0; i < width; i++)
//         {
//             Vec2f uv(intrs.uv[0][i], intrs.uv[1][i]);
//             Vec4f filterWidth(filterWidths[0][i], filterWidths[1][i], filterWidths[2][i], filterWidths[3][i]);
//
//             Vec3f r     = textures[i]->Evaluate(uv, filterWidth, intrs.faceIndex[i]);
//             result.x[i] = r.x;
//             result.y[i] = r.y;
//             result.z[i] = r.z;
//         }
//         return result;
//     }
// };

template <typename TextureShader>
struct BumpMap
{
    // p' = p + d * n, d is displacement, estimate shading normal by computing dp'du and dp'dv (using chain rule)
    TextureShader displacementShader;
    template <i32 width>
    static void Evaluate(SurfaceInteraction<width> &intrs, const BumpMap<TextureShader> **bumpMaps)
    {
        TextureShader *displacementShaders[width];
        for (u32 i = 0; i < width; i++)
        {
            displacementShaders[i] = &bumpMaps[i]->displacementShader;
        }

        LaneF32<width> dddu, dddv;
        LaneF32<width> displacement = TextureShader::Evaluate(intrs, dpdu, dpdv, displacementShaders);

        Vec3lf<width> dpdu = intrs.shading.dpdu + dddu * intrs.shading.n + displacement * intrs.shading.dndu;
        Vec3lf<width> dpdv = intrs.shading.dpdv + dddv * intrs.shading.n + displacement * intrs.shading.dndv;

        intrs.shading.n    = Cross(dpdu, dpdv);
        intrs.shading.dpdu = dpdu;
        intrs.shading.dpdv = dpdv;
    }
};

template <typename TextureShader, typename NormalShader>
struct DiffuseMaterial
{
    Texture reflectanceShader;

    template <i32 width>
    static DiffuseBSDF<IntN> GetBSDF(SurfaceInteractions<width> &intr)
    {
        DiffuseMaterial *materials = scene->materials.Get<DiffuseMaterial>();
        // TODO: vectorized texture evaluation?
        // TODO: sampled spectrum vectorized
        Vec4<LaneF32<width>> sampledSpectra;

        Lane4F32 sampledSpectrumArray[width];
        // for (u32 i = 0; i < width; i++)
        // {
        //     materials[i].texture->Evaluate(intr.faceIndices[i], sampledSpectrumArray[i].f);
        // }
        // Convert RGB to SRGB
        Vec4<LaneF32<width>> reflectance = rShaderGraph.Evaluate(intr);
        return DiffuseBSDF(reflectance);
    }

    DiffuseBSDF GetBSDF()
    {
        Vec3 result;
        texture->SampleTexture(0, result);
    }
};

template <typename BSDFShader, typename NormalShader>
struct Material
{
    BSDFShader bsdfShader;
    NormalShader normalShader;
    template <i32 width>
    static auto Evaluate(SurfaceInteractions<width> &intr)
    {
        auto bsdf = bsdfShader.GetBSDF(intr);
        NormalShader *normalShaders[width];
        for (u32 i = 0; i < width; i++)
        {
            // TODO: get index from id
            normalShaders[i] = scene->materials.Get<Material>()[intrs.materialIDs[i]].normalShader;
        }
        NormalShader::Evaluate(intrs, normalShaders);
        return bsdf;
    }
};

void InitializePtex()
{
    u32 maxFiles  = 100;
    size_t maxMem = gigabytes(4);
    cache         = Ptex::PtexCache::create(maxFiles, maxMem, true, 0, &errorHandler);
}

// TODO: one for each type of material
template <typename Material, i32 length = 512>
struct ShadingQueuePtex
{
    static const u32 queueFlushSize = length / 2;
    SurfaceInteraction queue[length];
    u32 count;
    void Flush() // SortKey *keys, SurfaceInteraction *values, u32 num)
    {
        TempArena temp = ScratchStart(0, 0);
        // SortKey *keys0 = PushArrayNoZero(temp.arena, SortKey, queueFlushSize);
        // SortKey *keys1 = PushArrayNoZero(temp.arena, SortKey, queueFlushSize);
        SortKey keys0[queueFlushSize];
        SortKey keys1[queueFlushSize];

        count -= queueFlushSize;
        // Create radix sort keys
        for (u32 i = 0; i < queueFlushSize; i++)
        {
            SurfaceInteraction &intr = queue[count + i];
            keys0[i].value           = intr.GenerateKey();
            keys0[i].index           = count + i;
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
        for (u32 i = 0, aosoaIndex = 0; i < queueFlushSize; i += IntN, aosoaIndex++)
        {
            const u32 prefetchDistance = IntN * 2;
            alignas(32) SurfaceInteraction *intrs[IntN];
            for (u32 j = 0; j < IntN; j++)
            {
                _mm_prefetch((char *)&queue[keys[i + prefetchDistance + j].index], _MM_HINT_T0);
                intrs[j] = queue[keys0[i + j].index];
            }
            SurfaceInteractionsN aosoaIntrs; //&out = aosoaIntrs[aosoaIndex];
                                             //
                                             // Transpose p, n, uv
            Transpose(intrs, aosoaIntrs);
            // Transpose8x8(Lane8F32::Load(&intrs[0]), Lane8F32::Load(&intrs[1]), Lane8F32::Load(&intrs[2]), Lane8F32::Load(&intrs[3]),
            //              Lane8F32::Load(&intrs[4]), Lane8F32::Load(&intrs[5]), Lane8F32::Load(&intrs[6]), Lane8F32::Load(&intrs[7]),
            //              out.p.x, out.p.y, out.p.z, out.n.x, out.n.y, out.n.z, out.uv.x, out.uv.y);

            // Transpose the rest
            // ...
            Material::BSDF bsdf = Material::Evaluate(aosoaIntrs);

            if constexpr (bsdf::IsSpecular)
            {
            }
            else
            {
                // Push to next event estimation queue
                RayStateHandle rayStateHandles[IntN];
                lightSampleQueue.Push(rayStateHandles, intrs);

                LaneIF32 pdf;
                Vec3IF32 wi;
                // TODO: how do I get wi?
                NEESample neeSample = Material::BSDF::EvaluateSample(-ray.d, wi, scatterPdf, TransportMode::Radiance);
                // occlusion ray queue
            }

            // TODO: things I need to simd:
            // - sampler
            // - ray
            // - path throughput weight
            // - path flags
            // - path eta scale for russian roulette
            Sampler *samplers[IntN];
            Ray2 *rays[IntN];
            LaneF32<IntN> u;
            Vec2lf<IntN> uv;

            BSDFSample<IntN> sample = bsdf.GenerateSample(-ray.d, u, uv, TransportMode ::Radiance, BSDFFlags::RT);
            MaskF32<IntN> mask;
            mask = Select(sample.pdf == 0.f, 1, 0);
            beta *= sample.f * AbsDot(intr.shading.n, sample.wi) / sample.pdf;

            // TODO: path flags, set specular bounce to true
            // pathFlags &= bsdf.IsSpecular();

            // Spawn a new ray, push to the ray queue
        }

        // Return bsdf lobes

        // Material calculation
        ScratchEnd(temp);
    }
    // mesh id, face index
};

template <>
bool SurfaceInteraction::ComputeShading(Scene2 *scene, BSDF &bsdf)
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

#if 0
void Li(Scene2 *scene, RayDifferential &ray, Sampler &sampler, u32 maxDepth, SampledWavelengths &lambda)
{
    u32 depth = 0;
    SampledSpectrum L(0.f);
    SampledSpectrum beta(1.f);

    bool specularBounce = false;
    f32 bsdfPdf         = 1.f;
    f32 etaScale        = 1.f;

    SurfaceInteraction prevSi;
    u32 prevLightIndex;

    for (;;)
    {
        if (depth >= maxDepth)
        {
            break;
        }
        SurfaceInteraction si;
        bool intersect = scene->Intersect(ray, si);

        // If no intersection, sample "infinite" lights (e.g environment maps, sun, etc.)
        if (!intersect)
        {
            // Eschew MIS when last bounce is specular or depth is zero (because either light wasn't previously sampled,
            // or it wasn't sampled with MIS)
            if (specularBounce || depth == 0)
            {
                for (u32 i = 0; i < scene->numInfiniteLights; i++)
                {
                    InfiniteLight *light = &scene->infiniteLights[i];
                    SampledSpectrum Le   = light->Le(ray.d);
                    L += beta * Le;
                }
            }
            else
            {
                for (u32 i = 0; i < scene->numInfiniteLights; i++)
                {
                    InfiniteLight *light = &scene->infiniteLights[i];
                    SampledSpectrum Le   = light->Le(ray.d);
                    // probability of sampling the light * probability of
                    // lightSampler->PMF(prevSi, light) *
                    f32 pmf      = 1.f / scene->numLights;
                    f32 lightPdf = pmf *
                                   light->PDF_Li(scene, prevSi.lightIndices, prevSi.p, ray.d); // find the pmf for the light
                    f32 w_l = PowerHeuristic(1, bsdfPdf, 1, lightPdf);
                    // NOTE: beta already contains the cosine, bsdf, and pdf terms
                    L += beta * w_l * Le;
                }
            }
            break;
            // sample infinite area lights, environment map, and return
        }
        // If intersected with a light
        if (si.lightIndices)
        {
            DiffuseAreaLight *light = &scene->lights[si.lightIndices];
            if (specularBounce || depth == 0)
            {
                SampledSpectrum Le = light->Le(si.n, -ray.d, lambda);
                L += beta * Le;
            }
            else
            {
                Assert(0);
                SampledSpectrum Le = light->Le(si.n, -ray.d, lambda);
                // probability of sampling the light * probability of sampling point on light
                f32 pmf      = 1.f / scene->numLights;
                f32 lightPdf = pmf *
                               light->PDF_Li(scene, prevSi.lightIndices, prevSi.p, si);
                f32 w_l = PowerHeuristic(1, bsdfPdf, 1, lightPdf);
                // NOTE: beta already contains the cosine, bsdf, and pdf terms
                L += beta * w_l * Le;
            }
        }

        BSDF *bsdf = si.GetBSDF();

        // Next Event Estimation
        // TODO: offset ray origin, don't sample lights if bsdf is specular

        // Choose light source for direct lighting calculation
        f32 lightU     = sampler.Get1D();
        u32 lightIndex = u32(Min(lightU * scene->numLights, scene->numLights - 1));
        Light *light   = &scene->lights[lightIndex];
        f32 pmf        = 1.f / scene->numLights;
        if (light)
        {
            Vec2f sample = sampler.Get2D();
            // Sample point on the light source
            LightSample ls = SampleLi(scene, lightIndex, si, sample);
            if (ls)
            {
                // Evaluate BSDF for light sample, check visibility with shadow ray
                SampledSpectrum Ld(0.f);
                SampledSpectrum f = bsdf->f(-ray.d, wo) * AbsDot(si.shading.n, wi);
                if (f && !scene->IntersectShadowRay())
                {
                    // Calculate contribution
                    f32 lightPdf = pmf * ls.pdf;

                    if (IsDeltaLight(light->type))
                    {
                        Ld = beta * f * ls.L / lightPdf;
                    }
                    else
                    {
                        f32 bsdfPdf = bsdf->PDF(wo, wi);
                        f32 w_l     = PowerHeuristic(1, lightPdf, 1, bsdfPdf);
                        Ld          = beta * f * w_l * ls.L / lightPdf;
                    }
                }
            }
        }

        // sample bsdf, calculate pdf
        beta *= bsdf->f * AbsDot(shading->n, bsdf->wi) / pdf;
        if (bsdf->IsSpecular()) specularBounce = true;

        // Spawn new ray
        prevSi = si;

        // Russian Roulette
        SampledSpectrum rrBeta = beta * etaScale;
        f32 q                  = MaxComponentValue(rrBeta);
        if (depth > 1 && q < 1.f)
        {
            if (sampler.Get1D() < Max(0.f, 1 - q)) break;

            beta /= q;
            // TODO: infinity check for beta
        }
    }
}
#endif

//////////////////////////////
// Volumes
//

void VolumeAggregate::Build(Arena *arena, Scene2 *scene)
{
    const f32 T = -1.f / std::log(0.5f);
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

struct NEESample
{
    SampledSpectrum L_beta_tray;
    SampledSpectrum p_l;
    SampledSpectrum p_u;
    bool delta;
};

NEESample VolumetricSampleEmitter(const SurfaceInteraction &intr, Ray2 &ray, Scene2 *scene, Sampler sampler,
                                  SampledSpectrum beta, const SampledSpectrum &p, const SampledWavelengths &lambda, Vec3f &wi);

f32 VisibleWavelengthsPDF(f32 lambda)
{
    if (lambda < LambdaMin || lambda > LambdaMax)
    {
        return 0;
    }
    return 0.0039398042f / Sqr(std::cosh(0.0072f * (lambda - 538)));
}

f32 SampleVisibleWavelengths(f32 u)
{
    return 538 - 138.888889f * std::atanh(0.85691062f - 1.82750197f * u);
}

// Importance sampling the
static SampledWavelengths SampleVisible(f32 u)
{
    SampledWavelengths swl;
    for (i32 i = 0; i < NSampledWavelengths; i++)
    {
        f32 up = u + f32(i) / NSampledWavelengths;
        if (up > 1) up -= 1;
        swl.lambda[i] = SampleVisibleWavelengths(up);
        swl.pdf[i]    = VisibleWavelengthsPDF(swl.lambda[i]);
    }
    return swl;
}

bool IsValidVolume(u32 volumeIndex) { return volumeIndex != invalidVolume; }
// Manually intersect every quad in every mesh
bool Intersect(Scene2 *scene, Ray2 &r, SurfaceInteraction &intr)
{
    f32 tHit      = pos_inf;
    f32 tMin      = tMinEpsilon;
    bool result   = false;
    u32 typeIndex = 0;
    u32 index     = 0;
    ForEachType(scene->primitives, [&](auto *array, u32 count) {
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

SampledSpectrum VolumetricIntegrator(Scene2 *scene, Ray2 &ray, Sampler sampler,
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

    for (;;)
    {
        SurfaceInteraction intr;
        // TODO: tMin epsilon (for now)
        bool intersect = Intersect(scene, ray, intr);

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
                        NEESample neeSample = VolumetricSampleEmitter(intr, ray, scene, sampler, beta, p_u, lambda, wi);
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
        if (!intr.ComputeShading(scene, bsdf))
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
            NEESample neeSample = VolumetricSampleEmitter(intr, ray, scene, sampler, beta, p_u, lambda, wi);
            f32 scatterPdf;
            SampledSpectrum f = bsdf.EvaluateSample(-ray.d, wi, scatterPdf, TransportMode::Radiance);
            neeSample.p_u *= scatterPdf;
            L += neeSample.L_beta_tray * f * AbsDot(Vec3f(intr.shading.n), wi) *
                 MISWeight(neeSample.p_l, neeSample.delta ? SampledSpectrum(0.f) : neeSample.p_u);
        }

        //////////////////////////////
        // BSDF Sampling
        //
        BSDFSample sample = bsdf.GenerateSample(-ray.d, sampler.Get1D(), sampler.Get2D(), TransportMode::Radiance, BSDFFlags::RT);
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

NEESample VolumetricSampleEmitter(const SurfaceInteraction &intr, Ray2 &ray, Scene2 *scene, Sampler sampler,
                                  SampledSpectrum beta, const SampledSpectrum &p, const SampledWavelengths &lambda, Vec3f &wi)
{
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
