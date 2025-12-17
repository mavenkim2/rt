#include "rt.h"

#include <algorithm>
#include "memory.h"
#include "containers.h"
#include "debug.h"
#include "thread_context.h"

#include "hash.h"
#include "color.h"
#include "sampling.h"
#include "parallel.h"
#include "integrate.h"
#include "graphics/vulkan.h"
#include "graphics/ptex.h"
#include "tests/test.h"
#include "image.h"
#include "../../third_party/streamline/include/sl.h"

namespace rt
{

inline f32 DegreesToRadians(f32 degrees) { return degrees * PI / 180.f; }

//////////////////////////////
// Intervals
//
bool IsInInterval(f32 min, f32 max, f32 x) { return x >= min && x <= max; }

static Vec3f BACKGROUND;

bool IsValidRay(Ray *r) { return r->t != (f32)U32Max; }

#if 0
// TODO: optimize this by multiplying matrix by extents: mul(mat, maxX - minX)
AABB Transform(const Mat4 &mat, const AABB &aabb)
{
    AABB result;
    Vec3f vecs[] = {
        Mul(mat, Vec3f(aabb.minX, aabb.minY, aabb.minZ)),
        Mul(mat, Vec3f(aabb.maxX, aabb.minY, aabb.minZ)),
        Mul(mat, Vec3f(aabb.maxX, aabb.maxY, aabb.minZ)),
        Mul(mat, Vec3f(aabb.minX, aabb.maxY, aabb.minZ)),
        Mul(mat, Vec3f(aabb.minX, aabb.minY, aabb.maxZ)),
        Mul(mat, Vec3f(aabb.maxX, aabb.minY, aabb.maxZ)),
        Mul(mat, Vec3f(aabb.maxX, aabb.maxY, aabb.maxZ)),
        Mul(mat, Vec3f(aabb.minX, aabb.maxY, aabb.maxZ)),
    };

    for (u32 i = 0; i < ArrayLength(vecs); i++)
    {
        Vec3f &p    = vecs[i];
        result.minX = result.minX < p.x ? result.minX : p.x;
        result.minY = result.minY < p.y ? result.minY : p.y;
        result.minZ = result.minZ < p.z ? result.minZ : p.z;

        result.maxX = result.maxX > p.x ? result.maxX : p.x;
        result.maxY = result.maxY > p.y ? result.maxY : p.y;
        result.maxZ = result.maxZ > p.z ? result.maxZ : p.z;
    }
    return result;
}

Light CreateQuadLight(Quad *quad)
{
    Light light;
    light.type      = PrimitiveType_Quad;
    light.primitive = quad;
    return light;
}

Light CreateSphereLight(Sphere *sphere)
{
    Light light;
    light.type      = PrimitiveType_Sphere;
    light.primitive = sphere;
    return light;
}

Vec3f Sample(const Light *light, const Vec3f &origin, Vec2f u)
{
    switch (light->type)
    {
        case PrimitiveType_Quad:
        {
            Quad *quad = (Quad *)light->primitive;
            return quad->Random(origin, u);
        }
        case PrimitiveType_Sphere:
        {
            Sphere *sphere = (Sphere *)light->primitive;
            return sphere->Random(origin, u);
        }
        default: Assert(0); return Vec3f(1, 0, 0);
    }
}

// Vec3f GenerateSampleFromLights(const Light *lights, const u32 numLights, const Vec3f &origin, f32 u)
// NOTE: bad uniform sampling
const Light *SampleLights(const Light *lights, const u32 numLights, f32 u)
{
    i32 randomIndex = (i32)(numLights * u); // RandomInt(0, numLights);
    Assert(randomIndex < (i32)numLights);
    return &lights[randomIndex]; // Vec3f result = GenerateLightSample(&lights[randomIndex], origin, );
}

f32 GetLightPDFValue(const Light *light, const Vec3f &origin, const Vec3f &direction)
{
    switch (light->type)
    {
        case PrimitiveType_Quad:
        {
            Quad *quad = (Quad *)light->primitive;
            return quad->PdfValue(origin, direction);
        }
        case PrimitiveType_Sphere:
        {
            Sphere *sphere = (Sphere *)light->primitive;
            return sphere->PdfValue(origin, direction);
        }
        default: Assert(0); return 0;
    }
}

f32 GetLightsPDFValue(const Light *lights, const u32 numLights, const Vec3f &origin, const Vec3f &direction)
{
    f32 pdfSum = 0.f;
    for (u32 i = 0; i < numLights; i++)
    {
        pdfSum += GetLightPDFValue(&lights[i], origin, direction);
    }
    pdfSum /= numLights;
    return pdfSum;
}

inline Vec3f GenerateCosSample(Vec3f normal, Vec2f u)
{
    Basis basis  = GenerateBasis(normal);
    Vec3f cosDir = ConvertToLocal(&basis, SampleCosineHemisphere(u)); // RandomCosineDirection());
    return cosDir;
}

inline f32 GetCosPDFValue(Vec3f dir, Vec3f normal)
{
    f32 cosTheta = Dot(Normalize(dir), normal);
    f32 cosPdf   = fmax(cosTheta / PI, 0.f);
    return cosPdf;
}

struct Perlin
{
    // f32 *randFloat;
    Vec3f *randVec;
    i32 *permX;
    i32 *permY;
    i32 *permZ;
    static const i32 pointCount = 256;

    void Init()
    {
        randVec = new Vec3f[pointCount];
        for (i32 i = 0; i < pointCount; i++)
        {
            randVec[i] = Normalize(RandomVec3(-1, 1));
        }

        auto GeneratePerm = [&]() -> i32 * {
            i32 *perm = new i32[pointCount];
            for (i32 i = 0; i < pointCount; i++)
            {
                perm[i] = i;
            }

            for (i32 i = pointCount - 1; i > 0; i--)
            {
                i32 target   = RandomInt(0, i);
                i32 temp     = perm[i];
                perm[i]      = perm[target];
                perm[target] = temp;
            }
            return perm;
        };

        permX = GeneratePerm();
        permY = GeneratePerm();
        permZ = GeneratePerm();
    }

    f32 Noise(const Vec3f &p) const
    {
        f32 u = p.x - floor(p.x);
        f32 v = p.y - floor(p.y);
        f32 w = p.z - floor(p.z);

        Vec3f c[2][2][2];
        {
            i32 i = i32(floor(p.x));
            i32 j = i32(floor(p.y));
            i32 k = i32(floor(p.z));

            for (i32 di = 0; di < 2; di++)
            {
                for (i32 dj = 0; dj < 2; dj++)
                {
                    for (i32 dk = 0; dk < 2; dk++)
                    {
                        c[di][dj][dk] = randVec[permX[(i + di) & 255] ^ permY[(j + dj) & 255] ^ permZ[(k + dk) & 255]];
                    }
                }
            }
        }

        f32 accum = 0.0;
        {
            f32 uu = u * u * (3 - 2 * u);
            f32 vv = v * v * (3 - 2 * v);
            f32 ww = w * w * (3 - 2 * w);
            for (i32 i = 0; i < 2; i++)
            {
                for (i32 j = 0; j < 2; j++)
                {
                    for (i32 k = 0; k < 2; k++)
                    {
                        Vec3f weightV(u - i, v - j, w - k);
                        accum += (i * uu + (1 - i) * (1 - uu)) *
                                 (j * vv + (1 - j) * (1 - vv)) *
                                 (k * ww + (1 - k) * (1 - ww)) *
                                 Dot(c[i][j][k], weightV);
                    }
                }
            }
        }
        return accum;
    }

    f32 Turbulence(Vec3f p, i32 depth) const
    {
        f32 accum  = 0.0;
        f32 weight = 1.0;
        for (i32 i = 0; i < depth; i++)
        {
            accum += weight * Noise(p);
            weight *= 0.5f;
            p *= 2.f;
        }

        return fabs(accum);
    }
};

struct Texture
{
    enum class Type
    {
        Solid,
        Checkered,
        Image,
        Noise
    } type;

    static Texture CreateSolid(const Vec3f &albedo)
    {
        Texture texture;
        texture.baseColor = albedo;
        texture.type      = Type::Solid;
        return texture;
    }
    static Texture CreateCheckered(f32 scale, const Vec3f &even, const Vec3f &odd)
    {
        Texture texture;
        texture.baseColor  = even;
        texture.baseColor2 = odd;
        texture.type       = Type::Checkered;
        texture.invScale   = 1.f / scale;
        return texture;
    }

    static Texture CreateImage(const char *filename)
    {
        Texture texture;
        texture.image = LoadFile(filename);
        texture.type  = Type::Image;
        return texture;
    }

    static Texture CreateNoise(f32 scale)
    {
        Texture texture;
        texture.perlin.Init();
        texture.type  = Type::Noise;
        texture.scale = scale;
        return texture;
    }

    Vec3f Value(const f32 u, const f32 v, const Vec3f &p) const
    {
        switch (type)
        {
            case Type::Solid:
            {
                return baseColor;
            }
            break;
            case Type::Checkered:
            {
                i32 x = i32(std::floor(p.x * invScale));
                i32 y = i32(std::floor(p.y * invScale));
                i32 z = i32(std::floor(p.z * invScale));
                return (x + y + z) % 2 == 0 ? baseColor : baseColor2;
            }
            break;
            case Type::Image:
            {
                Assert(image.width);
                Assert(image.height);
                i32 x = i32(u * image.width);
                i32 y = i32((1 - v) * image.height);

                u8 *data    = GetColor(&image, x, y);
                f32 divisor = 1 / 255.f;
                f32 r       = f32(data[0]) * divisor;
                f32 g       = f32(data[1]) * divisor;
                f32 b       = f32(data[2]) * divisor;
                return Vec3f(r, g, b);
            }
            break;
            case Type::Noise:
            {
                return Vec3f(.5f, .5f, .5f) * (1.f + sinf(scale * p.z + 10.f * perlin.Turbulence(p, 7)));
            }
            break;
            default: Assert(0); return Vec3f(0, 0, 0);
        }
    }

    Vec3f baseColor;

    // checkered
    Vec3f baseColor2;
    f32 invScale;

    // image
    Image image;

    // perlin
    Perlin perlin;
    f32 scale;
};

enum class MaterialType
{
    Lambert,
    Metal,
    Dielectric,
    DiffuseLight,
    Isotropic,
};

struct Material
{
    MaterialType type;
    Vec3f albedo;
    f32 fuzz;
    f32 refractiveIndex;

    Texture texture;

    static Material CreateLambert(Vec3f inAlbedo)
    {
        Material result;
        result.type    = MaterialType::Lambert;
        result.texture = Texture::CreateSolid(inAlbedo);
        // result.albedo = inAlbedo;
        return result;
    }

    static Material CreateLambert(Texture *texture)
    {
        Material result;
        result.type    = MaterialType::Lambert;
        result.texture = *texture;
        return result;
    }

    static Material CreateMetal(Vec3f inAlbedo, f32 inFuzz = 0.0)
    {
        Material result;
        result.type   = MaterialType::Metal;
        result.albedo = inAlbedo;
        result.fuzz   = inFuzz < 1 ? inFuzz : 1;
        return result;
    }

    static Material CreateDielectric(f32 inRefractiveIndex)
    {
        Material result;
        result.type            = MaterialType::Dielectric;
        result.refractiveIndex = inRefractiveIndex;
        return result;
    }

    static Material CreateDiffuseLight(Texture *texture)
    {
        Material result;
        result.type    = MaterialType::DiffuseLight;
        result.texture = *texture;
        return result;
    }

    static Material CreateDiffuseLight(Vec3f inAlbedo)
    {
        Material result;
        result.type    = MaterialType::DiffuseLight;
        result.texture = Texture::CreateSolid(inAlbedo);
        return result;
    }

    static Material CreateIsotropic(const Vec3f &albedo)
    {
        Material result;
        result.type    = MaterialType::Isotropic;
        result.texture = Texture::CreateSolid(albedo);
        return result;
    }

    static Material CreateIsotropic(Texture *texture)
    {
        Material result;
        result.type    = MaterialType::Isotropic;
        result.texture = *texture;
        return result;
    }

    bool LambertScatter(const Ray &r, const HitRecord &record, ScatterRecord &sRec, Vec2f u)
    {
        sRec.attenuation = texture.Value(record.u, record.v, record.p);
        sRec.skipPDFRay  = Ray(INVALID_VEC, INVALID_VEC, (f32)U32Max);
        sRec.sample      = GenerateCosSample(record.normal, u);
        return true;
    }

    bool MetalScatter(const Ray &r, const HitRecord &record, ScatterRecord &sRec, Vec2f u)
    {
        Vec3f reflectDir = Reflect(r.d, record.normal);
        reflectDir       = Normalize(reflectDir) + fuzz * RandomUnitVector(u);
        sRec.attenuation = albedo;
        sRec.skipPDFRay  = Ray(record.p, reflectDir, r.t);
        return true;
    }

    bool DielectricScatter(const Ray &r, const HitRecord &record, ScatterRecord &sRec, Vec2f u)
    {
        sRec.attenuation = Vec3f(1, 1, 1);
        f32 ri           = record.isFrontFace ? 1.f / refractiveIndex : refractiveIndex;

        Vec3f rayDir = Normalize(r.d);
        f32 cosTheta = Min(Dot(-rayDir, record.normal), 1.f);
        f32 sinTheta = Sqrt(1 - cosTheta * cosTheta);
        // total internal reflection
        bool cannotRefract = ri * sinTheta > 1.f;

        f32 f0          = (1 - ri) / (1 + ri);
        f0              = f0 * f0;
        f32 reflectance = f0 + (1 - f0) * powf(1 - cosTheta, 5.f);
        Vec3f direction = cannotRefract || reflectance > u.x // RandomFloat()
                              ? Reflect(rayDir, record.normal)
                              : Refract(rayDir, record.normal, ri);
        sRec.skipPDFRay = Ray(record.p, direction, r.t);

        return true;
    }

    bool IsotropicScatter(const Ray &r, const HitRecord &record, ScatterRecord &sRec, Vec2f u)
    {
        // scatteredRay     = Ray(record.p, RandomUnitVector(), r.t);
        sRec.attenuation = texture.Value(record.u, record.v, record.p);
        sRec.skipPDFRay  = Ray(INVALID_VEC, INVALID_VEC, (f32)U32Max);
        sRec.sample      = RandomUnitVector(u);
        return true;
    }

    inline bool Scatter(const Ray &r, const HitRecord &record, ScatterRecord &sRec, Vec2f u)
    {
        switch (type)
        {
            case MaterialType::Lambert: return LambertScatter(r, record, sRec, u);
            case MaterialType::Metal: return MetalScatter(r, record, sRec, u);
            case MaterialType::Dielectric: return DielectricScatter(r, record, sRec, u);
            case MaterialType::Isotropic: return IsotropicScatter(r, record, sRec, u);
            default: return false;
        }
    }

    inline Vec3f Emitted(const Ray &r, const HitRecord &record, f32 u, f32 v, const Vec3f &p) const
    {
        switch (type)
        {
            case MaterialType::DiffuseLight:
            {
                if (!record.isFrontFace)
                {
                    return Vec3f(0, 0, 0);
                }
                return texture.Value(u, v, p);
            }
            break;
            default: return Vec3f(0, 0, 0);
        }
    }

    inline f32 ScatteringPDF(const Ray &r, const HitRecord &record, Ray &scattered)
    {
        switch (type)
        {
            case MaterialType::Lambert:
            {
                f32 cosTheta = Dot(record.normal, Normalize(scattered.d));
                cosTheta     = fmax(cosTheta / PI, 0.f);
                return cosTheta;
            }
            case MaterialType::Isotropic:
            {
                return (1 / (4 * PI));
            }
            default: return 0;
        }
    }
};

#ifndef EMISSIVE
Vec3f RayColor(const Ray &r, const int depth, const BVH &bvh)
{
    if (depth <= 0)
        return Vec3f(0, 0, 0);

    Vec3f sphereCenter = Vec3f(0, 0, -1);
    HitRecord record;

    if (bvh.Hit(r, 0.001f, infinity, record))
    {
        Ray scattered;
        Vec3f attenuation;
        if (record.material->Scatter(r, record, attenuation, scattered))
        {
            return attenuation * RayColor(scattered, depth - 1, bvh);
        }
        return Vec3f(0, 0, 0);
    }

    const Vec3f NormalizedDirection = Normalize(r.d);
    f32 t                           = 0.5f * (NormalizedDirection.y + 1.f);
    return (1 - t) * Vec3f(1, 1, 1) + t * Vec3f(0.5f, 0.7f, 1.f);
}
#endif
#if 1
Vec3f RayColor(const Ray &r, Sampler sampler, const int depth, const Primitive &bvh, const Light *lights, const u32 numLights)
{
    if (depth <= 0)
        return Vec3f(0, 0, 0);

    HitRecord record;

    if (!bvh.Hit(r, 0.001f, infinity, record))
        return BACKGROUND;

    ScatterRecord sRec;
    // Ray scattered;
    Vec3f emissiveColor = record.material->Emitted(r, record, record.u, record.v, record.p);
    if (!record.material->Scatter(r, record, sRec, sampler.Get2D()))
        return emissiveColor;

    if (IsValidRay(&sRec.skipPDFRay))
        return sRec.attenuation * RayColor(sRec.skipPDFRay, sampler, depth - 1, bvh, lights, numLights);

    // Cosine importance sampling
    // TODO: this is hardcoded, this should be switched on based on the type of pdf
    f32 cosPdf = GetCosPDFValue(sRec.sample, record.normal);

    // Light importance sampling
    const Light *light = SampleLights(lights, numLights, sampler.Get1D());
    Assert(light);
    Vec3f randLightDir = Sample(light, record.p, sampler.Get2D());

    Vec3f scatteredDir;
    if (sampler.Get1D() < 0.5f) // RandomFloat() < 0.5f)
    {
        scatteredDir = sRec.sample;
    }
    else
    {
        scatteredDir = randLightDir;
    }
    Ray scattered = Ray(record.p, scatteredDir, r.t);
    f32 lightPdf  = GetLightsPDFValue(lights, numLights, record.p, scatteredDir);
    f32 pdf       = 0.5f * lightPdf + 0.5f * cosPdf;

    f32 scatteringPDF  = record.material->ScatteringPDF(r, record, scattered);
    Vec3f scatterColor = (sRec.attenuation * scatteringPDF * RayColor(scattered, sampler, depth - 1, bvh, lights, numLights)) / pdf;
    return emissiveColor + scatterColor;
}
#endif

void Integrate(const RayQueueItem *inRays, RayQueueItem *outRays, u32 numInRays, u32 *numOutRays, int depth,
               const Primitive &bvh, const Light *lights, const u32 numLights, Vec3f *outRadiance, Sampler sampler)
{
    TIMED_FUNCTION(integrationTime);

    *numOutRays = 0;
    for (u32 i = 0; i < numInRays; i++)
    {
        // TIMED_FUNCTION(samplingTime);
        const Ray r = inRays[i].ray;
        i32 index   = inRays[i].radianceIndex;

        if (depth <= 0)
        {
            outRadiance[index] *= Vec3f(0, 0, 0);
            continue;
        }

        HitRecord record;
        bool result = bvh.Hit(r, 0.001f, infinity, record);
        if (!result)
        {
            outRadiance[index] *= BACKGROUND;
            continue;
        }

        ScatterRecord sRec;
        // Ray scattered;

        Vec3f emissiveColor = record.material->Emitted(r, record, record.u, record.v, record.p);
        if (!record.material->Scatter(r, record, sRec, sampler.Get2D()))
        {
            outRadiance[index] *= emissiveColor;
            continue;
        }

        if (IsValidRay(&sRec.skipPDFRay))
        {
            outRadiance[index] *= sRec.attenuation;
            outRays[*numOutRays].ray           = sRec.skipPDFRay;
            outRays[*numOutRays].radianceIndex = index;
            *numOutRays += 1;
            continue;
        }

        // Cosine importance sampling
        // TODO: this is hardcoded, this should be switched on based on the type of pdf
        f32 cosPdf = GetCosPDFValue(sRec.sample, record.normal);

        // Light importance sampling
        const Light *light = SampleLights(lights, numLights, sampler.Get1D());
        Assert(light);
        Vec3f randLightDir = Sample(light, record.p, sampler.Get2D());

        Vec3f scatteredDir;
        if (sampler.Get1D() < 0.5f) // RandomFloat() < 0.5f)
        {
            scatteredDir = sRec.sample;
        }
        else
        {
            scatteredDir = randLightDir;
        }
        Ray scattered = Ray(record.p, scatteredDir, r.t);
        f32 lightPdf  = GetLightsPDFValue(lights, numLights, record.p, scatteredDir);
        f32 pdf       = 0.5f * lightPdf + 0.5f * cosPdf;

        f32 weight_l = lightPdf / (lightPdf + cosPdf);
        f32 weight_c = cosPdf / (lightPdf + cosPdf);

        f32 scatteringPDF                  = record.material->ScatteringPDF(r, record, scattered);
        Vec3f scatterColor                 = (sRec.attenuation * scatteringPDF) / pdf;
        outRays[*numOutRays].ray           = scattered; // sRec.skipPDFRay;
        outRays[*numOutRays].radianceIndex = index;
        *numOutRays += 1;
        outRadiance[index] *= emissiveColor + scatterColor;
    }
}

struct WorkItem
{
    u32 startX;
    u32 startY;
    u32 onePastEndX;
    u32 onePastEndY;
};

struct RenderParams
{
    Primitive bvh;
    Image *image;
    Light *lights;
    Vec3f cameraCenter;
    Vec3f pixel00;
    Vec3f pixelDeltaU;
    Vec3f pixelDeltaV;
    Vec3f defocusDiskU;
    Vec3f defocusDiskV;
    f32 defocusAngle;
    u32 maxDepth;
    u32 samplesPerPixel;
    u32 squareRootSamplesPerPixel;
    u32 numLights;
};

struct WorkQueue
{
    WorkItem *workItems;
    RenderParams *params;
    u64 volatile workItemIndex;
    u64 volatile tilesFinished;
    u32 workItemCount;
};

struct ThreadData
{
    WorkQueue *queue;
    u32 threadIndex;
};

bool RenderTile(WorkQueue *queue)
{
    u64 workItemIndex = InterlockedAdd(&queue->workItemIndex, 1);
    if (workItemIndex >= queue->workItemCount) return false;

    WorkItem *item = &queue->workItems[workItemIndex];

    i32 samplesPerPixel            = queue->params->samplesPerPixel;
    u32 squareRootSamplesPerPixel  = queue->params->squareRootSamplesPerPixel;
    f32 oneOverSqrtSamplesPerPixel = 1.f / squareRootSamplesPerPixel;
    Vec3f cameraCenter             = queue->params->cameraCenter;

    // ZSobolSampler sampler(samplesPerPixel, Vec2i(queue->params->image->width, queue->params->image->height),
    //                       RandomizeStrategy::FastOwen);
    IndependentSampler sampler(samplesPerPixel);
    for (u32 height = item->startY; height < item->onePastEndY; height++)
    {
        u32 *out = GetPixelPointer(queue->params->image, item->startX, height);
        for (u32 width = item->startX; width < item->onePastEndX; width++)
        {
            // SobolSampler sampler(samplesPerPixel, Vec2i(queue->params->image->width, queue->params->image->height),
            //                      RandomizeStrategy::Owen);

            Vec3f pixelColor(0, 0, 0);

            Vec2i pixel = Vec2i(width, height);
            for (u32 i = 0; i < squareRootSamplesPerPixel; i++)
            {
                for (u32 j = 0; j < squareRootSamplesPerPixel; j++)
                {
                    // const Vec3f offset = Vec3f(RandomFloat() - 0.5f, RandomFloat() - 0.5f, 0.f);
                    u32 sampleIndex = i * squareRootSamplesPerPixel + j;
                    sampler.StartPixelSample(pixel, sampleIndex);

                    Vec2f offsetSample      = sampler.Get2D();
                    const f32 offsetX       = ((i + offsetSample.x)) * oneOverSqrtSamplesPerPixel - 0.5f;
                    const f32 offsetY       = ((j + offsetSample.y)) * oneOverSqrtSamplesPerPixel - 0.5f;
                    const Vec3f offset      = Vec3f(offsetX, offsetY, 0.f);
                    const Vec3f pixelSample = queue->params->pixel00 + ((width + offset.x) * queue->params->pixelDeltaU) +
                                              ((height + offset.y) * queue->params->pixelDeltaV);
                    Vec3f rayOrigin;
                    if (queue->params->defocusAngle <= 0)
                    {
                        rayOrigin = cameraCenter;
                    }
                    else
                    {
                        Vec3f sample = RandomInUnitDisk();
                        rayOrigin    = cameraCenter + sample[0] * queue->params->defocusDiskU +
                                    sample[1] * queue->params->defocusDiskV;
                    }
                    const Vec3f rayDirection = pixelSample - rayOrigin;
                    const f32 rayTime        = sampler.Get1D();
                    Ray r(rayOrigin, rayDirection, rayTime);

                    pixelColor += RayColor(r, &sampler, queue->params->maxDepth, queue->params->bvh,
                                           queue->params->lights, queue->params->numLights);
                }
            }

            pixelColor /= (f32)samplesPerPixel;

            // NOTE: lazy NAN check
            if (pixelColor.x != pixelColor.x) pixelColor.x = 0.f;
            if (pixelColor.y != pixelColor.y) pixelColor.y = 0.f;
            if (pixelColor.z != pixelColor.z) pixelColor.z = 0.f;

            f32 r = 255.f * ExactLinearToSRGB(pixelColor.x);
            f32 g = 255.f * ExactLinearToSRGB(pixelColor.y);
            f32 b = 255.f * ExactLinearToSRGB(pixelColor.z);
            f32 a = 255.f;

            u32 color = (RoundFloatToU32(a) << 24) |
                        (RoundFloatToU32(r) << 16) |
                        (RoundFloatToU32(g) << 8) |
                        (RoundFloatToU32(b) << 0);
            *out++ = color;
        }
    }
    InterlockedAdd(&queue->tilesFinished, 1);
    return true;
}

bool RenderTileTest(WorkQueue *queue)
{
    TIMED_FUNCTION(misc);
    u64 workItemIndex = InterlockedAdd(&queue->workItemIndex, 1);
    if (workItemIndex >= queue->workItemCount) return false;

    WorkItem *item = &queue->workItems[workItemIndex];

    i32 samplesPerPixel            = queue->params->samplesPerPixel;
    Vec3f cameraCenter             = queue->params->cameraCenter;
    u32 squareRootSamplesPerPixel  = queue->params->squareRootSamplesPerPixel;
    f32 oneOverSqrtSamplesPerPixel = 1.f / squareRootSamplesPerPixel;

    TempArena temp = ScratchStart(0, 0);
    u32 numPixels  = (item->onePastEndY - item->startY) * (item->onePastEndX - item->startX);

    // Double buffered
    RayQueueItem *rayBuffer0 = PushArray(temp.arena, RayQueueItem, numPixels);
    RayQueueItem *rayBuffer1 = PushArray(temp.arena, RayQueueItem, numPixels);

    Vec3f *outRadiance  = PushArray(temp.arena, Vec3f, numPixels);
    Vec3f *tempRadiance = PushArray(temp.arena, Vec3f, numPixels);

    // ZSobolSampler sampler(samplesPerPixel, Vec2i(queue->params->image->width, queue->params->image->height),
    //                       RandomizeStrategy::FastOwen);
    IndependentSampler sampler(samplesPerPixel);

    u32 totalTileWidth = item->onePastEndX - item->startX;
    for (u32 i = 0; i < squareRootSamplesPerPixel; i++)
    {
        for (u32 j = 0; j < squareRootSamplesPerPixel; j++)
        {
            u32 rayCount = 0;
            for (u32 height = item->startY; height < item->onePastEndY; height++)
            {
                for (u32 width = item->startX; width < item->onePastEndX; width++)
                {
                    u32 sampleIndex = i * squareRootSamplesPerPixel + j;
                    Vec2i pixel     = Vec2i(width, height);
                    sampler.StartPixelSample(pixel, sampleIndex);

                    Vec2f offsetSample      = sampler.Get2D();
                    const f32 offsetX       = ((i + offsetSample.x)) * oneOverSqrtSamplesPerPixel - 0.5f;
                    const f32 offsetY       = ((j + offsetSample.y)) * oneOverSqrtSamplesPerPixel - 0.5f;
                    const Vec3f offset      = Vec3f(offsetX, offsetY, 0.f);
                    const Vec3f pixelSample = queue->params->pixel00 + ((width + offset.x) * queue->params->pixelDeltaU) +
                                              ((height + offset.y) * queue->params->pixelDeltaV);
                    Vec3f rayOrigin          = cameraCenter;
                    const Vec3f rayDirection = pixelSample - rayOrigin;
                    const f32 rayTime        = sampler.Get1D();
                    Ray r(rayOrigin, rayDirection, rayTime);

                    rayBuffer0[rayCount].ray           = r;
                    rayBuffer0[rayCount].radianceIndex = rayCount;
                    rayCount++;
                }
            }
            // for (u32 pixelIndex = 0; pixelIndex < numPixels; pixelIndex += LANE_WIDTH)
            // {
            //     // u32 sample
            //     u32 sampleIndicesArray[LANE_WIDTH];
            //     u32 widthArray[LANE_WIDTH];
            //     u32 heightArray[LANE_WIDTH];
            //     Vec2i pixelsArray[LANE_WIDTH];
            //
            //     u32 numLanes = Min<u32>(LANE_WIDTH, numPixels - pixelIndex);
            //     for (u32 laneIndex = 0; laneIndex < numLanes; laneIndex++)
            //     {
            //         sampleIndicesArray[laneIndex] = laneIndex * squareRootSamplesPerPixel + j;
            //         widthArray[laneIndex]         = item->startX + ((pixelIndex + laneIndex) % totalTileWidth);
            //         heightArray[laneIndex]        = item->startY + ((pixelIndex + laneIndex) / totalTileWidth);
            //         pixelsArray[laneIndex]        = Vec2i(widthArray[laneIndex], heightArray[laneIndex]);
            //     }
            //
            //     LaneU32 sampleIndices = Load(sampleIndicesArray);
            //     LaneU32 width         = Load(widthArray);
            //     LaneU32 height        = Load(heightArray);
            //     LaneVec2i pixels      = Load(pixelsArray);
            //
            //     // TODO: change this to work with simd
            //     sampler.StartPixelSample(pixel, sampleIndex);
            //
            //     Vec2f offsetSample      = sampler.Get2D();
            //     const f32 offsetX      = ((i + offsetSample.x)) * oneOverSqrtSamplesPerPixel - 0.5f;
            //     const f32 offsetY      = ((j + offsetSample.y)) * oneOverSqrtSamplesPerPixel - 0.5f;
            //     const Vec3f offset      = Vec3f(offsetX, offsetY, 0.f);
            //     const Vec3f pixelSample = queue->params->pixel00 + ((width + offset.x) * queue->params->pixelDeltaU) +
            //                              ((height + offset.y) * queue->params->pixelDeltaV);
            //     Vec3f rayOrigin          = cameraCenter;
            //     const Vec3f rayDirection = pixelSample - rayOrigin;
            //     const f32 rayTime       = sampler.Get1D();
            //     Ray r(rayOrigin, rayDirection, rayTime);
            //
            //     rayBuffer0[rayCount].ray           = r;
            //     rayBuffer0[rayCount].radianceIndex = rayCount;
            //     rayCount++;
            // }
            //
            Assert(rayCount == numPixels);
            int depth = queue->params->maxDepth;
            for (u32 index = 0; index < numPixels; index++)
            {
                tempRadiance[index] = Vec3f(1, 1, 1);
            }
            u32 numRays = numPixels;
            for (;;)
            {
                RayQueueItem *inRayBuffer  = ((queue->params->maxDepth - depth) & 1) ? rayBuffer1 : rayBuffer0;
                RayQueueItem *outRayBuffer = ((queue->params->maxDepth - depth) & 1) ? rayBuffer0 : rayBuffer1;
                Integrate(inRayBuffer, outRayBuffer, numRays, &numRays, depth, queue->params->bvh,
                          queue->params->lights, queue->params->numLights, tempRadiance, &sampler);
                if (depth == 0)
                    break;
                depth--;
            }
            for (u32 index = 0; index < numPixels; index++)
            {
                outRadiance[index] += tempRadiance[index];
            }
        }
    }

    u32 stride = item->onePastEndX - item->startX;
    for (u32 height = item->startY; height < item->onePastEndY; height++)
    {
        for (u32 width = item->startX; width < item->onePastEndX; width++)
        {
            u32 *out         = GetPixelPointer(queue->params->image, width, height);
            Vec3f pixelColor = outRadiance[(width - item->startX) + (height - item->startY) * stride];
            pixelColor /= (f32)samplesPerPixel;

            // NOTE: lazy NAN check
            if (pixelColor.x != pixelColor.x) pixelColor.x = 0.f;
            if (pixelColor.y != pixelColor.y) pixelColor.y = 0.f;
            if (pixelColor.z != pixelColor.z) pixelColor.z = 0.f;

            f32 r = 255.f * ExactLinearToSRGB(pixelColor.x);
            f32 g = 255.f * ExactLinearToSRGB(pixelColor.y);
            f32 b = 255.f * ExactLinearToSRGB(pixelColor.z);
            f32 a = 255.f;

            u32 color = (RoundFloatToU32(a) << 24) |
                        (RoundFloatToU32(r) << 16) |
                        (RoundFloatToU32(g) << 8) |
                        (RoundFloatToU32(b) << 0);
            *out = color;
        }
    }
    InterlockedAdd(&queue->tilesFinished, 1);

    ScratchEnd(temp);
    return true;
}

THREAD_ENTRY_POINT(WorkerThread)
{
    ThreadData *data = (ThreadData *)parameter;

    char name[20];
    sprintf_s(name, "Worker %u", data->threadIndex);
    SetThreadName(name);
    SetThreadIndex(data->threadIndex);

    // while (RenderTile(data->queue)) continue;
    while (RenderTileTest(data->queue)) continue;
}
#endif

} // namespace rt

using namespace rt;

int main(int argc, char *argv[])
{
    BuildPackMask();
    Arena *dataArena = ArenaAlloc();
    Arena *arena     = ArenaAlloc();
    InitThreadContext(arena, "[Main Thread]", 1);

    const u32 count = 3000000;

    Options options = {};
    bool setOptions = false;
    for (int i = 1; i < argc;)
    {
        string arg = Str8C(argv[i]);
        if (Contains(arg, "-pixel"))
        {
            if (i + 2 >= argc)
            {
                printf("Option -pixel requires two integer coordinates, specified as --pixel "
                       "x y");
            }
            string x = Str8C(argv[i + 1]);
            string y = Str8C(argv[i + 2]);
            if (!IsInt(x) || !IsInt(y))
            {
                printf("-pixel requires two integer coordinates");
            }

            options.pixelX = ConvertToUint(x);
            options.pixelY = ConvertToUint(y);
            setOptions     = true;
            i += 3;
        }
        else if (Contains(arg, "-validation"))
        {
            options.useValidation = true;
            i++;
        }
        else
        {
            options.filename = arg;
            i++;
        }
    }

    if (options.filename.size == 0 || !(GetFileExtension(options.filename) == "rtscene"))
    {
        printf("Must pass in a .rtscene file.\n");
        return 1;
    }

#ifdef USE_GPU
#ifdef DEBUG
    ValidationMode mode =
        options.useValidation ? ValidationMode::Verbose : ValidationMode::Disabled;
#else
    ValidationMode mode = ValidationMode::Disabled;
#endif
    Vulkan *v = PushStructConstruct(arena, Vulkan)(mode);
    device    = v;
#endif
    OS_Init();

    Spectra::Init(arena);
    RGBToSpectrumTable::Init(arena);
    RGBColorSpace::Init(arena);
    InitializePtex();

    u32 numProcessors      = OS_NumProcessors();
    threadLocalStatistics  = PushArray(arena, ThreadStatistics, numProcessors);
    threadMemoryStatistics = PushArray(arena, ThreadMemoryStatistics, numProcessors);
    scheduler.Init(numProcessors);

    TestRender(arena, &options);
}
