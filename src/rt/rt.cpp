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
#include "vulkan.h"
#include "tests/test.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"

namespace rt
{

Image LoadFile(const char *file, int numComponents)
{
    Image image;
    i32 nComponents;
    image.contents =
        (u8 *)stbi_load(file, &image.width, &image.height, &nComponents, numComponents);
    image.bytesPerPixel = nComponents;
    return image;
}

Image LoadHDR(const char *file)
{
    Image image;
    i32 nComponents;
    image.contents      = (u8 *)stbi_loadf(file, &image.width, &image.height, &nComponents, 0);
    image.bytesPerPixel = nComponents * sizeof(f32);
    return image;
}

inline f32 DegreesToRadians(f32 degrees) { return degrees * PI / 180.f; }

//////////////////////////////
// Intervals
//
bool IsInInterval(f32 min, f32 max, f32 x) { return x >= min && x <= max; }

static Vec3f BACKGROUND;

bool IsValidRay(Ray *r) { return r->t != (f32)U32Max; }

inline Vec3f LinearToSRGB(const Vec3f &v) { return Vec3f(Sqrt(v.x), Sqrt(v.y), Sqrt(v.z)); }

f32 ExactLinearToSRGB(f32 l)
{
    if (l < 0.0f)
    {
        l = 0.0f;
    }

    if (l > 1.0f)
    {
        l = 1.0f;
    }

    f32 s = l * 12.92f;
    if (l > 0.0031308f)
    {
        s = 1.055f * Pow(l, 1.0f / 2.4f) - 0.055f;
    }
    return s;
}

f32 *Image::GetSamplingDistribution(Arena *arena)
{
    u8 *ptr = contents;

    f32 *result = PushArrayNoZero(arena, f32, height * width);
    u32 count   = 0;
    for (i32 h = 0; h < height; h++)
    {
        for (i32 w = 0; w < width; w++)
        {
            Vec3f values    = SRGBToLinear(GetColor(this, w, h));
            f32 val         = (values[0] + values[1] + values[2]) / 3.f;
            result[count++] = val;
            Assert(result[count - 1] == result[count - 1]);
            ptr += bytesPerPixel;
        }
    }
    return result;
}

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
    OS_Init();

    Spectra::Init(arena);
    RGBToSpectrumTable::Init(arena);
    RGBColorSpace::Init(arena);
    InitializePtex();

    u32 numProcessors      = OS_NumProcessors();
    threadLocalStatistics  = PushArray(arena, ThreadStatistics, numProcessors);
    threadMemoryStatistics = PushArray(arena, ThreadMemoryStatistics, numProcessors);
    scheduler.Init(numProcessors);

    OS_Handle handle = OS_WindowInit(1920, 804);

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

    TestRender(arena, handle, &options);
    // WhiteFurnaceTest(arena, &options);

    // CameraRayTest(arena);
    // BVHTraverse4Test();

    // BVHIntersectionTest(arena);
    // VolumeRenderingTest(arena, "wdas_cloud_quarter.nvdb");
    // BVHSortingTest();

    // TriangleMesh mesh = LoadPLY(arena,
    // "data/island/pbrt-v4/isIronwoodA1/isIronwoodA1_geometry_00001.ply"); QuadMesh mesh =
    // LoadQuadPLY(arena,
    // "data/island/pbrt-v4/isIronwoodA1/isIronwoodA1_geometry_00001.ply");

    // TriangleMesh mesh = LoadPLY(arena,
    // "data/island/pbrt-v4/osOcean/osOcean_geometry_00001.ply"); TriangleMesh mesh =
    // LoadPLY(arena, "data/xyzrgb_statuette.ply");

    // QuadSBVHBuilderTest(dataArena, &mesh);
    // AOSSBVHBuilderTest(dataArena, &mesh);
    // PartitionFix();

    // PartialRebraidBuilderTest(dataArena);

    //////////////////////////////
    // Loading PBRT File Test
    //

//////////////////////////////
// SIMD Octahedral Encoding Test
//
#if 0
    const u32 num  = 100000000;
    Vec3f *normals  = (Vec3f *)malloc(sizeof(Vec3f) * num);
    u16 *results1x = (u16 *)malloc(sizeof(u16) * num);
    u16 *results1y = (u16 *)malloc(sizeof(u16) * num);

    u16 *results2x = (u16 *)malloc(sizeof(u16) * num);
    u16 *results2y = (u16 *)malloc(sizeof(u16) * num);

    for (u32 i = 0; i < num; i++)
    {
        normals[i] = Normalize(RandomVec3());
    }

    clock_t start = clock();

    LaneF32 absMask  = CastLaneF32FromLaneU32(LaneU32FromU32(0x7fffffff));
    LaneF32 signMask = CastLaneF32FromLaneU32(LaneU32FromU32(0x80000000));
    LaneF32 one      = LaneF32FromF32(1.f);
    LaneF32 oneHalf  = LaneF32FromF32(0.5f);
    LaneF32 u16Max   = LaneF32FromF32(65535.f);

    for (u32 i = 0; i < num; i += 4)
    {
        f32 x[] = {
            normals[i].x,
            normals[i + 1].x,
            normals[i + 2].x,
            normals[i + 3].x,
        };

        f32 y[] = {
            normals[i].y,
            normals[i + 1].y,
            normals[i + 2].y,
            normals[i + 3].y,
        };

        f32 z[] = {
            normals[i].z,
            normals[i + 1].z,
            normals[i + 2].z,
            normals[i + 3].z,
        };
        LaneF32 xV = Load(x);
        LaneF32 yV = Load(y);
        LaneF32 zV = Load(z);

        LaneF32 absX = Abs(xV);
        LaneF32 absY = Abs(yV);

        LaneF32 l1    = absX + absY + (zV & absMask);
        LaneF32 rcpL1 = rcp(l1);

        xV   = xV * rcpL1;
        yV   = yV * rcpL1;
        absX = xV & absMask;
        absY = yV & absMask;

        LaneF32 xResultZPos = (xV + one) * oneHalf * u16Max + oneHalf;
        LaneF32 xResultZNeg = (((one - absY) + one) * oneHalf * u16Max + oneHalf) ^ (xV & signMask);

        LaneF32 yResultZPos = (yV + one) * oneHalf * u16Max + oneHalf;
        LaneF32 yResultZNeg = (((one - absX) + one) * oneHalf * u16Max + oneHalf) ^ (yV & signMask);

        LaneF32 zSignMask = zV >= LaneF32Zero();

        LaneU32 xResult = TruncateU32ToU16(ConvertLaneF32ToLaneU32(Blend(xResultZNeg, xResultZPos, zSignMask)));
        LaneU32 yResult = TruncateU32ToU16(ConvertLaneF32ToLaneU32(Blend(yResultZNeg, yResultZPos, zSignMask)));

        StoreU16(results1x + i, xResult);
        StoreU16(results1y + i, yResult);
    }
    clock_t end = clock();

    printf("Total time simd: %dms\n", end - start);

    for (u32 i = 0; i < 10; i++)
    {
        printf("%hu ", results1x[i + 1000]);
        printf("%hu\n", results1y[i + 1000]);
    }

    printf("\n");
    start = clock();
    for (u32 i = 0; i < 100000000; i += 4)
    {
        OctahedralVector v1 = EncodeOctahedral(normals[i + 0]);
        OctahedralVector v2 = EncodeOctahedral(normals[i + 1]);
        OctahedralVector v3 = EncodeOctahedral(normals[i + 2]);
        OctahedralVector v4 = EncodeOctahedral(normals[i + 3]);

        results2x[i + 0] = v1.x;
        results2x[i + 1] = v2.x;
        results2x[i + 2] = v3.x;
        results2x[i + 3] = v4.x;

        results2y[i + 0] = v1.y;
        results2y[i + 1] = v2.y;
        results2y[i + 2] = v3.y;
        results2y[i + 3] = v4.y;
    }
    end = clock();

    for (u32 i = 0; i < 10; i++)
    {
        printf("%hu ", results2x[i + 1000]);
        printf("%hu\n", results2y[i + 1000]);
    }
    printf("Total time normal: %dms\n", end - start);
    // OctahedralVector result[] = {};
#endif

//////////////////////////////
// Main
//
#if 0
#if SPHERES
    const f32 aspectRatio     = 16.f / 9.f;
    const Vec3f lookFrom      = Vec3f(13, 2, 3);
    const Vec3f lookAt        = Vec3f(0, 0, 0);
    const Vec3f worldUp       = Vec3f(0, 1, 0);
    const f32 verticalFov     = 20;
    const f32 defocusAngle    = 0.6f;
    const f32 focusDist       = 10;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    const int imageWidth      = 400;
    BACKGROUND                = Vec3f(0.7f, 0.8f, 1.f);
#elif EARTH
    const f32 aspectRatio     = 16.f / 9.f;
    const Vec3f lookFrom      = Vec3f(0, 0, 12);
    const Vec3f lookAt        = Vec3f(0, 0, 0);
    const Vec3f worldUp       = Vec3f(0, 1, 0);
    const f32 verticalFov     = 20;
    const f32 defocusAngle    = 0;
    const f32 focusDist       = 10;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    const int imageWidth      = 400;
    BACKGROUND                = Vec3f(0.7f, 0.8f, 1.f);
#elif PERLIN
    const f32 aspectRatio     = 16.f / 9.f;
    const Vec3f lookFrom      = Vec3f(13, 2, 3);
    const Vec3f lookAt        = Vec3f(0, 0, 0);
    const Vec3f worldUp       = Vec3f(0, 1, 0);
    const f32 verticalFov     = 20;
    const f32 defocusAngle    = 0;
    const f32 focusDist       = 10;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    const int imageWidth      = 400;
    BACKGROUND                = Vec3f(0.7f, 0.8f, 1.f);
#elif QUADS
    const f32 aspectRatio     = 1.f;
    const Vec3f lookFrom      = Vec3f(0, 0, 9);
    const Vec3f lookAt        = Vec3f(0, 0, 0);
    const Vec3f worldUp       = Vec3f(0, 1, 0);
    const f32 verticalFov     = 80;
    const f32 defocusAngle    = 0;
    const f32 focusDist       = 10;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    const int imageWidth      = 400;
    BACKGROUND                = Vec3f(0.7f, 0.8f, 1.f);
#elif LIGHTS
    const f32 aspectRatio     = 16.f / 9.f;
    const Vec3f lookFrom      = Vec3f(26, 3, 6);
    const Vec3f lookAt        = Vec3f(0, 2, 0);
    const Vec3f worldUp       = Vec3f(0, 1, 0);
    const f32 verticalFov     = 20;
    const f32 defocusAngle    = 0;
    const f32 focusDist       = 10;
    const int imageWidth      = 400;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    BACKGROUND                = Vec3f(0, 0, 0);
#elif CORNELL
    const f32 aspectRatio  = 1.f;
    const Vec3f lookFrom   = Vec3f(278, 278, -800);
    const Vec3f lookAt     = Vec3f(278, 278, 0);
    const Vec3f worldUp    = Vec3f(0, 1, 0);
    const f32 verticalFov  = 40;
    const f32 defocusAngle = 0;
    const f32 focusDist    = 10;

    const int imageWidth      = 600;
    const int samplesPerPixel = 200;
    const int maxDepth        = 50;
    BACKGROUND                = Vec3f(0, 0, 0);
#elif CORNELL_SMOKE
    const f32 aspectRatio  = 1.0;
    const Vec3f lookFrom   = Vec3f(278, 278, -800);
    const Vec3f lookAt     = Vec3f(278, 278, 0);
    const Vec3f worldUp    = Vec3f(0, 1, 0);
    const f32 verticalFov  = 40;
    const f32 defocusAngle = 0;
    const f32 focusDist    = 10;

    const u32 imageWidth      = 600;
    const u32 samplesPerPixel = 200;
    const u32 maxDepth        = 50;
    BACKGROUND                = Vec3f(0, 0, 0);
#elif FINAL
    const f32 aspectRatio  = 1.0;
    const Vec3f lookFrom   = Vec3f(478, 278, -600);
    const Vec3f lookAt     = Vec3f(278, 278, 0);
    const Vec3f worldUp    = Vec3f(0, 1, 0);
    const f32 verticalFov  = 40;
    const f32 defocusAngle = 0;
    const f32 focusDist    = 10;

    const int imageWidth      = 600;
    const int samplesPerPixel = 4;
    const int maxDepth        = 4;
    BACKGROUND                = Vec3f(0, 0, 0);
#endif

    u32 imageHeight = u32(imageWidth / aspectRatio);
    imageHeight     = imageHeight < 1 ? 1 : imageHeight;
    f32 focalLength = Length(lookFrom - lookAt);
    f32 theta       = DegreesToRadians(verticalFov);
    f32 h           = Tan(theta / 2);

    Vec3f f = Normalize(lookFrom - lookAt);
    Vec3f s = Cross(worldUp, f);
    Vec3f u = Cross(f, s);

    f32 viewportHeight = 2 * h * focusDist;
    f32 viewportWidth  = viewportHeight * (f32(imageWidth) / imageHeight);
    Vec3f cameraCenter = lookFrom;

    Vec3f viewportU = viewportWidth * s;
    Vec3f viewportV = viewportHeight * -u;

    Vec3f pixelDeltaU = viewportU / (f32)imageWidth;
    Vec3f pixelDeltaV = viewportV / (f32)imageHeight;

    Vec3f viewportUpperLeft = cameraCenter - focusDist * f - viewportU / 2.f - viewportV / 2.f;
    Vec3f pixel00           = viewportUpperLeft + 0.5f * (pixelDeltaU + pixelDeltaV);

    f32 defocusRadius  = focusDist * Tan(DegreesToRadians(defocusAngle / 2));
    Vec3f defocusDiskU = defocusRadius * s;
    Vec3f defocusDiskV = defocusRadius * u;

    u32 squareRootSamplesPerPixel = (u32)Sqrt(samplesPerPixel);

    Scene scene = {};

#if SPHERES
    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            f32 chooseMat = RandomFloat();
            Vec3f center(a + 0.9f * RandomFloat(), 0.2f, b + 0.9f * RandomFloat());

            if ((center - Vec3f(4, 0.2f, 0)).length() > 0.9f)
            {
                Material *material = (Material *)malloc(sizeof(Material));
                // Diffuse
                if (chooseMat < 0.8)
                {
                    Vec3f albedo  = RandomVec3() * RandomVec3();
                    Vec3f center2 = center + Vec3f(0, RandomFloat(0, .5f), 0);
                    *material     = Material::CreateLambert(albedo);
                    scene.Add(Sphere(center, center2, 0.2f, material));
                }
                // Metal
                else if (chooseMat < 0.95)
                {
                    Vec3f albedo = RandomVec3(0.5f, 1);
                    f32 fuzz     = RandomFloat(0, 0.5f);
                    *material    = Material::CreateMetal(albedo, fuzz);
                    scene.Add(Sphere(center, 0.2f, material));
                }
                // Glass
                else
                {
                    *material = Material::CreateDielectric(1.5f);
                    scene.Add(Sphere(center, 0.2f, material));
                }
            }
        }
    }

    Texture checkered    = Texture::CreateCheckered(0.32f, Vec3f(.2f, .3f, .1f), Vec3f(.9f, .9f, .9f));
    Material materials[] = {
        Material::CreateDielectric(1.5f),
        Material::CreateLambert(Vec3f(0.4f, 0.2f, 0.1f)),
        Material::CreateMetal(Vec3f(0.7f, 0.6f, 0.5f), 0.f),
        Material::CreateLambert(&checkered),
    };

    scene.Add(Sphere(Vec3f(0, 1, 0), 1.f, &materials[0]));
    scene.Add(Sphere(Vec3f(-4, 1, 0), 1.f, &materials[1]));
    scene.Add(Sphere(Vec3f(4, 1, 0), 1.f, &materials[2]));

    // ground
    scene.Add(Sphere(Vec3f(0, -1000, 0), 1000, &materials[3]));

#elif EARTH
    Texture earth             = Texture::CreateImage("earthmap.jpg");
    Material surface          = Material::CreateLambert(&earth);

    scene.Add(Sphere(Vec3f(0, 0, 0), 2, &surface));
#elif PERLIN
    Texture noise             = Texture::CreateNoise(4.0);
    Material perlin           = Material::CreateLambert(&noise);

    scene.Add(Sphere(Vec3f(0, -1000, 0), 1000, &perlin));
    scene.Add(Sphere(Vec3f(0, 2, 0), 2, &perlin));
#elif QUADS
    Material materials[]      = {
        Material::CreateLambert(Vec3f(1.f, 0.2f, 0.2f)),
        Material::CreateLambert(Vec3f(0.2f, 1.f, 0.2f)),
        Material::CreateLambert(Vec3f(0.2f, 0.2f, 1.f)),
        Material::CreateLambert(Vec3f(1.f, 0.5f, 0.f)),
        Material::CreateLambert(Vec3f(0.2f, 0.8f, 0.8f)),
    };

    scene.Add(Quad(Vec3f(-3, -2, 5), Vec3f(0, 0, -4), Vec3f(0, 4, 0), &materials[0]));
    scene.Add(Quad(Vec3f(-2, -2, 0), Vec3f(4, 0, 0), Vec3f(0, 4, 0), &materials[1]));
    scene.Add(Quad(Vec3f(3, -2, 1), Vec3f(0, 0, 4), Vec3f(0, 4, 0), &materials[2]));
    scene.Add(Quad(Vec3f(-2, 3, 1), Vec3f(4, 0, 0), Vec3f(0, 0, 4), &materials[3]));
    scene.Add(Quad(Vec3f(-2, -3, 5), Vec3f(4, 0, 0), Vec3f(0, 0, -4), &materials[4]));
#elif LIGHTS
    Texture texture           = Texture::CreateNoise(4);
    Material lambert          = Material::CreateLambert(&texture);
    scene.Add(Sphere(Vec3f(0, -1000, 0), 1000, &lambert));
    scene.Add(Sphere(Vec3f(0, 2, 0), 2, &lambert));

    Material diffuse = Material::CreateDiffuseLight(Vec3f(4, 4, 4));
    scene.Add(Sphere(Vec3f(0, 7, 0), 2, &diffuse));
    scene.Add(Quad(Vec3f(3, 1, -2), Vec3f(2, 0, 0), Vec3f(0, 2, 0), &diffuse));
#elif CORNELL
    Material materials[]      = {
        Material::CreateLambert(Vec3f(.65f, .05f, .05f)),
        Material::CreateLambert(Vec3f(.73f, .73f, .73f)),
        Material::CreateLambert(Vec3f(.12f, .45f, .15f)),
        Material::CreateDiffuseLight(Vec3f(15, 15, 15)),
        Material::CreateMetal(Vec3f(.8f, .85f, .88f), 0.f),
        Material::CreateDielectric(1.5f),
    };

    Quad quads[] = {
        Quad(Vec3f(555, 0, 0), Vec3f(0, 555, 0), Vec3f(0, 0, 555), &materials[2]),
        Quad(Vec3f(343, 554, 332), Vec3f(-130, 0, 0), Vec3f(0, 0, -105), &materials[3]),
        Quad(Vec3f(0, 0, 0), Vec3f(555, 0, 0), Vec3f(0, 0, 555), &materials[1]),
        Quad(Vec3f(555, 555, 555), Vec3f(-555, 0, 0), Vec3f(0, 0, -555), &materials[1]),
        Quad(Vec3f(0, 0, 555), Vec3f(555, 0, 0), Vec3f(0, 555, 0), &materials[1]),
        Quad(Vec3f(0, 0, 0), Vec3f(0, 555, 0), Vec3f(0, 0, 555), &materials[0]),
    };

    Box boxes[] = {
        Box(Vec3f(0, 0, 0), Vec3f(165, 330, 165), &materials[1]),
        // Box(Vec3f(0, 0, 0), Vec3f(165, 165, 165), &materials[1]),
    };

    Sphere spheres[] = {
        Sphere(Vec3f(190, 90, 190), 90, &materials[5]),
    };

    HomogeneousTransform transforms[] = {
        {Vec3f(265, 0, 295), DegreesToRadians(15)},
        {Vec3f(130, 0, 65), DegreesToRadians(-18)}

    };

    Light lights[] = {
        CreateQuadLight(&quads[1]),
        CreateSphereLight(&spheres[0]),
    };

    scene.spheres        = spheres;
    scene.sphereCount    = ArrayLength(spheres);
    scene.quads          = quads;
    scene.quadCount      = ArrayLength(quads);
    scene.boxes          = boxes;
    scene.boxCount       = ArrayLength(boxes);
    scene.transforms     = transforms;
    scene.transformCount = ArrayLength(transforms);

    scene.FinalizePrimitives();
    scene.AddTransform(PrimitiveType_Box, 0, 0);
// scene.AddTransform(PrimitiveType_Box, 1, 1);
#elif CORNELL_SMOKE
    Material materials[]      = {
        Material::CreateLambert(Vec3f(.65f, .05f, .05f)),
        Material::CreateLambert(Vec3f(.73f, .73f, .73f)),
        Material::CreateLambert(Vec3f(.12f, .45f, .15f)),
        Material::CreateDiffuseLight(Vec3f(7, 7, 7)),
        Material::CreateIsotropic(Vec3f(0, 0, 0)),
        Material::CreateIsotropic(Vec3f(1, 1, 1)),
    };

    Quad quads[] = {
        Quad(Vec3f(555, 0, 0), Vec3f(0, 555, 0), Vec3f(0, 0, 555), &materials[2]),
        Quad(Vec3f(0, 0, 0), Vec3f(0, 555, 0), Vec3f(0, 0, 555), &materials[0]),
        Quad(Vec3f(113, 554, 127), Vec3f(330, 0, 0), Vec3f(0, 0, 305), &materials[3]),
        Quad(Vec3f(0, 0, 0), Vec3f(555, 0, 0), Vec3f(0, 0, 555), &materials[1]),
        Quad(Vec3f(0, 555, 0), Vec3f(555, 0, 0), Vec3f(0, 0, 555), &materials[1]),
        Quad(Vec3f(0, 0, 555), Vec3f(555, 0, 0), Vec3f(0, 555, 0), &materials[1]),
    };

    Box boxes[] = {
        Box(Vec3f(0, 0, 0), Vec3f(165, 330, 165), &materials[1]),
        Box(Vec3f(0, 0, 0), Vec3f(165, 165, 165), &materials[1]),
    };

    HomogeneousTransform transforms[] = {
        {Vec3f(265, 0, 295), DegreesToRadians(15)},
        {Vec3f(130, 0, 65), DegreesToRadians(-18)},
    };

    ConstantMedium media[] = {
        {0.01f, &materials[4]},
        {0.01f, &materials[5]},
    };

    scene.quads          = quads;
    scene.quadCount      = ArrayLength(quads);
    scene.boxes          = boxes;
    scene.boxCount       = ArrayLength(boxes);
    scene.transforms     = transforms;
    scene.transformCount = ArrayLength(transforms);
    scene.media          = media;
    scene.mediaCount     = ArrayLength(media);

    scene.FinalizePrimitives();
    scene.AddConstantMedium(PrimitiveType_Box, 0, 0);
    scene.AddTransform(PrimitiveType_Box, 0, 0);

    scene.AddConstantMedium(PrimitiveType_Box, 1, 1);
    scene.AddTransform(PrimitiveType_Box, 1, 1);

#elif FINAL
    Texture texture           = Texture::CreateImage("earthmap.jpg");
    Texture noise             = Texture::CreateNoise(0.2f);
    Material materials[]{
        Material::CreateLambert(Vec3f(0.48f, 0.83f, 0.53f)),
        Material::CreateDiffuseLight(Vec3f(7, 7, 7)),
        Material::CreateLambert(Vec3f(0.7f, 0.3f, 0.1f)),
        Material::CreateDielectric(1.5f),
        Material::CreateMetal(Vec3f(0.8f, 0.8f, 0.9f), 1.f),
        Material::CreateIsotropic(Vec3f(0.2f, 0.4f, 0.9f)),
        Material::CreateIsotropic(Vec3f(1, 1, 1)),
        Material::CreateLambert(&texture),
        Material::CreateLambert(&noise),
        Material::CreateLambert(Vec3f(.73f, .73f, .73f)),
    };

    const i32 boxesPerSide = 20;
    Box boxes[boxesPerSide * boxesPerSide];
    for (i32 i = 0; i < boxesPerSide; i++)
    {
        for (i32 j = 0; j < boxesPerSide; j++)
        {
            f32 w  = 100.f;
            f32 x0 = -1000.f + i * w;
            f32 z0 = -1000.f + j * w;
            f32 y0 = 0.f;
            f32 x1 = x0 + w;
            f32 y1 = RandomFloat(1, 101);
            f32 z1 = z0 + w;

            boxes[i * boxesPerSide + j] = Box(Vec3f(x0, y0, z0), Vec3f(x1, y1, z1), &materials[0]);
        }
    }

    Quad quads[] = {
        Quad(Vec3f(123, 554, 147), Vec3f(300, 0, 0), Vec3f(0, 0, 265), &materials[1]),
    };

    Light lights[] = {
        CreateQuadLight(&quads[0]),
    };

    Vec3f center1 = Vec3f(400, 400, 200);
    Vec3f center2 = center1 + Vec3f(30, 0, 0);

    const i32 ns           = 1000;
    Sphere spheres[8 + ns] = {
        Sphere(center1, center2, 50, &materials[2]),
        Sphere(Vec3f(260, 150, 45), 50, &materials[3]),
        Sphere(Vec3f(0, 150, 145), 50, &materials[4]),
        Sphere(Vec3f(360, 150, 145), 70, &materials[3]),
        Sphere(Vec3f(360, 150, 145), 70, &materials[3]),
        Sphere(Vec3f(0, 0, 0), 5000, &materials[3]),
        Sphere(Vec3f(400, 200, 400), 100, &materials[7]),
        Sphere(Vec3f(220, 280, 300), 80, &materials[8]),
    };

    HomogeneousTransform transforms[] = {
        {Vec3f(-100, 270, 395), DegreesToRadians(15)},
    };

    ConstantMedium media[] = {
        {0.2f, &materials[5]},
        {0.0001f, &materials[6]},
    };

    for (i32 j = 8; j < ArrayLength(spheres); j++)
    {
        spheres[j] = Sphere(RandomVec3(0, 165), 10, &materials[9]);
    }

    scene.spheres        = spheres;
    scene.sphereCount    = ArrayLength(spheres);
    scene.quads          = quads;
    scene.quadCount      = ArrayLength(quads);
    scene.boxes          = boxes;
    scene.boxCount       = ArrayLength(boxes);
    scene.transforms     = transforms;
    scene.transformCount = ArrayLength(transforms);
    scene.media          = media;
    scene.mediaCount     = ArrayLength(media);
    scene.FinalizePrimitives();

    scene.AddConstantMedium(PrimitiveType_Sphere, 4, 0);
    scene.AddConstantMedium(PrimitiveType_Sphere, 5, 1);

    for (i32 j = 8; j < ArrayLength(spheres); j++)
    {
        scene.AddTransform(PrimitiveType_Sphere, j, 0);
    }

#endif

    BVH bvh;
    bvh.Build(arena, &scene, 2);

    CompressedBVH4 compressedBVH = CreateCompressedBVH4(arena, &bvh);
    BVH4 bvh4                    = CreateBVH4(arena, &bvh);

    RenderParams params;
    params.pixel00                   = pixel00;
    params.pixelDeltaU               = pixelDeltaU;
    params.pixelDeltaV               = pixelDeltaV;
    params.cameraCenter              = cameraCenter;
    params.defocusDiskU              = defocusDiskU;
    params.defocusDiskV              = defocusDiskV;
    params.defocusAngle              = defocusAngle;
    params.bvh                       = &compressedBVH;
    params.maxDepth                  = maxDepth;
    params.samplesPerPixel           = samplesPerPixel;
    params.squareRootSamplesPerPixel = squareRootSamplesPerPixel;
    params.lights                    = lights;
    params.numLights                 = ArrayLength(lights);

    Image image;
    image.width         = imageWidth;
    image.height        = imageHeight;
    image.bytesPerPixel = sizeof(u32);
    image.contents      = (u8 *)malloc(GetImageSize(&image));
    params.image        = &image;

    u32 tileWidth     = 64;
    u32 tileHeight    = 64;
    u32 tileCountX    = (imageWidth + tileWidth - 1) / tileWidth;
    u32 tileCountY    = (imageHeight + tileHeight - 1) / tileHeight;
    WorkQueue queue   = {};
    u32 workItemTotal = tileCountX * tileCountY;
    queue.workItems   = (WorkItem *)malloc(sizeof(WorkItem) * workItemTotal);
    queue.params      = &params;
    for (u32 tileY = 0; tileY < tileCountY; tileY++)
    {
        u32 startY      = tileY * tileHeight;
        u32 onePastEndY = startY + tileHeight;
        onePastEndY     = onePastEndY > imageHeight ? imageHeight : onePastEndY;
        for (u32 tileX = 0; tileX < tileCountX; tileX++)
        {
            u32 startX      = tileX * tileWidth;
            u32 onePastEndX = startX + tileWidth;
            onePastEndX     = onePastEndX > imageWidth ? imageWidth : onePastEndX;

            WorkItem *workItem    = &queue.workItems[queue.workItemCount++];
            workItem->startX      = startX;
            workItem->startY      = startY;
            workItem->onePastEndX = onePastEndX;
            workItem->onePastEndY = onePastEndY;
        }
    }

    Assert(queue.workItemCount == workItemTotal);

    clock_t start = clock();
    for (u32 i = 1; i < OS_NumProcessors(); i++)
    {
        ThreadData *threadData  = PushStruct(arena, ThreadData);
        threadData->queue       = &queue;
        threadData->threadIndex = i;
        OS_CreateWorkThread(WorkerThread, threadData);
    }

    while (queue.tilesFinished < workItemTotal)
    {
        fprintf(stderr, "\rRaycasting %d%%...    ", 100 * (u32)queue.tilesFinished / workItemTotal);
        fflush(stdout);
        // RenderTile(&queue);
        RenderTileTest(&queue);
    }
    clock_t end = clock();

    fprintf(stderr, "\n");
    printf("Total time: %dms\n", end - start);
    WriteImage(&image, "image.bmp");
    // printf("Total ray AABB tests: %lld\n", statistics.rayAABBTests.load());

    // Aggregate statistics
    u64 primitiveIntersectionTime = 0;
    u64 bvhIntersectionTime       = 0;
    u64 integrationTime           = 0;
    u64 totalTime                 = 0;
    u64 totalSamplingTime         = 0;
    for (u32 i = 0; i < OS_NumProcessors(); i++)
    {
        primitiveIntersectionTime += threadLocalStatistics[i].primitiveIntersectionTime;
        bvhIntersectionTime += threadLocalStatistics[i].bvhIntersectionTime;
        integrationTime += threadLocalStatistics[i].integrationTime;
        totalTime += threadLocalStatistics[i].dumb;
        totalSamplingTime += threadLocalStatistics[i].samplingTime;
    }
    printf("Total primitive intersection time: %lldms\n", primitiveIntersectionTime);
    printf("Total integration time: %lldms\n", integrationTime);
    printf("Total sampling time time: %lldms\n", totalSamplingTime);
    printf("Total time: %lldms\n", totalTime);
    fprintf(stderr, "Done.");
#endif
}
