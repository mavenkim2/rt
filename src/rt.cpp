#include "rt.h"
#include "random.h"
#include "scene.h"
#include "bvh.h"
#include "primitive.h"
#include "parallel.h"
#include <algorithm>

#include "math.cpp"
#include "scene.cpp"
#include "bvh.cpp"

inline f32 DegreesToRadians(f32 degrees)
{
    return degrees * PI / 180.f;
}

//////////////////////////////
// Intervals
//
bool IsInInterval(f32 min, f32 max, f32 x)
{
    return x >= min && x <= max;
}

static vec3 BACKGROUND;

bool IsValidRay(Ray *r)
{
    return r->time() != (f32)U32Max;
}

inline vec3 LinearToSRGB(const vec3 &v)
{
    return vec3(sqrt(v.x), sqrt(v.y), sqrt(v.z));
}

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
        s = 1.055f * pow(l, 1.0f / 2.4f) - 0.055f;
    }
    return s;
}

struct Material;

AABB Transform(const mat4 &mat, const AABB &aabb)
{
    AABB result;
    vec3 vecs[] = {
        mul(mat, vec3(aabb.minX, aabb.minY, aabb.minZ)),
        mul(mat, vec3(aabb.maxX, aabb.minY, aabb.minZ)),
        mul(mat, vec3(aabb.maxX, aabb.maxY, aabb.minZ)),
        mul(mat, vec3(aabb.minX, aabb.maxY, aabb.minZ)),
        mul(mat, vec3(aabb.minX, aabb.minY, aabb.maxZ)),
        mul(mat, vec3(aabb.maxX, aabb.minY, aabb.maxZ)),
        mul(mat, vec3(aabb.maxX, aabb.maxY, aabb.maxZ)),
        mul(mat, vec3(aabb.minX, aabb.maxY, aabb.maxZ)),
    };

    for (u32 i = 0; i < ArrayLength(vecs); i++)
    {
        vec3 &p     = vecs[i];
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
    light.type      = PrimitiveType::Quad;
    light.primitive = quad;
    return light;
}

Light CreateSphereLight(Sphere *sphere)
{
    Light light;
    light.type      = PrimitiveType::Sphere;
    light.primitive = sphere;
    return light;
}

vec3 GenerateLightSample(const Light *light, const vec3 &origin)
{
    switch (light->type)
    {
        case PrimitiveType::Quad:
        {
            Quad *quad = (Quad *)light->primitive;
            return quad->Random(origin);
        }
        case PrimitiveType::Sphere:
        {
            Sphere *sphere = (Sphere *)light->primitive;
            return sphere->Random(origin);
        }
        default: assert(0); return vec3(1, 0, 0);
    }
}

f32 GetLightPDFValue(const Light *light, const vec3 &origin, const vec3 &direction)
{
    switch (light->type)
    {
        case PrimitiveType::Quad:
        {
            Quad *quad = (Quad *)light->primitive;
            return quad->PdfValue(origin, direction);
        }
        case PrimitiveType::Sphere:
        {
            Sphere *sphere = (Sphere *)light->primitive;
            return sphere->PdfValue(origin, direction);
        }
        default: assert(0); return 0;
    }
}

f32 GetLightsPDFValue(const Light *lights, const u32 numLights, const vec3 &origin, const vec3 &direction)
{
    f32 pdfSum = 0.f;
    for (u32 i = 0; i < numLights; i++)
    {
        pdfSum += GetLightPDFValue(&lights[i], origin, direction);
    }
    pdfSum /= numLights;
    return pdfSum;
}

vec3 GenerateSampleFromLights(const Light *lights, const u32 numLights, const vec3 &origin)
{
    i32 randomIndex = RandomInt(0, numLights);
    assert(randomIndex < (i32)numLights);
    vec3 result = GenerateLightSample(&lights[randomIndex], origin);
    return result;
}

inline vec3 GenerateCosSample(vec3 normal)
{
    Basis basis = GenerateBasis(normal);
    vec3 cosDir = ConvertToLocal(&basis, RandomCosineDirection());
    return cosDir;
}

inline f32 GetCosPDFValue(vec3 dir, vec3 normal)
{
    f32 cosTheta = dot(normalize(dir), normal);
    f32 cosPdf   = fmax(cosTheta / PI, 0.f);
    return cosPdf;
}

struct Perlin
{
    // f32 *randFloat;
    vec3 *randVec;
    i32 *permX;
    i32 *permY;
    i32 *permZ;
    static const i32 pointCount = 256;

    void Init()
    {
        randVec = new vec3[pointCount];
        for (i32 i = 0; i < pointCount; i++)
        {
            randVec[i] = normalize(RandomVec3(-1, 1));
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

    f32 Noise(const vec3 &p) const
    {
        f32 u = p.x - floor(p.x);
        f32 v = p.y - floor(p.y);
        f32 w = p.z - floor(p.z);

        vec3 c[2][2][2];
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
                        vec3 weightV(u - i, v - j, w - k);
                        accum += (i * uu + (1 - i) * (1 - uu)) *
                                 (j * vv + (1 - j) * (1 - vv)) *
                                 (k * ww + (1 - k) * (1 - ww)) *
                                 dot(c[i][j][k], weightV);
                    }
                }
            }
        }
        return accum;
    }

    f32 Turbulence(vec3 p, i32 depth) const
    {
        f32 accum  = 0.0;
        f32 weight = 1.0;
        for (i32 i = 0; i < depth; i++)
        {
            accum += weight * Noise(p);
            weight *= 0.5;
            p *= 2;
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

    static Texture CreateSolid(const vec3 &albedo)
    {
        Texture texture;
        texture.baseColor = albedo;
        texture.type      = Type::Solid;
        return texture;
    }
    static Texture CreateCheckered(f32 scale, const vec3 &even, const vec3 &odd)
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

    vec3 Value(const f32 u, const f32 v, const vec3 &p) const
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
                assert(image.width);
                assert(image.height);
                i32 x = i32(u * image.width);
                i32 y = i32((1 - v) * image.height);

                u8 *data    = GetColor(&image, x, y);
                f32 divisor = 1 / 255.f;
                f32 r       = f32(data[0]) * divisor;
                f32 g       = f32(data[1]) * divisor;
                f32 b       = f32(data[2]) * divisor;
                return vec3(r, g, b);
            }
            break;
            case Type::Noise:
            {
                return vec3(.5f, .5f, .5f) * (1.f + sinf(scale * p.z + 10.f * perlin.Turbulence(p, 7)));
            }
            break;
            default: assert(0); return vec3(0, 0, 0);
        }
    }

    vec3 baseColor;

    // checkered
    vec3 baseColor2;
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
    vec3 albedo;
    f32 fuzz;
    f32 refractiveIndex;

    Texture texture;

    static Material CreateLambert(vec3 inAlbedo)
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

    static Material CreateMetal(vec3 inAlbedo, f32 inFuzz = 0.0)
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

    static Material CreateDiffuseLight(vec3 inAlbedo)
    {
        Material result;
        result.type    = MaterialType::DiffuseLight;
        result.texture = Texture::CreateSolid(inAlbedo);
        return result;
    }

    static Material CreateIsotropic(const vec3 &albedo)
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

    bool LambertScatter(const Ray &r, const HitRecord &record, ScatterRecord &sRec)
    {
        sRec.attenuation = texture.Value(record.u, record.v, record.p);
        sRec.skipPDFRay  = Ray(INVALID_VEC, INVALID_VEC, (f32)U32Max);
        sRec.sample      = GenerateCosSample(record.normal);
        return true;
    }

    bool MetalScatter(const Ray &r, const HitRecord &record, ScatterRecord &sRec)
    {
        vec3 reflectDir  = Reflect(r.direction(), record.normal);
        reflectDir       = normalize(reflectDir) + fuzz * RandomUnitVector();
        sRec.attenuation = albedo;
        sRec.skipPDFRay  = Ray(record.p, reflectDir, r.time());
        return true;
    }

    bool DielectricScatter(const Ray &r, const HitRecord &record, ScatterRecord &sRec)
    {
        sRec.attenuation = vec3(1, 1, 1);
        f32 ri           = record.isFrontFace ? 1.f / refractiveIndex : refractiveIndex;

        vec3 rayDir  = normalize(r.direction());
        f32 cosTheta = fmin(dot(-rayDir, record.normal), 1.f);
        f32 sinTheta = sqrt(1 - cosTheta * cosTheta);
        // total internal reflection
        bool cannotRefract = ri * sinTheta > 1.f;

        f32 f0          = (1 - ri) / (1 + ri);
        f0              = f0 * f0;
        f32 reflectance = f0 + (1 - f0) * powf(1 - cosTheta, 5.f);
        vec3 direction  = cannotRefract || reflectance > RandomFloat()
                              ? Reflect(rayDir, record.normal)
                              : Refract(rayDir, record.normal, ri);
        sRec.skipPDFRay = Ray(record.p, direction, r.time());

        return true;
    }

    bool IsotropicScatter(const Ray &r, const HitRecord &record, ScatterRecord &sRec)
    {
        // scatteredRay     = Ray(record.p, RandomUnitVector(), r.time());
        sRec.attenuation = texture.Value(record.u, record.v, record.p);
        sRec.skipPDFRay  = Ray(INVALID_VEC, INVALID_VEC, (f32)U32Max);
        sRec.sample      = RandomUnitVector();
        return true;
    }

    inline bool Scatter(const Ray &r, const HitRecord &record, ScatterRecord &sRec)
    {
        switch (type)
        {
            case MaterialType::Lambert: return LambertScatter(r, record, sRec);
            case MaterialType::Metal: return MetalScatter(r, record, sRec);
            case MaterialType::Dielectric: return DielectricScatter(r, record, sRec);
            case MaterialType::Isotropic: return IsotropicScatter(r, record, sRec);
            default: return false;
        }
    }

    inline vec3 Emitted(const Ray &r, const HitRecord &record, f32 u, f32 v, const vec3 &p) const
    {
        switch (type)
        {
            case MaterialType::DiffuseLight:
            {
                if (!record.isFrontFace)
                {
                    return vec3(0, 0, 0);
                }
                return texture.Value(u, v, p);
            }
            break;
            default: return vec3(0, 0, 0);
        }
    }

    inline f32 ScatteringPDF(const Ray &r, const HitRecord &record, Ray &scattered)
    {
        switch (type)
        {
            case MaterialType::Lambert:
            {
                f32 cosTheta = dot(record.normal, normalize(scattered.direction()));
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
vec3 RayColor(const Ray &r, const int depth, const BVH &bvh)
{
    if (depth <= 0)
        return vec3(0, 0, 0);

    vec3 sphereCenter = vec3(0, 0, -1);
    HitRecord record;

    if (bvh.Hit(r, 0.001f, infinity, record))
    {
        Ray scattered;
        vec3 attenuation;
        if (record.material->Scatter(r, record, attenuation, scattered))
        {
            return attenuation * RayColor(scattered, depth - 1, bvh);
        }
        return vec3(0, 0, 0);
    }

    const vec3 normalizedDirection = normalize(r.direction());
    f32 t                          = 0.5f * (normalizedDirection.y + 1.f);
    return (1 - t) * vec3(1, 1, 1) + t * vec3(0.5f, 0.7f, 1.f);
}
#else
vec3 RayColor(const Ray &r, const int depth, const Primitive &bvh, const Light *lights, const u32 numLights)
{
    if (depth <= 0)
        return vec3(0, 0, 0);

    vec3 sphereCenter = vec3(0, 0, -1);
    HitRecord record;

    if (!bvh.Hit(r, 0.001f, infinity, record))
        return BACKGROUND;

    ScatterRecord sRec;
    // Ray scattered;
    vec3 emissiveColor = record.material->Emitted(r, record, record.u, record.v, record.p);
    if (!record.material->Scatter(r, record, sRec))
        return emissiveColor;

    if (IsValidRay(&sRec.skipPDFRay))
        return sRec.attenuation * RayColor(sRec.skipPDFRay, depth - 1, bvh, lights, numLights);

    // Cosine importance sampling
    // TODO: this is hardcoded, this should be switched on based on the type of pdf
    f32 cosPdf = GetCosPDFValue(sRec.sample, record.normal);

    // Light importance sampling
    vec3 randLightDir = GenerateSampleFromLights(lights, numLights, record.p);

    vec3 scatteredDir;
    if (RandomFloat() < 0.5f)
    {
        scatteredDir = sRec.sample;
    }
    else
    {
        scatteredDir = randLightDir;
    }
    Ray scattered = Ray(record.p, scatteredDir, r.time());
    f32 lightPdf  = GetLightsPDFValue(lights, numLights, record.p, scatteredDir);
    f32 pdf       = 0.5f * lightPdf + 0.5f * cosPdf;

    f32 scatteringPDF = record.material->ScatteringPDF(r, record, scattered);
    vec3 scatterColor = (sRec.attenuation * scatteringPDF * RayColor(scattered, depth - 1, bvh, lights, numLights)) / pdf;
    return emissiveColor + scatterColor;
}
#endif

bool RenderTile(WorkQueue *queue)
{
    u64 workItemIndex = InterlockedAdd(&queue->workItemIndex, 1);
    if (workItemIndex >= queue->workItemCount) return false;

    WorkItem *item = &queue->workItems[workItemIndex];

    i32 samplesPerPixel            = queue->params->samplesPerPixel;
    u32 squareRootSamplesPerPixel  = queue->params->squareRootSamplesPerPixel;
    f32 oneOverSqrtSamplesPerPixel = 1.f / squareRootSamplesPerPixel;
    vec3 cameraCenter              = queue->params->cameraCenter;

    for (u32 height = item->startY; height < item->onePastEndY; height++)
    {
        u32 *out = GetPixelPointer(queue->params->image, item->startX, height);
        for (u32 width = item->startX; width < item->onePastEndX; width++)
        {
            vec3 pixelColor(0, 0, 0);

            // for (i32 i = 0; i < samplesPerPixel; i++)
            for (u32 i = 0; i < squareRootSamplesPerPixel; i++)
            {
                for (u32 j = 0; j < squareRootSamplesPerPixel; j++)
                {
                    // const vec3 offset      = vec3(RandomFloat() - 0.5f, RandomFloat() - 0.5f, 0.f);
                    const f32 offsetX      = ((i + RandomFloat()) * oneOverSqrtSamplesPerPixel) - 0.5f;
                    const f32 offsetY      = ((j + RandomFloat()) * oneOverSqrtSamplesPerPixel) - 0.5f;
                    const vec3 offset      = vec3(offsetX, offsetY, 0.f);
                    const vec3 pixelSample = queue->params->pixel00 + ((width + offset.x) * queue->params->pixelDeltaU) +
                                             ((height + offset.y) * queue->params->pixelDeltaV);
                    vec3 rayOrigin;
                    if (queue->params->defocusAngle <= 0)
                    {
                        rayOrigin = cameraCenter;
                    }
                    else
                    {
                        vec3 sample = RandomInUnitDisk();
                        rayOrigin   = cameraCenter + sample[0] * queue->params->defocusDiskU +
                                    sample[1] * queue->params->defocusDiskV;
                    }
                    const vec3 rayDirection = pixelSample - rayOrigin;
                    const f32 rayTime       = RandomFloat();
                    Ray r(rayOrigin, rayDirection, rayTime);

                    pixelColor += RayColor(r, queue->params->maxDepth, queue->params->bvh,
                                           queue->params->lights, queue->params->numLights);
                }
            }

            pixelColor /= (f32)samplesPerPixel;

            // NOTE: lazy NAN check
            if (pixelColor.r != pixelColor.r) pixelColor.r = 0.f;
            if (pixelColor.g != pixelColor.g) pixelColor.g = 0.f;
            if (pixelColor.b != pixelColor.b) pixelColor.b = 0.f;

            f32 r = 255.f * ExactLinearToSRGB(pixelColor.r);
            f32 g = 255.f * ExactLinearToSRGB(pixelColor.g);
            f32 b = 255.f * ExactLinearToSRGB(pixelColor.b);
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

struct Sample
{
    f32 x;
    f32 pX;
};
bool CompareByX(const Sample &a, const Sample &b)
{
    return a.x < b.x;
}

int main(int argc, char *argv[])
{
#if 0
    const u32 n = 1;
    f32 sum     = 0.f;
    for (u32 i = 0; i < n; i++)
    {
        f32 x = 2.f * powf(RandomFloat(), 1.f / 3.f); // inverse of cdf
        sum += x * x / (3.f / 8.f * x * x);
    }
    printf("I = %f\n", sum / n);
#endif

#if 1
#if SPHERES
    const f32 aspectRatio     = 16.f / 9.f;
    const vec3 lookFrom       = vec3(13, 2, 3);
    const vec3 lookAt         = vec3(0, 0, 0);
    const vec3 worldUp        = vec3(0, 1, 0);
    const f32 verticalFov     = 20;
    const f32 defocusAngle    = 0.6f;
    const f32 focusDist       = 10;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    const int imageWidth      = 400;
    BACKGROUND                = vec3(0.7f, 0.8f, 1.f);
#elif EARTH
    const f32 aspectRatio     = 16.f / 9.f;
    const vec3 lookFrom       = vec3(0, 0, 12);
    const vec3 lookAt         = vec3(0, 0, 0);
    const vec3 worldUp        = vec3(0, 1, 0);
    const f32 verticalFov     = 20;
    const f32 defocusAngle    = 0;
    const f32 focusDist       = 10;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    const int imageWidth      = 400;
    BACKGROUND                = vec3(0.7f, 0.8f, 1.f);
#elif PERLIN
    const f32 aspectRatio     = 16.f / 9.f;
    const vec3 lookFrom       = vec3(13, 2, 3);
    const vec3 lookAt         = vec3(0, 0, 0);
    const vec3 worldUp        = vec3(0, 1, 0);
    const f32 verticalFov     = 20;
    const f32 defocusAngle    = 0;
    const f32 focusDist       = 10;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    const int imageWidth      = 400;
    BACKGROUND                = vec3(0.7f, 0.8f, 1.f);
#elif QUADS
    const f32 aspectRatio     = 1.f;
    const vec3 lookFrom       = vec3(0, 0, 9);
    const vec3 lookAt         = vec3(0, 0, 0);
    const vec3 worldUp        = vec3(0, 1, 0);
    const f32 verticalFov     = 80;
    const f32 defocusAngle    = 0;
    const f32 focusDist       = 10;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    const int imageWidth      = 400;
    BACKGROUND                = vec3(0.7f, 0.8f, 1.f);
#elif LIGHTS
    const f32 aspectRatio     = 16.f / 9.f;
    const vec3 lookFrom       = vec3(26, 3, 6);
    const vec3 lookAt         = vec3(0, 2, 0);
    const vec3 worldUp        = vec3(0, 1, 0);
    const f32 verticalFov     = 20;
    const f32 defocusAngle    = 0;
    const f32 focusDist       = 10;
    const int imageWidth      = 400;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    BACKGROUND                = vec3(0, 0, 0);
#elif CORNELL
    const f32 aspectRatio  = 1.f;
    const vec3 lookFrom    = vec3(278, 278, -800);
    const vec3 lookAt      = vec3(278, 278, 0);
    const vec3 worldUp     = vec3(0, 1, 0);
    const f32 verticalFov  = 40;
    const f32 defocusAngle = 0;
    const f32 focusDist    = 10;

    const int imageWidth      = 600;
    const int samplesPerPixel = 100;
    const int maxDepth        = 50;
    BACKGROUND                = vec3(0, 0, 0);
#elif CORNELL_SMOKE
    const f32 aspectRatio  = 1.0;
    const vec3 lookFrom    = vec3(278, 278, -800);
    const vec3 lookAt      = vec3(278, 278, 0);
    const vec3 worldUp     = vec3(0, 1, 0);
    const f32 verticalFov  = 40;
    const f32 defocusAngle = 0;
    const f32 focusDist    = 10;

    const u32 imageWidth      = 600;
    const u32 samplesPerPixel = 200;
    const u32 maxDepth        = 50;
    BACKGROUND                = vec3(0, 0, 0);
#elif FINAL
    const f32 aspectRatio  = 1.0;
    const vec3 lookFrom    = vec3(478, 278, -600);
    const vec3 lookAt      = vec3(278, 278, 0);
    const vec3 worldUp     = vec3(0, 1, 0);
    const f32 verticalFov  = 40;
    const f32 defocusAngle = 0;
    const f32 focusDist    = 10;

    const int imageWidth      = 400;
    const int samplesPerPixel = 250;
    const int maxDepth        = 4;
    BACKGROUND                = vec3(0, 0, 0);
#endif

    u32 imageHeight = u32(imageWidth / aspectRatio);
    imageHeight     = imageHeight < 1 ? 1 : imageHeight;
    f32 focalLength = (lookFrom - lookAt).length();
    f32 theta       = DegreesToRadians(verticalFov);
    f32 h           = tan(theta / 2);

    vec3 f = normalize(lookFrom - lookAt);
    vec3 s = cross(worldUp, f);
    vec3 u = cross(f, s);

    f32 viewportHeight = 2 * h * focusDist;
    f32 viewportWidth  = viewportHeight * (f32(imageWidth) / imageHeight);
    vec3 cameraCenter  = lookFrom;

    vec3 viewportU = viewportWidth * s;
    vec3 viewportV = viewportHeight * -u;

    vec3 pixelDeltaU = viewportU / imageWidth;
    vec3 pixelDeltaV = viewportV / (f32)imageHeight;

    vec3 viewportUpperLeft = cameraCenter - focusDist * f - viewportU / 2 - viewportV / 2;
    vec3 pixel00           = viewportUpperLeft + 0.5f * (pixelDeltaU + pixelDeltaV);

    f32 defocusRadius = focusDist * tan(DegreesToRadians(defocusAngle / 2));
    vec3 defocusDiskU = defocusRadius * s;
    vec3 defocusDiskV = defocusRadius * u;

    u32 squareRootSamplesPerPixel = (u32)sqrt(samplesPerPixel);

    Scene scene = {};

#if SPHERES
    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            f32 chooseMat = RandomFloat();
            vec3 center(a + 0.9f * RandomFloat(), 0.2f, b + 0.9f * RandomFloat());

            if ((center - vec3(4, 0.2f, 0)).length() > 0.9f)
            {
                Material *material = (Material *)malloc(sizeof(Material));
                // Diffuse
                if (chooseMat < 0.8)
                {
                    vec3 albedo  = RandomVec3() * RandomVec3();
                    vec3 center2 = center + vec3(0, RandomFloat(0, .5f), 0);
                    *material    = Material::CreateLambert(albedo);
                    scene.Add(Sphere(center, center2, 0.2f, material));
                }
                // Metal
                else if (chooseMat < 0.95)
                {
                    vec3 albedo = RandomVec3(0.5f, 1);
                    f32 fuzz    = RandomFloat(0, 0.5f);
                    *material   = Material::CreateMetal(albedo, fuzz);
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

    Texture checkered    = Texture::CreateCheckered(0.32f, vec3(.2f, .3f, .1f), vec3(.9f, .9f, .9f));
    Material materials[] = {
        Material::CreateDielectric(1.5f),
        Material::CreateLambert(vec3(0.4f, 0.2f, 0.1f)),
        Material::CreateMetal(vec3(0.7f, 0.6f, 0.5f), 0.f),
        Material::CreateLambert(&checkered),
    };

    scene.Add(Sphere(vec3(0, 1, 0), 1.f, &materials[0]));
    scene.Add(Sphere(vec3(-4, 1, 0), 1.f, &materials[1]));
    scene.Add(Sphere(vec3(4, 1, 0), 1.f, &materials[2]));

    // ground
    scene.Add(Sphere(vec3(0, -1000, 0), 1000, &materials[3]));

#elif EARTH
    Texture earth             = Texture::CreateImage("earthmap.jpg");
    Material surface          = Material::CreateLambert(&earth);

    scene.Add(Sphere(vec3(0, 0, 0), 2, &surface));
#elif PERLIN
    Texture noise             = Texture::CreateNoise(4.0);
    Material perlin           = Material::CreateLambert(&noise);

    scene.Add(Sphere(vec3(0, -1000, 0), 1000, &perlin));
    scene.Add(Sphere(vec3(0, 2, 0), 2, &perlin));
#elif QUADS
    Material materials[]      = {
        Material::CreateLambert(vec3(1.f, 0.2f, 0.2f)),
        Material::CreateLambert(vec3(0.2f, 1.f, 0.2f)),
        Material::CreateLambert(vec3(0.2f, 0.2f, 1.f)),
        Material::CreateLambert(vec3(1.f, 0.5f, 0.f)),
        Material::CreateLambert(vec3(0.2f, 0.8f, 0.8f)),
    };

    scene.Add(Quad(vec3(-3, -2, 5), vec3(0, 0, -4), vec3(0, 4, 0), &materials[0]));
    scene.Add(Quad(vec3(-2, -2, 0), vec3(4, 0, 0), vec3(0, 4, 0), &materials[1]));
    scene.Add(Quad(vec3(3, -2, 1), vec3(0, 0, 4), vec3(0, 4, 0), &materials[2]));
    scene.Add(Quad(vec3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4), &materials[3]));
    scene.Add(Quad(vec3(-2, -3, 5), vec3(4, 0, 0), vec3(0, 0, -4), &materials[4]));
#elif LIGHTS
    Texture texture           = Texture::CreateNoise(4);
    Material lambert          = Material::CreateLambert(&texture);
    scene.Add(Sphere(vec3(0, -1000, 0), 1000, &lambert));
    scene.Add(Sphere(vec3(0, 2, 0), 2, &lambert));

    Material diffuse = Material::CreateDiffuseLight(vec3(4, 4, 4));
    scene.Add(Sphere(vec3(0, 7, 0), 2, &diffuse));
    scene.Add(Quad(vec3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0), &diffuse));
#elif CORNELL
    Material materials[]      = {
        Material::CreateLambert(vec3(.65f, .05f, .05f)),
        Material::CreateLambert(vec3(.73f, .73f, .73f)),
        Material::CreateLambert(vec3(.12f, .45f, .15f)),
        Material::CreateDiffuseLight(vec3(15, 15, 15)),
        Material::CreateMetal(vec3(.8f, .85f, .88f), 0.f),
        Material::CreateDielectric(1.5f),
    };

    Quad quads[] = {
        Quad(vec3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), &materials[2]),
        Quad(vec3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), &materials[0]),
        Quad(vec3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), &materials[3]),
        Quad(vec3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), &materials[1]),
        Quad(vec3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), &materials[1]),
        Quad(vec3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), &materials[1]),
    };

    Box boxes[] = {
        Box(vec3(0, 0, 0), vec3(165, 330, 165), &materials[1]),
        // Box(vec3(0, 0, 0), vec3(165, 165, 165), &materials[1]),
    };

    Sphere spheres[] = {
        Sphere(vec3(190, 90, 190), 90, &materials[5]),
    };

    HomogeneousTransform transforms[] = {
        {vec3(265, 0, 295), DegreesToRadians(15)},
        {vec3(130, 0, 65), DegreesToRadians(-18)}

    };

    Light lights[] = {
        CreateQuadLight(&quads[2]),
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
    scene.AddTransform(PrimitiveType::Box, 0, 0);
    // scene.AddTransform(PrimitiveType::Box, 1, 1);
#elif CORNELL_SMOKE
    Material materials[]      = {
        Material::CreateLambert(vec3(.65f, .05f, .05f)),
        Material::CreateLambert(vec3(.73f, .73f, .73f)),
        Material::CreateLambert(vec3(.12f, .45f, .15f)),
        Material::CreateDiffuseLight(vec3(7, 7, 7)),
        Material::CreateIsotropic(vec3(0, 0, 0)),
        Material::CreateIsotropic(vec3(1, 1, 1)),
    };

    Quad quads[] = {
        Quad(vec3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), &materials[2]),
        Quad(vec3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), &materials[0]),
        Quad(vec3(113, 554, 127), vec3(330, 0, 0), vec3(0, 0, 305), &materials[3]),
        Quad(vec3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), &materials[1]),
        Quad(vec3(0, 555, 0), vec3(555, 0, 0), vec3(0, 0, 555), &materials[1]),
        Quad(vec3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), &materials[1]),
    };

    Box boxes[] = {
        Box(vec3(0, 0, 0), vec3(165, 330, 165), &materials[1]),
        Box(vec3(0, 0, 0), vec3(165, 165, 165), &materials[1]),
    };

    HomogeneousTransform transforms[] = {
        {vec3(265, 0, 295), DegreesToRadians(15)},
        {vec3(130, 0, 65), DegreesToRadians(-18)},
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
    scene.AddConstantMedium(PrimitiveType::Box, 0, 0);
    scene.AddTransform(PrimitiveType::Box, 0, 0);

    scene.AddConstantMedium(PrimitiveType::Box, 1, 1);
    scene.AddTransform(PrimitiveType::Box, 1, 1);

#elif FINAL
    Texture texture           = Texture::CreateImage("earthmap.jpg");
    Texture noise             = Texture::CreateNoise(0.2f);
    Material materials[]{
        Material::CreateLambert(vec3(0.48f, 0.83f, 0.53f)),
        Material::CreateDiffuseLight(vec3(7, 7, 7)),
        Material::CreateLambert(vec3(0.7f, 0.3f, 0.1f)),
        Material::CreateDielectric(1.5f),
        Material::CreateMetal(vec3(0.8f, 0.8f, 0.9f), 1.f),
        Material::CreateIsotropic(vec3(0.2f, 0.4f, 0.9f)),
        Material::CreateIsotropic(vec3(1, 1, 1)),
        Material::CreateLambert(&texture),
        Material::CreateLambert(&noise),
        Material::CreateLambert(vec3(.73f, .73f, .73f)),
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

            boxes[i * boxesPerSide + j] = Box(vec3(x0, y0, z0), vec3(x1, y1, z1), &materials[0]);
        }
    }

    Quad quads[] = {
        Quad(vec3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), &materials[1]),
    };

    vec3 center1 = vec3(400, 400, 200);
    vec3 center2 = center1 + vec3(30, 0, 0);

    const i32 ns           = 1000;
    Sphere spheres[8 + ns] = {
        Sphere(center1, center2, 50, &materials[2]),
        Sphere(vec3(260, 150, 45), 50, &materials[3]),
        Sphere(vec3(0, 150, 145), 50, &materials[4]),
        Sphere(vec3(360, 150, 145), 70, &materials[3]),
        Sphere(vec3(360, 150, 145), 70, &materials[3]),
        Sphere(vec3(0, 0, 0), 5000, &materials[3]),
        Sphere(vec3(400, 200, 400), 100, &materials[7]),
        Sphere(vec3(220, 280, 300), 80, &materials[8]),
    };

    HomogeneousTransform transforms[] = {
        {vec3(-100, 270, 395), DegreesToRadians(15)},
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

    scene.AddConstantMedium(PrimitiveType::Sphere, 4, 0);
    scene.AddConstantMedium(PrimitiveType::Sphere, 5, 1);

    for (i32 j = 8; j < ArrayLength(spheres); j++)
    {
        scene.AddTransform(PrimitiveType::Sphere, j, 0);
    }

#endif

    statistics.rayAABBTests      = 0;
    statistics.rayPrimitiveTests = 0;

    BVH bvh;
    bvh.Build(&scene, 2);

    // CompressedBVH4 compressedBVH = CreateCompressedBVH4(&bvh);
    BVH4 bvh4 = CreateBVH4(&bvh);

    RenderParams params;
    params.pixel00                   = pixel00;
    params.pixelDeltaU               = pixelDeltaU;
    params.pixelDeltaV               = pixelDeltaV;
    params.cameraCenter              = cameraCenter;
    params.defocusDiskU              = defocusDiskU;
    params.defocusDiskV              = defocusDiskV;
    params.defocusAngle              = defocusAngle;
    params.bvh                       = &bvh4; 
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

    assert(queue.workItemCount == workItemTotal);

    clock_t start = clock();
    for (u32 i = 0; i < GetCPUCoreCount(); i++)
    {
        CreateWorkThread(&queue);
    }

    while (queue.tilesFinished < workItemTotal)
    {
        fprintf(stderr, "\rRaycasting %d%%...    ", 100 * (u32)queue.tilesFinished / workItemTotal);
        fflush(stdout);
        RenderTile(&queue);
    }
    clock_t end = clock();

    fprintf(stderr, "\n");
    printf("Total time: %dms\n", end - start);
    WriteImage(&image, "image.bmp");
    // printf("Total ray AABB tests: %lld\n", statistics.rayAABBTests.load());
    // printf("Total ray primitive tests: %lld\n", statistics.rayPrimitiveTests.load());
    fprintf(stderr, "Done.");
#endif
}
