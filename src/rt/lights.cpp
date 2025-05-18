#include "lights.h"
#include "color.h"
#include "parallel.h"
#include "scene.h"
#include "spectrum.h"

namespace rt
{

LightSample::LightSample(const ShapeSample &sample, const SampledSpectrum &L, LightType type)
    : samplePoint(sample.p), wi(sample.w), n(sample.n), L(L), pdf(sample.pdf), lightType(type)
{
}

Vec3f EqualAreaSquareToSphere(Vec2f p)
{
    Assert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
    // Transform _p_ to $[-1,1]^2$ and compute absolute values
    float u = 2.f * p.x - 1.f, v = 2.f * p.y - 1.f;
    float up = Abs(u), vp = Abs(v);

    // Compute radius _r_ as signed distance from diagonal
    float signedDistance = 1.f - (up + vp);
    float d              = Abs(signedDistance);
    float r              = 1.f - d;

    // Compute angle $\phi$ for square to sphere mapping
    float phi = (r == 0 ? 1 : (vp - up) / r + 1.f) * PI / 4.f;

    // Find $z$ coordinate for spherical direction
    float z = Copysign(1 - Sqr(r), signedDistance);

    // Compute $\cos\phi$ and $\sin\phi$ for original quadrant and return vector
    float cosPhi = Copysign(Cos(phi), u);
    float sinPhi = Copysign(Sin(phi), v);
    return Vec3f(cosPhi * r * SafeSqrt(2.f - Sqr(r)), sinPhi * r * SafeSqrt(2.f - Sqr(r)), z);
}

// Via source code from Clarberg: Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD
Vec2f EqualAreaSphereToSquare(Vec3f d)
{
    Assert(LengthSquared(d) > .999 && LengthSquared(d) < 1.001);
    float x = Abs(d.x), y = Abs(d.y), z = Abs(d.z);

    // Compute the radius r
    float r = SafeSqrt(1 - z); // r = sqrt(1-|z|)

    // Compute the argument to atan (detect a=0 to avoid div-by-zero)
    float a = Max(x, y), b = Min(x, y);
    b = a == 0 ? 0 : b / a;

    // Polynomial approximation of atan(x)*2/pi, x=b
    // Coefficients for 6th degree minimax approximation of atan(x)*2/pi,
    // x=[0,1].
    const float t1 = 0.406758566246788489601959989e-5;
    const float t2 = 0.636226545274016134946890922156;
    const float t3 = 0.61572017898280213493197203466e-2;
    const float t4 = -0.247333733281268944196501420480;
    const float t5 = 0.881770664775316294736387951347e-1;
    const float t6 = 0.419038818029165735901852432784e-1;
    const float t7 = -0.251390972343483509333252996350e-1;
    float phi      = EvaluatePolynomial(b, t1, t2, t3, t4, t5, t6, t7);

    // Extend phi if the input is in the range 45-90 degrees (u<v)
    if (x < y) phi = 1 - phi;

    // Find (u,v) based on (r,phi)
    float v = phi * r;
    float u = r - v;

    if (d.z < 0)
    {
        // southern hemisphere -> mirror u,v
        Swap(u, v);
        u = 1 - u;
        v = 1 - v;
    }

    // Move (u,v) to the correct quadrant based on the signs of (x,y)
    u = Copysign(u, d.x);
    v = Copysign(v, d.y);

    // Transform (u,v) from [-1,1] to [0,1]
    return Vec2f(0.5f * (u + 1), 0.5f * (v + 1));
}

// NOTE: sample (over solid angle) the spherical rectangle obtained by projecting a planar
// rectangle onto the unit sphere centered at point p
// https://blogs.autodesk.com/media-and-entertainment/wp-content/uploads/sites/162/egsr2013_spherical_rectangle.pdf
// TODO: simd sin, cos, and arcsin

//////////////////////////////
// Diffuse Area Light
//
DiffuseAreaLight::DiffuseAreaLight(const DenselySampledSpectrum *Lemit,
                                   AffineSpace *renderFromLight, f32 scale, int geomID,
                                   int sceneID, LightType type)
    : Lemit(Lemit), renderFromLight(renderFromLight), scale(scale), geomID(geomID),
      sceneID(sceneID), type(type)
{
    luminance = SpectrumToPhotometric(*Lemit);
}

LightSample DiffuseAreaLight::SampleLi(SurfaceInteraction &intr, Vec2f &u,
                                       SampledWavelengths &lambda, bool allowIncompletePDF)
{
    ScenePrimitives **scenes = GetScenes();
    ShapeSample sample       = scenes[sceneID]->Sample(intr, renderFromLight, u, geomID);
    SampledSpectrum L        = Le(sample.n, sample.w, lambda);
    if (!L) return {};
    return LightSample(sample, L, type);
}

// TODO: I don't think this is going to be invoked for the moana scene
f32 DiffuseAreaLight::PDF_Li(const Vec3f &prevIntrP, const SurfaceInteraction &intr,
                             bool allowIncompletePDF)
{
    Assert(0);
    return 0.f;
}

SampledSpectrum DiffuseAreaLight::Le(const Vec3f &n, const Vec3f &w,
                                     const SampledWavelengths &lambda)
{
    if (Dot(n, w) < 0) return SampledSpectrum(0.f);
    return scale * Lemit->Sample(lambda);
}

// https://fpsunflower.github.io/ckulla/data/many-lights-hpg2018.pdf
// See page 8
f32 DiffuseAreaLight::Importance(const Vec3f &point, const Vec3f &n)
{
    Vec3f p[4];

    ScenePrimitives **scenes = GetScenes();
    Mesh *mesh               = (Mesh *)scenes[sceneID]->primitives + geomID;
    Assert(mesh->numVertices == 4 && !mesh->indices);

    p[0] = mesh->p[0];
    p[1] = mesh->p[1];
    p[2] = mesh->p[2];
    p[3] = mesh->p[3];
    if (renderFromLight)
    {
        for (int i = 0; i < 4; i++)
        {
            p[i] = TransformP(*renderFromLight, p[i]);
        }
    }
    p[0] -= point;
    p[1] -= point;
    p[2] -= point;
    p[3] -= point;

    f32 d = Dot(n, point);

    // TODO: differentiate between sphere vs hemisphere?
    // Clip to hemisphere by projecting onto plane
    for (int i = 0; i < 4; i++)
    {
        f32 signedDistance = Dot(p[i], n);
        p[i] -= Min(signedDistance, 0.f) * n;
    }

    f32 irradiance = 0.f;
    for (int i = 0; i < 4; i++)
    {
        int nextIndex = (i + 1) & 3;
        irradiance += AngleBetween(Normalize(p[i]), Normalize(p[nextIndex])) *
                      Max(Dot(Normalize(Cross(p[i], p[nextIndex])), n), 0.f);
    }
    irradiance *= .5 * InvPi * luminance * scale;
    return irradiance;
}

//////////////////////////////
// Distant Light
//

// SAMPLE_LI(DistantLight)
// {
//     const DistantLight *light = &scene->lights.Get<DistantLight>()[u32(lightIndices)];
//     Vec3f wi                  = -light->d;
//
//     return LightSample(light->scale * light->Lemit->Sample(lambda),
//                        Vec3f(intr.p) + 2.f * wi * light->sceneRadius, wi, 1.f,
//                        LightType::DeltaDirection);
// }

//////////////////////////////
// Infinite Lights
//
// SAMPLE_LI(UniformInfiniteLight)
// {
//     const UniformInfiniteLight *light =
//         &scene->lights.Get<UniformInfiniteLight>()[u32(lightIndices)];
//     if (allowIncompletePDF) return {};
//     Vec3f wi = SampleUniformSphere(u);
//     f32 pdf  = UniformSpherePDF();
//     return LightSample(light->scale * light->Lemit.Sample(lambda),
//                        Vec3f(intr.p) + 2.f * wi * light->sceneRadius, wi, pdf,
//                        LightType::Infinite);
// }

// PDF_LI_INF(UniformInfiniteLight) { return allowIncompletePDF ? 0.f : UniformSpherePDF(); }

// LE_INF(UniformInfiniteLight) { return light->scale * light->Lemit.Sample(lambda); }

ImageInfiniteLight::ImageInfiniteLight(Arena *arena, Image image,
                                       const AffineSpace *renderFromLight,
                                       const RGBColorSpace *imageColorSpace, f32 sceneRadius,
                                       f32 scale)
    : image(image), renderFromLight(renderFromLight), imageColorSpace(imageColorSpace),
      sceneRadius(sceneRadius), scale(scale)
{
    const f32 *values = image.GetSamplingDistribution(arena);
    distribution      = PiecewiseConstant2D(arena, values, image.width, image.height,
                                            Vec2f(0.f, 0.f), Vec2f(1.f, 1.f));

    u32 size     = image.width * image.height;
    f32 avg      = 0.f;
    f32 first    = values[0];
    bool allSame = true;

    for (u32 i = 0; i < size; i++)
    {
        if (allSame && i != 0 && values[i] != first)
        {
            allSame = false;
        }
        avg += values[i];
    }
    avg /= size;
    f32 *compensatedValues = PushArrayNoZero(arena, f32, size);
    if (allSame)
    {
        for (u32 i = 0; i < size; i++)
        {
            compensatedValues[i] = 1.f;
        }
    }
    else
    {
        for (u32 i = 0; i < size; i++)
        {
            compensatedValues[i] = Max(0.f, values[i] - avg);
        }
    }
    compensatedDistribution = PiecewiseConstant2D(
        arena, compensatedValues, image.width, image.height, Vec2f(0.f, 0.f), Vec2f(1.f, 1.f));

    CalculateSHFromEnvironmentMap();
}

void ImageInfiniteLight::CalculateSHFromEnvironmentMap()
{
    L2 c       = {};
    f32 weight = f32(FourPiTy()) / (image.width * image.height);

    ScratchArena scratch;

    int num = OS_NumProcessors();
    L2 *l2  = PushArray(scratch.temp.arena, L2, num);
    ParallelFor2D(Vec2i(0), Vec2i(image.width, image.height), Vec2i(32),
                  [&](int jobID, Vec2i start, Vec2i end) {
                      L2 c = {};
                      Assert(start.x >= 0 && start.y >= 0 && end.x >= 0 && end.y >= 0);
                      for (i32 h = start[1]; h < end[1]; h++)
                      {
                          for (i32 w = start[0]; w < end[0]; w++)
                          {
                              Vec2f uv(f32(w) / image.width, f32(h) / image.height);
                              Vec3f dir = EqualAreaSquareToSphere(uv);

                              Vec3f values = SRGBToLinear(GetColor(&image, w, h));

                              f32 lum  = CalculateLuminance(values) * scale;
                              L2 basis = EvaluateL2Basis(dir);

                              for (int i = 0; i < 9; i++)
                              {
                                  c[i] += lum * basis[i] * weight;
                              }
                          }
                      }
                      u32 index = GetThreadIndex();
                      l2[index] += c;
                  });

    for (int i = 0; i < num; i++)
    {
        c += l2[i];
    }
    coefficients = c;
}

SampledSpectrum ImageInfiniteLight::ImageLe(Vec2f uv, const SampledWavelengths &lambda) const
{
    Vec3f rgb = GetOctahedralRGB(&image, uv);

    RGBIlluminantSpectrum spec(*imageColorSpace, rgb);
    return scale * spec.Sample(lambda);
}

LightSample ImageInfiniteLight::SampleLi(SurfaceInteraction &intr, Vec2f &u,
                                         SampledWavelengths &lambda, bool allowIncompletePDF)
{
    Scene *scene = GetScene();

    f32 pdf;
    Vec2f uv;
    if (allowIncompletePDF)
    {
        uv = compensatedDistribution.Sample(u, &pdf);
    }
    else
    {
        uv = distribution.Sample(u, &pdf);
    }
    if (pdf == 0.f) return {};
    Vec3f wi = TransformV(*renderFromLight, EqualAreaSquareToSphere(uv));
    pdf /= 4 * PI;
    return LightSample(ImageLe(uv, lambda), Vec3f(intr.p) + 2.f * wi * sceneRadius, wi, pdf,
                       LightType::Infinite);
}

f32 ImageInfiniteLight::PDF_Li(const Vec3f &w, bool allowIncompletePDF)
{
    Vec3f wi = Normalize(ApplyInverseV(*renderFromLight, w));
    Vec2f uv = EqualAreaSphereToSquare(wi);
    f32 pdf;
    if (allowIncompletePDF)
    {
        pdf = compensatedDistribution.PDF(uv);
    }
    else
    {
        pdf = distribution.PDF(uv);
    }
    return pdf / (4 * PI);
}

SampledSpectrum ImageInfiniteLight::Le(const Vec3f &w, const SampledWavelengths &lambda)
{
    Vec3f wi = Normalize(ApplyInverseV(*renderFromLight, w));
    Vec2f uv = EqualAreaSphereToSquare(wi);
    return ImageLe(uv, lambda);
}

f32 ImageInfiniteLight::Importance(const Vec3f &p, const Vec3f &n)
{
    f32 irradiance = EvaluateIrradiance(coefficients, n);
    return irradiance;
}

// Light sampling
f32 UniformLightPDF(Scene *scene) { return 1.f / scene->lights.size(); }

Light *UniformLightSample(Scene *scene, f32 u, f32 *pmf = 0)
{
    size_t numLights = scene->lights.size();
    if (numLights == 0) return 0;
    size_t lightIndex = Min(size_t(u * numLights), numLights - 1);
    Assert(lightIndex >= 0 && lightIndex < numLights);
    if (pmf) *pmf = UniformLightPDF(scene);
    Light *light = scene->lights[lightIndex];
    ErrorExit(light, "lightIndex: %i, total: %llu\n", lightIndex, numLights);
    return light;
}

Light *ExhaustiveLightSample(Scene *scene, const Vec3f &p, const Vec3f &n, f32 u, f32 &pdf)
{
    f32 weight             = 0.f;
    Light *selectedLight   = 0;
    f32 selectedImportance = 0.f;

    for (auto &light : scene->lights)
    {
        f32 importance = light->Importance(p, n);
        weight += importance;
        f32 prob = importance / weight;
        if (u < prob)
        {
            selectedLight = light;
            u /= prob;
            selectedImportance = importance;
        }
        else
        {
            u = (u - prob) / (1 - prob);
        }
    }
    pdf = selectedImportance / weight;
    return selectedLight;
}

f32 ExhaustiveLightPDF(Scene *scene, Light *light, const Vec3f &p, const Vec3f &n)
{
    f32 selectedImportance = 0.f;
    f32 weight             = 0.f;
    for (auto &l : scene->lights)
    {
        f32 importance = light->Importance(p, n);
        weight += importance;
        if (l == light)
        {
            selectedImportance = importance;
        }
    }
    return selectedImportance / weight;
}

#define USE_EXHAUSTIVE_LIGHT_SAMPLER

Light *SampleLight(Scene *scene, SurfaceInteraction &intr, f32 u, f32 *pmf)
{
#ifdef USE_EXHAUSTIVE_LIGHT_SAMPLER
    return ExhaustiveLightSample(scene, intr.p, intr.n, u, *pmf);
#else
    return UniformLightSample(scene, u, pmf);
#endif
}

f32 LightPDF(Scene *scene, SurfaceInteraction &intr, Light *light)
{
#ifdef USE_EXHAUSTIVE_LIGHT_SAMPLER
    return ExhaustiveLightPDF(scene, light, intr.p, intr.n);
#else
    return UniformLightPDF(scene);
#endif
}

} // namespace rt
