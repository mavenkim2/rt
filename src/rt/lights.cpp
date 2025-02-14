#include "lights.h"
#include "color.h"

namespace rt
{

LightSample::LightSample(const ShapeSample &sample, const SampledSpectrum &L, LightType type)
    : samplePoint(sample.p), wi(sample.w), L(L), pdf(sample.pdf), lightType(type)
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

template <typename T>
T LinearPDF(T x, T a, T b)
{
    T mask   = x < 0 || x > 1;
    T result = Select(mask, T(0), 2 * Lerp(x, a, b) / (a + b));
    return result;
}

template <typename T>
T SampleLinear(T u, T a, T b)
{
    T mask = u == 0 && a == 0;
    T x    = Select(mask, T(0), u * (a + b) / (a + Sqrt(Lerp(u, a * a, b * b))));
    return Min(x, T(oneMinusEpsilon));
}

// p2 ---- p3
// |       |
// p0 ---- p1
template <typename T>
T Bilerp(const Vec2<T> &u, const Vec4<T> &w)
{
    T result = Lerp(u[0], Lerp(u[1], w[0], w[2]), Lerp(u[1], w[1], w[3]));
    return result;
}

//////////////////////////////
// Diffuse Area Light
//

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
    //     const DiffuseAreaLight *lights[IntN];
    //     for (u32 i = 0; i < IntN; i++)
    //     {
    //         lights[i] = &scene->GetAreaLights()[Get(lightIndices, i)];
    //     }
    //     Vec3lfn p[4];
    //     LaneNF32 area;
    //     // TODO: maybe have to spawn a ray??? but I feel like this is only called (at least
    //     for
    //     // now) when it already has intersected, and we need the pdf for MIS
    //     for (u32 i = 0; i < 4; i++)
    //     {
    //         Lane4F32 pI[IntN];
    //         for (u32 lightIndex = 0; lightIndex < IntN; lightIndex++)
    //         {
    //             const DiffuseAreaLight *light = lights[lightIndex];
    //
    //             pI[lightIndex]        = Lane4F32(TransformP(*light->renderFromLight,
    //             light->p[i])); Set(area, lightIndex) = light->area;
    //         }
    //         Transpose<IntN>(pI, p[i]);
    //     }
    //     Vec3lfn v00 = Normalize(p[0] - Vec3lfn(prevIntrP));
    //     Vec3lfn v10 = Normalize(p[1] - Vec3lfn(prevIntrP));
    //     Vec3lfn v01 = Normalize(p[3] - Vec3lfn(prevIntrP));
    //     Vec3lfn v11 = Normalize(p[2] - Vec3lfn(prevIntrP));
    //
    //     Vec3lfn eu = p[1] - p[0];
    //     Vec3lfn ev = p[3] - p[0];
    //     // If the solid angle is small
    //     LaneNF32 sphArea = SphericalQuadArea(v00, v10, v01, v11);
    //     MaskF32 mask     = sphArea < DiffuseAreaLight::MinSphericalArea;
    //
    //     LaneNF32 pdf;
    //     if (All(mask))
    //     {
    //         Vec3lfn n  = Normalize(Cross(eu, ev));
    //         Vec3lfn wi = prevIntrP - intr.p;
    //         pdf        = LengthSquared(wi) / (area * AbsDot(Normalize(wi), n));
    //     }
    //     else if (None(mask))
    //     {
    //         pdf = 1.f / sphArea;
    //
    //         NotImplemented;
    // #if 0
    //         Vec2lfn u = InvertSphericalRectangleSample(intrP, prevIntrP, eu, ev);
    //         Vec4lfn w(AbsDot(v00, intr.shading.n), AbsDot(v10, intr.shading.n),
    //                    AbsDot(v01, intr.shading.n), AbsDot(v11, intr.shading.n));
    //         pdf *= BilinearPDF(u, w);
    // #endif
    //     }
    //     else
    //     {
    //         NotImplemented;
    //         pdf = 0.f;
    //     }
    //     return pdf;
}

SampledSpectrum DiffuseAreaLight::Le(const Vec3f &n, const Vec3f &w,
                                     const SampledWavelengths &lambda)
{
    if (Dot(n, w) < 0) return SampledSpectrum(0.f);
    return scale * Lemit->Sample(lambda);
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
// TODO: the scene radius should not have to be stored per infinite light
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
    // avg                    = 0.3282774686813354f;
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

//////////////////////////////
// Morphism
//

f32 LightPDF(Scene *scene) { return 1.f / scene->lights.size(); }

Light *UniformLightSample(Scene *scene, f32 u, f32 *pmf = 0)
{
    size_t numLights = scene->lights.size();
    if (numLights == 0) return 0;
    size_t lightIndex = Min(size_t(u * numLights), numLights - 1);
    Assert(lightIndex >= 0 && lightIndex < numLights);
    if (pmf) *pmf = LightPDF(scene);
    Light *light = scene->lights[lightIndex];
    ErrorExit(light, "lightIndex: %i, total: %llu\n", lightIndex, numLights);
    return light;
}
} // namespace rt
