#include "lights.h"
#include "color.h"

namespace rt
{
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
Vec3lfn SampleSphericalRectangle(const Vec3lfn &p, const Vec3lfn &base, const Vec3lfn &eu,
                                 const Vec3lfn &ev, const Vec2lfn &samples, LaneNF32 *pdf)
{
    LaneNF32 euLength = Length(eu);
    LaneNF32 evLength = Length(ev);

    // Calculate local coordinate system where sampling is done
    // NOTE: rX and rY must be perpendicular
    Vec3lfn rX = eu / euLength;
    Vec3lfn rY = ev / evLength;
    Vec3lfn rZ = Cross(rX, rY);

    Vec3lfn d0  = base - p;
    LaneNF32 x0 = Dot(d0, rX);
    LaneNF32 y0 = Dot(d0, rY);
    LaneNF32 z0 = Dot(d0, rZ);
    if (z0 > 0)
    {
        z0 *= -1.f;
        rZ *= LaneNF32(-1.f);
    }

    LaneNF32 x1 = x0 + euLength;
    LaneNF32 y1 = y0 + evLength;

    Vec3lfn v00(x0, y0, z0);
    Vec3lfn v01(x0, y1, z0);
    Vec3lfn v10(x1, y0, z0);
    Vec3lfn v11(x1, y1, z0);

    // Compute normals to edges (i.e, normal of plane containing edge and p)
    Vec3lfn n0 = Normalize(Cross(v00, v10));
    Vec3lfn n1 = Normalize(Cross(v10, v11));
    Vec3lfn n2 = Normalize(Cross(v11, v01));
    Vec3lfn n3 = Normalize(Cross(v01, v00));

    // Calculate the angle between the plane normals
    LaneNF32 g0 = AngleBetween(-n0, n1);
    LaneNF32 g1 = AngleBetween(-n1, n2);
    LaneNF32 g2 = AngleBetween(-n2, n3);
    LaneNF32 g3 = AngleBetween(-n3, n0);

    // Compute solid angle subtended by rectangle
    LaneNF32 k = TwoPi * PI - g2 - g3;
    LaneNF32 S = g0 + g1 - k;
    *pdf       = 1.f / S;

    LaneNF32 b0 = n0.z;
    LaneNF32 b1 = n2.z;

    // Compute cu
    // LaneNF32 au = samples[0] * S + k;
    LaneNF32 au = samples[0] * (g0 + g1 - TwoPi) + (samples[0] - 1) * (g2 + g3);
    LaneNF32 fu = (Cos(au) * b0 - b1) / Sin(au);
    LaneNF32 cu = Clamp(Copysign(1 / Sqrt(fu * fu + b0 * b0), fu), -1.f, 1.f);

    // Compute xu
    LaneNF32 xu = -(cu * z0) / Sqrt(1.f - cu * cu);
    xu          = Clamp(xu, x0, x1);
    // Compute yv
    LaneNF32 d  = Sqrt(xu * xu + z0 * z0);
    LaneNF32 h0 = y0 / Sqrt(d * d + y0 * y0);
    LaneNF32 h1 = y1 / Sqrt(d * d + y1 * y1);
    // Linearly interpolate between h0 and h1
    LaneNF32 hv   = h0 + (h1 - h0) * samples[1];
    LaneNF32 hvsq = hv * hv;
    LaneNF32 yv   = (hvsq < 1 - 1e-6f) ? (hv * d / Sqrt(1 - hvsq)) : y1;
    // Convert back to world space
    return p + rX * xu + rY * yv + rZ * z0;
}

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

template <typename T>
T BilinearPDF(const Vec2<T> &u, const Vec4<T> &w)
{
    T zeroMask = u.x < 0 || u.x > 1 || u.y < 0 || u.y > 1;
    T denom    = w[0] + w[1] + w[2] + w[3];
    T oneMask  = denom == 0;
    T result   = Select(zeroMask, T(0), Select(oneMask, T(1), 4 * Bilerp(u, w) / denom));
    return result;
}

template <typename T>
Vec2<T> SampleBilinear(const Vec2<T> &u, const Vec4<T> &w)
{
    Vec2<T> result;
    result.y = SampleLinear(u[1], w[0] + w[1], w[2] + w[3]);
    result.x = SampleLinear(u[0], Lerp(result.y, w[0], w[2]), Lerp(result.y, w[1], w[3]));
    return result;
}

LaneNF32 SphericalQuadArea(const Vec3lfn &a, const Vec3lfn &b, const Vec3lfn &c,
                           const Vec3lfn &d)
{
    Vec3lfn axb = Normalize(Cross(a, b));
    Vec3lfn bxc = Normalize(Cross(b, c));
    Vec3lfn cxd = Normalize(Cross(c, d));
    Vec3lfn dxa = Normalize(Cross(d, a));

    LaneNF32 g0 = AngleBetween(-axb, bxc);
    LaneNF32 g1 = AngleBetween(-bxc, cxd);
    LaneNF32 g2 = AngleBetween(-cxd, dxa);
    LaneNF32 g3 = AngleBetween(-dxa, axb);
    return Abs(g0 + g1 + g2 + g3 - 2 * PI);
}

//////////////////////////////
// Diffuse Area Light
//
SAMPLE_LI(DiffuseAreaLight)
{
    const DiffuseAreaLight *lights[IntN];
    for (u32 i = 0; i < IntN; i++)
    {
        lights[i] = &scene->GetAreaLights()[Get(lightIndices, i)];
    }

    Vec3lfn p[4];
    LaneNF32 area;
    for (u32 i = 0; i < 4; i++)
    {
        Lane4F32 pI[IntN];
        for (u32 lightIndex = 0; lightIndex < IntN; lightIndex++)
        {
            const DiffuseAreaLight *light = lights[lightIndex];
            // Lane4F32 pTemp                = Lane4F32::LoadU(light->p + i);

            // TODO: maybe transform the vertices once
            pI[lightIndex]        = Lane4F32(TransformP(*light->renderFromLight, light->p[i]));
            Set(area, lightIndex) = light->area;
        }
        Transpose<IntN>(pI, p[i]);
    }

    Vec3lfn v00 = Normalize(p[0] - Vec3lfn(intr.p));
    Vec3lfn v10 = Normalize(p[1] - Vec3lfn(intr.p));
    Vec3lfn v01 = Normalize(p[3] - Vec3lfn(intr.p));
    Vec3lfn v11 = Normalize(p[2] - Vec3lfn(intr.p));

    Vec3lfn eu = p[1] - p[0];
    Vec3lfn ev = p[3] - p[0];
    Vec3lfn n  = Normalize(Cross(eu, ev));

    LightSample result;
    // If the solid angle is small
    MaskF32 mask = SphericalQuadArea(v00, v10, v01, v11) < DiffuseAreaLight::MinSphericalArea;
    Vec3lfn wi   = intr.p - result.samplePoint;
    if (All(mask))
    {
        result.samplePoint = Lerp(u[0], Lerp(u[1], p[0], p[3]), Lerp(u[1], p[1], p[2]));
        result.pdf         = LengthSquared(wi) / (LaneNF32(area) * AbsDot(Normalize(wi), n));
    }
    else if (None(mask))
    {
        LaneNF32 pdf;
        result.samplePoint = SampleSphericalRectangle(intr.p, p[0], eu, ev, u, &pdf);

        // add projected solid angle measure (n dot wi) to pdf
        Vec4lfn w(AbsDot(v00, intr.shading.n), AbsDot(v10, intr.shading.n),
                  AbsDot(v01, intr.shading.n), AbsDot(v11, intr.shading.n));
        Vec2lfn uNew = SampleBilinear(u, w);
        pdf *= BilinearPDF(uNew, w);
        result.pdf = pdf;
    }
    else
    {
        NotImplemented;
    }
    result.wi        = -wi;
    result.lightType = LightType::Area;
    result.L         = DiffuseAreaLight::Le(lights[0], n, wi, lambda);
    if (!result.L) return {};
    return result;
}

// TODO: I don't think this is going to be invoked for the moana scene
PDF_LI(DiffuseAreaLight)
{
    const DiffuseAreaLight *lights[IntN];
    for (u32 i = 0; i < IntN; i++)
    {
        lights[i] = &scene->GetAreaLights()[Get(lightIndices, i)];
    }
    Vec3lfn p[4];
    LaneNF32 area;
    // TODO: maybe have to spawn a ray??? but I feel like this is only called (at least for
    // now) when it already has intersected, and we need the pdf for MIS
    for (u32 i = 0; i < 4; i++)
    {
        Lane4F32 pI[IntN];
        for (u32 lightIndex = 0; lightIndex < IntN; lightIndex++)
        {
            const DiffuseAreaLight *light = lights[lightIndex];

            pI[lightIndex]        = Lane4F32(TransformP(*light->renderFromLight, light->p[i]));
            Set(area, lightIndex) = light->area;
        }
        Transpose<IntN>(pI, p[i]);
    }
    Vec3lfn v00 = Normalize(p[0] - Vec3lfn(prevIntrP));
    Vec3lfn v10 = Normalize(p[1] - Vec3lfn(prevIntrP));
    Vec3lfn v01 = Normalize(p[3] - Vec3lfn(prevIntrP));
    Vec3lfn v11 = Normalize(p[2] - Vec3lfn(prevIntrP));

    Vec3lfn eu = p[1] - p[0];
    Vec3lfn ev = p[3] - p[0];
    // If the solid angle is small
    LaneNF32 sphArea = SphericalQuadArea(v00, v10, v01, v11);
    MaskF32 mask     = sphArea < DiffuseAreaLight::MinSphericalArea;

    LaneNF32 pdf;
    if (All(mask))
    {
        Vec3lfn n  = Normalize(Cross(eu, ev));
        Vec3lfn wi = prevIntrP - intr.p;
        pdf        = LengthSquared(wi) / (area * AbsDot(Normalize(wi), n));
    }
    else if (None(mask))
    {
        pdf = 1.f / sphArea;

        NotImplemented;
#if 0
        Vec2lfn u = InvertSphericalRectangleSample(intrP, prevIntrP, eu, ev);
        Vec4lfn w(AbsDot(v00, intr.shading.n), AbsDot(v10, intr.shading.n),
                   AbsDot(v01, intr.shading.n), AbsDot(v11, intr.shading.n));
        pdf *= BilinearPDF(u, w);
#endif
    }
    else
    {
        NotImplemented;
        pdf = 0.f;
    }
    return pdf;
}

LE(DiffuseAreaLight)
{
    if (Dot(n, w) < 0) return SampledSpectrum(0.f);
    return light->scale * light->Lemit->Sample(lambda);
}

//////////////////////////////
// Distant Light
//

SAMPLE_LI(DistantLight)
{
    const DistantLight *light = &scene->lights.Get<DistantLight>()[u32(lightIndices)];
    Vec3f wi                  = -light->d;

    return LightSample(light->scale * light->Lemit->Sample(lambda),
                       Vec3f(intr.p) + 2.f * wi * light->sceneRadius, wi, 1.f,
                       LightType::DeltaDirection);
}

//////////////////////////////
// Infinite Lights
//
// TODO: the scene radius should not have to be stored per infinite light
SAMPLE_LI(UniformInfiniteLight)
{
    const UniformInfiniteLight *light =
        &scene->lights.Get<UniformInfiniteLight>()[u32(lightIndices)];
    if (allowIncompletePDF) return {};
    Vec3f wi = SampleUniformSphere(u);
    f32 pdf  = UniformSpherePDF();
    return LightSample(light->scale * light->Lemit->Sample(lambda),
                       Vec3f(intr.p) + 2.f * wi * light->sceneRadius, wi, pdf,
                       LightType::Infinite);
}

PDF_LI_INF(UniformInfiniteLight) { return allowIncompletePDF ? 0.f : UniformSpherePDF(); }

LE_INF(UniformInfiniteLight) { return light->scale * light->Lemit->Sample(lambda); }

// ImageInfiniteLight::ImageInfiniteLight(Arena *arena, Image image) : image(image)
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
}

SampledSpectrum ImageInfiniteLight::ImageLe(Vec2f uv, const SampledWavelengths &lambda) const
{
    Vec3f rgb = GetOctahedralRGB(&image, uv);

    RGBIlluminantSpectrum spec(*imageColorSpace, rgb);
    return scale * spec.Sample(lambda);
}

SAMPLE_LI(ImageInfiniteLight)
{
    const ImageInfiniteLight *light =
        &scene->lights.Get<ImageInfiniteLight>()[u32(lightIndices)];

    f32 pdf;
    Vec2f uv;
    if (allowIncompletePDF)
    {
        uv = light->compensatedDistribution.Sample(u, &pdf);
    }
    else
    {
        uv = light->distribution.Sample(u, &pdf);
    }
    if (pdf == 0.f) return {};
    Vec3f wi = TransformV(*light->renderFromLight, EqualAreaSquareToSphere(uv));
    pdf /= 4 * PI;
    return LightSample(light->ImageLe(uv, lambda),
                       Vec3f(intr.p) + 2.f * wi * light->sceneRadius, wi, pdf,
                       LightType::Infinite);
}

PDF_LI_INF(ImageInfiniteLight)
{
    Vec3f wi = Normalize(ApplyInverseV(*light->renderFromLight, w));
    Vec2f uv = EqualAreaSphereToSquare(wi);
    f32 pdf;
    if (allowIncompletePDF)
    {
        pdf = light->compensatedDistribution.PDF(uv);
    }
    else
    {
        pdf = light->distribution.PDF(uv);
    }
    return pdf / (4 * PI);
}

LE_INF(ImageInfiniteLight)
{
    Vec3f wi = Normalize(ApplyInverseV(*light->renderFromLight, w));
    Vec2f uv = EqualAreaSphereToSquare(wi);
    return light->ImageLe(uv, lambda);
}

//////////////////////////////
// Morphism
//
// TODO: find the class of each sample, add to a corresponding queue, when the queue is
// full enough, generate the samples
static LightSample SampleLi(Scene *scene, LightHandle lightHandle,
                            const SurfaceInteraction &intr, const SampledWavelengths &lambda,
                            Vec2f &u, bool allowIncompletePDF = false)
{
    LightClass lClass = lightHandle.GetType();
    switch (lClass)
    {
        case LightClass::DiffuseAreaLight:
            return DiffuseAreaLight::SampleLi(scene, lightHandle.GetIndex(), intr, lambda, u,
                                              allowIncompletePDF);
        case LightClass::DistantLight:
            return DistantLight::SampleLi(scene, lightHandle.GetIndex(), intr, lambda, u,
                                          allowIncompletePDF);
        case LightClass::UniformInfiniteLight:
            return UniformInfiniteLight::SampleLi(scene, lightHandle.GetIndex(), intr, lambda,
                                                  u, allowIncompletePDF);
        case LightClass::ImageInfiniteLight:
            return ImageInfiniteLight::SampleLi(scene, lightHandle.GetIndex(), intr, lambda, u,
                                                allowIncompletePDF);
        default: Assert(0); return LightSample();
    }
}

static f32 PDF_Li(Scene *scene, LightHandle lightHandle, Vec3f &prevIntrP,
                  SurfaceInteraction &intr, bool allowIncompletePDF = false)
{
    LightClass lClass = lightHandle.GetType();
    u32 lightIndex    = lightHandle.GetIndex();
    switch (lClass)
    {
        case LightClass::DiffuseAreaLight:
            Assert(0);
            return (f32)DiffuseAreaLight::PDF_Li(scene, lightIndex, prevIntrP, intr,
                                                 allowIncompletePDF);
        case LightClass::DistantLight:
            return (f32)DistantLight::PDF_Li(scene, lightIndex, prevIntrP, intr,
                                             allowIncompletePDF);
        // case LightClass_InfUnf: return (f32)UniformInfiniteLight::PDF_Li(scene,
        // lightIndex, prevIntrP, intr, allowIncompletePDF); case LightClass_InfImg: return
        // (f32)ImageInfiniteLight::PDF_Li(scene, lightIndex, prevIntrP, intr,
        // allowIncompletePDF);
        default: Assert(0); return 0.f;
    }
}

// NOTE: other pdfs cannot sample dirac delta distributions (unless they have the same
// direction)
static SampledSpectrum Le(Scene *scene, LightHandle lightHandle, Vec3f &n, Vec3f &w,
                          SampledWavelengths &lambda)
{
    LightClass lClass = lightHandle.GetType();
    u32 lightIndex    = lightHandle.GetIndex();
    switch (lClass)
    {
        case LightClass::DiffuseAreaLight:
            return DiffuseAreaLight::Le(&scene->GetAreaLights()[lightIndex], n, w, lambda);
        case LightClass::DistantLight:
            return DistantLight::Le(&scene->lights.Get<DistantLight>()[lightIndex], n, w,
                                    lambda);
        // case LightClass_InfUnf: return
        // UniformInfiniteLight::Le(&scene->uniformInfLights[lightIndex], n, w, lambda);
        // case LightClass_InfImg: return
        // ImageInfiniteLight::Le(&scene->imageInfLights[lightIndex], n, w, lambda);
        default: Assert(0); return SampledSpectrum(0.f);
    }
}

// void BuildLightPDF(Scene *scene)
// {
//     u32 total = 0;
//     for (u32 i = 0; i < LightClass_Count; i++)
//     {
//         total += scene->lightCount[i];
//         scene->lightPDF[i] = total;
//     }
// }

f32 LightPDF(Scene *scene) { return 1.f / scene->numLights; }

LightHandle UniformLightSample(Scene *scene, f32 u, f32 *pmf = 0)
{
    if (scene->numLights == 0) return LightHandle();
    u32 lightIndex = Min(u32(u * scene->numLights), scene->numLights - 1);
    Assert(lightIndex >= 0 && lightIndex < scene->numLights);
    u32 localIndex;
    u32 type = scene->lights.ConvertIndexToType(lightIndex, &localIndex);
    if (pmf) *pmf = LightPDF(scene);
    return LightHandle(LightClass(type), localIndex);
}
} // namespace rt
