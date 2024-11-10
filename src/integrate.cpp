#include "integrate.h"

namespace rt
{
// TODO to render moana:
// - shading, ptex, materials, textures
//      - ray differentials
// - bvh intersection and triangle intersection
// - creating objects from the parsed scene packets

// after that's done:
// - simd queues for everything (radiance evaluation, shading, ray streams?)
// - volumetric
//      - ratio tracking, residual ratio tracking, delta tracking, virtual density segments?
// - bdpt, metropolis, vcm/upbp, mcm?
// - subdivision surfaces

// NOTE: sample (over solid angle) the spherical rectangle obtained by projecting a planar rectangle onto
// the unit sphere centered at point p
// https://blogs.autodesk.com/media-and-entertainment/wp-content/uploads/sites/162/egsr2013_spherical_rectangle.pdf

// TODO: simd sin, cos, and arcsin
Vec3IF32 SampleSphericalRectangle(const Vec3IF32 &p, const Vec3IF32 &base, const Vec3IF32 &eu, const Vec3IF32 &ev,
                                  const Vec2IF32 &samples, LaneIF32 *pdf)
{
    LaneIF32 euLength = Length(eu);
    LaneIF32 evLength = Length(ev);

    // Calculate local coordinate system where sampling is done
    // NOTE: rX and rY must be perpendicular
    Vec3IF32 rX = eu / euLength;
    Vec3IF32 rY = ev / evLength;
    Vec3IF32 rZ = Cross(rX, rY);

    Vec3IF32 d0 = base - p;
    LaneIF32 x0 = Dot(d0, rX);
    LaneIF32 y0 = Dot(d0, rY);
    LaneIF32 z0 = Dot(d0, rZ);
    if (z0 > 0)
    {
        z0 *= -1.f;
        rZ *= LaneIF32(-1.f);
    }

    LaneIF32 x1 = x0 + euLength;
    LaneIF32 y1 = y0 + evLength;

    Vec3IF32 v00(x0, y0, z0);
    Vec3IF32 v01(x0, y1, z0);
    Vec3IF32 v10(x1, y0, z0);
    Vec3IF32 v11(x1, y1, z0);

    // Compute normals to edges (i.e, normal of plane containing edge and p)
    Vec3IF32 n0 = Normalize(Cross(v00, v10));
    Vec3IF32 n1 = Normalize(Cross(v10, v11));
    Vec3IF32 n2 = Normalize(Cross(v11, v01));
    Vec3IF32 n3 = Normalize(Cross(v01, v00));

    // Calculate the angle between the plane normals
    LaneIF32 g0 = AngleBetween(-n0, n1);
    LaneIF32 g1 = AngleBetween(-n1, n2);
    LaneIF32 g2 = AngleBetween(-n2, n3);
    LaneIF32 g3 = AngleBetween(-n3, n0);

    // Compute solid angle subtended by rectangle
    LaneIF32 k = LaneIF32(2.f * PI) - g2 - g3;
    LaneIF32 S = g0 + g1 - k;
    *pdf       = LaneIF32(1.f) / S;

    LaneIF32 b0 = n0.z;
    LaneIF32 b1 = n2.z;

    // Compute cu
    LaneIF32 au = samples[0] * S + k;
    LaneIF32 fu = ((Cos(au)) * b0 - b1) / (Sin(au));
    LaneIF32 cu = Clamp(Copysignf(1 / Sqrt(fu * fu + b0 * b0), fu), -1.f, 1.f);

    // Compute xu
    LaneIF32 xu = -(cu * z0) / LaneIF32(Sqrt(LaneIF32(1.f) - cu * cu));
    xu          = Clamp(xu, LaneIF32(-1.f), LaneIF32(1.f));
    // Compute yv
    LaneIF32 d  = Sqrt(xu * xu + z0 * z0);
    LaneIF32 h0 = y0 / LaneIF32(Sqrt(d * d + y0 * y0));
    LaneIF32 h1 = y1 / LaneIF32(Sqrt(d * d + y1 * y1));
    // Linearly interpolate between h0 and h1
    LaneIF32 hv   = h0 + (h1 - h0) * samples[1];
    LaneIF32 hvsq = hv * hv;
    LaneIF32 yv   = (hvsq < 1 - 1e-6f) ? (hv * d / Lane1F32(Sqrt(1 - hvsq))) : y1;
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

__forceinline void Transpose8x3(const Lane4F32 &inA, const Lane4F32 &inB, const Lane4F32 &inC, const Lane4F32 &inD,
                                const Lane4F32 &inE, const Lane4F32 &inF, const Lane4F32 &inG, const Lane4F32 &inH,
                                Lane8F32 &out0, Lane8F32 &out1, Lane8F32 &out2)
{
    Lane4F32 temp[6];
    Transpose4x3(inA, inB, inC, inD, temp[0], temp[1], temp[2]);
    Transpose4x3(inE, inF, inG, inH, temp[3], temp[4], temp[5]);
    out0 = Lane8F32(temp[0], temp[3]);
    out1 = Lane8F32(temp[1], temp[4]);
    out2 = Lane8F32(temp[2], temp[5]);
}

__forceinline void Transpose(const Lane4F32 lanes[IntN], Vec3IF32 &out)
{
#if IntN == 1
    out = ToVec3f(lanes[0]);
#elif IntN == 4
    Transpose4x3(lanes[0], lanes[1], lanes[2], lanes[3], out.x, out.y, out.z);
#elif IntN == 8
    Transpose8x3(lanes[0], lanes[1], lanes[2], lanes[3],
                 lanes[4], lanes[5], lanes[6], lanes[7],
                 out.x, out.y, out.z);
#endif
}

LaneIF32 SphericalQuadArea(const Vec3IF32 &a, const Vec3IF32 &b, const Vec3IF32 &c, const Vec3IF32 &d)
{
    Vec3IF32 axb = Normalize(Cross(a, b));
    Vec3IF32 bxc = Normalize(Cross(b, c));
    Vec3IF32 cxd = Normalize(Cross(c, d));
    Vec3IF32 dxa = Normalize(Cross(d, a));

    LaneIF32 g0 = AngleBetween(-axb, bxc);
    LaneIF32 g1 = AngleBetween(-bxc, cxd);
    LaneIF32 g2 = AngleBetween(-cxd, dxa);
    LaneIF32 g3 = AngleBetween(-dxa, axb);
    return Abs(g0 + g1 + g2 + g3 - 2 * PI);
}

// TODO: maybe consider one light x N interactions instead of N x N
LightSample SampleLi(const Scene2 *scene, const LaneIU32 lightIndices, const SurfaceInteraction &intr, const Vec2IF32 &u)
{
    DiffuseAreaLight *lights[IntN];
    for (u32 i = 0; i < IntN; i++)
    {
        lights[i] = &scene->lights[lightIndices[i]];
    }

    Vec3IF32 p[4];
    LaneIF32 area;
    for (u32 i = 0; i < 4; i++)
    {
        Lane4F32 pI[IntN];
        for (u32 lightIndex = 0; lightIndex < IntN; lightIndex++)
        {
            DiffuseAreaLight *light = lights[lightIndex];
            Lane4F32 pTemp          = Lane4F32::LoadU(light->p + i);

            // TODO: maybe transform the vertices once
            pI[lightIndex]   = Transform(*light->renderFromLight, pTemp);
            area[lightIndex] = light->area;
        }
        Transpose(pI, p[i]);
    }

    Vec3IF32 v00 = Normalize(p[0] - Vec3IF32(intr.p));
    Vec3IF32 v10 = Normalize(p[1] - Vec3IF32(intr.p));
    Vec3IF32 v01 = Normalize(p[3] - Vec3IF32(intr.p));
    Vec3IF32 v11 = Normalize(p[2] - Vec3IF32(intr.p));

    Vec3IF32 eu = p[1] - p[0];
    Vec3IF32 ev = p[3] - p[0];
    Vec3IF32 n  = Normalize(Cross(eu, ev));

    LightSample result;
    // If the solid angle is small
    MaskF32 mask = SphericalQuadArea(v00, v10, v01, v11) < DiffuseAreaLight::MinSphericalArea;
    if (All(mask))
    {
        result.samplePoint = Lerp(u[0], Lerp(u[1], p[0], p[3]), Lerp(u[1], p[1], p[2]));
        Vec3IF32 wi        = intr.p - result.samplePoint;
        result.n           = n;
        result.pdf         = LengthSquared(wi) / (area * AbsDot(Normalize(wi), n));
    }
    else if (None(mask))
    {
        LaneIF32 pdf;
        result.samplePoint = SampleSphericalRectangle(intr.p, p[0], eu, ev, u, &pdf);
        result.n           = n;

        // add projected solid angle measure (n dot wi) to pdf
        Vec4IF32 w(AbsDot(v00, intr.shading.n), AbsDot(v10, intr.shading.n),
                   AbsDot(v01, intr.shading.n), AbsDot(v11, intr.shading.n));
        Vec2IF32 uNew = SampleBilinear(u, w);
        pdf *= BilinearPDF(uNew, w);
        result.pdf = pdf;
    }
    else
    {
        NotImplemented;
    }
    return result;
}

// TODO: I don't think this is going to be invoked for the moana scene
LaneIF32 PDF_Li(const Scene2 *scene, const LaneIU32 lightIndices, const Vec3IF32 &prevIntrP, const SurfaceInteraction &intr)
{
    DiffuseAreaLight *lights[IntN];
    for (u32 i = 0; i < IntN; i++)
    {
        lights[i] = &scene->lights[lightIndices[i]];
    }
    Vec3IF32 p[4];
    LaneIF32 area;
    // TODO: maybe have to spawn a ray??? but I feel like this is only called (at least for now) when it already has intersected,
    // and we need the pdf for MIS
    for (u32 i = 0; i < 4; i++)
    {
        Lane4F32 pI[IntN];
        for (u32 lightIndex = 0; lightIndex < IntN; lightIndex++)
        {
            DiffuseAreaLight *light = lights[lightIndex];
            Lane4F32 pTemp          = Lane4F32::LoadU(light->p + i);

            pI[lightIndex]   = Transform(*light->renderFromLight, pTemp);
            area[lightIndex] = light->area;
        }
        Transpose(pI, p[i]);
    }
    Vec3IF32 v00 = Normalize(p[0] - Vec3IF32(prevIntrP));
    Vec3IF32 v10 = Normalize(p[1] - Vec3IF32(prevIntrP));
    Vec3IF32 v01 = Normalize(p[3] - Vec3IF32(prevIntrP));
    Vec3IF32 v11 = Normalize(p[2] - Vec3IF32(prevIntrP));

    Vec3IF32 eu = p[1] - p[0];
    Vec3IF32 ev = p[3] - p[0];
    // If the solid angle is small
    LaneIF32 sphArea = SphericalQuadArea(v00, v10, v01, v11);
    MaskF32 mask     = sphArea < DiffuseAreaLight::MinSphericalArea;

    LaneIF32 pdf;
    if (All(mask))
    {
        Vec3IF32 n  = Normalize(Cross(eu, ev));
        Vec3IF32 wi = prevIntrP - intr.p;
        pdf         = LengthSquared(wi) / (area * AbsDot(Normalize(wi), n));
    }
    else if (None(mask))
    {
        pdf = 1.f / sphArea;

        NotImplemented;
#if 0
        Vec2IF32 u = InvertSphericalRectangleSample(intrP, prevIntrP, eu, ev);
        Vec4IF32 w(AbsDot(v00, intr.shading.n), AbsDot(v10, intr.shading.n),
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
} // namespace rt
