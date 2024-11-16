#include "integrate.h"

namespace rt
{
// TODO to render moana:
// - infinite lights
// - volumetric
//      - ratio tracking, residual ratio tracking, delta tracking, virtual density segments?
// - shading, ptex, materials, textures
//      - ray differentials
// - bvh intersection and triangle intersection
// - creating objects from the parsed scene packets

// after that's done:
// - simd queues for everything (radiance evaluation, shading, ray streams?)
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
    LaneIF32 k = TwoPi * PI - g2 - g3;
    LaneIF32 S = g0 + g1 - k;
    *pdf       = 1.f / S;

    LaneIF32 b0 = n0.z;
    LaneIF32 b1 = n2.z;

    // Compute cu
    // LaneIF32 au = samples[0] * S + k;
    LaneIF32 au = samples[0] * (g0 + g1 - TwoPi) + (samples[0] - 1) * (g2 + g3);
    LaneIF32 fu = (Cos(au) * b0 - b1) / Sin(au);
    LaneIF32 cu = Clamp(Copysignf(1 / Sqrt(fu * fu + b0 * b0), fu), -1.f, 1.f);

    // Compute xu
    LaneIF32 xu = -(cu * z0) / Sqrt(1.f - cu * cu);
    xu          = Clamp(xu, x0, x1);
    // Compute yv
    LaneIF32 d  = Sqrt(xu * xu + z0 * z0);
    LaneIF32 h0 = y0 / Sqrt(d * d + y0 * y0);
    LaneIF32 h1 = y1 / Sqrt(d * d + y1 * y1);
    // Linearly interpolate between h0 and h1
    LaneIF32 hv   = h0 + (h1 - h0) * samples[1];
    LaneIF32 hvsq = hv * hv;
    LaneIF32 yv   = (hvsq < 1 - 1e-6f) ? (hv * d / Sqrt(1 - hvsq)) : y1;
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

// TODO: get Le
LightSample SampleLiArea(const Scene2 *scene, const LaneIU32 lightIndices, const SurfaceInteraction &intr, const Vec2IF32 &u)
{
    DiffuseAreaLight *lights[IntN];
    for (u32 i = 0; i < IntN; i++)
    {
        lights[i] = &scene->areaLights[lightIndices[i]];
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

// TODO: find the class of each sample, add to a corresponding queue, when the queue is full enough, generate the samples
LightSample SampleLi(const Scene2 *scene, const LightHandle lightHandle, const SurfaceInteraction &intr, const Vec2f &u)
{
    LightClass lClass = lightHandle.GetType();
    switch (lClass)
    {
        case LightClass_Area:
        {
            return SampleLiArea(scene, lightHandle.GetIndex(), intr, u);
        }
        break;
        case LightClass_InfImg:
        {
            NotImplemented;
            return LightSample();
        }
        break;
        default: Assert(0); return LightSample();
    }
}

// TODO: I don't think this is going to be invoked for the moana scene
LaneIF32 PDF_Li(const Scene2 *scene, const LaneIU32 lightIndices, const Vec3IF32 &prevIntrP, const SurfaceInteraction &intr)
{
    DiffuseAreaLight *lights[IntN];
    for (u32 i = 0; i < IntN; i++)
    {
        lights[i] = &scene->areaLights[lightIndices[i]];
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

void BuildLightPDF(Scene2 *scene)
{
    u32 total                        = 0;
    scene->lightPDF[LightClass_Area] = total;
    total += scene->numAreaLights;
    scene->lightPDF[LightClass_InfImg] = total;
    total += scene->numInfiniteLights;
}

f32 LightPDF(Scene2 *scene)
{
    return 1.f / scene->numLights;
}

LightHandle UniformLightSample(Scene2 *scene, f32 u, f32 *pmf = 0)
{
    u32 lightIndex = Min(u32(u * scene->numLights), scene->numLights - 1);
    for (u32 i = 0; i < LightClass_Count; i++)
    {
        if (lightIndex > scene->lightPDF[i])
        {
            LightHandle handle(LightClass(i), lightIndex - scene->lightPDF[i]);
            if (pmf) *pmf = LightPDF(scene);
            return handle;
        }
    }
    Assert(0);
    return LightHandle();
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
    for (u32 i = 0; i < scene->numVolumes; i++)
    {
        Shape *shape = scene->shapes[scene->volumes[i].shapeIndex];
        bounds.Extend(shape->GetBounds());
    }
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

    root = PushStruct(arena, OctreeNode);
    root->extinctionMin.SetInf();
    for (u32 i = 0; i < scene->numVolumes; i++)
    {
        struct StackEntry
        {
            OctreeNode *node;
            Bounds b;
        };
        Volume *volume = &scene->volumes[i];
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
            segment = RaySegment(entry.tMin, entry.tMax, node->extinctionMin, node->extinctionMax, cExtinct);
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

f32 HenyeyGreenstein(f32 cosTheta, f32 g)
{
    g         = Clamp(g, -.99f, .99f);
    f32 denom = 1 + Sqr(g) + 2 * g * cosTheta;
    return Inv4Pi * (1 - Sqr(g)) / (denom * SafeSqrt(denom));
}

f32 HenyeyGreenstein(Vec3f wo, Vec3f wi, f32 g) { return HenyeyGreenstein(Dot(wo, wi), g); }

Vec3f SampleHenyeyGreenstein(const Vec3f &wo, f32 g, Vec2f u, f32 *pdf = 0)
{
    f32 cosTheta;
    if (Abs(g) < 1e-3f)
        cosTheta = 1 - 2 * u[0];
    else
        cosTheta = -1 / (2 * g) *
                   (1 + Sqr(g) - Sqr((1 - Sqr(g)) / (1 + g - 2 * g * u[0])));

    f32 sinTheta = SafeSqrt(1 - Sqr(cosTheta));
    f32 phi      = TwoPi * u[1];

    // TODO: implement FromZ
    Vec3f wi;
    // Frame wFrame = Frame::FromZ(wo);
    // Vector3f wi  = wFrame.FromLocal(Vec3f(sinTheta * Cos(phi), sinTheta * Sin(phi), cosTheta));

    if (pdf) *pdf = HenyeyGreenstein(cosTheta, g);
    return wi;
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
SampledSpectrum SampleTMaj(Scene2 *scene, Ray2 &ray, f32 tMax, f32 xi, RNG &rng, const SampledWavelengths &lambda, const F &callback)
{
    tMax *= Length(ray.d);
    ray.d                      = Normalize(ray.d);
    VolumeAggregate &aggregate = scene->aggregate;
    // TODO: get this from the medium somehow
    SampledSpectrum cExtinct;

    VolumeAggregate::Iterator itr = aggregate.CreateIterator(&ray, cExtinct, intr.tHit);
    RaySegment segment;

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

        for (;;)
        {
            f32 t = t - (std::log(1 - xi) / cMaj);
            xi    = rng.Uniform<f32>();
            if (t > tMax)
            {
                f32 dT = Min(FLT_MAX, tMax) - tMin;
                tMaj *= FastExp(-dT * cSpectrumMaj);
                if constexpr (residualRatioTracking) tRay *= FastExp(-dT * segment.cMin[0]);
                break;
            }
            else
            {
                tMaj *= FastExp(-(t - tMin) * cSpectrumMaj);
                Vec3f p = ray(t);

                SampledSpectrum cAbsorb, cScatter, Le;
                PhaseFunction phase;
                if (!callback(p, cSpectrumMaj, tMaj, cAbsorb, cScatter, Le, phase))
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

NEESample VolumetricSampleLD(const SurfaceInteraction &intr, Ray2 &ray, Scene2 *scene, Sampler sampler,
                             SampledSpectrum beta, const SampledSpectrum &p, const SampledWavelengths &lambda, Vec3f &wi);

SampledSpectrum VolumetricIntegrator(Scene2 *scene, VolumeAggregate *aggregate, Ray2 &ray, Sampler sampler,
                                     SampledWavelengths &lambda, u32 maxDepth)
{
    // TODO:
    // 1. actually finish this
    // 2. majorant octree
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
        bool intersect = Intersect(ray, intr);
        if (intr.volumeIndices != 0)
        {
            bool scattered  = false;
            bool terminated = false;

            RNG rng(Hash(sampler.Get1D()), Hash(sampler.Get1D()));

            SampledSpectrum tMaj = SampleTMaj<false>(
                scene, ray, f32(intr.tHit), sampler.Get1D(), rng, lambda,
                [&](Vec3f p, const SampledSpectrum &cMaj, const SampledSpectrum &tMaj,
                    const SampledSpectrum &cAbsorb, const SampledSpectrum &cScatter,
                    const SampledSpectrum &Le, PhaseFunction &phase) {
                    if (!beta)
                    {
                    }
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
                        NEESample neeSample = VolumetricSampleLD(intr, ray, scene, sampler, beta, p_u, lambda, wi);
                        f32 scatterPdf;
                        SampledSpectrum f = phase.EvaluateSample(wo, wi, &scatterPdf);
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
            if (specularBounce || depth == 0)
            {
                for (u32 i = 0; i < scene->numInfiniteLights; i++)
                {
                    InfiniteLight *light = &scene->infiniteLights[i];
                    SampledSpectrum Le   = light->Le(ray.d);
                    L += beta * Le * MISWeight(p_u);
                }
            }
            else
            {
                for (u32 i = 0; i < scene->numInfiniteLights; i++)
                {
                    InfiniteLight *light = &scene->infiniteLights[i];
                    SampledSpectrum Le   = light->Le(ray.d);

                    f32 pdf      = LightPDF(scene);
                    f32 lightPdf = pdf * light->PDF_Li(scene); //..., prevSi.lightIndices, prevSi.p, ray.d);

                    p_l *= lightPdf;
                    L += beta * Le * MISWeight(p_u, p_l);
                }
            }
            break;
            // sample infinite area lights, environment map, and return
        }

        //////////////////////////////
        // Emitter Intersection
        //
        if (intr.lightIndices)
        {
            DiffuseAreaLight *light = &scene->lights[intr.lightIndices];
            SampledSpectrum Le      = light->Le(intr.n, -ray.d, lambda);
            if (depth == 0 || bsdf->IsSpecular())
            {
                L += beta * Le * MISWeight(p_u);
            }
            else
            {
                f32 pdf = LightPDF(scene);
                pdf *= light->PDF_Li(scene, intr.lightIndices, prevIntr.p, intr);
                p_l *= pdf;
                L += beta * Le * MISWeight(p_u, p_l);
            }
        }

        // BSDF bsdf = intr.ComputeShading();
        BSDF bsdf;
        if (!bsdf)
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
        if (!IsSpecular(bsdf->flags))
        {
            Vec3f wi;
            NEESample neeSample = VolumetricSampleLD(intr, ray, scene, sampler, beta, p_u, lambda, wi);
            f32 scatterPdf;
            SampledSpectrum f = bsdf.EvaluateSample(-ray.d, wi, &scatterPdf);
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
        specularBounce = IsSpecular(bsdf.flags);
        if (sample.IsTransmission())
        {
            etaScale *= Sqr(sample.eta);
        }
        ray.o = intr.p;
        ray.d = sample.wi;
    }
    return L;
}

NEESample VolumetricSampleLD(const SurfaceInteraction &intr, Ray2 &ray, Scene2 *scene, Sampler sampler,
                             SampledSpectrum beta, const SampledSpectrum &p, const SampledWavelengths &lambda, Vec3f &wi)
{
    f32 lightPdf;
    LightHandle lightHandle = UniformLightSample(scene, sampler.Get1D(), &lightPdf);
    Vec2f u                 = sampler.Get2D();
    if (!light) return {};
    LightSample sample = SampleLi(scene, lightHandle, intr, u);
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

    RNG rng(Hash(sampler.Get1D()), Hash(sampler.Get1D()));
    SampledSpectrum tMaj = SampleTMaj<true>(
        scene, ray, f32(intr.tHit), sampler.Get1D(), rng, lambda,
        [&](Vec3f p, const SampledSpectrum &cMaj, const SampledSpectrum &tMaj,
            const SampledSpectrum &cAbsorb, const SampledSpectrum &cScatter,
            const SampledSpectrum &Le, PhaseFunction &phase) {
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

    return NEESample{beta * tRay * sample.L, p_l, p_u, sample.isDelta};
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
