namespace rt
{
// TODO to render moana:
// - integrator <- currently working
// - uniform light sampling (more complicated ones later)
// - shading, ptex, materials
//      - ray differentials
// - bvh intersection and triangle intersection
// - creating objects from the parsed scene packets

// after that's done:
// - simd queues for everything (radiance evaluation, shading, ray streams?)

void Li(Scene *scene, RayDifferential &ray, u32 maxDepth)
{
    u32 depth = 0;
    SampledSpectrum L(0.f);
    SampledSpectrum beta(1.f);

    bool specularBounce = false;
    f32 bsdfPdf         = 1.f;
    f32 etaScale        = 1.f;

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
                    SampledSpectrum Le   = light->Li(-ray.d);
                    L += beta * Le;
                }
            }
            else
            {
                for (u32 i = 0; i < scene->numInfiniteLights; i++)
                {
                    InfiniteLight *light = &scene->infiniteLights[i];
                    SampledSpectrum Le   = light->Li(-ray.d);
                    // probability of sampling the light * probability of
                    f32 lightPdf = lightSampler->PMF(prev, light) * light->PDF_Li(); // find the pmf for the light
                    f32 w_l      = PowerHeuristic(1, bsdfPdf, 1, lightPdf);
                    // NOTE: beta already contains the cosine, bsdf, and pdf terms
                    L += beta * w_l * Le;
                }
            }
            break;
            // sample infinite area lights, environment map, and return
        }
        // If intersected with a light,
        if (si.light)
        {
            Light *light = si.light;
            if (specularBounce || depth == 0)
            {
                SampledSpectrum Le = light->Li(-ray.d);
                L += beta * Le;
            }
            else
            {
                SampledSpectrum Le = light->Li(-ray.d);
                // probability of sampling the light * probability of
                f32 lightPdf = lightSampler->PMF(prev, light) * light->PDF_Li(); // find the pmf for the light
                f32 w_l      = PowerHeuristic(1, bsdfPdf, 1, lightPdf);
                // NOTE: beta already contains the cosine, bsdf, and pdf terms
                L += beta * w_l * Le;
            }
        }

        BSDF *bsdf = si.GetBSDF();

        // Next Event Estimation
        // TODO: offset ray origin

        // Choose light source for direct lighting calculation
        Light *light = lightSampler->SampleLight(&si);
        if (light)
        {
            Vec2f sample = sampler.Get2D();
            // Sample point on the light source
            LightSample ls = light->Sample(sample);
            if (ls)
            {
                // Evaluate BSDF for light sample, check visibility with shadow ray
                SampledSpectrum Ld(0.f);
                SampledSpectrum f = bsdf->f(-ray.d, wo) * AbsDot(si.shading.n, wi);
                if (f && !scene->IntersectShadowRay())
                {
                    // Calculate contribution
                    f32 lightPdf = 0.f; // prob of choosing light * prob of choosing point on light;
                    if (light->type == LightType::Delta)
                    {
                        Ld = beta * lightPdf;
                    }
                    else
                    {
                        f32 bsdfPdf = bsdf->PDF(wo, wi);
                        f32 w_l     = PowerHeuristic(1, bsdfPdf, 1, lightPdf);
                        Ld          = beta * f * w_l * ls.L / lightPdf;
                    }
                }
            }
        }

        // sample bsdf, calculate pdf
        beta *= bsdf->f * AbsDot(shading->n, bsdf->wi) / pdf;
        if (bsdf->IsSpecular()) specularBounce = true;

        // Spawn new ray

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
} // namespace rt
