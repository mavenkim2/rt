#ifndef BXDF_SIMD_H
#define BXDF_SIMD_H

namespace rt
{

template <i32 N>
struct BSDFSampleN
{
    SampledSpectrumBase<N> f;
    Vec3lf<N> wi;
    LaneF32<N> pdf = LaneNF32(0);
    LaneU32<N> flags;
    f32 eta;

    BSDFSample() = default;

    BSDFSample(const SampledSpectrumBase<N> &f, const Vec3lf<N> &wi, const LaneF32<N> &pdf,
               const LaneU32<N> &flags, const LaneF32<N> &eta = 1.f)
        : f(f), wi(wi), pdf(pdf), flags(flags), eta(eta)
    {
    }

    MaskF32<N> IsReflective() { return rt::IsReflective(flags); }
    MaskF32<N> IsTransmissive() { return rt::IsTransmissive(flags); }
    MaskF32<N> IsDiffuse() { return rt::IsDiffuse(flags); }
    MaskF32<N> IsGlossy() { return rt::IsGlossy(flags); }
    MaskF32<N> IsSpecular() { return rt::IsSpecular(flags); }
    MaskF32<N> IsValid() { return !rt::IsValid(flags); }
};

template <i32 N>
__forceinline BSDFSample<N> Select(const MaskF32<N> &mask, const BSDFSample<N> &a,
                                   const BSDFSample<N> &b)
{
    return BSDFSample(Select(mask, a.f, b.f), Select(mask, a.wi, b.wi),
                      Select(mask, a.pdf, b.pdf), Select(mask, a.flags, b.flags),
                      Select(mask, a.eta, b.eta));
}

template <i32 N>
MaskF32<N> Refract(Vec3lf<N> wi, Vec3lf<N> n, LaneF32<N> eta, f32 *etap = 0, Vec3lf<N> *wt = 0)
{
    LaneF32<N> cosTheta_i = Dot(wi, n);

    MaskF32<N> mask = cosTheta_i > 0;
    n               = Select(mask, n, -n);
    eta             = Select(mask, eta, 1 / eta);
    cosTheta_i      = Select(mask, cosTheta_i, -cosTheta_i);

    LaneF32<N> sin2Theta_i = Max(0.f, 1 - cosTheta_i * cosTheta_i);
    // Snell's law
    LaneF32<N> sin2Theta_t = sin2Theta_i / (eta * eta);
    // Total internal eeflection

    mask = sin2Theta_t < 1;
    // if (sin2Theta_t >= 1) return false;
    LaneF32<N> cosTheta_t = SafeSqrt(1 - sin2Theta_t);
    if (etap)
    {
        *etap = eta;
    }
    if (wt)
    {
        *wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * n;
    }
    return mask;
}

template <i32 N>
LaneF32<N> FrDielectric(LaneF32<N> cosTheta_i, LaneF32<N> eta)
{
    cosTheta_i      = Clamp(cosTheta_i, -1.f, 1.f);
    LaneF32<N> mask = cosTheta_i < 0;
    cosTheta_i      = Select(mask, -cosTheta_i, cosTheta_i);
    eta             = Select(mask, 1.f / eta, eta);

    LaneF32<N> sin2Theta_i = Max(0.f, 1 - cosTheta_i * cosTheta_i);
    LaneF32<N> sin2Theta_t = sin2Theta_i / (eta * eta);
    LaneF32<N> cosTheta_t  = SafeSqrt(1 - sin2Theta_t);
    LaneF32<N> rParallel   = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    LaneF32<N> rPerp       = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    LaneF32<N> Fr          = 0.5f * (rPerp * rPerp + rParallel * rParallel);
    return Fr;
}

template <i32 N>
struct DielectricBxDFSIMD
{
    using Float  = LaneF32<N>;
    using Vec3lf = Vec3<Float>;
    using Vec2lf = Vec2<Float>;
    using Mask   = MaskBase<Float>;

    TrowbridgeReitzDistribution mfDistrib;
    // NOTE: spectrally varying IORs are handled by randomly sampling a single wavelength
    Float eta;

    DielectricBxDF() {}
    DielectricBxDF(const Float &eta, const TrowbridgeReitzDistribution &mfDistrib)
        : eta(eta), mfDistrib(mfDistrib)
    {
    }

    // TODO: simd this
    template <i32 N>
    BSDFSample GenerateSample(const Vec3lf &wo, const Float &uc, const Vec2lf &u,
                              TransportMode mode    = TransportMode::Radiance,
                              BxDFFlags sampleFlags = BxDFFlags::RT) const
    {
        Mask smoothMask = eta == Float(1) || mfDistrib.EffectivelySmooth();

        BSDFSample smoothSample, roughSample;
        if (Any(smoothMask))
        {
            // Sample perfect specular dielectric BSDF
            LaneF32<N> R = FrDielectric(CosTheta(wo), eta), T = 1 - R;
            // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
            LaneF32<N> pr = R, pt = T;
            if (!(sampleFlags & BxDFFlags::Reflection)) pr = 0;
            if (!(sampleFlags & BxDFFlags::Transmission)) pt = 0;

            Mask invalidMask = pr == 0 && pt == 0;

            BSDFSample reflect, refract;
            Mask mask = uc < pr / (pr + pt);
            // Branch  1
            {
                Vec3lf<N> wi(-wo.x, -wo.y, wo.z);
                SampledSpectrumBase<N> fr(R / AbsCosTheta(wi));
                reflect =
                    BSDFSample(fr, wi, pr / (pr + pt), (u32)BxDFFlags::SpecularReflection);
            }
            // Branch 2
            {
                Vec3lf<N> wi;
                f32 etap;
                Mask valid = Refract(wo, Vec3f(0, 0, 1), eta, &etap, &wi);
                invalidMask |= valid;

                SampledSpectrumBase<N> ft(T / AbsCosTheta(wi));
                // Account for non-symmetry with transmission to different medium
                if (mode == TransportMode::Radiance) ft /= Sqr(etap);

                refract = BSDFSample(ft, wi, pt / (pr + pt),
                                     (u32)BxDFFlags::SpecularTransmission, etap);
            }
            smoothSample     = Select(mask, reflect, refract);
            smoothSample.pdf = Select(invalidMask, Float(0), smoothSample.pdf);
            if (All(smoothMask)) return smoothSample;
        }
        {
            BSDFSample reflect, refract;
            // Sample rough dielectric BSDF
            Vec3lf wm = mfDistrib.Sample_wm(wo, u);
            Float R   = FrDielectric(Dot(wo, wm), eta);
            Float T   = 1 - R;
            // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
            Float pr = R, pt = T;
            if (!(sampleFlags & BxDFFlags::Reflection)) pr = 0;
            if (!(sampleFlags & BxDFFlags::Transmission)) pt = 0;

            Mask invalidMask = pr == 0 && pt == 0;

            Float pdf;
            Mask reflectMask = uc < pr / (pr + pt);
            {
                // Sample reflection at rough dielectric interface
                Vec3lf wi = Reflect(wo, wm);
                invalidMask |= !SameHemisphere(wo, wi);
                // Compute PDF of rough dielectric reflection
                pdf = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)) * pr / (pr + pt);

                Assert(!IsNaN(pdf));
                SampledSpectrum f(mfDistrib.D(wm) * mfDistrib.G(wo, wi) * R /
                                  (4 * CosTheta(wi) * CosTheta(wo)));
                reflect = BSDFSample(f, wi, pdf, (u32)BxDFFlags::GlossyReflection);
            }
            {
                // Sample transmission at rough dielectric interface
                Float etap;
                Vec3lf wi;
                bool tir = !Refract(wo, (Vec3f)wm, eta, &etap, &wi);
                invalidMask |= SameHemisphere(wo, wi) || wi.z == 0 || tir;
                // Compute PDF of rough dielectric transmission
                Float denom   = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
                Float dwm_dwi = AbsDot(wi, wm) / denom;
                pdf           = mfDistrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);

                Assert(!IsNaN(pdf));
                // Evaluate BRDF and return _BSDFSample_ for rough transmission
                SampledSpectrum ft(
                    T * mfDistrib.D(wm) * mfDistrib.G(wo, wi) *
                    Abs(Dot(wi, wm) * Dot(wo, wm) / (CosTheta(wi) * CosTheta(wo) * denom)));
                // Account for non-symmetry with transmission to different medium
                if (mode == TransportMode::Radiance) ft /= Sqr(etap);

                refract = BSDFSample(ft, wi, pdf, (u32)BxDFFlags::GlossyTransmission, etap);
            }
            roughSample     = Select(reflectMask, reflect, refract);
            roughSample.pdf = Select(invalidMask, Float(0), roughSample.pdf);
            if (None(smoothMask)) return roughSample;
        }
        return Select(smoothMask, smoothSample, roughSample);
    }

    SampledSpectrum EvaluateSample(const Vec3f &wo, const Vec3f &wi, f32 &pdf,
                                   TransportMode mode) const
    {
        if (eta == 1 || mfDistrib.EffectivelySmooth()) return SampledSpectrum(0.f);
        // Evaluate rough dielectric BSDF
        // Compute generalized half vector _wm_
        f32 cosTheta_o = CosTheta(wo), cosTheta_i = CosTheta(wi);
        bool reflect = cosTheta_i * cosTheta_o > 0;
        float etap   = 1;
        if (!reflect) etap = cosTheta_o > 0 ? eta : (1 / eta);
        Vec3f wm = wi * etap + wo;
        if (cosTheta_i == 0 || cosTheta_o == 0 || LengthSquared(wm) == 0) return {};
        wm = FaceForward(Normalize(wm), Vec3f(0, 0, 1));

        // Discard backfacing microfacets
        if (Dot(wm, wi) * cosTheta_i < 0 || Dot(wm, wo) * cosTheta_o < 0) return {};

        f32 F = FrDielectric(Dot(wo, wm), eta);
        f32 T = 1 - F;
        if (reflect)
        {
            // Compute reflection at rough dielectric interface
            pdf = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)) * F / (F + T);
            return SampledSpectrum(mfDistrib.D(wm) * mfDistrib.G(wo, wi) * F /
                                   std::abs(4 * cosTheta_i * cosTheta_o));
        }
        else
        {
            // Compute transmission at rough dielectric interface
            f32 denomPdf = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
            f32 denom    = denomPdf * cosTheta_i * cosTheta_o;
            f32 ft       = mfDistrib.D(wm) * (1 - F) * mfDistrib.G(wo, wi) *
                     std::abs(Dot(wi, wm) * Dot(wo, wm) / denom);

            f32 dwm_dwi = AbsDot(wi, wm) / denomPdf;
            pdf         = mfDistrib.PDF(wo, wm) * dwm_dwi * T / (F + T);
            // Account for non-symmetry with transmission to different medium
            if (mode == TransportMode::Radiance) ft /= Sqr(etap);

            return SampledSpectrum(ft);
        }
    }
    LaneNU32 Flags() const
    {
        LaneNU32 flags = Select(eta == 1.f, u32(BxDFFlags::Transmission), u32(BxDFFlags::RT));
        return flags | Select(mfDistrib.EffectivelySmooth(), u32(BxDFFlags::Specular),
                              u32(BxDFFlags::Glossy));
    }
    f32 PDF(Vec3f wo, Vec3f wi, TransportMode mode,
            BxDFFlags sampleFlags = BxDFFlags::RT) const
    {
        if (eta == 1 || mfDistrib.EffectivelySmooth()) return 0;
        // Evaluate sampling PDF of rough dielectric BSDF
        // Compute generalized half vector _wm_
        f32 cosTheta_o = CosTheta(wo), cosTheta_i = CosTheta(wi);
        bool reflect = cosTheta_i * cosTheta_o > 0;
        float etap   = 1;
        if (!reflect) etap = cosTheta_o > 0 ? eta : (1 / eta);
        Vec3f wm = wi * etap + wo;
        if (cosTheta_i == 0 || cosTheta_o == 0 || LengthSquared(wm) == 0) return {};
        wm = FaceForward(Normalize(wm), Vec3f(0, 0, 1));

        // Discard backfacing microfacets
        if (Dot(wm, wi) * cosTheta_i < 0 || Dot(wm, wo) * cosTheta_o < 0) return {};

        // Determine Fresnel reflectance of rough dielectric boundary
        f32 R = FrDielectric(Dot(wo, wm), eta);
        f32 T = 1 - R;

        // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
        f32 pr = R, pt = T;
        if (!(sampleFlags & BxDFFlags::Reflection)) pr = 0;
        if (!(sampleFlags & BxDFFlags::Transmission)) pt = 0;
        if (pr == 0 && pt == 0) return {};

        // Return PDF for rough dielectric
        f32 pdf;
        if (reflect)
        {
            // Compute PDF of rough dielectric reflection
            pdf = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)) * pr / (pr + pt);
        }
        else
        {
            // Compute PDF of rough dielectric transmission
            f32 denom   = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
            f32 dwm_dwi = AbsDot(wi, wm) / denom;
            pdf         = mfDistrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);
        }
        return pdf;
    }
};

template <typename TopBxDF, typename BottomBxDF>
struct CoatedBxDFSIMD
{
    TopBxDF top;
    BottomBxDF bot;
    SampledSpectrumN albedo;
    f32 g, thickness;
    u32 maxDepth, nSamples;
    CoatedBxDF() {}
    CoatedBxDF(TopBxDF top, BottomBxDF bot, SampledSpectrumN albedo, f32 g, f32 thickness,
               u32 maxDepth, u32 nSamples)
        : top(top), bot(bot), albedo(albedo), g(g), thickness(thickness), maxDepth(maxDepth),
          nSamples(nSamples)
    {
    }
    BSDFSample GenerateSample(const Vec3lfn &wOut, const LaneNF32 &uc, const Vec2lfn &u,
                              TransportMode mode    = TransportMode::Radiance,
                              BxDFFlags sampleFlags = BxDFFlags::RT) const
    {
        // NOTE: assumes two sided, meaning that it always first intersects the top layer
        bool flipWi = false;
        Vec3f wo    = wOut;
        if (wo.z < 0)
        {
            flipWi = true;
            wo     = -wo;
        }

        BSDFSample sampleInitial = top.GenerateSample(wo, uc, u, mode);
        if (sampleInitial.pdf == 0 || !sampleInitial.f || sampleInitial.wi.z == 0) return {};

        if (sampleInitial.IsReflective())
        {
            if (flipWi)
            {
                sampleInitial.wi = -sampleInitial.wi;
            }
            // TODO: maybe a bit hacky but whatever
            f32 pdf = PDF(wOut, sampleInitial.wi, mode);
            sampleInitial.f *= pdf / sampleInitial.pdf;
            sampleInitial.pdf = pdf;
            return sampleInitial;
        }
        bool specularPath = sampleInitial.IsSpecular();
        Vec3f w           = sampleInitial.wi;

        RNG rng(Hash(wo), Hash(uc, u));
        auto r = [&rng]() { return Min(rng.Uniform<f32>(), oneMinusEpsilon); };

        SampledSpectrum f = sampleInitial.f * AbsCosTheta(w);
        f32 pdf           = sampleInitial.pdf;

        f32 z = thickness;
        PhaseFunction p(g);

        for (u32 depth = 0; depth < maxDepth; depth++)
        {
            f32 rrBeta = f.MaxComponentValue() / pdf;
            if (depth > 3 && rrBeta < 0.25f)
            {
                f32 q = Max(0.f, 1.f - rrBeta);
                if (r() < q) return {};
                pdf *= 1 - q;
            }
            if (w.z == 0) return {};
            // NOTE: assume homogeneous
            if (albedo)
            {
                f32 sigma_t = 1;
                f32 dz      = SampleExponential(r(), sigma_t / AbsCosTheta(w));
                if (dz == 0) return {};
                f32 zp = w.z > 0 ? (z + dz) : (z - dz);

                if (0 < zp && zp < thickness)
                {
                    PhaseFunctionSample sample = p.GenerateSample(-w, Vec2f(r(), r()));
                    if (sample.pdf == 0 || sample.wi.z == 0) return {};
                    f *= albedo * sample.p;
                    pdf *= sample.pdf;
                    specularPath = false;
                    w            = sample.wi;
                    z            = zp;
                    continue;
                }
                z = Clamp(zp, 0.f, thickness);
            }
            else
            {
                // calculate homogeneous transmittance
                z = (z == thickness) ? 0 : thickness;

                f *= Tr(thickness, w);
            }
            f32 uc0           = r();
            BSDFSample sample = z == 0 ? bot.GenerateSample(-w, uc0, Vec2f(r(), r()), mode)
                                       : top.GenerateSample(-w, uc0, Vec2f(r(), r()), mode);
            if (sample.pdf == 0.f || !sample.f || sample.wi.z == 0) return {};
            f *= sample.f;
            pdf *= sample.pdf;
            specularPath &= sample.IsSpecular();
            w = sample.wi;
            if (sample.IsTransmissive())
            {
                BxDFFlags flags =
                    SameHemisphere(wo, w) ? BxDFFlags::Reflection : BxDFFlags::Transmission;
                flags |= specularPath ? BxDFFlags::Specular : BxDFFlags::Glossy;
                if (flipWi) w = -w;
                f32 absPdf = PDF(wOut, w, mode);
                f *= absPdf / pdf;
                return BSDFSample(f, w, absPdf, u32(flags), 1.f);
            }
            f *= AbsCosTheta(w);
        }
        return {};
    }

    SampledSpectrum EvaluateSample(const Vec3f &wOut, const Vec3f &wIn, f32 &outPdf,
                                   TransportMode mode) const
    {
        Vec3f wo = wOut;
        Vec3f wi = wIn;
        SampledSpectrum f(0.);
        // Estimate _LayeredBxDF_ value _f_ using random sampling
        // Set _wo_ and _wi_ for layered BSDF evaluation
        if (wo.z < 0)
        {
            wo = -wo;
            wi = -wi;
        }

        // enterinterface = top
        // Determine exit interface and exit $z$ for layered BSDF
        TopOrBottomBxDF<TopBxDF, BottomBxDF> exitInterface, nonExitInterface;
        if (!SameHemisphere(wo, wi))
        {
            exitInterface    = &bot;
            nonExitInterface = &top;
        }
        else
        {
            exitInterface    = &top;
            nonExitInterface = &bot;
        }
        f32 exitZ = (SameHemisphere(wo, wi)) ? thickness : 0;

        // Account for reflection at the entrance interface
        f32 _pdf;
        if (SameHemisphere(wo, wi)) f = f32(nSamples) * top.EvaluateSample(wo, wi, _pdf, mode);

        // Declare _RNG_ for layered BSDF evaluation
        RNG rng(Hash(wo), Hash(wi));
        auto r = [&rng]() { return Min(rng.Uniform<f32>(), oneMinusEpsilon); };

        for (u32 s = 0; s < nSamples; ++s)
        {
            // Sample random walk through layers to estimate BSDF value
            // Sample transmission direction through entrance interface
            f32 uc = r();
            BSDFSample wos =
                top.GenerateSample(wo, uc, Vec2f(r(), r()), mode, BxDFFlags::Transmission);
            if (!wos.f || wos.pdf == 0 || wos.wi.z == 0) continue;

            // Sample BSDF for virtual light from _wi_
            uc             = r();
            BSDFSample wis = exitInterface.GenerateSample(
                wi, uc, Vec2f(r(), r()), TransportMode(!u32(mode)), BxDFFlags::Transmission);
            if (!wis.f || wis.pdf == 0 || wis.wi.z == 0) continue;

            // Declare state for random walk through BSDF layers
            SampledSpectrum beta = wos.f * AbsCosTheta(wos.wi) / wos.pdf;
            f32 z                = thickness;
            Vec3f w              = wos.wi;
            PhaseFunction phase(g);

            for (u32 depth = 0; depth < maxDepth; ++depth)
            {
                // Possibly terminate layered BSDF random walk with Russian roulette
                if (depth > 3 && beta.MaxComponentValue() < 0.25f)
                {
                    f32 q = Max(0.f, 1 - beta.MaxComponentValue());
                    if (r() < q) break;
                    beta /= 1 - q;
                }

                // Account for media between layers and possibly scatter
                if (!albedo)
                {
                    // Advance to next layer boundary and update _beta_ for transmittance
                    z = (z == thickness) ? 0 : thickness;
                    beta *= Tr(thickness, w);
                }
                else
                {
                    // Sample medium scattering for layered BSDF evaluation
                    f32 sigma_t = 1;
                    f32 dz      = SampleExponential(r(), sigma_t / Abs(w.z));
                    f32 zp      = w.z > 0 ? (z + dz) : (z - dz);
                    if (z == zp) continue;
                    if (0 < zp && zp < thickness)
                    {
                        // Handle scattering event in layered BSDF medium
                        // Account for scattering through _exitInterface_ using _wis_
                        f32 wt             = 1;
                        SampledSpectrumN p = phase.EvaluateSample(-w, -wis.wi, &_pdf);
                        if (!IsSpecular(exitInterface.Flags()))
                            wt = PowerHeuristic(1, wis.pdf, 1, _pdf);
                        f += beta * albedo * p * wt * Tr(zp - exitZ, wis.wi) * wis.f / wis.pdf;

                        // Sample phase function and update layered path state
                        Vec2f u{r(), r()};
                        PhaseFunctionSample ps = phase.GenerateSample(-w, u);
                        if (ps.pdf == 0 || ps.wi.z == 0) continue;
                        beta *= albedo * ps.p / ps.pdf;
                        w = ps.wi;
                        z = zp;

                        // Possibly account for scattering through _exitInterface_
                        if (((z < exitZ && w.z > 0) || (z > exitZ && w.z < 0)) &&
                            !IsSpecular(exitInterface.Flags()))
                        {
                            // Account for scattering through _exitInterface_
                            SampledSpectrum fExit = exitInterface.EvaluateSample(-w, wi, mode);
                            if (fExit)
                            {
                                f32 exitPDF =
                                    exitInterface.PDF(-w, wi, mode, BxDFFlags::Transmission);
                                wt = PowerHeuristic(1, ps.pdf, 1, exitPDF);
                                f += beta * Tr(zp - exitZ, ps.wi) * fExit * wt;
                            }
                        }

                        continue;
                    }
                    z = Clamp(zp, 0.f, thickness);
                }

                // Account for scattering at appropriate interface
                if (z == exitZ)
                {
                    // Account for reflection at _exitInterface_
                    uc            = r();
                    BSDFSample bs = exitInterface.GenerateSample(-w, uc, Vec2f(r(), r()), mode,
                                                                 BxDFFlags::Reflection);
                    if (!bs.f || bs.pdf == 0 || bs.wi.z == 0) break;
                    beta *= bs.f * AbsCosTheta(bs.wi) / bs.pdf;
                    w = bs.wi;
                }
                else
                {
                    // Account for scattering at _nonExitInterface_
                    if (!IsSpecular(nonExitInterface.Flags()))
                    {
                        // Add NEE contribution along presampled _wis_ direction
                        f32 wt = 1;
                        if (!IsSpecular(exitInterface.Flags()))
                            wt = PowerHeuristic(1, wis.pdf, 1,
                                                nonExitInterface.PDF(-w, -wis.wi, mode));
                        f += beta * nonExitInterface.EvaluateSample(-w, -wis.wi, mode) *
                             AbsCosTheta(wis.wi) * wt * Tr(thickness, wis.wi) * wis.f /
                             wis.pdf;
                    }
                    // Sample new direction using BSDF at _nonExitInterface_
                    uc = r();
                    Vec2f u(r(), r());
                    BSDFSample bs = nonExitInterface.GenerateSample(-w, uc, u, mode,
                                                                    BxDFFlags::Reflection);
                    if (bs.pdf == 0 || !bs.f || bs.wi.z == 0) break;
                    beta *= bs.f * AbsCosTheta(bs.wi) / bs.pdf;
                    w = bs.wi;

                    if (!IsSpecular(exitInterface.Flags()))
                    {
                        // Add NEE contribution along direction from BSDF sample
                        SampledSpectrum fExit = exitInterface.EvaluateSample(-w, wi, mode);
                        if (fExit)
                        {
                            f32 wt = 1;
                            if (!IsSpecular(nonExitInterface.Flags()))
                            {
                                f32 exitPDF =
                                    exitInterface.PDF(-w, wi, mode, BxDFFlags::Transmission);
                                wt = PowerHeuristic(1, bs.pdf, 1, exitPDF);
                            }
                            f += beta * Tr(thickness, bs.wi) * fExit * wt;
                        }
                    }
                }
            }
        }

        outPdf = PDF(wOut, wIn, mode);
        return f / f32(nSamples);
    }

    f32 PDF(Vec3f wo, Vec3f wi, TransportMode mode,
            BxDFFlags sampleFlags = BxDFFlags::RT) const
    {
        // Set _wo_ and _wi_ for layered BSDF evaluation
        if (wo.z < 0)
        {
            wo = -wo;
            wi = -wi;
        }

        // Declare _RNG_ for layered PDF evaluation
        RNG rng(Hash(wi), Hash(wo));
        auto r = [&rng]() { return Min(rng.Uniform<f32>(), oneMinusEpsilon); };

        // Update _pdfSum_ for reflection at the entrance layer
        f32 pdfSum = 0;
        if (SameHemisphere(wo, wi))
        {
            BxDFFlags reflFlag = BxDFFlags::Reflection;
            pdfSum += nSamples * top.PDF(wo, wi, mode, reflFlag);
        }

        for (u32 s = 0; s < nSamples; ++s)
        {
            // Evaluate layered BSDF PDF sample
            if (SameHemisphere(wo, wi))
            {
                // Sample _tInterface_ to get direction into the layers
                BxDFFlags trans = BxDFFlags::Transmission;
                BSDFSample wos  = top.GenerateSample(wo, r(), {r(), r()}, mode, trans);
                BSDFSample wis =
                    top.GenerateSample(wi, r(), {r(), r()}, TransportMode(!u32(mode)), trans);

                // Update _pdfSum_ accounting for TRT scattering events
                if (wos.pdf > 0 && wos.f && wis.pdf > 0 && wis.f)
                {
                    if (!IsNonSpecular(top.Flags())) pdfSum += bot.PDF(-wos.wi, -wis.wi, mode);
                    else
                    {
                        // Use multiple importance sampling to estimate PDF product
                        BSDFSample rs = bot.GenerateSample(-wos.wi, r(), {r(), r()}, mode);
                        if (rs.pdf > 0 && rs.f)
                        {
                            if (IsSpecular(bot.Flags())) pdfSum += top.PDF(-rs.wi, wi, mode);
                            else
                            {
                                float rPDF = bot.PDF(-wos.wi, -wis.wi, mode);
                                float wt   = PowerHeuristic(1, wis.pdf, 1, rPDF);
                                pdfSum += wt * rPDF;

                                float tPDF = top.PDF(-rs.wi, wi, mode);
                                wt         = PowerHeuristic(1, rs.pdf, 1, tPDF);
                                pdfSum += wt * tPDF;
                            }
                        }
                    }
                }
            }
            else
            {
                // to == top ti == bot

                f32 uc = r();
                Vec2f u(r(), r());
                BSDFSample wos = top.GenerateSample(wo, uc, u, mode);
                if (!wos.f || wos.pdf == 0 || wos.wi.z == 0 || wos.IsReflective()) continue;

                uc             = r();
                u              = Vec2f(r(), r());
                BSDFSample wis = bot.GenerateSample(wi, uc, u, TransportMode(!u32(mode)));
                if (!wis.f || wis.pdf == 0 || wis.wi.z == 0 || wis.IsReflective()) continue;

                if (IsSpecular(top.Flags())) pdfSum += bot.PDF(-wos.wi, wi, mode);
                else if (IsSpecular(bot.Flags())) pdfSum += top.PDF(wo, -wis.wi, mode);
                else pdfSum += (top.PDF(wo, -wis.wi, mode) + bot.PDF(-wos.wi, wi, mode)) / 2;
            }
        }
        // Return mixture of PDF estimate and constant PDF
        return Lerp(0.9f, 1 / (4 * PI), pdfSum / nSamples);
    }

    static f32 Tr(f32 dz, Vec3f w)
    {
        if (Abs(dz) <= std::numeric_limits<f32>::min()) return 1;
        return FastExp(-Abs(dz / w.z));
    }

    LaneNU32 Flags() const
    {
        BxDFFlags topFlags = BxDFFlags(top.Flags()), bottomFlags = BxDFFlags(bot.Flags());

        BxDFFlags flags = BxDFFlags::Reflection;
        if (IsSpecular(topFlags)) flags = flags | BxDFFlags::Specular;

        if (IsDiffuse(topFlags) || IsDiffuse(bottomFlags) || albedo)
            flags = flags | BxDFFlags::Diffuse;
        else if (IsGlossy(topFlags) || IsGlossy(bottomFlags))
            flags = flags | BxDFFlags::Glossy;

        if (IsTransmissive(topFlags) && IsTransmissive(bottomFlags))
            flags = flags | BxDFFlags::Transmission;

        return u32(flags);
    }
};

} // namespace rt

#endif
