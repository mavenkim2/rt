#ifndef BSDF_H
#define BSDF_H

namespace rt
{

// TODO: multiple scattering, better hair models, light polarization (ew)

// NOTE: wi and wt point away from the surface point
bool Refract(Vec3f wi, Vec3f n, f32 eta, f32 *etap, Vec3f *wt)
{
    f32 cosTheta_i = Dot(wi, n);
    if (cosTheta_i < 0)
    {
        n          = -n;
        eta        = 1.f / eta;
        cosTheta_i = -cosTheta_i;
    }

    f32 sin2Theta_i = Max(0.f, 1 - cosTheta_i * cosTheta_i);
    // Snell's law
    f32 sin2Theta_t = sin2Theta_i / (eta * eta);
    // Total internal reflection
    if (sin2Theta_t >= 1) return false;
    f32 cosTheta_t = SafeSqrt(1 - sin2Theta_t);
    if (etap)
    {
        *etap = eta;
    }
    *wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * n;
    return true;
}

f32 FrDielectric(f32 cosTheta_i, f32 eta)
{
    cosTheta_i = Clamp(cosTheta_i, -1.f, 1.f);
    if (cosTheta_i < 0)
    {
        cosTheta_i = -cosTheta_i;
        eta        = 1.f / eta;
    }
    f32 sin2Theta_i = Max(0.f, 1 - cosTheta_i * cosTheta_i);
    f32 sin2Theta_t = sin2Theta_i / (eta * eta);
    if (sin2Theta_t >= 1) return 1.f;
    f32 cosTheta_t = SafeSqrt(1 - sin2Theta_t);
    f32 rParallel  = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    f32 rPerp      = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    f32 Fr         = 0.5f * (rPerp * rPerp + rParallel * rParallel);
    return Fr;
}

// Conductors have refractive index with imaginary component: eta - ik, where k is the absoroption coefficient
f32 FrComplex(f32 cosTheta_i, complex eta)
{
    cosTheta_i          = Clamp(cosTheta_i, 0.f, 1.f);
    f32 sin2Theta_i     = Max(0.f, 1 - cosTheta_i * cosTheta_i);
    complex sin2Theta_t = sin2Theta_i / (eta * eta);
    complex cosTheta_t  = Sqrt(1 - sin2Theta_t);
    complex rParallel   = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    complex rPerp       = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    f32 Fr              = 0.5f * (rPerp.Norm() + rParallel.Norm());
    return Fr;
}

SampledSpectrum FrComplex(f32 cosTheta_i, SampledSpectrum eta, SampledSpectrum k)
{
    SampledSpectrum result;
    for (i32 i = 0; i < NSampledWavelengths; ++i)
        result[i] = FrComplex(cosTheta_i, complex(eta[i], k[i]));
    return result;
}

// NOTE: BSDF calculations operate in the frame where the geometric normal is the z axis. Thus, the angle
// between the surface normal and a vector in that frame is just the z component.
f32 CosTheta(Vec3f w)
{
    return w.z;
}

f32 Cos2Theta(Vec3f w)
{
    return w.z * w.z;
}

f32 AbsCosTheta(Vec3f w)
{
    return Abs(w.z);
}

f32 Sin2Theta(Vec3f w)
{
    return Max(0.f, 1 - Cos2Theta(w));
}

f32 SinTheta(Vec3f w)
{
    return sqrtf(Sin2Theta(w));
}

f32 TanTheta(Vec3f w)
{
    return SinTheta(w) / CosTheta(w);
}

f32 Tan2Theta(Vec3f w)
{
    return Sin2Theta(w) / Cos2Theta(w);
}

f32 CosPhi(Vec3f w)
{
    f32 sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : Clamp(w.x / sinTheta, -1.f, 1.f);
}

f32 SinPhi(Vec3f w)
{
    f32 sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : Clamp(w.y / sinTheta, -1.f, 1.f);
}

bool SameHemisphere(Vec3f w, Vec3f wp)
{
    return w.z * wp.z > 0;
}

Vec3f FaceForward(Vec3f n, Vec3f v)
{
    return (Dot(n, v) < 0.f) ? -n : n;
}

struct TrowbridgeReitzDistribution
{
    TrowbridgeReitzDistribution(f32 alphaX, f32 alphaY) : alphaX(alphaX), alphaY(alphaY) {}
    f32 D(Vec3f wm) const
    {
        f32 tan2Theta = Tan2Theta(wm);
        if (IsInf(tan2Theta)) return 0;
        f32 cos2Theta = Cos2Theta(wm);
        f32 cos4Theta = cos2Theta * cos2Theta;
        f32 e         = tan2Theta * (Sqr(CosPhi(wm) / alphaX) +
                             Sqr(SinPhi(wm) / alphaY));
        return 1 / (PI * alphaX * alphaY * cos4Theta * Sqr(1 + e));
    }
    f32 G1(Vec3f w) const
    {
        return 1 / (1 + Lambda(w));
    }
    // NOTE: height based correlation; microfacets at a greater height are more likely to be visible
    f32 G(Vec3f wo, Vec3f wi) const
    {
        return 1 / (1 + Lambda(wo) + Lambda(wi));
    }
    f32 D(Vec3f w, Vec3f wm) const
    {
        return G1(w) / AbsCosTheta(w) * D(wm) * AbsDot(w, wm);
    }
    f32 Lambda(Vec3f w) const
    {
        f32 tan2Theta = Tan2Theta(w);
        if (IsInf(tan2Theta)) return 0;
        f32 alpha2 = Sqr(CosPhi(w) * alphaX) + Sqr(SinPhi(w) * alphaY);
        return (sqrtf(1 + alpha2 * tan2Theta) - 1) / 2;
    }
    f32 PDF(Vec3f w, Vec3f wm) const
    {
        return D(w, wm);
    }
    // NOTE: samples the visible normals instead of simply the distribution function
    Vec3f Sample_wm(Vec3f w, Vec2f u) const
    {
        Vec3f wh = Normalize(Vec3f(alphaX * w.x, alphaY * w.y, w.z));
        if (wh.z < 0)
        {
            wh = -wh;
        }
        // NOTE: this process involves the projection of a disk of uniformly distributed points onto a truncated ellipsoid.
        // The inverse of the scale factor of the ellipsoid is applied to the incoming direction in order to simplify to the
        // isotropic case.
        Vec2f p  = SampleUniformDiskPolar(u);
        Vec3f T1 = wh.z < 0.99999f ? Normalize(Cross(Vec3f(0, 0, 1), wh)) : Vec3f(1, 0, 0);
        Vec3f T2 = Cross(wh, T1);
        // NOTE: For a given x, the y component has a value in range [-h, h], where h = sqrt(1 - Sqr(x)). When the
        // projection is not perpendicular to the ellipsoid, the range shrinks to [-hcos(theta), h]. This requires an
        // affine transformation with scale 0.5(1 + cosTheta) and translation 0.5h(1 - cosTheta).

        f32 h = sqrtf(1 - p.x * p.x);
        p.y   = Lerp((1.f + wh.z) / 2.f, h, p.y);

        // Project point to hemisphere, transform to ellipsoid.
        f32 pz   = sqrtf(Max(0.f, 1 - p.x * p.x - p.y * p.y));
        Vec3f nh = p.x * T1 + p.y * T2 + pz * wh;
        return Normalize(Vec3f(alphaX * nh.x, alphaY * nh.y, Max(1e-6f, nh.z)));
    }
    bool EffectivelySmooth() const
    {
        return Max(alphaX, alphaY) < 1e-3f;
    }
    f32 alphaX;
    f32 alphaY;
};

enum class BSDFFlags
{
    Unset                = 0,
    Reflection           = 1 << 0,
    Transmission         = 1 << 1,
    Diffuse              = 1 << 2,
    Specular             = 1 << 3,
    Glossy               = 1 << 4,
    Invalid              = 1 << 5,
    DiffuseReflection    = Diffuse | Reflection,
    DiffuseTransmission  = Diffuse | Transmission,
    SpecularReflection   = Specular | Reflection,
    SpecularTransmission = Specular | Transmission,
    GlossyReflection     = Glossy | Reflection,
    GlossyTransmission   = Glossy | Transmission,
    RT                   = Reflection | Transmission,
    All                  = Reflection | Transmission | Diffuse | Specular | Glossy,
};

ENUM_CLASS_FLAGS(BSDFFlags)

inline b32 IsReflective(BSDFFlags f)
{
    return EnumHasAnyFlags(f, BSDFFlags::Reflection);
}
inline b32 IsTransmissive(BSDFFlags f)
{
    return EnumHasAnyFlags(f, BSDFFlags::Transmission);
}
inline b32 IsDiffuse(BSDFFlags f)
{
    return EnumHasAnyFlags(f, BSDFFlags::Diffuse);
}
inline b32 IsGlossy(BSDFFlags f)
{
    return EnumHasAnyFlags(f, BSDFFlags::Glossy);
}
inline b32 IsSpecular(BSDFFlags f)
{
    return EnumHasAnyFlags(f, BSDFFlags::Specular);
}
inline b32 IsNonSpecular(BSDFFlags f)
{
    return EnumHasAnyFlags(f, BSDFFlags::Diffuse | BSDFFlags::Glossy);
}
inline b32 IsValid(BSDFFlags f)
{
    return !EnumHasAnyFlags(f, BSDFFlags::Invalid);
}

struct BSDFSample
{
    SampledSpectrum f;
    Vec3f wi;
    f32 pdf;
    BSDFFlags flags;
    f32 eta;
    bool pdfIsProportional;

    BSDFSample() = default;

    BSDFSample(SampledSpectrum f, Vec3f wi, f32 pdf, BSDFFlags flags, f32 eta = 1.f, bool pdfIsProportional = false)
        : f(f), wi(wi), pdf(pdf), flags(flags), eta(eta), pdfIsProportional(pdfIsProportional) {}

    bool IsReflective() { return rt::IsReflective(flags); }
    bool IsTransmissive() { return rt::IsTransmissive(flags); }
    bool IsDiffuse() { return rt::IsDiffuse(flags); }
    bool IsGlossy() { return rt::IsGlossy(flags); }
    bool IsSpecular() { return rt::IsSpecular(flags); }
    bool IsValid() { return !(rt::IsValid(flags)); }
};

// NOTE: BTDFs are not generally symmetric
enum class TransportMode
{
    Importance,
    Radiance,
};

// Lambertian model, light is scattered in all directions equally
struct DiffuseBSDF : BSDFCRTP<DiffuseBSDF>
{
    SampledSpectrum R;
    DiffuseBSDF() = default;
    DiffuseBSDF(SampledSpectrum R) : R(R) {}

    SampledSpectrum EvaluateSample(Vec3f wo, Vec3f wi, f32 &pdf, TransportMode mode) const
    {
        if (!SameHemisphere(wo, wi))
        {
            pdf = 0.f;
            return SampledSpectrum(0.f);
        }
        pdf = CosineHemispherePDF(AbsCosTheta(wi));
        return R / PI;
    }

    BSDFSample GenerateSample(Vec3f wo, f32 uc, Vec2f u, TransportMode mode, BSDFFlags sampleFlags = BSDFFlags::RT) const
    {
        if (!EnumHasAnyFlags(sampleFlags, BSDFFlags::Reflection)) return {};
        Vec3f wi = SampleCosineHemisphere(u);
        if (wo.z < 0) wi.z *= -1;
        f32 pdf = CosineHemispherePDF(AbsCosTheta(wi));
        return BSDFSample(R * InvPi, wi, pdf, BSDFFlags::DiffuseReflection);
    }
    f32 PDF(Vec3f wo, Vec3f wi, TransportMode mode, BSDFFlags sampleFlags = BSDFFlags::RT) const
    {
        if (!EnumHasAnyFlags(sampleFlags, BSDFFlags::Reflection) || !SameHemisphere(wo, wi)) return 0.f;
        return CosineHemispherePDF(AbsCosTheta(wi));
    }
    BSDFFlags Flags() const
    {
        return R ? BSDFFlags::DiffuseReflection : BSDFFlags::Unset;
    }
};

struct ConductorBSDF : BSDFCRTP<ConductorBSDF>
{
    ConductorBSDF() = delete;
    ConductorBSDF(TrowbridgeReitzDistribution mfDistrib, SampledSpectrum eta, SampledSpectrum k)
        : mfDistrib(mfDistrib), eta(eta), k(k) {}
    TrowbridgeReitzDistribution mfDistrib;
    SampledSpectrum eta, k;

    SampledSpectrum EvaluateSample(Vec3f wo, Vec3f wi, f32 &pdf, TransportMode mode) const
    {
        if (!SameHemisphere(wo, wi))
        {
            pdf = 0.f;
            return {};
        }
        if (mfDistrib.EffectivelySmooth())
        {
            pdf = 0.f;
            return {};
        }
        f32 cosTheta_o = AbsCosTheta(wo);
        f32 cosTheta_i = AbsCosTheta(wi);
        if (cosTheta_i == 0 || cosTheta_o == 0)
        {
            pdf = 0.f;
            return {};
        }
        Vec3f wm = wi + wo;
        if (LengthSquared(wm) == 0.f)
        {
            pdf = 0.f;
            return {};
        }
        wm                 = Normalize(wm);
        SampledSpectrum Fr = FrComplex(AbsDot(wo, wm), eta, k);
        SampledSpectrum f  = mfDistrib.D(wm) * Fr * mfDistrib.G(wo, wi) / (4 * cosTheta_i * cosTheta_o);
        wm                 = FaceForward(Normalize(wm), Vec3f(0, 0, 1));
        pdf                = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm));
        return f;
    }
    BSDFSample GenerateSample(Vec3f wo, f32 uc, Vec2f u, TransportMode mode, BSDFFlags sampleFlags) const
    {
        if (!EnumHasAnyFlags(sampleFlags, BSDFFlags::RT)) return {};
        if (mfDistrib.EffectivelySmooth())
        {
            Vec3f wi(-wo.x, -wo.y, wo.z);
            SampledSpectrum f = FrComplex(AbsCosTheta(wi), eta, k) / AbsCosTheta(wi);
            return BSDFSample(f, wi, 1.f, BSDFFlags::SpecularReflection);
        }
        Vec3f wm = mfDistrib.Sample_wm(wo, u);
        Vec3f wi = Reflect(wo, wm);
        if (!SameHemisphere(wo, wi)) return {};
        f32 pdf        = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm));
        f32 cosTheta_o = AbsCosTheta(wo);
        f32 cosTheta_i = AbsCosTheta(wi);
        // NOTE: Fresnel term with respect to the microfacet normal, wm
        SampledSpectrum Fr = FrComplex(AbsDot(wo, wm), eta, k);
        // Torrance Sparrow BRDF Model (D * G * F / 4cos cos)
        SampledSpectrum f = mfDistrib.D(wm) * Fr * mfDistrib.G(wo, wi) / (4 * cosTheta_i * cosTheta_o);
        return BSDFSample(f, wi, pdf, BSDFFlags::GlossyReflection);
    }
    f32 PDF(Vec3f wo, Vec3f wi, TransportMode mode, BSDFFlags sampleFlags = BSDFFlags::RT) const
    {
        if (!EnumHasAnyFlags(sampleFlags, BSDFFlags::Reflection)) return 0.f;
        if (!SameHemisphere(wo, wi)) return 0.f;
        if (mfDistrib.EffectivelySmooth()) return 0.f;
        Vec3f wm = wo + wi;
        if (LengthSquared(wm) == 0.f) return 0.f;
        wm = FaceForward(Normalize(wm), Vec3f(0, 0, 1));
        return mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm));
    }

    BSDFFlags Flags() const
    {
        return mfDistrib.EffectivelySmooth() ? BSDFFlags::SpecularReflection : BSDFFlags::GlossyReflection;
    }
};

struct DielectricBSDF : BSDFCRTP<DielectricBSDF>
{
    DielectricBSDF() = delete;
    DielectricBSDF(f32 eta, TrowbridgeReitzDistribution mfDistrib) : eta(eta), mfDistrib(mfDistrib) {}
    SampledSpectrum EvaluateSample(Vec3f wo, Vec3f wi, f32 &pdf, TransportMode mode) const
    {
        if (eta == 1 || mfDistrib.EffectivelySmooth())
            return SampledSpectrum(0.f);
        f32 cosTheta_o = CosTheta(wo);
        f32 cosTheta_i = CosTheta(wi);
        bool reflect   = cosTheta_i * cosTheta_o > 0.f;
        f32 etap       = 1.f;
        if (!reflect)
            etap = cosTheta_o > 0.f ? eta : 1 / eta;
        Vec3f wm = wi * etap + wo;
        if (cosTheta_i == 0 || cosTheta_o == 0 || LengthSquared(wm) == 0.f)
            return {};
        wm = Normalize(wm);
        wm = wm.z < 0.f ? -wm : wm;
        if (Dot(wm, wi) * cosTheta_i < 0.f || Dot(wm, wo) * cosTheta_o < 0.f) return {};
        f32 F  = FrDielectric(Dot(wo, wm), eta);
        f32 T  = 1 - F;
        f32 pr = F;
        f32 pt = T;
        if (pr == 0.f && pt == 0.f) pdf = 0.f;
        if (reflect)
        {
            pdf = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)) * pr / (pr + pt);
            return SampledSpectrum(mfDistrib.D(wm) * mfDistrib.G(wo, wi) * F / Abs(4 * cosTheta_i * cosTheta_o));
        }
        else
        {
            f32 denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
            f32 ft    = mfDistrib.D(wm) * (1 - F) * mfDistrib.G(wo, wi) * Abs(Dot(wi, wm) * Dot(wo, wm) / (denom * cosTheta_i * cosTheta_o));

            f32 dwm_dwi = AbsDot(wi, wm) / denom;
            pdf         = mfDistrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);
            if (mode == TransportMode::Radiance)
                ft /= Sqr(etap);
            return SampledSpectrum(ft);
        }
    }
    BSDFSample GenerateSample(Vec3f wo, f32 uc, Vec2f u, TransportMode mode, BSDFFlags sampleFlags) const
    {
        // Sample specular BTDF
        if (eta == 1 || mfDistrib.EffectivelySmooth())
        {
            f32 R  = FrDielectric(CosTheta(wo), eta);
            f32 T  = 1 - R;
            f32 pr = R;
            f32 pt = T;
            if (!EnumHasAnyFlags(sampleFlags, BSDFFlags::Reflection)) pr = 0.f;
            if (!EnumHasAnyFlags(sampleFlags, BSDFFlags::Transmission)) pt = 0.f;
            if (pr == 0.f && pt == 0.f) return {};
            // Sample based on the amount of reflection / transmission
            // Specular reflection
            if (uc < pr / (pr + pt))
            {
                Vec3f wi(-wo.x, -wo.y, wo.z);
                SampledSpectrum fr(R / AbsCosTheta(wi));
                return BSDFSample(fr, wi, pr / (pr + pt), BSDFFlags::SpecularReflection);
            }
            // Specular transmission
            else
            {
                Vec3f wi;
                f32 etap;
                if (!Refract(wo, Vec3f(0, 0, 1), eta, &etap, &wi)) return {};
                SampledSpectrum ft(T / AbsCosTheta(wi));
                if (mode == TransportMode::Radiance) ft /= etap * etap;
                return BSDFSample(ft, wi, pt / (pr + pt), BSDFFlags::SpecularTransmission, etap);
            }
        }
        // Rough dielectric BSDF
        else
        {
            Vec3f wm = mfDistrib.Sample_wm(wo, u);
            f32 R    = FrDielectric(Dot(wo, wm), eta);
            f32 T    = 1 - R;
            f32 pr   = R;
            f32 pt   = T;
            if (!EnumHasAnyFlags(sampleFlags, BSDFFlags::Reflection)) pr = 0.f;
            if (!EnumHasAnyFlags(sampleFlags, BSDFFlags::Transmission)) pt = 0.f;
            if (pr == 0.f && pt == 0.f) return {};
            // Glossy reflection
            if (uc < pr / (pr + pt))
            {
                Vec3f wi = Reflect(wo, wm);
                if (!SameHemisphere(wo, wi)) return {};
                f32 pdf = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)) * pr / (pr + pt);
                SampledSpectrum f(mfDistrib.D(wm) * mfDistrib.G(wo, wi) * R / (4 * CosTheta(wi) * CosTheta(wo)));
                return BSDFSample(f, wi, pdf, BSDFFlags::GlossyReflection);
            }
            // Glossy transmission
            else
            {
                Vec3f wi;
                f32 etap;
                if (!Refract(wo, wm, eta, &etap, &wi) || SameHemisphere(wo, wi) || wi.z == 0.f) return {};
                f32 denom   = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
                f32 dwm_dwi = AbsDot(wi, wm) / denom;
                f32 pdf     = mfDistrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);
                SampledSpectrum ft(T * mfDistrib.D(wm) * mfDistrib.G(wo, wi) *
                                   Abs(Dot(wi, wm) * Dot(wo, wm) / (CosTheta(wi) * CosTheta(wo) * denom)));
                if (mode == TransportMode::Radiance) ft /= etap * etap;
                return BSDFSample(ft, wi, pdf, BSDFFlags::GlossyTransmission, etap);
            }
        }
    }
    BSDFFlags Flags() const
    {
        BSDFFlags flags = (eta == 1.f) ? BSDFFlags::Transmission : (BSDFFlags::RT);
        return flags | (mfDistrib.EffectivelySmooth() ? BSDFFlags::Specular : BSDFFlags::Glossy);
    }
    f32 PDF(Vec3f wo, Vec3f wi, TransportMode mode, BSDFFlags sampleFlags = BSDFFlags::RT) const
    {
        if (eta == 1.f || mfDistrib.EffectivelySmooth())
            return 0.f;
        f32 cosTheta_o = CosTheta(wo);
        f32 cosTheta_i = CosTheta(wi);
        bool reflect   = cosTheta_i * cosTheta_o > 0.f;
        f32 etap       = 1.f;
        if (!reflect)
            etap = cosTheta_o > 0.f ? eta : 1.f / eta;
        // Calculate the half angle, accounting for transmission
        Vec3f wm = wi * etap + wo;
        if (cosTheta_i == 0 || cosTheta_o == 0 || LengthSquared(wm) == 0.f) return {};
        wm = wm.z < 0.f ? -wm : wm;
        if (Dot(wm, wi) * cosTheta_i < 0 || Dot(wm, wo) * cosTheta_o < 0) return {};
        f32 R  = FrDielectric(Dot(wo, wm), eta);
        f32 T  = 1 - R;
        f32 pr = R;
        f32 pt = T;
        if (!EnumHasAnyFlags(sampleFlags, BSDFFlags::Reflection)) pr = 0.f;
        if (!EnumHasAnyFlags(sampleFlags, BSDFFlags::Transmission)) pt = 0.f;
        if (pr == 0.f && pt == 0.f) return 0.f;
        f32 pdf;
        if (reflect)
        {
            pdf = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)) * pr / (pr + pt);
        }
        else
        {
            f32 denom   = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
            f32 dwm_dwi = AbsDot(wi, wm) / denom;
            pdf         = mfDistrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);
        }
        return pdf;
    }

    TrowbridgeReitzDistribution mfDistrib;
    // NOTE: spectrally varying IORs are handled by randomly sampling a single wavelength
    f32 eta;
};

// NOTE: only models perfect specular scattering
struct ThinDielectricBSDF : BSDFCRTP<ThinDielectricBSDF>
{
    ThinDielectricBSDF() = default;
    ThinDielectricBSDF(f32 eta) : eta(eta) {}
    SampledSpectrum EvaluateSample(Vec3f wo, Vec3f wi, f32 &pdf, TransportMode mode) const
    {
        pdf = 0.f;
        return SampledSpectrum(0.f);
    }
    BSDFSample GenerateSample(Vec3f wo, f32 uc, Vec2f u, TransportMode mode, BSDFFlags sampleFlags) const
    {
        f32 R = FrDielectric(AbsCosTheta(wo), eta);
        f32 T = 1 - R;
        if (R < 1)
        {
            R += Sqr(T) * R / (1 - Sqr(R));
            T = 1 - R;
        }
        f32 pr = R;
        f32 pt = T;
        if (!EnumHasAnyFlags(sampleFlags, BSDFFlags::Reflection)) pr = 0.f;
        if (!EnumHasAnyFlags(sampleFlags, BSDFFlags::Transmission)) pt = 0.f;
        if (pr == 0.f && pt == 0.f) return {};
        // TODO: this is the same as in the dielectric case. Compress?
        if (uc < pr / (pr + pt))
        {
            Vec3f wi(-wo.x, -wo.y, wo.z);
            SampledSpectrum fr(R / AbsCosTheta(wi));
            return BSDFSample(fr, wi, pr / (pr + pt), BSDFFlags::SpecularReflection);
        }
        Vec3f wi = -wo;
        SampledSpectrum ft(T / AbsCosTheta(wi));
        return BSDFSample(ft, wi, pt / (pr + pt), BSDFFlags::SpecularTransmission);
    }
    f32 PDF(Vec3f wo, Vec3f wi, TransportMode mode, BSDFFlags sampleFlags) const
    {
        return 0.f;
    }
    BSDFFlags Flags() const
    {
        return BSDFFlags::Reflection | BSDFFlags::Transmission | BSDFFlags::Specular;
    }

    f32 eta;
};
} // namespace rt

#endif
