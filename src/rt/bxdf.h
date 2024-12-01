#ifndef BXDF_H
#define BXDF_H

namespace rt
{

// TODO: multiple scattering, better hair models, light polarization (ew)

// NOTE: wi and wt point away from the surface point
MaskF32 Refract(Vec3NF32 wi, Vec3NF32 n, LaneNF32 eta, LaneNF32 *etap, Vec3NF32 *wt)
{
    LaneNF32 cosTheta_i = Dot(wi, n);

    MaskF32 mask = cosTheta_i < 0;
    n            = Select(mask, -n, n);
    eta          = Select(mask, 1 / eta, eta);
    cosTheta_i   = Select(mask, -cosTheta_i, cosTheta_i);

    LaneNF32 sin2Theta_i = Max(0.f, 1 - cosTheta_i * cosTheta_i);
    // Snell's law
    LaneNF32 sin2Theta_t = sin2Theta_i / (eta * eta);
    // Total internal reflection

    mask = sin2Theta_t < 1;
    // if (sin2Theta_t >= 1) return false;
    LaneNF32 cosTheta_t = SafeSqrt(1 - sin2Theta_t);
    if (etap)
    {
        *etap = eta;
    }
    *wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * n;
    return mask;
}

LaneNF32 FrDielectric(const LaneNF32 &cosTheta_i, const LaneNF32 &eta)
{
    cosTheta_i           = Clamp(cosTheta_i, -1.f, 1.f);
    MaskF32 mask         = cosTheta_i < 0;
    cosTheta_i           = Select(mask, -cosTheta_i, cosTheta_i);
    eta                  = Select(mask, 1.f / eta, eta);
    LaneNF32 sin2Theta_i = Max(0.f, 1 - cosTheta_i * cosTheta_i);
    LaneNF32 sin2Theta_t = sin2Theta_i / (eta * eta);
    LaneNF32 cosTheta_t  = SafeSqrt(1 - sin2Theta_t);
    LaneNF32 rParallel   = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    LaneNF32 rPerp       = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    LaneNF32 Fr          = 0.5f * (rPerp * rPerp + rParallel * rParallel);
    return Fr;
}

// Conductors have refractive index with imaginary component: eta - ik, where k is the absoroption coefficient
LaneNF32 FrComplex(const LaneNF32 &cosTheta_i, const Complex<LaneNF32> &eta)
{
    cosTheta_i           = Clamp(cosTheta_i, 0.f, 1.f);
    LaneNF32 sin2Theta_i = Max(0.f, 1 - cosTheta_i * cosTheta_i);
    Complex sin2Theta_t  = sin2Theta_i / (eta * eta);
    Complex cosTheta_t   = Sqrt(1 - sin2Theta_t);
    Complex rParallel    = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    Complex rPerp        = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    LaneNF32 Fr          = 0.5f * (rPerp.Norm() + rParallel.Norm());
    return Fr;
}

SampledSpectrumN FrComplex(LaneNF32 cosTheta_i, SampledSpectrumN eta, SampledSpectrumN k)
{
    SampledSpectrumN result;
    for (i32 i = 0; i < NSampledWavelengths; ++i)
        result[i] = FrComplex(cosTheta_i, Complex(eta[i], k[i]));
    return result;
}

// NOTE: BSDF calculations operate in the frame where the geometric normal is the z axis. Thus, the angle
// between the surface normal and a vector in that frame is just the z component.
LaneNF32 CosTheta(Vec3NF32 w)
{
    return w.z;
}

LaneNF32 Cos2Theta(Vec3NF32 w)
{
    return w.z * w.z;
}

LaneNF32 AbsCosTheta(Vec3NF32 w)
{
    return Abs(w.z);
}

LaneNF32 Sin2Theta(Vec3NF32 w)
{
    return Max(0.f, 1 - Cos2Theta(w));
}

LaneNF32 SinTheta(Vec3NF32 w)
{
    return sqrtf(Sin2Theta(w));
}

LaneNF32 TanTheta(Vec3NF32 w)
{
    return SinTheta(w) / CosTheta(w);
}

LaneNF32 Tan2Theta(Vec3NF32 w)
{
    return Sin2Theta(w) / Cos2Theta(w);
}

LaneNF32 CosPhi(Vec3NF32 w)
{
    LaneNF32 sinTheta = SinTheta(w);
    return Select(sinTheta == 0, 1, Clamp(w.x / sinTheta, -1.f, 1.f));
}

LaneNF32 SinPhi(Vec3NF32 w)
{
    LaneNF32 sinTheta = SinTheta(w);
    return Select(sinTheta == 0, 0, Clamp(w.y / sinTheta, -1.f, 1.f));
}

MaskF32 SameHemisphere(Vec3NF32 w, Vec3NF32 wp)
{
    return w.z * wp.z > 0;
}

Vec3NF32 FaceForward(Vec3NF32 n, Vec3NF32 v)
{
    return Select(Dot(n, v) < 0.f, -n, n);
}

struct TrowbridgeReitzDistribution
{
    TrowbridgeReitzDistribution(f32 alphaX, f32 alphaY) : alphaX(alphaX), alphaY(alphaY) {}
    LaneNF32 D(const Vec3NF32 &wm) const
    {
        LaneNF32 tan2Theta = Tan2Theta(wm);
        if (IsInf(tan2Theta)) return 0;
        LaneNF32 cos2Theta = Cos2Theta(wm);
        LaneNF32 cos4Theta = cos2Theta * cos2Theta;
        LaneNF32 e         = tan2Theta * (Sqr(CosPhi(wm) / alphaX) +
                                  Sqr(SinPhi(wm) / alphaY));
        return 1 / (PI * alphaX * alphaY * cos4Theta * Sqr(1 + e));
    }
    LaneNF32 G1(Vec3NF32 w) const
    {
        return 1 / (1 + Lambda(w));
    }
    // NOTE: height based correlation; microfacets at a greater height are more likely to be visible
    LaneNF32 G(Vec3NF32 wo, Vec3NF32 wi) const
    {
        return 1 / (1 + Lambda(wo) + Lambda(wi));
    }
    LaneNF32 D(Vec3NF32 w, Vec3NF32 wm) const
    {
        return G1(w) / AbsCosTheta(w) * D(wm) * AbsDot(w, wm);
    }
    LaneNF32 Lambda(Vec3NF32 w) const
    {
        LaneNF32 tan2Theta = Tan2Theta(w);
        if (IsInf(tan2Theta)) return 0;
        LaneNF32 alpha2 = Sqr(CosPhi(w) * alphaX) + Sqr(SinPhi(w) * alphaY);
        return (sqrtf(1 + alpha2 * tan2Theta) - 1) / 2;
    }
    LaneNF32 PDF(Vec3NF32 w, Vec3NF32 wm) const
    {
        return D(w, wm);
    }
    // NOTE: samples the visible normals instead of simply the distribution function
    Vec3NF32 Sample_wm(Vec3NF32 w, Vec2NF32 u) const
    {
        Vec3NF32 wh = Normalize(Vec3NF32(alphaX * w.x, alphaY * w.y, w.z));
        if (wh.z < 0)
        {
            wh = -wh;
        }
        // NOTE: this process involves the projection of a disk of uniformly distributed points onto a truncated ellipsoid.
        // The inverse of the scale factor of the ellipsoid is applied to the incoming direction in order to simplify to the
        // isotropic case.
        Vec2NF32 p  = SampleUniformDiskPolar(u);
        Vec3NF32 T1 = wh.z < 0.99999f ? Normalize(Cross(Vec3NF32(0, 0, 1), wh)) : Vec3NF32(1, 0, 0);
        Vec3NF32 T2 = Cross(wh, T1);
        // NOTE: For a given x, the y component has a value in range [-h, h], where h = sqrt(1 - Sqr(x)). When the
        // projection is not perpendicular to the ellipsoid, the range shrinks to [-hcos(theta), h]. This requires an
        // affine transformation with scale 0.5(1 + cosTheta) and translation 0.5h(1 - cosTheta).

        LaneNF32 h = sqrtf(1 - p.x * p.x);
        p.y        = Lerp((1.f + wh.z) / 2.f, h, p.y);

        // Project point to hemisphere, transform to ellipsoid.
        LaneNF32 pz = sqrtf(Max(0.f, 1 - p.x * p.x - p.y * p.y));
        Vec3NF32 nh = p.x * T1 + p.y * T2 + pz * wh;
        return Normalize(Vec3NF32(alphaX * nh.x, alphaY * nh.y, Max(1e-6f, nh.z)));
    }
    bool EffectivelySmooth() const
    {
        return Max(alphaX, alphaY) < 1e-3f;
    }
    f32 alphaX;
    f32 alphaY;
};

enum class BxDFFlags
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

ENUM_CLASS_FLAGS(BxDFFlags)

inline b32 IsReflective(BxDFFlags f)
{
    return EnumHasAnyFlags(f, BxDFFlags::Reflection);
}
inline b32 IsTransmissive(BxDFFlags f)
{
    return EnumHasAnyFlags(f, BxDFFlags::Transmission);
}
inline b32 IsDiffuse(BxDFFlags f)
{
    return EnumHasAnyFlags(f, BxDFFlags::Diffuse);
}
inline b32 IsGlossy(BxDFFlags f)
{
    return EnumHasAnyFlags(f, BxDFFlags::Glossy);
}
inline b32 IsSpecular(BxDFFlags f)
{
    return EnumHasAnyFlags(f, BxDFFlags::Specular);
}
inline b32 IsNonSpecular(BxDFFlags f)
{
    return EnumHasAnyFlags(f, BxDFFlags::Diffuse | BxDFFlags::Glossy);
}
inline b32 IsValid(BxDFFlags f)
{
    return !EnumHasAnyFlags(f, BxDFFlags::Invalid);
}

struct BSDFSample
{
    SampledSpectrumN f;
    Vec3NF32 wi;
    LaneNF32 pdf;
    LaneNF32 eta;
    BxDFFlags flags;
    bool pdfIsProportional;

    BSDFSample() = default;

    BSDFSample(SampledSpectrumN f, Vec3NF32 wi, LaneNF32 pdf, BxDFFlags flags, LaneNF32 eta = 1.f, bool pdfIsProportional = false)
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
struct DiffuseBxDF : BxDFCRTP<DiffuseBxDF>
{
    SampledSpectrumN R;
    DiffuseBSDF() = default;
    DiffuseBSDF(SampledSpectrumN R) : R(R) {}

    SampledSpectrumN EvaluateSample(Vec3NF32 wo, Vec3NF32 wi, LaneNF32 &pdf, TransportMode mode) const
    {
        if (!SameHemisphere(wo, wi))
        {
            pdf = 0.f;
            return SampledSpectrumN(0.f);
        }
        pdf = CosineHemispherePDF(AbsCosTheta(wi));
        return R / PI;
    }

    BSDFSample GenerateSample(Vec3NF32 wo, LaneNF32 uc, Vec2NF32 u, TransportMode mode, BxDFFlags sampleFlags = BxDFFlags::RT) const
    {
        if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection)) return {};
        Vec3NF32 wi = SampleCosineHemisphere(u);
        if (wo.z < 0) wi.z *= -1;
        LaneNF32 pdf = CosineHemispherePDF(AbsCosTheta(wi));
        return BSDFSample(R * InvPi, wi, pdf, BxDFFlags::DiffuseReflection);
    }
    LaneNF32 PDF(Vec3NF32 wo, Vec3NF32 wi, TransportMode mode, BxDFFlags sampleFlags = BxDFFlags::RT) const
    {
        if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection) || !SameHemisphere(wo, wi)) return 0.f;
        return CosineHemispherePDF(AbsCosTheta(wi));
    }
    BxDFFlags Flags() const
    {
        return R ? BxDFFlags::DiffuseReflection : BxDFFlags::Unset;
    }
};

struct ConductorBSDF : BSDFCRTP<ConductorBSDF>
{
    ConductorBSDF() = delete;
    ConductorBSDF(TrowbridgeReitzDistribution mfDistrib, SampledSpectrumN eta, SampledSpectrumN k)
        : mfDistrib(mfDistrib), eta(eta), k(k) {}
    TrowbridgeReitzDistribution mfDistrib;
    SampledSpectrumN eta, k;

    SampledSpectrumN EvaluateSample(Vec3NF32 wo, Vec3NF32 wi, LaneNF32 &pdf, TransportMode mode) const
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
        LaneNF32 cosTheta_o = AbsCosTheta(wo);
        LaneNF32 cosTheta_i = AbsCosTheta(wi);
        if (cosTheta_i == 0 || cosTheta_o == 0)
        {
            pdf = 0.f;
            return {};
        }
        Vec3NF32 wm = wi + wo;
        if (LengthSquared(wm) == 0.f)
        {
            pdf = 0.f;
            return {};
        }
        wm                  = Normalize(wm);
        SampledSpectrumN Fr = FrComplex(AbsDot(wo, wm), eta, k);
        SampledSpectrumN f  = mfDistrib.D(wm) * Fr * mfDistrib.G(wo, wi) / (4 * cosTheta_i * cosTheta_o);
        wm                  = FaceForward(Normalize(wm), Vec3NF32(0, 0, 1));
        pdf                 = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm));
        return f;
    }
    BSDFSample GenerateSample(Vec3NF32 wo, LaneNF32 uc, Vec2NF32 u, TransportMode mode, BxDFFlags sampleFlags) const
    {
        if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::RT)) return {};
        if (mfDistrib.EffectivelySmooth())
        {
            Vec3NF32 wi(-wo.x, -wo.y, wo.z);
            SampledSpectrumN f = FrComplex(AbsCosTheta(wi), eta, k) / AbsCosTheta(wi);
            return BSDFSample(f, wi, 1.f, BxDFFlags::SpecularReflection);
        }
        Vec3NF32 wm = mfDistrib.Sample_wm(wo, u);
        Vec3NF32 wi = Reflect(wo, wm);
        if (!SameHemisphere(wo, wi)) return {};
        LaneNF32 pdf        = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm));
        LaneNF32 cosTheta_o = AbsCosTheta(wo);
        LaneNF32 cosTheta_i = AbsCosTheta(wi);
        // NOTE: Fresnel term with respect to the microfacet normal, wm
        SampledSpectrumN Fr = FrComplex(AbsDot(wo, wm), eta, k);
        // Torrance Sparrow BRDF Model (D * G * F / 4cos cos)
        SampledSpectrumN f = mfDistrib.D(wm) * Fr * mfDistrib.G(wo, wi) / (4 * cosTheta_i * cosTheta_o);
        return BSDFSample(f, wi, pdf, BxDFFlags::GlossyReflection);
    }
    LaneNF32 PDF(Vec3NF32 wo, Vec3NF32 wi, TransportMode mode, BxDFFlags sampleFlags = BxDFFlags::RT) const
    {
        if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection)) return 0.f;
        if (!SameHemisphere(wo, wi)) return 0.f;
        if (mfDistrib.EffectivelySmooth()) return 0.f;
        Vec3NF32 wm = wo + wi;
        if (LengthSquared(wm) == 0.f) return 0.f;
        wm = FaceForward(Normalize(wm), Vec3NF32(0, 0, 1));
        return mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm));
    }

    BxDFFlags Flags() const
    {
        return mfDistrib.EffectivelySmooth() ? BxDFFlags::SpecularReflection : BxDFFlags::GlossyReflection;
    }
};

struct DielectricBSDF : BSDFCRTP<DielectricBSDF>
{
    DielectricBSDF() = delete;
    DielectricBSDF(LaneNF32 eta, TrowbridgeReitzDistribution mfDistrib) : eta(eta), mfDistrib(mfDistrib) {}
    SampledSpectrumN EvaluateSample(Vec3NF32 wo, Vec3NF32 wi, LaneNF32 &pdf, TransportMode mode) const
    {
        if (eta == 1 || mfDistrib.EffectivelySmooth())
            return SampledSpectrumN(0.f);
        LaneNF32 cosTheta_o = CosTheta(wo);
        LaneNF32 cosTheta_i = CosTheta(wi);
        bool reflect        = cosTheta_i * cosTheta_o > 0.f;
        LaneNF32 etap       = 1.f;
        if (!reflect)
            etap = cosTheta_o > 0.f ? eta : 1 / eta;
        Vec3NF32 wm = wi * etap + wo;
        if (cosTheta_i == 0 || cosTheta_o == 0 || LengthSquared(wm) == 0.f)
            return {};
        wm = Normalize(wm);
        wm = wm.z < 0.f ? -wm : wm;
        if (Dot(wm, wi) * cosTheta_i < 0.f || Dot(wm, wo) * cosTheta_o < 0.f) return {};
        LaneNF32 F  = FrDielectric(Dot(wo, wm), eta);
        LaneNF32 T  = 1 - F;
        LaneNF32 pr = F;
        LaneNF32 pt = T;
        if (pr == 0.f && pt == 0.f) pdf = 0.f;
        if (reflect)
        {
            pdf = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)) * pr / (pr + pt);
            return SampledSpectrumN(mfDistrib.D(wm) * mfDistrib.G(wo, wi) * F / Abs(4 * cosTheta_i * cosTheta_o));
        }
        else
        {
            LaneNF32 denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
            LaneNF32 ft    = mfDistrib.D(wm) * (1 - F) * mfDistrib.G(wo, wi) * Abs(Dot(wi, wm) * Dot(wo, wm) / (denom * cosTheta_i * cosTheta_o));

            LaneNF32 dwm_dwi = AbsDot(wi, wm) / denom;
            pdf              = mfDistrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);
            if (mode == TransportMode::Radiance)
                ft /= Sqr(etap);
            return SampledSpectrumN(ft);
        }
    }
    BSDFSample GenerateSample(Vec3NF32 wo, LaneNF32 uc, Vec2NF32 u, TransportMode mode, BxDFFlags sampleFlags) const
    {
        // Sample specular BTDF
        if (eta == 1 || mfDistrib.EffectivelySmooth())
        {
            LaneNF32 R  = FrDielectric(CosTheta(wo), eta);
            LaneNF32 T  = 1 - R;
            LaneNF32 pr = R;
            LaneNF32 pt = T;
            if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection)) pr = 0.f;
            if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Transmission)) pt = 0.f;
            if (pr == 0.f && pt == 0.f) return {};
            // Sample based on the amount of reflection / transmission
            // Specular reflection
            if (uc < pr / (pr + pt))
            {
                Vec3NF32 wi(-wo.x, -wo.y, wo.z);
                SampledSpectrumN fr(R / AbsCosTheta(wi));
                return BSDFSample(fr, wi, pr / (pr + pt), BxDFFlags::SpecularReflection);
            }
            // Specular transmission
            else
            {
                Vec3NF32 wi;
                LaneNF32 etap;
                if (!Refract(wo, Vec3NF32(0, 0, 1), eta, &etap, &wi)) return {};
                SampledSpectrumN ft(T / AbsCosTheta(wi));
                if (mode == TransportMode::Radiance) ft /= etap * etap;
                return BSDFSample(ft, wi, pt / (pr + pt), BxDFFlags::SpecularTransmission, etap);
            }
        }
        // Rough dielectric BSDF
        else
        {
            Vec3NF32 wm = mfDistrib.Sample_wm(wo, u);
            LaneNF32 R  = FrDielectric(Dot(wo, wm), eta);
            LaneNF32 T  = 1 - R;
            LaneNF32 pr = R;
            LaneNF32 pt = T;
            if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection)) pr = 0.f;
            if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Transmission)) pt = 0.f;
            if (pr == 0.f && pt == 0.f) return {};
            // Glossy reflection
            if (uc < pr / (pr + pt))
            {
                Vec3NF32 wi = Reflect(wo, wm);
                if (!SameHemisphere(wo, wi)) return {};
                LaneNF32 pdf = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)) * pr / (pr + pt);
                SampledSpectrumN f(mfDistrib.D(wm) * mfDistrib.G(wo, wi) * R / (4 * CosTheta(wi) * CosTheta(wo)));
                return BSDFSample(f, wi, pdf, BxDFFlags::GlossyReflection);
            }
            // Glossy transmission
            else
            {
                Vec3NF32 wi;
                LaneNF32 etap;
                if (!Refract(wo, wm, eta, &etap, &wi) || SameHemisphere(wo, wi) || wi.z == 0.f) return {};
                LaneNF32 denom   = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
                LaneNF32 dwm_dwi = AbsDot(wi, wm) / denom;
                LaneNF32 pdf     = mfDistrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);
                SampledSpectrumN ft(T * mfDistrib.D(wm) * mfDistrib.G(wo, wi) *
                                    Abs(Dot(wi, wm) * Dot(wo, wm) / (CosTheta(wi) * CosTheta(wo) * denom)));
                if (mode == TransportMode::Radiance) ft /= etap * etap;
                return BSDFSample(ft, wi, pdf, BxDFFlags::GlossyTransmission, etap);
            }
        }
    }
    BxDFFlags Flags() const
    {
        BxDFFlags flags = (eta == 1.f) ? BxDFFlags::Transmission : (BxDFFlags::RT);
        return flags | (mfDistrib.EffectivelySmooth() ? BxDFFlags::Specular : BxDFFlags::Glossy);
    }
    LaneNF32 PDF(Vec3NF32 wo, Vec3NF32 wi, TransportMode mode, BxDFFlags sampleFlags = BxDFFlags::RT) const
    {
        if (eta == 1.f || mfDistrib.EffectivelySmooth())
            return 0.f;
        LaneNF32 cosTheta_o = CosTheta(wo);
        LaneNF32 cosTheta_i = CosTheta(wi);
        bool reflect        = cosTheta_i * cosTheta_o > 0.f;
        LaneNF32 etap       = 1.f;
        if (!reflect)
            etap = cosTheta_o > 0.f ? eta : 1.f / eta;
        // Calculate the half angle, accounting for transmission
        Vec3NF32 wm = wi * etap + wo;
        if (cosTheta_i == 0 || cosTheta_o == 0 || LengthSquared(wm) == 0.f) return {};
        wm = wm.z < 0.f ? -wm : wm;
        if (Dot(wm, wi) * cosTheta_i < 0 || Dot(wm, wo) * cosTheta_o < 0) return {};
        LaneNF32 R  = FrDielectric(Dot(wo, wm), eta);
        LaneNF32 T  = 1 - R;
        LaneNF32 pr = R;
        LaneNF32 pt = T;
        if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection)) pr = 0.f;
        if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Transmission)) pt = 0.f;
        if (pr == 0.f && pt == 0.f) return 0.f;
        LaneNF32 pdf;
        if (reflect)
        {
            pdf = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)) * pr / (pr + pt);
        }
        else
        {
            LaneNF32 denom   = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
            LaneNF32 dwm_dwi = AbsDot(wi, wm) / denom;
            pdf              = mfDistrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);
        }
        return pdf;
    }

    TrowbridgeReitzDistribution mfDistrib;
    // NOTE: spectrally varying IORs are handled by randomly sampling a single wavelength
    LaneNF32 eta;
};

// NOTE: only models perfect specular scattering
struct ThinDielectricBSDF : BSDFCRTP<ThinDielectricBSDF>
{
    ThinDielectricBSDF() = default;
    ThinDielectricBSDF(LaneNF32 eta) : eta(eta) {}
    SampledSpectrumN EvaluateSample(Vec3NF32 wo, Vec3NF32 wi, LaneNF32 &pdf, TransportMode mode) const
    {
        pdf = 0.f;
        return SampledSpectrumN(0.f);
    }
    BSDFSample GenerateSample(Vec3NF32 wo, LaneNF32 uc, Vec2NF32 u, TransportMode mode, BxDFFlags sampleFlags) const
    {
        LaneNF32 R = FrDielectric(AbsCosTheta(wo), eta);
        LaneNF32 T = 1 - R;
        if (R < 1)
        {
            R += Sqr(T) * R / (1 - Sqr(R));
            T = 1 - R;
        }
        LaneNF32 pr = R;
        LaneNF32 pt = T;
        if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection)) pr = 0.f;
        if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Transmission)) pt = 0.f;
        if (pr == 0.f && pt == 0.f) return {};
        // TODO: this is the same as in the dielectric case. Compress?
        if (uc < pr / (pr + pt))
        {
            Vec3NF32 wi(-wo.x, -wo.y, wo.z);
            SampledSpectrumN fr(R / AbsCosTheta(wi));
            return BSDFSample(fr, wi, pr / (pr + pt), BxDFFlags::SpecularReflection);
        }
        Vec3NF32 wi = -wo;
        SampledSpectrumN ft(T / AbsCosTheta(wi));
        return BSDFSample(ft, wi, pt / (pr + pt), BxDFFlags::SpecularTransmission);
    }
    LaneNF32 PDF(Vec3NF32 wo, Vec3NF32 wi, TransportMode mode, BxDFFlags sampleFlags) const
    {
        return 0.f;
    }
    BxDFFlags Flags() const
    {
        return BxDFFlags::Reflection | BxDFFlags::Transmission | BxDFFlags::Specular;
    }

    LaneNF32 eta;
};

} // namespace rt

#endif
