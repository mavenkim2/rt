#ifndef BXDF_H
#define BXDF_H

namespace rt
{

// TODO: multiple scattering, better hair models, light polarization (ew)

// NOTE: wi and wt point away from the surface point
MaskF32 Refract(Vec3lfn wi, Vec3lfn n, LaneNF32 eta, LaneNF32 *etap, Vec3lfn *wt)
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

LaneNF32 FrDielectric(LaneNF32 cosTheta_i, LaneNF32 eta)
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
LaneNF32 FrComplex(LaneNF32 cosTheta_i, const Complex<LaneNF32> &eta)
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
LaneNF32 CosTheta(Vec3lfn w)
{
    return w.z;
}

LaneNF32 Cos2Theta(Vec3lfn w)
{
    return w.z * w.z;
}

LaneNF32 AbsCosTheta(Vec3lfn w)
{
    return Abs(w.z);
}

LaneNF32 Sin2Theta(Vec3lfn w)
{
    return Max(0.f, 1 - Cos2Theta(w));
}

LaneNF32 SinTheta(Vec3lfn w)
{
    return sqrtf(Sin2Theta(w));
}

LaneNF32 TanTheta(Vec3lfn w)
{
    return SinTheta(w) / CosTheta(w);
}

LaneNF32 Tan2Theta(Vec3lfn w)
{
    return Sin2Theta(w) / Cos2Theta(w);
}

LaneNF32 CosPhi(Vec3lfn w)
{
    LaneNF32 sinTheta = SinTheta(w);
    return Select(sinTheta == 0, 1, Clamp(w.x / sinTheta, -1.f, 1.f));
}

LaneNF32 SinPhi(Vec3lfn w)
{
    LaneNF32 sinTheta = SinTheta(w);
    return Select(sinTheta == 0, 0, Clamp(w.y / sinTheta, -1.f, 1.f));
}

MaskF32 SameHemisphere(Vec3lfn w, Vec3lfn wp)
{
    return w.z * wp.z > 0;
}

Vec3lfn FaceForward(Vec3lfn n, Vec3lfn v)
{
    return Select(Dot(n, v) < 0.f, -n, n);
}

struct TrowbridgeReitzDistribution
{
    TrowbridgeReitzDistribution(const LaneNF32 &alphaX, const LaneNF32 &alphaY) : alphaX(alphaX), alphaY(alphaY) {}
    LaneNF32 D(const Vec3lfn &wm) const
    {
        LaneNF32 tan2Theta = Tan2Theta(wm);
        LaneNF32 cos2Theta = Cos2Theta(wm);
        LaneNF32 cos4Theta = cos2Theta * cos2Theta;
        LaneNF32 e         = tan2Theta * (Sqr(CosPhi(wm) / alphaX) +
                                  Sqr(SinPhi(wm) / alphaY));
        return 1 / (PI * alphaX * alphaY * cos4Theta * Sqr(1 + e));
    }
    LaneNF32 G1(const Vec3lfn &w) const
    {
        return 1 / (1 + Lambda(w));
    }
    // NOTE: height based correlation; microfacets at a greater height are more likely to be visible
    LaneNF32 G(const Vec3lfn &wo, const Vec3lfn &wi) const
    {
        return 1 / (1 + Lambda(wo) + Lambda(wi));
    }
    LaneNF32 D(const Vec3lfn &w, const Vec3lfn &wm) const
    {
        return G1(w) / AbsCosTheta(w) * D(wm) * AbsDot(w, wm);
    }
    LaneNF32 Lambda(const Vec3lfn &w) const
    {
        LaneNF32 tan2Theta = Tan2Theta(w);
        LaneNF32 alpha2    = Sqr(CosPhi(w) * alphaX) + Sqr(SinPhi(w) * alphaY);
        return Select(tan2Theta == LaneNF32(pos_inf), 0, (Sqrt(1 + alpha2 * tan2Theta) - 1) / 2);
    }
    LaneNF32 PDF(const Vec3lfn &w, const Vec3lfn &wm) const
    {
        return D(w, wm);
    }
    // NOTE: samples the visible normals instead of simply the distribution function
    Vec3lfn Sample_wm(const Vec3lfn &w, const Vec2lfn &u) const
    {
        Vec3lfn wh = Normalize(Vec3lfn(alphaX * w.x, alphaY * w.y, w.z));
        wh         = Select(wh.z < 0, -wh, wh);
        // NOTE: this process involves the projection of a disk of uniformly distributed points onto a truncated ellipsoid.
        // The inverse of the scale factor of the ellipsoid is applied to the incoming direction in order to simplify to the
        // isotropic case.
        Vec2lfn p  = SampleUniformDiskPolar(u);
        Vec3lfn T1 = Select(wh.z < 0.99999f, Normalize(Cross(Vec3lfn(0, 0, 1), wh)), Vec3lfn(1, 0, 0));
        Vec3lfn T2 = Cross(wh, T1);
        // NOTE: For a given x, the y component has a value in range [-h, h], where h = sqrt(1 - Sqr(x)). When the
        // projection is not perpendicular to the ellipsoid, the range shrinks to [-hcos(theta), h]. This requires an
        // affine transformation with scale 0.5(1 + cosTheta) and translation 0.5h(1 - cosTheta).

        LaneNF32 h = Sqrt(1 - p.x * p.x);
        p.y        = Lerp((1.f + wh.z) / 2.f, h, p.y);

        // Project point to hemisphere, transform to ellipsoid.
        LaneNF32 pz = Sqrt(Max(0.f, 1 - p.x * p.x - p.y * p.y));
        Vec3lfn nh  = p.x * T1 + p.y * T2 + pz * wh;
        return Normalize(Vec3lfn(alphaX * nh.x, alphaY * nh.y, Max(1e-6f, nh.z)));
    }
    MaskF32 EffectivelySmooth() const
    {
        return Max(alphaX, alphaY) < 1e-3f;
    }
    LaneNF32 alphaX;
    LaneNF32 alphaY;
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

inline MaskF32 IsReflective(const LaneNU32 &f) { return (f & LaneNU32(u32(BxDFFlags::Reflection))) != 0; }
inline MaskF32 IsTransmissive(const LaneNU32 &f) { return (f & LaneNU32(u32(BxDFFlags::Transmission))) != 0; }
inline MaskF32 IsDiffuse(const LaneNU32 &f) { return (f & LaneNU32(u32(BxDFFlags::Diffuse))) != 0; }
inline MaskF32 IsGlossy(const LaneNU32 &f) { return (f & LaneNU32(u32(BxDFFlags::Glossy))) != 0; }
inline MaskF32 IsSpecular(const LaneNU32 &f) { return (f & LaneNU32(u32(BxDFFlags::Specular))) != 0; }
inline MaskF32 IsNonSpecular(const LaneNU32 &f) { return (f & LaneNU32(u32(BxDFFlags::Diffuse | BxDFFlags::Glossy))) != 0; }
inline MaskF32 IsValid(const LaneNU32 &f) { return (f & LaneNU32(u32(BxDFFlags::Invalid))) == 0; }

struct BSDFSample
{
    SampledSpectrumN f;
    Vec3lfn wi;
    LaneNF32 pdf;
    LaneNF32 eta;
    LaneNU32 flags;
    bool pdfIsProportional;

    BSDFSample() = default;

    BSDFSample(const SampledSpectrumN &f, const Vec3lfn &wi, const LaneNF32 &pdf,
               const LaneNU32 &flags, const LaneNF32 &eta = 1.f, bool pdfIsProportional = false)
        : f(f), wi(wi), pdf(pdf), flags(flags), eta(eta), pdfIsProportional(pdfIsProportional) {}

    MaskF32 IsReflective() { return rt::IsReflective(flags); }
    MaskF32 IsTransmissive() { return rt::IsTransmissive(flags); }
    MaskF32 IsDiffuse() { return rt::IsDiffuse(flags); }
    MaskF32 IsGlossy() { return rt::IsGlossy(flags); }
    MaskF32 IsSpecular() { return rt::IsSpecular(flags); }
    MaskF32 IsValid() { return !(rt::IsValid(flags)); }
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
    DiffuseBxDF() = default;
    DiffuseBxDF(const SampledSpectrumN &R) : R(R) {}

    SampledSpectrumN EvaluateSample(const Vec3lfn &wo, const Vec3lfn &wi, LaneNF32 &pdf, TransportMode mode) const
    {
        MaskF32 mask = SameHemisphere(wo, wi);
        pdf          = Select(mask, CosineHemispherePDF(AbsCosTheta(wi)), 0.f);
        return R / PI;
    }

    BSDFSample GenerateSample(const Vec3lfn &wo, const LaneNF32 &uc, const Vec2lfn &u,
                              TransportMode mode = TransportMode::Radiance, BxDFFlags sampleFlags = BxDFFlags::RT) const
    {
        if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection)) return {};
        Vec3lfn wi   = SampleCosineHemisphere(u);
        wi.z         = Select(wo.z < 0, -wi.z, wi.z);
        LaneNF32 pdf = CosineHemispherePDF(AbsCosTheta(wi));
        return BSDFSample(R * InvPi, wi, pdf, LaneNU32(u32(BxDFFlags::DiffuseReflection)));
    }
    // LaneNF32 PDF(Vec3lfn wo, Vec3lfn wi, TransportMode mode, BxDFFlags sampleFlags = BxDFFlags::RT) const
    // {
    //     if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection) || !SameHemisphere(wo, wi)) return 0.f;
    //     return CosineHemispherePDF(AbsCosTheta(wi));
    // }

    LaneNU32 Flags() const
    {
        return Select(MaskF32(R), LaneNU32(u32(BxDFFlags::DiffuseReflection)), LaneNU32(u32(BxDFFlags::Unset)));
    }
};

struct DiffuseTransmissionBxDF : BxDFCRTP<DiffuseTransmissionBxDF>
{
    SampledSpectrumN R, T;
    DiffuseTransmissionBxDF() = default;
    DiffuseTransmissionBxDF(const SampledSpectrumN &R, const SampledSpectrumN &T) : R(R), T(T) {}

    SampledSpectrumN EvaluateSample(const Vec3lfn &wo, const Vec3lfn &wi, LaneNF32 &pdf, TransportMode mode) const
    {
        MaskF32 reflectMask = SameHemisphere(wo, wi);
        LaneNF32 pr         = R.MaxComponentValue();
        LaneNF32 pt         = T.MaxComponentValue();
        pdf                 = CosineHemispherePDF(AbsCosTheta(wi)) * Select(reflectMask, pr, pt) / (pr + pt);
        return Select(reflectMask, R, T) * InvPi;
    }

    BSDFSample GenerateSample(const Vec3lfn &wo, const LaneNF32 &uc, const Vec2lfn &u,
                              TransportMode mode = TransportMode::Radiance, BxDFFlags sampleFlags = BxDFFlags::RT) const
    {
        if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection)) return {};
        LaneNF32 pr = R.MaxComponentValue();
        LaneNF32 pt = T.MaxComponentValue();

        LaneNF32 sum = pr + pt;

        MaskF32 reflectMask = uc < pr / sum;
        Vec3lfn wi          = SampleCosineHemisphere(u);
        wi.z                = Select(reflectMask ^ wo.z > 0, -wi.z, wi.z);
        LaneNF32 pdf        = CosineHemispherePDF(AbsCosTheta(wi));
        return BSDFSample(Select(reflectMask, R, T) * InvPi, wi, pdf,
                          Select(reflectMask,
                                 LaneNU32(u32(BxDFFlags::DiffuseReflection)),
                                 LaneNU32(u32(BxDFFlags::DiffuseTransmission))));
    }
    LaneNU32 Flags() const
    {
        return Select(MaskF32(R), LaneNU32(u32(BxDFFlags::DiffuseReflection)), LaneNU32(u32(BxDFFlags::Unset))) |
               Select(MaskF32(T), LaneNU32(u32(BxDFFlags::DiffuseTransmission)), LaneNU32(u32(BxDFFlags::Unset)));
    }
};

struct ConductorBxDF : BxDFCRTP<ConductorBxDF>
{
    ConductorBxDF() = delete;
    ConductorBxDF(const TrowbridgeReitzDistribution &mfDistrib, const SampledSpectrumN &eta, const SampledSpectrumN &k)
        : mfDistrib(mfDistrib), eta(eta), k(k) {}
    TrowbridgeReitzDistribution mfDistrib;
    SampledSpectrumN eta, k;

    SampledSpectrumN EvaluateSample(const Vec3lfn &wo, const Vec3lfn &wi, LaneNF32 &pdf, TransportMode mode) const
    {
        MaskF32 mask = SameHemisphere(wo, wi);
        mask &= mfDistrib.EffectivelySmooth();

        LaneNF32 cosTheta_o = AbsCosTheta(wo);
        LaneNF32 cosTheta_i = AbsCosTheta(wi);
        mask &= cosTheta_i != 0 && cosTheta_o != 0;

        Vec3lfn wm = wi + wo;
        mask &= LengthSquared(wm) != 0.f;

        wm                  = Normalize(wm);
        SampledSpectrumN Fr = FrComplex(AbsDot(wo, wm), eta, k);
        SampledSpectrumN f  = mfDistrib.D(wm) * Fr * mfDistrib.G(wo, wi) / (4 * cosTheta_i * cosTheta_o);
        wm                  = FaceForward(Normalize(wm), Vec3lfn(0, 0, 1));

        pdf = Select(mask, mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)), 0.f);
        return f;
    }
    BSDFSample GenerateSample(const Vec3lfn &wo, const LaneNF32 &uc, const Vec2lfn &u,
                              TransportMode mode, BxDFFlags sampleFlags) const
    {
        if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection)) return {};

        // MaskF32 mask(true);
        SampledSpectrumN f;
        Vec3lfn wi;
        LaneNF32 pdf;
        // LaneNU32 flags;

        MaskF32 specularMask = mfDistrib.EffectivelySmooth();
        MaskF32 validSampleMask(true);
        if (Any(specularMask))
        {
            f   = FrComplex(AbsCosTheta(wi), eta, k) / AbsCosTheta(wi);
            wi  = Vec3lfn(-wo.x, -wo.y, wo.z);
            pdf = 1.f;
        }
        if (!All(specularMask))
        {
            Vec3lfn wm      = mfDistrib.Sample_wm(wo, u);
            wi              = Select(specularMask, wi, Reflect(wo, wm));
            validSampleMask = Select(specularMask, validSampleMask, SameHemisphere(wo, wi));
            // if (!SameHemisphere(wo, wi)) return {};
            pdf                 = Select(specularMask, pdf, mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)));
            LaneNF32 cosTheta_o = AbsCosTheta(wo);
            LaneNF32 cosTheta_i = AbsCosTheta(wi);
            // NOTE: Fresnel term with respect to the microfacet normal, wm
            SampledSpectrumN Fr = FrComplex(AbsDot(wo, wm), eta, k);
            // Torrance Sparrow BRDF Model (D * G * F / 4cos cos)
            f = Select(specularMask, f, mfDistrib.D(wm) * Fr * mfDistrib.G(wo, wi) / (4 * cosTheta_i * cosTheta_o));
        }
        LaneNU32 flags = Select(specularMask,
                                LaneNU32(u32(BxDFFlags::SpecularReflection)), LaneNU32(u32(BxDFFlags::GlossyReflection)));
        return BSDFSample(f, wi, pdf, flags);
    }
    // LaneNF32 PDF(Vec3lfn wo, Vec3lfn wi, TransportMode mode, BxDFFlags sampleFlags = BxDFFlags::RT) const
    // {
    //     if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection)) return 0.f;
    //     if (!SameHemisphere(wo, wi)) return 0.f;
    //     if (mfDistrib.EffectivelySmooth()) return 0.f;
    //     Vec3lfn wm = wo + wi;
    //     if (LengthSquared(wm) == 0.f) return 0.f;
    //     wm = FaceForward(Normalize(wm), Vec3lfn(0, 0, 1));
    //     return mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm));
    // }

    LaneNU32 Flags() const
    {
        return Select(mfDistrib.EffectivelySmooth(), LaneNU32(u32(BxDFFlags::SpecularReflection)),
                      LaneNU32(u32(BxDFFlags::GlossyReflection)));
    }
};

struct DielectricBxDF : BxDFCRTP<DielectricBxDF>
{
    TrowbridgeReitzDistribution mfDistrib;
    // NOTE: spectrally varying IORs are handled by randomly sampling a single wavelength
    LaneNF32 eta;

    DielectricBxDF() = delete;
    DielectricBxDF(const LaneNF32 &eta, const TrowbridgeReitzDistribution &mfDistrib) : eta(eta), mfDistrib(mfDistrib) {}
    SampledSpectrumN EvaluateSample(const Vec3lfn &wo, const Vec3lfn &wi, LaneNF32 &pdf, TransportMode mode) const
    {
        MaskF32 mask = !(eta == 1 || mfDistrib.EffectivelySmooth());
        if (None(mask))
        {
            pdf = 0.f;
            return {};
        }

        LaneNF32 cosTheta_o = CosTheta(wo);
        LaneNF32 cosTheta_i = CosTheta(wi);
        MaskF32 reflect     = cosTheta_i * cosTheta_o > 0.f;
        LaneNF32 etap       = Select(reflect, 1.f, Select(cosTheta_o > 0.f, eta, 1 / eta));

        Vec3lfn wm = wi * etap + wo;

        mask &= !(cosTheta_i == 0 || cosTheta_o == 0 || LengthSquared(wm) == 0.f);

        wm = Normalize(wm);
        wm = Select(wm.z < 0.f, -wm, wm);
        mask &= !(Dot(wm, wi) * cosTheta_i < 0.f || Dot(wm, wo) * cosTheta_o < 0.f);

        if (None(mask))
        {
            pdf = 0.f;
            return {};
        }

        LaneNF32 F  = FrDielectric(Dot(wo, wm), eta);
        LaneNF32 T  = 1 - F;
        LaneNF32 pr = F;
        LaneNF32 pt = T;

        SampledSpectrumN f;
        if (Any(reflect))
        {
            pdf = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)) * pr / (pr + pt);
            f   = SampledSpectrumN(mfDistrib.D(wm) * mfDistrib.G(wo, wi) * F / Abs(4 * cosTheta_i * cosTheta_o));
        }
        if (!All(reflect))
        {
            LaneNF32 denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
            SampledSpectrumN ft(mfDistrib.D(wm) * (1 - F) * mfDistrib.G(wo, wi) * Abs(Dot(wi, wm) * Dot(wo, wm) / (denom * cosTheta_i * cosTheta_o)));

            LaneNF32 dwm_dwi = AbsDot(wi, wm) / denom;
            pdf              = Select(reflect, pdf, mfDistrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt));
            ft               = Select(mode == TransportMode::Radiance, ft / Sqr(etap), ft);
            f                = Select(reflect, f, ft);
        }
        pdf = Select(mask, pdf, 0.f);
        return f;
    }
    BSDFSample GenerateSample(const Vec3lfn &wo, const LaneNF32 &uc, const Vec2lfn &u,
                              TransportMode mode, BxDFFlags sampleFlags) const
    {
        // Sample specular BTDF

        MaskF32 specularMask = eta == 1 || mfDistrib.EffectivelySmooth();
        SampledSpectrumN f;
        Vec3lfn wi;
        LaneNF32 pdf;
        LaneNU32 flags;
        LaneNF32 etap = 1.f;

        Vec3lfn wm       = Vec3lfn(0, 0, 1);
        bool anySpecular = Any(specularMask);
        bool anyGlossy   = !All(specularMask);

        LaneNF32 R;
        if (anySpecular)
        {
            wm = Select(specularMask, wm, mfDistrib.Sample_wm(wo, u));
            R  = FrDielectric(CosTheta(wo), eta);
        }
        if (anyGlossy)
        {
            R = Select(specularMask, R, FrDielectric(Dot(wo, wm), eta));
        }
        LaneNF32 T  = 1 - R;
        LaneNF32 pr = R;
        LaneNF32 pt = T;
        if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection)) pr = 0.f;
        if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Transmission)) pt = 0.f;
        if (pr == 0.f && pt == 0.f) return {};
        // Sample based on the amount of reflection / transmission
        // Reflection
        MaskF32 reflect = uc < pr / (pr + pt);
        if (Any(reflect))
        {
            MaskF32 specularReflectionMask = specularMask & reflect;
            if (Any(specularReflectionMask))
            {
                f     = SampledSpectrumN(R / AbsCosTheta(wi));
                wi    = Vec3lfn(-wo.x, -wo.y, wo.z);
                pdf   = pr / (pr + pt);
                flags = LaneNU32(u32(BxDFFlags::SpecularReflection));
            }
            MaskF32 glossyReflectionMask = !specularMask & reflect;
            if (Any(glossyReflectionMask))
            {
                f     = Select(glossyReflectionMask,
                               SampledSpectrumN(mfDistrib.D(wm) * mfDistrib.G(wo, wi) * R / (4 * CosTheta(wi) * CosTheta(wo))),
                               f);
                wi    = Select(glossyReflectionMask, Reflect(wo, wm), wi);
                pdf   = Select(glossyReflectionMask, mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)) * pr / (pr + pt), pdf);
                pdf   = Select(SameHemisphere(wo, wi), pdf, 0.f);
                flags = Select(glossyReflectionMask, LaneNU32(u32(BxDFFlags::GlossyReflection)), flags);
            }
        }
        // Transmission
        if (!All(reflect))
        {
            LaneNF32 tempEtap;
            Vec3lfn tempWi;
            SampledSpectrum ft;
            MaskF32 refractMask = Refract(wo, wm, eta, &tempEtap, &tempWi);

            MaskF32 specularTransmissionMask = specularMask & !reflect;
            if (Any(specularTransmissionMask))
            {
                ft  = SampledSpectrumN(T / AbsCosTheta(tempWi));
                pdf = Select(specularTransmissionMask, pt / (pr + pt), pdf);

                flags = Select(specularTransmissionMask, LaneNU32(u32(BxDFFlags::SpecularTransmission)), flags);
            }

            MaskF32 glossyTransmissionMask = !(specularMask | reflect);
            if (Any(glossyTransmissionMask))
            {
                refractMask &= Select(specularMask, !(SameHemisphere(wo, wi) | (wi.z == 0.f)), refractMask);

                LaneNF32 denom   = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
                LaneNF32 dwm_dwi = AbsDot(wi, wm) / denom;
                pdf              = Select(glossyTransmissionMask, mfDistrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt), pdf);
                ft               = Select(glossyTransmissionMask,
                                          SampledSpectrumN(T * mfDistrib.D(wm) * mfDistrib.G(wo, wi) *
                                                           Abs(Dot(wi, wm) * Dot(wo, wm) / (CosTheta(wi) * CosTheta(wo) * denom))),
                                          ft);
                flags            = Select(glossyTransmissionMask, LaneNU32(u32(BxDFFlags::GlossyTransmission)), flags);
            }

            if (mode == TransportMode::Radiance) ft /= etap * etap;

            f    = Select(reflect, f, ft);
            wi   = Select(reflect, wi, tempWi);
            pdf  = Select(reflect, pdf, Select(refractMask, pdf, 0.f));
            etap = Select(reflect, 1.f, tempEtap);
        }
        return BSDFSample(f, wi, pdf, flags, etap);
    }
    LaneNU32 Flags() const
    {
        LaneNU32 flags = Select(eta == 1.f, u32(BxDFFlags::Transmission), u32(BxDFFlags::RT));
        return flags | Select(mfDistrib.EffectivelySmooth(), u32(BxDFFlags::Specular), u32(BxDFFlags::Glossy));
    }
    // LaneNF32 PDF(Vec3lfn wo, Vec3lfn wi, TransportMode mode, BxDFFlags sampleFlags = BxDFFlags::RT) const
    // {
    //     if (eta == 1.f || mfDistrib.EffectivelySmooth())
    //         return 0.f;
    //     LaneNF32 cosTheta_o = CosTheta(wo);
    //     LaneNF32 cosTheta_i = CosTheta(wi);
    //     bool reflect        = cosTheta_i * cosTheta_o > 0.f;
    //     LaneNF32 etap       = 1.f;
    //     if (!reflect)
    //         etap = cosTheta_o > 0.f ? eta : 1.f / eta;
    //     // Calculate the half angle, accounting for transmission
    //     Vec3lfn wm = wi * etap + wo;
    //     if (cosTheta_i == 0 || cosTheta_o == 0 || LengthSquared(wm) == 0.f) return {};
    //     wm = wm.z < 0.f ? -wm : wm;
    //     if (Dot(wm, wi) * cosTheta_i < 0 || Dot(wm, wo) * cosTheta_o < 0) return {};
    //     LaneNF32 R  = FrDielectric(Dot(wo, wm), eta);
    //     LaneNF32 T  = 1 - R;
    //     LaneNF32 pr = R;
    //     LaneNF32 pt = T;
    //     if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection)) pr = 0.f;
    //     if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Transmission)) pt = 0.f;
    //     if (pr == 0.f && pt == 0.f) return 0.f;
    //     LaneNF32 pdf;
    //     if (reflect)
    //     {
    //         pdf = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)) * pr / (pr + pt);
    //     }
    //     else
    //     {
    //         LaneNF32 denom   = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
    //         LaneNF32 dwm_dwi = AbsDot(wi, wm) / denom;
    //         pdf              = mfDistrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);
    //     }
    //     return pdf;
    // }
};

// NOTE: only models perfect specular scattering
// struct ThinDielectricBxDF : BxDFCRTP<ThinDielectricBxDF>
// {
//     ThinDielectricBxDF() = default;
//     ThinDielectricBxDF(LaneNF32 eta) : eta(eta) {}
//     SampledSpectrumN EvaluateSample(Vec3lfn wo, Vec3lfn wi, LaneNF32 &pdf, TransportMode mode) const
//     {
//         pdf = 0.f;
//         return SampledSpectrumN(0.f);
//     }
//     BSDFSample GenerateSample(Vec3lfn wo, LaneNF32 uc, Vec2lfn u, TransportMode mode, BxDFFlags sampleFlags) const
//     {
//         LaneNF32 R = FrDielectric(AbsCosTheta(wo), eta);
//         LaneNF32 T = 1 - R;
//         if (R < 1)
//         {
//             R += Sqr(T) * R / (1 - Sqr(R));
//             T = 1 - R;
//         }
//         LaneNF32 pr = R;
//         LaneNF32 pt = T;
//         if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection)) pr = 0.f;
//         if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Transmission)) pt = 0.f;
//         if (pr == 0.f && pt == 0.f) return {};
//         // TODO: this is the same as in the dielectric case. Compress?
//         if (uc < pr / (pr + pt))
//         {
//             Vec3lfn wi(-wo.x, -wo.y, wo.z);
//             SampledSpectrumN fr(R / AbsCosTheta(wi));
//             return BSDFSample(fr, wi, pr / (pr + pt), BxDFFlags::SpecularReflection);
//         }
//         Vec3lfn wi = -wo;
//         SampledSpectrumN ft(T / AbsCosTheta(wi));
//         return BSDFSample(ft, wi, pt / (pr + pt), BxDFFlags::SpecularTransmission);
//     }
//     LaneNF32 PDF(Vec3lfn wo, Vec3lfn wi, TransportMode mode, BxDFFlags sampleFlags) const
//     {
//         return 0.f;
//     }
//     BxDFFlags Flags() const
//     {
//         return BxDFFlags::Reflection | BxDFFlags::Transmission | BxDFFlags::Specular;
//     }
//
//     LaneNF32 eta;
// };

} // namespace rt

#endif
