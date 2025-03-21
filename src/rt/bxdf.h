#ifndef BXDF_H
#define BXDF_H

#include "base.h"
#include "spectrum.h"
#include "sampling.h"
#include <cmath>
namespace rt
{

// TODO: multiple scattering, better hair models, light polarization (ew)
inline Vec3f Reflect(const Vec3f &v, const Vec3f &norm)
{
    return -v + 2 * Dot(v, norm) * norm;
}

// NOTE: wi and wt point away from the surface point
inline MaskF32 Refract(Vec3lfn wi, Vec3lfn n, LaneNF32 eta, LaneNF32 *etap = 0, Vec3lfn *wt = 0)
{
    LaneNF32 cosTheta_i = Dot(wi, n);

    MaskF32 mask = cosTheta_i > 0;
    n            = Select(mask, n, -n);
    eta          = Select(mask, eta, 1 / eta);
    cosTheta_i   = Select(mask, cosTheta_i, -cosTheta_i);

    LaneNF32 sin2Theta_i = Max(0.f, 1 - cosTheta_i * cosTheta_i);
    // Snell's law
    LaneNF32 sin2Theta_t = sin2Theta_i / (eta * eta);
    // Total internal eeflection

    mask = sin2Theta_t < 1;
    // if (sin2Theta_t >= 1) return false;
    LaneNF32 cosTheta_t = SafeSqrt(1 - sin2Theta_t);
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

inline LaneNF32 FrDielectric(LaneNF32 cosTheta_i, LaneNF32 eta)
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

// Conductors have refractive index with imaginary component: eta - ik, where k is the
// absoroption coefficient
inline LaneNF32 FrComplex(LaneNF32 cosTheta_i, const Complex<LaneNF32> &eta)
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

inline SampledSpectrumN FrComplex(LaneNF32 cosTheta_i, SampledSpectrumN eta, SampledSpectrumN k)
{
    SampledSpectrumN result;
    for (i32 i = 0; i < NSampledWavelengths; ++i)
        result[i] = FrComplex(cosTheta_i, Complex(eta[i], k[i]));
    return result;
}

// NOTE: BSDF calculations operate in the frame where the geometric normal is the z axis. Thus,
// the angle between the surface normal and a vector in that frame is just the z component.
inline LaneNF32 CosTheta(Vec3lfn w) { return w.z; }

inline LaneNF32 Cos2Theta(Vec3lfn w) { return w.z * w.z; }

inline LaneNF32 AbsCosTheta(Vec3lfn w) { return Abs(w.z); }
inline LaneNF32 Sin2Theta(Vec3lfn w) { return Max(0.f, 1 - Cos2Theta(w)); }
inline LaneNF32 SinTheta(Vec3lfn w) { return Sqrt(Sin2Theta(w)); }
inline LaneNF32 TanTheta(Vec3lfn w) { return SinTheta(w) / CosTheta(w); }
inline LaneNF32 Tan2Theta(Vec3lfn w) { return Sin2Theta(w) / Cos2Theta(w); }
inline LaneNF32 CosPhi(Vec3lfn w)
{
    LaneNF32 sinTheta = SinTheta(w);
    return Select(sinTheta == 0, 1, Clamp(w.x / sinTheta, -1.f, 1.f));
}

inline LaneNF32 SinPhi(Vec3lfn w)
{
    LaneNF32 sinTheta = SinTheta(w);
    return Select(sinTheta == 0, 0, Clamp(w.y / sinTheta, -1.f, 1.f));
}

inline MaskF32 SameHemisphere(Vec3lfn w, Vec3lfn wp) { return w.z * wp.z > 0; }

inline Vec3lfn FaceForward(Vec3lfn n, Vec3lfn v) { return Select(Dot(n, v) < 0.f, -n, n); }

struct TrowbridgeReitzDistribution
{
    TrowbridgeReitzDistribution() {}
    TrowbridgeReitzDistribution(const LaneNF32 &alphaX, const LaneNF32 &alphaY)
        : alphaX(alphaX), alphaY(alphaY)
    {
    }
    LaneNF32 D(const Vec3lfn &wm) const
    {
        LaneNF32 tan2Theta = Tan2Theta(wm);
        LaneNF32 cos2Theta = Cos2Theta(wm);
        LaneNF32 cos4Theta = cos2Theta * cos2Theta;
        LaneNF32 e         = tan2Theta * (Sqr(CosPhi(wm) / alphaX) + Sqr(SinPhi(wm) / alphaY));
        return 1 / (PI * alphaX * alphaY * cos4Theta * Sqr(1 + e));
    }
    LaneNF32 G1(const Vec3lfn &w) const { return 1 / (1 + Lambda(w)); }
    // NOTE: height based correlation; microfacets at a greater height are more likely to be
    // visible
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
        return Select(tan2Theta == LaneNF32(pos_inf), 0,
                      (Sqrt(1 + alpha2 * tan2Theta) - 1) / 2);
    }
    LaneNF32 PDF(const Vec3lfn &w, const Vec3lfn &wm) const { return D(w, wm); }
    // NOTE: samples the visible normals instead of simply the distribution function
    Vec3lfn Sample_wm(const Vec3lfn &w, const Vec2lfn &u) const
    {
        Vec3lfn wh = Normalize(Vec3lfn(alphaX * w.x, alphaY * w.y, w.z));
        wh         = Select(wh.z < 0, -wh, wh);
        // NOTE: this process involves the projection of a disk of uniformly distributed points
        // onto a truncated ellipsoid. The inverse of the scale factor of the ellipsoid is
        // applied to the incoming direction in order to simplify to the isotropic case.
        Vec2lfn p = SampleUniformDiskPolar(u);
        Vec3lfn T1 =
            Select(wh.z < 0.99999f, Normalize(Cross(Vec3lfn(0, 0, 1), wh)), Vec3lfn(1, 0, 0));
        Vec3lfn T2 = Cross(wh, T1);
        // NOTE: For a given x, the y component has a value in range [-h, h], where h = sqrt(1
        // - Sqr(x)). When the projection is not perpendicular to the ellipsoid, the range
        // shrinks to [-hcos(theta), h]. This requires an affine transformation with scale
        // 0.5(1 + cosTheta) and translation 0.5h(1 - cosTheta).

        LaneNF32 h = Sqrt(1 - p.x * p.x);
        p.y        = Lerp((1.f + wh.z) / 2.f, h, p.y);

        // Project point to hemisphere, transform to ellipsoid.
        LaneNF32 pz = Sqrt(Max(0.f, 1 - p.x * p.x - p.y * p.y));
        Vec3lfn nh  = p.x * T1 + p.y * T2 + pz * wh;
        return Normalize(Vec3lfn(alphaX * nh.x, alphaY * nh.y, Max(1e-6f, nh.z)));
    }
    MaskF32 EffectivelySmooth() const { return Max(alphaX, alphaY) < 1e-3f; }
    static LaneNF32 RoughnessToAlpha(const LaneNF32 &roughness) { return Sqrt(roughness); }
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

inline b32 IsReflective(BxDFFlags f) { return EnumHasAnyFlags(f, BxDFFlags::Reflection); }
inline b32 IsTransmissive(BxDFFlags f) { return EnumHasAnyFlags(f, BxDFFlags::Transmission); }
inline b32 IsDiffuse(BxDFFlags f) { return EnumHasAnyFlags(f, BxDFFlags::Diffuse); }
inline b32 IsGlossy(BxDFFlags f) { return EnumHasAnyFlags(f, BxDFFlags::Glossy); }
inline b32 IsSpecular(BxDFFlags f) { return EnumHasAnyFlags(f, BxDFFlags::Specular); }
inline b32 IsNonSpecular(BxDFFlags f)
{
    return EnumHasAnyFlags(f, BxDFFlags::Diffuse | BxDFFlags::Glossy);
}
inline b32 IsValid(BxDFFlags f) { return !EnumHasAnyFlags(f, BxDFFlags::Invalid); }

inline MaskF32 IsReflective(const LaneNU32 &f)
{
    return (f & LaneNU32((BxDFFlags::Reflection))) != 0;
}
inline MaskF32 IsTransmissive(const LaneNU32 &f)
{
    return (f & LaneNU32((BxDFFlags::Transmission))) != 0;
}
inline MaskF32 IsDiffuse(const LaneNU32 &f)
{
    return (f & LaneNU32((BxDFFlags::Diffuse))) != 0;
}
inline MaskF32 IsGlossy(const LaneNU32 &f) { return (f & LaneNU32((BxDFFlags::Glossy))) != 0; }
inline MaskF32 IsSpecular(const LaneNU32 &f)
{
    return (f & LaneNU32((BxDFFlags::Specular))) != 0;
}
inline MaskF32 IsNonSpecular(const LaneNU32 &f)
{
    return (f & LaneNU32((BxDFFlags::Diffuse | BxDFFlags::Glossy))) != 0;
}
inline MaskF32 IsValid(const LaneNU32 &f) { return (f & LaneNU32(BxDFFlags::Invalid)) != 0; }

struct BSDFSample
{
    SampledSpectrumN f;
    Vec3lfn wi;
    LaneNF32 pdf = LaneNF32(0);
    LaneNU32 flags;
    LaneNF32 eta;

    BSDFSample() = default;

    BSDFSample(const SampledSpectrumN &f, const Vec3lfn &wi, const LaneNF32 &pdf,
               const LaneNU32 &flags, const LaneNF32 &eta = 1.f)
        : f(f), wi(wi), pdf(pdf), flags(flags), eta(eta)
    {
    }

    MaskF32 IsReflective() { return rt::IsReflective(flags); }
    MaskF32 IsTransmissive() { return rt::IsTransmissive(flags); }
    MaskF32 IsDiffuse() { return rt::IsDiffuse(flags); }
    MaskF32 IsGlossy() { return rt::IsGlossy(flags); }
    MaskF32 IsSpecular() { return rt::IsSpecular(flags); }
    MaskF32 IsValid() { return !rt::IsValid(flags); }
};

// NOTE: BTDFs are not generally symmetric
enum class TransportMode
{
    Importance = 0,
    Radiance   = 1,
};

struct BxDF
{
    virtual SampledSpectrumBase<LaneNF32> EvaluateSample(const Vec3lfn &wo, const Vec3lfn &wi,
                                                         LaneNF32 &pdf,
                                                         TransportMode mode) const = 0;
    virtual BSDFSample GenerateSample(const Vec3lfn &wo, const LaneNF32 &uc, const Vec2lfn &u,
                                      TransportMode mode, BxDFFlags inFlags) const = 0;
    virtual f32 PDF(const Vec3f &wo, const Vec3f &wi, TransportMode mode,
                    BxDFFlags sampleFlags = BxDFFlags::RT) const                   = 0;
    virtual LaneNU32 Flags() const                                                 = 0;
};

// Lambertian model, light is scattered in all directions equally
struct DiffuseBxDF : BxDF
{
    SampledSpectrumN R;
    DiffuseBxDF() = default;
    DiffuseBxDF(const SampledSpectrumN &R) : R(R) {}

    virtual SampledSpectrumN EvaluateSample(const Vec3lfn &wo, const Vec3lfn &wi,
                                            LaneNF32 &pdf, TransportMode mode) const override
    {
        MaskF32 mask = SameHemisphere(wo, wi);
        pdf          = Select(mask, CosineHemispherePDF(AbsCosTheta(wi)), 0.f);
        return Select(mask, R * InvPi, SampledSpectrumN(0.f));
    }

    virtual BSDFSample GenerateSample(const Vec3lfn &wo, const LaneNF32 &uc, const Vec2lfn &u,
                                      TransportMode mode    = TransportMode::Radiance,
                                      BxDFFlags sampleFlags = BxDFFlags::RT) const override
    {
        if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection)) return {};
        Vec3lfn wi   = SampleCosineHemisphere(u);
        wi.z         = Select(wo.z < 0, -wi.z, wi.z);
        LaneNF32 pdf = CosineHemispherePDF(AbsCosTheta(wi));
        return BSDFSample(R * InvPi, wi, pdf, LaneNU32(u32(BxDFFlags::DiffuseReflection)));
    }
    virtual f32 PDF(const Vec3f &wo, const Vec3f &wi, TransportMode mode,
                    BxDFFlags sampleFlags = BxDFFlags::RT) const override
    {
        if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection) || !SameHemisphere(wo, wi))
            return 0.f;
        return CosineHemispherePDF(AbsCosTheta(wi));
    }

    virtual LaneNU32 Flags() const override
    {
        return Select(MaskF32(R), LaneNU32(u32(BxDFFlags::DiffuseReflection)),
                      LaneNU32(u32(BxDFFlags::Unset)));
    }
};

struct DiffuseTransmissionBxDF : BxDF
{
    SampledSpectrumN R, T;
    DiffuseTransmissionBxDF() = default;
    DiffuseTransmissionBxDF(const SampledSpectrumN &R, const SampledSpectrumN &T) : R(R), T(T)
    {
    }

    virtual SampledSpectrumN EvaluateSample(const Vec3lfn &wo, const Vec3lfn &wi,
                                            LaneNF32 &pdf, TransportMode mode) const override
    {
        MaskF32 reflectMask = SameHemisphere(wo, wi);
        LaneNF32 pr         = R.MaxComponentValue();
        LaneNF32 pt         = T.MaxComponentValue();
        pdf = CosineHemispherePDF(AbsCosTheta(wi)) * Select(reflectMask, pr, pt) / (pr + pt);
        return Select(reflectMask, R, T) * InvPi;
    }

    virtual BSDFSample GenerateSample(const Vec3lfn &wo, const LaneNF32 &uc, const Vec2lfn &u,
                                      TransportMode mode    = TransportMode::Radiance,
                                      BxDFFlags sampleFlags = BxDFFlags::RT) const override
    {
        if (!EnumHasAnyFlags(sampleFlags, BxDFFlags::Reflection)) return {};
        LaneNF32 pr = R.MaxComponentValue();
        LaneNF32 pt = T.MaxComponentValue();

        LaneNF32 sum = pr + pt;

        MaskF32 reflectMask = uc < pr / sum;
        Vec3lfn wi          = SampleCosineHemisphere(u);
        wi.z                = Select(reflectMask ^ (wo.z > 0), -wi.z, wi.z);
        LaneNF32 pdf        = CosineHemispherePDF(AbsCosTheta(wi));
        return BSDFSample(Select(reflectMask, R, T) * InvPi, wi, pdf,
                          Select(reflectMask, LaneNU32(u32(BxDFFlags::DiffuseReflection)),
                                 LaneNU32(u32(BxDFFlags::DiffuseTransmission))));
    }
    virtual f32 PDF(const Vec3f &wo, const Vec3f &wi, TransportMode mode,
                    BxDFFlags sampleFlags = BxDFFlags::RT) const override
    {
        Assert(0);
        return 0.f;
    }
    virtual LaneNU32 Flags() const override
    {
        return Select(MaskF32(R), LaneNU32(u32(BxDFFlags::DiffuseReflection)),
                      LaneNU32(u32(BxDFFlags::Unset))) |
               Select(MaskF32(T), LaneNU32(u32(BxDFFlags::DiffuseTransmission)),
                      LaneNU32(u32(BxDFFlags::Unset)));
    }
};

struct ConductorBxDF : BxDF
{
    ConductorBxDF() = delete;
    ConductorBxDF(const TrowbridgeReitzDistribution &mfDistrib, const SampledSpectrumN &eta,
                  const SampledSpectrumN &k)
        : mfDistrib(mfDistrib), eta(eta), k(k)
    {
    }
    TrowbridgeReitzDistribution mfDistrib;
    SampledSpectrumN eta, k;

    virtual SampledSpectrumN EvaluateSample(const Vec3lfn &wo, const Vec3lfn &wi,
                                            LaneNF32 &pdf, TransportMode mode) const override
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
        SampledSpectrumN f =
            mfDistrib.D(wm) * Fr * mfDistrib.G(wo, wi) / (4 * cosTheta_i * cosTheta_o);
        wm = FaceForward(Normalize(wm), Vec3lfn(0, 0, 1));

        pdf = Select(mask, mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)), 0.f);
        return f;
    }
    virtual BSDFSample GenerateSample(const Vec3lfn &wo, const LaneNF32 &uc, const Vec2lfn &u,
                                      TransportMode mode, BxDFFlags sampleFlags) const override
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
            pdf = Select(specularMask, pdf, mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)));
            LaneNF32 cosTheta_o = AbsCosTheta(wo);
            LaneNF32 cosTheta_i = AbsCosTheta(wi);
            // NOTE: Fresnel term with respect to the microfacet normal, wm
            SampledSpectrumN Fr = FrComplex(AbsDot(wo, wm), eta, k);
            // Torrance Sparrow BRDF Model (D * G * F / 4cos cos)
            f = Select(specularMask, f,
                       mfDistrib.D(wm) * Fr * mfDistrib.G(wo, wi) /
                           (4 * cosTheta_i * cosTheta_o));
        }
        LaneNU32 flags = Select(specularMask, LaneNU32(u32(BxDFFlags::SpecularReflection)),
                                LaneNU32(u32(BxDFFlags::GlossyReflection)));
        return BSDFSample(f, wi, pdf, flags);
    }
    virtual f32 PDF(const Vec3f &wo, const Vec3f &wi, TransportMode mode,
                    BxDFFlags sampleFlags) const override
    {
        if (!(sampleFlags & BxDFFlags::Reflection)) return 0;
        if (!SameHemisphere(wo, wi)) return 0;
        if (mfDistrib.EffectivelySmooth()) return 0;
        // Evaluate sampling PDF of rough conductor BRDF
        Vec3f wm = wo + wi;
        if (LengthSquared(wm) == 0) return 0;
        wm = FaceForward(Normalize(wm), Vec3f(0, 0, 1));
        return mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm));
    }

    virtual LaneNU32 Flags() const override
    {
        return Select(mfDistrib.EffectivelySmooth(),
                      LaneNU32(u32(BxDFFlags::SpecularReflection)),
                      LaneNU32(u32(BxDFFlags::GlossyReflection)));
    }
};

struct DielectricBxDF : BxDF
{
    TrowbridgeReitzDistribution mfDistrib;
    // NOTE: spectrally varying IORs are handled by randomly sampling a single wavelength
    LaneNF32 eta;

    DielectricBxDF() {}
    DielectricBxDF(const LaneNF32 &eta, const TrowbridgeReitzDistribution &mfDistrib)
        : eta(eta), mfDistrib(mfDistrib)
    {
    }

    virtual BSDFSample GenerateSample(const Vec3f &wo, const f32 &uc, const Vec2f &u,
                                      TransportMode mode    = TransportMode::Radiance,
                                      BxDFFlags sampleFlags = BxDFFlags::RT) const override
    {
        if (eta == 1 || mfDistrib.EffectivelySmooth())
        {
            // Sample perfect specular dielectric BSDF
            f32 R = FrDielectric(CosTheta(wo), eta), T = 1 - R;
            // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
            f32 pr = R, pt = T;
            if (!(sampleFlags & BxDFFlags::Reflection)) pr = 0;
            if (!(sampleFlags & BxDFFlags::Transmission)) pt = 0;
            if (pr == 0 && pt == 0) return {};

            if (uc < pr / (pr + pt))
            {
                // Sample perfect specular dielectric BRDF
                Vec3f wi(-wo.x, -wo.y, wo.z);
                SampledSpectrum fr(R / AbsCosTheta(wi));
                return BSDFSample(fr, wi, pr / (pr + pt), (u32)BxDFFlags::SpecularReflection);
            }
            else
            {
                // Sample perfect specular dielectric BTDF
                // Compute ray direction for specular transmission
                Vec3f wi;
                f32 etap;
                bool valid = Refract(wo, Vec3f(0, 0, 1), eta, &etap, &wi);
                if (!valid) return {};

                SampledSpectrum ft(T / AbsCosTheta(wi));
                // Account for non-symmetry with transmission to different medium
                if (mode == TransportMode::Radiance) ft /= Sqr(etap);

                return BSDFSample(ft, wi, pt / (pr + pt), (u32)BxDFFlags::SpecularTransmission,
                                  etap);
            }
        }
        else
        {
            // Sample rough dielectric BSDF
            Vec3f wm = mfDistrib.Sample_wm(wo, u);
            f32 R    = FrDielectric(Dot(wo, wm), eta);
            f32 T    = 1 - R;
            // Compute probabilities _pr_ and _pt_ for sampling reflection and transmission
            f32 pr = R, pt = T;
            if (!(sampleFlags & BxDFFlags::Reflection)) pr = 0;
            if (!(sampleFlags & BxDFFlags::Transmission)) pt = 0;
            if (pr == 0 && pt == 0) return {};

            f32 pdf;
            if (uc < pr / (pr + pt))
            {
                // Sample reflection at rough dielectric interface
                Vec3f wi = Reflect(wo, wm);
                if (!SameHemisphere(wo, wi)) return {};
                // Compute PDF of rough dielectric reflection
                pdf = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm)) * pr / (pr + pt);

                Assert(!IsNaN(pdf));
                SampledSpectrum f(mfDistrib.D(wm) * mfDistrib.G(wo, wi) * R /
                                  (4 * CosTheta(wi) * CosTheta(wo)));
                return BSDFSample(f, wi, pdf, (u32)BxDFFlags::GlossyReflection);
            }
            else
            {
                // Sample transmission at rough dielectric interface
                f32 etap;
                Vec3f wi;
                bool tir = !Refract(wo, (Vec3f)wm, eta, &etap, &wi);
                if (SameHemisphere(wo, wi) || wi.z == 0 || tir) return {};
                // Compute PDF of rough dielectric transmission
                f32 denom   = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
                f32 dwm_dwi = AbsDot(wi, wm) / denom;
                pdf         = mfDistrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);

                Assert(!IsNaN(pdf));
                // Evaluate BRDF and return _BSDFSample_ for rough transmission
                SampledSpectrum ft(
                    T * mfDistrib.D(wm) * mfDistrib.G(wo, wi) *
                    Abs(Dot(wi, wm) * Dot(wo, wm) / (CosTheta(wi) * CosTheta(wo) * denom)));
                // Account for non-symmetry with transmission to different medium
                if (mode == TransportMode::Radiance) ft /= Sqr(etap);

                return BSDFSample(ft, wi, pdf, (u32)BxDFFlags::GlossyTransmission, etap);
            }
        }
    }

    virtual SampledSpectrum EvaluateSample(const Vec3f &wo, const Vec3f &wi, f32 &pdf,
                                           TransportMode mode) const override
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
    virtual LaneNU32 Flags() const override
    {
        LaneNU32 flags = Select(eta == 1.f, u32(BxDFFlags::Transmission), u32(BxDFFlags::RT));
        return flags | Select(mfDistrib.EffectivelySmooth(), u32(BxDFFlags::Specular),
                              u32(BxDFFlags::Glossy));
    }
    virtual f32 PDF(const Vec3f &wo, const Vec3f &wi, TransportMode mode,
                    BxDFFlags sampleFlags = BxDFFlags::RT) const override
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

inline f32 SampleExponential(f32 u, f32 a) { return -std::log(1 - u) / a; }

template <typename TopBxDF, typename BottomBxDF>
class TopOrBottomBxDF
{
public:
    // TopOrBottomBxDF Public Methods
    TopOrBottomBxDF() = default;
    TopOrBottomBxDF &operator=(const TopBxDF *t)
    {
        top    = t;
        bottom = nullptr;
        return *this;
    }
    TopOrBottomBxDF &operator=(const BottomBxDF *b)
    {
        bottom = b;
        top    = nullptr;
        return *this;
    }

    SampledSpectrum EvaluateSample(Vec3f wo, Vec3f wi, TransportMode mode) const
    {
        f32 pdf;
        return top ? top->EvaluateSample(wo, wi, pdf, mode)
                   : bottom->EvaluateSample(wo, wi, pdf, mode);
    }

    BSDFSample GenerateSample(Vec3f wo, f32 uc, Vec2f u, TransportMode mode,
                              BxDFFlags sampleFlags = BxDFFlags::RT) const
    {
        return top ? top->GenerateSample(wo, uc, u, mode, sampleFlags)
                   : bottom->GenerateSample(wo, uc, u, mode, sampleFlags);
    }

    f32 PDF(Vec3f wo, Vec3f wi, TransportMode mode,
            BxDFFlags sampleFlags = BxDFFlags::RT) const
    {
        return top ? top->PDF(wo, wi, mode, sampleFlags)
                   : bottom->PDF(wo, wi, mode, sampleFlags);
    }

    BxDFFlags Flags() const
    {
        return top ? BxDFFlags(top->Flags()) : BxDFFlags(bottom->Flags());
    }

private:
    const TopBxDF *top       = nullptr;
    const BottomBxDF *bottom = nullptr;
};

template <typename TopBxDF, typename BottomBxDF>
struct CoatedBxDF : BxDF
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
    virtual BSDFSample GenerateSample(const Vec3lfn &wOut, const LaneNF32 &uc,
                                      const Vec2lfn &u,
                                      TransportMode mode    = TransportMode::Radiance,
                                      BxDFFlags sampleFlags = BxDFFlags::RT) const override
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

    virtual SampledSpectrum EvaluateSample(const Vec3f &wOut, const Vec3f &wIn, f32 &outPdf,
                                           TransportMode mode) const override
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

    virtual f32 PDF(const Vec3f &wOut, const Vec3f &wIn, TransportMode mode,
                    BxDFFlags sampleFlags = BxDFFlags::RT) const override
    {
        // Set _wo_ and _wi_ for layered BSDF evaluation
        Vec3f wo = wOut;
        Vec3f wi = wIn;
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

    virtual LaneNU32 Flags() const override
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

typedef CoatedBxDF<DielectricBxDF, DiffuseBxDF> CoatedDiffuseBxDF;
typedef CoatedBxDF<DielectricBxDF, ConductorBxDF> CoatedConductorBxDF;

#define SQRT_2     1.41421356237309504880f /* sqrt(2) */
#define INV_SQRT_2 0.7071067811865475244f  /* 1/sqrt(2) */
// https://drive.google.com/file/d/0BzvWIdpUpRx_cFVlUkFhWXdleEU/view?resourcekey=0-DUQ5GMyc-cvSNBKFCA3LlQ

// https://jcgt.org/published/0003/02/03/paper.pdf
inline f32 GTR2Aniso(const Vec3f &wm, f32 alphaX, f32 alphaY)
{
    return 1.f /
           (PI * alphaX * alphaY * (Sqr(wm.x / alphaX) + Sqr(wm.y / alphaY) + Sqr(wm.z)));
}

// Smith GGX masking function
inline f32 G1_GGX(const Vec3f &w, f32 alphaX, f32 alphaY)
{
    LaneNF32 tanTheta = TanTheta(w);
    LaneNF32 alpha    = (Sqr(CosPhi(w) * alphaX) + Sqr(SinPhi(w) * alphaY)) * tanTheta;
    f32 lambda        = Select(tanTheta == LaneNF32(pos_inf), 0,
                               (Copysign(1.f, alpha) * Sqrt(1 + alpha * tanTheta) - 1) / 2);
    return 1 / (1 + lambda);
}

inline f32 GGXD_wi(Vec3f wo, Vec3f wm, f32 alphaX, f32 alphaY)
{
    f32 NdotV = CosTheta(wo);
    // NOTE: G1(wo) / AbsCosTheta(wo)
    // TODO: this is probably wrong
    // f32 sign = Copysign(1, NdotV);
    // f32 term =
    //     2.f / (Abs(NdotV) + Sqrt(NdotV * NdotV + Sqr(wo.x * alphaX) + Sqr(wo.y * alphaY)));
    // return GTR2Aniso(wm, alphaX, alphaY) * Max(0.f, Dot(wo, wm)) * term;
    return GTR2Aniso(wm, alphaX, alphaY) * G1_GGX(wo, alphaX, alphaY) * Max(0.f, Dot(wo, wm)) /
           AbsCosTheta(wo);
}

inline f32 SchlickWeight(f32 cosValue)
{
    f32 c = Clamp(1.f - cosValue, 0.f, 1.f);
    return (c * c) * (c * c) * c;
}

inline f32 FrSchlick(f32 f0, f32 cosH, f32 etaP, f32 invEtaP)
{
    f32 cosTheta2_t = (1 - (1 - Sqr(cosH)) * Sqr(invEtaP));
    return Select(etaP > 1.f, Lerp(f0, SchlickWeight(cosH), 1.f),
                  Lerp(f0, SchlickWeight(SafeSqrt(cosTheta2_t)), 1.f));
}

inline f32 GTR1(f32 cosH, f32 a)
{
    f32 a2 = a * a;
    return (a2 - 1) / (PI * Log(a2) * (1 + (a2 - 1) * cosH * cosH));
}

inline Vec3f SampleGTR1(f32 a, const Vec2f &u)
{
    f32 a2     = a * a;
    f32 cosH2  = (1 - Pow(a2, 1 - u[0])) / (1 - a2);
    f32 sinH   = Clamp(Sqrt(1 - cosH2), 0.f, 1.f);
    f32 cosH   = Sqrt(cosH2);
    f32 phi    = 2 * PI * u[1];
    f32 sinPhi = Sin(phi);
    f32 cosPhi = Cos(phi);

    return Vec3f(cosPhi * sinH, sinPhi * sinH, cosH);
}

inline f32 SmithG(const Vec3f &v, const Vec3f &wm, f32 a)
{
    f32 a2        = a * a;
    f32 cosTheta  = CosTheta(wm);
    f32 cos2Theta = cosTheta * cosTheta;
    f32 tan2Theta = (1 - cos2Theta) / cos2Theta;

    return Select(Dot(v, wm) <= 0.f, 0.f, 2.f * Rcp(1.f + Sqrt(1 + a2 * tan2Theta)));
}

inline Vec3lfn SampleGGXVNDF(const Vec3lfn &w, const Vec2lfn &u, f32 alphaX, f32 alphaY)
{
    Vec3lfn wh = Normalize(Vec3lfn(alphaX * w.x, alphaY * w.y, w.z));
    wh         = Select(wh.z < 0, -wh, wh);
    Vec2lfn p  = SampleUniformDiskPolar(u);
    Vec3lfn T1 =
        Select(wh.z < 0.99999f, Normalize(Cross(Vec3lfn(0, 0, 1), wh)), Vec3lfn(1, 0, 0));
    Vec3lfn T2 = Cross(wh, T1);
    LaneNF32 h = Sqrt(1 - p.x * p.x);
    p.y        = Lerp((1.f + wh.z) / 2.f, h, p.y);

    // Project point to hemisphere, transform to ellipsoid.
    LaneNF32 pz = Sqrt(Max(0.f, 1 - p.x * p.x - p.y * p.y));
    Vec3lfn nh  = p.x * T1 + p.y * T2 + pz * wh;
    return Normalize(Vec3lfn(alphaX * nh.x, alphaY * nh.y, Max(1e-6f, nh.z)));
}

inline f32 SmithGAniso(const Vec3f &v, const Vec3f &wm, f32 alphaX, f32 alphaY)
{
    f32 tan2Theta = (Sqr(v.x * alphaX) + Sqr(v.y * alphaY)) / Sqr(v.z);
    return Select(Dot(v, wm) <= 0.f, 0.f, 2 * Rcp(1 + Sqrt(1.f + tan2Theta)));
}

struct DisneyMaterial
{
    f32 metallic;
    f32 roughness;
    f32 anisotropic;
    f32 specularTint;
    f32 sheen;
    f32 sheenTint;
    f32 clearcoat;
    f32 clearcoatGloss;
    f32 specTrans;
    f32 ior;
    // Solid
    Vec3f scatterDistance;
    // Thin
    f32 flatness;
    f32 diffTrans;
};

inline Vec3f Luminance(const Vec3f &color) { return Dot(Vec3f(.3f, .6f, 1.f), color); }

#if 0
struct DisneySolidBxDF
{
    Vec3f baseColor;
    f32 metallic;
    f32 specTrans;
    f32 roughness;
    f32 sheen;
    f32 sheenTint;
    f32 specularTint;
    f32 clearcoatGloss;
    f32 flatness;
    f32 anisotropy;
    f32 eta;
    bool isThin;

    void CalculateAnisotropicRoughness(f32 &alphaX, f32 &alphaY)
    {
        f32 aspect = Sqrt(1 - .9f * anisotropy);
        alphaX     = Sqr(roughness) / aspect;
        alphaY     = Sqr(roughness) / aspect;
    }

    Vec3f EvalDiffuse(f32 NdotL, f32 NdotV, f32 LdotH)
    {
        f32 Fo        = SchlickWeight(NdotL);
        f32 Fi        = SchlickWeight(NdotV);
        Vec3f diffuse = (1.f - 0.5f * Fo) * (1.f - 0.5f * Fi);
        f32 rr        = 2 * roughness * Sqr(LdotH);
        Vec3f retro   = rr * (Fo + Fi + Fo * Fi * (rr - 1));

        return (diffuse + retro) * baseColor * InvPi;
    }
    Vec3f EvalSheen(const Vec3f &c, f32 LdotH)
    {
        return sheen * Lerp(sheenTint, Vec3f(1.f), c) * SchlickWeight(LdotH);
    }

    f32 EvalDisneyClearcoat(const Vec3f &wo, const Vec3f &wi, const Vec3f &wm, f32 glossFactor,
                            f32 etaP, f32 invEtaP, f32 &pdf)
    {
        f32 NdotH = AbsCosTheta(wm);
        f32 VdotH = Dot(wo, wm);

        f32 D = GTR1(NdotH, glossFactor);
        f32 G = SmithG(wo, 0.25f) * SmithG(wi, 0.25f);
        // f32 fh    = FrDielectric(dotVH,  wm), 1.5f);
        f32 F = FrSchlick(.04f, VdotH, etaP, invEtaP);

        f32 jacobian = 1 / (4 * Abs(VdotH));
        pdf          = D * NdotH * jacobian;
        // reversePdf = D * dotNH / (4 * AbsDot(wi, wm));
        return clearcoat * D * G * F * 0.25f;
    }

    BSDFSample SampleDisneyClearcoat(const Vec3f &wo, const Vec2f &u, f32 clearcoat,
                                     f32 clearcoatGloss, f32 etaP, f32 invEtaP)
    {
        Vec3f wm = SampleGTR1(gloss, u);
        wm       = FaceForward(wm, wo);
        Vec3f wi = Reflect(wo, wm);
        if (!SameHemisphere(wo, wi)) return {};

        f32 pdf;
        SampledSpectrumN f = EvalDisneyClearcoat(wo, wi, wm, clearcoat, clearcoatGloss, pdf);
        return BSDFSample(f, wi, pdf, (u32)BxDFFlags::DiffuseReflection);
    }

    bool EvalDisneySpecTrans(const Vec3f &wo, f32 invEtaP, Vec3f &result, Vec3f &wi, f32 &pdf)
    {
        f32 denom = Sqr(wi + wo * invEtaP);

        f32 D        = GTR2Aniso(wm, alphaX, alphaY);
        f32 G1       = SmithGAniso(wo, wm, alphaX, alphaY);
        f32 G        = G1 * SmithGAniso(wi, wm, alphaX, alphaY);
        f32 absLdotH = AbsDot(wi, wm);
        f32 VdotH    = Dot(wo, wm);

        pdf    = D * G1 * Max(0.f, VdotH) * absLdotH / (denom * AbsCosTheta(wo));
        result = Sqrt(baseColor) D * G * (1 - F) * Abs(VdotH) * absLdotH * /
                 (denom * Abs(NdotV) * AbsCosTheta(wi));
        return true;
    }

    BSDFSample GenerateSample(const Vec3f &wo, const f32 &uc, const Vec2f &u,
                              TransportMode mode    = TransportMode::Radiance,
                              BxDFFlags sampleFlags = BxDFFlags::RT) const
    {
        f32 NdotV = CosTheta(wo);
        f32 Fr    = FrDielectric(NdotV, eta);

        f32 pDiffuse, pSpecRfl, pSpecTrans;
        f32 pClearcoat = .25f * clearcoat;
        f32 cdf[4];
        cdf[0] = pDiffuse;
        cdf[1] = cdf[0] + pSpecRfl;
        cdf[2] = cdf[1] + pSpecTrans;
        cdf[3] = cdf[2] + pClearcoat;

        f32 pdf;
        // Diffuse
        if (uc < weights[0])
        {
            // Remap random variable
            // uc = (uc - weights[0]) / (weights[1] - weights[0]);
            Vec3f wi = SampleCosineHemisphere(u);
            wi.z     = Select(wo.z < 0, -wi.z, wi.z);
            Vec3f wm = wo + wi;
            if (LengthSquared(wm) == 0.f) return {};
            wm = Normalize(wm);

            f32 NdotV = CosTheta(wi);
            f32 LdotH = Dot(wi, wm);

            pdf = CosineHemispherePDF(AbsCosTheta(wi));
            pdf *= pDiffuse;

            Vec3f result = EvalDiffuse(NdotL, NdotV, LdotH);

            // Sheen
            Vec3f lum = Luminance(baseColor);
            Vec3f c   = Select(lum > 0.f, baseColor / lum, 1.f);
            result += EvalSheen(c, LdotH);

            return BSDFSample(result, wi, pdf, LaneNU32(u32(BxDFFlags::DiffuseReflection)));
        }
        // Dielectric
        else if (uc < weights[1])
        {
            // Remap random variable
            // uc = (uc - weights[1]) / (weights[2] - weights[1]);
            f32 alphaX, alphaY;
            CalculateAnisotropicRoughness(alphaX, alphaY);
            Vec3f wm, wi;
            u32 flags;
            if (alphaX > 0.f && alphaY > 0.f)
            {
                wm    = SampleGGXVNDF(wo, u, alphaX, alphaY);
                wi    = Reflect(wo, wm);
                flags = (u32)(BxDFFlags::GlossyReflection);
            }
            else
            {
                wi    = Vec3f(-wo.x, -wo.y, wo.z);
                flags = (u32)(BxDFFlags::SpecularReflection);
            }
            pdf *= pSpecRfl;
            return BSDFSample(EvalDisneySpecRfl(), wi, pdf, flags);
        }
        // Specular BSDF
        else if (uc < weights[2])
        {
            uc = (uc - weights[2]) / (weights[3] - weights[2]);
            Vec3f result, wi;
            Vec3f wm = SampleGGXVNDF(wo, u, alphaX, alphaY);
            f32 etaP;
            if (!Refract(wo, wm, eta, &etaP, &wi) || wi.z == 0 || SameHemisphere(wo, wi))
                return false;
            f32 invEtaP = 1.f / etaP;

            f32 Fr = FrDielectric(CosTheta(wo), etaP);

            // Importance sample fresnel
            if (uc < Fr)
            {
            }
            else
            {
            }

            if (EvalDisneySpecTrans(wo, result, wi pdf))
            {
                pdf *= pSpecTrans;
                return BSDFSample(result, wi, pdf, flags);
            }
            return {};
        }
        // Clearcoat
        else
        {
            Vec3f wm = SampleGTR1(.25f, u);
            Vec3f wi = Reflect(wo, wm);
            f32 pdf;
            f32 glossFactor = Lerp(clearcoatGloss, .1f, .001f);
            EvalDisneyClearcoat(wo, wi, wm, glossFactor, etaP, invEtaP, pdf);
            pdf *= pClearcoat;
        }
    }

    Vec3f EvaluateSample(const Vec3f &wo, const Vec3f &wi, f32 &pdf, TransportMode mode) const
    {
        Vec3f result(0.f);
        // NOTE: V is wo, L is wi
        // etaP = ni/nt
        // invEtaP = nt/ni
        f32 bsdf     = (1 - metallic) * specTrans;
        f32 brdf     = (1 - metallic) * (1 - specTrans);
        f32 NdotV    = CosTheta(wo);
        f32 NdotL    = CosTheta(wi);
        bool reflect = NdotV * NdotL > 0.f;
        bool refract = NdotV * NdotL < 0.f;

        f32 invEta  = Rcp(eta);
        f32 etaP    = Select(NdotV > 0.0f, eta, invEta);
        f32 invEtaP = Select(NdotV > 0.0f, invEta, eta);

        Vec3f wm = wo + wi * etaP;
        if (LengthSquared(wm) == 0.f) return {};

        wm        = Normalize(wm);
        f32 R     = FrDielectric(Dot(wo, wm), eta);
        f32 LdotH = Dot(wi, wm);

        // Anisotropy
        f32 alphaX, alphaY;
        CalculateAnisotropicRoughness(alphaX, alphaY);

        f32 D  = GTR2Aniso(wm, alphaX, alphaY);
        f32 G1 = SmithGAniso(wo, alphaX, alphaY);
        f32 G  = G1 * SmithGAniso(wi, alphaX, alphaY);
        // Disney Diffuse
        {
            result += EvalDisneyDiffuse(NdotL, NdotV, LdotH);

            // TODO: I'm pretty sure this is an entirely separate model
            Assert(!isThin && flatness == 0.f);
#if 0
            if (isThin && flatness > 0.f)
            {
                // Fake SSS
                f32 Fss90 = rr / 2.f;
                f32 Fss   = Lerp(Fo, 1.f, Fss90) * Lerp(Fi, 1.f, Fss90);
                f32 fss   = 1.25f * (Fss * (1.f / (NdotV + NdotL) - .5f) + .5f);
                result += brdf * baseColor * InvPi * Lerp(diffuse + retro, fss, flatness);
            }
#endif
            // TODO: path traced subsurface scattering
            result += brdf * baseColor * InvPi * (diffuse + retro);
        }

        // Disney Sheen

        f32 lum    = Luminance(baseColor);
        f32 cTint  = Select(lum > 0.f, baseColor / lum, 1.f);
        f32 cSheen = Lerp(sheenTint, 1.f, cTint);
        f32 Fd     = SchlickWeight(LdotH);
        result += sheen * (1.f - metallic) * Fd * cSheen;

        // Disney specular reflection
        if (reflect)
        {
            // f32 F = brdf * R;
            // NOTE: if intersecting from the back, disable metallic lobe
            f32 F = brdf * R;
            if (NdotV >= 0.f)
            {
                if (metallic > 0.f) F += metallic * FrSchlick(baseColor, NdotV, etaP, invEtaP);
                if (specularTint > 0.f)
                {
                    f32 F0 = Sqr((1 - etaP) / (1 + etaP)) * Lerp(specularTint, 1.f, cTint);
                    F0     = Lerp(metallic, F0 *, baseColor);
                    F += (1 - metallic) * FrSchlick(F0, NdotV, etaP, invEtaP);
                }
            }

            result += D * G * F / (4 * Abs(NdotL) * Abs(NdotV));
        }
        // Disney specular transmission
        if (refract && specTrans > 0.0f)
        {
            f32 T = 1 - R;

            f32 denom = Sqr(LdotH + VdotH * invEtaP);

            pdf = D * G1 * Abs(LdotH) * AbsDot(VdotH) / (NdotV * denom);

            f32 VdotH = Dot(wo, wm);
            f32 specTransComp =
                D * G * T * Abs(VdotH) * Abs(LdotH) / (Abs(NdotV) * Abs(NdotL) * denom);
            result += Sqrt(baseColor) * bsdf * specTransComp *
                      Select(mode == TransportMode::Radiance, Sqr(invEtaP), 1.f);
        }

        // Disney Clearcoat
        {
            f32 clearcoatPdf;
            f32 gloss      = Lerp(clearcoatGloss, .1f, .001f);
            f32 clearcoatC = EvalDisneyClearcoat(wo, wi, wm, clearcoat, gloss, clearcoatPdf);
            result += Vec3f(clearcoatC);
        }

        return result;
    }
};
#endif

} // namespace rt

#endif
