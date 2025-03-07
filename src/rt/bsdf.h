#ifndef BSDF_H
#define BSDF_H

#include "bxdf.h"
#include "math/math_include.h"

namespace rt
{

template <typename BxDF>
struct BSDFBase
{
    BSDFBase() = default;
    BxDF bxdf;
    LinearSpace<LaneNF32> frame;

    BSDFBase(BxDF bxdf, const Vec3lfn &dpdus, const Vec3lfn &ns)
        : bxdf(bxdf), frame(LinearSpace3fn::FromXZ(Normalize(dpdus), ns))
    {
    }
    SampledSpectrumN EvaluateSample(Vec3lfn wo, Vec3lfn wi, LaneNF32 &pdf,
                                    TransportMode mode = TransportMode::Radiance) const
    {
        wi = frame.ToLocal(wi);
        wo = frame.ToLocal(wo);
        if (All(wo.z == 0)) return {};
        return bxdf->EvaluateSample(wo, wi, pdf, mode);
        // void *ptr              = GetPtr();
        // u32 tag                = GetTag();
        // SampledSpectrumN result = bsdfMethods[tag].EvaluateSample(ptr, wo, wi, pdf, mode);
        // return result;
    }

    BSDFSample GenerateSample(Vec3lfn wo, const LaneNF32 &uc, const Vec2lfn &u,
                              TransportMode mode = TransportMode::Radiance,
                              BxDFFlags inFlags  = BxDFFlags::RT) const
    {
        wo             = frame.ToLocal(wo);
        LaneNU32 flags = Flags();
        if (All(wo.z == 0) || All((bxdf->Flags() & LaneNU32(u32(inFlags))) == 0)) return {};
        // void *ptr         = GetPtr();
        // u32 tag           = GetTag();
        // BSDFSample result = bsdfMethods[tag].GenerateSample(ptr, wo, uc, u, mode,
        // sampleFlags);
        BSDFSample result = bxdf->GenerateSample(wo, uc, u, mode, inFlags);
        if (!result.IsValid() || All(result.f == SampledSpectrumN(0.f)) ||
            All(result.pdf == 0) || All(result.wi.z == 0))
            return {};
        result.wi = frame.FromLocal(result.wi);
        return result;
    }

    LaneNF32 PDF(Vec3lfn wo, Vec3lfn wi, TransportMode mode, BxDFFlags sampleFlags) const
    {
        wo = frame.ToLocal(wo);
        wi = frame.ToLocal(wi);
        if (All(wo.z == 0) || !EnumHasAnyFlags(bxdf->Flags(), sampleFlags)) return {};
        // void *ptr       = GetPtr();
        // u32 tag         = GetTag();
        // LaneNF32 result = bsdfMethods[tag].PDF(ptr, wo, wi, mode, sampleFlags);
        LaneNF32 result = bxdf->PDF(wo, wi, mode, sampleFlags);
        return result;
    }

    SampledSpectrumN rho(Vec3lfn wo, LaneNF32 *uc, Vec2lfn *u, u32 numSamples) const
    {
        SampledSpectrumN r(0.f);
        for (u32 i = 0; i < numSamples; i++)
        {
            BSDFSample sample =
                GenerateSample(wo, uc[i], u[i], TransportMode::Radiance, BSDFFlags::All);
            if (sample.IsValid())
            {
                r += sample.f * AbsCosTheta(sample.wi) / sample.pdf;
            }
        }
        return r / (f32)numSamples;
    }

    SampledSpectrumN rho(Vec2lfn *u1, LaneNF32 *uc, Vec2lfn *u2, u32 numSamples) const
    {
        SampledSpectrumN r(0.f);
        for (u32 i = 0; i < numSamples; i++)
        {
            Vec3lfn wo = SampleUniformHemisphere(u1[i]);
            if (All(wo.z == 0)) continue;
            LaneNF32 pdfo = UniformHemispherePDF();
            BSDFSample bs =
                GenerateSample(wo, uc[i], u2[i], TransportMode::Radiance, BSDFFlags::All);
            r += bs.f + AbsCosTheta(bs.wi) * AbsCosTheta(wo) / (pdfo * bs.pdf);
        }
        return r / (PI * numSamples);
    }

    LaneNU32 Flags() const { return bxdf->Flags(); }
};

typedef BSDFBase<BxDF *> BSDF;

} // namespace rt

#endif
