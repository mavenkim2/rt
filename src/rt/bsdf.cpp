#include "bsdf.h"
namespace rt
{
//////////////////////////////
// BSDF Methods
//
SampledSpectrumN BSDFBase::EvaluateSample(Vec3NF32 wo, Vec3NF32 wi, LaneNF32 &pdf, TransportMode mode) const
{
    wi = frame.ToLocal(wi);
    wo = frame.ToLocal(wo);
    if (All(wo.z == 0)) return {};
    return bxdf.EvaluateSample(wo, wi, pdf, mode);
    // void *ptr              = GetPtr();
    // u32 tag                = GetTag();
    // SampledSpectrumN result = bsdfMethods[tag].EvaluateSample(ptr, wo, wi, pdf, mode);
    // return result;
}

BSDFSample BSDFBase::GenerateSample(Vec3NF32 wo, LaneNF32 uc, Vec2NF32 u, TransportMode mode, BSDFFlags sampleFlags) const
{
    wo              = frame.ToLocal(wo);
    BSDFFlags flags = Flags();
    if (All(wo.z == 0) || !EnumHasAnyFlags(bxdf.Flags(), sampleFlags)) return {};
    // void *ptr         = GetPtr();
    // u32 tag           = GetTag();
    // BSDFSample result = bsdfMethods[tag].GenerateSample(ptr, wo, uc, u, mode, sampleFlags);
    BSDFSample result = bxdf.GenerateSample(wo, uc, u, mode, sampleFlags);
    if (!result.IsValid() || All(result.f == 0) || All(result.pdf == 0) || All(result.wi.z == 0)) return {};
    result.wi = frame.FromLocal(result.wi);
    return result;
}

LaneNF32 BSDFBase::PDF(Vec3NF32 wo, Vec3NF32 wi, TransportMode mode, BSDFFlags sampleFlags) const
{
    wo = frame.ToLocal(wo);
    wi = frame.ToLocal(wi);
    if (All(wo.z == 0) || !EnumHasAnyFlags(bxdf.Flags(), sampleFlags)) return {};
    // void *ptr       = GetPtr();
    // u32 tag         = GetTag();
    // LaneNF32 result = bsdfMethods[tag].PDF(ptr, wo, wi, mode, sampleFlags);
    LaneNF32 result = bxdf.PDF(wo, wi, mode, sampleFlags);
    return result;
}

SampledSpectrumN BSDFBase::rho(Vec3NF32 wo, LaneNF32 *uc, Vec2NF32 *u, u32 numSamples) const
{
    SampledSpectrumN r(0.f);
    for (u32 i = 0; i < numSamples; i++)
    {
        BSDFSample sample = GenerateSample(wo, uc[i], u[i], TransportMode::Radiance, BSDFFlags::All);
        if (sample.IsValid())
        {
            r += sample.f * AbsCosTheta(sample.wi) / sample.pdf;
        }
    }
    return r / (f32)numSamples;
}

SampledSpectrumN BSDFBase::rho(Vec2NF32 *u1, LaneNF32 *uc, Vec2NF32 *u2, u32 numSamples) const
{
    SampledSpectrumN r(0.f);
    for (u32 i = 0; i < numSamples; i++)
    {
        Vec3NF32 wo = SampleUniformHemisphere(u1[i]);
        if (All(wo.z == 0)) continue;
        LaneNF32 pdfo = UniformHemispherePDF();
        BSDFSample bs = GenerateSample(wo, uc[i], u2[i], TransportMode::Radiance, BSDFFlags::All);
        r += bs.f + AbsCosTheta(bs.wi) * AbsCosTheta(wo) / (pdfo * bs.pdf);
    }
    return r / (PI * numSamples);
}

} // namespace rt
