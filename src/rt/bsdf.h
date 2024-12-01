#ifndef BSDF_H
#define BSDF_H

namespace rt
{

template <typename BxDF>
struct BSDFBase
{
    BSDFBase() = default;
    BxDF bxdf;
    LinearSpace frame;

    BSDFBase(const Vec3NF32 &ns, const Vec3NF32 &dpdus) : frame(LinearSpace::FromXZ(Normalize(dpdus), ns)) {}

    SampledSpectrumN EvaluateSample(Vec3NF32 wo, Vec3NF32 wi, LaneNF32 &pdf, TransportMode mode) const;
    BSDFSample GenerateSample(Vec3NF32 wo, const LaneNF32 &uc, const Vec2NF32 &u, TransportMode mode, BxDFFlags inFlags) const;
    LaneNF32 PDF(Vec3NF32 wo, Vec3NF32 wi, TransportMode mode, BxDFFlags inFlags) const;
    // Hemispherical directional function
    SampledSpectrumN rho(Vec3NF32 wo, LaneNF32 *uc, Vec2NF32 *u, u32 numSamples) const;
    // Hemispherical hemispherical function
    SampledSpectrumN rho(Vec2NF32 *u1, LaneNF32 *uc, Vec2NF32 *u2, u32 numSamples) const;
};

typename BSDFBase<BxDF> BSDF;

} // namespace rt

#endif
