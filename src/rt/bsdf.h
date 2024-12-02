#ifndef BSDF_H
#define BSDF_H

namespace rt
{

template <typename BxDF>
struct BSDFBase
{
    BSDFBase() = default;
    BxDF bxdf;
    LinearSpace<LaneNF32> frame;

    BSDFBase(const Vec3lfn &ns, const Vec3lfn &dpdus) : frame(LinearSpace::FromXZ(Normalize(dpdus), ns)) {}

    SampledSpectrumN EvaluateSample(Vec3lfn wo, Vec3lfn wi, LaneNF32 &pdf, TransportMode mode) const;
    BSDFSample GenerateSample(Vec3lfn wo, const LaneNF32 &uc, const Vec2lfn &u, TransportMode mode, BxDFFlags inFlags) const;
    // LaneNF32 PDF(Vec3lfn wo, Vec3lfn wi, TransportMode mode, BxDFFlags inFlags) const;
    // Hemispherical directional function
    SampledSpectrumN rho(Vec3lfn wo, LaneNF32 *uc, Vec2lfn *u, u32 numSamples) const;
    // Hemispherical hemispherical function
    SampledSpectrumN rho(Vec2lfn *u1, LaneNF32 *uc, Vec2lfn *u2, u32 numSamples) const;
};

typename BSDFBase<BxDF> BSDF;

} // namespace rt

#endif
