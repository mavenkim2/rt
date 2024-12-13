#ifndef BSDF_H
#define BSDF_H

#include "bxdf.h"

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
    // BSDFBase(BxDF &&bxdf, const Vec3lfn &ns, const Vec3lfn &dpdus)
    //     : bxdf(std::move(bxdf)), frame(LinearSpace3fn::FromXZ(Normalize(dpdus), ns)) {}

    SampledSpectrumN EvaluateSample(Vec3lfn wo, Vec3lfn wi, LaneNF32 &pdf,
                                    TransportMode mode = TransportMode::Radiance) const;
    BSDFSample GenerateSample(Vec3lfn wo, const LaneNF32 &uc, const Vec2lfn &u,
                              TransportMode mode = TransportMode::Radiance,
                              BxDFFlags inFlags  = BxDFFlags::RT) const;
    // LaneNF32 PDF(Vec3lfn wo, Vec3lfn wi, TransportMode mode, BxDFFlags inFlags) const;
    // Hemispherical directional function
    SampledSpectrumN rho(Vec3lfn wo, LaneNF32 *uc, Vec2lfn *u, u32 numSamples) const;
    // Hemispherical hemispherical function
    SampledSpectrumN rho(Vec2lfn *u1, LaneNF32 *uc, Vec2lfn *u2, u32 numSamples) const;
    LaneNU32 Flags() const;
};

typedef BSDFBase<BxDF> BSDF;

} // namespace rt

#endif
