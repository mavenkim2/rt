//////////////////////////////
// Primitive Methods
//

namespace rt
{
PrimitiveMethods primitiveMethods[] = {
    {BVHHit},
    {BVH4Hit},
    {CompressedBVH4Hit},
};

//////////////////////////////
// Creation from scene description
//

Sampler Sampler::Create(Arena *arena, const ScenePacket *packet, const Vec2i fullResolution)
{
    i32 samplesPerPixel        = 16;
    RandomizeStrategy strategy = RandomizeStrategy::FastOwen;
    i32 seed                   = 0;

    b32 isHalton     = (packet->type == "halton"_sid);
    b32 isStratified = (packet->type == "stratified"_sid);

    // stratified sampler only
    bool jitter  = true;
    i32 xSamples = 4;
    i32 ySamples = 4;

    for (u32 i = 0; i < packet->parameterCount; i++)
    {
        switch (packet->parameterNames[i])
        {
            case "pixelsamples"_sid:
            {
                samplesPerPixel = packet->GetInt(i);
            }
            break;
            case "randomization"_sid:
            {
                if (Compare(packet->bytes[i], "none"))
                {
                    strategy = RandomizeStrategy::FastOwen;
                }
                else if (Compare(packet->bytes[i], "permutedigits"))
                {
                    strategy = RandomizeStrategy::PermuteDigits;
                }
                else if (Compare(packet->bytes[i], "owen"))
                {
                    Assert(!isHalton);
                    strategy = RandomizeStrategy::Owen;
                }
            }
            break;
            case "seed"_sid:
            {
                seed = packet->GetInt(i);
            }
            break;
            case "jitter"_sid:
            {
                Assert(isStratified);
                jitter = packet->GetBool(i);
            }
            break;
            case "xsamples"_sid:
            {
                Assert(isStratified);
                xSamples = packet->GetInt(i);
            }
            break;
            case "ysamples"_sid:
            {
                Assert(isStratified);
                ySamples = packet->GetInt(i);
            }
            break;
            default:
            {
                Error(0, "Invalid option encountered during Sampler creation\n");
            }
        }
    }
    switch (packet->type)
    {
        case "independent"_sid: return PushStructConstruct(arena, IndependentSampler)(samplesPerPixel);
        case "paddedsobol"_sid: return PushStructConstruct(arena, PaddedSobolSampler)(samplesPerPixel, strategy, seed);
        case "sobol"_sid: return PushStructConstruct(arena, SobolSampler)(samplesPerPixel, fullResolution, strategy, seed);
        case "stratified"_sid: return PushStructConstruct(arena, StratifiedSampler)(xSamples, ySamples, jitter, seed);
        case "halton"_sid: Error(0, "Halton sampler not implemented.");
        default: return PushStructConstruct(arena, ZSobolSampler)(samplesPerPixel, fullResolution, strategy);
    }
}

//////////////////////////////
// Spectrum Methods
//

f32 Spectrum::operator()(f32 lambda) const
{
    void *ptr  = GetPtr();
    f32 result = spectrumMethods[GetTag()].Evaluate(ptr, lambda);
    return result;
}

f32 Spectrum::MaxValue() const
{
    void *ptr  = GetPtr();
    f32 result = spectrumMethods[GetTag()].MaxValue(ptr);
    return result;
}

SampledSpectrum Spectrum::Sample(const SampledWavelengths &lambda) const
{
    void *ptr              = GetPtr();
    SampledSpectrum result = spectrumMethods[GetTag()].Sample(ptr, lambda);
    return result;
}

// template <class T>
// f32 SpectrumCRTP<T>::Evaluate(void *ptr, f32 lambda)
// {
//     return static_cast<T *>(ptr)->Evaluate(lambda);
// }

template <class T>
SampledSpectrum SpectrumCRTP<T>::Sample(void *ptr, const SampledWavelengths &lambda)
{
    return static_cast<T *>(ptr)->Sample(lambda);
}

SampledSpectrumN BxDF::EvaluateSample(const Vec3lfn &wo, const Vec3lfn &wi, LaneNF32 &pdf, TransportMode mode) const
{
    void *ptr = GetPtr();
    return bxdfMethods[GetTag()].EvaluateSample(ptr, wo, wi, pdf, mode);
}

BSDFSample BxDF::GenerateSample(const Vec3lfn &wo, const LaneNF32 &uc, const Vec2lfn &u, TransportMode mode, BxDFFlags inFlags) const
{
    void *ptr = GetPtr();
    return bxdfMethods[GetTag()].GenerateSample(ptr, wo, uc, u, mode, inFlags);
}

LaneNU32 BxDF::Flags() const
{
    void *ptr = GetPtr();
    return bxdfMethods[GetTag()].Flags(ptr);
}

template <class T>
SampledSpectrum BxDFCRTP<T>::EvaluateSample(void *ptr, const Vec3lfn &wo, const Vec3lfn &wi, LaneNF32 &pdf, TransportMode mode)
{
    return static_cast<T *>(ptr)->EvaluateSample(wo, wi, pdf, mode);
}
template <class T>
BSDFSample BxDFCRTP<T>::GenerateSample(void *ptr, const Vec3lfn &wo, const LaneNF32 &uc, const Vec2lfn &u, TransportMode mode, BxDFFlags flags)
{
    return static_cast<T *>(ptr)->GenerateSample(wo, uc, u, mode, flags);
}
// template <class T>
// f32 BxDFCRTP<T>::PDF(void *ptr, Vec3f wo, Vec3f wi, TransportMode mode, BxDFFlags flags)
// {
//     return static_cast<T *>(ptr)->PDF(wo, wi, mode, flags);
// }
template <class T>
LaneNU32 BxDFCRTP<T>::Flags(void *ptr)
{
    return static_cast<T *>(ptr)->Flags();
}
} // namespace rt
