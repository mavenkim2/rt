//////////////////////////////
// Primitive Methods
//

PrimitiveMethods primitiveMethods[] = {
    {BVHHit},
    {BVH4Hit},
    {CompressedBVH4Hit},
};

//////////////////////////////
// Creation from scene description
//

Sampler Sampler::Create(Arena *arena, const ScenePacket *packet, const vec2i fullResolution)
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

template <class T>
f32 SpectrumCRTP<T>::Evaluate(void *ptr, f32 lambda)
{
    return static_cast<T *>(ptr)->Evaluate(lambda);
}

template <class T>
f32 SpectrumCRTP<T>::MaxValue(void *ptr)
{
    return static_cast<T *>(ptr)->MaxValue();
}

template <class T>
SampledSpectrum SpectrumCRTP<T>::Sample(void *ptr, const SampledWavelengths &lambda)
{
    return static_cast<T *>(ptr)->Sample(lambda);
}

//////////////////////////////
// BSDF Methods
//

SampledSpectrum BSDF::f(vec3 wo, vec3 wi, TransportMode mode) const
{
    wi = frame.ToLocal(wi);
    wo = frame.ToLocal(wo);
    if (wo.z == 0) return {};
    void *ptr              = GetPtr();
    u32 tag                = GetTag();
    SampledSpectrum result = bsdfMethods[tag].f(ptr, wo, wi, mode);
    return result;
}

BSDFSample BSDF::Sample_f(vec3 wo, f32 uc, vec2 u, TransportMode mode = TransportMode::Radiance, BSDFFlags sampleFlags = BSDFFlags::RT) const
{
    wo              = frame.ToLocal(wo);
    BSDFFlags flags = Flags();
    if (wo.z == 0 || !EnumHasAnyFlags(Flags(), sampleFlags)) return {};
    void *ptr         = GetPtr();
    u32 tag           = GetTag();
    BSDFSample result = bsdfMethods[tag].Sample_f(ptr, wo, uc, u, mode, sampleFlags);
    if (!result.IsValid() || result.f == 0 || result.pdf == 0 || result.wi.z == 0) return {};
    result.wi = frame.FromLocal(result.wi);
    return result;
}

f32 BSDF::PDF(vec3 wo, vec3 wi, TransportMode mode, BSDFFlags sampleFlags) const
{
    wo = frame.ToLocal(wo);
    wi = frame.ToLocal(wi);
    if (wo.z == 0 || !EnumHasAnyFlags(Flags(), sampleFlags)) return {};
    void *ptr  = GetPtr();
    u32 tag    = GetTag();
    f32 result = bsdfMethods[tag].PDF(ptr, wo, wi, mode, sampleFlags);
    return result;
}

SampledSpectrum BSDF::rho(vec3 wo, f32 *uc, vec2 *u, u32 numSamples) const
{
    SampledSpectrum r(0.f);
    for (u32 i = 0; i < numSamples; i++)
    {
        BSDFSample sample = Sample_f(wo, uc[i], u[i]);
        if (sample.IsValid())
        {
            r += sample.f * AbsCosTheta(sample.wi) / sample.pdf;
        }
    }
    return r / (f32)numSamples;
}

SampledSpectrum BSDF::rho(vec2 *u1, f32 *uc, vec2 *u2, u32 numSamples) const
{
    SampledSpectrum r(0.f);
    for (u32 i = 0; i < numSamples; i++)
    {
        vec3 wo = SampleUniformHemisphere(u1[i]);
        if (wo.z == 0) continue;
        f32 pdfo      = UniformHemispherePDF();
        BSDFSample bs = Sample_f(wo, uc[i], u2[i]);
        r += bs.f + AbsCosTheta(bs.wi) * AbsCosTheta(wo) / (pdfo * bs.pdf);
    }
    return r / (PI * numSamples);
}

BSDFFlags BSDF::Flags() const
{
    void *ptr        = GetPtr();
    u32 tag          = GetTag();
    BSDFFlags result = bsdfMethods[tag].Flags(ptr);
    return result;
}

template <class T>
SampledSpectrum BSDFCRTP<T>::f(void *ptr, vec3 wo, vec3 wi, TransportMode mode)
{
    return static_cast<T *>(ptr)->f(wo, wi, mode);
}
template <class T>
BSDFSample BSDFCRTP<T>::Sample_f(void *ptr, vec3 wo, f32 uc, vec2 u, TransportMode mode, BSDFFlags flags)
{
    return static_cast<T *>(ptr)->Sample_f(wo, uc, u, mode, flags);
}
template <class T>
f32 BSDFCRTP<T>::PDF(void *ptr, vec3 wo, vec3 wi, TransportMode mode, BSDFFlags flags)
{
    return static_cast<T *>(ptr)->PDF(wo, wi, mode, flags);
}
template <class T>
BSDFFlags BSDFCRTP<T>::Flags(void *ptr)
{
    return static_cast<T *>(ptr)->Flags();
}
