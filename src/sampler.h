// NOTE: independent uniform samplers without any care for discrepancy
struct IndependentSampler : SamplerCRTP<IndependentSampler>
{
    IndependentSampler(i32 samplesPerPixel, i32 seed = 0) : samplesPerPixel(samplesPerPixel), seed(seed) {}

    i32 SamplesPerPixel() const
    {
        return samplesPerPixel;
    }
    void StartPixelSample(vec2i p, i32 sampleIndex, i32 dimension)
    {
        rng.SetSequence(Hash(p, seed));
        rng.Advance(sampleIndex * 65536ull + dimension);
    }
    f32 Get1D() { return rng.Uniform<f32>(); }
    vec2 Get2D() { return {rng.Uniform<f32>(), rng.Uniform<f32>()}; }
    vec2 GetPixel2D() { return Get2D(); }

    i32 samplesPerPixel, seed;
    RNG rng;
};

struct StratifiedSampler : SamplerCRTP<StratifiedSampler>
{
    StratifiedSampler(i32 xPixelSamples, i32 yPixelSamples, bool jitter, i32 seed = 0)
        : xSamples(xPixelSamples), ySamples(yPixelSamples), jitter(jitter), seed(seed) {}

    i32 SamplesPerPixel() const
    {
        return xSamples * ySamples;
    }
    void StartPixelSample(vec2i p, i32 index, i32 dimension)
    {
        pixel       = p;
        sampleIndex = index;
        dimensions  = dimension;
        rng.SetSequence(Hash(p, seed));
        rng.Advance(sampleIndex * 65536ull + dimension);
    }
    f32 Get1D()
    {
        u64 hash    = Hash(pixel, dimensions, seed);
        i32 stratum = PermutationElement(sampleIndex, SamplesPerPixel(), (u32)hash);
        dimensions++;
        f32 delta = jitter ? rng.Uniform<f32>() : 0.5f;
        return (stratum + delta) / SamplesPerPixel();
    }
    vec2 Get2D()
    {
        u64 hash    = Hash(pixel, dimensions, seed);
        i32 stratum = PermutationElement(sampleIndex, SamplesPerPixel(), (u32)hash);
        dimensions += 2;

        i32 x      = stratum % xSamples;
        i32 y      = stratum / xSamples;
        f32 deltaX = jitter ? rng.Uniform<f32>() : 0.5f;
        f32 deltaY = jitter ? rng.Uniform<f32>() : 0.5f;

        return vec2((x + deltaX) / xSamples, (y + deltaY) / ySamples);
    }
    vec2 GetPixel2D()
    {
        return Get2D();
    }

    i32 xSamples, ySamples, seed;
    b8 jitter;
    RNG rng;
    vec2i pixel;
    i32 dimensions, sampleIndex = 0;
};
