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

enum class RandomizeStrategy
{
    NoRandomize,
    PermuteDigits,
    Owen,
    FastOwen,
};

struct HaltonSampler : SamplerCRTP<HaltonSampler>
{
};

struct SobolSampler : SamplerCRTP<SobolSampler>
{
    SobolSampler(i32 samplesPerPixel, vec2i fullResolution, RandomizeStrategy randomize, i32 seed = 0)
        : samplesPerPixel(samplesPerPixel), seed(seed), randomize(randomize)
    {
        assert(IsPow2(samplesPerPixel));
        scale = Max(NextPowerOfTwo(fullResolution.x), NextPowerOfTwo(fullResolution.y));
    }
    i32 SamplesPerPixel() const
    {
        return samplesPerPixel;
    }
    void StartPixelSample(vec2i p, i32 sampleIndex, i32 d)
    {
        pixel      = p;
        dimension  = std::max<i32>(2, d);
        sobolIndex = SobolIntervalToIndex(Log2Int(scale), sampleIndex, pixel);
    }
    f32 Get1D()
    {
        if (dimension >= nSobolDimensions)
            dimension = 2;
        return SampleDimension(dimension++);
    }
    vec2 Get2D()
    {
        if (dimension + 1 >= nSobolDimensions)
            dimension = 2;
        vec2 u(SampleDimension(dimension), SampleDimension(dimension + 1));
        dimension += 2;
        return u;
    }
    vec2 GetPixel2D()
    {
        vec2 u(SobolSample(sobolIndex, 0, NoRandomizer), SobolSample(sobolIndex, 1, NoRandomizer));

        for (i32 dim = 0; dim < 2; dim++)
        {
            u[dim] = Clamp(u[dim] * scale - pixel[dim], 0.f, oneMinusEpsilon);
        }
        return u;
    }

    f32 SampleDimension(i32 d)
    {
        if (randomize == RandomizeStrategy::NoRandomize)
            return SobolSample(sobolIndex, d, NoRandomizer);
        u32 hash = (u32)Hash(d, seed);
        if (randomize == RandomizeStrategy::PermuteDigits)
            return SobolSample(sobolIndex, d, BinaryPermuteScrambler, hash);
        if (randomize == RandomizeStrategy::FastOwen)
            return SobolSample(sobolIndex, d, FastOwenScrambler, hash);

        // Default is owen scrambling
        return SobolSample(sobolIndex, d, OwenScrambler, hash);
    }

    i32 samplesPerPixel, scale, seed;
    RandomizeStrategy randomize;
    vec2i pixel;
    i32 dimension;
    i64 sobolIndex;
};

struct PaddedSobolSampler : SamplerCRTP<PaddedSobolSampler>
{
    PaddedSobolSampler(i32 samplesPerPixel, RandomizeStrategy randomize, i32 seed = 0)
        : samplesPerPixel(samplesPerPixel), seed(seed), randomize(randomize)
    {
        assert(IsPow2(samplesPerPixel));
    }
    i32 SamplesPerPixel() const
    {
        return samplesPerPixel;
    }
    void StartPixelSample(vec2i p, i32 index, i32 dim)
    {
        pixel       = p;
        sampleIndex = index;
        dimension   = dim;
    }
    f32 Get1D()
    {
        u64 hash  = Hash(pixel, dimension, seed);
        i32 index = PermutationElement(sampleIndex, samplesPerPixel, (u32)hash);
        int dim   = dimension++;
        return SampleDimension(0, index, hash >> 32);
    }
    vec2 Get2D()
    {
        u64 hash  = Hash(pixel, dimension, seed);
        i32 index = PermutationElement(sampleIndex, samplesPerPixel, (u32)hash);
        int dim   = dimension;
        dimension += 2;
        return vec2(SampleDimension(0, index, (u32)hash), SampleDimension(1, index, hash >> 32));
    }
    vec2 GetPixel2D()
    {
        return Get2D();
    }

    f32 SampleDimension(i32 dim, u32 a, u32 hash) const
    {
        if (randomize == RandomizeStrategy::NoRandomize)
            return SobolSample(a, dim, NoRandomizer);
        if (randomize == RandomizeStrategy::PermuteDigits)
            return SobolSample(a, dim, BinaryPermuteScrambler, hash);
        if (randomize == RandomizeStrategy::FastOwen)
            return SobolSample(a, dim, FastOwenScrambler, hash);

        // Default is owen scrambling
        return SobolSample(a, dim, OwenScrambler, hash);
    }

    i32 samplesPerPixel, seed;
    RandomizeStrategy randomize;
    vec2i pixel;
    i32 dimension, sampleIndex;
};

struct ZSobolSampler : SamplerCRTP<ZSobolSampler>
{
    ZSobolSampler(i32 samplesPerPixel, vec2i fullResolution, RandomizeStrategy randomize, i32 seed = 0)
        : randomize(randomize), seed(seed)
    {
        assert(IsPow2(samplesPerPixel));
        log2SamplesPerPixel     = Log2Int(samplesPerPixel);
        i32 res                 = NextPowerOfTwo(Max(fullResolution.x, fullResolution.y));
        i32 log4SamplesPerPixel = (log2SamplesPerPixel + 1) / 2;
        nBase4Digits            = log4SamplesPerPixel + Log2Int(res);
    }
    i32 SamplesPerPixel() const
    {
        return 1 << log2SamplesPerPixel;
    }
    void StartPixelSample(vec2i p, i32 index, i32 dim)
    {
        dimension   = dim;
        mortonIndex = (EncodeMorton2(p.x, p.y) << log2SamplesPerPixel) | index;
    }
    f32 Get1D()
    {
        u64 sampleIndex = GetSampleIndex();
        dimension++;
        if (randomize == RandomizeStrategy::NoRandomize)
            return SobolSample(sampleIndex, 0, NoRandomizer);

        u32 hash = (u32)Hash(dimension, seed);
        if (randomize == RandomizeStrategy::PermuteDigits)
            return SobolSample(sampleIndex, 0, BinaryPermuteScrambler, hash);
        if (randomize == RandomizeStrategy::FastOwen)
            return SobolSample(sampleIndex, 0, FastOwenScrambler, hash);

        // Default is owen scrambling
        return SobolSample(sampleIndex, 0, OwenScrambler, hash);
    }
    vec2 Get2D()
    {
        u64 sampleIndex = GetSampleIndex();
        dimension += 2;
        if (randomize == RandomizeStrategy::NoRandomize)
            return vec2(SobolSample(sampleIndex, 0, NoRandomizer), SobolSample(sampleIndex, 1, NoRandomizer));

        u64 hash          = Hash(dimension, seed);
        u32 sampleHash[2] = {u32(hash), u32(hash >> 32)};
        if (randomize == RandomizeStrategy::PermuteDigits)
        {
            return vec2(SobolSample(sampleIndex, 0, BinaryPermuteScrambler, sampleHash[0]),
                        SobolSample(sampleIndex, 1, BinaryPermuteScrambler, sampleHash[1]));
        }
        if (randomize == RandomizeStrategy::FastOwen)
        {
            return vec2(SobolSample(sampleIndex, 0, FastOwenScrambler, sampleHash[0]),
                        SobolSample(sampleIndex, 1, FastOwenScrambler, sampleHash[1]));
        }

        // Default is owen scrambling
        return vec2(SobolSample(sampleIndex, 0, OwenScrambler, sampleHash[0]),
                    SobolSample(sampleIndex, 1, OwenScrambler, sampleHash[1]));
    }
    vec2 GetPixel2D()
    {
        return Get2D();
    }
    u64 GetSampleIndex() const
    {
        static const u8 permutations[24][4] = {
            {0, 1, 2, 3},
            {0, 1, 3, 2},
            {0, 2, 1, 3},
            {0, 2, 3, 1},
            {0, 3, 2, 1},
            {0, 3, 1, 2},
            {1, 0, 2, 3},
            {1, 0, 3, 2},
            {1, 2, 0, 3},
            {1, 2, 3, 0},
            {1, 3, 2, 0},
            {1, 3, 0, 2},
            {2, 1, 0, 3},
            {2, 1, 3, 0},
            {2, 0, 1, 3},
            {2, 0, 3, 1},
            {2, 3, 0, 1},
            {2, 3, 1, 0},
            {3, 1, 2, 0},
            {3, 1, 0, 2},
            {3, 2, 1, 0},
            {3, 2, 0, 1},
            {3, 0, 2, 1},
            {3, 0, 1, 2},
        };
        u64 sampleIndex = 0;

        u32 isNotPow4 = log2SamplesPerPixel & 1;
        i32 lastDigit = isNotPow4 ? 1 : 0;
        for (i32 i = nBase4Digits - 1; i >= lastDigit; i--)
        {
            // Get the base 4 digit
            i32 digitShift = 2 * i - (isNotPow4 ? 1 : 0);
            i32 digit      = (mortonIndex >> digitShift) & 3;

            // Find the permutation of the digit based on the higher bits + the dimension
            u64 higherDigits = mortonIndex >> (digitShift + 2);
            i32 p            = (MixBits(higherDigits ^ (0x55555555u * dimension)) >> 24) % 24;
            digit            = permutations[p][digit];
            sampleIndex |= u64(digit) << digitShift;
        }
        if (isNotPow4)
        {
            i32 digit = mortonIndex & 1;
            sampleIndex |= digit ^ (MixBits((mortonIndex >> 1) ^ (0x55555555u * dimension)) & 1);
        }

        return sampleIndex;
    }

    i32 nBase4Digits,
        log2SamplesPerPixel, seed;
    i32 dimension;
    u64 mortonIndex;
    RandomizeStrategy randomize;
};
