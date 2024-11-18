namespace rt
{

f32 RadicalInverse(i32 baseIndex, u64 a)
{
    i32 base           = primes[baseIndex];
    f32 invBase        = 1.f / base;
    f32 invMult        = 1.f;
    u64 reversedDigits = 0;
    while (a)
    {
        u64 next       = a / baseIndex;
        u64 remainder  = a - next * baseIndex;
        reversedDigits = reversedDigits * baseIndex + remainder;
        invMult *= invBase;
        a = next;
    }
    f32 radicalInv = reversedDigits * invMult;
    return Min(radicalInv, oneMinusEpsilon);
}

u64 InverseRadicalInverse(u64 inverse, i32 base, i32 nDigits)
{
    u64 index = 0;
    for (i32 i = 0; i < nDigits; i++)
    {
        u64 next  = inverse / base;
        u64 digit = inverse - next * base;
        index     = index * base + digit;
        inverse   = next;
    }
    return index;
}

//////////////////////////////
// Scramblers
//

u32 NoRandomizer(u32 v, u32 seed = 0)
{
    return v;
}

u32 BinaryPermuteScrambler(u32 v, u32 permutation)
{
    return permutation ^ v;
}

// Each digit is scrambled based on the current digit + higher bits
u32 OwenScrambler(u32 v, u32 seed)
{
    if (seed & 1)
        v ^= 1u << 31;
    for (i32 b = 1; b < 32; b++)
    {
        u32 mask = (~0u) << (32 - b);
        if ((u32)MixBits((v & mask) ^ seed) & (1u << b))
            v ^= 1u << (31 - b);
    }
    return v;
}

u32 FastOwenScrambler(u32 v, u32 seed)
{
    v = ReverseBits32(v);
    v ^= v * 0x3d20adea;
    v += seed;
    v *= (seed >> 16) | 1;
    v ^= v * 0x05526c56;
    v ^= v * 0x53a22864;
    return ReverseBits32(v);
}

//////////////////////////////
// Sobol
//

inline f32 SobolSample(i64 a, i32 dimension, u32 (*randomizer)(u32, u32), u32 seed = 0)
{
    u32 v = 0;
    for (i32 i = dimension * sobolMatrixSize; a != 0; a >>= 1, i++)
    {
        if (a & 1)
            v ^= sobolMatrices32[i];
    }

    v = randomizer(v, seed);
    return Min(v * 0x1p-32f, oneMinusEpsilon);
}

inline u64 SobolIntervalToIndex(u32 log2Scale, u64 sampleIndex, Vec2i p)
{
    if (log2Scale == 0)
        return sampleIndex;

    u32 m2    = log2Scale << 1;
    u64 index = sampleIndex << m2;

    u64 delta = 0;
    for (i32 c = 0; sampleIndex; sampleIndex >>= 1, c++)
    {
        if (sampleIndex & 1)
            delta ^= vdCSobolMatrices[log2Scale - 1][c];
    }

    u64 b = (((u64)(p.x) << log2Scale) | (p.y)) ^ delta;

    for (i32 c = 0; b; b >>= 1, c++)
    {
        if (b & 1)
            index ^= vdCSobolMatricesInv[log2Scale - 1][c];
    }
    return index;
}
} // namespace rt

// Given pixel coordinate p, you're trying to find the index in the halton sequence that gives a value such that
// the value scaled by the max screen resolution (rounded up to the next pow of 2) gives the pixel coordinate you're
// looking for. To do this, note that multiplying a radical inverse number 0.d1(a)d2(a)d3(a)... by b^2 gives
// d1(a)d2(a).d3(a)... Thus, the inverse radical inverse of the two last digits of x gives the value in the halton
// sequence that corresponds with pixel coordinate p.
