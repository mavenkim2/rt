#include "low_discrepancy.h"

namespace rt
{

static constexpr f32 OneMinusEpsilon = 0x1.fffffep-1;

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
    return Min(radicalInv, OneMinusEpsilon);
}
f32 SobolSample(i64 a, i32 dimension, u32 (*randomizer)(u32, u32), u32 seed)
{
    u32 v = 0;

    i64 shift = 0;

    while (a)
    {
        u32 index = Bsf64(a);
        a &= a - 1;
        v ^= (sobolMatrices32[dimension * sobolMatrixSize + index]);
    }

    v = randomizer(v, seed);
    return Min(v * 0x1p-32f, OneMinusEpsilon);
}

u64 SobolIntervalToIndex(u32 log2Scale, u64 sampleIndex, Vec2i p)
{
    if (log2Scale == 0) return sampleIndex;

    u32 m2    = log2Scale << 1;
    u64 index = sampleIndex << m2;

    u64 delta = 0;
    for (i32 c = 0; sampleIndex; sampleIndex >>= 1, c++)
    {
        if (sampleIndex & 1) delta ^= vdCSobolMatrices[log2Scale - 1][c];
    }

    u64 b = (((u64)(p.x) << log2Scale) | (p.y)) ^ delta;

    for (i32 c = 0; b; b >>= 1, c++)
    {
        if (b & 1) index ^= vdCSobolMatricesInv[log2Scale - 1][c];
    }
    return index;
}
} // namespace rt
