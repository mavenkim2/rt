#ifndef LOW_DISCREPANCY_H_
#define LOW_DISCREPANCY_H_

#include "base.h"
#include "math/basemath.h"
#include "math/vec2.h"
#include "math/math.h"
#include "hash.h"
#include "tables/primes.hpp"
#include "tables/sobolmatrices.hpp"
#include <emmintrin.h>
#include <immintrin.h>

namespace rt
{

f32 RadicalInverse(i32 baseIndex, u64 a);

inline u64 InverseRadicalInverse(u64 inverse, i32 base, i32 nDigits)
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

inline u32 NoRandomizer(u32 v, u32 seed = 0) { return v; }

inline u32 BinaryPermuteScrambler(u32 v, u32 permutation) { return permutation ^ v; }

// Each digit is scrambled based on the current digit + higher bits
inline u32 OwenScrambler(u32 v, u32 seed)
{
    if (seed & 1) v ^= 1u << 31;
    for (i32 b = 1; b < 32; b++)
    {
        u32 mask = (~0u) << (32 - b);
        if ((u32)MixBits((v & mask) ^ seed) & (1u << b)) v ^= 1u << (31 - b);
    }
    return v;
}

inline u32 FastOwenScrambler(u32 v, u32 seed)
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

f32 SobolSample(i64 a, i32 dimension, u32 (*randomizer)(u32, u32), u32 seed = 0);
u64 SobolIntervalToIndex(u32 log2Scale, u64 sampleIndex, Vec2i p);
} // namespace rt

// Given pixel coordinate p, you're trying to find the index in the halton sequence that gives
// a value such that the value scaled by the max screen resolution (rounded up to the next pow
// of 2) gives the pixel coordinate you're looking for. To do this, note that multiplying a
// radical inverse number 0.d1(a)d2(a)d3(a)... by b^2 gives d1(a)d2(a).d3(a)... Thus, the
// inverse radical inverse of the two last digits of x gives the value in the halton sequence
// that corresponds with pixel coordinate p.
#endif
