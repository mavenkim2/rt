#include <limits>

inline f32 RandomFloat()
{
    return rand() / (RAND_MAX + 1.f);
}

inline f32 RandomFloat(f32 min, f32 max)
{
    return min + (max - min) * RandomFloat();
}

inline i32 RandomInt(i32 min, i32 max)
{
    return i32(RandomFloat(f32(min), f32(max)));
}

inline vec3 RandomVec3()
{
    return vec3(RandomFloat(), RandomFloat(), RandomFloat());
}

inline vec3 RandomVec3(f32 min, f32 max)
{
    return vec3(RandomFloat(min, max), RandomFloat(min, max), RandomFloat(min, max));
}

#if 0
inline vec3 RandomUnitVector()
{
    while (true)
    {
        vec3 result = RandomVec3(-1, 1);
        if (result.lengthSquared() < 1)
        {
            return normalize(result);
        }
    }
}
#endif

inline vec3 RandomUnitVector(vec2 u)
{
    return normalize(SampleUniformSphere(u));
}

inline vec3 RandomUnitVector()
{
    vec2 u = vec2(RandomFloat(), RandomFloat());
    return RandomUnitVector(u);
}

inline vec3 RandomOnHemisphere(const vec3 &normal)
{
    // NOTE: why can't you just normalize a vector that has a length > 1?
    vec3 result = RandomUnitVector();
    result      = dot(normal, result) > 0 ? result : -result;
    return result;
}

#if 0
inline vec3 RandomInUnitDisk()
{
    while (true)
    {
        vec3 p = vec3(RandomFloat(-1, 1), RandomFloat(-1, 1), 0);
        if (p.lengthSquared() < 1)
        {
            return p;
        }
    }
}
#endif
inline vec3 RandomInUnitDisk()
{
    vec2 u = vec2(RandomFloat(), RandomFloat());
    return vec3(SampleUniformDiskConcentric(u), 0.f);
}

inline vec3 RandomCosineDirection()
{
    f32 r1 = RandomFloat();
    f32 r2 = RandomFloat();

    f32 phi = 2 * PI * r1;
    f32 x   = cos(phi) * sqrt(r2);
    f32 y   = sin(phi) * sqrt(r2);
    f32 z   = sqrt(1 - r2);
    return vec3(x, y, z);
}

//////////////////////////////
// RNG class
//

// Pseudo RNG
// www.pcg-random.org/paper.html
// https://www.pbr-book.org/4ed/Utilities/Mathematical_Infrastructure#RNG

#define PCG32_DEFAULT_STATE  0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT           0x5851f42d4c957f2dULL
static constexpr f32 oneMinusEpsilon = 0x1.fffffep-1;

struct RNG
{
    RNG() : state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM) {}
    RNG(u64 seqIndex, u64 offset) { SetSequence(seqIndex, offset); }
    RNG(u64 seqIndex) { SetSequence(seqIndex); }

    void SetSequence(u64 sequenceIndex, u64 offset);
    void SetSequence(u64 sequenceIndex)
    {
        SetSequence(sequenceIndex, MixBits(sequenceIndex));
    }
    template <typename T>
    T Uniform();

    template <typename T>
    typename std::enable_if_t<std::is_integral_v<T>, T> Uniform(T b)
    {
        T threshold = (~b + 1u) % b;
        for (;;)
        {
            T r = Uniform<T>();
            if (r >= threshold)
                return r % b;
        }
    }
    void Advance(i64 iDelta);
    i64 operator-(const RNG &other) const;

    u64 state, inc;
};

template <typename T>
inline T RNG::Uniform()
{
    return T::unimplemented;
}

template <>
inline u32 RNG::Uniform<u32>();

template <>
inline u32 RNG::Uniform<u32>()
{
    u64 oldState   = state;
    state          = oldState * PCG32_MULT + inc;
    u32 xorShifted = (u32)(((oldState >> 18u) ^ oldState) >> 27u);
    u32 rot        = (u32)(oldState >> 59u);
    return (xorShifted >> rot) | (xorShifted << ((~rot + 1u) & 31));
}

template <>
inline u64 RNG::Uniform<u64>()
{
    u64 v0 = Uniform<u32>(), v1 = Uniform<u32>();
    return (v0 << 32) | v1;
}

template <>
inline i32 RNG::Uniform<i32>()
{
    u32 v = Uniform<u32>();
    if (v <= (u32)INT_MAX)
        return i32(v);
    return i32(v - INT_MIN) + INT_MIN;
}

inline void RNG::SetSequence(u64 sequenceIndex, u64 seed)
{
    state = 0u;
    inc   = (sequenceIndex << 1u) | 1u;
    Uniform<u32>();
    state += seed;
    Uniform<u32>();
}

template <>
inline f32 RNG::Uniform<f32>()
{
    return Min(oneMinusEpsilon, Uniform<u32>() * 0x1p-32f);
}

inline void RNG::Advance(i64 iDelta)
{
    u64 curMult = PCG32_MULT, curPlus = inc, accMult = 1u;
    u64 accPlus = 0u, delta = (u64)iDelta;
    while (delta > 0)
    {
        if (delta & 1)
        {
            accMult *= curMult;
            accPlus = accPlus * curMult + curPlus;
        }
        curPlus = (curMult + 1) * curPlus;
        curMult *= curMult;
        delta /= 2;
    }
    state = accMult * state + accPlus;
}

inline i64 RNG::operator-(const RNG &other) const
{
    assert(inc == other.inc);
    u64 curMult = PCG32_MULT, curPlus = inc, curState = other.state;
    u64 theBit = 1u, distance = 0u;
    while (state != curState)
    {
        if ((state & theBit) != (curState & theBit))
        {
            curState = curState * curMult + curPlus;
            distance |= theBit;
        }
        assert((state & theBit) == (curState & theBit));
        theBit <<= 1;
        curPlus = (curMult + 1ULL) * curPlus;
        curMult *= curMult;
    }
    return (i64)distance;
}
