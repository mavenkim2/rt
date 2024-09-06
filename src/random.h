#include <limits>
namespace rt
{

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

inline Vec3f RandomVec3()
{
    return Vec3f(RandomFloat(), RandomFloat(), RandomFloat());
}

inline Vec3f RandomVec3(f32 min, f32 max)
{
    return Vec3f(RandomFloat(min, max), RandomFloat(min, max), RandomFloat(min, max));
}

#if 0
inline Vec3f RandomUnitVector()
{
    while (true)
    {
        Vec3f result = RandomVec3(-1, 1);
        if (result.lengthSquared() < 1)
        {
            return normalize(result);
        }
    }
}
#endif

inline Vec3f RandomUnitVector(Vec2f u)
{
    return Normalize(SampleUniformSphere(u));
}

inline Vec3f RandomUnitVector()
{
    Vec2f u = Vec2f(RandomFloat(), RandomFloat());
    return RandomUnitVector(u);
}

inline Vec3f RandomOnHemisphere(const Vec3f &normal)
{
    // NOTE: why can't you just normalize a vector that has a length > 1?
    Vec3f result = RandomUnitVector();
    result       = Dot(normal, result) > 0 ? result : -result;
    return result;
}

#if 0
inline Vec3f RandomInUnitDisk()
{
    while (true)
    {
        Vec3f p = Vec3f(RandomFloat(-1, 1), RandomFloat(-1, 1), 0);
        if (p.lengthSquared() < 1)
        {
            return p;
        }
    }
}
#endif
inline Vec3f RandomInUnitDisk()
{
    Vec2f u = Vec2f(RandomFloat(), RandomFloat());
    return Vec3f(SampleUniformDiskConcentric(u), 0.f);
}

inline Vec3f RandomCosineDirection()
{
    f32 r1 = RandomFloat();
    f32 r2 = RandomFloat();

    f32 phi = 2 * PI * r1;
    f32 x   = Cos(phi) * Sqrt(r2);
    f32 y   = Sin(phi) * Sqrt(r2);
    f32 z   = Sqrt(1 - r2);
    return Vec3f(x, y, z);
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
} // namespace rt
