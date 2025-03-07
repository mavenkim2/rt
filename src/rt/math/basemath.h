#ifndef BASEMATH_H_
#define BASEMATH_H_

#include <cmath>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <immintrin.h>

#include "../base.h"

namespace rt
{

#define Radians(x) (PI / 180.f) * x
#define Degrees(x) (180.f / PI) * x

const f32 infinity = std::numeric_limits<f32>::infinity();

static const __m128 SIMDInfinity = _mm_set1_ps(infinity);

__forceinline f32 IsInf(f32 x) { return std::isinf(x); }

__forceinline f32 IsNaN(f32 x) { return std::isnan(x); }

__forceinline f64 IsNaN(f64 x) { return std::isnan(x); }

__forceinline f32 Sqr(f32 x) { return x * x; }

__forceinline int Abs(const int x) { return ::abs(x); }
__forceinline f32 Abs(const f32 x) { return ::fabsf(x); }
__forceinline f32 Sqrt(const f32 x) { return ::sqrtf(x); }
__forceinline f32 Rsqrt(const f32 x)
{
    const __m128 a = _mm_set_ss(x);
#if defined(__AVX512VL__)
    __m128 r = _mm_rsqrt14_ss(_mm_set_ss(0.0f), a);
#else
    __m128 r = _mm_rsqrt_ss(a);
#endif
    const __m128 c = _mm_add_ss(
        _mm_mul_ss(_mm_set_ss(1.5f), r),
        _mm_mul_ss(_mm_mul_ss(_mm_mul_ss(a, _mm_set_ss(-0.5f)), r), _mm_mul_ss(r, r)));
    return _mm_cvtss_f32(c);
}
__forceinline f32 Rcp(const f32 x)
{
    const __m128 a = _mm_set_ss(x);

#if defined(__AVX512VL__)
    const __m128 r = _mm_rcp14_ss(_mm_set_ss(0.0f), a);
#else
    const __m128 r = _mm_rcp_ss(a);
#endif

#if defined(__AVX2__)
    return _mm_cvtss_f32(_mm_mul_ss(r, _mm_fnmadd_ss(r, a, _mm_set_ss(2.0f))));
#else
    return _mm_cvtss_f32(_mm_mul_ss(r, _mm_sub_ss(_mm_set_ss(2.0f), _mm_mul_ss(r, a))));
#endif
}
__forceinline f32 Cos(const f32 x) { return ::cosf(x); }
__forceinline f32 Sin(const f32 x) { return ::sinf(x); }
__forceinline f32 Tan(const f32 x) { return ::tanf(x); }
__forceinline f32 Atan2(const f32 y, const f32 x) { return ::atan2f(y, x); }
__forceinline f32 Ceil(const f32 x) { return ::ceilf(x); }
__forceinline f32 Floor(const f32 x) { return ::floorf(x); }
__forceinline f32 Pow(const f32 x, const f32 y) { return ::powf(x, y); }
__forceinline f32 Log2f(const f32 x) { return ::log2f(x); }
__forceinline f32 Log(const f32 x) { return ::logf(x); }
__forceinline f32 Copysign(const f32 a, const f32 b) { return ::copysignf(a, b); }
__forceinline f32 ArcCos(const f32 x) { return ::acosf(x); }
__forceinline bool All(bool b) { return b; }
__forceinline bool None(bool b) { return !b; }
__forceinline bool Some(bool b) { return b; }
__forceinline bool Any(bool b) { return b; }

template <typename T, typename F>
T Lerp(F t, T a, T b)
{
    return (1 - t) * a + t * b;
}

template <typename T>
T Clamp(const T &x, const T &min, const T &max)
{
    return Max(Min(max, x), min);
}

__forceinline f32 Select(bool mask, f32 a, f32 b) { return mask ? a : b; }

__forceinline u32 Select(bool mask, u32 a, u32 b) { return mask ? a : b; }

__forceinline u32 Movemask(bool mask) { return mask; }

__forceinline bool Select(bool mask, bool a, bool b) { return mask ? a : b; }

inline int Log2Int(u64 v)
{
#if _WIN32
    unsigned long lz = 0;
    _BitScanReverse64(&lz, v);
    return lz;
#else
#error
#endif
}

__forceinline f32 SafeSqrt(f32 x) { return std::sqrt(Max(0.f, x)); }

template <int n>
constexpr f32 Pow(f32 v)
{
    if constexpr (n < 0) return 1 / Pow<-n>(v);
    f32 n2 = Pow<n / 2>(v);
    return n2 * n2 * Pow < n & 1 > (v);
}

template <>
constexpr f32 Pow<0>(f32 v)
{
    return 1;
}

template <>
constexpr f32 Pow<1>(f32 v)
{
    return v;
}

#ifdef __AVX2__
__forceinline f32 FMA(const f32 a, const f32 b, const f32 c)
{
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}
__forceinline f32 FMS(const f32 a, const f32 b, const f32 c)
{
    return _mm_cvtss_f32(_mm_fmsub_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}
#else
__forceinline f32 FMA(const f32 a, const f32 b, const f32 c) { return std::fma(a, b, c); }
__forceinline f32 FMS(const f32 a, const f32 b, const f32 c) { return std::fma(a, b, -c); }
#endif

template <typename f32, typename C>
constexpr f32 EvaluatePolynomial(f32 t, C c)
{
    return c;
}

template <typename f32, typename C, typename... Args>
constexpr f32 EvaluatePolynomial(f32 t, C c, Args... cRemaining)
{
    return FMA(t, EvaluatePolynomial(t, cRemaining...), c);
}

inline f32 BitsToFloat(u32 src)
{
    f32 dst;
    std::memcpy(&dst, &src, sizeof(dst));
    return dst;
}

inline u32 FloatToBits(f32 src)
{
    u32 dst;
    std::memcpy(&dst, &src, sizeof(dst));
    return dst;
}

inline f32 AsFloat(u32 src) { return _mm_cvtss_f32(_mm_castsi128_ps(_mm_cvtsi32_si128(src))); }

inline u32 AsUInt(f32 src) { return _mm_cvtsi128_si32(_mm_castps_si128(_mm_set_ss(src))); }

inline i32 Exponent(f32 v) { return (FloatToBits(v) >> 23) - 127; }

 inline f32 FastExp(f32 x)
{
    f32 xp  = x * 1.442695041f;
    f32 fxp = std::floor(xp), f = xp - fxp;
    i32 i = (i32)fxp;

    f32 twoToF = EvaluatePolynomial(f, 1.f, 0.695556856f, 0.226173572f, 0.0781455737f);

    i32 exponent = Exponent(twoToF) + i;
    if (exponent < -126) return 0;
    if (exponent > 127) return infinity;
    u32 bits = FloatToBits(twoToF);
    bits &= 0b10000000011111111111111111111111u;
    bits |= (exponent + 127) << 23;
    return BitsToFloat(bits);
}

inline constexpr i32 NextPowerOfTwo(i32 v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

inline u32 SafeTruncateU64ToU32(u64 val)
{
    u32 result = (u32)val;
    Assert(val <= 0xffffffff);
    return result;
}
inline u16 SafeTruncateU32(u32 val)
{
    u16 result = (u16)val;
    Assert(val == result);
    return result;
}

inline u8 SafeTruncateU32ToU8(u32 val)
{
    Assert(val <= 255);
    u8 result = (u8)val;
    return result;
}

__forceinline u32 RoundFloatToU32(f32 val)
{
    u32 result = (u32)(val + 0.5f);
    return result;
}

inline u32 ReverseBits32(u32 n)
{
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
    return n;
}

// TODO: vectorized elementary functions, https://shibata.naist.jp/~n-sibata/pdfs/isc10simd.pdf
__forceinline f32 SafeASin(f32 x)
{
    Assert(x >= -1.0001 && x <= 1.0001);
    return std::asin(Clamp(x, -1.f, 1.f));
}

static constexpr f32 MachineEpsilon = std::numeric_limits<f32>::epsilon() * 0.5f;
inline constexpr f32 gamma(int n) { return (n * MachineEpsilon) / (1 - n * MachineEpsilon); }

inline float NextFloatUp(float v)
{
    if (IsInf(v) && v > 0.f) return v;
    if (v == -0.f) v = 0.f;

    uint32_t ui = FloatToBits(v);
    if (v >= 0) ++ui;
    else --ui;
    return BitsToFloat(ui);
}

inline float NextFloatDown(float v)
{
    if (IsInf(v) && v < 0.) return v;
    if (v == 0.f) v = -0.f;

    uint32_t ui = FloatToBits(v);
    if (v > 0) --ui;
    else ++ui;
    return BitsToFloat(ui);
}

//////////////////////////////

inline u32 PopCount(u32 val)
{
#ifdef __SSE4_2__
    return _mm_popcnt_u32(val);
#elif _MSC_VER
    return __popcnt(val);
#else
    u32 count = 0;
    for (u32 i = 0; i < 32; i++)
    {
        if (val & (1 << i)) count++;
    }
    return count;
#endif
}

inline float Erf(f32 x) { return ::erf(x); }
inline float ErfInv(float a)
{
    // https://stackoverflow.com/a/49743348
    float p;
    float t = Log(Max(FMA(a, -a, 1), std::numeric_limits<f32>::min()));
    if (std::abs(t) > 6.125f)
    {                                   // maximum ulp error = 2.35793
        p = 3.03697567e-10f;            //  0x1.4deb44p-32
        p = FMA(p, t, 2.93243101e-8f);  //  0x1.f7c9aep-26
        p = FMA(p, t, 1.22150334e-6f);  //  0x1.47e512p-20
        p = FMA(p, t, 2.84108955e-5f);  //  0x1.dca7dep-16
        p = FMA(p, t, 3.93552968e-4f);  //  0x1.9cab92p-12
        p = FMA(p, t, 3.02698812e-3f);  //  0x1.8cc0dep-9
        p = FMA(p, t, 4.83185798e-3f);  //  0x1.3ca920p-8
        p = FMA(p, t, -2.64646143e-1f); // -0x1.0eff66p-2
        p = FMA(p, t, 8.40016484e-1f);  //  0x1.ae16a4p-1
    }
    else
    {                                   // maximum ulp error = 2.35456
        p = 5.43877832e-9f;             //  0x1.75c000p-28
        p = FMA(p, t, 1.43286059e-7f);  //  0x1.33b458p-23
        p = FMA(p, t, 1.22775396e-6f);  //  0x1.49929cp-20
        p = FMA(p, t, 1.12962631e-7f);  //  0x1.e52bbap-24
        p = FMA(p, t, -5.61531961e-5f); // -0x1.d70c12p-15
        p = FMA(p, t, -1.47697705e-4f); // -0x1.35be9ap-13
        p = FMA(p, t, 2.31468701e-3f);  //  0x1.2f6402p-9
        p = FMA(p, t, 1.15392562e-2f);  //  0x1.7a1e4cp-7
        p = FMA(p, t, -2.32015476e-1f); // -0x1.db2aeep-3
        p = FMA(p, t, 8.86226892e-1f);  //  0x1.c5bf88p-1
    }
    return a * p;
}

} // namespace rt

#endif
