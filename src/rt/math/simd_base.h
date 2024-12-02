#ifndef SIMD_BASE_H
#define SIMD_BASE_H

#include <limits>

namespace rt
{
#if 1
#ifdef __AVX512VL__
#define MAX_LANE_WIDTH 16
#elif defined(__AVX2__)
#define MAX_LANE_WIDTH 8
#elif defined(__SSE2__)
#define MAX_LANE_WIDTH 4
#else
#define MAX_LANE_WIDTH 1
#endif
#endif

static const float one_over_255  = 1.0f / 255.0f;
static const float min_rcp_input = 1E-18f; // for abs(x) >= min_rcp_input the newton raphson rcp calculation does not fail

/* we consider floating point numbers in that range as valid input numbers */
static float FLT_LARGE = 1.844E18f;

struct TrueTy
{
    __forceinline operator bool() const { return true; }
};

const constexpr TrueTy True = TrueTy();

struct FalseTy
{
    __forceinline operator bool() const { return false; }
};

const constexpr FalseTy False = FalseTy();

struct ZeroTy
{
    __forceinline operator double() const { return 0; }
    __forceinline operator float() const { return 0; }
    __forceinline operator long long() const { return 0; }
    __forceinline operator unsigned long long() const { return 0; }
    __forceinline operator long() const { return 0; }
    __forceinline operator unsigned long() const { return 0; }
    __forceinline operator int() const { return 0; }
    __forceinline operator unsigned int() const { return 0; }
    __forceinline operator short() const { return 0; }
    __forceinline operator unsigned short() const { return 0; }
    __forceinline operator char() const { return 0; }
    __forceinline operator unsigned char() const { return 0; }
};

const constexpr ZeroTy zero = ZeroTy();

struct OneTy
{
    __forceinline operator double() const { return 1; }
    __forceinline operator float() const { return 1; }
    __forceinline operator long long() const { return 1; }
    __forceinline operator unsigned long long() const { return 1; }
    __forceinline operator long() const { return 1; }
    __forceinline operator unsigned long() const { return 1; }
    __forceinline operator int() const { return 1; }
    __forceinline operator unsigned int() const { return 1; }
    __forceinline operator short() const { return 1; }
    __forceinline operator unsigned short() const { return 1; }
    __forceinline operator char() const { return 1; }
    __forceinline operator unsigned char() const { return 1; }
};

const constexpr OneTy one = OneTy();

struct NegInfTy
{
    __forceinline operator double() const { return -std::numeric_limits<double>::infinity(); }
    __forceinline operator float() const { return -std::numeric_limits<float>::infinity(); }
    __forceinline operator long long() const { return (std::numeric_limits<long long>::min)(); }
    __forceinline operator unsigned long long() const { return (std::numeric_limits<unsigned long long>::min)(); }
    __forceinline operator long() const { return (std::numeric_limits<long>::min)(); }
    __forceinline operator unsigned long() const { return (std::numeric_limits<unsigned long>::min)(); }
    __forceinline operator int() const { return (std::numeric_limits<int>::min)(); }
    __forceinline operator unsigned int() const { return (std::numeric_limits<unsigned int>::min)(); }
    __forceinline operator short() const { return (std::numeric_limits<short>::min)(); }
    __forceinline operator unsigned short() const { return (std::numeric_limits<unsigned short>::min)(); }
    __forceinline operator char() const { return (std::numeric_limits<char>::min)(); }
    __forceinline operator unsigned char() const { return (std::numeric_limits<unsigned char>::min)(); }
};

const constexpr NegInfTy neg_inf = NegInfTy();

struct PosInfTy
{
    __forceinline operator double() const { return std::numeric_limits<double>::infinity(); }
    __forceinline operator float() const { return std::numeric_limits<float>::infinity(); }
    __forceinline operator long long() const { return (std::numeric_limits<long long>::max)(); }
    __forceinline operator unsigned long long() const { return (std::numeric_limits<unsigned long long>::max)(); }
    __forceinline operator long() const { return (std::numeric_limits<long>::max)(); }
    __forceinline operator unsigned long() const { return (std::numeric_limits<unsigned long>::max)(); }
    __forceinline operator int() const { return (std::numeric_limits<int>::max)(); }
    __forceinline operator unsigned int() const { return (std::numeric_limits<unsigned int>::max)(); }
    __forceinline operator short() const { return (std::numeric_limits<short>::max)(); }
    __forceinline operator unsigned short() const { return (std::numeric_limits<unsigned short>::max)(); }
    __forceinline operator char() const { return (std::numeric_limits<char>::max)(); }
    __forceinline operator unsigned char() const { return (std::numeric_limits<unsigned char>::max)(); }
};

const constexpr PosInfTy inf     = PosInfTy();
const constexpr PosInfTy pos_inf = PosInfTy();

struct NaNTy
{
    __forceinline operator double() const { return std::numeric_limits<double>::quiet_NaN(); }
    __forceinline operator float() const { return std::numeric_limits<float>::quiet_NaN(); }
};

const constexpr NaNTy NaN = NaNTy();

struct UlpTy
{
    __forceinline operator double() const { return std::numeric_limits<double>::epsilon(); }
    __forceinline operator float() const { return std::numeric_limits<float>::epsilon(); }
};

const constexpr UlpTy ulp = UlpTy();

struct PiTy
{
    __forceinline operator double() const { return double(PI); }
    __forceinline operator float() const { return float(PI); }
};

const constexpr PiTy pi = PiTy();

struct OneOverPiTy
{
    __forceinline operator double() const { return double(InvPi); }
    __forceinline operator float() const { return float(InvPi); }
};

const constexpr OneOverPiTy one_over_pi = OneOverPiTy();

struct TwoPiTy
{
    __forceinline operator double() const { return double(2.0 * PI); }
    __forceinline operator float() const { return float(2.0 * PI); }
};

const constexpr TwoPiTy two_pi = TwoPiTy();

struct OneOverTwoPiTy
{
    __forceinline operator double() const { return double(0.5 * InvPi); }
    __forceinline operator float() const { return float(0.5 * InvPi); }
};

const constexpr OneOverTwoPiTy one_over_two_pi = OneOverTwoPiTy();

struct FourPiTy
{
    __forceinline operator double() const { return double(4.0 * PI); }
    __forceinline operator float() const { return float(4.0 * PI); }
};

const constexpr FourPiTy four_pi = FourPiTy();

struct OneOverFourPiTy
{
    __forceinline operator double() const { return double(0.25 * InvPi); }
    __forceinline operator float() const { return float(0.25 * InvPi); }
};

const constexpr OneOverFourPiTy one_over_four_pi = OneOverFourPiTy();

struct StepTy
{
    __forceinline operator double() const { return 0; }
    __forceinline operator float() const { return 0; }
    __forceinline operator long long() const { return 0; }
    __forceinline operator unsigned long long() const { return 0; }
    __forceinline operator long() const { return 0; }
    __forceinline operator unsigned long() const { return 0; }
    __forceinline operator int() const { return 0; }
    __forceinline operator unsigned int() const { return 0; }
    __forceinline operator short() const { return 0; }
    __forceinline operator unsigned short() const { return 0; }
    __forceinline operator char() const { return 0; }
    __forceinline operator unsigned char() const { return 0; }
};

const constexpr StepTy step = StepTy();

struct ReverseStepTy
{
};

const constexpr ReverseStepTy reverse_step = ReverseStepTy();

struct EmptyTy
{
};

const constexpr EmptyTy empty = EmptyTy();

struct FullTy
{
};

const constexpr FullTy full = FullTy();

struct UndefinedTy
{
};

const constexpr UndefinedTy undefined = UndefinedTy();

template <i32 N>
struct LaneF32_
{
    f32 values[N];

    const f32 &operator[](i32 i) const
    {
        Assert(i < N);
        return values[i];
    }

    f32 &operator[](i32 i)
    {
        Assert(i < N);
        return values[i];
    }
};

template <i32 N>
struct LaneF32Helper
{
    using Type = LaneF32_<N>;
};

template <>
struct LaneF32Helper<1>
{
    using Type = f32;
};

template <i32 N>
using LaneF32 = typename LaneF32Helper<N>::Type;

template <i32 K>
__forceinline LaneF32<K> Cos(const LaneF32<K> &a)
{
    alignas(4 * K) f32 result[4];
    for (u32 i = 0; i < 4; i++)
    {
        result[i] = Cos(a[i]);
    }
    return LaneF32<K>::Load(result);
}

template <i32 K>
__forceinline LaneF32<K> Sin(const LaneF32<K> &a)
{
    alignas(4 * K) f32 result[4];
    for (u32 i = 0; i < 4; i++)
    {
        result[i] = Sin(a[i]);
    }
    return LaneF32<K>::Load(result);
}

template <i32 K>
__forceinline LaneF32<K> Atan2(const LaneF32<K> &y, const LaneF32<K> &x)
{
    alignas(4 * K) f32 result[4];
    for (u32 i = 0; i < 4; i++)
    {
        result[i] = Atan2(y[i], x[i]);
    }
    return LaneF32<K>::Load(result);
}

// template <>
// struct LaneF32_<1>
// {
//     f32 value;
//     const f32 &operator[](i32 i) const
//     {
//         Assert(i == 0);
//         return value;
//     }
//     f32 &operator[](i32 i)
//     {
//         Assert(i == 0);
//         return value;
//     }
//     __forceinline LaneF32_() {}
//     __forceinline LaneF32_(f32 v) : value(v) {}
//     __forceinline LaneF32_(ZeroTy) : value(0.f) {}
//     __forceinline explicit operator f32() const { return value; }
//
//     static __forceinline LaneF32_ Load(const void *ptr)
//     {
//         Assert(ptr);
//         return LaneF32_(((f32 *)ptr)[0]);
//     }
//     template <bool b>
//     static __forceinline bool Mask() { return b; }
// };

template <i32 N>
struct LaneU32_
{
    u32 values[N];
    const u32 &operator[](i32 i) const
    {
        Assert(i < N);
        return values[i];
    }

    u32 &operator[](i32 i)
    {
        Assert(i < N);
        return values[i];
    }
};

template <i32 N>
struct LaneU32Helper
{
    using Type = LaneU32_<N>;
};

template <>
struct LaneU32Helper<1>
{
    using Type = u32;
};

template <i32 N>
using LaneU32 = typename LaneU32Helper<N>::Type;

// template <>
// struct LaneU32<1>
// {
//     u32 value;
//     const u32 &operator[](i32 i) const
//     {
//         Assert(i == 0);
//         return value;
//     }
//     u32 &operator[](i32 i)
//     {
//         Assert(i == 0);
//         return value;
//     }
//     __forceinline LaneU32() {}
//     __forceinline LaneU32(u32 v) : value(v) {}
//     __forceinline explicit operator u32() const { return value; }
// };

using Lane1F32 = LaneF32<1>;
using Lane4F32 = LaneF32<4>;
using Lane8F32 = LaneF32<8>;

using Lane1U32 = LaneU32<1>;
using Lane4U32 = LaneU32<4>;
using Lane8U32 = LaneU32<8>;

using LaneXF32 = LaneF32<MAX_LANE_WIDTH>;

// __forceinline Lane1F32 operator+(const Lane1F32 &a, const Lane1F32 &b) { return a.value + b.value; }
// __forceinline Lane1F32 operator-(const Lane1F32 &a, const Lane1F32 &b) { return a.value - b.value; }
// __forceinline Lane1F32 operator/(const Lane1F32 &a, const Lane1F32 &b) { return a.value / b.value; }
// __forceinline Lane1F32 operator*(const Lane1F32 &a, const Lane1F32 &b) { return a.value * b.value; }
// __forceinline Lane1F32 operator-(const Lane1F32 &a) { return -a.value; }
// __forceinline Lane1F32 &operator+=(Lane1F32 &a, const Lane1F32 &b)
// {
//     a = a + b;
//     return a;
// }
// __forceinline Lane1F32 &operator*=(Lane1F32 &a, const Lane1F32 &b)
// {
//     a = a * b;
//     return a;
// }
// __forceinline bool operator>(const Lane1F32 &a, const Lane1F32 &b) { return a.value > b.value; }
// __forceinline bool operator<(const Lane1F32 &a, const Lane1F32 &b) { return a.value < b.value; }
// __forceinline bool operator==(const Lane1F32 &a, const Lane1F32 &b) { return a.value == b.value; }
// __forceinline Lane1F32 Cos(const Lane1F32 &a) { return Cos(a.value); }
// __forceinline Lane1F32 Sin(const Lane1F32 &a) { return Sin(a.value); }
// __forceinline Lane1F32 Sqrt(const Lane1F32 &a) { return Sqrt(a.value); }
// __forceinline Lane1F32 Abs(const Lane1F32 &a) { return Abs(a.value); }
// __forceinline Lane1F32 Copysignf(const Lane1F32 &mag, const Lane1F32 &sign) { return Copysignf(mag.value, sign.value); }
// __forceinline Lane1F32 Clamp(const Lane1F32 &v, const Lane1F32 &min, const Lane1F32 &max) { return Clamp(v.value, min.value, max.value); }
// __forceinline Lane1F32 Select(const Lane1F32 mask, const Lane1F32 a, const Lane1F32 b) { return mask.value ? a : b; }
// __forceinline Lane1F32 FMA(const Lane1F32 a, const Lane1F32 b, const Lane1F32 c) { return std::fma(a.value, b.value, c.value); }
// __forceinline Lane1F32 FMS(const Lane1F32 a, const Lane1F32 b, const Lane1F32 c) { return std::fma(a.value, b.value, -c.value); }
// __forceinline Lane1F32 Rsqrt(const Lane1F32 a) { return 1.f / Sqrt(a.value); }
// __forceinline Lane1F32 SafeSqrt(const Lane1F32 a) { return 1.f / Sqrt(a.value); }
// __forceinline Lane1F32 Max(const Lane1F32 a, const Lane1F32 b) { return Max(a.value, b.value); }
// __forceinline Lane1F32 Min(const Lane1F32 a, const Lane1F32 b) { return Min(a.value, b.value); }
// __forceinline bool operator!=(const Lane1U32 &a, const Lane1U32 &b) { return a.value != b.value; }

// lane width for integration
#define IntN 1
#if IntN == 1
typedef bool MaskF32;
#else
typedef LaneF32<IntN> MaskF32;
#endif
typedef LaneF32<IntN> LaneNF32;
typedef LaneU32<IntN> LaneNU32;
template <typename T>
struct MaskBase;

template <>
struct MaskBase<f32>
{
    using Type = bool;
};

// template <>
// struct MaskBase<Lane1F32>
// {
//     using Type = bool;
// };

template <typename T>
struct MaskBase
{
    using Type = T;
};

template <typename T>
using Mask = typename MaskBase<T>::Type;

f32 &Set(f32 &val, u32 index)
{
    Assert(index == 0);
    return val;
}

u32 &Set(u32 &val, u32 index)
{
    Assert(index == 0);
    return val;
}

f32 Get(f32 val, u32 index)
{
    Assert(index == 0);
    return val;
}

u32 Get(u32 val, u32 index)
{
    Assert(index == 0);
    return val;
}

static const __m128 _mm_lookupmask_ps[16] = {
    _mm_castsi128_ps(_mm_set_epi32(0, 0, 0, 0)),
    _mm_castsi128_ps(_mm_set_epi32(0, 0, 0, -1)),
    _mm_castsi128_ps(_mm_set_epi32(0, 0, -1, 0)),
    _mm_castsi128_ps(_mm_set_epi32(0, 0, -1, -1)),
    _mm_castsi128_ps(_mm_set_epi32(0, -1, 0, 0)),
    _mm_castsi128_ps(_mm_set_epi32(0, -1, 0, -1)),
    _mm_castsi128_ps(_mm_set_epi32(0, -1, -1, 0)),
    _mm_castsi128_ps(_mm_set_epi32(0, -1, -1, -1)),
    _mm_castsi128_ps(_mm_set_epi32(-1, 0, 0, 0)),
    _mm_castsi128_ps(_mm_set_epi32(-1, 0, 0, -1)),
    _mm_castsi128_ps(_mm_set_epi32(-1, 0, -1, 0)),
    _mm_castsi128_ps(_mm_set_epi32(-1, 0, -1, -1)),
    _mm_castsi128_ps(_mm_set_epi32(-1, -1, 0, 0)),
    _mm_castsi128_ps(_mm_set_epi32(-1, -1, 0, -1)),
    _mm_castsi128_ps(_mm_set_epi32(-1, -1, -1, 0)),
    _mm_castsi128_ps(_mm_set_epi32(-1, -1, -1, -1)),
};

__forceinline u32 Bsf(u32 val)
{
    unsigned long result = 0;
    _BitScanForward(&result, val);
    return result;
}

__forceinline u32 Bsr(u32 val)
{
    unsigned long result = 0;
    _BitScanReverse(&result, val);
    return result;
}

__forceinline constexpr u32 BsfConst(const u32 val)
{
    if (val == 0) return 0;
    for (u32 i = 0; i < 32; i++)
    {
        if (val & (1 << i)) return i;
    }
    return 32;
}

} // namespace rt

#endif
