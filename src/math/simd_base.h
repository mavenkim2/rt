#ifndef SIMD_BASE_H
#define SIMD_BASE_H

#include <limits>

namespace rt
{
#define MAX_LANE_WIDTH 4
#if 0
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
struct LaneF32
{
    f32 values[N];

    const f32 &operator[](i32 i)
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
struct LaneU32
{
    u32 values[N];
    const u32 &operator[](i32 i)
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

using Lane4F32 = LaneF32<4>;
using Lane8F32 = LaneF32<8>;

using Lane4U32 = LaneU32<4>;
using Lane8U32 = LaneU32<8>;

using LaneXF32 = LaneF32<MAX_LANE_WIDTH>;

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