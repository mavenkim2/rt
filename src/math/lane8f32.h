#ifndef LANE8F32_H
#define LANE8F32_H

namespace rt
{

template <>
struct LaneF32<8>
{
    typedef f32 Type;
    union
    {
        __m256 v;
        struct
        {
            __m128 lo, hi;
        };
        f32 f[8];
    };
    __forceinline LaneF32() {}
    __forceinline LaneF32(__m256 v) : v(v) {}
    __forceinline LaneF32(__m128 v) : v(_mm256_castps128_ps256(v)) {}

    __forceinline LaneF32(const Lane8F32 &other) { v = other.v; }
    __forceinline LaneF32(const Lane4F32 &a, const Lane4F32 &b) : v(_mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1)) {}
    __forceinline LaneF32(f32 a) : v(_mm256_set1_ps(a)) {}
    __forceinline LaneF32(f32 a, f32 b, f32 c, f32 d, f32 e, f32 f, f32 g, f32 h) { v = _mm256_setr_ps(a, b, c, d, e, f, g, h); }

    // NOTE: necessary since otherwise it would be interpreted as an integer
    // __forceinline explicit LaneF32(const Lane8U32 &l)
    // {
    //     __m256i a = _mm_and_si128(l.v, _mm_set1_epi32(0x7fffffff));
    //     __m256i b = _mm_and_si128(_mm_srai_epi32(l.v, 31), _mm_set1_epi32(0x4f000000));
    //     __m256 a2 = _mm_cvtepi32_ps(a);
    //     __m256 b2 = _mm_castsi128_ps(b);
    //     v         = _mm_add_ps(a2, b2);
    // }

    // TODO: not technically correct for unsigned integers > than int max
    __forceinline explicit LaneF32(const Lane8U32 &l) { v = _mm256_cvtepi32_ps(l); }

    // __forceinline Lane4U32 AsUInt()
    // {
    //     Lane4U32 l;
    //     l.v = _mm_castps_si128(v);
    //     return l;
    // }

    __forceinline LaneF32(ZeroTy) { v = _mm256_setzero_ps(); }
    __forceinline LaneF32(PosInfTy) { v = _mm256_set1_ps(pos_inf); }
    __forceinline LaneF32(NegInfTy) { v = _mm256_set1_ps(neg_inf); }
    __forceinline LaneF32(NaNTy) { v = _mm256_set1_ps(NaN); }
    __forceinline LaneF32(TrueTy) { v = _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_OQ); }
    __forceinline LaneF32(FalseTy) { v = _mm256_setzero_ps(); }

    template <i32 i1>
    __forceinline static LaneF32 Mask()
    {
        if constexpr (i1)
        {
            return _mm256_cmpeq_ps(_mm256_setzero_ps(), _mm256_setzero_ps());
        }
        else
        {
            return _mm256_setzero_ps();
        }
    }

    __forceinline static LaneF32 Mask(bool i)
    {
        return Lane8F32(Lane4F32::Mask(i), Lane4F32::Mask(i));
    }
    __forceinline static LaneF32 Mask(u32 i)
    {
        Assert(i >= 0 && i < 256);
        return Lane8F32(Lane4F32::Mask(i & 15), Lane4F32::Mask(i >> 4));
    }

    __forceinline operator const __m256 &() const { return v; }
    __forceinline operator __m256 &() { return v; }
    __forceinline Lane8F32 &operator=(const Lane8F32 &other)
    {
        v = other.v;
        return *this;
    }

    __forceinline const f32 &operator[](i32 i) const
    {
        Assert(i < 8);
        return f[i];
    }

    __forceinline f32 &operator[](i32 i)
    {
        Assert(i < 8);
        return f[i];
    }
    static __forceinline Lane8F32 Load(const void *ptr)
    {
        return _mm256_load_ps((f32 *)ptr);
    }
    static __forceinline Lane8F32 LoadU(const void *ptr)
    {
        return _mm256_loadu_ps((f32 *)ptr);
    }
    static __forceinline void Store(void *ptr, const Lane8F32 &l)
    {
        _mm256_store_ps((f32 *)ptr, l);
    }
};

__forceinline Lane8F32 operator+(const Lane8F32 &a, const Lane8F32 &b) { return _mm256_add_ps(a, b); }
__forceinline Lane8F32 operator+(f32 a, const Lane8F32 &b) { return Lane8F32(a) + b; }
__forceinline Lane8F32 operator+(const Lane8F32 &a, f32 b) { return a + Lane8F32(b); }

__forceinline Lane8F32 operator-(const Lane8F32 &a, const Lane8F32 &b) { return _mm256_sub_ps(a, b); }
__forceinline Lane8F32 operator-(f32 a, const Lane8F32 &b) { return Lane8F32(a) - b; }
__forceinline Lane8F32 operator-(const Lane8F32 &a, f32 b) { return a - Lane8F32(b); }

__forceinline Lane8F32 operator*(const Lane8F32 &a, const Lane8F32 &b) { return _mm256_mul_ps(a, b); }
__forceinline Lane8F32 operator*(f32 a, const Lane8F32 &b) { return Lane8F32(a) * b; }
__forceinline Lane8F32 operator*(const Lane8F32 &a, f32 b) { return a * Lane8F32(b); }

__forceinline Lane8F32 operator/(const Lane8F32 &a, const Lane8F32 &b) { return _mm256_div_ps(a, b); }
__forceinline Lane8F32 operator/(f32 a, const Lane8F32 &b) { return Lane8F32(a) / b; }
__forceinline Lane8F32 operator/(const Lane8F32 &a, f32 b) { return a / Lane8F32(b); }

// Fused multiply add
__forceinline Lane8F32 FMA(const Lane8F32 &a, const Lane8F32 &b, const Lane8F32 &c)
{
#ifdef __AVX2__
    return _mm256_fmadd_ps(a, b, c);
#else
    return a * b + c;
#endif
}

// Fused multiply subtract
__forceinline Lane8F32 FMS(const Lane8F32 &a, const Lane8F32 &b, const Lane8F32 &c)
{
#ifdef __AVX2__
    return _mm256_fmsub_ps(a, b, c);
#else
    return a * b - c;
#endif
}

__forceinline Lane8F32 Sqrt(const Lane8F32 &a) { return _mm256_sqrt_ps(a); }
__forceinline Lane8F32 Rsqrt(const Lane8F32 &a)
{
#ifdef __AVX512VL__
    Lane8F32 r = _mm256_rsqrt14_ps(a);
#else
    Lane8F32 r = _mm256_rsqrt_ps(a);
#endif
#if defined(__AVX2__)
    r = _mm256_fmadd_ps(_mm256_set1_ps(1.5f), r, _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(a, _mm256_set1_ps(-0.5f)), r), _mm256_mul_ps(r, r)));
#else
    r          = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(1.5f), r), _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(a, _mm256_set1_ps(-0.5f)), r), _mm256_mul_ps(r, r)));
#endif
    return r;
}

__forceinline Lane8F32 Rcp(const Lane8F32 &a)
{
#ifdef __AVX512VL__
    Lane8F32 r = _mm256_rcp14_ps(a);
#else
    Lane8F32 r = _mm256_rcp_ps(a);
#endif

#if defined(__AVX2__)
    return _mm256_fmadd_ps(r, _mm256_fnmadd_ps(a, r, Lane8F32(1.0f)), r); // computes r + r * (1 - a * r)
#else
    return _mm256_add_ps(r, _mm256_mul_ps(r, _mm256_sub_ps(Lane8F32(1.0f), _mm256_mul_ps(a, r)))); // computes r + r * (1 - a * r)
#endif
}

__forceinline Lane8F32 Abs(const Lane8F32 &a) { return _mm256_and_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff))); }
__forceinline Lane8F32 Min(const Lane8F32 &a, const Lane8F32 &b) { return _mm256_min_ps(a, b); }
__forceinline Lane8F32 Max(const Lane8F32 &a, const Lane8F32 &b) { return _mm256_max_ps(a, b); }

__forceinline Lane8F32 operator^(const Lane8F32 &a, const Lane8F32 &b) { return _mm256_xor_ps(a, b); }
__forceinline Lane8F32 &operator^=(Lane8F32 &a, const Lane8F32 &b)
{
    a = a ^ b;
    return a;
}
__forceinline Lane8F32 operator<(const Lane8F32 &a, const Lane8F32 &b) { return _mm256_cmp_ps(a, b, _CMP_LT_OS); }
__forceinline Lane8F32 operator<=(const Lane8F32 &a, const Lane8F32 &b) { return _mm256_cmp_ps(a, b, _CMP_LE_OS); }
__forceinline Lane8F32 operator>(const Lane8F32 &a, const Lane8F32 &b) { return _mm256_cmp_ps(a, b, _CMP_NLE_US); }
__forceinline Lane8F32 operator>=(const Lane8F32 &a, const Lane8F32 &b) { return _mm256_cmp_ps(a, b, _CMP_NLT_US); }
__forceinline Lane8F32 operator==(const Lane8F32 &a, const Lane8F32 &b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
__forceinline Lane8F32 operator&(const Lane8F32 &a, const Lane8F32 &b) { return _mm256_and_ps(a, b); }
__forceinline Lane8F32 &operator&=(Lane8F32 &a, const Lane8F32 &b)
{
    a = a & b;
    return a;
}
__forceinline Lane8F32 operator|(const Lane8F32 &a, const Lane8F32 &b) { return _mm256_or_ps(a, b); }

__forceinline Lane8F32 Select(const Lane8F32 &mask, const Lane8F32 &a, const Lane8F32 &b)
{
    return _mm256_blendv_ps(b, a, mask);
}

__forceinline i32 Movemask(const Lane8F32 &a) { return _mm256_movemask_ps(a); }
__forceinline bool All(const Lane8F32 &a) { return _mm256_movemask_ps(a) == 0xf; }
__forceinline bool Any(const Lane8F32 &a) { return _mm256_movemask_ps(a) != 0; }
__forceinline bool None(const Lane8F32 &a) { return _mm256_movemask_ps(a) == 0; }

__forceinline Lane8F32 MaskAdd(const Lane8F32 &mask, const Lane8F32 &a, const Lane8F32 &b)
{
#if defined(__AVX512VL__)
    return _mm256_mask_add_ps(a, (__mm256ask8)Movemask(mask), a, b);
#else
    return Select(mask, a + b, a);
#endif
}
__forceinline Lane8F32 MaskMin(const Lane8F32 &mask, const Lane8F32 &a, const Lane8F32 &b)
{
#if defined(__AVX512VL__)
    return _mm256_mask_min_ps(a, (__mmask8)Movemask(mask), a, b);
#else
    return Select(mask, _mm256_min_ps(a, b), a);
#endif
}

__forceinline Lane8F32 MaskMin(const Lane8F32 &mask, const Lane8F32 &a, const Lane8F32 &b, const Lane8F32 &default)
{
#if defined(__AVX512VL__)
    return _mm256_mask_min_ps(default, (__mmask8)Movemask(mask), a, b);
#else
    return Select(mask, _mm256_min_ps(a, b), default);
#endif
}

__forceinline Lane8F32 MaskMax(const Lane8F32 &mask, const Lane8F32 &a, const Lane8F32 &b)
{
#if defined(__AVX512VL__)
    return _mm256_mask_max_ps(a, (__mmask8)Movemask(mask), a, b);
#else
    return Select(mask, _mm256_max_ps(a, b), a);
#endif
}

__forceinline Lane8F32 UnpackLo(const Lane8F32 &a, const Lane8F32 &b) { return _mm256_unpacklo_ps(a, b); }
__forceinline Lane8F32 UnpackHi(const Lane8F32 &a, const Lane8F32 &b) { return _mm256_unpackhi_ps(a, b); }

template <i32 a, i32 b, i32 c, i32 d, i32 e, i32 f, i32 g, i32 h>
__forceinline Lane8F32 Blend(const Lane8F32 &l, const Lane8F32 &r)
{
    StaticAssert((a | b | c | d | e | f | g | h) == 1, Blend01);
    static constexpr u32 mask = (h << 7) | (g << 6) | (f << 5) | (e << 4) | (d << 3) | (c << 2) | (b << 1) | a;
    return _mm256_blend_ps(l, r, mask);
}

template <i32 a, i32 b, i32 c, i32 d, i32 e, i32 f, i32 g, i32 h>
__forceinline Lane8F32 Shuffle(const Lane8F32 &l)
{
    static const __m256i shuf = _mm256_setr_epi32(a, b, c, d, e, f, g, h);
    return _mm256_permutevar8x32_ps(l, shuf);
}

// TODO: permutevar8x32 is AVX2
template <i32 a>
__forceinline Lane8F32 Shuffle(const Lane8F32 &l)
{
    StaticAssert(a >= 0 && a < 8, Shuf);
    static const __m256i shuf = _mm256_setr_epi32(a, a, a, a, a, a, a, a);
    return _mm256_permutevar8x32_ps(l, shuf);
}

__forceinline Lane8F32 Shuffle(const Lane8F32 &l, const __m256i &shuf) { return _mm256_permutevar8x32_ps(l, shuf); }

template <i32 a, i32 b>
__forceinline Lane8F32 Shuffle4(const Lane8F32 &l, const Lane8F32 &r)
{
    StaticAssert(a >= 0 && a <= 3 && b >= 0 && b <= 3, InvalidPermute2f128);
    return _mm256_permute2f128_ps(l, r, (b << 4) | (a));
}

template <i32 a, i32 b>
__forceinline Lane8F32 Shuffle4(const Lane8F32 &l)
{
    StaticAssert(a >= 0 && a <= 3 && b >= 0 && b <= 3, InvalidPermute2f128);
    return _mm256_permute2f128_ps(l, l, (b << 4) | (a));
}

// NOTE: Functions with permute don't cross 128-bit lanes and work on one ymm register.
// Functions with shuffle either take two ymm registers, or shuffle across lanes.
template <i32 a, i32 b, i32 c, i32 d>
__forceinline Lane8F32 Permute(const Lane8F32 &l)
{
    return _mm256_permute_ps(l, _MM_SHUFFLE(d, c, b, a));
}

__forceinline Lane8F32 Permute(const Lane8F32 &l, const __m256i &shuf) { return _mm256_permutevar_ps(l, shuf); }

template <i32 a>
__forceinline Lane8F32 Permute(const Lane8F32 &l)
{
    StaticAssert(a >= 0 && a <= 3, Perm);
    return _mm256_permute_ps(l, (a << 6) | (a << 4) | (a << 2) | (a));
}

template <i32 i>
__forceinline f32 Extract(const Lane8F32 &a)
{
    return _mm256_cvtss_f32(Shuffle<i>(a));
}

template <i32 i>
__forceinline Lane4F32 Extract4(const Lane8F32 &a) { return _mm256_extractf128_ps(a, i); }

template <>
__forceinline Lane4F32 Extract4<0>(const Lane8F32 &a) { return _mm256_castps256_ps128(a); }

template <>
__forceinline f32 Extract<0>(const Lane8F32 &a)
{
    return _mm256_cvtss_f32(a);
}

// __forceinline Lane8F32 Permute(const Lane8F32 &a, const Lane8U32 &b)
// {
// #if defined(__AVX__)
//     return _mm256_permutevar_ps(a, b);
// #elif defined(__SSS3__)
//     const u32 MUL = 0x04040404;
//     const u32 ADD = 0x03020100;
//
//     const __m256i i0 = _mm256_cvtsi32_si128((b[0] & 3) * MUL + ADD);
//     const __m256i i1 = _mm256_cvtsi32_si128((b[1] & 3) * MUL + ADD);
//     const __m256i i2 = _mm256_cvtsi32_si128((b[2] & 3) * MUL + ADD);
//     const __m256i i3 = _mm256_cvtsi32_si128((b[3] & 3) * MUL + ADD);
//
//     __m256i permutation = _mm256_unpacklo_epi64(_mm256_unpacklo_epi32(i0, i1), _mm256_unpacklo_epi32(i2, i3));
//     return _mm256_castsi128_ps(_mm256_shuffle_epi8(_mm256_castps_si128(a), permutation));
// #else
// #error TODO
// #endif
// }

__forceinline f32 ReduceMin(const Lane8F32 &l)
{
    Lane8F32 a = Min(l, Permute<1, 0, 3, 2>(l));
    Lane8F32 b = Min(a, Permute<2, 3, 0, 1>(a));
    Lane8F32 c = Min(b, Shuffle4<1, 0>(b));
    return _mm_cvtss_f32(_mm256_castps256_ps128(c));
}

__forceinline f32 ReduceMax(const Lane8F32 &l)
{
    Lane8F32 a = Max(l, Permute<1, 0, 3, 2>(l));
    Lane8F32 b = Max(a, Permute<2, 3, 0, 1>(a));
    Lane8F32 c = Max(b, Shuffle4<1, 0>(b));
    return _mm_cvtss_f32(_mm256_castps256_ps128(c));
}

__forceinline Lane8F32 Floor(const Lane8F32 &lane)
{
    return _mm256_round_ps(lane, _MM_FROUND_TO_NEG_INF);
}
__forceinline Lane8F32 Ceil(const Lane8F32 &lane)
{
    return _mm256_round_ps(lane, _MM_FROUND_TO_POS_INF);
}

__forceinline void Transpose8x8(const Lane8F32 &inA, const Lane8F32 &inB, const Lane8F32 &inC, const Lane8F32 &inD,
                                const Lane8F32 &inE, const Lane8F32 &inF, const Lane8F32 &inG, const Lane8F32 &inH,
                                Lane8F32 &outA, Lane8F32 &outB, Lane8F32 &outC, Lane8F32 &outD,
                                Lane8F32 &outE, Lane8F32 &outF, Lane8F32 &outG, Lane8F32 &outH)
{
    Lane8F32 a = UnpackLo(inA, inC);
    Lane8F32 b = UnpackHi(inA, inC);

    Lane8F32 c = UnpackLo(inB, inD);
    Lane8F32 d = UnpackHi(inB, inD);

    Lane8F32 t0 = UnpackLo(a, c);
    Lane8F32 t1 = UnpackHi(a, c);
    Lane8F32 t2 = UnpackLo(b, d);
    Lane8F32 t3 = UnpackHi(b, d);

    Lane8F32 e = UnpackLo(inE, inG);
    Lane8F32 f = UnpackHi(inE, inG);

    Lane8F32 g = UnpackLo(inF, inH);
    Lane8F32 h = UnpackHi(inF, inH);

    Lane8F32 t4 = UnpackLo(e, g);
    Lane8F32 t5 = UnpackHi(e, g);
    Lane8F32 t6 = UnpackLo(f, h);
    Lane8F32 t7 = UnpackHi(f, h);

    outA = Shuffle4<0, 2>(t0, t4);
    outB = Shuffle4<0, 2>(t1, t5);
    outC = Shuffle4<0, 2>(t2, t6);
    outD = Shuffle4<0, 2>(t3, t7);
    outE = Shuffle4<1, 3>(t0, t4);
    outF = Shuffle4<1, 3>(t1, t5);
    outG = Shuffle4<1, 3>(t2, t6);
    outH = Shuffle4<1, 3>(t3, t7);
}

__forceinline void Transpose8x6(const Lane8F32 &inA, const Lane8F32 &inB, const Lane8F32 &inC, const Lane8F32 &inD,
                                const Lane8F32 &inE, const Lane8F32 &inF, const Lane8F32 &inG, const Lane8F32 &inH,
                                Lane8F32 &outA, Lane8F32 &outB, Lane8F32 &outC, Lane8F32 &outD,
                                Lane8F32 &outE, Lane8F32 &outF)
{
    Lane8F32 a = UnpackLo(inA, inC);
    Lane8F32 b = UnpackHi(inA, inC); // v v _ _ v v _ _

    Lane8F32 c = UnpackLo(inB, inD);
    Lane8F32 d = UnpackHi(inB, inD); // v v _ _ v v _ _

    Lane8F32 t0 = UnpackLo(a, c);
    Lane8F32 t1 = UnpackHi(a, c);
    Lane8F32 t2 = UnpackLo(b, d);

    Lane8F32 e = UnpackLo(inE, inG);
    Lane8F32 f = UnpackHi(inE, inG);

    Lane8F32 g = UnpackLo(inF, inH);
    Lane8F32 h = UnpackHi(inF, inH);

    Lane8F32 t3 = UnpackLo(e, g);
    Lane8F32 t4 = UnpackHi(e, g);
    Lane8F32 t5 = UnpackLo(f, h);

    outA = Shuffle4<0, 2>(t0, t3);
    outB = Shuffle4<0, 2>(t1, t4);
    outC = Shuffle4<0, 2>(t2, t5);

    outD = Shuffle4<1, 3>(t0, t3);
    outE = Shuffle4<1, 3>(t1, t4);
    outF = Shuffle4<1, 3>(t2, t5);
}

__forceinline Lane8U32 Flooru(Lane8F32 lane)
{
    return _mm256_cvtps_epi32(Floor(lane));
}

#if 0
#if defined(__AVX512VL__)
template <i32 R>
__forceinline Lane8F32 Rotate(const Lane8F32 &a)
{
    if constexpr (R < 0)
    {
        return _mm256_castsi128_ps(_mm256_alignr_epi32(_mm256_castps_si128(a), _mm256_castps_si128(a), (-R) % 4));
    }
    else
    {
        return _mm256_castsi128_ps(_mm256_alignr_epi32(_mm256_castps_si128(a), _mm256_castps_si128(a), (4 - (R % 4)) % 4));
    }
}
template <i32 S, typename T>
__forceinline Lane8F32 ShiftUp(const Lane8F32 &a)
{
    StaticAssert(S >= 0 && S <= 4, ShiftMustBeBetween0And4);
    return _mm256_castsi128_ps(_mm256_alignr_epi32(Lane8F32(T()), _mm256_castps_si128(a), 4 - S));
}

template <i32 S, typename T>
__forceinline Lane8F32 ShiftDown(const Lane8F32 &a)
{
    StaticAssert(S >= 0 && S <= 4, ShiftMustBeBetween0And4);
    return _mm256_castsi128_ps(_mm256_alignr_epi32(_mm256_castps_si128(a), Lane8F32(T()), S));
}

#else
template <i32 R>
__forceinline Lane8F32 Rotate(const Lane8F32 &a)
{
    if constexpr (R == 4)
    {
        return a;
    }
    else
    {
        constexpr i32 S = (R > 0) ? (4 - (R % 4)) : -R;
        constexpr i32 A = (S + 0) % 4;
        constexpr i32 B = (S + 1) % 4;
        constexpr i32 C = (S + 2) % 4;
        constexpr i32 D = (S + 3) % 4;

        return Shuffle<A, B, C, D>(a);
    }
}

template <i32 S, typename T>
__forceinline Lane8F32 ShiftUp(const Lane8F32 &a)
{
    StaticAssert(S >= 0 && S <= 4, ShiftMustBeBetween0And4);
    constexpr u32 A        = S > 0 ? 0 : 0xffffffff;
    constexpr u32 B        = S > 1 ? 0 : 0xffffffff;
    constexpr u32 C        = S > 2 ? 0 : 0xffffffff;
    constexpr u32 D        = S > 3 ? 0 : 0xffffffff;
    const __m256 shiftMask = _mm256_castsi128_ps(Lane4U32(A, B, C, D));

    return Select(shiftMask, Rotate<S>(a), Lane8F32(T()));
}

template <i32 S, typename T>
__forceinline Lane8F32 ShiftDown(const Lane8F32 &a)
{
    StaticAssert(S >= 0 && S <= 4, ShiftMustBeBetween0And4);

    constexpr u32 A = S > 3 ? 0 : 0xffffffff;
    constexpr u32 B = S > 2 ? 0 : 0xffffffff;
    constexpr u32 C = S > 1 ? 0 : 0xffffffff;
    constexpr u32 D = S > 0 ? 0 : 0xffffffff;

    const __m256 shiftMask = _mm256_castsi128_ps(Lane4U32(A, B, C, D));
    return Select(shiftMask, Rotate<-S>(a), Lane8F32(T()));
}
#endif

template <i32 S>
__forceinline Lane8F32 ShiftUp(const Lane8F32 &a)
{
    return ShiftUp<S, ZeroTy>(a);
}

template <i32 S>
__forceinline Lane8F32 ShiftDown(const Lane8F32 &a) { return ShiftDown<S, ZeroTy>(a); }
#endif

} // namespace rt

#endif
