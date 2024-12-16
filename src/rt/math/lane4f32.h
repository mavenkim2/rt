#ifndef LANE4F32_H
#define LANE4F32_H

namespace rt
{

template <>
struct LaneF32_<4>
{
    typedef f32 Type;
    enum
    {
        N = 4,
    };
    union
    {
        __m128 v;
        f32 f[4];
        // NOTE: this is technically undefined behavior, but should be fine for most popular
        // compilers
        struct
        {
            f32 values[3];
            union
            {
                i32 integer;
                u32 u;
                f32 fl;
            };
        };
    };
    __forceinline LaneF32_() {}
    __forceinline LaneF32_(__m128 v) : v(v) {}
    __forceinline LaneF32_(const Lane4F32 &other) { v = other.v; }
    __forceinline LaneF32_(f32 a) { v = _mm_set1_ps(a); }
    __forceinline LaneF32_(f32 a, f32 b, f32 c, f32 d) { v = _mm_setr_ps(a, b, c, d); }

    // NOTE: necessary since otherwise it would be interpreted as an integer
    __forceinline explicit LaneF32_(const Lane4U32 &l)
    {
        __m128i a = _mm_and_si128(l.v, _mm_set1_epi32(0x7fffffff));
        __m128i b = _mm_and_si128(_mm_srai_epi32(l.v, 31), _mm_set1_epi32(0x4f000000));
        __m128 a2 = _mm_cvtepi32_ps(a);
        __m128 b2 = _mm_castsi128_ps(b);
        v         = _mm_add_ps(a2, b2);
    }

    __forceinline Lane4U32 AsUInt()
    {
        Lane4U32 l;
        l.v = _mm_castps_si128(v);
        return l;
    }

    __forceinline LaneF32_(ZeroTy) { v = _mm_setzero_ps(); }
    __forceinline LaneF32_(PosInfTy) { v = _mm_set1_ps(pos_inf); }
    __forceinline LaneF32_(NegInfTy) { v = _mm_set1_ps(neg_inf); }
    __forceinline LaneF32_(NaNTy) { v = _mm_set1_ps(NaN); }

    template <i32 i1>
    __forceinline static LaneF32_ Mask()
    {
        if constexpr (i1)
        {
            return _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps());
        }
        else
        {
            return _mm_setzero_ps();
        }
    }

    __forceinline static LaneF32_ Mask(bool i)
    {
        return _mm_lookupmask_ps[((size_t)i << 3) | ((size_t)i << 2) | ((size_t)i << 1) |
                                 ((size_t)i)];
    }

    __forceinline static LaneF32_ Mask(u32 i)
    {
        Assert(i >= 0 && i < 16);
        return _mm_lookupmask_ps[i]; //((size_t)i << 3) | ((size_t)i << 2) | ((size_t)i << 1) |
                                     //((size_t)i)];
    }

    __forceinline LaneF32_(TrueTy) { v = _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps()); }
    __forceinline LaneF32_(FalseTy) { v = _mm_setzero_ps(); }

    __forceinline operator const __m128 &() const { return v; }
    __forceinline operator __m128 &() { return v; }
    __forceinline Lane4F32 &operator=(const Lane4F32 &other)
    {
        v = other.v;
        return *this;
    }

    __forceinline const f32 &operator[](i32 i) const
    {
        Assert(i < 4);
        return f[i];
    }

    __forceinline f32 &operator[](i32 i)
    {
        Assert(i < 4);
        return f[i];
    }
    static __forceinline Lane4F32 Load(const void *ptr) { return _mm_load_ps((f32 *)ptr); }
    static __forceinline Lane4F32 LoadU(const void *ptr) { return _mm_loadu_ps((f32 *)ptr); }
    static __forceinline void Store(void *ptr, const Lane4F32 &l)
    {
        _mm_store_ps((f32 *)ptr, l);
    }
    static __forceinline void StoreU(void *ptr, const Lane4F32 &l)
    {
        _mm_storeu_ps((f32 *)ptr, l);
    }

#if defined(__AVX2__)
    static __forceinline void StoreU(const u32 mask, void *ptr, const Lane4F32 &l)
    {
        _mm_maskstore_ps((f32 *)ptr, _mm_castps_si128(Mask(mask)), l);
    }
#else
    static __forceinline void StoreU(const Lane4F32 &mask, void *ptr, const Lane4F32 &l)
    {
        _mm_storeu_ps((f32 *)ptr, Select(mask, l, Load(ptr)));
    }
#endif

    friend __forceinline Lane4F32 Select(const Lane4F32 &mask, const Lane4F32 &a,
                                         const Lane4F32 &b)
    {
#ifdef __SSE4_1__
        return _mm_blendv_ps(b, a, mask);
#else
        return _mm_or_ps(_mm_andnot_ps(mask, b), _mm_and_ps(mask, a));
#endif
    }
};

__forceinline Lane4F32 operator+(const Lane4F32 &a, const Lane4F32 &b)
{
    return _mm_add_ps(a, b);
}
__forceinline Lane4F32 operator+(f32 a, const Lane4F32 &b) { return Lane4F32(a) + b; }
__forceinline Lane4F32 operator+(const Lane4F32 &a, f32 b) { return a + Lane4F32(b); }

__forceinline Lane4F32 &operator+=(Lane4F32 &a, const Lane4F32 &b)
{
    a = a + b;
    return a;
}

__forceinline Lane4F32 operator-(const Lane4F32 &a, const Lane4F32 &b)
{
    return _mm_sub_ps(a, b);
}
__forceinline Lane4F32 operator-(f32 a, const Lane4F32 &b) { return Lane4F32(a) - b; }
__forceinline Lane4F32 operator-(const Lane4F32 &a, f32 b) { return a - Lane4F32(b); }

__forceinline Lane4F32 &operator-=(Lane4F32 &a, const Lane4F32 &b)
{
    a = a - b;
    return a;
}

__forceinline Lane4F32 operator*(const Lane4F32 &a, const Lane4F32 &b)
{
    return _mm_mul_ps(a, b);
}
__forceinline Lane4F32 operator*(f32 a, const Lane4F32 &b) { return Lane4F32(a) * b; }
__forceinline Lane4F32 operator*(const Lane4F32 &a, f32 b) { return a * Lane4F32(b); }

__forceinline Lane4F32 &operator*=(Lane4F32 &a, const Lane4F32 &b)
{
    a = a * b;
    return a;
}

__forceinline Lane4F32 operator/(const Lane4F32 &a, const Lane4F32 &b)
{
    return _mm_div_ps(a, b);
}
__forceinline Lane4F32 operator/(f32 a, const Lane4F32 &b) { return Lane4F32(a) / b; }
__forceinline Lane4F32 operator/(const Lane4F32 &a, f32 b) { return a / Lane4F32(b); }

// Fused multiply add
__forceinline Lane4F32 FMA(const Lane4F32 &a, const Lane4F32 &b, const Lane4F32 &c)
{
#ifdef __AVX2__
    return _mm_fmadd_ps(a, b, c);
#else
    return a * b + c;
#endif
}

// Fused multiply subtract
__forceinline Lane4F32 FMS(const Lane4F32 &a, const Lane4F32 &b, const Lane4F32 &c)
{
#ifdef __AVX2__
    return _mm_fmsub_ps(a, b, c);
#else
    return a * b - c;
#endif
}

__forceinline Lane4F32 Sqr(const Lane4F32 &a) { return a * a; }
__forceinline Lane4F32 Sqrt(const Lane4F32 &a) { return _mm_sqrt_ps(a); }

__forceinline Lane4F32 Rsqrt(const Lane4F32 &a)
{
#ifdef __AVX512VL__
    Lane4F32 r = _mm_rsqrt14_ps(a);
#else
    Lane4F32 r = _mm_rsqrt_ps(a);
#endif
#if defined(__AVX2__)
    r = _mm_fmadd_ps(
        _mm_set1_ps(1.5f), r,
        _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(a, _mm_set1_ps(-0.5f)), r), _mm_mul_ps(r, r)));
#else
    r = _mm_add_ps(
        _mm_mul_ps(_mm_set1_ps(1.5f), r),
        _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(a, _mm_set1_ps(-0.5f)), r), _mm_mul_ps(r, r)));
#endif
    return r;
}

__forceinline Lane4F32 Rcp(const Lane4F32 &a)
{
#ifdef __AVX512VL__
    Lane4F32 r = _mm_rcp14_ps(a);
#else
    Lane4F32 r = _mm_rcp_ps(a);
#endif

#if defined(__AVX2__)
    return _mm_fmadd_ps(r, _mm_fnmadd_ps(a, r, Lane4F32(1.0f)),
                        r); // computes r + r * (1 - a * r)
#else
    return _mm_add_ps(
        r, _mm_mul_ps(r, _mm_sub_ps(Lane4F32(1.0f),
                                    _mm_mul_ps(a, r)))); // computes r + r * (1 - a * r)
#endif
}

__forceinline Lane4F32 Abs(const Lane4F32 &a)
{
    return _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)));
}
__forceinline Lane4F32 Signmask(const Lane4F32 &a)
{
    return _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x80000000)));
}

__forceinline Lane4F32 FlipSign(const Lane4F32 &a)
{
    static const __m128 signFlipMask = _mm_setr_ps(-0.f, -0.f, -0.f, -0.f);
    return _mm_xor_ps(a, signFlipMask);
}
__forceinline Lane4F32 Copysign(const Lane4F32 &mag, const Lane4F32 &sign)
{
    return _mm_or_ps(Abs(mag), Signmask(sign));
}

__forceinline Lane4F32 operator-(const Lane4F32 &a) { return FlipSign(a); }

__forceinline Lane4F32 Min(const Lane4F32 &a, const Lane4F32 &b) { return _mm_min_ps(a, b); }

__forceinline Lane4F32 Max(const Lane4F32 &a, const Lane4F32 &b) { return _mm_max_ps(a, b); }
__forceinline Lane4F32 operator!(const Lane4F32 &a)
{
    return _mm_xor_ps(a, Lane4F32::Mask<true>());
}
__forceinline Lane4F32 operator^(const Lane4F32 &a, const Lane4F32 &b)
{
    return _mm_xor_ps(a, b);
}
__forceinline Lane4F32 operator<(const Lane4F32 &a, const Lane4F32 &b)
{
    return _mm_cmplt_ps(a, b);
}
__forceinline Lane4F32 operator<=(const Lane4F32 &a, const Lane4F32 &b)
{
    return _mm_cmple_ps(a, b);
}
__forceinline Lane4F32 operator>(const Lane4F32 &a, const Lane4F32 &b)
{
    return _mm_cmpgt_ps(a, b);
}
__forceinline Lane4F32 operator>=(const Lane4F32 &a, const Lane4F32 &b)
{
    return _mm_cmpge_ps(a, b);
}
__forceinline Lane4F32 operator==(const Lane4F32 &a, const Lane4F32 &b)
{
    return _mm_cmpeq_ps(a, b);
}
__forceinline Lane4F32 operator!=(const Lane4F32 &a, const Lane4F32 &b)
{
    return _mm_cmpneq_ps(a, b);
}
__forceinline Lane4F32 operator~(const Lane4F32 &a) { return _mm_xor_ps(a, Lane4F32(True)); }

__forceinline Lane4F32 operator==(const Lane4U32 &a, const Lane4U32 &b)
{
    return _mm_castsi128_ps(_mm_cmpeq_epi32(a, b));
}
__forceinline Lane4F32 operator!=(const Lane4U32 &a, const Lane4U32 &b) { return !(a == b); }

__forceinline Lane4F32 operator&(const Lane4F32 &a, const Lane4F32 &b)
{
    return _mm_and_ps(a, b);
}
__forceinline Lane4F32 &operator&=(Lane4F32 &a, const Lane4F32 &b)
{
    a = a & b;
    return a;
}
__forceinline Lane4F32 operator|(const Lane4F32 &a, const Lane4F32 &b)
{
    return _mm_or_ps(a, b);
}
__forceinline Lane4F32 &operator|=(Lane4F32 &a, const Lane4F32 &b)
{
    a = a | b;
    return a;
}

// __forceinline Lane4F32 Select(const Lane4F32 &mask, const Lane4F32 &a, const Lane4F32 &b)
// {
// #ifdef __SSE4_1__
//     return _mm_blendv_ps(b, a, mask);
// #else
//     return _mm_or_ps(_mm_andnot_ps(mask, b), _mm_and_ps(mask, a));
// #endif
// }

__forceinline Lane4U32 Select(const Lane4F32 &mask, const Lane4U32 &a, const Lane4U32 &b)
{
#ifdef __SSE4_1__
    return _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(b), _mm_castsi128_ps(a), mask));
#else
    return _mm_or_si128(_mm_andnot_si128(mask, b), _mm_and_si128(mask, a));
#endif
}

__forceinline Lane4F32 SafeSqrt(const Lane4F32 &a) { return _mm_sqrt_ps(Max(a, 0.f)); }

__forceinline i32 Movemask(const Lane4F32 &a) { return _mm_movemask_ps(a); }
__forceinline bool All(const Lane4F32 &a) { return _mm_movemask_ps(a) == 0xf; }
__forceinline bool Any(const Lane4F32 &a) { return _mm_movemask_ps(a) != 0; }
__forceinline bool None(const Lane4F32 &a) { return _mm_movemask_ps(a) == 0; }

__forceinline Lane4F32 MaskAdd(const Lane4F32 &mask, const Lane4F32 &a, const Lane4F32 &b)
{
#if defined(__AVX512VL__)
    return _mm_mask_add_ps(a, (__mmask8)Movemask(mask), a, b);
#else
    return Select(mask, a + b, a);
#endif
}
__forceinline Lane4F32 MaskMin(const Lane4F32 &mask, const Lane4F32 &a, const Lane4F32 &b)
{
#if defined(__AVX512VL__)
    return _mm_mask_min_ps(a, (__mmask8)Movemask(mask), a, b);
#else
    return Select(mask, _mm_min_ps(a, b), a);
#endif
}

__forceinline Lane4F32 MaskMax(const Lane4F32 &mask, const Lane4F32 &a, const Lane4F32 &b)
{
#if defined(__AVX512VL__)
    return _mm_mask_max_ps(a, (__mmask8)Movemask(mask), a, b);
#else
    return Select(mask, _mm_max_ps(a, b), a);
#endif
}

__forceinline Lane4F32 UnpackLo(const Lane4F32 &a, const Lane4F32 &b)
{
    return _mm_unpacklo_ps(a, b);
}
__forceinline Lane4F32 UnpackHi(const Lane4F32 &a, const Lane4F32 &b)
{
    return _mm_unpackhi_ps(a, b);
}

template <i32 a, i32 b, i32 c, i32 d>
__forceinline Lane4F32 Shuffle(const Lane4F32 &l)
{
    return _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(l), _MM_SHUFFLE(d, c, b, a)));
}

template <i32 a, i32 b, i32 c, i32 d>
__forceinline Lane4F32 ShuffleReverse(const Lane4F32 &l)
{
    return Shuffle<d, c, b, a>(l);
}

template <i32 i>
__forceinline Lane4F32 Shuffle(const Lane4F32 &a)
{
    return Shuffle<i, i, i, i>(a);
}

template <i32 a, i32 b, i32 c, i32 d>
__forceinline Lane4F32 Shuffle(const Lane4F32 &l, const Lane4F32 &r)
{
    return _mm_shuffle_ps(l, r, _MM_SHUFFLE(d, c, b, a));
}

template <i32 i>
__forceinline f32 Extract(const Lane4F32 &a)
{
    return _mm_cvtss_f32(Shuffle<i>(a));
}

template <>
__forceinline f32 Extract<0>(const Lane4F32 &a)
{
    return _mm_cvtss_f32(a);
}

template <i32 m>
__forceinline Lane4F32 Blend(const Lane4F32 &a, const Lane4F32 &b)
{
    return _mm_blend_ps(a, b, m);
}

__forceinline Lane4F32 Permute(const Lane4F32 &a, const Lane4U32 &b)
{
#if defined(__AVX__)
    return _mm_permutevar_ps(a, b);
#elif defined(__SSSE3__)
    const u32 MUL = 0x04040404;
    const u32 ADD = 0x03020100;

    const __m128i i0 = _mm_cvtsi32_si128((b[0] & 3) * MUL + ADD);
    const __m128i i1 = _mm_cvtsi32_si128((b[1] & 3) * MUL + ADD);
    const __m128i i2 = _mm_cvtsi32_si128((b[2] & 3) * MUL + ADD);
    const __m128i i3 = _mm_cvtsi32_si128((b[3] & 3) * MUL + ADD);

    __m128i permutation =
        _mm_unpacklo_epi64(_mm_unpacklo_epi32(i0, i1), _mm_unpacklo_epi32(i2, i3));
    return _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(a), permutation));
#else
#error TODO
#endif
}

__forceinline u32 TruncateToU8(const Lane4F32 &lane)
{
#ifdef __SSE4_1__
    __m128i m  = _mm_cvtps_epi32(lane);
    m          = _mm_packus_epi32(m, m);
    m          = _mm_packus_epi16(m, m);
    i32 result = _mm_cvtsi128_si32(m);
    return *(u32 *)(&result);
#elif __SSSE3__
    __m128i m                 = _mm_cvtps_epi32(lane);
    static const __m128i mask = _mm_setr_epi8(
        (u8)0, (u8)4, (u8)8, (u8)12, (u8)0x80, (u8)0x80, (u8)0x80, (u8)0x80, (u8)0x80,
        (u8)0x80, (u8)0x80, (u8)0x80, (u8)0x80, (u8)0x80, (u8)0x80, (u8)0x80);
    i32 result = _mm_cvvtsi128_si32(_mm_shuffle_epi8(m, mask));
    return *(u32 *)(&result);

#else
    u32 result = 0;
    for (i32 i = 0; i < 4; i++)
    {
        result |= (u32((u8)lane[i]) << (i * 8));
    }
    return result;
#endif
}

__forceinline Lane4F32 ReduceMinV(const Lane4F32 &l)
{
    Lane4F32 a = Min(l, Shuffle<1, 0, 3, 2>(l));
    return Min(a, Shuffle<2, 3, 0, 1>(a));
}

__forceinline f32 ReduceMin(const Lane4F32 &l) { return _mm_cvtss_f32(ReduceMinV(l)); }

__forceinline void TruncateToU8(u8 *out, const Lane4F32 &lane)
{
    u32 result  = TruncateToU8(lane);
    *(u32 *)out = result;
}

#ifdef __SSE4_1__
__forceinline Lane4F32 Floor(const Lane4F32 &lane)
{
    return _mm_round_ps(lane, _MM_FROUND_TO_NEG_INF);
}
__forceinline Lane4F32 Ceil(const Lane4F32 &lane)
{
    return _mm_round_ps(lane, _MM_FROUND_TO_POS_INF);
}

#else
__forceinline Lane4F32 Floor(const Lane4F32 &lane)
{
    return Lane4F32(floorf(lane[0]), floorf(lane[1]), floorf(lane[2]), floorf(lane[3]));
}
__forceinline Lane4F32 Ceil(const Lane4F32 &lane)
{
    return Lane4F32(ceilf(lane[0]), ceilf(lane[1]), ceilf(lane[2]), ceilf(lane[3]));
}
#endif

__forceinline Lane4U32 Flooru(Lane4F32 lane) { return _mm_cvtps_epi32(Floor(lane)); }

__forceinline Lane4F32 AsFloat(Lane4U32 lane) { return _mm_castsi128_ps(lane); }

#if defined(__AVX512VL__)
template <i32 R>
__forceinline Lane4F32 Rotate(const Lane4F32 &a)
{
    if constexpr (R % 4 == 0)
    {
        return a;
    }
    else if constexpr (R < 0)
    {
        return _mm_castsi128_ps(
            _mm_alignr_epi32(_mm_castps_si128(a), _mm_castps_si128(a), (-R) % 4));
    }
    else
    {
        return _mm_castsi128_ps(
            _mm_alignr_epi32(_mm_castps_si128(a), _mm_castps_si128(a), (4 - (R % 4)) % 4));
    }
}
template <i32 S, typename T>
__forceinline Lane4F32 ShiftUp(const Lane4F32 &a)
{
    StaticAssert(S >= 0 && S <= 4, ShiftMustBeBetween0And4);
    return _mm_castsi128_ps(_mm_alignr_epi32(Lane4F32(T()), _mm_castps_si128(a), 4 - S));
}

template <i32 S, typename T>
__forceinline Lane4F32 ShiftDown(const Lane4F32 &a)
{
    StaticAssert(S >= 0 && S <= 4, ShiftMustBeBetween0And4);
    return _mm_castsi128_ps(_mm_alignr_epi32(_mm_castps_si128(a), Lane4F32(T()), S));
}

#else
template <i32 R>
__forceinline Lane4F32 Rotate(const Lane4F32 &a)
{
    if constexpr (R % 4 == 0)
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
__forceinline Lane4F32 ShiftUp(const Lane4F32 &a)
{
    StaticAssert(S >= 0 && S <= 4, ShiftMustBeBetween0And4);
    constexpr u32 A        = S > 0 ? 0 : 0xffffffff;
    constexpr u32 B        = S > 1 ? 0 : 0xffffffff;
    constexpr u32 C        = S > 2 ? 0 : 0xffffffff;
    constexpr u32 D        = S > 3 ? 0 : 0xffffffff;
    const __m128 shiftMask = _mm_castsi128_ps(Lane4U32(A, B, C, D));

    return Select(shiftMask, Rotate<S>(a), Lane4F32(T()));
}

template <i32 S, typename T>
__forceinline Lane4F32 ShiftDown(const Lane4F32 &a)
{
    StaticAssert(S >= 0 && S <= 4, ShiftMustBeBetween0And4);

    constexpr u32 A = S > 3 ? 0 : 0xffffffff;
    constexpr u32 B = S > 2 ? 0 : 0xffffffff;
    constexpr u32 C = S > 1 ? 0 : 0xffffffff;
    constexpr u32 D = S > 0 ? 0 : 0xffffffff;

    const __m128 shiftMask = _mm_castsi128_ps(Lane4U32(A, B, C, D));
    return Select(shiftMask, Rotate<-S>(a), Lane4F32(T()));
}
#endif

template <i32 S>
__forceinline Lane4F32 ShiftUp(const Lane4F32 &a)
{
    return ShiftUp<S, ZeroTy>(a);
}

template <i32 S>
__forceinline Lane4F32 ShiftDown(const Lane4F32 &a)
{
    return ShiftDown<S, ZeroTy>(a);
}

// NOTE: the end of the output contains garbage
__forceinline Lane4F32 Compact(const u32 mask, const Lane4F32 &l)
{
#if defined(__AVX512VL__)
    return _mm_mask_compress_ps(_mm_setzero_ps(), Movemask(mask), l);
#elif defined(__AVX__)
    const u32 bitMask   = mask * 4;
    const u32 permute01 = (0x498E4DC349824100ull >> bitMask) & 0xf;
    const u32 permute23 = (0xE980300020000000ull >> bitMask) & 0xf;
    return _mm_permutevar_ps(l, _mm_srlv_epi32(_mm_set1_epi32((permute01 | (permute23 << 4))),
                                               _mm_setr_epi32(0, 2, 4, 6)));
#elif defined(__SSSE3__)
#error TODO
#endif
}

__forceinline void Transpose3x3(const Lane4F32 &a, const Lane4F32 &b, const Lane4F32 &c,
                                Lane4F32 &out1, Lane4F32 &out2, Lane4F32 &out3)
{
    Lane4F32 acXY = UnpackLo(a, c);
    Lane4F32 bcXY = UnpackLo(b, c);

    out1 = UnpackLo(acXY, b);
    out2 = UnpackHi(acXY, bcXY);
    out3 = Shuffle<0, 1, 2, 0>(UnpackHi(a, b), c);
}

__forceinline void Transpose4x3(const Lane4F32 &a, const Lane4F32 &b, const Lane4F32 &c,
                                const Lane4F32 &d, Lane4F32 &out1, Lane4F32 &out2,
                                Lane4F32 &out3)
{
    Lane4F32 acXY = UnpackLo(a, c);
    Lane4F32 acZ  = UnpackHi(a, c);
    Lane4F32 bdXY = UnpackLo(b, d);
    Lane4F32 bdZ  = UnpackHi(b, d);

    out1 = UnpackLo(acXY, bdXY);
    out2 = UnpackHi(acXY, bdXY);
    out3 = UnpackLo(acZ, bdZ);
}

__forceinline void Transpose4x4(const Lane4F32 &a, const Lane4F32 &b, const Lane4F32 &c,
                                const Lane4F32 &d, Lane4F32 &out1, Lane4F32 &out2,
                                Lane4F32 &out3, Lane4F32 &out4)
{
}

__forceinline void Transpose3x4(const Lane4F32 &a, const Lane4F32 &b, const Lane4F32 &c,
                                Lane4F32 &out0, Lane4F32 &out1, Lane4F32 &out2, Lane4F32 &out3)
{
    Lane4F32 t0 = UnpackLo(a, c);
    Lane4F32 t1 = UnpackHi(a, c);
    Lane4F32 t2 = Shuffle<2, 2, 1, 1>(b);
    Lane4F32 t3 = Shuffle<3>(b);
    out0        = UnpackLo(t0, b);
    out1        = UnpackHi(t0, t2);
    out2        = UnpackLo(t1, t2);
    out3        = UnpackHi(t1, t3);
}

f32 &Set(Lane4F32 &val, u32 index)
{
    Assert(index < 4);
    return val[index];
}

f32 Get(const Lane4F32 &val, u32 index)
{
    Assert(index < 4);
    return val[index];
}

} // namespace rt

#endif
