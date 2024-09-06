#ifndef LANE4F32_H
#define LANE4F32_H

namespace rt
{

template <>
struct LaneF32<4>
{
    union
    {
        __m128 v;
        f32 f[4];
    };
    __forceinline LaneF32() {}
    __forceinline LaneF32(__m128 v) : v(v) {}
    __forceinline LaneF32(const Lane4F32 &other) { v = other.v; }
    __forceinline LaneF32(f32 a) { v = _mm_set1_ps(a); }
    __forceinline LaneF32(f32 a, f32 b, f32 c, f32 d) { v = _mm_setr_ps(a, b, c, d); }

    // NOTE: necessary since otherwise it would be interpreted as an integer
    __forceinline explicit LaneF32(const Lane4U32 &l)
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

    __forceinline LaneF32(ZeroTy) { v = _mm_setzero_ps(); }
    __forceinline LaneF32(PosInfTy) { v = _mm_set1_ps(pos_inf); }
    __forceinline LaneF32(NegInfTy) { v = _mm_set1_ps(neg_inf); }

    template <i32 i1>
    __forceinline static LaneF32 Mask()
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

    __forceinline static LaneF32 Mask(bool i)
    {
        return _mm_lookupmask_ps[((size_t)i << 3) | ((size_t)i << 2) | ((size_t)i << 1) | ((size_t)i)];
    }

    __forceinline LaneF32(TrueTy) { v = _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps()); }
    __forceinline LaneF32(FalseTy) { v = _mm_setzero_ps(); }

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
    static __forceinline void Store(void *ptr, const Lane4F32 &l) { _mm_store_ps((f32 *)ptr, l); }
};

__forceinline Lane4F32 operator+(const Lane4F32 &a, const Lane4F32 &b) { return _mm_add_ps(a, b); }
__forceinline Lane4F32 operator+(f32 a, const Lane4F32 &b) { return Lane4F32(a) + b; }
__forceinline Lane4F32 operator+(const Lane4F32 &a, f32 b) { return a + Lane4F32(b); }

__forceinline Lane4F32 operator-(const Lane4F32 &a, const Lane4F32 &b) { return _mm_sub_ps(a, b); }
__forceinline Lane4F32 operator-(f32 a, const Lane4F32 &b) { return Lane4F32(a) - b; }
__forceinline Lane4F32 operator-(const Lane4F32 &a, f32 b) { return a - Lane4F32(b); }

__forceinline Lane4F32 operator*(const Lane4F32 &a, const Lane4F32 &b) { return _mm_mul_ps(a, b); }
__forceinline Lane4F32 operator*(f32 a, const Lane4F32 &b) { return Lane4F32(a) * b; }
__forceinline Lane4F32 operator*(const Lane4F32 &a, f32 b) { return a * Lane4F32(b); }

__forceinline Lane4F32 operator/(const Lane4F32 &a, const Lane4F32 &b) { return _mm_div_ps(a, b); }
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

__forceinline Lane4F32 Sqrt(const Lane4F32 &a) { return _mm_sqrt_ps(a); }
__forceinline Lane4F32 Rsqrt(const Lane4F32 &a)
{
#ifdef __AVX512VL__
    Lane4F32 r = _mm_rsqrt14_ps(a);
#else
    Lane4F32 r = _mm_rsqrt_ps(a);
#endif
#if defined(__AVX2__)
    r = _mm_fmadd_ps(_mm_set1_ps(1.5f), r, _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(a, _mm_set1_ps(-0.5f)), r), _mm_mul_ps(r, r)));
#else
    r          = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.5f), r), _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(a, _mm_set1_ps(-0.5f)), r), _mm_mul_ps(r, r)));
#endif
    return r;
}

__forceinline Lane4F32 rcp(const Lane4F32 &a)
{
#ifdef __AVX512VL__
    Lane4F32 r = _mm_rcp14_ps(a);
#else
    Lane4F32 r = _mm_rcp_ps(a);
#endif

#if defined(__AVX2__)
    return _mm_fmadd_ps(r, _mm_fnmadd_ps(a, r, Lane4F32(1.0f)), r); // computes r + r * (1 - a * r)
#else
    return _mm_add_ps(r, _mm_mul_ps(r, _mm_sub_ps(Lane4F32(1.0f), _mm_mul_ps(a, r)))); // computes r + r * (1 - a * r)
#endif
}

__forceinline Lane4F32 Abs(const Lane4F32 &a) { return _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff))); }
__forceinline Lane4F32 Min(const Lane4F32 &a, const Lane4F32 &b) { return _mm_min_ps(a, b); }
__forceinline Lane4F32 Max(const Lane4F32 &a, const Lane4F32 &b) { return _mm_max_ps(a, b); }
__forceinline Lane4F32 operator^(const Lane4F32 &a, const Lane4F32 &b) { return _mm_xor_ps(a, b); }
__forceinline Lane4F32 operator<(const Lane4F32 &a, const Lane4F32 &b) { return _mm_cmplt_ps(a, b); }
__forceinline Lane4F32 operator<=(const Lane4F32 &a, const Lane4F32 &b) { return _mm_cmple_ps(a, b); }
__forceinline Lane4F32 operator>(const Lane4F32 &a, const Lane4F32 &b) { return _mm_cmpgt_ps(a, b); }
__forceinline Lane4F32 operator>=(const Lane4F32 &a, const Lane4F32 &b) { return _mm_cmpge_ps(a, b); }
__forceinline Lane4F32 operator==(const Lane4F32 &a, const Lane4F32 &b) { return _mm_cmpeq_ps(a, b); }
__forceinline Lane4F32 operator&(const Lane4F32 &a, const Lane4F32 &b) { return _mm_and_ps(a, b); }
__forceinline Lane4F32 &operator&=(Lane4F32 &a, const Lane4F32 &b)
{
    a = a & b;
    return a;
}
__forceinline Lane4F32 operator|(const Lane4F32 &a, const Lane4F32 &b) { return _mm_or_ps(a, b); }

__forceinline Lane4F32 Select(const Lane4F32 &a, const Lane4F32 &b, const Lane4F32 &mask)
{
#ifdef __SSE4_1__
    return _mm_blendv_ps(a, b, mask);
#else
    return _mm_or_ps(_mm_andnot_ps(mask, b), _mm_and_ps(mask, a));
#endif
}

__forceinline i32 Movemask(const Lane4F32 &a) { return _mm_movemask_ps(a); }
__forceinline bool All(const Lane4F32 &a) { return _mm_movemask_ps(a) == 0xf; }
__forceinline bool Any(const Lane4F32 &a) { return _mm_movemask_ps(a) != 0; }
__forceinline bool None(const Lane4F32 &a) { return _mm_movemask_ps(a) == 0; }

__forceinline Lane4F32 UnpackLo(const Lane4F32 &a, const Lane4F32 &b) { return _mm_unpacklo_ps(a, b); }
__forceinline Lane4F32 UnpackHi(const Lane4F32 &a, const Lane4F32 &b) { return _mm_unpackhi_ps(a, b); }

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

__forceinline u32 TruncateToU8(const Lane4F32 &lane)
{
#ifdef __SSE4_1__
    __m128i m  = _mm_cvtps_epi32(lane);
    m          = _mm_packus_epi32(m, m);
    m          = _mm_packus_epi16(m, m);
    i32 result = _mm_cvtsi128_si32(m);
    return *(u32 *)(&result);
#else
    u32 result = 0;
    for (i32 i = 0; i < 4; i++)
    {
        result |= ((u8)lane[i] << (i * 8));
    }
    return result;
#endif
}

} // namespace rt

#endif
