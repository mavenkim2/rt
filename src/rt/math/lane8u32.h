#ifndef LANE8U32_H
#define LANE8U32_H

namespace rt
{
template <>
struct LaneU32_<8>
{
    union
    {
        __m256i v;
        u32 f[8];
    };
    __forceinline LaneU32_() {}
    __forceinline LaneU32_(__m256i v) : v(v) {}
    __forceinline LaneU32_(const Lane8U32 &other) { v = other.v; }
    __forceinline LaneU32_(u32 a) { v = _mm256_set1_epi32(a); }
    __forceinline LaneU32_(u32 a, u32 b, u32 c, u32 d, u32 e, u32 f, u32 g, u32 h) { v = _mm256_setr_epi32(a, b, c, d, e, f, g, h); }

    __forceinline LaneU32_(ZeroTy) { v = _mm256_setzero_si256(); }
    __forceinline LaneU32_(PosInfTy) { v = _mm256_set1_epi32(pos_inf); }
    __forceinline LaneU32_(NegInfTy) { v = _mm256_set1_epi32(neg_inf); }

    __forceinline operator const __m256i &() const
    {
        return v;
    }
    __forceinline operator __m256i &() { return v; }
    __forceinline Lane8U32 &operator=(const Lane8U32 &other)
    {
        v = other.v;
        return *this;
    }

    __forceinline const u32 &operator[](i32 i) const
    {
        Assert(i < 8);
        return f[i];
    }
    __forceinline u32 &operator[](i32 i)
    {
        Assert(i < 8);
        return f[i];
    }
    static __forceinline Lane8U32 Load(const void *ptr) { return _mm256_load_si256((const __m256i *)ptr); }
    static __forceinline Lane8U32 LoadU(const void *ptr) { return _mm256_loadu_si256((const __m256i *)ptr); }
    static __forceinline void Store(void *ptr, const Lane8U32 &l) { _mm256_store_si256((__m256i *)ptr, l); }
    static __forceinline void StoreU(void *ptr, const Lane8U32 &l) { _mm256_storeu_si256((__m256i *)ptr, l); }
    static __forceinline void Store(const __m256 &mask, void *ptr, const Lane8U32 &l)
    {
#ifdef __AVX512VL__
        _mm256_mask_store_epi32((__m256i)ptr, (__mmask8)Movemask(mask), l);
#else

        _mm256_store_si256((__m256i *)ptr, _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(_mm256_load_si256((__m256i *)ptr)),
                                                                                _mm256_castsi256_ps(l),
                                                                                mask)));
#endif
    }
    static __forceinline void StoreU(const __m256 &mask, void *ptr, const Lane8U32 &l)
    {
#ifdef __AVX512VL__
        _mm256_mask_storeu_epi32((__m256i)ptr, (__mmask8)Movemask(mask), l);
#else

        _mm256_storeu_si256((__m256i *)ptr, _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(_mm256_loadu_si256((__m256i *)ptr)),
                                                                                 _mm256_castsi256_ps(l),
                                                                                 mask)));
#endif
    }

    static __forceinline Lane8U32 Step(u32 start)
    {
        return LaneU32_(start + 0, start + 1, start + 2, start + 3, start + 4, start + 5, start + 6, start + 7);
    }
};

__forceinline Lane8U32 UnpackLo(const Lane8U32 &a, const Lane8U32 &b) { return _mm256_unpacklo_epi32(a, b); }
__forceinline Lane8U32 UnpackHi(const Lane8U32 &a, const Lane8U32 &b) { return _mm256_unpackhi_epi32(a, b); }

template <i32 i>
u32 Extract(const Lane8U32 &l)
{
    i32 result = _mm256_extract_epi32(l, i);
    return *(u32 *)(&result);
}

__forceinline Lane8U32 operator+(const Lane8U32 &a, const Lane8U32 &b) { return _mm256_add_epi32(a, b); }
__forceinline Lane8U32 operator+(const Lane8U32 &a, const u32 b) { return a + Lane8U32(b); }
__forceinline Lane8U32 operator+(const u32 a, const Lane8U32 &b) { return Lane8U32(a) + b; }
__forceinline Lane8U32 &operator+=(Lane8U32 &a, const Lane8U32 &b)
{
    a = a + b;
    return a;
}

__forceinline Lane8U32 operator-(const Lane8U32 &a, const Lane8U32 &b) { return _mm256_sub_epi32(a, b); }
__forceinline Lane8U32 operator-(const Lane8U32 &a, const u32 b) { return a - Lane8U32(b); }
__forceinline Lane8U32 operator-(const u32 a, const Lane8U32 &b) { return Lane8U32(a) - b; }

__forceinline Lane8U32 operator^(const Lane8U32 &a, const Lane8U32 &b)
{
    return _mm256_xor_si256(a, b);
}
__forceinline Lane8U32 operator^(const Lane8U32 &a, const u32 b)
{
    return a ^ Lane8U32(b);
}
__forceinline Lane8U32 operator^(const u32 a, const Lane8U32 &b)
{
    return Lane8U32(a) ^ b;
}
__forceinline Lane8U32 &operator^=(Lane8U32 &a, const Lane8U32 &b)
{
    a = a ^ b;
    return a;
}

__forceinline Lane8U32 operator&(const Lane8U32 &a, const Lane8U32 &b)
{
    return _mm256_and_si256(a, b);
}
__forceinline Lane8U32 operator&(const Lane8U32 &a, u32 inMask)
{
    return a & Lane8U32(inMask);
}
__forceinline Lane8U32 operator&(u32 inMask, const Lane8U32 &a)
{
    return Lane8U32(inMask) & a;
}
__forceinline Lane8U32 &operator&=(Lane8U32 &lane, const u32 val)
{
    lane = lane & Lane8U32(val);
    return lane;
}
__forceinline Lane8U32 operator|(const Lane8U32 &a, const Lane8U32 &b) { return _mm256_or_si256(a, b); }
__forceinline Lane8U32 operator<<(const Lane8U32 &a, const u32 inShift) { return _mm256_slli_epi32(a, inShift); }
__forceinline Lane8U32 operator>>(const Lane8U32 &a, const u32 inShift)
{
    return _mm256_srli_epi32(a, inShift);
}

#if defined(__AVX2__)
__forceinline Lane8U32 operator<<(const Lane8U32 &a, const Lane8U32 &b)
{
    return _mm256_sllv_epi32(a, b);
}
__forceinline Lane8U32 operator>>(const Lane8U32 &a, const Lane8U32 &b) { return _mm256_srlv_epi32(a, b); }
#else
__forceinline Lane8U32 operator<<(const Lane8U32 &a, const Lane8U32 &b)
{
}
#endif

__forceinline Lane8U32 Min(const Lane8U32 &a, const Lane8U32 &b)
{
#if defined(__SSE4_1__)
    return _mm256_min_epu32(a, b);
#else
#error TODO
#endif
}
__forceinline Lane8U32 Max(const Lane8U32 &a, const Lane8U32 &b)
{
#if defined(__SSE4_1__)
    return _mm256_max_epu32(a, b);
#else
#error TODO
#endif
}

__forceinline Lane8U32 PackU32(const Lane8U32 &a, const Lane8U32 &b) { return _mm256_packus_epi32(a, b); }
__forceinline Lane8U32 PackU16(const Lane8U32 &a, const Lane8U32 &b) { return _mm256_packus_epi16(a, b); }

} // namespace rt
#endif
