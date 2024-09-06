#ifndef LANE4U32_H
#define LANE4U32_H

namespace rt
{
template <>
struct LaneU32<4>
{
    union
    {
        __m128i v;
        u32 f[4];
    };
    __forceinline LaneU32() {}
    __forceinline LaneU32(__m128i v) : v(v) {}
    __forceinline LaneU32(const Lane4U32 &other) { v = other.v; }
    __forceinline LaneU32(u32 a) { v = _mm_set1_epi32(a); }
    __forceinline LaneU32(u32 a, u32 b, u32 c, u32 d) { v = _mm_setr_epi32(a, b, c, d); }

    __forceinline LaneU32(ZeroTy) { v = _mm_setzero_si128(); }
    __forceinline LaneU32(PosInfTy) { v = _mm_set1_epi32(pos_inf); }
    __forceinline LaneU32(NegInfTy) { v = _mm_set1_epi32(neg_inf); }

    __forceinline operator const __m128i &() const
    {
        return v;
    }
    __forceinline operator __m128i &() { return v; }
    __forceinline Lane4U32 &operator=(const Lane4U32 &other)
    {
        v = other.v;
        return *this;
    }

    __forceinline const u32 &operator[](i32 i) const
    {
        Assert(i < 4);
        return f[i];
    }
    __forceinline u32 &operator[](i32 i)
    {
        Assert(i < 4);
        return f[i];
    }
    static __forceinline Lane4U32 Load(const void *ptr) { return _mm_load_si128((const __m128i *)ptr); }
    static __forceinline Lane4U32 LoadU(const void *ptr) { return _mm_loadu_si128((const __m128i *)ptr); }
    static __forceinline void Store(void *ptr, const Lane4U32 &l) { _mm_store_si128((__m128i *)ptr, l); }
};

__forceinline Lane4U32 UnpackLo(const Lane4U32 &a, const Lane4U32 &b) { return _mm_unpacklo_epi32(a, b); }
__forceinline Lane4U32 UnpackHi(const Lane4U32 &a, const Lane4U32 &b) { return _mm_unpackhi_epi32(a, b); }

template <i32 i>
u32 Extract(const Lane4U32 &l)
{
    i32 result = _mm_extract_epi32(l, i);
    return *(u32 *)(&result);
}

__forceinline Lane4U32 operator+(const Lane4U32 &a, const Lane4U32 &b) { return _mm_add_epi32(a, b); }
__forceinline Lane4U32 operator+(const Lane4U32 &a, const u32 b) { return a + Lane4U32(b); }
__forceinline Lane4U32 operator+(const u32 a, const Lane4U32 &b) { return Lane4U32(a) + b; }

__forceinline Lane4U32 operator-(const Lane4U32 &a, const Lane4U32 &b) { return _mm_sub_epi32(a, b); }
__forceinline Lane4U32 operator-(const Lane4U32 &a, const u32 b) { return a - Lane4U32(b); }
__forceinline Lane4U32 operator-(const u32 a, const Lane4U32 &b) { return Lane4U32(a) - b; }

__forceinline Lane4U32 operator^(const Lane4U32 &a, const Lane4U32 &b)
{
    return _mm_xor_si128(a, b);
}
__forceinline Lane4U32 operator^(const Lane4U32 &a, const u32 b)
{
    return a ^ Lane4U32(b);
}
__forceinline Lane4U32 operator^(const u32 a, const Lane4U32 &b)
{
    return Lane4U32(a) ^ b;
}
__forceinline Lane4U32 &operator^=(Lane4U32 &a, const Lane4U32 &b)
{
    a = a ^ b;
    return a;
}

__forceinline Lane4U32 operator&(const Lane4U32 &a, const Lane4U32 &b)
{
    return _mm_and_si128(a, b);
}
__forceinline Lane4U32 operator&(const Lane4U32 &a, u32 inMask)
{
    return a & Lane4U32(inMask);
}
__forceinline Lane4U32 operator&(u32 inMask, const Lane4U32 &a)
{
    return Lane4U32(inMask) & a;
}
__forceinline Lane4U32 &operator&=(Lane4U32 &lane, const u32 val)
{
    lane = lane & Lane4U32(val);
    return lane;
}
__forceinline Lane4U32 operator|(const Lane4U32 &a, const Lane4U32 &b) { return _mm_or_si128(a, b); }
__forceinline Lane4U32 operator<<(const Lane4U32 &a, const u32 inShift) { return _mm_slli_epi32(a, inShift); }
__forceinline Lane4U32 operator>>(const Lane4U32 &a, const u32 inShift) { return _mm_srli_epi32(a, inShift); }
__forceinline Lane4U32 SignExtendU8ToU32(const Lane4U32 &l)
{
#ifdef __SSE4_1__
    return _mm_cvtepu8_epi32(l);
#elif __SSE3__
    static const __m128i mask = _mm_setr_epi8((u8)0, (u8)0xff, (u8)0xff, (u8)0xff,
                                              (u8)1, (u8)0xff, (u8)0xff, (u8)0xff,
                                              (u8)2, (u8)0xff, (u8)0xff, (u8)0xff,
                                              (u8)3, (u8)0xff, (u8)0xff, (u8)0xff);
    return _mm_shuffle_epi8(l, mask);
#else
    return _mm_setr_epi32(l[0] & 0xf, (l[0] >> 8) & 0xf, (l[0] >> 16) & 0xf, (l[0] >> 24) & 0xf);
#endif
}

} // namespace rt
#endif
