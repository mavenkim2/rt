#ifndef VEC_4_H
#define VEC_4_H

namespace rt
{
template <typename T>
struct Vec4
{
    union
    {
        T e[4];
        struct
        {
            T x, y, z, w;
        };
        struct
        {
            Vec3<T> xyz;
            T w;
        };
    };

    Vec4() : e{zero, zero, zero, zero} {}
    Vec4(T e0, T e1, T e2, T e3) : e{e0, e1, e2, e3} {}
    Vec4(const Vec4<T> &other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        w = other.w;
    }
    template <typename T1>
    Vec4(const Vec4<T1> &other) : x(T(other.x)), y(T(other.y)), z(T(other.z)) {}
    template <typename T1>
    Vec4 &operator=(const Vec4<T1> &other)
    {
        x = T(other.x);
        y = T(other.y);
        z = T(other.z);
        return *this;
    }
    T &operator[](i32 i)
    {
        Assert(i < 4);
        return e[i];
    }
    const T &operator[](i32 i) const
    {
        Assert(i < 4);
        return e[i];
    }
};

template <typename T>
__forceinline Vec4<T> operator-(const Vec4<T> &v) { return Vec4<T>(-v.x, -v.y, -v.z, -v.w); }

template <typename T>
__forceinline Vec4<T> operator+(const Vec4<T> &u, const Vec4<T> &v)
{
    return Vec4<T>(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w);
}

template <typename T>
__forceinline Vec4<T> &operator+=(Vec4<T> &a, const Vec4<T> &b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

template <typename T>
__forceinline Vec4<T> operator-(const Vec4<T> &u, const Vec4<T> &v)
{
    return Vec4<T>(u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w);
}

template <typename T>
__forceinline Vec4<T> &operator-=(Vec4<T> &a, const Vec4<T> &b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}

template <typename T>
__forceinline Vec4<T> operator*(const Vec4<T> &u, const Vec4<T> &v)
{
    return Vec4<T>(u.x * v.x, u.y * v.y, u.z * v.z, u.w * v.w);
}

template <typename T>
__forceinline Vec4<T> &operator*=(Vec4<T> &a, const T b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    return a;
}

template <typename T>
__forceinline Vec4<T> operator*(const Vec4<T> &u, T d)
{
    return Vec4<T>(u.x * d, u.y * d, u.z * d, u.w * d);
}

template <typename T>
__forceinline Vec4<T> operator*(T d, const Vec4<T> &u)
{
    return Vec4<T>(d * u.x, d * u.y, d * u.z, d * u.w);
}

template <typename T>
__forceinline Vec4<T> operator/(const Vec4<T> &v, T d)
{
    return v * (1 / d);
}

template <typename T>
__forceinline Vec4<T> &operator*=(Vec4<T> &a, const Vec4<T> &b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    return a;
}

template <typename T>
__forceinline Vec4<T> &operator/=(Vec4<T> &a, const T b)
{
    return a * (1 / b);
}

template <typename T>
__forceinline Vec4<T> Sqrt(const Vec4<T> &a)
{
    return Vec4<T>(sqrt(a.x), sqrt(a.y), sqrt(a.z), sqrt(a.w));
}

template <typename T>
__forceinline T Dot(const Vec4<T> &a, const Vec4<T> &b)
{
    return FMA(a.x, b.x, FMA(a.y, b.y, FMA(a.z, b.z, a.w * b.w)));
}

template <typename T>
__forceinline T AbsDot(const Vec4<T> &u, const Vec4<T> &v)
{
    return abs(Dot(u, v));
}

template <typename T>
__forceinline T Sqr(const Vec4<T> &a)
{
    return Dot(a, a);
}

template <typename T>
__forceinline T LengthSquared(const Vec4<T> &a)
{
    return Sqr(a);
}

template <typename T>
__forceinline Vec4<T> Rsqrt(const Vec4<T> &a)
{
    return Vec4<T>(Rsqrt(a.x), Rsqrt(a.y), Rsqrt(a.z), Rsqrt(a.w));
}

template <typename T>
__forceinline T Length(const Vec4<T> &a)
{
    return Sqrt(Sqr(a));
}

template <typename T>
__forceinline Vec4<T> Normalize(const Vec4<T> a)
{
    return a * Rsqrt(Sqr(a));
}

template <typename T>
__forceinline Vec4<T> Min(const Vec4<T> &a, const Vec4<T> &b)
{
    return Vec4<T>(Min(a.x, b.x), Min(a.y, b.y), Min(a.z, b.z), Min(a.w, b.w));
}

template <typename T>
__forceinline Vec4<T> Max(const Vec4<T> &a, const Vec4<T> &b)
{
    return Vec4<T>(Max(a.x, b.x), Max(a.y, b.y), Max(a.z, b.z), Min(a.w, b.w));
}

template <typename T>
inline Vec4<T> ClampZero(const Vec4<T> &v)
{
    return Vec4<T>(Max(zero, v.x), Max(zero, v.y), Max(zero, v.z), Max(zero, v.w));
}

typedef Vec4<f32> Vec4f;
template <i32 K>
using LaneVec4f = Vec4<LaneF32<K>>;
typedef LaneVec4f<4> Lane4Vec4f;

} // namespace rt
#endif