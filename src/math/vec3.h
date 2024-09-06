#include <cmath>
namespace rt
{
template <typename T>
struct Vec3
{
    union
    {
        struct
        {
            T x, y, z;
        };
        struct
        {
            T r, g, b;
        };
        struct
        {
            Vec2<T> xy;
            T z;
        };
        struct
        {
            T x;
            Vec2<T> yz;
        };
        T e[3];
    };
    // TODO: see if this breaks anything
    __forceinline Vec3() : x(zero), y(zero), z(zero) {}
    __forceinline Vec3(T e0, T e1, T e2) : x(e0), y(e1), z(e2) {}
    __forceinline Vec3(Vec2<T> v, T e2) : x(v.x), y(v.y), z(e2) {}

    __forceinline Vec3(const Vec3<T> &other) : x(other.x), y(other.y), z(other.z) {}

    template <typename T1>
    __forceinline Vec3(const Vec3<T1> &other) : x(T(other.x)), y(T(other.y)), z(T(other.z)) {}

    template <typename T1>
    __forceinline Vec3 &operator=(const Vec3<T1> &other)
    {
        x = T(other.x);
        y = T(other.y);
        z = T(other.z);
        return *this;
    }
    __forceinline const T &operator[](const i32 i) const
    {
        Assert(i < 3);
        return e[i];
    }
    __forceinline T &operator[](const i32 i)
    {
        Assert(i < 3);
        return e[i];
    }
};

template <typename T>
__forceinline Vec3<T> operator-(const Vec3<T> &v) { return Vec3<T>(-v.x, -v.y, -v.z); }

template <typename T>
__forceinline Vec3<T> operator+(const Vec3<T> &u, const Vec3<T> &v)
{
    return Vec3<T>(u.x + v.x, u.y + v.y, u.z + v.z);
}

template <typename T>
__forceinline Vec3<T> &operator+=(Vec3<T> &a, const Vec3<T> &b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

template <typename T>
__forceinline Vec3<T> operator-(const Vec3<T> &u, const Vec3<T> &v)
{
    return Vec3<T>(u.x - v.x, u.y - v.y, u.z - v.z);
}

template <typename T>
__forceinline Vec3<T> &operator-=(Vec3<T> &a, const Vec3<T> &b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

template <typename T>
__forceinline Vec3<T> operator*(const Vec3<T> &u, const Vec3<T> &v)
{
    return Vec3<T>(u.x * v.x, u.y * v.y, u.z * v.z);
}

template <typename T>
__forceinline Vec3<T> &operator*=(Vec3<T> &a, const T b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

template <typename T>
__forceinline Vec3<T> operator*(const Vec3<T> &u, T d)
{
    return Vec3<T>(u.x * d, u.y * d, u.z * d);
}

template <typename T>
__forceinline Vec3<T> operator*(T d, const Vec3<T> &u)
{
    return Vec3<T>(d * u.x, d * u.y, d * u.z);
}

template <typename T>
__forceinline Vec3<T> operator/(const Vec3<T> &v, T d)
{
    return Vec3<T>(v.x / d, v.y / d, v.z / d);
}

template <typename T>
__forceinline Vec3<T> &operator*=(Vec3<T> &a, const Vec3<T> &b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

template <typename T>
__forceinline Vec3<T> &operator/=(Vec3<T> &a, const T b)
{
    a = a / b;
    return a;
}

template <typename T>
__forceinline Vec3<T> Sqrt(const Vec3<T> &a)
{
    return Vec3<T>(Sqrt(a.x), Sqrt(a.y), Sqrt(a.z));
}

template <typename T>
__forceinline T Dot(const Vec3<T> &a, const Vec3<T> &b)
{
    return FMA(a.x, b.x, FMA(a.y, b.y, a.z * b.z));
}

template <typename T>
__forceinline T AbsDot(const Vec3<T> &u, const Vec3<T> &v)
{
    return Abs(Dot(u, v));
}

template <typename T>
__forceinline Vec3<T> Cross(const Vec3<T> &a, const Vec3<T> &b)
{
    return Vec3<T>(FMS(a.y, b.z, a.z * b.y), FMS(a.z, b.x, a.x * b.z), FMS(a.x, b.y, a.y * b.x));
}

template <typename T>
__forceinline T Sqr(const Vec3<T> &a)
{
    return Dot(a, a);
}

template <typename T>
__forceinline T LengthSquared(const Vec3<T> &a)
{
    return Sqr(a);
}

template <typename T>
__forceinline Vec3<T> Rsqrt(const Vec3<T> &a)
{
    return Vec3<T>(Rsqrt(a.x), Rsqrt(a.y), Rsqrt(a.z));
}

template <typename T>
__forceinline T Length(const Vec3<T> &a)
{
    return Sqrt(Sqr(a));
}

template <typename T>
__forceinline T Distance(const Vec3<T> &a, const Vec3<T> &b)
{
    return Length(a - b);
}

template <typename T>
__forceinline Vec3<T> Normalize(const Vec3<T> &a)
{
    return a * Rsqrt(Sqr(a));
}

template <typename T>
__forceinline Vec3<T> Min(const Vec3<T> &a, const Vec3<T> &b)
{
    return Vec3<T>(Min(a.x, b.x), Min(a.y, b.y), Min(a.z, b.z));
}

template <typename T>
__forceinline Vec3<T> Max(const Vec3<T> &a, const Vec3<T> &b)
{
    return Vec3<T>(Max(a.x, b.x), Max(a.y, b.y), Max(a.z, b.z));
}

template <typename T>
__forceinline Vec3<T> FMA(const Vec3<T> &a, const Vec3<T> &b, const Vec3<T> &c)
{
    return Vec3<T>(FMA(a.x, b.x, c.x), FMA(a.y, b.y, c.y), FMA(a.z, b.z, c.z));
}

template <typename T>
inline Vec3<T> ClampZero(const Vec3<T> &v)
{
    return Vec3<T>(Max(T(zero), v.x), Max(T(zero), v.y), Max(T(zero), v.z));
}

typedef Vec3<f32> Vec3f;

template <i32 K>
using LaneVec3f = Vec3<LaneF32<K>>;
typedef LaneVec3f<4> Lane4Vec3f;

// template <>
// __forceinline Vec3<Lane4F32>::Vec3(const Vec3f &a)
// {
//     x = a.x;
//     y = a.y;
//     z = a.z;
// }

inline Vec3f Reflect(const Vec3f &v, const Vec3f &norm)
{
    return v - 2 * Dot(v, norm) * norm;
}

inline Vec3f Refract(const Vec3f &uv, const Vec3f &n, f32 refractiveIndexRatio)
{
    f32 cosTheta   = Min(Dot(-uv, n), 1.f);
    Vec3f perp     = refractiveIndexRatio * (uv + cosTheta * n);
    Vec3f parallel = -Sqrt(Abs(1.f - LengthSquared(perp))) * n;
    return perp + parallel;
}
} // namespace rt
