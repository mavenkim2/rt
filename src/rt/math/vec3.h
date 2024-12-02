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
            Vec2<T> xy;
            T z_;
        };
        struct
        {
            T x_;
            Vec2<T> yz;
        };
        T e[3];
    };
    // TODO: see if this breaks anything
    __forceinline Vec3() : x(zero), y(zero), z(zero) {}
    __forceinline Vec3(T e0) : x(e0), y(e0), z(e0) {}
    __forceinline Vec3(T e0, T e1, T e2) : x(e0), y(e1), z(e2) {}
    __forceinline Vec3(Vec2<T> v, T e2) : x(v.x), y(v.y), z(e2) {}

    __forceinline Vec3(const Vec3<T> &other) : x(other.x), y(other.y), z(other.z) {}

    template <typename T1>
    __forceinline Vec3(const Vec3<T1> &other) : x(T(other.x)), y(T(other.y)), z(T(other.z)) {}

    __forceinline Vec3(PosInfTy) : x(pos_inf), y(pos_inf), z(pos_inf) {}
    __forceinline Vec3(NegInfTy) : x(neg_inf), y(neg_inf), z(neg_inf) {}

    __forceinline Vec3 &operator=(const Vec3<T> &other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    }
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
    __forceinline explicit operator Lane4F32() const
    {
        return Lane4F32(x, y, z, 0.f);
    }
};

template <typename T>
__forceinline bool operator==(const Vec3<T> &a, const Vec3<T> &b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

template <typename T>
__forceinline bool operator!=(const Vec3<T> &a, const Vec3<T> &b) { return a.x != b.x || a.y != b.y || a.z != b.z; }

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
__forceinline Vec3<T> operator/(const Vec3<T> &a, const Vec3<T> &b)
{
    return Vec3<T>(a.x / b.x, a.y / b.y, a.z / b.z);
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
__forceinline Vec3<T> operator>(const Vec3<T> &a, const Vec3<T> &b)
{
    return Vec3<T>(a.x > b.x, a.y > b.y, a.z > b.z);
}

template <typename T>
__forceinline Vec3<T> operator<(const Vec3<T> &a, const Vec3<T> &b)
{
    return Vec3<T>(a.x < b.x, a.y < b.y, a.z < b.z);
}

template <typename T>
__forceinline Vec3<T> Select(const Mask<T> &mask, const Vec3<T> &a, const Vec3<T> &b)
{
    return Vec3<T>(Select(mask, a.x, b.x), Select(mask, a.y, b.y), Select(mask, a.z, b.z));
}

template <typename T>
__forceinline Vec3<T> Select(const Vec3<T> &mask, const Vec3<T> &a, const Vec3<T> &b)
{
    return Vec3<T>(Select(mask.x, a.x, b.x), Select(mask.y, a.y, b.y), Select(mask.z, a.z, b.z));
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
__forceinline Vec3<T> Floor(const Vec3<T> &v)
{
    return Vec3<T>(Floor(v.x), Floor(v.y), Floor(v.z));
}

template <typename T>
__forceinline Vec3<T> Ceil(const Vec3<T> &v)
{
    return Vec3<T>(Ceil(v.x), Ceil(v.y), Ceil(v.z));
}

template <typename T>
__forceinline Vec3<T> FMA(const Vec3<T> &a, const Vec3<T> &b, const Vec3<T> &c)
{
    return Vec3<T>(FMA(a.x, b.x, c.x), FMA(a.y, b.y, c.y), FMA(a.z, b.z, c.z));
}
template <typename T>
__forceinline Vec3<T> FMS(const Vec3<T> &a, const Vec3<T> &b, const Vec3<T> &c)
{
    return Vec3<T>(FMS(a.x, b.x, c.x), FMS(a.y, b.y, c.y), FMS(a.z, b.z, c.z));
}

template <typename T>
inline Vec3<T> ClampZero(const Vec3<T> &v)
{
    return Vec3<T>(Max(T(zero), v.x), Max(T(zero), v.y), Max(T(zero), v.z));
}

typedef Vec3<f32> Vec3f;
typedef Vec3<u32> Vec3u;
typedef Vec3<i32> Vec3i;

template <i32 K>
using Vec3lf = Vec3<LaneF32<K>>;
typedef Vec3lf<4> Vec3lf4;
typedef Vec3lf<8> Vec3lf8;

template <i32 K>
using Vec3lu = Vec3<LaneU32<K>>;
typedef Vec3lu<4> Vec3lu4;
typedef Vec3lu<8> Vec3lu8;

typedef Vec3<LaneNF32> Vec3lfn;

template <template <i32> class T, i32 K>
struct SOASetVec3
{
    Vec3<T<K>> &v;
    u32 index;

    SOASetVec3(Vec3<T<K>> &v, u32 index) : v(v), index(index) {}

    __forceinline SOASetVec3 &operator=(const Vec3f &r)
    {
        Assert(index < K);
        Set(v[0], index) = r[0];
        Set(v[1], index) = r[1];
        Set(v[2], index) = r[2];
        return *this;
    }
};

template <template <i32> class T, i32 K>
__forceinline SOASetVec3<T, K> Set(Vec3<T<K>> &v, u32 index)
{
    return SOASetVec3<T, K>(v, index);
}

template <i32 K>
__forceinline Vec3lu<K> Flooru(const Vec3lf<K> &v)
{
    return Vec3lu<K>(Flooru(v.x), Flooru(v.y), Flooru(v.z));
}

__forceinline Vec3f ToVec3f(const Lane4F32 &l)
{
    return Vec3f(l[0], l[1], l[2]);
}

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
