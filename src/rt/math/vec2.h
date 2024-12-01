namespace rt
{
template <typename T>
struct Vec2
{
    union
    {
        T e[2];
        struct
        {
            T x, y;
        };
    };
    Vec2() : e{zero, zero} {}
    Vec2(T e0) : e{e0, e0} {}
    Vec2(T e0, T e1) : e{e0, e1} {}
    Vec2(const Vec2<T> &other) : x(other.x), y(other.y) {}
    template <typename T1>
    __forceinline Vec2(const Vec2<T1> &other) : x(T(other.x)), y(T(other.y)) {}

    T operator[](i32 i) const
    {
        Assert(i < 2);
        return e[i];
    }
    T &operator[](i32 i)
    {
        Assert(i < 2);
        return e[i];
    }
};

template <typename T>
__forceinline Vec2<T> operator-(const Vec2<T> &v) { return Vec2<T>(-v.x, -v.y); }

template <typename T>
__forceinline Vec2<T> operator+(const Vec2<T> &u, const Vec2<T> &v)
{
    return Vec2<T>(u.x + v.x, u.y + v.y);
}

template <typename T>
__forceinline Vec2<T> &operator+=(Vec2<T> &a, const Vec2<T> &b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

template <typename T>
__forceinline Vec2<T> operator-(const Vec2<T> &u, const Vec2<T> &v)
{
    return Vec2<T>(u.x - v.x, u.y - v.y);
}

template <typename T>
__forceinline Vec2<T> &operator-=(Vec2<T> &a, const Vec2<T> &b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

template <typename T>
__forceinline Vec2<T> operator*(const Vec2<T> &u, const Vec2<T> &v)
{
    return Vec2<T>(u.x * v.x, u.y * v.y);
}

template <typename T>
__forceinline Vec2<T> &operator*=(Vec2<T> &a, const T b)
{
    a.x *= b;
    a.y *= b;
    return a;
}

template <typename T>
__forceinline Vec2<T> operator*(const Vec2<T> &u, T d)
{
    return Vec2<T>(u.x * d, u.y * d);
}

template <typename T>
__forceinline Vec2<T> operator*(T d, const Vec2<T> &u)
{
    return Vec2<T>(d * u.x, d * u.y);
}

template <typename T>
__forceinline Vec2<T> operator/(const Vec2<T> &u, const Vec2<T> &v) { return Vec2<T>(u.x / v.x, u.y / v.y); }

template <typename T>
__forceinline Vec2<T> operator/(const Vec2<T> &v, T d)
{
    return v * (1 / d);
}

template <typename T>
__forceinline Vec2<T> &operator*=(Vec2<T> &a, const Vec2<T> &b)
{
    a.x *= b.x;
    a.y *= b.y;
    return a;
}

template <typename T>
__forceinline Vec2<T> &operator/=(Vec2<T> &a, const T b)
{
    return a * (1 / b);
}

template <typename T>
__forceinline Vec2<T> Sqrt(const Vec2<T> &a)
{
    return Vec2<T>(sqrt(a.x), sqrt(a.y));
}

template <typename T>
__forceinline T Dot(const Vec2<T> &a, const Vec2<T> &b)
{
    return FMA(a.x, b.x, a.y * b.y);
}

template <typename T>
__forceinline T AbsDot(const Vec2<T> &u, const Vec2<T> &v)
{
    return Abs(Dot(u, v));
}

template <typename T>
__forceinline T Sqr(const Vec2<T> &a)
{
    return Dot(a, a);
}

template <typename T>
__forceinline T LengthSquared(const Vec2<T> &a)
{
    return Sqr(a);
}

template <typename T>
__forceinline Vec2<T> Rsqrt(const Vec2<T> &a)
{
    return Vec2<T>(Rsqrt(a.x), Rsqrt(a.y));
}

template <typename T>
__forceinline T Length(const Vec2<T> &a)
{
    return Sqrt(Sqr(a));
}

template <typename T>
__forceinline Vec2<T> Normalize(const Vec2<T> a)
{
    return a * Rsqrt(Sqr(a));
}

template <typename T>
__forceinline Vec2<T> Min(const Vec2<T> &a, const Vec2<T> &b)
{
    return Vec2<T>(Min(a.x, b.x), Min(a.y, b.y));
}

template <typename T>
__forceinline Vec2<T> Max(const Vec2<T> &a, const Vec2<T> &b)
{
    return Vec2<T>(Max(a.x, b.x), Max(a.y, b.y));
}

template <typename T>
inline Vec2<T> ClampZero(const Vec2<T> &v)
{
    return Vec2<T>(Max(zero, v.x), Max(zero, v.y));
}

template <typename T>
__forceinline Vec2<T> Select(const Mask<T> &mask, const Vec2<T> &a, const Vec2<T> &b)
{
    return Vec2<T>(Select(mask, a.x, b.x), Select(mask, a.y, b.y));
}

template <typename T>
__forceinline Vec2<T> Select(const Vec2<T> &mask, const Vec2<T> &a, const Vec2<T> &b)
{
    return Vec2<T>(Select(mask.x, a.x, b.x), Select(mask.y, a.y, b.y));
}

typedef Vec2<u32> Vec2u;
typedef Vec2<i32> Vec2i;
typedef Vec2<f32> Vec2f;

template <i32 K>
using LaneVec2f = Vec2<LaneF32<K>>;
typedef Vec2<Lane4F32> Lane4Vec2f;

template <i32 K>
using LaneVec2u = Vec2<LaneU32<K>>;
typedef Vec2<Lane4U32> Lane4Vec2u;

} // namespace rt
